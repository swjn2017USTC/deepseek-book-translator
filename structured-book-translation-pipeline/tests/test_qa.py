"""P07 QA cascade tests (spec §13): planted-error matrix, critic honesty
contract, revision currency, constraint semantics and the D2 slow-provider
guard.  No test depends on a live network — every critic/translation call
goes through an offline double.

Planting model: the book is populated through the real contextual v2 path
(ContextualFakeClient), then the planted translations (the error rows in
``fixtures/qa_planted_errors.jsonl``) overwrite their rows directly — the way
a bad translation would be stored.  ``FakeQACriticClient`` implements
``complete`` (probe) + ``complete_json`` (verdicts) and meters
``wire_attempts`` per structured call, mirroring the real client's budget
surface.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from book_pipeline.cli import parser as cli_parser
from book_pipeline.contextual import (
    is_translation_current_v2,
    translate_v2,
)
from book_pipeline.contextual.invalidation import current_briefs, current_concept_versions
from book_pipeline.contextual.briefs import build_chapter_briefs, nodes_index
from book_pipeline.contextual.deps import load_concept_glossary
from book_pipeline.io_utils import read_jsonl, write_jsonl
from book_pipeline.llm_client import OpenAICompatibleClient
from book_pipeline.models import Segment, digest
from book_pipeline.qa import run_qa
from book_pipeline.qa import retranslate_likely_errors
from book_pipeline.qa import anomaly as anomaly_module
from book_pipeline.qa import deterministic as deterministic_module
from book_pipeline.qa import terminology as terminology_module
from book_pipeline.qa.schemas import QACriticVerdict
from book_pipeline.qa.settings import qa_settings
from book_pipeline.render import render_book
from test_contextual_translation import ContextualFakeClient, write_fixture_book

FIXTURES = Path(__file__).parent / "fixtures"
PLANTED = FIXTURES / "qa_planted_errors.jsonl"
SAMPLE_OCR = FIXTURES / "sample_ocr.json"

# The P06 rethinking demo concept glossary semantics (schema v2): Deng
# Xiaoping is hard -> 邓小平; the rest advisory.  Self-contained so the suite
# never depends on .spikes paths.
CONCEPT_GLOSSARY = {
    "schema_version": 2,
    "book_id": "rethinking-chinese-politics",
    "concepts": [
        {"concept_id": "concept-e4dc3368f941", "canonical_source": "party congress",
         "surface_forms": ["party congress"], "category": "concept",
         "senses": [{"sense_id": "default", "translation": "党代会", "constraint": "preferred",
                     "version": 1, "confidence": 0.9}]},
        {"concept_id": "concept-2934843b25bf", "canonical_source": "central committee",
         "surface_forms": ["central committee"], "category": "org",
         "senses": [{"sense_id": "default", "translation": "中央委员会", "constraint": "preferred",
                     "version": 1, "confidence": 0.9}]},
        {"concept_id": "concept-d605fc4a132c", "canonical_source": "Deng Xiaoping",
         "surface_forms": ["Deng Xiaoping"], "category": "person",
         "senses": [{"sense_id": "default", "translation": "邓小平", "constraint": "hard",
                     "version": 1, "confidence": 0.95}]},
        {"concept_id": "concept-71c89de31880", "canonical_source": "Hu Jintao",
         "surface_forms": ["Hu Jintao"], "category": "person",
         "senses": [{"sense_id": "default", "translation": "胡锦涛", "constraint": "preferred",
                     "version": 1, "confidence": 0.9}]},
        {"concept_id": "concept-56a10d2602e2", "canonical_source": "elite politics",
         "surface_forms": ["elite politics"], "category": "concept",
         "senses": [{"sense_id": "default", "translation": "精英政治", "constraint": "preferred",
                     "version": 1, "confidence": 0.9}]},
    ],
}


def fixture_rows():
    return [json.loads(line) for line in PLANTED.read_text(encoding="utf-8").splitlines()]


def segment_dicts(rows):
    out = []
    for index, row in enumerate(rows):
        out.append({
            "id": row["id"],
            "source_hash": digest(row["source_text"]),
            "kind": row.get("kind", "paragraph"),
            "source_text": row["source_text"],
            "page_json": row.get("page_json", index),
            "zone": row.get("zone", "mainmatter"),
            "parent_node_id": None,
            "level": None,
            "translatable": True,
            "metadata": {"label": "text"},
        })
    return out


def plant_errors(config, rows):
    """Overwrite the stored translations of the planted rows (the bad text a
    QA run audits); the error rows keep their original record fields."""
    path = Path(config["output_dir"]) / "translations.jsonl"
    by_id = {row["id"]: row["translated_text"] for row in rows}
    updated = []
    for record in read_jsonl(path):
        if record["segment_id"] in by_id:
            record["translated_text"] = by_id[record["segment_id"]]
        updated.append(record)
    write_jsonl(path, updated)


def qa_book(tmp_path, *, rows=None, concepts=None, qa_block=None, glossary_path=None,
            deps_enabled=True):
    """Full runnable book: planted segments translated through the real v2
    path, then the planted error texts overwritten; returns the loaded config
    (optionally with a ``qa{}`` block or a v1 glossary instead of concepts)."""
    rows = fixture_rows() if rows is None else rows
    config = write_fixture_book(
        tmp_path,
        segments=segment_dicts(rows),
        concept_overrides=concepts if concepts is not None else CONCEPT_GLOSSARY,
        deps_enabled=deps_enabled,
    )
    translate_v2(config, client=ContextualFakeClient())
    plant_errors(config, rows)
    if qa_block is not None:
        config["qa"] = qa_block
    if glossary_path is not None:
        config["glossary"] = str(glossary_path)
    return config


class FakeQACriticClient:
    """QA-4 double: probe (complete) + structured verdicts (complete_json).

    Modes: ``ok`` (programmable per-segment verdicts, default ``pass``),
    ``down`` (every wire call raises -> provider_unavailable),
    ``leaky`` (schema-valid verdict that echoes a foreign segment id -> the
    critic's echo check must reject it).  ``wire_attempts`` meters structured
    calls only (the probe is a gate, not a critic attempt — the live path
    probes with a separate client).
    """

    model = "fake-qa-critic"

    def __init__(self, verdicts=None, default="pass", mode="ok"):
        self.verdicts = {str(key): value for key, value in (verdicts or {}).items()}
        self.default = default
        self.mode = mode
        self.calls = []
        self.wire_attempts = 0

    def complete(self, system, user):
        if self.mode == "down":
            raise RuntimeError("provider down")
        payload = json.loads(user)
        if isinstance(payload, dict) and payload.get("probe") is not None:
            return json.dumps({"ok": True}), {"prompt_tokens": 1, "completion_tokens": 1}
        return json.dumps({"ok": True}), {"prompt_tokens": 1, "completion_tokens": 1}

    def _spec(self, segment_id):
        if segment_id in self.verdicts:
            value = self.verdicts[segment_id]
            if isinstance(value, dict):
                return dict(value)
            verdict, reason, confidence = value
            return {"verdict": verdict, "reason": reason, "confidence": confidence}
        return {"verdict": self.default, "reason": "no concrete defect found", "confidence": 0.95}

    def complete_json(self, system, user, model_cls, *, decode_retries=3):
        self.wire_attempts += 1
        if self.mode == "down":
            raise RuntimeError("provider down")
        payload = json.loads(user)
        target_id = payload["target"]["id"]
        self.calls.append(payload)
        spec = self._spec(target_id)
        if self.mode == "leaky":
            spec["segment_id"] = "qa-ok1"  # a real book id, never the target
        else:
            spec["segment_id"] = target_id
        try:
            return model_cls.model_validate(spec), {"prompt_tokens": 5, "completion_tokens": 7}
        except ValidationError as exc:
            raise RuntimeError(
                "Structured response failed validation after retries"
            ) from exc


def flags_for(report, segment_id, stage):
    return [
        flag for flag in report.get("flags") or []
        if flag["segment_id"] == segment_id and flag["stage"] == stage
    ]


def verdict_for(report, segment_id):
    for verdict in report.get("verdicts") or []:
        if verdict["segment_id"] == segment_id:
            return verdict
    return None


# ---------------------------------------------------------------------------
# 1. Planted-error matrix (R1-R9, plan §8.1) + only-flags rule
# ---------------------------------------------------------------------------

def test_planted_matrix_caught_with_expected_stage_verdicts(tmp_path):
    """R1-R9: every planted error is flagged exactly at the expected
    deterministic stage/check; stages marked '—' stay clean; QA-4 then judges
    the flagged segments (all pass here)."""
    config = qa_book(tmp_path)
    report = run_qa(config, client=FakeQACriticClient())

    assert report["stages"]["qa-1"]["status"] == "complete"
    assert report["stages"]["qa-2"]["status"] == "complete"
    assert report["stages"]["qa-3"]["status"] == "complete"
    assert report["stages"]["qa-4"]["status"] == "complete"
    assert report["summary"]["would_block"] == 0

    expected_checks = {
        "qa-e1": {"qa-1": ["footnotes"], "qa-2": [], "qa-3": []},
        "qa-e2": {"qa-1": ["urls"], "qa-2": [], "qa-3": []},
        "qa-e3": {"qa-1": [], "qa-2": ["hard_violation"], "qa-3": []},
        "qa-e4": {"qa-1": [], "qa-2": [], "qa-3": ["length_ratio"]},
        "qa-e5": {"qa-1": [], "qa-2": [], "qa-3": ["residual"]},
        "qa-e6": {"qa-1": ["years"], "qa-2": [], "qa-3": []},
        "qa-e7": {"qa-1": [], "qa-2": [], "qa-3": ["empty"]},
        "qa-e8": {"qa-1": [], "qa-2": [], "qa-3": ["proper_noun"]},
        "qa-e9a": {"qa-1": [], "qa-2": [], "qa-3": ["duplicate"]},
        "qa-e9b": {"qa-1": [], "qa-2": [], "qa-3": ["duplicate"]},
        "qa-e9c": {"qa-1": [], "qa-2": [], "qa-3": ["duplicate"]},
        "qa-ok1": {"qa-1": [], "qa-2": [], "qa-3": []},
    }
    for segment_id, expected in expected_checks.items():
        for stage, checks in expected.items():
            rows = flags_for(report, segment_id, stage)
            if checks:
                for check in checks:
                    assert any(row["check"] == check for row in rows), (
                        f"{segment_id} missing {stage}/{check}: {rows}"
                    )
            else:
                assert rows == [], f"{segment_id} unexpected {stage} flags: {rows}"

    # The hard glossary violation is a high-severity FAIL (ladder rule).
    hard = flags_for(report, "qa-e3", "qa-2")[0]
    assert hard["severity"] == "high"
    # QA-1 count-parity mismatches never gate (severity recorded, A1).
    assert all(row["severity"] in {"flag", "high"} for row in report["flags"])

    # translation_status.json carries the non-breaking QA extension.
    status = json.loads(
        (Path(config["output_dir"]) / "translation_status.json").read_text(encoding="utf-8")
    )
    assert status["qa_flagged_segments"] == 11
    assert status["qa_likely_error"] == 0
    assert status["qa_revised_segments"] == 0
    assert status["qa_last_run"] == report["created_at"]

    # QA artifacts exist.
    output = Path(config["output_dir"])
    assert (output / "qa_report.json").is_file()
    assert (output / "qa_audit.jsonl").is_file()
    assert (output / "qa_usage.jsonl").is_file()
    audit_rows = list(read_jsonl(output / "qa_audit.jsonl"))
    assert audit_rows  # one row per flag + per verdict


def test_qa4_receives_only_flagged_segments(tmp_path):
    """Only-flags rule: the critic sees exactly the union of QA-1/2/3 flagged
    segments — never the full book (18 translated rows in the book)."""
    fake = FakeQACriticClient()
    config = qa_book(tmp_path)
    report = run_qa(config, client=fake)
    flagged_ids = sorted({flag["segment_id"] for flag in report["flags"]})
    payload_ids = sorted(call["target"]["id"] for call in fake.calls)
    assert payload_ids == flagged_ids
    assert len(payload_ids) < 18  # full book has 12 rows -> a strict subset
    assert fake.wire_attempts == len(flagged_ids)
    for call in fake.calls:
        # bounded local context: at most 2 previous + 2 next source texts
        assert len(call["previous_source"]) <= 2
        assert len(call["next_source"]) <= 2
        assert all(len(text) <= 401 for text in call["previous_source"] + call["next_source"])
        assert call["target"]["id"] == call["target"]["id"]


def test_qa4_provider_down_records_provider_unavailable_with_deterministic_stages(tmp_path):
    """Provider down => QA-4 status provider_unavailable, every flagged
    segment needs_human/provider_unavailable, deterministic QA-1/2/3 still
    complete — zero fabricated verdicts."""
    config = qa_book(tmp_path)
    report = run_qa(config, client=FakeQACriticClient(mode="down"))
    assert report["stages"]["qa-1"]["status"] == "complete"
    assert report["stages"]["qa-2"]["status"] == "complete"
    assert report["stages"]["qa-3"]["status"] == "complete"
    assert report["stages"]["qa-4"]["status"] == "provider_unavailable"
    flagged_ids = {flag["segment_id"] for flag in report["flags"]}
    verdicts = report["verdicts"]
    assert len(verdicts) == len(flagged_ids)
    assert all(row["verdict"] == "needs_human" for row in verdicts)
    assert all(row["reason"] == "provider_unavailable" for row in verdicts)
    assert all(row["confidence"] == 0.0 for row in verdicts)
    assert set(row["segment_id"] for row in verdicts) == flagged_ids


def test_qa4_wire_budget_exhaustion_marks_remainder_needs_human(tmp_path):
    """Wire budget caps critic HTTP attempts; on exhaustion the remaining
    flagged segments get needs_human/wire_budget_exhausted (never a guess)."""
    config = qa_book(tmp_path, qa_block={"wire_budget": 3})
    fake = FakeQACriticClient()
    report = run_qa(config, client=fake)
    flagged_ids = sorted({flag["segment_id"] for flag in report["flags"]})
    assert len(flagged_ids) > 3
    assert fake.wire_attempts == 3
    assert report["stages"]["qa-4"]["wire_budget_exhausted"] is True
    exhausted = [
        row for row in report["verdicts"]
        if row["reason"] == "wire_budget_exhausted"
    ]
    assert len(exhausted) == len(flagged_ids) - 3
    assert all(row["verdict"] == "needs_human" for row in exhausted)


def test_qa4_leaky_critic_verdict_rejected(tmp_path):
    """A schema-valid verdict that echoes a foreign (context) segment id is
    rejected by the critic's echo check -> needs_human, leak suspected."""
    config = qa_book(tmp_path)
    report = run_qa(config, client=FakeQACriticClient(mode="leaky"))
    verdicts = report["verdicts"]
    assert verdicts
    assert all(row["verdict"] == "needs_human" for row in verdicts)
    assert all("leak suspected" in row["reason"] for row in verdicts)


def test_qa4_only_needs_persisted_flags_from_prior_run(tmp_path):
    """`--only qa-4` reads the flags persisted by a prior full run; without a
    prior run it fails with a clear 'run qa first' style error."""
    config = qa_book(tmp_path)
    with pytest.raises(ValueError, match="run `qa --config"):
        run_qa(config, client=FakeQACriticClient(), only="qa-4")
    # After a full run the persisted flags drive a standalone QA-4 rerun.
    run_qa(config, client=FakeQACriticClient())
    report = run_qa(config, client=FakeQACriticClient(), only="qa-4")
    assert report["stage_requested"] == "qa-4"
    assert report["stages"]["qa-4"]["status"] == "complete"
    assert {row["verdict"] for row in report["verdicts"]} == {"pass"}


def test_only_qa1_single_stage(tmp_path):
    config = qa_book(tmp_path)
    report = run_qa(config, client=FakeQACriticClient(), only="qa-1")
    assert set(report["stages"]) == {"qa-1"}
    assert report["partial"] is True
    assert flags_for(report, "qa-e1", "qa-1")  # footnote drop caught at QA-1


def test_only_qa3_surfaces_footnote_missing_anomaly(tmp_path):
    """QA-3 alone re-surfaces dropped footnote refs as footnote_missing (in
    the full cascade the same defect is reported once at QA-1)."""
    config = qa_book(tmp_path)
    report = run_qa(config, only="qa-3")
    assert set(report["stages"]) == {"qa-3"}
    assert any(
        flag["segment_id"] == "qa-e1" and flag["check"] == "footnote_missing"
        for flag in report["flags"]
    )


# ---------------------------------------------------------------------------
# 2. Revision currency + qa-retranslate (the only writer)
# ---------------------------------------------------------------------------

def test_retranslate_revision_currency_render_and_trail(tmp_path):
    """After retranslate_likely_errors: the revision row supersedes the
    original (same segment_id + source_hash + current v2 metadata), the
    original row is still in the file, render passes, and qa_report.json
    records the trail."""
    config = write_fixture_book(tmp_path)  # 18-segment contextual corpus
    translate_v2(config, client=ContextualFakeClient())
    config["qa"] = {"wire_budget": 40}
    fake = FakeQACriticClient(verdicts={
        "seg-ctx-0010": ("likely_error", "hard concept vanguard party not translated", 0.95),
    })
    report = run_qa(config, client=fake)
    target = "seg-ctx-0010"
    assert verdict_for(report, target)["verdict"] == "likely_error"

    translations_path = Path(config["output_dir"]) / "translations.jsonl"
    rows_before = list(read_jsonl(translations_path))
    original = next(row for row in rows_before if row["segment_id"] == target)

    result = retranslate_likely_errors(
        config, client=ContextualFakeClient(), segment_id=target
    )
    assert result["retranslated"] == 1
    assert result["segment_ids"] == [target]

    rows_after = list(read_jsonl(translations_path))
    target_rows = [row for row in rows_after if row["segment_id"] == target]
    assert len(target_rows) == 2  # original kept + revision appended
    revision = target_rows[-1]
    assert revision["source_hash"] == original["source_hash"]
    assert revision["status"] == "completed"
    assert revision["metadata"]["context_mode"] == "contextual_v2"
    assert revision["timestamp"] > original["timestamp"]
    assert revision["qa"]["revision_of"] == original["timestamp"]
    assert revision["qa"]["critic_verdict"] == "likely_error"
    assert revision["qa"]["flag"]["stage"] == "qa-2"

    # qa_report.json records the revision trail; translation_status.json
    # carries the qa_revised_segments extension (until the next translate run
    # refreshes the status file — translation_status is translate-owned).
    qa_report = json.loads(
        (Path(config["output_dir"]) / "qa_report.json").read_text(encoding="utf-8")
    )
    assert len(qa_report["revisions"]) == 1
    trail = qa_report["revisions"][0]
    assert trail["segment_id"] == target
    assert trail["original_timestamp"] == original["timestamp"]
    assert trail["verdict"] == "likely_error"
    status = json.loads(
        (Path(config["output_dir"]) / "translation_status.json").read_text(encoding="utf-8")
    )
    assert status["qa_revised_segments"] == 1

    # v2 currency: the revised segment is current (brief_hash + prompt + deps).
    segments = [Segment(**row) for row in read_jsonl(
        Path(config["output_dir"]) / "cleaned_segments.jsonl"
    )]
    structure = json.loads(Path(config["structure_json"]).read_text(encoding="utf-8"))
    nodes_by_id = nodes_index(structure)
    briefs = build_chapter_briefs(structure, segments)
    brief_hashes = current_briefs(briefs)
    concept_list = load_concept_glossary(
        Path(config["glossary_dependencies"]["concept_glossary_path"]))
    versions = current_concept_versions(concept_list)
    segment = next(segment for segment in segments if segment.id == target)
    completed = {row["segment_id"]: row for row in rows_after}
    assert is_translation_current_v2(
        segment, completed[target], brief_hashes=brief_hashes,
        concept_versions=versions, nodes_by_id=nodes_by_id,
    )

    # translate-v2 sees zero pending after the revision (currency via machinery).
    rerun = translate_v2(config, client=ContextualFakeClient())
    assert rerun["translated_this_run"] == 0
    assert rerun["ready_to_render"] is True

    # render passes and emits the revision's text.
    render_report = render_book(config)
    assert render_report["partial"] is False
    rendered = (Path(config["output_dir"]) / "book_translated.md").read_text(encoding="utf-8")
    assert revision["translated_text"] in rendered


def test_retranslate_bounded_by_segment_id_and_limit(tmp_path):
    config = write_fixture_book(tmp_path)
    translate_v2(config, client=ContextualFakeClient())
    config["qa"] = {"wire_budget": 40}
    fake = FakeQACriticClient(verdicts={
        "seg-ctx-0010": ("likely_error", "hard concept missing", 0.95),
        "seg-ctx-0018": ("likely_error", "hard concept missing", 0.95),
    })
    run_qa(config, client=fake)
    # --segment-id for a segment without a likely_error verdict -> clear error.
    with pytest.raises(ValueError, match="no persisted likely_error verdict"):
        retranslate_likely_errors(
            config, client=ContextualFakeClient(), segment_id="seg-ctx-0004"
        )
    # --limit caps the number of revisions.
    result = retranslate_likely_errors(
        config, client=ContextualFakeClient(), limit=1
    )
    assert result["retranslated"] == 1
    assert len(result["segment_ids"]) == 1
    # A rerun retranslates deliberately again (append-only, no auto-skip).
    result2 = retranslate_likely_errors(
        config, client=ContextualFakeClient(), limit=1
    )
    assert result2["retranslated"] == 1


def test_retranslate_blocks_without_provider_and_writes_nothing(tmp_path):
    """qa-retranslate with no live client blocks with an honest error and
    never appends a fabricated revision (D3)."""
    config = write_fixture_book(tmp_path)
    translate_v2(config, client=ContextualFakeClient())
    config["qa"] = {"wire_budget": 40}
    run_qa(config, client=FakeQACriticClient(verdicts={
        "seg-ctx-0010": ("likely_error", "hard concept missing", 0.95),
    }))
    config["provider"] = {"api_key_env": "QA_TEST_UNSET_KEY_9F3A", "requests_per_minute": 0}
    path = Path(config["output_dir"]) / "translations.jsonl"
    rows_before = list(read_jsonl(path))
    with pytest.raises(RuntimeError):
        retranslate_likely_errors(config, client=None)
    rows_after = list(read_jsonl(path))
    assert rows_after == rows_before  # nothing written


# ---------------------------------------------------------------------------
# 3. Glossary constraint semantics (x5) + v1 fallback
# ---------------------------------------------------------------------------

def _constraint_config(tmp_path, concepts):
    concept_path = tmp_path / "constraints.json"
    concept_path.write_text(
        json.dumps({"schema_version": 2, "book_id": "constraint-book", "concepts": concepts},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "glossary_dependencies": {"enabled": True, "concept_glossary_path": str(concept_path)},
        "_base_dir": str(tmp_path),
    }, concept_path


def _audit(pairs):
    """[(id, source, target)] -> audited (Segment, row) list."""
    audited = []
    for index, (segment_id, source, target) in enumerate(pairs):
        segment = Segment(
            id=segment_id, source_hash=digest(source), kind="paragraph",
            source_text=source, page_json=index, zone="mainmatter",
            parent_node_id=None,
        )
        audited.append((segment, {"segment_id": segment_id, "translated_text": target,
                                  "source_hash": digest(source), "status": "completed"}))
    return audited


def _concept(concept_id, surface, translation, constraint, version=1):
    return {
        "concept_id": concept_id,
        "canonical_source": surface,
        "surface_forms": [surface],
        "category": "concept",
        "senses": [{"sense_id": "default", "translation": translation,
                    "constraint": constraint, "version": version, "confidence": 0.9}],
    }


def test_constraint_hard_violation_is_deterministic_fail(tmp_path):
    concepts = [_concept("concept-hard-000001", "vanguard party", "先锋党", "hard")]
    config, _ = _constraint_config(tmp_path, concepts)
    audited = _audit([
        ("s1", "The vanguard party pressed its claim.", "先锋党提出了自己的主张。"),
        ("s2", "The vanguard party lost the vote.", "The vanguard party lost the vote."),
    ])
    result = terminology_module.run_terminology_audit(audited, config)
    flags = {flag["segment_id"]: flag for flag in result["flags"]}
    assert "s1" not in flags
    assert flags["s2"]["check"] == "hard_violation"
    assert flags["s2"]["severity"] == "high"
    assert result["metrics"]["hard_violations"] == 1


def test_constraint_sense_constrained_missing_translation_flags_critic(tmp_path):
    concepts = [_concept("concept-sense-00001", "commune", "公社", "sense_constrained")]
    config, _ = _constraint_config(tmp_path, concepts)
    audited = _audit([
        ("s1", "The commune governed the village.", "公社管理着村庄。"),
        ("s2", "The commune dissolved overnight.", "共同体在一夜之间解散了。"),
        ("s3", "The commune was strong.", "共同体力量强大。"),
    ])
    result = terminology_module.run_terminology_audit(audited, config)
    suspicious = [flag for flag in result["flags"] if flag["check"] == "suspicious_context"]
    # Absence of the canonical translation is critic-flagged (never a hard
    # fail): s2 chose a different valid sense rendering (共同体), s3 dropped
    # the canonical form too — both need context judgment.
    assert [flag["segment_id"] for flag in suspicious] == ["s2", "s3"]
    assert result["metrics"]["suspicious_contexts"] == 2
    assert all(flag["severity"] == "flag" for flag in suspicious)


def test_constraint_preferred_is_advisory_never_a_fail(tmp_path):
    concepts = [_concept("concept-pref-000001", "elite politics", "精英政治", "preferred")]
    config, _ = _constraint_config(tmp_path, concepts)
    audited = _audit([
        ("s1", "Elite politics shaped the court.", "精英政治塑造了宫廷。"),
        ("s2", "Elite politics mattered here.", "这里的上层政治很重要。"),
    ])
    result = terminology_module.run_terminology_audit(audited, config)
    assert result["flags"] == []
    metrics = result["metrics"]
    assert metrics["applicable"] == 2
    assert metrics["matched"] == 1
    assert metrics["term_success_rate"] == 0.5


def test_constraint_do_not_translate_surface_verbatim(tmp_path):
    concepts = [_concept("concept-dnt-0000001", "eunuch", "", "do_not_translate")]
    config, _ = _constraint_config(tmp_path, concepts)
    audited = _audit([
        ("s1", "The eunuch held the seal.", "太监保管着印玺。"),
        ("s2", "Another eunuch arrived.", "另一个太监来了。"),
        ("s3", "The eunuch vanished.", "阉人消失了。"),
    ])
    result = terminology_module.run_terminology_audit(audited, config)
    flagged = [flag for flag in result["flags"] if flag["check"] == "do_not_translate"]
    # The do-not-translate surface must appear verbatim; every segment that
    # rendered it into Chinese (太监/阉人) is deterministically flagged.
    assert [flag["segment_id"] for flag in flagged] == ["s1", "s2", "s3"]


def test_constraint_first_occurrence_only_first_segment_translated(tmp_path):
    concepts = [_concept("concept-first-00001", "dynasty", "王朝", "first_occurrence")]
    config, _ = _constraint_config(tmp_path, concepts)
    audited = _audit([
        ("s1", "The dynasty ruled the valley.", "王朝统治着这片山谷。"),
        ("s2", "The dynasty declined slowly.", "dynasty 后来逐渐衰落。"),
        ("s3", "The dynasty fell at last.", "王朝最终崩溃了。"),
    ])
    result = terminology_module.run_terminology_audit(audited, config)
    checks = {(flag["segment_id"], flag["check"]) for flag in result["flags"]}
    assert ("s1", "suspicious_context") not in checks  # first occurrence translated
    assert ("s2", "first_occurrence") not in checks    # later use kept source form
    assert ("s3", "first_occurrence") in checks        # later use dropped source form
    metrics = result["metrics"]
    assert metrics["applicable"] == 1  # only the first occurrence counts


def test_qa2_v1_glossary_fallback_and_duplicate_catch(tmp_path):
    """The '#9 uses v1-glossary path' leg: with a v1 glossary (no concept
    file) QA-2 runs success/consistency over {term, translation} rows and the
    duplicate rows are still caught by QA-3."""
    rows = [row for row in fixture_rows() if row["id"].startswith("qa-e9")]
    config = qa_book(
        tmp_path, rows=rows, deps_enabled=False,
        glossary_path=str(tmp_path / "v1_glossary.json"),
    )
    v1_glossary = [
        {"term": "archives", "translation": "档案"},
        {"term": "minutes", "translation": "会议记录"},
    ]
    (tmp_path / "v1_glossary.json").write_text(
        json.dumps(v1_glossary, ensure_ascii=False), encoding="utf-8")
    report = run_qa(config, client=FakeQACriticClient())
    assert report["stages"]["qa-2"]["glossary_source"] == "v1"
    for segment_id in ("qa-e9a", "qa-e9b", "qa-e9c"):
        assert any(
            flag["check"] == "duplicate" for flag in flags_for(report, segment_id, "qa-3")
        )


# ---------------------------------------------------------------------------
# 4. Deterministic check semantics (unit level)
# ---------------------------------------------------------------------------

def test_qa1_url_set_parity_catches_replacement(tmp_path):
    """URLs use set parity: swapping one URL for another keeps counts equal
    but must still flag (bytes are never legitimately translated)."""
    segment = Segment(id="u1", source_hash=digest("a"), kind="paragraph",
                      source_text="See https://example.org/a here.",
                      page_json=1, zone="mainmatter", parent_node_id=None)
    flags = deterministic_module.check_segment(segment, "See https://example.org/b here.")
    assert any(flag["check"] == "urls" for flag in flags)


def test_qa1_url_set_parity_ignores_attached_cjk_prose(tmp_path):
    segment = Segment(id="u2", source_hash=digest("a"), kind="footnote",
                      source_text="See http://example.org/a. Then continue.",
                      page_json=1, zone="backmatter", parent_node_id=None)
    translated = "参见 http://example.org/a。另见后文。"
    assert not any(
        flag["check"] == "urls"
        for flag in deterministic_module.check_segment(segment, translated)
    )


def test_qa1_classes_no_zero_length_and_overlap_documented(tmp_path):
    import book_pipeline.qa.deterministic as det

    assert det.roman_matches("IMPORTANT is not roman") == []
    assert det.roman_matches("Chapter II begins in MMXXIV") == ["II", "MMXXIV"]
    segment = Segment(id="r1", source_hash=digest("a"), kind="paragraph",
                      source_text="In 1978, 12% of the II Corps read Vogel (2011).",
                      page_json=1, zone="mainmatter", parent_node_id=None)
    # full preservation -> no flags
    good = "1978年，II军团的12%读过Vogel (2011)。"
    assert deterministic_module.check_segment(segment, good) == []
    # dropped year -> years (and numerals) flag
    bad = "那一年，II军团的12%读过Vogel (2011)。"
    checks = {flag["check"] for flag in deterministic_module.check_segment(segment, bad)}
    assert "years" in checks


def test_qa1_load_gate_rejects_duplicate_and_unknown_ids():
    segments = [
        Segment(id="dup", source_hash="a" * 64, kind="paragraph", source_text="x",
                page_json=1, zone="mainmatter", parent_node_id=None),
        Segment(id="dup", source_hash="b" * 64, kind="paragraph", source_text="y",
                page_json=2, zone="mainmatter", parent_node_id=None),
    ]
    with pytest.raises(ValueError, match="Duplicate segment id"):
        deterministic_module.load_gate(segments, {})
    segments = segments[:1]
    with pytest.raises(ValueError, match="unknown cleaned segment"):
        deterministic_module.load_gate(
            segments,
            {"ghost": {"segment_id": "ghost", "status": "completed"}},
        )
    # partial coverage is normal; untranslated segments are simply not audited.
    rows = deterministic_module.audit_rows(
        segments,
        {"ghost": {"segment_id": "ghost", "status": "completed",
                   "source_hash": "a" * 64, "translated_text": "译：x"}},
    )
    assert rows == []


def test_qa3_duplicate_requires_more_than_two_identical_rows():
    def segments_for(texts):
        return [
            (Segment(id=f"d{index}", source_hash=digest("s"), kind="paragraph",
                     source_text="source text number {index}", page_json=index,
                     zone="mainmatter", parent_node_id=None),
             {"segment_id": f"d{index}", "translated_text": text})
            for index, text in enumerate(texts)
        ]

    assert anomaly_module.duplicate_anomalies(segments_for(["同", "同"])) == []
    flags = anomaly_module.duplicate_anomalies(segments_for(["同", "同", "同"]))
    assert {flag["segment_id"] for flag in flags} == {"d0", "d1", "d2"}


def test_qa3_repeated_sentence_loop_within_segment():
    segment = Segment(id="loop", source_hash=digest("s"), kind="paragraph",
                      source_text="The drums beat the alarm again and again and again.",
                      page_json=1, zone="mainmatter", parent_node_id=None)
    flags = anomaly_module.anomalies_for(
        segment, "警报响了。警报响了。警报响了。再也没有别的消息。",
        qa_settings({}),
    )
    assert any(flag["check"] == "repeated_sentence" for flag in flags)


# ---------------------------------------------------------------------------
# 5. Schema + settings contract
# ---------------------------------------------------------------------------

def test_qa_critic_schema_rejects_extra_invalid_and_boundary_fields():
    QACriticVerdict(segment_id="s1", verdict="likely_error",
                    reason="dropped number", confidence=0.9)
    with pytest.raises(ValidationError):
        QACriticVerdict(segment_id="s1", verdict="likely_error",
                        reason="dropped number", confidence=0.9,
                        invented_key="leaked context")
    with pytest.raises(ValidationError):
        QACriticVerdict(segment_id="s1", verdict="probably_fine",
                        reason="dropped number", confidence=0.9)
    with pytest.raises(ValidationError):
        QACriticVerdict(segment_id="s1", verdict="pass",
                        reason="", confidence=0.9)
    with pytest.raises(ValidationError):
        QACriticVerdict(segment_id="s1", verdict="pass",
                        reason="x" * 401, confidence=0.9)
    with pytest.raises(ValidationError):
        QACriticVerdict(segment_id="s1", verdict="pass",
                        reason="ok", confidence=1.1)


def test_qa_settings_tolerant_parse_and_validation():
    settings = qa_settings({})
    assert settings.enabled is True
    assert settings.wire_budget == 20
    assert settings.max_len_ratio == 0.5
    assert settings.max_residual_run == 8
    assert settings.critic_role == "qa_critic"
    parsed = qa_settings({"qa": {"wire_budget": 5, "mystery_key": True,
                                 "max_residual_run": 12}})
    assert parsed.wire_budget == 5
    assert parsed.max_residual_run == 12
    with pytest.raises(ValueError, match="max_len_ratio"):
        qa_settings({"qa": {"max_len_ratio": 1.5}})


# ---------------------------------------------------------------------------
# 6. D2 slow-provider guard
# ---------------------------------------------------------------------------

def test_slow_provider_client_carries_timeout_1200_retries_8(monkeypatch):
    """Constructed clients honor the slow-provider values from the provider
    dict (llm_client reads timeout_seconds/retries — amendment A2)."""
    monkeypatch.setenv("TEST_LLM_KEY", "dummy-not-a-secret")
    client = OpenAICompatibleClient({
        "api_key_env": "TEST_LLM_KEY",
        "api_url": "https://example.invalid/v1/chat/completions",
        "model": "fixture-model",
        "timeout_seconds": 1200,
        "retries": 8,
    })
    assert client.timeout == 1200
    assert client.retries == 8


def test_initialize_project_template_emits_slow_provider(tmp_path):
    """initialize_project's provider template emits timeout_seconds 1200 /
    retries 8 (amendment A2)."""
    from book_pipeline.workflow import initialize_project

    project_dir = tmp_path / "new-book"
    initialized = initialize_project(
        input_json=SAMPLE_OCR,
        book_id="new-fixture-book",
        book_title="New Fixture Book",
        book_title_zh="新测试书",
        project_dir=project_dir,
    )
    translation_config = json.loads(
        Path(initialized["translation_config"]).read_text(encoding="utf-8")
    )
    provider = translation_config["provider"]
    assert provider["timeout_seconds"] == 1200
    assert provider["retries"] == 8


# ---------------------------------------------------------------------------
# 7. CLI wiring
# ---------------------------------------------------------------------------

def test_cli_qa_subcommands_parse():
    parser = cli_parser()
    args = parser.parse_args(["qa", "--config", "book.json", "--only", "qa-3", "--limit", "5"])
    assert args.command == "qa"
    assert args.only == "qa-3"
    assert args.limit == 5
    args = parser.parse_args(["qa", "--config", "book.json"])
    assert args.only is None
    args = parser.parse_args(
        ["qa-retranslate", "--config", "book.json", "--segment-id", "seg-1", "--limit", "2"])
    assert args.command == "qa-retranslate"
    assert args.segment_id == "seg-1"
