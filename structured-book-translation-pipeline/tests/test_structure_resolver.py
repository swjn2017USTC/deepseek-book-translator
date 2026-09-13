"""P03 v2 structure resolver tests (plan §5.9): pydantic strict intake,
window determinism, evidence reader, llm_roles selection, and the orchestrator
E2E (Pass A/B/C -> decisions/rescue -> fake chapter apply -> validation ->
repair -> status), rescue round-trip, and fail-closed fallback.

No real chapter binary and no live provider is ever invoked: the chapter
``apply-decisions`` subprocess is replaced by :func:`fake_chapter_apply`, and
LLM calls go to the JSON-mode :class:`FakeClient`.
"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from book_pipeline.io_utils import read_jsonl
from book_pipeline.structure_resolver.evidence_reader import BookEvidence
from book_pipeline.structure_resolver.local_window import build_windows
from book_pipeline.structure_resolver.orchestrator import (
    StructureResolver,
    load_v2_spec,
)
from book_pipeline.structure_resolver.schemas import (
    RescueCandidate,
    StructureDecision,
)

from conftest import (
    BOOK_ID,
    CANDIDATES,
    FakeClient,
    write_fixture_evidence,
    write_fixture_spec,
)

CAND_BY_ID = {spec["cid"]: spec for spec in CANDIDATES}
FIXTURE_CANDIDATE_IDS = set(CAND_BY_ID)
FIXTURE_BLOCK_IDS = {spec["blk"] for spec in CANDIDATES}


# ------------------------------------------------------------- fake chapter
def fake_chapter_apply(recorded):
    """Build a chapter-side ``apply-decisions`` stand-in.

    The returned callable takes ``(chapter_root, arguments)`` like the real
    subprocess driver, reads the written decisions/rescue under
    ``structure_dir/v2/``, simulates a deterministic validator (orphan sections
    are gate errors; unknown parent candidates are errors), and writes
    ``validation.json`` + ``book_structure.json`` into ``v2/``.
    """

    def _apply(chapter_root, arguments):
        args = list(arguments)
        assert args[0] == "apply-decisions"
        decisions_path = Path(args[args.index("--decisions") + 1])
        rescue_path = Path(args[args.index("--rescue") + 1])
        v2_dir = decisions_path.parent
        recorded.append(v2_dir)
        records = [row for row in read_jsonl(decisions_path) if row.get("candidate_id")]
        rescues = list(read_jsonl(rescue_path))
        # rescue must-exist: every rescued block id must exist in the fixture
        # page evidence
        for rescue in rescues:
            block_id = str(rescue.get("block_id"))
            if block_id not in FIXTURE_BLOCK_IDS:
                _write_validation(v2_dir, errors=[
                    {"code": "rescue_block_unknown", "node_id": None,
                     "message": f"block_id {block_id} not in OCR pages"}
                ])
                return
        errors = []
        for record in records:
            cid = str(record.get("candidate_id"))
            parent = record.get("parent_candidate_id")
            if parent and parent not in FIXTURE_CANDIDATE_IDS:
                errors.append({"code": "unknown_parent_candidate", "node_id": "node-" + cid[5:],
                               "message": f"parent {parent} unknown"})
            if record.get("decision") == "heading" and record.get("kind") in ("chapter", "section") \
                    and record.get("level") in (2, 3) and not parent:
                if record.get("kind") == "section":
                    errors.append({"code": "orphan_section", "node_id": "node-" + cid[5:],
                                   "message": f"section {cid} has no parent heading"})
        gate_failures = list(errors)
        _write_validation(v2_dir, errors=errors, gate_failures=gate_failures)
        nodes = [
            {"id": "node-" + record["candidate_id"][5:], "kind": record.get("kind") or "body",
             "level": record.get("level"), "candidate_id": record["candidate_id"]}
            for record in records
        ]
        (v2_dir / "book_structure.json").write_text(
            json.dumps({"schema_version": 3, "book_id": BOOK_ID, "nodes": nodes}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return _apply


def _write_validation(v2_dir, errors=None, gate_failures=None, warnings=None):
    payload = {
        "ok": not (errors or gate_failures),
        "errors": errors or [],
        "warnings": warnings or [],
        "gate_failures": gate_failures or [],
        "metrics": {"node_count": 0, "heading_count": 0},
    }
    (v2_dir / "validation.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _make_environment(tmp_path, client_options=None, **spec_overrides):
    """Structure dir + evidence + spec + FakeClient + fake chapter apply."""
    structure_dir = tmp_path / "structure"
    work_dir = tmp_path / "work"
    evidence_sha256 = write_fixture_evidence(structure_dir)
    spec = write_fixture_spec(structure_dir, work_dir, **spec_overrides)
    client = FakeClient(mode="json", **dict(client_options or {}))
    recorded = []
    run_chapter = fake_chapter_apply(recorded)
    resolver = StructureResolver(
        spec, window_size=spec_overrides.get("window_size", 20),
        client_factory=lambda role, model: client,
        run_chapter=run_chapter,
    )
    return {
        "structure_dir": structure_dir,
        "work_dir": work_dir,
        "evidence_sha256": evidence_sha256,
        "client": client,
        "recorded": recorded,
        "resolver": resolver,
    }


# ------------------------------------------------------- pydantic strict intake
def test_schema_rejects_bad_decision_enum():
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="paragraph",
                          confidence=0.9, evidence_refs=["a"])


def test_schema_rejects_bad_kind_and_bad_level():
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          kind="subsection", level=2, confidence=0.9, evidence_refs=["a"])
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          kind="chapter", level=0, confidence=0.9, evidence_refs=["a"])
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          kind="chapter", level=7, confidence=0.9, evidence_refs=["a"])


def test_schema_rejects_hallucinated_ids_and_free_text_titles():
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-invented-not-hex", decision="heading",
                          kind="chapter", level=2, confidence=0.9, evidence_refs=["a"])
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          kind="chapter", level=2, confidence=0.9, evidence_refs=["a"],
                          title_source="Fabricated Title")
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          kind="chapter", level=2, confidence=0.9, evidence_refs=["a"],
                          exact_text="Also forbidden")
    with pytest.raises(ValidationError):
        RescueCandidate(block_id="blk-hallucinated", confidence=0.7, evidence_refs=["a"])


def test_schema_requires_evidence_refs_and_heading_fields():
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          kind="chapter", level=2, confidence=0.9, evidence_refs=[])
    with pytest.raises(ValidationError):
        # heading without kind/level
        StructureDecision(candidate_id="cand-0000000000000001", decision="heading",
                          confidence=0.9, evidence_refs=["a"])
    with pytest.raises(ValidationError):
        # body carrying level
        StructureDecision(candidate_id="cand-0000000000000001", decision="body", level=2,
                          confidence=0.9, evidence_refs=["a"])


def test_schema_rejects_unknown_top_level_keys():
    with pytest.raises(ValidationError):
        StructureDecision(candidate_id="cand-0000000000000001", decision="body",
                          confidence=0.9, evidence_refs=["a"], node_id="node-x")


# --------------------------------------------------------------- window builder
def test_window_builder_is_deterministic(tmp_path):
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    evidence = BookEvidence(structure_dir)
    first = build_windows(evidence, window_size=20)
    second = build_windows(evidence, window_size=20)
    assert first == second
    assert len(first) >= 1
    # windows are non-overlapping, in order, and cover every candidate once
    seen = []
    for window in first:
        assert window["end_page"] - window["start_page"] <= 20
        for candidate in window["candidates"]:
            seen.append(candidate["candidate_id"])
    assert sorted(seen) == sorted(evidence.candidate_ids)
    # 40-page width collapses the fixture into one window as well
    assert len(build_windows(evidence, window_size=40)) == len(first)


def test_window_builder_rejects_out_of_range_size(tmp_path):
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    evidence = BookEvidence(structure_dir)
    with pytest.raises(ValueError):
        build_windows(evidence, window_size=10)
    with pytest.raises(ValueError):
        build_windows(evidence, window_size=41)


# ---------------------------------------------------------------- evidence reader
def test_evidence_reader_builds_candidate_index_and_pages(tmp_path):
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    evidence = BookEvidence(structure_dir)
    assert evidence.book_id == BOOK_ID
    assert evidence.candidate_count == len(CANDIDATES)
    assert set(evidence.candidates_by_id) == FIXTURE_CANDIDATE_IDS
    assert evidence.page_count == 8
    assert evidence.zone_of(5) == "backmatter"
    ordered = evidence.ordered_candidates()
    assert [c["candidate_id"] for c in ordered] == sorted(
        [c["candidate_id"] for c in ordered]
    )
    # sha256 is stable across loads of identical bytes
    assert BookEvidence(structure_dir).evidence_sha256 == evidence.evidence_sha256
    assert len(evidence.evidence_sha256) == 64


def test_evidence_reader_tolerates_missing_pdf_evidence(tmp_path):
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    (structure_dir / "evidence" / "pdf_evidence.json").unlink()
    evidence = BookEvidence(structure_dir)
    assert evidence.pdf_evidence is None
    assert evidence.heading_candidates


# ------------------------------------------------------------------- llm roles
def test_llm_roles_selection_across_passes(tmp_path):
    env = _make_environment(tmp_path, client_options={})
    resolver = env["resolver"]
    assert resolver._model("structure_global") == "deepseek-v4-flash"
    assert resolver._model("structure_local") == "deepseek-v4-flash"
    assert resolver._model("structure_review") == "deepseek-v4-flash"
    # role override via spec llm_roles
    env2 = _make_environment(tmp_path, llm_roles={"structure_local": "custom-local"})
    assert env2["resolver"]._model("structure_local") == "custom-local"


# ------------------------------------------------------------------- E2E flow
def test_orchestrator_e2e_ok_with_rescue_round_trip_and_usage_log(tmp_path):
    env = _make_environment(tmp_path, client_options={"rescue": "one"})
    resolver = env["resolver"]
    result = resolver.run()
    assert result["status"] == "structure_v2_ok"
    assert result["book_id"] == BOOK_ID
    assert env["recorded"], "fake chapter apply must have run"
    v2_dir = env["structure_dir"] / "v2"
    # decisions file: meta line first, then decision records
    lines = [json.loads(line) for line in (v2_dir / "structure_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "_meta" in lines[0]
    assert lines[0]["_meta"]["schema_version"] == 2
    assert lines[0]["_meta"]["evidence_sha256"] == env["evidence_sha256"]
    assert lines[0]["_meta"]["repair_total"] == 0
    records = [row for row in lines[1:]]
    assert records and all(row["schema_version"] == 2 for row in records)
    # rescue records reference an existing block id and carry rescue_model
    rescues = list(read_jsonl(v2_dir / "structure_rescue.jsonl"))
    assert rescues
    assert all(row["block_id"] in FIXTURE_BLOCK_IDS for row in rescues)
    assert all(row["rescue_model"] for row in rescues)
    assert all(row["prompts_version"] == 1 for row in rescues)
    # status + fake tree + validation written by the apply step
    assert json.loads((v2_dir / "status.json").read_text(encoding="utf-8"))["status"] == "structure_v2_ok"
    assert json.loads((v2_dir / "validation.json").read_text(encoding="utf-8"))["ok"] is True
    assert (v2_dir / "book_structure.json").is_file()
    # usage log written in work_dir with role/model/kind/usage
    usage_rows = list(read_jsonl(env["work_dir"] / "structure_v2_usage.jsonl"))
    kinds = {row["kind"] for row in usage_rows}
    roles = {row["role"] for row in usage_rows}
    assert "grammar" in kinds and "local" in kinds and "rescue" in kinds
    assert "structure_global" in roles and "structure_local" in roles
    assert all(row["model"] for row in usage_rows)


def test_orchestrator_repair_round_resolves_validation(tmp_path):
    env = _make_environment(
        tmp_path,
        client_options={"local": "bad", "repair": "fix"},
    )
    resolver = env["resolver"]
    result = resolver.run()
    assert result["status"] == "structure_v2_ok"
    assert result["repair_total"] == 1
    # repaired decision file reflects round 1 and no orphan sections remain
    v2_dir = env["structure_dir"] / "v2"
    lines = [json.loads(line) for line in (v2_dir / "structure_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines[0]["_meta"]["repair_total"] == 1
    repairs = [row for row in lines[1:] if row["repair_round"] == 1]
    assert repairs
    orphans = [
        row for row in lines[1:]
        if row["decision"] == "heading" and row.get("kind") == "section" and not row.get("parent_candidate_id")
    ]
    assert not orphans


def test_orchestrator_human_review_after_two_failed_repair_rounds(tmp_path):
    env = _make_environment(tmp_path, client_options={"local": "bad", "repair": "stubborn"})
    resolver = env["resolver"]
    result = resolver.run()
    assert result["status"] == "needs_human_review"
    assert result["repair_total"] == 2
    v2_dir = env["structure_dir"] / "v2"
    packets = list(read_jsonl(v2_dir / "human_review_packets.jsonl"))
    assert packets
    validation = json.loads((v2_dir / "validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is False


def test_orchestrator_provider_unavailable_fails_closed_with_flushed_decisions(tmp_path):
    env = _make_environment(tmp_path, client_options={"fail_on_pass": "rescue"})
    resolver = env["resolver"]
    result = resolver.run()
    assert result["status"] == "structure_llm_unavailable"
    v2_dir = env["structure_dir"] / "v2"
    lines = [json.loads(line) for line in (v2_dir / "structure_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "_meta" in lines[0]
    assert len(lines) > 1, "decisions from completed passes must be flushed"
    status = json.loads((v2_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "structure_llm_unavailable"
    # the meta line is still written even when Pass A itself is down
    env2 = _make_environment(tmp_path, client_options={"fail_on_pass": "grammar"})
    result2 = env2["resolver"].run()
    assert result2["status"] == "structure_llm_unavailable"
    meta = [json.loads(line) for line in (env2["structure_dir"] / "v2" / "structure_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][0]
    assert "_meta" in meta


def test_orchestrator_rejects_hallucinated_rescue_block(tmp_path):
    env = _make_environment(tmp_path, client_options={"rescue": "hallucinated"})
    resolver = env["resolver"]
    result = resolver.run()
    assert result["status"] == "structure_llm_unavailable"
    assert "block_id" in result.get("error", "")
    v2_dir = env["structure_dir"] / "v2"
    rescues = list(read_jsonl(v2_dir / "structure_rescue.jsonl"))
    assert rescues == []


def test_orchestrator_recovers_on_second_grammar_attempt(tmp_path):
    """A pydantic-rejected Pass A response is retried, not fatal."""
    env = _make_environment(tmp_path, client_options={"grammar": "bad-once"})
    resolver = env["resolver"]
    result = resolver.run()
    assert result["status"] == "structure_v2_ok"
    assert len(env["client"].calls) >= 2  # grammar + later passes


def test_cli_structure_v2_parses_and_smokes(tmp_path):
    """structure-v2 subcommand exists; with a fixture spec and a stubbed
    chapter/LLM layer it runs to a status without network or the real binary."""
    from book_pipeline.cli import main as cli_main

    structure_dir = tmp_path / "structure"
    work_dir = tmp_path / "work"
    write_fixture_evidence(structure_dir)
    spec_path = write_fixture_spec(structure_dir, work_dir, tmp_path=tmp_path)
    client = FakeClient(mode="json")
    recorded = []
    from book_pipeline.structure_resolver.orchestrator import StructureResolver

    captured = {}

    def stub_run(spec, **kwargs):
        resolver = StructureResolver(
            spec, window_size=kwargs.get("window_size", 20),
            client_factory=lambda role, model: client,
            run_chapter=fake_chapter_apply(recorded),
        )
        captured["status"] = resolver.run()["status"]
        return captured["status"]

    import book_pipeline.cli as cli
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "run_structure_v2", stub_run)
    try:
        cli_main(["structure-v2", "--spec", str(spec_path), "--window-size", "20"])
    finally:
        monkeypatch.undo()
    assert captured["status"] == "structure_v2_ok"


def test_validate_decision_records_rejects_hallucinated_candidates(tmp_path):
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    evidence = BookEvidence(structure_dir)
    from book_pipeline.structure_resolver.local_window import (
        StructureParseError,
        validate_decision_records,
    )
    window = build_windows(evidence, window_size=20)[0]
    bad = {"candidate_id": "cand-ffffffffffffffff", "decision": "body",
           "confidence": 0.5, "evidence_refs": ["a"]}
    with pytest.raises(StructureParseError, match="not in window"):
        validate_decision_records(window, [bad], [])


# ------------------------------------------------------------------- spec/CLI
def test_load_v2_spec_validates_and_rejects_missing(tmp_path):
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    spec_path = write_fixture_spec(structure_dir, tmp_path / "work", tmp_path=tmp_path)
    spec = load_v2_spec(spec_path)
    assert spec["book_id"] == BOOK_ID
    assert Path(spec["structure_dir"]).is_dir()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_v2_spec(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 2, "book_id": "x"}), encoding="utf-8")
    with pytest.raises(Exception):
        load_v2_spec(bad)


def test_orchestrator_page_range_bounds_windows(tmp_path):
    """spec.page_range (inclusive print-page bounds) restricts windowed
    candidates deterministically — the live bounded-run knob (plan §6)."""
    from book_pipeline.structure_resolver.local_window import MAX_WINDOWS
    from tests.conftest import FakeClient, write_fixture_evidence, write_fixture_spec

    assert MAX_WINDOWS == 10
    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    spec = write_fixture_spec(structure_dir, tmp_path / "work", page_range=[2, 4])
    client = FakeClient(mode="json")
    resolver = StructureResolver(
        spec, window_size=40,
        client_factory=lambda role, model: client,
        run_chapter=lambda *a: None,
    )
    windows = resolver._windows(BookEvidence(structure_dir))
    pages = {int(c["page_json"]) for w in windows for c in w["candidates"]}
    assert pages <= {2, 3, 4}


def test_orchestrator_max_windows_guard_raises_when_natural_exceeds(tmp_path):
    """window_limit is a fail-closed guard: a cap below the natural window
    count raises (no silent truncation of evidence)."""
    from tests.conftest import FakeClient, write_fixture_evidence, write_fixture_spec

    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    spec = write_fixture_spec(structure_dir, tmp_path / "work", max_windows=1)
    client = FakeClient(mode="json")
    resolver = StructureResolver(
        spec, window_size=40,
        client_factory=lambda role, model: client,
        run_chapter=lambda *a: None,
    )
    with pytest.raises(ValueError, match="hard cap"):
        resolver._windows(BookEvidence(structure_dir))


def test_orchestrator_wire_budget_is_enforced(monkeypatch, tmp_path):
    """Reviewer M1 regression: client-internal wire retries consume the request
    budget — a single logical step whose client reports > budget wire attempts
    must fail closed with ProviderUnavailable."""
    from book_pipeline.structure_resolver.orchestrator import LLM_REQUEST_BUDGET, ProviderUnavailable
    from tests.conftest import write_fixture_evidence, write_fixture_spec

    structure_dir = tmp_path / "structure"
    write_fixture_evidence(structure_dir)
    spec = write_fixture_spec(structure_dir, tmp_path / "work")

    class _WireHeavyClient:
        model = "heavy"
        wire_attempts = LLM_REQUEST_BUDGET + 1

        def complete_json(self, system, user, model_cls, *, decode_retries=3):
            from pydantic import TypeAdapter
            return TypeAdapter(model_cls).validate_python([]), {"prompt_tokens": 1, "completion_tokens": 1}

    resolver = StructureResolver(
        spec, window_size=40,
        client_factory=lambda role, model: _WireHeavyClient(),
        run_chapter=lambda *a: None,
    )
    result = resolver.run()  # run() maps ProviderUnavailable -> fail-closed status
    assert result["status"] == "structure_llm_unavailable"
    assert "wire request budget exceeded" in result["error"]
