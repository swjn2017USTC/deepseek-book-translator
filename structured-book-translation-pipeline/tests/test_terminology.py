"""Terminology resolver v2 tests (plan §6): deterministic miner, schema
rejection, known-term recall, polysemy non-merge, constraint clamp,
progress/resume + fail-closed, wire caps, offline zero-LLM, master provenance,
spec validation and the additive CLI path.

No test depends on a live network: every LLM-shaped call goes to the
terminology-JSON-mode :class:`TerminologyFakeClient` in
``terminology_fixtures.py``.
"""
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from book_pipeline.glossary import _normal
from book_pipeline.io_utils import read_jsonl
from book_pipeline.terminology.evaluate import (
    duplicate_concept_errors,
    needs_human_rows,
    polysemy_check,
)
from book_pipeline.terminology.miner import mine_candidates
from book_pipeline.terminology.orchestrator import (
    load_terminology_spec,
    run_terminology_v2,
)
from book_pipeline.terminology.schemas import (
    ConceptConsolidation,
    ConceptRecord,
    Sense,
    TermhoodJudgment,
)

from terminology_fixtures import (
    BOOK_ID,
    BOOK_TITLE,
    KNOWN_TERMS,
    MINING_DEFAULTS,
    TerminologyFakeClient,
    concept_id_for,
    make_terminology_spec,
    write_spec_file,
)

RUN_OPTIONS = {"term_miner": 25, "term_consolidator": 4}


def _run(spec, client=None, **kwargs):
    client = client or TerminologyFakeClient()
    return run_terminology_v2(spec, client_factory=lambda role, model: client, **kwargs)


def _work(spec):
    return Path(spec["work_dir"])


def _glossary_concepts(spec):
    path = _work(spec) / "concept_glossary.json"
    return json.loads(path.read_text(encoding="utf-8"))["concepts"]


# ------------------------------------------------------------ 1. miner
def test_miner_determinism_and_signal_coverage(tmp_path):
    spec = make_terminology_spec(tmp_path)
    segments = list(read_jsonl(Path(spec["cleaned_segments"])))
    mining = dict(MINING_DEFAULTS)
    mining["seed_terms"] = ["vanguard party"]
    first = mine_candidates(mining, segments)
    second = mine_candidates(mining, segments)
    dumped = [row.model_dump() for row in first]
    # Byte-stable determinism: same candidate set, same order, same ids.
    assert dumped == [row.model_dump() for row in second]
    assert len(dumped) == len({row["candidate_id"] for row in dumped})

    by_type = {}
    surfaces = set()
    for row in first:
        surfaces.add(row.surface)
        by_type.setdefault(row.source_type, set()).add(row.surface)
        expected_id = "term-" + hashlib.sha256(_normal(row.surface).encode("utf-8")).hexdigest()[:12]
        assert row.candidate_id == expected_id
        assert row.evidence_segment_ids and all(str(item).startswith("seg-") for item in row.evidence_segment_ids)
        assert row.occurrences >= 1
    # Every v1 signal family fires on its dedicated fixture text.
    assert "vanguard party" in by_type["manually_seeded_source_term"]
    assert any(surface.startswith("Chapter") for surface in by_type["heading_phrase"])
    assert "The Agrarian Question" in by_type["heading_subphrase"]
    assert "iron cage" in by_type["quoted_phrase"]
    assert "Karl Marx" in by_type["proper_name"]
    assert "agrarian question" in by_type["repeated_phrase"]
    assert "rural commune" in by_type["repeated_phrase"]
    assert "counter-revolutionary" in by_type["distinctive_word"]
    # Boundary guard: the corpus's only "civil society"-like text is
    # "incivil society"; no candidate may be mined for the clean surface.
    assert "civil society" not in surfaces


# ------------------------------------------------- 2. known-term recall
def test_known_term_recall_at_mining_and_after_termhood(tmp_path):
    spec = make_terminology_spec(tmp_path, known_terms=KNOWN_TERMS)
    result = _run(spec)
    assert result["status"] == "terminology_v2_ok"
    metrics = json.loads((_work(spec) / "terminology_metrics.json").read_text(encoding="utf-8"))
    recall = metrics["known_term_recall"]
    assert recall["known_term_count"] == 3
    assert recall["mined_count"] == 3
    assert recall["recall_at_mining"] == 1.0
    assert recall["recall_after_termhood"] == 1.0
    assert all(term["retained_after_termhood"] for term in recall["per_term"])

    # A known term the LLM rejects is measured as retention loss, not hidden.
    spec_reject = make_terminology_spec(tmp_path / "reject", known_terms=KNOWN_TERMS)
    client = TerminologyFakeClient(options={"reject_surfaces": ["agrarian question"]})
    result = _run(spec_reject, client)
    assert result["status"] == "terminology_v2_ok"
    metrics = json.loads((_work(spec_reject) / "terminology_metrics.json").read_text(encoding="utf-8"))
    recall = metrics["known_term_recall"]
    assert recall["recall_at_mining"] == 1.0
    assert recall["recall_after_termhood"] == pytest.approx(0.6667, abs=1e-4)
    by_surface = {term["surface"]: term for term in recall["per_term"]}
    assert by_surface["agrarian question"]["retained_after_termhood"] is False
    assert by_surface["agrarian question"]["loss_stage"] == "rejected_by_termhood"
    assert by_surface["rural commune"]["retained_after_termhood"] is True
    assert metrics["termhood"]["ordinary_phrase_reject_rate"] > 0


# ------------------------------------------------- 3. schema rejections
def test_termhood_schema_rejections():
    good_id = "term-" + "a" * 12
    with pytest.raises(ValidationError):
        TermhoodJudgment(candidate_id="term-nothex", is_term=True, confidence=0.9)
    with pytest.raises(ValidationError):  # hallucinated id shape
        TermhoodJudgment(candidate_id="term-ffffffffffffffff", is_term=True, confidence=0.9)
    with pytest.raises(ValidationError):  # reject without a reason
        TermhoodJudgment(candidate_id=good_id, is_term=False, confidence=0.9)
    with pytest.raises(ValidationError):  # invented key
        TermhoodJudgment(candidate_id=good_id, is_term=True, confidence=0.9, invented="x")
    with pytest.raises(ValidationError):  # bad category
        TermhoodJudgment(candidate_id=good_id, is_term=True, confidence=0.9, category="foo")
    with pytest.raises(ValidationError):  # confidence out of range
        TermhoodJudgment(candidate_id=good_id, is_term=True, confidence=1.2)
    with pytest.raises(ValidationError):  # possible_senses must be >= 1
        TermhoodJudgment(candidate_id=good_id, is_term=True, confidence=0.9, possible_senses=0)
    valid = TermhoodJudgment(
        candidate_id=good_id, is_term=False, confidence=0.4,
        reason="ordinary high-frequency phrase")
    assert valid.is_term is False


def test_consolidation_schema_rejections():
    with pytest.raises(ValidationError):  # invented top-level key
        ConceptConsolidation(concepts=[], merges=[], invented=1)
    with pytest.raises(ValidationError):  # bad concept id
        ConceptRecord(concept_id="concept-nope", canonical_source="x", surface_forms=["x"],
                      category="concept", senses=[Sense(sense_id="default", translation="译",
                                                        confidence=0.9)])
    with pytest.raises(ValidationError):  # unknown category
        ConceptRecord(concept_id="concept-" + "b" * 12, canonical_source="x", surface_forms=["x"],
                      category="mystery", senses=[Sense(sense_id="default", translation="译",
                                                        confidence=0.9)])
    with pytest.raises(ValidationError):  # bad sense id
        Sense(sense_id="sense-prime", translation="译", confidence=0.9)
    with pytest.raises(ValidationError):  # missing translation
        Sense(sense_id="default", translation="  ", confidence=0.9)


def test_hallucinated_termhood_id_fails_closed(tmp_path):
    spec = make_terminology_spec(tmp_path)
    client = TerminologyFakeClient(options={"hallucinate_id": True})
    result = _run(spec, client)
    assert result["status"] == "terminology_llm_unavailable"
    assert "outside window" in result["error"] or "duplicate judgments" in result["error"]
    assert not (_work(spec) / "concept_glossary.json").exists()
    progress = json.loads((_work(spec) / "term_mining_progress.json").read_text(encoding="utf-8"))
    # Judgment aborts at the first failing window; later windows stay pending.
    assert progress["windows"]["0"] == "termhood_error"
    assert set(progress["windows"].values()) <= {"pending", "termhood_error"}
    usage = list(read_jsonl(_work(spec) / "terminology_v2_usage.jsonl"))
    assert usage and usage[-1]["status"] == "error" and usage[-1]["role"] == "term_miner"


# ----------------------------------------------------- 4. polysemy contract
def test_polysemy_is_one_concept_with_n_senses(tmp_path):
    spec = make_terminology_spec(tmp_path)
    client = TerminologyFakeClient(options={"polysemy_surfaces": {"agrarian question": 2}})
    result = _run(spec, client)
    # Gray zone: a polysemous concept is listed for the human gate (§11.6 #2).
    assert result["status"] == "needs_human_review"
    concepts = _glossary_concepts(spec)
    matches = [c for c in concepts if c["canonical_source"] == "agrarian question"]
    assert len(matches) == 1  # never two concepts for one surface
    concept = matches[0]
    assert len(concept["senses"]) == 2
    assert [sense["sense_id"] for sense in concept["senses"]] == ["default", "sense-2"]
    assert concept["concept_id"] == concept_id_for("agrarian question")
    metrics = json.loads((_work(spec) / "terminology_metrics.json").read_text(encoding="utf-8"))
    assert metrics["consolidation"]["polysemous_concept_count"] == 1
    assert metrics["consolidation"]["sense_count_per_polysemous_concept"] == [2]
    assert metrics["consolidation"]["duplicate_concept_merge_error"] == 0
    template = list(read_jsonl(_work(spec) / "concept_decisions.jsonl"))
    assert any(row["concept_id"] == concept["concept_id"] and row["rank"] == 2 for row in template)


def test_duplicate_concepts_same_canonical_merge_not_fatal(tmp_path):
    """Same canonical under different draft ids merges (identity = canonical),
    so an LLM duplicating a concept is not fatal; the post-rewrite duplicate
    guard remains unit-tested against evaluate.duplicate_concept_errors."""
    spec = make_terminology_spec(tmp_path / "dup")
    client = TerminologyFakeClient(options={"duplicate_concepts": True})
    result = _run(spec, client)
    assert result["status"] == "terminology_v2_ok"
    metrics = result.get("metrics") or {}
    assert metrics.get("duplicate_concept_merge_error", 0) == 0
    concepts = _glossary_concepts(spec)
    # fixture duplicates every concept (same canonical, different draft ids):
    # the compiler must merge duplicates -> one concept per unique canonical
    norms = {_normal(c["canonical_source"]) for c in concepts}
    assert len(concepts) == len(norms) and len(concepts) >= 2
    # Metric function detects the error on the offending concept list.
    duplicate = ConceptRecord(
        concept_id="concept-" + "c" * 12, canonical_source="dupe", surface_forms=["dupe"],
        category="concept", senses=[Sense(sense_id="default", translation="译", confidence=0.9)])
    duplicate_two = ConceptRecord(
        concept_id="concept-" + "d" * 12, canonical_source="dupe", surface_forms=["dupe"],
        category="concept", senses=[Sense(sense_id="default", translation="译2", confidence=0.9)])
    assert len(duplicate_concept_errors([duplicate, duplicate_two])) == 1
    assert polysemy_check([duplicate, duplicate_two])["duplicate_concept_merge_error"] == 1


# ------------------------------------------------------- 5. constraint clamp
def test_constraint_clamp_hard_only_for_proper_or_master(tmp_path):
    spec = make_terminology_spec(tmp_path)
    client = TerminologyFakeClient(options={
        "person_surfaces": ["Vera Zasulich"],
        "sense_constraints": {
            "agrarian question": "hard",     # common noun -> clamp
            "Vera Zasulich": "hard",         # person, no master -> allowed (llm)
            "mode of production": "hard",    # master + evidence -> allowed (master)
        },
    })
    result = _run(spec, client)
    assert result["status"] == "terminology_v2_ok"
    concepts = {c["canonical_source"]: c for c in _glossary_concepts(spec)}
    assert concepts["agrarian question"]["senses"][0]["constraint"] == "preferred"
    assert concepts["agrarian question"]["constraint_origin"] == "deterministic"
    assert concepts["Vera Zasulich"]["senses"][0]["constraint"] == "hard"
    assert concepts["Vera Zasulich"]["constraint_origin"] == "llm"
    assert concepts["mode production"]["senses"][0]["constraint"] == "hard"
    assert concepts["mode production"]["constraint_origin"] == "master"
    metrics = json.loads((_work(spec) / "terminology_metrics.json").read_text(encoding="utf-8"))
    distribution = metrics["constraints"]["constraint_distribution"]
    assert distribution["hard"]["count"] == 2
    assert distribution["preferred"]["count"] >= 1
    assert metrics["constraints"]["constraint_origin_distribution"] == {
        "deterministic": 1, "llm": len(concepts) - 2, "master": 1}


def test_constraint_passthrough_and_single_sense_sense_constrained_clamp(tmp_path):
    spec = make_terminology_spec(tmp_path)
    client = TerminologyFakeClient(options={"sense_constraints": {
        "iron cage": "do_not_translate",
        "counter-revolutionary": "first_occurrence",
        "mode of production": "sense_constrained",  # one sense -> clamp to preferred
    }})
    result = _run(spec, client)
    assert result["status"] == "terminology_v2_ok"
    concepts = {c["canonical_source"]: c for c in _glossary_concepts(spec)}
    assert concepts["iron cage"]["senses"][0]["constraint"] == "do_not_translate"
    assert concepts["counter-revolutionary"]["senses"][0]["constraint"] == "first_occurrence"
    assert concepts["mode production"]["senses"][0]["constraint"] == "preferred"
    assert concepts["mode production"]["constraint_origin"] == "deterministic"


# ------------------------------------------- 6. progress/resume + fail-closed
def test_progress_resume_and_fail_closed(tmp_path):
    spec = make_terminology_spec(tmp_path)
    failing = TerminologyFakeClient(options={"fail_on_window": 1})
    first = _run(spec, failing)
    assert first["status"] == "terminology_llm_unavailable"
    assert "provider down" in first["error"]
    work = _work(spec)
    progress = json.loads((work / "term_mining_progress.json").read_text(encoding="utf-8"))
    assert progress["windows"] == {"0": "termhood_ok", "1": "termhood_error"}
    assert progress["consolidator_state"] == "pending"
    assert (work / "term_candidates_v2.jsonl").is_file()
    assert not (work / "concept_glossary.json").exists()

    healthy = TerminologyFakeClient()
    second = _run(spec, healthy)
    assert second["status"] == "terminology_v2_ok"
    usage = list(read_jsonl(work / "terminology_v2_usage.jsonl"))
    # Window 0 judged once (first run), window 1 resumed (second run),
    # consolidation once. No re-mining of completed windows.
    termhood_ok_rows = [row for row in usage if row["role"] == "term_miner" and row["status"] == "ok"]
    assert len(termhood_ok_rows) == 2
    consolidation_rows = [row for row in usage if row["role"] == "term_consolidator"]
    assert len(consolidation_rows) == 1
    assert healthy.calls and sum(
        1 for call in healthy.calls if "term_miner" in call["system"]) == 1
    rows = list(read_jsonl(work / "termhood_decisions.jsonl"))
    assert {str(row["window_index"]) for row in rows} == {"0", "1"}

    # A third full run is byte-idempotent: zero new LLM calls.
    usage_count = len(usage)
    idle = TerminologyFakeClient()
    third = _run(spec, idle)
    assert third["status"] == "terminology_v2_ok"
    assert len(list(read_jsonl(work / "terminology_v2_usage.jsonl"))) == usage_count
    assert idle.calls == []


# -------------------------------------------------------------- 7. wire caps
def test_wire_budget_fail_closed(tmp_path):
    spec = make_terminology_spec(tmp_path, wire_budget={"term_miner": 1, "term_consolidator": 4})
    client = TerminologyFakeClient()
    result = _run(spec, client)
    assert result["status"] == "terminology_llm_unavailable"
    assert "wire request budget exceeded" in result["error"]
    assert result["wire"]["term_miner"]["budget"] == 1
    assert result["wire"]["term_miner"]["used"] == 2
    assert not (_work(spec) / "concept_glossary.json").exists()


def test_wire_attempts_are_counted_per_http_post(tmp_path):
    spec = make_terminology_spec(tmp_path / "wire")
    client = TerminologyFakeClient(options={"wire_per_call": 30})
    result = _run(spec, client)
    assert result["status"] == "terminology_llm_unavailable"
    assert "wire request budget exceeded" in result["error"]
    assert result["wire"]["term_miner"]["used"] == 30


# ------------------------------------------------------- 8. offline zero-LLM
def test_offline_run_makes_zero_llm_calls(tmp_path):
    spec = make_terminology_spec(tmp_path)

    class ExplodingClient:
        wire_attempts = 0

        def complete_json(self, *args, **kwargs):
            raise AssertionError("offline run must not call the LLM client")

    result = _run(spec, ExplodingClient(), offline=True)
    assert result["status"] == "terminology_v2_ok"
    assert result["offline"] is True
    work = _work(spec)
    candidates = list(read_jsonl(work / "term_candidates_v2.jsonl"))
    assert len(candidates) > 10
    metrics = json.loads((work / "terminology_metrics.json").read_text(encoding="utf-8"))
    assert metrics["offline"] is True
    assert metrics["termhood"]["judged_count"] == 0
    assert metrics["constraints"]["constraint_distribution"] == {}
    assert metrics["known_term_recall"]["recall_at_mining"] == 1.0
    assert metrics["known_term_recall"]["recall_after_termhood"] is None
    assert metrics["wire"]["term_miner"]["used"] == 0
    assert not (work / "concept_glossary.json").exists()
    assert not (work / "concept_decisions.jsonl").exists()
    assert not (work / "terminology_v2_usage.jsonl").exists()
    progress = json.loads((work / "term_mining_progress.json").read_text(encoding="utf-8"))
    assert progress["consolidator_state"] == "skipped_offline"


# --------------------------------------------------- 9. master provenance
def test_master_provenance_retention(tmp_path):
    spec = make_terminology_spec(tmp_path)
    result = _run(spec)
    assert result["status"] == "terminology_v2_ok"
    concepts = {c["canonical_source"]: c for c in _glossary_concepts(spec)}
    concept = concepts["mode production"]
    assert concept["master_source"] == "生产方式"
    assert "production mode" in concept["master_aliases"]
    assert "mode of production" in concept["master_aliases"]
    assert concept["master_sha256"] == hashlib.sha256(
        Path(spec["mining"]["master_json"]).read_bytes()).hexdigest()
    manifest = json.loads((_work(spec) / "concept_glossary_manifest.json").read_text(encoding="utf-8"))
    assert manifest["master_sha256"] == concept["master_sha256"]
    assert manifest["concept_glossary_sha256"] and manifest["cleaned_segments_sha256"]
    assert manifest["counts"]["concept_count"] > 0
    metrics = json.loads((_work(spec) / "terminology_metrics.json").read_text(encoding="utf-8"))
    assert metrics["master_provenance"]["master_provenance_rate"] > 0
    assert metrics["master_provenance"]["master_sha256"] == concept["master_sha256"]


def test_master_conflict_concept_goes_to_human_gate(tmp_path):
    spec = make_terminology_spec(tmp_path)
    client = TerminologyFakeClient(options={"master_conflict_surfaces": ["mode of production"]})
    result = _run(spec, client)
    assert result["status"] == "needs_human_review"
    template = list(read_jsonl(_work(spec) / "concept_decisions.jsonl"))
    assert any(row["rank"] == 3 and row["priority"] == "conflicting_master_glossary"
               for row in template)


# ----------------------------------------------------- 10. spec validation
def test_load_terminology_spec_validation(tmp_path):
    spec = make_terminology_spec(tmp_path)
    path = write_spec_file(spec, tmp_path)
    loaded = load_terminology_spec(path)
    assert loaded["book_id"] == BOOK_ID
    assert loaded["_spec_path"] == str(path)
    valid_raw = json.loads(path.read_text(encoding="utf-8"))

    def _write(raw):
        bad_path = tmp_path / "bad_spec.json"
        bad_path.write_text(json.dumps(raw), encoding="utf-8")
        return bad_path

    raw = dict(valid_raw)
    raw["invented_top_level_key"] = True
    with pytest.raises(ValueError, match="invalid terminology spec"):
        load_terminology_spec(_write(raw))

    raw = dict(valid_raw)
    raw["cleaned_segments"] = str(tmp_path / "does-not-exist.jsonl")
    with pytest.raises(FileNotFoundError):
        load_terminology_spec(_write(raw))

    raw = dict(valid_raw)
    raw["structure_json"] = str(tmp_path / "missing_structure.json")
    with pytest.raises(FileNotFoundError):
        load_terminology_spec(_write(raw))

    raw = dict(valid_raw)
    raw["mining"] = dict(raw["mining"])
    raw["mining"]["master_json"] = str(tmp_path / "missing_master.json")
    with pytest.raises(FileNotFoundError):
        load_terminology_spec(_write(raw))


# ------------------------------------------------------------ 11. CLI offline
def test_cli_terminology_v2_offline(tmp_path, capsys):
    from book_pipeline.cli import main

    spec = make_terminology_spec(tmp_path)
    path = write_spec_file(spec, tmp_path)
    main(["terminology-v2", "--spec", str(path), "--offline"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["status"] == "terminology_v2_ok"
    assert payload["offline"] is True
    assert (tmp_path / "work" / "term_candidates_v2.jsonl").is_file()
    assert (tmp_path / "work" / "terminology_metrics.json").is_file()


# ------------------------------------------------------- determinism guard
def test_rerun_is_byte_idempotent_for_candidates_and_glossary(tmp_path):
    spec = make_terminology_spec(tmp_path)
    assert _run(spec)["status"] == "terminology_v2_ok"
    work = _work(spec)
    candidates_once = (work / "term_candidates_v2.jsonl").read_bytes()
    glossary_once = (work / "concept_glossary.json").read_bytes()
    decisions_once = (work / "termhood_decisions.jsonl").read_bytes()
    metrics_once = json.loads((work / "terminology_metrics.json").read_text(encoding="utf-8"))
    assert _run(spec, TerminologyFakeClient())["status"] == "terminology_v2_ok"
    assert (work / "term_candidates_v2.jsonl").read_bytes() == candidates_once
    assert (work / "concept_glossary.json").read_bytes() == glossary_once
    assert (work / "termhood_decisions.jsonl").read_bytes() == decisions_once
    metrics_twice = json.loads((work / "terminology_metrics.json").read_text(encoding="utf-8"))
    for key in ("mining", "termhood", "consolidation", "known_term_recall"):
        assert metrics_once[key] == metrics_twice[key]


def test_select_queue_protects_concept_sources_under_cap():
    """Reviewer P05-RECALL-001 regression: a tight queue cap must keep
    protected sources (master/seed/heading/repeated_phrase) and only shed
    lower-tier noise; known concept terms must not be dropped by the cap."""
    from book_pipeline.terminology.miner import TermCandidate, select_queue
    from book_pipeline.terminology.miner import QUEUE_PROTECTED_SOURCES

    def cand(surface, source_type, score=10.0, occurrences=5):
        return TermCandidate(
            candidate_id="term-" + hashlib.sha256(surface.encode()).hexdigest()[:12],
            surface=surface, source_type=source_type, occurrences=occurrences,
            evidence_segment_ids=["seg-1"], score=score)

    candidates = [
        cand("party congress", "repeated_phrase", score=20.0),
        cand("Book Title Term", "book_title_term", score=100.0),
        cand("ordinary quoted thing", "quoted_phrase", score=5.0),
        cand("Some Proper Name Here", "proper_name", score=4.0),
        cand("verydistinctivelongwordcandidate", "distinctive_word", score=2.0),
    ]
    # limit=2: exactly the two protected concept sources survive; all noise shed
    queue = select_queue(candidates, limit=2)
    surfaces = {c.surface for c in queue}
    assert "party congress" in surfaces and "Book Title Term" in surfaces
    assert len(queue) == 2
    for noise in ("ordinary quoted thing", "Some Proper Name Here",
                  "verydistinctivelongwordcandidate"):
        assert noise not in surfaces
    # limit=3: protected + one more by tier/score order (distinctive_word tier 2
    # sorts before quoted/proper) — the cap sheds lowest-priority noise only
    queue3 = select_queue(candidates, limit=3)
    assert len(queue3) == 3
    assert "verydistinctivelongwordcandidate" in {c.surface for c in queue3}
    # default unlimited queue returns everything
    assert len(select_queue(candidates, limit=0)) == len(candidates)


def test_known_term_recall_stages_are_honest():
    """Reviewer P05-RECALL-001/002 regression: staged recall with honest
    labels — queued-not-judged is never 'rejected_by_termhood'; a term
    dropped by a queue cap is labeled dropped_by_queue_cap; per-term rows
    carry candidate surface + evidence segment for auditability."""
    from book_pipeline.terminology.miner import TermCandidate
    from book_pipeline.terminology import evaluate

    def cand(surface, source_type="repeated_phrase"):
        return TermCandidate(
            candidate_id="term-" + hashlib.sha256(surface.encode()).hexdigest()[:12],
            surface=surface, source_type=source_type, occurrences=4,
            evidence_segment_ids=["seg-1"], score=10.0)

    full = [cand("alpha concept"), cand("beta phrase"), cand("gamma idea")]
    queue_ids = [full[0].candidate_id, full[2].candidate_id]   # beta dropped by cap
    known = [{"surface": "alpha concept"}, {"surface": "beta phrase"},
             {"surface": "gamma idea"}, {"surface": "delta absent"}]
    # offline: no termhood judgments -> retention stage None
    rec = evaluate.known_term_recall(full, None, known, queue_ids=queue_ids)
    assert rec["recall_at_mining"] == 0.75
    assert rec["recall_at_queue"] == 0.5
    assert rec["recall_after_termhood"] is None
    by = {r["surface"]: r for r in rec["per_term"]}
    assert by["alpha concept"]["loss_stage"] == "queued_not_judged_yet"
    assert by["beta phrase"]["loss_stage"] == "dropped_by_queue_cap"
    assert by["gamma idea"]["loss_stage"] == "queued_not_judged_yet"
    assert by["alpha concept"]["first_evidence_segment_id"] == "seg-1"
    assert by["delta absent"]["loss_stage"] == "not_mined"
    # judged: alpha accepted, gamma rejected -> honest labels
    rec2 = evaluate.known_term_recall(
        full, [full[0].candidate_id], known[:3], queue_ids=queue_ids)
    assert rec2["recall_after_termhood"] == 0.3333
    by2 = {r["surface"]: r for r in rec2["per_term"]}
    assert by2["alpha concept"]["loss_stage"] == "retained"
    assert by2["gamma idea"]["loss_stage"] == "rejected_by_termhood"


def test_compile_merges_duplicate_draft_with_polysemy_senses():
    """P07-pretest finding: deepseek-v4-flash emitted the same draft concept id
    twice (same canonical surface, distinct senses). Compiler must merge into
    one concept with renumbered senses — not hard-fail on the draft handle."""
    from book_pipeline.terminology.orchestrator import TerminologyResolver
    from book_pipeline.terminology.schemas import ConceptConsolidation, ConceptRecord, Sense

    resolver = TerminologyResolver.__new__(TerminologyResolver)  # bypass __init__
    resolver._master_index = {}
    resolver._master_path = None
    resolver._master_sha256 = None
    consolidation = ConceptConsolidation(concepts=[
        ConceptRecord(concept_id="concept-4a1b2c3d4e5f", canonical_source="civil society",
                      surface_forms=["civil society"], category="concept",
                      senses=[Sense(translation="市民社会", constraint="preferred",
                                    confidence=0.9, sense_id="default")]),
        ConceptRecord(concept_id="concept-4a1b2c3d4e5f", canonical_source="civil society",
                      surface_forms=["civil society", "civil-society"], category="concept",
                      senses=[Sense(translation="公民社会", constraint="preferred",
                                    confidence=0.8, sense_id="default")]),
    ])
    compiled = resolver._compile(consolidation)
    assert len(compiled) == 1
    c = compiled[0]
    assert _normal(c.canonical_source) == _normal("civil society")
    assert any(_normal(s) == _normal("civil-society") for s in c.surface_forms)
    assert len(c.senses) == 2
    assert c.senses[0].sense_id == "sense-1" and c.senses[1].sense_id == "sense-2"


def test_compile_keeps_different_canonicals_under_one_draft_id():
    """Live finding (deepseek-v4-flash reused one draft id for 'political system'
    and 'Impact on the Party System'): draft handles are NOT identity. Two
    distinct canonicals under one draft id must compile to two concepts;
    merges hitting the ambiguous id degrade to a recorded warning, never a
    guess or a crash."""
    from book_pipeline.terminology.orchestrator import TerminologyResolver
    from book_pipeline.terminology.schemas import ConceptConsolidation, ConceptRecord, Sense

    resolver = TerminologyResolver.__new__(TerminologyResolver)
    resolver._master_index = {}
    resolver._master_path = None
    resolver._master_sha256 = None
    consolidation = ConceptConsolidation(concepts=[
        ConceptRecord(concept_id="concept-4a1b2c3d4e5f", canonical_source="political system",
                      surface_forms=["political system"], category="concept",
                      senses=[Sense(sense_id="default", translation="政治体制", confidence=0.9)]),
        ConceptRecord(concept_id="concept-4a1b2c3d4e5f", canonical_source="Impact on the Party System",
                      surface_forms=["Impact on the Party System"], category="concept",
                      senses=[Sense(sense_id="default", translation="对党的体制的影响", confidence=0.9)]),
    ], merges=[{"concept_id": "concept-4a1b2c3d4e5f", "alias_surface": "the party system"}])
    compiled = resolver._compile(consolidation)
    assert len(compiled) == 2
    assert resolver._merge_ambiguity_warnings  # ambiguous merge recorded, not guessed
    canonicals = {_normal(c.canonical_source) for c in compiled}
    assert _normal("political system") in canonicals
    assert _normal("Impact on the Party System") in canonicals
    # distinct canonicals -> distinct deterministic ids
    ids = {c.concept_id for c in compiled}
    assert len(ids) == 2
