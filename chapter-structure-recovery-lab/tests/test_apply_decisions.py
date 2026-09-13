"""V2 apply-decisions tests (plan P03 section 4.1/4.6 + amendment A1).

Covers the strict JSONL intake, rescue must-exist re-evidence, the A1
deterministic candidate -> node mapping (rule 2 fallback, new-node creation,
occupied-block fail-closed guard), merge adjacency application, and full
``apply_decisions`` byte-determinism on the offline sample OCR fixture (plus
a Downloads-gated rerun on rethinking_chinese_politics evidence).
"""

import hashlib
import json
from pathlib import Path

import pytest

from chapter_recovery.apply_decisions import (
    apply_decisions,
    merge_target_candidate,
    read_decisions_jsonl,
    read_rescue_jsonl,
    rescue_re_evidence,
    validate_decision_v2,
    _apply_decisions_to_base,
    _map_candidate,
)
from chapter_recovery.decisions import DecisionValidationError, KINDS
from chapter_recovery.evidence_fusion import run_evidence_pipeline
from chapter_recovery.heading_text import canonical_heading_text
from chapter_recovery.models import Block, Node, stable_id
from chapter_recovery.page_map import PageMap

SHA = "a" * 64
FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


def err_codes(decision, candidates, page_count=100, allowed=None):
    return {
        item["code"]
        for item in validate_decision_v2(
            decision, candidates, page_count, allowed or set(KINDS), SHA
        )
    }


# ---------------------------------------------------------------------------
# Shared construction helpers
# ---------------------------------------------------------------------------


def candidate(cid, page=1, text="The Heading", blocks=("blk-2222222222222222",),
              bbox=(0.1, 0.2, 0.9, 0.3), zone="mainmatter",
              label="paragraph_title", numbering_data=None, page_print=None):
    return {
        "candidate_id": cid,
        "source_block_ids": list(blocks),
        "exact_text": text,
        "page_json": page,
        "page_print": page if page_print is None else page_print,
        "bbox": list(bbox),
        "zone": zone,
        "paddle_label": label,
        "paddle_relevel": None,
        "pdf_bookmark": None,
        "native_font": None,
        "numbering": numbering_data,
        "toc_matches": [],
        "repetition": {"count": 1, "nearby_count": 1},
        "before_context": None,
        "after_context": None,
        "deterministic_suppressions": [],
        "evidence_refs": (
            ["paddle:" + blocks[0] + ":" + label] if blocks else []
        ),
    }


def decision(cid, action="heading", kind=None, level=None, parent=None,
             confidence=0.98, **extra):
    record = {
        "schema_version": 2,
        "candidate_id": cid,
        "decision": action,
        "confidence": confidence,
        "evidence_refs": ["paddle:blk-2222222222222222:paragraph_title"],
        "decision_model": "test-model",
        "repair_round": 0,
        "prompts_version": 1,
    }
    if kind is not None:
        record["kind"] = kind
    if level is not None:
        record["level"] = level
    if parent is not None:
        record["parent_candidate_id"] = parent
    record.update(extra)
    return record


def base_node(node_id, kind="section", level=3, title="Heading", page=1,
              parent="book", block=None, ordinal=None, numbering_path=(),
              zone="mainmatter", bbox=(0.1, 0.1, 0.9, 0.2), ignored=False,
              source_block_ids=None, title_source=None):
    if block is None:
        block = "blk-" + hashlib.sha1(node_id.encode("utf-8")).hexdigest()[:16]
    return Node(
        id=node_id, kind=kind, level=level,
        title_source=title if title_source is None else title_source,
        page_json=page, page_print=page, parent_id=parent, block_id=block,
        source="paragraph_title", confidence=1.0,
        evidence=["paragraph_title"], bbox=bbox, zone=zone, ordinal=ordinal,
        numbering_path=list(numbering_path),
        source_block_ids=list(source_block_ids or [block]),
        ignored=ignored,
    )


def book_node(book_id="book"):
    return Node(
        id=book_id, kind="book", level=1, title_source="Sample Book",
        page_json=None, page_print=None, parent_id=None, block_id=None,
        source="config", confidence=1.0, evidence=["config"],
    )


def simple_tree():
    return [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                  page=1, block="blk-chapter"),
        base_node("section", kind="section", level=3, title="A section",
                  page=2, parent="chapter", block="blk-section"),
    ]


def candidates_index(*records):
    return {record["candidate_id"]: record for record in records}


# ---------------------------------------------------------------------------
# read_decisions_jsonl / read_rescue_jsonl
# ---------------------------------------------------------------------------


def test_read_decisions_jsonl_parses_meta_first(tmp_path):
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({
            "_meta": {
                "schema_version": 2, "book_id": "book-one",
                "evidence_sha256": SHA, "decision_model": "m",
                "prompts_version": 1, "generated_by": "test",
                "repair_total": 0,
            }
        })
        + "\n"
        + json.dumps(decision("cand-1111111111111111"))
        + "\n",
        encoding="utf-8",
    )
    packet = read_decisions_jsonl(path)
    assert packet["meta"]["book_id"] == "book-one"
    assert packet["meta"]["evidence_sha256"] == SHA
    assert len(packet["decisions"]) == 1
    assert packet["decisions"][0]["candidate_id"] == "cand-1111111111111111"


def test_read_decisions_jsonl_rejects_malformed_files(tmp_path):
    def write(lines):
        path = tmp_path / "decisions.jsonl"
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n",
                        encoding="utf-8")
        return path

    with pytest.raises(DecisionValidationError, match="missing file"):
        read_decisions_jsonl(tmp_path / "missing.jsonl")
    with pytest.raises(DecisionValidationError, match="_meta"):
        read_decisions_jsonl(write([decision("cand-1111111111111111")]))
    with pytest.raises(DecisionValidationError, match="schema_version"):
        read_decisions_jsonl(write([
            {"_meta": {"schema_version": 3, "book_id": "b",
                       "evidence_sha256": SHA, "prompts_version": 1}}
        ]))
    with pytest.raises(DecisionValidationError, match="unknown _meta"):
        read_decisions_jsonl(write([
            {"_meta": {"schema_version": 2, "book_id": "b",
                       "evidence_sha256": SHA, "prompts_version": 1,
                       "surprise": 1}}
        ]))
    with pytest.raises(DecisionValidationError, match="64-hex"):
        read_decisions_jsonl(write([
            {"_meta": {"schema_version": 2, "book_id": "b",
                       "evidence_sha256": "not-a-sha", "prompts_version": 1}}
        ]))
    with pytest.raises(DecisionValidationError, match="first line only"):
        read_decisions_jsonl(write([
            {"_meta": {"schema_version": 2, "book_id": "b",
                       "evidence_sha256": SHA, "prompts_version": 1}},
            {"_meta": {"schema_version": 2, "book_id": "b",
                       "evidence_sha256": SHA, "prompts_version": 1}},
        ]))


def test_read_rescue_jsonl_rejects_malformed_records(tmp_path):
    def write(line):
        path = tmp_path / "rescue.jsonl"
        path.write_text(json.dumps(line) + "\n", encoding="utf-8")
        return path

    good = {
        "block_id": "blk-1111111111111111", "reason": "missed_heading",
        "confidence": 0.7, "evidence_refs": ["paddle:blk-x:label"],
        "rescue_model": "m", "prompts_version": 1,
    }
    assert read_rescue_jsonl(write(good))[0]["block_id"].startswith("blk-")
    with pytest.raises(DecisionValidationError, match="unknown rescue keys"):
        read_rescue_jsonl(write({**good, "extra": 1}))
    with pytest.raises(DecisionValidationError, match="block_id"):
        read_rescue_jsonl(write({**good, "block_id": "node-x"}))
    with pytest.raises(DecisionValidationError, match="confidence"):
        read_rescue_jsonl(write({**good, "confidence": 1.5}))
    with pytest.raises(DecisionValidationError, match="reason"):
        read_rescue_jsonl(write({**good, "reason": "  "}))
    with pytest.raises(DecisionValidationError, match="evidence_refs"):
        read_rescue_jsonl(write({**good, "evidence_refs": []}))
    with pytest.raises(DecisionValidationError, match="rescue_model"):
        read_rescue_jsonl(write({**good, "rescue_model": ""}))


# ---------------------------------------------------------------------------
# validate_decision_v2 strict intake
# ---------------------------------------------------------------------------


def test_intake_rejects_missing_fields_and_wrong_schema():
    candidates = candidates_index(candidate("cand-1111111111111111"))
    codes = err_codes({"schema_version": 2, "candidate_id": "cand-1111111111111111"},
                      candidates)
    assert "missing_field" in codes
    assert "bad_decision" in codes
    for bad in (1, 3, 2.0, True, None):
        codes = err_codes(decision("cand-1111111111111111", schema_version=bad),
                          candidates)
        assert "schema_version_mismatch" in codes


def test_intake_rejects_unknown_and_forbidden_text_keys():
    candidates = candidates_index(candidate("cand-1111111111111111"))
    assert "unknown_top_level_key" in err_codes(
        decision("cand-1111111111111111", garbage=1), candidates)
    for key in ("exact_text", "title_source", "title", "source_text",
                "new_title", "text"):
        assert "forbidden_text_key" in err_codes(
            decision("cand-1111111111111111", **{key: "invented title"}),
            candidates)


def test_intake_rejects_hallucinated_ids():
    candidates = candidates_index(candidate("cand-1111111111111111"))
    assert "bad_candidate_id" in err_codes(
        decision("cand-not-hex"), candidates)
    assert "unknown_candidate_id" in err_codes(
        decision("cand-ffffffffffffffff"), candidates)
    assert "bad_candidate_id" in err_codes(
        decision(12345), candidates)
    assert "unknown_parent_candidate_id" in err_codes(
        decision("cand-1111111111111111", parent="cand-ffffffffffffffff"),
        candidates)
    assert "self_parent" in err_codes(
        decision("cand-1111111111111111", parent="cand-1111111111111111"),
        candidates)
    assert "bad_parent_id" in err_codes(
        decision("cand-1111111111111111", parent="node-x"), candidates)


def test_intake_rejects_bad_decisions_kinds_and_levels():
    candidates = candidates_index(candidate("cand-1111111111111111"))
    assert "bad_decision" in err_codes(
        decision("cand-1111111111111111", action="paragraph"), candidates)
    assert "bad_decision" in err_codes(
        decision("cand-1111111111111111", action=5), candidates)
    assert "bad_kind" in err_codes(
        decision("cand-1111111111111111", kind="subsection"), candidates)
    assert "bad_kind" in err_codes(
        decision("cand-1111111111111111", kind=None), candidates)
    for bad_level in (0, 7, 2.0, True, None):
        assert "bad_level" in err_codes(
            decision("cand-1111111111111111", level=bad_level), candidates)
    # kind/level only belong to heading decisions.
    assert "unexpected_kind" in err_codes(
        decision("cand-1111111111111111", action="body", kind="section"),
        candidates)
    assert "unexpected_level" in err_codes(
        decision("cand-1111111111111111", action="caption", level=2),
        candidates)
    assert "unexpected_parent" in err_codes(
        decision("cand-1111111111111111", action="body",
                 parent="cand-1111111111111111"),
        candidates)


def test_intake_rejects_bad_confidence_evidence_and_rounds():
    candidates = candidates_index(candidate("cand-1111111111111111"))
    for bad in (-0.1, 1.2, "0.9", True, None):
        assert "bad_confidence" in err_codes(
            decision("cand-1111111111111111", confidence=bad), candidates)
    for bad_evidence in (None, [], ["ok", 3]):
        assert "empty_evidence_refs" in err_codes(
            decision("cand-1111111111111111", evidence_refs=bad_evidence),
            candidates)
    for bad_round in (-1, 3, "0", True):
        assert "bad_repair_round" in err_codes(
            decision("cand-1111111111111111", repair_round=bad_round),
            candidates)
    assert "bad_prompts_version" in err_codes(
        decision("cand-1111111111111111", prompts_version="1"), candidates)
    assert "stale_evidence_sha256" in err_codes(
        decision("cand-1111111111111111", evidence_sha256="b" * 64),
        candidates)
    out_of_range = candidate("cand-1111111111111111", page=500)
    assert "page_out_of_range" in err_codes(
        decision("cand-1111111111111111"),
        candidates_index(out_of_range), page_count=100)


def test_intake_accepts_valid_records_and_null_optional_fields():
    candidates = candidates_index(
        candidate("cand-1111111111111111"),
        candidate("cand-2222222222222222"),
    )
    heading = decision("cand-1111111111111111", kind="chapter", level=2,
                       parent="cand-2222222222222222")
    assert validate_decision_v2(heading, candidates, 100, set(KINDS), SHA) == []
    body = decision("cand-1111111111111111", action="body")
    assert validate_decision_v2(body, candidates, 100, set(KINDS), SHA) == []
    body_null = decision("cand-1111111111111111", action="apparatus",
                         kind=None, level=None, parent=None)
    assert validate_decision_v2(body_null, candidates, 100, set(KINDS), SHA) == []
    caption = decision("cand-1111111111111111", action="caption")
    assert validate_decision_v2(caption, candidates, 100, set(KINDS), SHA) == []


def test_intake_merge_adjacency_rules():
    page = 5
    top = candidate("cand-1111111111111111", page=page, text="Left part",
                    blocks=("blk-1111111111111111",), bbox=(0.1, 0.05, 0.9, 0.1))
    middle = candidate("cand-2222222222222222", page=page, text="Middle part",
                       blocks=("blk-2222222222222222",), bbox=(0.1, 0.3, 0.9, 0.35))
    bottom = candidate("cand-3333333333333333", page=page, text="Right part",
                       blocks=("blk-3333333333333333",), bbox=(0.1, 0.8, 0.9, 0.85))
    other_page = candidate("cand-4444444444444444", page=8, text="Elsewhere",
                           blocks=("blk-4444444444444444",), bbox=(0.1, 0.1, 0.9, 0.15))
    index = candidates_index(top, middle, bottom, other_page)

    # bottom is adjacent to middle: explicit parent and auto both accepted.
    assert validate_decision_v2(
        decision("cand-3333333333333333", action="merge",
                 parent="cand-2222222222222222"),
        index, 100, set(KINDS), SHA,
    ) == []
    assert validate_decision_v2(
        decision("cand-3333333333333333", action="merge"), index, 100,
        set(KINDS), SHA,
    ) == []
    assert merge_target_candidate(
        decision("cand-3333333333333333", action="merge"), index
    )["candidate_id"] == "cand-2222222222222222"
    # bottom is NOT adjacent to top (middle sits between them).
    assert "merge_not_adjacent" in err_codes(
        decision("cand-3333333333333333", action="merge",
                 parent="cand-1111111111111111"),
        index)
    # a target on another page is not adjacent either.
    assert "merge_not_adjacent" in err_codes(
        decision("cand-3333333333333333", action="merge",
                 parent="cand-4444444444444444"),
        index)
    # merges need a block-backed candidate.
    no_blocks = candidate("cand-5555555555555555", page=page, text="Font only",
                          blocks=())
    assert "merge_without_blocks" in err_codes(
        decision("cand-5555555555555555", action="merge"), candidates_index(no_blocks))


# ---------------------------------------------------------------------------
# rescue must-exist + re-evidence
# ---------------------------------------------------------------------------


def sample_pages():
    first = Block(page_index=0, block_index=0, raw_block_id=None,
                  label="paragraph_title", content="The Real Chapter",
                  bbox=(0.0, 0.0, 1.0, 0.1), order=0, page_width=1.0,
                  page_height=1.0)
    body = Block(page_index=0, block_index=1, raw_block_id=None,
                 label="text", content="Some body sentence follows here.",
                 bbox=(0.0, 0.2, 1.0, 0.3), order=1, page_width=1.0,
                 page_height=1.0)
    return [[first, body]]


def test_rescue_re_evidence_rejects_unknown_block_ids():
    pages = sample_pages()
    page_map = PageMap(0, 1.0, 1, 1, {})
    with pytest.raises(DecisionValidationError, match="unknown"):
        rescue_re_evidence(["blk-ffffffffffffffff"], pages, page_map,
                           {"book_id": "test-book"})
    with pytest.raises(DecisionValidationError, match="malformed"):
        rescue_re_evidence(["blk-nope"], pages, page_map, {"book_id": "test-book"})


def test_rescue_re_evidence_synthesizes_candidate_from_block():
    pages = sample_pages()
    block = pages[0][0]
    page_map = PageMap(0, 1.0, 1, 1, {})
    records = rescue_re_evidence([block.id], pages, page_map,
                                 {"book_id": "test-book"})
    assert len(records) == 1
    record = records[0]
    assert record["source_block_ids"] == [block.id]
    assert record["exact_text"] == "The Real Chapter"
    assert record["page_json"] == 0
    assert record["page_print"] == 0
    assert record["paddle_label"] == "paragraph_title"
    assert record["zone"] == "mainmatter"
    expected_id = "cand-" + stable_id(
        "test-book", 0, "\x1f".join(sorted([block.id])), "The Real Chapter"
    )
    assert record["candidate_id"] == expected_id
    assert record["evidence_refs"] == [f"paddle:{block.id}:paragraph_title"]
    assert record["numbering"] is None
    assert record["bbox"] == [0.0, 0.0, 1.0, 0.1]
    # duplicate ids collapse to one candidate
    assert len(rescue_re_evidence([block.id, block.id], pages, page_map,
                                  {"book_id": "test-book"})) == 1
    # print page follows the page map offset
    shifted = rescue_re_evidence([block.id], pages, PageMap(2, 1.0, 1, 1, {}),
                                 {"book_id": "test-book"})
    assert shifted[0]["page_print"] == 2


# ---------------------------------------------------------------------------
# A1 mapping + engine application
# ---------------------------------------------------------------------------


def test_a1_rule2_page_title_mapping_when_blocks_do_not_overlap():
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Introduction",
                  page=5, block="blk-chapter"),
    ]
    record = candidate("cand-1111111111111111", page=5, text="INTRODUCTION",
                       blocks=("blk-other000000000000",), bbox=(0.1, 0.1, 0.9, 0.2))
    assert _map_candidate(record, nodes).id == "chapter"
    applied = _apply_decisions_to_base(
        list(nodes),
        candidates_index(record),
        [decision("cand-1111111111111111", kind="chapter", level=2)],
        PageMap(0, 1.0, 1, 1, {}),
        "book-one", "book",
    )
    assert len(applied["nodes"]) == 2  # mapped, not duplicated
    chapter = next(node for node in applied["nodes"] if node.id == "chapter")
    assert (chapter.kind, chapter.level) == ("chapter", 2)
    assert applied["touched"]["chapter"]


def test_a1_rule3_numbering_token_mapping():
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="1.2 The Part",
                  page=5, block="blk-chapter", ordinal="1.2",
                  numbering_path=["1", "2"]),
    ]
    record = candidate("cand-1111111111111111", page=5, text="Totally different",
                       blocks=("blk-other000000000000",),
                       numbering_data={"token": "1.2", "path": ["1", "2"],
                                       "depth": 2, "family": "decimal"})
    assert _map_candidate(record, nodes).id == "chapter"


def test_a1_rule1_block_overlap_takes_priority():
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                  page=5, block="blk-shared"),
        base_node("other", kind="chapter", level=2, title="1.2 Other",
                  page=5, block="blk-other", ordinal="1.2",
                  numbering_path=["1", "2"]),
    ]
    record = candidate("cand-1111111111111111", page=5, text="1.2 Other",
                       blocks=("blk-shared",),
                       numbering_data={"token": "1.2", "path": ["1", "2"],
                                       "depth": 2, "family": "decimal"})
    assert _map_candidate(record, nodes).id == "chapter"


def test_a1_new_node_creation_for_free_candidate():
    book = book_node()
    chapter = base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                        page=1, block="blk-chapter")
    nodes = [book, chapter]
    record = candidate("cand-1111111111111111", page=9, text="New Section",
                       blocks=("blk-free0000000000000",), bbox=(0.1, 0.1, 0.9, 0.2))
    applied = _apply_decisions_to_base(
        list(nodes),
        candidates_index(record),
        [decision("cand-1111111111111111", kind="section", level=3)],
        PageMap(0, 1.0, 1, 1, {}),
        "book-one", "book",
    )
    created = applied["nodes"][-1]
    assert created.id.startswith("node-")
    assert (created.kind, created.level) == ("section", 3)
    assert created.title_source == "New Section"
    assert created.parent_id == "chapter"  # nearest preceding container
    assert created.block_id == "blk-free0000000000000"
    assert created.source_block_ids == ["blk-free0000000000000"]
    assert created.zone == "mainmatter"
    assert created.confidence == 0.98
    assert "v2_decision:heading" in created.evidence
    assert applied["touched"][created.id]


def test_a1_occupied_blocks_fail_closed(monkeypatch):
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                  page=1, block="blk-chapter"),
        base_node("section", kind="section", level=3, title="A section",
                  page=2, parent="chapter", block="blk-occupied"),
    ]
    record = candidate("cand-1111111111111111", page=9,
                       text="Text that matches no node",
                       blocks=("blk-occupied",))
    # Simulate a mapping divergence (e.g. a future rule change): the block is
    # occupied by an unrelated active node and no base node resolves.
    monkeypatch.setattr("chapter_recovery.apply_decisions._map_candidate",
                        lambda *args, **kwargs: None)
    with pytest.raises(DecisionValidationError, match="occupied"):
        _apply_decisions_to_base(
            list(nodes),
            candidates_index(record),
            [decision("cand-1111111111111111", kind="section", level=3)],
            PageMap(0, 1.0, 1, 1, {}),
            "book-one", "book",
        )


def test_heading_parent_mapping_across_decisions():
    nodes = [book_node()]
    part = candidate("cand-1111111111111111", page=1, text="Part One",
                     blocks=("blk-part0000000000000",))
    section = candidate("cand-2222222222222222", page=9, text="A deep section",
                        blocks=("blk-sec00000000000000",))
    applied = _apply_decisions_to_base(
        list(nodes),
        candidates_index(part, section),
        [
            decision("cand-1111111111111111", kind="part", level=2),
            decision("cand-2222222222222222", kind="section", level=3,
                     parent="cand-1111111111111111"),
        ],
        PageMap(0, 1.0, 1, 1, {}),
        "book-one", "book",
    )
    by_id = {node.id: node for node in applied["nodes"]}
    part_node = next(node for node in applied["nodes"] if node.kind == "part")
    section_node = next(node for node in applied["nodes"] if node.kind == "section")
    assert section_node.parent_id == part_node.id
    assert part_node.parent_id == "book"
    assert by_id[part_node.id].kind == "part"


def test_body_decision_demotes_mapped_node_and_reparents_children():
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                  page=1, block="blk-chapter"),
        base_node("bad", kind="section", level=3, title="Not a heading",
                  page=2, parent="chapter", block="blk-bad"),
        base_node("child", kind="section", level=4, title="Inner",
                  page=3, parent="bad", block="blk-child"),
    ]
    record = candidate("cand-1111111111111111", page=2, text="Not a heading",
                       blocks=("blk-bad",))
    applied = _apply_decisions_to_base(
        list(nodes),
        candidates_index(record),
        [decision("cand-1111111111111111", action="body")],
        PageMap(0, 1.0, 1, 1, {}),
        "book-one", "book",
    )
    bad = next(node for node in applied["nodes"] if node.id == "bad")
    child = next(node for node in applied["nodes"] if node.id == "child")
    assert bad.ignored is True
    assert child.parent_id == "chapter"
    assert "v2_decision:body" in bad.evidence


def test_merge_decision_folds_source_into_adjacent_target_node():
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                  page=1, block="blk-chapter"),
        base_node("target", kind="section", level=3, title="Left part",
                  page=5, parent="chapter", block="blk-1111111111111111"),
        base_node("elsewhere", kind="section", level=3, title="Elsewhere",
                  page=6, parent="chapter", block="blk-5555555555555555"),
    ]
    source = candidate("cand-3333333333333333", page=5, text="Right part",
                       blocks=("blk-2222222222222222",), bbox=(0.1, 0.6, 0.9, 0.65))
    # The decision carries no explicit parent: intake/apply resolve the
    # adjacent preceding candidate.  Build the candidate index so the target
    # candidate record matches the target base node by block.
    target = candidate("cand-1111111111111111", page=5, text="Left part",
                       blocks=("blk-1111111111111111",), bbox=(0.1, 0.1, 0.9, 0.15))
    applied = _apply_decisions_to_base(
        list(nodes),
        candidates_index(target, source),
        [decision("cand-3333333333333333", action="merge")],
        PageMap(0, 1.0, 1, 1, {}),
        "book-one", "book",
    )
    merged = next(node for node in applied["nodes"] if node.id == "target")
    assert merged.title_source == "Left part Right part"
    assert "blk-2222222222222222" in merged.source_block_ids
    assert "v2_decision:merge" in merged.evidence
    assert len(applied["nodes"]) == 4  # no new node from the merged source


def test_merge_with_explicit_target_concatenates_titles():
    nodes = [
        book_node(),
        base_node("chapter", kind="chapter", level=2, title="Chapter 1",
                  page=1, block="blk-chapter"),
        base_node("target", kind="section", level=3, title="Article title",
                  page=5, parent="chapter", block="blk-1111111111111111"),
    ]
    target = candidate("cand-1111111111111111", page=5, text="Article title",
                       blocks=("blk-1111111111111111",), bbox=(0.1, 0.1, 0.9, 0.2))
    source = candidate("cand-2222222222222222", page=5, text="continued here",
                       blocks=("blk-2222222222222222",), bbox=(0.1, 0.4, 0.9, 0.5))
    applied = _apply_decisions_to_base(
        list(nodes),
        candidates_index(target, source),
        [decision("cand-2222222222222222", action="merge",
                  parent="cand-1111111111111111")],
        PageMap(0, 1.0, 1, 1, {}),
        "book-one", "book",
    )
    merged = next(node for node in applied["nodes"] if node.id == "target")
    assert merged.title_source == "Article title continued here"
    assert merged.source_block_ids == ["blk-1111111111111111", "blk-2222222222222222"]
    assert applied["nodes"][-1].id == "target"


# ---------------------------------------------------------------------------
# Full apply_decisions determinism (offline sample fixture)
# ---------------------------------------------------------------------------


def sample_input_available():
    config = json.loads((FIXTURES / "sample_config.json").read_text(encoding="utf-8"))
    return (FIXTURES / config["input_json"]).is_file()


def _evidence_sha(structure_dir):
    digest = hashlib.sha256()
    digest.update((structure_dir / "evidence" / "heading_candidates.jsonl").read_bytes())
    digest.update((structure_dir / "evidence" / "book_structure_evidence.json").read_bytes())
    return digest.hexdigest()


def _write_fixture_inputs(structure_dir, book_id, rescue_block_id=None):
    candidates = [
        json.loads(line)
        for line in (structure_dir / "evidence" / "heading_candidates.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    block_backed = [c for c in candidates if c.get("source_block_ids")]
    selected = block_backed[:2] or candidates[:2]
    lines = [
        json.dumps({
            "_meta": {
                "schema_version": 2, "book_id": book_id,
                "evidence_sha256": _evidence_sha(structure_dir),
                "decision_model": "fixture-model", "grammar_model": "fixture-grammar",
                "prompts_version": 1,
                "generated_by": "chapter tests", "repair_total": 0,
            }
        })
    ]
    for index, record in enumerate(selected):
        lines.append(json.dumps({
            "schema_version": 2,
            "candidate_id": record["candidate_id"],
            "decision": "heading" if index == 0 else "body",
            "kind": "chapter" if index == 0 else None,
            "level": 2 if index == 0 else None,
            "parent_candidate_id": None,
            "confidence": 0.99,
            "evidence_refs": ["fixture:" + record["candidate_id"]],
            "decision_model": "fixture-model",
            "repair_round": 0,
            "prompts_version": 1,
        }))
    decisions_path = structure_dir / "v2-fixtures" / "structure_decisions.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rescue_path = structure_dir / "v2-fixtures" / "structure_rescue.jsonl"
    if rescue_block_id is not None:
        rescue_path.write_text(json.dumps({
            "block_id": rescue_block_id, "reason": "missed_heading",
            "confidence": 0.7,
            "evidence_refs": ["fixture:" + rescue_block_id],
            "rescue_model": "fixture-model", "prompts_version": 1,
        }) + "\n", encoding="utf-8")
    else:
        rescue_path.write_text("", encoding="utf-8")
    return decisions_path, rescue_path


def _apply_twice_and_compare(config_path, structure_dir, decisions_path,
                             rescue_path):
    first = apply_decisions(config_path, decisions_path, rescue_path,
                            output_dir=structure_dir)
    before = {
        name: (structure_dir / "v2" / name).read_bytes()
        for name in ("book_structure.json", "validation.json",
                     "hierarchy_audit.json", "decision_provenance.json")
    }
    second = apply_decisions(config_path, decisions_path, rescue_path,
                             output_dir=structure_dir)
    after = {
        name: (structure_dir / "v2" / name).read_bytes()
        for name in ("book_structure.json", "validation.json",
                     "hierarchy_audit.json", "decision_provenance.json")
    }
    assert first["decisions"] == second["decisions"]
    assert first["nodes"] == second["nodes"]
    return before, after, first


def test_apply_decisions_deterministic_on_sample_fixture(tmp_path):
    if not sample_input_available():
        pytest.skip("sample OCR fixture is unavailable")
    config_path = FIXTURES / "sample_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    structure_dir = tmp_path / "structure"
    run_evidence_pipeline(
        config,
        FIXTURES / config["input_json"],
        structure_dir,
        config_base=FIXTURES,
    )
    decisions_path, rescue_path = _write_fixture_inputs(
        structure_dir, config["book_id"])
    before, after, result = _apply_twice_and_compare(
        config_path, structure_dir, decisions_path, rescue_path)
    for name in before:
        assert before[name] == after[name], f"{name} is not byte-deterministic"
    tree = json.loads(after["book_structure.json"].decode("utf-8"))
    assert tree["schema_version"] == 3
    assert tree["book_id"] == config["book_id"]
    validation = json.loads(after["validation.json"].decode("utf-8"))
    assert "v2" in validation
    assert validation["v2"]["checks"]
    provenance = json.loads(after["decision_provenance.json"].decode("utf-8"))
    assert provenance["schema_version"] == 1
    assert len(provenance["nodes"]) == len(tree["nodes"])
    origins = {record["decision_origin"] for record in provenance["nodes"]}
    assert origins == {"llm", "deterministic"}
    assert result["output_dir"].endswith("structure/v2")
    audit = json.loads(after["hierarchy_audit.json"].decode("utf-8"))
    assert audit["heading_level_counts"] == validation["metrics"]["heading_level_counts"]


def test_apply_decisions_rejects_stale_and_hallucinated_inputs(tmp_path):
    if not sample_input_available():
        pytest.skip("sample OCR fixture is unavailable")
    config_path = FIXTURES / "sample_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    structure_dir = tmp_path / "structure"
    run_evidence_pipeline(
        config, FIXTURES / config["input_json"], structure_dir,
        config_base=FIXTURES,
    )
    decisions_path, rescue_path = _write_fixture_inputs(
        structure_dir, config["book_id"])
    lines = decisions_path.read_text(encoding="utf-8").splitlines()
    stale = lines[0].replace(_evidence_sha(structure_dir), "c" * 64)
    stale_path = tmp_path / "stale_decisions.jsonl"
    stale_path.write_text("\n".join([stale, *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(DecisionValidationError, match="stale"):
        apply_decisions(config_path, stale_path, rescue_path,
                        output_dir=structure_dir)

    candidates = [
        json.loads(line)
        for line in (structure_dir / "evidence" / "heading_candidates.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hallucinated_meta = json.loads(lines[0])
    fake_decision = {
        "schema_version": 2,
        "candidate_id": "cand-ffffffffffffffff",  # not in the live set
        "decision": "heading", "kind": "chapter", "level": 2,
        "parent_candidate_id": None, "confidence": 0.9,
        "evidence_refs": ["fixture:fake"], "decision_model": "m",
        "repair_round": 0, "prompts_version": 1,
    }
    bad_path = tmp_path / "hallucinated_decisions.jsonl"
    bad_path.write_text(
        json.dumps(hallucinated_meta) + "\n"
        + json.dumps(fake_decision) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DecisionValidationError, match="not in the live candidate"):
        apply_decisions(config_path, bad_path, rescue_path,
                        output_dir=structure_dir)
    assert len(candidates) > 0  # sanity: evidence was actually produced


# ---------------------------------------------------------------------------
# Downloads-gated rerun on rethinking_chinese_politics
# ---------------------------------------------------------------------------


RETHINKING_CONFIG = ROOT / "configs/rethinking_chinese_politics.json"
RETHINKING_RUN = ROOT / "runs/rethinking_chinese_politics"


def rethinking_available():
    if not RETHINKING_CONFIG.is_file():
        return False
    config = json.loads(RETHINKING_CONFIG.read_text(encoding="utf-8"))
    evidence_dir = RETHINKING_RUN / "evidence"
    return (
        Path(config["input_json"]).expanduser().is_file()
        and (evidence_dir / "heading_candidates.jsonl").is_file()
        and (evidence_dir / "book_structure_evidence.json").is_file()
    )


@pytest.mark.skipif(not rethinking_available(),
                    reason="local rethinking OCR corpus + P02 evidence unavailable")
def test_apply_decisions_byte_deterministic_on_rethinking(tmp_path):
    config_path = RETHINKING_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    structure_dir = tmp_path / "structure"
    # Copy the read-only evidence into the tmp structure dir so the run never
    # writes into runs/<book>.
    evidence_src = RETHINKING_RUN / "evidence"
    evidence_dst = structure_dir / "evidence"
    evidence_dst.mkdir(parents=True, exist_ok=True)
    for name in ("heading_candidates.jsonl", "book_structure_evidence.json",
                 "pdf_evidence.json"):
        (evidence_dst / name).write_bytes((evidence_src / name).read_bytes())
    decisions_path, rescue_path = _write_fixture_inputs(
        structure_dir, config["book_id"])
    before, after, result = _apply_twice_and_compare(
        config_path, structure_dir, decisions_path, rescue_path)
    for name in before:
        assert before[name] == after[name], f"{name} is not byte-deterministic"
    tree = json.loads(after["book_structure.json"].decode("utf-8"))
    assert tree["schema_version"] == 3
    assert tree["book_id"] == config["book_id"]
