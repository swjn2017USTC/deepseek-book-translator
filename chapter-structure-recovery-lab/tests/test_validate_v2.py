"""V2 gate validator tests (plan P03 section 4.4/4.6; spec section 9.6).

The five additive checks N1..N5 live in ``validate_v2.validate_structure_v2``
and run on top of the untouched legacy 15 checks.  These tests construct
minimal inline trees that pass every legacy check and then isolate exactly
one v2 failure per test, so gate_failures assert on the intended code.
"""

import hashlib

import pytest

from chapter_recovery.models import Node
from chapter_recovery.validate import validate_structure
from chapter_recovery.validate_v2 import V2_CHECK_CODES, validate_structure_v2

SHA = "a" * 64


def book_node(title="Sample Book"):
    return Node(
        id="book", kind="book", level=1, title_source=title,
        page_json=None, page_print=None, parent_id=None, block_id=None,
        source="config", confidence=1.0, evidence=["config"],
    )


def heading(node_id, title, page, kind="section", level=3, parent="book",
            block=None, ordinal=None,
            numbering_path=(), zone="mainmatter", ignored=False,
            source_block_ids=None):
    if block is None:
        block = "blk-" + hashlib.sha1(node_id.encode("utf-8")).hexdigest()[:16]
    return Node(
        id=node_id, kind=kind, level=level, title_source=title,
        page_json=page, page_print=page, parent_id=parent, block_id=block,
        source="paragraph_title", confidence=1.0,
        evidence=["paragraph_title"], bbox=(0.1, 0.1, 0.9, 0.2), zone=zone,
        ordinal=ordinal, numbering_path=list(numbering_path),
        source_block_ids=list(source_block_ids or [block]),
        ignored=ignored,
    )


def gate_codes(result):
    return {finding["code"] for finding in result["gate_failures"]}


def error_codes(result):
    return {finding["code"] for finding in result["errors"]}


def test_v2_clean_tree_passes_and_records_v2_checks():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 1, kind="chapter", level=2),
        heading("section", "Inner section", 2, parent="chapter", level=3),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"}, {"prompts_version": 1})
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["gate_failures"] == []
    assert result["v2"]["checks"] == V2_CHECK_CODES
    assert result["v2"]["decision_meta"] == {"prompts_version": 1}
    # Legacy 15-check contract preserved: metrics still reported.
    assert result["metrics"]["node_count"] == 3


def test_n1_parent_before_child_is_gate_failure():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 10, kind="chapter", level=2),
        heading("section", "Late section", 5, parent="chapter", level=3),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"parent_before_child"}
    assert error_codes(result) == {"parent_before_child"}
    finding = result["gate_failures"][0]
    assert finding["parent_id"] == "chapter"
    assert finding["parent_page_json"] == 10
    assert finding["page_json"] == 5


def test_n2_sibling_numbering_decrease_is_gate_failure():
    chapter = heading("chapter", "Chapter 1", 1, kind="chapter", level=2)
    nodes = [
        book_node(),
        chapter,
        heading("s-one", "1.7 First", 5, parent="chapter", level=3,
                ordinal="1.7", numbering_path=["1", "7"]),
        heading("s-two", "1.3 Second", 6, parent="chapter", level=3,
                ordinal="1.3", numbering_path=["1", "3"]),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"sibling_numbering_monotonic"}
    finding = result["gate_failures"][0]
    assert finding["numbering"] == ["1", "3"]
    assert finding["previous_numbering"] == ["1", "7"]
    assert finding["parent_id"] == chapter.id


def test_n2_sibling_numbering_increase_passes():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 1, kind="chapter", level=2),
        heading("s-one", "1.3 First", 5, parent="chapter", level=3,
                ordinal="1.3", numbering_path=["1", "3"]),
        heading("s-two", "1.7 Second", 6, parent="chapter", level=3,
                ordinal="1.7", numbering_path=["1", "7"]),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is True
    assert "sibling_numbering_monotonic" not in gate_codes(result)


def test_n2_partial_numbering_evidence_is_ignored():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 1, kind="chapter", level=2),
        heading("s-one", "1.7 Numbered", 5, parent="chapter", level=3,
                ordinal="1.7", numbering_path=["1", "7"]),
        heading("s-two", "Unnumbered", 6, parent="chapter", level=3),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert "sibling_numbering_monotonic" not in gate_codes(result)


def test_n3_duplicate_block_occupation_is_gate_failure():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 1, kind="chapter", level=2),
        heading("s-one", "First section", 5, parent="chapter", level=3,
                block="blk-1111111111111111", source_block_ids=["blk-1111111111111111"]),
        heading("s-two", "Second section", 6, parent="chapter", level=3,
                block="blk-2222222222222222",
                source_block_ids=["blk-2222222222222222", "blk-1111111111111111"]),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"duplicate_block_occupation"}
    finding = result["gate_failures"][0]
    assert finding["block_id"] == "blk-1111111111111111"
    assert finding["count"] == 2
    assert set(finding["node_ids"]) == {"s-one", "s-two"}


def test_n4a_apparatus_reopen_into_mainmatter_is_gate_failure():
    nodes = [
        book_node(),
        heading("bib", "Bibliography", 100, kind="backmatter", level=2,
                zone="bibliography"),
        heading("later", "Epilogue", 105, kind="chapter", level=2),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"zone_consistency"}
    finding = result["gate_failures"][0]
    assert finding["reason"] == "apparatus_then_mainmatter"
    assert finding["apparatus_node_id"] == "bib"


def test_n4b_main_heading_in_apparatus_is_gate_failure():
    nodes = [
        book_node(),
        heading("note-ch", "Anmerkungen", 100, kind="chapter", level=2,
                zone="notes"),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"zone_consistency"}
    finding = result["gate_failures"][0]
    assert finding["reason"] == "main_heading_in_apparatus"
    assert finding["kind"] == "chapter"
    assert finding["zone"] == "notes"


def test_n5_orphan_section_with_null_parent_is_gate_failure():
    nodes = [
        book_node(),
        heading("orphan", "Orphaned section", 3, level=3, parent=None),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"orphan_section"}
    assert result["gate_failures"][0]["reason"] == "missing_parent"


def test_n5_deep_heading_parented_only_to_book_is_gate_failure():
    nodes = [
        book_node(),
        heading("deep", "Deep uncontained heading", 3, level=3, parent="book"),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"orphan_section"}
    assert result["gate_failures"][0]["reason"] == "root_parented_deep_heading"


def test_n5_deep_heading_under_invalid_parent_kind_is_gate_failure():
    """Reviewer P03-M2: a deep heading parented to a non-container node
    (backmatter/frontmatter/toc_entry) must fail N5 — an apparatus or demoted
    node can never legitimately carry a level >= 3 heading."""
    nodes = [
        book_node(),
        heading("back", "Notes", 50, kind="backmatter", level=2),
        heading("deep", "A section under apparatus", 51, kind="section",
                level=3, parent="back"),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert result["ok"] is False
    assert gate_codes(result) == {"orphan_section"}
    assert result["gate_failures"][0]["reason"] == "invalid_parent_kind"
    assert result["gate_failures"][0]["parent_kind"] == "backmatter"


def test_n5_deep_heading_under_chapter_parent_passes():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 1, kind="chapter", level=2),
        heading("sec", "1.1 A real section", 2, kind="section", level=3, parent="chapter"),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert "orphan_section" not in gate_codes(result)


def test_n5_level_two_chapter_under_book_passes():
    nodes = [
        book_node(),
        heading("chapter", "Chapter 1", 1, kind="chapter", level=2),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert "orphan_section" not in gate_codes(result)


def test_v2_failures_are_reported_in_both_errors_and_gate_failures():
    nodes = [
        book_node(),
        heading("s-one", "1.7 First", 5, kind="section", level=3,
                ordinal="1.7", numbering_path=["1", "7"]),
        heading("s-two", "1.3 Second", 6, kind="section", level=3,
                ordinal="1.3", numbering_path=["1", "3"]),
    ]
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    for code in ("sibling_numbering_monotonic", "orphan_section"):
        assert code in error_codes(result)
        assert code in gate_codes(result)
    assert result["ok"] is False


def test_legacy_checks_still_run_inside_v2():
    # missing_parent (legacy) fires even though no v2 check is triggered.
    nodes = [
        book_node(),
        heading("bad", "Lost child", 5, parent="missing-parent", level=3),
    ]
    legacy = validate_structure(nodes, [], {"book_title": "Sample Book"})
    result = validate_structure_v2(nodes, [], {"book_title": "Sample Book"})
    assert "missing_parent" in legacy["errors"][0]["code"]
    assert "missing_parent" in error_codes(result)
    assert result["v2"]["checks"] == V2_CHECK_CODES
