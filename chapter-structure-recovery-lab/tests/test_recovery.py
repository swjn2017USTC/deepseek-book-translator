import json
from pathlib import Path

from chapter_recovery.cli import run_analyze
from chapter_recovery.models import Node
from chapter_recovery.recover import (
    _normalize_standard_hierarchy,
    _repair_sequential_arabic_roman_hierarchy,
    _suppress_book_title_duplicates,
)
from chapter_recovery.validate import validate_structure


FIXTURES = Path(__file__).parent / "fixtures"


def test_recovers_toc_macro_nodes_and_hidden_sections(tmp_path):
    result = run_analyze(FIXTURES / "sample_config.json", tmp_path)
    assert result["validation_ok"] is True
    structure = (tmp_path / "recovered_outline.md").read_text(encoding="utf-8")
    assert "## Introduction" in structure
    assert "## First Chapter" in structure
    assert "## Conclusion" in structure
    assert "### A hidden section" in structure
    assert "### Another hidden section" in structure


def test_ocr_hash_depth_does_not_override_same_layout(tmp_path):
    run_analyze(FIXTURES / "sample_config.json", tmp_path)
    structure = (tmp_path / "recovered_outline.md").read_text(encoding="utf-8")
    assert "#### A hidden section" not in structure
    assert "##### Another hidden section" not in structure


def test_sequential_arabic_chapters_and_roman_subsections_ignore_date_number():
    def node(identifier, title, page, kind="section", level=4):
        return Node(
            id=identifier, kind=kind, level=level, title_source=title,
            page_json=page, page_print=page, parent_id="book", block_id=identifier,
            source="fixture", confidence=1.0, bbox=(0, 0, 100, 20),
        )

    nodes = [
        node("part", "Part One: Old Regime", 1, "toc_entry", 2),
        node("chapter-1", "1 The Dynasty", 2, "chapter", 2),
        node("section-i", "i The Tsar and His People", 3, "chapter", 2),
        node("chapter-2", "2 Unstable Pillars", 4, "section", 3),
        node("date", "3 December 1920", 5, "section", 3),
        node("section-ii", "ii The Statesman", 6, "chapter", 2),
    ]

    _repair_sequential_arabic_roman_hierarchy(nodes, "book")

    assert (nodes[0].kind, nodes[0].level, nodes[0].parent_id) == ("part", 2, "book")
    assert (nodes[1].kind, nodes[1].level, nodes[1].parent_id) == ("chapter", 2, "part")
    assert (nodes[2].kind, nodes[2].level, nodes[2].parent_id) == ("section", 3, "chapter-1")
    assert (nodes[3].kind, nodes[3].level, nodes[3].parent_id) == ("chapter", 2, "part")
    assert (nodes[4].kind, nodes[4].level) == ("section", 3)
    assert (nodes[5].kind, nodes[5].level, nodes[5].parent_id) == ("section", 3, "chapter-2")


def test_review_packets_include_page_toc_and_image_context(tmp_path):
    run_analyze(FIXTURES / "sample_config.json", tmp_path)
    packets = [
        json.loads(line)
        for line in (tmp_path / "review_packets.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for packet in packets:
        assert packet["review_status"] == "needs_human"
        assert "page_context" in packet
        assert "toc_context" in packet
        assert packet["page_image_reference"]["ocr_json"].endswith("sample_ocr.json")
        assert packet["page_image_reference"]["json_page"] == packet["page_json"]
        assert packet["decision_guardrail"].startswith("Review evidence only")


def test_book_title_repeats_are_suppressed_and_children_are_redirected():
    book = Node("book", "book", 1, "The Example Book", None, None, None, None, "config", 1.0, evidence=["config"])
    header = Node("header", "section", 3, "The Example Book", 4, 4, "book", "b1", "header", 0.95, evidence=["header"])
    child = Node("child", "section", 4, "A real section", 4, 4, "header", "b2", "paragraph_title", 0.95, evidence=["paragraph_title"])
    near_one = Node("near-1", "section", 3, "The Example B00k", 5, 5, "book", "b3", "header", 0.9, evidence=["header"])
    near_two = Node("near-2", "section", 3, "The Example B00k", 6, 6, "book", "b4", "header", 0.9, evidence=["header"])
    candidates = [{"source_text": "The Example Book", "page_json": 7, "evidence": []}]

    _suppress_book_title_duplicates([book, header, child, near_one, near_two], candidates, book.title_source)

    assert header.ignored is True
    assert near_one.ignored is True and near_two.ignored is True
    assert child.parent_id == "book"
    assert candidates[0]["suppressed"] is True
    assert candidates[0]["suppression_reason"] == "book_title_running_header"


def test_standard_hierarchy_recovers_h1_h2_h3_and_validation_metrics():
    book = Node("book", "book", 4, "Book", None, None, None, None, "config", 1.0, evidence=["config"])
    chapter = Node("chapter", "chapter", 5, "Chapter 1", 1, 1, "book", "b1", "toc", 1.0, evidence=["toc"])
    section = Node("section", "section", 2, "Inner section", 2, 2, "chapter", "b2", "paragraph_title", 1.0, evidence=["paragraph_title"])

    _normalize_standard_hierarchy([book, chapter, section], "book", False)
    validation = validate_structure(
        [book, chapter, section], [], {"book_title": "Book", "chapters_under_parts": False}
    )

    assert [book.level, chapter.level, section.level] == [1, 2, 3]
    assert validation["ok"] is True
    assert validation["metrics"]["heading_level_counts"] == {"H1": 1, "H2": 1, "H3": 1}
