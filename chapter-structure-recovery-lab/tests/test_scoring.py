import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "score_structure.py"
SPEC = importlib.util.spec_from_file_location("score_structure", MODULE_PATH)
SCORING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCORING)


def test_hybrid_scoring_matches_macro_nodes_by_kind_and_ordinal():
    predicted = {
        "nodes": [
            {"id": "book", "kind": "book", "level": 1},
            {
                "id": "part-1", "kind": "part", "ordinal": "I", "level": 2,
                "page_json": 10, "parent_id": "book",
            },
            {
                "id": "chapter-1", "kind": "chapter", "ordinal": "1", "level": 2,
                "page_json": 12, "parent_id": "part-1",
            },
            {
                "id": "section", "kind": "section", "level": 3,
                "page_json": 13, "parent_id": "chapter-1",
            },
        ]
    }
    gold = {
        "nodes": [
            {"title": "第1部分 第一编", "level": 1, "page_json": 20, "line": 1},
            {"title": "第1章 第一章", "level": 2, "page_json": 12, "line": 2},
            {"title": "内部小节", "level": 3, "page_json": 13, "line": 3},
        ]
    }

    result = SCORING.score_structure(predicted, gold)

    assert result["matched_by_page_and_order"] == 2
    assert result["hybrid"]["matched"] == 3
    assert result["hybrid"]["macro_identity_matched"] == 2
    assert result["hybrid"]["semantic_level_matches"] == 3
    assert result["hybrid"]["raw_markdown_level_matches"] == 2
    assert result["hybrid"]["parent_matches"] == 3


def test_number_parses_arabic_roman_and_chinese_ordinals():
    assert SCORING._number("12") == 12
    assert SCORING._number("XIX") == 19
    assert SCORING._number("十九") == 19
    assert SCORING._number("一百零二") == 102


def test_explicit_block_to_gold_line_identity_prevents_local_zip_mispairing():
    predicted = {
        "nodes": [
            {"id": "book", "kind": "book", "level": 1},
            {"id": "a", "block_id": "block-a", "kind": "section", "level": 3,
             "page_json": 13, "bbox": [0, 10, 10, 20], "parent_id": "book"},
            {"id": "b", "block_id": "block-b", "kind": "section", "level": 4,
             "page_json": 13, "bbox": [0, 30, 10, 40], "parent_id": "a"},
        ]
    }
    gold = {
        "nodes": [
            {"title": "乙", "level": 4, "page_json": 13, "order_on_page": 0, "line": 20},
            {"title": "甲", "level": 3, "page_json": 13, "order_on_page": 1, "line": 21},
        ]
    }
    identities = {
        "positive_headings": [
            {"block_id": "block-a", "manual_gold_line": 21},
            {"block_id": "block-b", "manual_gold_line": 20},
        ]
    }

    result = SCORING.score_structure(predicted, gold, identities)

    assert result["level_matches"] == 0
    assert result["hybrid"]["explicit_identity_matched"] == 2
    assert result["hybrid"]["explicit_identity_gold"] == 2
    assert result["hybrid"]["semantic_level_matches"] == 2
    assert result["hybrid"]["local_page_order_matched"] == 0


def test_explicit_identity_gold_rejects_ambiguous_manual_lines():
    predicted = {"nodes": [{"id": "book", "kind": "book", "level": 1}]}
    gold = {"nodes": [{"title": "甲", "level": 3, "page_json": 1, "line": 10}]}
    identities = {
        "nodes": [
            {"block_id": "one", "manual_gold_line": 10},
            {"block_id": "two", "manual_gold_line": 10},
        ]
    }

    try:
        SCORING.score_structure(predicted, gold, identities)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate identity gold must fail closed")
