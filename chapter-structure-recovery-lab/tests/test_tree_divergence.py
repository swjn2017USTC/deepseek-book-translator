"""P04 tree-divergence surrogate tests (plan P04 §3.2 + §17 spirit).

The A/B harness reports a deterministic ``ordered_relabel_reparent_edit_cost``
surrogate instead of a full ordered tree edit distance (Zhang-Shasha / zss is
explicitly deferred and documented in the metrics JSON).  These tests pin the
four operation cases (insert / delete / relabel / reparent), the cost scalar,
and determinism.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_ab.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ab", MODULE_PATH)
HARNESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HARNESS)


def _node(node_id, title, kind="chapter", level=2, parent_id="book-1",
          page_json=10, block_id=None, ignored=False):
    return {
        "id": node_id,
        "kind": kind,
        "level": level,
        "title_source": title,
        "page_json": page_json,
        "parent_id": parent_id,
        "block_id": block_id,
        "ignored": ignored,
    }


def _tree(nodes):
    book = _node("book-1", "The Book", kind="book", level=1, parent_id=None,
                 page_json=None)
    return {"schema_version": 2, "book_id": "b", "page_map": {}, "toc_entries": [],
            "nodes": [book] + nodes}


def _distance(a_nodes, b_nodes):
    return HARNESS.tree_distance(_tree(a_nodes), _tree(b_nodes))


def test_identical_trees_have_zero_cost():
    nodes = [
        _node("n1", "One", block_id="blk-1"),
        _node("n2", "Two", kind="section", level=3, parent_id="n1",
              page_json=11, block_id="blk-2"),
    ]
    result = _distance(nodes, nodes)
    assert result["cost"] == 0
    assert result["node_count_delta"] == 0
    assert result["added_node_ids"] == []
    assert result["removed_node_ids"] == []
    assert result["changed_level_count"] == 0
    assert result["changed_parent_count"] == 0


def test_insert_counts_one_add_and_one_cost():
    base = [_node("n1", "One", block_id="blk-1")]
    grown = [_node("n1", "One", block_id="blk-1"),
             _node("n2", "Two", block_id="blk-2")]
    result = _distance(base, grown)
    assert result["added_node_ids"] == ["n2"]
    assert result["removed_node_ids"] == []
    assert result["node_count_delta"] == 1
    assert result["cost"] == 1


def test_delete_counts_one_removed_and_one_cost():
    base = [_node("n1", "One", block_id="blk-1"),
            _node("n2", "Two", block_id="blk-2")]
    shrunk = [_node("n1", "One", block_id="blk-1")]
    result = _distance(base, shrunk)
    assert result["removed_node_ids"] == ["n2"]
    assert result["added_node_ids"] == []
    assert result["node_count_delta"] == -1
    assert result["cost"] == 1


def test_relabel_level_change_counts_one_cost():
    base = [_node("n1", "One", level=2, block_id="blk-1"),
            _node("n2", "Two", level=3, parent_id="n1", block_id="blk-2")]
    revised = [_node("n1", "One", level=2, block_id="blk-1"),
               _node("n2", "Two", level=4, parent_id="n1", block_id="blk-2")]
    result = _distance(base, revised)
    assert result["changed_level_count"] == 1
    assert result["changed_kind_count"] == 0
    assert result["changed_parent_count"] == 0
    assert result["cost"] == 1


def test_reparent_counts_one_cost():
    base = [
        _node("p1", "Part One", kind="part", level=1, parent_id="book-1",
              block_id="blk-p1"),
        _node("c1", "Chapter", kind="chapter", level=2, parent_id="p1",
              block_id="blk-c1"),
    ]
    revised = [
        _node("p1", "Part One", kind="part", level=1, parent_id="book-1",
              block_id="blk-p1"),
        _node("c1", "Chapter", kind="chapter", level=2, parent_id="book-1",
              block_id="blk-c1"),
    ]
    result = _distance(base, revised)
    assert result["changed_parent_count"] == 1
    assert result["cost"] == 1


def test_kind_change_counts_as_relabel():
    base = [_node("n1", "One", kind="section", level=3, block_id="blk-1")]
    revised = [_node("n1", "One", kind="chapter", level=3, block_id="blk-1")]
    result = _distance(base, revised)
    assert result["changed_kind_count"] == 1
    assert result["changed_level_count"] == 0
    assert result["cost"] == 1


def test_cost_is_sum_of_all_operations():
    base = [
        _node("p1", "Part One", kind="part", level=1, parent_id="book-1",
              block_id="blk-p1"),
        _node("c1", "One", level=2, parent_id="p1", block_id="blk-c1"),
        _node("gone", "Gone", level=3, parent_id="c1", block_id="blk-gone"),
    ]
    revised = [
        _node("p1", "Part One", kind="part", level=1, parent_id="book-1",
              block_id="blk-p1"),
        _node("c1", "One", level=3, parent_id="book-1", block_id="blk-c1"),
        _node("fresh", "Fresh", level=3, parent_id="c1", block_id="blk-fresh"),
    ]
    result = _distance(base, revised)
    # relabel (c1 level) + reparent (c1) + delete (gone) + insert (fresh)
    assert result["cost"] == 4
    assert result["added_node_ids"] == ["fresh"]
    assert result["removed_node_ids"] == ["gone"]
    assert result["changed_level_count"] == 1
    assert result["changed_parent_count"] == 1


def test_block_id_matching_beats_title_equality():
    # Same normalized title in both trees but different block ids: matched by
    # block_id when present, so a pure title collision is not misread.
    base = [_node("n1", "Part One: Things", kind="part", block_id="blk-1")]
    revised = [_node("n2", "Part One: Things", kind="part", block_id="blk-2")]
    result = _distance(base, revised)
    assert result["added_node_ids"] == ["n2"]
    assert result["removed_node_ids"] == ["n1"]


def test_normalized_title_fallback_matches_without_block_ids():
    base = [_node("n1", "1.  The Title", block_id=None)]
    revised = [_node("n1", "The Title", block_id=None)]
    result = _distance(base, revised)
    assert result["cost"] == 0
    assert result["added_node_ids"] == []
    assert result["removed_node_ids"] == []


def test_determinism_same_input_same_output():
    nodes = [
        _node("p1", "Part One", kind="part", level=1, parent_id="book-1",
              block_id="blk-p1"),
        _node("c1", "One", level=2, parent_id="p1", block_id="blk-c1"),
    ]
    first = _distance(nodes, nodes)
    second = _distance(nodes, nodes)
    assert first == second


def test_ignored_nodes_are_not_counted():
    base = [_node("n1", "One", block_id="blk-1")]
    revised = [
        _node("n1", "One", block_id="blk-1"),
        _node("n2", "Two", block_id="blk-2", ignored=True),
    ]
    result = _distance(base, revised)
    assert result["added_node_ids"] == []
    assert result["node_count_delta"] == 0
    assert result["cost"] == 0
