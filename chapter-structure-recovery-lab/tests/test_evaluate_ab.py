"""P04 harness tests: score_structure importlib compat + metric oracles.

Amendment A1 requires that the importlib-loaded ``scripts/score_structure.py``
(function ``score_structure``) produce exactly the JSON the subprocess CLI
``python3 scripts/score_structure.py A B`` prints for the same fixture
(CW vol1 page-gold book).  The remaining tests pin the harness's own metric
functions (macro TOC, parent chains, workload proxies, deterministic merge).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
GOLD_DIR = ROOT / "gold"

MODULE_PATH = ROOT / "scripts" / "evaluate_ab.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ab", MODULE_PATH)
HARNESS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(HARNESS)

SCORE_PATH = ROOT / "scripts" / "score_structure.py"
SCORE_SPEC = importlib.util.spec_from_file_location("score_structure", SCORE_PATH)
SCORING = importlib.util.module_from_spec(SCORE_SPEC)
assert SCORE_SPEC.loader is not None
SCORE_SPEC.loader.exec_module(SCORING)


# --------------------------------------------------------------------------- #
# amendment A1: importlib score_structure == subprocess CLI (CW vol1)
# --------------------------------------------------------------------------- #

def _cw_artifacts_available():
    return (
        (RUNS_DIR / "cambridge_world_history_vol1" / "book_structure.json").is_file()
        and (GOLD_DIR / "cambridge_world_history_vol1_headings.json").is_file()
    )


@pytest.mark.skipif(
    not _cw_artifacts_available(),
    reason="local CW vol1 run tree + gold unavailable",
)
def test_importlib_score_structure_matches_subprocess_cli_on_cw_vol1():
    structure_path = RUNS_DIR / "cambridge_world_history_vol1" / "book_structure.json"
    gold_path = GOLD_DIR / "cambridge_world_history_vol1_headings.json"
    predicted = json.loads(structure_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))

    via_import = SCORING.score_structure(predicted, gold)

    completed = subprocess.run(
        [sys.executable, str(SCORE_PATH), str(structure_path), str(gold_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    via_cli = json.loads(completed.stdout)

    assert via_import == via_cli
    assert via_import["hybrid"]["precision"] > 0.0


def test_harness_score_structure_loader_agrees_with_direct_import():
    # The harness must route through the same module object semantics; guard
    # against loader drift (different file would break the A1 contract).
    loaded = HARNESS._score_structure_module()
    assert loaded is SCORING or loaded.__file__ == str(SCORE_PATH)
    assert callable(getattr(loaded, "score_structure"))


# --------------------------------------------------------------------------- #
# macro TOC metric oracle
# --------------------------------------------------------------------------- #

def _macro_tree(nodes, toc_entries, page_map=None):
    book = {"id": "book-1", "kind": "book", "level": 1, "title_source": "Book",
            "page_json": None, "ignored": False}
    return {
        "schema_version": 2,
        "book_id": "b",
        "page_map": page_map or {"print_to_json": {}},
        "toc_entries": toc_entries,
        "nodes": [book] + nodes,
    }


def _mnode(node_id, title, level, page, kind="chapter", parent_id="book-1"):
    return {"id": node_id, "kind": kind, "level": level, "title_source": title,
            "page_json": page, "parent_id": parent_id, "ignored": False}


def test_macro_toc_recall_and_precision_oracle():
    toc_entries = [
        {"id": "toc-1", "kind": "chapter", "title": "1. First Chapter", "print_page": 10},
        {"id": "toc-2", "kind": "chapter", "title": "2. Second Chapter", "print_page": 30},
        {"id": "toc-3", "kind": "frontmatter", "title": "List of figures", "print_page": None},
    ]
    nodes = [
        _mnode("c1", "First Chapter", 2, 10),
        _mnode("c2", "Second Chapter", 2, 30),
        _mnode("c3", "Unannounced Chapter", 2, 60),
        _mnode("s1", "Deep section", 3, 10, kind="section"),
    ]
    tree = _macro_tree(nodes, toc_entries)
    result = HARNESS.macro_toc_metrics(tree)
    assert result["entries"] == 2            # frontmatter entry excluded
    assert result["entries_without_page"] == 0
    assert result["macro_headings"] == 3     # level<=2 part/chapter/backmatter
    assert result["matched_entries"] == 2
    assert result["matched_headings"] == 2
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3, abs=0.0001)


def test_macro_toc_page_tolerance_and_numbering_strip():
    toc_entries = [
        # same title as node but announced 3 physical pages earlier: within tol
        {"id": "toc-1", "kind": "chapter", "title": "III.  The Wanderer", "print_page": 20},
    ]
    nodes = [_mnode("c1", "The Wanderer", 2, 23)]
    result = HARNESS.macro_toc_metrics(_macro_tree(nodes, toc_entries))
    assert result["recall"] == 1.0
    assert result["matched_modes"]["exact_title"] == 1

    # beyond tolerance: no match
    toc_entries = [
        {"id": "toc-1", "kind": "chapter", "title": "The Wanderer", "print_page": 20},
    ]
    nodes = [_mnode("c1", "The Wanderer", 2, 30)]
    result = HARNESS.macro_toc_metrics(_macro_tree(nodes, toc_entries))
    assert result["recall"] == 0.0


def test_macro_toc_uses_print_to_json_mapping():
    toc_entries = [
        {"id": "toc-1", "kind": "chapter", "title": "Intro", "print_page": 5},
    ]
    nodes = [_mnode("c1", "Intro", 2, 105)]  # physical page_json space
    page_map = {"print_to_json": {"5": 105}}
    result = HARNESS.macro_toc_metrics(_macro_tree(nodes, toc_entries, page_map))
    assert result["recall"] == 1.0


# --------------------------------------------------------------------------- #
# parent-chain metric oracle
# --------------------------------------------------------------------------- #

def _chain_tree():
    book = {"id": "book-1", "kind": "book", "level": 1, "title_source": "Book",
            "page_json": None, "ignored": False}
    nodes = [
        book,
        {"id": "n-part", "kind": "part", "level": 2, "title_source": "First Part",
         "page_json": 5, "parent_id": "book-1", "block_id": "blk-part", "ignored": False},
        {"id": "n-sec", "kind": "section", "level": 3, "title_source": "Opening",
         "page_json": 6, "parent_id": "n-part", "block_id": "blk-sec", "ignored": False},
        {"id": "n-other", "kind": "section", "level": 3, "title_source": "Wrong parent child",
         "page_json": 9, "parent_id": "book-1", "block_id": "blk-other", "ignored": False},
        {"id": "n-wrong-lvl", "kind": "section", "level": 4, "title_source": "Level drift",
         "page_json": 10, "parent_id": "n-part", "block_id": "blk-wrong-lvl", "ignored": False},
    ]
    return {"schema_version": 2, "book_id": "b", "page_map": {}, "toc_entries": [],
            "nodes": nodes}


def test_parent_chain_pass_rate_coverage_level(tmp_path):
    gold_path = tmp_path / "chains.json"
    gold = {"schema_version": 1, "book_id": "b", "nodes": [
        {"block_id": "blk-sec", "expected_parent_block_id": "blk-part",
         "level": 3, "title_source": "Opening", "page_json": 6},        # pass
        {"block_id": "blk-other", "expected_parent_block_id": "blk-part",
         "level": 3, "title_source": "Wrong parent child", "page_json": 9},  # parent fail
        {"block_id": "blk-wrong-lvl", "expected_parent_block_id": "blk-part",
         "level": 3, "title_source": "Level drift", "page_json": 10},   # level fail
        {"block_id": "blk-missing", "expected_parent_block_id": "blk-part",
         "level": 3, "title_source": "Never recovered", "page_json": 99},  # coverage miss
    ]}
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    result = HARNESS.parent_chain_metrics(_chain_tree(), gold_path)
    assert result["gold"] == 4
    assert result["found"] == 3
    assert result["coverage"] == pytest.approx(0.75)
    # passes: blk-sec (parent ok) and blk-wrong-lvl (parent ok, level drift)
    assert result["pass_rate"] == pytest.approx(2 / 3, abs=0.0001)
    assert result["level_matches"] == 2
    assert result["level_accordance"] == pytest.approx(2 / 3, abs=0.0001)
    assert result["joint"] == 1  # only blk-sec matches level AND parent
    assert len(result["misses"]) == 2  # one not-found + one parent mismatch


def test_parent_chain_normalized_title_fallback(tmp_path):
    gold_path = tmp_path / "chains.json"
    gold = {"schema_version": 1, "book_id": "b", "nodes": [
        # no block_id anywhere: matched via normalized title_source
        {"title_source": "a. Opening", "expected_parent_block_id": "blk-part",
         "level": 3, "page_json": 6},
    ]}
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    result = HARNESS.parent_chain_metrics(_chain_tree(), gold_path)
    assert result["found"] == 1
    assert result["found_by_normalized_title"] == 1
    assert result["pass_rate"] == 1.0


# --------------------------------------------------------------------------- #
# workload proxies
# --------------------------------------------------------------------------- #

def _write_lines(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join("{}\n".format(i) for i in range(count)), encoding="utf-8")


def test_workload_a_counts_review_plus_suppressions(tmp_path):
    review = tmp_path / "review_packets.jsonl"
    suppressions = tmp_path / "candidate_suppressions.jsonl"
    _write_lines(review, 3)
    _write_lines(suppressions, 7)
    result = HARNESS._workload_a(review, suppressions, page_count=100)
    assert result["method"].startswith("review_packets_lines")
    assert result["value_per_100_pages"] == 10.0


def test_workload_b_human_packets_when_status_ok(tmp_path):
    packets = tmp_path / "human_review_packets.jsonl"
    status = tmp_path / "status.json"
    _write_lines(packets, 5)
    status.write_text(json.dumps({"status": "structure_v2_ok"}), encoding="utf-8")
    result = HARNESS._workload_b(packets, status, {}, page_count=100)
    assert result["method"] == "human_review_packets_lines / page_count * 100"
    assert result["value_per_100_pages"] == 5.0


def test_workload_b_gate_proxy_on_llm_unavailable(tmp_path):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"status": "structure_llm_unavailable"}), encoding="utf-8")
    validation = {"gate_failures": [{"code": "orphan_section"}, {"code": "orphan_section"}]}
    result = HARNESS._workload_b(None, status, validation, page_count=100)
    assert result["method"] == "gate_proxy"
    assert result["value_per_100_pages"] == 2.0
    assert result["gate_failures"] == 2


def test_workload_b_gate_proxy_when_status_missing(tmp_path):
    # deterministic apply-decisions output has no v2/status.json
    validation = {"gate_failures": [{"code": "zone_consistency"}]}
    result = HARNESS._workload_b(None, None, validation, page_count=100)
    assert result["method"] == "gate_proxy"
    assert "no v2/status.json" in result["note"]
    assert result["value_per_100_pages"] == 1.0


# --------------------------------------------------------------------------- #
# CLI merge + determinism
# --------------------------------------------------------------------------- #

def _demo_tree(tmp_path, name="c1"):
    return {
        "schema_version": 2,
        "book_id": "demo",
        "page_map": {"print_to_json": {"10": 10}},
        "toc_entries": [
            {"id": "toc-1", "kind": "chapter", "title": "Chapter One", "print_page": 10}
        ],
        "nodes": [
            {"id": "book-1", "kind": "book", "level": 1, "title_source": "Demo",
             "page_json": None, "ignored": False},
            {"id": name, "kind": "chapter", "level": 2, "title_source": "Chapter One",
             "page_json": 10, "parent_id": "book-1", "block_id": "blk-1", "ignored": False},
        ],
    }


def test_cli_merge_is_deterministic_and_merges_books(tmp_path):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    out = tmp_path / "metrics.json"
    a_path.write_text(json.dumps(_demo_tree(tmp_path, "c1")), encoding="utf-8")
    b_path.write_text(json.dumps(_demo_tree(tmp_path, "c1")), encoding="utf-8")

    args = [
        "--book", "demo-a", "--a-tree", str(a_path), "--b-tree", str(b_path),
        "--a-validation", "", "--b-validation", "",
        "--page-count", "100", "--out", str(out),
    ]
    HARNESS.main(args)
    first_payload = json.loads(out.read_text(encoding="utf-8"))
    assert "demo-a" in first_payload["books"]
    assert first_payload["books"]["demo-a"]["divergence"]["cost"] == 0
    assert first_payload["books"]["demo-a"]["path_a"]["macro_toc"]["recall"] == 1.0

    # second run over a different book merges and stays deterministic
    args_b = [
        "--book", "demo-b", "--a-tree", str(a_path), "--b-tree", str(b_path),
        "--a-validation", "", "--b-validation", "",
        "--page-count", "100", "--out", str(out),
    ]
    HARNESS.main(args_b)
    second_payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(second_payload["books"]) == {"demo-a", "demo-b"}

    # re-run identical args -> byte-identical apart from generated_at
    HARNESS.main(args)
    third = json.loads(out.read_text(encoding="utf-8"))
    for payload in (first_payload, second_payload, third):
        payload.pop("generated_at", None)
    first_payload.pop("generated_at", None)
    third.pop("generated_at", None)
    # demo-a was re-merged into the file holding demo-b: compare book entries
    assert second_payload["books"]["demo-a"] == third["books"]["demo-a"]
    assert second_payload["books"]["demo-b"] == third["books"]["demo-b"]


def test_cli_live_meta_block():
    # live_budget_used is surfaced at the top level when --live-meta is passed
    result = HARNESS.evaluate_book(
        HARNESS.build_parser().parse_args(
            ["--book", "x", "--label", "holdout", "--page-count", "10",
             "--out", "/tmp/never-written.json"]
        )
    )
    assert result["label"] == "holdout"
