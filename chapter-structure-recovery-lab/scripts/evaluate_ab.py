#!/usr/bin/env python3
"""P04 A/B generalization metric harness (spec §10; plan §3 + amendment A1).

Compares, per book:

* ``path_a``  = legacy deterministic/profile pipeline artifacts
* ``path_b``  = v2 evidence + (deterministic or live LLM) resolver artifacts

Metrics (all deterministic, additive, never call a provider):

* ``score_structure`` (CW vol1 only; page-gold hybrid via importlib-loaded
  ``scripts/score_structure.py`` per amendment A1);
* ``parent_chain`` (hegel band1/band2/rechts identity gold: pass rate,
  coverage, level accordance — matched by ``block_id`` with fallback
  normalized-title);
* ``macro_toc`` recall/precision (toc_entries vs level-1/2 active headings,
  normalized-title/numbering-token matching inside a page tolerance);
* ``workload_proxy_per_100_pages`` (A = review_packets + candidate_suppressions
  lines / pages * 100; B = human_review_packets lines when status is
  ok/needs_human_review, else gate_failures / pages * 100, ``method`` field
  records which proxy was used);
* ``wall_time_s`` / ``provider_tokens`` (B live runs only; else 0);
* ``divergence`` — deterministic tree surrogate per plan §3.2
  (relabel/reparent/insert/delete counts over active headings matched by
  block_id, fallback normalized title; full Zhang-Shasha TED deferred).

Determinism: re-running with identical inputs yields byte-identical
``--out`` JSON (only ``generated_at`` changes).

Usage::

    python3 scripts/evaluate_ab.py --book <book_id> \
        --a-tree runs/<book>/book_structure.json \
        --b-tree runs/<book>/v2/book_structure.json \
        --a-validation runs/<book>/validation.json \
        --b-validation runs/<book>/v2/validation.json \
        --a-review-packets runs/<book>/review_packets.jsonl \
        --a-suppressions runs/<book>/candidate_suppressions.jsonl \
        --b-human-packets runs/<book>/v2/human_review_packets.jsonl \
        --b-status runs/<book>/v2/status.json \
        [--gold gold/<book>_headings.json] \
        [--parent-gold gold/<book>_parent_chains.json] \
        [--page-count N] [--label dev|holdout] \
        --out docs/v2/generalization_metrics.json

Every file argument is optional; absent artifacts simply produce an empty
side/entry with a ``missing`` note (deterministic-B-only books such as
debates-on-stalinism have no legacy A tree and are still measurable).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# imports of repo machinery (all deterministic; lazy where possible)
# --------------------------------------------------------------------------- #

def _load_score_structure() -> Any:
    """Import ``scripts/score_structure.py`` via importlib (amendment A1).

    The scoring script is a standalone module, not a package; loading it
    through ``spec_from_file_location`` keeps it callable without
    sys.path mutation or top-level import side effects.
    """
    here = Path(__file__).resolve().parent
    module_path = here / "score_structure.py"
    spec = importlib.util.spec_from_file_location("score_structure", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_text_similarity() -> Any:
    """Lazy import of the repo token-similarity helpers (rapidfuzz or difflib)."""
    from chapter_recovery.text_similarity import token_set_ratio  # type: ignore

    return token_set_ratio


SCORE_STRUCTURE = None  # populated lazily on first CW-vol1 scoring call


def _score_structure_module() -> Any:
    global SCORE_STRUCTURE
    if SCORE_STRUCTURE is None:
        SCORE_STRUCTURE = _load_score_structure()
    return SCORE_STRUCTURE


# --------------------------------------------------------------------------- #
# normalization helpers
# --------------------------------------------------------------------------- #

_NUMBER_TOKEN = re.compile(
    r"^\s*[\[\(]?\s*"
    r"([0-9]+(?:\.[0-9]+)*|[一二三四五六七八九十百千零〇]+|[ivxlcdmIVXLCDM]{1,4}|[a-zA-Z])"
    r"(.*)$",
    re.DOTALL,
)
_LEADING_SEPARATOR = re.compile(r"^[ \t]*[\.\)\]\:\-–—]")
_ROMAN_ONLY = re.compile(r"^[ivxlcdmIVXLCDM]{1,4}$")
_TRAILING = re.compile(r"[\s\.\:\-\–—]+$")


def _first_after_space(rest: str) -> str:
    return rest.lstrip(" \t")[:1]


def _numbering_token_candidate(token: str, rest: str) -> bool:
    """True when ``token`` at the title start is a numbering label, not a word."""
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", token):
        # '1.'-style labels and '1 The ...' enumerations; never a bare year
        # followed by a lowercase continuation ("1945 and after").
        if _LEADING_SEPARATOR.match(rest):
            return True
        first = _first_after_space(rest)
        return bool(first and (first.isupper() or not first.isascii()))
    if re.fullmatch(r"[一二三四五六七八九十百千零〇]+", token):
        return True
    if _ROMAN_ONLY.match(token) and len(token) >= 1:
        if _LEADING_SEPARATOR.match(rest):
            return True
        first = _first_after_space(rest)
        case_uniform = token.isupper() or token.islower()
        if case_uniform and (len(token) >= 2 or token.islower()):
            # roman enumerations continue with an uppercase/foreign word
            return bool(first and (first.isupper() or not first.isascii()))
        return False
    if len(token) == 1:
        # single letter: 'a.'/'A.'/'a)' enumerations, or lowercase 'i ' before
        # an uppercase word; never the first letter of a real word.
        if _LEADING_SEPARATOR.match(rest):
            return True
        if token.islower():
            first = _first_after_space(rest)
            return bool(first and (first.isupper() or not first.isascii()))
    return False


def normalize_title_source(title: Any) -> str:
    """Canonical heading identity used for block-id-free tree matching.

    Casefold + NFKC, one leading numbering token stripped, interior
    whitespace collapsed, trailing punctuation stripped.  Deterministic.
    """
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    match = _NUMBER_TOKEN.match(text)
    if match and _numbering_token_candidate(match.group(1), match.group(2)):
        text = re.sub(r"^[\s\.\)\]\:\-–—]+", "", match.group(2))
    text = _TRAILING.sub("", text)
    return " ".join(text.casefold().split())


def _token_ratio(left: str, right: str) -> float:
    token_set_ratio = _load_text_similarity()
    return float(token_set_ratio(left, right))


# --------------------------------------------------------------------------- #
# small file helpers
# --------------------------------------------------------------------------- #

def _read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _count_lines(path: Optional[Path]) -> Optional[int]:
    if path is None or not Path(path).is_file():
        return None
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _active_headings(tree: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Active (not ignored, not the book root) heading nodes."""
    if not tree:
        return []
    return [
        node
        for node in tree.get("nodes", [])
        if not node.get("ignored") and node.get("kind") != "book"
    ]


def _node_key(node: Dict[str, Any]) -> str:
    block_id = node.get("block_id")
    if block_id:
        return f"block:{block_id}"
    return "title:" + normalize_title_source(node.get("title_source"))


def _page_map(tree: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if not tree:
        return {}
    raw = (tree.get("page_map") or {}).get("print_to_json") or {}
    mapping: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            mapping[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _toc_page(entry: Dict[str, Any], page_map: Dict[str, int]) -> Optional[int]:
    """Physical (page_json-space) page for one toc entry, if resolvable."""
    print_page = entry.get("print_page")
    page_label = str(entry.get("page_label") or "")
    page: Optional[int] = None
    if isinstance(print_page, int) and not isinstance(print_page, bool):
        page = print_page
    elif page_label.isdigit():
        page = int(page_label)
    if page is None:
        return None
    mapped = page_map.get(str(page))
    return mapped if mapped is not None else page


# --------------------------------------------------------------------------- #
# validation summary
# --------------------------------------------------------------------------- #

def _validation_summary(validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not validation:
        return {"missing": True}
    gate_failures = validation.get("gate_failures") or []
    codes = Counter(str(item.get("code")) for item in gate_failures)
    summary: Dict[str, Any] = {
        "ok": bool(validation.get("ok")),
        "gate_failures": len(gate_failures),
        "errors": len(validation.get("errors") or []),
        "warnings": len(validation.get("warnings") or []),
        "codes": dict(sorted(codes.items())),
    }
    metrics = validation.get("metrics")
    if isinstance(metrics, dict):
        summary["metrics"] = {
            key: metrics[key]
            for key in (
                "node_count", "heading_count", "macro_toc_expected",
                "macro_toc_aligned", "heading_level_counts", "zone_counts",
            )
            if key in metrics
        }
    return summary


# --------------------------------------------------------------------------- #
# macro TOC recall / precision (plan §3.1.3)
# --------------------------------------------------------------------------- #

MACRO_ENTRY_KINDS = {"chapter", "part", "backmatter"}
# level-1/2 active heading kinds that a main-matter TOC row can announce
# (frontmatter lists never carry numeric main-matter TOC pages).
MACRO_NODE_KINDS = {"part", "part_intro", "chapter", "backmatter"}
PAGE_TOLERANCE = 3
FUZZY_MATCH_FLOOR = 0.9


def macro_toc_metrics(tree: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Title+page macro TOC coverage surrogate for one side (A or B)."""
    if not tree:
        return {"missing": True}
    nodes = _active_headings(tree)
    page_map = _page_map(tree)
    entries = [
        entry
        for entry in (tree.get("toc_entries") or [])
        if entry.get("kind") in MACRO_ENTRY_KINDS
    ]
    resolvable: List[Dict[str, Any]] = []
    without_page = 0
    for entry in entries:
        page = _toc_page(entry, page_map)
        if page is None:
            without_page += 1
            continue
        resolvable.append({"entry": entry, "page": page})

    headings = [
        node
        for node in nodes
        if node.get("kind") in MACRO_NODE_KINDS
        and isinstance(node.get("page_json"), int)
    ]
    macro_headings = [
        node for node in headings if int(node.get("level") or 0) <= 2
    ]

    # Greedy unique best-first matching: exact normalized-title equality wins,
    # then fuzzy token-ratio, both inside the page tolerance.
    pairs: List[Tuple[float, float, int, int]] = []  # (title_score, dist, e, h)
    for e_index, row in enumerate(resolvable):
        entry = row["entry"]
        entry_norm = normalize_title_source(entry.get("title"))
        entry_page = row["page"]
        for h_index, node in enumerate(macro_headings):
            distance = abs(entry_page - int(node.get("page_json")))
            if distance > PAGE_TOLERANCE:
                continue
            node_norm = normalize_title_source(node.get("title_source"))
            if not entry_norm or not node_norm:
                continue
            if entry_norm == node_norm:
                title_score = 1.0
            else:
                title_score = _token_ratio(entry_norm, node_norm)
                if title_score < FUZZY_MATCH_FLOOR:
                    continue
            pairs.append((title_score, distance, e_index, h_index))

    pairs.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    matched_entries = set()
    matched_headings = set()
    matched_mode: Dict[str, int] = {"exact_title": 0, "fuzzy_title": 0}
    for title_score, _distance, e_index, h_index in pairs:
        if e_index in matched_entries or h_index in matched_headings:
            continue
        matched_entries.add(e_index)
        matched_headings.add(h_index)
        if title_score >= 1.0:
            matched_mode["exact_title"] += 1
        else:
            matched_mode["fuzzy_title"] += 1

    return {
        "method": (
            "toc_entries(kind in {chapter,part,backmatter}, numeric page) vs "
            "active level<=2 {part,part_intro,chapter,backmatter} headings; "
            "normalized-title/numbering-token matching within page tolerance"
        ),
        "tolerance_pages": PAGE_TOLERANCE,
        "fuzzy_floor": FUZZY_MATCH_FLOOR,
        "entries": len(resolvable),
        "entries_without_page": without_page,
        "macro_headings": len(macro_headings),
        "matched_entries": len(matched_entries),
        "matched_headings": len(matched_headings),
        "matched_modes": matched_mode,
        "recall": round(len(matched_entries) / len(resolvable), 4) if resolvable else None,
        "precision": round(len(matched_headings) / len(macro_headings), 4) if macro_headings else None,
    }


# --------------------------------------------------------------------------- #
# parent-chain (identity gold) metric (plan §3.1.2)
# --------------------------------------------------------------------------- #

def parent_chain_metrics(tree: Optional[Dict[str, Any]], gold_path: Optional[Path]) -> Dict[str, Any]:
    """hegel-family identity gold: pass rate / coverage / level accordance."""
    if not tree:
        return {"missing": True, "note": "no tree"}
    gold_data = _read_json(gold_path)
    if not gold_data:
        return {"missing": True, "note": "no identity gold file"}
    nodes = tree.get("nodes") or []
    by_block: Dict[str, Dict[str, Any]] = {}
    by_title: Dict[str, List[Dict[str, Any]]] = {}
    by_id = {node.get("id"): node for node in nodes}
    for node in nodes:
        block_id = node.get("block_id")
        if block_id:
            by_block.setdefault(str(block_id), node)
        by_title.setdefault(normalize_title_source(node.get("title_source")), []).append(node)

    found = 0
    found_by_block = 0
    found_by_title = 0
    pass_count = 0
    level_matches = 0
    joint = 0
    misses: List[Dict[str, Any]] = []
    gold_nodes = gold_data.get("nodes") or []
    for expected in gold_nodes:
        title = expected.get("title_source")
        title_norm = normalize_title_source(title)
        block_id = expected.get("block_id")
        node = by_block.get(str(block_id)) if block_id else None
        matched_by = "block_id"
        if node is None:
            candidates = by_title.get(title_norm) or []
            # unique title match preferred; ambiguity falls back to page.
            if len(candidates) == 1:
                node = candidates[0]
                matched_by = "normalized_title"
            elif len(candidates) > 1 and isinstance(expected.get("page_json"), int):
                same_page = [
                    candidate for candidate in candidates
                    if candidate.get("page_json") == expected["page_json"]
                ]
                if len(same_page) == 1:
                    node = same_page[0]
                    matched_by = "normalized_title_plus_page"
        if node is None:
            misses.append({
                "title_source": title,
                "block_id": block_id,
                "page_json": expected.get("page_json"),
                "reason": "not_found",
            })
            continue
        found += 1
        if matched_by == "block_id":
            found_by_block += 1
        else:
            found_by_title += 1
        parent = by_id.get(node.get("parent_id"))
        parent_ok = bool(
            parent
            and parent.get("block_id")
            and str(parent.get("block_id")) == str(expected.get("expected_parent_block_id"))
        )
        level_ok = (
            expected.get("level") is not None
            and node.get("level") == expected.get("level")
        )
        pass_count += int(parent_ok)
        level_matches += int(level_ok)
        joint += int(parent_ok and level_ok)
        if not parent_ok:
            misses.append({
                "title_source": title,
                "block_id": block_id,
                "page_json": expected.get("page_json"),
                "reason": "parent_mismatch",
                "actual_parent_block_id": parent.get("block_id") if parent else None,
                "expected_parent_block_id": expected.get("expected_parent_block_id"),
            })

    return {
        "method": (
            "gold node -> tree node by block_id (fallback normalized title_source); "
            "pass <=> b_node.parent_id == expected-parent node id; "
            "level_accordance <=> b_node.level == gold.level"
        ),
        "gold": len(gold_nodes),
        "found": found,
        "found_by_block_id": found_by_block,
        "found_by_normalized_title": found_by_title,
        "pass_rate": round(pass_count / found, 4) if found else None,
        "coverage": round(found / len(gold_nodes), 4) if gold_nodes else None,
        "level_matches": level_matches,
        "level_accordance": round(level_matches / found, 4) if found else None,
        "joint": joint,
        "misses": misses,
    }


# --------------------------------------------------------------------------- #
# score_structure (CW vol1 page-gold) via importlib (amendment A1)
# --------------------------------------------------------------------------- #

def score_structure_metrics(
    tree: Optional[Dict[str, Any]],
    gold_path: Optional[Path],
    identity_path: Optional[Path],
) -> Dict[str, Any]:
    if not tree:
        return {"missing": True, "note": "no tree"}
    gold = _read_json(gold_path)
    if not gold:
        return {"missing": True, "note": "no page gold file"}
    identity = _read_json(identity_path) if identity_path else None
    module = _score_structure_module()
    result = module.score_structure(tree, gold, identity)
    # Keep only the compact hybrid metric block + top-level headline numbers.
    return {
        "method": "score_structure.py hybrid (importlib-loaded; identity=None unless --identity-gold)",
        "score": {
            "predicted": result["predicted"],
            "gold": result["gold"],
            "precision": result["precision"],
            "recall": result["recall"],
            "level_accuracy_on_matched": result["level_accuracy_on_matched"],
            "parent_accuracy_on_comparable": result["parent_accuracy_on_comparable"],
            "hybrid": result["hybrid"],
        },
    }


# --------------------------------------------------------------------------- #
# workload proxies (plan §3.1.4)
# --------------------------------------------------------------------------- #

def _workload_a(
    review_packets: Optional[Path],
    suppressions: Optional[Path],
    page_count: Optional[int],
) -> Dict[str, Any]:
    review_lines = _count_lines(review_packets)
    suppression_lines = _count_lines(suppressions)
    if review_lines is None or suppression_lines is None:
        return {
            "missing": True,
            "note": "A-side workload files absent (deterministic-B-only book?)",
        }
    if not page_count:
        return {"missing": True, "note": "no page_count; workload unscalable"}
    value = (review_lines + suppression_lines) / page_count * 100.0
    return {
        "method": "review_packets_lines + candidate_suppressions_lines / page_count * 100",
        "value_per_100_pages": round(value, 4),
        "review_packets_lines": review_lines,
        "candidate_suppressions_lines": suppression_lines,
        "page_count": page_count,
    }


def _workload_b(
    human_packets: Optional[Path],
    status: Optional[Path],
    validation: Optional[Dict[str, Any]],
    page_count: Optional[int],
) -> Dict[str, Any]:
    if not page_count:
        return {"missing": True, "note": "no page_count; workload unscalable"}
    status_data = _read_json(status) if status else None
    status_value = None
    if isinstance(status_data, dict):
        status_value = status_data.get("status")
    packet_lines = _count_lines(human_packets) if status_value in ("structure_v2_ok", "needs_human_review") else None
    if status_value in ("structure_v2_ok", "needs_human_review"):
        lines = packet_lines if packet_lines is not None else 0
        return {
            "method": "human_review_packets_lines / page_count * 100",
            "status": status_value,
            "value_per_100_pages": round(lines / page_count * 100.0, 4),
            "human_review_packets_lines": lines,
            "page_count": page_count,
        }
    gate_failures = len((validation or {}).get("gate_failures") or [])
    note = ""
    if status_value is None:
        note = "no v2/status.json (deterministic apply-decisions output); gate proxy used"
    else:
        note = f"status {status_value!r}; LLM unavailable -> deterministic gate proxy"
    return {
        "method": "gate_proxy",
        "status": status_value,
        "note": note,
        "value_per_100_pages": round(gate_failures / page_count * 100.0, 4),
        "gate_failures": gate_failures,
        "page_count": page_count,
    }


# --------------------------------------------------------------------------- #
# tree divergence surrogate (plan §3.2)
# --------------------------------------------------------------------------- #

def tree_distance(a_tree: Optional[Dict[str, Any]], b_tree: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic relabel/reparent/insert/delete surrogate over active headings.

    Matched by block_id (fallback normalized title_source).  Full ordered
    tree edit distance (Zhang-Shasha / zss) is explicitly deferred.
    """
    if a_tree is None or b_tree is None:
        return {"missing": True, "note": "both A and B trees required"}
    a_nodes = _active_headings(a_tree)
    b_nodes = _active_headings(b_tree)

    def index(nodes: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        by_key: Dict[str, Dict[str, Any]] = {}
        ambiguous: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        for node in nodes:
            key = _node_key(node)
            if key in by_key:
                ambiguous.append(node)
            else:
                by_key[key] = node
            seen[key] = seen.get(key, 0) + 1
        for node in ambiguous:
            by_key.pop(_node_key(node), None)  # ambiguous identity: unmatchable
        return by_key, [node for node in nodes if node not in by_key.values()]

    a_by_key, _ = index(a_nodes)
    b_by_key, _ = index(b_nodes)

    added: List[str] = []
    removed: List[str] = []
    changed_level: List[str] = []
    changed_kind: List[str] = []
    changed_parent: List[str] = []

    for key, node in b_by_key.items():
        if key not in a_by_key:
            added.append(str(node.get("id")))
    for key, node in a_by_key.items():
        if key not in b_by_key:
            removed.append(str(node.get("id")))
    for key in sorted(set(a_by_key) & set(b_by_key)):
        a_node = a_by_key[key]
        b_node = b_by_key[key]
        if a_node.get("level") != b_node.get("level"):
            changed_level.append(str(b_node.get("id")))
        if a_node.get("kind") != b_node.get("kind"):
            changed_kind.append(str(b_node.get("id")))
        if a_node.get("parent_id") != b_node.get("parent_id"):
            changed_parent.append(str(b_node.get("id")))

    relabel = sorted(set(changed_level) | set(changed_kind))
    added_sorted = sorted(added)
    removed_sorted = sorted(removed)
    cost = len(relabel) + len(changed_parent) + len(added_sorted) + len(removed_sorted)

    return {
        "method": "ordered_relabel_reparent_edit_cost",
        "matches_by": "block_id_fallback_normalized_title_source",
        "node_count_delta": len(b_nodes) - len(a_nodes),
        "added_node_ids": added_sorted,
        "removed_node_ids": removed_sorted,
        "changed_level_count": len(changed_level),
        "changed_kind_count": len(changed_kind),
        "changed_parent_count": len(changed_parent),
        "cost": cost,
        "note": (
            "deterministic surrogate over active headings; full ordered "
            "tree edit distance (zss) explicitly deferred"
        ),
    }


# --------------------------------------------------------------------------- #
# per-book assembly
# --------------------------------------------------------------------------- #

def evaluate_book(args: argparse.Namespace) -> Dict[str, Any]:
    a_tree_path = Path(args.a_tree) if args.a_tree else None
    b_tree_path = Path(args.b_tree) if args.b_tree else None
    a_tree = _read_json(a_tree_path)
    b_tree = _read_json(b_tree_path)
    a_validation = _read_json(Path(args.a_validation)) if args.a_validation else None
    b_validation = _read_json(Path(args.b_validation)) if args.b_validation else None

    entry: Dict[str, Any] = {}
    entry["label"] = args.label or "dev"
    entry["scorable"] = {"style": "none"}
    if args.gold and a_tree_path and Path(args.gold).is_file():
        entry["scorable"] = {"style": "page_gold", "gold": str(Path(args.gold).name)}
    elif args.parent_gold and Path(args.parent_gold).is_file():
        entry["scorable"] = {"style": "identity_parent_chains", "gold": str(Path(args.parent_gold).name)}

    path_a: Dict[str, Any] = {}
    if a_tree is not None:
        path_a["validation"] = _validation_summary(a_validation)
        path_a["macro_toc"] = macro_toc_metrics(a_tree)
        if args.gold and Path(args.gold).is_file():
            identity_path = Path(args.identity_gold) if args.identity_gold else None
            path_a["gold_score_structure"] = score_structure_metrics(
                a_tree, Path(args.gold), identity_path
            )
        elif args.parent_gold and Path(args.parent_gold).is_file():
            path_a["gold_parent_chain"] = parent_chain_metrics(a_tree, Path(args.parent_gold))
        path_a["workload_proxy_per_100_pages"] = _workload_a(
            Path(args.a_review_packets) if args.a_review_packets else None,
            Path(args.a_suppressions) if args.a_suppressions else None,
            args.page_count,
        )
        path_a["wall_time_s"] = args.wall_a_sec
        path_a["provider_tokens"] = 0
    else:
        path_a["missing"] = True
        if args.label != "deterministic-b-only":
            path_a["note"] = "no legacy A tree for this book"

    path_b: Dict[str, Any] = {}
    if b_tree is not None:
        path_b["validation"] = _validation_summary(b_validation)
        path_b["macro_toc"] = macro_toc_metrics(b_tree)
        if args.gold and Path(args.gold).is_file():
            identity_path = Path(args.identity_gold) if args.identity_gold else None
            path_b["gold_score_structure"] = score_structure_metrics(
                b_tree, Path(args.gold), identity_path
            )
        elif args.parent_gold and Path(args.parent_gold).is_file():
            path_b["gold_parent_chain"] = parent_chain_metrics(b_tree, Path(args.parent_gold))
        path_b["workload_proxy_per_100_pages"] = _workload_b(
            Path(args.b_human_packets) if args.b_human_packets else None,
            Path(args.b_status) if args.b_status else None,
            b_validation,
            args.page_count,
        )
        path_b["wall_time_s"] = args.wall_b_sec
        path_b["provider_tokens"] = args.provider_tokens_b or 0
        path_b["provider_blocked"] = args.provider_blocked_b
    else:
        path_b["missing"] = True

    entry["path_a"] = path_a
    entry["path_b"] = path_b
    entry["divergence"] = tree_distance(a_tree, b_tree)
    entry["page_count"] = args.page_count
    return entry


# --------------------------------------------------------------------------- #
# JSON merge + CLI
# --------------------------------------------------------------------------- #

def _merge(out_path: Path, book: str, entry: Dict[str, Any], live_meta: Optional[str]) -> None:
    if out_path.is_file():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    books = payload.get("books")
    if not isinstance(books, dict):
        books = {}
    books[book] = entry
    payload["books"] = dict(sorted(books.items()))
    payload["schema_version"] = 1
    payload["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload["tree_distance_method_note"] = (
        "deterministic relabel/reparent edit cost surrogate over active headings "
        "(plan P04 §3.2); full Zhang-Shasha tree edit distance (zss) explicitly deferred"
    )
    if live_meta:
        try:
            parsed = json.loads(live_meta)
            if isinstance(parsed, dict):
                payload["live_budget_used"] = parsed
        except json.JSONDecodeError:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P04 A/B generalization metric harness (spec §10, plan P04 §3)"
    )
    parser.add_argument("--book", required=True, help="book id / metrics key")
    parser.add_argument("--label", choices=["dev", "holdout", "deterministic-b-only"])
    parser.add_argument("--a-tree")
    parser.add_argument("--b-tree")
    parser.add_argument("--a-validation")
    parser.add_argument("--b-validation")
    parser.add_argument("--a-review-packets")
    parser.add_argument("--a-suppressions")
    parser.add_argument("--b-human-packets")
    parser.add_argument("--b-status")
    parser.add_argument("--gold", help="page-gold headings JSON (CW vol1)")
    parser.add_argument("--identity-gold", help="optional explicit-identity gold (CW vol1 local patterns)")
    parser.add_argument("--parent-gold", help="identity parent-chain gold JSON (hegel family)")
    parser.add_argument("--page-count", type=int)
    parser.add_argument("--wall-a-sec", type=float)
    parser.add_argument("--wall-b-sec", type=float)
    parser.add_argument("--provider-tokens-b", type=int)
    parser.add_argument("--provider-blocked-b", action="store_true", default=False)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--live-meta",
        help='JSON string merged into top-level "live_budget_used": '
        '{"wire_attempts": n, "tokens": n, "books": {...}}',
    )
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    entry = evaluate_book(args)
    _merge(Path(args.out), args.book, entry, args.live_meta)
    print(json.dumps({"book": args.book, "merged_into": args.out}, ensure_ascii=False, indent=2))
    return entry


if __name__ == "__main__":
    main()
