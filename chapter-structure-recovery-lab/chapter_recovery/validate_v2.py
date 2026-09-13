"""V2 gate validator (plan P03 section 4.3/4.4; spec section 9.6).

``validate_structure_v2`` runs the existing 15 legacy checks
(``validate.validate_structure``) and appends five additive checks that only
run on the v2 ``apply-decisions`` path:

- N1 ``parent_before_child``            error + gate
- N2 ``sibling_numbering_monotonic``    error + gate
- N3 ``duplicate_block_occupation``     error + gate
- N4 ``zone_consistency``               error + gate
- N5 ``orphan_section``                 error + gate

``validate.py`` is untouched; the v2 checks are unreachable from the legacy
``analyze`` flow because nothing outside the v2 path imports this module.
The returned ``ok`` flag is recomputed: False iff errors or gate_failures are
non-empty (same convention as the legacy validator).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .models import Node, TocEntry
from .validate import validate_structure

APPARATUS_ZONES = {"notes", "bibliography", "index"}
MAIN_KINDS = {"part", "chapter", "section"}

V2_CHECK_CODES = [
    "parent_before_child",
    "sibling_numbering_monotonic",
    "duplicate_block_occupation",
    "zone_consistency",
    "orphan_section",
]

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_ONLY = re.compile(r"^[ivxlcdm]+$")


def _roman_int(token: str) -> int:
    total = 0
    previous = 0
    for char in reversed(token.casefold()):
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def _element_class(element: str) -> int:
    # Token order per plan 4.4: section-symbol < numeric < upper/lower letter.
    if element == "\u00a7":
        return 0
    if element.isdigit():
        return 1
    return 2


def _compare_elements(left: str, right: str) -> int:
    left_class = _element_class(left)
    right_class = _element_class(right)
    if left_class != right_class:
        return -1 if left_class < right_class else 1
    if left_class == 0:
        return 0
    if left_class == 1:
        left_value, right_value = int(left), int(right)
        return (left_value > right_value) - (left_value < right_value)
    if _ROMAN_ONLY.match(left) and _ROMAN_ONLY.match(right):
        left_value, right_value = _roman_int(left), _roman_int(right)
        return (left_value > right_value) - (left_value < right_value)
    left_value, right_value = left.casefold(), right.casefold()
    return (left_value > right_value) - (left_value < right_value)


def _compare_numbering_paths(left: List[str], right: List[str]) -> int:
    for left_element, right_element in zip(left, right):
        compared = _compare_elements(left_element, right_element)
        if compared:
            return compared
    return (len(left) > len(right)) - (len(left) < len(right))


def _document_order_key(node: Node) -> tuple:
    return (
        node.page_json if node.page_json is not None else -1,
        node.bbox[1] if node.bbox is not None else -1.0,
        node.id,
    )


def _add_failure(
    errors: List[Dict[str, Any]],
    gate_failures: List[Dict[str, Any]],
    finding: Dict[str, Any],
) -> None:
    errors.append(finding)
    gate_failures.append(finding)


def _check_parent_before_child(
    active: List[Node],
    by_id: Dict[str, Node],
    errors: List[Dict[str, Any]],
    gate_failures: List[Dict[str, Any]],
) -> None:
    """N1: an active parent's page must not follow its child's page."""
    for node in active:
        if node.kind == "book" or not node.parent_id:
            continue
        parent = by_id.get(node.parent_id)
        if parent is None or parent.page_json is None or node.page_json is None:
            continue
        if parent.page_json > node.page_json:
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "parent_before_child",
                    "node_id": node.id,
                    "parent_id": parent.id,
                    "parent_page_json": parent.page_json,
                    "page_json": node.page_json,
                },
            )


def _check_sibling_numbering_monotonic(
    active: List[Node],
    errors: List[Dict[str, Any]],
    gate_failures: List[Dict[str, Any]],
) -> None:
    """N2: sibling numbering tokens must not decrease in document order.

    Groups are sibling sets sharing a ``parent_id``.  A group is compared
    only when every heading in it carries a ``numbering_path``; partial
    numbering evidence is ignored (never an error).
    """
    groups: Dict[Optional[str], List[Node]] = {}
    for node in active:
        if node.kind == "book":
            continue
        groups.setdefault(node.parent_id, []).append(node)
    for parent_id, siblings in groups.items():
        numbered = [node for node in siblings if node.numbering_path]
        if len(numbered) < 2 or len(numbered) != len(siblings):
            continue
        ordered = sorted(numbered, key=_document_order_key)
        for left, right in zip(ordered, ordered[1:]):
            if _compare_numbering_paths(left.numbering_path, right.numbering_path) > 0:
                _add_failure(
                    errors,
                    gate_failures,
                    {
                        "code": "sibling_numbering_monotonic",
                        "parent_id": parent_id,
                        "node_id": right.id,
                        "previous_node_id": left.id,
                        "previous_numbering": left.numbering_path,
                        "numbering": right.numbering_path,
                    },
                )


def _check_duplicate_block_occupation(
    active: List[Node],
    errors: List[Dict[str, Any]],
    gate_failures: List[Dict[str, Any]],
) -> None:
    """N3: no active heading may share a source block with another.

    Counts ``block_id`` and every ``blk-*`` in ``source_block_ids`` across all
    active heading nodes (full block level, including rescued candidates).
    """
    occupied: Counter = Counter()
    owners: Dict[str, List[str]] = {}
    for node in active:
        if node.kind == "book":
            continue
        blocks = set(node.source_block_ids)
        if node.block_id:
            blocks.add(node.block_id)
        for block_id in blocks:
            occupied[block_id] += 1
            owners.setdefault(block_id, []).append(node.id)
    for block_id, count in occupied.items():
        if count > 1:
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "duplicate_block_occupation",
                    "block_id": block_id,
                    "count": count,
                    "node_ids": sorted(set(owners[block_id])),
                },
            )


def _check_zone_consistency(
    active: List[Node],
    errors: List[Dict[str, Any]],
    gate_failures: List[Dict[str, Any]],
) -> None:
    """N4: (a) no apparatus zone may reopen into mainmatter;
    (b) main kinds inside apparatus zones are gate failures."""
    for node in active:
        if (
            node.kind in MAIN_KINDS
            and node.zone in APPARATUS_ZONES
        ):
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "zone_consistency",
                    "node_id": node.id,
                    "kind": node.kind,
                    "zone": node.zone,
                    "reason": "main_heading_in_apparatus",
                },
            )

    ordered = sorted(
        [
            node
            for node in active
            if node.kind != "book" and node.page_json is not None
        ],
        key=_document_order_key,
    )
    apparatus_seen: Optional[Node] = None
    for node in ordered:
        if node.zone in APPARATUS_ZONES and apparatus_seen is None:
            apparatus_seen = node
        if apparatus_seen is not None and node.zone == "mainmatter":
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "zone_consistency",
                    "node_id": node.id,
                    "zone": node.zone,
                    "page_json": node.page_json,
                    "apparatus_node_id": apparatus_seen.id,
                    "apparatus_zone": apparatus_seen.zone,
                    "reason": "apparatus_then_mainmatter",
                },
            )
            break


def _check_orphan_section(
    active: List[Node],
    by_id: Dict[str, Node],
    errors: List[Dict[str, Any]],
    gate_failures: List[Dict[str, Any]],
) -> None:
    """N5: deep headings (level >= 3) must have a real container parent.

    Fails when the parent id is null, when the only parent is the book
    root (no part/chapter/section container between the root and a deep
    heading), or when the parent node is not a valid heading container
    (e.g. a backmatter/frontmatter/toc_entry node — a demoted or apparatus
    node can never legitimately carry a deep heading).  Unknown parents are
    left to the legacy ``missing_parent`` check.
    """
    valid_parent_kinds = {"part", "part_intro", "chapter", "section"}
    for node in active:
        if node.kind == "book" or node.level < 3:
            continue
        if not node.parent_id:
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "orphan_section",
                    "node_id": node.id,
                    "level": node.level,
                    "reason": "missing_parent",
                },
            )
            continue
        parent = by_id.get(node.parent_id)
        if parent is None:
            continue  # unknown id -> legacy missing_parent handles it
        if parent.kind == "book":
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "orphan_section",
                    "node_id": node.id,
                    "level": node.level,
                    "parent_id": parent.id,
                    "reason": "root_parented_deep_heading",
                },
            )
            continue
        if parent.kind not in valid_parent_kinds:
            _add_failure(
                errors,
                gate_failures,
                {
                    "code": "orphan_section",
                    "node_id": node.id,
                    "level": node.level,
                    "parent_id": parent.id,
                    "parent_kind": parent.kind,
                    "reason": "invalid_parent_kind",
                },
            )


def validate_structure_v2(
    nodes: List[Node],
    toc_entries: List[TocEntry],
    config: Optional[Dict[str, Any]] = None,
    decision_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the legacy 15 checks plus the v2 N1..N5 checks.

    ``decision_meta`` (the meta record of the decisions file) is
    informational only: it is recorded, verbatim, under ``result["v2"]`` so
    validation artifacts can be traced to the prompts/repair round that
    produced the tree.  It never changes check semantics.
    """
    result = validate_structure(nodes, toc_entries, config)
    errors: List[Dict[str, Any]] = result["errors"]
    warnings: List[Dict[str, Any]] = result["warnings"]
    gate_failures: List[Dict[str, Any]] = result["gate_failures"]

    active = [node for node in nodes if not node.ignored]
    by_id = {node.id: node for node in active}

    _check_parent_before_child(active, by_id, errors, gate_failures)
    _check_sibling_numbering_monotonic(active, errors, gate_failures)
    _check_duplicate_block_occupation(active, errors, gate_failures)
    _check_zone_consistency(active, errors, gate_failures)
    _check_orphan_section(active, by_id, errors, gate_failures)

    result["ok"] = not errors and not gate_failures
    result["v2"] = {
        "checks": list(V2_CHECK_CODES),
        "decision_meta": dict(decision_meta) if decision_meta else None,
    }
    # warnings is kept for parity with the legacy return contract; v2 checks
    # do not emit warnings today.
    del warnings
    return result
