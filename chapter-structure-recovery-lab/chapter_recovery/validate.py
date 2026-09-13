from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from .models import Node, TocEntry
from .alignment import normalized_title
from .heading_text import rejection_reason


def validate_structure(
    nodes: List[Node],
    toc_entries: List[TocEntry],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or {}
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    gate_failures: List[Dict[str, Any]] = []
    active = [node for node in nodes if not node.ignored]
    ids = [node.id for node in active]
    for node_id, count in Counter(ids).items():
        if count > 1:
            errors.append({"code": "duplicate_id", "node_id": node_id})

    id_set = set(ids)
    by_id = {node.id: node for node in active}
    book_title = normalized_title(str(config.get("book_title") or ""))
    strict_hierarchy = not bool(config.get("hierarchy_profile"))
    for node in active:
        if node.parent_id and node.parent_id not in id_set:
            errors.append(
                {"code": "missing_parent", "node_id": node.id, "parent_id": node.parent_id}
            )
        if not node.evidence:
            errors.append({"code": "missing_provenance", "node_id": node.id})
        suspicious = rejection_reason(node.title_source) if node.kind != "book" else None
        if suspicious:
            warnings.append({"code": "suspicious_heading_text", "node_id": node.id, "reason": suspicious})
        if node.zone in {"notes", "bibliography", "index"} and node.kind in {"part", "chapter", "section"}:
            warnings.append({"code": "main_heading_in_apparatus", "node_id": node.id, "zone": node.zone})
        if not 1 <= int(node.level) <= 6:
            failure = {"code": "heading_level_out_of_range", "node_id": node.id, "level": node.level}
            errors.append(failure)
            gate_failures.append(failure)
        if node.kind == "book" and node.level != 1:
            failure = {"code": "book_not_h1", "node_id": node.id, "level": node.level}
            errors.append(failure)
            gate_failures.append(failure)
        if node.kind != "book" and book_title and normalized_title(node.title_source) == book_title:
            failure = {"code": "active_book_title_duplicate", "node_id": node.id}
            errors.append(failure)
            gate_failures.append(failure)

        parent = by_id.get(node.parent_id or "")
        equal_level_parallel_macro = bool(
            parent
            and parent.kind == "part"
            and node.kind == "chapter"
            and not bool(config.get("chapters_under_parts", False))
        )
        if parent and node.kind in {"section", "part_intro"} and node.level <= parent.level:
            finding = {
                "code": "hierarchy_parent_not_shallower",
                "node_id": node.id,
                "parent_id": parent.id,
                "level": node.level,
                "parent_level": parent.level,
            }
            if strict_hierarchy:
                errors.append(finding)
                gate_failures.append(finding)
            else:
                warnings.append(finding)
        elif parent and node.level < parent.level and not equal_level_parallel_macro:
            warnings.append({
                "code": "hierarchy_level_inversion",
                "node_id": node.id,
                "parent_id": parent.id,
                "level": node.level,
                "parent_level": parent.level,
            })

    aligned_nodes = [
        node
        for node in active
        if node.toc_entry_id and node.alignment_status == "matched" and node.block_id
    ]
    assigned_blocks = Counter(node.block_id for node in aligned_nodes if node.block_id)
    for block_id, count in assigned_blocks.items():
        if count > 1:
            failure = {
                "code": "duplicate_toc_block_assignment",
                "block_id": block_id,
                "count": count,
            }
            errors.append(failure)
            gate_failures.append(failure)

    sequence_nodes = sorted(
        aligned_nodes,
        key=lambda node: int(node.alignment_details.get("sequence_index", -1)),
    )
    sequence_pages = [node.page_json for node in sequence_nodes if node.page_json is not None]
    if any(right < left for left, right in zip(sequence_pages, sequence_pages[1:])):
        failure = {"code": "global_alignment_not_monotone"}
        errors.append(failure)
        gate_failures.append(failure)

    title_keys = Counter(
        (" ".join(node.title_source.casefold().split()), node.page_json, node.parent_id)
        for node in active if node.kind != "book"
    )
    for key, count in title_keys.items():
        if count > 1:
            warnings.append({"code": "duplicate_heading", "title": key[0], "page_json": key[1], "count": count})

    ordered = [node for node in active if node.kind != "book" and node.page_json is not None]
    ordered.sort(key=lambda node: (node.page_json or -1, node.bbox[1] if node.bbox else -1))
    previous_level = 1
    for node in ordered:
        expected_deep_marker = "section_symbol_numbering_grammar" in node.evidence
        if node.level > previous_level + 1 and not expected_deep_marker:
            warnings.append(
                {
                    "code": "level_jump",
                    "node_id": node.id,
                    "previous_level": previous_level,
                    "level": node.level,
                }
            )
        previous_level = node.level

    arabic_macro = [
        entry
        for entry in toc_entries
        if entry.print_page is not None and entry.kind in {"chapter", "part", "backmatter"}
    ]
    numbered_ids = {entry.id for entry in arabic_macro}
    aligned = len(
        {
            node.toc_entry_id
            for node in active
            if node.toc_entry_id in numbered_ids
            and node.alignment_status == "matched"
            and node.block_id is not None
        }
    )
    if arabic_macro and aligned < len(arabic_macro):
        failure = {
            "code": "macro_toc_alignment_incomplete",
            "aligned": aligned,
            "expected": len(arabic_macro),
        }
        warnings.append(failure)
        gate_failures.append(failure)

    return {
        "ok": not errors and not gate_failures,
        "errors": errors,
        "warnings": warnings,
        "gate_failures": gate_failures,
        "metrics": {
            "node_count": len(active),
            "heading_count": len([node for node in active if node.kind != "book"]),
            "macro_toc_expected": len(arabic_macro),
            "macro_toc_aligned": aligned,
            "toc_alignment_matched": len(aligned_nodes),
            "toc_alignment_unmatched": len(
                [node for node in active if node.toc_entry_id and node.alignment_status == "unmatched"]
            ),
            "toc_alignment_low_confidence": len(
                [
                    node
                    for node in active
                    if node.toc_entry_id
                    and node.alignment_status == "matched"
                    and node.confidence < 0.9
                ]
            ),
            "unique_toc_block_assignments": len(assigned_blocks),
            "ignored_heading_count": len([node for node in nodes if node.ignored]),
            "book_title_duplicates_suppressed": len([
                node for node in nodes
                if node.ignored and "book_title_running_header_suppressed" in node.evidence
            ]),
            "heading_level_counts": {
                f"H{level}": count
                for level, count in sorted(Counter(node.level for node in active).items())
            },
            "zone_counts": dict(Counter(node.zone for node in active if node.kind != "book")),
        },
    }


def _strip_number(title: str) -> str:
    parts = title.split(maxsplit=1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return title
