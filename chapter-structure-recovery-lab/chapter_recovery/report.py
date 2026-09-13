from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit

from .models import Block, Node, TocEntry
from .page_map import PageMap


def write_outputs(
    output_dir: Path,
    config: Dict[str, Any],
    nodes: List[Node],
    toc_entries: List[TocEntry],
    page_map: PageMap,
    text_candidates: List[Dict[str, Any]],
    validation: Dict[str, Any],
    pages: List[List[Block]],
    raw_pages: List[Dict[str, Any]],
    input_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = {
        "schema_version": 2,
        "book_id": config["book_id"],
        "book_title": config["book_title"],
        "page_map": page_map.to_dict(),
        "toc_entries": [entry.to_dict() for entry in toc_entries],
        "nodes": [node.to_dict() for node in nodes],
    }
    (output_dir / "book_structure.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    hierarchy_audit = {
        "schema_version": 1,
        "heading_level_counts": validation.get("metrics", {}).get("heading_level_counts", {}),
        "book_title_duplicates_suppressed": validation.get("metrics", {}).get(
            "book_title_duplicates_suppressed", 0
        ),
        "hierarchy_findings": [
            item
            for item in [
                *validation.get("errors", []),
                *validation.get("warnings", []),
            ]
            if str(item.get("code", "")).startswith(
                ("hierarchy_", "heading_level_", "book_not_", "active_book_title_")
            )
        ],
    }
    (output_dir / "hierarchy_audit.json").write_text(
        json.dumps(hierarchy_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_outline(output_dir / "recovered_outline.md", nodes)
    suppressed = [item for item in text_candidates if item.get("suppressed")]
    reviewable = [item for item in text_candidates if not item.get("suppressed")]
    with (output_dir / "candidate_suppressions.jsonl").open("w", encoding="utf-8") as handle:
        for item in suppressed:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    packets = _review_packets(nodes, reviewable, float(config.get("review_threshold", 0.9)))
    contextual_packets = _contextualize_review_packets(
        packets, nodes, toc_entries, pages, raw_pages, input_path
    )
    with (output_dir / "review_packets.jsonl").open("w", encoding="utf-8") as handle:
        for packet in contextual_packets:
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")
    _write_report(
        output_dir / "structure_report.md",
        nodes,
        toc_entries,
        page_map,
        contextual_packets,
        suppressed,
        validation,
    )


def _write_outline(path: Path, nodes: Iterable[Node]) -> None:
    lines = []
    for node in nodes:
        if node.ignored:
            continue
        lines.append(f"{'#' * node.level} {node.title_source}")
        if node.kind != "book":
            lines.append(
                f"<!-- json_page={node.page_json} print_page={node.page_print} "
                f"kind={node.kind} confidence={node.confidence:.2f} source={node.source} -->"
            )
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _review_packets(
    nodes: List[Node], text_candidates: List[Dict[str, Any]], threshold: float
) -> List[Dict[str, Any]]:
    packets: List[Dict[str, Any]] = []
    for node in nodes:
        if node.kind == "book" or node.ignored or node.confidence >= threshold:
            continue
        packets.append(
            {
                "candidate_id": node.id,
                "candidate_type": (
                    "toc_alignment_unmatched"
                    if node.alignment_status == "unmatched"
                    else "accepted_low_confidence_node"
                ),
                "page_json": node.page_json,
                "source_text": node.title_source,
                "current_kind": node.kind,
                "current_level": node.level,
                "confidence": round(node.confidence, 4),
                "evidence": node.evidence,
                "alignment_details": node.alignment_details,
                "allowed_decisions": [
                    "accept", "reject", "change_level", "change_kind",
                    "merge_with_previous", "needs_human",
                ],
            }
        )
    packets.extend(text_candidates)
    return packets


def _safe_image_reference(value: Any) -> Any:
    """Keep a stable page-image pointer without copying expiring signed query data."""
    if not isinstance(value, str):
        return value
    parts = urlsplit(value)
    if parts.scheme in {"http", "https"}:
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    return value


def _block_summary(block: Block, role: str) -> Dict[str, Any]:
    return {
        "role": role,
        "block_id": block.id,
        "raw_block_id": block.raw_block_id,
        "label": block.label,
        "text": block.clean_content[:600],
        "bbox": list(block.bbox),
        "order": block.order,
    }


def _contextualize_review_packets(
    packets: List[Dict[str, Any]],
    nodes: List[Node],
    toc_entries: List[TocEntry],
    pages: List[List[Block]],
    raw_pages: List[Dict[str, Any]],
    input_path: Path,
) -> List[Dict[str, Any]]:
    node_by_id = {node.id: node for node in nodes}
    toc_positions = {entry.id: index for index, entry in enumerate(toc_entries)}
    result: List[Dict[str, Any]] = []
    for original in packets:
        packet = dict(original)
        packet.setdefault("review_status", "needs_human")
        node = node_by_id.get(str(packet.get("candidate_id") or ""))
        page_index = packet.get("page_json")
        page_blocks = (
            pages[page_index]
            if isinstance(page_index, int) and 0 <= page_index < len(pages)
            else []
        )
        target_ids = set(packet.get("source_block_ids") or [])
        if packet.get("block_id"):
            target_ids.add(packet["block_id"])
        if node:
            target_ids.update(node.source_block_ids)
            if node.block_id:
                target_ids.add(node.block_id)
        target_index = next(
            (index for index, block in enumerate(page_blocks) if block.id in target_ids),
            None,
        )
        if target_index is None:
            selected = list(enumerate(page_blocks[:5]))
        else:
            start = max(0, target_index - 2)
            stop = min(len(page_blocks), target_index + 3)
            selected = list(enumerate(page_blocks[start:stop], start=start))
        packet["page_context"] = {
            "target_block_found": target_index is not None,
            "blocks": [
                _block_summary(
                    block,
                    "current"
                    if index == target_index
                    else ("preceding" if target_index is not None and index < target_index else "following")
                    if target_index is not None
                    else "page_sample",
                )
                for index, block in selected
            ],
        }

        toc_id = node.toc_entry_id if node else packet.get("toc_entry_id")
        toc_index = toc_positions.get(toc_id)
        toc_context = []
        if toc_index is not None:
            for index in range(max(0, toc_index - 1), min(len(toc_entries), toc_index + 2)):
                entry = toc_entries[index]
                item = entry.to_dict()
                item["role"] = "current" if index == toc_index else (
                    "preceding" if index < toc_index else "following"
                )
                toc_context.append(item)
        packet["toc_context"] = {
            "toc_entry_id": toc_id,
            "entries": toc_context,
        }

        raw_page = (
            raw_pages[page_index]
            if isinstance(page_index, int) and 0 <= page_index < len(raw_pages)
            else {}
        )
        output_images = raw_page.get("outputImages") or {}
        packet["page_image_reference"] = {
            "ocr_json": str(input_path),
            "json_page": page_index,
            "source_json_pointer": f"$[{page_index}].inputImage" if page_index is not None else None,
            "input_image": _safe_image_reference(raw_page.get("inputImage")),
            "layout_image": _safe_image_reference(
                output_images.get("layout_det_res") if isinstance(output_images, dict) else None
            ),
            "signed_query_omitted": True,
        }
        if node:
            packet["node_context"] = {
                "node_id": node.id,
                "parent_id": node.parent_id,
                "block_id": node.block_id,
                "source_block_ids": node.source_block_ids,
                "bbox": list(node.bbox) if node.bbox is not None else None,
            }
        packet["decision_guardrail"] = (
            "Review evidence only. Reject or defer when evidence is insufficient; "
            "do not invent a title, page, kind, or level."
        )
        result.append(packet)
    return result


def _write_report(
    path: Path,
    nodes: List[Node],
    toc_entries: List[TocEntry],
    page_map: PageMap,
    packets: List[Dict[str, Any]],
    suppressed: List[Dict[str, Any]],
    validation: Dict[str, Any],
) -> None:
    active = [node for node in nodes if not node.ignored]
    lines = [
        "# 章节结构恢复报告",
        "",
        "## 摘要",
        "",
        f"- 结构节点：{len(active)}",
        f"- 目录条目：{len(toc_entries)}",
        f"- 待 coding agent/人工裁决：{len(packets)}",
        f"- 验证通过：{'是' if validation.get('ok') else '否'}",
        f"- 全局对齐：{validation.get('metrics', {}).get('toc_alignment_matched', 0)} 个匹配，"
        f"{validation.get('metrics', {}).get('toc_alignment_unmatched', 0)} 个未匹配",
        "",
        "## 页码映射",
        "",
        f"- 公式：印刷页 = JSON页 + ({page_map.offset})",
        f"- 置信度：{page_map.confidence:.2%}（{page_map.support}/{page_map.observations}）",
        "",
        f"- 自动排除伪标题/附录重复：{len([node for node in nodes if node.ignored])}",
        f"- 自动抑制文本标题候选：{len(suppressed)}",
        "",
        "## 恢复结构",
        "",
        "| JSON页 | 印刷页 | 区域 | 类型 | 级别 | 标题 | 置信度 | 来源 |",
        "|---:|---:|---|---|---:|---|---:|---|",
    ]
    for node in active:
        if node.kind == "book":
            continue
        title = node.title_source.replace("|", "\\|")
        lines.append(
            f"| {node.page_json if node.page_json is not None else ''} | "
            f"{node.page_print if node.page_print is not None else ''} | {node.zone} | {node.kind} | "
            f"{node.level} | {title} | {node.confidence:.2f} | {node.source} |"
        )
    lines.extend(["", "## 待裁决项目", ""])
    if not packets:
        lines.append("无。")
    else:
        for packet in packets:
            lines.append(
                f"- JSON页 {packet.get('page_json')}：{packet.get('source_text')} "
                f"（{float(packet.get('confidence', 0)):.2f}）"
            )
    lines.extend(["", "## 验证警告", ""])
    warnings = validation.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- `{warning.get('code')}`：`{json.dumps(warning, ensure_ascii=False)}`")
    else:
        lines.append("无。")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
