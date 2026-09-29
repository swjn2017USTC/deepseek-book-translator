from __future__ import annotations

"""Reviewable OCR source manuscript and confirmation gate.

The OCR cleaner remains the provenance layer. This module creates a readable
Markdown manuscript after chapter recovery, performs a conservative second
cross-page merge, and lets a human edit heading text/levels and source text
before glossary generation or translation.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Dict, List, Sequence, Tuple

from .clean import SENTENCE_END, bbox as raw_bbox, raw_blocks, require_structure_gate
from .config import resolve_path
from .io_utils import read_jsonl, write_json, write_jsonl
from .models import Segment, digest
from .provenance import file_sha256, write_clean_manifest


SEGMENT_MARKER = re.compile(r"^<!-- DBT:SEG ([A-Za-z0-9._:-]+) -->\s*$", re.M)
PAGE_NOTICE = re.compile(r"^> \*\*〔OCR/PDF 页：[^〕]+〕\*\*\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CONTINUATION_START = re.compile(r"^[a-zäöüßà-ÿ,;:—–\-\)\]\}»”’]")
CONTINUATION_END = re.compile(r"[,;:—–\-]\s*$")


def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    base = Path(config["_base_dir"]).expanduser().resolve()
    source_dir = base / "source"
    raw = dict(config.get("source_review") or {})
    raw.setdefault("mode", "manual")
    raw.setdefault("directory", str(source_dir))
    raw.setdefault("markdown", str(source_dir / "structured_source.md"))
    raw.setdefault("state", str(source_dir / "source_state.json"))
    raw.setdefault("generated_segments", str(source_dir / "generated_segments.jsonl"))
    raw.setdefault("cross_page_merge", True)
    raw.setdefault("aggressive_cross_page_merge", False)
    return raw


def _path(config: Dict[str, Any], key: str) -> Path:
    value = Path(str(_settings(config)[key])).expanduser()
    return value.resolve() if value.is_absolute() else (Path(config["_base_dir"]) / value).resolve()


def applicable(config: Dict[str, Any]) -> bool:
    # Historical OCR projects predate this gate. Do not silently rewrite their
    # cleaned-segment source of truth on upgrade; they opt in when a
    # source_review block is explicitly added (new v0.10 projects always have one).
    return (
        str(config.get("source_adapter") or "") != "epub_native_v1"
        and bool(config.get("input_json"))
        and isinstance(config.get("source_review"), dict)
    )


def _source_and_structure_hashes(config: Dict[str, Any]) -> Tuple[str, str]:
    return (
        file_sha256(resolve_path(config, "input_json")),
        file_sha256(resolve_path(config, "structure_json")),
    )


def source_status(config: Dict[str, Any]) -> Dict[str, Any]:
    if not applicable(config):
        return {"applicable": False, "mode": "not_applicable", "status": "not_applicable"}
    settings = _settings(config)
    state_path = _path(config, "state")
    markdown = _path(config, "markdown")
    if not state_path.is_file():
        return {"applicable": True, "mode": str(settings["mode"]), "status": "not_generated", "markdown": str(markdown)}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"applicable": True, "mode": str(settings["mode"]), "status": "invalid_state", "markdown": str(markdown)}
    source_sha, structure_sha = _source_and_structure_hashes(config)
    stale = state.get("source_sha256") != source_sha or state.get("structure_sha256") != structure_sha
    status = "stale" if stale else str(state.get("status") or "not_generated")
    if status == "confirmed" and markdown.is_file():
        confirmed = str(state.get("confirmed_markdown_sha256") or "")
        if confirmed and file_sha256(markdown) != confirmed:
            status = "edited_after_confirmation"
    return {
        "applicable": True,
        "mode": str(state.get("mode") or settings["mode"]),
        "status": status,
        "markdown": str(markdown),
        "state": str(state_path),
        "cross_page_merged": int(state.get("cross_page_merged") or 0),
        "segment_count": int(state.get("segment_count") or 0),
    }


def require_source_confirmation(config: Dict[str, Any]) -> Dict[str, Any]:
    status = source_status(config)
    if not status["applicable"]:
        return status
    if status["status"] != "confirmed":
        raise RuntimeError(
            "Structured source manuscript is not confirmed; review source/structured_source.md "
            "and run confirm-source, or choose source confirmation mode 'auto'"
        )
    return status


def _page_metrics(config: Dict[str, Any], segments: Sequence[Segment]) -> Dict[int, Dict[str, float]]:
    result: Dict[int, Dict[str, float]] = {}
    try:
        pages = json.loads(resolve_path(config, "input_json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return result
    for page_index, page in enumerate(pages if isinstance(pages, list) else []):
        boxes = [raw_bbox(block) for block in raw_blocks(page)]
        if not boxes:
            continue
        result[page_index] = {
            "width": max(float(box[2]) for box in boxes) or 1.0,
            "height": max(float(box[3]) for box in boxes) or 1.0,
        }
    by_page: Dict[int, List[float]] = {}
    for segment in segments:
        if segment.kind != "paragraph" or segment.page_json is None:
            continue
        box = (segment.metadata or {}).get("bbox")
        if isinstance(box, list) and len(box) == 4:
            by_page.setdefault(int(segment.page_json), []).append(float(box[0]))
    for page, values in by_page.items():
        if page in result and values:
            result[page]["paragraph_x1"] = float(median(values))
    return result


def _page_span(segment: Segment) -> Tuple[int, int]:
    if segment.page_json is None:
        return (-1, -1)
    start = int(segment.page_json)
    end = int((segment.metadata or {}).get("last_page", start))
    span = (segment.metadata or {}).get("page_span")
    if isinstance(span, list) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            pass
    return start, max(start, end)


def _geometry_continuation(
    previous: Segment,
    current: Segment,
    metrics: Dict[int, Dict[str, float]],
    *,
    aggressive: bool,
) -> bool:
    _, p_end = _page_span(previous)
    c_start, _ = _page_span(current)
    if p_end < 0 or c_start != p_end + 1:
        return False
    p_box = (previous.metadata or {}).get("bbox")
    c_box = (current.metadata or {}).get("bbox")
    if not (isinstance(p_box, list) and len(p_box) == 4 and isinstance(c_box, list) and len(c_box) == 4):
        return False
    p_metric, c_metric = metrics.get(p_end), metrics.get(c_start)
    if not p_metric or not c_metric:
        return False
    bottom = float(p_box[3]) / max(1.0, p_metric["height"])
    top = float(c_box[1]) / max(1.0, c_metric["height"])
    if bottom < 0.86 or top > 0.22:
        return False
    if aggressive:
        return True
    continuation = bool(
        not SENTENCE_END.search(previous.source_text)
        or CONTINUATION_END.search(previous.source_text)
        or CONTINUATION_START.search(current.source_text.lstrip())
    )
    typical_x = c_metric.get("paragraph_x1")
    if typical_x is not None:
        tolerance = 0.025 * max(1.0, c_metric["width"])
        if float(c_box[0]) > typical_x + tolerance:
            return False
    return continuation


def merge_cross_page_segments(segments: Sequence[Segment], config: Dict[str, Any]) -> Tuple[List[Segment], int]:
    settings = _settings(config)
    if not bool(settings.get("cross_page_merge", True)):
        return [deepcopy(segment) for segment in segments], 0
    metrics = _page_metrics(config, segments)
    aggressive = bool(settings.get("aggressive_cross_page_merge", False))
    output: List[Segment] = []
    merged = 0
    for source in segments:
        segment = deepcopy(source)
        if not output:
            output.append(segment)
            continue
        previous = output[-1]
        compatible = (
            previous.kind == "paragraph"
            and segment.kind == "paragraph"
            and previous.translatable == segment.translatable
            and previous.zone == segment.zone
            and previous.parent_node_id == segment.parent_node_id
        )
        if not compatible or not _geometry_continuation(previous, segment, metrics, aggressive=aggressive):
            output.append(segment)
            continue
        left, right = previous.source_text, segment.source_text
        separator = " "
        if left.endswith("-") and right[:1].islower():
            left, separator = left[:-1], ""
        previous.source_text = left + separator + right
        previous.source_hash = digest(previous.source_text)
        previous.source_block_ids.extend(block for block in segment.source_block_ids if block not in previous.source_block_ids)
        p_start, _ = _page_span(previous)
        _, c_end = _page_span(segment)
        previous.metadata = dict(previous.metadata or {})
        previous.metadata["last_page"] = c_end
        previous.metadata["page_span"] = [p_start, c_end]
        ids = list(previous.metadata.get("source_manuscript_merged_segment_ids") or [previous.id])
        ids.extend(
            value for value in (segment.metadata or {}).get("source_manuscript_merged_segment_ids", [segment.id])
            if value not in ids
        )
        previous.metadata["source_manuscript_merged_segment_ids"] = ids
        previous.metadata["source_manuscript_cross_page_merge"] = True
        merged += 1
    return output, merged


def _page_label(segment: Segment) -> str:
    start, end = _page_span(segment)
    if start < 0:
        return "无页码"
    return str(start + 1) if start == end else f"{start + 1}–{end + 1}"


def _markdown_for(config: Dict[str, Any], segments: Sequence[Segment]) -> str:
    lines = [
        f"# {config.get('book_title') or config.get('book_id')}｜结构化原文",
        "",
        "> 本文件是翻译前的可审阅原文。你可以修改标题文字、Markdown 标题层级（# 数量）和正文；",
        "> 请不要删除或改写隐藏的 DBT:SEG 锚点。保存后运行 confirm-source，",
        "> 后续术语审核、chapter context 与 micro-batch 都会使用确认后的版本。",
        "",
        "> 页码采用 OCR/PDF 文件中的 1-based 页序；跨页段落会显示页码范围。",
        "",
    ]
    for segment in segments:
        lines.append(f"<!-- DBT:SEG {segment.id} -->")
        lines.append(f"> **〔OCR/PDF 页：{_page_label(segment)}〕**")
        if segment.kind == "heading":
            level = min(6, max(1, int(segment.level or 2)))
            lines.append(f"{'#' * level} {segment.source_text}")
        elif segment.kind == "image":
            lines.append(f"[图像占位：block_ids={','.join(segment.source_block_ids)}]")
        else:
            lines.append(segment.source_text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_source_manuscript(config: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    if not applicable(config):
        return {"status": "not_applicable", "mode": "not_applicable"}
    require_structure_gate(config)
    current = source_status(config)
    if current.get("status") in {"confirmed", "needs_confirmation"} and not force:
        return current
    output_dir = resolve_path(config, "output_dir")
    segments_path = output_dir / "cleaned_segments.jsonl"
    if not segments_path.is_file():
        raise FileNotFoundError("Run clean before building structured_source.md")
    segments = [Segment(**row) for row in read_jsonl(segments_path)]
    merged_segments, merge_count = merge_cross_page_segments(segments, config)
    markdown = _path(config, "markdown")
    generated = _path(config, "generated_segments")
    state_path = _path(config, "state")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(generated, [segment.to_dict() for segment in merged_segments])
    markdown.write_text(_markdown_for(config, merged_segments), encoding="utf-8")
    source_sha, structure_sha = _source_and_structure_hashes(config)
    mode = str(_settings(config).get("mode") or "manual")
    if mode not in {"manual", "auto"}:
        raise ValueError("source_review.mode must be 'manual' or 'auto'")
    state = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "mode": mode,
        "status": "needs_confirmation",
        "markdown": str(markdown),
        "generated_segments": str(generated),
        "source_sha256": source_sha,
        "structure_sha256": structure_sha,
        "generated_markdown_sha256": file_sha256(markdown),
        "generated_segments_sha256": file_sha256(generated),
        "cross_page_merged": merge_count,
        "segment_count": len(merged_segments),
    }
    write_json(state_path, state)
    if mode == "auto":
        return confirm_source_manuscript(config)
    return {
        "applicable": True, "mode": mode, "status": "needs_confirmation",
        "markdown": str(markdown), "state": str(state_path),
        "cross_page_merged": merge_count, "segment_count": len(merged_segments),
    }


def _content_by_marker(markdown: str, expected: Sequence[str]) -> Dict[str, str]:
    matches = list(SEGMENT_MARKER.finditer(markdown))
    identifiers = [match.group(1) for match in matches]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("structured_source.md contains duplicate DBT:SEG anchors")
    expected_set = set(expected)
    if set(identifiers) != expected_set:
        missing = sorted(expected_set - set(identifiers))
        extra = sorted(set(identifiers) - expected_set)
        raise ValueError(f"DBT:SEG anchors changed; missing={missing[:8]} extra={extra[:8]}")
    result: Dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        lines = [line for line in markdown[match.end():end].strip().splitlines() if not PAGE_NOTICE.match(line.strip())]
        result[match.group(1)] = "\n".join(lines).strip()
    return result


def _rebuild_structure(structure: Dict[str, Any], segments: List[Segment]) -> None:
    nodes = [node for node in (structure.get("nodes") or []) if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    root = next((str(node.get("id")) for node in nodes if node.get("kind") == "book"), None)
    stack: List[Tuple[int, str]] = []
    for segment in segments:
        if segment.kind == "heading":
            node_id = str((segment.metadata or {}).get("node_id") or "")
            level = min(6, max(1, int(segment.level or 2)))
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else root
            segment.parent_node_id = parent
            node = by_id.get(node_id)
            if node is not None:
                node["title_source"] = segment.source_text
                node["level"] = level
                node["parent_id"] = parent
                if str(node.get("kind") or "") in {"chapter", "section"}:
                    node["kind"] = "chapter" if parent == root else "section"
                    segment.metadata["node_kind"] = node["kind"]
            if node_id:
                stack.append((level, node_id))
        elif stack and segment.zone in {"mainmatter", "notes", "bibliography", "backmatter"}:
            segment.parent_node_id = stack[-1][1]


def confirm_source_manuscript(config: Dict[str, Any]) -> Dict[str, Any]:
    if not applicable(config):
        return {"status": "not_applicable", "mode": "not_applicable"}
    state_path = _path(config, "state")
    markdown_path = _path(config, "markdown")
    generated_path = _path(config, "generated_segments")
    if not state_path.is_file() or not markdown_path.is_file() or not generated_path.is_file():
        raise FileNotFoundError("Build source/structured_source.md before confirming it")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    source_sha, structure_sha = _source_and_structure_hashes(config)
    if state.get("source_sha256") != source_sha or state.get("structure_sha256") != structure_sha:
        raise RuntimeError("Structured source manuscript is stale; regenerate after the latest source/structure change")
    base_segments = [Segment(**row) for row in read_jsonl(generated_path)]
    content = _content_by_marker(markdown_path.read_text(encoding="utf-8"), [segment.id for segment in base_segments])
    updated: List[Segment] = []
    for base in base_segments:
        segment = deepcopy(base)
        block = content[segment.id]
        if segment.kind == "heading":
            first = next((line.strip() for line in block.splitlines() if line.strip()), "")
            match = HEADING.match(first)
            if not match:
                raise ValueError(f"Heading {segment.id} must remain a Markdown heading (# Title)")
            segment.level = len(match.group(1))
            segment.source_text = match.group(2).strip()
        elif segment.kind == "image":
            segment.source_text = base.source_text
        else:
            if not block.strip() and segment.translatable:
                raise ValueError(f"Translatable segment {segment.id} became empty")
            segment.source_text = block.strip()
        segment.source_hash = digest(segment.source_text)
        updated.append(segment)

    structure_path = resolve_path(config, "structure_json")
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    _rebuild_structure(structure, updated)
    write_json(structure_path, structure)

    output_dir = resolve_path(config, "output_dir")
    segments_path = output_dir / "cleaned_segments.jsonl"
    report_path = output_dir / "cleaning_report.json"
    write_jsonl(segments_path, [segment.to_dict() for segment in updated])
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
        "schema_version": 1, "book_id": config["book_id"], "metrics": {},
    }
    metrics = dict(report.get("metrics") or {})
    metrics["segments_after_source_review"] = len(updated)
    metrics["source_review_cross_page_merges"] = int(state.get("cross_page_merged") or 0)
    report["metrics"] = metrics
    report["source_manuscript"] = {"markdown": str(markdown_path), "mode": str(state.get("mode") or "manual"), "confirmed": True}
    write_json(report_path, report)

    state["status"] = "confirmed"
    state["confirmed_markdown_sha256"] = file_sha256(markdown_path)
    state["structure_sha256"] = file_sha256(structure_path)
    state["segment_count"] = len(updated)
    write_json(state_path, state)
    write_clean_manifest(config, segments_path, report_path)
    state["confirmed_segments_sha256"] = file_sha256(segments_path)
    write_json(state_path, state)
    write_clean_manifest(config, segments_path, report_path)
    return {
        "applicable": True, "mode": str(state.get("mode") or "manual"), "status": "confirmed",
        "markdown": str(markdown_path), "state": str(state_path), "segment_count": len(updated),
        "cross_page_merged": int(state.get("cross_page_merged") or 0),
    }
