from __future__ import annotations

"""Reviewed-source Markdown gate between structure recovery and translation.

The generated Markdown is both human-readable and machine-roundtrippable:
stable BOOK_SEGMENT comments preserve identity while visible headings/text can
be corrected. Translation always consumes the approved canonical segments
rather than a display-only export.
"""

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import resolve_path, source_input_key
from .io_utils import read_jsonl, write_json, write_jsonl
from .models import Segment, digest
from .provenance import file_sha256, write_clean_manifest


SEGMENT_MARKER = re.compile(r"^<!-- BOOK_SEGMENT (\{.*\}) -->\s*$")
PAGE_MARKER = re.compile(r"^<!-- SOURCE_PAGE .* -->\s*$")
VISIBLE_PAGE = re.compile(r"^> \*\*\[Source page(?:s)?: .*\]\*\*\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _paths(config: Dict[str, Any], project_dir: Path) -> Dict[str, Path]:
    output = resolve_path(config, "output_dir")
    return {
        "markdown": project_dir / "source_review.md",
        "state": project_dir / "source_review_state.json",
        "segments": output / "cleaned_segments.jsonl",
        "report": output / "cleaning_report.json",
        "structure": resolve_path(config, "structure_json"),
    }


def _fingerprint(config: Dict[str, Any], project_dir: Path) -> Dict[str, str]:
    source = resolve_path(config, source_input_key(config))
    structure = resolve_path(config, "structure_json")
    baseline = project_dir / "structure" / "book_structure.json"
    return {
        "source_sha256": file_sha256(source),
        "baseline_structure_sha256": file_sha256(baseline) if baseline.is_file() else file_sha256(structure),
        "structure_sha256": file_sha256(structure),
    }


def _page_end(segment: Segment) -> Optional[int]:
    metadata = segment.metadata or {}
    value = metadata.get("page_end", metadata.get("last_page", segment.page_json))
    return int(value) if value is not None else segment.page_json


def _page_label(segment: Segment) -> str:
    start = segment.page_json
    end = _page_end(segment)
    if start is None:
        metadata = segment.metadata or {}
        href = str(metadata.get("href") or "")
        return href or "native"
    if end is not None and end != start:
        return f"json {start}–{end}"
    return f"json {start}"


def _marker(segment: Segment) -> str:
    metadata = segment.metadata or {}
    payload = {
        "id": segment.id,
        "kind": segment.kind,
        "page_json": segment.page_json,
        "page_end": _page_end(segment),
        "zone": segment.zone,
        "parent_node_id": segment.parent_node_id,
        "level": segment.level,
        "node_id": metadata.get("node_id"),
        "node_kind": metadata.get("node_kind"),
    }
    return "<!-- BOOK_SEGMENT " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + " -->"


def render_source_review(config: Dict[str, Any], project_dir: Path) -> Dict[str, Any]:
    paths = _paths(config, project_dir)
    segments = [Segment(**row) for row in read_jsonl(paths["segments"])]
    lines: List[str] = [
        f"# {config.get('book_title', config.get('book_id', 'Book'))}",
        "",
        "<!-- REVIEW_INSTRUCTIONS",
        "Edit visible source text and Markdown heading levels if needed.",
        "Do not delete or duplicate BOOK_SEGMENT comments; they preserve stable IDs.",
        "Page markers are informational and are ignored when importing.",
        "-->",
        "",
    ]
    last_page: Optional[Tuple[Optional[int], Optional[int]]] = None
    for segment in segments:
        span = (segment.page_json, _page_end(segment))
        if span != last_page:
            lines.append(
                "<!-- SOURCE_PAGE "
                + json.dumps({"page_json": span[0], "page_end": span[1]}, ensure_ascii=False, separators=(",", ":"))
                + " -->"
            )
            lines.append(f"> **[Source page{'s' if span[1] != span[0] else ''}: {_page_label(segment)}]**")
            lines.append("")
            last_page = span
        lines.append(_marker(segment))
        text = segment.source_text.strip()
        if segment.kind == "heading":
            level = min(6, max(1, int(segment.level or 2)))
            lines.append("#" * level + " " + text)
        else:
            lines.append(text)
        lines.append("")
    markdown = "\n".join(lines).rstrip() + "\n"
    paths["markdown"].write_text(markdown, encoding="utf-8")
    fp = _fingerprint(config, project_dir)
    state = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "status": "awaiting_choice",
        **fp,
        "markdown": str(paths["markdown"]),
        "markdown_sha256": file_sha256(paths["markdown"]),
        "segments_sha256": file_sha256(paths["segments"]),
        "review_mode": None,
    }
    write_json(paths["state"], state)
    return state


def source_review_state(config: Dict[str, Any], project_dir: Path) -> Optional[Dict[str, Any]]:
    paths = _paths(config, project_dir)
    if not paths["state"].is_file():
        return None
    try:
        state = json.loads(paths["state"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    fp = _fingerprint(config, project_dir)
    if (
        state.get("source_sha256") != fp["source_sha256"]
        or state.get("baseline_structure_sha256") != fp["baseline_structure_sha256"]
        or state.get("structure_sha256") != fp["structure_sha256"]
    ):
        return None
    if not paths["markdown"].is_file() or not paths["segments"].is_file():
        return None
    return state


def require_approved_source(config: Dict[str, Any], project_dir: Path) -> Dict[str, Any]:
    state = source_review_state(config, project_dir)
    if not state or state.get("status") != "approved":
        raise RuntimeError("Reviewed source Markdown is not approved; resolve the source-review gate before translation")
    paths = _paths(config, project_dir)
    if file_sha256(paths["segments"]) != state.get("segments_sha256"):
        raise RuntimeError("Approved source segments changed; re-apply or re-approve source_review.md")
    return state


def choose_source_review(config: Dict[str, Any], project_dir: Path, mode: str) -> Dict[str, Any]:
    if mode not in {"manual", "auto"}:
        raise ValueError("source review mode must be manual or auto")
    paths = _paths(config, project_dir)
    state = source_review_state(config, project_dir)
    if state is None:
        state = render_source_review(config, project_dir)
    state["review_mode"] = mode
    if mode == "manual":
        state["status"] = "awaiting_manual_review"
    else:
        state["status"] = "approved"
        state["markdown_sha256"] = file_sha256(paths["markdown"])
        state["segments_sha256"] = file_sha256(paths["segments"])
    write_json(paths["state"], state)
    return state


def _split_markdown(path: Path) -> List[Tuple[Dict[str, Any], str]]:
    rows: List[Tuple[Dict[str, Any], str]] = []
    current: Optional[Dict[str, Any]] = None
    body: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = SEGMENT_MARKER.match(raw)
        if match:
            if current is not None:
                rows.append((current, "\n".join(body).strip()))
            current = json.loads(match.group(1))
            body = []
            continue
        if current is None:
            continue
        if PAGE_MARKER.match(raw) or VISIBLE_PAGE.match(raw):
            continue
        body.append(raw)
    if current is not None:
        rows.append((current, "\n".join(body).strip()))
    return rows


def _update_reviewed_structure(structure: Dict[str, Any], segments: Sequence[Segment]) -> None:
    nodes = [node for node in (structure.get("nodes") or []) if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in nodes if node.get("id")}
    root = next((node for node in nodes if node.get("kind") == "book"), None)
    root_id = str(root.get("id")) if root else None
    stack: List[Tuple[int, str]] = []
    current_heading: Optional[str] = None
    for segment in segments:
        metadata = segment.metadata or {}
        if segment.kind == "heading":
            level = min(6, max(1, int(segment.level or 2)))
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else root_id
            node_id = str(metadata.get("node_id") or "")
            segment.parent_node_id = parent_id
            if node_id:
                node = by_id.get(node_id)
                if node is not None:
                    node["title_source"] = segment.source_text
                    node["level"] = level
                    node["parent_id"] = parent_id
                    node["manual_source_review"] = True
                stack.append((level, node_id))
                current_heading = node_id
            continue
        if current_heading:
            segment.parent_node_id = current_heading


def apply_source_review(
    config: Dict[str, Any],
    project_dir: Path,
    markdown_path: Optional[Path] = None,
) -> Dict[str, Any]:
    paths = _paths(config, project_dir)
    source = (markdown_path or paths["markdown"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    original = [Segment(**row) for row in read_jsonl(paths["segments"])]
    original_by_id = {segment.id: segment for segment in original}
    parsed = _split_markdown(source)
    ids = [str(meta.get("id") or "") for meta, _ in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("source_review.md contains duplicate BOOK_SEGMENT ids")
    if set(ids) != set(original_by_id):
        missing = sorted(set(original_by_id) - set(ids))
        extra = sorted(set(ids) - set(original_by_id))
        raise ValueError(f"source_review.md must preserve every segment marker; missing={missing[:5]} extra={extra[:5]}")

    reviewed: List[Segment] = []
    for meta, body in parsed:
        segment = original_by_id[str(meta["id"])]
        if segment.kind == "heading":
            match = HEADING.match(body)
            if not match:
                raise ValueError(f"Heading segment {segment.id} must remain a Markdown heading")
            segment.level = len(match.group(1))
            text = match.group(2).strip()
        else:
            text = body.strip()
        if segment.translatable and not text:
            raise ValueError(f"Translatable segment {segment.id} cannot be empty")
        segment.source_text = text
        segment.source_hash = digest(text)
        segment.metadata = dict(segment.metadata or {})
        segment.metadata["source_reviewed"] = True
        reviewed.append(segment)

    structure = json.loads(paths["structure"].read_text(encoding="utf-8"))
    _update_reviewed_structure(structure, reviewed)
    reviewed_structure = resolve_path(config, "output_dir") / "reviewed_structure.json"
    write_json(reviewed_structure, structure)
    write_jsonl(paths["segments"], [segment.to_dict() for segment in reviewed])

    # Translation uses the reviewed tree; the chapter analyzer keeps owning the
    # baseline structure file so a future baseline change invalidates this gate.
    config_path = Path(str(config["_config_path"])).expanduser().resolve()
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["structure_json"] = str(reviewed_structure)
    write_json(config_path, raw_config)
    config["structure_json"] = str(reviewed_structure)
    paths = _paths(config, project_dir)

    # Rebind cleaning provenance to the manually reviewed canonical source.
    write_clean_manifest(config, paths["segments"], paths["report"])
    if source != paths["markdown"]:
        paths["markdown"].write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    state = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "status": "approved",
        "review_mode": "manual",
        **_fingerprint(config, project_dir),
        "markdown": str(paths["markdown"]),
        "markdown_sha256": file_sha256(paths["markdown"]),
        "segments_sha256": file_sha256(paths["segments"]),
    }
    write_json(paths["state"], state)
    return state
