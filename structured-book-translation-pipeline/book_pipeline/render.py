from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .config import resolve_path
from .io_utils import read_jsonl, write_json
from .models import Segment
from .clean import require_structure_gate
from .provenance import require_fresh_cleaning
from .authenticity import reject_demo_translations
from .translate import translation_status


def render_book(config: Dict[str, Any], allow_partial: bool = False) -> Dict[str, Any]:
    require_structure_gate(config)
    require_fresh_cleaning(config)
    output_dir = resolve_path(config, "output_dir")
    status = translation_status(config)
    if not status["ready_to_render"] and not allow_partial:
        raise RuntimeError(f"Translation incomplete: {status['remaining_segments']} segments remain")
    segments = [Segment(**row) for row in read_jsonl(output_dir / "cleaned_segments.jsonl")]
    translation_rows = list(read_jsonl(output_dir / "translations.jsonl"))
    reject_demo_translations(config, translation_rows)
    translations = {str(row["segment_id"]): row for row in translation_rows}
    lines: List[str] = [f"# {config['book_title_zh']}", ""]
    if config.get("author"):
        lines += [str(config["author"]), ""]
    last_page = None
    footnotes: List[str] = []
    missing: List[str] = []
    for segment in segments:
        record = translations.get(segment.id)
        current = bool(record and record.get("source_hash") == segment.source_hash and record.get("status") == "completed")
        if segment.translatable and not current:
            missing.append(segment.id)
        text = str(record["translated_text"]) if current else segment.source_text
        if segment.page_json is not None and segment.page_json != last_page:
            lines += [f"<!-- PAGE_JSON: {segment.page_json:04d} -->", ""]
            last_page = segment.page_json
        if segment.kind == "heading":
            lines += [f"{'#' * max(2, min(6, int(segment.level or 2)))} {text}", ""]
        elif segment.kind == "footnote":
            footnotes.append(f"[^{segment.metadata['reference']}]: {text}")
        elif segment.kind == "image":
            ids = ",".join(segment.source_block_ids)
            lines += [f"<!-- IMAGE block_ids={ids} bbox={segment.metadata.get('bbox')} -->", ""]
        elif segment.kind == "caption":
            lines += [f"*{text}*", ""]
        else:
            lines += [text, ""]
    if footnotes:
        lines += ["---", "", "## 脚注", ""] + footnotes + [""]
    output_path = output_dir / "book_translated.md"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    report = {**status, "output": str(output_path), "partial": bool(missing), "missing_segment_ids": missing}
    write_json(output_dir / "render_report.json", report)
    return report
