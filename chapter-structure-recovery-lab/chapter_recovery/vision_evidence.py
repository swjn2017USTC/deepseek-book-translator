from __future__ import annotations

"""Deterministic consumer for externally collected vision structure evidence.

Network access belongs to the parent book workflow. This module only validates
and fuses a versioned sidecar into the existing OCR/PDF recovery pipeline, so a
vision result never becomes a second unaudited source of truth.
"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .alignment import title_similarity
from .models import Block, TocEntry, stable_id


ALLOWED_KINDS = {
    "part", "chapter", "section", "frontmatter", "backmatter", "toc_entry",
}
MATCHED_LABEL = "vision_heading"
UNMATCHED_LABEL = "vision_heading_unmatched"


def load_vision_evidence(
    path: Path,
    *,
    input_path: Optional[Path] = None,
) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version") or 0) != 1:
        raise ValueError("Unsupported vision evidence schema")
    if not isinstance(value.get("toc_pages", []), list):
        raise ValueError("vision toc_pages must be a list")
    if not isinstance(value.get("headings", []), list):
        raise ValueError("vision headings must be a list")
    if input_path is not None and value.get("input_sha256"):
        actual = sha256(input_path.read_bytes()).hexdigest()
        if actual != str(value.get("input_sha256")):
            raise RuntimeError("Vision evidence is stale for the current OCR JSON")
    pdf_value = value.get("pdf_path")
    pdf_sha = value.get("pdf_sha256")
    if pdf_value and pdf_sha:
        pdf_path = Path(str(pdf_value)).expanduser()
        if pdf_path.is_file():
            actual_pdf = sha256(pdf_path.read_bytes()).hexdigest()
            if actual_pdf != str(pdf_sha):
                raise RuntimeError("Vision evidence is stale for the current PDF")
    return value


def _kind(value: Any) -> str:
    kind = str(value or "toc_entry").strip().casefold()
    return kind if kind in ALLOWED_KINDS else "toc_entry"


def _page_label(item: Dict[str, Any]) -> str:
    value = str(item.get("page_label") or "").strip()
    if value:
        return value
    print_page = item.get("print_page")
    if isinstance(print_page, int) and print_page > 0:
        return str(print_page)
    return ""


def merge_vision_toc(
    parsed: List[TocEntry],
    evidence: Dict[str, Any],
    *,
    min_confidence: float = 0.72,
) -> List[TocEntry]:
    """Merge high-confidence vision TOC rows without replacing stronger OCR rows."""
    result = list(parsed)
    seen = {(item.title.casefold(), item.page_label.casefold()) for item in result}
    for page in evidence.get("toc_pages") or []:
        if not isinstance(page, dict) or not bool(page.get("is_toc")):
            continue
        try:
            page_json = int(page.get("page_json"))
        except (TypeError, ValueError):
            continue
        for item in page.get("entries") or []:
            if not isinstance(item, dict):
                continue
            try:
                confidence = float(item.get("confidence", page.get("confidence", 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_confidence:
                continue
            title = " ".join(str(item.get("title") or "").split()).strip()
            if not title or len(title) > 240:
                continue
            page_label = _page_label(item)
            print_page = int(page_label) if page_label.isdigit() else None
            kind = _kind(item.get("kind"))
            try:
                source_level = min(5, max(1, int(item.get("level") or 1)))
            except (TypeError, ValueError):
                source_level = 1

            best: Optional[TocEntry] = None
            best_score = 0.0
            for existing in result:
                score = title_similarity(title, existing.title)
                if page_label and existing.page_label and page_label == existing.page_label:
                    score += 0.06
                if kind == existing.kind:
                    score += 0.03
                if score > best_score:
                    best, best_score = existing, score
            if best is not None and best_score >= 0.78:
                if best.print_page is None and print_page is not None:
                    best.print_page = print_page
                    best.page_label = page_label
                if best.source_level is None:
                    best.source_level = source_level
                if "deepseek_vision_toc" not in best.provenance:
                    best.provenance = best.provenance + "+deepseek_vision_toc"
                    best.id = ""
                    best.__post_init__()
                continue

            key = (title.casefold(), page_label.casefold())
            if key in seen:
                continue
            seen.add(key)
            result.append(
                TocEntry(
                    title=title,
                    page_label=page_label,
                    print_page=print_page,
                    kind=kind,
                    source_page=page_json,
                    provenance="deepseek_vision_toc",
                    source_level=source_level,
                )
            )
    return result


def _bbox(item: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    value = item.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= v <= 1.0 for v in values):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    return values


def _best_block(page: Sequence[Block], title: str) -> Tuple[Optional[int], float]:
    best_index: Optional[int] = None
    best_score = 0.0
    for index, block in enumerate(page):
        if block.label in {"image", "image_body", "figure", "chart", "table", "formula"}:
            continue
        text = block.clean_content
        if not text or len(text) > 320:
            continue
        score = title_similarity(title, text)
        if score > best_score:
            best_index, best_score = index, score
    return best_index, best_score


def inject_vision_headings(
    pages: List[List[Block]],
    evidence: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, int]:
    """Promote OCR blocks corroborated by vision and add review-only misses.

    A vision title that matches OCR text is represented by the original OCR
    block/id with a stronger label and a level hint. A heading that vision sees
    but OCR cannot match is added only at a stricter confidence threshold and
    later receives sub-review confidence rather than automatic acceptance.
    """
    settings = dict(config.get("vision") or {})
    minimum = float(config.get("vision_heading_min_confidence", settings.get("heading_min_confidence", 0.72)))
    unmatched_minimum = float(config.get("vision_unmatched_heading_min_confidence", settings.get("unmatched_heading_min_confidence", 0.90)))
    match_floor = float(config.get("vision_heading_text_match", settings.get("heading_text_match", 0.58)))
    matched = 0
    synthetic_count = 0
    skipped = 0

    for ordinal, item in enumerate(evidence.get("headings") or []):
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
            page_index = int(item.get("page_json"))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if confidence < minimum or not (0 <= page_index < len(pages)):
            skipped += 1
            continue
        title = " ".join(str(item.get("title") or "").split()).strip()
        if not title or len(title) > 240:
            skipped += 1
            continue
        try:
            level = int(item.get("level") or 3)
        except (TypeError, ValueError):
            level = 3
        level = min(6, max(2, level))
        page = pages[page_index]
        block_index, score = _best_block(page, title)

        if block_index is not None and score >= match_floor:
            block = page[block_index]
            # Preserve the OCR wording. Vision contributes classification and a
            # hierarchy hint, not a rewritten source title.
            page[block_index] = replace(
                block,
                label=MATCHED_LABEL,
                content=("#" * level) + " " + block.clean_content,
            )
            matched += 1
            continue

        if confidence < unmatched_minimum:
            skipped += 1
            continue
        normalized_bbox = _bbox(item)
        width = page[0].page_width if page else 1000.0
        height = page[0].page_height if page else 1400.0
        if normalized_bbox is None:
            normalized_bbox = (0.10, 0.15, 0.90, 0.22)
        x1, y1, x2, y2 = normalized_bbox
        bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
        synthetic_block = Block(
                page_index=page_index,
                block_index=(max((block.block_index for block in page), default=-1) + 1),
                raw_block_id="vision-" + stable_id(page_index, ordinal, title),
                label=UNMATCHED_LABEL,
                content=("#" * level) + " " + title,
                bbox=bbox,
                order=None,
                page_width=width,
                page_height=height,
            )
        insertion = next(
            (index for index, block in enumerate(page) if block.bbox[1] > bbox[1]),
            len(page),
        )
        page.insert(insertion, synthetic_block)
        synthetic_count += 1

    return {"matched_ocr_blocks": matched, "synthetic_review_headings": synthetic_count, "skipped": skipped}
