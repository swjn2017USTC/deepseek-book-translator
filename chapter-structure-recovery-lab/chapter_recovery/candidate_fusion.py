"""Heading candidate fusion from four deterministic sources (P02, plan D7).

Sources, swept in fixed order:
  (1) Paddle macro blocks  -- label in {doc_title, paragraph_title, header}
  (2) PDF bookmarks/TOC    -- native outline, page resolved via page_map
  (3) Font-signal lines    -- native large-font lines (threshold parameterized)
  (4) text-labeled blocks  -- only when supported by native evidence

Candidates are evidence, never decisions: suppression rules only *annotate*
(``deterministic_suppressions``); nothing is dropped after source gates.
Duplicates (same resolved ``page_json`` + canonical text) merge per D7;
ordering is deterministic (A3: null-page candidates sort last).  ``bbox`` is
kept in normalized [0, 1] coordinates, rounded to 4 decimals.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .alignment import MACRO_LABELS, normalized_title, title_similarity
from .evidence_models import (
    Bookmark,
    EvidenceCandidate,
    PdfEvidenceResult,
    make_candidate_id,
)
from .heading_text import canonical_heading_text, numbering, rejection_reason
from .models import Block, TocEntry
from .page_map import PageMap
from .pdf_evidence import PdfLine, large_line_threshold
from .visual_context import heading_visual_context

SpanAccessor = Callable[[int, Sequence[float]], Optional[List[Dict[str, Any]]]]

APPARATUS_ZONES = {"notes", "bibliography", "index"}
LABEL_RANK = {"doc_title": 0, "paragraph_title": 1, "header": 2, "text": 3}
_TOC_SIMILARITY_FLOOR = 0.90
_TEXT_SUPPORT_SIMILARITY_FLOOR = 0.90
_TEXT_SUPPORT_OVERLAP_FLOOR = 0.50

BBox = Tuple[float, float, float, float]


def _round_bbox(bbox: Optional[BBox]) -> Optional[BBox]:
    if bbox is None:
        return None
    return tuple(round(float(v), 4) for v in bbox)  # type: ignore[return-value]


def _sweep_key(block: Block) -> Tuple[Any, ...]:
    return (
        block.page_index,
        block.order if block.order is not None else float("inf"),
        block.bbox[1],
        block.bbox[0],
        block.id,
    )


def _sweep_blocks(pages: List[List[Block]], labels: set) -> List[Block]:
    blocks = [block for page in pages for block in page if block.label in labels]
    blocks.sort(key=_sweep_key)
    return blocks


def _print_page_for(page_json: Optional[int], page_map: PageMap) -> Optional[int]:
    if page_json is None or page_map.offset is None:
        return None
    return int(page_json + page_map.offset)


def _block_context(
    block: Block, pages: List[List[Block]]
) -> Tuple[Optional[str], Optional[str]]:
    page_blocks = pages[block.page_index]
    try:
        index = page_blocks.index(block)
    except ValueError:
        return None, None
    before = page_blocks[index - 1].clean_content[:80] if index > 0 else None
    after = page_blocks[index + 1].clean_content[:80] if index + 1 < len(page_blocks) else None
    return before, after


def _running_header_titles(pages: List[List[Block]], threshold: int) -> Dict[str, int]:
    """Normalized titles of header blocks repeated >= threshold across >=80% spread."""
    stats: Dict[str, List[int]] = {}
    for page_index, blocks in enumerate(pages):
        for block in blocks:
            if block.label != "header":
                continue
            key = normalized_title(block.clean_content)
            if not key:
                continue
            item = stats.setdefault(key, [0, page_index, page_index])
            item[0] += 1
            item[1] = min(item[1], page_index)
            item[2] = max(item[2], page_index)
    page_count = max(1, len(pages))
    return {
        key: item[0]
        for key, item in stats.items()
        if item[0] >= threshold and (item[2] - item[1] + 1) / page_count >= 0.80
    }


def _annotations_for_block(
    block: Block, text: str, zone: str, running_headers: Dict[str, int], pages: List[List[Block]]
) -> List[str]:
    annotations: List[str] = []
    reason = rejection_reason(text)
    if reason:
        annotations.append(reason)
    visual = heading_visual_context(block, pages[block.page_index])
    if visual.rejection_reason:
        annotations.append(visual.rejection_reason)
    if zone in APPARATUS_ZONES:
        annotations.append("apparatus_zone")
    if block.label == "header" and normalized_title(text) in running_headers:
        annotations.append("running_header_repeat")
    return sorted(set(annotations))


def _resolve_bookmark_page(
    bookmark: Bookmark, page_map: PageMap, page_count: int
) -> Optional[int]:
    if bookmark.page <= 0:
        return None
    resolved = page_map.json_page(bookmark.page - 1)
    if resolved is None or not (0 <= resolved < page_count):
        return None
    return resolved


def _resolved_page_matches(
    block_page: int, bookmark: Bookmark, page_map: PageMap, page_count: int
) -> bool:
    resolved = _resolve_bookmark_page(bookmark, page_map, page_count)
    return resolved is not None and resolved == block_page


def _line_overlap_ratio(block: Block, line: PdfLine) -> float:
    """Fraction of the native line bbox covered by the OCR block bbox."""
    bx1, by1, bx2, by2 = block.normalized_bbox()
    lx1, ly1, lx2, ly2 = line.bbox
    ix = min(bx2, lx2) - max(bx1, lx1)
    iy = min(by2, ly2) - max(by1, ly1)
    intersection = max(0.0, ix) * max(0.0, iy)
    line_area = max(1e-9, (lx2 - lx1) * (ly2 - ly1))
    return intersection / line_area


def _text_support_reasons(
    block: Block,
    text: str,
    bookmarks: List[Bookmark],
    large_lines_by_page: Dict[int, List[PdfLine]],
    page_medians: Dict[int, float],
    page_map: PageMap,
    page_count: int,
    font_signal_min_size: float,
    font_signal_ratio: float,
    span_accessor: Optional[SpanAccessor],
) -> List[str]:
    """Blind-spot support check for text-labeled blocks (hegel p115 path).

    A text block qualifies iff a PDF bookmark with >= 0.90 title similarity
    resolves to its page, or a large native line overlaps >= 50% of its bbox,
    or (bounded rescue for lines dropped by the per-page cap) an on-demand
    span in its bbox is large enough.
    """
    reasons: List[str] = []
    for bookmark in bookmarks:
        if (
            title_similarity(text, bookmark.title) >= _TEXT_SUPPORT_SIMILARITY_FLOOR
            and _resolved_page_matches(block.page_index, bookmark, page_map, page_count)
        ):
            reasons.append("bookmark")
            break
    median = page_medians.get(block.page_index, 0.0)
    threshold = large_line_threshold(median, font_signal_min_size, font_signal_ratio)
    for line in large_lines_by_page.get(block.page_index, []):
        if line.font_size >= threshold and _line_overlap_ratio(block, line) >= _TEXT_SUPPORT_OVERLAP_FLOOR:
            reasons.append("font")
            break
    if "font" not in reasons and span_accessor is not None:
        spans = span_accessor(block.page_index, block.normalized_bbox())
        if spans and any(float(span["size"]) >= threshold for span in spans):
            reasons.append("font")
    return sorted(set(reasons))


def _paddle_candidates(
    pages: List[List[Block]],
    zones: Dict[int, str],
    running_headers: Dict[str, int],
    page_map: PageMap,
    labels: set,
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for block in _sweep_blocks(pages, labels):
        text = canonical_heading_text(block.clean_content)
        if not text:
            continue
        zone = zones.get(block.page_index, "mainmatter")
        before, after = _block_context(block, pages)
        candidates.append(
            EvidenceCandidate(
                candidate_id="",
                source_block_ids=[block.id],
                exact_text=text,
                page_json=block.page_index,
                page_print=_print_page_for(block.page_index, page_map),
                bbox=_round_bbox(block.normalized_bbox()),
                zone=zone,
                paddle_label=block.label,
                before_context=before,
                after_context=after,
                deterministic_suppressions=_annotations_for_block(
                    block, text, zone, running_headers, pages
                ),
                evidence_refs=[f"paddle:{block.id}:{block.label}"],
            )
        )
    return candidates


def _bookmark_candidates(
    bookmarks: List[Bookmark],
    zones: Dict[int, str],
    page_map: PageMap,
    page_count: int,
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for index, bookmark in enumerate(bookmarks):
        title = canonical_heading_text(bookmark.title)
        if not title:
            continue
        json_page = _resolve_bookmark_page(bookmark, page_map, page_count)
        zone = zones.get(json_page, "mainmatter") if json_page is not None else "mainmatter"
        annotations: List[str] = []
        reason = rejection_reason(title)
        if reason:
            annotations.append(reason)
        if zone in APPARATUS_ZONES:
            annotations.append("apparatus_zone")
        candidates.append(
            EvidenceCandidate(
                candidate_id="",
                source_block_ids=[],
                exact_text=title,
                page_json=json_page,
                page_print=bookmark.page,
                bbox=None,
                zone=zone,
                paddle_label=None,
                pdf_bookmark={
                    "level": bookmark.level,
                    "title": bookmark.title,
                    "page": bookmark.page,
                },
                deterministic_suppressions=sorted(set(annotations)),
                evidence_refs=[f"pdf:bookmark:{index}"],
            )
        )
    return candidates


def _font_candidates(
    pdf_evidence: PdfEvidenceResult,
    zones: Dict[int, str],
    page_map: PageMap,
    page_count: int,
    font_signal_min_size: float,
    font_signal_ratio: float,
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for page_evidence in pdf_evidence.pages:
        threshold = large_line_threshold(
            page_evidence.font_size_median, font_signal_min_size, font_signal_ratio
        )
        for idx, line in enumerate(page_evidence.large_lines):
            if line.font_size < threshold:
                continue
            text = canonical_heading_text(line.text)
            if not text or rejection_reason(text) is not None:
                continue
            page_json = line.page_index if line.page_index < page_count else None
            zone = zones.get(line.page_index, "mainmatter")
            candidates.append(
                EvidenceCandidate(
                    candidate_id="",
                    source_block_ids=[],
                    exact_text=text,
                    page_json=page_json,
                    page_print=(
                        _print_page_for(line.page_index, page_map)
                        if page_json is not None
                        else None
                    ),
                    bbox=line.bbox,
                    zone=zone,
                    native_font={
                        "family": line.font_family,
                        "size": line.font_size,
                        "bold": line.is_bold,
                        "italic": line.is_italic,
                    },
                    evidence_refs=[f"pdf:font:{line.page_index}:{idx}"],
                )
            )
    return candidates


def _text_support_candidates(
    pages: List[List[Block]],
    zones: Dict[int, str],
    running_headers: Dict[str, int],
    page_map: PageMap,
    bookmarks: List[Bookmark],
    large_lines_by_page: Dict[int, List[PdfLine]],
    page_medians: Dict[int, float],
    page_count: int,
    font_signal_min_size: float,
    font_signal_ratio: float,
    span_accessor: Optional[SpanAccessor],
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    text_blocks = _sweep_blocks(pages, {"text"})
    for block in text_blocks:
        text = canonical_heading_text(block.clean_content)
        if not text or rejection_reason(text) is not None:
            continue
        reasons = _text_support_reasons(
            block,
            text,
            bookmarks,
            large_lines_by_page,
            page_medians,
            page_map,
            page_count,
            font_signal_min_size,
            font_signal_ratio,
            span_accessor,
        )
        if not reasons:
            continue
        zone = zones.get(block.page_index, "mainmatter")
        before, after = _block_context(block, pages)
        refs = [f"paddle:{block.id}:text"]
        refs.extend(f"pdf:text_support:{block.id}:{reason}" for reason in reasons)
        candidates.append(
            EvidenceCandidate(
                candidate_id="",
                source_block_ids=[block.id],
                exact_text=text,
                page_json=block.page_index,
                page_print=_print_page_for(block.page_index, page_map),
                bbox=_round_bbox(block.normalized_bbox()),
                zone=zone,
                paddle_label="text",
                before_context=before,
                after_context=after,
                deterministic_suppressions=_annotations_for_block(
                    block, text, zone, running_headers, pages
                ),
                evidence_refs=sorted(set(refs)),
            )
        )
    return candidates


def _union_bbox(group: List[EvidenceCandidate]) -> Optional[BBox]:
    present = [cand.bbox for cand in group if cand.bbox is not None]
    if not present:
        return None
    return _round_bbox(
        (
            min(item[0] for item in present),
            min(item[1] for item in present),
            max(item[2] for item in present),
            max(item[3] for item in present),
        )
    )


def _strongest_fields(group: List[EvidenceCandidate]) -> Tuple[Any, ...]:
    """Per D7: strongest present source fills the label/bookmark/font fields."""
    macro_labels = [
        cand.paddle_label
        for cand in group
        if cand.paddle_label in MACRO_LABELS
    ]
    has_bookmark = any(cand.pdf_bookmark is not None for cand in group)
    has_font = any(cand.native_font is not None for cand in group)
    has_text_support = any(cand.paddle_label == "text" for cand in group)
    if macro_labels:
        macro_labels.sort(key=lambda label: LABEL_RANK[label])
        return ("paddle", macro_labels[0], None, None)
    if has_bookmark:
        bookmark = next(cand.pdf_bookmark for cand in group if cand.pdf_bookmark is not None)
        return ("bookmark", None, bookmark, None)
    if has_font:
        font = next(cand.native_font for cand in group if cand.native_font is not None)
        return ("font", None, None, font)
    if has_text_support:
        return ("text_support", "text", None, None)
    return ("paddle", None, None, None)


def _merge_candidates(raw: List[EvidenceCandidate], book_id: str) -> List[EvidenceCandidate]:
    """D7 dedup + merge on (page_json, canonical text casefold)."""
    groups: "Dict[Tuple[Optional[int], str], List[EvidenceCandidate]]" = {}
    order: List[Tuple[Optional[int], str]] = []
    for cand in raw:
        key = (cand.page_json, canonical_heading_text(cand.exact_text).casefold())
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(cand)

    merged: List[EvidenceCandidate] = []
    for key in order:
        group = groups[key]
        _, paddle_label, bookmark, font = _strongest_fields(group)
        block_ids = sorted({bid for cand in group for bid in cand.source_block_ids})
        refs = sorted({ref for cand in group for ref in cand.evidence_refs})
        suppressions = sorted(
            {reason for cand in group for reason in cand.deterministic_suppressions}
        )
        context_candidate = next(
            (cand for cand in group if cand.source_block_ids), group[0]
        )
        page_json = key[0]
        exact_text = canonical_heading_text(group[0].exact_text)
        candidate = EvidenceCandidate(
            candidate_id="",
            source_block_ids=block_ids,
            exact_text=exact_text,
            page_json=page_json,
            page_print=context_candidate.page_print,
            bbox=_union_bbox(group),
            zone=context_candidate.zone,
            paddle_label=paddle_label,
            pdf_bookmark=bookmark,
            native_font=font,
            before_context=context_candidate.before_context,
            after_context=context_candidate.after_context,
            deterministic_suppressions=suppressions,
            evidence_refs=refs,
        )
        candidate.candidate_id = make_candidate_id(
            book_id, page_json, block_ids, exact_text
        )
        merged.append(candidate)

    def _sort_key(cand: EvidenceCandidate) -> Tuple[Any, ...]:
        return (
            cand.page_json if cand.page_json is not None else float("inf"),
            cand.bbox[1] if cand.bbox is not None else 0.0,
            cand.bbox[0] if cand.bbox is not None else 0.0,
            cand.candidate_id,
        )

    merged.sort(key=_sort_key)
    return merged


def _numbering_dict(text: str) -> Optional[Dict[str, Any]]:
    parsed = numbering(text)
    if parsed is None:
        return None
    return {
        "token": parsed.token,
        "path": list(parsed.path),
        "depth": parsed.depth,
        "family": parsed.family,
    }


def _attach_toc_matches(
    candidates: List[EvidenceCandidate], toc_entries: List[TocEntry]
) -> None:
    if not toc_entries:
        return
    for candidate in candidates:
        matches: List[Dict[str, Any]] = []
        text = candidate.exact_text
        for entry in toc_entries:
            similarity = title_similarity(text, entry.title)
            if similarity < _TOC_SIMILARITY_FLOOR:
                continue
            kind = (
                "exact"
                if normalized_title(text) == normalized_title(entry.title)
                else "fuzzy"
            )
            matches.append(
                {
                    "toc_entry_id": entry.id,
                    "title": entry.title,
                    "page_label": entry.page_label,
                    "match_score": round(similarity, 4),
                    "match_kind": kind,
                }
            )
        candidate.toc_matches = matches


def _attach_repetition(
    candidates: List[EvidenceCandidate], repetition_window: int
) -> None:
    pages_by_text: "Dict[str, List[Optional[int]]]" = {}
    for candidate in candidates:
        key = canonical_heading_text(candidate.exact_text).casefold()
        pages_by_text.setdefault(key, []).append(candidate.page_json)
    for candidate in candidates:
        key = canonical_heading_text(candidate.exact_text).casefold()
        pages = pages_by_text[key]
        nearby = 0
        for other_page in pages:
            if (
                other_page is not None
                and candidate.page_json is not None
                and abs(other_page - candidate.page_json) <= repetition_window
            ):
                nearby += 1
        candidate.repetition = {"count": len(pages), "nearby_count": nearby}


def build_heading_candidates(
    pages: List[List[Block]],
    page_map: PageMap,
    zones: Dict[int, str],
    toc_entries: List[TocEntry],
    pdf_evidence: Optional[PdfEvidenceResult],
    span_accessor: Optional[SpanAccessor],
    config: Dict[str, Any],
    book_id: str,
) -> List[EvidenceCandidate]:
    """Union of the four deterministic candidate sources, merged and ordered.

    ``span_accessor`` is only consulted by the text-support gate (source d);
    pass None when no PDF is available.
    """
    page_count = len(pages)
    min_size = float(config.get("font_signal_min_size", 14.0))
    ratio = float(config.get("font_signal_ratio", 1.15))
    repetition_window = int(config.get("repetition_window", 10))
    running_threshold = int(config.get("running_header_threshold", 5))

    running_headers = _running_header_titles(pages, running_threshold)

    bookmarks = pdf_evidence.bookmarks if pdf_evidence is not None else []
    large_lines_by_page: Dict[int, List[PdfLine]] = {}
    page_medians: Dict[int, float] = {}
    if pdf_evidence is not None:
        for page_evidence in pdf_evidence.pages:
            large_lines_by_page[page_evidence.page_index] = list(page_evidence.large_lines)
            page_medians[page_evidence.page_index] = page_evidence.font_size_median

    raw: List[EvidenceCandidate] = []
    raw.extend(_paddle_candidates(pages, zones, running_headers, page_map, MACRO_LABELS))
    raw.extend(_bookmark_candidates(bookmarks, zones, page_map, page_count))
    if pdf_evidence is not None:
        raw.extend(
            _font_candidates(pdf_evidence, zones, page_map, page_count, min_size, ratio)
        )
    raw.extend(
        _text_support_candidates(
            pages,
            zones,
            running_headers,
            page_map,
            bookmarks,
            large_lines_by_page,
            page_medians,
            page_count,
            min_size,
            ratio,
            span_accessor,
        )
    )

    candidates = _merge_candidates(raw, book_id)
    for candidate in candidates:
        candidate.numbering = _numbering_dict(candidate.exact_text)
    _attach_toc_matches(candidates, toc_entries)
    _attach_repetition(candidates, repetition_window)
    return candidates
