"""Native PDF evidence extraction via PyMuPDF (P02, lazy fitz import).

Everything here is additive and deterministic.  ``fitz`` is imported lazily so
the module (and the whole ``chapter_recovery`` package) imports cleanly on
machines without PyMuPDF.

Extraction conventions
----------------------
- Per-page aggregates only plus a bounded ``large_lines`` list (<= 20 lines
  per page, canonical text <= 180 chars).  The full per-line span dump is
  deliberately OFF: spans are fetched on demand via ``make_span_accessor``.
- ``PdfLine.bbox`` is normalized to the containing PDF page ([0, 1] space).
- Font-signal threshold: a line is "large" iff its font size satisfies
  ``size >= max(font_signal_min_size, font_signal_ratio * page_font_median)``
  (defaults 14.0 pt and 1.15), parameterized per config so candidate fusion
  and extraction always agree.
- Bookmark pages are 1-based PDF page numbers (PyMuPDF convention); a page of
  -1 means the destination is external/unresolved and is preserved as -1.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evidence_models import Bookmark, PdfEvidenceResult, PdfLine, PdfPageEvidence

BOLD_FLAG = 1 << 4  # PyMuPDF span flag: bold
ITALIC_FLAG = 1 << 1  # PyMuPDF span flag: italic


def _fitz() -> Any:
    import fitz  # lazy: raises ImportError when PyMuPDF is absent

    return fitz


def large_line_threshold(
    page_median: float, font_signal_min_size: float, font_signal_ratio: float
) -> float:
    """Deterministic per-page font-signal floor (pt)."""
    return max(float(font_signal_min_size), float(font_signal_ratio) * float(page_median))


def discover_sibling_pdf(input_json: Path) -> Optional[Path]:
    """Exact stem swap only: ``input_json.with_suffix(".pdf")`` if present."""
    candidate = input_json.with_suffix(".pdf")
    return candidate if candidate.is_file() else None


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of an ascending-sorted sample."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[-1]
    frac = rank - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])


def _line_text(spans: Sequence[Dict[str, Any]]) -> str:
    """Collapse a physical line's spans into canonical single-spaced text.

    Span text usually already carries inter-word spaces; when a span boundary
    falls between words without a space character AND the glyphs are visibly
    separated (gap >= 0.25 em), a space is inserted so words never glue.
    """
    pieces: List[str] = []
    previous: Optional[Dict[str, Any]] = None
    for span in spans:
        text = span.get("text") or ""
        if previous is not None and not text.startswith((" ", "\u00a0")):
            gap = float(span["bbox"][0]) - float(previous["bbox"][2])
            size = float(previous.get("size") or 1.0)
            if gap >= 0.25 * size and not (previous.get("text") or "").endswith(
                (" ", "\u00a0")
            ):
                pieces.append(" ")
        pieces.append(text)
        previous = span
    return " ".join("".join(pieces).replace("\u00a0", " ").split())


def get_toc_bookmarks(pdf_path: Path) -> List[Bookmark]:
    """PDF outline -> Bookmark list (1-based pages; -1 when unresolved).

    Raises on open/read failures; callers catch and downgrade gracefully.
    """
    fitz = _fitz()
    bookmarks: List[Bookmark] = []
    with fitz.open(pdf_path) as document:
        for item in document.get_toc(simple=True):
            level, title, page = int(item[0]), str(item[1]), int(item[2])
            bookmarks.append(Bookmark(level=level, title=" ".join(title.split()), page=page))
    return bookmarks


def _extract_pages(
    document: Any,
    max_large_lines_per_page: int,
    font_signal_min_size: float,
    font_signal_ratio: float,
) -> List[PdfPageEvidence]:
    pages: List[PdfPageEvidence] = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        rect = page.rect
        width = float(rect.width) or 1.0
        height = float(rect.height) or 1.0
        text_dict = page.get_text("dict")

        sizes: List[float] = []
        char_count = 0
        span_count = 0
        bold_count = 0
        italic_count = 0
        families: Counter = Counter()
        line_candidates: List[Dict[str, Any]] = []

        for block in text_dict.get("blocks") or []:
            if block.get("type") not in (0, None):
                continue
            for line in block.get("lines") or []:
                spans = [s for s in (line.get("spans") or []) if (s.get("text") or "").strip()]
                if not spans:
                    continue
                line_text = _line_text(spans)
                line_bbox = tuple(float(v) for v in line.get("bbox") or (0, 0, 0, 0))
                if not line_text or len(line_text) > 180:
                    continue
                sizes_of_line = [float(s.get("size") or 0.0) for s in spans]
                max_size = max(sizes_of_line)
                leading = max(range(len(spans)), key=lambda i: (sizes_of_line[i], -i))
                lead_span = spans[leading]
                for span in spans:
                    text = span.get("text") or ""
                    char_count += len(text)
                    span_count += 1
                    size = float(span.get("size") or 0.0)
                    sizes.append(size)
                    flags = int(span.get("flags") or 0)
                    if flags & BOLD_FLAG:
                        bold_count += 1
                    if flags & ITALIC_FLAG:
                        italic_count += 1
                    families[str(span.get("font") or "")] += 1
                flags = int(lead_span.get("flags") or 0)
                line_candidates.append(
                    {
                        "text": line_text,
                        "bbox": (
                            min(max(line_bbox[0] / width, 0.0), 1.0),
                            min(max(line_bbox[1] / height, 0.0), 1.0),
                            min(max(line_bbox[2] / width, 0.0), 1.0),
                            min(max(line_bbox[3] / height, 0.0), 1.0),
                        ),
                        "font_size": float(lead_span.get("size") or 0.0),
                        "font_family": str(lead_span.get("font") or ""),
                        "is_bold": bool(flags & BOLD_FLAG),
                        "is_italic": bool(flags & ITALIC_FLAG),
                    }
                )

        sorted_sizes = sorted(sizes)
        median = _percentile(sorted_sizes, 50.0)
        threshold = large_line_threshold(median, font_signal_min_size, font_signal_ratio)
        qualifying = [item for item in line_candidates if item["font_size"] >= threshold]
        qualifying.sort(
            key=lambda item: (-item["font_size"], item["bbox"][1], item["bbox"][0], item["text"])
        )
        large_lines = [
            PdfLine(
                page_index=page_index,
                text=item["text"],
                bbox=tuple(round(v, 4) for v in item["bbox"]),
                font_size=round(item["font_size"], 2),
                font_family=item["font_family"],
                is_bold=item["is_bold"],
                is_italic=item["is_italic"],
            )
            for item in qualifying[:max_large_lines_per_page]
        ]
        top_families = sorted(families.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        pages.append(
            PdfPageEvidence(
                page_index=page_index,
                page_width=round(width, 2),
                page_height=round(height, 2),
                native_char_count=char_count,
                span_count=span_count,
                font_size_p10=round(_percentile(sorted_sizes, 10.0), 2),
                font_size_median=round(median, 2),
                font_size_p90=round(_percentile(sorted_sizes, 90.0), 2),
                top_font_families=[(name, int(count)) for name, count in top_families],
                bold_span_count=bold_count,
                italic_span_count=italic_count,
                large_lines=large_lines,
            )
        )
    return pages


def extract_book_pdf_evidence(
    pdf_path: Optional[Path],
    page_count: int,
    max_large_lines_per_page: int = 20,
    font_signal_min_size: float = 14.0,
    font_signal_ratio: float = 1.15,
) -> PdfEvidenceResult:
    """Graceful, never-raising native PDF evidence extraction.

    ``page_count`` (OCR JSON pages) is only a sanity/normalization hint:
    evidence is per PDF page and a count mismatch does not fail the run.
    """
    del page_count  # informational only (plan D3); evidence is per PDF page
    if pdf_path is None:
        return PdfEvidenceResult(ok=False, reason="pdf_missing", pages=[], bookmarks=[], page_count=0)
    try:
        fitz = _fitz()
    except ImportError:
        return PdfEvidenceResult(
            ok=False, reason="fitz_unavailable", pages=[], bookmarks=[], page_count=0
        )
    try:
        with fitz.open(pdf_path) as document:
            bookmarks: List[Bookmark] = []
            for item in document.get_toc(simple=True):
                bookmarks.append(
                    Bookmark(
                        level=int(item[0]),
                        title=" ".join(str(item[1]).split()),
                        page=int(item[2]),
                    )
                )
            pages = _extract_pages(
                document, max_large_lines_per_page, font_signal_min_size, font_signal_ratio
            )
        return PdfEvidenceResult(
            ok=True, reason="ok", pages=pages, bookmarks=bookmarks, page_count=len(pages)
        )
    except Exception:
        return PdfEvidenceResult(
            ok=False, reason="pdf_corrupt", pages=[], bookmarks=[], page_count=0
        )


def make_span_accessor(
    pdf_path: Optional[Path],
) -> Callable[[int, Sequence[float]], Optional[List[Dict[str, Any]]]]:
    """On-demand span fetcher for candidate fusion (bounded, one page at a time).

    Returns a callable ``(page_index, bbox) -> list[span dict]`` where ``bbox``
    is normalized [0, 1] page coordinates; spans are normalized, filtered to
    those intersecting ``bbox``, sorted by (size desc, y, x, text) and capped
    at 100.  Returns None (never raises) when the PDF is unavailable or the
    page cannot be read.  A single-entry page cache keeps repeated access
    cheap while bounding memory.
    """
    cache: Dict[str, Any] = {"document": None, "page_index": None, "page_dict": None, "rect": None}

    if pdf_path is None:
        return lambda page_index, bbox: None

    def accessor(page_index: int, bbox: Sequence[float]) -> Optional[List[Dict[str, Any]]]:
        if not bbox or len(bbox) != 4:
            return None
        try:
            if cache["document"] is None:
                fitz = _fitz()
                cache["document"] = fitz.open(pdf_path)
            if cache["page_index"] != page_index:
                page = cache["document"].load_page(int(page_index))
                cache["page_dict"] = page.get_text("dict")
                cache["rect"] = page.rect
                cache["page_index"] = page_index
            rect = cache["rect"]
            width = float(rect.width) or 1.0
            height = float(rect.height) or 1.0
        except Exception:
            return None
        x1, y1, x2, y2 = (float(v) for v in bbox)
        hits: List[Dict[str, Any]] = []
        for block in (cache["page_dict"].get("blocks") or []):
            if block.get("type") not in (0, None):
                continue
            for line in block.get("lines") or []:
                for span in line.get("spans") or []:
                    sb = span.get("bbox") or (0, 0, 0, 0)
                    sx1, sy1, sx2, sy2 = (float(v) for v in sb)
                    nx1, ny1, nx2, ny2 = (
                        sx1 / width,
                        sy1 / height,
                        sx2 / width,
                        sy2 / height,
                    )
                    if nx2 <= x1 or nx1 >= x2 or ny2 <= y1 or ny1 >= y2:
                        continue
                    hits.append(
                        {
                            "text": span.get("text") or "",
                            "size": round(float(span.get("size") or 0.0), 2),
                            "font": str(span.get("font") or ""),
                            "flags": int(span.get("flags") or 0),
                            "bbox": [round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4)],
                        }
                    )
        hits.sort(key=lambda item: (-item["size"], item["bbox"][1], item["bbox"][0], item["text"]))
        return hits[:100]

    return accessor
