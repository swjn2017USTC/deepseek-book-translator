"""Deterministic evidence dataclasses + serialization (P02 evidence layer).

Coordinates conventions shared by the evidence modules:

- ``bbox`` values are normalized page coordinates ``[0, 1]`` (both for
  ``PdfLine`` and for ``EvidenceCandidate``); serialization converts the
  ``BBox`` tuple to a JSON list and ``from_dict`` restores the tuple.
- ``page_json`` refers to the OCR JSON page index (0-based).  Bookmark and
  font-signal candidates are placed on that axis; candidates whose page could
  not be resolved keep ``page_json=null`` (see plan amendment A3).
- ``PdfLine.text`` and ``EvidenceCandidate.exact_text`` are canonical
  (whitespace-collapsed) heading texts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import BBox, stable_id


@dataclass
class Bookmark:
    level: int
    title: str
    page: int  # 1-based PDF page number (PyMuPDF convention), -1 if unresolved

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "title": self.title, "page": self.page}


@dataclass
class PdfLine:
    """A native PDF text line retained for font-signal evidence.

    ``bbox`` is normalized to the containing PDF page (all values in [0, 1]).
    ``font_size`` is in points; ``font_family``/``is_bold``/``is_italic``
    describe the largest span in the line.
    """

    page_index: int
    text: str
    bbox: BBox
    font_size: float
    font_family: str
    is_bold: bool
    is_italic: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "text": self.text,
            "bbox": list(self.bbox),
            "font_size": self.font_size,
            "font_family": self.font_family,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PdfLine":
        bbox = data.get("bbox")
        return cls(
            page_index=int(data["page_index"]),
            text=str(data["text"]),
            bbox=tuple(float(v) for v in bbox) if bbox is not None else (0.0, 0.0, 0.0, 0.0),
            font_size=float(data["font_size"]),
            font_family=str(data["font_family"]),
            is_bold=bool(data["is_bold"]),
            is_italic=bool(data["is_italic"]),
        )


@dataclass
class PdfPageEvidence:
    page_index: int
    page_width: float
    page_height: float
    native_char_count: int
    span_count: int
    font_size_p10: float
    font_size_median: float
    font_size_p90: float
    top_font_families: List[Tuple[str, int]]  # <=5, sorted by count desc
    bold_span_count: int
    italic_span_count: int
    large_lines: List[PdfLine]  # <=20 per page, already bounded at extraction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_index": self.page_index,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "native_char_count": self.native_char_count,
            "span_count": self.span_count,
            "font_size_p10": self.font_size_p10,
            "font_size_median": self.font_size_median,
            "font_size_p90": self.font_size_p90,
            "top_font_families": [list(item) for item in self.top_font_families],
            "bold_span_count": self.bold_span_count,
            "italic_span_count": self.italic_span_count,
            "large_lines": [line.to_dict() for line in self.large_lines],
        }


@dataclass
class PdfEvidenceResult:
    ok: bool
    reason: str  # "ok" | "pdf_missing" | "fitz_unavailable" | "pdf_corrupt"
    pages: List[PdfPageEvidence]
    bookmarks: List[Bookmark]
    page_count: int  # PDF page count (0 when no PDF was processed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "pages": [page.to_dict() for page in self.pages],
            "bookmarks": [bookmark.to_dict() for bookmark in self.bookmarks],
            "page_count": self.page_count,
        }


@dataclass
class EvidenceCandidate:
    candidate_id: str
    source_block_ids: List[str]
    exact_text: str
    page_json: Optional[int]
    page_print: Optional[int]
    bbox: Optional[BBox]
    zone: str = "mainmatter"
    paddle_label: Optional[str] = None
    paddle_relevel: Optional[Any] = None  # always None this phase
    pdf_bookmark: Optional[Dict[str, Any]] = None
    native_font: Optional[Dict[str, Any]] = None
    numbering: Optional[Dict[str, Any]] = None
    toc_matches: List[Dict[str, Any]] = field(default_factory=list)
    repetition: Dict[str, int] = field(default_factory=lambda: {"count": 0, "nearby_count": 0})
    before_context: Optional[str] = None
    after_context: Optional[str] = None
    deterministic_suppressions: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_block_ids": list(self.source_block_ids),
            "exact_text": self.exact_text,
            "page_json": self.page_json,
            "page_print": self.page_print,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "zone": self.zone,
            "paddle_label": self.paddle_label,
            "paddle_relevel": self.paddle_relevel,
            "pdf_bookmark": self.pdf_bookmark,
            "native_font": self.native_font,
            "numbering": self.numbering,
            "toc_matches": self.toc_matches,
            "repetition": dict(self.repetition),
            "before_context": self.before_context,
            "after_context": self.after_context,
            "deterministic_suppressions": list(self.deterministic_suppressions),
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceCandidate":
        bbox = data.get("bbox")
        return cls(
            candidate_id=str(data["candidate_id"]),
            source_block_ids=[str(item) for item in (data.get("source_block_ids") or [])],
            exact_text=str(data.get("exact_text") or ""),
            page_json=data.get("page_json"),
            page_print=data.get("page_print"),
            bbox=tuple(float(v) for v in bbox) if bbox is not None else None,
            zone=str(data.get("zone") or "mainmatter"),
            paddle_label=data.get("paddle_label"),
            paddle_relevel=data.get("paddle_relevel"),
            pdf_bookmark=data.get("pdf_bookmark"),
            native_font=data.get("native_font"),
            numbering=data.get("numbering"),
            toc_matches=list(data.get("toc_matches") or []),
            repetition=dict(data.get("repetition") or {"count": 0, "nearby_count": 0}),
            before_context=data.get("before_context"),
            after_context=data.get("after_context"),
            deterministic_suppressions=[
                str(item) for item in (data.get("deterministic_suppressions") or [])
            ],
            evidence_refs=[str(item) for item in (data.get("evidence_refs") or [])],
        )


def make_candidate_id(
    book_id: str,
    page_json: Optional[int],
    source_block_ids: Sequence[str],
    exact_text: str,
) -> str:
    """Stable candidate id per plan D4/A3.

    The page hash sentinel is -1 when ``page_json`` is None (the stored field
    stays null); ``source_block_ids`` are sorted so the id is invariant to
    evidence ordering.
    """
    return "cand-" + stable_id(
        book_id,
        page_json if page_json is not None else -1,
        "\x1f".join(sorted(set(source_block_ids))),
        exact_text,
    )


@dataclass
class EvidenceMetadata:
    schema_version: int
    book_id: str
    book_title: str
    generated_by: str
    package_version: str
    python_version: str
    input_sha256: str
    pdf_sha256: Optional[str]
    legacy_profile_used: bool
    pdf_evidence_ok: bool
    pdf_reason: str
    page_count: int  # OCR JSON page count (candidate space)
    candidate_count: int
    source_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "book_id": self.book_id,
            "book_title": self.book_title,
            "generated_by": self.generated_by,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "input_sha256": self.input_sha256,
            "pdf_sha256": self.pdf_sha256,
            "legacy_profile_used": self.legacy_profile_used,
            "pdf_evidence_ok": self.pdf_evidence_ok,
            "pdf_reason": self.pdf_reason,
            "page_count": self.page_count,
            "candidate_count": self.candidate_count,
            "source_counts": dict(self.source_counts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceMetadata":
        return cls(
            schema_version=int(data["schema_version"]),
            book_id=str(data["book_id"]),
            book_title=str(data["book_title"]),
            generated_by=str(data["generated_by"]),
            package_version=str(data["package_version"]),
            python_version=str(data["python_version"]),
            input_sha256=str(data["input_sha256"]),
            pdf_sha256=data.get("pdf_sha256"),
            legacy_profile_used=bool(data["legacy_profile_used"]),
            pdf_evidence_ok=bool(data["pdf_evidence_ok"]),
            pdf_reason=str(data["pdf_reason"]),
            page_count=int(data["page_count"]),
            candidate_count=int(data["candidate_count"]),
            source_counts={str(k): int(v) for k, v in (data.get("source_counts") or {}).items()},
        )
