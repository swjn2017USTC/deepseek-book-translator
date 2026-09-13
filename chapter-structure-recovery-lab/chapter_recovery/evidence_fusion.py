"""Evidence pipeline orchestrator (P02, plan section 4.4).

Reuses the existing leaf modules (paddleocr, page_map, toc, confirmed, zones,
style_model, alignment, heading_text, visual_context) as-is and adds the
deterministic evidence layer on top.  Never edits analyze outputs; artifacts
are written under ``<output_dir>/evidence/`` only.
"""

from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__ as _package_version
from .alignment import MACRO_LABELS
from .candidate_fusion import build_heading_candidates
from .confirmed import load_confirmed_toc, merge_confirmed_toc
from .evidence_models import EvidenceMetadata
from .heading_text import numbering
from .models import Block
from .page_map import infer_page_map
from .paddleocr import load_pages, normalize_blocks
from .pdf_evidence import (
    discover_sibling_pdf,
    extract_book_pdf_evidence,
    make_span_accessor,
)
from .structure_packets import (
    build_book_structure_evidence,
    build_numbering_summary,
    build_repeated_headers,
    build_style_clusters,
    candidate_sources,
    write_evidence_artifacts,
)
from .style_model import infer_internal_styles
from .toc import find_toc_pages, parse_toc_entries
from .zones import infer_page_zones


def _package_version_string() -> str:
    try:
        return _metadata.version("chapter-structure-recovery-lab")
    except Exception:
        return _package_version


def _resolve_config_path(base: Optional[Path], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base is None:
        return path
    return (base / path).resolve()


def _heading_blocks(pages: List[List[Block]]) -> List[Block]:
    """Style-analysis input: macro heading blocks + §-numbered text blocks."""
    blocks: List[Block] = []
    for page in pages:
        for block in page:
            if block.label in MACRO_LABELS:
                blocks.append(block)
                continue
            if block.label == "text":
                parsed = numbering(block.clean_content)
                if parsed is not None and parsed.family == "section_symbol":
                    blocks.append(block)
    return blocks


def _source_counts(candidates) -> Dict[str, int]:
    counts: Dict[str, int] = {tag: 0 for tag in ("paddle", "text_support", "bookmark", "font")}
    for candidate in candidates:
        for tag in candidate_sources(candidate):
            counts[tag] += 1
    return counts


def _generated_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_evidence_pipeline(
    config: Dict[str, Any],
    input_path: Path,
    output_dir: Path,
    pdf_path: Optional[Path] = None,
    config_base: Optional[Path] = None,
    emit_timestamp: bool = False,
) -> Dict[str, Any]:
    """Build the deterministic evidence packet for one book.

    ``pdf_path`` may be an explicit override; when None the sibling-PDF
    discovery rule (exact stem swap) applies.  ``config_base`` is the config
    file's directory, used only to resolve a relative ``confirmed_toc`` path.
    """
    raw_pages = load_pages(input_path)
    pages = normalize_blocks(raw_pages)
    page_map = infer_page_map(pages)
    toc_pages = find_toc_pages(pages, config.get("toc_search_pages", [0, 30]))
    toc_entries = parse_toc_entries(pages, toc_pages)
    if config.get("confirmed_toc"):
        confirmed_path = _resolve_config_path(config_base, config["confirmed_toc"])
        toc_entries = merge_confirmed_toc(toc_entries, load_confirmed_toc(confirmed_path))
    zones = infer_page_zones(pages)

    resolved_pdf = pdf_path
    if resolved_pdf is None:
        resolved_pdf = discover_sibling_pdf(input_path)
    font_min_size = float(config.get("font_signal_min_size", 14.0))
    font_ratio = float(config.get("font_signal_ratio", 1.15))
    pdf_result = extract_book_pdf_evidence(
        resolved_pdf,
        len(pages),
        font_signal_min_size=font_min_size,
        font_signal_ratio=font_ratio,
    )
    span_accessor = (
        make_span_accessor(resolved_pdf)
        if pdf_result.ok and resolved_pdf is not None
        else None
    )
    candidates = build_heading_candidates(
        pages,
        page_map,
        zones,
        toc_entries,
        pdf_result if pdf_result.ok else None,
        span_accessor,
        config,
        str(config["book_id"]),
    )

    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    pdf_sha256: Optional[str] = None
    if resolved_pdf is not None and resolved_pdf.is_file():
        pdf_sha256 = hashlib.sha256(resolved_pdf.read_bytes()).hexdigest()
    legacy_profile_used = bool(
        config.get("hierarchy_profile") or config.get("text_candidate_profile")
    )
    source_counts = _source_counts(candidates)
    metadata = EvidenceMetadata(
        schema_version=1,
        book_id=str(config["book_id"]),
        book_title=str(config.get("book_title") or config["book_id"]),
        generated_by="chapter_recovery.evidence_v2",
        package_version=_package_version_string(),
        python_version=platform.python_version(),
        input_sha256=input_sha256,
        pdf_sha256=pdf_sha256,
        legacy_profile_used=legacy_profile_used,
        pdf_evidence_ok=pdf_result.ok,
        pdf_reason=pdf_result.reason,
        page_count=len(pages),
        candidate_count=len(candidates),
        source_counts=source_counts,
    )

    style_clusters = build_style_clusters(infer_internal_styles(_heading_blocks(pages)))
    numbering_summary = build_numbering_summary(candidates)
    repeated_headers = build_repeated_headers(
        pages, threshold=int(config.get("running_header_threshold", 5))
    )
    book_evidence = build_book_structure_evidence(
        metadata,
        candidates,
        toc_entries,
        pdf_result.bookmarks,
        page_map,
        zones,
        numbering_summary,
        repeated_headers,
        style_clusters,
    )
    write_evidence_artifacts(
        output_dir,
        metadata,
        pdf_result,
        candidates,
        book_evidence,
        generated_at=_generated_at() if emit_timestamp else None,
    )

    return {
        "output_dir": str(output_dir),
        "book_id": str(config["book_id"]),
        "pages": len(pages),
        "candidates": len(candidates),
        "toc_entries": len(toc_entries),
        "legacy_profile_used": legacy_profile_used,
        "pdf_evidence_ok": pdf_result.ok,
        "pdf_reason": pdf_result.reason,
        "source_counts": source_counts,
    }
