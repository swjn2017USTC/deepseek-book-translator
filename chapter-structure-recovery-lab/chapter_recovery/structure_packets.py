"""Compressed whole-book evidence packet + artifact writing (P02).

Artifacts (all under ``runs/<book>/evidence/``):

- ``pdf_evidence.json``           -- per-page native aggregates (+ bounded lines)
- ``heading_candidates.jsonl``    -- one merged candidate object per line
- ``book_structure_evidence.json`` -- compressed packet for orchestrators/LLMs

Writes are atomic (sibling ``.tmp`` then ``os.replace``) and byte-deterministic:
no timestamps unless the caller opts in via ``generated_at``.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .alignment import normalized_title
from .evidence_models import (
    Bookmark,
    EvidenceCandidate,
    EvidenceMetadata,
    PdfEvidenceResult,
)
from .models import Block, TocEntry
from .page_map import PageMap


def candidate_sources(candidate: EvidenceCandidate) -> List[str]:
    """Evidence sources of a merged candidate, derived from its refs.

    Ref grammar: ``paddle:<block_id>:<label>``, ``pdf:bookmark:<idx>``,
    ``pdf:font:<page>:<idx>``, ``pdf:text_support:<block_id>:<reason>``.
    Returns the canonical order [paddle, text_support, bookmark, font].
    """
    tags = set()
    for ref in candidate.evidence_refs:
        parts = ref.split(":")
        if parts[:1] == ["paddle"]:
            label = parts[-1]
            if label in {"doc_title", "paragraph_title", "header"}:
                tags.add("paddle")
            elif label == "text":
                tags.add("text_support")
        elif parts[:2] == ["pdf", "bookmark"]:
            tags.add("bookmark")
        elif parts[:2] == ["pdf", "font"]:
            tags.add("font")
        elif parts[:2] == ["pdf", "text_support"]:
            tags.add("text_support")
    order = ["paddle", "text_support", "bookmark", "font"]
    return [tag for tag in order if tag in tags]


def build_numbering_summary(candidates: List[EvidenceCandidate]) -> Dict[str, Any]:
    families: Counter = Counter()
    representatives: Dict[str, List[str]] = {}
    for candidate in candidates:
        parsed = candidate.numbering
        if not parsed:
            continue
        family = str(parsed["family"])
        families[family] += 1
        token = str(parsed["token"])
        if token not in representatives.setdefault(family, []):
            representatives[family].append(token)
    return {
        "families": {family: int(families[family]) for family in sorted(families)},
        "representative_tokens": {
            family: sorted(tokens)[:5] for family, tokens in sorted(representatives.items())
        },
    }


def build_repeated_headers(
    pages: List[List[Block]], threshold: int = 5
) -> List[Dict[str, Any]]:
    seen: Dict[str, List[Any]] = {}
    for page_index, blocks in enumerate(pages):
        for block in blocks:
            if block.label != "header":
                continue
            key = normalized_title(block.clean_content)
            if not key:
                continue
            item = seen.setdefault(key, [key, 0, page_index, "header"])
            item[1] += 1
            item[2] = min(item[2], page_index)
    rows = [item for item in seen.values() if item[1] >= threshold]
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [
        {
            "normalized_title": item[0],
            "count": int(item[1]),
            "first_page_json": int(item[2]),
            "label": str(item[3]),
        }
        for item in rows
    ]


def build_style_clusters(style_decisions: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_level: Dict[int, List[str]] = {}
    for block_id, decision in style_decisions.items():
        by_level.setdefault(int(decision.level), []).append(str(block_id))
    clusters: List[Dict[str, Any]] = []
    for level in sorted(by_level):
        block_ids = sorted(set(by_level[level]))
        clusters.append(
            {
                "level": level,
                "count": len(block_ids),
                "sample_block_ids": block_ids[:8],
            }
        )
    return clusters


def _bookmark_tree(bookmarks: List[Bookmark]) -> List[Dict[str, Any]]:
    """Flat PDF outline -> nested tree; implicit root holds level-1 items."""
    root: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []

    def node_for(bookmark: Bookmark) -> Dict[str, Any]:
        return {
            "level": bookmark.level,
            "title": bookmark.title,
            "page": bookmark.page,
            "children": [],
        }

    for bookmark in bookmarks:
        entry = node_for(bookmark)
        while stack and stack[-1]["level"] >= bookmark.level:
            stack.pop()
        if stack:
            stack[-1]["children"].append(entry)
        else:
            root.append(entry)
        stack.append(entry)
    return root


def build_book_structure_evidence(
    metadata: EvidenceMetadata,
    candidates: List[EvidenceCandidate],
    toc_entries: List[TocEntry],
    bookmarks: List[Bookmark],
    page_map: PageMap,
    zones: Dict[int, str],
    numbering_summary: Dict[str, Any],
    repeated_headers: List[Dict[str, Any]],
    style_clusters: List[Dict[str, Any]],
) -> Dict[str, Any]:
    page_count = max(1, metadata.page_count)
    return {
        "schema_version": metadata.schema_version,
        "metadata": metadata.to_dict(),
        "toc_entries": [entry.to_dict() for entry in toc_entries],
        "bookmark_tree": _bookmark_tree(bookmarks),
        "numbering_summary": numbering_summary,
        "style_clusters": style_clusters,
        "zones": {
            str(page_index): zones[page_index]
            for page_index in sorted(zones)
        },
        "repeated_headers": repeated_headers,
        "candidate_count": len(candidates),
        "page_density": round(len(candidates) / page_count, 3),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "page_json": candidate.page_json,
                "exact_text": candidate.exact_text,
                "primary_sources": candidate_sources(candidate),
            }
            for candidate in candidates
        ],
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json_text(value: Any, indent: Optional[int] = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent) + "\n"


def write_evidence_artifacts(
    output_dir: Path,
    metadata: EvidenceMetadata,
    pdf_evidence: PdfEvidenceResult,
    candidates: List[EvidenceCandidate],
    book_evidence: Dict[str, Any],
    generated_at: Optional[str] = None,
) -> Dict[str, Path]:
    """Write the three evidence artifacts atomically; return their paths."""
    evidence_dir = output_dir / "evidence"
    pdf_root: Dict[str, Any] = pdf_evidence.to_dict()
    packet_root = dict(book_evidence)
    if generated_at is not None:
        pdf_root["generated_at"] = generated_at
        packet_root["generated_at"] = generated_at

    paths = {
        "pdf_evidence": evidence_dir / "pdf_evidence.json",
        "heading_candidates": evidence_dir / "heading_candidates.jsonl",
        "book_structure_evidence": evidence_dir / "book_structure_evidence.json",
    }
    _atomic_write(paths["pdf_evidence"], _json_text(pdf_root, indent=2))
    lines = "".join(_json_text(candidate.to_dict()) for candidate in candidates)
    _atomic_write(paths["heading_candidates"], lines)
    _atomic_write(paths["book_structure_evidence"], _json_text(packet_root, indent=2))
    return paths
