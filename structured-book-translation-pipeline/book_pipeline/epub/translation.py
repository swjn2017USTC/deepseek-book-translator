"""EPUB-specific translation contracts shared by legacy and contextual paths."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict

from ..models import Segment
from .segments import reconstruct_segment_slots

_NUMBER = re.compile(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)")
_URL = re.compile(r"https?://[^\s<]+")
_NOTE_REF = re.compile(r"(?:\[\^[^\]]+\]|\[\[EPUB:\d+:(?:OPEN|CLOSE):sup\]\])")


def is_epub_segment(segment: Segment) -> bool:
    return (segment.metadata or {}).get("source_adapter") == "epub_native_v1"


def validate_epub_translation(segment: Segment, translated_text: str) -> None:
    """Fail closed before a checkpoint can contain an unreconstructable EPUB slot."""
    if not is_epub_segment(segment):
        return
    # This checks the exact token sequence, balanced nesting, and that each
    # translatable slot owns exactly one unambiguous span in the reply.
    slots = reconstruct_segment_slots(segment, translated_text)
    source_values = {
        str(slot["id"]): str(slot.get("value") or "")
        for slot in (segment.metadata or {}).get("slots") or []
    }
    for slot_id, value in slots.items():
        # Only a slot that *had* content may not be emptied. A whitespace-only
        # slot is the separator between inline elements (``</em> <strong>``);
        # requiring a non-blank value there would reject every correct
        # translation of a paragraph that has one.
        if not str(value).strip() and source_values.get(slot_id, "").strip():
            raise ValueError(f"EPUB response emptied a text slot for {segment.id}")
    # Plain-language numbers may legitimately be localized (for example,
    # ``late 1800s`` -> ``19世纪末``). Numeric fidelity remains a QA signal,
    # but must not make the EPUB slot rewriter reject an otherwise
    # reconstructable translation. URLs and note/sup markers are structural
    # invariants and stay byte/token checked here.
    for label, pattern in (("URL", _URL), ("footnote reference", _NOTE_REF)):
        if Counter(pattern.findall(segment.source_text)) != Counter(pattern.findall(translated_text)):
            raise ValueError(f"EPUB {label} mismatch for {segment.id}")


def epub_cache_metadata(segment: Segment, *, prompt_version: int, chapter_dependency: str, glossary_dependencies: Dict[str, int] | None = None) -> Dict[str, Any]:
    """Persist document provenance while invalidating at a locator's text scope."""
    metadata = segment.metadata or {}
    table = {
        key: metadata.get(key)
        for key in ("row", "column", "rowspan", "colspan", "cell_locator", "table_locator")
        if key in metadata
    }
    return {
        "source_adapter": "epub_native_v1",
        # Whole-document hash is export provenance. It must not invalidate
        # siblings when one XHTML text slot changes; source_hash does that.
        "source_document_sha256": metadata.get("source_document_sha256"),
        "locator": metadata.get("locator"),
        "structural_hash": metadata.get("structural_hash"),
        "prompt_version": prompt_version,
        "chapter_dependency": chapter_dependency,
        "glossary_dependencies": dict(glossary_dependencies or {}),
        "epub_invariants": {
            "link_targets": list(metadata.get("link_targets") or []),
            "footnote_targets": list(metadata.get("footnote_targets") or []),
            "table": table,
        },
    }


def epub_cache_current(segment: Segment, metadata: Dict[str, Any], *, prompt_version: int, chapter_dependency: str, concept_versions: Dict[str, int] | None = None) -> bool:
    if not is_epub_segment(segment):
        return True
    expected = epub_cache_metadata(segment, prompt_version=prompt_version, chapter_dependency=chapter_dependency)
    # source_document_sha256 remains a record/export provenance key. Comparing
    # it here would make every sibling in a changed XHTML stale.
    for key in ("source_adapter", "locator", "structural_hash", "prompt_version", "chapter_dependency"):
        if metadata.get(key) != expected[key]:
            return False
    if metadata.get("epub_invariants") != expected["epub_invariants"]:
        return False
    dependencies = metadata.get("glossary_dependencies")
    if not isinstance(dependencies, dict):
        return False
    if concept_versions is None:
        return not dependencies
    return all(concept_versions.get(key) == value for key, value in dependencies.items())
