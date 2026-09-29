from __future__ import annotations

"""Structure-aware micro-batching for contextual translation.

Persistence/cache granularity stays per Segment.  This module only chooses the
wire/request unit, so batching never changes stable segment IDs or DOM/OCR
provenance.
"""

from typing import Any, Dict, List, Sequence

from ..models import Segment
from ..translate import STRUCTURAL_TOKEN
from .windows import chapter_order


_SINGLETON_KINDS = {"table", "table_cell", "attribute"}
_SMALL_KINDS = {"footnote", "caption", "definition_term", "definition_description"}


def _label(segment: Segment) -> str:
    return str((segment.metadata or {}).get("label") or "").casefold()


def batch_class(segment: Segment) -> str:
    if segment.kind in _SINGLETON_KINDS:
        return "singleton"
    if segment.kind in _SMALL_KINDS or _label(segment) == "reference_content":
        return "small"
    # Dense protected-token payloads have a larger validation blast radius.
    if len(STRUCTURAL_TOKEN.findall(segment.source_text)) >= 6:
        return "singleton"
    return "prose"


def build_micro_batches(
    all_segments: Sequence[Segment],
    pending: Sequence[Segment],
    chapter_of: Dict[str, str],
    settings: Dict[str, Any],
) -> List[List[Segment]]:
    micro = dict(settings.get("micro_batch") or {})
    enabled = bool(micro.get("enabled", True))
    max_segments = max(1, int(micro.get("max_segments", 4)))
    max_chars = max(1, int(micro.get("max_chars", 6000)))
    small_max = max(1, int(micro.get("small_kind_max_segments", 2)))
    if not enabled:
        max_segments = 1

    pending_ids = {segment.id for segment in pending}
    source_order = {segment.id: index for index, segment in enumerate(all_segments)}
    chapters: Dict[str, List[Segment]] = {}
    for segment in all_segments:
        if not segment.translatable or not segment.source_text.strip():
            continue
        chapter = chapter_of.get(segment.id) or ""
        chapters.setdefault(chapter, []).append(segment)

    batches: List[List[Segment]] = []
    for chapter_segments in chapters.values():
        ordered = chapter_order(chapter_segments)
        current: List[Segment] = []
        characters = 0
        current_class = ""
        current_zone = ""

        def flush() -> None:
            nonlocal current, characters, current_class, current_zone
            if current:
                batches.append(current)
            current = []
            characters = 0
            current_class = ""
            current_zone = ""

        for segment in ordered:
            if segment.id not in pending_ids:
                # Do not batch across an already-completed segment: its existing
                # target translation is useful boundary context and should not
                # be silently skipped inside a new wire batch.
                flush()
                continue
            kind = batch_class(segment)
            limit = 1 if kind == "singleton" else (small_max if kind == "small" else max_segments)
            incompatible = bool(
                current
                and (
                    kind != current_class
                    or segment.zone != current_zone
                    or len(current) >= limit
                    or characters + len(segment.source_text) > max_chars
                )
            )
            if incompatible:
                flush()
            if not current:
                current_class = kind
                current_zone = segment.zone
            current.append(segment)
            characters += len(segment.source_text)
            if kind == "singleton" or len(current) >= limit or characters >= max_chars:
                flush()
        flush()

    batches.sort(key=lambda batch: source_order.get(batch[0].id, 10**12))
    return batches


def estimated_request_count(
    all_segments: Sequence[Segment],
    pending: Sequence[Segment],
    chapter_of: Dict[str, str],
    settings: Dict[str, Any],
) -> int:
    return len(build_micro_batches(all_segments, pending, chapter_of, settings))
