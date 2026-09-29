from __future__ import annotations

"""Planning helpers for contextual micro-batches.

Segments remain the persistence/cache/QA atom. A batch is only a wire unit, so
request efficiency can improve without weakening per-segment invalidation or
DOM/OCR provenance.
"""

from typing import Any, Dict, Iterable, List, Sequence

from ..models import Segment


SENSITIVE_KINDS = {
    "table", "table_cell", "attribute", "footnote", "caption",
}


def sensitive(segment: Segment) -> bool:
    if segment.kind in SENSITIVE_KINDS:
        return True
    label = str((segment.metadata or {}).get("label") or "")
    return label in {"reference_content", "table", "formula"}


def plan_micro_batches(
    pending: Sequence[Segment],
    *,
    chapter_of: Dict[str, str],
    positions: Dict[str, Dict[str, int]],
    max_segments: int,
    max_chars: int,
    sensitive_max_segments: int = 1,
) -> List[List[Segment]]:
    if max_segments < 1 or max_chars < 1 or sensitive_max_segments < 1:
        raise ValueError("micro-batch limits must be positive")
    batches: List[List[Segment]] = []
    current: List[Segment] = []
    chars = 0
    current_chapter = ""
    current_sensitive = False
    last_position = -2

    def flush() -> None:
        nonlocal current, chars, current_chapter, current_sensitive, last_position
        if current:
            batches.append(current)
        current = []
        chars = 0
        current_chapter = ""
        current_sensitive = False
        last_position = -2

    for segment in pending:
        chapter = chapter_of.get(segment.id) or ""
        position = (positions.get(chapter) or {}).get(segment.id)
        if position is None:
            flush()
            batches.append([segment])
            continue
        is_sensitive = sensitive(segment)
        cap = sensitive_max_segments if is_sensitive else max_segments
        compatible = bool(
            current
            and chapter == current_chapter
            and is_sensitive == current_sensitive
            and position == last_position + 1
            and len(current) < cap
            and chars + len(segment.source_text) <= max_chars
        )
        if current and not compatible:
            flush()
        if not current:
            current_chapter = chapter
            current_sensitive = is_sensitive
        current.append(segment)
        chars += len(segment.source_text)
        last_position = position
        if len(current) >= cap or chars >= max_chars:
            flush()
    flush()
    return batches


def batch_plan_summary(batches: Sequence[Sequence[Segment]]) -> Dict[str, Any]:
    sizes = [len(batch) for batch in batches]
    return {
        "batches": len(batches),
        "segments": sum(sizes),
        "average_batch_size": round(sum(sizes) / len(sizes), 4) if sizes else 0.0,
        "max_batch_size": max(sizes, default=0),
        "singleton_batches": sum(size == 1 for size in sizes),
    }
