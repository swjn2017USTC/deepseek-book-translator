"""Deterministic context windows for contextual translation v2 (spec §12.4).

Ordering is a pure function of the chapter's segment list: ``(page_json, file
index)`` ascending — the same page-then-block order the cleaned segments file
is written in.  Windows are bounded by the configurable ``previous_segments`` /
``next_segments`` caps and can never cross a chapter boundary, because callers
compute windows over the *chapter's own* ordered member list only.

``previous_translation_text`` mirrors v1's ``previous_tail`` rule (translate.py)
but sources it from the nearest preceding completed translation of the same
chapter instead of the last translated batch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

from ..models import Segment


def chapter_order(segments: Sequence[Segment]) -> List[Segment]:
    """Stable chapter ordering: ``(page_json or -1, original file index)``.

    Deterministic across repeated invocations and independent of any batch or
    pending state; no randomness.
    """
    return [
        segment
        for _, segment in sorted(
            enumerate(segments),
            key=lambda item: (
                (item[1].page_json if item[1].page_json is not None else -1),
                item[0],
            ),
        )
    ]


def window(
    segments: Sequence[Segment],
    index: int,
    *,
    prev: int,
    next_: int,
) -> Tuple[List[str], List[str]]:
    """Return ``(previous_source_texts, next_source_texts)`` for the segment at
    ``index`` inside the already-ordered ``segments`` list (see ``chapter_order``).

    The slice is bounded by the list edges and by ``prev`` / ``next_``; it never
    reaches outside the caller's chapter member list, so cross-chapter leakage
    is impossible by construction.  ``segments`` must contain only text-bearing
    members (callers drop empty/image segments).
    """
    if prev < 0 or next_ < 0:
        raise ValueError("prev and next_ must be non-negative")
    if index < 0 or index >= len(segments):
        raise IndexError(f"segment index {index} out of range for {len(segments)} members")
    previous = [segment.source_text for segment in segments[max(0, index - prev):index]]
    following = [segment.source_text for segment in segments[index + 1:index + 1 + next_]]
    return previous, following


def previous_translation_text(
    rows: Sequence[Optional[Dict[str, Any]]],
    *,
    max_chars: int,
    sentence_finish: Pattern[str],
) -> str:
    """Tail (≤ ``max_chars``) of the nearest preceding completed translation.

    ``rows`` must hold one entry per chapter-ordered segment *before* the
    target (None where the neighbour has no current translation), oldest first.
    Returns "" when no preceding translation exists or when the preceding
    translation ends in sentence-finish punctuation — the v1 ``previous_tail``
    continuation rule (translate.SENTENCE_FINISH).
    """
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    for row in reversed(list(rows)):
        if not row:
            continue
        text = str(row.get("translated_text") or "")
        break
    else:  # no preceding current translation in this chapter
        return ""
    if not text:
        return ""
    if sentence_finish.search(text):
        return ""
    return text[-max_chars:]
