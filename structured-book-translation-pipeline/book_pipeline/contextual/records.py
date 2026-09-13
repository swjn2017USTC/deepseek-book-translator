"""v1-compatible superset translation records for contextual translation v2.

Rows appended to the shared ``translations.jsonl`` carry every v1 field —
``glossary_sha256`` filled exactly as v1 (``glossary_cache_sha``) so v1
consumers (``translation_status``, ``render_book``, v1 ``translate`` resume)
see v2 output as current — plus a single reserved ``metadata`` dict with the v2
cache keys (D5).  ``models.TranslationRecord`` is untouched.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from ..models import Segment


def record_row(
    segment: Segment,
    translated_text: str,
    client_model: str,
    glossary_sha256: str,
    *,
    chapter_id: str,
    brief_hash: str,
    prompt_version: int,
    deps: Dict[str, int],
) -> Dict[str, Any]:
    """Build the v1-compatible superset record for one completed segment."""
    return {
        "segment_id": segment.id,
        "source_hash": segment.source_hash,
        "translated_text": translated_text,
        "model": client_model,
        "status": "completed",
        "attempt": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "glossary_sha256": glossary_sha256,
        "metadata": {
            "chapter_id": chapter_id,
            "brief_hash": brief_hash,
            "prompt_version": prompt_version,
            "context_mode": "contextual_v2",
            "glossary_dependencies": dict(deps),
        },
    }
