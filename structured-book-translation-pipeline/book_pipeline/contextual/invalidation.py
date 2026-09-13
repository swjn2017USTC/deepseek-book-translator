"""Fine cache invalidation for contextual translation v2 (spec §12.6 / D4).

``is_translation_current_v2`` replaces v1's book-global glossary-SHA check with
per-segment concept versions plus per-chapter brief hashes:

* source_hash must match (source edited -> stale);
* status must be ``completed``;
* ``metadata.prompt_version`` must equal the current PROMPTS_VERSION;
* ``metadata.brief_hash`` must equal the current hash of the segment's chapter;
* every recorded dependency ``{concept_id: version}`` must equal the concept's
  current version — a bumped/addressed concept invalidates exactly the segments
  that mention it, and a brand-new concept invalidates nothing.

v1 ``is_translation_current`` is untouched and keeps book-global semantics for
the legacy path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models import Segment
from .briefs import chapter_id_for_segment, zone_bucket_id
from .deps import concept_version
from .prompts import PROMPTS_VERSION


def current_briefs(briefs_by_id: Dict[str, Any]) -> Dict[str, str]:
    """``chapter_id -> brief_hash`` over an already-built brief map."""
    return {chapter_id: brief.brief_hash for chapter_id, brief in briefs_by_id.items()}


def current_concept_versions(concepts: list) -> Dict[str, int]:
    """``concept_id -> current version`` (max sense version) per concept."""
    versions: Dict[str, int] = {}
    for concept in concepts:
        concept_id = str(concept.get("concept_id") or "").strip()
        if concept_id:
            versions[concept_id] = concept_version(concept)
    return versions


def brief_key_for_segment(
    segment: Segment,
    nodes_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Chapter (or synthetic zone bucket) whose brief governs this segment."""
    if nodes_by_id:
        chapter_id = chapter_id_for_segment(segment, nodes_by_id)
        if chapter_id is not None:
            return chapter_id
        metadata = segment.metadata or {}
        if metadata.get("node_kind") == "chapter" and metadata.get("node_id"):
            return str(metadata["node_id"])
    return zone_bucket_id(segment.zone)


def is_translation_current_v2(
    segment: Segment,
    row: Optional[Dict[str, Any]],
    *,
    brief_hashes: Dict[str, str],
    concept_versions: Dict[str, int],
    nodes_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    """v2 currency: source + status + prompt_version + chapter brief hash +
    per-dependency concept versions (spec §3 contract)."""
    if not row:
        return False
    if row.get("source_hash") != segment.source_hash or row.get("status") != "completed":
        return False
    metadata = row.get("metadata") or {}
    if metadata.get("context_mode") != "contextual_v2":
        return False
    if metadata.get("prompt_version") != PROMPTS_VERSION:
        return False
    if metadata.get("brief_hash") != brief_hashes.get(brief_key_for_segment(segment, nodes_by_id)):
        return False
    dependencies = metadata.get("glossary_dependencies") or {}
    if not isinstance(dependencies, dict):
        return False
    return all(
        concept_versions.get(concept_id) == version
        for concept_id, version in dependencies.items()
    )
