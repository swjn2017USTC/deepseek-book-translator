"""Per-segment concept dependency resolution (spec §12.6 / D4).

Replaces the book-global glossary-SHA invalidation of v1 with per-segment
``{concept_id: version}`` dependency maps.  Matching reuses v1's own phrase
boundary matcher (``glossary._phrase_pattern``) so there is exactly one
matching convention in the repo — read-only import, no shims added.

A concept's dependency version is ``max(sense.version)`` across its senses:
any sense add/bump invalidates every segment that uses the concept (safe,
simple, deterministic).  With no concept file (today's default state) the
dependency map is empty and invalidation is driven by source_hash + brief_hash
+ prompt_version — still finer than v1's book-global SHA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..glossary import _phrase_pattern  # noqa: W0212 — read-only convention reuse (P06 D4/R1)
from ..models import Segment


def load_concept_glossary(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load the P05 concept glossary (schema v2) into a concept list.

    Returns ``[]`` when the path is None or the file is missing (fallback: no
    per-segment concept dependencies).  Raises ``ValueError`` on malformed
    files or a schema version other than 2 — never silently misreads.
    """
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Concept glossary is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Concept glossary must be a JSON object: {path}")
    if raw.get("schema_version") != 2:
        raise ValueError(
            f"Concept glossary schema_version must be 2, got {raw.get('schema_version')!r}: {path}"
        )
    concepts = raw.get("concepts")
    if not isinstance(concepts, list):
        raise ValueError(f"Concept glossary 'concepts' must be a list: {path}")
    return [concept for concept in concepts if isinstance(concept, dict)]


def concept_version(concept: Dict[str, Any]) -> int:
    """Dependency version of a concept = max sense.version (>= 1)."""
    senses = concept.get("senses")
    if not isinstance(senses, list):
        return 0
    versions = [
        int(sense.get("version", 1))
        for sense in senses
        if isinstance(sense, dict) and sense.get("version") is not None
    ]
    return max(versions, default=0)


def _sense(concept: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Preferred sense for rendering: ``default`` if present else max version."""
    senses = [sense for sense in (concept.get("senses") or []) if isinstance(sense, dict)]
    if not senses:
        return None
    for sense in senses:
        if sense.get("sense_id") == "default":
            return sense
    return max(senses, key=lambda sense: int(sense.get("version", 1)))


def concept_term(concept: Dict[str, Any], source_text: str) -> str:
    """Prompt-facing term: the surface form that actually matched, falling back
    to the canonical source / first surface form."""
    for surface in _surfaces(concept):
        if surface and _phrase_pattern(surface).search(source_text):
            return surface
    canonical = str(concept.get("canonical_source") or "").strip()
    if canonical:
        return canonical
    for surface in concept.get("surface_forms") or []:
        if str(surface).strip():
            return str(surface).strip()
    return ""


def _surfaces(concept: Dict[str, Any]) -> List[str]:
    surfaces: List[str] = []
    for value in (concept.get("surface_forms") or []):
        if str(value).strip():
            surfaces.append(str(value).strip())
    for value in (concept.get("master_aliases") or []):
        if str(value).strip():
            surfaces.append(str(value).strip())
    return surfaces


def concept_sense(concept: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _sense(concept)


def matched_surface(concept: Dict[str, Any], source_text: str) -> Optional[str]:
    """First surface form / master alias present in ``source_text`` (phrase
    boundary), else None."""
    for surface in _surfaces(concept):
        if _phrase_pattern(surface).search(source_text):
            return surface
    return None


def resolve_concept_deps(
    segment: Segment,
    concepts: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Phrase-boundary match of each concept's surface forms + master aliases
    against ``segment.source_text`` -> ``{concept_id: concept_version}``.

    Empty dict when ``concepts`` is empty (fallback state).  Order follows the
    concept file so payload/record rendering is deterministic.
    """
    if not concepts:
        return {}
    dependencies: Dict[str, int] = {}
    for concept in concepts:
        concept_id = str(concept.get("concept_id") or "").strip()
        if not concept_id:
            continue
        matched = any(
            surface and _phrase_pattern(surface).search(segment.source_text)
            for surface in _surfaces(concept)
        )
        if matched:
            dependencies[concept_id] = concept_version(concept)
    return dependencies


def glossary_subset(
    segment: Segment,
    concepts: List[Dict[str, Any]],
    dependencies: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Prompt glossary entries for the concepts this segment depends on.

    Each entry is trimmed to id/term/translation/constraint (token economy,
    §18); the term is the matched surface form.
    """
    entries: List[Dict[str, Any]] = []
    for concept in concepts:
        concept_id = str(concept.get("concept_id") or "").strip()
        if concept_id not in dependencies:
            continue
        sense = concept_sense(concept) or {}
        entries.append({
            "concept_id": concept_id,
            "term": concept_term(concept, segment.source_text),
            "translation": str(sense.get("translation") or ""),
            "constraint": str(sense.get("constraint") or "preferred"),
        })
    return entries
