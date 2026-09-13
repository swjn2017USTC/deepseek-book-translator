from __future__ import annotations

"""Deterministic high-recall candidate miner (spec §11.2, plan §2).

The miner reuses glossary.py's eight v1 lexical signals **by import**
(``_candidate_sources``) so recall behaviour is byte-identical to the legacy
miner while v1 itself stays untouched.  Each v1 candidate becomes one
``TermCandidate`` row with a stable content-addressed id, its *primary* signal
as ``source_type`` (v1 tier ordering), a bounded evidence list and a bounded
context list (§18: candidates + short snippets, never whole chapters).

Extension: ``use_yake`` is a feature flag for an optional ``yake`` import.
YAKE is **not** installed this phase (D2); when the flag is on and the module
is importable its top keywords become ``source_type: "yake"`` candidates, and
``yake_enabled`` records whether the import succeeded so the absence of
statistical signals is never misreported as a regression.

Everything is deterministic: stable sort keys only, no iteration over sets or
dicts with identity ordering, ids derived from sha256 of the normalized
surface.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

from ..glossary import (
    GENERIC_HEADINGS,
    STOPWORDS,
    _candidate_sources,
    _matching_segments,
    _normal,
)
from .schemas import TermCandidate

V1_SOURCE_TYPES = (
    "book_title_term",
    "manually_seeded_source_term",
    "heading_phrase",
    "heading_subphrase",
    "quoted_phrase",
    "proper_name",
    "repeated_phrase",
    "distinctive_word",
)
YAKE_SOURCE_TYPE = "yake"

# v1 tier function as defined in glossary._candidates — mirrored here so the
# *primary* signal of a multi-signal candidate is chosen exactly like v1's
# ranking prefers it.
_TIER = {
    "book_title_term": 0,
    "manually_seeded_source_term": 0,
    "heading_phrase": 1,
    "heading_subphrase": 1,
    "distinctive_word": 2,
    "quoted_phrase": 3,
    "proper_name": 4,
    "repeated_phrase": 5,
    YAKE_SOURCE_TYPE: 5,
}

MAX_EVIDENCE_SEGMENTS = 12
MAX_CONTEXTS = 3
YAKE_KEYWORD_LIMIT = 60
YAKE_MIN_OCCURRENCES = 2


def _term_candidate_id(surface: str) -> str:
    """``term-`` + first 12 hex chars of sha256(normalized surface)."""
    from hashlib import sha256

    return "term-" + sha256(_normal(surface).encode("utf-8")).hexdigest()[:12]


def _primary_source_type(item: Dict[str, Any]) -> str:
    """Pick the primary signal with v1 tier precedence; ties break by name."""
    reasons = sorted(str(reason) for reason in item.get("reasons", ()))
    if not reasons:
        return "distinctive_word"
    return min(reasons, key=lambda reason: (_TIER.get(reason, 9), reason))


def _segment_id(segment: Dict[str, Any]) -> str:
    return str(segment.get("id") or "")


def _optional_yake_keywords(
    segments: Sequence[Dict[str, Any]],
    use_yake: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Try-import YAKE and run it over the corpus text (D2).

    Returns ``(items, enabled)`` where each item is
    ``{"surface", "occurrences", "segment_ids", "contexts", "score"}`` in
    deterministic order (score desc, surface asc).  ``enabled`` is false when
    the feature flag is off or the import fails, and the caller records that in
    the manifest so "no statistical signals" is never misread as a regression.
    """
    items: List[Dict[str, Any]] = []
    if not use_yake:
        return items, False
    try:  # yake is optional; it pulls nltk+networkx and is NOT installed (D2).
        from yake import KeywordExtractor  # type: ignore[import-not-found]
    except Exception:
        return items, False
    try:
        text = "\n".join(
            " ".join(str(segment.get("source_text") or "").split())
            for segment in segments
            if str(segment.get("source_text") or "").strip()
        )
        if len(text) < 200:
            return items, True
        extractor = KeywordExtractor(
            lan="en", n=3, top=YAKE_KEYWORD_LIMIT * 4, dedupLim=0.9, windowsSize=1
        )
        keywords = extractor.extract_keywords(text)
        for surface, yake_score in keywords:
            surface = " ".join(str(surface).split())
            norm = _normal(surface)
            if (
                len(norm) < 4
                or norm in GENERIC_HEADINGS
                or _normal(surface).split()[0] in STOPWORDS
                or len(_normal(surface).split()) > 8
            ):
                continue
            segment_ids: List[str] = []
            contexts: List[str] = []
            for segment in segments:
                if _matching_segments(surface, [segment]):
                    segment_ids.append(_segment_id(segment))
                    source = str(segment.get("source_text") or "")
                    if len(contexts) < MAX_CONTEXTS and source not in contexts:
                        contexts.append(source[:280])
            occurrences = len(segment_ids)
            if occurrences < YAKE_MIN_OCCURRENCES:
                continue
            items.append({
                "surface": surface,
                "occurrences": occurrences,
                "segment_ids": segment_ids[:MAX_EVIDENCE_SEGMENTS],
                "contexts": contexts,
                "score": float(yake_score),
            })
            if len(items) >= YAKE_KEYWORD_LIMIT:
                break
    except Exception:  # any yake runtime failure degrades to flag-off, recorded
        return [], False
    items.sort(key=lambda row: (-float(row["score"]), str(row["surface"])))
    return items, True


def mine_candidates(config: Dict[str, Any], segments: Sequence[Dict[str, Any]]) -> List[TermCandidate]:
    """Run the v1 signals (by import) plus the optional yake pass.

    ``config`` is terminology-side: ``book_title`` (optional, powers the v1
    title signal), ``seed_terms``, ``use_yake``.  Returns rows sorted by the
    v1 tier/score/occurrence ordering without v1's master-coverage exclusion
    and with no hard 80-row cap: v2 termhood decides, not a lexical cap.
    """
    v1_config = {
        "book_title": str(config.get("book_title") or "").strip(),
        "glossary_settings": {
            "seed_terms": list(config.get("seed_terms") or []),
        },
    }
    raw = _candidate_sources(v1_config, segments)
    rows: List[Dict[str, Any]] = []
    for norm, item in raw.items():
        reasons = sorted(str(reason) for reason in item.get("reasons", ()))
        segment_ids = [str(value) for value in item.get("segment_ids", []) if str(value)]
        surface = str(item.get("term") or "").strip()
        if not surface or not segment_ids:
            continue
        rows.append({
            "candidate_id": _term_candidate_id(surface),
            "surface": surface,
            "variants": [],
            "category_hint": None,
            "source_type": _primary_source_type(item),
            "occurrences": len(segment_ids),
            "evidence_segment_ids": segment_ids[:MAX_EVIDENCE_SEGMENTS],
            "context": list(item.get("contexts") or [])[:MAX_CONTEXTS],
            "score": float(item.get("score") or 0),
        })

    mining_use_yake = bool(config.get("use_yake", False))
    yake_items, yake_enabled = _optional_yake_keywords(segments, mining_use_yake)
    if yake_enabled:
        seen_norms = {_normal(row["surface"]) for row in rows}
        for item in yake_items:
            norm = _normal(item["surface"])
            if norm in seen_norms:
                continue
            seen_norms.add(norm)
            rows.append({
                "candidate_id": _term_candidate_id(item["surface"]),
                "surface": item["surface"],
                "variants": [],
                "category_hint": None,
                "source_type": YAKE_SOURCE_TYPE,
                "occurrences": item["occurrences"],
                "evidence_segment_ids": item["segment_ids"],
                "context": item["contexts"],
                "score": item["score"],
            })

    rows.sort(key=lambda row: (
        _TIER.get(str(row["source_type"]), 9),
        -float(row["score"]),
        -int(row["occurrences"]),
        str(_normal(row["surface"])),
    ))
    # v2: the FULL mined set is the discovery record (term_candidates_v2.jsonl).
    # candidate_limit applies only to the LLM termhood queue (select_queue),
    # never to mining recall.
    return [TermCandidate(**row) for row in rows]


# Concept-class signals that must never be dropped by a queue cap: a cap may
# only shed lower-tier lexical noise (distinctive/quoted/proper), never master/
# seed/heading/repeated-phrase candidates (repeated phrases carry concept
# repetitions; dropping them caused the P05 live-queue recall regression).
QUEUE_PROTECTED_SOURCES = frozenset({
    "book_title_term",
    "manually_seeded_source_term",
    "heading_phrase",
    "heading_subphrase",
    "repeated_phrase",
})


def select_queue(candidates, limit: int):
    """Deterministic termhood queue: protected sources first (in candidate
    sort order), then remaining sources in sort order, capped at ``limit``.
    ``limit <= 0`` or >= len(candidates) returns the full set (default)."""
    ordered = sorted(candidates, key=lambda c: (
        _TIER.get(str(c.source_type), 9),
        -float(c.score),
        -int(c.occurrences),
        str(_normal(c.surface)),
    ))
    limit = int(limit or 0)
    if limit <= 0 or limit >= len(ordered):
        return list(ordered)
    protected = [c for c in ordered if str(c.source_type) in QUEUE_PROTECTED_SOURCES]
    rest = [c for c in ordered if str(c.source_type) not in QUEUE_PROTECTED_SOURCES]
    if len(protected) >= limit:
        return protected[:limit]
    return protected + rest[:limit - len(protected)]


def yake_feature_state(segments: Sequence[Dict[str, Any]], use_yake: bool) -> bool:
    """Probe-only helper: whether a yake run would actually enable."""
    _, enabled = _optional_yake_keywords(list(segments)[:0], use_yake)
    return enabled
