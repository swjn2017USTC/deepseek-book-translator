"""QA-2 terminology audit (spec §13, plan §3.2).

Four deterministic outputs over the audited segments:

1. **term success rate** — ``matched / applicable``: the fraction of
   applicable concept contexts whose canonical translation actually appears
   in the target text (verbatim containment, the same convention
   ``contextual.ladder`` uses).
2. **consistency** — one source surface must map to one translation across
   the book; a segment using a minority translation for a surface is flagged.
3. **hard-constraint violations** — deterministic FAIL rows reusing
   ``contextual.ladder._violations`` (read-only): a ``hard`` concept whose
   surface occurs in the source but whose canonical translation is absent
   from the target.
4. **sense-constrained suspicious contexts** — ``sense_constrained`` concepts
   whose surface is present but the canonical translation is absent: flagged
   for the QA-4 critic (context judgment is required — never a hard fail).

Constraint semantics (``terminology.schemas.GlossaryConstraint``):
``hard`` -> deterministic violation; ``sense_constrained`` -> critic-flagged;
``preferred`` -> advisory only (success-rate contributor, never a flag);
``do_not_translate`` -> the source surface must appear verbatim in the target;
``first_occurrence`` -> the translation is expected only on the first
book-ordered segment containing the surface, later segments expect the source
form verbatim.

Glossary source resolution: the P06 concept glossary (schema v2,
``glossary_dependencies.concept_glossary_path`` — P06 fixture paths
tolerated) when enabled and present; else the v1 glossary term list
(``config["glossary"]`` — the compiled ``{term, translation}`` file v1
translation actually consumes; v1 rows are advisory/preferred, contributing
success rate + consistency only); else the audit completes with an empty term
set (deterministic stages never hard-require a glossary).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import resolve_path
from ..contextual import ladder as _ladder  # noqa: W0212 — _violations rule reuse by design
from ..contextual.deps import (
    concept_sense,
    load_concept_glossary,
    matched_surface,
    resolve_concept_deps,
)
from ..glossary import _phrase_pattern  # noqa: W0212 — read-only convention reuse (P06 precedent)
from ..models import Segment


def concept_glossary_path(config: Dict[str, Any]) -> Optional[Path]:
    """P06 concept-glossary path when dependency mode is enabled (fixture
    absolute paths tolerated); ``None`` otherwise."""
    settings = config.get("glossary_dependencies") or {}
    if not settings.get("enabled"):
        return None
    value = settings.get("concept_glossary_path")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_base_dir"]) / path).resolve()


def glossary_terms(config: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Resolve the QA-2 glossary source -> ``(mode, terms)`` with mode in
    ``"concept"`` / ``"v1"`` / ``"none"``.

    A malformed concept file raises ``ValueError`` (never silently misreads);
    a missing v1 file yields ``"none"`` so deterministic stages still
    complete with zero applicable terms.
    """
    concept_path = concept_glossary_path(config)
    if concept_path is not None and concept_path.is_file():
        return "concept", load_concept_glossary(concept_path)
    v1_path = resolve_path(config, "glossary") if config.get("glossary") else None
    if v1_path is not None and v1_path.is_file():
        raw = json.loads(v1_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            rows = [
                row for row in raw
                if isinstance(row, dict) and str(row.get("term") or "").strip()
            ]
            return "v1", rows
    return "none", []


def _expectation(concept: Dict[str, Any]) -> Tuple[str, str]:
    """``(constraint, canonical_translation)`` from the concept's governing
    sense (default sense else max version — the same sense ``glossary_subset``
    presents to the translator)."""
    sense = concept_sense(concept) or {}
    constraint = str(sense.get("constraint") or "preferred")
    return constraint, str(sense.get("translation") or "")


def _surface_in_source(concept: Dict[str, Any], source_text: str) -> Optional[str]:
    """Matched surface form (phrase boundary) or ``None``."""
    return matched_surface(concept, source_text)


def run_terminology_audit(
    audited: Sequence[Tuple[Segment, Dict[str, Any]]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run QA-2 over ``audited`` (cleaned-order ``(segment, completed row)``).

    Returns ``{"flags": [...], "metrics": {...}}``.  ``metrics`` carries
    ``term_success_rate`` / ``applicable`` / ``matched`` /
    ``hard_violations`` / ``consistency_flags`` / ``suspicious_contexts`` /
    ``glossary_source``.
    """
    mode, terms = glossary_terms(config)
    flags: List[Dict[str, Any]] = []
    concepts_by_id: Dict[str, Dict[str, Any]] = {}
    if mode == "concept":
        concepts_by_id = {
            str(concept.get("concept_id") or ""): concept
            for concept in terms
            if str(concept.get("concept_id") or "").strip()
        }
    v1_patterns = [
        (row, _phrase_pattern(str(row.get("term") or "")))
        for row in terms
    ] if mode == "v1" else []

    # Book-ordered first occurrence per concept (first_occurrence semantics).
    first_occurrence_seen: Dict[str, int] = {}
    # surface -> uses of actual translations (consistency book map).
    uses_by_surface: Dict[str, List[Dict[str, str]]] = {}

    applicable = 0
    matched_count = 0
    hard_violations = 0
    consistency_flags = 0
    suspicious_contexts = 0

    def flag(segment: Segment, check: str, severity: str, detail: str) -> None:
        flags.append({
            "segment_id": segment.id,
            "stage": "qa-2",
            "check": check,
            "severity": severity,
            "expected": 1,
            "actual": 0,
            "detail": detail,
        })

    def record_use(surface: str, segment_id: str, translation: str) -> None:
        uses_by_surface.setdefault(surface, []).append(
            {"segment_id": segment_id, "translation": translation}
        )

    for ordinal, (segment, row) in enumerate(audited):
        target = str(row.get("translated_text") or "")
        source = segment.source_text

        if mode == "concept":
            dependencies = resolve_concept_deps(segment, terms)
            # 1) hard violations — the exact ladder rule, read-only reused.
            for concept_id in _ladder._violations(  # noqa: W0212 — rule reuse
                segment, target, dependencies, concepts_by_id
            ):
                concept = concepts_by_id[concept_id]
                constraint, translation = _expectation(concept)
                surface = matched_surface(concept, source) or str(
                    concept.get("canonical_source") or concept_id
                )
                hard_violations += 1
                flag(
                    segment,
                    "hard_violation",
                    "high",
                    f"hard concept {concept_id!r} ({surface!r} in source) expects "
                    f"translation {translation!r} in target; absent",
                )
            # 2) per-dependency contexts (skipping already-flagged hard rows).
            for concept in terms:
                concept_id = str(concept.get("concept_id") or "").strip()
                if not concept_id or concept_id not in dependencies:
                    continue
                constraint, translation = _expectation(concept)
                surface = matched_surface(concept, source)
                if surface is None:
                    continue
                if constraint == "hard" and concept_id in _ladder._violations(
                    segment, target, dependencies, concepts_by_id
                ):
                    continue  # already flagged above
                ordinal_first = concept_id not in first_occurrence_seen
                if concept_id not in first_occurrence_seen:
                    first_occurrence_seen[concept_id] = ordinal

                if constraint == "do_not_translate":
                    # Source surface must survive verbatim.
                    if surface not in target:
                        flag(
                            segment,
                            "do_not_translate",
                            "high",
                            f"do-not-translate surface {surface!r} must appear "
                            "verbatim in the target; absent",
                        )
                    continue
                if constraint == "first_occurrence" and not ordinal_first:
                    # Later occurrences expect the source form verbatim.
                    if surface not in target:
                        flag(
                            segment,
                            "first_occurrence",
                            "flag",
                            f"concept {concept_id!r} first-occurrence rule: "
                            f"later use of {surface!r} expects the source form "
                            "verbatim in the target; absent",
                        )
                    continue
                # Translation-bearing context: contributes to the success rate.
                if constraint != "first_occurrence" or ordinal_first:
                    applicable += 1
                    if translation and translation in target:
                        matched_count += 1
                        record_use(surface, segment.id, translation)
                    elif constraint == "sense_constrained":
                        suspicious_contexts += 1
                        flag(
                            segment,
                            "suspicious_context",
                            "flag",
                            f"sense-constrained concept {concept_id!r} "
                            f"({surface!r}): surface present but expected "
                            f"translation {translation!r} absent",
                        )
                    elif constraint == "first_occurrence" and ordinal_first:
                        # First occurrence without its translation: suspicious
                        # (never a hard fail — the surface may be re-introduced
                        # in its source form by design).
                        if translation and translation not in target:
                            suspicious_contexts += 1
                            flag(
                                segment,
                                "suspicious_context",
                                "flag",
                                f"first-occurrence concept {concept_id!r} "
                                f"({surface!r}) appears first here but its "
                                f"translation {translation!r} is absent",
                            )

        elif mode == "v1":
            for row_item, pattern in v1_patterns:
                if not pattern.search(source):
                    continue
                surface = str(row_item.get("term") or "")
                translation = str(row_item.get("translation") or "")
                if not translation:
                    continue
                applicable += 1
                if translation in target:
                    matched_count += 1
                    record_use(surface, segment.id, translation)

    # Book-level consistency: surface -> {distinct translations used}; every
    # use of a minority variant is a consistency flag.
    segment_by_id = {segment.id: segment for segment, _ in audited}
    for surface, uses in uses_by_surface.items():
        variants: Dict[str, List[str]] = {}
        for use in uses:
            variants.setdefault(use["translation"], []).append(use["segment_id"])
        if len(variants) <= 1:
            continue
        majority = max(variants.items(), key=lambda item: len(item[1]))[0]
        for translation, segment_ids in variants.items():
            if translation == majority:
                continue
            consistency_flags += 1
            for segment_id in segment_ids:
                flag(
                    segment_by_id[segment_id],
                    "consistency",
                    "flag",
                    f"surface {surface!r} translated {translation!r} here but "
                    f"{majority!r} elsewhere; book variants={sorted(variants)}",
                )

    return {
        "flags": flags,
        "metrics": {
            "glossary_source": mode,
            "applicable": applicable,
            "matched": matched_count,
            "term_success_rate": round(matched_count / applicable, 6) if applicable else 1.0,
            "hard_violations": hard_violations,
            "consistency_flags": consistency_flags,
            "suspicious_contexts": suspicious_contexts,
        },
    }
