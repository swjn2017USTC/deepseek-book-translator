from __future__ import annotations

"""Evaluation metrics of the terminology resolver v2 (spec §5.3 + §11 gate).

Deterministic-only module: every metric is computed from artifacts already on
disk, so offline sampling runs and live runs report the same shapes.  Metrics
surfaced (plan §6/D6):

* ``candidate_count_by_source`` — one count per candidate row under its primary
  v1 signal (or ``yake``), so the sums equal the mined candidate count.
* termhood accept/reject/needs_human counts + rates, incl.
  ``ordinary_phrase_reject_rate`` (rejected/judged) as the documented
  non-human precision proxy.
* concept/sense counts incl. ``polysemous_concept_count`` and
  ``sense_count_per_polysemous_concept``; ``duplicate_concept_merge_error`` is
  counted here and *enforced* to 0 by the orchestrator (hard fail otherwise).
* ``constraint_distribution`` and ``constraint_origin_distribution`` after the
  deterministic compiler clamp (D4).
* ``known_term_recall`` measured at mining AND after termhood retention.
* ``master_provenance_rate`` over accepted concepts.

The §11 "candidate precision 人工抽样" gate needs a human and is deliberately
**not** claimed here; the proxies above plus a shipped sampling protocol are
the honest PARTIAL evidence for P05 (deferred to the P09 human gate).
"""

from typing import Any, Dict, List, Optional, Sequence

from ..glossary import _normal
from .schemas import ConceptRecord, GlossaryConstraint, TermCandidate

# §11.6 human-review ordering groups (baked into concept_decisions.jsonl).
HUMAN_REVIEW_RANKS = {
    "high_impact_low_confidence": 1,
    "polysemous": 2,
    "conflicting_master_glossary": 3,
    "new_author_specific_concept": 4,
    "ordinary_name_or_entity": 5,
}
# Termhood needs_human threshold: accepted but low-confidence candidates go to
# the human gate instead of being silently trusted.
HUMAN_CONFIDENCE_THRESHOLD = 0.5


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def compute_mining_metrics(
    candidates: Sequence[TermCandidate],
    segments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    by_source: Dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.source_type or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    ordered = {source: by_source[source] for source in sorted(by_source)}
    return {
        "candidate_count": len(candidates),
        "candidate_count_by_source": ordered,
        "segments_count": len(segments),
    }


def known_term_recall(
    candidates: Sequence[TermCandidate],
    accepted_ids: Optional[Sequence[str]],
    known_terms: Sequence[Dict[str, Any]],
    queue_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Recall of ``sampling.known_terms`` staged honestly.

    Stages: (1) ``mined`` over the FULL discovery set (never capped);
    (2) ``in_queue`` over the termhood LLM queue (``queue_ids``; None/absent
    means the queue is the full set, e.g. default unlimited queue);
    (3) ``retained_after_termhood`` only when ``accepted_ids`` (the actual
    judged-and-accepted ids from termhood_decisions.jsonl) is provided.

    Label rules (no mislabeling): a mined term not in the judged set is
    ``queued_not_judged_yet`` (never ``rejected_by_termhood`` — rejection
    requires a recorded verdict); a mined term absent from the queue is
    ``dropped_by_queue_cap``; judged-rejected is ``rejected_by_termhood``.
    """
    by_norm: Dict[str, TermCandidate] = {}
    for candidate in candidates:
        norm = _normal(candidate.surface)
        by_norm.setdefault(norm, candidate)
    queue = set(queue_ids or ())
    queue_full = queue_ids is None
    accepted = set(accepted_ids or ())
    judged = accepted_ids is not None
    rows: List[Dict[str, Any]] = []
    mined_total = 0
    queued_total = 0
    retained_total = 0
    for known in known_terms:
        surface = str(known.get("surface") or "").strip()
        norm = _normal(surface)
        candidate = by_norm.get(norm)
        mined = candidate is not None
        candidate_id = candidate.candidate_id if candidate is not None else None
        in_queue = bool(candidate_id and (queue_full or candidate_id in queue))
        retained = bool(candidate_id and candidate_id in accepted)
        if mined:
            mined_total += 1
        if in_queue:
            queued_total += 1
        if retained:
            retained_total += 1
        if not mined:
            stage = "not_mined"
        elif in_queue and retained:
            stage = "retained"
        elif in_queue and judged and candidate_id in accepted:
            stage = "retained"
        elif in_queue and judged:
            stage = "rejected_by_termhood"
        elif in_queue:
            stage = "queued_not_judged_yet"
        else:
            stage = "dropped_by_queue_cap"
        rows.append({
            "surface": surface,
            "category": str(known.get("category") or ""),
            "mined": mined,
            "in_queue": in_queue,
            "candidate_id": candidate_id,
            "candidate_surface": candidate.surface if candidate is not None else None,
            "matched_source_type": candidate.source_type if candidate is not None else None,
            "first_evidence_segment_id": (
                candidate.evidence_segment_ids[0] if candidate is not None
                and candidate.evidence_segment_ids else None
            ),
            "retained_after_termhood": retained if judged else None,
            "loss_stage": stage,
        })
    total = len(known_terms)
    result: Dict[str, Any] = {
        "known_term_count": total,
        "mined_count": mined_total,
        "queued_count": queued_total if queue_ids is not None else None,
        "retained_count": retained_total if judged else None,
        "recall_at_mining": _rate(mined_total, total),
        "recall_at_queue": _rate(queued_total, total) if queue_ids is not None else None,
        "recall_after_termhood": _rate(retained_total, total) if judged else None,
        "per_term": rows,
    }
    if accepted_ids is None:
        result["note"] = ("termhood not run (offline): retention stage not measured; "
                          "queue stage reported when a queue limit is configured")
    return result


def termhood_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rows are the flushed termhood_decisions.jsonl records."""
    judged = len(rows)
    accepted = sum(1 for row in rows if row.get("verdict") == "accept")
    rejected = sum(1 for row in rows if row.get("verdict") == "reject")
    needs_human = sum(1 for row in rows if row.get("verdict") == "needs_human")
    return {
        "judged_count": judged,
        "accepted": accepted,
        "rejected": rejected,
        "needs_human": needs_human,
        "accept_rate": _rate(accepted, judged),
        "reject_rate": _rate(rejected, judged),
        "needs_human_rate": _rate(needs_human, judged),
        # Proxy for precision (spec §5.3 candidate precision): share of judged
        # candidates the LLM rejected as ordinary high-frequency phrases.
        "ordinary_phrase_reject_rate": _rate(rejected, judged),
    }


def _surface_norms(concept: ConceptRecord) -> List[str]:
    norms = [_normal(concept.canonical_source)]
    for surface in concept.surface_forms:
        norm = _normal(str(surface))
        if norm:
            norms.append(norm)
    return norms


def duplicate_concept_errors(concepts: Sequence[ConceptRecord]) -> List[Dict[str, Any]]:
    """Detect two concepts sharing one normalized surface (merge error).

    Surface lists are deduplicated *within* a concept first (canonical plus
    identical surface_forms entries are one identity, never a merge error).
    Errors then mean the consolidator produced two concepts for one surface —
    either the same concept id twice (identical canonical sources are
    deterministic-id collisions after the compiler rewrite) or one surface
    under two different ids — the §11.4 same-surface-many-senses violation:
    N senses must live in ONE concept's ``senses[]``.
    """
    owner: Dict[str, str] = {}
    seen_ids: Dict[str, str] = {}
    errors: List[Dict[str, Any]] = []
    for concept in concepts:
        if concept.concept_id in seen_ids:
            errors.append({
                "normalized_surface": _normal(concept.canonical_source),
                "concept_ids": [concept.concept_id, concept.concept_id],
                "note": "same concept id emitted twice by the consolidator",
            })
        seen_ids.setdefault(concept.concept_id, concept.canonical_source)
        for norm in dict.fromkeys(_surface_norms(concept)):
            prior = owner.get(norm)
            if prior is not None and prior != concept.concept_id:
                errors.append({
                    "normalized_surface": norm,
                    "concept_ids": sorted({prior, concept.concept_id}),
                })
            else:
                owner[norm] = concept.concept_id
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for error in errors:
        key = (error["normalized_surface"], tuple(error["concept_ids"]))
        deduped.setdefault(key, error)
    return sorted(deduped.values(), key=lambda error: (error["normalized_surface"], str(error["concept_ids"])))


def polysemy_check(concepts: Sequence[ConceptRecord]) -> Dict[str, Any]:
    concept_count = len(concepts)
    sense_count = sum(len(concept.senses) for concept in concepts)
    polysemous = [concept for concept in concepts if len(concept.senses) > 1]
    errors = duplicate_concept_errors(concepts)
    return {
        "concept_count": concept_count,
        "sense_count": sense_count,
        "polysemous_concept_count": len(polysemous),
        "sense_count_per_polysemous_concept": sorted(
            (len(concept.senses) for concept in polysemous), reverse=True
        ),
        "avg_senses_per_concept": round(sense_count / concept_count, 4) if concept_count else 0.0,
        # Same-surface polysemy must map to ONE concept with N senses; two
        # concepts sharing a surface is a duplicate-merge error (target 0,
        # enforced fail-closed by the orchestrator).
        "duplicate_concept_merge_error": len(errors),
        "duplicate_concept_merge_details": errors,
    }


def constraint_distribution(concepts: Sequence[ConceptRecord]) -> Dict[str, Any]:
    counts = {constraint.value: 0 for constraint in GlossaryConstraint}
    origin_counts: Dict[str, int] = {}
    for concept in concepts:
        origin = str(concept.constraint_origin or "llm")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        for sense in concept.senses:
            counts[str(sense.constraint.value)] += 1
    total = sum(counts.values())
    distribution = {
        value: {"count": counts[value], "rate": _rate(counts[value], total)}
        for value in sorted(counts)
    }
    return {
        "sense_count": total,
        "constraint_distribution": distribution,
        "constraint_origin_distribution": {
            origin: origin_counts[origin] for origin in sorted(origin_counts)
        },
        "hard_sense_rate": _rate(counts["hard"], total),
    }


def master_provenance_rate(concepts: Sequence[ConceptRecord]) -> Dict[str, Any]:
    provened = [
        concept for concept in concepts
        if concept.master_source or concept.master_aliases
    ]
    return {
        "concept_count": len(concepts),
        "master_provenanced_concept_count": len(provened),
        "master_provenance_rate": _rate(len(provened), len(concepts)),
    }


def needs_human_rows(
    concepts: Sequence[ConceptRecord],
    needs_human_surfaces: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """§11.6 review-template rows: only the genuinely high-value gray zone.

    Goal: the LLM rejects ordinary phrases automatically; the human gate then
    sees a *small* ranked list, never the full candidate set.  A concept is
    sent for review when it is (1) high-impact + low-confidence (any sense
    confidence below the threshold, or its surface was a termhood
    ``needs_human`` verdict), (2) polysemous (one surface, N senses), or
    (3) a conflicting master-glossary entry (master source present but the
    consolidated translation differs).  Ordinary non-master author terms the
    LLM accepted confidently are machine-approved and are NOT listed here.
    Rows are ordered by rank 1 -> 3 then confidence ascending.
    """
    gray_norms = {_normal(str(surface)) for surface in (needs_human_surfaces or [])}
    rows: List[Dict[str, Any]] = []
    for concept in concepts:
        min_confidence = min(sense.confidence for sense in concept.senses)
        surface_norms = {_normal(concept.canonical_source)}
        surface_norms.update(_normal(surface) for surface in concept.surface_forms)
        rank = None
        reason = ""
        if min_confidence < HUMAN_CONFIDENCE_THRESHOLD:
            rank = HUMAN_REVIEW_RANKS["high_impact_low_confidence"]
            reason = f"lowest sense confidence {min_confidence} below {HUMAN_CONFIDENCE_THRESHOLD}"
        elif surface_norms & gray_norms:
            rank = HUMAN_REVIEW_RANKS["high_impact_low_confidence"]
            reason = "termhood judged this surface needs human review (low confidence)"
        elif len(concept.senses) > 1:
            rank = HUMAN_REVIEW_RANKS["polysemous"]
            reason = f"{len(concept.senses)} in-book senses on one surface"
        elif bool(concept.master_source) and any(
            str(sense.translation).strip() != str(concept.master_source).strip()
            for sense in concept.senses
        ):
            rank = HUMAN_REVIEW_RANKS["conflicting_master_glossary"]
            reason = "consolidator translation differs from master glossary"
        if rank is None:
            continue
        rows.append({
            "concept_id": concept.concept_id,
            "decision": "needs_human",
            "canonical_source": concept.canonical_source,
            "category": concept.category,
            "rank": rank,
            "priority": next(label for label, value in HUMAN_REVIEW_RANKS.items() if value == rank),
            "reason": reason,
            "senses_count": len(concept.senses),
            "min_confidence": min_confidence,
        })
    rows.sort(key=lambda row: (int(row["rank"]), float(row["min_confidence"]), str(row["concept_id"])))
    return rows
