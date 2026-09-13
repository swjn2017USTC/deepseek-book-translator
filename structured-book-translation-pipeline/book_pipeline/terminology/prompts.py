"""Prompt builders for the terminology resolver v2 (spec §11.3/§11.4, §18).

``PROMPTS_VERSION`` stamps every judgment/summary record so prompt drift is
traceable.  Prompts carry **bounded context only** (§18): mined candidates plus
short source snippets and matching segment ids — never whole chapters.  The
consolidator receives only accepted-candidate summaries plus master provenance
hints ("只看候选摘要"), never full book text.
"""

import json
from typing import Any, Dict, List, Optional, Sequence


PROMPTS_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _trim(text: Optional[str], limit: int = 200) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ------------------------------------------------------------------ termhood
def term_miner_system_prompt(book_id: str) -> str:
    return (
        "You are the term_miner for one chapter window of a book. Your job is "
        "the termhood judgment: decide whether each mined expression, if its "
        "translation were inconsistent across the book, would actually damage "
        "concept comprehension or proper-name consistency.\n"
        "Rules:\n"
        "1. Judge ONLY candidates listed in the window payload; never invent "
        "candidate_id values or new surfaces.\n"
        "2. is_term true = genuine term of this book worth a stable "
        "translation. is_term false = ordinary high-frequency phrase or "
        "generic wording that needs no glossary entry; reason is then "
        "required.\n"
        "3. category must be one of: concept, person, place, org, event, "
        "work, term. Use null when unsure.\n"
        "4. possible_senses: 1 for a single-sense term; 2+ only when the "
        "surface genuinely carries distinct in-book senses that need "
        "different translations.\n"
        "5. confidence is a float 0..1. Return ONLY a JSON list of judgment "
        "objects {candidate_id, is_term, category, canonical_surface, "
        "variants, definition_in_book, translation_sensitive, "
        "possible_senses, contexts, confidence, reason}.\n"
        "6. One judgment object per listed candidate, nothing more. "
        f"Book: {book_id}."
    )


def term_miner_user_prompt(window: Dict[str, Any]) -> str:
    """One window: candidates + bounded snippets/segment ids only."""
    payload = {
        "book_id": window.get("book_id"),
        "window_index": window.get("window_index"),
        "window_label": window.get("label"),
        "window_title": _trim(window.get("title"), 160),
        "segments_in_window": window.get("segment_count"),
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "surface": candidate["surface"],
                "variants": list(candidate.get("variants") or []),
                "category_hint": candidate.get("category_hint"),
                "source_type": candidate.get("source_type"),
                "occurrences": candidate.get("occurrences"),
                "contexts": list(candidate.get("context") or [])[:3],
                "evidence_segment_ids": list(candidate.get("evidence_segment_ids") or [])[:12],
            }
            for candidate in window.get("candidates", [])
        ],
    }
    return _json(payload)


# ------------------------------------------------------------- consolidation
def consolidator_system_prompt(book_id: str) -> str:
    return (
        "You are the term_consolidator for one whole book. Input is the set of "
        "termhood-accepted candidate summaries (never full text). Build the "
        "book concept glossary by:\n"
        "1. merging spelling variants, British/American forms, abbreviations "
        "and person-name forms into one concept (as alias surfaces or "
        "surface_forms);\n"
        "2. modelling same-surface polysemy as ONE concept with one Sense per "
        "in-book sense — never two concepts for one surface;\n"
        "3. preferring the master glossary translation when a master hint is "
        "present and nothing in the evidence contradicts it;\n"
        "4. keeping author-specific concepts when they carry real in-book "
        "meaning.\n"
        "Constraint guidance (§11.5): 'hard' ONLY for person/place/org/event/"
        "work names or master-backed terms with evidence; 'sense_constrained' "
        "for polysemous concepts; 'preferred' (overridable) is the default; "
        "'do_not_translate' for formulas/verbatim titles; 'first_occurrence' "
        "for gloss-on-first-mention. Do NOT make every term hard.\n"
        "Output ONLY one JSON object: {concepts: [{concept_id "
        "('concept-'+12 hex lowercase, deterministic only as a draft), "
        "canonical_source, surface_forms, category (concept/person/place/org/"
        "event/work/term), senses: [{sense_id ('default' or 'sense-<n>'), "
        "translation, constraint, definition_in_book, evidence_segments, "
        "confidence, version:1}], master_source, master_aliases}], merges: "
        "[{concept_id, alias_surface}]}. Never invent evidence segments; reuse "
        "only segment ids from the summaries. "
        f"Book: {book_id}."
    )


def consolidator_user_prompt(summaries: Sequence[Dict[str, Any]]) -> str:
    payload = {
        "candidate_summaries": [
            {
                "candidate_id": summary.get("candidate_id"),
                "surface": summary.get("surface"),
                "category": summary.get("category"),
                "canonical_surface": summary.get("canonical_surface"),
                "definition_in_book": _trim(summary.get("definition_in_book"), 300),
                "translation_sensitive": bool(summary.get("translation_sensitive")),
                "possible_senses": summary.get("possible_senses"),
                "confidence": summary.get("confidence"),
                "contexts": [_trim(context, 160) for context in (summary.get("contexts") or [])[:2]],
                "evidence_segment_ids": list(summary.get("evidence_segment_ids") or [])[:8],
                "master": summary.get("master"),
            }
            for summary in summaries
        ],
    }
    return _json(payload)
