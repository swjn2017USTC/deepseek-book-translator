"""Prompt builders for the v2 structure resolver (plan §5.4).

``PROMPTS_VERSION`` stamps every prompt and every decision/rescue record so
prompt drift is traceable.  Prompts carry **compressed evidence only** — no
full book text is ever sent upstream: candidate context is trimmed to
id/source_block_ids/exact_text/page/zone/label/numbering/repetition counts and
short before/after snippets, and rescue/repair prompts reuse the same
compressed windows.
"""

import json
from typing import Any, Dict, List, Optional, Sequence


PROMPTS_VERSION = 1


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _trim(text: Optional[str], limit: int = 90) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _zone_ranges(zones: Dict[int, str], page_count: int) -> List[Dict[str, Any]]:
    """Compress a page->zone map into contiguous [start, end) ranges."""
    ordered = sorted(int(page) for page in zones)
    ranges: List[Dict[str, Any]] = []
    if not ordered:
        return ranges
    start = ordered[0]
    previous = ordered[0]
    previous_zone = zones[previous]
    for page in ordered[1:]:
        if page != previous + 1 or zones[page] != previous_zone:
            ranges.append({"zone": previous_zone, "start": start, "end": previous + 1})
            start = page
        previous = page
        previous_zone = zones[page]
    ranges.append({"zone": previous_zone, "start": start, "end": previous + 1})
    return ranges


def grammar_user_prompt(evidence: Any) -> str:
    """Pass A: derive the book grammar from the evidence summary only."""
    metadata = evidence.metadata
    summary = {
        "book_id": evidence.book_id,
        "book_title": evidence.book_title,
        "page_count": evidence.page_count,
        "pdf_evidence_ok": bool(evidence.pdf_evidence and evidence.pdf_evidence.get("ok")),
        "pdf_reason": (evidence.pdf_evidence or {}).get("reason"),
        "candidate_count": evidence.candidate_count,
        "toc_entries": [
            {"kind": entry.get("kind"), "title": _trim(entry.get("title"), 120)}
            for entry in evidence.toc_entries
        ],
        "numbering_summary": evidence.numbering_summary,
        "style_clusters": evidence.style_clusters,
        "zone_ranges": _zone_ranges(evidence.zones, evidence.page_count),
        "repeated_headers": evidence.repeated_headers,
        "page_density": evidence.page_density,
        "source_counts": metadata.get("source_counts"),
        "frontmatter_page_0": evidence.zone_of(0) if evidence.zones else None,
    }
    return _json(summary)


def grammar_system_prompt(book_id: str) -> str:
    return (
        "You derive the global structure grammar of a scanned book from its "
        "deterministic evidence summary. Return ONLY a compact JSON object with "
        "these fields: book_id, frontmatter_kinds (list), backmatter_kinds "
        "(list), numbering_families (list of observed numbering family names), "
        "chapter_family (the family used for top-level chapters, or null), "
        "section_family (the family used for lower-level sections, or null), "
        "notes (short list of structural conventions relevant to deciding "
        "headings). Never invent candidate ids, titles, or page numbers. "
        f"Book: {book_id}."
    )


def local_window_system_prompt(book_id: str) -> str:
    return (
        "You classify OCR heading candidates of one page window of a book into "
        "structure decisions. Rules:\n"
        "1. Only decide candidates listed in the window; never invent "
        "candidate_id, exact_text, title, title_source, or source text.\n"
        "2. decision is one of heading/body/caption/apparatus/merge. kind is "
        "required for heading only: frontmatter/part/part_intro/chapter/"
        "section/backmatter/toc_entry. level is required for heading only: an "
        "integer 1..6.\n"
        "3. parent_candidate_id may reference a decided heading candidate in "
        "the same window or an earlier window; use null for top-level.\n"
        "4. confidence is a float 0..1; evidence_refs is a non-empty list of "
        "the candidate's own evidence refs.\n"
        "5. Do not emit titles, rewritten text, or any field other than the "
        "allowed decision fields.\n"
        "6. Return ONLY a JSON list of decision objects. Book: " + book_id + "."
    )


def local_window_user_prompt(
    grammar: Any,
    window: Dict[str, Any],
) -> str:
    """Pass B: decide every candidate in one deterministic window."""
    payload = {
        "book_id": window["book_id"],
        "window_index": window["index"],
        "window_pages": [window["start_page"], window["end_page"] - 1],
        "grammar": grammar.model_dump() if grammar is not None else None,
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "source_block_ids": list(candidate.get("source_block_ids") or []),
                "exact_text": _trim(candidate.get("exact_text"), 160),
                "page_json": candidate.get("page_json"),
                "page_print": candidate.get("page_print"),
                "zone": candidate.get("zone") or "mainmatter",
                "paddle_label": candidate.get("paddle_label"),
                "numbering": candidate.get("numbering"),
                "repetition": candidate.get("repetition"),
                "toc_matches": len(candidate.get("toc_matches") or []),
                "before_context": _trim(candidate.get("before_context")),
                "after_context": _trim(candidate.get("after_context")),
                "evidence_refs": list(candidate.get("evidence_refs") or []),
            }
            for candidate in window["candidates"]
        ],
        "output": "one decision object per candidate: "
        "{candidate_id, decision, kind, level, parent_candidate_id, "
        "confidence, evidence_refs}",
    }
    return _json(payload)


def rescue_system_prompt(book_id: str) -> str:
    return (
        "You detect headings that the deterministic machinery missed. You may "
        "only name blocks that already exist in the provided directory; never "
        "invent block ids or titles. Return ONLY a JSON list of rescue objects "
        "{block_id, reason, confidence, evidence_refs} where reason is "
        "'missed_heading'. Empty list [] when nothing needs rescue. "
        f"Book: {book_id}."
    )


def rescue_user_prompt(book_id: str, window: Dict[str, Any], window_index: int) -> str:
    blocks: List[Dict[str, Any]] = []
    seen = set()
    for candidate in window["candidates"]:
        for block_id in candidate.get("source_block_ids") or []:
            if block_id in seen:
                continue
            seen.add(block_id)
            blocks.append({
                "block_id": block_id,
                "text_prefix": _trim(candidate.get("exact_text"), 60),
                "paddle_label": candidate.get("paddle_label"),
                "page_json": candidate.get("page_json"),
                "bbox": candidate.get("bbox"),
                "zone": candidate.get("zone") or "mainmatter",
            })
    payload = {
        "book_id": book_id,
        "window_index": window_index,
        "pages": [window["start_page"], window["end_page"] - 1],
        "block_directory": blocks,
    }
    return _json(payload)


def _error_codes(validation: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    for key in ("errors", "gate_failures"):
        for item in validation.get(key) or []:
            if isinstance(item, dict) and item.get("code"):
                codes.append(str(item["code"]))
            elif isinstance(item, str):
                codes.append(item)
    return codes


def repair_system_prompt(book_id: str) -> str:
    return (
        "You repair structure decisions that failed a deterministic validator. "
        "Rules:\n"
        "1. Only revise decisions for candidates named in the review request; "
        "never invent candidate ids or titles.\n"
        "2. A revised decision is the SAME shape as the original: "
        "{candidate_id, decision, kind, level, parent_candidate_id, "
        "confidence, evidence_refs}.\n"
        "3. You never change source text; you only change the decision fields "
        "(decision/kind/level/parent_candidate_id/confidence).\n"
        "4. Every emitted candidate_id must exist in the candidate directory.\n"
        "5. Return ONLY a JSON list of the revised decision objects. "
        f"Book: {book_id}."
    )


def repair_user_prompt(
    book_id: str,
    window_index: int,
    repair_round: int,
    prior_decisions: Sequence[Dict[str, Any]],
    validation: Dict[str, Any],
    window: Dict[str, Any],
) -> str:
    """Conflict-neighborhood repair (Pass R): prior decisions + validator
    errors + the compressed window they came from."""
    payload = {
        "book_id": book_id,
        "window_index": window_index,
        "repair_round": repair_round,
        "validator_errors": validation.get("errors") or [],
        "validator_gate_failures": validation.get("gate_failures") or [],
        "error_codes": _error_codes(validation),
        "decisions_to_review": [
            {
                "candidate_id": decision.get("candidate_id"),
                "decision": decision.get("decision"),
                "kind": decision.get("kind"),
                "level": decision.get("level"),
                "parent_candidate_id": decision.get("parent_candidate_id"),
                "confidence": decision.get("confidence"),
                "evidence_refs": decision.get("evidence_refs"),
                "repair_round": decision.get("repair_round", 0),
            }
            for decision in prior_decisions
        ],
        "neighborhood_candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "exact_text": _trim(candidate.get("exact_text"), 120),
                "page_json": candidate.get("page_json"),
                "zone": candidate.get("zone") or "mainmatter",
                "paddle_label": candidate.get("paddle_label"),
                "numbering": candidate.get("numbering"),
            }
            for candidate in window["candidates"]
        ],
    }
    return _json(payload)
