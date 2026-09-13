from __future__ import annotations

"""Pass C: heading rescue (plan §5.4/§3.5).

Per deterministic window, the provider reviews a compressed directory of the
blocks that produced candidates in that window and names blocks that look like
headings the deterministic machinery missed.  Intake enforces the cross-repo
contract hard rule: **existing ``blk-*`` ids only** — a hallucinated block id
is rejected (the chapter side re-establishes full evidence for returned blocks
and synthesizes a fresh ``cand-*`` before the normal decision flow).
"""

from typing import Any, Dict, List, Sequence, Tuple

from .prompts import rescue_system_prompt, rescue_user_prompt
from .schemas import RescueCandidate


class RescueParseError(RuntimeError):
    """A rescue window returned blocks that are not in the live block set.

    A RuntimeError (not ValueError) so the orchestrator fails closed with the
    specific intake message instead of retrying on a deterministic rejection."""


def window_block_ids(window: Dict[str, Any]) -> List[str]:
    """Deterministic, ordered union of source block ids visible in a window."""
    seen: List[str] = []
    for candidate in window["candidates"]:
        for block_id in candidate.get("source_block_ids") or []:
            if str(block_id) not in seen:
                seen.append(str(block_id))
    return seen


def validate_rescue_records(
    window: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Intake of one window's rescue records: existing block ids only."""
    existing = set(window_block_ids(window))
    errors: List[str] = []
    valid: List[Dict[str, Any]] = []
    for record in records:
        data = dict(record)
        block_id = str(data.get("block_id") or "")
        if block_id not in existing:
            errors.append(f"block_id {block_id!r} not in the window's block directory")
            continue
        valid.append(data)
    if errors:
        raise RescueParseError("; ".join(errors))
    return valid


def run_rescue_pass(
    evidence: Any,
    window: Dict[str, Any],
    client: Any,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One rescue call per window; returns ``(validated_records, usage)``."""
    rescued, usage = client.complete_json(
        rescue_system_prompt(evidence.book_id),
        rescue_user_prompt(evidence.book_id, window, window["index"]),
        List[RescueCandidate],
    )
    records = [
        {
            "block_id": str(item.block_id),
            "reason": item.reason,
            "confidence": float(item.confidence),
            "evidence_refs": list(item.evidence_refs),
        }
        for item in rescued
    ]
    return validate_rescue_records(window, records), dict(usage)
