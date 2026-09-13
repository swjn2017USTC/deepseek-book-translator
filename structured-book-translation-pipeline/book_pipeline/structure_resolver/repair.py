from __future__ import annotations

"""Pass R: bounded conflict-neighborhood repair (plan §5.4, §5.5 step 9).

When ``apply_decisions`` returns a failing v2 gate, the orchestrator sends each
conflict window (windows whose decisions could not produce a passing tree) to a
``structure_review``-role call.  The reviewer sees the previous decisions, the
validator errors and the compressed window context, and may **only change
decision fields** — never source text, never free-text titles, never invented
candidate ids.  Revised records replace the prior round's records for the same
window; ``repair_round`` is stamped by the orchestrator.
"""

from typing import Any, Dict, List, Sequence, Tuple

from .prompts import repair_system_prompt, repair_user_prompt


def repair_window_decision_ids(
    prior_decisions: Sequence[Dict[str, Any]],
    validation: Dict[str, Any],
) -> List[str]:
    """Which candidate decisions belong to a failing neighborhood?

    The chapter validator keys errors by node ids; the deterministic
    provenance file (written by ``apply-decisions``) maps node ids back to the
    candidate ids the LLM decided.  When provenance is unavailable we fall back
    to every decided candidate in the round (conservative conflict
    neighborhood)."""
    node_ids: List[str] = []
    for key in ("errors", "gate_failures"):
        for item in validation.get(key) or []:
            if isinstance(item, dict) and item.get("node_id"):
                node_ids.append(str(item["node_id"]))
    if node_ids:
        return node_ids
    return [str(decision["candidate_id"]) for decision in prior_decisions]


def run_repair(
    evidence: Any,
    window: Dict[str, Any],
    prior_decisions: Sequence[Dict[str, Any]],
    validation: Dict[str, Any],
    client: Any,
    model: str,
    repair_round: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One reviewer call for one conflict window; returns revised records."""
    system_prompt = repair_system_prompt(evidence.book_id)
    user_prompt = repair_user_prompt(
        evidence.book_id,
        window["index"],
        repair_round,
        prior_decisions,
        validation,
        window,
    )
    revised, usage = client.complete_json(system_prompt, user_prompt, List[Dict[str, Any]])
    # StructureDecision-shaped intake happens in the orchestrator, which owns
    # the model/round stamping; here we only ensure a list of dicts came back.
    records: List[Dict[str, Any]] = []
    for item in revised or []:
        if not isinstance(item, dict):
            raise ValueError("repair response items must be JSON objects")
        records.append(dict(item))
    return records, dict(usage)
