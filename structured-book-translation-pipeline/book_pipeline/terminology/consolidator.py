from __future__ import annotations

"""Book-level concept consolidation runner (spec §11.4)."""

from typing import Any, Dict, List

from .prompts import consolidator_system_prompt, consolidator_user_prompt
from .schemas import ConceptConsolidation


def run_consolidation(client: Any, model: str, summaries: List[Dict[str, Any]]) -> ConceptConsolidation:
    """One pro-model call folding accepted candidate summaries into concepts.

    Returns a pydantic-validated ``ConceptConsolidation`` (``extra="forbid"``);
    the deterministic compiler clamps constraints and enforces the polysemy
    non-merge contract afterwards.
    """
    book_id = ""
    for summary in summaries:
        if summary.get("book_id"):
            book_id = str(summary["book_id"])
            break
    system = consolidator_system_prompt(book_id)
    user = consolidator_user_prompt(summaries)
    parsed, _usage = client.complete_json(system, user, ConceptConsolidation)
    return parsed
