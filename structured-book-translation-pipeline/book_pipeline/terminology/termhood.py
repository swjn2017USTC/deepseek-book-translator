from __future__ import annotations

"""Chapter-window termhood judgment runner (spec §11.3)."""

from typing import Any, Dict, List

from .prompts import term_miner_system_prompt, term_miner_user_prompt
from .schemas import TermhoodJudgment


def run_termhood_window(client: Any, model: str, window: Dict[str, Any]) -> List[TermhoodJudgment]:
    """One flash-model call judging every candidate assigned to ``window``.

    Returns pydantic-validated judgments (``extra="forbid"``); the caller
    enforces exact coverage and rejects hallucinated ids deterministically.
    """
    system = term_miner_system_prompt(str(window.get("book_id") or ""))
    user = term_miner_user_prompt(window)
    parsed, usage = client.complete_json(system, user, List[TermhoodJudgment])
    return list(parsed)
