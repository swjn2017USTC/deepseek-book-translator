from __future__ import annotations

"""Pass A: global book grammar (plan §5.4).

Exactly one LLM call with role ``structure_global`` on the compressed
``book_structure_evidence.json`` summary.  The parsed ``BookGrammar`` is
advisory context for Pass B/C windows — never an artifact and never a place
where candidate-level decisions or free-text titles may appear.
"""

from typing import Any, Dict, Tuple

from .prompts import grammar_system_prompt, grammar_user_prompt
from .schemas import BookGrammar


def run_grammar_pass(
    evidence: Any,
    client: Any,
    model: str,
) -> Tuple[BookGrammar, Dict[str, Any]]:
    """Call the provider once for the grammar summary; returns
    ``(grammar, usage)``."""
    grammar, usage = client.complete_json(
        grammar_system_prompt(evidence.book_id),
        grammar_user_prompt(evidence),
        BookGrammar,
    )
    if not isinstance(grammar, BookGrammar):
        raise ValueError("Pass A did not return a BookGrammar instance")
    return grammar, dict(usage)
