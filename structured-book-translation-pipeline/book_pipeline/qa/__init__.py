"""QA cascade for the structured-book translation pipeline (spec §13, P07).

Public surface: ``run_qa`` (read-only cascade QA-1 -> QA-2 -> QA-3 -> QA-4,
artifacts: qa_report.json / qa_audit.jsonl / qa_usage.jsonl + the
translation_status extension), ``retranslate_likely_errors`` (the ONLY writer:
appends revision rows for persisted ``likely_error`` verdicts), and the
validated ``QACriticVerdict``/``QAVerdict`` schema models.

Design contract (hard): QA never edits translated text; retranslation keeps
the original record and appends a revision record with the same
segment_id/source_hash; provider-down => QA-4 status ``provider_unavailable``
with deterministic QA-1/2/3 still complete; verdicts are never fabricated.
"""

from __future__ import annotations

from .orchestrator import run_qa
from .retranslate import retranslate_likely_errors
from .schemas import QACriticVerdict, QAVerdict

__all__ = [
    "run_qa",
    "retranslate_likely_errors",
    "QACriticVerdict",
    "QAVerdict",
]
