from __future__ import annotations

"""Fail-closed fallback for provider outages (plan §4/§5, D7).

When any LLM step of the terminology resolver v2 fails (provider down, wire
budget exhausted, deterministic rejection of hallucinated output) the
orchestrator:

* records the sentinel status ``terminology_llm_unavailable`` in
  ``<work_dir>/status.json`` and in the returned result;
* keeps every artifact already flushed (candidates, partial termhood
  decisions, usage log, progress manifest);
* never emits a partial ``concept_glossary.json`` as approved: the run is
  reported as **blocked, not faked**, and offline deterministic sampling
  remains fully available.

Offline runs and successful full runs record ``terminology_v2_ok``; a run that
completed but produced gray-zone concepts for the human gate records
``needs_human_review`` (§11.6 template written, glossary flagged not yet
human-approved).
"""

from pathlib import Path
from typing import Any, Dict, Optional

from ..io_utils import write_json

TERMINOLOGY_LLM_UNAVAILABLE = "terminology_llm_unavailable"
TERMINOLOGY_V2_OK = "terminology_v2_ok"
NEEDS_HUMAN_REVIEW = "needs_human_review"


def write_term_status(
    work_dir: Any,
    status: str,
    *,
    book_id: str = "",
    detail: str = "",
    counts: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> Dict[str, Any]:
    """Persist ``<work_dir>/status.json`` and return the status record."""
    payload = {
        "schema_version": 1,
        "book_id": book_id,
        "status": status,
        "detail": detail,
        "counts": counts or {},
        "error": error,
    }
    write_json(_status_path(work_dir), payload)
    return payload


def _status_path(work_dir: Any) -> Path:
    return Path(work_dir) / "status.json"
