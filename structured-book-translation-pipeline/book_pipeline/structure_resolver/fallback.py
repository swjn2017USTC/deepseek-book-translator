from __future__ import annotations

"""Fail-closed fallback for provider outages (plan §5.4/§5.5 step 11, §8).

When any LLM step of the v2 structure resolver raises a provider-unavailable
error the orchestrator:

* records the sentinel status ``structure_llm_unavailable`` in
  ``structure/v2/status.json`` and in the returned result;
* keeps every artifact already flushed (meta line, partial decisions, rescue
  records, usage log);
* never claims the v2 structure was produced: the deterministic legacy
  pipeline remains the runnable path and v2 is reported as **blocked, not
  faked**.

This module holds the sentinel constant and the status writer used by the
orchestrator and by the CLI.
"""

from pathlib import Path
from typing import Any, Dict

from ..io_utils import write_json

STRUCTURE_LLM_UNAVAILABLE = "structure_llm_unavailable"
STRUCTURE_V2_OK = "structure_v2_ok"
NEEDS_HUMAN_REVIEW = "needs_human_review"


def write_status(
    structure_dir: Any,
    status: str,
    *,
    book_id: str = "",
    detail: str = "",
    repair_total: int = 0,
    decision_count: int = 0,
    rescue_count: int = 0,
    error: str = "",
) -> Dict[str, Any]:
    """Persist ``structure/v2/status.json`` and return the status record."""
    payload = {
        "schema_version": 1,
        "book_id": book_id,
        "status": status,
        "detail": detail,
        "error": error,
        "repair_total": repair_total,
        "decision_count": decision_count,
        "rescue_count": rescue_count,
    }
    write_json(_status_path(structure_dir), payload)
    return payload


def _status_path(structure_dir: Any) -> Path:
    return Path(structure_dir) / "v2" / "status.json"
