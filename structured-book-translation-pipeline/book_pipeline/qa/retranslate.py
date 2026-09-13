"""Selective retranslation of ``likely_error`` segments (spec §13, plan D3/D7).

This module is the **only writer** in the QA cascade.  ``qa`` runs are
read-only; only ``qa-retranslate`` appends revision rows to
``translations.jsonl``, and only for verdicts already persisted as
``likely_error`` in ``qa_report.json``.  It reuses the contextual v2
single-target seam read-only: ``_prepare`` -> context via ``_context`` ->
``_translate_one`` -> ``_commit_target`` (same imports ``ladder.py`` already
uses).  ``--rollback`` is out of scope (documented, not built).

Revision-row contract (scout §1.2/§8.3 + plan §5.4/§6): same ``segment_id`` +
same ``source_hash`` as the segment, ``status: completed``, current v2
metadata (``chapter_id``/``brief_hash``/``prompt_version``/
``glossary_dependencies`` from the run's own prepared state — so
``is_translation_current_v2`` stays true and render still passes), plus a new
``qa`` field tying the revision to its flag/verdict.  The original row is
kept in the file; last-write-wins by append order supersedes it.  Without a
live provider the command blocks with an honest error — a revision is never
fabricated.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

from ..config import resolve_path
from ..contextual.orchestrator import _commit_target, _prepare, _translate_one
from ..io_utils import write_json
from ..llm_client import OpenAICompatibleClient
from .critic import _probe_client
from .critic import _probe_ok as _probe_ok

__all__ = ["retranslate_likely_errors", "retranslate_segment_ids"]


def _probe_or_raise(client: Optional[Any], config: Dict[str, Any]) -> Any:
    """Return a live-capable client or raise an honest block message.

    With an injected offline client the probe runs against it; with no client
    the live path probes with the 150s probe client, then builds the real
    slow-policy client (1200s / 8 retries from the config provider).
    """
    if client is not None:
        if not _probe_ok(client):
            raise RuntimeError(
                "qa-retranslate blocked: provider probe failed; no revision was "
                "written (never fabricate)"
            )
        return client
    try:
        probe_client = _probe_client(config)
        if not _probe_ok(probe_client):
            raise RuntimeError(
                "qa-retranslate blocked: provider probe failed; no revision was "
                "written (never fabricate)"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"qa-retranslate blocked: provider unavailable ({exc}); no revision "
            "was written (never fabricate)"
        ) from exc
    return OpenAICompatibleClient(dict(config.get("provider") or {}))


def _load_report(config: Dict[str, Any]) -> Dict[str, Any]:
    path = resolve_path(config, "output_dir") / "qa_report.json"
    if not path.is_file():
        raise FileNotFoundError(
            "qa-retranslate requires a qa_report.json from a prior `qa` run; "
            "run `qa --config <config>` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _select_verdicts(
    report: Dict[str, Any],
    *,
    segment_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Likely-error verdicts in report order, filtered/bounded."""
    verdicts = [
        verdict for verdict in report.get("verdicts") or []
        if verdict.get("verdict") == "likely_error"
    ]
    if segment_id is not None:
        verdicts = [verdict for verdict in verdicts if verdict["segment_id"] == segment_id]
        if not verdicts:
            raise ValueError(
                f"segment {segment_id!r} has no persisted likely_error verdict; "
                "qa-retranslate only rewrites likely_error segments"
            )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        verdicts = verdicts[:limit]
    return verdicts


def _segment_flags(report: Dict[str, Any], segment_id: str) -> List[Dict[str, Any]]:
    return [
        flag for flag in report.get("flags") or []
        if flag["segment_id"] == segment_id
    ]


def retranslate_segment_ids(
    config: Dict[str, Any],
    client: Optional[Any] = None,
    *,
    segment_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Retranslate persisted ``likely_error`` segments (CLI ``qa-retranslate``).

    Only this command writes revision rows.  Blocks (raises) without a live
    provider; returns a summary dict with ``retranslated`` counts and the
    report path otherwise.
    """
    report_path = resolve_path(config, "output_dir") / "qa_report.json"
    report = _load_report(config)
    verdicts = _select_verdicts(report, segment_id=segment_id, limit=limit)
    if not verdicts:
        return {
            "schema_version": 1,
            "book_id": config["book_id"],
            "retranslated": 0,
            "segment_ids": [],
            "reason": "no likely_error verdicts in qa_report.json",
            "report_file": str(report_path),
        }

    live = _probe_or_raise(client, config)
    prepared = _prepare(config)
    prepared["client"] = live
    prepared["delay"] = 0.0  # revisions are few and explicit; no batch pacing

    # Original rows are captured before any append so the trail records the
    # true pre-revision state (last-write-wins would otherwise shadow it).
    originals = {
        segment_id: dict(row)
        for segment_id, row in prepared["completed"].items()
        if row.get("status") == "completed"
    }

    from ..translate import _completed, _segments

    output_dir = resolve_path(config, "output_dir")
    segments = _segments(output_dir / "cleaned_segments.jsonl")
    by_id = {segment.id: segment for segment in segments}
    completed = _completed(output_dir / "translations.jsonl")
    translations_path = output_dir / "translations.jsonl"
    now_stamp = datetime.now(timezone.utc).isoformat()

    revised: List[Dict[str, Any]] = []
    for verdict in verdicts:
        target_id = verdict["segment_id"]
        segment = by_id.get(target_id)
        original = originals.get(target_id) or completed.get(target_id)
        if segment is None:
            raise RuntimeError(
                f"Cannot retranslate {target_id}: unknown cleaned segment; rerun clean"
            )
        if original is None or original.get("status") != "completed":
            raise RuntimeError(
                f"Cannot retranslate {target_id}: no completed translation row to revise"
            )
        outcome = _translate_one(segment, prepared, batch_index=1)
        record = outcome["record"]
        flag_rows = _segment_flags(report, target_id)
        record["qa"] = {
            "revision_of": str(original.get("timestamp") or ""),
            "flag": (
                {"stage": flag_rows[0]["stage"], "check": flag_rows[0]["check"]}
                if flag_rows else {"stage": "qa-4", "check": "critic"}
            ),
            "critic_verdict": verdict.get("verdict", "likely_error"),
            "retranslated_at": str(record.get("timestamp") or now_stamp),
        }
        _commit_target(prepared, segment, outcome)
        revised.append({
            "segment_id": target_id,
            "original_timestamp": str(original.get("timestamp") or ""),
            "revision_timestamp": str(record.get("timestamp") or ""),
            "verdict": verdict.get("verdict", "likely_error"),
            "reason": str(verdict.get("reason") or ""),
            "flag": record["qa"]["flag"],
        })

    # Persist the revision trail into qa_report.json.
    report["revisions"] = list(report.get("revisions") or []) + revised
    report["last_retranslated_at"] = now_stamp
    write_json(report_path, report)

    # Refresh translation_status.json qa_revised_segments.
    status_path = output_dir / "translation_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["qa_revised_segments"] = len(report["revisions"])
        status["qa_last_retranslated_at"] = now_stamp
        write_json(status_path, status)

    return {
        "schema_version": 1,
        "book_id": config["book_id"],
        "retranslated": len(revised),
        "segment_ids": [item["segment_id"] for item in revised],
        "report_file": str(report_path),
        "translation_file": str(translations_path),
        "revisions": revised,
    }


retranslate_likely_errors = retranslate_segment_ids
