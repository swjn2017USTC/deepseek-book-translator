"""QA cascade orchestrator (spec §13, plan §6).

Fixed order QA-1 -> QA-2 -> QA-3 -> QA-4.  A full ``run_qa`` never mutates a
translation: it reads ``cleaned_segments.jsonl`` + ``translations.jsonl``
through the same gates/loaders translation uses and writes only QA artifacts
(``qa_report.json``, ``qa_audit.jsonl``, an extended ``translation_status.json``).
``qa-retranslate`` is the single writer of revision rows.

``--only`` semantics: a partial run executes exactly the requested stage;
``qa-4`` alone reads the flags persisted by a prior full run
(``output_dir/qa_report.json``) and rejects with a clear error when there is
nothing persisted.  Partial runs merge into the previous report so a
``--only qa-3`` follow-up never destroys the full run's flags/verdicts.

Severity model (amendment A1): every flag carries ``severity: flag|high``;
no QA stage blocks rendering or translation in this phase — the report
distinguishes ``flagged`` (all rows) from ``would_block`` (always 0).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Sequence

from ..clean import require_structure_gate
from ..config import resolve_path
from ..contextual.briefs import nodes_index
from ..contextual.invalidation import brief_key_for_segment
from ..io_utils import append_jsonl, write_json
from ..models import Segment
from ..provenance import require_fresh_cleaning
from ..translate import _completed, _segments, translation_status
from . import anomaly as anomaly_module
from . import critic as critic_module
from . import deterministic as deterministic_module
from . import terminology as terminology_module
from .settings import qa_settings

VALID_STAGES = {"qa-1", "qa-2", "qa-3", "qa-4"}


def _load(config: Dict[str, Any]) -> Dict[str, Any]:
    require_structure_gate(config)
    require_fresh_cleaning(config)
    output_dir = resolve_path(config, "output_dir")
    segments = _segments(output_dir / "cleaned_segments.jsonl")
    completed = _completed(output_dir / "translations.jsonl")
    deterministic_module.load_gate(segments, completed)
    audited = deterministic_module.audit_rows(segments, completed)
    translatable = [segment for segment in segments if segment.translatable]
    return {
        "segments": segments,
        "completed": completed,
        "audited": audited,
        "untranslated": len(translatable) - len(audited),
    }


def _chapter_map(config: Dict[str, Any], segments: Sequence[Segment]) -> Dict[str, str]:
    structure = json.loads(resolve_path(config, "structure_json").read_text(encoding="utf-8"))
    nodes_by_id = nodes_index(structure)
    chapter_of: Dict[str, str] = {}
    for segment in segments:
        if segment.translatable or segment.source_text.strip():
            chapter_of[segment.id] = brief_key_for_segment(segment, nodes_by_id)
    return chapter_of


def _previous_report(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = resolve_path(config, "output_dir") / "qa_report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_rows(
    config: Dict[str, Any],
    segments: Sequence[Segment],
    flags: Sequence[Dict[str, Any]],
    verdicts: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """qa_audit.jsonl rows: one per flag plus one per QA-4 verdict (append-only
    log, plan §5.2)."""
    source_hash = {segment.id: segment.source_hash for segment in segments}
    now = datetime.now(timezone.utc).isoformat()
    rows: List[Dict[str, Any]] = []
    for flag in flags:
        rows.append({
            "segment_id": flag["segment_id"],
            "source_hash": source_hash.get(flag["segment_id"], ""),
            "stage": flag["stage"],
            "check": flag["check"],
            "verdict": "flag",
            "detail": str(flag.get("detail") or ""),
            "severity": flag.get("severity", "flag"),
            "timestamp": now,
        })
    for verdict in verdicts:
        rows.append({
            "segment_id": verdict["segment_id"],
            "source_hash": source_hash.get(verdict["segment_id"], ""),
            "stage": "qa-4",
            "check": "critic",
            "verdict": verdict["verdict"],
            "detail": str(verdict.get("reason") or ""),
            "confidence": float(verdict.get("confidence") or 0.0),
            "timestamp": now,
        })
    return rows


def _group_flags(flags: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One group per flagged segment (deterministic first-flag order) — the
    QA-4 input: flagged segments only, never the full book."""
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for flag in flags:
        by_id.setdefault(flag["segment_id"], []).append(flag)
    return [
        {"segment_id": segment_id, "flags": member_flags}
        for segment_id, member_flags in by_id.items()
    ]


def run_qa(
    config: Dict[str, Any],
    client: Optional[Any] = None,
    *,
    only: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the QA cascade (or the single ``only`` stage) and write artifacts.

    ``client`` injects an offline critic double; ``limit`` caps live critic
    calls / judged flagged segments.  Never mutates translations.
    """
    if only is not None and only not in VALID_STAGES:
        raise ValueError(f"only must be one of {sorted(VALID_STAGES)}, got {only!r}")
    loaded = _load(config)
    segments = loaded["segments"]
    audited = loaded["audited"]
    previous = _previous_report(config)
    created_at = datetime.now(timezone.utc).isoformat()
    output_dir = resolve_path(config, "output_dir")

    all_flags: List[Dict[str, Any]] = []
    verdicts: List[Dict[str, Any]] = []
    stages: Dict[str, Dict[str, Any]] = {}
    if previous:
        stages = {key: dict(value) for key, value in (previous.get("stages") or {}).items()}

    # ------------------------------------------------------------- QA-1
    if only is None or only == "qa-1":
        qa1_flags: List[Dict[str, Any]] = []
        for segment, row in audited:
            qa1_flags.extend(
                deterministic_module.check_segment(
                    segment, str(row.get("translated_text") or "")
                )
            )
        stages["qa-1"] = {
            "status": "complete",
            "flags": len(qa1_flags),
            "checks": deterministic_module.count_class_totals(qa1_flags),
        }
        all_flags.extend(qa1_flags)

    # ------------------------------------------------------------- QA-2
    if only is None or only == "qa-2":
        terminology_result = terminology_module.run_terminology_audit(audited, config)
        qa2_flags = terminology_result["flags"]
        metrics = terminology_result["metrics"]
        stages["qa-2"] = {
            "status": "complete",
            "flags": len(qa2_flags),
            "term_success_rate": metrics["term_success_rate"],
            "applicable": metrics["applicable"],
            "matched": metrics["matched"],
            "hard_violations": metrics["hard_violations"],
            "consistency_flags": metrics["consistency_flags"],
            "suspicious_contexts": metrics["suspicious_contexts"],
            "glossary_source": metrics["glossary_source"],
        }
        all_flags.extend(qa2_flags)

    # ------------------------------------------------------------- QA-3
    if only is None or only == "qa-3":
        settings = qa_settings(config)
        qa3_flags: List[Dict[str, Any]] = []
        for segment, row in audited:
            qa3_flags.extend(
                anomaly_module.anomalies_for(
                    segment, str(row.get("translated_text") or ""), settings
                )
            )
        qa3_flags.extend(anomaly_module.duplicate_anomalies(audited))
        qa1_checks = {
            flag["segment_id"]: flag["check"]
            for flag in all_flags
            if flag["stage"] == "qa-1"
        }
        if previous:  # a prior full run already surfaced footnote drops at QA-1
            for flag in (previous.get("flags") or []):
                if flag.get("stage") == "qa-1":
                    qa1_checks.setdefault(flag["segment_id"], flag["check"])
        qa3_flags.extend(anomaly_module.footnote_missing_flags(audited, qa1_checks))
        anomaly_totals: Dict[str, int] = {}
        for flag in qa3_flags:
            anomaly_totals[flag["check"]] = anomaly_totals.get(flag["check"], 0) + 1
        stages["qa-3"] = {
            "status": "complete",
            "flags": len(qa3_flags),
            "anomalies": anomaly_totals,
        }
        all_flags.extend(qa3_flags)

    # ------------------------------------------------------------- QA-4
    qa4_outcome: Optional[Dict[str, Any]] = None
    if only is None or only == "qa-4":
        if only == "qa-4":
            # QA-4 alone: read the deterministic flags persisted by a prior run.
            persisted_flags = (previous or {}).get("flags") or []
            if not persisted_flags:
                raise ValueError(
                    "qa-4 requires a prior full qa run (qa_report.json with flags); "
                    "run `qa --config <config>` first"
                )
            qa4_input = _group_flags(persisted_flags)
        else:
            qa4_input = _group_flags(all_flags)
        settings = qa_settings(config)
        translated = {
            segment.id: str(row.get("translated_text") or "")
            for segment, row in audited
        }
        chapter_of = _chapter_map(config, segments)
        qa4_outcome = critic_module.run_critic(
            config,
            qa4_input,
            segments=segments,
            translated=translated,
            chapter_of=chapter_of,
            settings=settings,
            client=client,
            limit=limit,
        )
        stages["qa-4"] = qa4_outcome["stage"]
        verdicts = qa4_outcome["verdicts"]

    # Full run recomputes the flag list; partial runs merge into the previous
    # report (per-stage replacement keeps the full run's other stages).
    if only is None:
        previous_revisions = (previous or {}).get("revisions") or []
        revisions: List[Dict[str, Any]] = list(previous_revisions)
        flags_for_report = all_flags
    else:
        previous_flags = [
            flag for flag in ((previous or {}).get("flags") or [])
            if flag["stage"] != only
        ]
        flags_for_report = previous_flags + all_flags
        revisions = list((previous or {}).get("revisions") or [])
        if only != "qa-4":
            # Deterministic-only partial run: verdicts from the prior run are
            # unchanged and carried forward in the report.
            verdicts = list((previous or {}).get("verdicts") or [])

    flagged_ids = sorted({flag["segment_id"] for flag in flags_for_report})
    likely_error_ids = [
        verdict["segment_id"]
        for verdict in verdicts
        if verdict.get("verdict") == "likely_error"
    ]
    report: Dict[str, Any] = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "created_at": created_at,
        "audited_segments": len(audited),
        "untranslated_segments": loaded["untranslated"],
        "stages": stages,
        "flags": flags_for_report,
        "verdicts": verdicts,
        "revisions": revisions,
        "summary": {
            "flagged_segments": len(flagged_ids),
            "would_block": 0,
            "note": "no QA stage blocks rendering or translation in this phase; "
                    "flags concentrate human/QA-4 attention (amendment A1)",
        },
    }
    if only is not None:
        report["partial"] = True
        report["stage_requested"] = only
        if only == "qa-4":
            report["unjudged_flags"] = stages["qa-4"].get("unjudged_flags", 0)

    write_json(output_dir / "qa_report.json", report)
    audit = _audit_rows(config, segments, all_flags, verdicts)
    if audit:
        append_jsonl(output_dir / "qa_audit.jsonl", audit)

    # translation_status.json extension (scout §1.5): non-breaking keys.
    try:
        status = translation_status(config)
    except Exception:
        status = {"book_id": config["book_id"]}
    status.update({
        "qa_flagged_segments": len(flagged_ids),
        "qa_likely_error": len(likely_error_ids),
        "qa_revised_segments": len(revisions),
        "qa_last_run": created_at,
    })
    write_json(output_dir / "translation_status.json", status)

    report["artifacts"] = {
        "qa_report": str(output_dir / "qa_report.json"),
        "qa_audit": str(output_dir / "qa_audit.jsonl"),
        "qa_usage": str(output_dir / "qa_usage.jsonl"),
        "translation_status": str(output_dir / "translation_status.json"),
    }
    if qa4_outcome is not None and qa4_outcome["usage_rows"]:
        report["qa_4_wire_attempts"] = getattr(client, "wire_attempts", None)
    return report
