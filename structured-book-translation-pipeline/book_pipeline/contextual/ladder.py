"""Smoke ladder for contextual translation v2 (spec §12.7 / D6).

``provider_probe`` is a single cheap complete round-trip that gates every live
run: when the provider is DOWN the ladder writes one honest ``BLOCKED`` record
(``{"blocked": true, "reason": ...}``) and translates nothing — zero fabricated
translations.  ``run_ladder`` records one row per translated segment with
success/retries/latency/tokens/best-effort glossary violations/sample.

``glossary_violations`` is a deterministic placeholder (full terminology QA is
P07): a concept with constraint ``hard`` whose surface occurs in the source but
whose canonical translation is absent from the produced target text.  Reported,
never enforced.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .deps import concept_sense
from .orchestrator import _commit_target, _prepare, _translate_one


LADDER_STEPS = {"10": 10, "100": 100, "500": 500, "all": None}


def provider_probe(client: Any) -> bool:
    """Single cheap complete round-trip; True iff a non-empty reply arrives."""
    try:
        content, _ = client.complete("probe", '{"probe": true, "reply": "ok"}')
        return bool(content)
    except Exception:
        return False


def _violations(
    segment,
    target_text: str,
    dependencies: Dict[str, int],
    concepts_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    violations: List[str] = []
    for concept_id in sorted(dependencies):
        concept = concepts_by_id.get(concept_id)
        if concept is None:
            continue
        sense = concept_sense(concept) or {}
        if sense.get("constraint") != "hard":
            continue
        translation = str(sense.get("translation") or "")
        if translation and translation not in target_text:
            violations.append(concept_id)
    return violations


def run_ladder(
    config: Dict[str, Any],
    client: Any,
    level: str,
) -> Dict[str, Any]:
    """Run ``level`` ladder steps (10/100/500/all) through the contextual v2
    path, writing ``translation_ladder.jsonl`` rows next to the shared
    ``translations.jsonl``.  Probe failure -> a BLOCKED record, nothing else.
    """
    if level not in LADDER_STEPS:
        raise ValueError(f"Unknown ladder level {level!r}; choose from {sorted(LADDER_STEPS)}")
    requested = LADDER_STEPS[level]
    prepared = _prepare(config)
    prepared["client"] = client
    # P07-pretest fix: only translatable segments are pending — non-translatable
    # placeholders (e.g. empty image/table rows) have no brief and must never be
    # fed to the translator (ladder previously failed 10/10 on them).
    pending = [segment for segment in prepared["segments"]
               if segment.translatable and not prepared["current"](segment)]
    step_count = len(pending) if requested is None else min(requested, len(pending))
    output_dir = prepared["output_dir"]
    ladder_path = output_dir / "translation_ladder.jsonl"
    blocked_reason = _probe_reason(client)
    if blocked_reason is not None:
        row = {"blocked": True, "reason": f"provider probe failed: {blocked_reason}"}
        _append_ladder_row(ladder_path, row)
        return {
            "schema_version": 1,
            "book_id": config["book_id"],
            "pipeline": "contextual_v2",
            "ladder_level": level,
            "requested_steps": requested,
            "executed_steps": 0,
            "succeeded": 0,
            "failed": 0,
            "blocked": True,
            "reason": row["reason"],
            "ladder_file": str(ladder_path),
            "usage_file": str(prepared["usage_path"]),
            "translation_file": str(prepared["progress_path"]),
        }
    concepts_by_id = {concept.get("concept_id"): concept for concept in prepared["concepts"] if concept.get("concept_id")}
    executed = 0
    succeeded = 0
    failed = 0
    for step, segment in enumerate(pending[:step_count] if step_count else [], 1):
        started = time.monotonic()
        try:
            outcome = _translate_one(segment, prepared, batch_index=step)
            _commit_target(prepared, segment, outcome)
            executed += 1
            succeeded += 1
            latency = round(time.monotonic() - started, 3)
            usage = outcome["usage"] or {}
            row = {
                "step": step,
                "success": True,
                "retries": max(0, outcome["attempts"] - 1),
                "latency_s": latency,
                "tokens": {
                    "prompt": usage.get("prompt_tokens"),
                    "completion": usage.get("completion_tokens"),
                },
                "glossary_violations": _violations(
                    segment,
                    outcome["record"]["translated_text"],
                    outcome["record"]["metadata"]["glossary_dependencies"],
                    concepts_by_id,
                ),
                "sample": {
                    "source_text": segment.source_text,
                    "target_text": outcome["record"]["translated_text"],
                },
            }
            _append_ladder_row(ladder_path, row)
        except RuntimeError as exc:
            executed += 1
            failed += 1
            row = {
                "step": step,
                "success": False,
                "retries": 10,
                "latency_s": round(time.monotonic() - started, 3),
                "tokens": {"prompt": None, "completion": None},
                "glossary_violations": [],
                "sample": {"source_text": segment.source_text, "target_text": ""},
                "error": str(exc),
            }
            _append_ladder_row(ladder_path, row)
        if prepared["delay"]:
            time.sleep(prepared["delay"])
    return {
        "schema_version": 1,
        "book_id": config["book_id"],
        "pipeline": "contextual_v2",
        "ladder_level": level,
        "requested_steps": requested,
        "executed_steps": executed,
        "succeeded": succeeded,
        "failed": failed,
        "blocked": False,
        "ladder_file": str(ladder_path),
        "usage_file": str(prepared["usage_path"]),
        "translation_file": str(prepared["progress_path"]),
    }


def _probe_reason(client: Any) -> Optional[str]:
    try:
        ok = provider_probe(client)
    except Exception as exc:  # constructor-level failures are reported too
        return str(exc) or exc.__class__.__name__
    return None if ok else "probe returned no usable reply"


def _append_ladder_row(path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
