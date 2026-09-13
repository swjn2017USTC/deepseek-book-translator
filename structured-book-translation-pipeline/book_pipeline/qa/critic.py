"""QA-4 LLM critic for flagged segments only (spec §13, plan §4).

Honesty contract:

- **flagged-only**: the critic receives exclusively the union of QA-1/2/3
  flags — never a full book re-read (prohibited by spec §13).
- **probe gate**: ``contextual.ladder.provider_probe`` with a 150s-timeout
  client (a slow-but-alive provider must not be falsely declared down; the
  probe is one cheap round-trip, so it fails fast rather than burning the
  slow 1200s/8-retry live budget).  Probe failure => every QA-4 row is
  ``{verdict: needs_human, reason: "provider_unavailable", confidence: 0}``,
  the stage status is ``provider_unavailable``, deterministic QA-1/2/3
  results are still emitted.  Never a fabricated verdict.
- **wire budget**: ``qa.wire_budget`` (default 20) caps total critic HTTP
  attempts (``client.wire_attempts``, counting client-internal retries); on
  exhaustion the remaining flags become ``needs_human`` /
  ``wire_budget_exhausted``.
- **judge only**: the model may never output a translation or edit; schema
  ``extra="forbid"`` plus an echo check (the verdict's ``segment_id`` must
  equal the flagged id) reject invented keys and leaked context ids.
- **usage log**: ``qa_usage.jsonl`` rows in the ``usage.jsonl`` shape spirit.

Role resolution: ``config["llm_roles"][qa.critic_role]`` falling back to
``provider.model`` (``resolve_model`` precedent: structure/terminology
resolvers).  Only flagged segments are sent; each call carries the flagged
segment plus bounded previous/next source context from the same chapter plus
the flag's expectation detail.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Sequence

from ..config import resolve_path
from ..contextual import ladder as ladder_module
from ..io_utils import append_jsonl
from ..llm_client import OpenAICompatibleClient, resolve_model
from ..models import Segment
from .schemas import QACriticVerdict
from .settings import QaSettings

PROBE_TIMEOUT_SECONDS = 150

SYSTEM_PROMPT = (
    "You are a translation QA critic for a Chinese translation of an English "
    "scholarly book. You judge ONLY — never output a translation, never edit, "
    "rewrite or re-translate any text, never propose replacement text.\n"
    "Rules:\n"
    "1. Verdict must be exactly one of: pass | acceptable_variant | "
    "likely_error | needs_human.\n"
    "2. likely_error ONLY when there is a concrete, nameable defect: a dropped "
    "token/marker/number, a mistranslated hard glossary term, an untranslated "
    "residual source passage, an empty or duplicated translation.\n"
    "3. acceptable_variant covers legitimate rewording, reordered digits, "
    "transliterated names and context-justified term choices.\n"
    "4. needs_human when you cannot judge confidently or the case needs a "
    "human glossary decision.\n"
    "5. confidence reflects your own certainty in this verdict (0.0-1.0), not "
    "the difficulty of the sentence.\n"
    "6. Respond with exactly one JSON object, no code fences: "
    '{"segment_id": "<the target segment id>", "verdict": "...", '
    '"reason": "<= 400 chars, concise", "confidence": 0.0}.\n'
    "7. Never reference or echo any previous_source/next_source segment id or "
    "text; segment_id must be the target segment id."
)


def _probe_ok(client: Any) -> bool:
    """Honest single round-trip probe (never swallows the outcome into a
    verdict — a failed probe is provider_unavailable, not a judgment)."""
    try:
        return ladder_module.provider_probe(client)
    except Exception:
        return False


def _critic_client(config: Dict[str, Any], role: str) -> OpenAICompatibleClient:
    provider = dict(config.get("provider") or {})
    provider["model"] = resolve_model(config, role)
    return OpenAICompatibleClient(provider)


def _probe_client(config: Dict[str, Any]) -> OpenAICompatibleClient:
    provider = dict(config.get("provider") or {})
    provider["timeout_seconds"] = PROBE_TIMEOUT_SECONDS
    provider["retries"] = 1
    return OpenAICompatibleClient(provider)


def _local_context(
    flagged: Dict[str, Any],
    segments: Sequence[Segment],
    chapter_of: Dict[str, str],
) -> Dict[str, List[str]]:
    """Bounded previous/next source windows for one flagged segment (same
    chapter only, at most 2 each side, each text capped at 400 chars)."""
    segment_id = flagged["segment_id"]
    chapter_id = chapter_of.get(segment_id)
    members = [
        segment for segment in segments
        if chapter_of.get(segment.id) == chapter_id
        and segment.kind != "image"
        and segment.source_text.strip()
    ]
    index = next(
        (position for position, segment in enumerate(members) if segment.id == segment_id),
        None,
    )
    if index is None:
        return {"previous_source": [], "next_source": []}
    trim = lambda text: text if len(text) <= 400 else text[:400] + "…"
    return {
        "previous_source": [
            trim(member.source_text) for member in members[max(0, index - 2):index]
        ],
        "next_source": [
            trim(member.source_text) for member in members[index + 1:index + 3]
        ],
    }


def _unavailable_rows(flags: Sequence[Dict[str, Any]], reason: str) -> List[Dict[str, Any]]:
    return [
        {
            "segment_id": flag["segment_id"],
            "verdict": "needs_human",
            "reason": reason,
            "confidence": 0.0,
        }
        for flag in flags
    ]


def run_critic(
    config: Dict[str, Any],
    flags: Sequence[Dict[str, Any]],
    *,
    segments: Sequence[Segment],
    translated: Dict[str, str],
    chapter_of: Dict[str, str],
    settings: QaSettings,
    client: Optional[Any] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the QA-4 critic over one item per *flagged segment*.

    ``flags`` holds per-segment groups ``{"segment_id": id, "flags": [flag
    rows...]}`` in deterministic (file) order — the union of QA-1/2/3 flags,
    never the full book (only-flags rule).  ``client`` is the caller-injected
    double for offline runs (tests/demo); when ``None`` the live path probes
    with a 150s client and then builds the role-model client from the config
    provider.  ``limit`` caps how many flagged segments are judged this run
    (token economy); the remainder stays flagged-but-unjudged and is reported
    as ``unjudged_flags``.
    """
    stage: Dict[str, Any] = {
        "status": "disabled",
        "verdicts": {"pass": 0, "acceptable_variant": 0, "likely_error": 0, "needs_human": 0},
    }
    verdict_rows: List[Dict[str, Any]] = []
    if not settings.enabled:
        stage["status"] = "disabled"
        stage["detail"] = "qa.enabled=false in config"
        return {"stage": stage, "verdicts": verdict_rows, "usage_rows": []}
    if not flags:
        stage["status"] = "complete"
        stage["detail"] = "no flagged segments"
        return {"stage": stage, "verdicts": verdict_rows, "usage_rows": []}

    ordered = list(flags)
    judged = ordered if limit is None else ordered[:limit]
    if limit is not None and len(ordered) > limit:
        stage["unjudged_flags"] = len(ordered) - limit

    if client is None:
        try:
            probe_client = _probe_client(config)
            probe_ok = _probe_ok(probe_client)
        except Exception:
            probe_ok = False
        if not probe_ok:
            stage["status"] = "provider_unavailable"
            stage["detail"] = "provider probe failed; deterministic stages still complete"
            verdict_rows = _unavailable_rows(ordered, "provider_unavailable")
            stage["verdicts"]["needs_human"] = len(verdict_rows)
            return {"stage": stage, "verdicts": verdict_rows, "usage_rows": []}
        client = _critic_client(config, settings.critic_role)
    else:
        if not _probe_ok(client):
            stage["status"] = "provider_unavailable"
            stage["detail"] = "provider probe failed; deterministic stages still complete"
            verdict_rows = _unavailable_rows(ordered, "provider_unavailable")
            stage["verdicts"]["needs_human"] = len(verdict_rows)
            return {"stage": stage, "verdicts": verdict_rows, "usage_rows": []}

    stage["status"] = "complete"
    model_name = getattr(client, "model", resolve_model(config, settings.critic_role))
    wire_budget = settings.wire_budget
    qa_model = model_name
    qa_batch = f"qa-4:{config['book_id']}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    output_dir = resolve_path(config, "output_dir")
    usage_path = output_dir / "qa_usage.jsonl"
    usage_rows: List[Dict[str, Any]] = []
    tally = {"pass": 0, "acceptable_variant": 0, "likely_error": 0, "needs_human": 0}
    exhausted = False
    for group in judged:
        if exhausted or getattr(client, "wire_attempts", 0) >= wire_budget:
            exhausted = True
            row = {
                "segment_id": group["segment_id"],
                "verdict": "needs_human",
                "reason": "wire_budget_exhausted",
                "confidence": 0.0,
            }
            verdict_rows.append(row)
            tally["needs_human"] += 1
            continue
        payload = {
            "target": _flagged_target(group, segments, translated),
            **_local_context(group, segments, chapter_of),
            "flag": {
                "checks": [
                    {
                        key: item[key]
                        for key in ("stage", "check", "severity", "expected", "actual", "detail")
                        if key in item
                    }
                    for item in group["flags"]
                ]
            },
        }
        usage: Dict[str, Any] = {}
        try:
            verdict, usage = client.complete_json(
                SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                QACriticVerdict,
            )
            if verdict.segment_id != group["segment_id"]:
                raise RuntimeError(
                    f"critic echoed segment id {verdict.segment_id!r} for flag on "
                    f"{group['segment_id']!r}; leak suspected"
                )
            row = {
                "segment_id": verdict.segment_id,
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "confidence": verdict.confidence,
            }
            verdict_rows.append(row)
            tally[row["verdict"]] += 1
        except RuntimeError as exc:
            row = {
                "segment_id": group["segment_id"],
                "verdict": "needs_human",
                "reason": f"critic_error: {str(exc)[:180]}",
                "confidence": 0.0,
            }
            verdict_rows.append(row)
            tally["needs_human"] += 1
        usage_rows.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "qa_batch": qa_batch,
            "qa_model": qa_model,
            "qa_segment_ids": [group["segment_id"]],
            "usage": dict(usage),
            "verdict_counts": dict(tally),
        })
    if exhausted:
        stage["wire_budget_exhausted"] = True
        stage["detail"] = (
            f"wire budget {wire_budget} exhausted; remaining flags marked "
            "needs_human/wire_budget_exhausted"
        )
    if usage_rows:
        append_jsonl(usage_path, usage_rows)
    stage["verdicts"] = tally
    return {"stage": stage, "verdicts": verdict_rows, "usage_rows": usage_rows}


def _flagged_target(
    flag: Dict[str, Any],
    segments: Sequence[Segment],
    translated: Dict[str, str],
) -> Dict[str, Any]:
    segment = next(
        (item for item in segments if item.id == flag["segment_id"]),
        None,
    )
    if segment is None:
        return {
            "id": flag["segment_id"],
            "kind": "paragraph",
            "source_text": "",
            "translated_text": "",
        }
    return {
        "id": segment.id,
        "kind": segment.kind,
        "source_text": segment.source_text,
        "translated_text": translated.get(segment.id, ""),
    }
