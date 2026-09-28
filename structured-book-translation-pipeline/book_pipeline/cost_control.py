from __future__ import annotations

"""Cost telemetry for long-running DeepSeek book translation.

Prices are defaults for DeepSeek Flash in RMB per 1M tokens and are intentionally
configurable through provider.pricing_rmb_per_million so future price changes do
not require code changes.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_FLASH_PRICING_RMB_PER_MILLION = {
    "off_peak": {"cache_hit_input": 0.02, "cache_miss_input": 1.0, "output": 4.0},
    "peak": {"cache_hit_input": 0.04, "cache_miss_input": 2.0, "output": 8.0},
}


def pricing(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    raw = (config.get("provider") or {}).get("pricing_rmb_per_million") or {}
    result = json.loads(json.dumps(DEFAULT_FLASH_PRICING_RMB_PER_MILLION))
    for period in ("off_peak", "peak"):
        result[period].update({key: float(value) for key, value in (raw.get(period) or {}).items()})
    return result


def is_peak_beijing(now: Optional[datetime] = None) -> bool:
    """DeepSeek Flash weekday peak windows in Beijing time.

    A timezone-aware datetime is preferred. A naive datetime is treated as
    Beijing local time so CLI/Windows users get intuitive behaviour.
    """
    if now is None:
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
        except Exception:
            now = datetime.now()
    elif now.tzinfo is not None:
        try:
            from zoneinfo import ZoneInfo
            now = now.astimezone(ZoneInfo("Asia/Shanghai"))
        except Exception:
            pass
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 <= minutes < 12 * 60) or (14 * 60 <= minutes < 18 * 60)


def _usage_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def usage_totals(path: Path) -> Dict[str, Any]:
    prompt = completion = hit = miss = reasoning = requests = 0
    parse_attempts = segments = 0
    for row in _usage_rows(path):
        usage = row.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        hit += int(usage.get("prompt_cache_hit_tokens") or 0)
        if "prompt_cache_miss_tokens" in usage:
            miss += int(usage.get("prompt_cache_miss_tokens") or 0)
        else:
            miss += max(0, int(usage.get("prompt_tokens") or 0) - int(usage.get("prompt_cache_hit_tokens") or 0))
        details = usage.get("completion_tokens_details") or {}
        reasoning += int(details.get("reasoning_tokens") or 0)
        requests += int(usage.get("request_count") or row.get("api_completion_requests") or 1)
        parse_attempts += int(row.get("parse_attempts") or 1)
        segments += len(row.get("segment_ids") or [])
    accounted = hit + miss
    return {
        "segments": segments,
        "api_completion_requests": requests,
        "parse_attempts": parse_attempts,
        "prompt_tokens": prompt,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "visible_completion_tokens": max(0, completion - reasoning),
        "cache_hit_ratio": round(hit / accounted, 6) if accounted else None,
        "reasoning_ratio": round(reasoning / completion, 6) if completion else None,
        "retry_rate": round(max(0, requests - segments) / requests, 6) if requests else 0.0,
    }


def estimate_rmb(totals: Dict[str, Any], rates: Dict[str, float]) -> float:
    return round(
        (
            int(totals.get("prompt_cache_hit_tokens") or 0) * rates["cache_hit_input"]
            + int(totals.get("prompt_cache_miss_tokens") or 0) * rates["cache_miss_input"]
            + int(totals.get("completion_tokens") or 0) * rates["output"]
        ) / 1_000_000,
        6,
    )


def cost_report(config: Dict[str, Any], usage_path: Path, remaining_segments: int) -> Dict[str, Any]:
    totals = usage_totals(usage_path)
    rates = pricing(config)
    current_period = "peak" if is_peak_beijing() else "off_peak"
    completed = int(totals["segments"])
    current_cost = estimate_rmb(totals, rates[current_period])
    average = (current_cost / completed) if completed else None
    projected_remaining = round(average * remaining_segments, 4) if average is not None else None
    projected_total = round(current_cost + projected_remaining, 4) if projected_remaining is not None else None
    warnings = []
    if totals["reasoning_ratio"] and totals["reasoning_ratio"] > 0.05:
        warnings.append("reasoning_tokens_detected")
    if totals["retry_rate"] > 0.05:
        warnings.append("high_retry_rate")
    if totals["cache_hit_ratio"] is not None and totals["cache_hit_ratio"] < 0.25:
        warnings.append("low_cache_hit_ratio")
    return {
        "pricing_period_beijing": current_period,
        "rates_rmb_per_million": rates[current_period],
        "estimated_cost_rmb_so_far": current_cost,
        "estimated_cost_rmb_if_off_peak": estimate_rmb(totals, rates["off_peak"]),
        "estimated_cost_rmb_if_peak": estimate_rmb(totals, rates["peak"]),
        "average_cost_rmb_per_completed_segment": round(average, 6) if average is not None else None,
        "projected_remaining_cost_rmb_at_current_average": projected_remaining,
        "projected_total_cost_rmb_at_current_average": projected_total,
        "usage": totals,
        "warnings": warnings,
    }
