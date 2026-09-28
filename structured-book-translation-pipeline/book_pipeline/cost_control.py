from __future__ import annotations

"""Cost telemetry for long-running DeepSeek book translation.

Built-in rates cover DeepSeek Flash only. Other/custom models must provide a
complete provider.pricing_rmb_per_million override before the code will emit a
currency estimate; this fails closed instead of silently applying Flash prices.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


FLASH_MODEL_ALIASES = {
    "deepseek-flash",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
}
DEFAULT_FLASH_PRICING_EFFECTIVE_DATE = "2026-09-10"
DEFAULT_FLASH_PRICING_RMB_PER_MILLION = {
    "off_peak": {"cache_hit_input": 0.02, "cache_miss_input": 1.0, "output": 4.0},
    "peak": {"cache_hit_input": 0.04, "cache_miss_input": 2.0, "output": 8.0},
}
_REQUIRED_RATE_KEYS = {"cache_hit_input", "cache_miss_input", "output"}


def _provider(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("provider") or {})


def effective_model(config: Dict[str, Any]) -> str:
    provider = _provider(config)
    model_env = str(provider.get("model_env") or "DEEPSEEK_MODEL")
    return str(os.environ.get(model_env) or provider.get("model") or "deepseek-flash")


def _complete_explicit_pricing(config: Dict[str, Any]) -> bool:
    raw = _provider(config).get("pricing_rmb_per_million") or {}
    return all(
        isinstance(raw.get(period), dict)
        and _REQUIRED_RATE_KEYS.issubset(raw[period])
        for period in ("off_peak", "peak")
    )


def pricing(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    raw = _provider(config).get("pricing_rmb_per_million") or {}
    result = json.loads(json.dumps(DEFAULT_FLASH_PRICING_RMB_PER_MILLION))
    for period in ("off_peak", "peak"):
        result[period].update({
            key: float(value)
            for key, value in (raw.get(period) or {}).items()
            if key in _REQUIRED_RATE_KEYS
        })
    return result


def is_peak_beijing(now: Optional[datetime] = None) -> bool:
    """DeepSeek Flash weekday peak windows in fixed UTC+8 Beijing time."""
    beijing = timezone(timedelta(hours=8))
    if now is None:
        now = datetime.now(beijing)
    elif now.tzinfo is not None:
        now = now.astimezone(beijing)
    # Naive datetimes supplied by tests/callers are interpreted as Beijing.
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
    parse_attempts = usage_records = segment_references = failed_records = 0
    unique_segment_ids = set()
    completed_segment_ids = set()
    failed_segment_ids = set()
    models = set()

    for row in _usage_rows(path):
        usage = row.get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        hit += int(usage.get("prompt_cache_hit_tokens") or 0)
        if "prompt_cache_miss_tokens" in usage:
            miss += int(usage.get("prompt_cache_miss_tokens") or 0)
        else:
            miss += max(
                0,
                int(usage.get("prompt_tokens") or 0)
                - int(usage.get("prompt_cache_hit_tokens") or 0),
            )
        details = usage.get("completion_tokens_details") or {}
        reasoning += int(details.get("reasoning_tokens") or 0)

        has_billable_usage = any(
            int(usage.get(key) or 0) > 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "request_count")
        )
        request_count = int(
            usage.get("request_count")
            or row.get("api_completion_requests")
            or (1 if has_billable_usage else 0)
        )
        requests += request_count
        if request_count:
            usage_records += 1
        parse_attempts += int(row.get("parse_attempts") or (1 if request_count else 0))

        segment_ids = [str(value) for value in (row.get("segment_ids") or [])]
        segment_references += len(segment_ids)
        unique_segment_ids.update(segment_ids)
        if row.get("status") == "failed_validation":
            failed_records += 1
            failed_segment_ids.update(segment_ids)
        else:
            completed_segment_ids.update(segment_ids)
        if row.get("model"):
            models.add(str(row["model"]))

    accounted = hit + miss
    return {
        "segments": segment_references,
        "unique_segment_ids": len(unique_segment_ids),
        "completed_segment_ids": len(completed_segment_ids),
        "failed_segment_ids": len(failed_segment_ids),
        "usage_records": usage_records,
        "failed_validation_records": failed_records,
        "models": sorted(models),
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
        "retry_rate": round(max(0, requests - usage_records) / requests, 6) if requests else 0.0,
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


def cost_report(
    config: Dict[str, Any],
    usage_path: Path,
    remaining_segments: int,
    *,
    completed_segments: Optional[int] = None,
) -> Dict[str, Any]:
    totals = usage_totals(usage_path)
    model = effective_model(config)
    provider = _provider(config)
    explicit_pricing = _complete_explicit_pricing(config)
    observed_models = set(totals["models"])
    observed_models_supported = not observed_models or observed_models.issubset(FLASH_MODEL_ALIASES)
    pricing_supported = explicit_pricing or (
        model in FLASH_MODEL_ALIASES and observed_models_supported
    )
    current_period = "peak" if is_peak_beijing() else "off_peak"

    warnings = []
    if totals["reasoning_ratio"] and totals["reasoning_ratio"] > 0.05:
        warnings.append("reasoning_tokens_detected")
    if totals["retry_rate"] > 0.05:
        warnings.append("high_retry_rate")
    if totals["cache_hit_ratio"] is not None and totals["cache_hit_ratio"] < 0.25:
        warnings.append("low_cache_hit_ratio")

    denominator = (
        int(completed_segments)
        if completed_segments is not None
        else int(totals["completed_segment_ids"])
    )

    if not pricing_supported:
        warnings.append("unknown_or_mixed_model_pricing")
        return {
            "model": model,
            "pricing_supported": False,
            "pricing_source": None,
            "pricing_period_beijing": current_period,
            "rates_rmb_per_million": None,
            "estimated_cost_rmb_so_far": None,
            "estimated_cost_rmb_if_off_peak": None,
            "estimated_cost_rmb_if_peak": None,
            "average_cost_rmb_per_completed_segment": None,
            "projected_remaining_cost_rmb_at_current_average": None,
            "projected_total_cost_rmb_at_current_average": None,
            "usage": totals,
            "warnings": warnings,
        }

    rates = pricing(config)
    pricing_source = (
        "config_override"
        if provider.get("pricing_rmb_per_million")
        else f"builtin_deepseek_flash_effective_{DEFAULT_FLASH_PRICING_EFFECTIVE_DATE}"
    )
    current_cost = estimate_rmb(totals, rates[current_period])
    average = (current_cost / denominator) if denominator else None
    projected_remaining = round(average * remaining_segments, 4) if average is not None else None
    projected_total = round(current_cost + projected_remaining, 4) if projected_remaining is not None else None

    return {
        "model": model,
        "pricing_supported": True,
        "pricing_source": pricing_source,
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
