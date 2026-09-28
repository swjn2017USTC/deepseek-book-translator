import json
from datetime import datetime
from pathlib import Path

from book_pipeline.cost_control import (
    DEFAULT_FLASH_PRICING_RMB_PER_MILLION,
    cost_report,
    estimate_rmb,
    is_peak_beijing,
    usage_totals,
)


def test_peak_windows_use_beijing_weekdays():
    assert is_peak_beijing(datetime(2026, 9, 28, 10, 0)) is True
    assert is_peak_beijing(datetime(2026, 9, 28, 13, 0)) is False
    assert is_peak_beijing(datetime(2026, 9, 28, 15, 0)) is True
    assert is_peak_beijing(datetime(2026, 9, 27, 10, 0)) is False


def test_usage_totals_count_cache_reasoning_and_retry_requests(tmp_path):
    path = tmp_path / "usage.jsonl"
    rows = [
        {
            "segment_ids": ["s1"],
            "parse_attempts": 2,
            "usage": {
                "prompt_tokens": 200,
                "prompt_cache_hit_tokens": 120,
                "prompt_cache_miss_tokens": 80,
                "completion_tokens": 100,
                "completion_tokens_details": {"reasoning_tokens": 40},
                "request_count": 2,
            },
        },
        {
            "segment_ids": ["s2"],
            "parse_attempts": 1,
            "usage": {
                "prompt_tokens": 100,
                "prompt_cache_hit_tokens": 50,
                "prompt_cache_miss_tokens": 50,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "request_count": 1,
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    totals = usage_totals(path)
    assert totals["segments"] == 2
    assert totals["api_completion_requests"] == 3
    assert totals["prompt_tokens"] == 300
    assert totals["prompt_cache_hit_tokens"] == 170
    assert totals["prompt_cache_miss_tokens"] == 130
    assert totals["completion_tokens"] == 150
    assert totals["reasoning_tokens"] == 40
    assert totals["visible_completion_tokens"] == 110
    assert totals["retry_rate"] > 0


def test_estimate_rmb_uses_cache_hit_miss_and_output_rates():
    totals = {
        "prompt_cache_hit_tokens": 1_000_000,
        "prompt_cache_miss_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
    }
    peak = DEFAULT_FLASH_PRICING_RMB_PER_MILLION["peak"]
    assert estimate_rmb(totals, peak) == peak["cache_hit_input"] + peak["cache_miss_input"] + peak["output"]


def test_cost_report_projects_from_completed_segments(tmp_path):
    path = Path(tmp_path) / "usage.jsonl"
    path.write_text(json.dumps({
        "segment_ids": ["s1"],
        "usage": {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 500,
            "prompt_cache_miss_tokens": 500,
            "completion_tokens": 500,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "request_count": 1,
        },
    }) + "\n", encoding="utf-8")
    report = cost_report(
        {"provider": {"model": "deepseek-flash"}},
        path,
        remaining_segments=9,
        completed_segments=1,
    )
    assert report["usage"]["segments"] == 1
    assert report["projected_total_cost_rmb_at_current_average"] is not None
    assert report["estimated_cost_rmb_if_peak"] >= report["estimated_cost_rmb_if_off_peak"]


def test_unknown_model_pricing_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    path = Path(tmp_path) / "usage.jsonl"
    path.write_text(json.dumps({
        "model": "custom-expensive-model",
        "segment_ids": ["s1"],
        "status": "completed",
        "usage": {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 1000,
            "completion_tokens": 500,
            "request_count": 1,
        },
    }) + "\n", encoding="utf-8")
    report = cost_report(
        {"provider": {"model": "custom-expensive-model"}},
        path,
        remaining_segments=9,
        completed_segments=1,
    )
    assert report["pricing_supported"] is False
    assert report["estimated_cost_rmb_so_far"] is None
    assert "unknown_or_mixed_model_pricing" in report["warnings"]


def test_failed_usage_cost_does_not_inflate_completed_denominator(tmp_path):
    path = Path(tmp_path) / "usage.jsonl"
    rows = [
        {
            "model": "deepseek-flash",
            "segment_ids": ["failed"],
            "status": "failed_validation",
            "usage": {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000,
                "completion_tokens": 1000,
                "request_count": 3,
            },
        },
        {
            "model": "deepseek-flash",
            "segment_ids": ["done"],
            "status": "completed",
            "usage": {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1000,
                "completion_tokens": 1000,
                "request_count": 1,
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = cost_report(
        {"provider": {"model": "deepseek-flash"}},
        path,
        remaining_segments=1,
        completed_segments=1,
    )
    assert report["usage"]["completed_segment_ids"] == 1
    assert report["usage"]["failed_segment_ids"] == 1
    assert report["average_cost_rmb_per_completed_segment"] == report["estimated_cost_rmb_so_far"]


def test_mixed_historical_models_disable_builtin_flash_pricing(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    path = Path(tmp_path) / "usage.jsonl"
    rows = [
        {
            "model": "deepseek-flash",
            "segment_ids": ["s1"],
            "status": "completed",
            "usage": {
                "prompt_tokens": 10,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10,
                "completion_tokens": 10,
                "request_count": 1,
            },
        },
        {
            "model": "some-other-model",
            "segment_ids": ["s2"],
            "status": "completed",
            "usage": {
                "prompt_tokens": 10,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10,
                "completion_tokens": 10,
                "request_count": 1,
            },
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = cost_report(
        {"provider": {"model": "deepseek-flash"}},
        path,
        remaining_segments=1,
        completed_segments=2,
    )
    assert report["pricing_supported"] is False
    assert "unknown_or_mixed_model_pricing" in report["warnings"]
