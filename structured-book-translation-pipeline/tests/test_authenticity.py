"""Demo/fake translation authenticity guard tests (P09 user-reported defect).

render/export must fail closed on demo/fake translation rows unless the caller
explicitly opts in via config allow_demo_translations=true.
"""
import json
from pathlib import Path

import pytest

from book_pipeline.authenticity import (
    DEMO_MODEL,
    current_rows,
    demo_current_segments,
    is_demo_row,
    reject_demo_translations,
)

REAL = {"segment_id": "seg-aaa", "model": "deepseek-v4-flash", "translated_text": "中文"}
DEMO = {"segment_id": "seg-bbb", "model": "demo-fake-contextual", "translated_text": "译：English"}
FAKE = {"segment_id": "seg-ccc", "model": "fake-model", "translated_text": "译：English"}


def test_demo_model_detection():
    assert DEMO_MODEL.search("demo-fake-contextual")
    assert DEMO_MODEL.search("fake-model")
    assert not DEMO_MODEL.search("deepseek-v4-flash")
    assert is_demo_row(DEMO) and is_demo_row(FAKE)
    assert not is_demo_row(REAL)


def test_metadata_marker_detection():
    row = dict(REAL)
    row["metadata"] = {"demo_only": True}
    assert is_demo_row(row)


def test_current_rows_last_write_wins_and_demo_detection():
    rows = [dict(REAL, segment_id="seg-x"), dict(REAL, segment_id="seg-x", translated_text="修订")]
    latest = current_rows(rows)
    assert latest["seg-x"]["translated_text"] == "修订"
    assert demo_current_segments([DEMO, REAL]) == ["seg-bbb"]


def test_reject_demo_fails_closed_by_default():
    with pytest.raises(RuntimeError, match="demo or fake translations"):
        reject_demo_translations({}, [REAL, DEMO])
    # superseded demo row no longer blocks
    rows = [DEMO, dict(DEMO, model="deepseek-v4-flash", translated_text="真译文")]
    reject_demo_translations({}, rows)  # no raise (latest is real)


def test_reject_demo_allowed_when_opted_in():
    reject_demo_translations({"allow_demo_translations": True}, [REAL, DEMO, FAKE])


def test_all_real_passes():
    reject_demo_translations({}, [REAL])
