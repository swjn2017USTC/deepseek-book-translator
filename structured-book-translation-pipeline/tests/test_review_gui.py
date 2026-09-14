from pathlib import Path

import pytest

from book_pipeline.io_utils import write_jsonl
from book_pipeline.publish import build_book_tex
from book_pipeline.review_gui import decision_from_fields, load_review


def test_review_decisions_require_evidence_and_reasons():
    with pytest.raises(ValueError, match="译名"):
        decision_from_fields("glossary", "glo-abcdefabcdef", {
            "decision": "include", "translation": "", "evidence": "",
            "category": "", "reason": "",
        })
    included = decision_from_fields("glossary", "glo-abcdefabcdef", {
        "decision": "include", "translation": "术语", "evidence": "p1: heading",
        "category": "book_specific", "reason": "",
    })
    assert included["evidence"] == ["p1: heading"]
    with pytest.raises(ValueError, match="目标层级"):
        decision_from_fields("structure", "candidate-1", {
            "decision": "change_level", "reason": "page_evidence", "evidence": "page_context.blocks[0]",
            "level": "9", "kind": "", "merge_with_node_id": "", "translation": "", "category": "",
        })


def test_review_load_rejects_stale_decisions(tmp_path: Path):
    write_jsonl(tmp_path / "glossary_candidates.jsonl", [{"candidate_id": "glo-a", "term": "a"}])
    write_jsonl(tmp_path / "glossary_decisions.template.jsonl", [{"candidate_id": "glo-a", "decision": "needs_human"}])
    write_jsonl(tmp_path / "glossary_decisions.jsonl", [{"candidate_id": "glo-old", "decision": "reject", "reason": "old"}])
    candidates, decisions, _ = load_review(tmp_path, "glossary")
    assert candidates[0]["term"] == "a"
    assert decisions == [{"candidate_id": "glo-a", "decision": "needs_human"}]


def test_no_pending_structure_review_needs_no_template(tmp_path: Path):
    (tmp_path / "structure").mkdir()
    (tmp_path / "structure" / "review_packets.jsonl").write_text("", encoding="utf-8")
    candidates, decisions, _ = load_review(tmp_path, "structure")
    assert candidates == decisions == []


def test_windows_pdf_font_profile():
    tex = build_book_tex("样例", "作者", font_profile="windows")
    assert r"\setCJKmainfont{SimSun}" in tex
    assert r"\setCJKsansfont{Microsoft YaHei}" in tex
