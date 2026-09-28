from __future__ import annotations

import json
from pathlib import Path

from book_pipeline.glossary_llm import auto_review_glossary
from book_pipeline.io_utils import read_jsonl, write_json, write_jsonl


class FakeGlossaryClient:
    model = "fake-glossary-model"

    def __init__(self):
        self.calls = 0

    def complete_json_text(self, _system: str, user: str):
        self.calls += 1
        payload = json.loads(user)
        decisions = []
        for row in payload["candidates"]:
            candidate_id = row["candidate_id"]
            if candidate_id == "glo-111111111111":
                if "first_decision" in row:
                    decisions.append({
                        "candidate_id": candidate_id,
                        "decision": "include",
                        "translation": "最终术语",
                        "category": "concept",
                        "alternatives": ["备选术语"],
                        "note": "second-pass adjudication",
                        "reason": "",
                        "confidence": 0.94,
                    })
                else:
                    decisions.append({
                        "candidate_id": candidate_id,
                        "decision": "include",
                        "translation": "初译",
                        "category": "concept",
                        "alternatives": [],
                        "note": "",
                        "reason": "",
                        "confidence": 0.55,
                    })
            else:
                decisions.append({
                    "candidate_id": candidate_id,
                    "decision": "reject",
                    "translation": "",
                    "category": "",
                    "alternatives": [],
                    "note": "",
                    "reason": "普通搭配，不需要固定进全书术语表",
                    "confidence": 0.97,
                })
        return json.dumps({"decisions": decisions}, ensure_ascii=False), {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


def _config(tmp_path: Path) -> dict:
    glossary = tmp_path / "glossary.json"
    manifest = tmp_path / "glossary_manifest.json"
    candidates = tmp_path / "glossary_candidates.jsonl"
    write_json(glossary, [])
    write_json(manifest, {"status": "needs_review"})
    write_jsonl(candidates, [
        {
            "candidate_id": "glo-111111111111",
            "term": "social reproduction",
            "reasons": ["repeated_phrase"],
            "occurrences": 5,
            "sample_contexts": ["The crisis of social reproduction reshaped the argument."],
            "evidence_segment_ids": ["seg-a", "seg-b"],
        },
        {
            "candidate_id": "glo-222222222222",
            "term": "in this chapter",
            "reasons": ["repeated_phrase"],
            "occurrences": 6,
            "sample_contexts": ["In this chapter we examine the evidence."],
            "evidence_segment_ids": ["seg-c"],
        },
    ])
    return {
        "_base_dir": str(tmp_path),
        "book_id": "test-book",
        "book_title": "Test Book",
        "book_title_zh": "测试书",
        "author": "Author",
        "source_lang": "英语",
        "target_lang": "简体中文",
        "domain": "学术人文社科",
        "glossary": str(glossary),
        "glossary_settings": {
            "manifest": str(manifest),
            "candidates": str(candidates),
            "llm_review": {
                "batch_size": 12,
                "structured_retries": 2,
                "adjudicate_below": 0.78,
            },
        },
    }


def test_llm_glossary_review_adjudicates_low_confidence_and_writes_audit(tmp_path):
    client = FakeGlossaryClient()
    report = auto_review_glossary(_config(tmp_path), client=client)

    assert report["status"] == "reviewed"
    assert report["candidate_count"] == 2
    assert report["included"] == 1
    assert report["rejected"] == 1
    assert report["adjudicated"] == 1
    assert client.calls == 2

    decisions = list(read_jsonl(tmp_path / "glossary_decisions.jsonl"))
    included = next(row for row in decisions if row["decision"] == "include")
    rejected = next(row for row in decisions if row["decision"] == "reject")
    assert included["translation"] == "最终术语"
    assert included["evidence"] == ["segment:seg-a", "segment:seg-b"]
    assert rejected["reason"]

    audit = list(read_jsonl(tmp_path / "glossary_llm_review.jsonl"))
    first = next(row for row in audit if row["candidate_id"] == "glo-111111111111")
    assert first["adjudicated"] is True
    assert first["final_decision"]["confidence"] == 0.94
    assert report["usage"]["request_count"] == 2
