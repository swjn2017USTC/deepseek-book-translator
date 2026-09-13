import json
from pathlib import Path

import pytest

from chapter_recovery.evidence_fusion import run_evidence_pipeline
from chapter_recovery.paddleocr import load_pages, normalize_blocks

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_OCR = ROOT / "tests" / "fixtures" / "sample_ocr.json"


def sample_config(**overrides):
    config = {
        "book_id": "sample-book",
        "book_title": "Sample Book",
        "toc_search_pages": [0, 2],
        "toc_alignment_window": 1,
        "review_threshold": 0.9,
    }
    config.update(overrides)
    return config


def run(config, output):
    return run_evidence_pipeline(config, SAMPLE_OCR, output)


def test_provenance_sha_fields(tmp_path):
    first = run(sample_config(), tmp_path / "a")
    assert first["pdf_evidence_ok"] is False
    assert first["pdf_reason"] == "pdf_missing"
    assert first["legacy_profile_used"] is False
    assert first["pages"] > 0
    assert first["candidates"] > 0
    packet = json.loads(
        (tmp_path / "a" / "evidence" / "book_structure_evidence.json").read_text(encoding="utf-8")
    )
    meta = packet["metadata"]
    assert len(meta["input_sha256"]) == 64
    assert meta["pdf_sha256"] is None
    assert meta["legacy_profile_used"] is False
    assert meta["page_count"] == first["pages"]
    assert meta["candidate_count"] == first["candidates"]
    # same input twice -> stable hash
    second = run(sample_config(), tmp_path / "b")
    packet_b = json.loads(
        (tmp_path / "b" / "evidence" / "book_structure_evidence.json").read_text(encoding="utf-8")
    )
    assert packet_b["metadata"]["input_sha256"] == meta["input_sha256"]


def test_legacy_profile_flag(tmp_path):
    result = run(
        sample_config(hierarchy_profile="german_legal_commentary"),
        tmp_path / "profiled",
    )
    assert result["legacy_profile_used"] is True
    result_text = run(sample_config(text_candidate_profile="continuous_prose_monograph"), tmp_path / "tp")
    assert result_text["legacy_profile_used"] is True


def test_coverage_invariant(tmp_path):
    output = tmp_path / "run"
    run(sample_config(), output)
    lines = [
        line
        for line in (output / "evidence" / "heading_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(lines) > 0
    for line in lines:
        candidate = json.loads(line)
        assert candidate["source_block_ids"] or any(
            ref.startswith("pdf:") for ref in candidate["evidence_refs"]
        )


def test_determinism_same_inputs(tmp_path):
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    run(sample_config(), output_a)
    run(sample_config(), output_b)
    for name in ("pdf_evidence.json", "heading_candidates.jsonl", "book_structure_evidence.json"):
        bytes_a = (output_a / "evidence" / name).read_bytes()
        bytes_b = (output_b / "evidence" / name).read_bytes()
        assert bytes_a == bytes_b, f"{name} differs across identical runs"
        assert b"generated_at" not in bytes_a  # no timestamps by default


def test_pipeline_loads_realistic_paddle_json(tmp_path):
    pages = normalize_blocks(load_pages(SAMPLE_OCR))
    assert len(pages) >= 5
    output = tmp_path / "pipeline"
    result = run(sample_config(), output)
    pdf_path = output / "evidence" / "pdf_evidence.json"
    pdf_doc = json.loads(pdf_path.read_text(encoding="utf-8"))
    assert pdf_doc["reason"] == "pdf_missing"
    assert result["output_dir"] == str(output)
