import json
from pathlib import Path

import pytest

from book_pipeline.io_utils import read_jsonl, write_json, write_jsonl
from book_pipeline.models import Segment, digest
from book_pipeline.source_manuscript import (
    build_source_manuscript,
    confirm_source_manuscript,
    require_source_confirmation,
    source_status,
)


def _config(tmp_path: Path):
    structure_dir = tmp_path / "structure"
    output_dir = tmp_path / "work"
    structure_dir.mkdir()
    output_dir.mkdir()
    input_path = tmp_path / "book.json"
    pages = [
        {
            "prunedResult": {
                "parsing_res_list": [
                    {"block_id": 1, "block_label": "paragraph_title", "block_content": "Chapter One", "block_bbox": [100, 50, 600, 100]},
                    {"block_id": 2, "block_label": "text", "block_content": "A long paragraph reaches the bottom:", "block_bbox": [100, 700, 900, 950]},
                    {"block_id": 3, "block_label": "number", "block_content": "1", "block_bbox": [480, 980, 520, 1000]},
                ]
            }
        },
        {
            "prunedResult": {
                "parsing_res_list": [
                    {"block_id": 4, "block_label": "text", "block_content": "continuation on the next page.", "block_bbox": [100, 40, 900, 180]},
                    {"block_id": 5, "block_label": "number", "block_content": "2", "block_bbox": [480, 980, 520, 1000]},
                ]
            }
        },
    ]
    input_path.write_text(json.dumps(pages), encoding="utf-8")

    structure_path = structure_dir / "book_structure.json"
    write_json(structure_path, {
        "schema_version": 2,
        "book_id": "source-review",
        "nodes": [
            {"id": "book-root", "kind": "book", "level": 0, "title_source": "", "parent_id": None},
            {"id": "chapter-1", "kind": "chapter", "level": 2, "title_source": "Chapter One", "parent_id": "book-root", "page_json": 0, "zone": "mainmatter"},
        ],
    })
    validation = structure_dir / "validation.json"
    write_json(validation, {"ok": True})
    write_jsonl(structure_dir / "review_packets.jsonl", [])

    segments = [
        Segment(
            id="seg-heading", source_hash=digest("Chapter One"), kind="heading",
            source_text="Chapter One", page_json=0, zone="mainmatter",
            parent_node_id="book-root", level=2, translatable=True,
            source_block_ids=["blk-title"],
            metadata={"node_id": "chapter-1", "node_kind": "chapter", "bbox": [100, 50, 600, 100], "last_page": 0},
        ),
        Segment(
            id="seg-p1", source_hash=digest("A long paragraph reaches the bottom:"), kind="paragraph",
            source_text="A long paragraph reaches the bottom:", page_json=0, zone="mainmatter",
            parent_node_id="chapter-1", translatable=True, source_block_ids=["blk-p1"],
            metadata={"bbox": [100, 700, 900, 950], "last_page": 0},
        ),
        Segment(
            id="seg-p2", source_hash=digest("continuation on the next page."), kind="paragraph",
            source_text="continuation on the next page.", page_json=1, zone="mainmatter",
            parent_node_id="chapter-1", translatable=True, source_block_ids=["blk-p2"],
            metadata={"bbox": [100, 40, 900, 180], "last_page": 1},
        ),
    ]
    write_jsonl(output_dir / "cleaned_segments.jsonl", [segment.to_dict() for segment in segments])
    write_json(output_dir / "cleaning_report.json", {"schema_version": 1, "book_id": "source-review", "metrics": {"segments": 3}})

    config_path = tmp_path / "translation_config.json"
    config = {
        "_base_dir": str(tmp_path),
        "_config_path": str(config_path),
        "book_id": "source-review",
        "book_title": "Source Review",
        "book_title_zh": "源稿审核",
        "source_adapter": "ocr_json_v1",
        "input_json": str(input_path),
        "structure_json": str(structure_path),
        "output_dir": str(output_dir),
        "quality": {
            "validation_json": str(validation),
            "review_packets": str(structure_dir / "review_packets.jsonl"),
            "allow_pending_review": False,
        },
        "source_review": {
            "mode": "manual",
            "directory": str(tmp_path / "source"),
            "markdown": str(tmp_path / "source" / "structured_source.md"),
            "state": str(tmp_path / "source" / "source_state.json"),
            "generated_segments": str(tmp_path / "source" / "generated_segments.jsonl"),
            "cross_page_merge": True,
            "aggressive_cross_page_merge": False,
        },
    }
    # provenance expects the config path to exist.
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    config_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
    return config


def test_structured_source_cross_page_merge_edit_and_confirmation(tmp_path):
    config = _config(tmp_path)
    generated = build_source_manuscript(config)

    assert generated["status"] == "needs_confirmation"
    assert generated["cross_page_merged"] == 1
    markdown = Path(generated["markdown"])
    text = markdown.read_text(encoding="utf-8")
    assert "〔OCR/PDF 页：1–2〕" in text

    generated_segments = list(read_jsonl(tmp_path / "source" / "generated_segments.jsonl"))
    assert len(generated_segments) == 2
    merged = next(row for row in generated_segments if row["kind"] == "paragraph")
    assert merged["metadata"]["page_span"] == [0, 1]
    assert merged["metadata"]["source_manuscript_merged_segment_ids"] == ["seg-p1", "seg-p2"]

    text = text.replace("## Chapter One", "### Edited Chapter")
    text = text.replace("continuation on the next page.", "revised continuation on the next page.")
    markdown.write_text(text, encoding="utf-8")

    confirmed = confirm_source_manuscript(config)
    assert confirmed["status"] == "confirmed"
    assert require_source_confirmation(config)["status"] == "confirmed"

    structure = json.loads((tmp_path / "structure" / "book_structure.json").read_text(encoding="utf-8"))
    chapter = next(node for node in structure["nodes"] if node.get("id") == "chapter-1")
    assert chapter["title_source"] == "Edited Chapter"
    assert chapter["level"] == 3

    rows = list(read_jsonl(tmp_path / "work" / "cleaned_segments.jsonl"))
    assert len(rows) == 2
    heading = next(row for row in rows if row["kind"] == "heading")
    paragraph = next(row for row in rows if row["kind"] == "paragraph")
    assert heading["source_text"] == "Edited Chapter"
    assert "revised continuation" in paragraph["source_text"]
    assert paragraph["parent_node_id"] == "chapter-1"

    manifest = json.loads((tmp_path / "work" / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_manuscript"] == str(markdown)
    assert manifest["source_confirmation_state"]

    markdown.write_text(text + "\n", encoding="utf-8")
    assert source_status(config)["status"] == "edited_after_confirmation"
    with pytest.raises(RuntimeError, match="not confirmed"):
        require_source_confirmation(config)
