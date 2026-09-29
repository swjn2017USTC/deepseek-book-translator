import json
from pathlib import Path

from book_pipeline.config import load_config
from book_pipeline.io_utils import read_jsonl, write_json, write_jsonl
from book_pipeline.models import Segment, digest
from book_pipeline.provenance import write_clean_manifest
from book_pipeline.source_review import (
    apply_source_review,
    choose_source_review,
    render_source_review,
    require_approved_source,
)


def _fixture(tmp_path: Path):
    project = tmp_path
    structure_dir = project / "structure"
    work = project / "work"
    structure_dir.mkdir()
    work.mkdir()
    source = project / "input.json"
    source.write_text("[]", encoding="utf-8")
    structure = {
        "schema_version": 2,
        "book_id": "review-book",
        "nodes": [
            {"id": "book-review", "kind": "book", "level": 0, "title_source": "Book", "parent_id": None},
            {"id": "chapter-1", "kind": "chapter", "level": 2, "title_source": "Chapter One",
             "parent_id": "book-review", "page_json": 1, "zone": "mainmatter"},
        ],
    }
    structure_path = structure_dir / "book_structure.json"
    write_json(structure_path, structure)
    segments = [
        Segment("seg-h", digest("Chapter One"), "heading", "Chapter One", 1, "mainmatter",
                "book-review", 2, True, ["b1"], {"node_id": "chapter-1", "node_kind": "chapter"}),
        Segment("seg-p", digest("A sentence continuing across pages"), "paragraph",
                "A sentence continuing across pages", 1, "mainmatter", "chapter-1",
                None, True, ["b2", "b3"], {"page_start": 1, "page_end": 2, "cross_page_stitch": True}),
    ]
    segments_path = work / "cleaned_segments.jsonl"
    write_jsonl(segments_path, [segment.to_dict() for segment in segments])
    report_path = work / "cleaning_report.json"
    write_json(report_path, {"schema_version": 1, "book_id": "review-book", "metrics": {"segments": 2}})
    config_path = project / "translation_config.json"
    write_json(config_path, {
        "book_id": "review-book",
        "book_title": "Review Book",
        "book_title_zh": "审核书",
        "input_json": str(source),
        "structure_json": str(structure_path),
        "output_dir": str(work),
        "quality": {"validation_json": str(structure_dir / "validation.json")},
    })
    config = load_config(config_path)
    write_clean_manifest(config, segments_path, report_path)
    return project, config


def test_source_review_round_trip_updates_canonical_text_structure_and_page_span(tmp_path):
    project, config = _fixture(tmp_path)
    state = render_source_review(config, project)
    markdown = Path(state["markdown"])
    text = markdown.read_text(encoding="utf-8")
    assert "Source pages: json 1–2" in text
    assert "BOOK_SEGMENT" in text

    choose_source_review(config, project, "manual")
    edited = text.replace("## Chapter One", "### Revised Chapter")
    edited = edited.replace("A sentence continuing across pages", "A corrected sentence across pages.")
    markdown.write_text(edited, encoding="utf-8")

    approved = apply_source_review(config, project)
    reloaded = load_config(project / "translation_config.json")
    approved_now = require_approved_source(reloaded, project)
    assert approved["status"] == "approved"
    assert approved_now["review_mode"] == "manual"
    assert Path(reloaded["structure_json"]).name == "reviewed_structure.json"

    rows = list(read_jsonl(project / "work" / "cleaned_segments.jsonl"))
    heading = next(row for row in rows if row["id"] == "seg-h")
    paragraph = next(row for row in rows if row["id"] == "seg-p")
    assert heading["source_text"] == "Revised Chapter"
    assert heading["level"] == 3
    assert paragraph["source_text"] == "A corrected sentence across pages."
    assert paragraph["source_hash"] == digest(paragraph["source_text"])

    structure = json.loads(Path(reloaded["structure_json"]).read_text(encoding="utf-8"))
    node = next(node for node in structure["nodes"] if node["id"] == "chapter-1")
    assert node["title_source"] == "Revised Chapter"
    assert node["level"] == 3
    assert node["manual_source_review"] is True
