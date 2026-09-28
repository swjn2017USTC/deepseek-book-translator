from chapter_recovery.models import Block, TocEntry
from chapter_recovery.vision_evidence import (
    inject_vision_headings,
    merge_vision_toc,
)


def _block(page, index, label, text, bbox):
    return Block(
        page_index=page,
        block_index=index,
        raw_block_id=index,
        label=label,
        content=text,
        bbox=bbox,
        order=index,
        page_width=1000,
        page_height=1400,
    )


def test_vision_toc_fills_missing_page_and_adds_omitted_entry():
    parsed = [
        TocEntry(
            title="Chapter One",
            page_label="",
            print_page=None,
            kind="chapter",
            source_page=2,
        )
    ]
    evidence = {
        "schema_version": 1,
        "toc_pages": [
            {
                "page_json": 2,
                "is_toc": True,
                "confidence": 0.98,
                "entries": [
                    {
                        "title": "Chapter One",
                        "page_label": "7",
                        "kind": "chapter",
                        "level": 1,
                        "confidence": 0.98,
                    },
                    {
                        "title": "1.1 Hidden Section",
                        "page_label": "9",
                        "kind": "section",
                        "level": 2,
                        "confidence": 0.94,
                    },
                ],
            }
        ],
        "headings": [],
    }

    merged = merge_vision_toc(parsed, evidence)

    chapter = next(item for item in merged if item.title == "Chapter One")
    hidden = next(item for item in merged if item.title == "1.1 Hidden Section")
    assert chapter.print_page == 7
    assert "deepseek_vision_toc" in chapter.provenance
    assert hidden.print_page == 9
    assert hidden.kind == "section"
    assert hidden.provenance == "deepseek_vision_toc"
    assert hidden.source_level == 2


def test_vision_heading_grounding_preserves_ocr_text_and_marks_unmatched_for_review():
    pages = [
        [
            _block(0, 0, "text", "Ordinary paragraph.", (100, 200, 900, 400)),
        ],
        [
            _block(1, 0, "text", "A Hidden Section", (120, 180, 520, 230)),
            _block(1, 1, "text", "Body paragraph follows here with enough words.", (100, 300, 900, 700)),
        ],
    ]
    evidence = {
        "schema_version": 1,
        "toc_pages": [],
        "headings": [
            {
                "page_json": 1,
                "title": "A Hidden Section",
                "kind": "section",
                "level": 4,
                "bbox": [0.12, 0.12, 0.52, 0.17],
                "confidence": 0.96,
            },
            {
                "page_json": 1,
                "title": "OCR Completely Missed This Heading",
                "kind": "section",
                "level": 3,
                "bbox": [0.10, 0.50, 0.80, 0.56],
                "confidence": 0.95,
            },
        ],
    }

    stats = inject_vision_headings(pages, evidence, {"vision": {}})

    assert stats["matched_ocr_blocks"] == 1
    assert stats["synthetic_review_headings"] == 1
    grounded = next(block for block in pages[1] if block.label == "vision_heading")
    synthetic = next(block for block in pages[1] if block.label == "vision_heading_unmatched")
    assert grounded.clean_content == "A Hidden Section"
    assert grounded.source_level == 4
    assert synthetic.clean_content == "OCR Completely Missed This Heading"
    assert synthetic.source_level == 3


def test_vision_evidence_rejects_stale_ocr_source(tmp_path):
    import json
    import pytest

    from chapter_recovery.vision_evidence import load_vision_evidence

    source = tmp_path / "book.json"
    source.write_text("[]", encoding="utf-8")
    sidecar = tmp_path / "vision.json"
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "input_sha256": "0" * 64,
        "toc_pages": [],
        "headings": [],
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="stale"):
        load_vision_evidence(sidecar, input_path=source)
