from __future__ import annotations

import base64
import json
from pathlib import Path

from book_pipeline.structure_vision import collect_structure_vision


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQMcAAAAASUVORK5CYII="
)


class FakeVisionClient:
    model = "fake-vision"

    def __init__(self):
        self.calls = 0

    def complete_multimodal_json_text(self, _system, content):
        self.calls += 1
        instruction = content[0]["text"]
        page_ids = [
            int(block["text"].split("\n", 1)[0].split("=", 1)[1])
            for block in content
            if block.get("type") == "text" and block["text"].startswith("PAGE_JSON=")
        ]
        if "table-of-contents" in instruction:
            rows = []
            for page in page_ids:
                rows.append({
                    "page_json": page,
                    "is_toc": page == 0,
                    "confidence": 0.99,
                    "entries": (
                        [{
                            "title": "Chapter One",
                            "page_label": "1",
                            "kind": "chapter",
                            "level": 1,
                            "confidence": 0.98,
                        }]
                        if page == 0
                        else []
                    ),
                })
            payload = {"pages": rows}
        else:
            payload = {
                "pages": [
                    {
                        "page_json": page,
                        "headings": (
                            [{
                                "title": "A Hidden Section",
                                "kind": "section",
                                "level": 3,
                                "bbox": [0.1, 0.1, 0.8, 0.18],
                                "confidence": 0.96,
                            }]
                            if page == 1
                            else []
                        ),
                    }
                    for page in page_ids
                ]
            }
        return json.dumps(payload), {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        }


def test_collect_structure_vision_uses_local_page_images_and_finds_hidden_heading(tmp_path: Path):
    image = tmp_path / "page.png"
    image.write_bytes(PNG_1X1)
    source = tmp_path / "book.json"
    source.write_text(json.dumps([
        {
            "inputImage": str(image),
            "prunedResult": {
                "parsing_res_list": [
                    {
                        "block_id": 1,
                        "block_label": "paragraph_title",
                        "block_content": "Contents",
                        "block_bbox": [100, 100, 400, 160],
                        "block_order": 1,
                    },
                    {
                        "block_id": 2,
                        "block_label": "content",
                        "block_content": "Chapter One 1",
                        "block_bbox": [100, 220, 800, 700],
                        "block_order": 2,
                    },
                ]
            },
        },
        {
            "inputImage": str(image),
            "prunedResult": {
                "parsing_res_list": [
                    {
                        "block_id": 1,
                        "block_label": "text",
                        "block_content": "A Hidden Section",
                        "block_bbox": [100, 150, 500, 200],
                        "block_order": 1,
                    },
                    {
                        "block_id": 2,
                        "block_label": "text",
                        "block_content": "This is body text following the hidden section heading.",
                        "block_bbox": [100, 260, 900, 800],
                        "block_order": 2,
                    },
                    {
                        "block_id": 3,
                        "block_label": "number",
                        "block_content": "1",
                        "block_bbox": [850, 1300, 900, 1340],
                        "block_order": 3,
                    },
                ]
            },
        },
    ]), encoding="utf-8")
    config = {
        "book_id": "vision-test",
        "book_title": "Vision Test",
        "input_json": str(source),
        "output_dir": str(tmp_path / "out"),
        "toc_search_pages": [0, 1],
        "vision": {
            "toc_scan_pages": [0, 1],
            "toc_batch_size": 4,
            "body_batch_size": 4,
            "max_body_pages": 10,
        },
    }
    client = FakeVisionClient()

    result = collect_structure_vision(
        config,
        {"model": "deepseek-flash"},
        config_base=tmp_path,
        client=client,
    )

    assert client.calls == 2
    assert any(row["page_json"] == 0 and row["is_toc"] for row in result["toc_pages"])
    assert any(row["title"] == "A Hidden Section" and row["page_json"] == 1 for row in result["headings"])
    assert result["scan"]["image_sources"]["ocr_local_image"] == 2
    assert result["usage"]["request_count"] == 2
