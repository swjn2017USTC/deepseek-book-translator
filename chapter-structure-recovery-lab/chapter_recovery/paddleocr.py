from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .models import Block


BLOCK_PATH = ("prunedResult", "parsing_res_list")


def _raw_blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    value: Any = page
    for key in BLOCK_PATH:
        if not isinstance(value, dict):
            return []
        value = value.get(key, {})
    return value if isinstance(value, list) else []


def load_pages(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected the OCR JSON top level to be a list of pages")
    return data


def _canvas_size(pages: Iterable[Dict[str, Any]]) -> Tuple[float, float]:
    max_x = 1.0
    max_y = 1.0
    for page in pages:
        for block in _raw_blocks(page):
            bbox = block.get("block_bbox") or [0, 0, 0, 0]
            if isinstance(bbox, list) and len(bbox) == 4:
                max_x = max(max_x, float(bbox[2]))
                max_y = max(max_y, float(bbox[3]))
    return max_x, max_y


def normalize_blocks(pages: List[Dict[str, Any]]) -> List[List[Block]]:
    page_width, page_height = _canvas_size(pages)
    normalized: List[List[Block]] = []
    for page_index, page in enumerate(pages):
        blocks: List[Block] = []
        for block_index, raw in enumerate(_raw_blocks(page)):
            bbox_raw = raw.get("block_bbox") or [0, 0, 0, 0]
            if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
                bbox_raw = [0, 0, 0, 0]
            bbox = tuple(float(v) for v in bbox_raw)
            order = raw.get("block_order")
            blocks.append(
                Block(
                    page_index=page_index,
                    block_index=block_index,
                    raw_block_id=raw.get("block_id"),
                    label=str(raw.get("block_label") or "unknown"),
                    content=str(raw.get("block_content") or ""),
                    bbox=bbox,  # type: ignore[arg-type]
                    order=order if isinstance(order, int) else None,
                    page_width=page_width,
                    page_height=page_height,
                )
            )
        blocks.sort(
            key=lambda item: (
                item.order is None,
                item.order if item.order is not None else item.bbox[1],
                item.bbox[0],
            )
        )
        normalized.append(blocks)
    return normalized
