from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
import re
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Block:
    page_index: int
    block_index: int
    raw_block_id: Optional[object]
    label: str
    content: str
    bbox: BBox
    order: Optional[int]
    page_width: float
    page_height: float
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "blk-" + stable_id(
                self.page_index,
                self.raw_block_id,
                self.block_index,
                self.label,
                self.content,
            )

    @property
    def clean_content(self) -> str:
        text = re.sub(r"^(?:\s*#{1,6}\s*)+", "", self.content.strip())
        return " ".join(text.split()).strip()

    @property
    def source_level(self) -> Optional[int]:
        stripped = self.content.lstrip()
        first = re.match(r"#+", stripped)
        count = len(first.group(0)) if first else 0
        return count or None

    def normalized_bbox(self) -> BBox:
        width = self.page_width or 1.0
        height = self.page_height or 1.0
        x1, y1, x2, y2 = self.bbox
        return x1 / width, y1 / height, x2 / width, y2 / height

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        data["bbox_normalized"] = list(self.normalized_bbox())
        data["clean_content"] = self.clean_content
        data["source_level"] = self.source_level
        return data


@dataclass
class TocEntry:
    title: str
    page_label: str
    print_page: Optional[int]
    kind: str
    source_page: int
    id: str = ""
    provenance: str = "ocr_toc"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = "toc-" + stable_id(
                self.source_page, self.title, self.page_label, self.kind
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Node:
    id: str
    kind: str
    level: int
    title_source: str
    page_json: Optional[int]
    page_print: Optional[int]
    parent_id: Optional[str]
    block_id: Optional[str]
    source: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    bbox: Optional[BBox] = None
    source_level: Optional[int] = None
    ignored: bool = False
    zone: str = "mainmatter"
    ordinal: Optional[str] = None
    numbering_path: List[str] = field(default_factory=list)
    source_block_ids: List[str] = field(default_factory=list)
    toc_entry_id: Optional[str] = None
    alignment_status: Optional[str] = None
    alignment_score: Optional[float] = None
    alignment_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data
