from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional


def digest(*parts: object) -> str:
    return sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


@dataclass
class Segment:
    id: str
    source_hash: str
    kind: str
    source_text: str
    page_json: Optional[int]
    zone: str
    parent_node_id: Optional[str]
    level: Optional[int] = None
    translatable: bool = True
    source_block_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranslationRecord:
    segment_id: str
    source_hash: str
    translated_text: str
    model: str
    status: str
    attempt: int
    timestamp: str
    glossary_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
