from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import Block


ARABIC_PAGE = re.compile(r"^\s*(\d{1,4})\s*$")


@dataclass
class PageMap:
    offset: Optional[int]
    confidence: float
    support: int
    observations: int
    print_to_json: Dict[int, int]

    def json_page(self, print_page: int) -> Optional[int]:
        if print_page in self.print_to_json:
            return self.print_to_json[print_page]
        if self.offset is None:
            return None
        return print_page - self.offset

    def to_dict(self) -> Dict[str, object]:
        return {
            "formula": "print_page = json_page + offset",
            "offset": self.offset,
            "confidence": round(self.confidence, 4),
            "support": self.support,
            "observations": self.observations,
            "print_to_json": {str(k): v for k, v in sorted(self.print_to_json.items())},
        }


def infer_page_map(pages: List[List[Block]]) -> PageMap:
    observations = []
    for blocks in pages:
        for block in blocks:
            if block.label != "number":
                continue
            match = ARABIC_PAGE.match(block.clean_content)
            if not match:
                continue
            _, y1, _, y2 = block.normalized_bbox()
            if not (y1 <= 0.12 or y2 >= 0.85):
                continue
            print_page = int(match.group(1))
            if print_page <= 0:
                continue
            observations.append((block.page_index, print_page))

    if not observations:
        return PageMap(None, 0.0, 0, 0, {})

    offsets = Counter(print_page - json_page for json_page, print_page in observations)
    offset, support = offsets.most_common(1)[0]
    consistent = [item for item in observations if item[1] - item[0] == offset]
    print_to_json = {print_page: json_page for json_page, print_page in consistent}
    return PageMap(
        offset=offset,
        confidence=support / len(observations),
        support=support,
        observations=len(observations),
        print_to_json=print_to_json,
    )
