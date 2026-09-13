from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .heading_text import canonical_heading_text
from .models import Block


VISUAL_LABELS = {"image", "image_body", "figure", "chart", "table"}
CAPTION_PREFIX = re.compile(
    r"^(?:fig(?:ure)?|map|table|chart|plate|abb(?:ildung)?|karte|tafel)\s*[.:]?\s*\w",
    re.I,
)


@dataclass(frozen=True)
class VisualContext:
    rejection_reason: Optional[str] = None
    confidence_cap: Optional[float] = None
    evidence: List[str] = field(default_factory=list)


def _intersection_ratio(inner: Block, outer: Block) -> float:
    ix1, iy1, ix2, iy2 = inner.normalized_bbox()
    ox1, oy1, ox2, oy2 = outer.normalized_bbox()
    width = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    height = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(1e-9, (ix2 - ix1) * (iy2 - iy1))
    return width * height / inner_area


def _horizontal_overlap(left: Block, right: Block) -> float:
    lx1, _, lx2, _ = left.normalized_bbox()
    rx1, _, rx2, _ = right.normalized_bbox()
    overlap = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    return overlap / max(1e-9, min(lx2 - lx1, rx2 - rx1))


def _vertical_gap(left: Block, right: Block) -> float:
    _, ly1, _, ly2 = left.normalized_bbox()
    _, ry1, _, ry2 = right.normalized_bbox()
    if ly2 < ry1:
        return ry1 - ly2
    if ry2 < ly1:
        return ly1 - ry2
    return 0.0


def heading_visual_context(heading: Block, page_blocks: Sequence[Block]) -> VisualContext:
    """Classify strong caption evidence without rejecting nearby real sections."""
    visuals = [block for block in page_blocks if block.label in VISUAL_LABELS]
    if not visuals:
        return VisualContext()
    text = canonical_heading_text(heading.clean_content)
    caption_like = bool(CAPTION_PREFIX.match(text))
    evidence: List[str] = []
    for visual in visuals:
        overlap = _intersection_ratio(heading, visual)
        vx1, vy1, vx2, vy2 = visual.normalized_bbox()
        visual_area = max(0.0, (vx2 - vx1) * (vy2 - vy1))
        if overlap >= 0.80 and visual_area >= 0.08:
            return VisualContext(
                rejection_reason="visual_embedded_caption",
                evidence=[f"embedded_in_{visual.label}", "visual_caption_geometry"],
            )
        if _horizontal_overlap(heading, visual) >= 0.35 and _vertical_gap(heading, visual) <= 0.035:
            evidence.append(f"adjacent_{visual.label}")
    if evidence and caption_like:
        return VisualContext(
            rejection_reason="visual_adjacent_caption",
            evidence=evidence + ["caption_prefix"],
        )
    if evidence:
        return VisualContext(confidence_cap=0.90, evidence=evidence + ["visual_adjacency_ambiguous"])
    return VisualContext()
