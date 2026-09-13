from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from statistics import median
from typing import Dict, List

from .models import Block
from .heading_text import numbering
from .heading_text import canonical_heading_text


@dataclass
class StyleDecision:
    level: int
    confidence: float
    evidence: List[str]


def infer_internal_styles(blocks: List[Block]) -> Dict[str, StyleDecision]:
    """Infer levels from repeated layout, treating OCR hashes as weak evidence.

    Height is the most stable signal available in the current PaddleOCR schema.
    Mirrored page margins make raw x positions unsafe across odd/even pages.  A
    distinctly smaller repeated height class may be a subsection; singleton
    classes stay at section level and are marked for later review.
    """
    if not blocks:
        return {}
    heights = {
        block.id: max(0.0001, block.normalized_bbox()[3] - block.normalized_bbox()[1])
        for block in blocks
    }
    typical = median(heights.values())
    smaller = [block for block in blocks if heights[block.id] < typical * 0.78]
    smaller_ids = {block.id for block in smaller} if len(smaller) >= 2 else set()
    source_levels = {block.source_level for block in blocks if block.source_level}
    hashes_disagree = len(source_levels) > 1
    title_frequency = Counter(
        canonical_heading_text(block.clean_content).casefold() for block in blocks
    )

    decisions: Dict[str, StyleDecision] = {}
    for block in blocks:
        parsed = numbering(block.clean_content)
        level = 4 if block.id in smaller_ids else 3
        evidence = ["repeated_layout_height", "ocr_level_is_weak"]
        confidence = 0.93
        if parsed and parsed.family == "decimal":
            level = min(6, 2 + parsed.depth)
            confidence = 0.98
            evidence.append("hierarchical_numbering_grammar")
        elif parsed and parsed.family == "upper_letter":
            level = 3
            confidence = 0.96
            evidence.append("upper_letter_numbering_grammar")
        elif parsed and parsed.family == "lower_letter":
            level = 4
            confidence = 0.94
            evidence.append("lower_letter_numbering_grammar")
        elif parsed and parsed.family == "section_symbol":
            level = 5
            confidence = 0.99
            evidence.append("section_symbol_numbering_grammar")
        elif parsed and parsed.family in {"teil", "part", "book", "volume"}:
            level = 2
            confidence = 0.97
            evidence.append("labelled_numbering_grammar")
        elif parsed and parsed.family in {"kapitel", "chapter"}:
            level = 3
            confidence = 0.97
            evidence.append("labelled_numbering_grammar")
        if hashes_disagree:
            evidence.append("ocr_levels_disagree_within_style")
        semantic_local_apparatus = canonical_heading_text(block.clean_content).casefold()
        if (
            semantic_local_apparatus in {
                "further reading", "further readings", "bibliographical essay"
            }
            and title_frequency[semantic_local_apparatus] >= 3
        ):
            level = 3
            confidence = 0.96
            evidence.append("repeated_chapter_local_apparatus_heading")
        if block.id in smaller_ids and not parsed:
            evidence.append("repeated_smaller_heading_style")
            if "repeated_chapter_local_apparatus_heading" not in evidence:
                confidence = 0.88
        decisions[block.id] = StyleDecision(level, confidence, evidence)
    return decisions
