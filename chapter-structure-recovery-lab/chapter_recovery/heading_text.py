from __future__ import annotations

import re
from html import unescape
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .models import Block


STACKED_HASHES = re.compile(r"^(?:\s*#{1,6}\s*)+")
DECORATIVE = re.compile(r"^[\s*~◇◆△▲▽▼○●□■·•—–_\-=]+$")
NUMERIC_ONLY = re.compile(r"^(?:p(?:age)?\.?\s*)?[ivxlcdm\d]+$", re.I)
HIERARCHICAL = re.compile(r"^(\d+(?:\.\d+){1,5})(?:[.)])?\s+(.+)$")
SIMPLE_NUMBER = re.compile(r"^(\d+|[IVXLCDM]+)(?:[.)])?\s+(.+)$", re.I)
LETTER_NUMBER = re.compile(r"^([A-Z]|[a-z])(?:[.)])\s+(.+)$")
LABEL_NUMBER = re.compile(
    r"^(part|teil|book|volume|chapter|kapitel)\s+([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b[:.]?\s*(.*)$",
    re.I,
)
SECTION_SYMBOL = re.compile(r"^§\s*([0-9Il|]+)$", re.I)


@dataclass(frozen=True)
class Numbering:
    token: str
    path: Tuple[str, ...]
    depth: int
    family: str


def canonical_heading_text(text: str) -> str:
    """Remove OCR/Markdown wrappers without rewriting the source title."""
    value = STACKED_HASHES.sub("", text.strip())
    value = unescape(re.sub(r"<[^>]+>", " ", value))
    value = re.sub(r"^\$?\s*\\?underline\s*\{([^{}]+)\}\s*\$?", r"\1", value)
    return " ".join(value.replace("\u00a0", " ").split()).strip()


def numbering(text: str) -> Optional[Numbering]:
    value = canonical_heading_text(text)
    match = SECTION_SYMBOL.match(value)
    if match:
        token = match.group(1).translate(str.maketrans({"I": "1", "i": "1", "l": "1", "|": "1"}))
        return Numbering(f"§{token}", ("§", token), 2, "section_symbol")
    match = LABEL_NUMBER.match(value)
    if match:
        family, token = match.group(1).casefold(), match.group(2)
        return Numbering(token, (family, token.casefold()), 1, family)
    match = HIERARCHICAL.match(value)
    if match:
        path = tuple(match.group(1).split("."))
        return Numbering(match.group(1), path, len(path), "decimal")
    match = LETTER_NUMBER.match(value)
    if match:
        token = match.group(1)
        return Numbering(token, (token,), 1, "upper_letter" if token.isupper() else "lower_letter")
    match = SIMPLE_NUMBER.match(value)
    if match:
        return Numbering(match.group(1), (match.group(1).casefold(),), 1, "simple")
    return None


def rejection_reason(text: str) -> Optional[str]:
    value = canonical_heading_text(text)
    if not value:
        return "empty"
    if DECORATIVE.fullmatch(value):
        return "decorative_only"
    if NUMERIC_ONLY.fullmatch(value):
        return "folio_or_numeric_only"
    if len(value) > 180:
        return "too_long"
    words = value.split()
    if len(words) > 24:
        return "sentence_like"
    if value.endswith(("?", "!", "。", "？", "！")) and len(words) > 7:
        return "quotation_or_sentence"
    return None


def should_merge(left: Block, right: Block, toc_titles: Sequence[str] = ()) -> bool:
    if left.label != "paragraph_title" or right.label != "paragraph_title":
        return False
    a, b = canonical_heading_text(left.content), canonical_heading_text(right.content)
    if rejection_reason(a) or rejection_reason(b):
        return False
    joined = f"{a} {b}"
    targets = {canonical_heading_text(title).casefold() for title in toc_titles}
    if left.page_index == right.page_index:
        gap = right.normalized_bbox()[1] - left.normalized_bbox()[3]
        if gap > 0.035:
            return False
    elif right.page_index == left.page_index + 1:
        # A cross-page merge needs both a physical page-boundary signal and a
        # textual continuation signal.  This deliberately rejects merely
        # consecutive headings on adjacent pages.
        if left.normalized_bbox()[3] < 0.80 or right.normalized_bbox()[1] > 0.20:
            return False
        if numbering(a) and numbering(b):
            return False
        if not a.endswith((":", "—", "–", "-")) and joined.casefold() not in targets:
            return False
    else:
        return False
    if a.endswith((":", "—", "–", "-")):
        return True
    return joined.casefold() in targets


def merge_heading_blocks(blocks: Sequence[Block], toc_titles: Sequence[str] = ()) -> List[Tuple[Block, List[str]]]:
    """Return representative blocks plus source fragments; IDs remain auditable."""
    output: List[Tuple[Block, List[str]]] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        fragments = [block.id]
        if index + 1 < len(blocks) and should_merge(block, blocks[index + 1], toc_titles):
            following = blocks[index + 1]
            block = Block(
                page_index=block.page_index,
                block_index=block.block_index,
                raw_block_id=block.raw_block_id,
                label=block.label,
                content=f"{canonical_heading_text(block.content)} {canonical_heading_text(following.content)}",
                bbox=(
                    block.bbox[0],
                    block.bbox[1],
                    max(block.bbox[2], following.bbox[2]),
                    following.bbox[3] if block.page_index == following.page_index else block.bbox[3],
                ),
                order=block.order,
                page_width=block.page_width,
                page_height=block.page_height,
                id=block.id,
            )
            fragments.append(following.id)
            index += 1
        output.append((block, fragments))
        index += 1
    return output
