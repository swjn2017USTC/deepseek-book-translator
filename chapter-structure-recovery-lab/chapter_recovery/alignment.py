from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .heading_text import rejection_reason
from .models import Block, TocEntry
from .page_map import PageMap
from .toc import title_without_number


MACRO_LABELS = {"doc_title", "paragraph_title", "header"}
TERMINAL_PUNCTUATION = (".", "?", "!", ";", "。", "？", "！", "；")
STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with",
}


def normalized_title(text: str) -> str:
    text = title_without_number(text)
    text = re.sub(r"^(?:\s*#{1,6}\s*)+", "", text.strip())
    text = re.sub(
        r"^(?:part|book|volume|teil)\s+(?:\d+|[ivxlcdm]+)\s*[:.·–—-]*\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[^\w]+", " ", text.casefold(), flags=re.UNICODE)
    normalized = " ".join(text.split())
    return re.sub(r"^list of\s+", "", normalized)


def title_similarity(left: str, right: str) -> float:
    a = normalized_title(left)
    b = normalized_title(right)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    aset = {token for token in a.split() if token not in STOPWORDS}
    bset = {token for token in b.split() if token not in STOPWORDS}
    token = len(aset & bset) / max(1, len(aset | bset))
    overlap = len(aset & bset)
    containment = max(overlap / max(1, len(aset)), overlap / max(1, len(bset)))
    containment_score = containment * 0.94 if overlap >= 2 else 0.0
    return max(sequence, token, containment_score)


@dataclass
class TocAlignment:
    entry: TocEntry
    original_index: int
    sequence_index: int
    target_page: Optional[int]
    block: Optional[Block] = None
    score: float = 0.0
    confidence: float = 0.0
    status: str = "unmatched"
    components: Dict[str, Any] = field(default_factory=dict)


def _short_text_candidate(block: Block) -> bool:
    if block.label != "text":
        return False
    text = block.clean_content
    if not (1 <= len(text) <= 180) or len(text.split()) > 24:
        return False
    if text.endswith(TERMINAL_PUNCTUATION) or rejection_reason(text):
        return False
    _, y1, _, y2 = block.normalized_bbox()
    return y1 >= 0.08 and y2 <= 0.92


def _candidate_blocks(pages: List[List[Block]], toc_pages: set[int]) -> List[Block]:
    candidates: List[Block] = []
    for page_index, blocks in enumerate(pages):
        if page_index in toc_pages:
            continue
        for block in blocks:
            if block.label in MACRO_LABELS or _short_text_candidate(block):
                if not rejection_reason(block.clean_content):
                    candidates.append(block)
    candidates.sort(
        key=lambda block: (
            block.page_index,
            block.order if block.order is not None else block.block_index,
            block.bbox[1],
            block.block_index,
        )
    )
    return candidates


def _target_page(entry: TocEntry, page_map: PageMap) -> Optional[int]:
    if entry.print_page is None:
        return None
    page = page_map.json_page(entry.print_page)
    return page if page is None else max(0, page)


def _label_quality(block: Block) -> float:
    return {
        "doc_title": 1.0,
        "paragraph_title": 0.92,
        "text": 0.62,
        "header": 0.45,
    }.get(block.label, 0.0)


def _kind_quality(entry: TocEntry, block: Block) -> float:
    text = block.clean_content.strip()
    low = text.casefold()
    if entry.kind == "part":
        if re.match(r"^(part|book|volume|teil)\b", low) or (text.isupper() and len(text) >= 3):
            return 1.0
        return 0.65 if block.label == "text" else 0.35
    if entry.kind in {"chapter", "part_intro"}:
        if block.label == "doc_title" or re.match(r"^(?:chapter\s+)?(?:\d+|[ivxlcdm]+)\b", low):
            return 1.0
        return 0.75 if block.label == "paragraph_title" else 0.50
    if entry.kind == "frontmatter":
        return 1.0 if block.page_index < 80 else 0.45
    if entry.kind == "backmatter":
        return 1.0 if low in {"index", "bibliography", "references", "works cited", "notes"} else 0.65
    return 0.75


def _page_quality(target_page: Optional[int], candidate_page: int, window: int) -> Tuple[float, Optional[int]]:
    if target_page is None:
        return 0.50, None
    distance = abs(candidate_page - target_page)
    scale = max(2.0, float(window) + 1.0)
    return math.exp(-distance / scale), distance


def _pair_score(
    entry: TocEntry,
    block: Block,
    target_page: Optional[int],
    window: int,
) -> Optional[Tuple[float, Dict[str, Any]]]:
    similarity = title_similarity(entry.title, block.clean_content)
    if similarity < (0.42 if target_page is not None else 0.62):
        return None
    entry_tokens = {token for token in normalized_title(entry.title).split() if token not in STOPWORDS}
    block_tokens = {token for token in normalized_title(block.clean_content).split() if token not in STOPWORDS}
    overlap_count = len(entry_tokens & block_tokens)
    if similarity < 0.90 and min(len(entry_tokens), len(block_tokens)) >= 2 and overlap_count < 2:
        return None
    page_quality, distance = _page_quality(target_page, block.page_index, window)
    if distance is not None and distance > max(24, window * 8) and similarity < 0.86:
        return None
    label_quality = _label_quality(block)
    kind_quality = _kind_quality(entry, block)
    partial_title_bonus = (
        0.03 if 0.60 <= similarity < 0.90 and overlap_count >= 2
        else 0.0
    )
    score = (
        0.72 * similarity
        + 0.18 * page_quality
        + 0.06 * label_quality
        + 0.04 * kind_quality
        + partial_title_bonus
    )
    structural_title_bonus = 0.0
    if similarity >= 0.94:
        if block.label == "doc_title":
            structural_title_bonus = 0.14
        elif block.label == "paragraph_title":
            structural_title_bonus = 0.05
        elif entry.kind == "part" and block.label == "text" and block.clean_content.isupper():
            # Part title pages are often exported as one large all-caps text
            # block, while an earlier Introduction repeats the same wording as
            # a paragraph_title.  The title-page block is the structural anchor.
            structural_title_bonus = 0.14
    score += structural_title_bonus
    return min(1.0, score), {
        "title_similarity": round(similarity, 4),
        "page_quality": round(page_quality, 4),
        "page_distance": distance,
        "label_quality": round(label_quality, 4),
        "kind_quality": round(kind_quality, 4),
        "structural_title_bonus": round(structural_title_bonus, 4),
        "partial_title_bonus": round(partial_title_bonus, 4),
        "distinctive_token_overlap": overlap_count,
    }


def _provisional_page(
    entry: TocEntry,
    target_page: Optional[int],
    candidates: Sequence[Block],
    window: int,
    page_count: int,
) -> float:
    best: Optional[Tuple[float, Block, Dict[str, Any]]] = None
    best_key: Optional[Tuple[float, ...]] = None
    for block in candidates:
        pair = _pair_score(entry, block, target_page, window)
        if pair is None:
            continue
        score, components = pair
        if target_page is None:
            key = (
                float(components["title_similarity"]),
                score,
                1.0 if float(components.get("partial_title_bonus", 0.0)) > 0 else 0.0,
            )
        else:
            key = (score,)
        if best is None or best_key is None or key > best_key:
            best = (score, block, components)
            best_key = key
    if best:
        similarity = float(best[2]["title_similarity"])
        distinctive_partial = float(best[2].get("partial_title_bonus", 0.0)) > 0 and best[0] >= 0.62
        if similarity >= 0.72 or distinctive_partial:
            return float(best[1].page_index)
    if target_page is not None:
        return float(target_page)
    if entry.kind == "frontmatter":
        return -1.0
    if entry.kind == "backmatter":
        return float(page_count + 1)
    return float(page_count) / 2.0


def align_toc_entries(
    pages: List[List[Block]],
    entries: List[TocEntry],
    page_map: PageMap,
    config: Dict[str, Any],
) -> List[TocAlignment]:
    """Globally align TOC entries to a monotone, one-to-one body-title sequence."""
    if not entries:
        return []
    window = int(config.get("toc_alignment_window", 2))
    minimum = float(config.get("global_alignment_min_score", 0.62))
    toc_pages = {entry.source_page for entry in entries if entry.source_page >= 0}
    candidates = _candidate_blocks(pages, toc_pages)
    candidate_title_frequency = Counter(normalized_title(block.clean_content) for block in candidates)
    targets = [_target_page(entry, page_map) for entry in entries]

    ordered = list(range(len(entries)))
    provisional = [
        _provisional_page(entry, targets[index], candidates, window, len(pages))
        for index, entry in enumerate(entries)
    ]
    ordered.sort(key=lambda index: (provisional[index], index))

    row_count = len(ordered)
    column_count = len(candidates)
    scores: List[List[Optional[Tuple[float, Dict[str, Any]]]]] = []
    for original_index in ordered:
        entry = entries[original_index]
        target = targets[original_index]
        scores.append([_pair_score(entry, block, target, window) for block in candidates])

    dp = [[0.0] * (column_count + 1) for _ in range(row_count + 1)]
    action = [["entry"] * (column_count + 1) for _ in range(row_count + 1)]
    for i in range(1, row_count + 1):
        for j in range(1, column_count + 1):
            best = dp[i - 1][j]
            chosen = "entry"
            if dp[i][j - 1] > best + 1e-12:
                best = dp[i][j - 1]
                chosen = "candidate"
            pair = scores[i - 1][j - 1]
            if pair is not None and pair[0] >= minimum:
                reward = pair[0] - minimum + 0.02
                matched = dp[i - 1][j - 1] + reward
                if matched > best + 1e-12:
                    best = matched
                    chosen = "match"
            dp[i][j] = best
            action[i][j] = chosen

    selected: Dict[int, Tuple[int, float, Dict[str, Any]]] = {}
    i, j = row_count, column_count
    while i > 0 and j > 0:
        chosen = action[i][j]
        if chosen == "match":
            pair = scores[i - 1][j - 1]
            assert pair is not None
            selected[i - 1] = (j - 1, pair[0], pair[1])
            i -= 1
            j -= 1
        elif chosen == "candidate":
            j -= 1
        else:
            i -= 1

    results: List[Optional[TocAlignment]] = [None] * len(entries)
    for sequence_index, original_index in enumerate(ordered):
        entry = entries[original_index]
        target = targets[original_index]
        choice = selected.get(sequence_index)
        if choice is None:
            results[original_index] = TocAlignment(
                entry=entry,
                original_index=original_index,
                sequence_index=sequence_index,
                target_page=target,
                confidence=0.45 if target is not None else 0.30,
                components={
                    "algorithm": "global_monotone_dp_v1",
                    "sequence_index": sequence_index,
                    "provisional_page": round(provisional[original_index], 2),
                    "reason": "no_unique_candidate_above_threshold",
                },
            )
            continue
        candidate_index, score, components = choice
        block = candidates[candidate_index]
        details = dict(components)
        details.update(
            {
                "algorithm": "global_monotone_dp_v1",
                "candidate_index": candidate_index,
                "sequence_index": sequence_index,
                "provisional_page": round(provisional[original_index], 2),
                "order_repaired": sequence_index != original_index,
                "candidate_title_frequency": candidate_title_frequency[
                    normalized_title(block.clean_content)
                ],
            }
        )
        confidence = score
        similarity = float(details["title_similarity"])
        distance = details["page_distance"]
        if similarity >= 0.94 and distance is not None and distance <= window:
            confidence = max(confidence, 0.95)
        elif similarity >= 0.94 and block.label in {"doc_title", "paragraph_title"}:
            confidence = max(confidence, 0.93)
        elif similarity >= 0.94 and block.label == "text":
            confidence = max(confidence, 0.91)
        elif (
            similarity >= 0.94
            and block.label == "header"
            and details["candidate_title_frequency"] == 1
        ):
            confidence = max(confidence, 0.92)
        results[original_index] = TocAlignment(
            entry=entry,
            original_index=original_index,
            sequence_index=sequence_index,
            target_page=target,
            block=block,
            score=score,
            confidence=min(0.99, confidence),
            status="matched",
            components=details,
        )
    return [result for result in results if result is not None]
