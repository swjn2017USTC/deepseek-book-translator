from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .alignment import TocAlignment, align_toc_entries, normalized_title, title_similarity
from .models import Block, Node, TocEntry, stable_id
from .heading_text import canonical_heading_text, merge_heading_blocks, numbering, rejection_reason
from .page_map import PageMap
from .style_model import infer_internal_styles
from .toc import title_without_number
from .visual_context import heading_visual_context
from .zones import infer_page_zones, suppress_apparatus_duplicates, zone_for_heading


TERMINAL_PUNCTUATION = (".", "?", "!", ":", ";", "。", "？", "！", "；")
STRUCTURAL_TEXT = re.compile(r"^(?:part|book|volume|chapter|teil|kapitel)\b", re.I)
GERMAN_SECTION = re.compile(r"^(?:ERSTER|ZWEITER|DRITTER)\s+ABSCHNITT\b", re.I)
GERMAN_UPPER_LETTER = re.compile(r"^[A-H]\.(?:\s|$)")
GERMAN_ROMAN = re.compile(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.(?:\s|$)", re.I)
GERMAN_LOWER_LETTER = re.compile(r"^[a-h]\)(?:\s|$)")
ARABIC_TITLE = re.compile(r"^(\d+)(?:[.)])?\s+\S")
ROMAN_TITLE = re.compile(r"^([ivxlcdm]+)(?:[.)])?\s+\S", re.I)
NUMBERED_DATE = re.compile(
    r"^\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    re.I,
)


def _normalized_title(text: str) -> str:
    return normalized_title(text)


def _suppress_book_title_duplicates(
    nodes: List[Node],
    text_candidates: List[Dict[str, Any]],
    book_title: str,
) -> None:
    """Remove OCR headings that merely repeat the configured book title.

    Exact repeats are redundant even on a title page because the book node is
    already the sole H1. Near-repeats are suppressed only when the OCR labelled
    them as a header or when the same form recurs on multiple pages. Every
    decision remains visible in the structure/suppression audit.
    """
    target = _normalized_title(book_title)
    if not target:
        return
    active = [node for node in nodes if node.kind != "book" and not node.ignored]
    repeated_forms: Dict[str, Set[int]] = {}
    for node in active:
        key = _normalized_title(node.title_source)
        if key and title_similarity(key, target) >= 0.86 and node.page_json is not None:
            repeated_forms.setdefault(key, set()).add(node.page_json)

    redirected: Dict[str, Optional[str]] = {}
    for node in active:
        key = _normalized_title(node.title_source)
        exact = bool(key and key == target)
        recurring_near_match = bool(
            key
            and title_similarity(key, target) >= 0.86
            and len(repeated_forms.get(key, set())) >= 2
        )
        header_near_match = bool(
            key and title_similarity(key, target) >= 0.86 and node.source == "header"
        )
        if not (exact or recurring_near_match or header_near_match):
            continue
        node.ignored = True
        node.evidence.append("book_title_running_header_suppressed")
        if exact:
            node.evidence.append("exact_book_title_duplicate")
        elif recurring_near_match:
            node.evidence.append("repeated_near_book_title_duplicate")
        else:
            node.evidence.append("header_near_book_title_duplicate")
        redirected[node.id] = node.parent_id

    for node in nodes:
        if node.parent_id in redirected:
            node.parent_id = redirected[node.parent_id]
            node.evidence.append("parent_redirected_after_book_title_suppression")

    candidate_forms: Dict[str, Set[int]] = {}
    for item in text_candidates:
        key = _normalized_title(str(item.get("source_text") or ""))
        page = item.get("page_json")
        if key and isinstance(page, int) and title_similarity(key, target) >= 0.86:
            candidate_forms.setdefault(key, set()).add(page)
    for item in text_candidates:
        key = _normalized_title(str(item.get("source_text") or ""))
        if not key:
            continue
        if key == target or (
            title_similarity(key, target) >= 0.86
            and len(candidate_forms.get(key, set())) >= 2
        ):
            item["suppressed"] = True
            item["suppression_reason"] = "book_title_running_header"
            evidence = item.setdefault("evidence", [])
            if "book_title_running_header" not in evidence:
                evidence.append("book_title_running_header")


def _normalize_standard_hierarchy(
    nodes: List[Node], book_node_id: str, chapters_under_parts: bool
) -> None:
    """Convert ordinary book semantics into stable H1-H6 levels.

    Specialized Hegel/numbering profiles run their own state machines and do
    not use this pass. This pass only repairs objectively impossible/evidently
    macro levels; deeper style-model decisions are preserved.
    """
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        if node.ignored:
            continue
        old_level = node.level
        if node.kind == "book":
            node.level = 1
        elif node.kind in {"part", "frontmatter", "backmatter"}:
            node.level = 2
        elif node.kind == "chapter":
            parent = by_id.get(node.parent_id or "")
            node.level = 3 if chapters_under_parts and parent and parent.kind == "part" else 2
        elif node.kind == "part_intro":
            node.level = 3
        elif node.kind == "section":
            parent = by_id.get(node.parent_id or "")
            minimum = 3 if parent is None or parent.id == book_node_id else min(6, parent.level + 1)
            node.level = max(minimum, min(6, node.level))
        else:
            node.level = max(2, min(6, node.level))
        if node.level != old_level:
            node.evidence.append(f"standard_hierarchy_level_repaired:{old_level}->{node.level}")


def _macro_level(entry: TocEntry) -> int:
    parsed = numbering(entry.title)
    if entry.kind == "section" and parsed and parsed.family == "decimal":
        return min(6, 2 + parsed.depth)
    if entry.kind in {"section", "part_intro"}:
        return 3
    return 2


def _trim_appended_byline(entry: TocEntry, block: Block) -> Tuple[str, bool]:
    """Keep an exact OCR title prefix when the same block ends in an all-caps byline."""
    block_title = block.clean_content
    entry_title = canonical_heading_text(entry.title)
    if not entry_title or not block_title.casefold().startswith(entry_title.casefold()):
        return block_title, False
    if len(block_title) == len(entry_title):
        return block_title, False
    if not block_title[len(entry_title)].isspace():
        return block_title, False
    tail = block_title[len(entry_title):].strip()
    letters = [character for character in tail if character.isalpha()]
    words = tail.split()
    if (
        2 <= len(words) <= 24
        and letters
        and tail == tail.upper()
        and not any(character.isdigit() for character in tail)
    ):
        return block_title[:len(entry_title)].rstrip(), True
    return block_title, False


def _node_from_alignment(
    book_id: str,
    alignment: TocAlignment,
    book_node_id: str,
) -> Node:
    entry = alignment.entry
    block = alignment.block
    matched = alignment.status == "matched" and block is not None
    page_json = block.page_index if matched else alignment.target_page
    evidence = ["toc", entry.provenance, "global_sequence_alignment"]
    if entry.print_page is not None:
        evidence.append("page_map")
    if matched:
        evidence.append(block.label)
        evidence.append("unique_block_assignment")
    else:
        evidence.append("global_alignment_unmatched")
    if alignment.components.get("order_repaired"):
        evidence.append("heterogeneous_toc_order_repaired")
    confidence = alignment.confidence
    source = "toc_unmatched"
    title = title_without_number(entry.title)
    similarity = float(alignment.components.get("title_similarity", 0.0))
    block_numbering = numbering(block.clean_content) if matched else None
    part_fragment_only = (
        entry.kind == "part"
        and matched
        and (not block_numbering or block_numbering.family not in {"part", "teil", "book", "volume"})
    )
    if matched and similarity >= 0.58 and not part_fragment_only:
        source = block.label
        title, removed_appended_byline = _trim_appended_byline(entry, block)
        if removed_appended_byline:
            evidence.append("appended_byline_trimmed_to_toc_prefix")
    elif matched:
        source = "toc_aligned"

    return Node(
        id="node-" + stable_id(book_id, entry.id, page_json, title),
        kind=entry.kind,
        level=_macro_level(entry),
        title_source=title,
        page_json=page_json,
        page_print=entry.print_page,
        parent_id=book_node_id,
        block_id=block.id if matched else None,
        source=source,
        confidence=confidence,
        evidence=evidence,
        bbox=block.bbox if matched else None,
        source_level=block.source_level if matched else None,
        zone="frontmatter" if entry.kind == "frontmatter" else ("backmatter" if entry.kind == "backmatter" else "mainmatter"),
        ordinal=(numbering(entry.title).token if numbering(entry.title) else None),
        numbering_path=(list(numbering(entry.title).path) if numbering(entry.title) else []),
        source_block_ids=[block.id] if matched else [],
        toc_entry_id=entry.id,
        alignment_status=alignment.status,
        alignment_score=round(alignment.score, 4),
        alignment_details=alignment.components,
    )


def _chapter_parent(nodes: Sequence[Node], page_index: int) -> Optional[str]:
    parents = [
        node
        for node in nodes
        if node.kind in {"chapter", "part"}
        and node.page_json is not None
        and node.page_json <= page_index
        and not node.ignored
    ]
    if not parents:
        return None
    parents.sort(key=lambda node: (node.page_json or -1, node.kind == "chapter"))
    return parents[-1].id


def _internal_confidence(block: Block) -> Tuple[float, List[str]]:
    x1, y1, x2, y2 = block.normalized_bbox()
    height = y2 - y1
    width = x2 - x1
    parsed = numbering(block.clean_content)
    if block.label == "text" and parsed and parsed.family == "section_symbol":
        return 0.99, ["text", "explicit_section_symbol"]
    evidence = [block.label]
    score = 0.86
    if height <= 0.07:
        score += 0.04
        evidence.append("single_or_short_line")
    if width <= 0.82:
        score += 0.02
        evidence.append("narrower_than_body")
    if block.source_level:
        evidence.append("ocr_markdown_level_weak")
    return min(0.96, score), evidence


def _text_candidates(
    pages: List[List[Block]],
    used_blocks: Set[str],
    body_range: Tuple[int, int],
    zones: Dict[int, str],
    nodes: Sequence[Node],
    profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    output = []
    start, end = body_range
    titles_by_page: Dict[int, List[str]] = {}
    for node in nodes:
        if node.page_json is not None and not node.ignored and node.kind != "book":
            titles_by_page.setdefault(node.page_json, []).append(node.title_source)
    for page_index in range(max(0, start), min(len(pages), end + 1)):
        if zones.get(page_index, "mainmatter") != "mainmatter":
            continue
        blocks = pages[page_index]
        for position, block in enumerate(blocks):
            if block.id in used_blocks or block.label != "text":
                continue
            text = " ".join(block.clean_content.split())
            if not (2 <= len(text) <= 100) or text.endswith(TERMINAL_PUNCTUATION):
                continue
            words = text.split()
            if not (1 <= len(words) <= 14):
                continue
            x1, y1, x2, y2 = block.normalized_bbox()
            height = y2 - y1
            width = x2 - x1
            if height > 0.065 or width > 0.82:
                continue
            above_gap = 0.0
            below_gap = 0.0
            if position > 0:
                above_gap = max(0.0, y1 - blocks[position - 1].normalized_bbox()[3])
            if position + 1 < len(blocks):
                below_gap = max(0.0, blocks[position + 1].normalized_bbox()[1] - y2)
            layout_score = min(1.0, (above_gap + below_gap) / 0.06)
            if layout_score < 0.35:
                continue
            item = {
                "candidate_id": "candidate-" + stable_id(block.id, "text-heading"),
                "block_id": block.id,
                "page_json": page_index,
                "source_text": text,
                "suggested_level": 3,
                "confidence": round(0.45 + 0.25 * layout_score, 4),
                "evidence": ["short_text", "surrounding_whitespace"],
                "bbox": list(block.bbox),
                "allowed_decisions": [
                    "accept", "reject", "change_level", "change_kind",
                    "merge_with_previous", "needs_human",
                ],
            }
            suppression_reason = None
            if any(
                title_similarity(text, title) >= 0.86
                for title in titles_by_page.get(page_index, [])
            ):
                suppression_reason = "same_page_active_title_duplicate"
            elif re.match(r"^\$?\s*\^\{?\d+", text):
                suppression_reason = "explicit_footnote_marker"
            elif _is_chapter_byline(text, block, nodes):
                suppression_reason = "chapter_title_page_byline"
            elif _is_epigraph_attribution(text):
                suppression_reason = "epigraph_attribution"
            elif _is_visual_legend_line(block, blocks):
                suppression_reason = "visual_legend_line"
            elif profile == "continuous_prose_monograph":
                suppression_reason = _continuous_prose_suppression(
                    text, block, blocks, position
                )
            if suppression_reason:
                item["suppressed"] = True
                item["suppression_reason"] = suppression_reason
                item["evidence"].append(suppression_reason)
            output.append(item)
    return output


def _continuous_prose_suppression(
    text: str, block: Block, page_blocks: Sequence[Block], position: int
) -> Optional[str]:
    """Suppress prose/quotation fragments for editions with isolated text lines.

    This profile is opt-in because some book families use short unnumbered text
    as genuine headings. Explicit structural labels and numbering are never
    suppressed here.
    """
    if STRUCTURAL_TEXT.match(text) or numbering(text):
        return None
    words = text.split()
    _, y1, _, _ = block.normalized_bbox()
    previous = page_blocks[:position]
    following = page_blocks[position + 1:]
    previous_body = any(
        item.label in {"text", "abstract"} and len(item.clean_content) >= 140
        for item in previous[-3:]
    )
    following_body = any(
        item.label in {"text", "abstract"} and len(item.clean_content) >= 140
        for item in following[:3]
    )
    if y1 >= 0.70 and len(words) >= 7 and not following_body and (
        previous_body or (y1 >= 0.84 and ". " in text)
    ):
        return "page_bottom_body_continuation"
    stripped = text.rstrip("'\"”’ ")
    if y1 <= 0.15 and len(words) >= 8 and (
        text[:1].islower() or stripped.endswith(TERMINAL_PUNCTUATION)
    ):
        return "page_top_body_continuation"
    if stripped.endswith(TERMINAL_PUNCTUATION) and len(words) >= 7:
        return "body_sentence_or_quotation"
    if re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        text,
        re.I,
    ):
        return "letter_dateline"
    if re.match(r"^(?:Your loving|Yours(?: sincerely| faithfully)?\b)", text, re.I):
        return "letter_closing"

    nearby_short_lines = [
        item
        for item in page_blocks
        if item.label == "text"
        and item.id != block.id
        and len(item.clean_content) <= 120
        and abs(item.normalized_bbox()[1] - y1) <= 0.16
    ]
    if len(words) <= 12 and len(nearby_short_lines) >= 2:
        return "verse_or_quotation_cluster"
    return None


def _recover_structural_text_headings(
    pages: List[List[Block]],
    nodes: List[Node],
    used_blocks: Set[str],
    zones: Dict[int, str],
    body_range: Tuple[int, int],
    page_map: PageMap,
    book_id: str,
    book_node_id: str,
    profile: Optional[str],
) -> None:
    if profile != "continuous_prose_monograph":
        return
    start, end = body_range
    for page_index in range(max(0, start), min(len(pages), end + 1)):
        if zones.get(page_index, "mainmatter") != "mainmatter":
            continue
        for block in pages[page_index]:
            text = canonical_heading_text(block.clean_content)
            parsed = numbering(text)
            if (
                block.id in used_blocks
                or block.label != "text"
                or not STRUCTURAL_TEXT.match(text)
                or parsed is None
                or parsed.family not in {"part", "book", "volume", "chapter", "kapitel"}
                or len(text) > 140
            ):
                continue
            kind = "chapter" if parsed.family in {"chapter", "kapitel"} else "part"
            nodes.append(Node(
                id="node-" + stable_id(book_id, block.id, "structural-text-heading"),
                kind=kind,
                level=2,
                title_source=text,
                page_json=page_index,
                page_print=(page_index + page_map.offset) if page_map.offset is not None else None,
                parent_id=book_node_id,
                block_id=block.id,
                source="text",
                confidence=0.99,
                evidence=["text", "explicit_structural_text", "profile:continuous_prose_monograph"],
                bbox=block.bbox,
                source_level=block.source_level,
                zone="mainmatter",
                ordinal=parsed.token,
                numbering_path=list(parsed.path),
                source_block_ids=[block.id],
            ))
            used_blocks.add(block.id)


def _is_chapter_byline(text: str, block: Block, nodes: Sequence[Node]) -> bool:
    if STRUCTURAL_TEXT.match(text) or any(character.isdigit() for character in text):
        return False
    chapter_boxes = [
        node.bbox
        for node in nodes
        if node.kind == "chapter"
        and node.page_json == block.page_index
        and not node.ignored
        and node.bbox is not None
    ]
    if not chapter_boxes:
        return False
    _, y1, _, _ = block.normalized_bbox()
    below_chapter = any(
        y1 >= float(box[3]) / block.page_height
        and y1 - float(box[3]) / block.page_height <= 0.10
        for box in chapter_boxes
    )
    if not below_chapter or y1 > 0.42:
        return False
    words = text.split()
    letters = [character for character in text if character.isalpha()]
    if (
        4 <= len(words) <= 24
        and letters
        and text == text.upper()
        and "," in text
        and "and" in {word.strip(".,;:()[]").casefold() for word in words}
    ):
        return True
    if 2 <= len(words) <= 7 and letters and text == text.upper():
        return True
    low = text.casefold()
    if "with assistance from" in low or "translation by" in low:
        return True
    if not (3 <= len(words) <= 7 and "and" in low.split()):
        return False
    connectors = {"and", "de", "del", "van", "von"}
    name_words = [word.strip(".,;:()[]") for word in words]
    return all(
        word.casefold() in connectors
        or (word and word[0].isupper())
        for word in name_words
    )


def _is_epigraph_attribution(text: str) -> bool:
    value = text.strip()
    if re.search(r"\$\s*\^\{?\d+\}?\s*\$|[⁰¹²³⁴⁵⁶⁷⁸⁹]+$", value):
        return True
    if any(character.isdigit() for character in value):
        return False
    if "," in value and len(value.split()) <= 10:
        prefix = value.split(",", 1)[0]
        if ":" in prefix:
            return False
        prefix_words = [word.strip(".()[]") for word in prefix.split()]
        if 1 <= len(prefix_words) <= 4 and all(
            word and (word[0].isupper() or len(word) <= 2)
            for word in prefix_words
        ):
            return True
    words = value.split()
    if 1 <= len(words) <= 3:
        latin_words = [word.strip(".,;:()[]") for word in words if any(ch.isalpha() and ch.isascii() for ch in word)]
        if latin_words and all(word[0].isupper() for word in latin_words if word):
            return True
    return False


def _is_visual_legend_line(block: Block, page_blocks: Sequence[Block]) -> bool:
    """Recognize one line in a multi-line legend between an image and caption.

    The test is deliberately conjunctive: punctuation alone is never enough.
    This keeps ordinary colon-bearing section titles eligible for review.
    """
    text = " ".join(block.clean_content.split())
    if block.label != "text" or ":" not in text or "," not in text:
        return False
    _, y1, _, y2 = block.bbox
    preceding_images = [
        item
        for item in page_blocks
        if item.label in {"image", "chart"}
        and item.bbox[3] <= y1
        and y1 - item.bbox[3] <= block.page_height * 0.16
    ]
    following_captions = [
        item
        for item in page_blocks
        if item.label in {"figure_title", "image_caption"}
        and item.bbox[1] >= y2
        and item.bbox[1] - y2 <= block.page_height * 0.08
    ]
    if not preceding_images or not following_captions:
        return False
    image = max(preceding_images, key=lambda item: item.bbox[3])
    caption = min(following_captions, key=lambda item: item.bbox[1])
    legend_lines = [
        item
        for item in page_blocks
        if item.label == "text"
        and image.bbox[3] <= item.bbox[1]
        and item.bbox[3] <= caption.bbox[1]
        and ":" in item.clean_content
        and "," in item.clean_content
        and len(item.clean_content) <= 120
        and abs(item.bbox[0] - block.bbox[0]) <= block.page_width * 0.05
    ]
    return len(legend_lines) >= 3


def _recover_part_label_fragments(
    pages: List[List[Block]], nodes: List[Node], used_blocks: Set[str]
) -> None:
    for part in [node for node in nodes if node.kind == "part" and node.page_json is not None and node.bbox and not node.ignored]:
        page_blocks = pages[int(part.page_json)]
        part_top = float(part.bbox[1]) / max(1.0, page_blocks[0].page_height if page_blocks else 1.0)
        for block in page_blocks:
            parsed = numbering(block.clean_content)
            if (
                block.label != "text"
                or block.id in used_blocks
                or parsed is None
                or parsed.family not in {"part", "teil", "book", "volume"}
                or block.normalized_bbox()[3] > part_top
                or part_top - block.normalized_bbox()[3] > 0.10
            ):
                continue
            if part.ordinal and parsed.token.casefold() != part.ordinal.casefold():
                continue
            if not re.match(r"^(?:part|teil|book|volume)\b", part.title_source, re.I):
                part.title_source = f"{block.clean_content} {part.title_source}".strip()
                part.evidence.append("part_label_fragment_merged")
            else:
                part.evidence.append("part_label_duplicate_consumed")
            part.source_block_ids.append(block.id)
            used_blocks.add(block.id)


def _recover_chapter_title_page_structure(
    pages: List[List[Block]],
    nodes: List[Node],
    used_blocks: Set[str],
    page_map: PageMap,
    book_id: str,
) -> None:
    """Use title/byline/body order to distinguish subtitles from first sections."""
    chapters = [
        node
        for node in nodes
        if node.kind == "chapter"
        and node.page_json is not None
        and node.bbox is not None
        and not node.ignored
    ]
    for chapter in chapters:
        page_index = int(chapter.page_json)
        page_blocks = pages[page_index]
        chapter_bottom = float(chapter.bbox[3]) / max(1.0, page_blocks[0].page_height if page_blocks else 1.0)
        text_blocks = [
            block
            for block in page_blocks
            if block.label == "text"
            and block.id not in used_blocks
            and chapter_bottom <= block.normalized_bbox()[1] <= 0.40
            and 2 <= len(block.clean_content) <= 160
            and len(block.clean_content.split()) <= 18
            and not block.clean_content.endswith(TERMINAL_PUNCTUATION)
        ]
        bylines = [block for block in text_blocks if _is_chapter_byline(block.clean_content, block, nodes)]
        if bylines:
            first_byline = min(bylines, key=lambda block: block.normalized_bbox()[1])
            byline_top = first_byline.normalized_bbox()[1]
            subtitle_blocks = [
                block
                for block in text_blocks
                if block.normalized_bbox()[3] <= byline_top
                and not _is_chapter_byline(block.clean_content, block, nodes)
            ]
            for block in sorted(subtitle_blocks, key=lambda item: item.normalized_bbox()[1]):
                if title_similarity(chapter.title_source, block.clean_content) >= 0.86:
                    continue
                chapter.title_source = f"{chapter.title_source} {block.clean_content}".strip()
                chapter.source_block_ids.append(block.id)
                chapter.evidence.append("chapter_title_page_subtitle_merged")
                used_blocks.add(block.id)

            byline_bottom = max(block.normalized_bbox()[3] for block in bylines)
            section_blocks = [
                block for block in text_blocks
                if block.normalized_bbox()[1] > byline_bottom
            ]
        else:
            embedded_author = re.search(
                r"\b[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ.'’-]+(?:\s+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ.'’-]+){1,4}$",
                chapter.title_source,
            )
            section_blocks = text_blocks if embedded_author else []

        for block in section_blocks:
            later_body = any(
                following.label in {"abstract", "text"}
                and following.normalized_bbox()[1] > block.normalized_bbox()[3]
                and len(following.clean_content.split()) > 18
                for following in page_blocks
            )
            if (
                not later_body
                or len(block.clean_content.split()) > 10
                or _is_epigraph_attribution(block.clean_content)
            ):
                continue
            nodes.append(
                Node(
                    id="node-" + stable_id(book_id, block.id, "title-page-section"),
                    kind="section",
                    level=3,
                    title_source=block.clean_content,
                    page_json=page_index,
                    page_print=(page_index + page_map.offset)
                    if page_map.offset is not None
                    else None,
                    parent_id=chapter.id,
                    block_id=block.id,
                    source="text",
                    confidence=0.95,
                    evidence=["chapter_title_page", "after_byline", "before_body"],
                    bbox=block.bbox,
                    source_level=block.source_level,
                    zone="mainmatter",
                    source_block_ids=[block.id],
                )
            )
            used_blocks.add(block.id)


def _promote_text_candidates_below_doc_title(
    candidates: List[Dict[str, Any]],
    nodes: List[Node],
    page_map: PageMap,
    book_id: str,
) -> List[Dict[str, Any]]:
    remaining: List[Dict[str, Any]] = []
    for item in candidates:
        if item.get("suppressed"):
            remaining.append(item)
            continue
        page_index = int(item["page_json"])
        y1 = float(item["bbox"][1])
        parents = [
            node
            for node in nodes
            if node.page_json == page_index
            and node.kind == "section"
            and node.source == "doc_title"
            and node.bbox is not None
            and float(node.bbox[3]) <= y1
            and y1 - float(node.bbox[3]) <= 60.0
            and not node.ignored
        ]
        if not parents or _is_epigraph_attribution(str(item["source_text"])):
            remaining.append(item)
            continue
        parent = max(parents, key=lambda node: float(node.bbox[3]) if node.bbox else -1.0)
        nodes.append(
            Node(
                id="node-" + stable_id(book_id, item["block_id"], "doc-title-child"),
                kind="section",
                level=min(6, parent.level + 1),
                title_source=str(item["source_text"]),
                page_json=page_index,
                page_print=(page_index + page_map.offset)
                if page_map.offset is not None
                else None,
                parent_id=parent.id,
                block_id=str(item["block_id"]),
                source="text",
                confidence=0.95,
                evidence=["immediately_below_internal_doc_title", "short_text"],
                bbox=tuple(float(value) for value in item["bbox"]),
                zone="mainmatter",
                source_block_ids=[str(item["block_id"])],
            )
        )
    return remaining


def _recover_text_children_below_doc_titles(
    pages: List[List[Block]],
    nodes: List[Node],
    used_blocks: Set[str],
    page_map: PageMap,
    book_id: str,
) -> None:
    """Recover a short subheading directly below an internal doc_title.

    Raw same-page coordinates are used here instead of the corpus-wide
    normalized canvas, which can be distorted by a single malformed OCR page.
    """
    parents = [
        node
        for node in nodes
        if node.kind == "section"
        and node.source == "doc_title"
        and node.page_json is not None
        and node.bbox is not None
        and not node.ignored
    ]
    for parent in parents:
        page_index = int(parent.page_json)
        page_blocks = pages[page_index]
        following = sorted(
            [
                block
                for block in page_blocks
                if block.label == "text"
                and block.id not in used_blocks
                and block.bbox[1] >= float(parent.bbox[3])
            ],
            key=lambda block: (block.bbox[1], block.bbox[0]),
        )
        if not following:
            continue
        child = following[0]
        text = " ".join(child.clean_content.split())
        has_body_after = any(
            block.label in {"abstract", "text"}
            and block.bbox[1] > child.bbox[3]
            and len(block.clean_content.split()) > 18
            for block in page_blocks
        )
        if (
            child.bbox[1] - float(parent.bbox[3]) > 60.0
            or not (2 <= len(text) <= 120)
            or len(text.split()) > 10
            or text.endswith(TERMINAL_PUNCTUATION)
            or _is_epigraph_attribution(text)
            or not has_body_after
        ):
            continue
        nodes.append(
            Node(
                id="node-" + stable_id(book_id, child.id, "doc-title-child"),
                kind="section",
                level=min(6, parent.level + 1),
                title_source=text,
                page_json=page_index,
                page_print=(page_index + page_map.offset)
                if page_map.offset is not None
                else None,
                parent_id=parent.id,
                block_id=child.id,
                source="text",
                confidence=0.95,
                evidence=[
                    "immediately_below_internal_doc_title",
                    "same_page_raw_coordinates",
                    "body_text_follows",
                ],
                bbox=child.bbox,
                zone="mainmatter",
                source_block_ids=[child.id],
            )
        )
        used_blocks.add(child.id)


def apply_overrides(
    nodes: List[Node],
    override_path: Optional[Path],
    text_candidates: Optional[List[Dict[str, Any]]] = None,
    page_map: Optional[PageMap] = None,
    book_id: Optional[str] = None,
    book_node_id: Optional[str] = None,
) -> None:
    if not override_path or not override_path.exists():
        return
    data = json.loads(override_path.read_text(encoding="utf-8"))
    overrides = data.get("overrides") if isinstance(data, dict) else data
    if not isinstance(overrides, list):
        raise ValueError("Overrides must be a JSON list or compiled decision bundle")
    for override in overrides:
        operation = override.get("operation")
        if operation:
            if operation in {"promote_candidate", "reject_candidate", "merge_candidate"}:
                if text_candidates is None or page_map is None or not book_id or not book_node_id:
                    raise ValueError("Compiled candidate overrides require live recovery context")
                _apply_compiled_candidate_override(
                    nodes,
                    text_candidates,
                    override,
                    page_map,
                    book_id,
                    book_node_id,
                )
            else:
                _apply_compiled_override(nodes, override)
            continue
        match = override.get("match", {})
        updates = override.get("set", {})
        for node in nodes:
            if any(getattr(node, key, None) != value for key, value in match.items()):
                continue
            for key, value in updates.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            if updates:
                node.evidence.append("manual_override")
                node.confidence = 1.0


def _apply_compiled_override(nodes: List[Node], override: Dict[str, Any]) -> None:
    operation = override.get("operation")
    by_id = {node.id: node for node in nodes}
    review = override.get("review") or {}
    reason = str(review.get("reason_code") or "reviewed")
    evidence = f"review_decision:{reason}"

    if operation in {"accept_node", "reject_node", "update_node"}:
        node_id = override.get("node_id")
        node = by_id.get(node_id)
        if node is None:
            raise ValueError(f"Compiled override references an unknown node: {node_id}")
        if operation == "accept_node":
            node.confidence = 1.0
        elif operation == "reject_node":
            node.ignored = True
            for child in nodes:
                if child.parent_id == node.id:
                    child.parent_id = node.parent_id
                    child.evidence.append("parent_redirected_after_review_rejection")
        else:
            updates = override.get("set") or {}
            if not isinstance(updates, dict) or set(updates) - {"level", "kind"}:
                raise ValueError("Compiled update_node may only set level or kind")
            for key, value in updates.items():
                setattr(node, key, value)
            node.confidence = 1.0
        node.evidence.extend(["compiled_review_override", evidence])
        return

    if operation == "merge_nodes":
        source = by_id.get(override.get("source_id"))
        target = by_id.get(override.get("target_id"))
        if source is None or target is None or source.id == target.id:
            raise ValueError("Compiled merge_nodes requires two distinct existing nodes")
        target.title_source = " ".join(
            value for value in (target.title_source.strip(), source.title_source.strip()) if value
        )
        target.source_block_ids = list(dict.fromkeys([
            *target.source_block_ids,
            *source.source_block_ids,
            *([target.block_id] if target.block_id else []),
            *([source.block_id] if source.block_id else []),
        ]))
        if target.page_json == source.page_json and target.bbox and source.bbox:
            target.bbox = (
                min(target.bbox[0], source.bbox[0]),
                min(target.bbox[1], source.bbox[1]),
                max(target.bbox[2], source.bbox[2]),
                max(target.bbox[3], source.bbox[3]),
            )
        target.confidence = 1.0
        target.evidence.extend(["compiled_review_merge", evidence])
        source.ignored = True
        source.evidence.extend(["merged_by_review_decision", evidence])
        for child in nodes:
            if child.parent_id == source.id:
                child.parent_id = target.id
                child.evidence.append("parent_redirected_after_review_merge")
        return

    raise ValueError(f"Unknown compiled override operation: {operation}")


def _candidate_parent(
    nodes: Sequence[Node], page_json: int, bbox: Sequence[float], level: int, book_node_id: str
) -> str:
    y = float(bbox[1])
    preceding = [
        node
        for node in nodes
        if not node.ignored
        and node.page_json is not None
        and node.kind not in {"book", "toc_entry"}
        and node.level < level
        and (
            node.page_json < page_json
            or (
                node.page_json == page_json
                and (node.bbox is None or float(node.bbox[1]) <= y)
            )
        )
    ]
    if not preceding:
        return book_node_id
    parent = max(
        preceding,
        key=lambda node: (
            node.page_json if node.page_json is not None else -1,
            float(node.bbox[1]) if node.bbox else -1.0,
            node.level,
        ),
    )
    return parent.id


def _apply_compiled_candidate_override(
    nodes: List[Node],
    text_candidates: List[Dict[str, Any]],
    override: Dict[str, Any],
    page_map: PageMap,
    book_id: str,
    book_node_id: str,
) -> None:
    """Apply a reviewed text candidate only when its live source still matches."""
    operation = override.get("operation")
    candidate_id = override.get("candidate_id")
    candidate = next(
        (item for item in text_candidates if item.get("candidate_id") == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Compiled override references an unknown candidate: {candidate_id}")
    for key in ("block_id", "page_json", "source_text", "bbox", "suggested_level"):
        if candidate.get(key) != override.get(key):
            raise ValueError(
                f"Compiled candidate override is stale or altered at {key}: {candidate_id}"
            )
    review = override.get("review") or {}
    reason = str(review.get("reason_code") or "reviewed")
    review_evidence = f"review_decision:{reason}"

    if operation == "reject_candidate":
        candidate["suppressed"] = True
        candidate["suppression_reason"] = "review_rejected_heading"
        candidate.setdefault("evidence", []).extend(
            ["compiled_review_rejection", review_evidence]
        )
        return

    if operation == "merge_candidate":
        target = next(
            (
                node
                for node in nodes
                if node.id == override.get("target_id") and not node.ignored
            ),
            None,
        )
        if target is None:
            raise ValueError("Compiled merge_candidate references an unknown active target")
        candidate_page = int(candidate["page_json"])
        candidate_bbox = tuple(float(value) for value in candidate["bbox"])
        if target.page_json is None or target.page_json > candidate_page or (
            target.page_json == candidate_page
            and target.bbox is not None
            and float(target.bbox[1]) > candidate_bbox[1]
        ):
            raise ValueError("merge_with_previous target must precede the candidate")
        target.title_source = " ".join(
            value
            for value in (target.title_source.strip(), candidate["source_text"].strip())
            if value
        )
        target.source_block_ids = list(
            dict.fromkeys([
                *target.source_block_ids,
                *([target.block_id] if target.block_id else []),
                candidate["block_id"],
            ])
        )
        if target.page_json == candidate_page and target.bbox:
            target.bbox = (
                min(target.bbox[0], candidate_bbox[0]),
                min(target.bbox[1], candidate_bbox[1]),
                max(target.bbox[2], candidate_bbox[2]),
                max(target.bbox[3], candidate_bbox[3]),
            )
        target.confidence = 1.0
        target.evidence.extend(["compiled_review_candidate_merge", review_evidence])
        text_candidates.remove(candidate)
        return

    if operation != "promote_candidate":
        raise ValueError(f"Unknown compiled candidate override operation: {operation}")
    level = override.get("level")
    kind = override.get("kind")
    allowed_kinds = {
        "frontmatter", "part", "part_intro", "chapter", "section", "backmatter"
    }
    if not isinstance(level, int) or isinstance(level, bool) or not 2 <= level <= 6:
        raise ValueError("Compiled promote_candidate has an invalid level")
    if kind not in allowed_kinds:
        raise ValueError("Compiled promote_candidate has an invalid kind")
    page_json = int(candidate["page_json"])
    bbox = tuple(float(value) for value in candidate["bbox"])
    node_id = "node-" + stable_id(book_id, candidate["block_id"], "review-promoted-heading")
    if any(node.id == node_id for node in nodes):
        raise ValueError(f"Compiled candidate promotion would duplicate node: {node_id}")
    parent_id = _candidate_parent(nodes, page_json, bbox, level, book_node_id)
    parsed = numbering(candidate["source_text"])
    nodes.append(
        Node(
            id=node_id,
            kind=kind,
            level=level,
            title_source=candidate["source_text"],
            page_json=page_json,
            page_print=(page_json + page_map.offset) if page_map.offset is not None else None,
            parent_id=parent_id,
            block_id=candidate["block_id"],
            source="text",
            confidence=1.0,
            evidence=[
                *candidate.get("evidence", []),
                "compiled_review_candidate_promotion",
                review_evidence,
            ],
            bbox=bbox,
            zone="mainmatter",
            ordinal=parsed.token if parsed else None,
            numbering_path=list(parsed.path) if parsed else [],
            source_block_ids=[candidate["block_id"]],
        )
    )
    text_candidates.remove(candidate)


def recover_structure(
    pages: List[List[Block]],
    toc_entries: List[TocEntry],
    page_map: PageMap,
    config: Dict[str, Any],
    override_path: Optional[Path] = None,
) -> Tuple[List[Node], List[Dict[str, Any]]]:
    book_id = str(config["book_id"])
    book_node_id = "book-" + stable_id(book_id)
    nodes = [
        Node(
            id=book_node_id,
            kind="book",
            level=1,
            title_source=str(config["book_title"]),
            page_json=None,
            page_print=None,
            parent_id=None,
            block_id=None,
            source="config",
            confidence=1.0,
            evidence=["config"],
        )
    ]
    used_blocks: Set[str] = set()
    for alignment in align_toc_entries(pages, toc_entries, page_map, config):
        node = _node_from_alignment(book_id, alignment, book_node_id)
        nodes.append(node)
        if node.block_id:
            used_blocks.add(node.block_id)

    chapters_under_parts = bool(config.get("chapters_under_parts", False))
    _repair_numbered_parents(nodes, book_node_id, chapters_under_parts)

    macro_pages = [
        node.page_json
        for node in nodes
        if node.page_json is not None and node.kind in {"chapter", "part", "backmatter"}
    ]
    body_start = min(macro_pages) if macro_pages else 0
    backmatter_pages = [
        node.page_json
        for node in nodes
        if node.page_json is not None and node.kind == "backmatter"
    ]
    body_end = min(backmatter_pages) - 1 if backmatter_pages else len(pages) - 1
    zones = infer_page_zones(pages, body_start)

    _recover_part_label_fragments(pages, nodes, used_blocks)
    _recover_chapter_title_page_structure(
        pages, nodes, used_blocks, page_map, book_id
    )

    for node in nodes:
        if node.page_json is not None and node.kind not in {"frontmatter", "backmatter"}:
            node.zone = zones.get(node.page_json, "mainmatter")

    raw_internal_blocks = []
    for page_index in range(max(0, body_start), min(len(pages), body_end + 1)):
        for block in pages[page_index]:
            parsed = numbering(block.clean_content)
            explicit_section_text = (
                block.label == "text"
                and parsed is not None
                and parsed.family == "section_symbol"
            )
            explicit_apparatus_boundary = (
                block.label in {"doc_title", "paragraph_title"}
                and zone_for_heading(block.clean_content) is not None
            )
            if (
                block.label in {"paragraph_title", "doc_title"}
                or explicit_section_text
                or explicit_apparatus_boundary
            ) and block.id not in used_blocks:
                raw_internal_blocks.append(block)

    merged = merge_heading_blocks(raw_internal_blocks, [entry.title for entry in toc_entries])
    internal_blocks = [block for block, _ in merged]
    fragments_by_id = {block.id: fragments for block, fragments in merged}
    fragment_pages = {block.id: block.page_index for block in raw_internal_blocks}

    style_decisions = infer_internal_styles(internal_blocks)
    for block in internal_blocks:
        zone = zones.get(block.page_index, "mainmatter")
        visual_context = heading_visual_context(block, pages[block.page_index])
        reason = rejection_reason(block.clean_content) or visual_context.rejection_reason
        boundary_zone = zone_for_heading(block.clean_content)
        if zone != "mainmatter" and boundary_zone == zone and not reason:
            nodes.append(
                Node(
                    id="node-" + stable_id(book_id, block.id, "backmatter-boundary"),
                    kind="backmatter",
                    level=2,
                    title_source=canonical_heading_text(block.clean_content),
                    page_json=block.page_index,
                    page_print=(block.page_index + page_map.offset)
                    if page_map.offset is not None
                    else None,
                    parent_id=book_node_id,
                    block_id=block.id,
                    source=block.label,
                    confidence=0.99,
                    evidence=["apparatus_boundary_heading", f"zone:{zone}"],
                    bbox=block.bbox,
                    source_level=block.source_level,
                    zone=zone,
                    source_block_ids=fragments_by_id.get(block.id, [block.id]),
                )
            )
            used_blocks.update(fragments_by_id.get(block.id, [block.id]))
            continue
        if zone != "mainmatter" or reason:
            evidence = [
                reason or "outside_mainmatter",
                f"zone:{zone}",
                "auto_rejected",
                *visual_context.evidence,
            ]
            nodes.append(
                Node(
                    id="node-" + stable_id(book_id, block.id, "rejected-heading"),
                    kind="section",
                    level=3,
                    title_source=canonical_heading_text(block.clean_content),
                    page_json=block.page_index,
                    page_print=(block.page_index + page_map.offset) if page_map.offset is not None else None,
                    parent_id=book_node_id,
                    block_id=block.id,
                    source="paragraph_title",
                    confidence=0.99,
                    evidence=evidence,
                    bbox=block.bbox,
                    source_level=block.source_level,
                    ignored=True,
                    zone=zone,
                    source_block_ids=fragments_by_id.get(block.id, [block.id]),
                )
            )
            used_blocks.update(fragments_by_id.get(block.id, [block.id]))
            continue
        parent_id = _chapter_parent(nodes, block.page_index) or book_node_id
        confidence, evidence = _internal_confidence(block)
        evidence.extend(visual_context.evidence)
        if visual_context.confidence_cap is not None:
            confidence = min(confidence, visual_context.confidence_cap)
        style = style_decisions[block.id]
        confidence = min(confidence, style.confidence)
        evidence.extend(style.evidence)
        fragments = fragments_by_id.get(block.id, [block.id])
        if len(fragments) > 1:
            evidence.append("adjacent_heading_fragments_merged")
            if any(fragment_pages.get(fragment_id) != block.page_index for fragment_id in fragments):
                evidence.append("cross_page_heading_fragments_merged")
        parsed_number = numbering(block.clean_content)
        nodes.append(
            Node(
                id="node-" + stable_id(book_id, block.id, "section"),
                kind="section",
                level=style.level,
                title_source=block.clean_content,
                page_json=block.page_index,
                page_print=(block.page_index + page_map.offset)
                if page_map.offset is not None
                else None,
                parent_id=parent_id,
                block_id=block.id,
                source=block.label,
                confidence=confidence,
                evidence=evidence,
                bbox=block.bbox,
                source_level=block.source_level,
                zone=zone,
                ordinal=parsed_number.token if parsed_number else None,
                numbering_path=list(parsed_number.path) if parsed_number else [],
                source_block_ids=fragments,
            )
        )
        used_blocks.update(fragments)

    _recover_text_children_below_doc_titles(
        pages, nodes, used_blocks, page_map, book_id
    )
    _recover_structural_text_headings(
        pages,
        nodes,
        used_blocks,
        zones,
        (body_start, max(body_start, body_end)),
        page_map,
        book_id,
        book_node_id,
        config.get("text_candidate_profile"),
    )
    _repair_numbered_parents(nodes, book_node_id, chapters_under_parts)
    _repair_unnumbered_section_parents(nodes, book_node_id)
    _repair_section_symbol_parents(nodes, book_node_id)
    if config.get("hierarchy_profile") == "german_legal_commentary":
        _repair_german_legal_hierarchy(nodes, book_node_id)
    elif config.get("hierarchy_profile") == "geist_interleaved_outline":
        _repair_geist_interleaved_outline(nodes, book_node_id)
    elif config.get("hierarchy_profile") == "geist_part_dependent_outline":
        _repair_geist_part_dependent_outline(nodes, book_node_id)
    elif config.get("hierarchy_profile") == "sequential_arabic_chapters_roman_sections":
        _repair_sequential_arabic_roman_hierarchy(nodes, book_node_id)
    elif not config.get("hierarchy_profile"):
        _normalize_standard_hierarchy(nodes, book_node_id, chapters_under_parts)
    _deduplicate_nodes(nodes)
    suppress_apparatus_duplicates(nodes)

    text_candidates = _text_candidates(
        pages,
        used_blocks,
        (body_start, max(body_start, body_end)),
        zones,
        nodes,
        config.get("text_candidate_profile"),
    )
    text_candidates = _promote_text_candidates_below_doc_title(
        text_candidates, nodes, page_map, book_id
    )
    apply_overrides(
        nodes,
        override_path,
        text_candidates=text_candidates,
        page_map=page_map,
        book_id=book_id,
        book_node_id=book_node_id,
    )
    if config.get("suppress_book_title_duplicates", True):
        _suppress_book_title_duplicates(
            nodes, text_candidates, str(config.get("book_title") or "")
        )
    nodes.sort(
        key=lambda node: (
            node.page_json is not None,
            node.page_json if node.page_json is not None else -1,
            node.bbox[1] if node.bbox else -1,
            node.level,
        )
    )
    return nodes, text_candidates


def _repair_numbered_parents(
    nodes: List[Node], book_node_id: str, chapters_under_parts: bool = False
) -> None:
    """Use semantic decimal prefixes before unreliable OCR hash depth."""
    active_stack: Dict[Tuple[str, ...], Node] = {}
    current_chapter = book_node_id
    current_part = book_node_id
    for node in sorted(nodes, key=lambda item: (item.page_json if item.page_json is not None else -1, item.bbox[1] if item.bbox else -1)):
        if node.kind == "part" and not node.ignored:
            node.parent_id = book_node_id
            node.level = 2
            current_part = node.id
            current_chapter = node.id
            active_stack = {}
            if node.numbering_path:
                active_stack[tuple(node.numbering_path)] = node
            continue
        if node.kind == "chapter" and not node.ignored:
            node.parent_id = current_part
            node.level = (
                3
                if chapters_under_parts and current_part != book_node_id
                else 2
            )
            current_chapter = node.id
            active_stack = {}
            if node.numbering_path:
                active_stack[tuple(node.numbering_path)] = node
            continue
        if node.kind != "section" or node.ignored or not node.numbering_path:
            continue
        path = tuple(node.numbering_path)
        if len(path) > 1:
            parent = active_stack.get(path[:-1])
            node.parent_id = parent.id if parent else current_chapter
            if parent:
                node.level = min(6, parent.level + 1)
            active_stack[path] = node
        elif node.kind == "section":
            node.parent_id = current_chapter
            chapter = next((item for item in nodes if item.id == current_chapter), None)
            if chapter:
                node.level = min(6, chapter.level + 1)


def _repair_sequential_arabic_roman_hierarchy(
    nodes: List[Node], book_node_id: str
) -> None:
    """Recover a continuous Arabic chapter series with Roman subsections.

    A date such as ``3 December 1920`` is not promoted when the next expected
    chapter is 16.  This keeps the rule structural rather than merely lexical.
    """
    current_part = book_node_id
    current_chapter: Optional[str] = None
    expected_chapter = 1
    ordered = sorted(
        [node for node in nodes if node.page_json is not None and not node.ignored],
        key=lambda item: (
            item.page_json if item.page_json is not None else -1,
            item.bbox[1] if item.bbox else -1,
        ),
    )
    for node in ordered:
        parsed = numbering(node.title_source)
        if parsed and parsed.family in {"part", "book", "volume", "teil"}:
            node.kind = "part"
            node.level = 2
            node.parent_id = book_node_id
            current_part = node.id
            current_chapter = None
            node.evidence.append("hierarchy_profile:sequential_arabic_roman")
            continue
        canonical = canonical_heading_text(node.title_source)
        arabic = ARABIC_TITLE.match(canonical)
        if arabic and NUMBERED_DATE.match(canonical):
            arabic = None
        if arabic and int(arabic.group(1)) == expected_chapter:
            node.kind = "chapter"
            node.level = 2
            node.parent_id = current_part
            current_chapter = node.id
            expected_chapter += 1
            node.evidence.append("hierarchy_profile:sequential_arabic_chapter")
            continue
        roman = ROMAN_TITLE.match(canonical_heading_text(node.title_source))
        if roman and current_chapter:
            node.kind = "section"
            node.level = 3
            node.parent_id = current_chapter
            node.evidence.append("hierarchy_profile:roman_subsection")


def _repair_section_symbol_parents(nodes: List[Node], book_node_id: str) -> None:
    """Attach § paragraphs to the nearest visible enclosing heading."""
    stack: List[Node] = []
    ordered = sorted(
        [node for node in nodes if node.page_json is not None and not node.ignored],
        key=lambda item: (
            item.page_json if item.page_json is not None else -1,
            item.bbox[1] if item.bbox else -1,
            item.level,
        ),
    )
    for node in ordered:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        parsed = numbering(node.title_source)
        if parsed and parsed.family == "section_symbol":
            node.parent_id = stack[-1].id if stack else book_node_id
        stack.append(node)


def _repair_german_legal_hierarchy(nodes: List[Node], book_node_id: str) -> None:
    """Restore the printed Teil/Abschnitt/A./I./a)/§ nesting used by Hegel editions."""
    stack: List[Node] = []
    ordered = sorted(
        [node for node in nodes if node.page_json is not None and not node.ignored],
        key=lambda item: (
            item.page_json if item.page_json is not None else -1,
            item.bbox[1] if item.bbox else -1,
            item.level,
        ),
    )
    for node in ordered:
        title = canonical_heading_text(node.title_source)
        parsed = numbering(title)
        old_level = node.level
        if node.kind in {"frontmatter", "backmatter", "part"}:
            node.level = 2
        elif title.casefold() in {"naturrecht und staatswissenschaft", "vorrede"}:
            node.level = 2
        elif title.casefold() in {"einleitung", "einteilung"}:
            node.level = 3
        elif title.casefold().startswith("übergang "):
            node.level = 3
        elif re.match(r"^\d+\.(?:\s|$)", title):
            node.level = 2
        elif GERMAN_SECTION.match(title):
            node.level = 3
        elif GERMAN_ROMAN.match(title):
            node.level = 5
        elif GERMAN_UPPER_LETTER.match(title):
            node.level = 4
        elif GERMAN_LOWER_LETTER.match(title):
            node.level = min(6, (stack[-1].level + 1) if stack else 5)
        elif parsed and parsed.family == "section_symbol":
            # Preserve the established canonical § depth used by downstream
            # translation rendering; these paragraph markers are leaves.
            node.level = 5

        section_symbol_leaf = bool(parsed and parsed.family == "section_symbol")
        if section_symbol_leaf:
            parent = next(
                (candidate for candidate in reversed(stack) if candidate.level < node.level),
                None,
            )
            node.parent_id = parent.id if parent else book_node_id
        else:
            while stack and stack[-1].level >= node.level:
                stack.pop()
            node.parent_id = stack[-1].id if stack else book_node_id
        if node.level != old_level or "german_legal_hierarchy_profile" not in node.evidence:
            node.evidence.append("german_legal_hierarchy_profile")
        # Numbered Hegel paragraphs are leaves, not containers for the next §
        # or for a following a)/b)/c) subheading.
        if not section_symbol_leaf:
            stack.append(node)


def _repair_geist_interleaved_outline(nodes: List[Node], book_node_id: str) -> None:
    """Interleave source A./a./III. containers with numbered commentary sections."""
    stack: List[Node] = []
    ordered = sorted(
        [node for node in nodes if node.page_json is not None and not node.ignored],
        key=lambda item: (
            item.page_json if item.page_json is not None else -1,
            item.bbox[1] if item.bbox else -1,
            item.level,
        ),
    )
    for node in ordered:
        title = canonical_heading_text(node.title_source)
        parsed = numbering(title)
        old_level, old_parent = node.level, node.parent_id

        # Parenthesized (AA)/(BB)/(CC) labels are parallel source markers,
        # not containers that should displace the current Teil.
        if node.kind == "toc_entry":
            node.parent_id = book_node_id
            continue
        if node.kind in {"frontmatter", "backmatter"}:
            node.level = 2
            stack = []
        elif node.kind == "part":
            node.level = 2
        elif re.match(r"^(?:[A-Ca-c])\.(?:\s|$)", title):
            node.level = 3
        elif GERMAN_ROMAN.match(title):
            node.level = 3
        elif node.ordinal and str(node.ordinal).upper() in {"I", "II"}:
            # OCR may render source Roman I/II as Arabic 1/11.  In this
            # edition these are H4 source-outline siblings, not commentary roots.
            node.level = 4
        elif parsed and parsed.family == "decimal":
            node.level = min(6, 2 + parsed.depth)
        elif node.kind == "chapter" or re.match(r"^Kapitel\b", title, re.I):
            node.level = 3

        while stack and stack[-1].level >= node.level:
            stack.pop()
        node.parent_id = stack[-1].id if stack else book_node_id
        if node.level != old_level or node.parent_id != old_parent:
            node.evidence.append("geist_interleaved_outline_profile")
        stack.append(node)


def _repair_geist_part_dependent_outline(
    nodes: List[Node], book_node_id: str
) -> None:
    """Restore the different commentary grammars used by each Geist Band 1 Teil.

    Teil 1 already follows decimal numbering and is deliberately left alone.
    Teil 2 nests integer commentary under Kapitel/source headings, with later
    commentary nested under A./B. source-outline headings.  Teil 3 uses the
    deeper A./a./integer interleave.  The state machine relies only on printed
    numbering and order; it contains no title strings or block IDs.
    """
    ordered = sorted(
        [node for node in nodes if node.page_json is not None and not node.ignored],
        key=lambda item: (
            item.page_json if item.page_json is not None else -1,
            item.bbox[1] if item.bbox else -1,
            item.level,
        ),
    )
    current_part: Optional[Node] = None
    part_number: Optional[int] = None
    source_root: Optional[Node] = None
    upper_outline: Optional[Node] = None
    lower_outline: Optional[Node] = None
    integer_nodes: Dict[int, Node] = {}

    def set_structure(node: Node, level: int, parent: Optional[Node]) -> None:
        old_level, old_parent = node.level, node.parent_id
        node.level = level
        node.parent_id = parent.id if parent else book_node_id
        if node.level != old_level or node.parent_id != old_parent:
            node.evidence.append("geist_part_dependent_outline_profile")

    for node in ordered:
        title = canonical_heading_text(node.title_source)
        parsed = numbering(title)

        if node.kind == "toc_entry":
            node.parent_id = book_node_id
            continue
        if node.kind in {"frontmatter", "backmatter"}:
            current_part = None
            part_number = None
            source_root = upper_outline = lower_outline = None
            integer_nodes = {}
            continue
        if node.kind == "part":
            current_part = node
            try:
                part_number = int(str(node.ordinal))
            except (TypeError, ValueError):
                part_number = None
            source_root = upper_outline = lower_outline = None
            integer_nodes = {}
            set_structure(node, 2, None)
            continue
        if current_part is None or part_number not in {2, 3}:
            continue

        is_kapitel = bool(re.match(r"^Kapitel\b", title, re.I))
        upper_match = re.match(r"^[A-H]\.(?:\s|$)", title)
        lower_match = re.match(r"^[a-h]\.(?:\s|$)", title)
        ordinal_text = str(node.ordinal or "")
        integer = int(ordinal_text) if ordinal_text.isdigit() else None

        if is_kapitel:
            set_structure(node, 3, current_part)
            source_root = node
            upper_outline = lower_outline = None
            continue

        if part_number == 2 and node.kind == "section" and parsed is None:
            # The printed source title immediately following Kapitel is an H3
            # sibling and becomes the container for its commentary sequence.
            set_structure(node, 3, current_part)
            source_root = node
            upper_outline = lower_outline = None
            continue

        if upper_match:
            parent = source_root if part_number == 2 else current_part
            set_structure(node, 4, parent)
            upper_outline = node
            lower_outline = None
            continue

        if lower_match and part_number == 3:
            set_structure(node, 5, upper_outline or current_part)
            lower_outline = node
            continue

        if parsed and parsed.family == "decimal":
            try:
                root_number = int(parsed.path[0])
            except (TypeError, ValueError):
                root_number = -1
            parent = integer_nodes.get(root_number)
            if parent:
                set_structure(node, min(6, parent.level + 1), parent)
            continue

        if integer is None:
            continue
        if part_number == 2:
            if upper_outline is not None and integer >= 20:
                set_structure(node, 5, upper_outline)
            else:
                set_structure(node, 4, source_root or current_part)
        else:
            if lower_outline is not None:
                set_structure(node, 6, lower_outline)
            elif upper_outline is not None:
                set_structure(node, 5, upper_outline)
            else:
                set_structure(node, 4, current_part)
        integer_nodes[integer] = node


def _repair_unnumbered_section_parents(
    nodes: List[Node], book_node_id: str
) -> None:
    """Attach unnumbered H4+ headings to the nearest preceding shallower section."""
    current_container = book_node_id
    section_stack: List[Node] = []
    ordered = sorted(
        [node for node in nodes if node.page_json is not None and not node.ignored],
        key=lambda item: (
            item.page_json if item.page_json is not None else -1,
            item.bbox[1] if item.bbox else -1,
            item.level,
        ),
    )
    for node in ordered:
        if node.kind in {"part", "chapter"}:
            current_container = node.id
            section_stack = []
            continue
        if node.kind != "section":
            continue
        while section_stack and section_stack[-1].level >= node.level:
            section_stack.pop()
        parsed = numbering(node.title_source)
        if parsed is None and node.level >= 4:
            expected_parent = (
                section_stack[-1].id if section_stack else current_container
            )
            if node.parent_id != expected_parent:
                node.parent_id = expected_parent
                node.evidence.append("non_numbered_level_parent_repaired")
        section_stack.append(node)


def _deduplicate_nodes(nodes: List[Node]) -> None:
    seen: Dict[Tuple[str, Optional[int], str], Node] = {}
    replacements: Dict[str, str] = {}
    for node in nodes:
        if node.kind == "book" or node.ignored:
            continue
        key = (_normalized_title(node.title_source), node.page_json, node.zone)
        if not key[0]:
            continue
        if key in seen:
            node.ignored = True
            node.evidence.append("same_page_duplicate_suppressed")
            replacements[node.id] = seen[key].id
        else:
            seen[key] = node
    for node in nodes:
        if node.parent_id in replacements:
            node.parent_id = replacements[node.parent_id]
            node.evidence.append("parent_redirected_after_dedup")
