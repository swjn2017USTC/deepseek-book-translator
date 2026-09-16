from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from .models import Block, TocEntry


TOC_TITLE = re.compile(r"^(contents|table of contents|inhalt|inhaltsverzeichnis|目录)$", re.I)
TOC_STOP_TITLE = re.compile(
    r"^(tables|list of tables|figures|list of figures|abbreviations|"
    r"acknowledg(?:e)?ments|表格|图表|插图|缩略语|致谢)$",
    re.I,
)
ENTRY = re.compile(r"^(.*?)(?:\s+page)?\s+([ivxlcdm]+|\d{1,4})$", re.I)
ENTRY_WITH_AUTHOR = re.compile(
    r"^(.*?\D)\s+(\d{1,4})(?:\s+[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ .,'’\-]+)?$"
)
LEADER = re.compile(r"[.·•…]{2,}")
WRAP_END = re.compile(r"\b(?:and|of|the|in|den|der|die|das|des|und|von|zu)\s*$", re.I)
WRAP_START = re.compile(r"^(?:and|of|the|in|im|am|und|von|zu|zur|zum)\b", re.I)
CHAPTER = re.compile(
    r"^(?:chapter\s+)?([0-9]+|[ivxlcdm]+)\s*(?:\\?[.·])?\s+(.+)$",
    re.I,
)
PART_ONLY = re.compile(
    r"^(?:(part|book|volume|teil)\s+([ivxlcdm]+|\d+)\b|"
    r"(?:erster|zweiter|dritter|vierter|fünfter|sechster)\s+teil\b).*$",
    re.I,
)
GLUED_PART_CHAPTER = re.compile(
    r"^((?:part|book|volume|teil)\s+(?:[ivxlcdm]+|\d+)\b.*?)\s+"
    r"((?:chapter\s+)?(?:\d+|"
    # A real glued chapter number is a canonical Roman numeral standing alone
    # ("II The Republic"); a bare [ivxlcdm]+ also matched the first letters of
    # an ordinary word.  Live regression: the TOC line
    # "PART II: IDENTITIES: GENDER, NATION, CIVIL SOCIETY 59" was split into a
    # part plus a bogus chapter "CIVIL SOCIETY" (the "CIVIL" prefix matched as a
    # numeral), which left one macro TOC entry permanently unaligned
    # (what-was-socialism: macro_toc_alignment_incomplete).
    r"(?=[IVXLCDM])M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
    r"(?![A-Za-z]))\s*(?:\\?[.·])?\s+.+)$",
    re.I,
)
OCR_MIXED_PAGE = re.compile(
    r"(?<=\s)([Il|]+\d{1,3})(?=\s+[A-ZÀ-ÖØ-Þ]|$)"
)


def find_toc_pages(
    pages: List[List[Block]], search_range: Sequence[int]
) -> List[int]:
    start, end = int(search_range[0]), int(search_range[1])
    found = []
    for page_index in range(max(0, start), min(len(pages), end + 1)):
        if any(TOC_TITLE.match(block.clean_content) for block in pages[page_index]):
            found.append(page_index)
    return found


def _kind(title: str, page_label: str) -> str:
    normalized = title.strip()
    low = normalized.casefold()
    if re.match(r"^introduction\s+to\s+part\b", low):
        return "part_intro"
    if re.match(r"^\d+(?:\.\d+)+\s+", normalized) or re.match(r"^[A-Za-z][.)]\s+", normalized):
        return "section"
    if re.match(r"^(?:kapitel|chapter)\s+(?:\d+|[ivxlcdm]+)\b", low):
        return "chapter"
    if CHAPTER.match(normalized) or low in {"introduction", "general introduction", "volume introduction", "einleitung", "导论", "绪论"}:
        return "chapter"
    if PART_ONLY.match(normalized):
        return "part"
    if re.match(r"^(?:erster|zweiter|dritter|vierter|fünfter|sechster)\s+abschnitt\b", low):
        return "section"
    if low in {
        "conclusion",
        "conclusions",
        "notes",
        "endnotes",
        "bibliography",
        "references",
        "works cited",
        "further reading",
        "bibliographical essay",
        "index",
        "literatur",
        "personenregister",
        "sachregister",
        "结论",
        "参考文献",
        "索引",
    }:
        return "backmatter"
    if low in {"vorwort", "preface", "acknowledgments", "acknowledgements"}:
        return "frontmatter"
    if not page_label.isdigit():
        return "frontmatter"
    return "toc_entry"


def _segments(text: str) -> Iterable[str]:
    value = text.replace("\u00a0", " ")
    lines = [line for line in re.split(r"\n+", value) if line.strip()]
    if len(lines) > 1:
        # In genuine multiline TOC blocks each physical line is one entry.
        # Dot leaders separate its title and page; they are not entry breaks.
        cleaned_lines = [" ".join(LEADER.sub(" ", line).split()) for line in lines]
        raw_segments = []
        index = 0
        while index < len(cleaned_lines):
            current = cleaned_lines[index]
            if index + 1 < len(cleaned_lines) and not ENTRY.match(current):
                following = cleaned_lines[index + 1]
                following_match = ENTRY.match(following)
                following_title = following_match.group(1).strip() if following_match else following
                if (
                    current.endswith((":", "—", "–", "-"))
                    or WRAP_END.search(current)
                    or WRAP_START.match(following_title)
                ):
                    raw_segments.append(f"{current} {following}")
                    index += 2
                    continue
            raw_segments.append(current)
            index += 1
    else:
        # Some OCR exports flatten several entries onto one line and retain
        # only wide spaces between them.
        cleaned = LEADER.sub("  ", value)
        raw_segments = re.split(r"\s{2,}", cleaned)
    for segment in raw_segments:
        segment = " ".join(segment.split()).strip()
        if segment:
            yield segment


def parse_toc_entries(pages: List[List[Block]], toc_pages: List[int]) -> List[TocEntry]:
    entries: List[TocEntry] = []
    seen = set()
    ordered_pages = sorted(set(toc_pages))
    # Real exports mix content and text rows on the same TOC page.
    allowed_labels = {"content", "text", "doc_title", "paragraph_title"}
    for page_index in ordered_pages:
        for block in pages[page_index]:
            if block.label not in allowed_labels:
                continue
            block_text = (
                block.clean_content
                if block.label in {"doc_title", "paragraph_title"}
                else block.content
            )
            for segment in _segments(block_text):
                if TOC_TITLE.match(segment):
                    continue
                segment = OCR_MIXED_PAGE.sub(
                    lambda match: match.group(1).translate(
                        str.maketrans({"I": "1", "l": "1", "|": "1"})
                    ),
                    segment,
                )
                match = ENTRY.match(segment) or ENTRY_WITH_AUTHOR.match(segment)
                if not match:
                    if PART_ONLY.match(segment) or CHAPTER.match(segment):
                        _append_entry(entries, seen, segment, "", page_index)
                    continue
                title = match.group(1).strip(" .")
                page_label = match.group(2)
                glued = GLUED_PART_CHAPTER.match(title)
                glued_second = glued.group(2).casefold() if glued else ""
                wrapped_part_continuation = bool(
                    re.match(r"^\d+\s*(?:\\?[.·])?\s*band\b", glued_second)
                    or glued_second.startswith(("im ", "in ", "und "))
                )
                if glued and not wrapped_part_continuation:
                    _append_entry(entries, seen, glued.group(1), "", page_index)
                    _append_entry(entries, seen, glued.group(2), page_label, page_index)
                    continue
                _append_entry(entries, seen, title, page_label, page_index)
    return entries


def _append_entry(
    entries: List[TocEntry], seen: set, title: str, page_label: str, source_page: int
) -> None:
    if not title or len(title) > 240:
        return
    if page_label.isdigit() and re.search(r"\b(?:18|19|20)\d{2}\s*[–—-]\s*$", title):
        # A wrapped year range such as "1934–" + "41" is not a page number.
        title = f"{title}{page_label}"
        page_label = ""
    key = (title.casefold(), page_label.casefold())
    if key in seen:
        return
    seen.add(key)
    entries.append(
        TocEntry(
            title=title,
            page_label=page_label,
            print_page=int(page_label) if page_label.isdigit() else None,
            kind="part" if PART_ONLY.match(title) else _kind(title, page_label),
            source_page=source_page,
        )
    )


def title_without_number(title: str) -> str:
    match = CHAPTER.match(title.strip())
    return match.group(2).strip() if match else title.strip()
