from __future__ import annotations

import re
from typing import Dict, List, Optional

from .models import Block, Node
from .heading_text import canonical_heading_text


ZONE_HEADINGS = {
    "notes": re.compile(r"^(notes|endnotes|anmerkungen|注释|注)$", re.I),
    # A chapter-local "Bibliographical Essay" is not a global zone boundary.
    "bibliography": re.compile(
        r"^(references|bibliography|works cited|literatur|literaturverzeichnis|参考文献)$",
        re.I,
    ),
    "index": re.compile(
        r"^(index|name index|subject index|personenregister|sachregister|namenregister|索引)$",
        re.I,
    ),
}


def zone_for_heading(text: str) -> Optional[str]:
    value = canonical_heading_text(text)
    return next(
        (zone for zone, pattern in ZONE_HEADINGS.items() if pattern.match(value)),
        None,
    )


def infer_page_zones(pages: List[List[Block]], main_start: int = 0) -> Dict[int, str]:
    zones: Dict[int, str] = {page: ("frontmatter" if page < main_start else "mainmatter") for page in range(len(pages))}
    current = "mainmatter"
    apparatus_floor = max(main_start + 5, int(len(pages) * 0.55))
    for page_index in range(main_start, len(pages)):
        for block in pages[page_index]:
            if block.label not in {"doc_title", "paragraph_title", "header"}:
                continue
            matched = zone_for_heading(block.content)
            if matched and page_index >= apparatus_floor:
                current = matched
                break
        zones[page_index] = current
    return zones


def suppress_apparatus_duplicates(nodes: List[Node]) -> None:
    main_titles = {
        re.sub(r"[^\w]+", " ", node.title_source.casefold()).strip()
        for node in nodes
        if node.zone == "mainmatter" and node.kind in {"part", "chapter", "section"}
    }
    for node in nodes:
        normalized = re.sub(r"[^\w]+", " ", node.title_source.casefold()).strip()
        if node.zone in {"notes", "bibliography", "index"} and normalized in main_titles:
            node.ignored = True
            node.evidence.append("apparatus_duplicate_suppressed")
