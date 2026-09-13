from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .alignment import title_similarity
from .models import TocEntry


def _items(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        yield from (item for item in data if isinstance(item, dict))
        return
    if not isinstance(data, dict):
        return
    for key in ("entries", "toc_plan", "front_matter", "parts", "chapters"):
        value = data.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield {"title": item, "_container": key}
                elif isinstance(item, dict):
                    enriched = dict(item)
                    enriched["_container"] = key
                    yield enriched
                    nested = item.get("chapters")
                    if isinstance(nested, list):
                        for child in nested:
                            if isinstance(child, str):
                                yield {"title": child, "_container": "chapters"}
                            elif isinstance(child, dict):
                                yield {**child, "_container": "chapters"}


def _title(item: Dict[str, Any]) -> str:
    for key in ("title_src", "title", "matched_heading", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _page(item: Dict[str, Any]) -> Optional[int]:
    for key in ("page", "print_page", "page_start", "start_page"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _kind(item: Dict[str, Any], title: str) -> str:
    explicit = str(item.get("kind") or "").casefold()
    level = str(item.get("level") or "").casefold()
    low = title.casefold()
    container = str(item.get("_container") or "")
    number = str(item.get("number") or "").casefold()
    if container == "front_matter":
        return "frontmatter"
    if container == "parts" or number.startswith(("part ", "teil ")):
        return "part"
    if container == "chapters" or (number and number.isdigit()):
        return "chapter"
    if explicit in {"part", "chapter", "frontmatter", "backmatter"}:
        return explicit
    if level in {"part", "chapter", "frontmatter", "backmatter"}:
        return level
    if low.startswith(("part ", "teil ", "book ", "volume ")):
        return "part"
    if low.startswith("chapter ") or low == "introduction":
        return "chapter"
    if low in {"notes", "references", "bibliography", "works cited", "index", "further reading"}:
        return "backmatter"
    return "toc_entry"


def load_confirmed_toc(path: Path) -> List[TocEntry]:
    """Normalize the several human-confirmed TOC schemas used in the corpus."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries: List[TocEntry] = []
    seen = set()
    for item in _items(data):
        title = _title(item)
        if not title:
            continue
        page = _page(item)
        kind = _kind(item, title)
        key = (title.casefold(), page, kind)
        if key in seen:
            continue
        seen.add(key)
        entries.append(TocEntry(title, str(page or ""), page, kind, -1, provenance="confirmed_toc"))
    return entries


def merge_confirmed_toc(parsed: List[TocEntry], confirmed: List[TocEntry]) -> List[TocEntry]:
    """Confirmed entries are the macro prior; OCR-only entries fill omissions.

    Match every confirmed item, including items that already have page numbers.
    Earlier versions only consumed OCR matches for page-less items and therefore
    appended a second copy of several chapters.
    """
    if not confirmed:
        return parsed
    result = list(confirmed)
    used_parsed = set()
    for confirmed_item in result:
        best_index, best_score = None, 0.0
        for index, parsed_item in enumerate(parsed):
            if index in used_parsed:
                continue
            score = title_similarity(confirmed_item.title, parsed_item.title)
            if confirmed_item.kind == parsed_item.kind:
                score += 0.04
            if (
                confirmed_item.print_page is not None
                and parsed_item.print_page is not None
                and confirmed_item.print_page == parsed_item.print_page
            ):
                score += 0.06
            if score > best_score:
                best_index, best_score = index, score
        if best_index is not None and best_score >= 0.68:
            match = parsed[best_index]
            if confirmed_item.print_page is None:
                confirmed_item.print_page = match.print_page
                confirmed_item.page_label = match.page_label
            confirmed_item.source_page = match.source_page
            confirmed_item.provenance = "confirmed_toc+ocr_toc"
            confirmed_item.id = ""
            confirmed_item.__post_init__()
            used_parsed.add(best_index)
    known = {(item.title.casefold(), item.print_page) for item in result}
    for index, item in enumerate(parsed):
        if index in used_parsed:
            continue
        if (item.title.casefold(), item.print_page) not in known:
            result.append(item)
    return result
