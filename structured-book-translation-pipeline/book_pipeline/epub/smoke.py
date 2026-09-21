"""Deterministic smoke selection for native EPUB projects.

E07 is the only phase allowed to spend a small, fixed translation budget on a real book, and the
point of that budget is *structural* coverage: a normal paragraph, a paragraph carrying inline
markup, a heading, a table cell, a footnote reference, a footnote body, and an attribute slot
(``alt``). Taking the first N segments in spine order would cover none of that — a real book opens
with front matter — so this module chooses the segments deterministically instead.

Determinism is the contract: the selection is a pure function of ``cleaned_segments.jsonl`` and the
requested limit. Nothing here writes to the project, fabricates coverage, or inspects translated
text; the caller decides what to do with the IDs. The same input always yields the same segments,
so a reviewer can recompute the selection and check the report against it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

TOKEN = re.compile(r"\[\[EPUB:\d+:(?:OPEN|CLOSE):[A-Za-z0-9_-]+\]\]")
# Note documents in real EPUB2 books are named per chapter (`ch19fn.html`, `notes.html`, …).
NOTES_DOCUMENT = re.compile(r"(?:^|/)[^/]*fn\d*\.x?html?$", re.IGNORECASE)
NOTE_TARGET = re.compile(r"[^/]*fn\d*\.x?html?", re.IGNORECASE)

# Fixed order: the capabilities are reported in this order, and the first matching segment in
# document order wins for each one. `fill` is appended for any remaining budget.
SMOKE_CAPABILITIES = (
    "heading",
    "paragraph_plain",
    "paragraph_inline",
    "table_cell",
    "footnote_reference",
    "footnote_body",
    "attribute",
)


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _has_tokens(row: Mapping[str, Any]) -> bool:
    return bool(TOKEN.search(str(row.get("source_text") or "")))


def _link_targets(row: Mapping[str, Any]) -> List[str]:
    targets = _metadata(row).get("link_targets")
    return [str(target) for target in targets] if isinstance(targets, list) else []


def _document(row: Mapping[str, Any]) -> str:
    return str(_metadata(row).get("href") or "")


def _is_footnote_reference(row: Mapping[str, Any]) -> bool:
    """A segment that links *into* a note document (the reference site in the body)."""
    document = _document(row)
    for target in _link_targets(row):
        if NOTE_TARGET.search(target) and target not in document:
            return True
    return False


def _is_footnote_body(row: Mapping[str, Any]) -> bool:
    """A segment that lives inside a note document (the note itself)."""
    return bool(NOTES_DOCUMENT.search(_document(row)))


def _capability(row: Mapping[str, Any]) -> str | None:
    kind = str(row.get("kind") or "")
    if kind == "heading":
        return "heading"
    if kind == "paragraph":
        if _is_footnote_body(row):
            return "footnote_body"
        if _has_tokens(row):
            return "paragraph_inline"
        return "paragraph_plain"
    if kind == "table_cell":
        return "table_cell"
    if kind == "attribute":
        return "attribute"
    return None


def _matches(capability: str, row: Mapping[str, Any]) -> bool:
    if not row.get("translatable", True):
        return False
    kind = str(row.get("kind") or "")
    if capability == "heading":
        return kind == "heading"
    if capability == "table_cell":
        return kind == "table_cell"
    if capability == "attribute":
        return kind == "attribute"
    if capability == "paragraph_plain":
        return kind == "paragraph" and not _has_tokens(row)
    if capability == "paragraph_inline":
        return kind == "paragraph" and _has_tokens(row) and not _is_footnote_body(row)
    if capability == "footnote_reference":
        return _is_footnote_reference(row)
    if capability == "footnote_body":
        return _is_footnote_body(row)
    return False


def select_smoke_segments(rows: Sequence[Mapping[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """Choose at most ``limit`` segments covering every capability, in document order.

    The first translatable segment satisfying each capability (document order) is taken. The
    remaining budget is filled round-robin over documents in first-appearance order, so the smoke
    spans several chapters instead of spending its budget on the title page. The result is sorted
    by the caller's input order, so translations run in spine order.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []

    chosen: Dict[int, str] = {}
    for capability in SMOKE_CAPABILITIES:
        if len(chosen) >= limit:
            break
        for index, row in enumerate(rows):
            if index in chosen:
                continue
            if _matches(capability, row):
                chosen[index] = capability
                break

    # Fill round-robin over documents: one segment per document per pass, earliest first.
    documents: List[str] = []
    for row in rows:
        document = _document(row)
        if document not in documents:
            documents.append(document)
    progress = True
    while len(chosen) < limit and progress:
        progress = False
        for document in documents:
            if len(chosen) >= limit:
                break
            for index, row in enumerate(rows):
                if index in chosen or not row.get("translatable", True):
                    continue
                if _document(row) != document:
                    continue
                chosen[index] = "fill"
                progress = True
                break

    selection: List[Dict[str, Any]] = []
    for index in sorted(chosen):
        row = rows[index]
        metadata = _metadata(row)
        selection.append({
            "capability": chosen[index],
            "segment_id": str(row.get("id") or ""),
            "kind": str(row.get("kind") or ""),
            "href": _document(row),
            "locator": str(metadata.get("locator") or ""),
            "protected_tokens": len(TOKEN.findall(str(row.get("source_text") or ""))),
        })
    return selection


def load_segment_rows(path: Path) -> List[Dict[str, Any]]:
    """Read ``cleaned_segments.jsonl`` into plain dicts, skipping blank lines."""
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_smoke_segment_ids(rows: Sequence[Mapping[str, Any]], limit: int = 10) -> List[str]:
    """Convenience wrapper: the selected segment IDs in document order."""
    return [entry["segment_id"] for entry in select_smoke_segments(rows, limit=limit)]


def missing_capabilities(selection: Sequence[Mapping[str, Any]]) -> List[str]:
    """Capabilities the selection failed to cover (empty list means full structural coverage)."""
    covered = {str(entry.get("capability")) for entry in selection}
    return [capability for capability in SMOKE_CAPABILITIES if capability not in covered]
