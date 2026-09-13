"""Deterministic chapter briefs for contextual translation v2 (spec §12.3).

Every translatable segment is covered by exactly one brief:

* segments whose nearest ancestor structure node with ``kind == "chapter"``
  (walked through the ``parent_node_id`` -> ``parent_id`` chain, with chapter
  heading segments — whose parent is the book root — resolved through their
  ``metadata.node_id``) map to that chapter node;
* unattached segments (frontmatter/backmatter/index/... with no chapter
  ancestor) map to a synthetic per-zone bucket ``zone:<zone>``.

``brief_hash`` binds ONLY to deterministic canonical structural fields
(§3 contract), so hashes are reproducible offline and LLM enrichment — when it
arrives — can never perturb cache semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..io_utils import write_json
from ..models import Segment


@dataclass
class ChapterBrief:
    chapter_id: str
    title_source: str
    kind: str                    # "chapter" for structure chapters; "zone_bucket" for synthetic buckets
    level: Optional[int]
    zone: str
    page_range: Tuple[int, int]  # (min, max) page_json over member segments
    section_titles: List[str]    # ordered descendant section title_source values
    segment_count: int
    translatable_count: int
    brief_hash: str = ""         # sha256 over the canonical structural fields


# Canonical structural fields the brief hash binds to (spec §3).
HASH_FIELDS = (
    "chapter_id",
    "title_source",
    "kind",
    "level",
    "zone",
    "page_range",
    "section_titles",
    "segment_count",
)


def nodes_index(structure: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes = structure.get("nodes") if isinstance(structure, dict) else None
    return {str(node.get("id")): node for node in nodes if isinstance(node, dict)} if isinstance(nodes, list) else {}


def zone_bucket_id(zone: str) -> str:
    """Synthetic brief key for segments with no chapter ancestor."""
    return f"zone:{zone}"


def _chapter_heading_node_id(segment: Segment) -> Optional[str]:
    metadata = segment.metadata or {}
    if metadata.get("node_kind") == "chapter" and metadata.get("node_id"):
        return str(metadata["node_id"])
    return None


def chapter_id_for_segment(
    segment: Segment,
    nodes_by_id: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Nearest ancestor node with ``kind == "chapter"``, or None when the
    segment is unattached to any chapter.

    Walks ``parent_node_id`` up the ``parent_id`` chain.  Chapter heading
    segments attach to the book root, so their own chapter node (recorded in
    ``metadata.node_id`` when ``metadata.node_kind == "chapter"``) is used.
    """
    current_id = segment.parent_node_id
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        node = nodes_by_id.get(current_id)
        if node is None:
            break
        if node.get("kind") == "chapter":
            return str(node["id"])
        current_id = node.get("parent_id")
    heading_id = _chapter_heading_node_id(segment)
    if heading_id and nodes_by_id.get(heading_id, {}).get("kind") == "chapter":
        return heading_id
    return None


def brief_hash(brief_fields: Dict[str, Any]) -> str:
    """sha256 over the canonical key-sorted structural fields (JSON, UTF-8).

    Key sorting + ``ensure_ascii=False`` makes the digest independent of key
    insertion order and non-structural decorations.
    """
    payload = {key: brief_fields.get(key) for key in HASH_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _descendant_section_titles(chapter_id: str, nodes: Sequence[Dict[str, Any]], nodes_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    titles: List[Tuple[int, int, str]] = []
    for order, node in enumerate(nodes):
        if node.get("kind") != "section":
            continue
        parent_id = node.get("parent_id")
        while parent_id:
            if parent_id == chapter_id:
                titles.append((int(node.get("page_json") if node.get("page_json") is not None else -1), order, str(node.get("title_source") or "")))
                break
            parent_node = nodes_by_id.get(parent_id)
            if parent_node is None:
                break
            parent_id = parent_node.get("parent_id")
    titles.sort(key=lambda item: (item[0], item[1]))
    return [title for _, _, title in titles]


def build_chapter_briefs(
    structure: Dict[str, Any],
    segments: Sequence[Segment],
) -> Dict[str, ChapterBrief]:
    """Build one deterministic brief per chapter node plus synthetic zone
    buckets, so every translatable segment is covered."""
    nodes = [node for node in (structure.get("nodes") or []) if isinstance(node, dict)]
    nodes_by_id = nodes_index(structure)
    members: Dict[str, List[Segment]] = {}
    unattached: Dict[str, List[Segment]] = {}
    for segment in segments:
        chapter_id = chapter_id_for_segment(segment, nodes_by_id)
        if chapter_id is not None:
            members.setdefault(chapter_id, []).append(segment)
        elif segment.translatable:
            unattached.setdefault(segment.zone, []).append(segment)

    briefs: Dict[str, ChapterBrief] = {}
    for node in nodes:
        if node.get("kind") != "chapter":
            continue
        chapter_id = str(node["id"])
        chapter_members = members.get(chapter_id, [])
        translatable = [segment for segment in chapter_members if segment.translatable]
        pages = [int(segment.page_json) for segment in chapter_members if segment.page_json is not None]
        page_range = (min(pages), max(pages)) if pages else (0, 0)
        brief = ChapterBrief(
            chapter_id=chapter_id,
            title_source=str(node.get("title_source") or ""),
            kind="chapter",
            level=node.get("level"),
            zone=str(node.get("zone") or ""),
            page_range=page_range,
            section_titles=_descendant_section_titles(chapter_id, nodes, nodes_by_id),
            segment_count=len(chapter_members),
            translatable_count=len(translatable),
        )
        brief.brief_hash = brief_hash({
            "chapter_id": brief.chapter_id,
            "title_source": brief.title_source,
            "kind": brief.kind,
            "level": brief.level,
            "zone": brief.zone,
            "page_range": list(brief.page_range),
            "section_titles": brief.section_titles,
            "segment_count": brief.segment_count,
        })
        briefs[brief.chapter_id] = brief

    for zone, zone_segments in sorted(unattached.items()):
        translatable = [segment for segment in zone_segments if segment.translatable]
        pages = [int(segment.page_json) for segment in zone_segments if segment.page_json is not None]
        page_range = (min(pages), max(pages)) if pages else (0, 0)
        bucket_id = zone_bucket_id(zone)
        brief = ChapterBrief(
            chapter_id=bucket_id,
            title_source=zone.title(),
            kind="zone_bucket",
            level=None,
            zone=zone,
            page_range=page_range,
            section_titles=[],
            segment_count=len(translatable),
            translatable_count=len(translatable),
        )
        brief.brief_hash = brief_hash({
            "chapter_id": brief.chapter_id,
            "title_source": brief.title_source,
            "kind": brief.kind,
            "level": brief.level,
            "zone": brief.zone,
            "page_range": list(brief.page_range),
            "section_titles": brief.section_titles,
            "segment_count": brief.segment_count,
        })
        briefs[bucket_id] = brief
    return briefs


def brief_payload(brief: ChapterBrief) -> Dict[str, Any]:
    """§12.2 prompt-facing chapter_brief (page range key ``pages``)."""
    return {
        "title_source": brief.title_source,
        "kind": brief.kind,
        "zone": brief.zone,
        "pages": [brief.page_range[0], brief.page_range[1]],
        "section_titles": list(brief.section_titles),
        "segment_count": brief.segment_count,
        "brief_hash": brief.brief_hash,
    }


def write_chapter_briefs(
    output_dir: Path,
    briefs: Dict[str, ChapterBrief],
    *,
    book_id: str = "",
) -> Path:
    """Write ``chapter_briefs.json`` (§3 artifact schema) and return its path."""
    rows = []
    for chapter_id in sorted(briefs):
        brief = briefs[chapter_id]
        rows.append({
            "chapter_id": brief.chapter_id,
            "title_source": brief.title_source,
            "kind": brief.kind,
            "level": brief.level,
            "zone": brief.zone,
            "page_range": [brief.page_range[0], brief.page_range[1]],
            "section_titles": list(brief.section_titles),
            "segment_count": brief.segment_count,
            "translatable_count": brief.translatable_count,
            "brief_hash": brief.brief_hash,
            "llm_enrichment": None,
        })
    document = {
        "schema_version": 1,
        "book_id": book_id,
        "prompts_version": 1,
        "briefs": rows,
    }
    path = Path(output_dir) / "chapter_briefs.json"
    write_json(path, document)
    return path
