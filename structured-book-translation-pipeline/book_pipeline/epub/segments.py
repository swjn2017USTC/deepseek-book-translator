from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
import zipfile

from ..io_utils import write_json, write_jsonl
from ..models import Segment, digest
from .package import local_name, parse_container, parse_package
from .security import ArchiveLimits, parse_xml, read_member, validate_archive, validate_ocf_mimetype
from .tokens import compile_inline, reconstruct_slots, validate_tokens

BLOCK_TAGS = {"p": "paragraph", "li": "list_item", "blockquote": "blockquote", "dt": "definition_term", "dd": "definition_description", "caption": "caption", "figcaption": "caption", "td": "table_cell", "th": "table_cell", "text": "svg_text"}
SKIP_TAGS = {"script", "style", "code", "pre", "math", "path"}
ATTRIBUTE_SLOTS = ("alt", "title", "aria-label")
CONTENT_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "image/svg+xml"}


def element_paths(root: Any) -> Dict[int, str]:
    output: Dict[int, str] = {}
    def walk(element: Any, prefix: str) -> None:
        output[id(element)] = prefix
        counts: Dict[str, int] = {}
        for child in element:
            tag = local_name(child.tag)
            counts[tag] = counts.get(tag, 0) + 1
            walk(child, f"{prefix}/{tag}[{counts[tag]}]")
    walk(root, f"/{local_name(root.tag)}[1]")
    return output


def _parents(root: Any) -> Dict[int, Any]:
    return {id(child): parent for parent in root.iter() for child in parent}


def _epub_type(element: Any) -> str:
    return " ".join(str(value) for key, value in element.attrib.items() if local_name(key) == "type").casefold()


def is_footnote_element(element: Any) -> bool:
    markers = " ".join((_epub_type(element), str(element.get("role") or ""), str(element.get("class") or ""))).casefold()
    return "footnote" in markers or "endnote" in markers


def is_noteref_element(element: Any) -> bool:
    return "noteref" in " ".join((_epub_type(element), str(element.get("role") or ""), str(element.get("class") or ""))).casefold()


def _kind(element: Any) -> Optional[str]:
    tag = local_name(element.tag)
    if tag in {f"h{level}" for level in range(1, 7)}:
        return "heading"
    if is_footnote_element(element):
        return "footnote"
    return BLOCK_TAGS.get(tag)


def _skipped(element: Any, parents: Mapping[int, Any]) -> bool:
    current: Optional[Any] = element
    while current is not None:
        if local_name(current.tag) in SKIP_TAGS or is_noteref_element(current):
            return True
        current = parents.get(id(current))
    return False


def _selected_blocks(root: Any, parents: Mapping[int, Any]) -> List[Any]:
    selected: List[Any] = []
    selected_ids: Set[int] = set()
    for element in root.iter():
        if _skipped(element, parents) or not _kind(element):
            continue
        current = parents.get(id(element))
        while current is not None:
            if id(current) in selected_ids:
                break
            current = parents.get(id(current))
        else:
            selected.append(element)
            selected_ids.add(id(element))
    return selected


def _id_counts(root: Any) -> Counter:
    return Counter(str(element.get("id")) for element in root.iter() if element.get("id"))


def _locator(element: Any, paths: Mapping[int, str], ids: Mapping[str, int]) -> str:
    element_id = str(element.get("id") or "")
    return f"id:{element_id}" if element_id and ids[element_id] == 1 else f"path:{paths[id(element)]}"


def _segment_id(book_id: str, href: str, locator: str, slot_kind: str, ordinal: int = 0) -> str:
    return f"seg-{digest(book_id, href, locator, slot_kind, ordinal)[:20]}"


def _parent_heading(element: Any, headings: List[Tuple[Any, str]], parents: Mapping[int, Any]) -> Optional[str]:
    ancestors = set()
    current = parents.get(id(element))
    while current is not None:
        ancestors.add(id(current))
        current = parents.get(id(current))
    for heading, identifier in reversed(headings):
        if id(heading) in ancestors:
            return identifier
    return headings[-1][1] if headings else None


def _zone(element: Any, parents: Mapping[int, Any]) -> str:
    current: Optional[Any] = element
    while current is not None:
        if is_footnote_element(current):
            return "notes"
        current = parents.get(id(current))
    return "mainmatter"


def _table_metadata(element: Any, parents: Mapping[int, Any], paths: Mapping[int, str]) -> Dict[str, Any]:
    if local_name(element.tag) not in {"td", "th"}:
        return {}
    row = parents.get(id(element))
    table = row
    while table is not None and local_name(table.tag) != "table":
        table = parents.get(id(table))
    rows = [candidate for candidate in table.iter() if local_name(candidate.tag) == "tr"] if table is not None else []
    occupied: Set[Tuple[int, int]] = set()
    for row_index, candidate_row in enumerate(rows):
        column = 0
        for cell in (child for child in candidate_row if local_name(child.tag) in {"td", "th"}):
            while (row_index, column) in occupied:
                column += 1
            rowspan = int(cell.get("rowspan") or 1)
            colspan = int(cell.get("colspan") or 1)
            if cell is element:
                return {"row": row_index, "column": column, "rowspan": rowspan, "colspan": colspan, "cell_locator": paths[id(element)], "table_locator": paths[id(table)] if table is not None else ""}
            for future_row in range(row_index, row_index + rowspan):
                for future_column in range(column, column + colspan):
                    occupied.add((future_row, future_column))
            column += colspan
    return {}


def _reference_metadata(element: Any) -> Dict[str, List[str]]:
    links: List[str] = []
    footnotes: List[str] = []
    for node in element.iter():
        href = str(node.get("href") or "")
        if href:
            links.append(href)
            if is_noteref_element(node):
                footnotes.append(href)
    return {"link_targets": links, "footnote_targets": footnotes}


def _structural_hash(element: Any) -> str:
    """Hash a segment's DOM skeleton, excluding independently translated attrs."""
    parts: List[str] = []
    def walk(node: Any) -> None:
        parts.append(local_name(node.tag))
        parts.extend(
            f"{local_name(key)}={value}"
            for key, value in sorted(node.attrib.items())
            if local_name(key) not in ATTRIBUTE_SLOTS
        )
        for child in node:
            walk(child)
    walk(element)
    return digest(*parts)


def _metadata(rootfile: str, spine_index: int, href: str, locator: str, slots: List[Dict[str, str]], template: List[str], document_hash: str, language: str = "", **extra: Any) -> Dict[str, Any]:
    return {"source_adapter": "epub_native_v1", "rootfile": rootfile, "spine_index": spine_index, "href": href, "locator": locator, "slots": slots, "template": template, "source_document_sha256": document_hash, "language": language, **extra}


def _attribute_segments(book_id: str, href: str, spine_index: int, rootfile: str, root: Any, paths: Mapping[int, str], parents: Mapping[int, Any], ids: Mapping[str, int], document_hash: str, headings: List[Tuple[Any, str]], language: str) -> List[Segment]:
    output: List[Segment] = []
    for element in root.iter():
        if _skipped(element, parents):
            continue
        for attribute in ATTRIBUTE_SLOTS:
            value = str(element.get(attribute) or "")
            if not value.strip():
                continue
            locator = _locator(element, paths, ids)
            slot = {"id": "s0", "kind": "attribute", "path": paths[id(element)], "attribute": attribute, "value": value}
            segment_id = _segment_id(book_id, href, locator, f"attribute:{attribute}")
            metadata = _metadata(rootfile, spine_index, href, locator, [slot], ["s0"], document_hash, language=language, attribute=attribute)
            metadata["structural_hash"] = _structural_hash(element)
            metadata["chapter_dependency"] = _parent_heading(element, headings, parents) or f"document:{href}"
            output.append(Segment(segment_id, digest(value), "attribute", value, None, _zone(element, parents), _parent_heading(element, headings, parents), translatable=True, metadata=metadata))
    return output


def segment_epub(path: Path, book_id: str, output_dir: Optional[Path] = None, *, limits: ArchiveLimits = ArchiveLimits()) -> Dict[str, Any]:
    source = path.expanduser().resolve()
    segments: List[Segment] = []
    skipped = Counter()
    documents: List[Dict[str, Any]] = []
    with zipfile.ZipFile(source) as archive:
        members = validate_archive(archive, limits)
        validate_ocf_mimetype(archive, members)
        rootfile = parse_container(archive, members)[0]["path"]
        package = parse_package(archive, members, rootfile)
        manifest = {item["id"]: item for item in package["manifest"]}
        for spine_index, itemref in enumerate(package["spine"]):
            item = manifest[itemref["idref"]]
            if item["media_type"].casefold() not in CONTENT_MEDIA_TYPES:
                skipped["non_content_spine"] += 1
                continue
            if "nav" in item["properties"].split():
                skipped["navigation_document"] += 1
                continue
            data = read_member(archive, members, item["path"])
            root = parse_xml(data, label=item["path"], limits=limits)
            paths, parents, ids, document_hash = element_paths(root), _parents(root), _id_counts(root), sha256(data).hexdigest()
            language = next((str(value) for key, value in root.attrib.items() if local_name(key) == "lang"), "")
            headings: List[Tuple[Any, str]] = []
            for element in _selected_blocks(root, parents):
                kind = _kind(element)
                assert kind is not None
                source_text, slots, template = compile_inline(element, paths, lambda node: local_name(node.tag) in SKIP_TAGS or local_name(node.tag) == "comment" or is_noteref_element(node))
                if not source_text.strip():
                    skipped[f"empty_{kind}"] += 1
                    continue
                locator = _locator(element, paths, ids)
                identifier = _segment_id(book_id, item["path"], locator, "text")
                parent_heading = _parent_heading(element, headings, parents)
                metadata = _metadata(
                    rootfile, spine_index, item["path"], locator, slots, template,
                    document_hash, language=language, **_table_metadata(element, parents, paths),
                    **_reference_metadata(element),
                )
                metadata["structural_hash"] = _structural_hash(element)
                metadata["chapter_dependency"] = identifier if kind == "heading" else (parent_heading or f"document:{item['path']}")
                segment = Segment(identifier, digest(source_text), kind, source_text, None, _zone(element, parents), parent_heading, level=int(local_name(element.tag)[1:]) if kind == "heading" else None, translatable=True, metadata=metadata)
                segments.append(segment)
                if kind == "heading":
                    headings.append((element, identifier))
            segments.extend(_attribute_segments(book_id, item["path"], spine_index, rootfile, root, paths, parents, ids, document_hash, headings, language))
            documents.append({"href": item["path"], "spine_index": spine_index, "sha256": document_hash})
    source_hash = sha256(source.read_bytes()).hexdigest()
    report = {"schema_version": 1, "book_id": book_id, "source_path": str(source), "source_sha256": source_hash, "metrics": {"segments": len(segments), "kinds": dict(sorted(Counter(segment.kind for segment in segments).items())), "skipped": dict(sorted(skipped.items()))}}
    source_manifest = {"schema_version": 1, "adapter": "epub_native_v1", "book_id": book_id, "source_path": str(source), "source_sha256": source_hash, "rootfile": rootfile, "documents": documents}
    result = {"segments": segments, "cleaning_report": report, "source_manifest": source_manifest}
    if output_dir is not None:
        destination = output_dir.expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        write_jsonl(destination / "cleaned_segments.jsonl", [segment.to_dict() for segment in segments])
        write_json(destination / "cleaning_report.json", report)
        write_json(destination / "source_manifest.json", source_manifest)
    return result


def reconstruct_segment_slots(segment: Segment, candidate: str) -> Dict[str, str]:
    validate_tokens(segment.source_text, candidate)
    return reconstruct_slots(segment.metadata["template"], segment.metadata["slots"], candidate)


def apply_segment_slots(root: Any, segment: Segment, candidate: str, paths: Optional[Mapping[int, str]] = None) -> None:
    """Write one reconstructed translation back into its recorded DOM slots.

    ``paths`` lets a caller that rewrites a whole document reuse one path map
    instead of recomputing it for every segment; it must come from ``root``.
    """
    values = reconstruct_segment_slots(segment, candidate)
    mapping = paths if paths is not None else element_paths(root)
    by_path = {mapping[id(element)]: element for element in root.iter()}
    for slot in segment.metadata["slots"]:
        element = by_path.get(slot["path"])
        if element is None:
            raise ValueError(f"EPUB slot path is missing for {segment.id}: {slot['path']}")
        value = values[slot["id"]]
        if slot["kind"] == "text":
            element.text = value or None
        elif slot["kind"] == "tail":
            # An empty tail must be ``None``, not ``""``: the source block ended
            # in an element, so its tail is None, and writing "" would show up
            # as a structural change to an untouched node.
            element.tail = value or None
        else:
            element.set(slot["attribute"], value)
