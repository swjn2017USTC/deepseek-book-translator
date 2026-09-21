"""DOM write-back of current translations into a rendered EPUB tree.

The source container is never modified. Current translations are written back
slot by slot through the E02 locators, navigation labels follow the translated
titles, and every other member is copied byte for byte. Documents whose
translation equals the source text are copied verbatim, so a no-op or partial
render leaves untouched XHTML byte-identical.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import zipfile
from xml.etree import ElementTree as ET

from ..models import Segment
from .package import fragment, local_name, manifest_index, parse_container, parse_package, resolve_href
from .security import ArchiveLimits, decode_xml_text, parse_xml, read_member, validate_archive, validate_ocf_mimetype
from .segments import apply_segment_slots, element_paths
from .tokens import TOKEN
from .translation import is_epub_segment, validate_epub_translation


class EPUBRewriteError(RuntimeError):
    """Raised when a translated tree cannot be produced without breaking invariants."""


XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"
DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
OPS_NAMESPACE = "http://www.idpf.org/2007/ops"
OPF_NAMESPACE = "http://www.idpf.org/2007/opf"
NCX_NAMESPACE = "http://www.daisy.org/z3986/2005/ncx/"
XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
# ``ElementTree`` rewrites every namespace it does not know as ``ns0``/``ns1``
# (and would turn the XHTML default namespace into ``html:``), so a rewritten
# document installs the conventional EPUB prefixes for the duration of one
# serialization. Prefixes carry no meaning in XML; an unknown namespace still
# serializes to a valid document, which the ADR explicitly allows.
PREFERRED_NAMESPACES: Tuple[Tuple[str, str], ...] = (
    ("svg", "http://www.w3.org/2000/svg"),
    ("xlink", "http://www.w3.org/1999/xlink"),
    ("m", "http://www.w3.org/1998/Math/MathML"),
    ("ncx", NCX_NAMESPACE),
    ("opf", OPF_NAMESPACE),
    ("dc", DC_NAMESPACE),
    ("epub", OPS_NAMESPACE),
)
NAMESPACE_LOCK = threading.Lock()
XML_ENCODING = re.compile(r"encoding\s*=\s*(['\"])[^'\"]*\1")
CONTAINER_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "image/svg+xml"}
CONTAINER_SUFFIXES = {".xhtml", ".html", ".htm", ".svg"}
# Navigation types whose labels are page numbers or a fixed target name and must
# keep their source text (plan §3.7).
UNTRANSLATABLE_NAV_TYPES = {"page-list", "page_list"}


def root_namespace(root: Any) -> Optional[str]:
    """Namespace URI of the document element, or ``None`` when it has none."""
    tag = root.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return tag[1:].split("}", 1)[0]
    return None


def _namespace_registry(default_namespace: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    entries: List[Tuple[str, str]] = []
    if default_namespace:
        entries.append(("", default_namespace))
    for prefix, uri in PREFERRED_NAMESPACES:
        if prefix != "" and uri != default_namespace:
            entries.append((prefix, uri))
    return tuple(entries)


@contextmanager
def namespace_registry(default_namespace: Optional[str] = XHTML_NAMESPACE) -> Iterator[None]:
    """Install the document's prefixes for one serialization, then restore them."""
    with NAMESPACE_LOCK:
        snapshot = dict(ET._namespace_map)
        try:
            for prefix, uri in _namespace_registry(default_namespace):
                ET.register_namespace(prefix, uri)
            yield
        finally:
            ET._namespace_map.clear()
            ET._namespace_map.update(snapshot)


def _namespace_used_by_attribute(root: Any, uri: str) -> bool:
    needle = f"{{{uri}}}"
    return any(str(key).startswith(needle) for element in root.iter() for key in element.attrib)


def _prologue_end(text: str) -> int:
    """Index of the first real element start tag.

    A naive ``<`` search breaks on legal pre-root content: an XML declaration,
    a DOCTYPE with an internal subset, comments, or processing instructions may
    all contain ``<``/``tag``-looking text. Everything the XML prolog may hold
    is skipped here so the rewritten document keeps it verbatim.
    """
    index = 0
    length = len(text)
    while index < length:
        start = text.find("<", index)
        if start < 0:
            return length
        if text.startswith("<!--", start):
            end = text.find("-->", start + 4)
            if end < 0:
                return length
            index = end + 3
        elif text.startswith("<![CDATA[", start):
            end = text.find("]]>", start + 9)
            if end < 0:
                return length
            index = end + 3
        elif text.startswith("<!", start):
            # DOCTYPE (possibly with an internal subset) or another declaration
            depth = 0
            cursor = start + 2
            while cursor < length:
                character = text[cursor]
                if character == "[":
                    depth += 1
                elif character == "]":
                    depth = max(0, depth - 1)
                elif character == ">" and depth == 0:
                    break
                cursor += 1
            index = cursor + 1
        elif text.startswith("<?", start):
            end = text.find("?>", start + 2)
            if end < 0:
                return length
            index = end + 2
        else:
            return start
    return length


def serialize_document(
    root: Any, original: bytes, label: str, *, default_namespace: Optional[str] = None
) -> bytes:
    """Re-serialize a rewritten document, keeping its declaration and DOCTYPE.

    Whitespace, attribute order and namespace prefixes may change (ADR
    ``Preservation and rewrite invariants``); the encoding is normalized to
    UTF-8 and the XML declaration is rewritten to match it. The default
    namespace is taken from the document element itself, so a document never
    gains or loses one. A default namespace is additionally dropped when an
    *attribute* lives in it: XML default namespaces apply to elements only, so
    ``ElementTree`` would drop the prefix and silently move ``opf:role`` out of
    the OPF namespace. Such a document keeps explicit prefixes instead, which
    preserves the attribute's meaning.
    """
    text = decode_xml_text(original, label)
    prologue = XML_ENCODING.sub('encoding="utf-8"', text[: _prologue_end(text)])
    if default_namespace is None:
        default_namespace = root_namespace(root)
    if default_namespace and _namespace_used_by_attribute(root, default_namespace):
        default_namespace = None
    with namespace_registry(default_namespace):
        body = ET.tostring(root, encoding="utf-8")
    return prologue.encode("utf-8") + body


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _is_content_member(name: str, media_types: Mapping[str, str]) -> bool:
    if media_types.get(name, "").casefold() in CONTAINER_MEDIA_TYPES:
        return True
    return Path(name).suffix.casefold() in CONTAINER_SUFFIXES


def _member_kind(
    name: str,
    *,
    rootfile: str,
    navigation: Mapping[str, List[str]],
    media_types: Mapping[str, str],
) -> str:
    """Classification of one container member, independent of whether it changed."""
    if name == "mimetype":
        return "mimetype"
    if name == "META-INF/container.xml":
        return "container"
    if name == rootfile:
        return "package_document"
    if name in navigation["ncx"] or name in navigation["nav"] or name in navigation["page_maps"]:
        return "navigation_document"
    if _is_content_member(name, media_types):
        return "content_document"
    return "opaque"


def navigation_members(package: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Locate NCX, EPUB3 nav and page-map documents from the OPF manifest."""
    ncx: List[str] = []
    nav: List[str] = []
    page_maps: List[str] = []
    for item in package["manifest"]:
        properties = set(str(item["properties"]).split())
        if item["media_type"].casefold() == "application/x-dtbncx+xml":
            ncx.append(item["path"])
        if "nav" in properties:
            nav.append(item["path"])
        if "page-map" in properties or "page-map" in str(item["href"]).casefold():
            page_maps.append(item["path"])
    return {"ncx": sorted(set(ncx)), "nav": sorted(set(nav)), "page_maps": sorted(set(page_maps))}


def embedded_cover(
    package: Mapping[str, Any],
    rootfile: str,
    members: Mapping[str, Any],
    read: Callable[[str], bytes],
) -> Dict[str, Any]:
    """Identify the input's embedded cover (guide, cover-image, OPF2 meta)."""
    manifest = manifest_index(package)
    declarations: List[str] = []
    paths: Dict[str, List[str]] = {}

    def declare(path: str, source: str) -> None:
        declarations.append(source)
        paths.setdefault(path, []).append(source)

    for entry in package["guide"]:
        if str(entry["type"]).casefold() == "cover" and entry["href"]:
            declare(resolve_href(rootfile, str(entry["href"])), "guide:cover")
    for item in manifest.values():
        if "cover-image" in set(str(item["properties"]).split()):
            declare(str(item["path"]), f"manifest:{item['id']}")
    for entry in package["metadata"].get("meta", []):
        if str(entry["name"]).casefold() == "cover" and entry["value"]:
            item = manifest.get(str(entry["value"]))
            if item is not None:
                declare(str(item["path"]), f"meta:{entry['id'] or 'cover'}")
    missing = sorted(path for path in paths if path not in members)
    if missing:
        raise EPUBRewriteError(f"embedded cover target is missing: {missing}")
    return {
        "paths": sorted(paths),
        "declared_by": sorted(declarations),
        "sha256": {path: _sha256(read(path)) for path in sorted(paths)},
    }


def _label_text_nodes(element: Any) -> List[Tuple[Any, str]]:
    """Collect every non-empty text node inside a navigation label."""
    nodes: List[Tuple[Any, str]] = []
    for node in element.iter():
        if (node.text or "").strip():
            nodes.append((node, "text"))
        for child in node:
            if (child.tail or "").strip():
                nodes.append((child, "tail"))
    return nodes


def _target_key(document: str, href: str) -> Tuple[str, str]:
    return resolve_href(document, href), fragment(href)


def heading_labels(
    segments: Sequence[Segment], translations: Mapping[str, str]
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str], List[Dict[str, Any]]]:
    """Navigation labels for translated headings.

    Returns fragment keys ``(href, id)`` for headings navigation can target
    directly, document keys ``(href, source heading text)`` so a bare-document
    nav entry is synchronized from the exact heading it names - including a
    heading with no ``id``, which cannot be a fragment target but is still legal
    as a document-level one - and the ambiguous keys that could not be used
    because several headings in one document share the same source text.
    """
    fragments: Dict[Tuple[str, str], str] = {}
    candidates: Dict[Tuple[str, str], set] = {}
    ambiguous: List[Dict[str, Any]] = []
    for segment in segments:
        metadata = segment.metadata or {}
        locator = str(metadata.get("locator") or "")
        if segment.kind != "heading" or segment.id not in translations:
            continue
        label = TOKEN.sub("", translations[segment.id]).strip()
        if not label:
            continue
        href = str(metadata.get("href") or "")
        if locator.startswith("id:"):
            fragments[(href, locator[3:])] = label
        candidates.setdefault((href, segment.source_text.strip()), set()).add(label)
    documents: Dict[Tuple[str, str], str] = {}
    for key, labels in sorted(candidates.items()):
        if len(labels) == 1:
            documents[key] = next(iter(labels))
        else:
            ambiguous.append({"href": key[0], "label": key[1], "translations": sorted(labels)})
    return fragments, documents, ambiguous


def nav_anchors(root: Any) -> List[Tuple[str, str, str, Any]]:
    """``(href, full label, nav type, anchor)`` for every EPUB3 nav anchor.

    The enclosing ``<nav epub:type>`` decides the policy: ``toc`` and
    ``landmarks`` may follow a translated heading label, while ``page-list``
    entries are page numbers and must never be translated (plan §3.7).
    """
    anchors: List[Tuple[str, str, str, Any]] = []
    for container in root.iter():
        if local_name(container.tag) != "nav":
            continue
        nav_type = "toc"
        for key, value in container.attrib.items():
            if local_name(key) == "type" and str(value).strip():
                nav_type = str(value).strip().split()[0]
        for anchor in container.iter():
            if local_name(anchor.tag) == "a" and anchor.get("href"):
                anchors.append(
                    (str(anchor.get("href")), "".join(anchor.itertext()).strip(), nav_type, anchor)
                )
    return anchors


def navigation_targets(root: Any, document: str, kind: str) -> List[Tuple[str, str]]:
    """Ordered ``(href, label text)`` pairs for an NCX or EPUB3 nav document.

    A list, not a mapping: real books repeat the same href with different labels
    (a chapter and its sub-entry pointing at the same anchor), and collapsing
    those would let one desynchronized label hide behind another.
    """
    labels: List[Tuple[str, str]] = []
    if kind == "ncx":
        for nav_point in root.iter():
            if local_name(nav_point.tag) != "navPoint":
                continue
            target = ""
            text = None
            for child in nav_point:
                name = local_name(child.tag)
                if name == "content" and child.get("src"):
                    target = str(child.get("src"))
                elif name == "navLabel":
                    text = next((node for node in child if local_name(node.tag) == "text"), None)
            if target and text is not None:
                # Full text, including nested nodes: the entry's label is what a
                # reader sees, and that is the evidence a heading is matched on.
                labels.append((target, "".join(text.itertext()).strip()))
        return labels
    labels.extend((href, label) for href, label, _, _ in nav_anchors(root))
    return labels


def resolve_label(
    document: str,
    href: str,
    source_label: Optional[str],
    fragments: Mapping[Tuple[str, str], str],
    documents: Mapping[Tuple[str, str], str],
) -> Optional[str]:
    """Translation for one navigation target, or ``None`` when it has none.

    A fragment target is matched by its exact fragment; a bare document target
    is matched only against a heading whose *source label* the navigation entry
    already names, so a document-level entry is never relabelled by guesswork.
    """
    resolved, anchor = _target_key(document, href)
    if (resolved, anchor) in fragments:
        return fragments[(resolved, anchor)]
    if not anchor and source_label is not None:
        return documents.get((resolved, source_label.strip()))
    return None


def _update_ncx_labels(
    root: Any,
    document: str,
    fragments: Mapping[Tuple[str, str], str],
    documents: Mapping[Tuple[str, str], str],
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    paths = element_paths(root)
    for nav_point in root.iter():
        if local_name(nav_point.tag) != "navPoint":
            continue
        target = ""
        label_node = None
        for child in nav_point:
            name = local_name(child.tag)
            if name == "content" and child.get("src"):
                target = str(child.get("src"))
            elif name == "navLabel":
                label_node = next((node for node in child if local_name(node.tag) == "text"), None)
        if not target:
            continue
        source_label = None if label_node is None else "".join(label_node.itertext()).strip()
        value = resolve_label(document, target, source_label, fragments, documents)
        if value is None:
            continue
        if label_node is None:
            # A translated target with no navLabel text cannot be synchronized;
            # leaving it stale would silently desynchronize the navigation.
            raise EPUBRewriteError(f"NCX label for {target!r} is missing and cannot be synchronized")
        nodes = _label_text_nodes(label_node)
        if len(nodes) != 1 or len(list(label_node)):
            raise EPUBRewriteError(f"NCX label for {target!r} cannot be synchronized with its translated heading")
        if label_node.text == value:
            continue
        changes.append(
            {
                "document": document,
                "target": target,
                "from": label_node.text,
                "to": value,
                "path": paths[id(label_node)],
                "kind": "text",
            }
        )
        label_node.text = value
    return changes


def _update_nav_labels(
    root: Any,
    document: str,
    fragments: Mapping[Tuple[str, str], str],
    documents: Mapping[Tuple[str, str], str],
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    paths = element_paths(root)
    for target, source_label, nav_type, anchor in nav_anchors(root):
        if nav_type in UNTRANSLATABLE_NAV_TYPES:
            # A page-list label is a page number: translating it would corrupt
            # the page-list (plan §3.7).
            continue
        value = resolve_label(document, target, source_label, fragments, documents)
        if value is None:
            continue
        nodes = _label_text_nodes(anchor)
        if len(nodes) != 1 or len(list(anchor)):
            # Inline markup or several text nodes cannot be replaced without
            # inventing markup; refuse rather than leave navigation stale.
            raise EPUBRewriteError(f"nav label for {target!r} cannot be synchronized with its translated heading")
        node, kind = nodes[0]
        current = node.text if kind == "text" else node.tail
        if current == value:
            continue
        changes.append(
            {
                "document": document,
                "target": target,
                "from": current,
                "to": value,
                "path": paths[id(node)],
                "kind": kind,
                "nav_type": nav_type,
            }
        )
        if kind == "text":
            node.text = value
        else:
            node.tail = value
    return changes


def _metadata_fingerprint(package: Mapping[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    metadata = package["metadata"]
    return {key: [dict(entry) for entry in metadata[key]] for key in sorted(metadata)}


def _apply_metadata(root: Any, metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Apply the allowed metadata changes and record each one."""
    container = next((element for element in root.iter() if local_name(element.tag) == "metadata"), None)
    if container is None:
        raise EPUBRewriteError("OPF package has no metadata element")
    changes: List[Dict[str, Any]] = []
    language = str(metadata.get("language") or "")
    if language:
        element = next((node for node in container.iter() if local_name(node.tag) == "language"), None)
        if element is None:
            raise EPUBRewriteError("OPF metadata has no dc:language to update")
        if (element.text or "") != language:
            changes.append({"element": "dc:language", "action": "set", "from": element.text, "to": language})
            element.text = language
    title_zh = str(metadata.get("title_zh") or "")
    if title_zh:
        taken = {str(node.get("id")) for node in container.iter() if local_name(node.tag) == "title"}
        identifier = "title-zh"
        while identifier in taken:
            identifier = f"{identifier}-x"
        element = ET.SubElement(
            container, f"{{{DC_NAMESPACE}}}title", {f"{{{XML_NAMESPACE}}}lang": "zh-CN", "id": identifier}
        )
        element.text = title_zh
        changes.append({"element": "dc:title", "action": "add", "id": identifier, "lang": "zh-CN", "value": title_zh})
    return changes


def _require_separate_output(source: Path, destination: Path) -> None:
    """Refuse to render over the input container or over its directory.

    The rendered tree is replaced with ``rmtree`` + ``os.replace``, so the
    source EPUB must never live inside the destination: both an exact match and
    the source's own directory (whose removal would delete the input) are
    rejected. A tree merely nested somewhere below the source's directory is
    safe and stays allowed.
    """
    if destination == source or destination in source.parents:
        raise EPUBRewriteError(f"rendered tree must not contain the source EPUB: {destination}")


def rewrite_epub_tree(
    source: Path,
    tree_dir: Path,
    *,
    book_id: str,
    segments: Sequence[Segment],
    records: Mapping[str, Mapping[str, Any]],
    source_manifest: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    partial: bool = False,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Dict[str, Any]:
    """Write current translations into a rendered tree beside the source EPUB.

    Fails closed before writing anything when the source changed, a translation
    is missing or stale (unless ``partial``), or a stored reply does not
    reconstruct its recorded slots exactly.
    """
    source_path = source.expanduser().resolve()
    source_digest = _sha256(source_path.read_bytes())
    expected = str((source_manifest or {}).get("source_sha256") or "")
    if expected and expected != source_digest:
        raise EPUBRewriteError(f"source EPUB changed since prepare: expected {expected}, found {source_digest}")

    epub_segments = [segment for segment in segments if is_epub_segment(segment) and segment.translatable]
    destination = tree_dir.expanduser().resolve()
    _require_separate_output(source_path, destination)
    temporary = destination.with_name(f".{destination.name}.tmp")

    with zipfile.ZipFile(source_path) as archive:
        members = validate_archive(archive, limits)
        validate_ocf_mimetype(archive, members)
        rootfile = parse_container(archive, members)[0]["path"]
        package = parse_package(archive, members, rootfile)
        manifest = manifest_index(package)
        spine: Dict[str, int] = {}
        for index, itemref in enumerate(package["spine"]):
            spine.setdefault(manifest[itemref["idref"]]["path"], index)
        media_types = {item["path"]: item["media_type"] for item in package["manifest"]}
        recorded_documents = {
            str(entry["href"]): str(entry["sha256"])
            for entry in (source_manifest or {}).get("documents", [])
        }

        grouped: Dict[str, List[Segment]] = {}
        for segment in epub_segments:
            href = str((segment.metadata or {}).get("href") or "")
            if href not in members:
                raise EPUBRewriteError(f"segment {segment.id} references a document missing from the source: {href!r}")
            grouped.setdefault(href, []).append(segment)

        document_hashes = {href: _sha256(read_member(archive, members, href)) for href in grouped}
        for href, document_segments in grouped.items():
            recorded = {str((segment.metadata or {}).get("source_document_sha256") or "") for segment in document_segments}
            recorded.add(recorded_documents.get(href, document_hashes[href]))
            if recorded != {document_hashes[href]}:
                raise EPUBRewriteError(f"source document changed since prepare: {href}")

        missing: List[str] = []
        stale: List[str] = []
        translations: Dict[str, str] = {}
        for segment in epub_segments:
            row = records.get(segment.id)
            if not row:
                missing.append(segment.id)
                continue
            if row.get("status") != "completed" or row.get("source_hash") != segment.source_hash:
                stale.append(segment.id)
                continue
            translations[segment.id] = str(row.get("translated_text") or "")
        if (missing or stale) and not partial:
            raise EPUBRewriteError(f"translation coverage is not current: {len(missing)} missing, {len(stale)} stale")

        # Validate every stored reply against its recorded slots before writing.
        for segment in epub_segments:
            if segment.id in translations:
                validate_epub_translation(segment, translations[segment.id])

        changed: Dict[str, bytes] = {}
        applied_slots: Dict[str, List[Dict[str, Any]]] = {}
        navigation = navigation_members(package)
        for href in sorted(grouped):
            applied = [segment for segment in grouped[href] if segment.id in translations]
            if not applied or all(translations[segment.id] == segment.source_text for segment in applied):
                continue
            data = read_member(archive, members, href)
            root = parse_xml(data, label=href, limits=limits)
            paths = element_paths(root)
            for segment in applied:
                apply_segment_slots(root, segment, translations[segment.id], paths)
            # The exact slot locations this rewrite was allowed to change; the
            # validator uses them to hold every other DOM fact unchanged.
            applied_slots[href] = [
                {
                    "segment_id": segment.id,
                    "path": str(slot["path"]),
                    "kind": str(slot["kind"]),
                    "attribute": str(slot["attribute"]),
                }
                for segment in applied
                for slot in (segment.metadata or {}).get("slots", [])
                if translations[segment.id] != segment.source_text
            ]
            rendered = serialize_document(root, data, href)
            if rendered != data:
                changed[href] = rendered

        fragments, document_labels, ambiguous_headings = heading_labels(epub_segments, translations)
        label_changes: Dict[str, List[Dict[str, Any]]] = {"ncx": [], "nav": []}
        unmatched_targets: List[Dict[str, Any]] = []
        for kind, updater in (("ncx", _update_ncx_labels), ("nav", _update_nav_labels)):
            for path in navigation[kind]:
                if path in changed:
                    raise EPUBRewriteError(f"navigation document also carries translated segments: {path}")
                data = read_member(archive, members, path)
                root = parse_xml(data, label=path, limits=limits)
                for target, label in navigation_targets(root, path, kind):
                    if resolve_label(path, target, label, fragments, document_labels) is None:
                        # A target without a translated heading of its own (a
                        # document-level landmark label such as "Start") keeps
                        # its source label; it is recorded so the report never
                        # claims a full label synchronization.
                        unmatched_targets.append({"document": path, "type": kind, "target": target, "label": label})
                changes = updater(root, path, fragments, document_labels)
                if changes:
                    changed[path] = serialize_document(root, data, path)
                label_changes[kind].extend(changes)

        metadata_report: Dict[str, Any] = {"preserved": _metadata_fingerprint(package), "changes": []}
        if metadata:
            data = read_member(archive, members, rootfile)
            root = parse_xml(data, label=rootfile, limits=limits)
            metadata_report["changes"] = _apply_metadata(root, metadata)
            if metadata_report["changes"]:
                changed[rootfile] = serialize_document(root, data, rootfile)

        cover = embedded_cover(package, rootfile, members, lambda name: read_member(archive, members, name))

        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        resources: List[Dict[str, Any]] = []
        for name in sorted(members):
            payload = changed.get(name)
            if payload is None:
                payload = read_member(archive, members, name)
                source_digest_member = _sha256(payload)
            else:
                source_digest_member = document_hashes.get(name) or _sha256(read_member(archive, members, name))
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            resources.append(
                {
                    "path": name,
                    "kind": _member_kind(
                        name, rootfile=rootfile, navigation=navigation, media_types=media_types
                    ),
                    "source_sha256": source_digest_member,
                    "output_sha256": _sha256(payload),
                    "changed": name in changed,
                }
            )

    shutil.rmtree(destination, ignore_errors=True)
    os.replace(temporary, destination)

    digests = {resource["path"]: resource["output_sha256"] for resource in resources}
    documents: List[Dict[str, Any]] = []
    for href in sorted(grouped):
        document_segments = grouped[href]
        applied = [segment for segment in document_segments if segment.id in translations]
        documents.append(
            {
                "href": href,
                "spine_index": spine.get(href),
                "segments": {
                    "text": sum(segment.kind != "attribute" for segment in document_segments),
                    "attribute": sum(segment.kind == "attribute" for segment in document_segments),
                },
                "translated_segments": len(applied),
                "slots": sum(len(segment.metadata.get("slots") or []) for segment in applied),
                "applied_slots": applied_slots.get(href, []),
                "source_sha256": document_hashes[href],
                "output_sha256": digests.get(href, document_hashes[href]),
                "changed": href in changed,
            }
        )

    return {
        "schema_version": 1,
        "adapter": "epub_native_v1",
        "book_id": book_id,
        "partial": bool(partial),
        "source": {
            "path": str(source_path),
            "sha256": source_digest,
            "rootfile": rootfile,
            "package_version": package["version"],
            "zip_entries": len(members),
        },
        "tree": {"path": str(destination), "members": len(members), "changed": sorted(changed)},
        "gate": {
            "translatable_segments": len(epub_segments),
            "translated_segments": len(translations),
            "missing": sorted(missing),
            "stale": sorted(stale),
        },
        "documents": documents,
        "navigation": {
            "labels": label_changes,
            "heading_targets": len(fragments),
            "document_targets": len(document_labels),
            "untranslated_targets": unmatched_targets,
            "ambiguous_headings": ambiguous_headings,
        },
        "metadata": metadata_report,
        "resources": resources,
        "embedded_cover": cover,
    }
