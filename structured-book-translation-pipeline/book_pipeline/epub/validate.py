"""Structural, resource and link validation for a rendered EPUB.

Every check compares the source container with the rendered tree and fails
closed: opaque resources must be byte-identical, rewritten documents must keep
their DOM skeleton, and the package, spine, navigation targets, ids, tables and
footnote graph must not move. EPUBCheck results are reported exactly as found;
a missing tool is ``validation_incomplete``, never a fabricated pass.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import zipfile

from .inventory import document_data, inspect_tree
from .package import local_name, parse_container, parse_navigation, parse_package
from .rewrite import (
    UNTRANSLATABLE_NAV_TYPES,
    XML_NAMESPACE,
    nav_anchors,
    navigation_members,
    navigation_targets,
)
from .security import ArchiveLimits, DirectoryArchive, directory_members, parse_xml, read_member, validate_archive, validate_ocf_mimetype

CONTENT_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "image/svg+xml"}
CONTENT_SUFFIXES = {".xhtml", ".html", ".htm", ".svg"}
REFERENCE_ATTRIBUTES = {"href", "src"}


class EPUBValidationError(RuntimeError):
    """Raised when a rendered tree or packed EPUB breaks an EPUB invariant."""


def _sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def _is_content(path: str, media_types: Mapping[str, str]) -> bool:
    if media_types.get(path, "").casefold() in CONTENT_MEDIA_TYPES:
        return True
    return Path(path).suffix.casefold() in CONTENT_SUFFIXES


def document_shape(root: Any, allowed_slots: Optional[Set[Tuple[str, str, str]]] = None) -> Dict[str, Any]:
    """Canonical structural fingerprint of one content document.

    Only the exact DOM locations a rewrite recorded as written slots may differ
    between source and output. Every other fact is captured verbatim: the
    element sequence, each element's name and namespace-qualified attributes,
    text and tail values outside written slots (so untranslated ``title``,
    ``script``, ``style``, ``code``, ``pre``, ``math`` and plain-``div`` text
    must survive untouched), ids, references, tables, footnotes and ``sup``
    counts. ``allowed_slots`` holds ``(element path, slot kind, attribute)``.
    """
    allowed = allowed_slots or set()

    def permitted(path: str, kind: str, attribute: str = "") -> bool:
        return (path, kind, attribute) in allowed

    elements: List[str] = []
    attributes: List[List[List[str]]] = []
    text: List[List[str]] = []
    ids: List[str] = []
    references: List[List[str]] = []
    tables: List[List[List[List[int]]]] = []
    footnotes: List[str] = []
    noterefs: List[str] = []
    superscripts = 0
    text_nodes = 0

    # Element paths must be computed the same way the rewrite computed them.
    from .segments import element_paths

    paths = element_paths(root)
    for element in root.iter():
        path = paths[id(element)]
        name = local_name(element.tag)
        # The qualified tag, not the local name: an element must not silently
        # move into (or out of) a namespace during serialization.
        elements.append(str(element.tag))
        entries = []
        for key, value in element.attrib.items():
            # Compare the namespace URI, never the prefix: serialization may
            # rename prefixes, but an attribute must not leave its namespace.
            # Slot identity is namespace-qualified too, so an allowed unqualified
            # ``title`` slot never permits a *namespaced* ``x:title``.
            if permitted(path, "attribute", str(key)):
                continue
            entries.append([str(key), str(value)])
            if local_name(key) in REFERENCE_ATTRIBUTES and str(value):
                references.append([name, str(key), str(value)])
        attributes.append(sorted(entries))
        text.append(
            [
                "" if permitted(path, "text") else str(element.text or ""),
                *[
                    f"{local_name(child.tag)}:{'' if permitted(paths[id(child)], 'tail') else child.tail or ''}"
                    for child in element
                ],
            ]
        )
        identifier = str(element.get("id") or "")
        if identifier:
            ids.append(identifier)
        if name == "sup":
            superscripts += 1
        # Count only text the rewrite was *not* allowed to touch. A permitted
        # slot may legitimately gain or lose text -- a translation can merge two
        # source nodes into one, and a small-caps run's leading capital moves
        # between the preceding tail and the ``<small>`` text -- so a global
        # node count would fail a rewrite whose every element, attribute, id and
        # reference is in fact unchanged. The ``text`` array above applies the
        # same rule to the values themselves.
        if (element.text or "").strip() and not permitted(path, "text"):
            text_nodes += 1
        for child in element:
            if (child.tail or "").strip() and not permitted(paths[id(child)], "tail"):
                text_nodes += 1
        markers = " ".join(
            str(value) for key, value in element.attrib.items() if local_name(key) in {"type", "role", "class"}
        ).casefold()
        if identifier and ("footnote" in markers or "endnote" in markers):
            footnotes.append(identifier)
        if "noteref" in markers:
            anchor = str(element.get("href") or "")
            if anchor:
                noterefs.append(anchor)
    for table in root.iter():
        if local_name(table.tag) != "table":
            continue
        rows: List[List[List[int]]] = []
        for row in table.iter():
            if local_name(row.tag) != "tr":
                continue
            rows.append(
                [
                    [int(cell.get("rowspan") or 1), int(cell.get("colspan") or 1), len(cell)]
                    for cell in row
                    if local_name(cell.tag) in {"td", "th"}
                ]
            )
        tables.append(rows)
    return {
        "elements": elements,
        "attributes": attributes,
        "text": text,
        "ids": sorted(ids),
        "references": sorted(references),
        "tables": tables,
        "footnotes": sorted(footnotes),
        "noterefs": sorted(noterefs),
        "superscripts": superscripts,
        "text_nodes": text_nodes,
    }


def _shape_differences(source: Mapping[str, Any], output: Mapping[str, Any]) -> List[Dict[str, Any]]:
    differences = []
    for key in sorted(set(source) | set(output)):
        if source.get(key) != output.get(key):
            differences.append({"field": key, "source": source.get(key), "output": output.get(key)})
    return differences


def _document_shapes(
    archive: Any,
    members: Mapping[str, Any],
    package: Mapping[str, Any],
    allowed_by_document: Optional[Mapping[str, Set[Tuple[str, str, str]]]] = None,
    navigation_documents: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    """Shape every XML document a rewrite may touch.

    Content documents plus the navigation/package-adjacent XML (NCX, EPUB3 nav,
    page-map) are fingerprinted. Navigation documents carry no translated slots
    of their own; the only locations a rewrite may change there are the recorded
    label slots, which arrive through ``allowed_by_document``.
    """
    shapes: Dict[str, Dict[str, Any]] = {}
    allowed_by_document = allowed_by_document or {}
    targets = [
        str(item["path"]) for item in package["manifest"]
        if str(item["media_type"]).casefold() in CONTENT_MEDIA_TYPES or _is_content(str(item["path"]), {})
    ]
    targets.extend(str(path) for path in navigation_documents)
    for path in dict.fromkeys(targets):
        if path not in members:
            raise EPUBValidationError(f"XML document is missing from the container: {path}")
        shapes[path] = document_shape(
            parse_xml(read_member(archive, members, path), label=path),
            allowed_by_document.get(path),
        )
    return shapes


def _package_facts(package: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = [
        [str(item["id"]), str(item["href"]), str(item["media_type"]), str(item["properties"]), str(item["fallback"])]
        for item in package["manifest"]
    ]
    return {
        "version": package["version"],
        "unique_identifier": package["unique_identifier"],
        "manifest": manifest,
        "spine": [[str(entry["idref"]), str(entry["linear"])] for entry in package["spine"]],
        "guide": [[str(entry["type"]), str(entry["href"])] for entry in package["guide"]],
    }


def _navigation_targets(navigation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "ncx": [[str(entry["source"]), str(entry["href"])] for entry in navigation["ncx"]],
        "nav": {
            key: [[str(entry["source"]), str(entry["href"])] for entry in entries]
            for key, entries in navigation["nav"].items()
        },
        "page_maps": list(navigation["page_maps"]),
        "page_map_links": [[str(entry["source"]), str(entry["href"])] for entry in navigation["page_map_links"]],
    }


def _metadata_entries(package: Mapping[str, Any]) -> List[List[str]]:
    entries = []
    for key in sorted(package["metadata"]):
        for entry in package["metadata"][key]:
            entries.append([key, str(entry["value"]), str(entry["id"]), str(entry["property"]), str(entry["name"])])
    return entries


def _attribute_names(root: Any) -> Counter:
    """Namespace-qualified attribute names, with multiplicity.

    Serialization may rename prefixes, but an attribute must not leave its
    namespace: ``opf:role`` becoming ``role`` changes the qualified name and is
    exactly the corruption this counts. Used for package and navigation XML,
    which carry no translated slots.
    """
    return Counter(str(key) for element in root.iter() for key in element.attrib)


def navigation_labels(root: Any, document: str, kind: str) -> List[Tuple[str, str, str]]:
    """Ordered ``(nav type, href, full label)`` triples for a nav document."""
    if kind == "ncx":
        return [("ncx", href, label) for href, label in navigation_targets(root, document, kind)]
    return [(nav_type, href, label) for href, label, nav_type, _ in nav_anchors(root)]


def _label_differences(
    source_labels: Sequence[Tuple[str, str, str]],
    output_labels: Sequence[Tuple[str, str, str]],
    recorded: Sequence[Mapping[str, Any]],
    document: str,
) -> List[Dict[str, Any]]:
    """Label changes that the rewrite manifest did not record.

    Compared as an ordered sequence, so a duplicated target keeps one entry per
    occurrence and a mutation to any single label is visible. A page-list label
    is a page number: it may neither be translated nor mutated.
    """
    expected = {
        str(change["target"]): str(change["to"])
        for change in recorded
        if str(change.get("document")) == document
    }
    if len(source_labels) != len(output_labels):
        return [
            {
                "document": document,
                "detail": "navigation entry count changed",
                "source": len(source_labels),
                "output": len(output_labels),
            }
        ]
    differences: List[Dict[str, Any]] = []
    for index, ((source_type, source_href, source_label), (output_type, output_href, output_label)) in enumerate(
        zip(source_labels, output_labels)
    ):
        if (source_type, source_href, source_label) == (output_type, output_href, output_label):
            continue
        if (source_type, source_href) != (output_type, output_href):
            differences.append(
                {
                    "index": index,
                    "document": document,
                    "detail": "navigation target changed",
                    "source": [source_type, source_href],
                    "output": [output_type, output_href],
                }
            )
            continue
        if source_type in UNTRANSLATABLE_NAV_TYPES:
            differences.append(
                {
                    "index": index,
                    "document": document,
                    "detail": "page-list label must not change",
                    "source": source_label,
                    "output": output_label,
                }
            )
            continue
        if expected.get(source_href) == output_label:
            continue
        differences.append(
            {
                "index": index,
                "document": document,
                "target": source_href,
                "from": source_label,
                "to": output_label,
                "recorded": expected.get(source_href),
            }
        )
    return differences


def _metadata_differences(
    source_package: Mapping[str, Any], output_package: Mapping[str, Any], recorded: Sequence[Mapping[str, Any]]
) -> Dict[str, List[List[str]]]:
    """Metadata entries added or removed outside the recorded change list."""
    source_entries = {tuple(entry) for entry in _metadata_entries(source_package)}
    output_entries = {tuple(entry) for entry in _metadata_entries(output_package)}
    allowed_removed: set = set()
    allowed_added: set = set()
    for change in recorded:
        element = str(change.get("element") or "")
        key = element.split(":", 1)[-1]
        action = str(change.get("action") or "")
        if action == "set":
            allowed_removed |= {entry for entry in source_entries if entry[0] == key and entry[1] == str(change.get("from"))}
            allowed_added |= {entry for entry in output_entries if entry[0] == key and entry[1] == str(change.get("to"))}
        elif action == "add":
            allowed_added |= {
                entry
                for entry in output_entries
                if entry[0] == key and entry[1] == str(change.get("value")) and str(change.get("id")) in {"", entry[2]}
            }
    return {
        "removed": sorted(list(entry) for entry in source_entries - output_entries - allowed_removed),
        "added": sorted(list(entry) for entry in output_entries - source_entries - allowed_added),
    }


def _link_diff(
    source_graph: Mapping[str, Any], output_graph: Mapping[str, Any]
) -> Dict[str, Any]:
    def key(entry: Mapping[str, Any]) -> Tuple[str, ...]:
        return (str(entry["source"]), str(entry["href"]), str(entry.get("element", "")), str(entry.get("attribute", "")))

    source_links = {key(entry) for entry in source_graph["links"]}
    output_links = {key(entry) for entry in output_graph["links"]}
    return {
        "links": {"source": len(source_links), "output": len(output_links)},
        "added": sorted(list(entry) for entry in output_links - source_links),
        "removed": sorted(list(entry) for entry in source_links - output_links),
        "internal_links": {
            "source": len(source_graph["internal_links"]),
            "output": len(output_graph["internal_links"]),
        },
        "broken_source": source_graph["broken_links"],
        "broken_output": output_graph["broken_links"],
        "duplicate_ids_source": source_graph["duplicate_ids"],
        "duplicate_ids_output": output_graph["duplicate_ids"],
    }


def new_broken_links(
    source_link_diff: Mapping[str, Any], output_broken: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Broken links the rewrite introduced, ignoring pre-existing source damage."""
    existing = {
        (str(entry["source"]), str(entry["href"]))
        for entry in source_link_diff["broken_source"]
    }
    return [
        dict(entry)
        for entry in output_broken
        if (str(entry["source"]), str(entry["href"])) not in existing
    ]


def validate_rendered_tree(
    source: Path,
    tree_dir: Path,
    *,
    rewrite_report: Optional[Mapping[str, Any]] = None,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Dict[str, Any]:
    """Compare a rendered tree with its source container, fail-closed by report.

    Returns the report (``status`` ``PASS`` or ``FAIL`` with ``failures``)
    instead of raising, so a failing render still leaves machine evidence; the
    caller must refuse to repackage a non-``PASS`` report.
    """
    source_path = source.expanduser().resolve()
    root = tree_dir.expanduser().resolve()
    failures: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}

    members_tree = directory_members(root)
    tree = DirectoryArchive(root)
    validate_ocf_mimetype(tree, members_tree)

    with zipfile.ZipFile(source_path) as archive:
        members_source = validate_archive(archive, limits)
        validate_ocf_mimetype(archive, members_source)

        added = sorted(set(members_tree) - set(members_source))
        removed = sorted(set(members_source) - set(members_tree))
        if added or removed:
            # Deeper comparisons assume a closed member set: a missing manifest
            # target would abort inside the package parser without evidence.
            return {
                "schema_version": 1,
                "status": "FAIL",
                "source_path": str(source_path),
                "tree_path": str(root),
                "checks": {"mimetype": {"first": True, "stored": True}, "members": {"added": added, "removed": removed}},
                "failures": [{"check": "members", "added": added, "removed": removed}],
                "aborted_after": "members",
                "structural_diff": {"documents": [], "differences": []},
                "resource_diff": {"opaque_changed": removed, "added": added, "removed": removed},
                "link_diff": {},
            }

        if read_member(archive, members_source, "mimetype") != read_member(tree, members_tree, "mimetype"):
            failures.append({"check": "mimetype", "detail": "mimetype bytes changed"})
        checks["mimetype"] = {"first": True, "stored": True, "bytes": "application/epub+zip"}

        rootfiles_source = parse_container(archive, members_source)
        rootfiles_tree = parse_container(tree, members_tree)
        if rootfiles_source != rootfiles_tree:
            failures.append({"check": "container", "source": rootfiles_source, "output": rootfiles_tree})

        source_package = parse_package(archive, members_source, rootfiles_source[0]["path"])
        output_package = parse_package(tree, members_tree, rootfiles_tree[0]["path"])
        source_facts = _package_facts(source_package)
        output_facts = _package_facts(output_package)
        package_differences = [
            {"field": key, "source": source_facts[key], "output": output_facts[key]}
            for key in sorted(source_facts)
            if source_facts[key] != output_facts[key]
        ]
        if package_differences:
            failures.append({"check": "package", "differences": package_differences})
        checks["package"] = {
            "version_preserved": source_facts["version"] == output_facts["version"],
            "manifest_items": len(source_package["manifest"]),
            "spine_items": len(source_package["spine"]),
            "guide_references": len(source_package["guide"]),
        }
        if source_facts["version"] != output_facts["version"]:
            failures.append({"check": "package_version", "detail": "input package version must be preserved"})

        source_navigation = parse_navigation(archive, members_source, source_package)
        output_navigation = parse_navigation(tree, members_tree, output_package)
        source_targets = _navigation_targets(source_navigation)
        output_targets = _navigation_targets(output_navigation)
        if source_targets != output_targets:
            failures.append({"check": "navigation_targets", "source": source_targets, "output": output_targets})
        declared_changed = set(str(name) for name in (rewrite_report or {}).get("tree", {}).get("changed", []))
        navigation = navigation_members(source_package)
        recorded_labels = ((rewrite_report or {}).get("navigation", {}).get("labels") or {})
        label_differences: List[Dict[str, Any]] = []
        for kind in ("ncx", "nav"):
            for path in navigation[kind]:
                source_labels = navigation_labels(
                    parse_xml(read_member(archive, members_source, path), label=path), path, kind
                )
                output_labels = navigation_labels(
                    parse_xml(read_member(tree, members_tree, path), label=path), path, kind
                )
                label_differences.extend(
                    _label_differences(source_labels, output_labels, recorded_labels.get(kind) or [], path)
                )
        if label_differences:
            failures.append({"check": "navigation_labels", "differences": label_differences})
        checks["navigation"] = {
            "ncx_documents": navigation["ncx"],
            "nav_documents": navigation["nav"],
            "page_maps": navigation["page_maps"],
            "ncx_targets": len(source_targets["ncx"]),
            "nav_targets": sum(len(entries) for entries in source_targets["nav"].values()),
            "label_changes": recorded_labels,
            "label_differences": label_differences,
        }
        if source_targets["ncx"] != output_targets["ncx"]:
            failures.append({"check": "ncx_targets", "detail": "NCX target hrefs changed"})
        if source_targets["nav"] != output_targets["nav"]:
            failures.append({"check": "nav_targets", "detail": "EPUB3 nav hrefs changed"})

        media_types = {str(item["path"]): str(item["media_type"]) for item in source_package["manifest"]}
        # Package and navigation XML carry no translated slots. A navigation
        # document may only be relabelled, so its attribute names must match
        # exactly; the package document may gain the attributes of a recorded
        # metadata addition (``id``/``xml:lang``) but nothing else. Either way an
        # attribute must never lose its namespace.
        recorded_metadata = (rewrite_report or {}).get("metadata", {}).get("changes", [])
        namespace_documents = [source_package["path"]] + navigation["ncx"] + navigation["nav"] + navigation["page_maps"]
        namespace_differences: List[Dict[str, Any]] = []
        for path in dict.fromkeys(namespace_documents):
            source_names = _attribute_names(parse_xml(read_member(archive, members_source, path), label=path))
            output_names = _attribute_names(parse_xml(read_member(tree, members_tree, path), label=path))
            missing = sorted((source_names - output_names).elements())
            added = sorted((output_names - source_names).elements())
            if path == source_package["path"] and recorded_metadata:
                allowed_added = {f"{{{XML_NAMESPACE}}}lang", "id"}
                added = [name for name in added if name not in allowed_added]
            if missing or added:
                namespace_differences.append({"path": path, "missing_attribute_names": missing, "added_attribute_names": added})
        if namespace_differences:
            failures.append({"check": "xml_attributes", "documents": namespace_differences})
        checks["xml_attributes"] = {"documents": len(list(dict.fromkeys(namespace_documents)))}

        allowed_by_document = {
            str(document["href"]): {
                (str(slot["path"]), str(slot["kind"]), str(slot["attribute"]))
                for slot in document.get("applied_slots") or []
            }
            for document in (rewrite_report or {}).get("documents", [])
        }
        for kind in ("ncx", "nav"):
            for change in recorded_labels.get(kind) or []:
                allowed_by_document.setdefault(str(change["document"]), set()).add(
                    (str(change["path"]), str(change["kind"]), "")
                )
        navigation_documents = navigation["ncx"] + navigation["nav"] + navigation["page_maps"]
        source_shapes = _document_shapes(
            archive, members_source, source_package, allowed_by_document, navigation_documents
        )
        output_shapes = _document_shapes(
            tree, members_tree, output_package, allowed_by_document, navigation_documents
        )
        structural_differences: List[Dict[str, Any]] = []
        for path in sorted(set(source_shapes) | set(output_shapes)):
            differences = _shape_differences(source_shapes.get(path, {}), output_shapes.get(path, {}))
            if differences:
                structural_differences.append({"path": path, "differences": differences})
        if structural_differences:
            failures.append({"check": "dom_structure", "documents": [entry["path"] for entry in structural_differences]})
        checks["dom"] = {
            "documents": len(source_shapes),
            "tables": sum(len(shape["tables"]) for shape in source_shapes.values()),
            "cells": sum(len(row) for shape in source_shapes.values() for table in shape["tables"] for row in table),
            "noterefs": sum(len(shape["noterefs"]) for shape in source_shapes.values()),
            "footnotes": sum(len(shape["footnotes"]) for shape in source_shapes.values()),
            "superscripts": sum(shape["superscripts"] for shape in source_shapes.values()),
        }

        source_ids, source_documents = document_data(archive, members_source, source_package)
        output_ids, output_documents = document_data(tree, members_tree, output_package)
        if source_ids != output_ids:
            differences = sorted(
                path for path in set(source_ids) | set(output_ids) if source_ids.get(path) != output_ids.get(path)
            )
            failures.append({"check": "document_ids", "documents": differences})
        link_diff = _link_diff(source_documents["graph"], output_documents["graph"])
        introduced = new_broken_links(link_diff, link_diff["broken_output"])
        if introduced:
            failures.append({"check": "broken_links", "links": introduced})
        if link_diff["duplicate_ids_output"] != link_diff["duplicate_ids_source"]:
            failures.append({"check": "duplicate_ids", "source": link_diff["duplicate_ids_source"], "output": link_diff["duplicate_ids_output"]})

        resource_diff: Dict[str, Any] = {
            "opaque_unchanged": 0,
            "opaque_changed": [],
            "content_changed": [],
            "content_unchanged": [],
            "package_changed": [],
            "navigation_changed": [],
            "added": added,
            "removed": removed,
        }
        for name in sorted(set(members_source) & set(members_tree)):
            source_bytes = read_member(archive, members_source, name)
            output_bytes = read_member(tree, members_tree, name)
            identical = source_bytes == output_bytes
            if name == "mimetype":
                continue
            if name == source_package["path"]:
                if not identical:
                    resource_diff["package_changed"].append(name)
                    if name not in declared_changed:
                        failures.append({"check": "declared_change", "path": name, "detail": "package changed without a declaration"})
                continue
            if name in navigation["ncx"] or name in navigation["nav"] or name in navigation["page_maps"]:
                if not identical:
                    resource_diff["navigation_changed"].append(name)
                    if name not in declared_changed:
                        failures.append({"check": "declared_change", "path": name, "detail": "navigation changed without a declaration"})
                continue
            if _is_content(name, media_types):
                if identical:
                    resource_diff["content_unchanged"].append(name)
                    if name in declared_changed:
                        failures.append({"check": "declared_change", "path": name, "detail": "declared as rewritten but byte-identical"})
                else:
                    resource_diff["content_changed"].append(name)
                    if name not in declared_changed:
                        failures.append({"check": "declared_change", "path": name, "detail": "changed without a rewrite manifest declaration"})
                continue
            if identical:
                resource_diff["opaque_unchanged"] += 1
            else:
                resource_diff["opaque_changed"].append(name)
        if resource_diff["opaque_changed"]:
            failures.append({"check": "opaque_resources", "paths": resource_diff["opaque_changed"]})

        recorded_metadata = (rewrite_report or {}).get("metadata", {}).get("changes", [])
        metadata_diff = _metadata_differences(source_package, output_package, recorded_metadata)
        if metadata_diff["removed"] or metadata_diff["added"]:
            failures.append({"check": "metadata", **metadata_diff})
        checks["metadata"] = {
            "recorded_changes": recorded_metadata,
            **metadata_diff,
        }

    try:
        inspect_tree(root, limits=limits)
        checks["reinspection"] = {"status": "PASS"}
    except Exception as error:  # noqa: BLE001 - the report must record the exact failure
        checks["reinspection"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
        failures.append({"check": "reinspection", "detail": checks["reinspection"]["error"]})

    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "source_path": str(source_path),
        "tree_path": str(root),
        "checks": checks,
        "failures": failures,
        "structural_diff": {
            "documents": [
                {
                    "path": path,
                    "changed": path in declared_changed,
                    "elements": len(source_shapes[path]["elements"]),
                    "ids": len(source_shapes[path]["ids"]),
                    "references": len(source_shapes[path]["references"]),
                    "cells": sum(len(row) for table in source_shapes[path]["tables"] for row in table),
                }
                for path in sorted(source_shapes)
            ],
            "differences": structural_differences,
        },
        "resource_diff": resource_diff,
        "link_diff": link_diff,
    }


def _epubcheck_version(tool: str) -> str:
    if tool.casefold().endswith(".jar"):
        from ..publish import java_executable

        java = java_executable()
        if java is None:
            return ""
        command = [java, "-jar", tool, "--version"]
    else:
        command = [tool, "--version"]
    try:
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (completed.stdout or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def epubcheck_signature(result: Mapping[str, Any]) -> Dict[str, int]:
    """Count EPUBCheck messages by ``id/severity`` for baseline comparison.

    Comparing by message identity rather than by location keeps the comparison
    meaningful across a rewrite: paths and line numbers legitimately move, while
    a *new* class of error is exactly what the baseline policy must catch.
    """
    assessment = result.get("assessment") or {}
    signature: Dict[str, int] = {}
    for message in assessment.get("messages") or []:
        key = f"{message.get('severity')}:{message.get('id')}"
        signature[key] = signature.get(key, 0) + 1
    return signature


def new_epubcheck_messages(
    baseline: Mapping[str, Any], output: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Messages the output introduced, split into errors and non-errors.

    The ADR forbids the output from adding *errors*; a new warning or usage
    notice is recorded but does not by itself make a publication non-conformant
    (EPUBCheck itself still exits 0 for those). Keeping the two apart avoids
    both a fabricated pass on errors and a false failure on a benign notice.
    """
    before = epubcheck_signature(baseline)
    after = epubcheck_signature(output)
    errors: List[Dict[str, Any]] = []
    notices: List[Dict[str, Any]] = []
    for key, count in sorted(after.items()):
        extra = count - before.get(key, 0)
        if extra <= 0:
            continue
        severity, _, message_id = key.partition(":")
        entry = {"severity": severity, "id": message_id, "count": extra}
        (errors if severity in {"FATAL", "ERROR"} else notices).append(entry)
    return errors, notices


def epubcheck_verdict(baseline: Mapping[str, Any], output: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the baseline-invalid policy (ADR ``Baseline-invalid policy``).

    Reports a status; it never raises, so the caller decides whether the policy
    outcome blocks a deliverable.

    Severity policy, matching the ADR sentence "the output must not add errors"
    and the plan's invariant that a deliverable adds no new errors:

    * output EPUBCheck-green -> ``PASS``
    * input invalid, output adds no new errors -> ``PRESERVED_BASELINE_INVALID_NEEDS_REVIEW``
    * output adds FATAL/ERROR messages -> ``REGRESSED`` (never a pass)
    * tool or runtime missing/unverified -> ``validation_incomplete``

    A newly introduced WARNING or USAGE notice is recorded in ``new_notices``
    but does not fail the publication: those are not conformance errors and
    EPUBCheck itself still exits 0 for them, so failing on them would reject
    publications the official checker calls valid.
    """
    output_assessment = output.get("assessment") or {}
    baseline_assessment = baseline.get("assessment") or {}
    if output.get("dependency_missing") or baseline.get("dependency_missing"):
        return {
            "status": "validation_incomplete",
            "dependency_missing": output.get("dependency_missing") or baseline.get("dependency_missing"),
        }
    if not output_assessment or not baseline_assessment:
        return {"status": "validation_incomplete", "dependency_missing": "epubcheck"}
    introduced, notices = new_epubcheck_messages(baseline, output)
    clean = output_assessment.get("fatal", 0) == 0 and output_assessment.get("error", 0) == 0
    baseline_clean = baseline_assessment.get("fatal", 0) == 0 and baseline_assessment.get("error", 0) == 0
    if introduced:
        status = "REGRESSED"
    elif clean:
        status = "PASS"
    elif baseline_clean:
        status = "REGRESSED"
    else:
        status = "PRESERVED_BASELINE_INVALID_NEEDS_REVIEW"
    return {
        "status": status,
        "baseline": {key: baseline_assessment.get(key) for key in ("fatal", "error", "warning", "usage")},
        "output": {key: output_assessment.get(key) for key in ("fatal", "error", "warning", "usage")},
        "new_messages": introduced,
        "new_notices": notices,
    }


def run_epubcheck(epub_path: Path, *, config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Run the configured EPUBCheck and report exactly what happened.

    A missing tool is ``validation_incomplete`` with ``dependency_missing`` set;
    it is never reported as a pass.
    """
    from ..publish import (
        _epubcheck_discover,
        _publish_settings,
        _run_epubcheck,
        pinned_epubcheck_sha256,
    )

    epub_path = Path(epub_path).expanduser().resolve()
    settings = _publish_settings(dict(config or {}))
    if not settings["epubcheck"]["enabled"]:
        return {"validation": "skipped"}
    tool = _epubcheck_discover(settings)
    if tool is None:
        return {
            "validation": "validation_incomplete",
            "dependency_missing": "epubcheck",
            "detail": "no epubcheck executable or jar found; install EPUBCheck or set publish.epubcheck.path",
        }

    def incomplete(detail: str, source: Mapping[str, Any]) -> Dict[str, Any]:
        """Report missing evidence, keeping the explanation as ``detail``.

        The raw result carries its own (often empty) ``detail``, so the spread
        comes first and the verdict fields are applied after it - otherwise the
        explanation would be overwritten by that empty value.
        """
        return {
            **{key: value for key, value in source.items() if key != "assessment"},
            "validation": "validation_incomplete",
            "dependency_missing": str(source.get("dependency_missing") or "epubcheck"),
            "detail": detail,
        }

    result = _run_epubcheck(
        epub_path, tool, epub_path.parent, pinned_sha256=pinned_epubcheck_sha256(settings),
    )
    if result.get("dependency_missing") or result.get("exit_code") is None:
        return incomplete(str(result.get("detail") or "the checker could not be run"), result)
    # Defence in depth: a verdict is only ever derived from a structured
    # EPUBCheck assessment. Without one there is no evidence the artifact was
    # validated, whatever the exit code says.
    assessment = result.get("assessment")
    if not isinstance(assessment, dict) or "checker_version" not in assessment:
        return incomplete("the checker produced no usable EPUBCheck assessment", result)
    # Evidence provenance: the assessment must be about the artifact we asked
    # about. EPUBCheck echoes the path it was given *relative to its working
    # directory*, so it is resolved against the run directory before comparing.
    checked = str(assessment.get("checked_path") or "")
    if not checked:
        # Provenance is not optional: an assessment that does not name the file
        # it is about cannot support a verdict, however clean it looks.
        return incomplete("the assessment does not say which file was checked", result)
    candidate = Path(checked)
    if not candidate.is_absolute():
        candidate = epub_path.parent / candidate
    if candidate.resolve() != epub_path.resolve():
        return incomplete(f"the assessment is about {checked!r}, not {epub_path}", result)
    # A non-zero exit with no message at all means the checker analysed nothing
    # (an unreadable or missing file), never that the artifact is clean.
    total = sum(int(assessment.get(key) or 0) for key in ("fatal", "error", "warning", "usage"))
    if result["exit_code"] != 0 and total == 0:
        return incomplete("the checker failed without analysing the artifact", result)
    version = str(assessment.get("checker_version") or "") or _epubcheck_version(tool)
    outcome = "valid" if result["exit_code"] == 0 else "invalid"
    # The raw result comes first so that the computed verdict and version can
    # never be replaced by keys the checker result happens to carry.
    return {**result, "validation": outcome, "version": version}
