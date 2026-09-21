from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Set, Tuple
import re
import zipfile

from .package import fragment, local_name, parse_container, parse_navigation, parse_package, resolve_href
from .security import ArchiveLimits, EPUBSecurityError, parse_xml, read_member, validate_archive, validate_ocf_mimetype

XHTML_MEDIA_TYPES = {"application/xhtml+xml", "text/html"}
REMOTE_PREFIXES = ("//",)
URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
CSS_REMOTE_URI = re.compile(r"url\(\s*(['\"]?)([^)'\"\s]+)\1\s*\)", re.IGNORECASE)

# Attributes that make the package load something: `href`/`src`/`data`/`poster` are resource
# references unless the element is a hyperlink. `a` and `area` point the reader somewhere; the
# package never fetches them, so a book that cites `http://www.oed.com/` is not a book with
# remote resources. (SVG `<image xlink:href>` is still a resource: the element is `image`.)
LINK_TAGS = {"a", "area"}
RESOURCE_ATTRIBUTES = ("href", "src", "data", "poster")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_xhtml(item: Mapping[str, str]) -> bool:
    return item["media_type"].casefold() in XHTML_MEDIA_TYPES or PurePosixPath(item["path"]).suffix.casefold() in {".xhtml", ".html", ".htm"}


def _is_remote(value: str) -> bool:
    return value.strip().startswith(REMOTE_PREFIXES) or bool(URI_SCHEME.match(value.strip()))


def _body_text_nodes(root: Any) -> int:
    def walk(element: Any, in_body: bool, excluded: bool) -> int:
        tag = local_name(element.tag).casefold()
        body = in_body or tag == "body"
        blocked = excluded or tag in {"script", "style", "code", "pre", "math", "nav", "title"}
        count = int(body and not blocked and bool((element.text or "").strip()))
        return count + sum(walk(child, body, blocked) for child in element)

    return walk(root, False, False)


def document_data(archive: zipfile.ZipFile, members: Mapping[str, Any], package: Mapping[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    ids: Dict[str, Set[str]] = {}
    links: List[Dict[str, str]] = []
    duplicate_ids: List[Dict[str, str]] = []
    scripted = False
    text_nodes = 0
    spine_ids = {entry["idref"] for entry in package["spine"]}
    for item in package["manifest"]:
        if not _is_xhtml(item):
            continue
        path = item["path"]
        root = parse_xml(read_member(archive, members, path), label=path)
        found_ids: Set[str] = set()
        for element in root.iter():
            element_id = str(element.get("id") or "")
            if element_id:
                if element_id in found_ids:
                    duplicate_ids.append({"path": path, "id": element_id})
                found_ids.add(element_id)
            tag = local_name(element.tag).casefold()
            scripted = scripted or tag == "script"
            for key, value in element.attrib.items():
                attribute = local_name(key)
                if attribute in {"href", "src"} and value:
                    links.append({"source": path, "href": str(value), "element": tag, "attribute": attribute})
        if item["id"] in spine_ids and "nav" not in item["properties"].split():
            text_nodes += _body_text_nodes(root)
        ids[path] = found_ids
    broken: List[Dict[str, str]] = []
    for link in links:
        href = link["href"]
        if _is_remote(href):
            continue
        target = resolve_href(link["source"], href)
        anchor = fragment(href)
        if target not in members or (anchor and (target not in ids or anchor not in ids[target])):
            broken.append({"source": link["source"], "href": href, "target": target, "fragment": anchor})
    internal_links = [
        link for link in links
        if link["element"] == "a" and link["attribute"] == "href"
        and not _is_remote(link["href"]) and not link["href"].startswith("#")
    ]
    return ids, {
        "graph": {
            "schema_version": 1,
            "links": sorted(links, key=lambda item: (item["source"], item["href"], item["element"], item["attribute"])),
            "internal_links": sorted(internal_links, key=lambda item: (item["source"], item["href"])),
            "broken_links": sorted(broken, key=lambda item: (item["source"], item["href"])),
            "duplicate_ids": sorted(duplicate_ids, key=lambda item: (item["path"], item["id"])),
        },
        "signals": {"scripted": scripted or any("scripted" in item["properties"].split() for item in package["manifest"]), "translatable_text_nodes": text_nodes},
    }


def _remote_resources(archive: zipfile.ZipFile, members: Mapping[str, Any], package: Mapping[str, Any]) -> List[Dict[str, str]]:
    remote: List[Dict[str, str]] = []
    for item in package["manifest"]:
        path = item["path"]
        media_type = item["media_type"].casefold()
        data = read_member(archive, members, path)
        if media_type == "text/css" or PurePosixPath(path).suffix.casefold() == ".css":
            for match in CSS_REMOTE_URI.finditer(data.decode("utf-8", errors="replace")):
                if _is_remote(match.group(2)):
                    remote.append({"path": path, "attribute": "css:url", "value": match.group(2)})
        elif _is_xhtml(item) or media_type == "image/svg+xml" or PurePosixPath(path).suffix.casefold() == ".svg":
            root = parse_xml(data, label=path)
            for element in root.iter():
                tag = local_name(element.tag).casefold()
                if tag in LINK_TAGS:
                    continue
                for key, value in element.attrib.items():
                    attribute = local_name(key).casefold()
                    if attribute in RESOURCE_ATTRIBUTES and _is_remote(str(value)):
                        remote.append({"path": path, "attribute": attribute, "value": str(value)})
    return sorted(remote, key=lambda item: (item["path"], item["attribute"], item["value"]))


def _require_targets(members: Mapping[str, Any], ids: Mapping[str, Set[str]], package: Mapping[str, Any], navigation: Mapping[str, Any]) -> None:
    targets: List[Dict[str, str]] = [{"source": package["path"], "href": entry["href"]} for entry in package["guide"]]
    targets.extend(navigation["ncx"])
    targets.extend(navigation["page_map_links"])
    for entries in navigation["nav"].values():
        targets.extend(entries)
    for target in targets:
        href = target["href"]
        if _is_remote(href):
            raise EPUBSecurityError(f"navigation target must be local: {href!r}")
        resolved = resolve_href(target["source"], href)
        anchor = fragment(href)
        if resolved not in members:
            raise EPUBSecurityError(f"navigation target is missing: {resolved!r}")
        if anchor and (resolved not in ids or anchor not in ids[resolved]):
            raise EPUBSecurityError(f"navigation fragment is missing: {href!r}")


def _compatibility(package: Mapping[str, Any], rootfiles: List[Mapping[str, str]], signals: Mapping[str, Any], members: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = package["manifest"]
    encryption = "META-INF/encryption.xml" in members
    fixed_layout = any(
        (entry["property"] == "rendition:layout" or entry["name"] == "rendition:layout") and entry["value"].casefold() == "pre-paginated"
        for entry in package["metadata"].get("meta", [])
    )
    smil = any(item["media_type"].casefold() == "application/smil+xml" or item["media_overlay"] for item in manifest)
    flags = {
        "encrypted": encryption,
        "fixed_layout": fixed_layout,
        "media_overlay": smil,
        "scripted": bool(signals["scripted"]),
        "remote_resources": bool(signals["remote_resources"]),
        "multiple_renditions": len(rootfiles) > 1,
        "image_only": int(signals["translatable_text_nodes"]) == 0,
    }
    if encryption:
        status = "BLOCKED_ENCRYPTED_EPUB"
    elif flags["image_only"]:
        status = "NEEDS_OCR"
    elif fixed_layout:
        status = "NEEDS_USER_REVIEW_FIXED_LAYOUT"
    elif smil or flags["scripted"] or flags["remote_resources"] or flags["multiple_renditions"]:
        status = "NEEDS_COMPATIBILITY_REVIEW"
    else:
        status = "COMPATIBLE_REFLOWABLE"
    return {"schema_version": 1, "status": status, "flags": flags, "remote_resources": signals["remote_resources"]}


def _inspect(
    archive: Any,
    members: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
) -> Dict[str, Any]:
    """Inspect an already-validated container (ZIP archive or rendered tree)."""
    rootfiles = parse_container(archive, members)
    packages = [parse_package(archive, members, rootfile["path"]) for rootfile in rootfiles]
    primary = packages[0]
    navigation = parse_navigation(archive, members, primary)
    ids, documents = document_data(archive, members, primary)
    _require_targets(members, ids, primary, navigation)
    documents["signals"]["remote_resources"] = _remote_resources(archive, members, primary)
    inventory = {
        "schema_version": 1,
        "source_path": source_path, "source_sha256": source_sha256, "zip_entries": len(members),
        "rootfiles": rootfiles, "packages": packages, "navigation": navigation,
        "mimetype": {"first": True, "stored": True},
        "entry_sha256": {name: sha256(read_member(archive, members, name)).hexdigest() for name in sorted(members)},
    }
    return {
        "epub_inventory": inventory,
        "source_link_graph": documents["graph"],
        "compatibility_report": _compatibility(primary, rootfiles, documents["signals"], members),
    }


def inspect_epub(path: Path, *, limits: ArchiveLimits = ArchiveLimits()) -> Dict[str, Any]:
    """Inspect a source EPUB without extracting or modifying it."""
    source = path.expanduser().resolve()
    try:
        with zipfile.ZipFile(source) as archive:
            members = validate_archive(archive, limits)
            validate_ocf_mimetype(archive, members)
            return _inspect(
                archive, members,
                source_path=str(source), source_sha256=_sha256(source),
            )
    except zipfile.BadZipFile as error:
        raise EPUBSecurityError(f"invalid ZIP container: {source}") from error


def inspect_tree(tree_dir: Path, *, limits: ArchiveLimits = ArchiveLimits()) -> Dict[str, Any]:
    """Inspect a rendered tree through the same checks as an input container."""
    from .security import DirectoryArchive, directory_members

    root = tree_dir.expanduser().resolve()
    members = directory_members(root)
    archive = DirectoryArchive(root)
    validate_ocf_mimetype(archive, members)
    digest = sha256()
    for name in sorted(members):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(read_member(archive, members, name)).hexdigest()))
    return _inspect(
        archive, members,
        source_path=str(root), source_sha256=digest.hexdigest(),
    )


def write_inspection_reports(path: Path, output_dir: Path, *, limits: ArchiveLimits = ArchiveLimits()) -> Dict[str, Path]:
    """Write deterministic inspection reports; source input remains read-only."""
    reports = inspect_epub(path, limits=limits)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for name in ("epub_inventory", "source_link_graph", "compatibility_report"):
        output = destination / f"{name}.json"
        output.write_text(json.dumps(reports[name], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs[name] = output
    return outputs
