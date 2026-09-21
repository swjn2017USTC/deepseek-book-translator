from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Tuple
import posixpath

from .security import EPUBSecurityError, normalize_member_name, parse_xml, read_member

CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
def local_name(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else "comment"


def text_value(element: Any) -> str:
    return "".join(element.itertext()).strip()


def resolve_href(base: str, href: str) -> str:
    value = href.split("#", 1)[0]
    if not value:
        return normalize_member_name(base)
    return normalize_member_name(posixpath.normpath(posixpath.join(posixpath.dirname(base), value)))


def fragment(href: str) -> str:
    return href.partition("#")[2]


def parse_container(archive: Any, members: Mapping[str, Any]) -> List[Dict[str, str]]:
    root = parse_xml(read_member(archive, members, "META-INF/container.xml"), label="META-INF/container.xml")
    rootfiles = []
    for element in root.iter():
        if local_name(element.tag) != "rootfile":
            continue
        path = str(element.get("full-path") or "")
        media_type = str(element.get("media-type") or "")
        if not path:
            raise EPUBSecurityError("container rootfile has no full-path")
        normalized = normalize_member_name(path)
        if normalized not in members:
            raise EPUBSecurityError(f"container rootfile is missing: {normalized!r}")
        rootfiles.append({"path": normalized, "media_type": media_type})
    if not rootfiles:
        raise EPUBSecurityError("container.xml contains no rootfile")
    return sorted(rootfiles, key=lambda item: (item["path"], item["media_type"]))


def _metadata(package: Any) -> Dict[str, List[Dict[str, str]]]:
    values: Dict[str, List[Dict[str, str]]] = {}
    for element in package.iter():
        if local_name(element.tag) != "metadata":
            continue
        for child in list(element):
            key = local_name(child.tag)
            values.setdefault(key, []).append({
                "value": str(child.get("content") or text_value(child)),
                "id": str(child.get("id") or ""),
                "property": str(child.get("property") or ""),
                "name": str(child.get("name") or ""),
            })
    return {key: values[key] for key in sorted(values)}


def parse_package(archive: Any, members: Mapping[str, Any], rootfile: str) -> Dict[str, Any]:
    package = parse_xml(read_member(archive, members, rootfile), label=rootfile)
    if local_name(package.tag) != "package":
        raise EPUBSecurityError(f"rootfile is not an OPF package: {rootfile!r}")
    manifest: Dict[str, Dict[str, str]] = {}
    spine: List[Dict[str, str]] = []
    guide: List[Dict[str, str]] = []
    for element in package.iter():
        name = local_name(element.tag)
        if name == "item" and element.get("id") and element.get("href"):
            item_id = str(element.get("id"))
            if item_id in manifest:
                raise EPUBSecurityError(f"duplicate OPF manifest id: {item_id!r}")
            href = str(element.get("href"))
            resolved = resolve_href(rootfile, href)
            if resolved not in members:
                raise EPUBSecurityError(f"OPF manifest target is missing: {resolved!r}")
            manifest[item_id] = {
                "id": item_id,
                "href": href,
                "path": resolved,
                "media_type": str(element.get("media-type") or ""),
                "properties": str(element.get("properties") or ""),
                "media_overlay": str(element.get("media-overlay") or ""),
                "fallback": str(element.get("fallback") or ""),
            }
        elif name == "itemref" and element.get("idref"):
            spine.append({"idref": str(element.get("idref")), "linear": str(element.get("linear") or "yes")})
        elif name == "reference" and element.get("href"):
            guide.append({"type": str(element.get("type") or ""), "title": str(element.get("title") or ""), "href": str(element.get("href"))})
    for entry in spine:
        if entry["idref"] not in manifest:
            raise EPUBSecurityError(f"OPF spine references missing manifest id: {entry['idref']!r}")
    return {
        "path": rootfile,
        "version": str(package.get("version") or ""),
        "unique_identifier": str(package.get("unique-identifier") or ""),
        "metadata": _metadata(package),
        "manifest": [manifest[key] for key in sorted(manifest)],
        "spine": spine,
        "guide": guide,
    }


def manifest_index(package: Mapping[str, Any]) -> Dict[str, Dict[str, str]]:
    return {str(item["id"]): dict(item) for item in package["manifest"]}


def _nav_links(root: Any, source: str) -> List[Dict[str, str]]:
    links = []
    for element in root.iter():
        if local_name(element.tag) == "a" and element.get("href"):
            links.append({"href": str(element.get("href")), "label": text_value(element), "source": source})
    return links


def parse_navigation(archive: Any, members: Mapping[str, Any], package: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = manifest_index(package)
    rootfile = str(package["path"])
    ncx = []
    nav: Dict[str, List[Dict[str, str]]] = {"toc": [], "page_list": [], "landmarks": []}
    page_maps = []
    page_map_links = []
    for item in manifest.values():
        media_type = item["media_type"].casefold()
        properties = set(item["properties"].split())
        if media_type == "application/x-dtbncx+xml":
            root = parse_xml(read_member(archive, members, item["path"]), label=item["path"])
            for node in root.iter():
                if local_name(node.tag) == "content" and node.get("src"):
                    ncx.append({"href": str(node.get("src")), "source": item["path"]})
        if "nav" in properties:
            root = parse_xml(read_member(archive, members, item["path"]), label=item["path"])
            for element in root.iter():
                if local_name(element.tag) != "nav":
                    continue
                nav_type = " ".join(value for key, value in element.attrib.items() if key.rsplit("}", 1)[-1] == "type")
                target = "page_list" if "page-list" in nav_type else "landmarks" if "landmarks" in nav_type else "toc"
                nav[target].extend(_nav_links(element, item["path"]))
        if "page-map" in item["properties"] or "page-map" in item["href"].casefold():
            page_maps.append(item["path"])
            root = parse_xml(read_member(archive, members, item["path"]), label=item["path"])
            for element in root.iter():
                for key, value in element.attrib.items():
                    if local_name(key) == "href" and value:
                        page_map_links.append({"href": str(value), "source": item["path"]})
    return {"ncx": ncx, "nav": {key: nav[key] for key in sorted(nav)}, "page_maps": sorted(page_maps), "page_map_links": sorted(page_map_links, key=lambda item: (item["source"], item["href"]))}
