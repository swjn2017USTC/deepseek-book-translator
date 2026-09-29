from __future__ import annotations

"""Tolerant EPUB -> canonical reviewed-source normalization.

This adapter deliberately does *not* preserve arbitrary source DOM/CSS for
translation. It extracts reading-order semantics and reusable local assets into
the same Segment/Markdown pipeline used by OCR projects, then publication
rebuilds a clean EPUB3 from the translated Markdown.
"""

from collections import Counter
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import html
import posixpath
import re
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote, urlsplit
import zipfile

from ..models import Segment, digest
from .security import ArchiveLimits, EPUBSecurityError, normalize_member_name, validate_archive


CONTENT_SUFFIXES = {".xhtml", ".html", ".htm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp"}
IMAGE_MEDIA = {
    "image/jpeg", "image/png", "image/gif", "image/svg+xml", "image/webp",
}
SKIP_TAGS = {"script", "style", "code", "math", "noscript"}
BLOCK_KINDS = {
    "p": "paragraph",
    "li": "list_item",
    "blockquote": "blockquote",
    "figcaption": "caption",
    "caption": "caption",
    "dt": "definition_term",
    "dd": "definition_description",
    "pre": "paragraph",
}
REMOTE_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _decode_markup(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _attrs(attrs: Sequence[Tuple[str, Optional[str]]]) -> Dict[str, str]:
    return {str(key).casefold(): str(value or "") for key, value in attrs}


def _local(tag: str) -> str:
    return tag.casefold().split(":")[-1]


def _resolve_member(base: str, href: str) -> Optional[str]:
    value = html.unescape(str(href or "")).strip()
    if not value or value.startswith("//") or REMOTE_SCHEME.match(value):
        return None
    parsed = urlsplit(value)
    raw_path = unquote(parsed.path)
    if not raw_path:
        return normalize_member_name(base)
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(base), raw_path))
    try:
        return normalize_member_name(joined)
    except EPUBSecurityError:
        return None


class _OPFParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.manifest: Dict[str, Dict[str, str]] = {}
        self.spine: List[str] = []
        self.cover_id = ""
        self.guide_cover = ""
        self.metadata: Dict[str, List[str]] = {"title": [], "creator": [], "language": []}
        self.meta_rows: List[Dict[str, str]] = []
        self._capture: Optional[str] = None
        self._capture_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        self._start(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        self._start(tag, attrs)

    def _start(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        name = _local(tag)
        row = _attrs(attrs)
        if name == "item" and row.get("id") and row.get("href"):
            self.manifest[row["id"]] = {
                "id": row["id"],
                "href": row["href"],
                "media_type": row.get("media-type", ""),
                "properties": row.get("properties", ""),
            }
        elif name == "itemref" and row.get("idref"):
            self.spine.append(row["idref"])
        elif name == "meta":
            self.meta_rows.append(row)
            if row.get("name", "").casefold() == "cover" and row.get("content"):
                self.cover_id = row["content"]
        elif name == "reference" and "cover" in row.get("type", "").casefold() and row.get("href"):
            self.guide_cover = row["href"]
        elif name in {"title", "creator", "language"}:
            self._capture = name
            self._capture_parts = []

    def handle_endtag(self, tag: str) -> None:
        name = _local(tag)
        if self._capture == name:
            value = " ".join("".join(self._capture_parts).split()).strip()
            if value:
                self.metadata[name].append(value)
            self._capture = None
            self._capture_parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._capture_parts.append(data)


class _ContentParser(HTMLParser):
    """Extract simple book semantics from XHTML or HTML-ish content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[Dict[str, Any]] = []
        self.document_title = ""
        self._title_parts: List[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._footnote_depth = 0
        self._current: Optional[Dict[str, Any]] = None

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        self._start(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        self._start(tag, attrs)
        if _local(tag) in {"img", "image", "br", "hr"}:
            return
        self.handle_endtag(tag)

    def _start(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        name = _local(tag)
        row = _attrs(attrs)
        if name in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if name == "title":
            self._in_title = True
            self._title_parts = []
            return

        epub_type = " ".join(
            value for key, value in row.items()
            if key in {"epub:type", "type", "role", "class"}
        ).casefold()
        if name in {"aside", "section"} and ("footnote" in epub_type or "endnote" in epub_type):
            self._footnote_depth += 1

        if name == "br" and self._current is not None:
            self._current["parts"].append("\n")
            return

        if name in {"img", "image"}:
            src = row.get("src") or row.get("href") or row.get("xlink:href") or ""
            if src:
                self.blocks.append({
                    "kind": "image",
                    "src": src,
                    "alt": row.get("alt", ""),
                    "id": row.get("id", ""),
                })
            return

        kind: Optional[str] = None
        level: Optional[int] = None
        if re.fullmatch(r"h[1-6]", name):
            kind = "heading"
            level = int(name[1:])
        elif name in BLOCK_KINDS:
            kind = "footnote" if self._footnote_depth else BLOCK_KINDS[name]

        # Avoid nested block duplication (e.g. <li><p>...</p></li>).
        if kind and self._current is None:
            self._current = {
                "tag": name,
                "kind": kind,
                "level": level,
                "id": row.get("id", ""),
                "parts": [],
            }

    def handle_endtag(self, tag: str) -> None:
        name = _local(tag)
        if name in SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if name == "title":
            self._in_title = False
            value = " ".join("".join(self._title_parts).split()).strip()
            if value:
                self.document_title = value
            return
        if self._current is not None and name == self._current["tag"]:
            text = _clean_text("".join(self._current["parts"]))
            if text:
                row = dict(self._current)
                row["text"] = text
                row.pop("parts", None)
                self.blocks.append(row)
            self._current = None
        if name in {"aside", "section"} and self._footnote_depth:
            self._footnote_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        if self._current is not None:
            self._current["parts"].append(data)


def _clean_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _discover_rootfile(archive: zipfile.ZipFile, members: Mapping[str, Any]) -> str:
    container = members.get("META-INF/container.xml")
    if container is not None:
        text = _decode_markup(archive.read(container))
        match = re.search(r"full-path\s*=\s*['\"]([^'\"]+\.opf)['\"]", text, re.I)
        if match:
            candidate = normalize_member_name(unquote(match.group(1)))
            if candidate in members:
                return candidate
    opfs = sorted(name for name in members if name.casefold().endswith(".opf"))
    if len(opfs) == 1:
        return opfs[0]
    if opfs:
        # Prefer the shallowest path; ties are deterministic.
        return min(opfs, key=lambda value: (value.count("/"), len(value), value.casefold()))
    raise EPUBSecurityError("EPUB contains no OPF package document")


def _parse_opf(archive: zipfile.ZipFile, members: Mapping[str, Any], rootfile: str) -> _OPFParser:
    parser = _OPFParser()
    parser.feed(_decode_markup(archive.read(members[rootfile])))
    parser.close()
    if not parser.manifest:
        raise EPUBSecurityError("OPF package contains no manifest items")
    if not parser.spine:
        # Tolerant fallback for damaged/simple EPUBs: use manifest document order.
        parser.spine = [
            item_id for item_id, item in parser.manifest.items()
            if item["media_type"].casefold() in {"application/xhtml+xml", "text/html"}
            or PurePosixPath(item["href"]).suffix.casefold() in CONTENT_SUFFIXES
        ]
    return parser


def validate_epub_source(path: Path, *, limits: ArchiveLimits = ArchiveLimits()) -> Dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise EPUBSecurityError(f"invalid EPUB/ZIP container: {source}")
    with zipfile.ZipFile(source) as archive:
        members = validate_archive(archive, limits)
        if "META-INF/encryption.xml" in members:
            raise EPUBSecurityError("encrypted/obfuscated EPUB resources are not supported by canonical import")
        rootfile = _discover_rootfile(archive, members)
        package = _parse_opf(archive, members, rootfile)
        return {
            "rootfile": rootfile,
            "manifest_items": len(package.manifest),
            "spine_items": len(package.spine),
            "strict_ocf_mimetype": (
                bool(archive.infolist())
                and archive.infolist()[0].filename == "mimetype"
                and archive.infolist()[0].compress_type == zipfile.ZIP_STORED
                and "mimetype" in members
                and archive.read(members["mimetype"]) == b"application/epub+zip"
            ),
        }


def _safe_asset_name(member: str, data: bytes) -> str:
    suffix = PurePosixPath(member).suffix.casefold()
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".bin"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", PurePosixPath(member).stem).strip("-._") or "asset"
    return f"{sha256(data).hexdigest()[:12]}-{stem[:80]}{suffix}"


def _copy_asset(
    archive: zipfile.ZipFile,
    members: Mapping[str, Any],
    member: str,
    project_dir: Path,
    cache: Dict[str, str],
) -> Optional[str]:
    if member in cache:
        return cache[member]
    if member not in members:
        return None
    data = archive.read(members[member])
    suffix = PurePosixPath(member).suffix.casefold()
    if suffix not in IMAGE_SUFFIXES:
        return None
    relative = Path("assets") / "epub" / _safe_asset_name(member, data)
    destination = project_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(data)
    cache[member] = relative.as_posix()
    return cache[member]


def _cover_member(
    archive: zipfile.ZipFile,
    members: Mapping[str, Any],
    rootfile: str,
    package: _OPFParser,
) -> Optional[str]:
    def item_member(item: Mapping[str, str]) -> Optional[str]:
        return _resolve_member(rootfile, item.get("href", ""))

    candidates: List[str] = []
    if package.cover_id and package.cover_id in package.manifest:
        member = item_member(package.manifest[package.cover_id])
        if member:
            candidates.append(member)
    for item in package.manifest.values():
        if "cover-image" in item.get("properties", "").split():
            member = item_member(item)
            if member:
                candidates.append(member)
    if package.guide_cover:
        member = _resolve_member(rootfile, package.guide_cover)
        if member:
            candidates.append(member)
    for item in package.manifest.values():
        value = " ".join((item.get("id", ""), item.get("href", ""))).casefold()
        if "cover" in value:
            member = item_member(item)
            if member:
                candidates.append(member)

    for candidate in candidates:
        if candidate not in members:
            continue
        if PurePosixPath(candidate).suffix.casefold() in IMAGE_SUFFIXES:
            return candidate
        if PurePosixPath(candidate).suffix.casefold() in CONTENT_SUFFIXES:
            parser = _ContentParser()
            parser.feed(_decode_markup(archive.read(members[candidate])))
            parser.close()
            image = next((row for row in parser.blocks if row.get("kind") == "image"), None)
            if image:
                resolved = _resolve_member(candidate, str(image.get("src") or ""))
                if resolved in members:
                    return resolved
    return None


def normalize_epub(
    path: Path,
    book_id: str,
    project_dir: Path,
    *,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Dict[str, Any]:
    """Normalize an EPUB into canonical Segments + structure + local assets."""
    source = path.expanduser().resolve()
    project = project_dir.expanduser().resolve()
    asset_cache: Dict[str, str] = {}
    segments: List[Segment] = []
    nodes: List[Dict[str, Any]] = [{
        "id": f"book-{book_id}",
        "kind": "book",
        "level": 0,
        "title_source": "",
        "parent_id": None,
        "page_json": None,
        "zone": "mainmatter",
        "adapter": "epub_markdown_v2",
    }]
    documents: List[Dict[str, Any]] = []
    warnings: List[str] = []
    skipped = Counter()
    extracted_images: List[Dict[str, str]] = []

    with zipfile.ZipFile(source) as archive:
        members = validate_archive(archive, limits)
        if "META-INF/encryption.xml" in members:
            raise EPUBSecurityError("encrypted/obfuscated EPUB resources are not supported by canonical import")
        rootfile = _discover_rootfile(archive, members)
        package = _parse_opf(archive, members, rootfile)
        cover_member = _cover_member(archive, members, rootfile, package)
        manifest = package.manifest
        root_id = f"book-{book_id}"

        for spine_index, item_id in enumerate(package.spine):
            item = manifest.get(item_id)
            if item is None:
                warnings.append(f"spine item {item_id!r} missing from manifest; skipped")
                continue
            href = _resolve_member(rootfile, item.get("href", ""))
            if href is None or href not in members:
                warnings.append(f"spine target {item.get('href')!r} missing; skipped")
                continue
            suffix = PurePosixPath(href).suffix.casefold()
            media = item.get("media_type", "").casefold()
            if media not in {"application/xhtml+xml", "text/html"} and suffix not in CONTENT_SUFFIXES:
                skipped["non_text_spine"] += 1
                continue
            if "nav" in item.get("properties", "").split():
                skipped["navigation_document"] += 1
                continue

            data = archive.read(members[href])
            parser = _ContentParser()
            try:
                parser.feed(_decode_markup(data))
                parser.close()
            except Exception as exc:
                warnings.append(f"{href}: tolerant HTML parser recovered partially after {type(exc).__name__}")

            blocks = list(parser.blocks)
            if not any(row.get("kind") == "heading" for row in blocks):
                title = parser.document_title.strip()
                if title and title.casefold() not in {"untitled", "contents", "table of contents"}:
                    blocks.insert(0, {"kind": "heading", "text": title, "level": 2, "id": ""})

            stack: List[Tuple[int, str]] = []
            document_segment_ids: List[str] = []
            for ordinal, block in enumerate(blocks):
                kind = str(block.get("kind") or "paragraph")
                if kind == "image":
                    member = _resolve_member(href, str(block.get("src") or ""))
                    if member is None:
                        skipped["remote_or_invalid_image"] += 1
                        continue
                    asset = _copy_asset(archive, members, member, project, asset_cache)
                    if not asset:
                        skipped["missing_or_unsupported_image"] += 1
                        continue
                    alt = _clean_text(str(block.get("alt") or ""))
                    text = f"![{alt}]({asset})"
                    segment_id = f"seg-{digest(book_id, href, ordinal, 'image')[:20]}"
                    parent = stack[-1][1] if stack else root_id
                    segment = Segment(
                        segment_id, digest(text), "image", text, None, "mainmatter",
                        parent, translatable=False,
                        source_block_ids=[f"{href}#image-{ordinal}"],
                        metadata={
                            "source_adapter": "epub_markdown_v2",
                            "href": href,
                            "spine_index": spine_index,
                            "source_member": member,
                            "asset_path": asset,
                            "alt": alt,
                        },
                    )
                    segments.append(segment)
                    document_segment_ids.append(segment_id)
                    extracted_images.append({"source_member": member, "asset_path": asset})
                    continue

                text = _clean_text(str(block.get("text") or ""))
                if not text:
                    skipped[f"empty_{kind}"] += 1
                    continue
                level = int(block.get("level") or 2) if kind == "heading" else None
                segment_id = f"seg-{digest(book_id, href, ordinal, kind)[:20]}"
                if kind == "heading":
                    assert level is not None
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    parent = stack[-1][1] if stack else root_id
                else:
                    parent = stack[-1][1] if stack else root_id
                zone = "notes" if kind == "footnote" else "mainmatter"
                metadata = {
                    "source_adapter": "epub_markdown_v2",
                    "href": href,
                    "spine_index": spine_index,
                    "original_element_id": str(block.get("id") or ""),
                }
                if kind == "footnote":
                    metadata["reference"] = f"epub-note-{spine_index + 1}-{ordinal + 1}"
                segment = Segment(
                    segment_id, digest(text), kind, text, None, zone, parent,
                    level=level, translatable=True,
                    source_block_ids=[f"{href}#{ordinal}"], metadata=metadata,
                )
                segments.append(segment)
                document_segment_ids.append(segment_id)
                if kind == "heading":
                    node_kind = "chapter" if not stack else "section"
                    nodes.append({
                        "id": segment_id,
                        "kind": node_kind,
                        "level": level,
                        "title_source": text,
                        "parent_id": parent,
                        "page_json": None,
                        "zone": zone,
                        "href": href,
                        "spine_index": spine_index,
                        "adapter": "epub_markdown_v2",
                    })
                    segment.metadata["node_id"] = segment_id
                    segment.metadata["node_kind"] = node_kind
                    stack.append((level, segment_id))

            documents.append({
                "href": href,
                "spine_index": spine_index,
                "sha256": sha256(data).hexdigest(),
                "segments": len(document_segment_ids),
            })

        if not any(segment.translatable for segment in segments):
            raise EPUBSecurityError("EPUB canonical importer found no translatable text; use OCR for image-only books")

        cover_asset: Optional[str] = None
        if cover_member:
            cover_asset = _copy_asset(archive, members, cover_member, project, asset_cache)

    source_hash = _sha256(source)
    structure = {
        "schema_version": 1,
        "book_id": book_id,
        "adapter": "epub_markdown_v2",
        "nodes": nodes,
    }
    report = {
        "schema_version": 1,
        "book_id": book_id,
        "adapter": "epub_markdown_v2",
        "source_path": str(source),
        "source_sha256": source_hash,
        "metrics": {
            "segments": len(segments),
            "translatable_segments": sum(segment.translatable for segment in segments),
            "kinds": dict(sorted(Counter(segment.kind for segment in segments).items())),
            "skipped": dict(sorted(skipped.items())),
            "documents": len(documents),
            "images": len(extracted_images),
        },
        "warnings": warnings,
    }
    source_manifest = {
        "schema_version": 1,
        "adapter": "epub_markdown_v2",
        "book_id": book_id,
        "source_path": str(source),
        "source_sha256": source_hash,
        "rootfile": rootfile,
        "documents": documents,
        "assets": extracted_images,
        "cover_asset": cover_asset,
        "metadata": package.metadata,
        "strict_ocf_mimetype": validate_epub_source(source, limits=limits)["strict_ocf_mimetype"],
    }
    return {
        "segments": segments,
        "structure": structure,
        "cleaning_report": report,
        "source_manifest": source_manifest,
        "cover_asset": cover_asset,
        "warnings": warnings,
    }
