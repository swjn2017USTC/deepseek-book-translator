from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict
import stat
import unicodedata
import zipfile


class EPUBSecurityError(ValueError):
    """Raised when an EPUB archive or XML document is unsafe to inspect."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_entries: int = 10_000
    max_file_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024
    max_compression_ratio: int = 100
    max_xml_bytes: int = 32 * 1024 * 1024
    max_xml_depth: int = 512


def normalize_member_name(name: str) -> str:
    normalized_input = unicodedata.normalize("NFC", name)
    if not normalized_input or "\\" in normalized_input or any(ord(char) < 32 or ord(char) == 127 for char in normalized_input):
        raise EPUBSecurityError(f"unsafe ZIP member path: {name!r}")
    if normalized_input.startswith("/") or normalized_input.startswith("//") or (len(normalized_input) >= 2 and normalized_input[1] == ":"):
        raise EPUBSecurityError(f"absolute ZIP member path: {name!r}")
    parts = normalized_input.split("/")
    if any(part in {"", ".", ".."} for part in parts[:-1]) or parts[-1] in {".", ".."}:
        raise EPUBSecurityError(f"unsafe ZIP member path: {name!r}")
    normalized = str(PurePosixPath(normalized_input))
    if normalized in {".", ""} or normalized.startswith("../"):
        raise EPUBSecurityError(f"unsafe ZIP member path: {name!r}")
    return normalized


def validate_archive(archive: zipfile.ZipFile, limits: ArchiveLimits = ArchiveLimits()) -> Dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > limits.max_entries:
        raise EPUBSecurityError(f"ZIP entry limit exceeded: {len(infos)} > {limits.max_entries}")
    members: Dict[str, zipfile.ZipInfo] = {}
    folded: Dict[str, str] = {}
    total = 0
    for info in infos:
        normalized = normalize_member_name(info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise EPUBSecurityError(f"symbolic-link ZIP member: {info.filename!r}")
        if info.file_size > limits.max_file_bytes:
            raise EPUBSecurityError(f"ZIP member exceeds file-size limit: {info.filename!r}")
        total += info.file_size
        if total > limits.max_total_bytes:
            raise EPUBSecurityError("ZIP expanded size exceeds total-size limit")
        if info.file_size and (not info.compress_size or info.file_size / info.compress_size > limits.max_compression_ratio):
            raise EPUBSecurityError(f"ZIP member exceeds compression-ratio limit: {info.filename!r}")
        if normalized in members:
            raise EPUBSecurityError(f"duplicate normalized ZIP member: {normalized!r}")
        key = normalized.casefold()
        if key in folded:
            raise EPUBSecurityError(f"case-colliding ZIP members: {folded[key]!r}, {normalized!r}")
        members[normalized] = info
        folded[key] = normalized
    return members


def read_member(archive: zipfile.ZipFile, members: Dict[str, zipfile.ZipInfo], name: str) -> bytes:
    normalized = normalize_member_name(name)
    if normalized not in members:
        raise EPUBSecurityError(f"required ZIP member is missing: {normalized!r}")
    return archive.read(members[normalized])


def validate_ocf_mimetype(archive: zipfile.ZipFile, members: Dict[str, zipfile.ZipInfo]) -> None:
    infos = archive.infolist()
    if not infos or infos[0].filename != "mimetype":
        raise EPUBSecurityError("OCF mimetype must be the first ZIP member")
    if infos[0].compress_type != zipfile.ZIP_STORED:
        raise EPUBSecurityError("OCF mimetype must use STORE compression")
    if read_member(archive, members, "mimetype") != b"application/epub+zip":
        raise EPUBSecurityError("OCF mimetype content is invalid")


def decode_xml_text(data: bytes, label: str) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise EPUBSecurityError(f"unsupported or invalid XML encoding in {label!r}") from error


def _reject_unsafe_dtd(text: str, label: str) -> None:
    upper = text.upper()
    start = upper.find("<!DOCTYPE")
    if start < 0:
        return
    if "<!ENTITY" in upper:
        raise EPUBSecurityError(f"unsafe XML entity declaration in {label!r}")
    end = text.find(">", start)
    if end < 0 or "[" in text[start:end]:
        raise EPUBSecurityError(f"unsafe XML DTD in {label!r}")


def parse_xml(data: bytes, *, label: str, limits: ArchiveLimits = ArchiveLimits()):
    from xml.etree import ElementTree as ET

    if len(data) > limits.max_xml_bytes:
        raise EPUBSecurityError(f"XML exceeds size limit in {label!r}")
    text = decode_xml_text(data, label)
    _reject_unsafe_dtd(text, label)
    try:
        root = ET.fromstring(text, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    except ET.ParseError as error:
        raise EPUBSecurityError(f"invalid XML in {label!r}: {error}") from error
    stack = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        if depth > limits.max_xml_depth:
            raise EPUBSecurityError(f"XML exceeds depth limit in {label!r}")
        stack.extend((child, depth + 1) for child in element)
    return root


@dataclass(frozen=True)
class TreeMember:
    """Minimal ``ZipInfo`` shape needed by OCF member checks on a tree."""

    filename: str
    compress_type: int = zipfile.ZIP_STORED
    file_size: int = 0


class DirectoryArchive:
    """Read a rendered tree through the same member contract as a ZIP.

    Validation and repackaging must inspect a rendered tree exactly like the
    source container so ``parse_package``/``parse_navigation`` can be shared.
    ``read_member`` only needs ``members[name]`` as the handle passed to
    ``archive.read``, so a rendered tree maps each normalized member name to
    itself.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def read(self, name: str) -> bytes:
        return (self.root / name).read_bytes()

    def infolist(self) -> list:
        names = sorted(directory_members(self.root))
        ordered = ["mimetype"] + [name for name in names if name != "mimetype"]
        return [TreeMember(name, file_size=(self.root / name).stat().st_size) for name in ordered]


def directory_members(root: Path) -> Dict[str, str]:
    """Map normalized member names to their relative paths in a rendered tree."""
    base = root.expanduser().resolve()
    if not base.is_dir():
        raise EPUBSecurityError(f"rendered tree is missing: {base}")
    members: Dict[str, str] = {}
    folded: Dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink():
            raise EPUBSecurityError(f"symbolic-link tree member: {path}")
        relative = path.relative_to(base).as_posix()
        normalized = normalize_member_name(relative)
        key = normalized.casefold()
        if key in folded:
            raise EPUBSecurityError(f"case-colliding tree members: {folded[key]!r}, {normalized!r}")
        members[normalized] = relative
        folded[key] = normalized
    return members
