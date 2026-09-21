"""OCF-compliant packaging of a rendered EPUB tree.

The container contract is fixed by the OCF specification: ``mimetype`` is the
first ZIP entry, uncompressed (``STORE``), and its content is exactly
``application/epub+zip`` with no trailing bytes. Everything else is stored with
deflate. Output is written atomically, so a failed package never replaces a
previous artifact.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping
import zipfile

from .security import ArchiveLimits, EPUBSecurityError, directory_members, read_member, DirectoryArchive, validate_archive, validate_ocf_mimetype

MIMETYPE = b"application/epub+zip"


class EPUBRepackError(RuntimeError):
    """Raised when a rendered tree cannot be packaged as a valid OCF container."""


def _existing_package_compression(source: Path, members: Mapping[str, Any]) -> Dict[str, int]:
    """Remember the source's per-member compression so repackaging stays stable."""
    if not source.is_file():
        return {}
    with zipfile.ZipFile(source) as archive:
        return {name: archive.getinfo(name).compress_type for name in members if name != "mimetype"}


def pack_epub(
    tree_dir: Path,
    output: Path,
    *,
    source: Path | None = None,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Dict[str, Any]:
    """Write one rendered tree to ``output`` as an OCF-valid EPUB."""
    root = tree_dir.expanduser().resolve()
    destination = output.expanduser().resolve()
    if destination == root or root in destination.parents:
        raise EPUBRepackError(f"packaged EPUB must not be written inside its tree: {destination}")
    if source is not None and destination == source.expanduser().resolve():
        raise EPUBRepackError(f"packaged EPUB must not overwrite the source EPUB: {destination}")
    members = directory_members(root)
    archive_view = DirectoryArchive(root)
    validate_ocf_mimetype(archive_view, members)
    if read_member(archive_view, members, "mimetype") != MIMETYPE:
        raise EPUBRepackError("mimetype content must be exactly 'application/epub+zip'")
    if "META-INF/container.xml" not in members:
        raise EPUBRepackError("META-INF/container.xml is missing from the rendered tree")

    compression = _existing_package_compression(source, members) if source else {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    order: List[str] = ["mimetype"] + [name for name in sorted(members) if name != "mimetype"]
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.writestr(
                zipfile.ZipInfo("mimetype", date_time=(1980, 1, 1, 0, 0, 0)),
                MIMETYPE,
                compress_type=zipfile.ZIP_STORED,
            )
            for name in order[1:]:
                archive.writestr(
                    name,
                    (root / name).read_bytes(),
                    compress_type=compression.get(name, zipfile.ZIP_DEFLATED),
                )
        # Re-validate the written artifact with the same reader used on inputs.
        with zipfile.ZipFile(temporary) as archive:
            written = validate_archive(archive, limits)
            validate_ocf_mimetype(archive, written)
            if sorted(written) != sorted(members):
                raise EPUBRepackError("packaged member set differs from the rendered tree")
            for name in written:
                if archive.read(name) != (root / name).read_bytes():
                    raise EPUBRepackError(f"packaged member bytes differ from the rendered tree: {name}")
            infos = archive.infolist()
        os.replace(temporary, destination)
    except EPUBSecurityError as error:
        temporary.unlink(missing_ok=True)
        raise EPUBRepackError(str(error)) from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "schema_version": 1,
        "path": str(destination),
        "entries": len(order),
        "bytes": destination.stat().st_size,
        "mimetype": {
            "first": infos[0].filename == "mimetype",
            "stored": infos[0].compress_type == zipfile.ZIP_STORED,
            "content": MIMETYPE.decode("ascii"),
        },
        "compression": {
            "stored": sum(1 for info in infos if info.compress_type == zipfile.ZIP_STORED),
            "deflated": sum(1 for info in infos if info.compress_type == zipfile.ZIP_DEFLATED),
        },
        "tree": str(root),
    }
