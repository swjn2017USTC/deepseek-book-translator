from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict

from .config import resolve_path, source_input_key
from .io_utils import write_json


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_clean_manifest(config: Dict[str, Any], segments_path: Path, report_path: Path) -> Dict[str, Any]:
    output_dir = resolve_path(config, "output_dir")
    input_key = source_input_key(config)
    input_path = resolve_path(config, input_key)
    manifest = {
        "schema_version": 2,
        "book_id": config["book_id"],
        "config": config["_config_path"],
        "source_input_key": input_key,
        "source_path": str(input_path),
        "source_sha256": file_sha256(input_path),
        "structure_json": str(resolve_path(config, "structure_json")),
        "structure_sha256": file_sha256(resolve_path(config, "structure_json")),
        "segments_file": str(segments_path),
        "segments_sha256": file_sha256(segments_path),
        "cleaning_report": str(report_path),
        "cleaning_report_sha256": file_sha256(report_path),
    }
    # Preserve the historical OCR manifest fields for existing tooling.
    if input_key == "input_json":
        manifest["input_json"] = str(input_path)
        manifest["input_sha256"] = manifest["source_sha256"]
    else:
        manifest["input_epub"] = str(input_path)
        manifest["input_epub_sha256"] = manifest["source_sha256"]
    write_json(output_dir / "pipeline_manifest.json", manifest)
    return manifest


def require_fresh_cleaning(config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = resolve_path(config, "output_dir")
    manifest_path = output_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run clean before translate or render")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) < 2:
        raise RuntimeError("Cleaning manifest predates source-hash binding; rerun clean")
    input_key = source_input_key(config)
    input_hash_key = "source_sha256"
    if not manifest.get(input_hash_key):
        input_hash_key = "input_sha256" if input_key == "input_json" else "input_epub_sha256"
    input_label = "OCR input" if input_key == "input_json" else "EPUB input"
    checks = (
        (resolve_path(config, input_key), input_hash_key, input_label),
        (resolve_path(config, "structure_json"), "structure_sha256", "chapter structure"),
        (output_dir / "cleaned_segments.jsonl", "segments_sha256", "cleaned segments"),
        (output_dir / "cleaning_report.json", "cleaning_report_sha256", "cleaning report"),
    )
    for path, key, label in checks:
        if not path.exists() or file_sha256(path) != manifest.get(key):
            raise RuntimeError(f"Stale or changed {label}; rerun clean before translation")
    if manifest.get("book_id") != config.get("book_id"):
        raise RuntimeError("Cleaning manifest belongs to a different book")
    return manifest
