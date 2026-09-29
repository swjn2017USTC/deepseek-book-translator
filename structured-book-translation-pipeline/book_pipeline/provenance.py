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
    review = dict(config.get("source_review") or {})
    if input_key == "input_json" and review:
        base = Path(config["_base_dir"])
        markdown = Path(str(review.get("markdown") or (base / "source" / "structured_source.md"))).expanduser()
        state = Path(str(review.get("state") or (base / "source" / "source_state.json"))).expanduser()
        if not markdown.is_absolute():
            markdown = (base / markdown).resolve()
        if not state.is_absolute():
            state = (base / state).resolve()
        if markdown.is_file():
            manifest["source_manuscript"] = str(markdown)
            manifest["source_manuscript_sha256"] = file_sha256(markdown)
        if state.is_file():
            manifest["source_confirmation_state"] = str(state)
            manifest["source_confirmation_state_sha256"] = file_sha256(state)
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
    if manifest.get("source_manuscript"):
        manuscript = Path(str(manifest["source_manuscript"])).expanduser()
        if not manuscript.is_file() or file_sha256(manuscript) != manifest.get("source_manuscript_sha256"):
            raise RuntimeError("Structured source manuscript changed after confirmation; run confirm-source again")
    if manifest.get("source_confirmation_state"):
        state = Path(str(manifest["source_confirmation_state"])).expanduser()
        if not state.is_file() or file_sha256(state) != manifest.get("source_confirmation_state_sha256"):
            raise RuntimeError("Source confirmation state changed; run confirm-source again")
    if manifest.get("book_id") != config.get("book_id"):
        raise RuntimeError("Cleaning manifest belongs to a different book")
    return manifest
