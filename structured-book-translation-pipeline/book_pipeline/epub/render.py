"""Render a translated EPUB: DOM write-back, validation, OCF repackaging.

This is the EPUB counterpart of ``render.py``'s Markdown rendering. A formal
translated tree is produced only when the translation is 100% current, the
compatibility gate passes, and the source is unchanged; the packed EPUB is
written to ``exports/`` only after the structural/resource/link validation
reports ``PASS``. A partial tree is an explicitly marked test artifact: it goes
to its own directory and its EPUB name carries ``partial``.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from ..config import resolve_path
from ..io_utils import read_jsonl, write_json
from ..models import Segment
from .inventory import inspect_epub
from .repack import pack_epub
from .rewrite import EPUBRewriteError, rewrite_epub_tree
from .translation import is_epub_segment
from .validate import epubcheck_verdict, run_epubcheck, validate_rendered_tree

RENDERED_TREE = "epub_rendered"
PARTIAL_TREE = "epub_partial_rendered"
PARTIAL_SUFFIX = ".partial"
CHINESE_TARGETS = {"简体中文", "繁體中文", "中文", "zh", "zh-cn", "zh-hans", "zh-hant"}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "source_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"EPUB source manifest not found; run prepare first: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("adapter") != "epub_native_v1":
        raise EPUBRewriteError(f"source manifest belongs to another adapter: {manifest.get('adapter')!r}")
    return manifest


def current_segment_ids(config: Dict[str, Any]) -> Set[str]:
    """Current coverage ids under the config's active translation mode."""
    # Imported lazily: ``translate`` imports ``epub.translation`` at module load,
    # so a module-level import here would be circular.
    from ..translate import is_translation_current

    output_dir = resolve_path(config, "output_dir")
    segments = [Segment(**row) for row in read_jsonl(output_dir / "cleaned_segments.jsonl")]
    if (config.get("translation") or {}).get("context_mode") == "contextual_v2":
        from ..contextual.orchestrator import current_segment_ids

        return set(current_segment_ids(config))
    glossary_sha256 = ""
    if config.get("glossary"):
        from ..glossary import glossary_cache_sha

        try:
            glossary_sha256 = glossary_cache_sha(config)
        except (FileNotFoundError, RuntimeError, ValueError):
            glossary_sha256 = "__not_ready__"
    completed = {}
    for row in read_jsonl(output_dir / "translations.jsonl"):
        completed[str(row.get("segment_id"))] = row
    return {
        segment.id
        for segment in segments
        if segment.translatable and is_translation_current(segment, completed.get(segment.id), glossary_sha256)
    }


def render_epub(
    config: Dict[str, Any],
    *,
    allow_partial: bool = False,
    project_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Produce the translated EPUB tree, validation evidence, and a packed EPUB."""
    output_dir = resolve_path(config, "output_dir")
    source_manifest = _source_manifest(output_dir)
    source = Path(str(source_manifest["source_path"])).expanduser().resolve()

    # Compatibility gate: fixed layout, encryption, SMIL, scripted content,
    # remote resources and image-only input must never reach a formal render.
    reports = inspect_epub(source)
    compatibility = reports["compatibility_report"]
    if compatibility["status"] != "COMPATIBLE_REFLOWABLE":
        raise EPUBRewriteError(f"compatibility gate blocks EPUB rendering: {compatibility['status']}")

    segments = [Segment(**row) for row in read_jsonl(output_dir / "cleaned_segments.jsonl")]
    epub_segments = [segment for segment in segments if is_epub_segment(segment) and segment.translatable]
    current = current_segment_ids(config)
    records: Dict[str, Mapping[str, Any]] = {}
    for row in read_jsonl(output_dir / "translations.jsonl"):
        identifier = str(row.get("segment_id"))
        if identifier in current:
            records[identifier] = row
    missing = [segment.id for segment in epub_segments if segment.id not in records]
    if missing and not allow_partial:
        raise EPUBRewriteError(f"EPUB render requires current translations: {len(missing)} of {len(epub_segments)} missing or stale")

    metadata: Dict[str, Any] = {"title_zh": str(config.get("book_title_zh") or "")}
    if str(config.get("target_lang") or "").casefold() in CHINESE_TARGETS:
        metadata["language"] = "zh-CN"

    tree = output_dir / (PARTIAL_TREE if allow_partial else RENDERED_TREE)
    rewrite = rewrite_epub_tree(
        source,
        tree,
        book_id=str(config["book_id"]),
        segments=segments,
        records=records,
        source_manifest=source_manifest,
        metadata=metadata,
        partial=bool(allow_partial),
    )
    if allow_partial:
        rewrite["partial_marker"] = "PARTIAL_TRANSLATION_NOT_FOR_DELIVERY"

    validation = validate_rendered_tree(source, tree, rewrite_report=rewrite)
    write_json(output_dir / "epub_rewrite_manifest.json", rewrite)
    if validation["status"] != "PASS":
        write_json(
            output_dir / "epub_validation_report.json",
            {**validation, "partial": bool(allow_partial), "epubcheck": {"validation": "not_run"}, "repack": None},
        )
        raise EPUBRewriteError(
            "EPUB validation failed before packaging: "
            + ", ".join(str(entry["check"]) for entry in validation["failures"])
        )

    exports_dir = (project_dir.expanduser().resolve() if project_dir else output_dir) / "exports"
    if allow_partial:
        # A partial tree never reaches a deliverable path; the name is the marker.
        final_path = output_dir / f"{config['book_id']}{PARTIAL_SUFFIX}.zh-CN.epub"
    else:
        final_path = exports_dir / f"{config['book_id']}.zh-CN.epub"
    # Package to a staging path first: a required EPUBCheck failure must never
    # leave a deliverable artifact behind in exports/.
    staged = output_dir / f".{config['book_id']}.zh-CN.staged.epub"
    staged.unlink(missing_ok=True)
    repack = pack_epub(tree, staged, source=source)

    epubcheck = run_epubcheck(staged, config=config)
    require_epubcheck = bool((((config.get("publish") or {}).get("epubcheck")) or {}).get("require"))
    # ADR baseline-invalid policy: the source's own EPUBCheck findings are the
    # baseline, and a deliverable may never add errors to it. The comparison is
    # recorded either way, so an invalid source can never be reported as PASS.
    baseline = run_epubcheck(source, config=config)
    if baseline["validation"] != "skipped" and epubcheck["validation"] != "skipped":
        baseline_verdict = epubcheck_verdict(baseline, epubcheck)
    else:
        baseline_verdict = {"status": "validation_incomplete", "dependency_missing": "epubcheck"}
    if baseline_verdict["status"] == "REGRESSED":
        # Adding an EPUBCheck error to a book is never acceptable, whether or not
        # a clean pass was required: the ADR's baseline-invalid policy promises
        # the output adds no errors, so a regression fails closed unconditionally.
        staged.unlink(missing_ok=True)
        write_json(
            output_dir / "epub_validation_report.json",
            {
                **validation, "partial": bool(allow_partial), "epubcheck": epubcheck,
                "epubcheck_baseline": baseline, "baseline_verdict": baseline_verdict, "repack": repack,
            },
        )
        raise EPUBRewriteError(
            "EPUBCheck baseline policy failed: the output introduced new messages "
            f"{baseline_verdict.get('new_messages')}"
        )
    # EPUBCheck is required for a *deliverable*; a partial render is not one. It is the
    # inspection artifact used precisely when the source is baseline-invalid, where no output can
    # ever be "valid" by construction. The verdict is still recorded in full, the artifact carries
    # the partial marker, and export re-checks EPUBCheck before anything reaches exports/.
    if require_epubcheck and not allow_partial and epubcheck["validation"] not in {"valid", "skipped"}:
        staged.unlink(missing_ok=True)
        write_json(
            output_dir / "epub_validation_report.json",
            {
                **validation, "partial": bool(allow_partial), "epubcheck": epubcheck,
                "epubcheck_baseline": baseline, "baseline_verdict": baseline_verdict, "repack": repack,
            },
        )
        raise EPUBRewriteError(f"EPUBCheck is required but did not pass: {epubcheck['validation']}")

    if not allow_partial:
        exports_dir.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final_path)
    destination = final_path

    # An embedded cover has two parts with different contracts: the cover *asset*
    # must be byte-identical (it is never regenerated, re-encoded or replaced),
    # while the cover *page* may be rewritten when its own text slots (for
    # example an img alt) are translated - and that change is already verified
    # against the rewrite manifest by the DOM/declaration checks.
    cover = rewrite["embedded_cover"]
    resources = {entry["path"]: entry for entry in rewrite["resources"]}
    cover_paths = list(cover["paths"])
    cover_assets = [
        path for path in cover_paths
        if resources[path]["kind"] not in {"content_document", "navigation_document"}
    ]
    cover_pages = [path for path in cover_paths if path not in cover_assets]
    cover["assets"] = cover_assets
    cover["documents"] = cover_pages
    cover["preserved"] = all(
        resources[path]["source_sha256"] == resources[path]["output_sha256"] for path in cover_assets
    ) and all(not resources[path]["changed"] for path in cover_assets)
    cover["documents_changed"] = [path for path in cover_pages if resources[path]["changed"]]
    cover["searched_externally"] = False

    report = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "adapter": "epub_native_v1",
        "output": str(destination),
        "tree": str(tree),
        "partial": bool(allow_partial),
        "partial_marker": "PARTIAL_TRANSLATION_NOT_FOR_DELIVERY" if allow_partial else "",
        "coverage": {
            "translatable_segments": len(epub_segments),
            "current_segments": len(records),
            "missing_segment_ids": missing,
        },
        "compatibility_status": compatibility["status"],
        "validation": validation["status"],
        "epubcheck": epubcheck,
        "baseline_verdict": baseline_verdict,
        "cover": cover,
        "written": {
            "rewrite_manifest": str(output_dir / "epub_rewrite_manifest.json"),
            "validation_report": str(output_dir / "epub_validation_report.json"),
        },
    }
    write_json(
        output_dir / "epub_validation_report.json",
        {
            **validation,
            "partial": bool(allow_partial),
            "epubcheck": epubcheck,
            "epubcheck_baseline": baseline,
            "baseline_verdict": baseline_verdict,
            "repack": repack,
            "cover": cover,
            "source_sha256": _sha256_file(source),
            "epub_sha256": _sha256_file(destination),
        },
    )
    write_json(output_dir / "render_report.json", report)
    return report
