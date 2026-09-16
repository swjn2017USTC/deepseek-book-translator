from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

from .clean import clean_book, require_structure_gate
from .config import load_config, resolve_path
from .cover_generator import generate_typographic_cover
from .io_utils import read_jsonl, write_json, write_jsonl
from .glossary import compile_glossary_decisions, generate_glossary, require_ready_glossary
from .provenance import require_fresh_cleaning
from .publish import export_book, register_cover
from .render import render_book
from .translate import OpenAICompatibleClient, _batches, _completed, _segments, is_translation_current, translate_book, translation_status


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAPTER_ROOT = ROOT.parent / "chapter-structure-recovery-lab"
BOOK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _project_manifest(project: Path) -> Dict[str, Any]:
    path = project.expanduser().resolve()
    if path.is_dir():
        path = path / "project.json"
    if not path.is_file():
        raise FileNotFoundError(f"New-book project manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["_manifest_path"] = str(path)
    return value


def _translation_config(project: Dict[str, Any]) -> Dict[str, Any]:
    return load_config(Path(project["translation_config"]))


def _nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _validate_ocr_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        while True:
            character = handle.read(1)
            if not character:
                raise ValueError("OCR JSON is empty")
            if not character.isspace():
                if character != "[":
                    raise ValueError("OCR JSON must be a top-level page array")
                return


def initialize_project(
    *,
    input_json: Path,
    book_id: str,
    book_title: str,
    book_title_zh: str,
    project_dir: Optional[Path] = None,
    source_lang: str = "英语",
    target_lang: str = "简体中文",
    domain: str = "学术人文社科",
    author: str = "",
    toc_search_end: int = 30,
) -> Dict[str, Any]:
    if not BOOK_ID.fullmatch(book_id):
        raise ValueError("book_id may contain only lowercase letters, digits, hyphen, and underscore")
    if toc_search_end < 0:
        raise ValueError("toc_search_end must be non-negative")
    input_path = input_json.expanduser().resolve()
    _validate_ocr_path(input_path)
    project_path = (project_dir or (ROOT / "books" / book_id)).expanduser().resolve()
    source_library_setting = os.environ.get("BOOK_SOURCE_LIBRARY_ROOT")
    source_library = Path(source_library_setting).expanduser().resolve() if source_library_setting else None
    if source_library and (project_path == source_library or source_library in project_path.parents):
        raise ValueError("New-book project must not be inside the source library")
    manifest_path = project_path / "project.json"
    if project_path.exists() and (not project_path.is_dir() or any(project_path.iterdir())):
        raise FileExistsError(f"Project directory is not empty: {project_path}")
    project_path.mkdir(parents=True, exist_ok=True)
    structure_dir = project_path / "structure"
    work_dir = project_path / "work"
    chapter_config_path = project_path / "chapter_config.json"
    translation_config_path = project_path / "translation_config.json"
    glossary_path = project_path / "glossary.json"
    write_json(chapter_config_path, {
        "book_id": book_id,
        "book_title": book_title,
        "input_json": str(input_path),
        "output_dir": str(structure_dir),
        "toc_search_pages": [0, int(toc_search_end)],
        "toc_alignment_window": 3,
        "review_threshold": 0.9,
        "suppress_book_title_duplicates": True,
    })
    write_json(glossary_path, [])
    write_json(project_path / "glossary_master.json", [])
    write_json(translation_config_path, {
        "schema_version": 1,
        "book_id": book_id,
        "book_title": book_title,
        "book_title_zh": book_title_zh,
        "author": author,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "domain": domain,
        "input_json": str(input_path),
        "structure_json": str(structure_dir / "book_structure.json"),
        "output_dir": str(work_dir),
        **({"source_library_root": str(source_library)} if source_library else {}),
        "translate_zones": ["frontmatter", "mainmatter", "backmatter", "notes", "bibliography"],
        "quality": {
            "validation_json": str(structure_dir / "validation.json"),
            "review_packets": str(structure_dir / "review_packets.jsonl"),
            "allow_pending_review": False,
        },
        "glossary": str(glossary_path),
        "glossary_settings": {
            "mode": "master_subset_plus_review",
            "master_json": str(project_path / "glossary_master.json"),
            "manifest": str(project_path / "glossary_manifest.json"),
            "candidates": str(project_path / "glossary_candidates.jsonl"),
            "decision_template": str(project_path / "glossary_decisions.template.jsonl"),
            "additions": str(project_path / "glossary_additions.json"),
            "report": str(project_path / "GLOSSARY_REPORT.md"),
            "candidate_limit": 80,
            "require_approval": True,
            "bind_translation_cache": True,
        },
        "chunk": {"max_chars": 6000, "max_segments": 10, "max_glossary_terms": 40},
        "translation": {
            "context_mode": "contextual_v2",
            "previous_segments": 1,
            "next_segments": 1,
            "previous_translation_max_chars": 300,
            "skip_failed_segments": True,
            # Public defaults stay serial because DeepSeek account limits vary.
            # Users may raise this after a successful smoke ladder.
            "max_workers": 1,
        },
        "provider": {
            "api_url": "https://api.deepseek.com/chat/completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": "deepseek-v4-flash",
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "requests_per_minute": 20,
            "timeout_seconds": 1200,
            "retries": 8,
            "max_tokens": 8192,
            "temperature": 0.2,
        },
        "publish": {
            "epubcheck": {"enabled": True, "require": True, "path": ""},
            "pdf": {"tocdepth": 3},
        },
    })
    manifest = {
        "schema_version": 1,
        "book_id": book_id,
        "project_dir": str(project_path),
        # A frozen one-file executable is unpacked into a temporary directory.
        # Store no temporary package path so the project remains portable to a
        # later EXE session or a source checkout, both of which know their own
        # default chapter package location.
        "chapter_root": None if getattr(sys, "frozen", False) else str(DEFAULT_CHAPTER_ROOT),
        "chapter_config": str(chapter_config_path),
        "translation_config": str(translation_config_path),
        "structure_dir": str(structure_dir),
        "work_dir": str(work_dir),
        "cover_dir": str(project_path / "cover"),
        "exports_dir": str(project_path / "exports"),
        "glossary": str(glossary_path),
    }
    write_json(manifest_path, manifest)
    for schema_name in ("cover_candidates.schema.json", "glossary_decision.schema.json"):
        schema = ROOT / "schemas" / schema_name
        if schema.is_file():
            (project_path / schema_name).write_text(schema.read_text(encoding="utf-8"), encoding="utf-8")
    # Keep the runbook portable; commands are meant to be run from the
    # structured-book-translation-pipeline directory in a source checkout.
    script = "new_book.py"
    runbook = f"""# {book_title_zh}：新书翻译运行说明

1. 只做章节恢复和清理（不会调用模型）：

   `python3 {script} prepare --project {project_path}`

2. 若状态为 `needs_structure_review`，人工编辑 `structure_decisions.template.jsonl`，再运行：

   `python3 {script} compile-reviews --project {project_path} --decisions {project_path / 'structure_decisions.jsonl'}`

   然后重新执行 prepare。不得在裁决中创造标题文字。

3. 若状态为 `needs_glossary_review`，人工审核或让 coding agent 审核 `glossary_candidates.jsonl`，完成
   `glossary_decisions.template.jsonl` 的每一项后另存为 `glossary_decisions.jsonl`，再运行：

   `python3 {script} compile-glossary --project {project_path} --decisions {project_path / 'glossary_decisions.jsonl'}`

   收录术语必须有译名和证据；排除项必须写原因。然后重新执行 prepare。

4. 在当前终端设置 `DEEPSEEK_API_KEY` 后做零网络预检：

   `python3 {script} preflight --project {project_path}`

5. 建议先累计翻译 10 个片段，再逐步扩大：

   `python3 {script} translate --project {project_path} --target-completed 10`

   断点续跑时把累计目标改为 100、500 等。确认后整本运行使用显式 `--all`。

6. 完成率达到 100% 后渲染：

   `python3 {script} render --project {project_path}`

7. 用本地排版生成器创建并登记封面：

   `python3 {script} generate-cover --project {project_path} --theme auto`

   如果改用自备图片，核对来源与使用权后再用 `set-cover` 登记。

8. 生成带目录的 PDF 和带封面的 EPUB：

   `python3 {script} export --project {project_path} --format both`
"""
    (project_path / "RUNBOOK.md").write_text(runbook, encoding="utf-8")
    return {**manifest, "status": "initialized", "next_command": f"python3 {script} prepare --project {project_path}"}


def _run_chapter(project: Dict[str, Any], arguments: Sequence[str]) -> None:
    if getattr(sys, "frozen", False):
        # A one-file PyInstaller GUI cannot launch ``sys.executable -m``:
        # sys.executable is the GUI executable itself. The bundled package is
        # dispatched in-process instead.
        from chapter_recovery.cli import main as chapter_main

        # Windowed PyInstaller applications have no stdout/stderr stream, while
        # the CLI prints a JSON summary. Capture it so that print() stays safe.
        captured = io.StringIO()
        with redirect_stdout(captured), redirect_stderr(captured):
            chapter_main(list(arguments))
        return
    chapter_root = Path(project.get("chapter_root") or DEFAULT_CHAPTER_ROOT).expanduser().resolve()
    if not (chapter_root / "chapter_recovery").is_dir():
        raise FileNotFoundError(f"Chapter recovery project not found: {chapter_root}")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [sys.executable, "-m", "chapter_recovery", *arguments],
        cwd=chapter_root,
        env=environment,
        check=True,
    )


def _write_decision_template(review_path: Path, destination: Path) -> None:
    rows = [
        {
            "candidate_id": row["candidate_id"],
            "decision": "needs_human",
            "reason_code": "insufficient_evidence",
            "evidence_refs": ["page_context.blocks[0]"],
        }
        for row in read_jsonl(review_path)
    ]
    write_jsonl(destination, rows)


def prepare_project(project_path: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    _run_chapter(project, ["analyze", "--config", project["chapter_config"]])
    structure_dir = Path(project["structure_dir"])
    validation_path = structure_dir / "validation.json"
    review_path = structure_dir / "review_packets.jsonl"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    pending = _nonempty_lines(review_path)
    if not validation.get("ok") or pending:
        template = Path(project["project_dir"]) / "structure_decisions.template.jsonl"
        _write_decision_template(review_path, template)
        result = {
            "schema_version": 1,
            "book_id": project["book_id"],
            "status": "needs_structure_review",
            "validation_ok": bool(validation.get("ok")),
            "pending_structure_reviews": pending,
            "review_packets": str(review_path),
            "decision_template": str(template),
            "translation_started": False,
        }
    else:
        config = _translation_config(project)
        cleaning = clean_book(config)
        glossary = generate_glossary(config)
        status = translation_status(config)
        result = {
            "schema_version": 1,
            "book_id": project["book_id"],
            "status": "ready_to_translate" if glossary.get("approved") else "needs_glossary_review",
            "validation_ok": True,
            "pending_structure_reviews": 0,
            "cleaning_metrics": cleaning["metrics"],
            "translation": status,
            "glossary": glossary,
            "translation_started": False,
        }
    write_json(Path(project["project_dir"]) / "workflow_status.json", result)
    return result


def compile_reviews(project_path: Path, decisions: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    review_path = Path(project["structure_dir"]) / "review_packets.jsonl"
    output = Path(project["project_dir"]) / "structure_overrides.json"
    chapter_config_path = Path(project["chapter_config"])
    chapter_config = json.loads(chapter_config_path.read_text(encoding="utf-8"))
    arguments = [
        "compile-decisions", "--review-packets", str(review_path),
        "--decisions", str(decisions.expanduser().resolve()), "--output", str(output),
    ]
    existing_override = chapter_config.get("overrides")
    if existing_override:
        existing_path = Path(str(existing_override)).expanduser()
        if not existing_path.is_absolute():
            existing_path = (chapter_config_path.parent / existing_path).resolve()
        if existing_path != output:
            arguments += ["--base-overrides", str(existing_path)]
    _run_chapter(project, arguments)
    chapter_config["overrides"] = str(output)
    write_json(chapter_config_path, chapter_config)
    return {"book_id": project["book_id"], "status": "reviews_compiled", "overrides": str(output), "next": "rerun prepare"}


def generate_project_glossary(project_path: Path, force: bool = False) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    config = _translation_config(project)
    require_structure_gate(config)
    require_fresh_cleaning(config)
    return generate_glossary(config, force=force)


def compile_project_glossary(project_path: Path, decisions: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    config = _translation_config(project)
    result = compile_glossary_decisions(config, decisions)
    result["next"] = "rerun prepare"
    return result


def project_status(project_path: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    structure_dir = Path(project["structure_dir"])
    validation_path = structure_dir / "validation.json"
    review_path = structure_dir / "review_packets.jsonl"
    validation_ok = bool(validation_path.exists() and json.loads(validation_path.read_text(encoding="utf-8")).get("ok"))
    pending = _nonempty_lines(review_path)
    config = _translation_config(project)
    output_dir = resolve_path(config, "output_dir")
    cleaned = (output_dir / "cleaned_segments.jsonl").exists()
    fresh = False
    stale_reason: Optional[str] = None
    if cleaned:
        try:
            require_structure_gate(config)
            require_fresh_cleaning(config)
            fresh = True
        except (FileNotFoundError, RuntimeError) as exc:
            stale_reason = str(exc)
    glossary_ready = False
    glossary_reason: Optional[str] = None
    if cleaned and fresh:
        try:
            require_ready_glossary(config)
            glossary_ready = True
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            glossary_reason = str(exc)
    translation = translation_status(config) if cleaned else None
    if not validation_ok or pending:
        status = "needs_structure_review"
    elif not cleaned or not fresh:
        status = "needs_prepare"
    elif not glossary_ready:
        status = "needs_glossary_review"
    elif translation and translation["ready_to_render"]:
        status = "ready_to_render"
    else:
        status = "ready_to_translate"
    return {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": status,
        "validation_ok": validation_ok,
        "pending_structure_reviews": pending,
        "cleaning_fresh": fresh,
        "stale_reason": stale_reason,
        "glossary_ready": glossary_ready,
        "glossary_reason": glossary_reason,
        "translation": translation,
    }


def preflight_project(project_path: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    config = _translation_config(project)
    require_structure_gate(config)
    require_fresh_cleaning(config)
    glossary_sha256 = require_ready_glossary(config)
    cache_sha256 = glossary_sha256 if config.get("glossary_settings", {}).get("bind_translation_cache", False) else ""
    client = OpenAICompatibleClient(dict(config.get("provider") or {}))
    output_dir = resolve_path(config, "output_dir")
    segments = _segments(output_dir / "cleaned_segments.jsonl")
    completed = _completed(output_dir / "translations.jsonl")
    required = [segment for segment in segments if segment.translatable]
    pending = [
        segment for segment in required
        if not (
            is_translation_current(segment, completed.get(segment.id), cache_sha256)
        )
    ]
    chunk = config.get("chunk", {})
    batches = list(_batches(pending, int(chunk.get("max_chars", 10000)), int(chunk.get("max_segments", 20))))
    status = translation_status(config)
    return {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": "preflight_passed",
        "model": client.model,
        "api_key_present": True,
        "external_requests_made": 0,
        "required_segments": status["required_segments"],
        "completed_segments": status["completed_segments"],
        "remaining_segments": status["remaining_segments"],
        "estimated_remaining_batches": len(batches),
        "remaining_translatable_characters": sum(len(segment.source_text) for segment in pending),
    }


def translate_project(project_path: Path, target_completed: Optional[int], all_segments: bool) -> Dict[str, Any]:
    if not all_segments and target_completed is None:
        raise ValueError("Choose an explicit --target-completed value or --all")
    project = _project_manifest(project_path)
    config = _translation_config(project)
    target = None if all_segments else target_completed
    if (config.get("translation") or {}).get("context_mode") == "contextual_v2":
        from .contextual import translate_v2

        return translate_v2(config, target_completed=target)
    return translate_book(config, target_completed=target)


def render_project(project_path: Path, allow_partial: bool = False) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    return render_book(_translation_config(project), allow_partial=allow_partial)


def set_project_cover(
    project_path: Path,
    image_path: Path,
    *,
    rights_status: str,
    source_page_url: str = "",
    source_image_url: str = "",
    license_name: str = "",
    selected_by: str = "user",
    edition_evidence: str = "",
) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    return register_cover(
        Path(project["project_dir"]), image_path,
        rights_status=rights_status,
        source_page_url=source_page_url,
        source_image_url=source_image_url,
        license_name=license_name,
        selected_by=selected_by,
        edition_evidence=edition_evidence,
    )


def generate_project_cover(project_path: Path, theme: str = "auto") -> Dict[str, Any]:
    project = _project_manifest(project_path)
    return generate_typographic_cover(Path(project["project_dir"]), theme=theme)


def export_project(project_path: Path, output_format: str = "both") -> Dict[str, Any]:
    project = _project_manifest(project_path)
    formats = ("pdf", "epub") if output_format == "both" else (output_format,)
    return export_book(_translation_config(project), Path(project["project_dir"]), formats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and translate a new OCR JSON book safely")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a self-contained new-book project")
    init.add_argument("--input-json", type=Path, required=True)
    init.add_argument("--book-id", required=True)
    init.add_argument("--book-title", required=True)
    init.add_argument("--book-title-zh", required=True)
    init.add_argument("--project-dir", type=Path)
    init.add_argument("--source-lang", default="英语")
    init.add_argument("--target-lang", default="简体中文")
    init.add_argument("--domain", default="学术人文社科")
    init.add_argument("--author", default="")
    init.add_argument("--toc-search-end", type=int, default=30)
    for name in ("prepare", "status", "preflight", "render"):
        command = commands.add_parser(name)
        command.add_argument("--project", type=Path, required=True)
        if name == "render":
            command.add_argument("--allow-partial", action="store_true")
    compile_command = commands.add_parser("compile-reviews")
    compile_command.add_argument("--project", type=Path, required=True)
    compile_command.add_argument("--decisions", type=Path, required=True)
    glossary = commands.add_parser("glossary", help="generate the master subset and candidate review packets")
    glossary.add_argument("--project", type=Path, required=True)
    glossary.add_argument("--force", action="store_true")
    compile_glossary = commands.add_parser("compile-glossary", help="compile reviewed terminology decisions")
    compile_glossary.add_argument("--project", type=Path, required=True)
    compile_glossary.add_argument("--decisions", type=Path, required=True)
    translate = commands.add_parser("translate")
    translate.add_argument("--project", type=Path, required=True)
    choice = translate.add_mutually_exclusive_group(required=True)
    choice.add_argument("--target-completed", type=int)
    choice.add_argument("--all", action="store_true", dest="all_segments")
    cover = commands.add_parser("set-cover", help="register an approved local cover with provenance")
    cover.add_argument("--project", type=Path, required=True)
    cover.add_argument("--image", type=Path, required=True)
    cover.add_argument("--rights-status", required=True, choices=sorted({
        "public_domain", "licensed", "official_product_cover",
        "user_provided", "user_approved_personal_use",
    }))
    cover.add_argument("--source-page-url", default="")
    cover.add_argument("--source-image-url", default="")
    cover.add_argument("--license", dest="license_name", default="")
    cover.add_argument("--selected-by", default="user")
    cover.add_argument("--edition-evidence", default="")
    generated_cover = commands.add_parser(
        "generate-cover", help="generate and register a local typographic cover"
    )
    generated_cover.add_argument("--project", type=Path, required=True)
    generated_cover.add_argument(
        "--theme", choices=("auto", "ink", "ocean", "ember", "forest", "plum"), default="auto"
    )
    export = commands.add_parser("export", help="convert the completed book to PDF/EPUB")
    export.add_argument("--project", type=Path, required=True)
    export.add_argument("--format", choices=("both", "pdf", "epub"), default="both")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        result = initialize_project(
            input_json=args.input_json, book_id=args.book_id,
            book_title=args.book_title, book_title_zh=args.book_title_zh,
            project_dir=args.project_dir, source_lang=args.source_lang,
            target_lang=args.target_lang, domain=args.domain, author=args.author,
            toc_search_end=args.toc_search_end,
        )
    elif args.command == "prepare":
        result = prepare_project(args.project)
    elif args.command == "compile-reviews":
        result = compile_reviews(args.project, args.decisions)
    elif args.command == "glossary":
        result = generate_project_glossary(args.project, args.force)
    elif args.command == "compile-glossary":
        result = compile_project_glossary(args.project, args.decisions)
    elif args.command == "status":
        result = project_status(args.project)
    elif args.command == "preflight":
        result = preflight_project(args.project)
    elif args.command == "translate":
        result = translate_project(args.project, args.target_completed, args.all_segments)
    elif args.command == "set-cover":
        result = set_project_cover(
            args.project, args.image, rights_status=args.rights_status,
            source_page_url=args.source_page_url, source_image_url=args.source_image_url,
            license_name=args.license_name, selected_by=args.selected_by,
            edition_evidence=args.edition_evidence,
        )
    elif args.command == "generate-cover":
        result = generate_project_cover(args.project, args.theme)
    elif args.command == "export":
        result = export_project(args.project, args.format)
    else:
        result = render_project(args.project, args.allow_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
