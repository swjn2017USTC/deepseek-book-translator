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
from .models import Segment
from .provenance import write_clean_manifest
from .epub import inspect_epub, render_epub, segment_epub
from .epub.tokens import strip_tokens
from .glossary import compile_glossary_decisions, generate_glossary, require_ready_glossary
from .glossary_llm import auto_review_glossary
from .provenance import require_fresh_cleaning
from .publish import export_book, register_cover
from .render import render_book
from .structure_vision import collect_structure_vision
from .source_manuscript import (
    build_source_manuscript,
    confirm_source_manuscript,
    require_source_confirmation,
    source_status,
)
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


def _validate_epub_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Native inspection validates ZIP/OCF/package safety without modifying the source.
    inspect_epub(path)


def _is_epub_project(project: Dict[str, Any]) -> bool:
    return str(project.get("source_adapter") or "") == "epub_native_v1"


def _is_epub_config(config: Dict[str, Any]) -> bool:
    return str(config.get("source_adapter") or "") == "epub_native_v1" or bool(config.get("input_epub"))


def initialize_project(
    *,
    input_json: Optional[Path] = None,
    input_epub: Optional[Path] = None,
    book_id: str,
    book_title: str,
    book_title_zh: str,
    project_dir: Optional[Path] = None,
    source_lang: str = "英语",
    target_lang: str = "简体中文",
    domain: str = "学术人文社科",
    author: str = "",
    toc_search_end: int = 30,
    thinking_mode: str = "disabled",
    source_confirmation: str = "manual",
) -> Dict[str, Any]:
    if (input_json is None) == (input_epub is None):
        raise ValueError("Choose exactly one source: input_json or input_epub")
    if not BOOK_ID.fullmatch(book_id):
        raise ValueError("book_id may contain only lowercase letters, digits, hyphen, and underscore")
    if toc_search_end < 0:
        raise ValueError("toc_search_end must be non-negative")
    if thinking_mode not in {"disabled", "low", "high", "max"}:
        raise ValueError("thinking_mode must be one of: disabled, low, high, max")
    if source_confirmation not in {"manual", "auto"}:
        raise ValueError("source_confirmation must be 'manual' or 'auto'")

    source_adapter = "epub_native_v1" if input_epub is not None else "ocr_json_v1"
    input_path = (input_epub if input_epub is not None else input_json).expanduser().resolve()
    if input_epub is not None:
        _validate_epub_path(input_path)
    else:
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

    chapter_config_value: Optional[str] = None
    if input_epub is None:
        write_json(chapter_config_path, {
            "book_id": book_id,
            "book_title": book_title,
            "input_json": str(input_path),
            "output_dir": str(structure_dir),
            "toc_search_pages": [0, int(toc_search_end)],
            "toc_alignment_window": 3,
            "review_threshold": 0.9,
            "suppress_book_title_duplicates": True,
            "vision": {
                "enabled": True,
                "model": "deepseek-flash",
                "toc_scan_pages": [0, int(toc_search_end)],
                "toc_batch_size": 4,
                "body_batch_size": 4,
                "max_body_pages": 120,
                "scan_all_body_pages": False,
                "allow_remote_input_images": False,
                "render_dpi": 120,
                "max_tokens": 4096,
                "temperature": 0.0,
                "structured_retries": 3,
                "toc_page_min_confidence": 0.70,
                "toc_entry_min_confidence": 0.72,
                "heading_min_confidence": 0.72,
                "unmatched_heading_min_confidence": 0.90,
                "heading_text_match": 0.58,
            },
        })
        chapter_config_value = str(chapter_config_path)
    else:
        # Keep config loading valid before prepare while making it explicit that
        # no OCR chapter recovery has run yet. prepare replaces these placeholders.
        structure_dir.mkdir(parents=True, exist_ok=True)
        write_json(structure_dir / "book_structure.json", {
            "schema_version": 1,
            "book_id": book_id,
            "adapter": "epub_native_v1",
            "nodes": [],
        })
        write_json(structure_dir / "validation.json", {
            "schema_version": 1,
            "ok": False,
            "adapter": "epub_native_v1",
            "status": "not_prepared",
        })
        write_jsonl(structure_dir / "review_packets.jsonl", [])

    write_json(glossary_path, [])
    write_json(project_path / "glossary_master.json", [])

    translation_config: Dict[str, Any] = {
        "schema_version": 1,
        "book_id": book_id,
        "book_title": book_title,
        "book_title_zh": book_title_zh,
        "author": author,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "domain": domain,
        "source_adapter": source_adapter,
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
        "source_review": {
            "mode": source_confirmation,
            "directory": str(project_path / "source"),
            "markdown": str(project_path / "source" / "structured_source.md"),
            "state": str(project_path / "source" / "source_state.json"),
            "generated_segments": str(project_path / "source" / "generated_segments.jsonl"),
            "cross_page_merge": True,
            "aggressive_cross_page_merge": False,
        },
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
            "llm_review": {
                "enabled": True,
                "batch_size": 12,
                "structured_retries": 3,
                "adjudicate_below": 0.78,
            },
        },
        "chunk": {"max_chars": 6000, "max_segments": 10, "max_glossary_terms": 40},
        "translation": {
            "context_mode": "contextual_v2",
            "previous_segments": 1,
            "next_segments": 1,
            "previous_translation_max_chars": 300,
            "skip_failed_segments": True,
            "max_workers": 4,
            "max_parse_retries": 3,
            "micro_batch": {
                "enabled": True,
                "max_segments": 4,
                "max_chars": 6000,
                "small_kind_max_segments": 2,
                "binary_fallback": True,
            },
        },
        "provider": {
            "api_url": "https://api.deepseek.com/chat/completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "model": "deepseek-flash",
            "thinking_mode": thinking_mode,
            "native_json_mode": True,
            "auth_header": "Authorization",
            "auth_scheme": "Bearer",
            "requests_per_minute": 0,
            "timeout_seconds": 1200,
            "retries": 8,
            "max_tokens": 8192,
            "temperature": 0.2,
        },
        "publish": {
            "epubcheck": {"enabled": True, "require": True, "path": ""},
            "pdf": {"tocdepth": 3},
        },
    }
    translation_config["input_epub" if input_epub is not None else "input_json"] = str(input_path)
    write_json(translation_config_path, translation_config)

    manifest = {
        "schema_version": 1,
        "book_id": book_id,
        "source_adapter": source_adapter,
        "source_path": str(input_path),
        "project_dir": str(project_path),
        "chapter_root": (
            None
            if input_epub is not None or getattr(sys, "frozen", False)
            else str(DEFAULT_CHAPTER_ROOT)
        ),
        "chapter_config": chapter_config_value,
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

    script = "new_book.py"
    if input_epub is not None:
        runbook = f"""# {book_title_zh}：Native EPUB 翻译运行说明

1. 检查 EPUB 包并生成原生分段、兼容性报告和上下文结构（不会调用模型）：

   `python3 {script} prepare --project {project_path}`

   只有 `COMPATIBLE_REFLOWABLE` 会进入正式翻译；加密、纯图片、固定版式、
   scripted/remote-resource 等兼容性风险会被 fail-closed 阻止。

2. 若状态为 `needs_glossary_review`，设置 `DEEPSEEK_API_KEY` 后默认让 LLM 自动完成术语判断与译名：

   `python3 {script} auto-glossary --project {project_path}`

   该步骤会保存 `glossary_llm_review.jsonl` 审计记录并自动编译决定；人工术语审核仅作为异常兜底。然后重新执行 prepare。

3. 设置 `DEEPSEEK_API_KEY` 后做零网络预检，再按累计目标翻译：

   `python3 {script} preflight --project {project_path}`
   `python3 {script} translate --project {project_path} --target-completed 10`
   `python3 {script} translate --project {project_path} --target-completed 100`
   `python3 {script} translate --project {project_path} --all`

4. 完成率 100% 后原位回写 DOM、验证结构并重新打包：

   `python3 {script} render --project {project_path}`

   正式结果写入 `{project_path / 'exports' / (book_id + '.zh-CN.epub')}`。
   原 EPUB 始终只读；内嵌封面和非文本资源保持原文件。
"""
    else:
        runbook = f"""# {book_title_zh}：新书翻译运行说明

1. 先做一次纯离线章节恢复和清理：

   `python3 {script} prepare --project {project_path}`

2. 设置 `DEEPSEEK_API_KEY` 后运行 Vision 章节增强：

   `python3 {script} vision-structure --project {project_path}`

   默认优先读取与 OCR JSON 同名的 PDF 并用 PyMuPDF 渲染页面；没有同名 PDF 时退回 OCR JSON 的 `inputImage`。也可以显式指定：
   `python3 {script} vision-structure --project {project_path} --pdf /path/to/source.pdf`

   Vision 会识别目录页、抄录目录层级，并扫描 OCR/PDF 规则筛出的正文嫌疑页以补充目录未列出的次级节。结果保存在 `structure/vision_structure.json`，并绑定 OCR SHA-256。

3. 若增强后仍为 `needs_structure_review`，人工编辑 `structure_decisions.template.jsonl`，再运行：

   `python3 {script} compile-reviews --project {project_path} --decisions {project_path / 'structure_decisions.jsonl'}`

   然后重新执行 prepare。Vision 无 OCR 文本匹配的标题会保持低置信度等待人工复核，不得为了过门禁直接批准。

4. 章节门禁通过后，若状态为 `needs_glossary_review`，设置 `DEEPSEEK_API_KEY` 并默认自动完成术语判断与译名：

   `python3 {script} auto-glossary --project {project_path}`

   LLM 会对所有候选做 include/reject、给出统一译名与置信度；低置信度项自动进入第二轮复核。决定和 usage 会留档，人工术语审核只作为异常兜底。然后重新执行 prepare。

5. 在当前终端设置 `DEEPSEEK_API_KEY` 后做零网络预检：

   `python3 {script} preflight --project {project_path}`

6. 建议先累计翻译 10 个片段，再逐步扩大：

   `python3 {script} translate --project {project_path} --target-completed 10`

   断点续跑时把累计目标改为 100、500 等。确认后整本运行使用显式 `--all`。

7. 完成率达到 100% 后渲染：

   `python3 {script} render --project {project_path}`

8. 用本地排版生成器创建并登记封面：

   `python3 {script} generate-cover --project {project_path} --theme auto`

9. 生成带目录的 PDF 和带封面的 EPUB：

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


def _build_epub_context_structure(book_id: str, segments: Sequence[Segment]) -> Dict[str, Any]:
    """Build a lightweight heading tree for contextual windows/cache semantics.

    The EPUB DOM/OPF remains the publication authority. This tree only gives
    contextual_v2 stable chapter/section ancestry without flattening the book.
    """
    root_id = f"book-{book_id}"
    nodes: list[Dict[str, Any]] = [{
        "id": root_id,
        "kind": "book",
        "level": 0,
        "title_source": "",
        "parent_id": None,
        "page_json": None,
        "zone": "mainmatter",
        "adapter": "epub_native_v1",
    }]
    stacks: Dict[str, list[tuple[int, str]]] = {}
    for segment in segments:
        metadata = segment.metadata or {}
        href = str(metadata.get("href") or "")
        if segment.kind == "attribute":
            # Attribute slots are emitted after document text segmentation, so
            # their original parent hint is not a reliable reading-order cue.
            segment.parent_node_id = None
            continue
        if segment.kind != "heading":
            continue
        level = int(segment.level or 1)
        stack = stacks.setdefault(href, [])
        while stack and stack[-1][0] >= level:
            stack.pop()
        kind = "chapter" if not stack else "section"
        parent_id = stack[-1][1] if stack else root_id
        segment.parent_node_id = parent_id
        segment.metadata["node_id"] = segment.id
        segment.metadata["node_kind"] = kind
        nodes.append({
            "id": segment.id,
            "kind": kind,
            "level": level,
            "title_source": strip_tokens(segment.source_text),
            "parent_id": parent_id,
            "page_json": None,
            "zone": segment.zone,
            "href": href,
            "spine_index": metadata.get("spine_index"),
            "adapter": "epub_native_v1",
        })
        stack.append((level, segment.id))
    return {
        "schema_version": 1,
        "book_id": book_id,
        "adapter": "epub_native_v1",
        "nodes": nodes,
    }


def _prepare_epub_project(project: Dict[str, Any]) -> Dict[str, Any]:
    config = _translation_config(project)
    source = resolve_path(config, "input_epub")
    structure_dir = Path(project["structure_dir"])
    structure_dir.mkdir(parents=True, exist_ok=True)
    work_dir = resolve_path(config, "output_dir")
    work_dir.mkdir(parents=True, exist_ok=True)

    reports = inspect_epub(source)
    for name in ("epub_inventory", "source_link_graph", "compatibility_report"):
        write_json(structure_dir / f"{name}.json", reports[name])
    compatibility = reports["compatibility_report"]
    review_path = structure_dir / "review_packets.jsonl"
    write_jsonl(review_path, [])

    if compatibility.get("status") != "COMPATIBLE_REFLOWABLE":
        validation = {
            "schema_version": 1,
            "ok": False,
            "adapter": "epub_native_v1",
            "status": "blocked_epub_compatibility",
            "compatibility_status": compatibility.get("status"),
            "compatibility_report": str(structure_dir / "compatibility_report.json"),
        }
        write_json(structure_dir / "validation.json", validation)
        result = {
            "schema_version": 1,
            "book_id": project["book_id"],
            "status": "blocked_epub_compatibility",
            "validation_ok": False,
            "compatibility": compatibility,
            "translation_started": False,
        }
        write_json(Path(project["project_dir"]) / "workflow_status.json", result)
        return result

    segmented = segment_epub(source, str(config["book_id"]))
    segments = list(segmented["segments"])
    structure = _build_epub_context_structure(str(config["book_id"]), segments)
    structure_path = structure_dir / "book_structure.json"
    validation_path = structure_dir / "validation.json"
    write_json(structure_path, structure)
    write_json(validation_path, {
        "schema_version": 1,
        "ok": True,
        "adapter": "epub_native_v1",
        "status": "compatible",
        "compatibility_status": compatibility.get("status"),
        "checks": ["ocf", "package", "navigation", "links", "resources"],
    })

    cleaning = dict(segmented["cleaning_report"])
    cleaning["compatibility_status"] = compatibility.get("status")
    segments_path = work_dir / "cleaned_segments.jsonl"
    report_path = work_dir / "cleaning_report.json"
    write_jsonl(segments_path, [segment.to_dict() for segment in segments])
    write_json(report_path, cleaning)
    write_json(work_dir / "source_manifest.json", segmented["source_manifest"])
    write_clean_manifest(config, segments_path, report_path)

    glossary = generate_glossary(config)
    status = translation_status(config)
    result = {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": "ready_to_translate" if glossary.get("approved") else "needs_glossary_review",
        "validation_ok": True,
        "pending_structure_reviews": 0,
        "source_adapter": "epub_native_v1",
        "compatibility": compatibility,
        "cleaning_metrics": cleaning["metrics"],
        "translation": status,
        "glossary": glossary,
        "translation_started": False,
    }
    write_json(Path(project["project_dir"]) / "workflow_status.json", result)
    return result


def prepare_project(project_path: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    if _is_epub_project(project):
        return _prepare_epub_project(project)
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


def vision_enhance_project(
    project_path: Path,
    pdf_path: Optional[Path] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Collect DeepSeek Vision evidence, bind it to chapter config, and re-prepare."""
    project = _project_manifest(project_path)
    if _is_epub_project(project):
        raise ValueError("Native EPUB projects already carry structural markup; vision chapter recovery is for OCR/PDF projects")
    chapter_config_path = Path(project["chapter_config"]).expanduser().resolve()
    chapter_config = json.loads(chapter_config_path.read_text(encoding="utf-8"))
    translation_config = _translation_config(project)
    settings = dict(chapter_config.get("vision") or {})
    if settings and not bool(settings.get("enabled", True)):
        raise RuntimeError("Structure vision is disabled in chapter_config.json")
    evidence = collect_structure_vision(
        chapter_config,
        dict(translation_config.get("provider") or {}),
        config_base=chapter_config_path.parent,
        pdf_override=pdf_path,
        client=client,
    )
    structure_dir = Path(project["structure_dir"]).expanduser().resolve()
    structure_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = structure_dir / "vision_structure.json"
    write_json(evidence_path, evidence)
    chapter_config["vision_evidence"] = str(evidence_path)
    if pdf_path is not None:
        chapter_config["pdf_path"] = str(pdf_path.expanduser().resolve())
    write_json(chapter_config_path, chapter_config)
    prepared = prepare_project(project_path)
    return {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": "vision_structure_collected",
        "model": evidence.get("model"),
        "toc_pages": len([row for row in evidence.get("toc_pages", []) if row.get("is_toc")]),
        "vision_headings": len(evidence.get("headings", [])),
        "scanned_toc_pages": len((evidence.get("scan") or {}).get("toc_pages_scanned") or []),
        "scanned_body_pages": len((evidence.get("scan") or {}).get("body_pages_scanned") or []),
        "usage": evidence.get("usage"),
        "evidence_path": str(evidence_path),
        "prepared": prepared,
    }

def compile_reviews(project_path: Path, decisions: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    if _is_epub_project(project):
        raise ValueError("Native EPUB projects do not use OCR structure review decisions")
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


def auto_review_project_glossary(
    project_path: Path,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Resolve the glossary gate with the configured LLM, then compile it.

    The deterministic candidate generator and existing compiler remain the
    source of truth. The model only produces auditable include/reject decisions
    and translations for generated candidates.
    """
    project = _project_manifest(project_path)
    config = _translation_config(project)
    require_structure_gate(config)
    require_fresh_cleaning(config)
    manifest = generate_glossary(config)
    if manifest.get("approved"):
        return {
            "schema_version": 1,
            "book_id": project["book_id"],
            "status": "already_approved",
            "review": None,
            "compiled": manifest,
        }
    settings = ((config.get("glossary_settings") or {}).get("llm_review") or {})
    if settings and not bool(settings.get("enabled", True)):
        raise RuntimeError("LLM glossary review is disabled in glossary_settings.llm_review")
    review = auto_review_glossary(config, client=client)
    decisions = Path(str(review["decisions_path"])).expanduser().resolve()
    compiled = compile_glossary_decisions(config, decisions)
    if not compiled.get("approved"):
        raise RuntimeError(
            f"LLM glossary review did not clear the gate: {compiled.get('unresolved_terms')} unresolved"
        )
    return {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": "approved",
        "review": review,
        "compiled": compiled,
    }


def project_status(project_path: Path) -> Dict[str, Any]:
    project = _project_manifest(project_path)
    structure_dir = Path(project["structure_dir"])
    validation_path = structure_dir / "validation.json"
    review_path = structure_dir / "review_packets.jsonl"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    validation_ok = bool(validation.get("ok"))
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
    vision_path = structure_dir / "vision_structure.json"
    vision = None
    if vision_path.is_file():
        try:
            value = json.loads(vision_path.read_text(encoding="utf-8"))
            vision = {
                "model": value.get("model"),
                "toc_pages": len([row for row in value.get("toc_pages", []) if row.get("is_toc")]),
                "headings": len(value.get("headings", [])),
                "usage": value.get("usage"),
                "path": str(vision_path),
            }
        except (OSError, ValueError, TypeError):
            vision = {"path": str(vision_path), "error": "unreadable"}
    if _is_epub_project(project) and validation.get("status") == "not_prepared":
        status = "needs_prepare"
    elif _is_epub_project(project) and not validation_ok:
        status = "blocked_epub_compatibility"
    elif not validation_ok or pending:
        status = "needs_structure_review"
    elif not cleaned or not fresh:
        status = "needs_prepare"
    elif not glossary_ready:
        status = "needs_glossary_review"
    elif translation and translation["ready_to_render"]:
        status = "ready_to_render"
    else:
        status = "ready_to_translate"
    cost = None
    if translation and (config.get("translation") or {}).get("context_mode") == "contextual_v2":
        from .cost_control import cost_report

        cost = cost_report(
            config,
            output_dir / "usage.jsonl",
            int(translation.get("remaining_segments") or 0),
            completed_segments=int(translation.get("completed_segments") or 0),
        )
    return {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": status,
        "validation_ok": validation_ok,
        "pending_structure_reviews": pending,
        "source_adapter": project.get("source_adapter"),
        "compatibility_status": validation.get("compatibility_status"),
        "cleaning_fresh": fresh,
        "stale_reason": stale_reason,
        "glossary_ready": glossary_ready,
        "glossary_reason": glossary_reason,
        "translation": translation,
        "cost": cost,
        "vision_structure": vision,
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
    contextual = (config.get("translation") or {}).get("context_mode") == "contextual_v2"
    if contextual:
        estimated_requests = len(pending)
    else:
        chunk = config.get("chunk", {})
        estimated_requests = len(list(_batches(
            pending, int(chunk.get("max_chars", 10000)), int(chunk.get("max_segments", 20))
        )))
    status = translation_status(config)
    return {
        "schema_version": 1,
        "book_id": project["book_id"],
        "status": "preflight_passed",
        "model": client.model,
        "thinking_mode": client.thinking_mode,
        "api_key_present": True,
        "external_requests_made": 0,
        "required_segments": status["required_segments"],
        "completed_segments": status["completed_segments"],
        "remaining_segments": status["remaining_segments"],
        "estimated_remaining_requests": estimated_requests,
        "estimated_remaining_batches": estimated_requests,
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
    config = _translation_config(project)
    if _is_epub_config(config):
        return render_epub(
            config,
            allow_partial=allow_partial,
            project_dir=Path(project["project_dir"]),
        )
    return render_book(config, allow_partial=allow_partial)


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
    config = _translation_config(project)
    if _is_epub_config(config):
        if output_format == "pdf":
            raise ValueError("Native EPUB projects preserve the source package and export EPUB only")
        report = render_epub(config, allow_partial=False, project_dir=Path(project["project_dir"]))
        return {
            "schema_version": 1,
            "book_id": config["book_id"],
            "source_adapter": "epub_native_v1",
            "outputs": {"epub": {"path": report["output"], "verification": report}},
            "skipped_formats": ["pdf"] if output_format == "both" else [],
        }
    formats = ("pdf", "epub") if output_format == "both" else (output_format,)
    return export_book(config, Path(project["project_dir"]), formats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and translate an OCR JSON or native EPUB book safely")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a self-contained new-book project")
    source = init.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path)
    source.add_argument("--input-epub", type=Path)
    init.add_argument("--book-id", required=True)
    init.add_argument("--book-title", required=True)
    init.add_argument("--book-title-zh", required=True)
    init.add_argument("--project-dir", type=Path)
    init.add_argument("--source-lang", default="英语")
    init.add_argument("--target-lang", default="简体中文")
    init.add_argument("--domain", default="学术人文社科")
    init.add_argument("--author", default="")
    init.add_argument("--toc-search-end", type=int, default=30)
    init.add_argument("--thinking-mode", choices=("disabled", "low", "high", "max"), default="disabled")
    for name in ("prepare", "status", "preflight", "render"):
        command = commands.add_parser(name)
        command.add_argument("--project", type=Path, required=True)
        if name == "render":
            command.add_argument("--allow-partial", action="store_true")
    vision = commands.add_parser(
        "vision-structure",
        help="use DeepSeek Vision to recover TOC pages and hidden body headings",
    )
    vision.add_argument("--project", type=Path, required=True)
    vision.add_argument("--pdf", type=Path)
    compile_command = commands.add_parser("compile-reviews")
    compile_command.add_argument("--project", type=Path, required=True)
    compile_command.add_argument("--decisions", type=Path, required=True)
    glossary = commands.add_parser("glossary", help="generate the master subset and candidate review packets")
    glossary.add_argument("--project", type=Path, required=True)
    glossary.add_argument("--force", action="store_true")
    compile_glossary = commands.add_parser("compile-glossary", help="compile reviewed terminology decisions")
    compile_glossary.add_argument("--project", type=Path, required=True)
    compile_glossary.add_argument("--decisions", type=Path, required=True)
    auto_glossary = commands.add_parser(
        "auto-glossary",
        help="use DeepSeek to decide and compile all terminology candidates",
    )
    auto_glossary.add_argument("--project", type=Path, required=True)
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
            input_json=args.input_json, input_epub=args.input_epub, book_id=args.book_id,
            book_title=args.book_title, book_title_zh=args.book_title_zh,
            project_dir=args.project_dir, source_lang=args.source_lang,
            target_lang=args.target_lang, domain=args.domain, author=args.author,
            toc_search_end=args.toc_search_end,
            thinking_mode=args.thinking_mode,
        )
    elif args.command == "prepare":
        result = prepare_project(args.project)
    elif args.command == "vision-structure":
        result = vision_enhance_project(args.project, args.pdf)
    elif args.command == "compile-reviews":
        result = compile_reviews(args.project, args.decisions)
    elif args.command == "glossary":
        result = generate_project_glossary(args.project, args.force)
    elif args.command == "compile-glossary":
        result = compile_project_glossary(args.project, args.decisions)
    elif args.command == "auto-glossary":
        result = auto_review_project_glossary(args.project)
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
