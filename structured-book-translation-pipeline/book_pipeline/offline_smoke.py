from __future__ import annotations

"""Disposable, synthetic end-to-end check; never calls DeepSeek or publishes a book."""

import json
import os
from pathlib import Path
import sys
import zipfile
from tempfile import TemporaryDirectory
from typing import Any

from .config import load_config
from .io_utils import read_jsonl, write_json, write_jsonl
from .review_gui import ReviewWindow
from .render import render_book
from .contextual import translate_v2
from .source_review import choose_source_review
from .workflow import (
    auto_review_project_glossary, export_project, generate_project_cover,
    initialize_project, preflight_project, prepare_project, vision_enhance_project,
)


class SyntheticClient:
    model = "demo-fake-offline-smoke"

    def complete(self, _system: str, user: str) -> tuple[str, dict[str, int]]:
        payload = json.loads(user)
        rows = [[row["n"], "译：" + row["text"]] for row in payload["targets"]]
        return json.dumps({"t": rows}, ensure_ascii=False), {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }


class SyntheticStructureVisionClient:
    model = "demo-fake-structure-vision"

    def complete_multimodal_json_text(self, _system: str, content: list[dict[str, Any]]):
        page_ids = [
            int(block["text"].split("\n", 1)[0].split("=", 1)[1])
            for block in content
            if block.get("type") == "text" and str(block.get("text") or "").startswith("PAGE_JSON=")
        ]
        instruction = str(content[0].get("text") or "")
        if "table-of-contents" in instruction:
            payload = {
                "pages": [
                    {"page_json": page, "is_toc": False, "confidence": 0.99, "entries": []}
                    for page in page_ids
                ]
            }
        else:
            payload = {
                "pages": [{"page_json": page, "headings": []} for page in page_ids]
            }
        return json.dumps(payload), {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }


class SyntheticGlossaryClient:
    model = "demo-fake-glossary-review"

    def complete_json_text(self, _system: str, user: str) -> tuple[str, dict[str, int]]:
        payload = json.loads(user)
        decisions = [
            {
                "candidate_id": row["candidate_id"],
                "decision": "reject",
                "translation": "",
                "category": "",
                "alternatives": [],
                "note": "",
                "reason": "Synthetic fixture: not a reusable book-level term",
                "confidence": 0.99,
            }
            for row in payload["candidates"]
        ]
        return json.dumps({"decisions": decisions}, ensure_ascii=False), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def _sample_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "examples" / "sample_ocr.json"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2] / "examples" / "sample_ocr.json"


def _write_messy_epub(path: Path) -> Path:
    """Build a noncanonical-but-readable EPUB for frozen importer smoke."""
    container = """<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    package = """<package version="3.0"><metadata><dc:title>Smoke EPUB</dc:title>
<meta name="cover" content="cover"/></metadata><manifest>
<item id="cover" href="Images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
<item id="ch1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>
</manifest><spine><itemref idref="ch1"/></spine></package>"""
    chapter = """<html><head><title>Smoke Chapter</title></head><body>
<h1>Smoke Chapter</h1><p>Hello <em>messy EPUB.</p><p>Second paragraph.</p>
</body></html>"""
    with zipfile.ZipFile(path, "w") as archive:
        # Deliberately violate the strict OCF mimetype placement/compression rule.
        archive.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OPS/package.opf", package, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OPS/Text/ch1.xhtml", chapter, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OPS/Images/cover.jpg", b"\xff\xd8\xffFROZEN-SMOKE-COVER", compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_DEFLATED)
    return path


def run_offline_smoke() -> dict[str, Any]:
    sample = _sample_path()
    if not sample.is_file():
        raise FileNotFoundError(f"Missing bundled synthetic sample: {sample}")
    previous_key = os.environ.get("DEEPSEEK_API_KEY")
    try:
        with TemporaryDirectory(prefix="deepseek-book-smoke-") as temporary:
            temporary_path = Path(temporary)
            source = temporary_path / "source.json"
            source.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
            sample_pages = json.loads(source.read_text(encoding="utf-8"))
            import fitz

            pdf = fitz.open()
            for page_index in range(len(sample_pages)):
                page = pdf.new_page(width=612, height=792)
                page.insert_text((72, 72), f"Synthetic page {page_index + 1}", fontsize=10)
            pdf.save(temporary_path / "source.pdf")
            pdf.close()

            project = temporary_path / "sample-book"
            initialize_project(input_json=source, book_id="synthetic-sample",
                               book_title="Synthetic Sample", book_title_zh="合成样例",
                               author="Demo", project_dir=project)
            prepared = prepare_project(project)
            vision = vision_enhance_project(project, client=SyntheticStructureVisionClient())
            prepared = vision["prepared"]
            if prepared["status"] != "needs_source_review_choice":
                raise AssertionError(f"Unexpected first gate: {prepared['status']}")
            source_review_path = project / "source_review.md"
            if not source_review_path.is_file() or "BOOK_SEGMENT" not in source_review_path.read_text(encoding="utf-8"):
                raise AssertionError("Reviewed-source Markdown was not generated")
            config_path = project / "translation_config.json"
            choose_source_review(load_config(config_path), project, "auto")
            prepared = prepare_project(project)
            if prepared["status"] != "needs_glossary_review":
                raise AssertionError(f"Source-review gate did not clear: {prepared['status']}")
            candidates = list(read_jsonl(project / "glossary_candidates.jsonl"))
            if not candidates:
                raise AssertionError("Synthetic sample must exercise the glossary review gate")
            automatic = auto_review_project_glossary(project, SyntheticGlossaryClient())
            compiled = automatic["compiled"]
            prepared = prepare_project(project)
            if prepared["status"] != "ready_to_translate":
                raise AssertionError(f"Review gate did not clear: {prepared['status']}")
            if automatic["review"]["candidate_count"] != len(candidates):
                raise AssertionError("Automatic glossary review did not cover all candidates")
            cover = generate_project_cover(project)
            if not Path(cover["image_path"]).is_file():
                raise AssertionError("Cover not generated")
            # A dummy key is local to this process. preflight makes zero requests.
            os.environ["DEEPSEEK_API_KEY"] = "offline-smoke-not-a-real-key"
            preflight = preflight_project(project)
            if preflight["external_requests_made"] != 0:
                raise AssertionError("Preflight contacted an API")
            if os.name == "nt":
                import tkinter as tk

                root = tk.Tk()
                root.withdraw()
                review_window = ReviewWindow(root, project, "glossary", lambda *_: None)
                root.update_idletasks()
                review_window.window.destroy()
                root.destroy()
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["allow_demo_translations"] = True
            config["provider"]["requests_per_minute"] = 0
            write_json(config_path, config)
            loaded = load_config(config_path)
            translated = translate_v2(loaded, SyntheticClient())
            if not translated["ready_to_render"]:
                raise AssertionError("Synthetic translation did not finish")
            rendered = render_book(loaded)
            markdown = Path(rendered["output"])
            if not markdown.is_file() or "译：" not in markdown.read_text(encoding="utf-8"):
                raise AssertionError("Markdown output missing")
            config.pop("allow_demo_translations")
            write_json(config_path, config)
            try:
                export_project(project, "epub")
            except RuntimeError as exc:
                if "demo or fake" not in str(exc):
                    raise
            else:
                raise AssertionError("Synthetic translations reached the publish path")

            # Frozen EPUB smoke: a deliberately noncanonical package must still
            # normalize locally into reviewed Markdown with source cover reuse.
            epub_source = _write_messy_epub(temporary_path / "messy-source.epub")
            epub_project = temporary_path / "epub-book"
            initialized_epub = initialize_project(
                input_epub=epub_source,
                book_id="synthetic-epub",
                book_title="Smoke EPUB",
                book_title_zh="合成 EPUB",
                author="Demo",
                project_dir=epub_project,
            )
            if initialized_epub["source_adapter"] != "epub_markdown_v2":
                raise AssertionError("New EPUB project did not select canonical adapter")
            prepared_epub = prepare_project(epub_project)
            if prepared_epub.get("status") != "needs_source_review_choice":
                raise AssertionError(f"Unexpected EPUB gate: {prepared_epub.get('status')}")
            epub_review = (epub_project / "source_review.md").read_text(encoding="utf-8")
            if (
                "EPUB source: OPS/Text/ch1.xhtml" not in epub_review
                or "Hello messy EPUB." not in epub_review
                or "[[EPUB:" in epub_review
            ):
                raise AssertionError("Canonical EPUB reviewed-source output is invalid")
            epub_cover = json.loads((epub_project / "cover" / "cover.json").read_text(encoding="utf-8"))
            if epub_cover.get("selected_by") != "source_epub":
                raise AssertionError("Source EPUB cover was not reused")
            return {
                "status": "ok", "sample_pages": len(sample_pages),
                "vision_structure_collected": True,
                "vision_pdf_rendered": bool(
                    ((json.loads((project / "structure" / "vision_structure.json").read_text(encoding="utf-8")).get("scan") or {}).get("image_sources") or {}).get("pdf_render")
                ),
                "source_review_generated": True,
                "source_review_mode": "auto",
                "glossary_candidates_reviewed": len(candidates),
                "glossary_review_mode": "llm_auto",
                "micro_batch_plan": translated.get("micro_batch_plan"),
                "review_status": compiled["status"],
                "cover_generated": True, "preflight_requests": 0,
                "translated_segments": translated["completed_segments"],
                "markdown_rendered": True, "fake_export_blocked": True,
                "gui_review_constructed": os.name == "nt",
                "pdf_epub_generated": False,
                "epub_canonical_imported": True,
                "epub_source_cover_reused": True,
            }
    finally:
        if previous_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = previous_key
