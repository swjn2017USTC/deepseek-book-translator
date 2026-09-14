from __future__ import annotations

"""Disposable, synthetic end-to-end check; never calls DeepSeek or publishes a book."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from .config import load_config
from .io_utils import read_jsonl, write_json, write_jsonl
from .review_gui import ReviewWindow, decision_from_fields
from .render import render_book
from .translate import translate_book
from .workflow import (
    compile_project_glossary, export_project, generate_project_cover,
    initialize_project, preflight_project, prepare_project,
)


class SyntheticClient:
    model = "demo-fake-offline-smoke"

    def complete(self, _system: str, user: str) -> tuple[str, dict[str, int]]:
        payload = json.loads(user)
        rows = [{"id": row["id"], "translated_text": "译：" + row["text"]} for row in payload["segments"]]
        return json.dumps(rows, ensure_ascii=False), {"prompt_tokens": 0, "completion_tokens": 0}


def _sample_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "examples" / "sample_ocr.json"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2] / "examples" / "sample_ocr.json"


def run_offline_smoke() -> dict[str, Any]:
    sample = _sample_path()
    if not sample.is_file():
        raise FileNotFoundError(f"Missing bundled synthetic sample: {sample}")
    previous_key = os.environ.get("DEEPSEEK_API_KEY")
    try:
        with TemporaryDirectory(prefix="deepseek-book-smoke-") as temporary:
            project = Path(temporary) / "sample-book"
            initialize_project(input_json=sample, book_id="synthetic-sample",
                               book_title="Synthetic Sample", book_title_zh="合成样例",
                               author="Demo", project_dir=project)
            prepared = prepare_project(project)
            if prepared["status"] != "needs_glossary_review":
                raise AssertionError(f"Unexpected first gate: {prepared['status']}")
            candidates = list(read_jsonl(project / "glossary_candidates.jsonl"))
            if not candidates:
                raise AssertionError("Synthetic sample must exercise the glossary review gate")
            decisions = [decision_from_fields("glossary", row["candidate_id"], {
                "decision": "reject", "reason": "Synthetic fixture: not a reusable book term",
                "translation": "", "category": "", "evidence": "",
            }) for row in candidates]
            decisions_path = project / "glossary_decisions.jsonl"
            write_jsonl(decisions_path, decisions)
            compiled = compile_project_glossary(project, decisions_path)
            prepared = prepare_project(project)
            if prepared["status"] != "ready_to_translate":
                raise AssertionError(f"Review gate did not clear: {prepared['status']}")
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
            config_path = project / "translation_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["allow_demo_translations"] = True
            config["provider"]["requests_per_minute"] = 0
            write_json(config_path, config)
            loaded = load_config(config_path)
            translated = translate_book(loaded, SyntheticClient())
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
            return {
                "status": "ok", "sample_pages": len(json.loads(sample.read_text(encoding="utf-8"))),
                "glossary_candidates_reviewed": len(candidates),
                "review_status": compiled["status"],
                "cover_generated": True, "preflight_requests": 0,
                "translated_segments": translated["completed_segments"],
                "markdown_rendered": True, "fake_export_blocked": True,
                "gui_review_constructed": os.name == "nt",
                "pdf_epub_generated": False,
            }
    finally:
        if previous_key is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = previous_key
