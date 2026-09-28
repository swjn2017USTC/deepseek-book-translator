from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

import book_pipeline.workflow as workflow
from book_pipeline.config import load_config
from book_pipeline.contextual.parser import parse_target_only
from book_pipeline.io_utils import read_jsonl
from book_pipeline.models import Segment
from book_pipeline.workflow import initialize_project, prepare_project, project_status


def _minimal_epub(path: Path) -> Path:
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package = """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" unique-identifier="bookid" xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:test:native-epub</dc:identifier>
    <dc:title>Native EPUB Test</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
  </spine>
</package>
"""
    nav = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Contents</title></head>
  <body><nav epub:type="toc"><ol><li><a href="ch1.xhtml#ch1">Chapter One</a></li></ol></nav></body>
</html>
"""
    chapter = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
  <head><title>Chapter One</title></head>
  <body>
    <h1 id="ch1">Chapter <em>One</em></h1>
    <p>Hello <em>world</em>.</p>
    <p title="A greeting">A second paragraph.</p>
  </body>
</html>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/content.opf", package, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/nav.xhtml", nav, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/ch1.xhtml", chapter, compress_type=zipfile.ZIP_DEFLATED)
    return path


def _init(tmp_path: Path) -> Path:
    source = _minimal_epub(tmp_path / "source.epub")
    project = tmp_path / "project"
    initialize_project(
        input_epub=source,
        book_id="native-epub-test",
        book_title="Native EPUB Test",
        book_title_zh="原生 EPUB 测试",
        author="Test Author",
        source_lang="英语",
        target_lang="简体中文",
        project_dir=project,
    )
    return project


def test_native_epub_init_prepare_and_status(tmp_path):
    project = _init(tmp_path)
    manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
    config_json = json.loads((project / "translation_config.json").read_text(encoding="utf-8"))

    assert manifest["source_adapter"] == "epub_native_v1"
    assert config_json["source_adapter"] == "epub_native_v1"
    assert "input_epub" in config_json
    assert "input_json" not in config_json
    assert project_status(project)["status"] == "needs_prepare"

    prepared = prepare_project(project)
    assert prepared["source_adapter"] == "epub_native_v1"
    assert prepared["compatibility"]["status"] == "COMPATIBLE_REFLOWABLE"
    assert prepared["status"] in {"needs_glossary_review", "ready_to_translate"}

    config = load_config(project / "translation_config.json")
    assert Path(config["input_epub"]).name == "source.epub"

    rows = list(read_jsonl(project / "work" / "cleaned_segments.jsonl"))
    assert rows
    assert all((row.get("metadata") or {}).get("source_adapter") == "epub_native_v1" for row in rows)
    assert any("[[EPUB:" in str(row.get("source_text") or "") for row in rows)

    structure = json.loads((project / "structure" / "book_structure.json").read_text(encoding="utf-8"))
    assert structure["adapter"] == "epub_native_v1"
    assert any(node.get("kind") == "chapter" for node in structure["nodes"])

    candidate_path = project / "glossary_candidates.jsonl"
    if candidate_path.exists():
        candidates = list(read_jsonl(candidate_path))
        assert all("EPUB" not in str(row.get("term") or "") for row in candidates)


def test_native_epub_contextual_parser_preserves_protected_tokens(tmp_path):
    project = _init(tmp_path)
    prepare_project(project)
    rows = list(read_jsonl(project / "work" / "cleaned_segments.jsonl"))
    target = Segment(**next(row for row in rows if "[[EPUB:" in str(row.get("source_text") or "")))

    valid = json.dumps(
        {"id": target.id, "translated_text": target.source_text},
        ensure_ascii=False,
    )
    assert parse_target_only(valid, target) == target.source_text

    changed = target.source_text.replace("[[EPUB:", "[[BROKEN:", 1)
    invalid = json.dumps({"id": target.id, "translated_text": changed}, ensure_ascii=False)
    with pytest.raises(ValueError, match="EPUB"):
        parse_target_only(invalid, target)


def test_native_epub_render_and_export_dispatch_to_native_renderer(tmp_path, monkeypatch):
    project = _init(tmp_path)
    calls = []

    def fake_render(config, *, allow_partial=False, project_dir=None):
        calls.append((config["source_adapter"], allow_partial, Path(project_dir)))
        return {"output": str(Path(project_dir) / "exports" / "native-epub-test.zh-CN.epub")}

    monkeypatch.setattr(workflow, "render_epub", fake_render)

    rendered = workflow.render_project(project, allow_partial=True)
    assert rendered["output"].endswith("native-epub-test.zh-CN.epub")
    assert calls[-1][0] == "epub_native_v1"
    assert calls[-1][1] is True

    exported = workflow.export_project(project, "epub")
    assert exported["source_adapter"] == "epub_native_v1"
    assert exported["outputs"]["epub"]["path"].endswith("native-epub-test.zh-CN.epub")

    with pytest.raises(ValueError, match="EPUB only"):
        workflow.export_project(project, "pdf")


def test_cli_init_accepts_exactly_one_epub_or_json_source():
    parser = workflow.build_parser()
    args = parser.parse_args([
        "init",
        "--input-epub", "book.epub",
        "--book-id", "book",
        "--book-title", "Book",
        "--book-title-zh", "书",
    ])
    assert args.input_epub == Path("book.epub")
    assert args.input_json is None
