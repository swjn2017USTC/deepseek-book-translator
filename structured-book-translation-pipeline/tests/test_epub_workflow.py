from __future__ import annotations

import json
from pathlib import Path
import zipfile

import book_pipeline.workflow as workflow
from book_pipeline.config import load_config
from book_pipeline.io_utils import read_jsonl
from book_pipeline.workflow import initialize_project, prepare_project, project_status


def _messy_epub(path: Path, *, with_cover: bool = True) -> Path:
    """A deliberately noncanonical but safely readable EPUB.

    - mimetype is compressed and not the first ZIP entry
    - chapter markup is HTML-ish rather than strict XML
    - source contains an embedded cover and body image
    """
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    cover_meta = '<meta name="cover" content="cover-image"/>' if with_cover else ""
    cover_item = '<item id="cover-image" href="Images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>' if with_cover else ""
    package = f"""<package version="3.0">
  <metadata>
    <dc:title>Messy EPUB</dc:title>
    <dc:creator>Test Author</dc:creator>
    {cover_meta}
  </metadata>
  <manifest>
    {cover_item}
    <item id="ch1" href="Text/ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="fig1" href="Images/figure.png" media-type="image/png"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>"""
    # Deliberately mismatched em/p nesting: HTMLParser recovers this, an XML
    # round-trip parser would reject it.
    chapter = """<html><head><title>Chapter One</title></head><body>
<h1>Chapter One</h1>
<p>Hello <em>world.</p>
<p>A second paragraph with formatting oddities.</p>
<img src="../Images/figure.png" alt="Figure one">
</body></html>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OPS/package.opf", package, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OPS/Text/ch1.xhtml", chapter, compress_type=zipfile.ZIP_DEFLATED)
        if with_cover:
            archive.writestr("OPS/Images/cover.jpg", b"\xff\xd8\xffSOURCE-COVER", compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OPS/Images/figure.png", b"\x89PNG\r\n\x1a\nFIGURE", compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_DEFLATED)
    return path


def _init(tmp_path: Path, *, with_cover: bool = True) -> Path:
    source = _messy_epub(tmp_path / "source.epub", with_cover=with_cover)
    project = tmp_path / "project"
    initialize_project(
        input_epub=source,
        book_id="canonical-epub-test",
        book_title="Messy EPUB",
        book_title_zh="标准化 EPUB 测试",
        author="Test Author",
        source_lang="英语",
        target_lang="简体中文",
        project_dir=project,
    )
    return project


def test_epub_defaults_to_canonical_markdown_adapter(tmp_path):
    project = _init(tmp_path)
    manifest = json.loads((project / "project.json").read_text(encoding="utf-8"))
    config_json = json.loads((project / "translation_config.json").read_text(encoding="utf-8"))

    assert manifest["source_adapter"] == "epub_markdown_v2"
    assert config_json["source_adapter"] == "epub_markdown_v2"
    assert project_status(project)["status"] == "needs_prepare"

    prepared = prepare_project(project)
    assert prepared["source_adapter"] == "epub_markdown_v2"
    assert prepared["status"] == "needs_source_review_choice"
    assert Path(prepared["epub_import_report"]).is_file()

    review = (project / "source_review.md").read_text(encoding="utf-8")
    assert "EPUB source: OPS/Text/ch1.xhtml" in review
    assert "Chapter One" in review
    assert "Hello world." in review
    assert "[[EPUB:" not in review
    assert "BOOK_SEGMENT" in review

    rows = list(read_jsonl(project / "work" / "cleaned_segments.jsonl"))
    assert rows
    assert all((row.get("metadata") or {}).get("source_adapter") == "epub_markdown_v2" for row in rows)
    assert not any("[[EPUB:" in str(row.get("source_text") or "") for row in rows)
    image = next(row for row in rows if row["kind"] == "image")
    assert image["translatable"] is False
    assert image["source_text"].startswith("![Figure one](assets/epub/")
    assert (project / image["metadata"]["asset_path"]).is_file()

    structure = json.loads((project / "structure" / "book_structure.json").read_text(encoding="utf-8"))
    assert structure["adapter"] == "epub_markdown_v2"
    assert any(node.get("kind") == "chapter" for node in structure["nodes"])


def test_epub_import_tolerates_noncanonical_mimetype_and_reuses_source_cover(tmp_path):
    project = _init(tmp_path)
    prepared = prepare_project(project)

    source_manifest = json.loads((project / "work" / "source_manifest.json").read_text(encoding="utf-8"))
    assert source_manifest["strict_ocf_mimetype"] is False
    assert source_manifest["cover_asset"]

    cover = json.loads((project / "cover" / "cover.json").read_text(encoding="utf-8"))
    assert cover["selected_by"] == "source_epub"
    assert cover["rights_status"] == "user_approved_personal_use"
    assert Path(cover["image_path"]).is_file()
    assert prepared["cover"]["sha256"] == cover["sha256"]



def test_epub_without_reusable_cover_generates_local_fallback(tmp_path):
    project = _init(tmp_path, with_cover=False)
    prepared = prepare_project(project)

    source_manifest = json.loads((project / "work" / "source_manifest.json").read_text(encoding="utf-8"))
    assert source_manifest["cover_asset"] is None
    cover = json.loads((project / "cover" / "cover.json").read_text(encoding="utf-8"))
    assert cover["selected_by"] == "local_generator"
    assert cover["rights_status"] == "generated"
    assert Path(cover["image_path"]).is_file()
    assert prepared["cover"]["sha256"] == cover["sha256"]

def test_canonical_epub_render_and_export_use_common_markdown_publisher(tmp_path, monkeypatch):
    project = _init(tmp_path)
    prepare_project(project)
    calls = []

    def fake_render(config, allow_partial=False):
        calls.append(("render", config["source_adapter"], allow_partial))
        return {"output": str(Path(config["_base_dir"]) / "work" / "book_translated.md")}

    def fake_export(config, project_dir, formats):
        calls.append(("export", config["source_adapter"], tuple(formats)))
        return {"outputs": {"epub": {"path": str(Path(project_dir) / "exports" / "canonical-epub-test.epub")}}}

    def fail_native(*_args, **_kwargs):
        raise AssertionError("epub_markdown_v2 must not use native DOM renderer")

    monkeypatch.setattr(workflow, "render_book", fake_render)
    monkeypatch.setattr(workflow, "export_book", fake_export)
    monkeypatch.setattr(workflow, "render_epub", fail_native)

    rendered = workflow.render_project(project, allow_partial=True)
    assert rendered["output"].endswith("book_translated.md")
    exported = workflow.export_project(project, "epub")
    assert exported["outputs"]["epub"]["path"].endswith("canonical-epub-test.epub")
    assert calls == [
        ("render", "epub_markdown_v2", True),
        ("export", "epub_markdown_v2", ("epub",)),
    ]


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
