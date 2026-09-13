"""P08 publication layer tests (spec §14, plan decision 7).

Convention: tests that execute a real external tool are guarded with
``skipif``; deterministic absent/present cases use monkeypatch so they
always run. The Lua filters under test live in ``pandoc_lua/`` at the repo
root and are exercised through the production module helpers so asset names
cannot drift from the code that loads them.
"""

import base64
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import book_pipeline.publish as publish_module
from book_pipeline.clean import block_id, clean_book, raw_blocks
from book_pipeline.config import load_config
from book_pipeline.io_utils import read_jsonl, write_jsonl
from book_pipeline.publish import (
    _apply_text_level_clean,
    _epubcheck_discover,
    _escape_table_cell_list_markers,
    _filters_dir,
    _pandoc_args_for_tex,
    _publish_settings,
    _required_toc_depth,
    _run_epub_validation,
    _verify_bookmark_depth,
    _verify_toc_depth,
    build_book_tex,
    clean_for_latex,
    export_book,
    register_cover,
)
from book_pipeline.render import render_book
from book_pipeline.translate import translate_book


FIXTURES = Path(__file__).parent / "fixtures"

PANDOC = shutil.which("pandoc")
require_pandoc = pytest.mark.skipif(PANDOC is None, reason="pandoc not installed")


def _epubcheck_present() -> bool:
    return _epubcheck_discover(_publish_settings({})) is not None


require_epubcheck = pytest.mark.skipif(
    PANDOC is None or not _epubcheck_present(),
    reason="epubcheck (or pandoc) not installed",
)


def _pandoc_native(source: str, filters=(), metadata=()) -> str:
    command = [PANDOC, "-f", "markdown-raw_tex-yaml_metadata_block", "-t", "native"]
    for name in filters:
        command += ["--lua-filter", str(_filters_dir() / name)]
    for key, value in metadata:
        command += ["--metadata", f"{key}={value}"]
    completed = subprocess.run(
        command, input=source, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


# ---------------------------------------------------------------------------
# Test 1: legacy vs Lua-filter byte parity on the synthetic edge fixture
# ---------------------------------------------------------------------------

PARITY_FIXTURE = """# 测试书

测试作者

<!-- PAGE_JSON: 0001 -->

## 第一章 *重点* 与 `代码`

正文 ¹ 且 α ≤ β 且 $^{1}$ $^2$ 结尾 12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890。

<!-- NOTRANSLATE:BEGIN keep -->
原文段落
<!-- NOTRANSLATE:END -->

> ## 引言内的标题

### 第一节 **粗体**

内容。


<!-- PAGE_JSON:0002 -->

结论。
"""


@require_pandoc
def test_lua_filter_parity_legacy_vs_filter(tmp_path):
    def body_tex(config, directory: Path) -> str:
        text, filter_args = _pandoc_args_for_tex(config, PARITY_FIXTURE, "测试书", "测试作者")
        prepared = directory / "input.md"
        prepared.write_text(text, encoding="utf-8")
        output = directory / "body.tex"
        command = [
            PANDOC, str(prepared), "-f", "markdown-raw_tex-yaml_metadata_block",
            "-t", "latex", "-o", str(output), "--top-level-division=chapter",
        ]
        if filter_args:
            command += ["--metadata", "title=测试书", "--metadata", "author=测试作者"]
            command += filter_args
        completed = subprocess.run(
            command, cwd=directory, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        return output.read_text(encoding="utf-8")

    legacy_dir = tmp_path / "legacy"
    filter_dir = tmp_path / "filter"
    legacy_dir.mkdir()
    filter_dir.mkdir()
    legacy = body_tex({}, legacy_dir)
    filtered = body_tex({"publish": {"lua_filters": True}}, filter_dir)

    assert filtered == legacy
    # Sanity: the shared result really went through the whole pipeline.
    assert "\\chapter{" in legacy
    assert "\\section{" in legacy
    assert "PAGE_JSON" not in legacy
    assert "NOTRANSLATE" not in legacy
    assert "引言内的标题" in legacy


# ---------------------------------------------------------------------------
# Tests 2-4: per-filter contracts via `pandoc ... -t native`
# ---------------------------------------------------------------------------


@require_pandoc
def test_promote_headings_filter_contract():
    source = "# 书名\n\n## 章一\n\n### 节一\n\n#### 小节\n"
    out = _pandoc_native(source, filters=("promote_headings.lua",))
    levels = [
        int(match)
        for match in re.findall(r"Header\n\s+(\d+)\s*(?:\n\s*)?\(", out)
    ]
    assert levels == [1, 1, 2, 3]  # title H1 untouched; H2->H1; H3->H2; H4->H3


@require_pandoc
def test_strip_comments_filter_contract():
    source = (
        "正文一。\n\n"
        "<!-- PAGE_JSON: 0001 -->\n\n"
        "正文二。\n\n"
        "<!-- PAGE_JSON:0002 -->\n\n"
        "<!-- NOTRANSLATE:BEGIN zone-x -->\n"
        "原文内容\n"
        "<!-- NOTRANSLATE:END -->\n\n"
        "<!-- IMAGE block_ids=7 bbox=[1,2] -->\n\n"
        "正文三。\n"
    )
    out = _pandoc_native(source, filters=("strip_comments.lua",))
    assert "PAGE_JSON" not in out
    assert "NOTRANSLATE" not in out
    assert "IMAGE block_ids=7" in out  # IMAGE comments preserved, as in legacy


@require_pandoc
def test_drop_title_page_filter_contract():
    source = (
        "# Book Title\n\n"
        "Jane Author\n\n"
        "Intro paragraph.\n\n"
        "# Another Chapter\n\n"
        "## A Section\n"
    )
    out = _pandoc_native(
        source,
        filters=("drop_title_page.lua",),
        metadata=[("title", "Book Title"), ("author", "Jane Author")],
    )
    assert 'Str "Book' not in out and 'Str "Title"' not in out
    assert 'Str "Jane' not in out and 'Str "Author"' not in out
    assert 'Str "Intro"' in out
    assert 'Str "Another"' in out and 'Str "Chapter"' in out  # later H1 kept
    assert 'Str "Section"' in out


# ---------------------------------------------------------------------------
# Test 5: clean_for_latex byte-identical regression guard (golden string
# captured from the pre-refactor implementation)
# ---------------------------------------------------------------------------

GOLDEN_MARKDOWN = """# 测试书

测试作者

<!-- PAGE_JSON: 0001 -->

## 第一章 *重点*

正文 ¹ 且 α ≤ β 且 $^{1}$ $^2$ 超长数字 12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 结尾。

<!-- NOTRANSLATE:BEGIN keep -->
原文段落

<!-- NOTRANSLATE:END -->

### 第一节 `代码`

内容。


<!-- PAGE_JSON:0002 -->

结论。
"""

# Byte-identical output of the historical clean_for_latex on GOLDEN_MARKDOWN
# (sha256 230d74c20a4a0cc7a954fcb39d99e6cbc9e20bca31fb261f4f4d60375cb3c89a).
GOLDEN_EXPECTED = (
    '# 第一章 *重点*\n\n'
    '正文 $^{1}$ 且 $\\alpha$ $\\leq$ $\\beta$ 且 $^{1}$ $^{2}$ 超长数字  结尾。\n\n\n'
    '原文段落\n\n\n'
    '## 第一节 `代码`\n\n'
    '内容。\n\n\n'
    '结论。\n'
)


def test_escape_table_cell_list_markers_for_epub():
    """PaddleOCR tables are raw HTML; a cell starting with a Markdown list
    marker makes Pandoc emit an unclosed <ol><li> inside the raw <td>
    (EPUBCheck FATAL RSC-016 + cascading RSC-012).  Live regression:
    the-dictators-dilemma ch005/ch007/ch010."""
    markdown = (
        "<table><tr>"
        "<td style='text-align: center;'>1. 即使我可以在世界上任意选择国家。</td>"
        "<td style='text-align: center;'>26.4</td>"
        "<td>- 项目二</td>"
        "</tr></table>\n"
    )
    cleaned = _escape_table_cell_list_markers(markdown)
    assert "\\1. 即使我可以在世界上任意选择国家。" in cleaned
    assert "\\- 项目二" in cleaned
    assert ">26.4<" in cleaned  # numeric data cells stay untouched


def test_clean_for_latex_still_byte_identical():
    prepared = clean_for_latex(GOLDEN_MARKDOWN, "测试书", "测试作者")
    assert prepared == GOLDEN_EXPECTED
    assert hashlib.sha256(prepared.encode("utf-8")).hexdigest() == (
        "230d74c20a4a0cc7a954fcb39d99e6cbc9e20bca31fb261f4f4d60375cb3c89a"
    )
    # The two preparation paths share the identical text-level transforms.
    assert _apply_text_level_clean(GOLDEN_MARKDOWN) != prepared  # title/H1 present
    assert "PAGE_JSON" not in prepared
    assert "NOTRANSLATE" not in prepared


# ---------------------------------------------------------------------------
# Tests 6-8: PDF TOC/bookmark depth checks (pure units)
# ---------------------------------------------------------------------------


def test_required_toc_depth_formula():
    assert _required_toc_depth("# a\n# b\n## c\n") == 1
    assert _required_toc_depth("# 书名\n") == 0  # title-only document
    assert _required_toc_depth("") == 0  # empty clamps to 0
    assert _required_toc_depth("### 深层\n") == 2
    assert _required_toc_depth("只是正文\n") == 0


def test_toc_depth_check_passes_and_fails():
    deep = (
        "\\contentsline {chapter}{A}{1}\n"
        "\\contentsline {section}{B}{2}\n"
        "\\contentsline {section}{\\numberline {1}C}{3}{section.1}\n"
    )
    assert _verify_toc_depth(deep, required_depth=1, heading_count=3) == {
        "toc_depth": 1, "toc_entries": 3,
    }
    shallow = "\\contentsline {chapter}{A}{1}\n"
    with pytest.raises(RuntimeError, match="TOC depth"):
        _verify_toc_depth(shallow, required_depth=1, heading_count=3)
    with pytest.raises(RuntimeError, match="missing headings"):
        _verify_toc_depth(shallow, required_depth=0, heading_count=6)
    with pytest.raises(RuntimeError, match="TOC depth"):
        _verify_toc_depth("", required_depth=0, heading_count=1)


def test_bookmark_depth_check_contiguity():
    good = (
        "\\BOOKMARK [0][-]{chapter.1}{A}{}\n"
        "\\BOOKMARK [1][-]{section.1}{B}{}\n"
        "\\BOOKMARK [1][-]{section.2}{C}{}\n"
    )
    assert _verify_bookmark_depth(good, required_depth=1) == {
        "bookmark_max_depth": 1, "bookmark_entries": 3,
    }
    jump = (
        "\\BOOKMARK [0][-]{chapter.1}{A}{}\n"
        "\\BOOKMARK [2][-]{subsection.1}{B}{}\n"
    )
    with pytest.raises(RuntimeError, match="skips levels"):
        _verify_bookmark_depth(jump, required_depth=1)
    shallow = "\\BOOKMARK [0][-]{chapter.1}{A}{}\n"
    with pytest.raises(RuntimeError, match="bookmark depth"):
        _verify_bookmark_depth(shallow, required_depth=1)


# ---------------------------------------------------------------------------
# Tests 9-10: EPUBCheck wrapper status contract
# ---------------------------------------------------------------------------


def test_epubcheck_absent_records_validation_incomplete(monkeypatch, tmp_path):
    monkeypatch.setattr(publish_module, "_epubcheck_discover", lambda settings: None)

    def must_not_run(*args, **kwargs):
        raise AssertionError("epubcheck must not run when discovery returns None")

    monkeypatch.setattr(publish_module, "_run_epubcheck", must_not_run)
    settings = _publish_settings({})
    result = _run_epub_validation(tmp_path / "book.epub", settings, tmp_path)
    assert result == {"validation": "validation_incomplete"}
    # require=true fails closed before an artifact could be published.
    strict = _publish_settings({"publish": {"epubcheck": {"require": True}}})
    with pytest.raises(RuntimeError, match="required"):
        _run_epub_validation(tmp_path / "book.epub", strict, tmp_path)
    # enabled=false records "skipped" without any discovery.
    off = _publish_settings({"publish": {"epubcheck": {"enabled": False}}})
    assert _run_epub_validation(tmp_path / "book.epub", off, tmp_path) == {
        "validation": "skipped",
    }


@require_epubcheck
def test_epubcheck_present_valid(tmp_path):
    source = tmp_path / "book.md"
    source.write_text("# 标题\n\n正文。\n", encoding="utf-8")
    epub = tmp_path / "book.epub"
    completed = subprocess.run(
        [PANDOC, str(source), "--to=epub3", "--standalone", "--output", str(epub)],
        cwd=tmp_path, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    settings = _publish_settings({})
    assert _epubcheck_discover(settings) is not None
    result = _run_epub_validation(epub, settings, tmp_path)
    assert result["validation"] == "valid"
    assert result["epubcheck"]["exit_code"] == 0
    strict = _publish_settings({"publish": {"epubcheck": {"require": True}}})
    assert _run_epub_validation(epub, strict, tmp_path)["validation"] == "valid"


# ---------------------------------------------------------------------------
# Test 11: export never reverse-modifies canonical book_translated.md
# ---------------------------------------------------------------------------

COVER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FakeClient:
    model = "p08-publish-fixture"

    def complete(self, system, user):
        payload = json.loads(user)
        translated = [
            {"id": item["id"], "translated_text": "译：" + item["text"]}
            for item in payload["segments"]
        ]
        return json.dumps(translated, ensure_ascii=False), {
            "prompt_tokens": 1, "completion_tokens": 1,
        }


def make_exportable_config(tmp_path: Path) -> dict:
    ocr_path = FIXTURES / "sample_ocr.json"
    pages = json.loads(ocr_path.read_text(encoding="utf-8"))
    first_id = block_id(0, 0, raw_blocks(pages[0])[0])
    section_id = block_id(2, 0, raw_blocks(pages[2])[0])
    structure_dir = tmp_path / "structure"
    structure_dir.mkdir(parents=True)
    structure = {
        "nodes": [
            {"id": "book-1", "kind": "book", "ignored": False},
            {"id": "chapter-1", "kind": "chapter", "level": 2, "title_source": "First Chapter", "page_json": 0, "parent_id": "book-1", "bbox": [80, 100, 700, 145], "source_block_ids": [first_id], "ignored": False, "zone": "mainmatter"},
            {"id": "section-1", "kind": "section", "level": 3, "title_source": "A real internal section", "page_json": 2, "parent_id": "chapter-1", "bbox": [80, 100, 700, 145], "source_block_ids": [section_id], "ignored": False, "zone": "mainmatter"},
        ]
    }
    (structure_dir / "book_structure.json").write_text(json.dumps(structure), encoding="utf-8")
    (structure_dir / "validation.json").write_text(
        json.dumps({"ok": True, "errors": [], "gate_failures": []}), encoding="utf-8"
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "book_id": "fixture-book",
        "book_title": "Fixture Book",
        "book_title_zh": "测试书",
        "author": "测试作者",
        "input_json": str(ocr_path),
        "structure_json": str(structure_dir / "book_structure.json"),
        "output_dir": str(tmp_path / "output"),
        "source_library_root": str(tmp_path / "read-only-library"),
        "translate_zones": ["mainmatter"],
        "quality": {"validation_json": str(structure_dir / "validation.json"), "allow_pending_review": False},
        "allow_demo_translations": True,
        "provider": {"requests_per_minute": 0},
        "chunk": {"max_chars": 1000, "max_segments": 20},
    }), encoding="utf-8")
    return load_config(config_path)


@require_pandoc
def test_export_does_not_modify_canonical_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_module, "_epubcheck_discover", lambda settings: None)
    config = make_exportable_config(tmp_path)
    clean_book(config)
    translate_book(config, FakeClient())
    render_book(config)
    markdown = Path(config["output_dir"]) / "book_translated.md"
    before = hashlib.sha256(markdown.read_bytes()).hexdigest()
    assert markdown.read_text(encoding="utf-8").startswith("# 测试书")

    project = tmp_path / "book-project"
    cover = tmp_path / "cover.png"
    cover.write_bytes(COVER_PNG)
    register_cover(
        project, cover, rights_status="user_provided",
        selected_by="P08 canonical-md test", edition_evidence="generated fixture",
    )
    report = export_book(config, project, ("epub",))

    after = hashlib.sha256(markdown.read_bytes()).hexdigest()
    assert after == before
    assert report["source_markdown_sha256"] == before
    assert report["outputs"]["epub"]["verification"]["validation"] == (
        "validation_incomplete"
    )
    assert Path(report["outputs"]["epub"]["path"]).is_file()


# ---------------------------------------------------------------------------
# Test 12: publish{} config defaults and load_config tolerance
# ---------------------------------------------------------------------------


def test_publish_config_defaults_and_absence_tolerated(tmp_path):
    assert _publish_settings({}) == {
        "lua_filters": False,
        "toc_depth": 4,
        "epub_chapter_level": 2,
        "pdf": {"tocdepth": 3},
        "epubcheck": {"enabled": True, "require": False, "path": ""},
    }
    config = {
        "publish": {
            "lua_filters": True,
            "toc_depth": 6,
            "epub_chapter_level": 3,
            "pdf": {"tocdepth": 1},
            "epubcheck": {"enabled": False, "require": True, "path": "/tmp/epubcheck.jar"},
        }
    }
    settings = _publish_settings(config)
    assert settings["lua_filters"] is True
    assert settings["toc_depth"] == 6
    assert settings["epub_chapter_level"] == 3
    assert settings["pdf"]["tocdepth"] == 1
    assert settings["epubcheck"] == {
        "enabled": False, "require": True, "path": "/tmp/epubcheck.jar",
    }
    # Partial publish{} sections keep the untouched defaults.
    partial = _publish_settings({"publish": {"pdf": {"tocdepth": 5}}})
    assert partial["toc_depth"] == 4 and partial["epubcheck"]["require"] is False

    # load_config accepts (and carries through) the full additive publish{}
    # block plus unknown future keys.
    ocr_path = FIXTURES / "sample_ocr.json"
    structure = tmp_path / "structure.json"
    structure.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "book_id": "cfg-book",
        "book_title": "Config Book",
        "book_title_zh": "配置书",
        "input_json": str(ocr_path),
        "structure_json": str(structure),
        "output_dir": str(tmp_path / "output"),
        "publish": {
            "lua_filters": False,
            "toc_depth": 4,
            "epub_chapter_level": 2,
            "pdf": {"tocdepth": 3},
            "epubcheck": {"enabled": True, "require": False, "path": ""},
            "future_setting": 1,
        },
    }), encoding="utf-8")
    loaded = load_config(config_path)
    assert loaded["publish"]["pdf"]["tocdepth"] == 3
    assert loaded["publish"]["future_setting"] == 1
