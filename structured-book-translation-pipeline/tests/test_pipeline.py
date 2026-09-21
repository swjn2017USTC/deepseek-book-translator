import json
import base64
from pathlib import Path

import pytest

import book_pipeline.llm_client as llm_client_module
from book_pipeline.clean import block_id, clean_book, raw_blocks
from book_pipeline.config import load_config
from book_pipeline.glossary import compile_glossary_decisions, generate_glossary
from book_pipeline.io_utils import read_jsonl, write_jsonl
from book_pipeline.migrate import (
    PLAIN_STRUCTURAL_HEADING,
    align_existing_headings,
    audit_existing_translation,
    parse_existing_markdown,
)
from book_pipeline.migration_preview import create_migration_preview
from book_pipeline.render import render_book
from book_pipeline.publish import (
    build_book_tex,
    clean_for_latex,
    export_book,
    load_registered_cover,
    postprocess_body_tex,
    register_cover,
)
from book_pipeline.translate import OpenAICompatibleClient, translate_book, translation_status
from book_pipeline.workflow import (
    initialize_project,
    compile_project_glossary,
    preflight_project,
    prepare_project,
    project_status,
    translate_project,
)


FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient:
    model = "fake-model"

    def complete(self, system, user):
        payload = json.loads(user)
        translated = [{"id": item["id"], "translated_text": "译：" + item["text"]} for item in payload["segments"]]
        return json.dumps(translated, ensure_ascii=False), {"prompt_tokens": 10, "completion_tokens": 20}


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}],
            "usage": {},
        }).encode("utf-8")


def make_config(tmp_path):
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
    (structure_dir / "validation.json").write_text(json.dumps({"ok": True, "errors": [], "gate_failures": []}), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "book_id": "fixture-book",
        "book_title": "Fixture Book",
        "book_title_zh": "测试书",
        "input_json": str(ocr_path),
        "structure_json": str(structure_dir / "book_structure.json"),
        "output_dir": str(tmp_path / "output"),
        "source_library_root": str(tmp_path / "read-only-library"),
        "translate_zones": ["mainmatter"],
        "quality": {"validation_json": str(structure_dir / "validation.json"), "allow_pending_review": False},
        "allow_demo_translations": True,
        "provider": {"requests_per_minute": 0},
        "chunk": {"max_chars": 1000, "max_segments": 20}
    }), encoding="utf-8")
    return load_config(config_path)


def test_clean_translate_resume_and_render(tmp_path):
    config = make_config(tmp_path)
    report = clean_book(config)
    assert report["metrics"]["unresolved_title_blocks"] == 0
    assert report["metrics"]["footnotes_total"] == 1
    segments = list(read_jsonl(Path(config["output_dir"]) / "cleaned_segments.jsonl"))
    source = "\n".join(item["source_text"] for item in segments)
    assert "Publisher footer" not in source
    assert "A cross-page paragraph continues here[^fn-p0001-1]." in source
    assert any(item["kind"] == "image" for item in segments)
    assert any(item["kind"] == "table" for item in segments)

    first = translate_book(config, FakeClient())
    assert first["ready_to_render"] is True
    line_count = len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl")))
    second = translate_book(config, FakeClient())
    assert second["remaining_segments"] == 0
    assert len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))) == line_count

    rendered = render_book(config)
    text = Path(rendered["output"]).read_text(encoding="utf-8")
    assert "## 译：First Chapter" in text
    assert "### 译：A real internal section" in text
    assert "[^fn-p0001-1]: 译：The source footnote." in text


def test_empty_footnote_body_is_not_required_for_coverage(tmp_path):
    config = make_config(tmp_path)
    pages = json.loads((FIXTURES / "sample_ocr.json").read_text(encoding="utf-8"))
    pages[1]["prunedResult"]["parsing_res_list"][3]["block_content"] = "1."
    input_path = tmp_path / "empty-footnote.json"
    input_path.write_text(json.dumps(pages), encoding="utf-8")
    config["input_json"] = str(input_path)
    clean_book(config)
    footnotes = [
        row for row in read_jsonl(Path(config["output_dir"]) / "cleaned_segments.jsonl")
        if row["kind"] == "footnote"
    ]
    assert len(footnotes) == 1
    assert footnotes[0]["source_text"] == ""
    assert footnotes[0]["translatable"] is False


def test_cover_registration_is_provenance_bound_and_export_refuses_incomplete_book(tmp_path):
    project = tmp_path / "book-project"
    image = tmp_path / "cover.png"
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    ))
    manifest = register_cover(
        project, image,
        rights_status="official_product_cover",
        source_page_url="https://publisher.example/books/fixture",
        source_image_url="https://publisher.example/covers/fixture.png",
        selected_by="OMP",
        edition_evidence="Exact title and ISBN match",
    )
    assert len(manifest["sha256"]) == 64
    assert load_registered_cover(project)["selected_by"] == "OMP"

    registered = Path(manifest["image_path"])
    registered.write_bytes(b"not the approved image")
    with pytest.raises((ValueError, RuntimeError), match="extension|changed"):
        load_registered_cover(project)

    config = make_config(tmp_path / "incomplete")
    clean_book(config)
    with pytest.raises(RuntimeError, match="100% completed"):
        export_book(config, project, ("pdf",))


def test_cambridge_style_tex_sources_have_promoted_headings_and_clickable_toc():
    markdown = """# 测试书

测试作者

<!-- PAGE_JSON: 0001 -->

## 第一章

正文¹，且 α ≤ β。

### 第一节

内容。
"""
    prepared = clean_for_latex(markdown, "测试书", "测试作者")
    assert "# 测试书" not in prepared
    assert "测试作者" not in prepared
    assert "<!-- PAGE_JSON" not in prepared
    assert "# 第一章" in prepared
    assert "## 第一节" in prepared
    assert "$^{1}$" in prepared
    assert "$\\alpha$" in prepared and "$\\leq$" in prepared

    body = postprocess_body_tex("\\chapter{第一\n章}\n\\section{第一节}\n\\section{第二节}\n")
    assert "\\chapter{第一 章}" in body
    assert body.count("\\clearpage") == 0  # sections flow; only chapters break pages

    book_tex = build_book_tex("书名_一", "作者 & 编者")
    assert "\\tableofcontents" in book_tex
    assert "fontset=none" in book_tex
    assert "\\setCJKmainfont{Songti SC}" in book_tex
    assert "\\setCJKsansfont{Heiti SC}" in book_tex
    assert "\\usepackage{longtable,booktabs,array,calc}" in book_tex
    assert "\\newcounter{none}" in book_tex
    assert "\\usepackage[unicode=true,hidelinks]{hyperref}" in book_tex
    assert "\\hypersetup{pdftitle=" in book_tex
    assert "书名\\_一" in book_tex
    assert "作者 \\& 编者" in book_tex


def test_clean_manifest_blocks_translation_after_structure_changes(tmp_path):
    config = make_config(tmp_path)
    clean_book(config)
    structure_path = Path(config["structure_json"])
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    structure["nodes"][1]["title_source"] = "Changed after clean"
    structure_path.write_text(json.dumps(structure), encoding="utf-8")

    with pytest.raises(RuntimeError, match="chapter structure"):
        translate_book(config, FakeClient())
    with pytest.raises(RuntimeError, match="chapter structure"):
        render_book(config, allow_partial=True)


def test_new_book_workflow_initializes_and_prepares_without_translation(tmp_path, monkeypatch):
    project_dir = tmp_path / "new-book"
    initialized = initialize_project(
        input_json=FIXTURES / "sample_ocr.json",
        book_id="new-fixture-book",
        book_title="New Fixture Book",
        book_title_zh="新测试书",
        project_dir=project_dir,
    )

    translation_config = json.loads(
        Path(initialized["translation_config"]).read_text(encoding="utf-8")
    )
    assert translation_config["provider"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert translation_config["provider"]["api_url"] == "https://api.deepseek.com/chat/completions"
    assert translation_config["provider"]["model"] == "deepseek-v4-flash"
    assert translation_config["provider"]["auth_header"] == "Authorization"
    assert translation_config["provider"]["auth_scheme"] == "Bearer"
    assert not any(key in translation_config["provider"] for key in ("api_key", "token", "secret"))
    translation = translation_config["translation"]
    assert translation["context_mode"] == "contextual_v2"
    assert (translation["previous_segments"], translation["next_segments"]) == (1, 1)
    assert translation["previous_translation_max_chars"] == 300
    assert translation["skip_failed_segments"] is True
    assert translation["max_workers"] == 1
    assert translation_config["publish"]["epubcheck"] == {
        "enabled": True, "require": True, "path": ""
    }
    assert translation_config["publish"]["pdf"]["tocdepth"] == 3
    assert Path(initialized["glossary"]).read_text(encoding="utf-8").strip() == "[]"
    assert (project_dir / "RUNBOOK.md").is_file()
    assert (project_dir / "glossary_decision.schema.json").is_file()

    prepared = prepare_project(project_dir)

    assert prepared["status"] == "needs_glossary_review"
    candidates = list(read_jsonl(project_dir / "glossary_candidates.jsonl"))
    assert candidates
    decisions = project_dir / "glossary_decisions.jsonl"
    write_jsonl(decisions, [
        {"candidate_id": row["candidate_id"], "decision": "reject", "reason": "fixture non-term"}
        for row in candidates
    ])
    compiled = compile_project_glossary(project_dir, decisions)
    assert compiled["status"] == "approved"
    prepared = prepare_project(project_dir)
    status = project_status(project_dir)

    assert prepared["status"] == "ready_to_translate"
    assert prepared["translation_started"] is False
    assert prepared["translation"]["completed_segments"] == 0
    assert status["status"] == "ready_to_translate"
    assert status["cleaning_fresh"] is True
    import book_pipeline.contextual as contextual

    called = {}

    def fake_translate_v2(config, *, target_completed=None, **_kwargs):
        called.update(mode=config["translation"]["context_mode"], target=target_completed)
        return {"status": "contextual-dispatched"}

    monkeypatch.setattr(contextual, "translate_v2", fake_translate_v2)
    assert translate_project(project_dir, target_completed=10, all_segments=False) == {
        "status": "contextual-dispatched"
    }
    assert called == {"mode": "contextual_v2", "target": 10}
    manifest = json.loads(
        (Path(initialized["work_dir"]) / "pipeline_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 2
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["structure_sha256"]) == 64
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy-local-preflight-only")
    preflight = preflight_project(project_dir)
    assert preflight["status"] == "preflight_passed"
    assert preflight["external_requests_made"] == 0
    assert preflight["estimated_remaining_batches"] > 0
    with pytest.raises(ValueError, match="target-completed"):
        translate_project(project_dir, target_completed=None, all_segments=False)


def test_glossary_generation_uses_boundaries_and_invalidates_translation_cache(tmp_path):
    config = make_config(tmp_path)
    clean_book(config)
    master = tmp_path / "master.json"
    master.write_text(json.dumps([
        {"term": "Foot", "translation": "脚", "category": "concept"},
        {"term": "First Chapter", "translation": "第一章", "category": "term"},
        {"term": "Publisher", "translation": "出版者", "category": "concept"},
    ]), encoding="utf-8")
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]\n", encoding="utf-8")
    config["glossary"] = str(glossary)
    config["glossary_settings"] = {
        "mode": "master_subset_plus_review",
        "master_json": str(master),
        "manifest": str(tmp_path / "glossary_manifest.json"),
        "candidates": str(tmp_path / "glossary_candidates.jsonl"),
        "decision_template": str(tmp_path / "glossary_decisions.template.jsonl"),
        "additions": str(tmp_path / "glossary_additions.json"),
        "report": str(tmp_path / "GLOSSARY_REPORT.md"),
        "require_approval": True,
        "bind_translation_cache": True,
    }
    generated = generate_glossary(config)
    terms = {row["term"] for row in json.loads(glossary.read_text(encoding="utf-8"))}
    assert "First Chapter" in terms
    assert "Foot" not in terms
    assert "Publisher" not in terms

    candidates = list(read_jsonl(tmp_path / "glossary_candidates.jsonl"))
    assert all(row["evidence_segment_ids"] for row in candidates)
    assert all(row["occurrences"] >= len(row["evidence_segment_ids"]) for row in candidates)
    decisions = tmp_path / "glossary_decisions.jsonl"
    write_jsonl(decisions, [
        {"candidate_id": row["candidate_id"], "decision": "reject", "reason": "fixture non-term"}
        for row in candidates
    ])
    approved = compile_glossary_decisions(config, decisions)
    assert approved["approved"] is True
    translated = translate_book(config, FakeClient())
    assert translated["completed_segments"] == translated["required_segments"]

    glossary.write_text(glossary.read_text(encoding="utf-8").replace("第一章", "首章"), encoding="utf-8")
    stale = translation_status(config)
    assert stale["glossary_ready"] is False
    assert stale["completed_segments"] == 0


def test_structure_gate_and_secret_safety(tmp_path):
    config = make_config(tmp_path)
    validation = Path(config["quality"]["validation_json"])
    validation.write_text(json.dumps({"ok": False}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="quality gate failed"):
        clean_book(config)

    raw = json.loads(Path(config["_config_path"]).read_text(encoding="utf-8"))
    raw["provider"]["api_key"] = "forbidden-secret"
    Path(config["_config_path"]).write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Secret-like value"):
        load_config(Path(config["_config_path"]))


def test_status_before_translation(tmp_path):
    config = make_config(tmp_path)
    clean_book(config)
    status = translation_status(config)
    assert status["completed_segments"] == 0
    assert status["ready_to_render"] is False
    with pytest.raises(RuntimeError, match="Translation incomplete"):
        render_book(config)


def test_target_completed_is_a_resume_safe_total_cap(tmp_path):
    config = make_config(tmp_path)
    clean_book(config)

    first = translate_book(config, FakeClient(), target_completed=2)
    assert first["completed_segments"] == 2
    line_count = len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl")))

    second = translate_book(config, FakeClient(), target_completed=2)
    assert second["completed_segments"] == 2
    assert len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))) == line_count

    with pytest.raises(ValueError, match="either limit or target_completed"):
        translate_book(config, FakeClient(), limit=1, target_completed=3)


def test_configured_auth_header_preserves_exact_case(monkeypatch):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "dummy-not-a-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = request.header_items()
        return FakeHTTPResponse()

    monkeypatch.setattr(llm_client_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient({
        "api_key_env": "TEST_DEEPSEEK_KEY",
        "api_url": "https://example.invalid/v1/chat/completions",
        "model": "fixture-model",
        "auth_header": "x-custom-auth",
        "auth_scheme": "",
    })
    client.complete("system", "user")

    header_names = [name for name, _ in captured["headers"]]
    header_values = dict(captured["headers"])
    assert "x-custom-auth" in header_names
    assert "X-custom-auth" not in header_names
    assert "Authorization" not in header_names
    assert header_values["x-custom-auth"] == "dummy-not-a-secret"


def test_default_authorization_scheme_is_bearer(monkeypatch):
    monkeypatch.setenv("TEST_DEFAULT_KEY", "dummy-not-a-secret")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        return FakeHTTPResponse()

    monkeypatch.setattr(llm_client_module, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient({
        "api_key_env": "TEST_DEFAULT_KEY",
        "api_url": "https://example.invalid/v1/chat/completions",
        "model": "fixture-model",
    })
    client.complete("system", "user")

    assert captured["headers"]["Authorization"] == "Bearer dummy-not-a-secret"


@pytest.mark.parametrize("bad_key", [" key", "key\n", "'key'", '"key"'])
def test_provider_rejects_malformed_environment_key(monkeypatch, bad_key):
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", bad_key)
    with pytest.raises(RuntimeError, match="API key environment variable"):
        OpenAICompatibleClient({
            "api_key_env": "TEST_DEEPSEEK_KEY",
            "api_url": "https://example.invalid/v1/chat/completions",
            "model": "fixture-model",
            "auth_header": "x-custom-auth",
            "auth_scheme": "",
        })


def test_rejects_output_in_source_library_and_pending_reviews(tmp_path):
    config = make_config(tmp_path)
    config_path = Path(config["_config_path"])
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["output_dir"] = str(Path(raw["source_library_root"]) / "a-book")
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be inside"):
        load_config(config_path)

    config = make_config(tmp_path / "pending")
    review_path = tmp_path / "pending" / "structure" / "review_packets.jsonl"
    review_path.write_text('{"candidate_id":"one"}\n', encoding="utf-8")
    config["quality"]["review_packets"] = str(review_path)
    with pytest.raises(RuntimeError, match="unresolved review packets"):
        clean_book(config)


def test_schema_v2_cross_page_heading_provenance_is_consumed(tmp_path):
    config = make_config(tmp_path)
    pages = json.loads((FIXTURES / "sample_ocr.json").read_text(encoding="utf-8"))
    left = {
        "block_id": 90,
        "block_order": 90,
        "block_label": "paragraph_title",
        "block_content": "A SPLIT:",
        "block_bbox": [80, 1120, 700, 1170],
    }
    right = {
        "block_id": 91,
        "block_order": 0,
        "block_label": "paragraph_title",
        "block_content": "HEADING",
        "block_bbox": [80, 30, 700, 75],
    }
    pages[0]["prunedResult"]["parsing_res_list"].append(left)
    pages[1]["prunedResult"]["parsing_res_list"].append(right)
    ocr_path = tmp_path / "cross_page_ocr.json"
    ocr_path.write_text(json.dumps(pages), encoding="utf-8")

    left_id = block_id(0, 4, left)
    right_id = block_id(1, 4, right)
    structure_path = Path(config["structure_json"])
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    structure["schema_version"] = 2
    structure["nodes"].append({
        "id": "section-cross-page",
        "kind": "section",
        "level": 3,
        "title_source": "A SPLIT: HEADING",
        "page_json": 0,
        "parent_id": "chapter-1",
        "bbox": left["block_bbox"],
        "source_block_ids": [left_id, right_id],
        "alignment_status": None,
        "ignored": False,
        "zone": "mainmatter",
    })
    structure_path.write_text(json.dumps(structure), encoding="utf-8")

    config_path = Path(config["_config_path"])
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["input_json"] = str(ocr_path)
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    config = load_config(config_path)

    report = clean_book(config)
    assert report["metrics"]["unresolved_title_blocks"] == 0
    segments = list(read_jsonl(Path(config["output_dir"]) / "cleaned_segments.jsonl"))
    merged = next(item for item in segments if item["id"].startswith("seg-") and item["source_text"] == "A SPLIT: HEADING")
    assert merged["source_block_ids"] == [left_id, right_id]


def test_upstream_candidate_suppression_does_not_delete_source_text(tmp_path):
    config = make_config(tmp_path)
    pages = json.loads((FIXTURES / "sample_ocr.json").read_text(encoding="utf-8"))
    byline = {
        "block_id": 92,
        "block_order": 92,
        "block_label": "text",
        "block_content": "AUTHOR NAME",
        "block_bbox": [300, 155, 500, 180],
    }
    pages[0]["prunedResult"]["parsing_res_list"].append(byline)
    ocr_path = tmp_path / "suppressed_candidate_ocr.json"
    ocr_path.write_text(json.dumps(pages), encoding="utf-8")

    structure_path = Path(config["structure_json"])
    suppression_path = structure_path.with_name("candidate_suppressions.jsonl")
    suppression_path.write_text(
        json.dumps({
            "block_id": block_id(0, 4, byline),
            "source_text": "AUTHOR NAME",
            "suppressed": True,
            "suppression_reason": "chapter_title_page_byline",
        }) + "\n",
        encoding="utf-8",
    )
    config_path = Path(config["_config_path"])
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["input_json"] = str(ocr_path)
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")

    report = clean_book(load_config(config_path))
    assert report["structure_gate"]["pending_reviews"] == 0
    segments = list(read_jsonl(Path(config["output_dir"]) / "cleaned_segments.jsonl"))
    assert any(item["source_text"] == "AUTHOR NAME" for item in segments)


def test_existing_translation_audit_is_read_only_and_proposes_levels(tmp_path):
    config = make_config(tmp_path)
    existing = tmp_path / "read-only-library" / "fixture-book" / "book_translated.md"
    existing.parent.mkdir(parents=True)
    existing.write_text(
        "# 旧书名\n\n"
        "<!-- PAGE: 0000 -->\n"
        "## 第1章 旧章名\n"
        "正文绝不能被改写。\n\n"
        "<!-- PAGE_JSON: 0002 -->\n"
        "#### 一个内部小节\n"
        "更多正文。\n",
        encoding="utf-8",
    )
    before = existing.read_bytes()
    config["existing_translation"] = str(existing)

    report = audit_existing_translation(config)

    assert existing.read_bytes() == before
    assert report["audit_only"] is True
    assert report["source_library_unchanged"] is True
    assert report["metrics"]["matched"] == 2
    assert report["metrics"]["proposed_level_changes"] == 1
    assert report["metrics"]["unmatched_existing_headings"] == 1
    assert not (Path(config["output_dir"]) / "book_translated.md").exists()
    records = json.loads(
        (Path(config["output_dir"]) / "migration_alignment.json").read_text(encoding="utf-8")
    )["records"]
    assert len({record["node_id"] for record in records}) == len(records)
    assert len({record["existing_heading_id"] for record in records}) == len(records)
    assert next(record for record in records if record["node_id"] == "section-1")["target_level"] == 3


def test_migration_alignment_is_global_ordered_and_one_to_one():
    nodes = [
        {"id": "n1", "page_json": 4, "level": 2, "title_source": "Chapter 1", "kind": "chapter", "identity_kind": "chapter", "identity_ordinal": "1"},
        {"id": "n2", "page_json": 4, "level": 2, "title_source": "Chapter 2", "kind": "chapter", "identity_kind": "chapter", "identity_ordinal": "2"},
    ]
    headings = [
        {"heading_id": "h1", "page_json": 4, "level": 2, "title": "第1章", "identity_kind": "chapter", "identity_ordinal": "1"},
        {"heading_id": "h2", "page_json": 4, "level": 2, "title": "第2章", "identity_kind": "chapter", "identity_ordinal": "2"},
    ]

    aligned = align_existing_headings(nodes, headings)

    assert aligned["pairs"] == [(0, 0), (1, 1)]
    assert aligned["unmatched_nodes"] == []
    assert aligned["unmatched_headings"] == []

    chinese_numeral_heading = dict(headings[1], title="二 第二节", identity_ordinal="2")
    translated_ordinal = align_existing_headings([nodes[1]], [chinese_numeral_heading])
    assert translated_ordinal["pairs"] == [(0, 0)]


def test_nearby_unnumbered_h2_chapter_is_reviewable_but_not_high_confidence():
    nodes = [{
        "id": "chapter-20", "page_json": 437, "level": 2,
        "title_source": "Jerusalem: capital city created in stone and in imagination",
        "kind": "chapter", "identity_kind": "chapter", "identity_ordinal": "20",
    }]
    headings = [{
        "heading_id": "h-jerusalem", "page_json": 434, "level": 2,
        "title": "耶路撒冷：以石头和想象力创造的首都",
        "identity_kind": None, "identity_ordinal": None,
    }]

    aligned = align_existing_headings(nodes, headings)

    assert aligned["pairs"] == [(0, 0)]


def test_further_reading_cross_language_keyword_supports_ordered_alignment():
    nodes = [{
        "id": "reading", "page_json": 435, "level": 5,
        "title_source": "FURTHER READINGS", "kind": "section",
        "identity_kind": None, "identity_ordinal": None,
    }]
    headings = [{
        "heading_id": "h-reading", "page_json": 434, "level": 5,
        "title": "延伸阅读", "identity_kind": None, "identity_ordinal": None,
    }]

    aligned = align_existing_headings(nodes, headings)

    assert aligned["pairs"] == [(0, 0)]


def test_unique_same_page_h2_chapter_is_high_despite_extra_internal_headings(tmp_path):
    config = make_config(tmp_path)
    structure_path = Path(config["structure_json"])
    structure_path.write_text(json.dumps({
        "nodes": [
            {"id": "book-1", "kind": "book", "ignored": False},
            {
                "id": "chapter-1", "kind": "chapter", "level": 2,
                "title_source": "An unnumbered translated chapter", "page_json": 4,
                "parent_id": "book-1", "bbox": [80, 100, 700, 145],
                "source_block_ids": ["chapter-source"], "ignored": False,
                "zone": "mainmatter",
            },
        ]
    }), encoding="utf-8")
    existing = tmp_path / "old.md"
    existing.write_text(
        "<!-- PAGE: 0004 -->\n## 一个未编号的译文章标题\n### 本页多出的内部标题\n",
        encoding="utf-8",
    )
    config["existing_translation"] = str(existing)

    audit_existing_translation(config)
    record = json.loads(
        (Path(config["output_dir"]) / "migration_alignment.json").read_text(encoding="utf-8")
    )["records"][0]

    assert record["confidence"] == "high"
    assert "complete_same_page_macro_sequence" in record["evidence"]


def test_migration_preview_changes_only_heading_prefix_and_is_reversible(tmp_path):
    config = make_config(tmp_path)
    existing = tmp_path / "read-only-library" / "fixture-book" / "book_translated.md"
    existing.parent.mkdir(parents=True)
    original = (
        "# 旧书名\r\n\r\n"
        "<!-- PAGE: 0001 -->\r\n"
        "#### 第1章 旧章名\r\n"
        "正文绝不能被改写。\r\n\r\n"
        "<!-- PAGE_JSON: 0002 -->\r\n"
        "#### 一个内部小节\r\n"
        "更多正文。\r\n"
    ).encode("utf-8")
    existing.write_bytes(original)
    config["existing_translation"] = str(existing)
    audit_existing_translation(config)

    review_path = Path(config["output_dir"]) / "migration_review_packets.jsonl"
    reviews = list(read_jsonl(review_path))
    decisions = []
    for review in reviews:
        if review["reason"] == "migration_alignment_not_high_confidence":
            decisions.append({
                "candidate_id": review["candidate_id"],
                "decision": "accept_mapping",
                "reason": "人工核对编号与相邻页一致",
            })
        elif review["reason"] == "existing_heading_unmatched":
            decisions.append({
                "candidate_id": review["candidate_id"],
                "decision": "keep_unchanged",
                "reason": "保留书名",
            })
        else:
            decisions.append({
                "candidate_id": review["candidate_id"],
                "decision": "leave_unmapped",
                "reason": "本次不映射",
            })
    decisions_path = tmp_path / "migration_decisions.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )

    report = create_migration_preview(config, decisions_path)

    assert existing.read_bytes() == original
    assert report["ready_for_adoption"] is True
    assert report["non_heading_sha256_equal"] is True
    assert report["heading_text_sequence_equal"] is True
    preview = Path(report["preview"]).read_bytes()
    assert b"\r\n" in preview
    assert "## 第1章 旧章名\r\n".encode("utf-8") in preview
    assert "### 一个内部小节\r\n".encode("utf-8") in preview
    patch = json.loads(
        (Path(config["output_dir"]) / "migration_patch.json").read_text(encoding="utf-8")
    )
    reversed_lines = preview.decode("utf-8").splitlines(keepends=True)
    for item in reversed(patch["patches"]):
        assert reversed_lines[item["line"] - 1] == item["reverse"]["before"]
        reversed_lines[item["line"] - 1] = item["reverse"]["after"]
    assert "".join(reversed_lines).encode("utf-8") == original

    structure_path = Path(config["structure_json"])
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    structure["audit_stale_test"] = True
    structure_path.write_text(json.dumps(structure), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Structure changed after migration audit"):
        create_migration_preview(config, decisions_path)


def test_migration_decisions_fail_closed_on_unknown_fields(tmp_path):
    config = make_config(tmp_path)
    existing = tmp_path / "old.md"
    existing.write_text("<!-- PAGE: 0000 -->\n## 第1章 旧章名\n", encoding="utf-8")
    config["existing_translation"] = str(existing)
    audit_existing_translation(config)
    review = next(read_jsonl(Path(config["output_dir"]) / "migration_review_packets.jsonl"))
    decisions = tmp_path / "bad.jsonl"
    decisions.write_text(
        json.dumps({
            "candidate_id": review["candidate_id"],
            "decision": "needs_human",
            "reason": "证据不足",
            "title": "禁止提供新标题",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown migration decision fields"):
        create_migration_preview(config, decisions)


def test_plain_numbered_part_line_can_be_promoted_without_text_change(tmp_path):
    config = make_config(tmp_path)
    structure_path = Path(config["structure_json"])
    structure_path.write_text(json.dumps({
        "nodes": [
            {"id": "book-1", "kind": "book", "ignored": False},
            {
                "id": "part-1", "kind": "part", "level": 2,
                "title_source": "Part One: Old Regime", "page_json": 0,
                "parent_id": "book-1", "bbox": [80, 100, 700, 145],
                "source_block_ids": ["part-source"], "ignored": False,
                "zone": "mainmatter", "ordinal": "One",
            },
        ]
    }), encoding="utf-8")
    existing = tmp_path / "old-parts.md"
    existing.write_text(
        "<!-- PAGE: 0000 -->\n第一部分：旧制度\n正文保持不变。\n",
        encoding="utf-8",
    )
    original = existing.read_bytes()
    config["existing_translation"] = str(existing)

    audit = audit_existing_translation(config)
    preview = create_migration_preview(config)

    assert audit["metrics"]["high_confidence"] == 1
    assert preview["ready_for_adoption"] is True
    assert preview["changed_heading_levels"] == 1
    assert Path(preview["preview"]).read_text(encoding="utf-8").startswith(
        "<!-- PAGE: 0000 -->\n## 第一部分：旧制度\n"
    )
    assert existing.read_bytes() == original
    assert preview["non_heading_sha256_equal"] is True
    assert preview["heading_text_sequence_equal"] is True


@pytest.mark.parametrize("part_line", ["第一部分 旧制度", "第二部分"])
def test_plain_numbered_part_line_without_colon_is_structural(tmp_path, part_line):
    existing = tmp_path / "old-parts.md"
    existing.write_text(
        f"<!-- PAGE: 0046 -->\n{part_line}\n第一部分研究的是城市史。\n",
        encoding="utf-8",
    )

    parsed = parse_existing_markdown(existing)

    assert [heading["title"] for heading in parsed["headings"]] == [part_line]
    assert parsed["headings"][0]["identity_kind"] == "part"
    assert "第一部分研究的是城市史。\n" in "".join(
        line
        for line in existing.read_text(encoding="utf-8").splitlines(keepends=True)
        if not PLAIN_STRUCTURAL_HEADING.match(line.strip())
    )


def test_human_level_override_and_false_heading_demotion_are_reversible(tmp_path):
    config = make_config(tmp_path)
    structure_path = Path(config["structure_json"])
    structure_path.write_text(json.dumps({
        "nodes": [
            {"id": "book-1", "kind": "book", "ignored": False},
            {
                "id": "section-roman", "kind": "section", "level": 2,
                "title_source": "iii The Heir", "page_json": 1,
                "parent_id": "book-1", "bbox": [80, 100, 700, 145],
                "source_block_ids": ["section-source"], "ignored": False,
                "zone": "mainmatter",
            },
        ]
    }), encoding="utf-8")
    existing = tmp_path / "old-reviewed.md"
    original = (
        "<!-- PAGE: 0000 -->\n"
        "## 三 继承人\n"
        "## 这不是标题\n"
        "正文保持不变。\n"
    ).encode("utf-8")
    existing.write_bytes(original)
    config["existing_translation"] = str(existing)
    audit_existing_translation(config)
    reviews = list(read_jsonl(Path(config["output_dir"]) / "migration_review_packets.jsonl"))
    alignment = next(row for row in reviews if row["reason"] == "migration_alignment_not_high_confidence")
    unmatched = next(row for row in reviews if row["reason"] == "existing_heading_unmatched")
    decisions = tmp_path / "reviewed.jsonl"
    decisions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [
        {
            "candidate_id": alignment["candidate_id"],
            "decision": "accept_mapping",
            "target_level": 3,
            "reason": "人工确认对应关系，并确认罗马数字小节使用 H3",
        },
        {
            "candidate_id": unmatched["candidate_id"],
            "decision": "demote_to_body",
            "reason": "人工确认该行不是标题，只移除 Markdown 标记",
        },
    ]), encoding="utf-8")

    report = create_migration_preview(config, decisions)
    preview = Path(report["preview"]).read_text(encoding="utf-8")

    assert "### 三 继承人\n" in preview
    assert "\n这不是标题\n" in preview
    assert "## 这不是标题" not in preview
    assert report["demoted_false_headings"] == 1
    assert report["protected_body_sha256_equal"] is True
    assert report["heading_text_sequence_equal"] is True
    assert report["ready_for_adoption"] is True
    assert existing.read_bytes() == original
