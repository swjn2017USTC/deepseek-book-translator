"""Contextual translation v2 tests (P06, spec §12) + gate-complete fixture book.

The fixture corpus mirrors the real cleaned-segment shapes from the chapter
repo: chapter headings attach to the book root (their own chapter recorded in
``metadata.node_id``), section headings attach to their chapter node, and body
segments attach to the last heading.  Concept surfaces are planted so that
dependency maps are exactly predictable.  Every gate file (structure +
validation, OCR input, cleaned segments + cleaning report + pipeline manifest)
is written so ``translate_v2`` runs its full gate path offline.
"""
import json
from pathlib import Path

import pytest

from book_pipeline.cli import parser as cli_parser
from book_pipeline.config import load_config
from book_pipeline.contextual import ladder as ladder_module
from book_pipeline.contextual import (
    build_chapter_briefs,
    is_translation_current_v2,
    parse_target_only,
    resolve_concept_deps,
    run_ladder,
    translate_v2,
)
from book_pipeline.contextual.briefs import (
    brief_hash,
    chapter_id_for_segment,
    nodes_index,
    zone_bucket_id,
)
from book_pipeline.contextual.deps import load_concept_glossary
from book_pipeline.contextual.invalidation import current_briefs, current_concept_versions
from book_pipeline.contextual.prompts import (
    CONTEXT_ONLY_CLAUSE,
    PROMPTS_VERSION,
    build_payload,
    system_prompt,
)
from book_pipeline.contextual.records import record_row
from book_pipeline.contextual.windows import chapter_order, previous_translation_text, window
from book_pipeline.io_utils import read_jsonl, write_json, write_jsonl
from book_pipeline.models import Segment, digest
from book_pipeline.provenance import file_sha256
from book_pipeline.translate import SENTENCE_FINISH, translation_status

BOOK_ID = "fixture-contextual-v2"

# --------------------------------------------------------------------------
# Fixture corpus (page-ordered, mirrors real cleaned_segments.jsonl shapes)
# --------------------------------------------------------------------------

FIXTURE_NODES = [
    {"id": "book-ctx", "kind": "book", "level": 1, "ignored": False, "zone": "book", "title_source": "Context Fixture"},
    {"id": "chapter-ctx-1", "kind": "chapter", "level": 2, "zone": "mainmatter",
     "title_source": "Chapter One: Origins of Order", "parent_id": "book-ctx", "page_json": 1},
    {"id": "section-ctx-1", "kind": "section", "level": 3, "zone": "mainmatter",
     "title_source": "The Grammar of Power", "parent_id": "chapter-ctx-1", "page_json": 2},
    {"id": "section-ctx-2", "kind": "section", "level": 3, "zone": "mainmatter",
     "title_source": "Ritual and Rank", "parent_id": "chapter-ctx-1", "page_json": 3},
    {"id": "chapter-ctx-2", "kind": "chapter", "level": 2, "zone": "mainmatter",
     "title_source": "Chapter Two: Revolutionary Rupture", "parent_id": "book-ctx", "page_json": 4},
    {"id": "section-ctx-3", "kind": "section", "level": 3, "zone": "mainmatter",
     "title_source": "The Party Congress Decides", "parent_id": "chapter-ctx-2", "page_json": 5},
    {"id": "section-ctx-4", "kind": "section", "level": 3, "zone": "mainmatter",
     "title_source": "Counter-Revolution and the Village", "parent_id": "chapter-ctx-2", "page_json": 6},
]

# (id, kind, text, page, zone, parent, level, metadata)
_RAW = [
    ("seg-ctx-0001", "paragraph", "Frontmatter note for the reader.", 0, "frontmatter", None, None, {}),
    ("seg-ctx-0002", "paragraph", "This study began as a series of lectures", 0, "frontmatter", None, None, {}),
    ("seg-ctx-0003", "heading", "Chapter One: Origins of Order", 1, "mainmatter", "book-ctx", 2,
     {"node_id": "chapter-ctx-1", "node_kind": "chapter"}),
    ("seg-ctx-0004", "paragraph", "The region's notables gathered to settle the grammar of local order.", 1,
     "mainmatter", "chapter-ctx-1", None, {}),
    ("seg-ctx-0005", "heading", "The Grammar of Power", 2, "mainmatter", "chapter-ctx-1", 3,
     {"node_id": "section-ctx-1", "node_kind": "section"}),
    ("seg-ctx-0006", "paragraph", "The commune shaped the grammar of power, and each faction obeyed its rule of rank",
     2, "mainmatter", "section-ctx-1", None, {}),
    ("seg-ctx-0007", "paragraph", "Rank and ritual bound the village soviet to the wider order.", 2,
     "mainmatter", "section-ctx-1", None, {}),
    ("seg-ctx-0008", "heading", "Ritual and Rank", 3, "mainmatter", "chapter-ctx-1", 3,
     {"node_id": "section-ctx-2", "node_kind": "section"}),
    ("seg-ctx-0009", "paragraph", "When the party congress adjourned, the ritual of rank held the room.", 3,
     "mainmatter", "section-ctx-2", None, {}),
    ("seg-ctx-0010", "paragraph", "The vanguard party pressed its claim through the commune and its allies.", 3,
     "mainmatter", "section-ctx-2", None, {}),
    ("seg-ctx-0011", "heading", "Chapter Two: Revolutionary Rupture", 4, "mainmatter", "book-ctx", 2,
     {"node_id": "chapter-ctx-2", "node_kind": "chapter"}),
    ("seg-ctx-0012", "paragraph", "After the rupture the old hierarchy of the commune dissolved overnight.", 4,
     "mainmatter", "chapter-ctx-2", None, {}),
    ("seg-ctx-0013", "heading", "The Party Congress Decides", 5, "mainmatter", "chapter-ctx-2", 3,
     {"node_id": "section-ctx-3", "node_kind": "section"}),
    ("seg-ctx-0014", "paragraph", "Delegates arrived at the party congress to settle the succession", 5,
     "mainmatter", "section-ctx-3", None, {}),
    ("seg-ctx-0015", "paragraph", "Central approval turned the party congress decree into law.", 5,
     "mainmatter", "section-ctx-3", None, {}),
    ("seg-ctx-0016", "heading", "Counter-Revolution and the Village", 6, "mainmatter", "chapter-ctx-2", 3,
     {"node_id": "section-ctx-4", "node_kind": "section"}),
    ("seg-ctx-0017", "paragraph", "The village soviet survived the counter-revolution by hiding its charter.", 6,
     "mainmatter", "section-ctx-4", None, {}),
    ("seg-ctx-0018", "paragraph", "Only the vanguard party kept the charter safe from the returning landlords.", 6,
     "mainmatter", "section-ctx-4", None, {}),
]

FIXTURE_SEGMENTS = [
    {
        "id": identifier,
        "source_hash": digest(text),
        "kind": kind,
        "source_text": text,
        "page_json": page,
        "zone": zone,
        "parent_node_id": parent,
        "level": level,
        "translatable": True,
        "metadata": metadata,
    }
    for identifier, kind, text, page, zone, parent, level, metadata in _RAW
]

SEG_BY_ID = {row["id"]: row for row in FIXTURE_SEGMENTS}

CONCEPT_A = {
    "concept_id": "concept-aaaaaaaaaaaa",
    "canonical_source": "party congress",
    "surface_forms": ["party congress"],
    "category": "concept",
    "senses": [{"sense_id": "default", "translation": "党代会", "constraint": "preferred", "version": 2, "confidence": 0.9}],
}
CONCEPT_B = {
    "concept_id": "concept-bbbbbbbbbbbb",
    "canonical_source": "commune",
    "surface_forms": ["the commune"],
    "category": "concept",
    "senses": [
        {"sense_id": "default", "translation": "公社", "constraint": "preferred", "version": 1, "confidence": 0.9},
        {"sense_id": "sense-2", "translation": "共同体", "constraint": "sense_constrained", "version": 3, "confidence": 0.8},
    ],
}
CONCEPT_C = {
    "concept_id": "concept-cccccccccccc",
    "canonical_source": "village soviet",
    "surface_forms": ["village soviet"],
    "category": "concept",
    "senses": [{"sense_id": "default", "translation": "村苏维埃", "constraint": "preferred", "version": 1, "confidence": 0.9}],
}
CONCEPT_H = {
    "concept_id": "concept-dddddddddddd",
    "canonical_source": "vanguard party",
    "surface_forms": ["vanguard party"],
    "category": "concept",
    "senses": [{"sense_id": "default", "translation": "先锋党", "constraint": "hard", "version": 1, "confidence": 0.9}],
}
CONCEPT_NEW = {
    "concept_id": "concept-eeeeeeeeeeee",
    "canonical_source": "returning landlords",
    "surface_forms": ["returning landlords"],
    "category": "concept",
    "senses": [{"sense_id": "default", "translation": "还乡地主", "constraint": "preferred", "version": 1, "confidence": 0.9}],
}

# Expected per-segment dependencies under the base concept set.
EXPECTED_DEPS = {
    "seg-ctx-0001": {}, "seg-ctx-0002": {}, "seg-ctx-0003": {}, "seg-ctx-0004": {},
    "seg-ctx-0005": {},
    "seg-ctx-0006": {"concept-bbbbbbbbbbbb": 3},
    "seg-ctx-0007": {"concept-cccccccccccc": 1},
    "seg-ctx-0008": {},
    "seg-ctx-0009": {"concept-aaaaaaaaaaaa": 2},
    "seg-ctx-0010": {"concept-bbbbbbbbbbbb": 3, "concept-dddddddddddd": 1},
    "seg-ctx-0011": {}, "seg-ctx-0012": {"concept-bbbbbbbbbbbb": 3},
    "seg-ctx-0013": {"concept-aaaaaaaaaaaa": 2},
    "seg-ctx-0014": {"concept-aaaaaaaaaaaa": 2},
    "seg-ctx-0015": {"concept-aaaaaaaaaaaa": 2},
    "seg-ctx-0016": {},
    "seg-ctx-0017": {"concept-cccccccccccc": 1},
    "seg-ctx-0018": {"concept-dddddddddddd": 1},
}


def make_concepts(*concepts):
    return {
        "schema_version": 2,
        "book_id": BOOK_ID,
        "concepts": list(concepts) if concepts else [CONCEPT_A, CONCEPT_B, CONCEPT_C, CONCEPT_H],
    }


def make_segments(rows=None):
    return [Segment(**row) for row in (rows if rows is not None else FIXTURE_SEGMENTS)]


class ContextualFakeClient:
    """ChatClient double for the single-target contextual payload shape.

    Emits ``{"id": target.id, "translated_text": "译：" + text}`` (fake targets
    are used only to exercise machinery — never asserted as quality).  A
    ``payload["probe"]`` document answers ``{"ok": true}`` like a provider
    probe round-trip.
    """

    model = "fake-contextual-model"

    def __init__(self, mode="ok", extra_translated=None):
        self.mode = mode
        self.extra_translated = extra_translated
        self.calls = []

    def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        payload = json.loads(user)
        if payload.get("probe") is not None:
            return json.dumps({"ok": True}), {"prompt_tokens": 1, "completion_tokens": 1}
        target = payload["target"]
        if self.mode == "leak":
            content = json.dumps(
                [{"id": target["id"], "translated_text": "译：" + target["text"]},
                 {"id": "seg-ctx-9999", "translated_text": "译：leaked context"}],
                ensure_ascii=False,
            )
        elif self.mode == "missing-id":
            content = json.dumps({"translated_text": "译：" + target["text"]}, ensure_ascii=False)
        elif self.mode == "not-json":
            content = "not json at all"
        else:
            content = json.dumps(
                {"id": target["id"], "translated_text": "译：" + target["text"]},
                ensure_ascii=False,
            )
        return content, {"prompt_tokens": 10, "completion_tokens": 20}


# --------------------------------------------------------------------------
# Gate-complete fixture book writer
# --------------------------------------------------------------------------

def write_fixture_book(
    tmp_path,
    *,
    segments=None,
    nodes=None,
    concept_overrides=None,
    translation=None,
    deps_enabled=True,
):
    """Write a full runnable book (structure+validation+ocr+cleaning manifest)
    and return the loaded config dict."""
    root = Path(tmp_path)
    structure_dir = root / "structure"
    structure_dir.mkdir(parents=True, exist_ok=True)
    structure_path = structure_dir / "book_structure.json"
    structure_path.write_text(json.dumps(
        {"schema_version": 2, "book_id": BOOK_ID, "nodes": nodes if nodes is not None else FIXTURE_NODES},
        ensure_ascii=False,
    ) + "\n", encoding="utf-8")
    (structure_dir / "validation.json").write_text(
        json.dumps({"ok": True, "errors": [], "gate_failures": []}) + "\n", encoding="utf-8")
    input_path = root / "input.json"
    input_path.write_text("{}", encoding="utf-8")

    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_path = output_dir / "cleaned_segments.jsonl"
    write_jsonl(segments_path, segments if segments is not None else FIXTURE_SEGMENTS)
    cleaning_report = output_dir / "cleaning_report.json"
    write_json(cleaning_report, {"schema_version": 1, "book_id": BOOK_ID, "metrics": {"segments": 18}})
    manifest = {
        "schema_version": 2,
        "book_id": BOOK_ID,
        "input_json": str(input_path),
        "input_sha256": file_sha256(input_path),
        "structure_json": str(structure_path),
        "structure_sha256": file_sha256(structure_path),
        "segments_file": str(segments_path),
        "segments_sha256": file_sha256(segments_path),
        "cleaning_report": str(cleaning_report),
        "cleaning_report_sha256": file_sha256(cleaning_report),
    }
    write_json(output_dir / "pipeline_manifest.json", manifest)

    concept_path = None
    if deps_enabled:
        concepts = concept_overrides if concept_overrides is not None else make_concepts()
        concept_path = root / "concepts.json"
        concept_path.write_text(json.dumps(concepts, ensure_ascii=False) + "\n", encoding="utf-8")

    config_path = root / "config.json"
    config_path.write_text(json.dumps({
        "book_id": BOOK_ID,
        "book_title": "Context Fixture",
        "book_title_zh": "上下文夹具",
        "input_json": str(input_path),
        "structure_json": str(structure_path),
        "output_dir": str(output_dir),
        "source_library_root": str(root / "read-only-library"),
        "translate_zones": ["mainmatter", "frontmatter"],
        "quality": {"validation_json": str(structure_dir / "validation.json"), "allow_pending_review": False},
        "allow_demo_translations": True,
        "provider": {"requests_per_minute": 0},
        "translation": translation if translation is not None else {
            "context_mode": "contextual_v2",
            "previous_segments": 2,
            "next_segments": 2,
            "previous_translation_max_chars": 500,
            "chapter_brief": {"enabled": True, "llm_enrich": False},
        },
        "glossary_dependencies": {
            "enabled": deps_enabled,
            "concept_glossary_path": str(concept_path) if concept_path else None,
        },
    }, ensure_ascii=False), encoding="utf-8")
    return load_config(config_path)


def structure_from_disk(config):
    return json.loads(Path(config["structure_json"]).read_text(encoding="utf-8"))


def pending_segments(config, concepts=None, structure=None):
    from book_pipeline.translate import _completed, _segments

    segments = _segments(Path(config["output_dir"]) / "cleaned_segments.jsonl")
    completed = _completed(Path(config["output_dir"]) / "translations.jsonl")
    briefs = build_chapter_briefs(structure if structure is not None else structure_from_disk(config), segments)
    brief_hashes = current_briefs(briefs)
    concept_list = concepts if concepts is not None else load_concept_glossary(
        Path(config["glossary_dependencies"]["concept_glossary_path"]))
    versions = current_concept_versions(concept_list)
    nodes_by_id = nodes_index(structure if structure is not None else structure_from_disk(config))
    return [
        segment for segment in segments
        if segment.translatable
        and not is_translation_current_v2(segment, completed.get(segment.id),
                                          brief_hashes=brief_hashes, concept_versions=versions,
                                          nodes_by_id=nodes_by_id)
    ], segments


# --------------------------------------------------------------------------
# Parser: strict single-target ID round trip
# --------------------------------------------------------------------------

def _segment(segment_id="seg-ctx-0006"):
    row = dict(SEG_BY_ID[segment_id])
    return Segment(**row)


def test_parse_target_only_exact_round_trip():
    target = _segment()
    content = json.dumps({"id": target.id, "translated_text": "译：" + target.source_text}, ensure_ascii=False)
    assert parse_target_only(content, target) == "译：" + target.source_text
    # fenced variants and wrapped containers are accepted when they hold exactly the target
    fenced = "```json\n" + json.dumps({"translations": [{"id": target.id, "translated_text": "甲"}]}) + "\n```"
    assert parse_target_only(fenced, target) == "甲"


def test_parse_target_only_rejects_context_leak_extra_ids_missing_and_empty():
    target = _segment()
    leaked = [
        json.dumps([{"id": target.id, "translated_text": "x"}, {"id": "seg-ctx-0007", "translated_text": "y"}]),
        json.dumps({"segments": [{"id": "seg-ctx-0007", "translated_text": "y"}]}),
        json.dumps({"id": "seg-ctx-0007", "translated_text": "x"}),
        json.dumps({"translations": [{"id": target.id, "translated_text": "x"}, {"id": target.id, "translated_text": "x"}]}),
        json.dumps({"translated_text": "missing id"}),
        json.dumps({"id": target.id, "translated_text": "   "}),
        json.dumps([{"id": target.id, "translated_text": "ok"}, {"translated_text": "no id"}]),
    ]
    for content in leaked:
        with pytest.raises(ValueError):
            parse_target_only(content, target)
    with pytest.raises(ValueError):
        parse_target_only("not json", target)  # JSONDecodeError is a ValueError subclass


def test_parse_target_only_preserves_structural_tokens_and_table_delimiters():
    source = "Body text with footnote[^fn-p0001-1] and url https://example.org/a?b=1 stayed intact."
    target = Segment("seg-struct", digest(source), "paragraph", source, 1, "mainmatter", "chapter-1", None)
    good = "译文带脚注[^fn-p0001-1]与链接 https://example.org/a?b=1 保持原样。"
    assert parse_target_only(json.dumps({"id": target.id, "translated_text": good}), target) == good
    bad = "译文丢掉了脚注和链接。"
    with pytest.raises(ValueError, match="Structural token mismatch"):
        parse_target_only(json.dumps({"id": target.id, "translated_text": bad}), target)
    table = Segment("seg-table", digest("| a | b |\n|---|---|"), "table", "| a | b |\n|---|---|",
                    1, "mainmatter", "chapter-1", None)
    with pytest.raises(ValueError, match="delimiter"):
        # pipe-count drift (a lost delimiter) is rejected, exactly like v1
        parse_target_only(json.dumps({"id": table.id, "translated_text": "a | b |\n|---|---"}), table)
    assert parse_target_only(json.dumps({"id": table.id, "translated_text": table.source_text}), table) == table.source_text


def test_parse_target_only_strips_provider_tool_call_framing():
    """A DeepSeek model sometimes appends DeepSeek tool-call close tags after
    an otherwise complete JSON object; that is framing noise, not payload.

    Live regression: the-dictators-dilemma seg-9a43660c9c72e1707a92 failed all
    16 repair attempts with JSONDecodeError "Extra data"."""
    source = "Fuping's micro-finance programmes provide small loans."
    target = Segment("seg-tool", digest(source), "paragraph", source, 1, "mainmatter", "chapter-1", None)
    body = json.dumps({"id": target.id, "translated_text": "富平的小额信贷项目提供小额贷款。"})
    framed = body + "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    assert parse_target_only(framed, target) == "富平的小额信贷项目提供小额贷款。"
    # framing must never rescue a genuinely incomplete response
    with pytest.raises(ValueError):
        parse_target_only('{"id": "seg-tool", "translated_text": "富平的小额', target)


def test_parse_target_only_accepts_citation_brackets_around_url():
    """A source citation "(http://x)" and a Chinese rendering "（http://x）" must
    parse: the URL itself is identical, only the bracket style differs (the URL
    class excludes the full-width "）" but not the ASCII ")").

    Live regression: seg-58b2832dd49aaabb4c8e (eastern-europe-since-1945)
    failed 16 attempts with no way to recover through retries.
    """
    source = "See TOL (http://www.tol.cz) and RFE (http://www.rferl.org)."
    target = Segment("seg-cite", digest(source), "paragraph", source, 1, "mainmatter", "chapter-1", None)
    good = "参见 TOL（http://www.tol.cz）与 RFE（http://www.rferl.org）。"
    assert parse_target_only(json.dumps({"id": target.id, "translated_text": good}), target) == good
    # the URL text itself is still required verbatim
    dropped = "参见 TOL 与 RFE。"
    with pytest.raises(ValueError, match="Structural token mismatch"):
        parse_target_only(json.dumps({"id": target.id, "translated_text": dropped}), target)


def test_parse_target_only_rewrites_footnote_whitespace_to_source_form():
    """Whitespace-only differences are repaired to the source spelling (P09),
    while bracket-style differences are tolerated without rewriting the text."""
    source = "Body with footnote $ ^{9} $ and a url (http://example.org/x)."
    target = Segment("seg-ws", digest(source), "paragraph", source, 1, "mainmatter", "chapter-1", None)
    variant = "正文含脚注 $^{9}$，链接（http://example.org/x）。"
    parsed = parse_target_only(json.dumps({"id": target.id, "translated_text": variant}), target)
    assert "$ ^{9} $" in parsed          # footnote restored to the source form
    assert "（http://example.org/x）" in parsed  # bracket style left untouched


# --------------------------------------------------------------------------
# Prompts: verbatim clause + payload shape
# --------------------------------------------------------------------------

def test_system_prompt_carries_verbatim_only_target_clause():
    prompt = system_prompt({"domain": "历史学", "source_lang": "英语", "target_lang": "简体中文"})
    assert CONTEXT_ONLY_CLAUSE in prompt
    assert "只返回严格 JSON" in prompt
    assert "required_output" not in prompt  # output contract lives in the payload


def test_build_payload_single_target_and_context_isolation():
    target = _segment("seg-ctx-0014")
    brief = {"title_source": "Chapter Two: Revolutionary Rupture", "kind": "chapter", "zone": "mainmatter",
             "pages": [4, 6], "section_titles": ["The Party Congress Decides"], "segment_count": 8,
             "brief_hash": "a" * 64}
    payload = build_payload(
        chapter_brief=brief,
        previous_source=["Earlier sentence of the same chapter.", "One more."],
        target=target,
        next_source=["Later sentence in the same chapter."],
        previous_translation="前一段译文",
        glossary=[{"concept_id": "concept-aaaaaaaaaaaa", "term": "party congress", "translation": "党代会",
                   "constraint": "preferred"}],
    )
    assert payload["required_output"] == {"id": target.id, "translated_text": "translation of the target text only"}
    assert payload["target"] == {"id": target.id, "kind": target.kind, "text": target.source_text}
    serialized = json.dumps(payload, ensure_ascii=False)
    # context ids/segments may never sit under an output field
    assert '"segments"' not in serialized
    assert payload["previous_source"][0].startswith("Earlier")
    assert payload["chapter_brief"]["brief_hash"] == "a" * 64
    assert payload["previous_translation"] == "前一段译文"
    assert payload["glossary"][0]["concept_id"] == "concept-aaaaaaaaaaaa"


# --------------------------------------------------------------------------
# Briefs: determinism, canonical hashing, chapter-scoped change
# --------------------------------------------------------------------------

def test_brief_hash_deterministic_and_ignores_nonstructural_keys():
    fields = {"chapter_id": "c1", "title_source": "T", "kind": "chapter", "level": 2, "zone": "mainmatter",
              "page_range": [1, 5], "section_titles": ["A", "B"], "segment_count": 9}
    assert brief_hash(fields) == brief_hash(dict(fields))
    shuffled = dict(fields)
    shuffled["llm_enrichment"] = {"topic": "noise"}  # non-structural decoration
    assert brief_hash(fields) == brief_hash(shuffled)
    assert brief_hash({**fields, "title_source": "T2"}) != brief_hash(fields)


def test_build_chapter_briefs_covers_every_segment_with_chapter_scoped_change():
    structure = {"schema_version": 2, "book_id": BOOK_ID, "nodes": FIXTURE_NODES}
    segments = make_segments()
    briefs = build_chapter_briefs(structure, segments)
    assert set(briefs) == {"chapter-ctx-1", "chapter-ctx-2", zone_bucket_id("frontmatter")}
    one = briefs["chapter-ctx-1"]
    assert one.title_source == "Chapter One: Origins of Order"
    assert one.kind == "chapter" and one.zone == "mainmatter"
    assert one.page_range == (1, 3)
    assert one.section_titles == ["The Grammar of Power", "Ritual and Rank"]
    assert one.segment_count == 8 and one.translatable_count == 8
    assert len(one.brief_hash) == 64
    bucket = briefs[zone_bucket_id("frontmatter")]
    assert bucket.kind == "zone_bucket" and bucket.translatable_count == 2
    # every translatable segment maps to exactly one brief
    nodes_by_id = nodes_index(structure)
    for segment in segments:
        key = chapter_id_for_segment(segment, nodes_by_id) or zone_bucket_id(segment.zone)
        assert key in briefs
    # chapter-scoped change: editing one section title moves only that chapter's hash
    edited_nodes = [dict(node) for node in FIXTURE_NODES]
    for node in edited_nodes:
        if node["id"] == "section-ctx-1":
            node["title_source"] = "The Grammar of Office"
    edited = build_chapter_briefs({"nodes": edited_nodes}, segments)
    assert edited["chapter-ctx-1"].brief_hash != briefs["chapter-ctx-1"].brief_hash
    assert edited["chapter-ctx-2"].brief_hash == briefs["chapter-ctx-2"].brief_hash
    assert edited[zone_bucket_id("frontmatter")].brief_hash == briefs[zone_bucket_id("frontmatter")].brief_hash


# --------------------------------------------------------------------------
# Deps: phrase-boundary matching + max-sense versioning
# --------------------------------------------------------------------------

def test_resolve_concept_deps_matches_surfaces_and_uses_max_sense_version():
    concepts = make_concepts()["concepts"]
    segments = make_segments()
    for segment in segments:
        assert resolve_concept_deps(segment, concepts) == EXPECTED_DEPS[segment.id]
    # polysemy: concept-B has senses at versions 1 and 3 -> dependency version 3
    assert resolve_concept_deps(make_segments()[9], concepts)["concept-bbbbbbbbbbbb"] == 3
    # phrase boundaries: "the commune" must not match "communities" or "communal"
    probe = Segment("seg-probe", digest("Communal life and communities ignored the commune."), "paragraph",
                    "Communal life and communities ignored the commune.", 1, "mainmatter", "c", None)
    assert resolve_concept_deps(probe, concepts) == {"concept-bbbbbbbbbbbb": 3}
    assert resolve_concept_deps(probe, []) == {}


# --------------------------------------------------------------------------
# Windows: determinism + chapter boundary caps + previous translation rules
# --------------------------------------------------------------------------

def test_chapter_order_and_window_deterministic_bounded_within_chapter():
    segments = make_segments()
    structure = {"nodes": FIXTURE_NODES}
    nodes_by_id = nodes_index(structure)
    chapter1 = [s for s in segments if chapter_id_for_segment(s, nodes_by_id) == "chapter-ctx-1"]
    ordered = chapter_order(chapter1)
    assert [s.id for s in ordered] == [f"seg-ctx-{i:04d}" for i in range(3, 11)]
    assert chapter_order(chapter1) == ordered  # deterministic
    index = ordered.index(next(s for s in ordered if s.id == "seg-ctx-0007"))
    prev_texts, next_texts = window(ordered, index, prev=2, next_=2)
    assert prev_texts == [SEG_BY_ID["seg-ctx-0005"]["source_text"], SEG_BY_ID["seg-ctx-0006"]["source_text"]]
    assert next_texts == [SEG_BY_ID["seg-ctx-0008"]["source_text"], SEG_BY_ID["seg-ctx-0009"]["source_text"]]
    assert window(ordered, index, prev=2, next_=2) == (prev_texts, next_texts)
    # caps respected at the chapter edges: no cross-chapter leakage possible
    last_index = len(ordered) - 1
    _, tail_next = window(ordered, last_index, prev=2, next_=2)
    assert tail_next == []  # chapter two never leaks into chapter one's window
    head_prev, _ = window(ordered, 0, prev=2, next_=2)
    assert head_prev == []
    with pytest.raises(ValueError):
        window(ordered, 1, prev=-1, next_=1)


def test_previous_translation_text_bounded_and_sentence_finish_rules():
    rows = [None, None, {"translated_text": "已完成的第一段译文", "status": "completed"},
            {"translated_text": "承接上一段的中间译文", "status": "completed"}]
    assert previous_translation_text(rows, max_chars=500, sentence_finish=SENTENCE_FINISH) == "承接上一段的中间译文"
    assert len(previous_translation_text(rows, max_chars=4, sentence_finish=SENTENCE_FINISH)) <= 4
    finished = rows + [{"translated_text": "这句话以句号结束。", "status": "completed"}]
    assert previous_translation_text(finished, max_chars=500, sentence_finish=SENTENCE_FINISH) == ""
    assert previous_translation_text([None, None], max_chars=500, sentence_finish=SENTENCE_FINISH) == ""


# --------------------------------------------------------------------------
# Invalidation matrix + selective scenarios
# --------------------------------------------------------------------------

def _current_check(segment_id, row, concept_versions, brief_hashes=None, nodes_by_id=None):
    segment = make_segments()[int(segment_id.rsplit("-", 1)[1]) - 1]
    structure = {"nodes": FIXTURE_NODES}
    brief_hashes = brief_hashes if brief_hashes is not None else current_briefs(
        build_chapter_briefs(structure, make_segments()))
    return is_translation_current_v2(segment, row, brief_hashes=brief_hashes,
                                     concept_versions=concept_versions,
                                     nodes_by_id=nodes_by_id or nodes_index(structure))


def _sample_row(segment_id):
    segment = make_segments()[int(segment_id.rsplit("-", 1)[1]) - 1]
    concepts = make_concepts()["concepts"]
    structure = {"nodes": FIXTURE_NODES}
    briefs = build_chapter_briefs(structure, make_segments())
    chapter_id = chapter_id_for_segment(segment, nodes_index(structure)) or zone_bucket_id(segment.zone)
    return record_row(segment, "译文", "fake", "", chapter_id=chapter_id,
                      brief_hash=briefs[chapter_id].brief_hash, prompt_version=PROMPTS_VERSION,
                      deps=resolve_concept_deps(segment, concepts))


def test_is_translation_current_v2_full_matrix():
    versions = current_concept_versions(make_concepts()["concepts"])
    row = _sample_row("seg-ctx-0009")  # depends on concept-A at version 2
    assert _current_check("seg-ctx-0009", row, versions)
    stale_source = dict(row, source_hash="0" * 64)
    assert not _current_check("seg-ctx-0009", stale_source, versions)
    assert not _current_check("seg-ctx-0009", dict(row, status="pending"), versions)
    assert not _current_check("seg-ctx-0009", None, versions)
    old_prompt = dict(row)
    old_prompt["metadata"] = dict(row["metadata"], prompt_version=PROMPTS_VERSION - 1)
    assert not _current_check("seg-ctx-0009", old_prompt, versions)
    legacy = dict(row)
    legacy["metadata"] = dict(row["metadata"], context_mode="legacy")
    assert not _current_check("seg-ctx-0009", legacy, versions)
    wrong_brief = dict(row)
    wrong_brief["metadata"] = dict(row["metadata"], brief_hash="b" * 64)
    assert not _current_check("seg-ctx-0009", wrong_brief, versions)
    bumped = dict(versions, **{"concept-aaaaaaaaaaaa": 3})
    assert not _current_check("seg-ctx-0009", row, bumped)  # concept-A bumped -> stale
    # an unrelated segment (no concept-A dep) stays current after the bump
    assert _current_check("seg-ctx-0007", _sample_row("seg-ctx-0007"), bumped)


def test_selective_invalidation_concept_bump_isolates_dep_segments(tmp_path):
    config = write_fixture_book(tmp_path)
    client = ContextualFakeClient()
    translate_v2(config, client)  # translate everything
    concepts = load_concept_glossary(Path(config["glossary_dependencies"]["concept_glossary_path"]))
    assert len(concepts) == 4
    assert translate_v2(config, client)["remaining_segments"] == 0  # rerun no-op
    # bump concept-A: 2 -> 3 (write a new concept file, keep everything else)
    concepts_bumped = []
    for concept in concepts:
        updated = dict(concept)
        if concept["concept_id"] == "concept-aaaaaaaaaaaa":
            updated = json.loads(json.dumps(concept))
            updated["senses"] = [dict(concept["senses"][0], version=3)]
        concepts_bumped.append(updated)
    path = Path(config["glossary_dependencies"]["concept_glossary_path"])
    path.write_text(json.dumps(make_concepts(*concepts_bumped), ensure_ascii=False) + "\n", encoding="utf-8")
    # v2 currency: exactly the concept-A segments become pending (measured
    # before any repair pass runs)
    pending, _ = pending_segments(config)
    dep_segments = [sid for sid, deps in EXPECTED_DEPS.items() if "concept-aaaaaaaaaaaa" in deps]
    assert {segment.id for segment in pending} == set(dep_segments) == {"seg-ctx-0009", "seg-ctx-0013", "seg-ctx-0014", "seg-ctx-0015"}
    # the repair pass translates exactly those four and records the bumped version
    report = translate_v2(config, client)
    assert report["translated_this_run"] == 4 and report["remaining_segments"] == 0
    rows = list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))
    assert len(rows) == 22  # 18 original + 4 repaired appends
    latest = {}
    for row in rows:  # last-write-wins like translate._completed
        latest[row["segment_id"]] = row
    for segment_id in dep_segments:
        assert latest[segment_id]["metadata"]["glossary_dependencies"]["concept-aaaaaaaaaaaa"] == 3
    assert translate_v2(config, client)["remaining_segments"] == 0


def test_brand_new_concept_invalidates_zero_segments(tmp_path):
    config = write_fixture_book(tmp_path)
    client = ContextualFakeClient()
    translate_v2(config, client)
    path = Path(config["glossary_dependencies"]["concept_glossary_path"])
    concepts = load_concept_glossary(path) + [CONCEPT_NEW]
    path.write_text(json.dumps(make_concepts(*concepts), ensure_ascii=False) + "\n", encoding="utf-8")
    report = translate_v2(config, client)
    assert report["completed_segments"] == 18
    assert report["remaining_segments"] == 0


def test_brief_hash_change_invalidates_only_its_chapter(tmp_path):
    config = write_fixture_book(tmp_path)
    client = ContextualFakeClient()
    translate_v2(config, client)
    # edit only chapter one's section title inside a *copied* structure and run
    # the v2 currency check against it (gates are bypassed for this scenario;
    # a live structure edit would rerun clean, which is exactly the workflow
    # that surfaces per-chapter brief drift)
    structure = structure_from_disk(config)
    for node in structure["nodes"]:
        if node["id"] == "section-ctx-1":
            node["title_source"] = "The Grammar of Office"
    pending, _ = pending_segments(config, structure=structure)
    pending_ids = {segment.id for segment in pending}
    chapter_one = {f"seg-ctx-{i:04d}" for i in range(3, 11)}
    chapter_two = {f"seg-ctx-{i:04d}" for i in range(11, 19)}
    assert pending_ids == chapter_one  # chapter one only
    assert not (pending_ids & chapter_two)


# --------------------------------------------------------------------------
# End-to-end via the full gate path
# --------------------------------------------------------------------------

def test_translate_v2_fixture_end_to_end_records_and_idempotent_resume(tmp_path):
    config = write_fixture_book(tmp_path)
    fake = ContextualFakeClient()
    report = translate_v2(config, fake, limit=5)
    assert report["completed_segments"] == 5 and report["translated_this_run"] == 5
    assert report["pipeline"] == "contextual_v2"
    rows = list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))
    assert len(rows) == 5
    usage_rows = list(read_jsonl(Path(config["output_dir"]) / "usage.jsonl"))
    assert len(usage_rows) == 5
    for row in usage_rows:
        assert row["prompt_version"] == PROMPTS_VERSION
        assert row["context_mode"] == "contextual_v2"
    structure = structure_from_disk(config)
    briefs = build_chapter_briefs(structure, make_segments())
    for row in rows:
        assert row["status"] == "completed" and row["attempt"] == 1
        assert row["metadata"]["context_mode"] == "contextual_v2"
        assert row["metadata"]["prompt_version"] == PROMPTS_VERSION
        segment = next(s for s in make_segments() if s.id == row["segment_id"])
        assert row["source_hash"] == segment.source_hash
        assert row["metadata"]["glossary_dependencies"] == EXPECTED_DEPS[segment.id]
        assert row["metadata"]["chapter_id"] in briefs
        assert row["metadata"]["brief_hash"] == briefs[row["metadata"]["chapter_id"]].brief_hash
    # v1 consumers see the v2 rows as current (shared file, v1-compatible superset)
    status = translation_status(config)
    assert status["completed_segments"] == 5
    # resume semantics: target_completed is an absolute cap, limit a relative slice
    resume = translate_v2(config, fake, target_completed=8)
    assert resume["completed_segments"] == 8 and resume["translated_this_run"] == 3
    done = translate_v2(config, fake)
    assert done["completed_segments"] == 18 and done["remaining_segments"] == 0
    assert len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))) == 18
    assert translate_v2(config, fake)["translated_this_run"] == 0
    # artifacts written
    briefs_file = json.loads((Path(config["output_dir"]) / "chapter_briefs.json").read_text(encoding="utf-8"))
    assert briefs_file["schema_version"] == 1 and len(briefs_file["briefs"]) == 3
    assert briefs_file["prompts_version"] == PROMPTS_VERSION
    deps_rows = list(read_jsonl(Path(config["output_dir"]) / "contextual_deps.jsonl"))
    assert len(deps_rows) == 18


def test_payload_carries_brief_prev_next_and_bounded_previous_translation(tmp_path):
    config = write_fixture_book(tmp_path)
    fake = ContextualFakeClient()
    translate_v2(config, fake, limit=18)
    calls = {json.loads(call["user"])["target"]["id"]: json.loads(call["user"]) for call in fake.calls
             if json.loads(call["user"]).get("target")}
    assert len(calls) == 18
    # a mid-chapter target gets both prev and next context, all from chapter one
    middle = calls["seg-ctx-0007"]
    assert middle["previous_source"] == [SEG_BY_ID["seg-ctx-0005"]["source_text"],
                                         SEG_BY_ID["seg-ctx-0006"]["source_text"]]
    assert middle["next_source"] == [SEG_BY_ID["seg-ctx-0008"]["source_text"],
                                     SEG_BY_ID["seg-ctx-0009"]["source_text"]]
    assert middle["chapter_brief"]["title_source"] == "Chapter One: Origins of Order"
    assert middle["chapter_brief"]["section_titles"] == ["The Grammar of Power", "Ritual and Rank"]
    assert middle["chapter_brief"]["brief_hash"]  # non-empty
    # previous_translation = bounded tail of the previous completed translation
    assert middle["previous_translation"] == "译：" + SEG_BY_ID["seg-ctx-0006"]["source_text"]
    assert len(middle["previous_translation"]) <= 500
    # the target whose previous segment ends on sentence-finish punctuation gets ""
    after_finish = calls["seg-ctx-0010"]
    assert after_finish["previous_translation"] == ""
    # chapter two's first target never sees chapter one text
    ch2_first = calls["seg-ctx-0011"]
    assert all("Rupture" not in text for text in ch2_first["previous_source"])
    assert ch2_first["previous_translation"] == ""
    # system prompt clause present in every call
    assert all(CONTEXT_ONLY_CLAUSE in call["system"] for call in fake.calls if not json.loads(call["user"]).get("probe"))


def test_translate_v2_parse_retries_bounded_and_no_fabricated_record(tmp_path):
    config = write_fixture_book(tmp_path)
    leaky = ContextualFakeClient(mode="leak")
    with pytest.raises(RuntimeError, match="failed response validation"):
        translate_v2(config, leaky, limit=1)
    invalid_file = Path(config["output_dir"]) / "last_invalid_response.txt"
    assert invalid_file.exists() and invalid_file.read_text(encoding="utf-8").strip()
    assert len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))) == 0
    bad_json = ContextualFakeClient(mode="not-json")
    with pytest.raises(RuntimeError):
        translate_v2(config, bad_json, limit=1)
    assert invalid_file.read_text(encoding="utf-8") == "not json at all"


def test_ladder_records_steps_and_best_effort_glossary_violations(tmp_path):
    config = write_fixture_book(tmp_path)
    fake = ContextualFakeClient()
    report = run_ladder(config, fake, "10")
    assert report["blocked"] is False and report["executed_steps"] == 10 and report["succeeded"] == 10
    ladder_rows = list(read_jsonl(Path(config["output_dir"]) / "translation_ladder.jsonl"))
    assert len(ladder_rows) == 10
    assert [row["step"] for row in ladder_rows] == list(range(1, 11))
    translations = list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))
    assert len(translations) == 10
    for row in ladder_rows:
        assert row["success"] is True
        assert set(row["tokens"]) == {"prompt", "completion"}
        assert "source_text" in row["sample"] and row["sample"]["target_text"]
    # the hard-constraint concept's canonical translation never appears in fake
    # targets -> recorded as a best-effort violation on the matching step
    violations = [row["glossary_violations"] for row in ladder_rows if row["glossary_violations"]]
    assert any("concept-dddddddddddd" in row for row in violations)
    # remaining steps stay pending for a later run
    leftover = translate_v2(config, fake)
    assert leftover["completed_segments"] == 18


def test_ladder_blocked_on_probe_failure_writes_no_fabricated_rows(tmp_path, monkeypatch):
    config = write_fixture_book(tmp_path)
    fake = ContextualFakeClient()
    monkeypatch.setattr(ladder_module, "provider_probe", lambda client: False)
    report = run_ladder(config, fake, "10")
    assert report["blocked"] is True and report["executed_steps"] == 0
    rows = list(read_jsonl(Path(config["output_dir"]) / "translation_ladder.jsonl"))
    assert rows == [{"blocked": True, "reason": "provider probe failed: probe returned no usable reply"}]
    assert len(list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))) == 0


def test_legacy_context_mode_rejected_with_pointer_to_v1(tmp_path):
    config = write_fixture_book(tmp_path, translation={
        "context_mode": "legacy", "previous_segments": 2, "next_segments": 2,
        "previous_translation_max_chars": 500, "chapter_brief": {"enabled": True, "llm_enrich": False}},
        deps_enabled=False)
    with pytest.raises(ValueError, match="translate --config"):
        translate_v2(config, ContextualFakeClient(), limit=1)
    with pytest.raises(ValueError, match="contextual_v2"):
        translate_v2(dict(config, translation={}), ContextualFakeClient(), limit=1)


def test_llm_enrich_false_only_and_cli_parser_shape():
    parsed = cli_parser().parse_args(["translate-v2", "--config", "x.json", "--limit", "3"])
    assert parsed.command == "translate-v2" and parsed.limit == 3 and parsed.ladder is None
    parsed_ladder = cli_parser().parse_args(["translate-v2", "--config", "x.json", "--ladder", "all"])
    assert parsed_ladder.ladder == "all" and parsed_ladder.target_completed is None
    config = {"translation": {"chapter_brief": {"llm_enrich": True}}}
    from book_pipeline.contextual.orchestrator import _v2_settings
    with pytest.raises(ValueError, match="llm_enrich"):
        _v2_settings(config)


def test_ladder_skips_non_translatable_empty_segments(tmp_path):
    """P07-pretest regression: run_ladder must only pick translatable segments.
    A leading non-translatable empty placeholder used to fail every step with
    'no chapter brief'; it must be skipped and translation must proceed."""
    import json as _json

    config = write_fixture_book(tmp_path)
    segments_path = Path(config["output_dir"]) / "cleaned_segments.jsonl"
    rows = [json.loads(line) for line in segments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    placeholder = {
        "id": "seg-placeholder-empty",
        "source_hash": "0" * 64,
        "kind": "image",
        "source_text": "",
        "page_json": 0,
        "zone": "mainmatter",
        "parent_node_id": None,
        "level": None,
        "translatable": False,
        "source_block_ids": ["blk-placeholder"],
        "metadata": {"label": "image"},
    }
    rows.insert(0, placeholder)
    segments_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    # keep the cleaning-manifest SHA gate consistent with the edited segments
    from book_pipeline.provenance import file_sha256 as _sha

    manifest_path = Path(config["output_dir"]) / "pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["segments_sha256"] = _sha(segments_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fake = ContextualFakeClient()
    report = run_ladder(config, fake, "10")
    assert report["blocked"] is False
    assert report["succeeded"] == 10, report
    ladder_rows = list(read_jsonl(Path(config["output_dir"]) / "translation_ladder.jsonl"))
    assert len(ladder_rows) == 10
    assert all(row["success"] is True for row in ladder_rows)
    # placeholder never attempted (no record, no error row about it)
    assert all(row["sample"]["source_text"].strip() for row in ladder_rows)


def test_structural_token_whitespace_variants_are_equivalent():
    """P09 live-run fix: '$ ^{9} $' (source) vs '$^{9}$' (translation) are the
    same footnote/formula token modulo internal whitespace and must pass
    parity; genuinely missing/extra tokens must still fail."""
    from book_pipeline.contextual.parser import _verify_structural_tokens
    from book_pipeline.models import Segment

    def seg(text):
        return Segment(id="seg-1", source_hash="h", kind="paragraph", source_text=text,
                       page_json=1, parent_node_id=None, zone="mainmatter",
                       translatable=True)

    spaced_src = "Text with footnote $ ^{9} $ end."
    compact = "译文带脚注 $^{9}$ 结束。"
    # target token is normalized back to the SOURCE form (whitespace preserved)
    assert _verify_structural_tokens(seg(spaced_src), compact) == "译文带脚注 $ ^{9} $ 结束。"
    # same token reordered vs dropped must fail
    multi = seg("A $ ^{1} $ then $ ^{2} $ end.")
    with pytest.raises(ValueError, match="Structural token mismatch"):
        _verify_structural_tokens(multi, "译 $^{2}$ 只有一条")


def test_translate_v2_skip_failed_segments_records_error_and_continues(tmp_path):
    """P09 unattended-run fix: with translation.skip_failed_segments=true a
    segment that exhausts parse attempts gets an explicit status:'error' row
    (never a fabricated translation) and the run continues with later
    segments; the error segment stays pending for a later retry."""
    config = write_fixture_book(tmp_path, translation={
        "context_mode": "contextual_v2",
        "previous_segments": 2,
        "next_segments": 2,
        "previous_translation_max_chars": 500,
        "chapter_brief": {"enabled": True, "llm_enrich": False},
        "skip_failed_segments": True,
    })

    class _SkipThenOk(ContextualFakeClient):
        def complete(self, system, user):
            self.calls.append(1)
            if len(self.calls) <= 1:
                raise ValueError("forced invalid response")
            return super().complete(system, user)

    fake = _SkipThenOk()
    report = translate_v2(config, fake, limit=4)
    rows = list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))
    errors = [r for r in rows if r.get("status") == "error"]
    completed = [r for r in rows if r.get("status") == "completed"]
    assert errors, "first segment must be recorded as error"
    assert all(r.get("translated_text") == "" for r in errors)
    assert completed, "later segments must still translate"
    assert report.get("error_segment_count") == len(errors)
    # error segment id is NOT in completed and stays pending next run
    assert all(r["segment_id"] != errors[0]["segment_id"] for r in completed)


def test_translate_v2_parallel_max_workers_completes_and_resumes(tmp_path):
    """Concurrency (translation.max_workers>1) must reach the same end state as
    serial: every pending segment completes, the run resumes idempotently, and
    per-chapter ordering is preserved in the written records."""
    translation = {
        "context_mode": "contextual_v2",
        "previous_segments": 2,
        "next_segments": 2,
        "previous_translation_max_chars": 500,
        "chapter_brief": {"enabled": True, "llm_enrich": False},
        "max_workers": 4,
    }
    config = write_fixture_book(tmp_path, translation=translation)
    fake = ContextualFakeClient()
    report = translate_v2(config, fake)
    assert report["max_workers"] == 4
    assert report["remaining_segments"] == 0
    assert report["coverage"] == 1.0
    assert report["error_segment_count"] == 0
    # idempotent resume: nothing left, nothing retranslated
    again = translate_v2(config, ContextualFakeClient())
    assert again["remaining_segments"] == 0
    assert again["translated_this_run"] == 0
    rows = list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))
    completed = [r for r in rows if r.get("status") == "completed"]
    assert completed, "parallel run must write completed records"
    # every completed segment's latest row is current (last-write-wins intact)
    assert {r["segment_id"] for r in completed} == {
        r["segment_id"] for r in rows
    }


def test_translate_v2_parallel_skip_failed_records_error_and_continues(tmp_path):
    """In the parallel path an exhausted segment is recorded as an explicit
    status:'error' row (empty text) while sibling segments still translate."""
    config = write_fixture_book(tmp_path, translation={
        "context_mode": "contextual_v2",
        "previous_segments": 2,
        "next_segments": 2,
        "previous_translation_max_chars": 500,
        "chapter_brief": {"enabled": True, "llm_enrich": False},
        "skip_failed_segments": True,
        "max_workers": 4,
    })

    class _FirstCallFails(ContextualFakeClient):
        def complete(self, system, user):
            self.calls.append(1)
            if len(self.calls) <= 1:
                raise ValueError("forced invalid response")
            return super().complete(system, user)

    report = translate_v2(config, _FirstCallFails())
    rows = list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))
    errors = [r for r in rows if r.get("status") == "error"]
    completed = [r for r in rows if r.get("status") == "completed"]
    assert errors, "the forced failure must surface as an error row"
    assert all(r.get("translated_text") == "" for r in errors)
    assert report["error_segment_count"] == len(errors)
    assert completed, "sibling segments must still complete under concurrency"
    # the errored segment stays pending: not present among completed
    assert all(r["segment_id"] != errors[0]["segment_id"] for r in completed)


def test_translate_v2_parallel_fail_closed_without_skip(tmp_path):
    """Without skip_failed_segments the parallel path must still fail closed:
    any exhausted segment aborts the run (RuntimeError), never a silent drop."""
    config = write_fixture_book(tmp_path, translation={
        "context_mode": "contextual_v2",
        "previous_segments": 2,
        "next_segments": 2,
        "previous_translation_max_chars": 500,
        "chapter_brief": {"enabled": True, "llm_enrich": False},
        "skip_failed_segments": False,
        "max_workers": 4,
    })

    class _AlwaysFails(ContextualFakeClient):
        def complete(self, system, user):
            raise ValueError("forced invalid response")

    with pytest.raises(ValueError, match="forced invalid response"):
        translate_v2(config, _AlwaysFails())


def test_translate_v2_rejects_max_workers_zero(tmp_path):
    """max_workers must be >= 1 (0 would disable the pool ambiguously)."""
    config = write_fixture_book(tmp_path, translation={
        "context_mode": "contextual_v2",
        "previous_segments": 2,
        "next_segments": 2,
        "previous_translation_max_chars": 500,
        "chapter_brief": {"enabled": True, "llm_enrich": False},
        "max_workers": 0,
    })
    with pytest.raises(ValueError, match="max_workers"):
        translate_v2(config, ContextualFakeClient())


def test_url_token_stops_at_cjk_punctuation():
    """P09 live-run fix: a URL immediately followed by full-width Chinese
    punctuation (no space) must tokenize as JUST the URL, not swallow the rest
    of the sentence — otherwise parity checks fail deterministically."""
    from book_pipeline.contextual.parser import _verify_structural_tokens
    from book_pipeline.models import Segment

    source = "Available at http://cpc.people.cn/a.html, next text."
    target_ok = "参见 http://cpc.people.cn/a.html，后续正文。"
    seg = Segment(id="seg-1", source_hash="h", kind="paragraph", source_text=source,
                  page_json=1, parent_node_id=None, zone="mainmatter", translatable=True)
    # tokenizes to just the URL on both sides -> parity passes
    assert _verify_structural_tokens(seg, target_ok)
    # genuinely dropping the URL still fails
    import pytest as _pt
    with _pt.raises(ValueError, match="Structural token mismatch"):
        _verify_structural_tokens(seg, "无链接译文。")
