"""Shared fixtures/helpers for the P03 structure-resolver tests.

The synthetic evidence pack mirrors the exact shapes the chapter repo's
``evidence_v2`` pipeline emits (``heading_candidates.jsonl`` +
``book_structure_evidence.json`` + optional ``pdf_evidence.json``) so the
resolver reads it identically to real runs.  All ids are 16-hex stable ids
formatted per the cross-repo contract (``cand-*`` / ``blk-*``).

``FakeClient`` implements a JSON mode (structured completion for the resolver)
while preserving the plain-text mode used by the translation tests.
"""
import json
from pathlib import Path

from book_pipeline.io_utils import write_jsonl

BOOK_ID = "fixture-historiography"
BOOK_TITLE = "Fixture Historiography"

# Deterministic fixture candidate set: page -> list of candidates.
CANDIDATES = [
    # page 0 frontmatter: book title
    {"cid": "cand-0000000000000001", "blk": "blk-000000000000000a", "text": "FIXTURE HISTORIOGRAPHY",
     "page_json": 0, "page_print": -12, "zone": "frontmatter", "label": "doc_title", "numbering": None,
     "before": None, "after": "Edited by A. Scholar", "ev": ["paddle:blk-000000000000000a:doc_title"]},
    # page 1: chapter 1
    {"cid": "cand-0000000000000002", "blk": "blk-000000000000000b", "text": "Chapter One: Origins",
     "page_json": 1, "page_print": 1, "zone": "mainmatter", "label": "heading", "numbering": {"token": "1", "path": ["1"], "depth": 1, "family": "chapter"},
     "before": None, "after": "The field began with disputes over method.", "ev": ["paddle:blk-000000000000000b:heading"]},
    # page 2: section under chapter 1 + running header
    {"cid": "cand-0000000000000003", "blk": "blk-000000000000000c", "text": "1.1 The German School",
     "page_json": 2, "page_print": 2, "zone": "mainmatter", "label": "heading", "numbering": {"token": "1.1", "path": ["1", "1"], "depth": 2, "family": "section"},
     "before": None, "after": "Ranke's heirs insisted on archives.", "ev": ["paddle:blk-000000000000000c:heading"]},
    {"cid": "cand-0000000000000004", "blk": "blk-000000000000000d", "text": "Origins and methods",
     "page_json": 2, "page_print": 2, "zone": "mainmatter", "label": "header", "numbering": None,
     "before": "Chapter One: Origins", "after": "1.1 The German School", "ev": ["paddle:blk-000000000000000d:header"]},
    # page 3: section 1.2 + chapter 2
    {"cid": "cand-0000000000000005", "blk": "blk-000000000000000e", "text": "1.2 The Annales Turn",
     "page_json": 3, "page_print": 3, "zone": "mainmatter", "label": "heading", "numbering": {"token": "1.2", "path": ["1", "2"], "depth": 2, "family": "section"},
     "before": None, "after": "Bloch and Febvre broadened the archive.", "ev": ["paddle:blk-000000000000000e:heading"]},
    {"cid": "cand-0000000000000006", "blk": "blk-000000000000000f", "text": "Chapter Two: Revolutions",
     "page_json": 3, "page_print": 3, "zone": "mainmatter", "label": "heading", "numbering": {"token": "2", "path": ["2"], "depth": 1, "family": "chapter"},
     "before": None, "after": "A second generation returned to politics.", "ev": ["paddle:blk-000000000000000f:heading"]},
    # page 4: caption-like block (should stay body/caption)
    {"cid": "cand-0000000000000007", "blk": "blk-0000000000000010", "text": "Figure 2.1: A revolutionary poster",
     "page_json": 4, "page_print": 4, "zone": "mainmatter", "label": "caption", "numbering": None,
     "before": "poster reproduction", "after": "Source: private collection.", "ev": ["paddle:blk-0000000000000010:caption"]},
    {"cid": "cand-0000000000000008", "blk": "blk-0000000000000011", "text": "2.1 Comparative Revolutions",
     "page_json": 4, "page_print": 4, "zone": "mainmatter", "label": "heading", "numbering": {"token": "2.1", "path": ["2", "1"], "depth": 2, "family": "section"},
     "before": None, "after": "England, France, Russia share a template.", "ev": ["paddle:blk-0000000000000011:heading"]},
    # page 5: backmatter begins
    {"cid": "cand-0000000000000009", "blk": "blk-0000000000000012", "text": "Notes",
     "page_json": 5, "page_print": 5, "zone": "backmatter", "label": "heading", "numbering": None,
     "before": None, "after": "1. On the German school see Ranke (1874).", "ev": ["paddle:blk-0000000000000012:heading"]},
    {"cid": "cand-0000000000000010", "blk": "blk-0000000000000013", "text": "Bibliography",
     "page_json": 5, "page_print": 5, "zone": "backmatter", "label": "heading", "numbering": None,
     "before": None, "after": "Anderson, B. Imagined Communities.", "ev": ["paddle:blk-0000000000000013:heading"]},
]

# page zones keyed by page index for the evidence summary
PAGE_ZONES = {0: "frontmatter", 1: "mainmatter", 2: "mainmatter", 3: "mainmatter", 4: "mainmatter", 5: "backmatter", 6: "backmatter", 7: "backmatter"}


def _candidate_row(spec):
    return {
        "candidate_id": spec["cid"],
        "source_block_ids": [spec["blk"]],
        "exact_text": spec["text"],
        "page_json": spec["page_json"],
        "page_print": spec["page_print"],
        "bbox": [0.1, 0.1, 0.8, 0.2],
        "zone": spec["zone"],
        "paddle_label": spec["label"],
        "paddle_relevel": None,
        "pdf_bookmark": None,
        "native_font": None,
        "numbering": spec["numbering"],
        "toc_matches": [],
        "repetition": {"count": 1, "nearby_count": 1},
        "before_context": spec["before"],
        "after_context": spec["after"],
        "deterministic_suppressions": [],
        "evidence_refs": list(spec["ev"]),
    }


def write_fixture_evidence(structure_dir):
    """Write the fixture evidence pack into ``structure_dir/evidence/``.

    Returns the sha256 over heading_candidates.jsonl + book_structure_evidence.json
    bytes (the staleness key the meta record carries)."""
    import hashlib

    structure_dir = Path(structure_dir)
    evidence_dir = structure_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    rows = [_candidate_row(spec) for spec in CANDIDATES]
    candidates_path = evidence_dir / "heading_candidates.jsonl"
    write_jsonl(candidates_path, rows)
    summary = {
        "schema_version": 1,
        "metadata": {
            "schema_version": 1,
            "book_id": BOOK_ID,
            "book_title": BOOK_TITLE,
            "generated_by": "chapter_recovery.evidence_v2",
            "package_version": "0.0.0-fixture",
            "python_version": "3.9.6",
            "input_sha256": "0" * 64,
            "pdf_sha256": None,
            "legacy_profile_used": False,
            "pdf_evidence_ok": False,
            "pdf_reason": "pdf_missing",
            "page_count": 8,
            "candidate_count": len(rows),
            "source_counts": {"paddle": len(rows)},
        },
        "toc_entries": [
            {"title": "Chapter One: Origins", "page_label": "", "print_page": 1, "kind": "chapter", "source_page": 1, "id": "toc-0000000000000001", "provenance": "ocr_toc"},
            {"title": "Chapter Two: Revolutions", "page_label": "", "print_page": 3, "kind": "chapter", "source_page": 3, "id": "toc-0000000000000002", "provenance": "ocr_toc"},
        ],
        "bookmark_tree": [],
        "numbering_summary": {
            "families": {"chapter": 2, "section": 3},
            "representative_tokens": {"chapter": ["1", "2"], "section": ["1.1", "1.2", "2.1"]},
        },
        "style_clusters": [{"level": 2, "count": len(rows), "sample_block_ids": [spec["blk"] for spec in CANDIDATES[:3]]}],
        "zones": {str(page): zone for page, zone in PAGE_ZONES.items()},
        "repeated_headers": [],
        "candidate_count": len(rows),
        "page_density": 0.5,
        "candidates": [{"candidate_id": spec["cid"], "exact_text": spec["text"], "page_json": spec["page_json"], "primary_sources": ["paddle"]} for spec in CANDIDATES],
    }
    summary_path = evidence_dir / "book_structure_evidence.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    pdf = {"ok": False, "reason": "pdf_missing", "pages": [], "bookmarks": [], "page_count": 0}
    (evidence_dir / "pdf_evidence.json").write_text(json.dumps(pdf) + "\n", encoding="utf-8")
    digest = hashlib.sha256()
    digest.update(candidates_path.read_bytes())
    digest.update(summary_path.read_bytes())
    return digest.hexdigest()


def write_fixture_spec(structure_dir, work_dir, tmp_path=None, **overrides):
    """Write a minimal v2 target spec; returns the spec dict."""
    structure_dir = Path(structure_dir).resolve()
    work_dir = Path(work_dir).resolve()
    chapter_config = structure_dir / "chapter_config.json"
    chapter_config.write_text(json.dumps({
        "book_id": BOOK_ID, "book_title": BOOK_TITLE,
        "input_json": str(structure_dir / "ocr.json"), "output_dir": str(structure_dir),
    }), encoding="utf-8")
    spec = {
        "schema_version": 1,
        "book_id": BOOK_ID,
        "chapter_config": str(chapter_config),
        "structure_dir": str(structure_dir),
        "work_dir": str(work_dir),
        "provider": {
            "api_url": "https://example.invalid/v1/chat/completions",
            "api_key_env": "TEST_FIXTURE_KEY",
            "model": "deepseek-v4-flash",
            "auth_header": "x-custom-auth",
            "auth_scheme": "",
            "requests_per_minute": 0,
        },
        "llm_roles": {
            "structure_global": "deepseek-v4-flash",
            "structure_local": "deepseek-v4-flash",
            "structure_review": "deepseek-v4-flash",
        },
    }
    spec.update(overrides)
    if tmp_path is not None:
        spec_path = Path(tmp_path) / "v2_target.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False) + "\n", encoding="utf-8")
        return spec_path
    return spec


class FakeClient:
    """ChatClient double with both translation (plain) and resolver (JSON) modes.

    JSON mode implements ``complete_json`` structurally: it inspects the system
    prompt to identify the pass (grammar / local window / rescue / repair) and
    the user payload for the candidate/block context, then returns a model
    instance for ``model_cls`` (pydantic-validated, like the real client) plus
    a usage dict.
    """

    model = "fake-model"

    def __init__(self, mode="plain", resolver=None, **options):
        self.mode = mode
        self.resolver = resolver
        self.options = options
        self.calls = []

    # ---- plain mode (translation tests)
    def complete(self, system, user):
        if self.mode == "json":
            raise RuntimeError("use complete_json in json mode")
        payload = json.loads(user)
        translated = [{"id": item["id"], "translated_text": "译：" + item["text"]} for item in payload["segments"]]
        return json.dumps(translated, ensure_ascii=False), {"prompt_tokens": 10, "completion_tokens": 20}

    # ---- json mode (resolver tests)
    def complete_json(self, system, user, model_cls, *, decode_retries=3):
        if self.mode != "json":
            raise RuntimeError("plain mode has no complete_json")
        self.calls.append({"system": system, "user": user, "model_cls": model_cls})
        if self.options.get("raise_runtime"):
            raise RuntimeError(self.options.get("raise_message", "provider down"))
        payload = json.loads(user) if isinstance(user, str) else user
        if "frontmatter_kinds" in system or "derive the global structure grammar" in system:
            if self.options.get("fail_on_pass") == "grammar":
                raise RuntimeError("provider down on grammar pass")
            value = self._grammar(payload)
        elif "missed" in system and "block_directory" in payload:
            if self.options.get("fail_on_pass") == "rescue":
                raise RuntimeError("provider down on rescue pass")
            value = self._rescue(payload)
        elif "repair" in system or "You repair structure decisions" in system:
            if self.options.get("fail_on_pass") == "repair":
                raise RuntimeError("provider down on repair pass")
            value = self._repair(payload)
        elif "classify OCR heading candidates" in system:
            if self.options.get("fail_on_pass") == "local":
                raise RuntimeError("provider down on local pass")
            value = self._local(payload)
        else:
            value = []
        from pydantic import TypeAdapter
        adapter = TypeAdapter(model_cls)
        return adapter.validate_python(value), {"prompt_tokens": 5, "completion_tokens": 5}

    # ------------------------------------------------------------------ JSON
    def _grammar(self, payload):
        strategy = self.options.get("grammar", "ok")
        grammar_calls = [c for c in self.calls if "derive the global structure grammar" in c["system"]]
        if strategy == "bad-once":
            if len(grammar_calls) > 1:
                return {"book_id": payload.get("book_id", BOOK_ID), "frontmatter_kinds": ["title_page"],
                        "backmatter_kinds": ["notes", "bibliography"], "numbering_families": ["chapter", "section"],
                        "chapter_family": "chapter", "section_family": "section", "notes": []}
            return {"unexpected": True}
        if strategy == "always-bad":
            return {"unexpected": True}
        return {"book_id": payload.get("book_id", BOOK_ID), "frontmatter_kinds": ["title_page"],
                "backmatter_kinds": ["notes", "bibliography"], "numbering_families": ["chapter", "section"],
                "chapter_family": "chapter", "section_family": "section", "notes": []}

    def _local(self, payload):
        strategy = self.options.get("local", "ok")
        candidates = payload["candidates"]
        chapters = {
            candidate["candidate_id"]: candidate for candidate in candidates
            if candidate.get("page_json") in {1, 3}
            and (candidate.get("exact_text") or "").upper().startswith("CHAPTER")
        }
        decisions = []
        for candidate in candidates:
            cid = candidate["candidate_id"]
            page = candidate.get("page_json")
            text = candidate.get("exact_text", "") or ""
            numbering = candidate.get("numbering")
            evidence_refs = candidate.get("evidence_refs") or ["paddle:blk-x:label"]
            if strategy == "bad" and page == 1:
                decisions.append({"candidate_id": cid, "decision": "heading", "kind": "section",
                                  "level": 3, "parent_candidate_id": None, "confidence": 0.9,
                                  "evidence_refs": evidence_refs})
            elif page == 5 and (text.upper().startswith("NOTES") or text.upper().startswith("BIBLIOGRAPHY")):
                decisions.append({"candidate_id": cid, "decision": "heading", "kind": "backmatter",
                                  "level": 2, "parent_candidate_id": None, "confidence": 0.95,
                                  "evidence_refs": evidence_refs})
            elif page in {1, 3} and (text.upper().startswith("CHAPTER") or text.upper().startswith("FIXTURE")):
                decisions.append({"candidate_id": cid, "decision": "heading", "kind": "chapter",
                                  "level": 2, "parent_candidate_id": None, "confidence": 0.95,
                                  "evidence_refs": evidence_refs})
            elif numbering and numbering.get("family") == "section":
                parent = None
                if numbering.get("path") and len(numbering.get("path", [])) >= 1:
                    chapter_token = numbering["path"][0]
                    for chapter_cid, chapter in chapters.items():
                        chapter_num = (chapter.get("numbering") or {}).get("token")
                        if str(chapter_num) == str(chapter_token):
                            parent = chapter_cid
                            break
                decisions.append({"candidate_id": cid, "decision": "heading", "kind": "section",
                                  "level": 3, "parent_candidate_id": parent, "confidence": 0.9,
                                  "evidence_refs": evidence_refs})
            elif candidate.get("paddle_label") == "caption":
                decisions.append({"candidate_id": cid, "decision": "caption", "kind": None,
                                  "level": None, "parent_candidate_id": None, "confidence": 0.85,
                                  "evidence_refs": evidence_refs})
            else:
                decisions.append({"candidate_id": cid, "decision": "body", "kind": None,
                                  "level": None, "parent_candidate_id": None, "confidence": 0.8,
                                  "evidence_refs": evidence_refs})
        return decisions

    def _rescue(self, payload):
        strategy = self.options.get("rescue", "none")
        target = self.options.get("rescue_block")
        if strategy == "one":
            blocks = payload.get("block_directory", [])
            if not blocks:
                return []
            if target is not None:
                block = next((b for b in blocks if b["block_id"] == target), None)
            else:
                block = next((b for b in blocks if b.get("paddle_label") in {"heading", "doc_title"}), None)
            block = block or blocks[0]
            return [{"block_id": block["block_id"], "reason": "missed_heading", "confidence": 0.7,
                     "evidence_refs": ["paddle:%s:header" % block["block_id"]]}]
        if strategy == "hallucinated":
            return [{"block_id": "blk-ffffffffffffffff", "reason": "missed_heading", "confidence": 0.7,
                     "evidence_refs": ["paddle:blk-ffffffffffffffff:header"]}]
        return []

    def _repair(self, payload):
        strategy = self.options.get("repair", "never")
        candidates = {c["candidate_id"]: c for c in payload.get("neighborhood_candidates", [])}
        reviewed = []
        for decision in payload.get("decisions_to_review", []):
            cid = decision["candidate_id"]
            candidate = candidates.get(cid, {})
            is_orphan_section = (
                decision.get("decision") == "heading"
                and decision.get("kind") == "section"
                and decision.get("level") == 3
                and decision.get("parent_candidate_id") is None
            )
            if strategy == "fix" and is_orphan_section:
                reviewed.append({"candidate_id": cid, "decision": "body", "kind": None, "level": None,
                                 "parent_candidate_id": None, "confidence": decision.get("confidence", 0.6),
                                 "evidence_refs": decision.get("evidence_refs") or candidate.get("evidence_refs") or ["paddle:blk-x:label"]})
            else:
                reviewed.append(dict(decision))
        return reviewed
