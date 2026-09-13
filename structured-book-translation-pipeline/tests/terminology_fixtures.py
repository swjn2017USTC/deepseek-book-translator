"""Terminology resolver v2 test fixtures (plan §6).

Synthetic cleaned-segment corpus + optional chapter structure + a tiny master
glossary + ``TerminologyFakeClient`` in terminology-JSON mode.  The fake
dispatches on the system-prompt role markers (``term_miner`` /
``term_consolidator``) exactly like the resolver does in production, returns
pydantic-validated models through ``TypeAdapter(model_cls)``, increments
``wire_attempts`` per call (mirroring ``OpenAICompatibleClient``), and records
every call for assertions.  No test ever touches a live provider.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import TypeAdapter

from book_pipeline.glossary import _normal
from book_pipeline.io_utils import write_jsonl

BOOK_ID = "fixture-rural-history"
BOOK_TITLE = "A Rural History of Revolution"

# Deterministic mini-corpus.  Repeated surfaces are chosen so the v1 signals
# fire exactly once each on their dedicated paragraphs (boundaries respected).
FIXTURE_SEGMENTS = [
    # frontmatter-free: chapter one + body
    {"id": "seg-00000000000000000001", "kind": "heading", "source_text": "Chapter One: The Agrarian Question",
     "page_json": 1, "zone": "mainmatter", "level": 2, "parent_node_id": "chapter-1",
     "metadata": {"node_id": "chapter-1", "node_kind": "chapter"}},
    {"id": "seg-00000000000000000002", "kind": "paragraph",
     "source_text": "The agrarian question divided the socialist movement in Russia.",
     "page_json": 1, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-1", "metadata": {}},
    {"id": "seg-00000000000000000003", "kind": "paragraph",
     "source_text": "Each faction answered the agrarian question from its own reading of village life.",
     "page_json": 1, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-1", "metadata": {}},
    {"id": "seg-00000000000000000004", "kind": "paragraph",
     "source_text": "Even the most orthodox thinkers treated the agrarian question as the pivot of strategy.",
     "page_json": 2, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-1", "metadata": {}},
    {"id": "seg-00000000000000000005", "kind": "paragraph",
     "source_text": "Debates on the agrarian question echoed through every party congress of the era.",
     "page_json": 2, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-1", "metadata": {}},
    {"id": "seg-00000000000000000006", "kind": "paragraph",
     "source_text": "One critic spoke of the \u201ciron cage\u201d of the market and the shrinking rural world.",
     "page_json": 2, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-1", "metadata": {}},
    # chapter two + body (mode of production x4 -> repeated phrase; masters hit)
    {"id": "seg-00000000000000000007", "kind": "heading", "source_text": "Chapter Two: Rural Class Formation",
     "page_json": 3, "zone": "mainmatter", "level": 2, "parent_node_id": "chapter-2",
     "metadata": {"node_id": "chapter-2", "node_kind": "chapter"}},
    {"id": "seg-00000000000000000008", "kind": "paragraph",
     "source_text": "Karl Marx analysed the mode of production in England, not in Russia.",
     "page_json": 3, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000009", "kind": "paragraph",
     "source_text": "The mode of production debate shaped how historians read the rural commune.",
     "page_json": 3, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000010", "kind": "paragraph",
     "source_text": "Under tsarism the mode of production stayed agrarian while the state industrialised.",
     "page_json": 4, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000011", "kind": "paragraph",
     "source_text": "Every comparison of the mode of production returned to the rural commune and its fate.",
     "page_json": 4, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000012", "kind": "paragraph",
     "source_text": "A counter-revolutionary wave followed the revolution across the countryside.",
     "page_json": 4, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000013", "kind": "paragraph",
     "source_text": "The vanguard party claimed the workers' mandate in the name of the soviets.",
     "page_json": 5, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000014", "kind": "paragraph",
     "source_text": "Vera Zasulich asked whether the rural commune could leap toward socialism.",
     "page_json": 5, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    # boundary probe: the master term "civil society" must never match the
    # non-word "incivil society" (lookbehind boundary of _phrase_pattern)
    {"id": "seg-00000000000000000015", "kind": "paragraph",
     "source_text": "Some historians speak of incivil society when describing the police state.",
     "page_json": 5, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
    {"id": "seg-00000000000000000016", "kind": "paragraph",
     "source_text": "Others insist that civil society grew quietly inside the zemstvo network; the rural commune reappears in every chapter of the debate.",
     "page_json": 6, "zone": "mainmatter", "level": None, "parent_node_id": "chapter-2", "metadata": {}},
]

# A proper seed corpus for the deterministic-miner unit (extra segments).
SEED_SURFACES = ["vanguard party"]
MINING_DEFAULTS = {
    "book_title": BOOK_TITLE,
    "seed_terms": SEED_SURFACES,
    "candidate_limit": 0,
    "use_yake": False,
}

MASTER_ENTRIES = [
    {"term": "mode of production", "translation": "生产方式", "category": "concept",
     "source_aliases": ["production mode"], "alternatives": ["生产方式"]},
    {"term": "civil society", "translation": "市民社会", "category": "concept",
     "source_aliases": [], "alternatives": ["公民社会"]},
    {"term": "Karl Marx", "translation": "卡尔·马克思", "category": "person",
     "source_aliases": ["Marx"]},
]

KNOWN_TERMS = [
    {"surface": "agrarian question", "category": "concept"},
    {"surface": "rural commune", "category": "concept"},
    {"surface": "counter-revolutionary", "category": "term"},
]


def _structure_nodes() -> List[Dict[str, Any]]:
    return [
        {"id": "book-1", "kind": "book", "ignored": False, "zone": "book", "level": 1},
        {"id": "chapter-1", "kind": "chapter", "level": 2, "zone": "mainmatter",
         "title_source": "Chapter One: The Agrarian Question", "parent_id": "book-1", "page_json": 1},
        {"id": "chapter-2", "kind": "chapter", "level": 2, "zone": "mainmatter",
         "title_source": "Chapter Two: Rural Class Formation", "parent_id": "book-1", "page_json": 3},
    ]


def write_terminology_fixture(
    tmp_path,
    *,
    segments: Optional[List[Dict[str, Any]]] = None,
    master_entries: Optional[List[Dict[str, Any]]] = None,
    with_structure: bool = True,
    book_id: str = BOOK_ID,
) -> Dict[str, Path]:
    """Write cleaned segments + optional structure + master under tmp_path."""
    root = Path(tmp_path)
    cleaned = root / "cleaned_segments.jsonl"
    write_jsonl(cleaned, segments if segments is not None else FIXTURE_SEGMENTS)
    structure_path = root / "book_structure.json"
    if with_structure:
        structure_path.write_text(json.dumps(
            {"schema_version": 3, "book_id": book_id, "nodes": _structure_nodes()},
            ensure_ascii=False,
        ) + "\n", encoding="utf-8")
    master_path = root / "master.json"
    master_path.write_text(json.dumps(
        master_entries if master_entries is not None else MASTER_ENTRIES,
        ensure_ascii=False,
    ), encoding="utf-8")
    return {"cleaned": cleaned, "structure": structure_path if with_structure else None,
            "master": master_path}


def make_terminology_spec(
    tmp_path,
    *,
    work_dir: Optional[Path] = None,
    known_terms: Optional[List[Dict[str, Any]]] = None,
    with_structure: bool = True,
    master_json: bool = True,
    **overrides,
) -> Dict[str, Any]:
    """Write a fixture and return a terminology spec dict (not a path)."""
    paths = write_terminology_fixture(tmp_path, with_structure=with_structure)
    work = Path(work_dir) if work_dir is not None else Path(tmp_path) / "work"
    spec = {
        "schema_version": 1,
        "book_id": BOOK_ID,
        "cleaned_segments": str(paths["cleaned"]),
        "structure_json": str(paths["structure"]) if with_structure and paths["structure"] else None,
        "work_dir": str(work),
        "provider": {},
        "llm_roles": {},
        "mining": dict(MINING_DEFAULTS),
        "sampling": {"known_terms": known_terms if known_terms is not None else KNOWN_TERMS},
        "wire_budget": {"term_miner": 25, "term_consolidator": 4},
    }
    if master_json:
        spec["mining"]["master_json"] = str(paths["master"])
    for key, value in overrides.items():
        if key in {"mining", "sampling", "provider", "llm_roles", "wire_budget"}:
            spec[key].update(value)
        else:
            spec[key] = value
    return spec


def write_spec_file(spec: Dict[str, Any], tmp_path) -> Path:
    path = Path(tmp_path) / "terminology_v2_spec.json"
    path.write_text(json.dumps(spec, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


class TerminologyFakeClient:
    """ChatClient double for the terminology resolver (terminology JSON mode).

    Dispatches on system-prompt role markers exactly like the production
    client flow: ``term_miner`` windows -> ``TermhoodJudgment`` list,
    ``term_consolidator`` -> ``ConceptConsolidation``.  Results are validated
    through ``TypeAdapter(model_cls)`` so a malformed option payload raises
    pydantic ``ValidationError`` exactly like a malformed provider response.

    Options (all normalized-surface keyed, unless noted):
    * reject_surfaces — is_term:false with a reason (ordinary phrase)
    * low_confidence_surfaces -> float confidence (< 0.5 => needs_human verdict)
    * polysemy_surfaces -> int sense count (one concept, N senses)
    * person_surfaces — judged category "person"
    * hallucinate_id — termhood returns an invented term-* candidate id
    * fail_on_window — raise RuntimeError("provider down") for that window
    * fail_consolidator — raise during consolidation
    * wire_per_call — wire_attempts per complete_json (default 1)
    * sense_constraints — {surface: constraint value} the consolidator proposes
      (default "preferred"); "hard" proposals exercise the compiler clamp (D4)
    * master_conflict_surfaces — consolidator picks a translation different
      from the master hint
    * duplicate_concepts — consolidator emits two concepts with the same
      canonical source (deterministic merge-error scenario)
    * invalid_json — termhood returns something pydantic rejects
    """

    model = "fake-terminology-model"

    def __init__(self, mode: str = "terminology", options: Optional[Dict[str, Any]] = None):
        self.mode = mode
        self.options = dict(options or {})
        self.calls: List[Dict[str, Any]] = []
        self.wire_attempts = 0
        self.wire_per_call = max(int(self.options.get("wire_per_call", 1)), 1)

    # -- ChatClient protocol ------------------------------------------------
    def complete(self, system: str, user: str):
        if self.mode == "terminology":
            raise RuntimeError("use complete_json in terminology mode")
        return json.dumps([]), {"prompt_tokens": 1, "completion_tokens": 1}

    def complete_json(self, system, user, model_cls, *, decode_retries=3):
        self.wire_attempts += self.wire_per_call
        payload = json.loads(user) if isinstance(user, str) else user
        self.calls.append({"system": system, "user": payload, "model_cls": model_cls})
        if "term_miner" in system and "term_consolidator" not in system:
            return self._termhood(payload, model_cls)
        if "term_consolidator" in system:
            return self._consolidation(payload, model_cls)
        raise RuntimeError("unknown terminology role prompt")

    # -- builders -------------------------------------------------------------
    def _surface(self, text: Any) -> str:
        return _normal(str(text or ""))

    @staticmethod
    def _join(norm: str) -> str:
        # v1's WORD tokenizer drops 2-letter tokens, so mined surfaces read
        # "mode production" where the book says "mode of production".  Option
        # surfaces are matched on either normalized form.
        return " ".join(word for word in norm.split() if len(word) > 2)

    def _match_any(self, candidates, surface: str) -> bool:
        norm = self._surface(surface)
        join = self._join(norm)
        for candidate in candidates:
            other = self._surface(candidate)
            if other == norm or self._join(other) == join:
                return True
        return False

    def _match_value(self, mapping, surface: str, default=None):
        norm = self._surface(surface)
        join = self._join(norm)
        for raw, value in mapping.items():
            other = self._surface(raw)
            if other == norm or self._join(other) == join:
                return value
        return default

    def _termhood(self, payload, model_cls):
        if self.options.get("fail_on_window") is not None \
                and str(payload.get("window_index")) == str(self.options["fail_on_window"]):
            raise RuntimeError(f"provider down on window {payload.get('window_index')}")
        reject = list(self.options.get("reject_surfaces", []))
        low_conf = dict(self.options.get("low_confidence_surfaces") or {})
        polysemy = dict(self.options.get("polysemy_surfaces") or {})
        person = list(self.options.get("person_surfaces", []))
        hallucinated = self.options.get("hallucinate_id")
        out = []
        for candidate in payload.get("candidates", []):
            cid = candidate["candidate_id"]
            if hallucinated:
                cid = "term-" + "f" * 12  # schema-valid, not in any window
            surface = str(candidate.get("surface") or "")
            is_term = not self._match_any(reject, surface)
            confidence = float(self._match_value(low_conf, surface, 0.9))
            sense_count = int(self._match_value(polysemy, surface, 1))
            judgment = {
                "candidate_id": cid,
                "is_term": is_term,
                "category": "person" if self._match_any(person, surface) else "concept",
                "canonical_surface": surface,
                "translation_sensitive": is_term,
                "possible_senses": sense_count,
                "confidence": confidence,
            }
            if not is_term:
                judgment["reason"] = "ordinary high-frequency phrase, not a term of this book"
            out.append(judgment)
        return TypeAdapter(model_cls).validate_python(out), {
            "prompt_tokens": 5, "completion_tokens": 5}

    def _consolidation(self, payload, model_cls):
        if self.options.get("fail_consolidator"):
            raise RuntimeError("provider down on consolidation")
        constraints = dict(self.options.get("sense_constraints") or {})
        conflict = list(self.options.get("master_conflict_surfaces") or [])
        polysemy = dict(self.options.get("polysemy_surfaces") or {})
        duplicate = self.options.get("duplicate_concepts")
        concepts = []
        index = 0
        for summary in payload.get("candidate_summaries", []):
            index += 1
            surface = str(summary.get("canonical_surface")
                          or summary.get("surface") or "")
            master = summary.get("master") or {}
            constraint = str(self._match_value(constraints, surface, "preferred"))
            sense_count = int(self._match_value(polysemy, surface, 1))
            senses = []
            for sense_number in range(1, sense_count + 1):
                sense_id = "default" if sense_number == 1 else f"sense-{sense_number}"
                if self._match_any(conflict, surface) and master.get("translation"):
                    translation = "DIFFERENT:" + str(master.get("translation") or "")
                elif master.get("translation") and sense_count == 1:
                    # Well-behaved consolidation: follow the master glossary.
                    translation = str(master.get("translation") or "")
                else:
                    translation = f"译{sense_number}:{surface}"
                senses.append({
                    "sense_id": sense_id,
                    "translation": translation,
                    "constraint": constraint,
                    "evidence_segments": list(summary.get("evidence_segment_ids") or [])[:2],
                    "confidence": 0.9,
                    "version": 1,
                })
            row = {
                "concept_id": f"concept-{index:012x}",
                "canonical_source": surface,
                "surface_forms": [surface],
                "category": str(summary.get("category") or "concept"),
                "senses": senses,
                "master_source": master.get("translation"),
                "master_aliases": list(master.get("aliases") or []),
            }
            concepts.append(row)
            if duplicate:
                concepts.append({
                    "concept_id": f"concept-{index + 1000:012x}",
                    "canonical_source": surface,
                    "surface_forms": [surface],
                    "category": str(summary.get("category") or "concept"),
                    "senses": [dict(senses[0])],
                    "master_source": master.get("translation"),
                    "master_aliases": list(master.get("aliases") or []),
                })
        return TypeAdapter(model_cls).validate_python(
            {"concepts": concepts, "merges": []}
        ), {"prompt_tokens": 5, "completion_tokens": 5}


def concept_id_for(surface: str) -> str:
    """Deterministic concept id the compiler derives from a canonical source."""
    return "concept-" + hashlib.sha256(_normal(surface).encode("utf-8")).hexdigest()[:12]
