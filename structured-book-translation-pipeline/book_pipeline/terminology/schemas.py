from __future__ import annotations

"""Pydantic models of the terminology resolver v2 contract (plan §2/§3, spec §11).

The v2 flow is additive: deterministic candidate mining -> per-window LLM
termhood judgment -> book-level concept consolidation -> a deterministic
constraint compiler that writes ``concept_glossary.json`` (schema v2, §3.3).

Models consumed from the LLM (``TermhoodJudgment``, ``Sense``,
``ConceptRecord``, ``ConceptConsolidation``) use ``extra="forbid"`` so a model
that invents keys (free-text titles, unknown top-level fields) or hallucinated
ids is rejected by pydantic before anything reaches an artifact.  The mined
``TermCandidate`` row and the spec/progress documents are deterministic-side
models and also forbid unknown keys so schema drift fails loudly.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TERM_CANDIDATE_ID_PATTERN = r"^term-[0-9a-f]{12}$"
CONCEPT_ID_PATTERN = r"^concept-[0-9a-f]{12}$"
MASTER_SHA_PATTERN = r"^[0-9a-f]{64}$"
CONCEPT_SCHEMA_VERSION = 2

# Canonical concept categories used by prompts, the compiler clamp (D4) and
# the §11.6 human-review ordering.  Master glossary categories are mapped onto
# this set when building provenance hints (see orchestrator.MASTER_CATEGORY_MAP).
CONCEPT_CATEGORIES = {"concept", "person", "place", "org", "event", "work", "term"}
# D4: only these surface categories may hold a ``hard`` constraint without a
# master precedent; everything else proposed as ``hard`` is clamped to
# ``preferred`` by the deterministic compiler.
PROPER_NAME_CATEGORIES = {"person", "place", "org", "event", "work"}


class GlossaryConstraint(str, Enum):
    HARD = "hard"
    SENSE_CONSTRAINED = "sense_constrained"
    PREFERRED = "preferred"
    DO_NOT_TRANSLATE = "do_not_translate"
    FIRST_OCCURRENCE = "first_occurrence"


CONSTRAINT_ORIGINS = {"llm", "master", "deterministic"}


class TermCandidate(BaseModel):
    """One deterministically mined candidate row (also the
    ``term_candidates_v2.jsonl`` schema)."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=TERM_CANDIDATE_ID_PATTERN)
    surface: str
    variants: List[str] = Field(default_factory=list)
    category_hint: Optional[str] = None
    source_type: str  # one of the 8 v1 signals, or "yake"
    occurrences: int = Field(ge=1)
    evidence_segment_ids: List[str] = Field(default_factory=list)
    context: List[str] = Field(default_factory=list)  # bounded snippets (§18)
    score: float

    @field_validator("surface")
    @classmethod
    def _surface_nonempty(cls, value: Any) -> Any:
        text = str(value).strip()
        if not text:
            raise ValueError("surface must not be empty")
        return text

    @field_validator("source_type")
    @classmethod
    def _source_type_valid(cls, value: Any) -> Any:
        allowed = {
            "book_title_term", "manually_seeded_source_term", "heading_phrase",
            "heading_subphrase", "quoted_phrase", "proper_name",
            "repeated_phrase", "distinctive_word", "yake",
        }
        if value not in allowed:
            raise ValueError(f"unknown source_type {value!r}")
        return value


class TermhoodJudgment(BaseModel):
    """LLM judgment of one mined candidate (spec §11.3; ``extra="forbid"``).

    ``is_term: false`` rejects an ordinary high-frequency phrase and then
    requires ``reason``.  ``possible_senses >= 1`` drives the deterministic
    polysemy non-merge contract (§11.4): a surface with N senses maps to one
    concept with N ``senses[]`` entries, never N one-to-one concepts.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=TERM_CANDIDATE_ID_PATTERN)
    is_term: bool
    category: Optional[str] = None
    canonical_surface: Optional[str] = None
    variants: List[str] = Field(default_factory=list)
    definition_in_book: Optional[str] = None
    translation_sensitive: bool = False
    possible_senses: int = Field(default=1, ge=1)
    contexts: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def _reject_requires_reason(self) -> "TermhoodJudgment":
        if self.is_term is False and not str(self.reason or "").strip():
            raise ValueError("reason is required when is_term is false")
        if self.is_term is False and self.possible_senses != 1:
            raise ValueError("rejected ordinary phrases must report possible_senses == 1")
        if self.is_term is True and self.category:
            category = str(self.category).casefold()
            if category not in CONCEPT_CATEGORIES:
                raise ValueError(
                    f"category must be one of {sorted(CONCEPT_CATEGORIES)} "
                    f"when is_term is true, got {self.category!r}"
                )
        if self.canonical_surface is not None and not str(self.canonical_surface).strip():
            raise ValueError("canonical_surface must not be empty")
        return self


class Sense(BaseModel):
    """One sense of a concept (spec §3.3)."""

    model_config = ConfigDict(extra="forbid")

    sense_id: str = Field(pattern=r"^(default|sense-[0-9]+)$")
    translation: str
    constraint: GlossaryConstraint = GlossaryConstraint.PREFERRED
    definition_in_book: Optional[str] = None
    evidence_segments: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    version: int = Field(default=1, ge=1)

    @field_validator("translation")
    @classmethod
    def _translation_nonempty(cls, value: Any) -> Any:
        text = str(value).strip()
        if not text:
            raise ValueError("translation must not be empty")
        return text


class ConceptRecord(BaseModel):
    """One concept entry of ``concept_glossary.json`` (spec §3.3).

    ``extra="forbid"``: the consolidator LLM may only emit the fields below;
    provenance fields (``master_source``/``master_aliases``/``master_sha256``)
    are written by the deterministic compiler from the master index, never by
    the model.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = CONCEPT_SCHEMA_VERSION
    concept_id: str = Field(pattern=CONCEPT_ID_PATTERN)
    canonical_source: str
    surface_forms: List[str] = Field(min_length=1)
    category: str
    senses: List[Sense] = Field(min_length=1)
    master_source: Optional[str] = None
    master_aliases: List[str] = Field(default_factory=list)
    master_sha256: Optional[str] = Field(default=None, pattern=MASTER_SHA_PATTERN)
    constraint_origin: str = "llm"

    @field_validator("canonical_source", "category")
    @classmethod
    def _nonempty(cls, value: Any) -> Any:
        text = str(value).strip()
        if not text:
            raise ValueError("must not be empty")
        return text

    @field_validator("constraint_origin")
    @classmethod
    def _origin_valid(cls, value: Any) -> Any:
        if value not in CONSTRAINT_ORIGINS:
            raise ValueError(f"constraint_origin must be one of {sorted(CONSTRAINT_ORIGINS)}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _category_canonical(self) -> "ConceptRecord":
        category = str(self.category).casefold()
        if category not in CONCEPT_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(CONCEPT_CATEGORIES)}, got {self.category!r}"
            )
        return self


class MergeOp(BaseModel):
    """Consolidator alias op: fold ``alias_surface`` into one existing concept.

    Merges are structural (spelling variant / abbreviation / person form), not
    sense merges: same-surface polysemy must be modelled inside one concept's
    ``senses[]``, which is why merge ops carry no senses themselves.
    """

    model_config = ConfigDict(extra="forbid")

    concept_id: str = Field(pattern=CONCEPT_ID_PATTERN)
    alias_surface: str

    @field_validator("alias_surface")
    @classmethod
    def _alias_nonempty(cls, value: Any) -> Any:
        text = str(value).strip()
        if not text:
            raise ValueError("alias_surface must not be empty")
        return text


class ConceptConsolidation(BaseModel):
    """Book-level consolidator output (spec §11.4; ``extra="forbid"``)."""

    model_config = ConfigDict(extra="forbid")

    concepts: List[ConceptRecord] = Field(default_factory=list)
    merges: List[MergeOp] = Field(default_factory=list)


class TermProgress(BaseModel):
    """``term_mining_progress.json``: per-window resume state."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    book_id: str
    windows: Dict[str, str] = Field(default_factory=dict)  # window_index -> state
    consolidator_state: str = "pending"
    segments_sha256: str = ""
    term_candidates_sha256: str = ""

    @field_validator("windows")
    @classmethod
    def _window_states(cls, value: Any) -> Any:
        allowed = {"pending", "termhood_ok", "termhood_error"}
        bad = {key: state for key, state in value.items() if state not in allowed}
        if bad:
            raise ValueError(f"window states must be in {sorted(allowed)}, got {bad}")
        return value

    @field_validator("consolidator_state")
    @classmethod
    def _consolidator_state_valid(cls, value: Any) -> Any:
        allowed = {"pending", "ok", "error", "skipped_offline"}
        if value not in allowed:
            raise ValueError(f"consolidator_state must be one of {sorted(allowed)}, got {value!r}")
        return value


class TerminologySpec(BaseModel):
    """``terminology-v2 --spec`` document (plan §5 / D8).

    Deliberately decoupled from ``load_config``: the spec carries every input
    the v2 terminology flow needs (cleaned segments, optional structure JSON
    for chapter windowing, work dir, provider/role models, mining/sampling/
    wire-budget options) and forbids unknown top-level keys so a mistyped
    field fails at load time instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    book_id: str
    cleaned_segments: str
    structure_json: Optional[str] = None
    work_dir: str
    provider: Dict[str, Any] = Field(default_factory=dict)
    llm_roles: Dict[str, str] = Field(default_factory=dict)
    mining: Dict[str, Any] = Field(default_factory=dict)
    sampling: Dict[str, Any] = Field(default_factory=dict)
    wire_budget: Dict[str, int] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _spec_version(cls, value: Any) -> Any:
        if value != 1:
            raise ValueError("terminology spec schema_version must be 1")
        return value

    @field_validator("book_id", "cleaned_segments", "work_dir")
    @classmethod
    def _nonempty_str(cls, value: Any) -> Any:
        text = str(value).strip()
        if not text:
            raise ValueError("must not be empty")
        return text
