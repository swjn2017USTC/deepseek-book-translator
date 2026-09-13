from __future__ import annotations

"""Pydantic mirror of the v2 cross-repo contract (plan §3.2-§3.5, §9.3).

These models are the translation-side strict intake for LLM decisions, rescue
records, decision metadata and the Pass A grammar summary.  Decision/rescue/meta
use ``extra="forbid"`` so a model that invents free-text title fields
(``title_source``/``exact_text``/``title``/``source_text``/``new_title``/``text``)
or unknown top-level keys is rejected by pydantic before anything reaches the
chapter side.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CANDIDATE_ID_PATTERN = r"^cand-[0-9a-f]{16}$"
BLOCK_ID_PATTERN = r"^blk-[0-9a-f]{16}$"
DECISION_SCHEMA_VERSION = 2

RESCUE_REASON = "missed_heading"
REPAIR_ROUND_MAX = 2

FORBIDDEN_TEXT_KEYS = ("title", "title_source", "source_text", "new_title", "exact_text", "text")


class StructureDecision(BaseModel):
    """One v2 decision record (plan §3.2/§3.3/§3.4)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = DECISION_SCHEMA_VERSION
    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)
    decision: str
    kind: Optional[str] = None
    level: Optional[int] = None
    parent_candidate_id: Optional[str] = Field(default=None, pattern=CANDIDATE_ID_PATTERN)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(min_length=1)
    decision_model: Optional[str] = None
    repair_round: int = Field(default=0, ge=0, le=REPAIR_ROUND_MAX)
    prompts_version: Optional[int] = None

    @field_validator("decision")
    @classmethod
    def _decision_enum(cls, value: Any) -> Any:
        if value not in {"heading", "body", "caption", "apparatus", "merge"}:
            raise ValueError(f"decision must be one of heading/body/caption/apparatus/merge, got {value!r}")
        return value

    @field_validator("kind")
    @classmethod
    def _kind_enum(cls, value: Any) -> Any:
        if value is None:
            return value
        if value not in {"frontmatter", "part", "part_intro", "chapter", "section", "backmatter", "toc_entry"}:
            raise ValueError(f"kind must be one of frontmatter/part/part_intro/chapter/section/backmatter/toc_entry, got {value!r}")
        return value

    @field_validator("level")
    @classmethod
    def _level_range(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("level must be an integer")
        if value < 1 or value > 6:
            raise ValueError("level must be within 1..6")
        return value

    @model_validator(mode="after")
    def _heading_requires_kind_level(self) -> "StructureDecision":
        heading = self.decision == "heading"
        if heading and not self.kind:
            raise ValueError("kind is required when decision == 'heading'")
        if heading and self.level is None:
            raise ValueError("level is required when decision == 'heading'")
        if not heading and self.kind is not None:
            raise ValueError("kind must be null unless decision == 'heading'")
        if not heading and self.level is not None:
            raise ValueError("level must be null unless decision == 'heading'")
        if self.parent_candidate_id is not None and not self.parent_candidate_id.startswith("cand-"):
            raise ValueError("parent_candidate_id must be a cand-* id or null")
        return self


class RescueCandidate(BaseModel):
    """One v2 rescue record (plan §3.5)."""

    model_config = ConfigDict(extra="forbid")

    block_id: str = Field(pattern=BLOCK_ID_PATTERN)
    reason: str = "missed_heading"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_refs: List[str] = Field(min_length=1)
    rescue_model: Optional[str] = None
    prompts_version: Optional[int] = None

    @field_validator("reason")
    @classmethod
    def _reason_enum(cls, value: Any) -> Any:
        if value != "missed_heading":
            raise ValueError(f"reason must be 'missed_heading', got {value!r}")
        return value


class DecisionsMeta(BaseModel):
    """First line of ``structure_decisions.jsonl`` (plan §3.2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = DECISION_SCHEMA_VERSION
    book_id: str
    evidence_sha256: str
    decision_model: str
    grammar_model: str
    prompts_version: int
    generated_by: str = "book_pipeline.structure_resolver"
    generated_at: str
    repair_total: int = 0


def structure_decision_schema_version() -> int:
    return DECISION_SCHEMA_VERSION


class BookGrammar(BaseModel):
    """Pass A output: the global structure grammar of one book (plan §9.3).

    ``extra="allow"`` by design: Pass A summarizes *book-level* conventions and
    different providers may name fields differently; the orchestrator consumes
    only the fields below and ignores additional keys.  No candidate-level
    decisions or free-text heading titles may be emitted here.
    """

    model_config = ConfigDict(extra="allow")

    book_id: Optional[str] = None
    frontmatter_kinds: List[str] = Field(default_factory=list)
    backmatter_kinds: List[str] = Field(default_factory=list)
    numbering_families: List[str] = Field(default_factory=list)
    chapter_family: Optional[str] = None
    section_family: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class V2TargetSpec(BaseModel):
    """``structure-v2 --spec`` document (amendment A2)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    book_id: str
    chapter_config: str
    structure_dir: str
    work_dir: str
    chapter_root: Optional[str] = None
    page_range: Optional[List[int]] = None
    max_windows: Optional[int] = None
    provider: Dict[str, Any] = Field(default_factory=dict)
    llm_roles: Dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _spec_version(cls, value: Any) -> Any:
        if value != 1:
            raise ValueError("v2 target spec schema_version must be 1")
        return value
