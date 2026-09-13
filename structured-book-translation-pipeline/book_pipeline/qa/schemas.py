"""QA cascade schema models (spec §13, P07).

``QACriticVerdict`` is the strict LLM-facing contract for the QA-4 critic:
``extra="forbid"`` so a model that invents keys (context ids, rewrite
suggestions, translations) is rejected by pydantic before it can influence any
artifact.  Per-segment flag rows are plain dicts at the orchestrator boundary
(they carry heterogeneous ``expected``/``actual`` fields), but every persisted
verdict is validated through this model so the report/audit trail can never
contain an unvalidated critic response.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QAVerdict(str, Enum):
    """QA-4 judgment vocabulary (spec §13).  Only ``LIKELY_ERROR`` may enter
    selective retranslation."""

    PASS = "pass"
    ACCEPTABLE_VARIANT = "acceptable_variant"
    LIKELY_ERROR = "likely_error"
    NEEDS_HUMAN = "needs_human"


class QACriticVerdict(BaseModel):
    """One validated critic judgment of exactly one flagged segment."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(min_length=1)
    verdict: QAVerdict
    reason: str = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("reason")
    @classmethod
    def _reason_nonempty(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("reason must not be empty")
        return text
