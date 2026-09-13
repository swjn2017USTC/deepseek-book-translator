"""Terminology resolver v2 (spec §11): high-recall deterministic candidate
mining -> per-window LLM termhood judgment -> book-level concept consolidation
-> deterministic constraint compiler -> ``concept_glossary.json`` (schema v2).

Additive package: v1 ``glossary.py``, ``clean.py``, translation-cache semantics
and chapter-repo code are untouched; this package reads them but never writes
through them.
"""

from .orchestrator import (
    ProviderUnavailable,
    load_terminology_spec,
    run_terminology_v2,
)

__all__ = ["ProviderUnavailable", "load_terminology_spec", "run_terminology_v2"]
