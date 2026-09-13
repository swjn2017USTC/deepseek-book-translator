"""Contextual translation v2 (spec §12): single-target, context-aware
translation with chapter briefs + bounded source/translation windows and
per-segment concept-dependency cache invalidation.

Purely additive package: v1 ``translate.py`` / ``models.py`` / ``glossary.py``
and the chapter repo are untouched.  The CLI command is ``translate-v2``
(``--config``), which requires ``translation.context_mode == "contextual_v2"``;
v1 books keep using ``translate --config``.
"""

from .ladder import LADDER_STEPS, provider_probe, run_ladder
from .orchestrator import require_contextual_mode, translate_v2
from .invalidation import is_translation_current_v2
from .briefs import build_chapter_briefs
from .deps import resolve_concept_deps
from .parser import parse_target_only

__all__ = [
    "LADDER_STEPS",
    "provider_probe",
    "run_ladder",
    "translate_v2",
    "require_contextual_mode",
    "is_translation_current_v2",
    "build_chapter_briefs",
    "resolve_concept_deps",
    "parse_target_only",
]
