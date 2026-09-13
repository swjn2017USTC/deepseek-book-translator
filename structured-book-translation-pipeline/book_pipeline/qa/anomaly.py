"""QA-3 anomaly detection (spec §13, plan §3.3).

Thresholds are parameterized through ``qa.settings.QaSettings`` (plan §D5
defaults): ``max_len_ratio`` 0.5, hard short-cutoff 8/64,
``max_residual_run`` 8.  Every anomaly is a FLAG feeding QA-4 + the flagged
list; none blocks rendering (amendment A1 severity model: ``empty`` and
``footnote_missing`` carry ``high`` — parse-gate conditions on stored rows —
everything else ``flag``).

Residual source-language detection uses only the stdlib: a token is
source-script when every character is a Latin letter (en -> zh corpora);
the check flags runs of more than ``max_residual_run`` consecutive
source-script tokens.  No Lingua/langdetect dependency (MacBook-Air friendly).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from ..models import Segment
from .deterministic import FOOTNOTE_REF
from .settings import QaSettings

# A source-script word: Latin letters with optional internal apostrophes /
# hyphens.  FP intent: legitimately kept English citations / proper nouns /
# DO_NOT_TRANSLATE terms read as residual source runs.
_SOURCE_WORD = re.compile(r"^[A-Za-z]+(?:['’\-][A-Za-z]+)*$")

# English sentence-initial tokens that would otherwise glue capitalized
# multi-word runs into fake proper nouns ("The Chinese Communist Party" ->
# run starts at "Chinese").  FP intent documented: mid-run capitalized
# conjunctions inside real names break the span, at worst splitting one flag
# into fewer/different spans — never fabricating a name.
_SENTENCE_STARTERS = {
    "The", "In", "On", "At", "By", "For", "As", "A", "An", "To", "From",
    "But", "And", "This", "That", "These", "Those", "It", "He", "She",
    "They", "We", "You", "When", "While", "Since", "After", "Before",
    "With", "However", "Moreover", "Yet", "During", "Within", "Without",
    "Between", "Among", "Against", "Through", "Because", "Although", "If",
    "Then", "There", "Here", "What", "Why", "How", "Only", "Also", "Even",
    "Still", "Just", "Indeed", "Clearly", "Perhaps", "One", "Two", "Both",
    "Each", "Most", "Many", "Much", "Little", "Few", "No", "Not", "More",
    "New", "Out", "Up", "Down", "Over", "Under", "Next", "Last", "Today",
    "Recently", "Earlier", "Later", "Finally", "Meanwhile", "Soon", "Now",
}

_CAPITALIZED = re.compile(r"^[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]+$")
_SENTENCE_SPLIT = re.compile(r"[。！？!?；;]+")


def _residual_run_length(target: str) -> int:
    """Longest run of consecutive source-script (Latin-letter) words."""
    longest = 0
    current = 0
    for token in target.split():
        if _SOURCE_WORD.match(token):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _capitalized_runs(source: str) -> List[str]:
    """Maximal runs of >= 2 capitalized words that are not led by a
    sentence-initial token (proper-noun candidates)."""
    tokens = source.split()
    runs: List[str] = []
    current: List[str] = []
    for token in tokens:
        word = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ’'\-]+$", "", token)
        if _CAPITALIZED.match(word) and word not in _SENTENCE_STARTERS:
            current.append(word)
        else:
            if len(current) >= 2:
                runs.append(" ".join(current))
            current = []
    if len(current) >= 2:
        runs.append(" ".join(current))
    return runs


def _span_survives(span: str, target: str) -> bool:
    """True when the whole span or any 2-word window of it appears verbatim."""
    if span in target:
        return True
    words = span.split()
    return any(
        " ".join(words[index:index + 2]) in target
        for index in range(len(words) - 1)
    )


def anomalies_for(
    segment: Segment,
    target: str,
    settings: QaSettings,
) -> List[Dict[str, Any]]:
    """Per-segment QA-3 anomalies (footnote_missing surfaced only when no
    QA-1 flag already carries the drop — see orchestrator dedupe)."""
    source = segment.source_text
    flags: List[Dict[str, Any]] = []

    def flag(check: str, severity: str, detail: str) -> None:
        flags.append({
            "segment_id": segment.id,
            "stage": "qa-3",
            "check": check,
            "severity": severity,
            "expected": 1,
            "actual": 0,
            "detail": detail,
        })

    # empty translation (belt-and-suspenders — the parsers already reject)
    if not target.strip():
        flag(
            "empty",
            "high",
            f"empty translation for source {len(source)} chars",
        )
        return flags

    # length ratio
    if source and len(target) / len(source) < settings.max_len_ratio:
        flag(
            "length_ratio",
            "flag",
            f"length ratio {len(target)}/{len(source)} = "
            f"{len(target) / len(source):.3f} < {settings.max_len_ratio}",
        )

    # suspiciously short
    if len(target) < 8 and len(source) > 64:
        flag(
            "short",
            "flag",
            f"target {len(target)} chars for source {len(source)} chars",
        )

    # residual source-language run
    run = _residual_run_length(target)
    if run > settings.max_residual_run:
        flag(
            "residual",
            "flag",
            f"residual {run} consecutive source-script words "
            f"(> {settings.max_residual_run})",
        )

    # disappeared proper noun (capitalized multi-word span)
    if source and target:
        missing = [
            span for span in _capitalized_runs(source)
            if not _span_survives(span, target)
        ]
        if missing:
            flag(
                "proper_noun",
                "flag",
                "capitalized multi-word span absent from target: "
                + "; ".join(missing[:5]),
            )

    # repeated sentence loop within one segment
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(target)
        if len(sentence.strip()) >= 4
    ]
    normalized = [re.sub(r"\s+", " ", sentence).casefold() for sentence in sentences]
    seen: Dict[str, int] = {}
    repeated: List[str] = []
    for sentence in normalized:
        seen[sentence] = seen.get(sentence, 0) + 1
        if seen[sentence] == 3:
            repeated.append(sentence)
    if repeated:
        flag(
            "repeated_sentence",
            "flag",
            f"same target sentence repeated >= 3 times: {repeated[:3]}",
        )

    return flags


def duplicate_anomalies(
    audited: Sequence[Tuple[Segment, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Book-level duplicate check: identical ``translated_text`` on more than
    two distinct segment ids (>2 repeats -> >= 3 rows).  FP intent: repeated
    short boilerplate (identical captions/notes) trips the threshold; the
    QA-4 critic absorbs legitimate repeats."""
    by_text: Dict[str, List[Tuple[Segment, Dict[str, Any]]]] = {}
    for segment, row in audited:
        text = str(row.get("translated_text") or "")
        if not text.strip():
            continue
        by_text.setdefault(text, []).append((segment, row))
    flags: List[Dict[str, Any]] = []
    for text, members in by_text.items():
        if len(members) <= 2:
            continue
        detail_ids = ", ".join(segment.id for segment, _ in members)
        for segment, _ in members:
            flags.append({
                "segment_id": segment.id,
                "stage": "qa-3",
                "check": "duplicate",
                "severity": "flag",
                "expected": 1,
                "actual": len(members),
                "detail": (
                    f"identical translated_text on {len(members)} segments "
                    f"({detail_ids}); text starts {text[:80]!r}"
                ),
            })
    return flags


def footnote_missing_flags(
    audited: Sequence[Tuple[Segment, Dict[str, Any]]],
    qa1_checks: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Footnote-reference drops surfaced as QA-3 anomalies (plan §3.3).

    ``qa1_checks`` maps ``segment_id -> QA-1 check names already flagged``;
    when QA-1 already carried the drop (its ``footnotes``/``structural``
    class), the cascade reports the defect once at QA-1 and this function
    stays silent so the planted matrix keeps one deterministic verdict per
    defect.  Standalone ``--only qa-3`` runs (no QA-1) surface these rows.
    """
    flags: List[Dict[str, Any]] = []
    for segment, row in audited:
        if segment.id in qa1_checks:
            continue
        target = str(row.get("translated_text") or "")
        source_refs = len(FOOTNOTE_REF.findall(segment.source_text))
        target_refs = len(FOOTNOTE_REF.findall(target))
        if source_refs > target_refs:
            flags.append({
                "segment_id": segment.id,
                "stage": "qa-3",
                "check": "footnote_missing",
                "severity": "high",
                "expected": source_refs,
                "actual": target_refs,
                "detail": (
                    f"footnote refs {source_refs} in source, {target_refs} in "
                    "target"
                ),
            })
    return flags
