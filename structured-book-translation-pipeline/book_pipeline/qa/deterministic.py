"""QA-1 deterministic count-parity invariants (spec §13, plan §3.1/§3.4).

Every check computes a per-class count over the source and the translated text
and flags ``count(source) != count(target)`` — count parity, never value
equality (page numbers and digits legitimately reflow across sentence
boundaries during translation).  URLs use **set parity** (each distinct source
URL must appear verbatim in the target) because URLs are byte-sensitive and
never legitimately translated.  Severity per amendment A1: every QA-1
mismatch is a FLAG that feeds QA-4 + the flagged list — deterministic QA-1
never blocks rendering by itself; ``severity`` distinguishes classes whose
loss indicates stored-text corruption (footnote refs / URLs / formulas /
comments / table delimiters are byte-preservation rules of this repo's
translation contract -> ``high``) from digit-classes that may legitimately
reflow (``flag``).

False-positive intent is documented per regex: the classes are deliberately
broad (e.g. uppercase Roman tokens include the pronoun ``I``; ``numerals``
overlaps ``years`` and ``percentages``) — a legitimately translated-away
occurrence produces a flag the QA-4 critic is expected to absorb as
``acceptable_variant``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from ..models import Segment
from ..translate import STRUCTURAL_TOKEN  # read-only regex reuse; v1 untouched


# --------------------------------------------------------------------------
# Module-level check regexes (FP intent in each docstring)
# --------------------------------------------------------------------------

# Inline math like ``$^1$`` / ``$^{12}$`` — the same formula token the v1/v2
# parsers enforce.  FP intent: superscript-style dollar spans in prose.
FORMULA = re.compile(r"\$\s*\^\{?\d+\}?\s*\$")

# Footnote reference markers ``[^fn-0013-1]`` (not bodies).  FP intent: none
# in practice — markers are structural and must survive byte-identical.
FOOTNOTE_REF = re.compile(r"\[\^[^\]]+\]")

# URL with scheme.  FP intent: trailing-sentence punctuation is normalized off
# before comparison, matching translate.parse_translation's normalize step.
URL = re.compile(r"https?://\S+")

# Parenthetical citation containing a 19xx/20xx year, e.g. "(Vogel 2011)".
# FP intent: any parenthetical date phrase ("(the 1980s reforms)") matches;
# the check counts occurrences, it does not judge citation validity.
CITATION_PAREN_YEAR = re.compile(r"\([^)]*\b(?:19|20)\d{2}\b[^)]*\)")

# Bare bracketed page/pager references, e.g. "[14]" / "[12, 34]" / "[5; 7]".
# FP intent: any bracketed numeral (list indices, note numbers) matches.
CITATION_BRACKETED = re.compile(r"\[\d+(?:[,;]\s*\d+)*\]")

# 4-digit years and era-suffixed years ("1990", "2020", "500 BCE", "44 BC").
# Digit runs use digit-run boundaries, not ``\b``: Chinese characters are
# Unicode word chars, so ``\b`` would miss "1978年" (no word boundary between
# the digit and 年).  FP intent: IDs/catalogue numbers containing 4 digits
# also match.
YEARS = re.compile(r"(?<![\d])(?:19|20)\d{2}(?![\d])|\b\d{1,4}\s*(?:BCE?|CE)\b")

# Number tokens with optional thousand/decimal groups ("12", "1,234", "3.5").
# Digit-run boundaries keep CJK-adjacent numbers ("到1978年") countable; FP
# intent: any digit run matches; legitimately reflowed digit boundaries
# inside a sentence trip parity even when the value survived.
NUMERALS = re.compile(r"(?<![\d])\d+(?:[.,]\d+)*(?![\d])")

# Percentage tokens ("12%" / "12.5%").  FP intent: overlap with NUMERALS is
# intentional — a dropped percent sign is its own defect class.
PERCENTAGES = re.compile(r"(?<![\d])\d+(?:\.\d+)?\s*%")

# Uppercase Roman numeral tokens as whole words ("I", "IV", "XIV", "MMXXIV",
# "CDXLIV").  Letter boundaries (not ``\b``) so CJK-adjacent numerals
# ("II军团") stay countable.  A single all-optional ``(?:M{0,3})...`` pattern
# cannot be used with ``findall`` (zero-length matches at every word
# boundary), so the class is a token regex plus an exact full-token
# validator.  FP intent: standalone English words spelled with Roman letters
# ("I" as a pronoun) match; intended signal is heading and volume numerals
# kept verbatim in the translated text.
ROMAN_TOKEN = re.compile(r"(?<![A-Za-z])[IVXLCDM]+(?![A-Za-z])")
_ROMAN_VALIDATOR = re.compile(
    r"^(?:M{0,3})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$"
)


def roman_matches(text: str) -> List[str]:
    """Whole-word uppercase Roman numeral tokens (no zero-length matches)."""
    return [
        token for token in ROMAN_TOKEN.findall(text)
        if _ROMAN_VALIDATOR.match(token)
    ]

# Markdown table delimiter.  Count parity matters only for table-kind
# segments (same rule as translate.parse_translation).
TABLE_PIPE = re.compile(r"\|")

# HTML comments — part of STRUCTURAL_TOKEN with no dedicated sub-class; a
# mismatch that no other class explains is attributed to this one.
HTML_COMMENT = re.compile(r"<!--.*?-->")

# Checks whose mismatch indicates byte-preservation loss rather than a
# plausibly legitimate reflow: severity "high" (still only a flag — QA-1
# never gates by itself, amendment A1).
_HIGH_SEVERITY_CLASSES = {"structural", "footnotes", "urls", "formulas", "table_delimiters"}


def severity_for(check: str) -> str:
    return "high" if check in _HIGH_SEVERITY_CLASSES else "flag"


def load_gate(
    segments: Sequence[Segment],
    completed: Dict[str, Dict[str, Any]],
) -> None:
    """Structural load gate for the QA audit (plan §3.4): unique segment ids
    and every completed translation row bound to a real segment.

    Raises ``ValueError`` (never a flag — this is input corruption, not a
    translatable defect) on duplicate segment ids or on completed rows whose
    ``segment_id`` does not exist among the cleaned segments.  Partial
    translation coverage is normal (``--limit`` runs) and is not an error;
    the audit universe is the intersection.
    """
    seen: Dict[str, int] = {}
    for segment in segments:
        seen[segment.id] = seen.get(segment.id, 0) + 1
        if seen[segment.id] > 1:
            raise ValueError(f"Duplicate segment id in cleaned segments: {segment.id}")
    for segment_id, row in completed.items():
        if row.get("status") != "completed":
            continue
        if segment_id not in seen:
            raise ValueError(
                f"Translation row {segment_id!r} references an unknown cleaned segment; "
                "rerun clean before qa"
            )


def audit_rows(
    segments: Sequence[Segment],
    completed: Dict[str, Dict[str, Any]],
) -> List[Tuple[Segment, Dict[str, Any]]]:
    """Audit universe: translatable segments with a *current* completed row
    (source_hash match; last-write-wins dedup already applied by
    ``translate._completed``).  Untranslated segments are out of scope for the
    cascade — they have no target text to audit."""
    rows: List[Tuple[Segment, Dict[str, Any]]] = []
    for segment in segments:
        if not segment.translatable:
            continue
        row = completed.get(segment.id)
        if not row or row.get("status") != "completed":
            continue
        if row.get("source_hash") != segment.source_hash:
            continue  # stale row (source edited since) — pending, not audited
        rows.append((segment, row))
    return rows


def check_segment(segment: Segment, translated_text: str) -> List[Dict[str, Any]]:
    """Run every QA-1 class over one audited segment.

    Returns flag rows ``{"segment_id", "stage": "qa-1", "check", "severity",
    "expected", "actual", "detail"}``.  The STRUCTURAL_TOKEN family emits at
    most one row per segment, attributed to the first differing specific class
    (footnotes/urls/formulas) or ``structural`` when only HTML comments
    drifted; ``detail`` names every differing class with its counts.
    """
    source = segment.source_text
    flags: List[Dict[str, Any]] = []

    def mismatch(
        check: str,
        expected: int,
        actual: int,
        detail: str,
    ) -> None:
        flags.append({
            "segment_id": segment.id,
            "stage": "qa-1",
            "check": check,
            "severity": severity_for(check),
            "expected": expected,
            "actual": actual,
            "detail": detail,
        })

    # --- STRUCTURAL_TOKEN family (footnotes | comments | urls | formulas) ---
    source_family = STRUCTURAL_TOKEN.findall(source)
    target_family = STRUCTURAL_TOKEN.findall(translated_text)
    if len(source_family) != len(target_family):
        differing: List[str] = []
        for name, pattern in (
            ("footnotes", FOOTNOTE_REF),
            ("urls", URL),
            ("formulas", FORMULA),
        ):
            if len(pattern.findall(source)) != len(pattern.findall(translated_text)):
                differing.append(name)
        comments_drift = len(HTML_COMMENT.findall(source)) != len(HTML_COMMENT.findall(translated_text))
        check = differing[0] if differing else "structural"
        if not differing and comments_drift:
            check = "structural"
        extra = "; HTML comments drifted" if comments_drift and check == "structural" else ""
        mismatch(
            check,
            len(source_family),
            len(target_family),
            "structural token family mismatch: "
            + (", ".join(differing) + " counts differ" if differing else "HTML comments differ")
            + f" (structural {len(source_family)} vs {len(target_family)})" + extra,
        )

    # --- URL set parity: every distinct source URL must appear verbatim ---
    def _urls(text: str) -> List[str]:
        return [token.rstrip(".,;:!?。；！？") for token in URL.findall(text)]

    source_urls = _urls(source)
    target_urls = _urls(translated_text)
    if (
        set(source_urls) != set(target_urls)
        and not any(flag["check"] == "urls" for flag in flags)
    ):
        missing = sorted(set(source_urls) - set(target_urls))
        unexpected = sorted(set(target_urls) - set(source_urls))
        mismatch(
            "urls",
            len(source_urls),
            len(target_urls),
            f"URL set parity: missing={missing or 'none'} unexpected={unexpected or 'none'}",
        )

    # --- footnote refs count parity ---
    source_count = len(FOOTNOTE_REF.findall(source))
    target_count = len(FOOTNOTE_REF.findall(translated_text))
    if source_count != target_count and not any(
        flag["check"] == "footnotes" for flag in flags
    ):
        mismatch(
            "footnotes",
            source_count,
            target_count,
            f"footnote refs {source_count} vs {target_count}",
        )

    # --- formulas count parity ---
    source_count = len(FORMULA.findall(source))
    target_count = len(FORMULA.findall(translated_text))
    if source_count != target_count and not any(
        flag["check"] == "formulas" for flag in flags
    ):
        mismatch(
            "formulas",
            source_count,
            target_count,
            f"formula tokens {source_count} vs {target_count}",
        )

    # --- citation markers (parenthetical-year or bracketed) ---
    source_count = len(CITATION_PAREN_YEAR.findall(source)) + len(CITATION_BRACKETED.findall(source))
    target_count = len(CITATION_PAREN_YEAR.findall(translated_text)) + len(CITATION_BRACKETED.findall(translated_text))
    if source_count != target_count:
        mismatch(
            "citation_markers",
            source_count,
            target_count,
            f"citation markers {source_count} vs {target_count}",
        )

    # --- years ---
    source_count = len(YEARS.findall(source))
    target_count = len(YEARS.findall(translated_text))
    if source_count != target_count:
        mismatch(
            "years",
            source_count,
            target_count,
            f"year tokens {source_count} vs {target_count}",
        )

    # --- numerals ---
    source_count = len(NUMERALS.findall(source))
    target_count = len(NUMERALS.findall(translated_text))
    if source_count != target_count:
        mismatch(
            "numerals",
            source_count,
            target_count,
            f"numeral tokens {source_count} vs {target_count}",
        )

    # --- percentages ---
    source_count = len(PERCENTAGES.findall(source))
    target_count = len(PERCENTAGES.findall(translated_text))
    if source_count != target_count:
        mismatch(
            "percentages",
            source_count,
            target_count,
            f"percentage tokens {source_count} vs {target_count}",
        )

    # --- Roman numerals ---
    source_roman = roman_matches(source)
    target_roman = roman_matches(translated_text)
    if len(source_roman) != len(target_roman):
        mismatch(
            "roman_numerals",
            len(source_roman),
            len(target_roman),
            f"Roman numeral tokens {len(source_roman)} vs {len(target_roman)}",
        )

    # --- table delimiters (table-kind segments only, parser rule) ---
    if segment.kind == "table":
        source_count = len(TABLE_PIPE.findall(source))
        target_count = len(TABLE_PIPE.findall(translated_text))
        if source_count != target_count:
            mismatch(
                "table_delimiters",
                source_count,
                target_count,
                f"table delimiter '|' {source_count} vs {target_count}",
            )

    return flags


def count_class_totals(flags: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """``check -> number of flagged segments`` for the qa-1 report block."""
    totals: Dict[str, int] = {}
    for flag in flags:
        totals[flag["check"]] = totals.get(flag["check"], 0) + 1
    return totals
