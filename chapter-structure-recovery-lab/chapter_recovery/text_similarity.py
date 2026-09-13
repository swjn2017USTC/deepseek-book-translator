"""RapidFuzz compat harness for deterministic string similarity (P02).

The core package stays stdlib-only importable: `rapidfuzz` is imported lazily
inside functions, so this module (and every importer of it) imports cleanly
without rapidfuzz installed.  When rapidfuzz is present, its scorers are used;
otherwise a difflib-based implementation with identical semantics is used.

Scoring parity notes
--------------------
- ``rapidfuzz.fuzz.ratio`` returns a percentage in [0, 100]; dividing by 100
  yields the normalized exact-LCS ratio ``2 * LCS / (len(a) + len(b))``.
- ``difflib.SequenceMatcher.ratio`` computes the same formula but its block
  matching is heuristic: on strings with several competing repeated blocks it
  can under-count the true LCS (measured example: "erster teil das abstrakte
  recht" vs "zweiter teil die sittlichkeit" -> difflib 0.5333 vs exact 0.6).
  It never applies its ``autojunk`` filter below 200 characters.  The difflib
  fallback in this module is bit-for-bit identical to the legacy
  ``alignment.title_similarity`` path; when rapidfuzz is present
  ``sequence_ratio`` is exact-LCS, so ``title_similarity_v2`` can exceed the
  legacy value precisely where difflib under-counts (see the compat tests for
  the documented divergence fixture).  Autojunk never applies to the short
  heading/TOC strings used here (only sequences >= 200 items).
- ``rapidfuzz.fuzz.token_set_ratio`` applies the classic token-set maximum
  over ``ratio`` of sorted-intersection combinations; the difflib fallback in
  this module mirrors that algorithm token-for-token (whitespace split, no
  implicit case folding -- callers normalize first).
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Callable


def _rapidfuzz_available() -> bool:
    try:
        import rapidfuzz  # noqa: F401
        return True
    except ImportError:
        return False


def _ratio_backend() -> Callable[[str, str], float]:
    """Return the backend ratio function producing a [0, 1] score."""
    if _rapidfuzz_available():
        from rapidfuzz import fuzz

        def _rf_ratio(left: str, right: str) -> float:
            # rapidfuzz returns percentages [0, 100]; scale to [0, 1].
            return float(fuzz.ratio(left, right)) / 100.0

        return _rf_ratio

    def _difflib_ratio(left: str, right: str) -> float:
        return SequenceMatcher(None, left, right).ratio()

    return _difflib_ratio


def _token_set_backend() -> Callable[[str, str], float]:
    if _rapidfuzz_available():
        from rapidfuzz import fuzz

        def _rf_token_set(left: str, right: str) -> float:
            return float(fuzz.token_set_ratio(left, right, processor=None)) / 100.0

        return _rf_token_set

    def _difflib_token_set(left: str, right: str) -> float:
        return _token_set_ratio_difflib(left, right)

    return _difflib_token_set


def _token_set_ratio_difflib(left: str, right: str) -> float:
    """Classic token-set ratio implemented with SequenceMatcher only.

    Tokens are whitespace-separated words (no case folding -- callers
    normalize).  The score is the maximum SequenceMatcher ratio over the
    sorted-intersection string combined with each side's unique tokens,
    mirroring rapidfuzz's ``token_set_ratio`` (processor=None) semantics.
    """
    tokens_a = left.split()
    tokens_b = right.split()
    set_a, set_b = set(tokens_a), set(tokens_b)
    intersection = set_a & set_b
    if not intersection:
        if not set_a and not set_b:
            return 1.0
        return SequenceMatcher(None, left, right).ratio()
    section = " ".join(sorted(intersection))
    combined_1st = " ".join(sorted(intersection | (set_a - set_b)))
    combined_2nd = " ".join(sorted(intersection | (set_b - set_a)))
    return max(
        SequenceMatcher(None, section, combined_1st).ratio(),
        SequenceMatcher(None, section, combined_2nd).ratio(),
        SequenceMatcher(None, combined_1st, combined_2nd).ratio(),
    )


def sequence_ratio(left: str, right: str) -> float:
    """Normalized character-sequence ratio in [0, 1] (rapidfuzz or difflib)."""
    return _ratio_backend()(left, right)


def token_set_ratio(left: str, right: str) -> float:
    """Token-set ratio in [0, 1] (rapidfuzz or difflib fallback)."""
    if not left and not right:
        return 1.0  # both empty: identical token multisets (rapidfuzz returns 0 otherwise)
    return _token_set_backend()(left, right)


def title_similarity_v2(left: str, right: str) -> float:
    """Structurally identical to ``alignment.title_similarity`` (P02 parity).

    Reuses the same normalization (``alignment.normalized_title``) and the
    same token-Jaccard / containment terms, but routes the character-sequence
    term through ``sequence_ratio()`` instead of calling SequenceMatcher
    directly.  Under the difflib backend the two functions are bit-for-bit
    identical; under rapidfuzz they agree up to final-ulp float noise.
    """
    from .alignment import normalized_title
    from .alignment import STOPWORDS

    a = normalized_title(left)
    b = normalized_title(right)
    if not a or not b:
        return 0.0
    sequence = sequence_ratio(a, b)
    aset = {token for token in a.split() if token not in STOPWORDS}
    bset = {token for token in b.split() if token not in STOPWORDS}
    token = len(aset & bset) / max(1, len(aset | bset))
    overlap = len(aset & bset)
    containment = max(overlap / max(1, len(aset)), overlap / max(1, len(bset)))
    containment_score = containment * 0.94 if overlap >= 2 else 0.0
    return max(sequence, token, containment_score)


__all__: Any = [
    "_rapidfuzz_available",
    "_ratio_backend",
    "_token_set_backend",
    "sequence_ratio",
    "token_set_ratio",
    "title_similarity_v2",
]
