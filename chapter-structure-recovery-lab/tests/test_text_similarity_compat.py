from difflib import SequenceMatcher

import pytest

import chapter_recovery.text_similarity as text_similarity
from chapter_recovery.alignment import normalized_title, title_similarity
from chapter_recovery.text_similarity import (
    sequence_ratio,
    title_similarity_v2,
    token_set_ratio,
)

FIXTURES = [
    ("", ""),
    ("", "Einleitung"),
    ("Einleitung", "Einleitung"),
    ("§ 41 Einleitung", "Einleitung"),
    ("Einleitung", "Schluss"),
    ("Größe", "Größe"),
    ("Größe", "Grosse"),
    ("DIE EHE", "DIE EHE"),
    ("Die bürgerliche Gesellschaft", "Die buergerliche Gesellschaft"),
    ("III. Die Souveränität gegen aussen", "II. Die Souveränität gegen aussen"),
    ("Erster Teil: Das abstrakte Recht", "Das abstrakte Recht"),
    ("第一章 导论", "第一章"),
    ("绪论", "绪 论"),
    ("Naturrecht und Staatswissenschaft im Grundrisse", "Naturrecht und Staatswissenschaft"),
    ("Zum Argumentationsgang des Textes", "Zum Argumentationsgang"),
    ("A. BESITZNAHME", "A. Besitznahme"),
]

# Measured difflib-heuristic divergence (P02 finding): difflib's recursive
# block matcher under-counts the true LCS when several competing repeated
# blocks exist, so rapidfuzz's exact-LCS ratio exceeds SequenceMatcher.ratio
# here.  This pair is deliberately NOT part of the parity fixtures above;
# the divergence is pinned by the dedicated test below instead of being
# hidden.  The migration-relevant consequence: switching the char-ratio term
# to rapidfuzz can only RAISE title_similarity on such repetitive pairs,
# never lower it (difflib under-counts; rapidfuzz is exact).
HEURISTIC_DIVERGENCE_PAIR = (
    "Erster Teil: Das abstrakte Recht",
    "Zweiter Teil: Die Sittlichkeit",
)


def test_title_similarity_v2_matches_legacy():
    # runs under whichever backend is active (difflib fallback or rapidfuzz);
    # agreement must hold up to final-ulp float noise either way.
    for left, right in FIXTURES:
        assert title_similarity_v2(left, right) == pytest.approx(
            title_similarity(left, right), abs=1e-12
        ), (left, right)


def test_difflib_heuristic_divergence_is_documented():
    """Pin the known heuristic difference instead of hiding it.

    difflib's SequenceMatcher is not an exact LCS matcher; on the pair above
    it returns 0.5333 while the exact LCS ratio (rapidfuzz) is 0.6.  The
    legacy reference is difflib, so the parity fixtures exclude exactly this
    class of repetitive-overlap pair, and any future cutover must expect
    title_similarity to rise (never fall) on such strings.
    """
    left, right = HEURISTIC_DIVERGENCE_PAIR
    legacy = title_similarity(left, right)
    assert legacy == SequenceMatcher(None, normalized_title(left), normalized_title(right)).ratio()
    if text_similarity._rapidfuzz_available():
        v2 = title_similarity_v2(left, right)
        assert v2 > legacy  # exact LCS beats the heuristic matcher here
        assert v2 == pytest.approx(0.6, abs=1e-12)
    else:
        # without rapidfuzz, v2 == legacy bit-for-bit even on this pair
        assert title_similarity_v2(left, right) == legacy


def test_sequence_ratio_rapidfuzz_equals_difflib():
    rapidfuzz = pytest.importorskip("rapidfuzz")
    for left, right in FIXTURES:
        # rapidfuzz reports percentages [0, 100]; difflib reports [0, 1]
        assert rapidfuzz.fuzz.ratio(left, right) / 100.0 == pytest.approx(
            SequenceMatcher(None, left, right).ratio(), abs=1e-12
        ), (left, right)
        assert sequence_ratio(left, right) == pytest.approx(
            SequenceMatcher(None, left, right).ratio(), abs=1e-12
        ), (left, right)


def test_token_set_ratio_consistency():
    assert token_set_ratio("a b c", "a b c") == pytest.approx(1.0)
    assert token_set_ratio("", "") == pytest.approx(1.0)
    # disjoint tokens AND disjoint letters: both backends fall back to the
    # plain sequence ratio, which is 0 here
    assert token_set_ratio("qqqq", "zzzz") == pytest.approx(0.0)
    # partial word overlap ("alpha"/"delta" share no tokens, but letter-level
    # similarity survives in the fallback): both backends agree
    assert token_set_ratio("alpha beta", "gamma delta") == pytest.approx(8 / 21, abs=1e-12)
    subset = token_set_ratio("einleitung der rechtsphilosophie", "einleitung")
    assert subset > 0.5  # intersection alone already scores 1.0 in the classic formula
    assert subset <= 1.0


def test_difflib_fallback_is_original(monkeypatch):
    monkeypatch.setattr(text_similarity, "_rapidfuzz_available", lambda: False)
    for left, right in FIXTURES:
        assert sequence_ratio(left, right) == SequenceMatcher(None, left, right).ratio()
        assert title_similarity_v2(left, right) == title_similarity(left, right)
