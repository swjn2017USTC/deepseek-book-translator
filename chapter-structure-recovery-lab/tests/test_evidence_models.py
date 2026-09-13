import pytest

from chapter_recovery.evidence_models import (
    EvidenceCandidate,
    EvidenceMetadata,
    make_candidate_id,
)


def sample_candidate(**overrides):
    base = dict(
        candidate_id="cand-0123456789abcdef",
        source_block_ids=["blk-aaa", "blk-bbb"],
        exact_text="§ 41 Einleitung",
        page_json=115,
        page_print=115,
        bbox=(0.1, 0.2, 0.8, 0.25),
        zone="mainmatter",
        paddle_label="paragraph_title",
        paddle_relevel=None,
        pdf_bookmark=None,
        native_font=None,
        numbering={"token": "§41", "path": ["§", "41"], "depth": 2, "family": "section_symbol"},
        toc_matches=[
            {
                "toc_entry_id": "toc-abc",
                "title": "Einleitung",
                "page_label": "115",
                "match_score": 1.0,
                "match_kind": "exact",
            }
        ],
        repetition={"count": 1, "nearby_count": 1},
        before_context="Text davor",
        after_context="Text danach",
        deterministic_suppressions=[],
        evidence_refs=["paddle:blk-aaa:paragraph_title"],
    )
    base.update(overrides)
    return EvidenceCandidate(**base)


def test_candidate_serialization_roundtrip():
    candidate = sample_candidate()
    restored = EvidenceCandidate.from_dict(candidate.to_dict())
    assert restored == candidate
    # json round trip: None -> null -> None and tuple bbox -> list -> tuple
    assert restored.bbox == (0.1, 0.2, 0.8, 0.25)
    assert isinstance(restored.bbox, tuple)


def test_null_field_roundtrip_is_lossless():
    candidate = sample_candidate(
        page_json=None,
        page_print=None,
        bbox=None,
        paddle_label=None,
        numbering=None,
        before_context=None,
        after_context=None,
        pdf_bookmark={"level": 2, "title": "Einleitung", "page": 116},
    )
    data = candidate.to_dict()
    assert data["bbox"] is None
    assert data["page_json"] is None
    restored = EvidenceCandidate.from_dict(data)
    assert restored == candidate


def test_candidate_ids_unique_and_stable():
    ids = set()
    for page in (10, 11):
        for text in ("Einleitung", "Vorrede"):
            ids.add(make_candidate_id("b", page, ["blk-1", "blk-2"], text))
    assert len(ids) == 4  # vary text/page => unique

    first = make_candidate_id("book", 115, ["blk-1", "blk-2", "blk-3"], "Einleitung")
    reordered = make_candidate_id("book", 115, ["blk-3", "blk-1", "blk-2"], "Einleitung")
    again = make_candidate_id("book", 115, ["blk-2", "blk-3", "blk-1"], "Einleitung")
    assert first == reordered == again
    assert first.startswith("cand-")


def test_candidate_id_null_page_sentinel_is_stable():
    a = make_candidate_id("book", None, ["blk-1"], "Cover")
    b = make_candidate_id("book", -1, ["blk-1"], "Cover")
    c = make_candidate_id("book", None, ["blk-1"], "Cover")
    zero = make_candidate_id("book", 0, ["blk-1"], "Cover")
    assert a == b == c  # null-page candidates always hash the -1 sentinel
    assert a != zero  # a real page 0 is distinct from an unresolved page


def test_evidence_metadata_roundtrip():
    metadata = EvidenceMetadata(
        schema_version=1,
        book_id="hegel-rechts-commentary",
        book_title="Hegels Grundlinien",
        generated_by="chapter_recovery.evidence_v2",
        package_version="0.15.0",
        python_version="3.9.6",
        input_sha256="a" * 64,
        pdf_sha256=None,
        legacy_profile_used=True,
        pdf_evidence_ok=True,
        pdf_reason="ok",
        page_count=1137,
        candidate_count=2400,
        source_counts={"paddle": 2200, "bookmark": 62, "font": 180, "text_support": 4},
    )
    restored = EvidenceMetadata.from_dict(metadata.to_dict())
    assert restored == metadata
    assert restored.pdf_sha256 is None
