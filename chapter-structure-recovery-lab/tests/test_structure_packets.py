import json

from chapter_recovery.evidence_models import (
    Bookmark,
    EvidenceCandidate,
    EvidenceMetadata,
    PdfEvidenceResult,
    PdfLine,
    PdfPageEvidence,
)
from chapter_recovery.models import TocEntry
from chapter_recovery.page_map import PageMap
from chapter_recovery.structure_packets import (
    build_book_structure_evidence,
    build_numbering_summary,
    build_repeated_headers,
    build_style_clusters,
    write_evidence_artifacts,
)
from chapter_recovery.style_model import StyleDecision


def metadata(candidate_count=3, page_count=100):
    return EvidenceMetadata(
        schema_version=1,
        book_id="b",
        book_title="Book",
        generated_by="chapter_recovery.evidence_v2",
        package_version="0.15.0",
        python_version="3.9.6",
        input_sha256="a" * 64,
        pdf_sha256="b" * 64,
        legacy_profile_used=False,
        pdf_evidence_ok=True,
        pdf_reason="ok",
        page_count=page_count,
        candidate_count=candidate_count,
        source_counts={"paddle": 1, "text_support": 0, "bookmark": 1, "font": 1},
    )


def candidate(cid, text, page, blocks=(), refs=()):
    numbering_value = None
    if text.startswith("§"):
        numbering_value = {
            "token": text,
            "path": ["§", text.lstrip("§")],
            "depth": 2,
            "family": "section_symbol",
        }
    return EvidenceCandidate(
        candidate_id=cid,
        source_block_ids=list(blocks),
        exact_text=text,
        page_json=page,
        page_print=page,
        bbox=(0.1, 0.1, 0.5, 0.15),
        numbering=numbering_value,
        evidence_refs=list(refs),
    )


def test_book_evidence_shape():
    candidates = [
        candidate("cand-1", "Einleitung", 41, blocks=["blk-1"], refs=["paddle:blk-1:paragraph_title"]),
        candidate("cand-2", "§41", 41, refs=["pdf:bookmark:0"]),
        candidate("cand-3", "KANT", 42, refs=["pdf:font:42:0"]),
    ]
    toc_entries = [TocEntry("Einleitung", "41", 41, "chapter", 0)]
    bookmarks = [
        Bookmark(1, "Vorwort", 10),
        Bookmark(2, "Vorrede", 42),
        Bookmark(1, "Index", 500),
    ]
    numbering_summary = build_numbering_summary(candidates)
    repeated_headers = []
    style_clusters = [{"level": 3, "count": 12, "sample_block_ids": ["blk-a", "blk-b"]}]
    packet = build_book_structure_evidence(
        metadata(candidate_count=3, page_count=100),
        candidates,
        toc_entries,
        bookmarks,
        PageMap(offset=0, confidence=1.0, support=1, observations=1, print_to_json={}),
        {page: "mainmatter" for page in range(50)},
        numbering_summary,
        repeated_headers,
        style_clusters,
    )
    assert packet["schema_version"] == 1
    assert packet["metadata"]["book_id"] == "b"
    assert packet["candidate_count"] == 3
    assert packet["page_density"] == 0.03  # 3 / 100
    assert len(packet["candidates"]) == 3
    assert sorted(packet["candidates"][0].keys()) == [
        "candidate_id",
        "exact_text",
        "page_json",
        "primary_sources",
    ]
    assert packet["candidates"][0]["primary_sources"] == ["paddle"]
    assert packet["candidates"][1]["primary_sources"] == ["bookmark"]
    assert packet["candidates"][2]["primary_sources"] == ["font"]
    assert [e["title"] for e in packet["toc_entries"]] == ["Einleitung"]
    assert packet["zones"]["0"] == "mainmatter"
    # bookmark tree nesting: 1->2 under Vorwort; second level-1 is a sibling
    tree = packet["bookmark_tree"]
    assert [node["title"] for node in tree] == ["Vorwort", "Index"]
    assert [node["title"] for node in tree[0]["children"]] == ["Vorrede"]
    assert packet["numbering_summary"]["families"] == {"section_symbol": 1}
    assert packet["style_clusters"] == style_clusters
    assert packet["repeated_headers"] == []


def test_numbering_summary_and_repeated_headers():
    letter = candidate("cand-3", "A. Besitznahme", 20, refs=["pdf:bookmark:2"])
    letter.numbering = {
        "token": "A",
        "path": ["A"],
        "depth": 1,
        "family": "upper_letter",
    }
    candidates = [
        candidate("cand-1", "§1", 1, refs=["pdf:bookmark:0"]),
        candidate("cand-2", "§1", 10, refs=["pdf:bookmark:1"]),
        letter,
    ]
    summary = build_numbering_summary(candidates)
    assert summary["families"] == {"section_symbol": 2, "upper_letter": 1}
    assert summary["representative_tokens"]["section_symbol"] == ["§1"]


def test_style_clusters_group_and_cap_samples():
    decisions = {
        f"blk-{index:02d}": StyleDecision(level=3, confidence=0.9, evidence=[])
        for index in range(12)
    }
    clusters = build_style_clusters(decisions)
    assert clusters == [
        {"level": 3, "count": 12, "sample_block_ids": [f"blk-{i:02d}" for i in range(8)]}
    ]


def test_atomic_write_and_encoding(tmp_path):
    umlaut = candidate("cand-1", "Größe der Familie", 10, blocks=["blk-1"], refs=["paddle:blk-1:paragraph_title"])
    umlaut.before_context = "Zur Größe"
    meta = metadata(candidate_count=1, page_count=10)
    pdf_result = PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[
            PdfPageEvidence(
                page_index=0,
                page_width=612.0,
                page_height=792.0,
                native_char_count=5,
                span_count=1,
                font_size_p10=10.0,
                font_size_median=10.0,
                font_size_p90=10.0,
                top_font_families=[("Dolly-Roman", 1)],
                bold_span_count=0,
                italic_span_count=0,
                large_lines=[
                    PdfLine(0, "KANT", (0.1, 0.1, 0.9, 0.2), 16.0, "Dolly-Roman", True, False)
                ],
            )
        ],
        bookmarks=[Bookmark(1, "Größe", 1)],
        page_count=1,
    )
    packet = build_book_structure_evidence(
        meta,
        [umlaut],
        [],
        pdf_result.bookmarks,
        PageMap(offset=0, confidence=1.0, support=1, observations=1, print_to_json={}),
        {0: "mainmatter"},
        {"families": {}, "representative_tokens": {}},
        [],
        [],
    )
    write_evidence_artifacts(tmp_path, meta, pdf_result, [umlaut], packet)

    evidence_dir = tmp_path / "evidence"
    assert (evidence_dir / "pdf_evidence.json").is_file()
    assert (evidence_dir / "heading_candidates.jsonl").is_file()
    assert (evidence_dir / "book_structure_evidence.json").is_file()
    assert list(evidence_dir.glob("*.tmp")) == []
    packet_raw = (evidence_dir / "book_structure_evidence.json").read_text(encoding="utf-8")
    assert json.loads(packet_raw)["candidate_count"] == 1
    # ensure_ascii=False: umlauts must be stored verbatim, not \u-escaped
    assert "Größe" in (evidence_dir / "heading_candidates.jsonl").read_text(encoding="utf-8")
    assert "Größe" in packet_raw
    assert "\\u00" not in packet_raw
    pdf_raw = (evidence_dir / "pdf_evidence.json").read_text(encoding="utf-8")
    assert json.loads(pdf_raw)["ok"] is True
    assert "KANT" in pdf_raw
