import pytest

from chapter_recovery.candidate_fusion import build_heading_candidates
from chapter_recovery.evidence_models import Bookmark, PdfEvidenceResult, PdfLine, PdfPageEvidence
from chapter_recovery.models import Block, TocEntry
from chapter_recovery.page_map import PageMap
from chapter_recovery.zones import infer_page_zones


def block(page, index, label, content, y1=0, y2=40, order=None):
    return Block(page, index, index, label, content, (0, y1, 700, y2), order, 800, 1200)


def pdf_page(index, median=9.0, lines=()):
    return PdfPageEvidence(
        page_index=index,
        page_width=612.0,
        page_height=792.0,
        native_char_count=0,
        span_count=0,
        font_size_p10=0.0,
        font_size_median=median,
        font_size_p90=0.0,
        top_font_families=[],
        bold_span_count=0,
        italic_span_count=0,
        large_lines=list(lines),
    )


def page_map(offset=0):
    return PageMap(offset=offset, confidence=1.0, support=1, observations=1, print_to_json={})


def build(pages, pdf=None, toc_entries=(), config=None, book_id="test-book"):
    zones = infer_page_zones(pages)
    return build_heading_candidates(
        pages,
        page_map(),
        zones,
        list(toc_entries),
        pdf,
        None,
        config or {},
        book_id,
    )


def empty_pdf(pages, bookmarks=()):
    return PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[pdf_page(index) for index in range(len(pages))],
        bookmarks=list(bookmarks),
        page_count=len(pages),
    )


def test_paddle_sources_collected():
    pages = [
        [
            block(0, 0, "doc_title", "BOOK TITLE", 0, 30),
            block(0, 1, "paragraph_title", "Chapter One", 60, 90),
            block(0, 2, "header", "RUNNING HEAD", 100, 110),
            block(0, 3, "text", "Loose sentence text here", 200, 230),
            block(0, 4, "content", "some body content", 300, 320),
        ]
    ]
    candidates = build(pages)
    assert [c.exact_text for c in candidates] == ["BOOK TITLE", "Chapter One", "RUNNING HEAD"]
    assert all(c.paddle_label in {"doc_title", "paragraph_title", "header"} for c in candidates)
    assert all("Loose sentence text here" != c.exact_text for c in candidates)


def test_source_union_and_dedup():
    heading = block(3, 1, "paragraph_title", "Einleitung", 100, 140)
    pages = [[] for _ in range(4)]
    pages[3] = [heading]
    line = PdfLine(3, "Einleitung", (0.05, 0.085, 0.85, 0.115), 16.0, "Dolly-Roman", False, False)
    pdf = PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[pdf_page(3, median=9.0, lines=[line])],
        bookmarks=[Bookmark(1, "Einleitung", 4)],  # PDF page 4 -> json page 3
        page_count=4,
    )
    candidates = build(pages, pdf=pdf)
    assert len(candidates) == 1
    merged = candidates[0]
    assert merged.exact_text == "Einleitung"
    assert merged.page_json == 3
    assert merged.source_block_ids == [heading.id]
    expected_refs = {
        f"paddle:{heading.id}:paragraph_title",
        "pdf:bookmark:0",
        "pdf:font:3:0",
    }
    assert set(merged.evidence_refs) == expected_refs
    # strongest source present is paddle -> bookmark/font fields are dropped
    assert merged.paddle_label == "paragraph_title"
    assert merged.pdf_bookmark is None
    assert merged.native_font is None
    assert merged.bbox == (0.0, pytest.approx(0.0833), 0.875, pytest.approx(0.1167))


def _normalize_page_order(blocks):
    """Same stable sort normalize_blocks applies to a page's blocks."""
    return sorted(
        blocks,
        key=lambda item: (
            item.order is None,
            item.order if item.order is not None else item.bbox[1],
            item.bbox[0],
        ),
    )


def test_deterministic_ordering():
    pages = [[] for _ in range(6)]
    pages[0] = [
        block(0, 0, "header", "KANT", 100, 110),
        block(0, 1, "paragraph_title", "Einleitung", 200, 240),
    ]
    pages[1] = [block(1, 0, "paragraph_title", "Erster Abschnitt", 100, 140)]
    lines0 = [
        PdfLine(0, "KANT", (0.2, 0.09, 0.8, 0.1), 16.0, "X-Roman", False, False),
        PdfLine(0, "Einleitung", (0.1, 0.17, 0.9, 0.2), 16.0, "X-Roman", False, False),
    ]
    lines1 = [PdfLine(1, "Erster Abschnitt", (0.1, 0.08, 0.9, 0.12), 16.0, "X-Roman", False, False)]
    pdf = PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[pdf_page(0, lines=lines0), pdf_page(1, lines=lines1)],
        bookmarks=[
            Bookmark(1, "Einleitung", 1),
            Bookmark(2, "Erster Abschnitt", 2),
        ],
        page_count=6,
    )
    run_a = build(pages, pdf=pdf)

    # Equivalent inputs: per-page blocks appear in different OCR order (then
    # get normalized the way paddleocr does) and pdf page records are
    # permuted.  Bookmark list order is canonical PDF outline order.
    shuffled_pages = [[] for _ in range(6)]
    shuffled_pages[0] = _normalize_page_order(list(reversed(pages[0])))
    shuffled_pages[1] = _normalize_page_order(list(reversed(pages[1])))
    shuffled_pdf = PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[pdf_page(1, lines=lines1), pdf_page(0, lines=lines0)],
        bookmarks=[
            Bookmark(1, "Einleitung", 1),
            Bookmark(2, "Erster Abschnitt", 2),
        ],
        page_count=6,
    )
    run_b = build(shuffled_pages, pdf=shuffled_pdf)
    assert [candidate.candidate_id for candidate in run_a] == [
        candidate.candidate_id for candidate in run_b
    ]
    assert [candidate.to_dict() for candidate in run_a] == [
        candidate.to_dict() for candidate in run_b
    ]
    # sanity: all three sources merged into the expected candidates
    assert [candidate.exact_text for candidate in run_a] == [
        "KANT",
        "Einleitung",
        "Erster Abschnitt",
    ]


def test_suppression_reasons():
    pages = [[] for _ in range(10)]
    for page_index in range(1, 9):
        pages[page_index] = [block(page_index, 0, "header", "THE PHILOSOPHY OF RIGHT", 20, 30)]
    pages[9] = [block(9, 0, "paragraph_title", "a b c d e f g h i j k l m n o p q r s t u v w x y", 50, 90)]
    candidates = build(pages)
    headers = [c for c in candidates if c.paddle_label == "header"]
    assert len(headers) == 8
    assert all("running_header_repeat" in c.deterministic_suppressions for c in headers)
    sentence = [c for c in candidates if c.exact_text.startswith("a b c d e f g")]
    assert len(sentence) == 1
    assert "sentence_like" in sentence[0].deterministic_suppressions


def test_toc_matches_and_numbering():
    pages = [
        [
            block(0, 0, "paragraph_title", "Einleitung", 100, 140),
            block(0, 1, "paragraph_title", "§41", 200, 220),
        ]
    ]
    entry = TocEntry(title="Einleitung", page_label="41", print_page=41, kind="chapter", source_page=0)
    candidates = build(pages, toc_entries=[entry])
    by_text = {c.exact_text: c for c in candidates}
    intro = by_text["Einleitung"]
    assert intro.numbering is None
    assert len(intro.toc_matches) == 1
    match = intro.toc_matches[0]
    assert match["toc_entry_id"] == entry.id
    assert match["title"] == "Einleitung"
    assert match["page_label"] == "41"
    assert match["match_score"] == pytest.approx(1.0)
    assert match["match_kind"] == "exact"
    section = by_text["§41"]
    assert section.numbering == {
        "token": "§41",
        "path": ["§", "41"],
        "depth": 2,
        "family": "section_symbol",
    }
    assert section.toc_matches == []


def test_repetition_window():
    pages = [[] for _ in range(41)]
    for page_index in (5, 9, 40):
        pages[page_index] = [block(page_index, 0, "paragraph_title", "Introduction", 100, 140)]
    candidates = build(pages, config={"repetition_window": 10})
    by_page = {c.page_json: c for c in candidates}
    assert by_page[5].repetition == {"count": 3, "nearby_count": 2}
    assert by_page[9].repetition == {"count": 3, "nearby_count": 2}
    assert by_page[40].repetition == {"count": 3, "nearby_count": 1}


def test_text_label_needs_pdf_support():
    pages = [[] for _ in range(5)]
    orphan = block(2, 0, "text", "Orphan Text Block", 100, 140)
    intro = block(3, 0, "text", "Einleitung in die Rechtsphilosophie", 100, 140)
    intro_line = block(4, 0, "text", "Zur Einführung", 100, 140)
    pages[2] = [orphan]
    pages[3] = [intro]
    pages[4] = [intro_line]
    support_line = PdfLine(4, "Zur Einführung", (0.05, 0.085, 0.85, 0.115), 16.0, "Dolly-Roman", False, False)
    pdf = PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[
            pdf_page(3, median=9.0),
            pdf_page(4, median=9.0, lines=[support_line]),
        ],
        bookmarks=[Bookmark(1, "Einleitung der Rechtsphilosophie", 4)],
        page_count=5,
    )
    candidates = build(pages, pdf=pdf)
    assert len(candidates) == 3
    # unsupported text block is excluded
    assert all("Orphan Text Block" != c.exact_text for c in candidates)
    # bookmark-supported text block keeps its own candidate (distinct OCR text)
    supported = next(c for c in candidates if c.exact_text == "Einleitung in die Rechtsphilosophie")
    assert supported.paddle_label == "text"
    assert supported.source_block_ids == [intro.id]
    assert f"paddle:{intro.id}:text" in supported.evidence_refs
    assert f"pdf:text_support:{intro.id}:bookmark" in supported.evidence_refs
    # font-supported text block merges with its font line into one candidate
    font_supported = [c for c in candidates if c.exact_text == "Zur Einführung"]
    assert len(font_supported) == 1
    merged = font_supported[0]
    assert merged.source_block_ids == [intro_line.id]
    assert f"pdf:text_support:{intro_line.id}:font" in merged.evidence_refs
    assert "pdf:font:4:0" in merged.evidence_refs
    assert merged.native_font == {
        "family": "Dolly-Roman",
        "size": 16.0,
        "bold": False,
        "italic": False,
    }


def test_unresolved_bookmark_candidate_is_kept_null_and_sorts_last():
    """A3: unresolved bookmark pages keep page_json=null (id hash uses the -1
    sentinel only) and null-page candidates sort after all resolved ones."""
    pages = [[] for _ in range(5)]
    heading = block(0, 0, "paragraph_title", "Einleitung", 100, 140)
    pages[0] = [heading]
    # "Einleitung" bookmark resolves to page 0 (merges with the block);
    # "Nachwort" bookmark page 999 resolves outside the book -> page_json None
    pdf = PdfEvidenceResult(
        ok=True,
        reason="ok",
        pages=[pdf_page(0, median=9.0)],
        bookmarks=[Bookmark(1, "Einleitung", 1), Bookmark(2, "Nachwort", 999)],
        page_count=5,
    )
    candidates = build(pages, pdf=pdf)
    assert len(candidates) == 2
    resolved = next(c for c in candidates if c.exact_text == "Einleitung")
    unresolved = next(c for c in candidates if c.exact_text == "Nachwort")
    assert resolved.page_json == 0
    assert resolved.source_block_ids == [heading.id]
    assert unresolved.page_json is None  # stored null, not 0/-1
    assert unresolved.page_print == 999  # bookmark.page recorded (A3)
    assert unresolved.pdf_bookmark == {"level": 2, "title": "Nachwort", "page": 999}
    # null-page candidate sorts last (A3 ordering override of D7's 0-fallback)
    assert candidates[0] is resolved
    assert candidates[1] is unresolved
