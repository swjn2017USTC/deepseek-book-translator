from chapter_recovery.alignment import align_toc_entries
from chapter_recovery.confirmed import merge_confirmed_toc
from chapter_recovery.models import Block, TocEntry
from chapter_recovery.page_map import PageMap


def block(page, index, label, content):
    return Block(page, index, index, label, content, (100, 120, 700, 180), index, 800, 1200)


def test_global_alignment_prefers_unique_monotone_doc_titles_over_wrong_page_header():
    pages = [[] for _ in range(9)]
    pages[0] = [block(0, 1, "text", "First Chapter 3    Second Chapter 2")]
    pages[2] = [block(2, 1, "header", "Second Chapter")]
    pages[3] = [block(3, 1, "doc_title", "First Chapter")]
    pages[7] = [block(7, 1, "doc_title", "Second Chapter")]
    entries = [
        TocEntry("First Chapter", "3", 3, "chapter", 0),
        TocEntry("Second Chapter", "2", 2, "chapter", 0),
    ]
    page_map = PageMap(0, 1.0, 2, 2, {2: 2, 3: 3})

    aligned = align_toc_entries(pages, entries, page_map, {"toc_alignment_window": 2})

    assert [item.block.page_index for item in aligned if item.block] == [3, 7]
    assert len({item.block.id for item in aligned if item.block}) == 2
    assert all(item.status == "matched" for item in aligned)


def test_global_alignment_never_assigns_one_body_block_twice():
    pages = [[], [block(1, 1, "doc_title", "Repeated Title")]]
    entries = [
        TocEntry("Repeated Title", "", None, "chapter", -1),
        TocEntry("Repeated Title", "", None, "chapter", -1),
    ]
    aligned = align_toc_entries(pages, entries, PageMap(None, 0.0, 0, 0, {}), {})

    assert sum(item.status == "matched" for item in aligned) == 1
    assert sum(item.status == "unmatched" for item in aligned) == 1


def test_distinctive_partial_title_can_anchor_page_less_introduction():
    pages = [[] for _ in range(5)]
    pages[1] = [block(1, 1, "doc_title", "Beyond Totalitarianism")]
    pages[3] = [
        block(3, 1, "text", "After Totalitarianism – Stalinism and Nazism Compared")
    ]
    entry = TocEntry("Introduction: After Totalitarianism", "", None, "chapter", -1)

    aligned = align_toc_entries(pages, [entry], PageMap(None, 0.0, 0, 0, {}), {})

    assert aligned[0].status == "matched"
    assert aligned[0].block.page_index == 3
    assert aligned[0].confidence < 0.9


def test_page_less_exact_title_beats_later_distinctive_partial_title():
    pages = [[] for _ in range(6)]
    pages[1] = [block(1, 1, "text", "PART II EARLY CITIES AND INFORMATION TECHNOLOGIES")]
    pages[4] = [block(4, 1, "text", "EARLY CITIES AS CREATIONS")]
    entry = TocEntry(
        "PART II: EARLY CITIES AND INFORMATION TECHNOLOGIES", "", None, "part", -1
    )

    aligned = align_toc_entries(
        pages, [entry], PageMap(None, 0.0, 0, 0, {}), {}
    )

    assert aligned[0].status == "matched"
    assert aligned[0].block.page_index == 1
    assert aligned[0].components["title_similarity"] == 1.0


def test_confirmed_entries_with_pages_consume_fuzzy_ocr_duplicates():
    confirmed = [TocEntry("A History of Revolution", "41", 41, "chapter", -1, provenance="confirmed_toc")]
    parsed = [TocEntry("1. A History of Revolution", "41", 41, "chapter", 4)]

    merged = merge_confirmed_toc(parsed, confirmed)

    assert len(merged) == 1
    assert merged[0].provenance == "confirmed_toc+ocr_toc"
    assert merged[0].source_page == 4


def test_part_title_page_beats_earlier_introduction_repeat():
    pages = [[] for _ in range(9)]
    pages[2] = [
        block(2, 1, "paragraph_title", "Part I: Historiography, methods, and themes")
    ]
    pages[6] = [
        block(6, 1, "text", "PART I HISTORIOGRAPHY, METHODS, AND THEMES")
    ]
    entry = TocEntry(
        "PART I: HISTORIOGRAPHY, METHODS, AND THEMES", "", None, "part", 0
    )

    aligned = align_toc_entries(
        pages, [entry], PageMap(None, 0.0, 0, 0, {}), {}
    )

    assert aligned[0].status == "matched"
    assert aligned[0].block.page_index == 6
    assert aligned[0].components["structural_title_bonus"] == 0.14
