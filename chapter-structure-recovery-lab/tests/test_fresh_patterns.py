import json

from chapter_recovery.confirmed import load_confirmed_toc
from chapter_recovery.heading_text import (
    canonical_heading_text,
    merge_heading_blocks,
    numbering,
    rejection_reason,
)
from chapter_recovery.models import Block
from chapter_recovery.models import Node
from chapter_recovery.page_map import PageMap
from chapter_recovery.recover import (
    _is_chapter_byline,
    _is_visual_legend_line,
    _repair_unnumbered_section_parents,
    _recover_text_children_below_doc_titles,
)
from chapter_recovery.style_model import infer_internal_styles
from chapter_recovery.toc import parse_toc_entries
from chapter_recovery.visual_context import heading_visual_context
from chapter_recovery.zones import infer_page_zones


def block(page, index, label, content, y1=0, y2=40):
    return Block(page, index, index, label, content, (0, y1, 700, y2), index, 800, 1200)


def test_repairs_stacked_markdown_prefix_and_rejects_noise():
    assert canonical_heading_text("## #####  A damaged heading") == "A damaged heading"
    assert rejection_reason("◇ ◇ ◇") == "decorative_only"
    assert rejection_reason("### 22") == "folio_or_numeric_only"


def test_numbering_grammar_preserves_deep_commentary_levels():
    parsed = numbering("3.1.1 Selbstbewusstsein")
    assert parsed is not None
    assert parsed.path == ("3", "1", "1")
    assert parsed.depth == 3
    assert numbering("a) Einzelheit").family == "lower_letter"
    assert numbering("Teil II Geist").family == "teil"
    section = numbering("§4I")
    assert section is not None
    assert section.family == "section_symbol"
    assert section.path == ("§", "41")


def test_merges_auditable_split_heading_fragments():
    items = [
        block(3, 1, "paragraph_title", "FOUNDING THE NEW STATE:", 100, 130),
        block(3, 2, "paragraph_title", "WAR, PEACE, AND TERROR", 135, 165),
    ]
    merged = merge_heading_blocks(items)
    assert len(merged) == 1
    assert merged[0][0].clean_content == "FOUNDING THE NEW STATE: WAR, PEACE, AND TERROR"
    assert len(merged[0][1]) == 2


def test_merges_cross_page_heading_only_with_boundary_and_continuation_evidence():
    items = [
        block(3, 1, "paragraph_title", "EMPIRE AND REVOLUTION:", 1080, 1180),
        block(4, 1, "paragraph_title", "A COMPARATIVE HISTORY", 40, 110),
    ]
    merged = merge_heading_blocks(items)
    assert len(merged) == 1
    assert merged[0][0].clean_content == "EMPIRE AND REVOLUTION: A COMPARATIVE HISTORY"
    assert merged[0][1] == [items[0].id, items[1].id]


def test_does_not_merge_independent_headings_across_pages():
    items = [
        block(3, 1, "paragraph_title", "Resistance", 1080, 1180),
        block(4, 1, "paragraph_title", "Science culture", 40, 110),
    ]
    assert len(merge_heading_blocks(items)) == 2


def test_splits_part_and_chapter_glued_in_toc_and_keeps_year_range():
    pages = [[
        block(0, 1, "paragraph_title", "Contents"),
        block(0, 2, "text", "PART I STALINISM 1934–41 1. Origins and outcomes 23 AUTHOR NAME"),
    ]]
    entries = parse_toc_entries(pages, [0])
    assert [(item.kind, item.print_page) for item in entries] == [("part", None), ("chapter", 23)]
    assert "1934–41" in entries[0].title


def test_does_not_treat_wrapped_year_suffix_as_page_number():
    pages = [[block(0, 1, "text", "2 Social identity in Soviet Russia, 1934– 41")]]
    entry = parse_toc_entries(pages, [0])[0]
    assert entry.title.endswith("1934–41")
    assert entry.print_page is None


def test_loads_multiple_confirmed_toc_schemas(tmp_path):
    path = tmp_path / "toc_confirmed.json"
    path.write_text(json.dumps({
        "front_matter": [{"title": "Preface", "page": 7}],
        "parts": [{"title": "PART I", "page_start": 1}],
        "chapters": [{"title_src": "Chapter 1 Beginning", "print_page": 3}],
    }), encoding="utf-8")
    entries = load_confirmed_toc(path)
    assert [item.kind for item in entries] == ["frontmatter", "part", "chapter"]
    assert all(item.provenance == "confirmed_toc" for item in entries)


def test_zone_switch_prevents_notes_repetitions_from_becoming_mainmatter():
    pages = [[] for _ in range(20)]
    pages[12] = [block(12, 1, "paragraph_title", "Notes")]
    pages[15] = [block(15, 1, "paragraph_title", "Chapter 1 Beginning")]
    zones = infer_page_zones(pages, 2)
    assert zones[11] == "mainmatter"
    assert zones[12] == "notes"
    assert zones[15] == "notes"


def test_german_apparatus_headings_switch_bibliography_and_index_zones():
    pages = [[] for _ in range(20)]
    pages[12] = [block(12, 1, "paragraph_title", "Literatur")]
    pages[16] = [block(16, 1, "doc_title", "Personenregister")]
    zones = infer_page_zones(pages, 2)
    assert zones[12] == "bibliography"
    assert zones[15] == "bibliography"
    assert zones[16] == "index"


def test_rejects_heading_embedded_in_full_page_map_but_keeps_table_section_heading():
    image = Block(5, 1, 1, "image", "<img>", (20, 60, 780, 1140), None, 800, 1200)
    map_caption = block(5, 2, "paragraph_title", "Territorial Losses 1850–1900", 900, 960)
    assert heading_visual_context(map_caption, [image, map_caption]).rejection_reason == "visual_embedded_caption"

    table = Block(6, 2, 2, "table", "<table>", (100, 280, 700, 1100), None, 800, 1200)
    heading = block(6, 1, "paragraph_title", "Abbreviations", 120, 160)
    context = heading_visual_context(heading, [heading, table])
    assert context.rejection_reason is None
    assert "adjacent_table" not in context.evidence


def test_repeated_chapter_local_apparatus_keeps_section_level_and_confidence():
    headings = [
        block(page, page, "paragraph_title", "FURTHER READING", 100, 120)
        for page in range(3)
    ] + [
        block(page + 3, page + 3, "paragraph_title", f"Main topic {page}", 100, 140)
        for page in range(3)
    ]
    decisions = infer_internal_styles(headings)
    for heading in headings[:3]:
        assert decisions[heading.id].level == 3
        assert decisions[heading.id].confidence == 0.96
        assert "repeated_chapter_local_apparatus_heading" in decisions[heading.id].evidence

    plural = [
        block(page + 10, page, "paragraph_title", "FURTHER READINGS", 100, 120)
        for page in range(3)
    ]
    plural_decisions = infer_internal_styles(plural)
    assert all(plural_decisions[item.id].confidence == 0.96 for item in plural)


def test_multi_author_uppercase_byline_is_not_a_heading_candidate():
    chapter = Node(
        id="chapter", kind="chapter", level=2, title_source="5 Cities",
        page_json=5, page_print=5, parent_id="book", block_id="title",
        source="doc_title", confidence=1.0, bbox=(200, 200, 600, 300),
    )
    byline = block(
        5, 2, "text",
        "JOHN BAINES, MIRIAM STARK, THOMAS GARRISON AND STEPHEN HOUSTON",
        320, 375,
    )

    assert _is_chapter_byline(byline.clean_content, byline, [chapter]) is True


def test_visual_legend_requires_image_caption_sandwich_and_repeated_lines():
    image = Block(5, 0, 0, "image", "<img>", (100, 100, 700, 500), None, 800, 1200)
    lines = [
        block(5, index, "text", content, 510 + index * 22, 530 + index * 22)
        for index, content in enumerate(
            ["African: L, L1, L2", "Asian: A, B, C", "Native American: A, B, C"]
        )
    ]
    caption = Block(
        5, 9, 9, "figure_title", "Map 5.1 Migration", (100, 590, 700, 650), None, 800, 1200
    )
    assert _is_visual_legend_line(lines[-1], [image, *lines, caption]) is True
    assert _is_visual_legend_line(lines[-1], [*lines, caption]) is False


def test_doc_title_child_uses_same_page_raw_coordinates():
    title = block(7, 1, "doc_title", "The domestication of fire", 100, 130)
    child = block(7, 2, "text", "Fire and hominins before domestication", 148, 171)
    body = block(
        7,
        3,
        "text",
        "This is ordinary body prose with enough words to prove that a real paragraph follows the short heading on the page.",
        182,
        420,
    )
    parent = Node(
        id="parent",
        kind="section",
        level=3,
        title_source=title.clean_content,
        page_json=7,
        page_print=1,
        parent_id="chapter",
        block_id=title.id,
        source="doc_title",
        confidence=0.92,
        bbox=title.bbox,
    )
    nodes = [parent]
    used = {title.id}
    _recover_text_children_below_doc_titles(
        [[] for _ in range(7)] + [[title, child, body]],
        nodes,
        used,
        PageMap(offset=-6, confidence=1.0, support=1, observations=1, print_to_json={}),
        "book",
    )
    assert len(nodes) == 2
    assert nodes[1].title_source == child.clean_content
    assert nodes[1].level == 4
    assert nodes[1].parent_id == parent.id
    assert "same_page_raw_coordinates" in nodes[1].evidence


def test_unnumbered_deeper_heading_uses_nearest_shallower_section_parent():
    chapter = Node("chapter", "chapter", 2, "Chapter", 1, 1, "book", None, "toc", 1.0, bbox=(0, 10, 1, 20))
    section = Node("section", "section", 3, "Main section", 2, 2, "chapter", None, "paragraph_title", 0.92, bbox=(0, 10, 1, 20))
    subsection = Node("subsection", "section", 4, "Smaller topic", 3, 3, "chapter", None, "paragraph_title", 0.88, bbox=(0, 10, 1, 20))
    sibling = Node("sibling", "section", 3, "Next main section", 4, 4, "chapter", None, "paragraph_title", 0.92, bbox=(0, 10, 1, 20))
    nodes = [chapter, section, subsection, sibling]

    _repair_unnumbered_section_parents(nodes, "book")

    assert subsection.parent_id == section.id
    assert "non_numbered_level_parent_repaired" in subsection.evidence
    assert sibling.parent_id == chapter.id
