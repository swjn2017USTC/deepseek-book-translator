from chapter_recovery.models import Node
from chapter_recovery.recover import _repair_geist_part_dependent_outline


def make_node(node_id, title, page, kind="section", ordinal=None):
    return Node(
        id=node_id,
        kind=kind,
        level=3,
        title_source=title,
        page_json=page,
        page_print=page,
        parent_id="book",
        block_id=f"block-{node_id}",
        source="paragraph_title",
        confidence=1.0,
        bbox=(0, 10, 100, 20),
        ordinal=ordinal,
    )


def test_geist_band1_profile_switches_grammar_by_part():
    book = make_node("book", "Book", 0, kind="book")
    part2 = make_node("part2", "Teil 2", 2, kind="part", ordinal="2")
    toc_marker = make_node("toc", "(A.) Bewusstsein", 3, kind="toc_entry")
    kapitel = make_node("kapitel", "Kapitel I", 4, kind="chapter", ordinal="I")
    source = make_node("source", "Die sinnliche Gewissheit", 5)
    nineteen = make_node("n19", "19. Commentary", 6, kind="chapter", ordinal="19")
    upper2 = make_node("upper2", "A. Source outline", 7, ordinal="A")
    twenty = make_node("n20", "20. Commentary", 8, kind="chapter", ordinal="20")
    part3 = make_node("part3", "Teil 3", 20, kind="part", ordinal="3")
    upper3 = make_node("upper3", "A. Beobachtende Vernunft", 21, ordinal="A")
    twenty_eight = make_node("n28", "28. Commentary", 22, kind="chapter", ordinal="28")
    lower3 = make_node("lower3", "a. Beobachtung der Natur", 23, ordinal="a")
    twenty_nine = make_node("n29", "29. Commentary", 24, kind="chapter", ordinal="29")
    nodes = [
        book, part2, toc_marker, kapitel, source, nineteen, upper2, twenty,
        part3, upper3, twenty_eight, lower3, twenty_nine,
    ]

    _repair_geist_part_dependent_outline(nodes, book.id)

    assert toc_marker.parent_id == book.id
    assert (nineteen.level, nineteen.parent_id) == (4, source.id)
    assert (upper2.level, upper2.parent_id) == (4, source.id)
    assert (twenty.level, twenty.parent_id) == (5, upper2.id)
    assert (upper3.level, upper3.parent_id) == (4, part3.id)
    assert (twenty_eight.level, twenty_eight.parent_id) == (5, upper3.id)
    assert (lower3.level, lower3.parent_id) == (5, upper3.id)
    assert (twenty_nine.level, twenty_nine.parent_id) == (6, lower3.id)
