from pathlib import Path

from chapter_recovery.models import Block
from chapter_recovery.toc import find_toc_pages, parse_toc_entries


def block(page, index, label, content):
    return Block(page, index, index, label, content, (0, 0, 700, 40), index, 800, 1200)


def test_parses_text_style_multpage_toc_with_inline_authors():
    pages = [
        [block(0, 1, "paragraph_title", "## Contents"),
         block(0, 2, "text", "1 · Introduction 1  AUTHOR NAME"),
         block(0, 3, "text", "PART I: METHODS")],
        [block(1, 1, "header", "Contents"),
         block(1, 2, "text", "2 Writing history 41 WRITER NAME")],
    ]
    entries = parse_toc_entries(pages, [0, 1])
    assert [(entry.kind, entry.print_page) for entry in entries] == [
        ("chapter", 1),
        ("part", None),
        ("chapter", 41),
    ]


def test_tolerates_ocr_number_punctuation_and_roman_misread():
    pages = [[
        block(0, 1, "paragraph_title", "## Contents"),
        block(0, 2, "text", "3\\. A chapter 56 AUTHOR NAME"),
        block(0, 3, "text", "II · Another chapter 261 AUTHOR NAME"),
    ]]
    entries = parse_toc_entries(pages, [0])
    assert [entry.kind for entry in entries] == ["chapter", "chapter"]


def test_parses_structural_toc_labels_page_less_chapter_and_mixed_ocr_page():
    pages = [[
        block(0, 0, "paragraph_title", "Contents"),
        block(0, 1, "doc_title", "I · Introduction: early cities AUTHOR NAME"),
        block(0, 2, "text", "6 · Urbanization and communication II3 HANS NISSEN"),
        block(0, 3, "paragraph_title", "## PART IV: POWER"),
    ]]

    entries = parse_toc_entries(pages, [0])

    assert [(entry.kind, entry.print_page) for entry in entries] == [
        ("chapter", None), ("chapter", 113), ("part", None)
    ]
    assert all(entry.title.casefold() != "contents" for entry in entries)


def test_parses_german_inhalt_with_teil_kapitel_and_deep_sections():
    pages = [[
        block(0, 1, "paragraph_title", "Inhalt"),
        block(
            0,
            2,
            "content",
            "TEIL 1: EINFÜHRUNGEN ..... 13\n"
            "1. Phänomenologie und Dialektik ..... 17\n"
            "1.1 Philosophie und Wissenschaft 21\n"
            "Kapitel II: Die Wahrnehmung ..... 35\n"
            "Literatur ..... 99",
        ),
    ]]
    toc_pages = find_toc_pages(pages, [0, 1])
    entries = parse_toc_entries(pages, toc_pages)

    assert toc_pages == [0]
    assert [entry.kind for entry in entries] == [
        "part", "chapter", "section", "chapter", "backmatter"
    ]


def test_joins_wrapped_german_part_title_without_joining_next_chapter():
    pages = [[
        block(0, 1, "paragraph_title", "Inhalt"),
        block(
            0,
            2,
            "content",
            "TEIL 4: EINFÜHRUNG IN DEN\n"
            "2. BAND ..... 13\n"
            "1. Vorbestimmung ..... 13\n"
            "TEIL 5: GEISTIGE PRAXISFORMEN\n"
            "IM KOOPERATIVEN VOLLZUG ..... 115",
        ),
    ]]
    entries = parse_toc_entries(pages, [0])

    assert [(entry.kind, entry.print_page, entry.title) for entry in entries] == [
        ("part", 13, "TEIL 4: EINFÜHRUNG IN DEN 2. BAND"),
        ("chapter", 13, "1. Vorbestimmung"),
        ("part", 115, "TEIL 5: GEISTIGE PRAXISFORMEN IM KOOPERATIVEN VOLLZUG"),
    ]
