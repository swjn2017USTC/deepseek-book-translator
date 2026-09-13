import pytest

from chapter_recovery.pdf_evidence import (
    discover_sibling_pdf,
    extract_book_pdf_evidence,
    large_line_threshold,
)


def test_large_line_threshold_formula():
    assert large_line_threshold(9.0, 14.0, 1.15) == pytest.approx(14.0)  # min size dominates
    assert large_line_threshold(16.0, 14.0, 1.15) == pytest.approx(18.4)  # ratio dominates
    assert large_line_threshold(0.0, 14.0, 1.15) == pytest.approx(14.0)


def test_missing_pdf_fallback():
    result = extract_book_pdf_evidence(None, 10)
    assert result.ok is False
    assert result.reason == "pdf_missing"
    assert result.pages == []
    assert result.bookmarks == []
    assert result.page_count == 0


def test_corrupt_pdf_fallback(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is definitely not a pdf file")
    result = extract_book_pdf_evidence(broken, 10)
    assert result.ok is False
    assert result.reason in {"pdf_corrupt", "fitz_unavailable"}
    assert result.pages == []
    assert result.bookmarks == []
    assert result.page_count == 0


def test_sibling_discovery_rule(tmp_path):
    json_path = tmp_path / "book.json"
    pdf_path = tmp_path / "book.pdf"
    json_path.write_text("[]", encoding="utf-8")
    assert discover_sibling_pdf(json_path) is None
    pdf_path.write_bytes(b"%PDF-1.4")
    assert discover_sibling_pdf(json_path) == pdf_path
    # a differently-stemmed PDF is NOT discovered (exact stem swap only)
    variant = tmp_path / "book (1997).json"
    variant.write_text("[]", encoding="utf-8")
    assert discover_sibling_pdf(variant) is None


def _write_synthetic_pdf(path):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for index in range(2):
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 100), "Body text line one", fontsize=10)
        page.insert_text((72, 130), "Body text line two", fontsize=10)
    page0 = document.load_page(0)
    page0.insert_text((72, 60), "Chapter One: The Beginning", fontsize=20, fontname="helv")
    page1 = document.load_page(1)
    page1.insert_text((72, 60), "Section One A", fontsize=16, fontname="helv")
    document.set_toc(
        [
            [1, "Chapter One: The Beginning", 1],
            [2, "Section One A", 2],
        ]
    )
    document.save(path)
    document.close()


def test_synthetic_pdf_extraction(tmp_path):
    fitz = pytest.importorskip("fitz")  # noqa: F841 -- gating import
    pdf_path = tmp_path / "synthetic.pdf"
    _write_synthetic_pdf(pdf_path)
    result = extract_book_pdf_evidence(pdf_path, page_count=2)
    assert result.ok is True
    assert result.reason == "ok"
    assert result.page_count == 2
    assert len(result.pages) == 2
    for page in result.pages:
        assert page.font_size_median > 0
        assert page.page_width > 0 and page.page_height > 0
    page0_lines = [line for line in result.pages[0].large_lines if "Chapter One" in line.text]
    assert page0_lines, "expected an oversized span on page 0 to be retained"
    assert page0_lines[0].font_size >= 16.0
    titles = [(b.level, b.title, b.page) for b in result.bookmarks]
    assert titles == [
        (1, "Chapter One: The Beginning", 1),
        (2, "Section One A", 2),
    ]
    # body-size lines must not qualify as large lines at the default threshold
    body_like = [
        line
        for page in result.pages
        for line in page.large_lines
        if line.text.startswith("Body text line")
    ]
    assert body_like == []
