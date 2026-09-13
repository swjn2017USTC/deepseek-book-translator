from pathlib import Path

from chapter_recovery.paddleocr import load_pages, normalize_blocks
from chapter_recovery.page_map import infer_page_map


FIXTURES = Path(__file__).parent / "fixtures"


def test_infers_print_page_offset_from_margin_numbers():
    pages = normalize_blocks(load_pages(FIXTURES / "sample_ocr.json"))
    page_map = infer_page_map(pages)
    assert page_map.offset == -1
    assert page_map.json_page(3) == 4
    assert page_map.support >= 5
