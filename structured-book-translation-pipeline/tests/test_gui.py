from pathlib import Path

import pytest

from book_pipeline.gui import self_test, validate_book_id
from book_pipeline.workflow import initialize_project


@pytest.mark.parametrize("value", ["book", "book-2026", "book_v2", "a1"])
def test_validate_book_id_accepts_portable_names(value):
    assert validate_book_id(value) == value


@pytest.mark.parametrize("value", ["", "Book", "book name", "书名", "../book"])
def test_validate_book_id_rejects_unsafe_names(value):
    with pytest.raises(ValueError):
        validate_book_id(value)


def test_executable_self_test_imports_runtime_dependencies(monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository / "chapter-structure-recovery-lab"))
    result = self_test()
    assert result["status"] == "ok"
    assert result["chapter_recovery"] is True
    assert result["pillow"]


def test_frozen_project_manifest_does_not_store_temporary_package_path(tmp_path, monkeypatch):
    source = tmp_path / "book.json"
    source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr("book_pipeline.workflow.sys.frozen", True, raising=False)

    result = initialize_project(
        input_json=source,
        book_id="portable-book",
        book_title="Portable Book",
        book_title_zh="可移植书籍",
        project_dir=tmp_path / "project",
    )

    assert result["chapter_root"] is None
    runbook = (tmp_path / "project" / "RUNBOOK.md").read_text(encoding="utf-8")
    assert "python3 new_book.py prepare" in runbook
