"""Workspace-discovery invariants for the sibling chapter repo (plan A1).

Proves that chapter-root resolution is path-independent:
  * DEFAULT_CHAPTER_ROOT is derived from book_pipeline's own location
    (ROOT.parent / "chapter-structure-recovery-lab"), never from a hard-coded
    absolute /Users/... literal in the source line that defines it;
  * _run_chapter honors a project-provided chapter_root (manifest override)
    and runs `python3 -m chapter_recovery analyze` with cwd = that root;
  * _run_chapter raises FileNotFoundError when the chapter_recovery package
    directory is missing from the resolved root.

No test here depends on a fixed /home/user/... path to pass: the runtime
value of DEFAULT_CHAPTER_ROOT legitimately lives under the user's home on this host,
but the derivation is relative and the source line contains no absolute literal.
"""
import sys
from pathlib import Path

import pytest

from book_pipeline import workflow


def test_default_chapter_root_derivation():
    assert workflow.DEFAULT_CHAPTER_ROOT == workflow.ROOT.parent / "chapter-structure-recovery-lab"


def test_default_chapter_root_source_has_no_absolute_literal():
    source = Path(workflow.__file__).read_text(encoding="utf-8")
    line = next(
        line for line in source.splitlines()
        if line.startswith("DEFAULT_CHAPTER_ROOT =")
    )
    assert "/Users" not in line
    assert "wahrfreiheit" not in line


def test_run_chapter_uses_manifest_chapter_root(tmp_path, monkeypatch):
    (tmp_path / "chapter_recovery").mkdir()
    recorded = {}

    def recorder(command, **kwargs):
        recorded["command"] = command
        recorded["cwd"] = kwargs.get("cwd")
        recorded["env"] = kwargs.get("env")

    monkeypatch.setattr(workflow.subprocess, "run", recorder)
    workflow._run_chapter({"chapter_root": str(tmp_path)}, ["analyze"])

    assert recorded["cwd"] == tmp_path.resolve()
    assert recorded["command"] == [sys.executable, "-m", "chapter_recovery", "analyze"]
    assert recorded["env"].get("PYTHONDONTWRITEBYTECODE") == "1"


def test_run_chapter_raises_when_chapter_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        workflow._run_chapter({"chapter_root": str(tmp_path)}, ["analyze"])
