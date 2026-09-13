"""Static no-network import audit (plan amendment A2).

Reads the source text of the six new P02 evidence modules plus cli.py and
asserts none import network/LLM clients at module level.  fitz/rapidfuzz are
lazy imports inside functions by design, so a top-level ``import fitz`` or
``import rapidfuzz`` in any audited module is a violation and fails here.
This is the grep-able evidence for the spec exit gate "no external LLM".
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "chapter_recovery"

AUDITED_MODULES = [
    "evidence_models.py",
    "pdf_evidence.py",
    "candidate_fusion.py",
    "evidence_fusion.py",
    "structure_packets.py",
    "text_similarity.py",
    "cli.py",
]

# Network/LLM clients that must never be imported anywhere at module level.
BANNED_TOP_LEVEL = ("requests", "urllib", "http", "socket", "openai", "httpx")
BANNED_ANYWHERE = ("requests", "openai", "httpx", "socket")


def top_level_lines(source):
    """Module-level lines only: column-0 statements (skips function bodies)."""
    return [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.startswith(("#", " ", "\t"))
    ]


@pytest.mark.parametrize("filename", AUDITED_MODULES)
def test_no_network_or_llm_imports(filename):
    source = (PACKAGE / filename).read_text(encoding="utf-8")
    top = top_level_lines(source)
    imports = [
        line for line in top if line.startswith("import ") or line.startswith("from ")
    ]
    assert imports, f"{filename}: expected module-level imports"
    for line in imports:
        lowered = line.casefold()
        for banned in BANNED_TOP_LEVEL:
            assert f" {banned} " not in f" {lowered} ", f"{filename}: {line}"
            assert not lowered.startswith(f"from {banned}"), f"{filename}: {line}"
            assert not lowered.startswith(f"import {banned}"), f"{filename}: {line}"
    for banned in BANNED_ANYWHERE:
        assert banned not in source.casefold(), f"{filename}: {banned} referenced"


@pytest.mark.parametrize("filename", AUDITED_MODULES)
def test_no_top_level_lazy_backend_imports(filename):
    source = (PACKAGE / filename).read_text(encoding="utf-8")
    top = top_level_lines(source)
    for line in top:
        assert not line.startswith("import fitz"), f"{filename}: top-level fitz import"
        assert not line.startswith("import rapidfuzz"), (
            f"{filename}: top-level rapidfuzz import"
        )
