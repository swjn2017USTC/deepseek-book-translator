# Public release preparation notes

- Kept the two projects as sibling directories because the translation workflow discovers the structure-recovery package by relative path.
- Replaced runtime provider defaults with DeepSeek's official Chat Completions endpoint, `DEEPSEEK_API_KEY`, Bearer authentication, and the documented default model used by this copy.
- Added a deterministic local typographic cover generator. It renders a 1600x2400 PNG from project metadata and records dimensions, theme, provenance, and SHA-256 without third-party artwork.
- Added a Tkinter GUI for project creation, preparation, cover generation, preflight, bounded translation, status, render, export, and project-folder access.
- Added a PyInstaller spec, local PowerShell builder, and GitHub Actions Windows build with executable self-test and tagged-release upload.
- Added `CODING_AGENT_PROMPT.md`, a complete evidence-gated agent workflow from OCR metadata inference through export.
- Removed original `books/`, `runs/`, `work/`, `smoke/`, `audits/`, `corpus/`, `gold/`, reports, OMP instructions, and machine-specific configs.
- Replaced machine-specific source-library and glossary defaults. Each initialized book gets an empty local `glossary_master.json`; `BOOK_SOURCE_LIBRARY_ROOT` is optional.
- Kept source packages, schemas, CLI entry points, synthetic fixtures, and offline unit tests.
- Synchronized the evidence-gated glossary candidate output, including source segment evidence and occurrence counts, plus deterministic rejection of generic heading fragments.
- Synchronized bibliography/reference title handling so book and article titles are translated while authors, journals, publishers, URLs, and DOIs remain verifiable in the original form.
- Added bounded QA triage and review-packet export helpers so large automated QA runs produce a small, explicit human queue instead of requiring manual inspection of every signal.
- EPUB publication metadata now declares `zh-CN` for readers and validators.
- QA URL extraction now stops at attached full-width punctuation or CJK prose, preventing false high-severity URL mismatches in Chinese translations.
- Did not add a license. The project owner must choose one only after confirming rights to all retained source code.
