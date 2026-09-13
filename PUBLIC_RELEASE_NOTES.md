# Public release preparation notes

- Kept the two projects as sibling directories because the translation workflow discovers the structure-recovery package by relative path.
- Replaced runtime provider defaults with DeepSeek's official Chat Completions endpoint, `DEEPSEEK_API_KEY`, Bearer authentication, and the current documented default model used by this copy.
- Removed original `books/`, `runs/`, `work/`, `smoke/`, `audits/`, `corpus/`, `gold/`, reports, OMP instructions, and machine-specific configs. Those items included real-book-derived artifacts, private paths, internal workflow notes, or provider history.
- Replaced machine-specific source-library and glossary defaults. Each initialized book gets an empty local `glossary_master.json`; `BOOK_SOURCE_LIBRARY_ROOT` is optional.
- Kept source packages, schemas, CLI entry points, synthetic fixtures, and offline unit tests.
- Did not add a license. The project owner must choose one only after confirming rights to all retained source code.
- Initialized a local Git repository on `main` and created the first commit. No GitHub repository, remote, or push was created; the root README contains the final remote/push commands.
