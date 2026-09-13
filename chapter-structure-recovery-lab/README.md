# Chapter Structure Recovery Lab

Deterministic chapter recovery from PaddleOCR JSON. The public copy contains the package, schemas, and synthetic tests. It does not need an API key. For a full new-book workflow, use the sibling `structured-book-translation-pipeline/` project and the repository root README.

Direct CLI: `python -m chapter_recovery analyze --config /path/to/chapter_config.json`. Review `validation.json` and `review_packets.jsonl` before sending content to translation.
