# Chapter Structure Recovery Lab

Auditable chapter recovery from PaddleOCR JSON. The core analyzer remains deterministic and can run without an API key; the parent book workflow can optionally collect a versioned DeepSeek Vision sidecar first, then fuse that evidence into the same deterministic tree builder. For a full new-book workflow, use the sibling `structured-book-translation-pipeline/` project and the repository root README.

Direct CLI: `python -m chapter_recovery analyze --config /path/to/chapter_config.json`. Review `validation.json` and `review_packets.jsonl` before sending content to translation.


## Vision-assisted evidence

The sibling translation workflow exposes:

```bash
python3 new_book.py prepare --project books/my-book
python3 new_book.py vision-structure --project books/my-book
```

Vision collection is separate from the analyzer. It prefers a same-stem PDF rendered through PyMuPDF, otherwise it can use PaddleOCR `inputImage` values. It identifies TOC pages and entries, then inspects a bounded shortlist of body pages for secondary headings omitted from the TOC. The sidecar is stored as `structure/vision_structure.json` and is bound to the OCR JSON SHA-256.

The deterministic analyzer consumes the sidecar only after validation. Same-page Vision/OCR text matches may be promoted as grounded heading evidence; an unmatched Vision title is deliberately kept below the normal automatic-accept threshold and remains a human-review candidate. PaddleOCR layout labels such as `table of contents` are also consumed directly.
