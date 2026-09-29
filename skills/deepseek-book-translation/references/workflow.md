# Workflow reference

Run commands from `structured-book-translation-pipeline/`. Use absolute paths for user input and the project directory.

## New book (OCR/PDF or EPUB)

Inspect enough front matter and contents pages to infer the original title, a provisional Chinese title, author/editor when supported, source language, domain, `book_id`, and a reasonable `toc_search_end`. Keep the author empty when evidence is absent.

```bash
python3 new_book.py init \
  --input-json '<OCR_JSON>' \
  --book-id '<BOOK_ID>' \
  --book-title '<ORIGINAL_TITLE>' \
  --book-title-zh '<CHINESE_TITLE>' \
  --author '<AUTHOR_OR_EMPTY>' \
  --source-lang '<SOURCE_LANGUAGE>' \
  --target-lang '简体中文' \
  --domain '<DOMAIN>' \
  --toc-search-end '<PAGE_LIMIT>'
python3 new_book.py prepare --project '<PROJECT_DIR>'

# OCR/PDF projects: add multimodal structure evidence before human review.
python3 new_book.py vision-structure --project '<PROJECT_DIR>'
# If the PDF is not beside the OCR JSON with the same stem:
# python3 new_book.py vision-structure --project '<PROJECT_DIR>' --pdf '<SOURCE_PDF>'

python3 new_book.py status --project '<PROJECT_DIR>'
```

Do not re-run `init` for a non-empty existing project. Read its status and resume.

For an EPUB, do **not** route it through OCR or the legacy DOM-preserving adapter by default. New projects use the local canonical importer:

```bash
python3 new_book.py init --input-epub '<SOURCE_EPUB>' \
  --book-id '<BOOK_ID>' --book-title '<ORIGINAL_TITLE>' \
  --book-title-zh '<CHINESE_TITLE>' --source-lang '<SOURCE_LANGUAGE>' \
  --target-lang '简体中文' --domain '<DOMAIN>'
python3 new_book.py prepare --project '<PROJECT_DIR>'
python3 new_book.py status --project '<PROJECT_DIR>'
```

Expected adapter: `epub_markdown_v2`. Inspect:

- `structure/epub_import_report.json`;
- `work/source_manifest.json`;
- `source_review.md`;
- `assets/epub/`;
- `cover/cover.json`.

The importer keeps archive safety limits but tolerates common source-package/XHTML quirks and extracts reading-order semantics instead of preserving arbitrary DOM/CSS. It should not emit `[[EPUB:...]]` tokens. A source JPEG/PNG cover is reused automatically; otherwise a local typographic cover is generated. Local body images are copied into project assets.

If the EPUB is image-only or semantic extraction clearly loses substantial text, report that and switch the book to the OCR path rather than modifying translation prompts for the package dialect. Existing projects explicitly marked `epub_native_v1` may resume the legacy adapter, but do not initialize new projects that way.

## Structure vision gate

For OCR/PDF projects, `prepare` remains the zero-model deterministic baseline. `vision-structure` is a separate billable network step that uses DeepSeek Vision before human structure review.

The Vision collector prefers an explicit or same-stem PDF and renders selected pages with PyMuPDF; otherwise it uses local/data-URL OCR `inputImage` references when available. Remote http(s) image references are ignored unless `vision.allow_remote_input_images=true` is explicitly enabled. It scans the configured front-matter range for true TOC pages, extracts TOC entries/hierarchy, then scans a bounded shortlist of body pages selected from PaddleOCR heading labels, numbering, short-title geometry, PyMuPDF large-font evidence and TOC page targets. Set `chapter_config.json -> vision.scan_all_body_pages=true` only when maximal recall is worth the added image/API cost.

The result is `structure/vision_structure.json`. It is source-hash bound and records model, scanned pages, TOC rows, body headings and token usage. The deterministic analyzer consumes that sidecar. Vision headings grounded to same-page OCR text may gain high confidence; headings with no OCR text match remain below the normal automatic-accept threshold and must stay reviewable.

## Structure gate

When status is `needs_structure_review`, inspect `structure/review_packets.jsonl`, the OCR blocks it cites, `validation.json`, and `recovered_outline.md`. Write one decision for every candidate to `structure_decisions.jsonl`, using only allowed actions and real evidence references. Then run:

```bash
python3 new_book.py compile-reviews --project '<PROJECT_DIR>' \
  --decisions '<PROJECT_DIR>/structure_decisions.jsonl'
python3 new_book.py prepare --project '<PROJECT_DIR>'
```

Repeat only when new evidence-backed packets remain. Do not convert an uncertain packet to `accept` just to clear the gate.

## Reviewed source gate

After structure recovery/review is clear, rerun `prepare`. OCR writes page markers and post-structure cross-page stitching; EPUB import writes `EPUB source: <spine href>` markers. Both produce the same `source_review.md` contract with Markdown heading hierarchy and hidden stable `BOOK_SEGMENT` markers.

Before glossary or translation, ask the user whether they want to manually inspect/correct this file.

Manual path:

```bash
python3 new_book.py source-review --project '<PROJECT_DIR>' --mode manual
```

Tell the user the exact `source_review.md` path and stop. Do not call glossary or translation while they are reviewing. When the user explicitly says the review is complete:

```bash
python3 new_book.py apply-source-review --project '<PROJECT_DIR>'
```

The importer updates canonical segment text/source hashes and heading levels/parents, writes `work/reviewed_structure.json`, and then continues automatic terminology review.

Fully automatic path:

```bash
python3 new_book.py source-review --project '<PROJECT_DIR>' --mode auto
```

This approves the generated Markdown and continues terminology automation. Any later edit to `source_review.md` invalidates approval until it is reapplied.

## Glossary gate

Once the structure gate is clear, terminology review is automatic by default. With `DEEPSEEK_API_KEY` available, run:

```bash
python3 new_book.py auto-glossary --project '<PROJECT_DIR>'
python3 new_book.py prepare --project '<PROJECT_DIR>'
```

The deterministic candidate generator still defines the review set. DeepSeek must decide every candidate as `include` or `reject`; includes require a stable Chinese translation. Low-confidence first-pass decisions are automatically adjudicated a second time. The command writes `glossary_decisions.jsonl`, `glossary_llm_review.jsonl`, and `glossary_llm_usage.json`, then compiles the normal glossary gate.

Do not ask the user to manually review ordinary terminology candidates. Use the manual `compile-glossary` path only when the provider is unavailable, structured validation repeatedly fails, or the user explicitly wants to override an automatic decision. Proceed only at `ready_to_translate`.

## Cover, preflight, and contextual translation

For OCR projects, generate a cover when needed. For `epub_markdown_v2`, `prepare` already reuses the source JPEG/PNG cover or generates a fallback automatically.

```bash
# OCR only when no registered cover already exists:
python3 new_book.py generate-cover --project '<PROJECT_DIR>' --theme auto
python3 new_book.py preflight --project '<PROJECT_DIR>'
python3 new_book.py translate --project '<PROJECT_DIR>' --target-completed 10
```

The generated configuration selects contextual micro-batching: bounded 1/1 external same-chapter context, a 300-character previous-translation tail, up to 4 consecutive targets / 6000 source characters per ordinary batch, singleton handling for high-structure-risk items, and up to 4 chapter lanes. Valid siblings are salvaged when one target fails; only unresolved targets are recursively split/retried. Inspect the ten translations for completeness, terminology, title handling, and structural-token preservation. If they pass, continue cumulatively:

```bash
python3 new_book.py translate --project '<PROJECT_DIR>' --target-completed 100
python3 new_book.py translate --project '<PROJECT_DIR>' --target-completed 500
python3 new_book.py translate --project '<PROJECT_DIR>' --all
```

Preflight reports the actual planned batch count and average batch size. `provider.requests_per_minute=0` disables proactive pacing by default; bounded provider 429/network backoff remains active. Reduce `translation.max_workers` or set an explicit RPM cap if an account/provider needs stricter throttling.

## QA and publication

At full coverage:

```bash
python3 -m book_pipeline qa --config '<PROJECT_DIR>/translation_config.json'
python3 new_book.py render --project '<PROJECT_DIR>'
python3 new_book.py export --project '<PROJECT_DIR>' --format both
```

Review `likely_error` findings against the source; a flag is not proof of an error. Use the QA retranslation path only for verified failures. New projects require EPUBCheck for a deliverable EPUB; discovery checks the configured path, `EPUBCHECK_JAR`, `epubcheck` on PATH, and common package-manager locations. PDF also requires Pandoc, XeLaTeX, and Chinese fonts. If dependencies are missing, deliver Markdown and report the missing formats accurately.

For `epub_markdown_v2`, publication is intentionally standardized rather than source-DOM-preserving:

```bash
python3 new_book.py render --project '<PROJECT_DIR>'
python3 new_book.py export --project '<PROJECT_DIR>' --format epub
```

The translated Markdown is the publication source. Pandoc rebuilds EPUB3, using the registered source/fallback cover and extracted local images; EPUBCheck remains the formal delivery gate. Do not claim source CSS, fonts, scripts, complex links or interactive behavior were preserved unless independently verified.

Existing `epub_native_v1` projects keep their legacy package-preserving publication contract for resume compatibility only.

## Existing local provider profile

The same workflow can operate in a private checkout whose configuration names another OpenAI-compatible provider. Keep that provider's endpoint, model, authentication header, rate limits, and environment-variable name in the private project or an untracked local Skill profile. Do not copy credentials, personal absolute paths, measured private-provider limits, or private book data into this public Skill.
