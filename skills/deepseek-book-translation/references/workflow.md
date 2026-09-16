# Workflow reference

Run commands from `structured-book-translation-pipeline/`. Use absolute paths for user input and the project directory.

## New book

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
python3 new_book.py status --project '<PROJECT_DIR>'
```

Do not re-run `init` for a non-empty existing project. Read its status and resume.

## Structure gate

When status is `needs_structure_review`, inspect `structure/review_packets.jsonl`, the OCR blocks it cites, `validation.json`, and `recovered_outline.md`. Write one decision for every candidate to `structure_decisions.jsonl`, using only allowed actions and real evidence references. Then run:

```bash
python3 new_book.py compile-reviews --project '<PROJECT_DIR>' \
  --decisions '<PROJECT_DIR>/structure_decisions.jsonl'
python3 new_book.py prepare --project '<PROJECT_DIR>'
```

Repeat only when new evidence-backed packets remain. Do not convert an uncertain packet to `accept` just to clear the gate.

## Glossary gate

Review every row in `glossary_candidates.jsonl` against its contexts. `include` requires a stable Chinese translation and evidence; `reject` requires a reason. Compile the complete decision file:

```bash
python3 new_book.py compile-glossary --project '<PROJECT_DIR>' \
  --decisions '<PROJECT_DIR>/glossary_decisions.jsonl'
python3 new_book.py prepare --project '<PROJECT_DIR>'
```

Proceed only at `ready_to_translate`.

## Cover, preflight, and contextual translation

```bash
python3 new_book.py generate-cover --project '<PROJECT_DIR>' --theme auto
python3 new_book.py preflight --project '<PROJECT_DIR>'
python3 new_book.py translate --project '<PROJECT_DIR>' --target-completed 10
```

The generated configuration selects the contextual v2 translator with bounded 1/1 same-chapter context, a 300-character previous-translation tail, resumable failures, and one public-safe worker. Inspect the ten translations for completeness, terminology, title handling, and structural-token preservation. If they pass, continue cumulatively:

```bash
python3 new_book.py translate --project '<PROJECT_DIR>' --target-completed 100
python3 new_book.py translate --project '<PROJECT_DIR>' --target-completed 500
python3 new_book.py translate --project '<PROJECT_DIR>' --all
```

Account limits vary. Increase `translation.max_workers` only after a successful smoke run and only within the provider's documented limits. Never copy private-provider concurrency measurements into a public default.

## QA and publication

At full coverage:

```bash
python3 -m book_pipeline qa --config '<PROJECT_DIR>/translation_config.json'
python3 new_book.py render --project '<PROJECT_DIR>'
python3 new_book.py export --project '<PROJECT_DIR>' --format both
```

Review `likely_error` findings against the source; a flag is not proof of an error. Use the QA retranslation path only for verified failures. New projects require EPUBCheck for a deliverable EPUB; discovery checks the configured path, `EPUBCHECK_JAR`, `epubcheck` on PATH, and common package-manager locations. PDF also requires Pandoc, XeLaTeX, and Chinese fonts. If dependencies are missing, deliver Markdown and report the missing formats accurately.

## Existing local provider profile

The same workflow can operate in a private checkout whose configuration names another OpenAI-compatible provider. Keep that provider's endpoint, model, authentication header, rate limits, and environment-variable name in the private project or an untracked local Skill profile. Do not copy credentials, personal absolute paths, measured private-provider limits, or private book data into this public Skill.
