# Structured Book Translation Pipeline

处理 PaddleOCR JSON/PDF 或 EPUB，通过 DeepSeek 官方 API 断点翻译。OCR 使用章节恢复 + Vision；新 EPUB 使用 tolerant `epub_markdown_v2` importer，先转成 canonical segments / reviewed-source Markdown / local assets，再与 OCR 共享术语、micro-batch 翻译、QA 与 Pandoc EPUB3 出版。

主要入口：

```bash
python3 new_book.py --help
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 ../deepseek_book_translator_gui.py
```

默认使用 `DEEPSEEK_API_KEY`、`https://api.deepseek.com/chat/completions`、`deepseek-flash` 和 Bearer 鉴权。OCR/PDF 项目可运行 `python3 new_book.py vision-structure --project <PROJECT>`；Vision 模型可用 `DEEPSEEK_VISION_MODEL` 单独覆盖。密钥不会写入项目。章节结构审核保持人工门禁；结构稳定后先生成带页码的 `source_review.md`，让用户选择人工核对或自动批准。批准版本成为 canonical source，之后术语自动审核并使用同章 contextual micro-batch（默认最多 4 targets/6000 字符）翻译。完整 CLI、Coding Agent 和 Windows EXE 文档见仓库根目录 README。
# EPUB canonical import layer

New EPUB projects use `book_pipeline.epub.normalize`, not source-DOM patching. The importer keeps archive safety checks, discovers OPF/spine tolerantly, extracts semantic text and local images, marks source spine locations in `source_review.md`, and automatically reuses a source JPEG/PNG cover or generates a fallback.

After review and translation, `render_book` produces the same canonical Markdown used by OCR projects and `export_book` rebuilds a clean EPUB3 with Pandoc + EPUBCheck. Source CSS/fonts/scripts and arbitrary interactive DOM semantics are intentionally not publication invariants.

The legacy `epub_native_v1` implementation remains in the package only so existing in-progress projects can resume without migration.
