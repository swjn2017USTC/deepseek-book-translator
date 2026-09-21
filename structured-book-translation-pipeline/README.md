# Structured Book Translation Pipeline

清理 PaddleOCR 文本、通过 DeepSeek 官方 API 断点翻译，并渲染结构化书籍。相邻的 `chapter-structure-recovery-lab/` 提供章节树。

主要入口：

```bash
python3 new_book.py --help
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 ../deepseek_book_translator_gui.py
```

默认使用 `DEEPSEEK_API_KEY`、`https://api.deepseek.com/chat/completions`、`deepseek-v4-flash` 和 Bearer 鉴权。密钥不会写入项目。完整 CLI、Coding Agent 和 Windows EXE 文档见仓库根目录 README。
# Native EPUB structural layer

The public pipeline now includes the same fail-closed native EPUB structural
layer used by the private workflow. `book_pipeline.epub` can inspect an EPUB,
segment XHTML while preserving protected inline tokens, validate translated
DOM/resource/link invariants, and repack a standards-compliant EPUB. It is
provider-neutral: translation still uses the public DeepSeek configuration,
and no private USTC credentials or book artifacts are included.

The existing OCR/Pandoc workflow remains available for PDF/OCR sources. For a
native EPUB project, use the EPUB helpers before publishing and require a
100% current translation plus EPUBCheck validation; partial renders are
explicitly non-release artifacts.
