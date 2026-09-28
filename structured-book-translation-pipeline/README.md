# Structured Book Translation Pipeline

处理 PaddleOCR JSON 或 native reflowable EPUB，通过 DeepSeek 官方 API 断点翻译，并渲染/回写结构化书籍。OCR 输入使用相邻的 `chapter-structure-recovery-lab/` 恢复章节树；EPUB 使用 package/DOM-preserving 适配器。

主要入口：

```bash
python3 new_book.py --help
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 ../deepseek_book_translator_gui.py
```

默认使用 `DEEPSEEK_API_KEY`、`https://api.deepseek.com/chat/completions`、`deepseek-flash` 和 Bearer 鉴权。密钥不会写入项目。章节结构审核保持人工门禁；结构通过后默认运行 `python3 new_book.py auto-glossary --project <PROJECT>`，让 DeepSeek 自动完成术语 include/reject 与统一译名，并写出可审计记录。完整 CLI、Coding Agent 和 Windows EXE 文档见仓库根目录 README。
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
