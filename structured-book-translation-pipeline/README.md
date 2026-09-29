# Structured Book Translation Pipeline

处理 PaddleOCR JSON 或 native reflowable EPUB，通过 DeepSeek 官方 API 断点翻译，并渲染/回写结构化书籍。OCR 输入先用 `chapter-structure-recovery-lab` + DeepSeek Vision/PyMuPDF 恢复章节，再生成带页码、可人工修改确认的 `source/structured_source.md`；确认后的 Markdown 成为 OCR 术语/上下文/翻译 source of truth。EPUB 始终以 package/DOM slots 为 source of truth，不经过可编辑 Markdown 重建。

主要入口：

```bash
python3 new_book.py --help
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 ../deepseek_book_translator_gui.py
```

默认使用 `DEEPSEEK_API_KEY`、`https://api.deepseek.com/chat/completions`、`deepseek-flash` 和 Bearer 鉴权。OCR/PDF 项目可运行 `python3 new_book.py vision-structure --project <PROJECT>`；Vision 模型可用 `DEEPSEEK_VISION_MODEL` 单独覆盖。密钥不会写入项目。章节结构审核保持人工门禁；结构通过后默认运行 `python3 new_book.py auto-glossary --project <PROJECT>`，让 DeepSeek 自动完成术语 include/reject 与统一译名，并写出可审计记录。完整 CLI、Coding Agent 和 Windows EXE 文档见仓库根目录 README。
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


## OCR source confirmation

新 OCR 项目初始化时可选择：

```bash
python3 new_book.py init ... --source-confirmation manual
# 或
python3 new_book.py init ... --source-confirmation auto
```

章节结构门禁通过后 `prepare` 会做一次后结构跨页合并并生成 `source/structured_source.md`。manual 模式停在 `needs_source_confirmation`；人工核对保存后运行 `python3 new_book.py confirm-source --project <PROJECT>`。确认后再改 Markdown 会被 provenance 检测并重新阻止翻译。

`preflight` 会按实际 micro-batch 规则返回 `estimated_remaining_requests` 和 `estimated_targets_per_request`，用于在真正调用模型前观察合批收益。
