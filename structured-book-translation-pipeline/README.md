# Structured Book Translation Pipeline

清理 PaddleOCR 文本、通过 DeepSeek 官方 API 断点翻译，并渲染结构化书籍。相邻的 `chapter-structure-recovery-lab/` 提供章节树。

主要入口：

```bash
python3 new_book.py --help
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 ../deepseek_book_translator_gui.py
```

默认使用 `DEEPSEEK_API_KEY`、`https://api.deepseek.com/chat/completions`、`deepseek-v4-flash` 和 Bearer 鉴权。密钥不会写入项目。完整 CLI、Coding Agent 和 Windows EXE 文档见仓库根目录 README。
