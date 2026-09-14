# 三页合成 OCR 入门样例

[`sample_ocr.json`](sample_ocr.json) 是人工编写的三页 PaddleOCR 风格 JSON，包含章节标题、跨页段落、脚注、图像占位符和 Markdown 表格。它不含真实书籍文本，可以用来熟悉流程。

在仓库根目录安装依赖后，先运行完全离线的自动检查：

```bash
python deepseek_book_translator_gui.py --offline-smoke
```

预期 `status` 为 `ok`，页面数为 3，术语审核项为 4，封面生成、零网络预检、6 段模拟翻译和 Markdown 渲染完成；正式 EPUB 导出会拒绝这些模拟译文。这项检查不会调用 DeepSeek，不需要真实 API key，也不会保留书籍项目。

手动体验 GUI 时，运行 `python deepseek_book_translator_gui.py` 或 Windows EXE。选择本文件夹的 `sample_ocr.json`，填入一个仓库外的空项目目录，Book ID 用 `synthetic-sample`，原文书名用 `Synthetic Sample`，中文书名用 `合成样例`，作者可填 `Demo`。点击“初始化项目”→“准备/检查”。当前样例无需章节裁决，但会出现术语审核门禁。点击“术语审核”，逐项查看证据；这个合成样例里的候选词是标题片段或识别噪声，可以选择 `reject` 并写明理由。保存后点击“编译决定并重新准备”，状态应为 `ready_to_translate`。然后可生成封面。

真正在线翻译时，请使用自己有权处理的 OCR 文本和自己的 DeepSeek key。自动检查中的“译：”文本只是管线测试，**不是真实译文**，不能作为书籍成品。PDF/EPUB 还要求系统安装 Pandoc、XeLaTeX 和中文字体。Windows EXE 的冻结版离线检查由 GitHub Actions 运行；实际付费 API 调用和 PDF/EPUB 出版工具不在公开 CI 中执行。
