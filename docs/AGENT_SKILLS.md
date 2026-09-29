# Agent Skill 方案

本仓库提供可安装的标准 Skill：[`skills/deepseek-book-translation`](../skills/deepseek-book-translation/SKILL.md)。它把端到端工作流、证据门禁、密钥边界和交付报告变成 coding agent 可发现的能力；`CODING_AGENT_PROMPT.md` 仍适合不支持 Skill 的 agent。

## 安装公开版 Skill

将 Skill 文件夹复制或软链接到 agent 的 Skill 目录。例如 Codex 使用：

```bash
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/deepseek-book-translation" ~/.codex/skills/deepseek-book-translation
```

然后可以说：

```text
使用 $deepseek-book-translation，把 /绝对路径/book.json（或 book.epub）翻译成中文书籍项目。
```

## 本地版是否也能做成 Skill

可以。公开版与本地版应共享章节、术语、翻译、QA 和出版门禁，但使用两个配置层：

- 公开 Skill 固定面向 DeepSeek 官方 API，并只读取 `DEEPSEEK_API_KEY`；支持
  PaddleOCR/PDF-derived 输入和原生 EPUB 输入，但原生 EPUB 必须走
  package/DOM-preserving 适配器及完整结构校验；OCR/PDF 路径在人工章节门禁前增加 DeepSeek Vision sidecar。章节门禁通过后生成带页码/跨页范围的 `source/structured_source.md`，开始时询问用户选择 manual（停住等人工核对）或 auto（自动确认）；确认后的 Markdown 成为术语与翻译的 source of truth。新项目正文默认同章 micro-batch（最多 4 target/6000 字符、局部 salvage、二分 fallback、最多 4 chapter lanes），术语候选仍默认由 DeepSeek 自动 include/reject 并决定统一译名；
- 本地 Skill 从私有项目配置读取 provider、模型、认证头、速率限制和本机工具路径；这些内容放在未跟踪的本地 reference 或私有 Skill 中。

不要把本地 API key、个人绝对路径、私有 provider 参数、真实 OCR 或批量运行记录复制到公开 Skill。若本地项目继续快速演化，可以让私有 Skill 的 `references/workflow.md` 指向本地项目的版本化运行说明，同时保留公开 Skill 作为稳定、可移植的发行版。
