# DeepSeek Book Translator

这是一个可自行运行的书籍 OCR 结构恢复与翻译流水线。两个 Python 项目保持并列：`chapter-structure-recovery-lab/` 从 PaddleOCR JSON 恢复可审计的章节树，`structured-book-translation-pipeline/` 在结构和术语审核通过后，调用 DeepSeek 官方 Chat Completions API 翻译，并按结构树渲染 Markdown。新书工作目录、OCR 原文、译文及导出文件都在 Git 忽略范围内。

本公开版只包含程序、schema 和合成测试样例；没有原开发环境中的真实书籍 OCR、译文、封面、运行记录或私有配置。它没有使用任何原项目中的 API key。基于这两个项目已有的代码制作；制作过程只读取原目录，所有改动都写在这个新目录中。

## 安装

需要 Python 3.9+。在**本文件所在目录**运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./chapter-structure-recovery-lab -e ./structured-book-translation-pipeline
python -m pip install pytest
```

PDF/EPUB 导出还需要原流水线调用的 Pandoc、XeLaTeX 与可用的中文字体；Markdown 翻译流程不依赖它们。

## 处理自己的书

准备 PaddleOCR 导出的 JSON：顶层须为页面数组。只处理你有权处理和发送给第三方 API 的文本。从仓库根目录运行：

```bash
cd structured-book-translation-pipeline
python new_book.py init \
  --input-json /绝对路径/自己的书.json \
  --book-id my-book \
  --book-title 'Original Title' \
  --book-title-zh '中文书名'
python new_book.py prepare --project books/my-book
python new_book.py status --project books/my-book
```

`prepare` 会先运行章节恢复，然后检查 `validation.json` 与 `review_packets.jsonl`。如果返回 `needs_structure_review`，按照生成的 `books/my-book/RUNBOOK.md` 完成证据审核并编译裁决，再重新 `prepare`。如果返回 `needs_glossary_review`，审核 `glossary_candidates.jsonl`，编译术语裁决，再重新 `prepare`。生成的 `glossary_master.json` 初始是空列表；可以加入你自己有权使用的术语数据。

在**运行命令的同一个终端**提供自己的 DeepSeek key：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python new_book.py preflight --project books/my-book
python new_book.py translate --project books/my-book --target-completed 10
python new_book.py status --project books/my-book
```

`preflight` 不发送网络请求；它验证密钥格式、质量门禁和剩余任务。`translate` 才会把待译文本发送至 DeepSeek 官方 API，产生可能计费的请求。`--target-completed 10` 是累计上限，重跑可断点续跑。检查结果后可逐步提高上限，或明确执行 `--all`。完成后运行：

```bash
python new_book.py render --project books/my-book
```

如需 PDF/EPUB，先用 `new_book.py set-cover --help` 登记有使用权的封面，再运行 `new_book.py export --project books/my-book --format both`。

## DeepSeek 配置与安全

默认请求发往 DeepSeek 官方 `https://api.deepseek.com/chat/completions`，默认模型为 `deepseek-v4-flash`，鉴权为 `Authorization: Bearer <key>`。key 只从 `DEEPSEEK_API_KEY` 环境变量读取；程序拒绝把密钥值写进 JSON 配置。模型可用性及 API 参数以 [DeepSeek 官方文档](https://api-docs.deepseek.com/guides/function_calling) 为准。需要另一个官方模型时，设置 `DEEPSEEK_MODEL`，或修改新建项目的 `translation_config.json` 中 `provider.model`；v2 结构/术语流程还可在 `llm_roles` 中指定模型。

不要把 `.env`、OCR、译文、术语库、封面或导出文件提交到 Git。根目录 `.gitignore` 已排除常见位置；如果把这些内容放到其他位置，提交前仍须人工检查 `git status` 与 `git diff --cached`。公开代码的授权方式需要项目所有者自行决定；此目录暂不附加许可证。

## 验证与发布

在仓库根目录运行离线测试：

```bash
(cd chapter-structure-recovery-lab && python3 -m pytest -q)
(cd structured-book-translation-pipeline && python3 -m pytest -q)
```

真实书籍回归与旧运行记录未包含在公开版，因此相应测试被排除。没有使用真实 DeepSeek key 做在线调用；发布前建议用自己的 key 和有权处理的短样本完成一次小规模翻译验证。

此交付目录已在 `main` 分支创建首个本地提交。检查内容并在 GitHub 创建空仓库后，从本目录运行（把 URL 改为自己的仓库）：

```bash
git status --short
git log -1 --oneline
git remote add origin git@github.com:YOUR_NAME/YOUR_REPO.git
git push -u origin main
```
