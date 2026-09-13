# DeepSeek Book Translator

一个面向长篇书籍的可审计翻译工作台：从 PaddleOCR JSON 恢复章节树，经结构与术语门禁后调用 DeepSeek 官方 API 翻译，支持断点续跑、本地生成排版封面，并输出 Markdown、PDF 与 EPUB。

本仓库提供三种入口：

- **CLI**：适合希望逐步控制结构审核、术语和翻译批次的用户；
- **Coding Agent**：把预制提示词交给 Codex、Claude Code 等能够读写文件和运行命令的 agent；
- **Windows EXE**：通过图形界面选择 OCR JSON、填写书名和参数并运行主要步骤。

仓库只包含程序、schema 和合成测试样例，没有真实书籍 OCR、译文、封面、运行记录、API key 或私人配置。书籍项目、译文和导出物默认在 Git 忽略范围内。

## 能力

- 从 PaddleOCR 页面数组恢复章节、标题层级、父子关系与稳定 ID；
- 用验证报告和审核包阻止不可靠结构直接进入翻译；
- 生成并审核书籍术语表，将术语版本绑定到翻译缓存；
- 通过 `DEEPSEEK_API_KEY` 调用 DeepSeek 官方 Chat Completions API；
- 按累计目标断点续跑，保留逐段译文和 token usage；
- 校验脚注、HTML 注释、公式、URL、Markdown 表格等结构令牌；
- 根据书名、原文书名和作者，在本地生成 1600×2400 排版封面，不使用第三方美术素材；
- 渲染 Markdown，并在安装出版工具后导出带封面的 EPUB 与带目录/书签的 PDF。

## 方式一：CLI

需要 Python 3.9+。在仓库根目录安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./chapter-structure-recovery-lab -e ./structured-book-translation-pipeline
python -m pip install pytest
```

Windows PowerShell 激活虚拟环境使用：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\chapter-structure-recovery-lab -e .\structured-book-translation-pipeline
```

准备 PaddleOCR 导出的 JSON，顶层须为页面数组。只处理你有权处理和发送给第三方 API 的文本。进入翻译子项目并初始化：

```bash
cd structured-book-translation-pipeline
python3 new_book.py init \
  --input-json /绝对路径/自己的书.json \
  --book-id my-book \
  --book-title 'Original Title' \
  --book-title-zh '中文书名' \
  --author 'Author Name'
python3 new_book.py prepare --project books/my-book
python3 new_book.py status --project books/my-book
```

`prepare` 先运行章节恢复，再检查 `validation.json` 和 `review_packets.jsonl`。返回 `needs_structure_review` 时，按项目中的 `RUNBOOK.md` 完成结构裁决；返回 `needs_glossary_review` 时，审核术语候选并编译决定。门禁通过后设置自己的 key：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python3 new_book.py preflight --project books/my-book
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 new_book.py translate --project books/my-book --target-completed 10
python3 new_book.py status --project books/my-book
```

`preflight` 不联网。`translate` 才发送文本并可能产生费用；建议按 `10 → 100 → 500 → --all` 逐步扩大。完成后运行：

```bash
python3 new_book.py render --project books/my-book
python3 translate_book.py qa --config books/my-book/translation_config.json
```

封面主题可选 `auto`、`ink`、`ocean`、`ember`、`forest`、`plum`。如果用户提供了有权使用的图片，也可以用 `new_book.py set-cover --help` 登记并替换自动封面。

PDF/EPUB 导出还需要 Pandoc、XeLaTeX、可用的中文字体和可选 epubcheck：

```bash
python3 new_book.py export --project books/my-book --format both
```

## 方式二：Coding Agent

打开 [CODING_AGENT_PROMPT.md](CODING_AGENT_PROMPT.md)，把开头的 `<OCR_JSON_ABSOLUTE_PATH>` 替换成 OCR JSON 的绝对路径，然后把整段提示词交给能够访问本仓库和终端的 coding agent。

Agent 会先从书名页、版权页和目录推断书名、作者、语言、领域及 `book_id`，再依次执行：初始化、章节恢复、结构审核、术语审核、本地封面生成、零网络预检、分级翻译、QA、Markdown 渲染与可选 PDF/EPUB 导出。证据不足的结构或术语必须留给用户，不允许为了跑通而猜测。

启动 agent 前应在 agent 进程能够继承的终端设置 `DEEPSEEK_API_KEY`。不要把 key 粘贴给 agent，也不要写进提示词或配置文件。

## 方式三：Windows EXE

Windows 图形程序名为 `DeepSeekBookTranslator.exe`。它提供：

- OCR JSON 文件选择；
- 项目目录、Book ID、原文/中文书名、作者、语言和领域输入；
- DeepSeek 模型及隐藏显示的 API key 输入；
- 初始化、准备、自动封面、零网络预检、分批翻译、状态、Markdown 渲染和 PDF/EPUB 导出按钮；
- 运行日志和项目目录快捷打开。

### 下载预编译 EXE

每次推送 `main` 后，GitHub Actions 的 **Build Windows EXE** 工作流会在真实 `windows-latest` 环境中运行测试、构建并执行 `--self-test`。在仓库的 **Actions → Build Windows EXE → Artifacts** 下载 `DeepSeekBookTranslator-Windows` 并解压即可。

带 `v*` 标签的构建还会把 EXE 附加到对应 GitHub Release。当前程序没有代码签名证书，Windows SmartScreen 可能显示“未知发布者”；可以检查对应提交的 Actions 构建记录后再运行。

### 在 Windows 本机构建

安装 Python 3.11 后，在 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows\build_windows.ps1
```

生成文件位于 `dist\DeepSeekBookTranslator.exe`。EXE 自带 Python 运行时、两个项目和 Pillow；PDF/EPUB 所需的 Pandoc、XeLaTeX 与字体仍需另行安装。

也可以在有桌面环境的 Python 安装中直接启动 GUI：

```bash
python3 deepseek_book_translator_gui.py
```

## 自动封面

`generate-cover` 根据项目元数据确定配色并绘制本地几何图形和排版，不调用网络或图像生成 API。生成文件会写入项目的 `cover/`，并登记：

- 图片 SHA-256；
- 1600×2400 像素尺寸；
- 使用的主题和生成器版本；
- `rights_status: generated`；
- `uses_third_party_art: false`。

系统会优先寻找 Windows 的 Microsoft YaHei/SimHei、macOS 的 PingFang 或 Linux 的 Noto CJK 字体。可以通过 `BOOK_COVER_FONT` 指定本地字体文件。

## DeepSeek 配置与安全

默认端点为 `https://api.deepseek.com/chat/completions`，默认模型为 `deepseek-v4-flash`，鉴权为 `Authorization: Bearer <key>`。key 只从 `DEEPSEEK_API_KEY` 环境变量读取；配置加载器拒绝配置文件中的密钥值。模型也可以通过 `DEEPSEEK_MODEL` 或书籍项目的 `provider.model` 设置。模型名称和 API 参数以 [DeepSeek 官方文档](https://api-docs.deepseek.com/guides/function_calling) 为准。

GUI 输入的 key 只放在当前进程内存和环境中，不保存到书籍项目。不要提交 `.env`、OCR、译文、术语库、封面或导出文件；使用自定义目录时仍应在 push 前检查 `git status` 和 `git diff --cached`。

## 测试

```bash
(cd chapter-structure-recovery-lab && python3 -m pytest -q)
(cd structured-book-translation-pipeline && python3 -m pytest -q)
python3 deepseek_book_translator_gui.py --self-test
```

真实书籍回归和旧运行记录没有收入公开仓库，因此少量依赖私有语料的测试会跳过。在线 DeepSeek 翻译必须由用户使用自己的 key 和有权处理的样本验证。

## GitHub 发布

当前仓库已配置 SSH remote。提交本次更新后可以运行：

```bash
git push origin main
```

创建版本标签会触发 Windows EXE 构建并创建或更新对应 Release：

```bash
git tag v0.2.0
git push origin v0.2.0
```

本仓库暂未附加许可证。公开代码不等于授予再使用权；仓库所有者应在确认全部代码权属后选择许可证。
