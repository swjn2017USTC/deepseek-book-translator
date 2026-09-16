# DeepSeek Book Translator

一个面向长篇书籍的可审计翻译工作台：从 PaddleOCR JSON 恢复章节树，经结构与术语门禁后调用 DeepSeek 官方 API 翻译，支持断点续跑、本地生成排版封面，并输出 Markdown、PDF 与 EPUB。

本仓库提供三种入口：

- **CLI**：适合希望逐步控制结构审核、术语和翻译批次的用户；
- **Coding Agent**：把预制提示词交给 Codex、Claude Code 等能够读写文件和运行命令的 agent；
- **Agent Skill**：安装仓库内的标准 Skill，让支持 Skill 的 agent 自动发现完整工作流；
- **Windows EXE**：通过图形界面选择 OCR JSON、填写书名和参数并运行主要步骤。

仓库只包含程序、schema 和合成测试样例，没有真实书籍 OCR、译文、封面、运行记录、API key 或私人配置。书籍项目、译文和导出物默认在 Git 忽略范围内。

第一次使用可先看 [三页合成 OCR 入门样例](examples/README.md)。本项目代码采用 [MIT 许可证](LICENSE)；书籍原文、译文和用户提供的封面不因本仓库许可证而自动获得使用授权。贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。

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

`prepare` 先运行章节恢复，再检查 `validation.json` 和 `review_packets.jsonl`。返回 `needs_structure_review` 时，按项目中的 `RUNBOOK.md` 完成结构裁决；返回 `needs_glossary_review` 时，审核术语候选并编译决定。也可以启动 GUI，在“章节审核”“术语审核”窗口逐项查看证据、保存裁决、编译并重新准备。门禁通过后设置自己的 key：

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

PDF/EPUB 导出还需要 Pandoc、XeLaTeX、可用的中文字体和 EPUBCheck。新项目默认要求 EPUBCheck 通过才交付 EPUB；可用 `EPUBCHECK_JAR` 指向 jar：

```bash
python3 new_book.py export --project books/my-book --format both
```

## 方式二：Coding Agent

打开 [CODING_AGENT_PROMPT.md](CODING_AGENT_PROMPT.md)，把开头的 `<OCR_JSON_ABSOLUTE_PATH>` 替换成 OCR JSON 的绝对路径，然后把整段提示词交给能够访问本仓库和终端的 coding agent。

Agent 会先从书名页、版权页和目录推断书名、作者、语言、领域及 `book_id`，再依次执行：初始化、章节恢复、结构审核、术语审核、本地封面生成、零网络预检、分级翻译、QA、Markdown 渲染与可选 PDF/EPUB 导出。证据不足的结构或术语必须留给用户，不允许为了跑通而猜测。

启动 agent 前应在 agent 进程能够继承的终端设置 `DEEPSEEK_API_KEY`。不要把 key 粘贴给 agent，也不要写进提示词或配置文件。

新项目默认使用 contextual v2：同章前后各 1 段上下文、最多 300 字上一段译文、失败段可续跑；公开默认保持单 worker，避免替用户假定 DeepSeek 账户速率额度。

## 方式三：Agent Skill

仓库内置 OpenAI/Codex Agent Skill：[`skills/deepseek-book-translation`](skills/deepseek-book-translation/SKILL.md)。它保留标准的 `SKILL.md`、`agents/openai.yaml` 和按需加载的 `references/`，用于让 agent 自动识别并执行本仓库的完整翻译流程。

推荐先预览 Skill 内容，再安装到 Codex 的用户级 Skill 目录：

```bash
gh skill preview swjn2017USTC/deepseek-book-translator deepseek-book-translation
gh skill install swjn2017USTC/deepseek-book-translator \
  deepseek-book-translation \
  --agent codex \
  --scope user
```

也可以通过 skills.sh 的安装器选择 Codex：

```bash
npx skills add swjn2017USTC/deepseek-book-translator
```

如果本机 GitHub CLI 尚不支持 `gh skill`，可克隆仓库后手动安装：

```bash
git clone https://github.com/swjn2017USTC/deepseek-book-translator.git
mkdir -p ~/.codex/skills
cp -R deepseek-book-translator/skills/deepseek-book-translation ~/.codex/skills/
```

安装后重新启动 Codex 会话，设置让 Codex 进程可以继承的 `DEEPSEEK_API_KEY`，然后直接提出任务，例如：

```text
使用 $deepseek-book-translation，把 /绝对路径/book.json 翻译成中文书籍项目。请从书中证据推断元数据，并保留所有结构与术语审核门禁。
```

Skill 会定位本仓库，读取工作流说明，依次完成初始化、章节恢复、结构与术语审核、零网络预检、分批翻译、QA、封面和出版物导出。它不会保存或提交 API key；证据不足的书籍信息、结构和术语会保持待审核状态。通过 `gh skill` 安装的版本可使用下面的命令检查并更新：

```bash
gh skill update deepseek-book-translation
```

Skill 的文件结构、配置边界和本地私有版的区别见 [Agent Skill 方案](docs/AGENT_SKILLS.md)。

## 方式四：Windows EXE

Windows 图形程序名为 `DeepSeekBookTranslator.exe`。它提供：

- OCR JSON 文件选择；
- 项目目录、Book ID、原文/中文书名、作者、语言和领域输入；
- DeepSeek 模型及隐藏显示的 API key 输入；
- 初始化、准备、逐项章节/术语审核、自动封面、零网络预检、分批翻译、状态、Markdown 渲染和 PDF/EPUB 导出按钮；
- 运行日志和项目目录快捷打开。

### 下载预编译 EXE

每次推送 `main` 后，GitHub Actions 的 **Build Windows EXE** 工作流会在真实 `windows-latest` 环境中运行测试、构建、GUI 自检及冻结 EXE 的合成 OCR 离线流程检查。在仓库的 **Actions → Build Windows EXE → Artifacts** 下载 `DeepSeekBookTranslator-Windows` 并解压即可。

带 `v*` 标签的构建还会把 EXE 附加到对应 GitHub Release。当前程序没有代码签名证书，Windows SmartScreen 可能显示“未知发布者”；可以检查对应提交的 Actions 构建记录后再运行。

### 在 Windows 本机构建

安装 Python 3.11 后，在 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\windows\build_windows.ps1
```

生成文件位于 `dist\DeepSeekBookTranslator.exe`。EXE 自带 Python 运行时、两个项目、三页合成样例和 Pillow；PDF/EPUB 所需的 Pandoc、XeLaTeX、字体和 EPUBCheck 仍需另行安装。Windows PDF 默认使用 SimSun、Microsoft YaHei 和 Consolas 字体。

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
python3 deepseek_book_translator_gui.py --offline-smoke
```

`--offline-smoke` 会在临时目录使用合成 OCR，经过初始化、章节恢复、术语门禁与裁决、封面、零网络预检、模拟翻译和 Markdown 渲染，并检查模拟译文无法进入正式导出。它不联网、不保存 key，也不验证真实 DeepSeek 翻译或 PDF/EPUB 工具。真实书籍回归和旧运行记录没有收入公开仓库，因此少量依赖私有语料的测试会跳过。在线 DeepSeek 翻译必须由用户使用自己的 key 和有权处理的样本验证。

## GitHub 发布

当前仓库已配置 SSH remote。提交本次更新后可以运行：

```bash
git push origin main
```

创建新的版本标签会触发 Windows EXE 构建并创建或更新对应 Release，例如：

```bash
git tag v0.4.0
git push origin v0.4.0
```
