# DeepSeek Book Translator

一个面向长篇书籍的可审计翻译工作台：OCR/PDF 路径用确定性规则、PDF bookmark/font evidence 与 DeepSeek Vision 联合恢复章节树，native EPUB 则直接保留原包结构翻译；经结构/兼容性与术语门禁后调用 DeepSeek 官方 API，支持断点续跑，并输出 Markdown、PDF 或原生回写后的 EPUB。

本仓库提供四种入口：

- **CLI**：适合希望逐步控制结构审核、术语和翻译批次的用户；
- **Coding Agent**：把预制提示词交给 Codex、Claude Code 等能够读写文件和运行命令的 agent；
- **Agent Skill**：安装仓库内的标准 Skill，让支持 Skill 的 agent 自动发现完整工作流；
- **Windows EXE**：通过图形界面选择 OCR JSON 或 reflowable EPUB、填写书名和参数；章节识别保留人工审核，术语默认由 DeepSeek 自动审核并决定译名。

仓库只包含程序、schema 和合成测试样例，没有真实书籍 OCR、译文、封面、运行记录、API key 或私人配置。书籍项目、译文和导出物默认在 Git 忽略范围内。

第一次使用可先看 [三页合成 OCR 入门样例](examples/README.md)。本项目代码采用 [MIT 许可证](LICENSE)；书籍原文、译文和用户提供的封面不因本仓库许可证而自动获得使用授权。贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，漏洞请按 [SECURITY.md](SECURITY.md) 私下报告。

## Contributors

- [@swjn2017USTC](https://github.com/swjn2017USTC) — 项目发起人和主要维护者

## 能力

- 从 PaddleOCR 页面数组恢复章节、标题层级、父子关系与稳定 ID；
- 对 OCR/PDF 项目增加 DeepSeek Vision 结构增强：识别真正的目录页、抄录目录层级并用页码映射定位正文，同时扫描规则/PDF 字体 evidence 选出的正文嫌疑页，补出目录未列出的次级节；
- 直接翻译 native EPUB：保留 OPF/spine/nav、DOM 层级、链接、脚注、表格、内嵌封面和非文本资源，只回写可翻译文本槽；
- 用验证报告和审核包阻止不可靠结构直接进入翻译；
- 生成书籍术语候选，并默认由 DeepSeek 自动执行 include/reject、统一译名和低置信度二次复核；全部决定与 usage 可审计并绑定到翻译缓存；
- 通过 `DEEPSEEK_API_KEY` 调用 DeepSeek 官方 Chat Completions API；
- 按累计目标断点续跑，保留逐段译文、完整重试 token usage、缓存命中率、推理 token 与成本估算；
- 校验脚注、HTML 注释、公式、URL、Markdown 表格等结构令牌；
- 根据书名、原文书名和作者，在本地生成 1600×2400 排版封面，不使用第三方美术素材；
- 渲染 Markdown，并在安装出版工具后导出带封面的 EPUB 与带目录/书签的 PDF。

## OCR/PDF 章节恢复架构

v0.9 起，章节恢复采用“多证据、单一树构建器”的方式，而不是把另一个文档解析器生成的树直接覆盖现有结果：

1. **PaddleOCR/PP-DocLayout evidence**：直接使用 `doc_title`、`paragraph_title`、`table of contents`、页码和 bbox；没有“Contents”文字但 layout model 已标成 TOC 的页面也能进入目录解析。
2. **PyMuPDF evidence**：读取 PDF bookmark、字体/大字号行，并负责把 PDF 页渲染成 Vision 输入。
3. **DeepSeek Vision**：识别真实 TOC 页、目录层级和页码；正文只扫描规则/PDF evidence 筛出的候选页，寻找 TOC 没列出的二级/三级标题。
4. **现有 deterministic recovery**：把 Vision sidecar 与 OCR/PDF 证据放回原来的 page-map、global monotone TOC alignment、numbering/style hierarchy 和 validator 中统一落树。
5. **人工结构门禁**：Vision 与 OCR 同页文本匹配时可以成为强证据；Vision 独自看到、OCR 完全没有文本匹配的标题只生成低置信度 review candidate。

这种设计吸收了 Docling heading-hierarchy 常用的 bookmark → numbering → visual style 信号顺序，但没有把整个 Docling 依赖栈打进 Windows EXE，以免和现有 chapter recovery 重复。需要最大召回时可在 `chapter_config.json` 中把 `vision.scan_all_body_pages` 设为 `true`；默认上限是 120 个正文候选页。

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

# 设置 key 后，用 Vision 在人工章节审核前补强目录和隐藏小节。
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python3 new_book.py vision-structure --project books/my-book
python3 new_book.py status --project books/my-book
```

`prepare` 保持纯离线：PaddleOCR label、页码映射、全局 TOC 对齐、编号和版式规则先给出 baseline。随后 `vision-structure` 才会联网并产生 API 费用。它优先读取与 OCR JSON 同名的 PDF，用 PyMuPDF 渲染页面；如果没有同名 PDF，则退回 OCR JSON 的 `inputImage`。也可以显式传入 `--pdf /path/to/source.pdf`。

Vision 会批量扫描前言区识别真正的 TOC 页并抄录目录项；正文阶段默认不会盲扫全书，而是结合 PaddleOCR 的 `paragraph_title/doc_title/table of contents` 类别、短标题几何、编号、PyMuPDF 大字号行和 TOC 目标页，先筛出候选页再看图。若追求最大召回，可在 `chapter_config.json` 设 `vision.scan_all_body_pages=true`。输出 `structure/vision_structure.json` 绑定 OCR SHA-256，并记录模型、扫描页、TOC、正文标题与 usage。

Vision 与 OCR 同页文本能匹配的标题可获得高置信证据；Vision 看见但 OCR 文本完全没有匹配的标题会保持低于自动通过阈值，仍交给人工章节审核。返回 `needs_structure_review` 时按 `RUNBOOK.md` 裁决；章节门禁通过后若返回 `needs_glossary_review`，继续自动术语审核：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python3 new_book.py auto-glossary --project books/my-book
python3 new_book.py prepare --project books/my-book
```

`auto-glossary` 会让 DeepSeek 覆盖全部候选，自动 include/reject 并决定统一译名；低置信度项自动做第二轮复核，并保存 `glossary_llm_review.jsonl` 与 `glossary_llm_usage.json`。人工术语窗口只作为异常或覆盖决定时的兜底。门禁通过后继续：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python3 new_book.py preflight --project books/my-book
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 new_book.py translate --project books/my-book --target-completed 10
python3 new_book.py status --project books/my-book
```

`preflight` 不联网，并会给出 contextual v2 的真实预计请求数（该模式下通常 1 个待翻译片段对应 1 次模型请求）。`translate` 才发送文本并可能产生费用；建议按 `10 → 100 → 500 → --all` 逐步扩大。10/100 段后优先检查 `cost.usage.reasoning_tokens`、`cache_hit_ratio`、`retry_rate` 与 `projected_total_cost_rmb_at_current_average`；若 reasoning token 非零或重试率异常，应先排查配置再继续整本。完成后运行：

```bash
python3 new_book.py render --project books/my-book
python3 translate_book.py qa --config books/my-book/translation_config.json
```

封面主题可选 `auto`、`ink`、`ocean`、`ember`、`forest`、`plum`。如果用户提供了有权使用的图片，也可以用 `new_book.py set-cover --help` 登记并替换自动封面。

PDF/EPUB 导出还需要 Pandoc、XeLaTeX、可用的中文字体和 EPUBCheck。新项目默认要求 EPUBCheck 通过才交付 EPUB；可用 `EPUBCHECK_JAR` 指向 jar：

```bash
python3 new_book.py export --project books/my-book --format both
```


### 直接翻译 EPUB（保留原 EPUB 结构）

如果手头已经有可重排（reflowable）的 EPUB，不需要先转成 PDF、截图或 OCR JSON。native EPUB 路径会直接读取 OCF/OPF、spine、导航与 XHTML，把正文和可翻译属性拆成稳定 segment，翻译后再写回原 DOM 槽位并重新打包。原始 EPUB 始终只读。

初始化时把 `--input-json` 换成 `--input-epub`：

```bash
cd structured-book-translation-pipeline
python3 new_book.py init \
  --input-epub /绝对路径/book.epub \
  --book-id my-epub-book \
  --book-title 'Original Title' \
  --book-title-zh '中文书名' \
  --author 'Author Name' \
  --source-lang '英语' \
  --target-lang '简体中文'

python3 new_book.py prepare --project books/my-epub-book
python3 new_book.py status --project books/my-epub-book
```

`prepare` 会先检查 ZIP/OCF、OPF manifest/spine、导航目标、内部链接、远程资源、脚本、fixed-layout、media overlay、加密状态等。当前正式翻译路径只接受 `COMPATIBLE_REFLOWABLE`；加密 EPUB、纯图片 EPUB、fixed-layout 或其他需要兼容性复核的包会 fail closed，并在 `structure/compatibility_report.json` 中说明原因。纯图片 EPUB 应先走 OCR 路径。

EPUB 中的 inline 标签会被编译成内部保护令牌，例如 `[[EPUB:0:OPEN:em]]`。这些令牌不是正文，不要手工删除或修改；翻译器会逐次验证它们的身份、顺序、嵌套关系以及 URL、脚注、表格和 DOM slot 是否仍可无歧义回写。

EPUB 也使用同一套自动术语门禁。`prepare` 返回 `needs_glossary_review` 时先自动审核，然后开始翻译：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python3 new_book.py auto-glossary --project books/my-epub-book
python3 new_book.py prepare --project books/my-epub-book
python3 new_book.py preflight --project books/my-epub-book
python3 new_book.py translate --project books/my-epub-book --target-completed 10
python3 new_book.py status --project books/my-epub-book

# 小样确认后逐步扩大
python3 new_book.py translate --project books/my-epub-book --target-completed 100
python3 new_book.py translate --project books/my-epub-book --all
```

完成率达到 100% 后直接 native render：

```bash
python3 new_book.py render --project books/my-epub-book
```

正式结果写入：

```text
books/my-epub-book/exports/my-epub-book.zh-CN.epub
```

也可以使用 `python3 new_book.py export --project books/my-epub-book --format epub`。native EPUB 路径不会通过 Pandoc 重新生成一本新书，也不会生成 PDF；它会保留原包中的图片、字体、CSS、封面和其他 opaque resource。只有被记录为可翻译的 XHTML 文本/属性槽以及必要的中文标题、语言 metadata 会发生受控变更。

正式 EPUB 交付默认要求 EPUBCheck。可把 `EPUBCHECK_JAR` 指向 EPUBCheck jar，或让 `epubcheck` 位于 PATH。**native EPUB 翻译本身不需要 Pandoc 或 XeLaTeX**；这两项只用于 OCR/Markdown 路径的 PDF/EPUB 出版。

native EPUB 输入现已同时通过 CLI、Coding Agent、Agent Skill 和 Windows EXE 暴露。Windows GUI 的同一个输入框可直接选择 `.json` 或 `.epub`；EPUB 会走 package/DOM-preserving 路径，不经过 OCR/Pandoc 重建。

## 方式二：Coding Agent

打开 [CODING_AGENT_PROMPT.md](CODING_AGENT_PROMPT.md)，把开头的 `<OCR_JSON_ABSOLUTE_PATH>` 替换成 OCR JSON 的绝对路径，然后把整段提示词交给能够访问本仓库和终端的 coding agent。

Agent 会先从书名页、版权页和目录推断书名、作者、语言、领域及 `book_id`，再依次执行：初始化、离线章节恢复、DeepSeek Vision 结构增强、人工结构审核、LLM 自动术语审核、本地封面生成、零网络预检、分级翻译、QA、渲染与导出。证据不足的章节结构必须留给用户；普通术语候选默认由 `auto-glossary` 自动 include/reject 和给出译名。

启动 agent 前应在 agent 进程能够继承的终端设置 `DEEPSEEK_API_KEY`。不要把 key 粘贴给 agent，也不要写进提示词或配置文件。

新项目默认使用 contextual v2：同章前后各 1 段上下文、最多 300 字上一段译文、失败段可续跑；默认 `thinking_mode=disabled`，翻译 JSON 使用 provider 原生 JSON mode，并将解析重试上限设为 3。公开默认保持单 worker，避免替用户假定 DeepSeek 账户速率额度。每次完成翻译后会在状态结果中附带 `cost` 诊断，汇总 cache hit/miss、reasoning tokens、retry rate 以及按当前均值估算的整本成本。

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
使用 $deepseek-book-translation，把 /绝对路径/book.json（或 book.epub）翻译成中文书籍项目。请从书中证据推断元数据，章节结构保持人工门禁，术语使用 DeepSeek 自动审核。
```

Skill 会定位本仓库，读取工作流说明；OCR/PDF 项目依次完成离线章节恢复、DeepSeek Vision 结构增强、人工结构审核，再进入 LLM 自动术语审核、零网络预检、分批翻译、QA、封面和出版物导出。它不会保存或提交 API key；结构证据不足时仍保持待审核，术语自动决定会写入独立审计与 usage 文件。通过 `gh skill` 安装的版本可使用下面的命令检查并更新：

```bash
gh skill update deepseek-book-translation
```

Skill 的文件结构、配置边界和本地私有版的区别见 [Agent Skill 方案](docs/AGENT_SKILLS.md)。

## 方式四：Windows EXE

Windows 图形程序名为 `DeepSeekBookTranslator.exe`。它提供：

- OCR JSON / reflowable EPUB 文件选择；
- 项目目录、Book ID、原文/中文书名、作者、语言和领域输入；
- DeepSeek 模型、思考模式（默认 disabled）及隐藏显示的 API key 输入；
- 初始化、**离线准备 + DeepSeek Vision 章节增强**、人工章节审核、DeepSeek 自动术语审核、可选人工术语复核、自动封面、零网络预检、分批翻译、状态、渲染和导出按钮；
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

生成文件位于 `dist\DeepSeekBookTranslator.exe`。EXE 自带 Python 运行时、两个项目、三页合成样例、Pillow 和 PyMuPDF，因此可直接从同名 PDF 渲染 Vision 页；PDF/EPUB 所需的 Pandoc、XeLaTeX、字体和 EPUBCheck 仍需另行安装。Windows PDF 默认使用 SimSun、Microsoft YaHei 和 Consolas 字体。

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

默认端点为 `https://api.deepseek.com/chat/completions`，正文翻译与 Vision 默认都使用 `deepseek-flash`，鉴权为 `Authorization: Bearer <key>`。Vision 模型可以通过 `DEEPSEEK_VISION_MODEL` 独立覆盖，默认显式关闭 thinking。key 只从 `DEEPSEEK_API_KEY` 环境变量读取；配置加载器拒绝配置文件中的密钥值。模型也可以通过 `DEEPSEEK_MODEL` 或书籍项目的 `provider.model` 设置。翻译默认显式关闭 thinking；可在项目配置的 `provider.thinking_mode` 或环境变量 `DEEPSEEK_THINKING_MODE` 中改为 `low`、`high`、`max`。对于普通长篇翻译建议保持 `disabled`，只在确有推理需要时开启。模型名称和 API 参数以 [DeepSeek 官方文档](https://api-docs.deepseek.com/guides/thinking_mode) 为准。

GUI 输入的 key 只放在当前进程内存和环境中，不保存到书籍项目。不要提交 `.env`、OCR、译文、术语库、封面或导出文件；使用自定义目录时仍应在 push 前检查 `git status` 和 `git diff --cached`。

## 测试

```bash
(cd chapter-structure-recovery-lab && python3 -m pytest -q)
(cd structured-book-translation-pipeline && python3 -m pytest -q)
python3 deepseek_book_translator_gui.py --self-test
python3 deepseek_book_translator_gui.py --offline-smoke
```

`--offline-smoke` 会在临时目录使用合成 OCR，经过初始化、章节恢复、**模拟 LLM 自动术语审核与编译**、封面、零网络预检、模拟翻译和 Markdown 渲染，并检查模拟译文无法进入正式导出。它不联网、不保存 key，也不验证真实 DeepSeek 翻译或 PDF/EPUB 工具。真实书籍回归和旧运行记录没有收入公开仓库，因此少量依赖私有语料的测试会跳过。在线 DeepSeek 翻译必须由用户使用自己的 key 和有权处理的样本验证。

## GitHub 发布

当前仓库已配置 SSH remote。提交本次更新后可以运行：

```bash
git push origin main
```

创建新的版本标签会触发 Windows EXE 构建并创建或更新对应 Release，例如：

```bash
git tag v0.5.0
git push origin v0.5.0
```
