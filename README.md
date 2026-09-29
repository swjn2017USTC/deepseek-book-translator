# DeepSeek Book Translator

一个面向长篇书籍的可审计翻译工作台：OCR/PDF 先恢复章节树，EPUB 先容错标准化为统一 reviewed-source Markdown/canonical segments；两类输入随后共享术语、contextual micro-batch 翻译、QA 与出版流程，并输出 Markdown、PDF 或标准化重建的 EPUB3。

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
- EPUB 默认先做 tolerant semantic import：按 spine 阅读顺序抽取标题、段落、列表、引用、脚注与本地图片到 canonical segments/Markdown，不再把源 DOM token 带进翻译 prompt；
- 用验证报告和审核包阻止不可靠结构直接进入翻译；
- 生成书籍术语候选，并默认由 DeepSeek 自动执行 include/reject、统一译名和低置信度二次复核；全部决定与 usage 可审计并绑定到翻译缓存；
- 通过 `DEEPSEEK_API_KEY` 调用 DeepSeek 官方 Chat Completions API；
- 章节稳定后生成带源页码的 `source_review.md`，可选择人工核对/修补标题层级和原文；批准版本成为 canonical translation source；
- 普通同章连续文本默认使用最多 4 target / 6000 字符的 contextual micro-batch，共享上下文；高结构风险片段保持 singleton，失败时 partial salvage + recursive fallback；
- 按累计目标断点续跑，保留逐段译文、完整重试 token usage、缓存命中率、推理 token、batch 计划与成本估算；
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

## Reviewed Source 与翻译效率

v0.10 把“segment”与“API request”解耦。segment 仍是断点续跑、source hash、QA、出版和缓存失效的原子单位，但多个同章连续 segment 可以作为一个 wire micro-batch 请求。v0.11 又把 EPUB 输入也统一到这套 canonical segment 模型。

章节结构/Vision/人工结构门禁通过后，`prepare` 会先做保守跨页 paragraph 合并，再生成 `source_review.md`。文件包含可读的 Markdown 标题层级、`Source page(s)` 标记，以及不可删除/复制的隐藏 `BOOK_SEGMENT` 标记。此时流程必须选择：

```bash
# 人工核对：暂停后续自动流程
python3 new_book.py source-review --project books/my-book --mode manual
# 编辑 source_review.md 后：
python3 new_book.py apply-source-review --project books/my-book

# 或完全自动：
python3 new_book.py source-review --project books/my-book --mode auto
```

人工导入会更新 canonical segment 文本/source hash 和标题层级，并写出 `work/reviewed_structure.json`；以后 micro-batch 使用的就是这个批准版本。批准后继续修改 Markdown 会自动使 gate 失效，必须再次应用，避免“看起来改过但模型实际没读”的情况。

默认翻译参数为普通文本最多 4 targets / 6000 source chars，每个 batch 只带边界外 1 段 previous/next context；table、table cell、attribute、footnote、caption/reference 等高风险项默认 singleton。若一批只坏 1 项，有效 sibling 会立即保留，只对失败目标递归二分/降级重试。跨章节默认最多 4 workers；`requests_per_minute=0` 表示不做额外固定节拍，HTTP client 仍对 429/网络失败做有界退避。

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

`prepare` 保持纯离线：PaddleOCR label、页码映射、全局 TOC 对齐、编号和版式规则先给出 baseline。随后 `vision-structure` 才会联网并产生 API 费用。它优先读取显式指定或与 OCR JSON 同名的 PDF，用 PyMuPDF 渲染页面；如果没有 PDF，则退回 OCR JSON 的本地路径/data-URL `inputImage`。远程 `http(s)` 图片默认不信任、不发送给模型，只有显式设置 `vision.allow_remote_input_images=true` 才启用。也可以在 CLI 传入 `--pdf /path/to/source.pdf`；Windows GUI 则有单独的“Vision PDF（可选）”选择框。

Vision 会批量扫描前言区识别真正的 TOC 页并抄录目录项；正文阶段默认不会盲扫全书，而是结合 PaddleOCR 的 `paragraph_title/doc_title/table of contents` 类别、短标题几何、编号、PyMuPDF 大字号行和 TOC 目标页，先筛出候选页再看图。若追求最大召回，可在 `chapter_config.json` 设 `vision.scan_all_body_pages=true`。输出 `structure/vision_structure.json` 绑定 OCR SHA-256，并记录模型、扫描页、TOC、正文标题与 usage。

Vision 与 OCR 同页文本能匹配的标题可获得高置信证据；Vision 看见但 OCR 文本完全没有匹配的标题会保持低于自动通过阈值，仍交给人工章节审核。章节门禁通过后再次 `prepare` 会生成 `source_review.md` 并返回 `needs_source_review_choice`。先选择人工核对或自动批准；只有批准后的 source 才会进入自动术语与翻译。

```bash
# 人工核对
python3 new_book.py source-review --project books/my-book --mode manual
# 编辑 source_review.md 后
python3 new_book.py apply-source-review --project books/my-book

# 或全自动
python3 new_book.py source-review --project books/my-book --mode auto
```

批准 source 后术语默认自动完成。`auto-glossary` 会让 DeepSeek 覆盖全部候选，自动 include/reject 并决定统一译名；低置信度项自动做第二轮复核，并保存 `glossary_llm_review.jsonl` 与 `glossary_llm_usage.json`。人工术语窗口只作为异常或覆盖决定时的兜底。门禁通过后继续：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek 官方 API key'
python3 new_book.py preflight --project books/my-book
python3 new_book.py generate-cover --project books/my-book --theme auto
python3 new_book.py translate --project books/my-book --target-completed 10
python3 new_book.py status --project books/my-book
```

`preflight` 不联网，并按真实 micro-batch planner 给出预计请求数、平均/最大 batch size 与 singleton 数；预计请求通常显著少于待翻译 segment 数。`translate` 才发送文本并可能产生费用；建议按 `10 → 100 → 500 → --all` 逐步扩大。10/100 段后优先检查 `cost.usage.reasoning_tokens`、`cache_hit_ratio`、`retry_rate` 与 `projected_total_cost_rmb_at_current_average`；若 reasoning token 非零或重试率异常，应先排查配置再继续整本。完成后运行：

```bash
python3 new_book.py render --project books/my-book
python3 translate_book.py qa --config books/my-book/translation_config.json
```

封面主题可选 `auto`、`ink`、`ocean`、`ember`、`forest`、`plum`。如果用户提供了有权使用的图片，也可以用 `new_book.py set-cover --help` 登记并替换自动封面。

PDF/EPUB 导出还需要 Pandoc、XeLaTeX、可用的中文字体和 EPUBCheck。新项目默认要求 EPUBCheck 通过才交付 EPUB；可用 `EPUBCHECK_JAR` 指向 jar：

```bash
python3 new_book.py export --project books/my-book --format both
```


### EPUB：先标准化为 reviewed-source Markdown，再重新出版

v0.11 起，**新 EPUB 项目默认不再对源 XHTML/DOM/CSS 做原位回写**。原因很简单：不同来源 EPUB 的 OPF、XHTML、inline 标签、属性、脚注实现和 CSS 差异会不断渗透进解析器和翻译 prompt，维护成本高，而且会产生大量无意义 token。

新的默认路径是：

```text
source EPUB
  → safe ZIP + tolerant OPF/spine discovery
  → tolerant HTML semantic extraction
  → canonical segments + assets/epub/
  → source_review.md
  → 用户可选人工审核
  → 共用 glossary + contextual micro-batch
  → book_translated.md
  → Pandoc EPUB3 rebuild + EPUBCheck
```

初始化方式不变：

```bash
python3 new_book.py init \
  --input-epub /绝对路径/book.epub \
  --book-id my-epub-book \
  --book-title 'Original Title' \
  --book-title-zh '中文书名' \
  --author 'Author Name'

python3 new_book.py prepare --project books/my-epub-book
```

新项目的 `source_adapter` 为 `epub_markdown_v2`。importer 保留 ZIP 路径/压缩炸弹等安全检查，但不再要求源 EPUB 的 `mimetype` 必须严格位于第一项且 STORE，也不要求正文 XHTML 必须是可无损 XML round-trip 的形式。正文使用 tolerant HTML 语义抽取，识别标准 `h1-h6`、paragraph/list/blockquote/caption/footnote，以及常见 class/id 形式的章节标题；格式很差但仍有 body text 时会保底抽出正文。纯图片书仍应走 OCR。

`prepare` 会写出：

- `source_review.md`：统一、可人工检查的整理版原书；EPUB 用 `EPUB source: OPS/Text/ch03.xhtml` 标明原 spine 位置；
- `structure/epub_import_report.json`：抽取统计与 warning；
- `work/source_manifest.json`：源 EPUB hash、spine document、资源与 cover provenance；
- `assets/epub/`：需要保留到成品里的本地正文图片。

源 EPUB 的 JPEG/PNG 封面会优先自动复用并登记 provenance；如果找不到可复用封面，则自动生成现有的本地排版封面。正文图片会在最终 Markdown 中作为标准 Markdown image 重新嵌入 EPUB。

接下来与 OCR 流程相同：

```bash
# 人工核对
python3 new_book.py source-review --project books/my-epub-book --mode manual
# 修改 source_review.md 后：
python3 new_book.py apply-source-review --project books/my-epub-book

# 或完全自动
python3 new_book.py source-review --project books/my-epub-book --mode auto

python3 new_book.py preflight --project books/my-epub-book
python3 new_book.py translate --project books/my-epub-book --target-completed 10
python3 new_book.py translate --project books/my-epub-book --all
python3 new_book.py render --project books/my-epub-book
python3 new_book.py export --project books/my-epub-book --format epub
```

这里的取舍是明确的：**不保证 1:1 保留源 EPUB 的 CSS、DOM、字体、脚本、复杂链接和特殊交互效果；保证的是正文阅读顺序、章节语义、抽取出来的图片、封面和可审核的文本内容进入一个稳定、统一的出版流水线。** 对翻译书来说，这通常比维护任意 EPUB 方言的原位回写更可靠，也让翻译 prompt 与源 EPUB 实现彻底解耦。

旧项目如果已经使用 `epub_native_v1`，仍保留 legacy DOM-preserving 路径以便断点续跑；但新项目和 Skill 不再默认选择它。

## 方式二：Coding Agent

打开 [CODING_AGENT_PROMPT.md](CODING_AGENT_PROMPT.md)，把开头的 `<BOOK_INPUT_ABSOLUTE_PATH>` 替换成 OCR JSON 或 reflowable EPUB 的绝对路径；OCR 项目若有对应 PDF，也让 agent 使用它做 Vision 结构增强。然后把整段提示词交给能够访问本仓库和终端的 coding agent。

Agent 会先从输入证据推断书名、作者、语言、领域及 `book_id`。OCR/PDF 执行章节恢复 + Vision；EPUB 则先本地标准化为 reviewed-source Markdown、抽取图片并复用/生成封面；随后两者共享 source-review 门禁、自动术语、零网络预检、micro-batch 翻译、QA 与出版。证据不足的章节结构必须留给用户；普通术语候选默认由 `auto-glossary` 自动 include/reject 和给出译名。

启动 agent 前应在 agent 进程能够继承的终端设置 `DEEPSEEK_API_KEY`。不要把 key 粘贴给 agent，也不要写进提示词或配置文件。

新项目默认使用 contextual micro-batch：同章边界外前后各 1 段上下文、最多 300 字上一段译文、普通 batch 最多 4 targets/6000 字符、高风险项 singleton、失败段可续跑；默认 `thinking_mode=disabled`，JSON mode 开启，最多 4 个章节 lanes 并行，固定 RPM pacing 默认关闭。每次完成翻译后会在状态结果中附带 `cost` 诊断，汇总 cache hit/miss、reasoning tokens、retry rate 以及按当前均值估算的整本成本。

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

- OCR JSON / reflowable EPUB 文件选择，并为 OCR 项目提供可选的对应 PDF 选择框供 Vision 直接渲染；
- 项目目录、Book ID、原文/中文书名、作者、语言和领域输入；
- DeepSeek 模型、思考模式（默认 disabled）及隐藏显示的 API key 输入；
- 初始化、离线准备 + Vision、人工章节审核、**打开/应用带页码的原文MD**、自动术语、可选人工术语复核、零网络预检、micro-batch 翻译、状态、渲染和导出按钮；
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

`--offline-smoke` 会在临时目录使用合成 OCR，经过初始化、章节/Vision、**reviewed-source gate、模拟 LLM 自动术语审核、contextual micro-batch**、封面、零网络预检和 Markdown 渲染，并检查模拟译文无法进入正式导出。它不联网、不保存 key，也不验证真实 DeepSeek 翻译或 PDF/EPUB 工具。真实书籍回归和旧运行记录没有收入公开仓库，因此少量依赖私有语料的测试会跳过。在线 DeepSeek 翻译必须由用户使用自己的 key 和有权处理的样本验证。

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
