# Coding Agent 端到端书籍翻译提示词

把下面整段提示词交给能够读取文件和运行终端命令的 coding agent，并把第一行中的路径换成你的 OCR JSON 或 EPUB 绝对路径。Agent 应在本仓库内工作。

---

需要处理的书籍输入：`<BOOK_INPUT_ABSOLUTE_PATH>`（PaddleOCR JSON 或 reflowable EPUB）

请使用当前仓库的 `chapter-structure-recovery-lab` 与 `structured-book-translation-pipeline`，把这本 OCR JSON/PDF 或 reflowable EPUB 处理为结构化中文译稿。你负责执行完整工作流、检查产物和报告仍需人工决定的问题。不要修改 OCR 原文件，不要把 OCR、译文、封面、API key 或其他私有数据提交到 Git。

## 必须遵守的边界

1. 先读仓库根目录 `README.md`、两个子项目的 `README.md`、本书生成后的 `RUNBOOK.md`，再运行命令。
2. 只处理用户有权处理并发送给 DeepSeek API 的文本。
3. API key 只能从进程环境变量 `DEEPSEEK_API_KEY` 读取。不要要求用户把 key 粘贴进对话，不要打印、记录或写入任何文件；如果变量未设置，停在零网络预检之前，给出设置命令。
4. OCR/PDF 项目先跑纯离线 `prepare`，再在人工章节审核前跑 `vision-structure`。Vision 是补强证据：可以识别 TOC 页、目录层级和遗漏小节，但不得为了过门禁而猜测章节、改写源标题或删除正文。Vision 无 OCR 文本匹配的标题必须继续保留给人审。
5. **开始 OCR/PDF 项目前先问用户一次**：章节恢复完成后，是要停下来人工核对/修补带页码的 `source/structured_source.md`，还是全自动继续？前者初始化用 `--source-confirmation manual`，后者用 `auto`。manual 模式必须在 `needs_source_confirmation` 停住，等用户明确说已经核对保存后才运行 `confirm-source`。该 Markdown 的标题/层级/正文在确认后成为术语与翻译的 source of truth；隐藏 DBT:SEG 锚点不得修改。
6. 术语候选默认交给仓库的 `auto-glossary` 让 DeepSeek 自动完成 include/reject 与统一译名；不要让用户逐条审核普通术语。模型决定必须保留审计记录。
7. 翻译使用 contextual v2 micro-batch，并按累计目标 `10 → 100 → 500 → all` 逐级进行。新项目默认普通同章连续文本最多 4 target/6000 字符共用一次请求、最多 4 个 chapter lane 并发；表格/attribute/结构令牌密集段自动 singleton。批内有效项立即保存，只重试失败项，整体 JSON 损坏时二分 fallback。不要删除 checkpoint 从头重翻。
7. 封面默认用本仓库的本地排版生成器，不下载第三方美术素材。用户明确提供合法图片时，才用 `set-cover` 登记其权利与来源。
8. PDF/EPUB 导出依赖 Pandoc、XeLaTeX、中文字体和 EPUBCheck；新项目的 EPUB 必须校验通过。缺少依赖时仍须交付 Markdown，并准确报告未生成的格式。

## 第一步：推断书目信息

只读检查 OCR JSON 的前 20–40 页和明显的书名页、版权页、目录页。推断并记录：

- 原文书名；
- 中文书名建议；
- 作者或编者；
- 源语言与目标语言；
- 学科领域；
- 适合作为目录名的 `book_id`，仅使用小写字母、数字、连字符或下划线；
- 如能可靠识别，记录 ISBN、版本和出版年份供报告使用。

证据冲突或关键书名无法可靠确定时，先向用户提出一个简短问题。不要因次要元数据缺失而停工；可以把作者留空并明确标注未确认。

## 第二步：初始化独立项目

从 `structured-book-translation-pipeline` 目录运行：

```bash
python3 new_book.py init \
  --input-json "<OCR_JSON_ABSOLUTE_PATH>" \
  --book-id "<INFERRED_BOOK_ID>" \
  --book-title "<INFERRED_ORIGINAL_TITLE>" \
  --book-title-zh "<INFERRED_CHINESE_TITLE>" \
  --author "<INFERRED_AUTHOR>" \
  --source-lang "<INFERRED_SOURCE_LANGUAGE>" \
  --target-lang "简体中文" \
  --domain "<INFERRED_DOMAIN>" \
  --source-confirmation "<manual-or-auto>"
```

把命令输出的项目目录保存为 `<PROJECT_DIR>`。确认 `project.json`、`chapter_config.json`、`translation_config.json` 和空术语表存在。配置文件中只能出现 `DEEPSEEK_API_KEY` 这个环境变量名，不能出现 key 值。

## 第三步：章节恢复、Vision 增强与结构审核

先建立纯离线 baseline：

```bash
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

对 OCR/PDF 项目，确认 `DEEPSEEK_API_KEY` 已设置后，在人工审核前运行：

```bash
python3 new_book.py vision-structure --project "<PROJECT_DIR>"
# 如果同名 PDF 不在 OCR JSON 旁边：
# python3 new_book.py vision-structure --project "<PROJECT_DIR>" --pdf "<SOURCE_PDF>"
python3 new_book.py status --project "<PROJECT_DIR>"
```

检查 `structure/vision_structure.json`：确认 Vision 实际扫描了 TOC 页和正文候选页，并记录模型与 usage。默认正文扫描由 OCR 标题类别、编号、短标题版式、PyMuPDF 大字号行和 TOC 页码共同 shortlist，不要无理由开启整书逐页 Vision。只有当召回仍明显不足且用户接受额外成本时，才把 `chapter_config.json -> vision.scan_all_body_pages` 改为 true 后重跑。

读取 `structure/validation.json`、`structure/review_packets.jsonl`、`structure/structure_report.md` 和 `structure/recovered_outline.md`。

- 如果验证通过且审核包为空，继续。
- 如果出现 `needs_structure_review`，逐项依据 OCR 原块、PDF/页码、Vision sidecar、相邻标题、目录证据和父链审核。
- 决定只能引用审核包中已有证据；不能创造源书不存在的标题文字。
- 将完整决定写入项目目录的 `structure_decisions.jsonl`，再运行：

```bash
python3 new_book.py compile-reviews \
  --project "<PROJECT_DIR>" \
  --decisions "<PROJECT_DIR>/structure_decisions.jsonl"
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

重复直到门禁通过，或明确列出必须由用户处理的未决项。不得把未决项静默标为通过。

## 第四步：生成并确认结构化原文

章节结构门禁通过后再次运行：

```bash
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

OCR/PDF 项目会把清理后的原文按确认的章节树重排，并对页尾→下一页页首的连续正文做第二遍保守跨页合并，生成：

```text
<PROJECT_DIR>/source/structured_source.md
```

文件带 OCR/PDF 的 1-based 页码或跨页范围。用户可以直接阅读，也可以修改标题文字、Markdown 标题层级和正文。不要删除或修改隐藏的 DBT:SEG 锚点。

- 若初始化选择 `manual`，状态必须是 `needs_source_confirmation`；告诉用户文件路径并停止后续工作。等用户明确说已经核对/保存后再运行：
  `python3 new_book.py confirm-source --project "<PROJECT_DIR>"`
- 若选择 `auto`，程序会自动确认该文件并继续。

确认后的文件 hash 会写入 provenance；之后若再次编辑，翻译会重新锁住，必须再次 `confirm-source`。后续 chapter brief、术语匹配和 micro-batch 必须基于确认后的 segment，不得重新回到原 OCR block 边界。

## 第五步：LLM 自动术语审核

确认 `DEEPSEEK_API_KEY` 已设置，然后直接运行：

```bash
python3 new_book.py auto-glossary --project "<PROJECT_DIR>"
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

自动流程会读取全部 `glossary_candidates.jsonl`，让 DeepSeek 对每项执行 include/reject，并为 include 项给出全书统一中文译名。低置信度项会自动进行第二轮复核。检查 `glossary_llm_review.jsonl` 与 `glossary_llm_usage.json` 是否完整覆盖候选；正常情况下不要让用户逐条术语审核。只有提供方不可用、JSON 结构持续失败或用户明确要求覆盖决定时，才使用人工 `compile-glossary` 路径。

只有状态为 `ready_to_translate` 才能继续。

## 第六步：生成封面

```bash
python3 new_book.py generate-cover --project "<PROJECT_DIR>" --theme auto
```

检查 `cover/cover.json` 与生成的 PNG：尺寸应为 1600×2400，`rights_status` 应为 `generated`，并包含 SHA-256、书名和作者证据。封面只使用元数据、字体与本地几何图形。

## 第七步：预检与分级翻译

确认当前进程已经设置 `DEEPSEEK_API_KEY` 后运行：

```bash
python3 new_book.py preflight --project "<PROJECT_DIR>"
```

预检必须报告 `external_requests_made: 0`，并检查 `estimated_remaining_requests`、`estimated_targets_per_request` 与 `micro_batch` 是否符合预期。随后依次运行并在每一级检查 `translation_status.json`、`translations.jsonl`、`usage.jsonl` 与最新错误文件：

```bash
python3 new_book.py translate --project "<PROJECT_DIR>" --target-completed 10
python3 new_book.py translate --project "<PROJECT_DIR>" --target-completed 100
python3 new_book.py translate --project "<PROJECT_DIR>" --target-completed 500
python3 new_book.py translate --project "<PROJECT_DIR>" --all
```

若全书不足某一级，直接进入下一状态检查。请求失败时先查明 401/403、429、超时、JSON 格式或结构令牌问题；保留已有完成记录，修复后断点续跑。不要删除进度文件从头重翻。

## 第八步：质量检查、渲染与导出

翻译达到 100% 后运行：

```bash
python3 new_book.py render --project "<PROJECT_DIR>"
python3 translate_book.py qa --config "<PROJECT_DIR>/translation_config.json"
```

检查标题层级、脚注、表格、HTML 注释、公式、URL、专名、术语一致性、漏译和异常重复。QA 标记的 likely_error 不得直接忽略；有充分依据时使用流水线的 QA 重译命令，仍有歧义则报告用户。

如果出版依赖齐全，再运行：

```bash
python3 new_book.py export --project "<PROJECT_DIR>" --format both
```

核对 `exports/export_report.json`：EPUB 应包含登记封面；PDF 应包含目录和书签。若缺少外部工具，保留已完成 Markdown，不伪造导出成功。

## 最终报告

最终回复必须包含：项目目录、推断的书目信息及证据、Vision 模型/扫描页/usage、结构门禁结果、结构化原文确认模式与路径、跨页合并数量、micro-batch 计划/实际请求数/平均 targets per request/fallback 情况、术语收录/排除数量、翻译完成率、DeepSeek 模型和实际 usage/成本汇总、QA 结果、封面与 Markdown/PDF/EPUB 路径，以及任何尚未解决的问题。区分已经执行的结果和仅供参考的建议。

---
