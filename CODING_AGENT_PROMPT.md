# Coding Agent 端到端书籍翻译提示词

把下面整段提示词交给能够读取文件和运行终端命令的 coding agent，并把第一行中的路径换成你的 OCR JSON 或 EPUB 绝对路径。Agent 应在本仓库内工作。

---

需要处理的书籍输入：`<BOOK_INPUT_ABSOLUTE_PATH>`（PaddleOCR JSON/PDF 或 EPUB）

请使用当前仓库的 `chapter-structure-recovery-lab` 与 `structured-book-translation-pipeline`，把这本 OCR JSON/PDF 或 EPUB 处理为结构化中文译稿。EPUB 新项目必须先标准化为 canonical reviewed-source Markdown，不要默认做源 DOM 原位回写。你负责执行完整工作流、检查产物和报告仍需人工决定的问题。不要修改 OCR 原文件，不要把 OCR、译文、封面、API key 或其他私有数据提交到 Git。

## 必须遵守的边界

1. 先读仓库根目录 `README.md`、两个子项目的 `README.md`、本书生成后的 `RUNBOOK.md`，再运行命令。
2. 只处理用户有权处理并发送给 DeepSeek API 的文本。
3. API key 只能从进程环境变量 `DEEPSEEK_API_KEY` 读取。不要要求用户把 key 粘贴进对话，不要打印、记录或写入任何文件；如果变量未设置，停在零网络预检之前，给出设置命令。
4. OCR/PDF 项目先跑纯离线 `prepare`，再在人工章节审核前跑 `vision-structure`。EPUB 项目不要跑 Vision；`prepare` 应使用 `epub_markdown_v2` 本地抽取 spine 文本、章节、图片和封面，生成 `epub_import_report.json` 与 `source_review.md`。Vision 是补强证据：可以识别 TOC 页、目录层级和遗漏小节，但不得为了过门禁而猜测章节、改写源标题或删除正文。Vision 无 OCR 文本匹配的标题必须继续保留给人审。
5. 章节结构稳定后，必须先生成并处理 `source_review.md` 门禁：主动询问用户是否要人工核对这份带页码的整理版原文。用户选择人工时运行 `source-review --mode manual`，给出文件路径并停止；只有用户明确说已核对完成，才能运行 `apply-source-review`。用户选择不人工核对时才运行 `source-review --mode auto` 并继续。
6. 术语候选默认交给仓库的 `auto-glossary` 让 DeepSeek 自动完成 include/reject 与统一译名；不要让用户逐条审核普通术语。模型决定必须保留审计记录。只有 API/结构化输出持续失败或用户明确要求覆盖某个决定时，才退回人工术语复核。
7. 翻译使用项目配置中的 contextual micro-batch，并按累计目标 `10 → 100 → 500 → all` 逐级进行。默认普通文本最多 4 targets/6000 字符共享上下文，高结构风险项 singleton，跨章节最多 4 workers。检查 preflight 的真实 batch 计划、失败记录和 usage；partial salvage 后只重试未通过 target。
8. OCR 默认使用本地排版封面；EPUB 默认优先复用源包 JPEG/PNG 封面，找不到才自动生成。不要为了适配某一本 EPUB 去修改翻译 prompt/加入 DOM token；优先修 importer，纯图片或正文抽取严重失败时改走 OCR。
9. PDF/EPUB 导出依赖 Pandoc、XeLaTeX、中文字体和 EPUBCheck；新项目的 EPUB 必须校验通过。缺少依赖时仍须交付 Markdown，并准确报告未生成的格式。

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
  --domain "<INFERRED_DOMAIN>"
```

把命令输出的项目目录保存为 `<PROJECT_DIR>`。确认 `project.json`、`chapter_config.json`、`translation_config.json` 和空术语表存在。配置文件中只能出现 `DEEPSEEK_API_KEY` 这个环境变量名，不能出现 key 值。

## 第三步：输入标准化 / 章节恢复

先建立纯离线 baseline：

```bash
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

如果输入是 EPUB，检查 `project.json -> source_adapter` 必须为 `epub_markdown_v2`，并检查 `structure/epub_import_report.json`、`work/source_manifest.json`、`assets/epub/` 和 `cover/cover.json`。确认 `source_review.md` 中没有 `[[EPUB:...]]` token，正文顺序与主要章节没有明显缺失。EPUB 不进入下面的 Vision/structure-review 步骤；直接进入第四步 reviewed-source 审核。若 semantic importer 明显漏掉大量正文，优先修 importer 或改走 OCR，不要给翻译 prompt 加 EPUB-specific 规则。


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

## 第四步：整理版原文 Markdown 审核

章节门禁完全通过后再次运行：

```bash
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

应生成 `<PROJECT_DIR>/source_review.md` 并返回 `needs_source_review_choice`。该文件是按最终章节结构和跨页合并整理出的原文电子版，含 Markdown 标题层级、源页码标记和隐藏 `BOOK_SEGMENT` 标记。

**现在必须询问用户是否要人工核对。**

如果用户选择人工核对：

```bash
python3 new_book.py source-review --project "<PROJECT_DIR>" --mode manual
```

把确切文件路径告诉用户，然后停止本次自动流程。用户可以修改可见原文和标题层级，但不要删除/复制 `BOOK_SEGMENT` 注释。只有用户之后明确说“核对好了/继续”时，才运行：

```bash
python3 new_book.py apply-source-review --project "<PROJECT_DIR>"
```

如果用户明确不需要人工核对，则运行：

```bash
python3 new_book.py source-review --project "<PROJECT_DIR>" --mode auto
```

批准后的 reviewed source 才能进入术语与翻译；禁止直接绕过这个 gate。

## 第五步：LLM 自动术语审核

确认 `DEEPSEEK_API_KEY` 已设置，然后直接运行：

```bash
python3 new_book.py auto-glossary --project "<PROJECT_DIR>"
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

自动流程会读取全部 `glossary_candidates.jsonl`，让 DeepSeek 对每项执行 include/reject，并为 include 项给出全书统一中文译名。低置信度项会自动进行第二轮复核。检查 `glossary_llm_review.jsonl` 与 `glossary_llm_usage.json` 是否完整覆盖候选；正常情况下不要让用户逐条术语审核。只有提供方不可用、JSON 结构持续失败或用户明确要求覆盖决定时，才使用人工 `compile-glossary` 路径。

只有状态为 `ready_to_translate` 才能继续。

## 第六步：确认封面

OCR 项目如无登记封面，运行 `generate-cover`。EPUB 项目在 `prepare` 时已自动复用源 JPEG/PNG 封面；没有可复用封面才自动生成。检查 `cover/cover.json` 的 `selected_by`/`rights_status`/SHA-256，不要无故覆盖源 EPUB 封面。

## 第七步：预检与分级翻译

确认当前进程已经设置 `DEEPSEEK_API_KEY` 后运行：

```bash
python3 new_book.py preflight --project "<PROJECT_DIR>"
```

预检必须报告 `external_requests_made: 0`，并检查 `micro_batch_plan` 的预计 batches、average_batch_size 与 singleton_batches；若预计请求数仍接近 segment 数，应先检查为何大量内容被降级为 singleton。随后依次运行并在每一级检查 `translation_status.json`、`translations.jsonl`、`usage.jsonl` 与最新错误文件：

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

最终回复必须包含：项目目录、输入 adapter；OCR 的 Vision/结构证据或 EPUB 的 import report/抽取资源/封面 provenance；`source_review.md` 审核模式与 hash；术语、micro-batch、usage、QA，以及标准化 Markdown/PDF/EPUB 产物。区分已经执行的结果和仅供参考的建议。

---
