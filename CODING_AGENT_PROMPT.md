# Coding Agent 端到端书籍翻译提示词

把下面整段提示词交给能够读取文件和运行终端命令的 coding agent，并把第一行中的路径换成你的 OCR JSON 绝对路径。Agent 应在本仓库内工作。

---

需要处理的 PaddleOCR JSON：`<OCR_JSON_ABSOLUTE_PATH>`

请使用当前仓库的 `chapter-structure-recovery-lab` 与 `structured-book-translation-pipeline`，把这本书从 OCR JSON 处理为结构化中文译稿。你负责执行完整工作流、检查产物和报告仍需人工决定的问题。不要修改 OCR 原文件，不要把 OCR、译文、封面、API key 或其他私有数据提交到 Git。

## 必须遵守的边界

1. 先读仓库根目录 `README.md`、两个子项目的 `README.md`、本书生成后的 `RUNBOOK.md`，再运行命令。
2. 只处理用户有权处理并发送给 DeepSeek API 的文本。
3. API key 只能从进程环境变量 `DEEPSEEK_API_KEY` 读取。不要要求用户把 key 粘贴进对话，不要打印、记录或写入任何文件；如果变量未设置，停在零网络预检之前，给出设置命令。
4. 不得为了过门禁而猜测章节、改写源标题或删除正文。结构证据不足时保留 `needs_human`，列出候选、上下文和建议，不擅自决定。
5. 不得自动批准未经核对的术语。根据原文上下文决定 include/reject；存在多种合理译法时保留 `needs_human` 并询问用户。
6. 翻译必须按累计目标 `10 → 100 → 500 → all` 逐级进行。每一级检查状态、失败记录、结构令牌、术语一致性和异常译文；不要绕过断点续跑或质量门禁。
7. 封面默认用本仓库的本地排版生成器，不下载第三方美术素材。用户明确提供合法图片时，才用 `set-cover` 登记其权利与来源。
8. PDF/EPUB 导出依赖 Pandoc、XeLaTeX、中文字体和可选 epubcheck。缺少依赖时仍须交付 Markdown，并准确报告未生成的格式。

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

## 第三步：章节恢复与结构审核

```bash
python3 new_book.py prepare --project "<PROJECT_DIR>"
python3 new_book.py status --project "<PROJECT_DIR>"
```

读取 `structure/validation.json`、`structure/review_packets.jsonl`、`structure/structure_report.md` 和 `structure/recovered_outline.md`。

- 如果验证通过且审核包为空，继续。
- 如果出现 `needs_structure_review`，逐项依据 OCR 原块、页码、相邻标题、目录证据和父链审核。
- 决定只能引用审核包中已有证据；不能创造源书不存在的标题文字。
- 将完整决定写入项目目录的 `structure_decisions.jsonl`，再运行：

```bash
python3 new_book.py compile-reviews \
  --project "<PROJECT_DIR>" \
  --decisions "<PROJECT_DIR>/structure_decisions.jsonl"
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

重复直到门禁通过，或明确列出必须由用户处理的未决项。不得把未决项静默标为通过。

## 第四步：术语审核

读取 `glossary_candidates.jsonl`、`glossary_decisions.template.jsonl` 和候选出现的源片段。优先处理人物、机构、地名、核心概念和高频专门术语。已有可靠通行译名时 include；普通词、误抽取和无术语价值的片段 reject；有争议时 needs_human。

把覆盖全部候选的决定写入 `glossary_decisions.jsonl`，然后运行：

```bash
python3 new_book.py compile-glossary \
  --project "<PROJECT_DIR>" \
  --decisions "<PROJECT_DIR>/glossary_decisions.jsonl"
python3 new_book.py prepare --project "<PROJECT_DIR>"
```

只有状态为 `ready_to_translate` 才能继续。

## 第五步：生成封面

```bash
python3 new_book.py generate-cover --project "<PROJECT_DIR>" --theme auto
```

检查 `cover/cover.json` 与生成的 PNG：尺寸应为 1600×2400，`rights_status` 应为 `generated`，并包含 SHA-256、书名和作者证据。封面只使用元数据、字体与本地几何图形。

## 第六步：预检与分级翻译

确认当前进程已经设置 `DEEPSEEK_API_KEY` 后运行：

```bash
python3 new_book.py preflight --project "<PROJECT_DIR>"
```

预检必须报告 `external_requests_made: 0`。随后依次运行并在每一级检查 `translation_status.json`、`translations.jsonl`、`usage.jsonl` 与最新错误文件：

```bash
python3 new_book.py translate --project "<PROJECT_DIR>" --target-completed 10
python3 new_book.py translate --project "<PROJECT_DIR>" --target-completed 100
python3 new_book.py translate --project "<PROJECT_DIR>" --target-completed 500
python3 new_book.py translate --project "<PROJECT_DIR>" --all
```

若全书不足某一级，直接进入下一状态检查。请求失败时先查明 401/403、429、超时、JSON 格式或结构令牌问题；保留已有完成记录，修复后断点续跑。不要删除进度文件从头重翻。

## 第七步：质量检查、渲染与导出

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

最终回复必须包含：项目目录、推断的书目信息及证据、结构门禁结果、人工决定数量、术语收录/排除/未决数量、翻译完成率、DeepSeek 模型和实际 usage 汇总、QA 结果、封面路径与哈希、生成的 Markdown/PDF/EPUB 路径，以及任何尚未解决的问题。区分已经执行的结果和仅供参考的建议。

---
