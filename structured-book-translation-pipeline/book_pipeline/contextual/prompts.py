"""Contextual translation v2 prompts (spec §12.2).

Every v2 prompt is single-target: one target segment per LLM call plus a
bounded chapter brief, bounded previous/next source windows and a bounded
previous translation.  The system prompt carries a hard, verbatim-only-target
clause (``CONTEXT_ONLY_CLAUSE``) so context ids can never legitimately appear
in the model output; the strict parser (``parser.parse_target_only``) rejects
any response that leaks them anyway.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models import Segment

PROMPTS_VERSION = 2  # stamps every v2 prompt payload, usage row and record.

# Verbatim clause asserted by tests and asserted verbatim in the system prompt.
CONTEXT_ONLY_CLAUSE = (
    "previous/next/previous_translation 只用于理解上下文；只输出 targets 的译文；"
    "不要翻译、不要输出任何 previous/next 或 context 片段。"
)


def system_prompt(config: Dict[str, Any]) -> str:
    domain = str(config.get("domain", "学术人文社科"))
    source_lang = str(config.get("source_lang", "原文"))
    target_lang = str(config.get("target_lang", "简体中文"))
    return (
        f"你是{domain}领域的专业译者。将{source_lang}完整翻译为{target_lang}。\n"
        "规则：\n"
        "1. chapter_brief、previous_source、next_source、previous_translation 是理解语境、指代、术语与文风的辅助上下文，全部来自同一章。\n"
        f"2. {CONTEXT_ONLY_CLAUSE}\n"
        "3. 不增删、概括或解释；保留论证、限定、引文、数字和专名。只翻译 targets 中每个 text 字段。\n"
        "4. 严格使用提供的 glossary 条目；专名首次出现可用“译名（原文）”，同一段再次出现只用译名。\n"
        "5. 参考文献、书目和脚注中的书名与文章标题也必须翻译为目标语言；不可因其属于书目而整句原样复制。作者名、期刊名、出版社名、URL、DOI 和书目信息保留可核对的原文形式。\n"
        "6. 保留全部脚注引用、HTML 注释、公式、URL、表格分隔符以及 [[EPUB:...]] 令牌；所有结构令牌必须逐字节原样复制，不得改写、重排、包裹为 Markdown 链接或添加标点。\n"
        "7. 不添加 Markdown 标题井号、页码、前言或说明。\n"
        '8. 只返回严格 JSON 对象：{"t":[[0,"译文"],[1,"译文"]]}。每个局部 n 恰好返回一次，顺序可不同；不要返回稳定 segment id、上下文文本或解释。'
    )


def build_payload(
    *,
    chapter_brief: Dict[str, Any],
    previous_source: List[str],
    targets: List[Segment],
    next_source: List[str],
    previous_translation: str,
    glossary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a compact multi-target payload with stable-prefix fields first."""
    return {
        "chapter_brief": dict(chapter_brief),
        "previous_source": [str(text) for text in previous_source],
        "targets": [
            {"n": index, "kind": target.kind, "text": target.source_text}
            for index, target in enumerate(targets)
        ],
        "next_source": [str(text) for text in next_source],
        "previous_translation": previous_translation,
        "glossary": [dict(entry) for entry in glossary],
        "required_output": {"t": [[0, "translation for target n=0"]]},
    }
