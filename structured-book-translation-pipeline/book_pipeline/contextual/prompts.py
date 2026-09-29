"""Prompt builders for contextual translation.

The legacy singleton prompt remains for the smoke ladder and compatibility
tests. Normal v0.10 translation uses a compact same-chapter micro-batch payload:
stable chapter context first, then only the external boundary context and a
small list of local-indexed targets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..models import Segment

PROMPTS_VERSION = 2

CONTEXT_ONLY_CLAUSE = (
    "previous/next/previous_translation 只用于理解上下文；只输出 target 的译文；"
    "不要翻译、不要输出任何 previous/next 或 context 片段的 id。"
)

BATCH_CONTEXT_ONLY_CLAUSE = (
    "previous_source、next_source、previous_translation 只用于理解边界语境；"
    "只翻译 targets，绝不输出或翻译边界上下文。"
)


def system_prompt(config: Dict[str, Any]) -> str:
    """Strict singleton prompt kept for ladder/safe fallback compatibility."""
    domain = str(config.get("domain", "学术人文社科"))
    source_lang = str(config.get("source_lang", "原文"))
    target_lang = str(config.get("target_lang", "简体中文"))
    return (
        f"你是{domain}领域的专业译者。将{source_lang}完整翻译为{target_lang}。\n"
        "规则：\n"
        "1. chapter_brief、previous_source、next_source、previous_translation 是理解语境、指代、术语与文风的辅助上下文，全部来自同一章。\n"
        f"2. {CONTEXT_ONLY_CLAUSE}\n"
        "3. 不增删、概括或解释；保留论证、限定、引文、数字和专名。只翻译 target 的 text 字段。\n"
        "4. 严格使用提供的 glossary 条目；专名首次出现可用“译名（原文）”，同一段再次出现只用译名。\n"
        "5. 参考文献、书目和脚注中的书名与文章标题也必须翻译为目标语言；作者、期刊、出版社、URL、DOI 可保留可核对原文。\n"
        "6. 保留全部脚注引用、HTML 注释、公式、URL、表格分隔符以及 [[EPUB:...]] 令牌。\n"
        "7. 不添加 Markdown 标题井号、页码、前言或说明。\n"
        '8. 只返回严格 JSON 对象：{"id":"<target id>","translated_text":"<译文>"}。'
    )


def batch_system_prompt(config: Dict[str, Any]) -> str:
    domain = str(config.get("domain", "学术人文社科"))
    source_lang = str(config.get("source_lang", "原文"))
    target_lang = str(config.get("target_lang", "简体中文"))
    return (
        f"你是{domain}领域的专业译者。将{source_lang}完整翻译为{target_lang}。\n"
        "同一个请求中的 targets 是同一章内连续原文块，它们彼此就是上下文。\n"
        f"{BATCH_CONTEXT_ONLY_CLAUSE}\n"
        "不增删、概括或解释；保留论证、限定、引文、数字和专名。严格使用 glossary。\n"
        "参考文献、书目和脚注中的书名/文章标题必须翻译；作者、期刊、出版社、URL、DOI 可保留原文。\n"
        "全部脚注引用、HTML 注释、公式、URL、表格分隔符和 [[EPUB:...]] 令牌必须逐字节保留。\n"
        "不要添加 Markdown 标题井号、页码、前言、解释或额外文本。\n"
        '只返回严格 JSON 对象 {"t":[[0,"译文"],[1,"译文"]]}；'
        "数组第一项必须是输入 targets 的局部 n，且每个 n 恰好出现一次。"
    )


def prompt_chapter_context(chapter_brief: Dict[str, Any]) -> Dict[str, Any]:
    """Only semantic fields belong in the provider prompt.

    brief_hash/page range/segment counts remain on disk for invalidation and
    provenance but are not useful translation context.
    """
    return {
        key: chapter_brief[key]
        for key in ("title_source", "kind", "zone", "section_titles")
        if chapter_brief.get(key) not in (None, "", [])
    }


def build_payload(
    *,
    chapter_brief: Dict[str, Any],
    previous_source: List[str],
    target: Segment,
    next_source: List[str],
    previous_translation: str,
    glossary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "chapter_brief": dict(chapter_brief),
        "previous_source": [str(text) for text in previous_source],
        "target": {"id": target.id, "kind": target.kind, "text": target.source_text},
        "next_source": [str(text) for text in next_source],
        "previous_translation": previous_translation,
        "glossary": [dict(entry) for entry in glossary],
        "required_output": {
            "id": target.id,
            "translated_text": "translation of the target text only",
        },
    }


def build_batch_payload(
    *,
    chapter_brief: Dict[str, Any],
    previous_source: List[str],
    targets: Sequence[Segment],
    next_source: List[str],
    previous_translation: str,
    glossary: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        # Keep this first: identical same-chapter requests share a longer
        # provider-prefix for automatic context caching.
        "chapter_context": prompt_chapter_context(chapter_brief),
        "previous_source": [str(text) for text in previous_source],
        "targets": [
            {"n": index, "kind": segment.kind, "text": segment.source_text}
            for index, segment in enumerate(targets)
        ],
        "next_source": [str(text) for text in next_source],
        "previous_translation": previous_translation,
        "glossary": [dict(entry) for entry in glossary],
        "required_output": {"t": [[0, "translation"], ["...", "..."]]},
    }
