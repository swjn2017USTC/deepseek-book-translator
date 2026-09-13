from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import resolve_path
from .io_utils import append_jsonl, read_jsonl, write_json
from .llm_client import ChatClient, OpenAICompatibleClient, _ExactHeaderName  # noqa: F401 (re-exported below)
from .models import Segment, TranslationRecord
from .clean import require_structure_gate
from .glossary import glossary_cache_sha, require_ready_glossary
from .provenance import require_fresh_cleaning

# The client implementation moved to llm_client.py (P03 extraction); these
# names are re-exported from translate.py so existing internal importers
# (workflow.py, tests) keep working unchanged.  Import direction is
# translate -> llm_client, never the reverse.
__all__ = ["ChatClient", "OpenAICompatibleClient", "_ExactHeaderName"]


STRUCTURAL_TOKEN = re.compile(
    r"(?:\[\^[^\]]+\]|<!--.*?-->|"
    r"https?://[^\s\u4e00-\u9fff，。；！？、）】》]+|"
    r"\$\s*\^\{?\d+\}?\s*\$)"
)


def _segments(path: Path) -> List[Segment]:
    return [Segment(**row) for row in read_jsonl(path)]


def _completed(path: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[str(row.get("segment_id"))] = row
    return latest


def is_translation_current(segment: Segment, row: Optional[Dict[str, Any]], glossary_sha256: str = "") -> bool:
    if not row:
        return False
    if row.get("source_hash") != segment.source_hash or row.get("status") != "completed":
        return False
    return not glossary_sha256 or row.get("glossary_sha256") == glossary_sha256


def _batches(segments: Sequence[Segment], max_chars: int, max_segments: int) -> Iterable[List[Segment]]:
    batch: List[Segment] = []
    characters = 0
    key: Optional[Tuple[Optional[str], str]] = None
    for segment in segments:
        segment_key = (segment.parent_node_id, segment.zone)
        if batch and (characters + len(segment.source_text) > max_chars or len(batch) >= max_segments or segment_key != key):
            yield batch
            batch, characters = [], 0
        key = segment_key
        batch.append(segment)
        characters += len(segment.source_text)
    if batch:
        yield batch


def _glossary_subset(batch: Sequence[Segment], glossary: Sequence[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    joined = "\n".join(segment.source_text for segment in batch).casefold()
    found = [
        item for item in glossary
        if item.get("term") and any(
            str(term).casefold() in joined
            for term in [item["term"], *(item.get("source_aliases") or [])]
        )
    ]
    found.sort(key=lambda item: -len(item["term"]))
    # Provenance stays on disk; only translation-relevant fields consume prompt
    # context and no source URLs or matched-segment lists are sent upstream.
    keys = ("term", "translation", "category", "alternatives", "note", "source_aliases")
    return [{key: item[key] for key in keys if item.get(key)} for item in found[:limit]]


def system_prompt(config: Dict[str, Any]) -> str:
    return f"""你是{config.get('domain', '学术人文社科')}领域的专业译者。将{config.get('source_lang', '原文')}完整翻译为{config.get('target_lang', '简体中文')}。
规则：
1. 不增删、概括或解释；保留论证、限定、引文、数字和专名。
2. 严格使用提供的术语表。专名首次出现可用“译名（原文）”，同一段再次出现只用译名。
3. 输入是结构化片段，不要添加 Markdown 标题井号、页码、前言或说明。
4. 保留全部脚注引用、HTML 注释、公式、URL 和表格分隔符；所有结构令牌（脚注、HTML 注释、URL、公式）必须逐字节原样复制，不得改写、包裹为 Markdown 链接或添加标点。
5. 标题只翻译原文，不改层级，不自行补标题。
6. 只返回严格 JSON；每个输入 id 恰好返回一次。"""


def user_prompt(config: Dict[str, Any], batch: Sequence[Segment], glossary: Sequence[Dict[str, str]], previous_tail: str) -> str:
    payload = [{"id": segment.id, "kind": segment.kind, "text": segment.source_text} for segment in batch]
    instructions = {
        "book": config["book_title"],
        "target_title": config["book_title_zh"],
        "glossary": glossary,
        "previous_translation_tail": previous_tail,
        "segments": payload,
        "required_output": [{"id": "same input id", "translated_text": "translation"}],
    }
    return json.dumps(instructions, ensure_ascii=False)


def parse_translation(content: str, batch: Sequence[Segment]) -> Dict[str, str]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    data = json.loads(cleaned)
    if isinstance(data, dict):
        if "translations" in data:
            data = data["translations"]
        elif "segments" in data:
            data = data["segments"]
        elif "id" in data and any(key in data for key in ("translated_text", "translation", "text")):
            data = [data]
        else:
            # Some compatible models return an object keyed by segment id.
            data = [
                {"id": identifier, **(value if isinstance(value, dict) else {"translated_text": value})}
                for identifier, value in data.items()
            ]
    if not isinstance(data, list):
        raise ValueError("Translation response must be a JSON list")
    expected = {segment.id for segment in batch}
    translated = {
        str(item.get("id")): str(
            item.get("translated_text")
            if item.get("translated_text") is not None
            else item.get("translation")
            if item.get("translation") is not None
            else item.get("text") or ""
        )
        for item in data
        if isinstance(item, dict)
    }
    if len(expected) == 1 and len(data) == 1:
        item = data[0]
        values = [
            value for key, value in item.items()
            if key not in {"id", "segment_id"} and isinstance(value, str) and value.strip()
        ]
        if len(values) == 1:
            # Singleton batches make an ID rewrite unambiguous.
            translated = {next(iter(expected)): values[0]}
    if set(translated) != expected or any(not translated[identifier].strip() for identifier in expected):
        raise ValueError("Translation response IDs or values do not match the batch")
    for segment in batch:
        source_tokens = STRUCTURAL_TOKEN.findall(segment.source_text)
        translated_text = translated[segment.id]
        translated_tokens = STRUCTURAL_TOKEN.findall(translated_text)
        if len(batch) == 1 and len(source_tokens) == len(translated_tokens):
            for source_token, translated_token in zip(source_tokens, translated_tokens):
                normalize = lambda token: token.rstrip(".,;:!?。；！？")
                if normalize(source_token) == normalize(translated_token) and source_token != translated_token:
                    translated_text = translated_text.replace(translated_token, source_token)
            translated[segment.id] = translated_text
        required = [token.rstrip(".,;:!?。；！？") for token in source_tokens]
        produced = [token.rstrip(".,;:!?。；！？") for token in STRUCTURAL_TOKEN.findall(translated_text)]
        if required != produced:
            raise ValueError(f"Structural token mismatch for {segment.id}")
        if segment.kind == "table" and translated[segment.id].count("|") != segment.source_text.count("|"):
            raise ValueError(f"Markdown table delimiter mismatch for {segment.id}")
    return translated


def translate_book(
    config: Dict[str, Any],
    client: Optional[ChatClient] = None,
    limit: Optional[int] = None,
    target_completed: Optional[int] = None,
) -> Dict[str, Any]:
    require_structure_gate(config)
    require_fresh_cleaning(config)
    require_ready_glossary(config)
    glossary_sha256 = glossary_cache_sha(config)
    output_dir = resolve_path(config, "output_dir")
    segments_path = output_dir / "cleaned_segments.jsonl"
    if not segments_path.exists():
        raise FileNotFoundError("Run clean before translate")
    progress_path = output_dir / "translations.jsonl"
    usage_path = output_dir / "usage.jsonl"
    segments = _segments(segments_path)
    completed = _completed(progress_path)
    pending = [segment for segment in segments if segment.translatable and not is_translation_current(segment, completed.get(segment.id), glossary_sha256)]
    if limit is not None and target_completed is not None:
        raise ValueError("Use either limit or target_completed, not both")
    if target_completed is not None:
        if target_completed < 0:
            raise ValueError("target_completed must be non-negative")
        current_completed = sum(
            segment.translatable
            and is_translation_current(segment, completed.get(segment.id), glossary_sha256)
            for segment in segments
        )
        pending = pending[:max(0, target_completed - current_completed)]
    elif limit is not None:
        pending = pending[:limit]
    if not pending:
        report = translation_status(config)
        write_json(output_dir / "translation_status.json", report)
        return report
    client = client or OpenAICompatibleClient(dict(config.get("provider") or {}))
    glossary: List[Dict[str, str]] = []
    if config.get("glossary"):
        glossary = json.loads(resolve_path(config, "glossary").read_text(encoding="utf-8"))
    chunk = config.get("chunk", {})
    max_chars = int(chunk.get("max_chars", 10000))
    max_segments = 1
    requests_per_minute = float(config.get("provider", {}).get("requests_per_minute", 20))
    delay = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
    previous_tail = ""
    for batch_index, batch in enumerate(_batches(pending, max_chars, max_segments), 1):
        subset = _glossary_subset(batch, glossary, int(chunk.get("max_glossary_terms", 40)))
        last_error: Optional[Exception] = None
        translated: Dict[str, str] = {}
        usage: Dict[str, Any] = {}
        for attempt in range(1, 11):
            try:
                content, usage = client.complete(system_prompt(config), user_prompt(config, batch, subset, previous_tail))
                translated = parse_translation(content, batch)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                (output_dir / "last_invalid_response.txt").write_text(content, encoding="utf-8")
        if not translated:
            raise RuntimeError(f"Batch {batch_index} failed response validation") from last_error
        now = datetime.now(timezone.utc).isoformat()
        records = [TranslationRecord(
            segment.id, segment.source_hash, translated[segment.id], client.model,
            "completed", 1, now, glossary_sha256,
        ).to_dict() for segment in batch]
        append_jsonl(progress_path, records)
        append_jsonl(usage_path, [{"timestamp": now, "batch": batch_index, "model": client.model, "segment_ids": [segment.id for segment in batch], "usage": usage}])
        previous_tail = records[-1]["translated_text"][-500:] if not SENTENCE_FINISH.search(records[-1]["translated_text"]) else ""
        if delay:
            time.sleep(delay)
    report = translation_status(config)
    write_json(output_dir / "translation_status.json", report)
    return report


SENTENCE_FINISH = re.compile(r"[。！？.!?»”’\"']\s*$")


def translation_status(config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = resolve_path(config, "output_dir")
    segments = _segments(output_dir / "cleaned_segments.jsonl")
    completed = _completed(output_dir / "translations.jsonl")
    required = [segment for segment in segments if segment.translatable]
    glossary_ready = True
    glossary_error: Optional[str] = None
    try:
        glossary_sha256 = glossary_cache_sha(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        glossary_ready = False
        glossary_error = str(exc)
        glossary_sha256 = "__not_ready__"
    current = [segment for segment in required if is_translation_current(segment, completed.get(segment.id), glossary_sha256)]
    return {
        "schema_version": 1,
        "book_id": config["book_id"],
        "required_segments": len(required),
        "completed_segments": len(current),
        "remaining_segments": len(required) - len(current),
        "coverage": round(len(current) / len(required), 6) if required else 1.0,
        "ready_to_render": glossary_ready and len(current) == len(required),
        "glossary_ready": glossary_ready,
        "glossary_error": glossary_error,
        "glossary_sha256": glossary_sha256 if glossary_ready else None,
    }
