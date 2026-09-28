from __future__ import annotations

"""LLM-assisted terminology review with an auditable, fail-closed contract.

The glossary candidate generator remains deterministic and evidence-based. This
module only decides whether each generated candidate belongs in the book-level
terminology table and, when included, supplies a Chinese translation. Every
accepted decision is tied back to real candidate segment IDs; model-generated
evidence is never trusted.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import resolve_path
from .io_utils import read_jsonl, write_json, write_jsonl
from .llm_client import ChatClient, OpenAICompatibleClient, aggregate_usage


FENCE_MARK = chr(96) * 3
FENCE = re.compile(
    rf"^\\s*{re.escape(FENCE_MARK)}(?:json)?\\s*|\\s*{re.escape(FENCE_MARK)}\\s*$",
    re.I,
)
ALLOWED_DECISIONS = {"include", "reject"}
DEFAULT_BATCH_SIZE = 12
DEFAULT_RETRIES = 3
DEFAULT_ADJUDICATE_BELOW = 0.78


def _settings(config: Mapping[str, Any]) -> Dict[str, Any]:
    return dict((config.get("glossary_settings") or {}).get("llm_review") or {})


def _setting_path(config: Dict[str, Any], key: str, default_name: str) -> Path:
    value = (config.get("glossary_settings") or {}).get(key)
    if value:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (Path(config["_base_dir"]) / path).resolve()
    return resolve_path(config, "glossary").parent / default_name


def _review_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    return {
        "manifest": _setting_path(config, "manifest", "glossary_manifest.json"),
        "candidates": _setting_path(config, "candidates", "glossary_candidates.jsonl"),
        "decisions": resolve_path(config, "glossary").parent / "glossary_decisions.jsonl",
        "audit": resolve_path(config, "glossary").parent / "glossary_llm_review.jsonl",
        "usage": resolve_path(config, "glossary").parent / "glossary_llm_usage.json",
    }


def _candidate_payload(
    row: Mapping[str, Any],
    *,
    first_decision: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "candidate_id": str(row.get("candidate_id") or ""),
        "term": str(row.get("term") or ""),
        "reasons": list(row.get("reasons") or []),
        "occurrences": int(row.get("occurrences") or 0),
        "sample_contexts": [str(value) for value in (row.get("sample_contexts") or [])],
        "evidence_segment_ids": [str(value) for value in (row.get("evidence_segment_ids") or [])],
    }
    if first_decision is not None:
        payload["first_decision"] = dict(first_decision)
    return payload


def _system_prompt(config: Mapping[str, Any], *, adjudication: bool = False) -> str:
    domain = str(config.get("domain") or "学术人文社科")
    source_lang = str(config.get("source_lang") or "原文")
    target_lang = str(config.get("target_lang") or "简体中文")
    mode = (
        "这是第二轮复核。请重新审视低置信度候选，并最终在 include/reject 二选一；不要返回 needs_human。"
        if adjudication
        else "这是第一轮术语审计。每个候选都必须在 include/reject 二选一；不要返回 needs_human。"
    )
    return f"""你是{domain}书籍的专业术语编辑。源语言是{source_lang}，目标语言是{target_lang}。
{mode}
判断目标：只有需要跨章节保持稳定译法的概念、人物、机构、地名、作品名、专门术语才 include。
普通短语、偶然搭配、完整句子、泛化标题、无术语价值的高频 n-gram 应 reject。
若不确定某候选是否值得进入术语表，优先 reject；reject 只是不把它固定进术语表，并不删除原文。
include 时必须给出适合作为全书统一译法的中文 translation；专名采用通行或稳健译法，无法可靠确定时使用透明、可回译的音译/直译，不得虚构事实。
只依据给出的书籍元数据、候选上下文和统计证据判断，不要声称查阅了外部资料。
confidence 是 0 到 1 的数值，表示你对该 include/reject 与译名判断的把握。
只返回一个严格 JSON 对象：
{{"decisions":[{{"candidate_id":"glo-...","decision":"include|reject","translation":"","category":"","alternatives":[],"note":"","reason":"","confidence":0.0}}]}}
必须恰好覆盖输入中的每个 candidate_id 一次，不要输出任何额外 candidate_id。"""


def _user_prompt(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    first_decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    candidates = [
        _candidate_payload(
            row,
            first_decision=(first_decisions or {}).get(str(row.get("candidate_id") or "")),
        )
        for row in rows
    ]
    payload = {
        "book": {
            "title": str(config.get("book_title") or ""),
            "title_zh": str(config.get("book_title_zh") or ""),
            "author": str(config.get("author") or ""),
            "domain": str(config.get("domain") or ""),
        },
        "candidates": candidates,
    }
    return json.dumps(payload, ensure_ascii=False)


def _normalize_response(
    content: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    cleaned = FENCE.sub("", content.strip()).strip()
    data = json.loads(cleaned)
    decisions = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("LLM glossary response must contain a decisions list")

    expected = [str(row.get("candidate_id") or "") for row in rows]
    expected_set = set(expected)
    if len(expected) != len(expected_set) or "" in expected_set:
        raise ValueError("Glossary candidate IDs are invalid")

    parsed: Dict[str, Dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("Every LLM glossary decision must be an object")
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id not in expected_set or candidate_id in parsed:
            raise ValueError(f"Unexpected or duplicate candidate id: {candidate_id!r}")
        decision = str(item.get("decision") or "").strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"Invalid glossary decision for {candidate_id}: {decision!r}")
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Missing confidence for {candidate_id}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence out of range for {candidate_id}")
        translation = str(item.get("translation") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if decision == "include" and not translation:
            raise ValueError(f"Included term is missing translation: {candidate_id}")
        if decision == "reject" and not reason:
            raise ValueError(f"Rejected term is missing reason: {candidate_id}")
        alternatives = item.get("alternatives") or []
        if not isinstance(alternatives, list) or any(not isinstance(value, str) for value in alternatives):
            raise ValueError(f"Alternatives must be strings: {candidate_id}")
        parsed[candidate_id] = {
            "candidate_id": candidate_id,
            "decision": decision,
            "translation": translation,
            "category": str(item.get("category") or "").strip(),
            "alternatives": [value.strip() for value in alternatives if value.strip()],
            "note": str(item.get("note") or "").strip(),
            "reason": reason,
            "confidence": confidence,
        }
    if set(parsed) != expected_set:
        missing = sorted(expected_set - set(parsed))
        raise ValueError(f"LLM glossary response did not cover every candidate: {missing}")
    return parsed


def _complete_json_text(
    client: ChatClient,
    system: str,
    user: str,
) -> Tuple[str, Dict[str, Any]]:
    complete_json_text = getattr(client, "complete_json_text", None)
    if callable(complete_json_text):
        return complete_json_text(system, user)
    return client.complete(system, user)


def _request_batch(
    client: ChatClient,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    retries: int,
    adjudication: bool = False,
    first_decisions: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    usages: List[Dict[str, Any]] = []
    last_error: Optional[BaseException] = None
    for _attempt in range(1, retries + 1):
        content, usage = _complete_json_text(
            client,
            _system_prompt(config, adjudication=adjudication),
            _user_prompt(config, rows, first_decisions=first_decisions),
        )
        usages.append(dict(usage or {}))
        try:
            return _normalize_response(content, rows), aggregate_usage(usages)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError("LLM glossary review failed structured validation after retries") from last_error


def _batches(
    rows: Sequence[Mapping[str, Any]],
    size: int,
) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def _real_evidence(candidate: Mapping[str, Any]) -> List[str]:
    segment_ids = [
        str(value)
        for value in (candidate.get("evidence_segment_ids") or [])
        if str(value)
    ]
    if segment_ids:
        return [f"segment:{value}" for value in segment_ids[:3]]
    return [f"candidate:{candidate.get('candidate_id')}:generated-source-context"]


def _compile_decision(
    candidate: Mapping[str, Any],
    final: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    if final["decision"] == "include":
        row: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "decision": "include",
            "translation": str(final["translation"]),
            "category": str(final.get("category") or "book_specific"),
            "evidence": _real_evidence(candidate),
        }
        if final.get("alternatives"):
            row["alternatives"] = list(final["alternatives"])
        if final.get("note"):
            row["note"] = str(final["note"])
        return row
    return {
        "candidate_id": candidate_id,
        "decision": "reject",
        "reason": str(final["reason"]),
    }


def auto_review_glossary(
    config: Dict[str, Any],
    client: Optional[ChatClient] = None,
) -> Dict[str, Any]:
    """Decide every glossary candidate with the configured LLM.

    A normal glossary_decisions.jsonl file is written for the existing compiler.
    Low-confidence first-pass decisions receive a second adjudication pass.
    """
    paths = _review_paths(config)
    if not paths["manifest"].is_file() or not paths["candidates"].is_file():
        raise FileNotFoundError("Run prepare/glossary generation before LLM terminology review")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    candidates = list(read_jsonl(paths["candidates"]))
    if not candidates:
        write_jsonl(paths["decisions"], [])
        write_jsonl(paths["audit"], [])
        report = {
            "schema_version": 1,
            "book_id": config["book_id"],
            "status": "no_candidates",
            "candidate_count": 0,
            "model": None,
            "usage": aggregate_usage([]),
        }
        write_json(paths["usage"], report)
        return report

    settings = _settings(config)
    batch_size = int(settings.get("batch_size", DEFAULT_BATCH_SIZE))
    retries = int(settings.get("structured_retries", DEFAULT_RETRIES))
    threshold = float(settings.get("adjudicate_below", DEFAULT_ADJUDICATE_BELOW))
    if batch_size < 1 or batch_size > 40:
        raise ValueError("glossary_settings.llm_review.batch_size must be between 1 and 40")
    if retries < 1 or retries > 5:
        raise ValueError("glossary_settings.llm_review.structured_retries must be between 1 and 5")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("glossary_settings.llm_review.adjudicate_below must be between 0 and 1")

    review_client = client or OpenAICompatibleClient(dict(config.get("provider") or {}))
    first: Dict[str, Dict[str, Any]] = {}
    usage_rows: List[Dict[str, Any]] = []
    for batch in _batches(candidates, batch_size):
        decisions, usage = _request_batch(review_client, config, batch, retries=retries)
        first.update(decisions)
        usage_rows.append(usage)

    low = [
        row
        for row in candidates
        if first[str(row["candidate_id"])]["confidence"] < threshold
    ]
    adjudicated: Dict[str, Dict[str, Any]] = {}
    if low:
        for batch in _batches(low, batch_size):
            first_subset = {
                str(row["candidate_id"]): first[str(row["candidate_id"])]
                for row in batch
            }
            decisions, usage = _request_batch(
                review_client,
                config,
                batch,
                retries=retries,
                adjudication=True,
                first_decisions=first_subset,
            )
            adjudicated.update(decisions)
            usage_rows.append(usage)

    final = {
        candidate_id: adjudicated.get(candidate_id, decision)
        for candidate_id, decision in first.items()
    }
    decisions = [
        _compile_decision(row, final[str(row["candidate_id"])])
        for row in candidates
    ]
    write_jsonl(paths["decisions"], decisions)

    audit = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        audit.append({
            "candidate_id": candidate_id,
            "term": candidate.get("term"),
            "first_decision": first[candidate_id],
            "final_decision": final[candidate_id],
            "adjudicated": candidate_id in adjudicated,
            "source_evidence": _real_evidence(candidate),
            "model": getattr(review_client, "model", ""),
        })
    write_jsonl(paths["audit"], audit)

    total_usage = aggregate_usage(usage_rows)
    included = sum(row["decision"] == "include" for row in decisions)
    rejected = len(decisions) - included
    low_final = sum(
        float(final[str(row["candidate_id"])]["confidence"]) < threshold
        for row in candidates
    )
    report = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "status": "reviewed",
        "candidate_count": len(candidates),
        "included": included,
        "rejected": rejected,
        "adjudicated": len(adjudicated),
        "low_confidence_after_adjudication": low_final,
        "confidence_threshold": threshold,
        "model": getattr(review_client, "model", ""),
        "decisions_path": str(paths["decisions"]),
        "audit_path": str(paths["audit"]),
        "source_manifest_status": manifest.get("status"),
        "usage": total_usage,
    }
    write_json(paths["usage"], report)
    return report
