#!/usr/bin/env python3
"""Export the bounded QA human queue as review-friendly Markdown and JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = str(row.get("id") or row.get("segment_id"))
            rows[key] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    queue = [json.loads(line) for line in (work / "qa_human_queue.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = load_jsonl(work / "cleaned_segments.jsonl")
    translations = load_jsonl(work / "translations.jsonl")
    report = json.loads((work / "qa_report.json").read_text(encoding="utf-8"))
    flags = {}
    for row in report.get("flags", []):
        flags.setdefault(str(row["segment_id"]), []).append(row)
    verdicts = {str(row["segment_id"]): row for row in report.get("verdicts", [])}

    records = []
    md = ["# QA 人工审核队列（20 个）", "", "生成自 `qa_human_queue.jsonl`。每条保留原文、当前译文和机器风险证据；本文件不自动修改译文。", ""]
    for index, item in enumerate(queue, 1):
        sid = str(item["segment_id"])
        source = segments.get(sid, {})
        translation = translations.get(sid, {})
        record = {
            "review_index": index,
            "segment_id": sid,
            "page_json": source.get("page_json"),
            "kind": source.get("kind"),
            "label": (source.get("metadata") or {}).get("label"),
            "checks": item.get("checks", []),
            "queue_reasons": item.get("reasons", []),
            "source_text": source.get("source_text", ""),
            "translated_text": translation.get("translated_text", ""),
            "qa_flags": flags.get(sid, []),
            "qa4_verdict": verdicts.get(sid),
        }
        records.append(record)
        md.extend([
            f"## {index}. `{sid}`",
            "",
            f"- 页码：{record['page_json']}  |  类型：`{record['kind']}`  |  检查：`{', '.join(record['checks'])}`",
            f"- 风险：{'；'.join(record['queue_reasons'])}",
            "",
            "### 原文",
            "",
            str(record["source_text"]).strip(),
            "",
            "### 当前译文",
            "",
            str(record["translated_text"]).strip(),
            "",
            "### 机器证据",
            "",
            "```json",
            json.dumps({"qa_flags": record["qa_flags"], "qa4_verdict": record["qa4_verdict"]}, ensure_ascii=False, indent=2),
            "```",
            "",
        ])
    (work / "qa_human_queue_review.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (work / "qa_human_queue_review.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"records": len(records), "markdown": str(work / "qa_human_queue_review.md"), "json": str(work / "qa_human_queue_review.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
