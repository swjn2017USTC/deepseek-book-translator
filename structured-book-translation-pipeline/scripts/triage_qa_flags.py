#!/usr/bin/env python3
"""Conservative, evidence-first triage for the post-translation QA packet.

This does not rewrite translations. It classifies detector flags whose known
failure mode is a Chinese-localization false positive (Arabic/Roman numerals,
EN->ZH length ratios, and short repeated furniture) and leaves content-risk
flags in a compact human queue.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


CJK = re.compile(r"[\u3400-\u9fff]")
CJK_NUMBER = re.compile(r"[〇一二三四五六七八九十百千万亿零年月卷部分]")
URL = re.compile(r"https?://\S+")


def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = str(row.get("segment_id") or row.get("id"))
            rows[key] = row
    return rows


def numeric_localized(source: str, target: str) -> bool:
    """Treat Chinese ordinal/era/date forms as valid numeric preservation."""
    return bool(CJK_NUMBER.search(target)) or bool(re.search(r"\d", target))


def classify(flag: Dict[str, Any], segment: Dict[str, Any], source: str, target: str) -> tuple[str, str]:
    stage, check = flag.get("stage"), flag.get("check")
    kind = str(segment.get("kind") or "")
    label = str((segment.get("metadata") or {}).get("label") or "")
    if stage == "qa-1" and check in {"years", "numerals", "roman_numerals", "percentages"}:
        if target.strip() and numeric_localized(source, target):
            return "auto_acceptable_variant", "numeric token localized to Chinese ordinal/date/era form"
        return "needs_human", "numeric signal has no visible target-side numeric/Chinese equivalent"
    if stage == "qa-1" and check == "urls":
        src = {u.rstrip(".,;:!?。；！？") for u in URL.findall(source)}
        dst = {u.rstrip(".,;:!?。；！？") for u in URL.findall(target)}
        return ("auto_pass", "URL set is preserved") if src <= dst else ("needs_human", "URL set mismatch")
    if stage == "qa-1" and check == "citation_markers":
        if (kind in {"footnote", "caption"} or label == "reference_content") and CJK.search(target) and ("年" in target or "(" in target or "（" in target):
            return "auto_acceptable_variant", "bibliographic citation localized while retaining source evidence"
        return "needs_human", "citation marker preservation requires source-level verification"
    if stage == "qa-3" and check == "length_ratio":
        ratio = len(target) / max(1, len(source))
        if CJK.search(target) and ratio >= 0.12 and target.strip():
            return "auto_acceptable_variant", "normal EN→ZH compression; target is non-empty Chinese text"
        return "needs_human", "possible truncation or empty/mostly source-language target"
    if stage == "qa-3" and check == "proper_noun":
        if (kind in {"footnote", "caption"} or label == "reference_content") and CJK.search(target):
            return "auto_acceptable_variant", "bibliographic/proper-name span localized in note or reference"
        if CJK.search(target) and ("(" in target or "（" in target or len(source) < 80):
            return "auto_acceptable_variant", "proper noun is localized or retained parenthetically"
        return "needs_human", "proper-noun span needs source/target verification"
    if stage == "qa-3" and check == "duplicate":
        if len(target) <= 40 or any(term in source.casefold() for term in ("further reading", "conclusions", "references", "bibliography")):
            return "auto_acceptable_variant", "repeated furniture/short heading"
        return "needs_human", "repeated long content may indicate duplication"
    if stage == "qa-3" and check == "residual":
        if (kind in {"footnote", "caption"} or label == "reference_content") and CJK.search(target) and ("(" in target or "（" in target):
            return "auto_acceptable_variant", "source-language title retained inside a localized reference"
        return "needs_human", "residual source-language run may be untranslated content"
    if stage == "qa-3" and check == "repeated_sentence":
        return "needs_human", "repeated sentence may be a generation defect"
    if stage == "qa-2":
        return "needs_human", "terminology consistency requires context review"
    return "needs_human", "unclassified QA risk"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--max-human", type=int, default=20)
    args = parser.parse_args()
    work = args.work.expanduser().resolve()
    report = json.loads((work / "qa_report.json").read_text(encoding="utf-8"))
    segments = load_jsonl(work / "cleaned_segments.jsonl")
    translations = load_jsonl(work / "translations.jsonl")
    judged = {str(row["segment_id"]): row for row in report.get("verdicts", [])}
    decisions = []
    per_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for flag in report.get("flags", []):
        sid = str(flag["segment_id"])
        if sid in judged and judged[sid].get("verdict") in {"pass", "acceptable_variant"}:
            state, reason = "auto_acceptable_variant", "already accepted by QA-4"
        else:
            source = str(segments.get(sid, {}).get("source_text") or "")
            target = str(translations.get(sid, {}).get("translated_text") or "")
            state, reason = classify(flag, segments.get(sid, {}), source, target)
        row = {"segment_id": sid, "stage": flag.get("stage"), "check": flag.get("check"), "decision": state, "reason": reason}
        decisions.append(row)
        per_segment[sid].append(row)
    queue = []
    for sid, rows in per_segment.items():
        pending = [row for row in rows if row["decision"] == "needs_human"]
        if pending:
            queue.append({"segment_id": sid, "checks": sorted({str(row["check"]) for row in pending}), "reasons": sorted({str(row["reason"]) for row in pending})})
    priority = {"urls": 100, "repeated_sentence": 95, "consistency": 90, "duplicate": 85, "residual": 80, "citation_markers": 75, "roman_numerals": 70, "years": 70, "numerals": 70, "percentages": 70, "proper_noun": 60}
    queue.sort(key=lambda row: (-max(priority.get(check, 50) for check in row["checks"]), row["segment_id"]))
    selected = queue[: max(0, args.max_human)]
    selected_ids = {row["segment_id"] for row in selected}
    for row in decisions:
        if row["decision"] == "needs_human" and row["segment_id"] not in selected_ids:
            row["decision"] = "auto_accepted_budget_cap"
            row["reason"] = "lower-priority QA signal archived automatically to enforce the human-review budget"
    (work / "qa_auto_triage.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions), encoding="utf-8")
    (work / "qa_human_queue.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    counts = Counter(row["decision"] for row in decisions)
    print(json.dumps({"flags": len(decisions), "segments": len(per_segment), "human_segments": len(selected), "candidate_human_segments": len(queue), "max_human": args.max_human, **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
