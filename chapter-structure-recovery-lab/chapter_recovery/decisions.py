from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DECISIONS = {
    "accept", "reject", "change_level", "change_kind",
    "merge_with_previous", "needs_human",
}
KINDS = {
    "frontmatter", "part", "part_intro", "chapter", "section",
    "backmatter", "toc_entry",
}
REASON_CODES = {
    "same_style", "numbering_parent", "running_header", "caption",
    "body_sentence", "apparatus_repeat", "split_title", "toc_evidence",
    "page_evidence", "insufficient_evidence", "other_verified",
}
COMMON_KEYS = {"candidate_id", "decision", "reason_code", "evidence_refs"}
DECISION_KEYS = {
    "accept": set(),
    "reject": set(),
    "change_level": {"level"},
    "change_kind": {"kind"},
    "merge_with_previous": {"merge_with_node_id"},
    "needs_human": set(),
}
FORBIDDEN_TEXT_KEYS = {"title", "title_source", "source_text", "new_title"}


class DecisionValidationError(ValueError):
    pass


def _records_hash(records: Iterable[Dict[str, Any]]) -> str:
    payload = json.dumps(list(records), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_decision(
    decision: Dict[str, Any], packet: Dict[str, Any]
) -> Dict[str, Any]:
    candidate_id = decision.get("candidate_id")
    if candidate_id != packet.get("candidate_id"):
        raise DecisionValidationError("candidate_id does not match its review packet")
    action = decision.get("decision")
    if action not in DECISIONS:
        raise DecisionValidationError(f"unsupported decision: {action}")
    allowed = set(packet.get("allowed_decisions") or DECISIONS)
    if action not in allowed:
        raise DecisionValidationError(f"decision is not allowed for {candidate_id}: {action}")
    extra = set(decision) - COMMON_KEYS - DECISION_KEYS[action]
    if extra:
        if extra & FORBIDDEN_TEXT_KEYS:
            raise DecisionValidationError("decisions must not supply or rewrite title text")
        raise DecisionValidationError(f"unexpected decision fields: {sorted(extra)}")
    missing = DECISION_KEYS[action] - set(decision)
    if missing:
        raise DecisionValidationError(f"missing decision fields: {sorted(missing)}")
    reason = decision.get("reason_code")
    if reason not in REASON_CODES:
        raise DecisionValidationError(f"unsupported reason_code: {reason}")
    evidence = decision.get("evidence_refs")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(value, str) and value.strip() for value in evidence
    ):
        raise DecisionValidationError("evidence_refs must be a non-empty string list")
    if action == "change_level":
        level = decision.get("level")
        if not isinstance(level, int) or isinstance(level, bool) or not 2 <= level <= 6:
            raise DecisionValidationError("level must be an integer from 2 through 6")
    if action == "change_kind" and decision.get("kind") not in KINDS:
        raise DecisionValidationError(f"unsupported kind: {decision.get('kind')}")
    if action == "merge_with_previous":
        target = decision.get("merge_with_node_id")
        if not isinstance(target, str) or not target.startswith("node-"):
            raise DecisionValidationError("merge_with_node_id must reference an existing node ID")
        if target == candidate_id:
            raise DecisionValidationError("a node cannot be merged with itself")
    return dict(decision)


def _candidate_snapshot(packet: Dict[str, Any]) -> Dict[str, Any]:
    """Copy only source fields already present in a non-node review packet."""
    block_id = packet.get("block_id")
    page_json = packet.get("page_json")
    source_text = packet.get("source_text")
    bbox = packet.get("bbox")
    level = packet.get("suggested_level", 3)
    if not isinstance(block_id, str) or not block_id.startswith("blk-"):
        raise DecisionValidationError("non-node candidate is missing a stable block_id")
    if not isinstance(page_json, int) or isinstance(page_json, bool) or page_json < 0:
        raise DecisionValidationError("non-node candidate is missing page_json")
    if not isinstance(source_text, str) or not source_text.strip():
        raise DecisionValidationError("non-node candidate is missing source_text")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox
    ):
        raise DecisionValidationError("non-node candidate is missing a four-number bbox")
    if not isinstance(level, int) or isinstance(level, bool) or not 2 <= level <= 6:
        raise DecisionValidationError("non-node candidate has an invalid suggested_level")
    return {
        "candidate_id": packet["candidate_id"],
        "block_id": block_id,
        "page_json": page_json,
        "source_text": source_text,
        "bbox": bbox,
        "suggested_level": level,
        "source_evidence": list(packet.get("evidence") or []),
    }


def compile_decisions(
    review_packets: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    base_overrides: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    packets = {}
    for packet in review_packets:
        candidate_id = packet.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise DecisionValidationError("review packet is missing candidate_id")
        if candidate_id in packets:
            raise DecisionValidationError(f"duplicate review packet: {candidate_id}")
        packets[candidate_id] = packet

    submitted = {}
    for raw in decisions:
        candidate_id = raw.get("candidate_id")
        if candidate_id not in packets:
            raise DecisionValidationError(f"decision has no review packet: {candidate_id}")
        if candidate_id in submitted:
            raise DecisionValidationError(f"duplicate decision: {candidate_id}")
        submitted[candidate_id] = validate_decision(raw, packets[candidate_id])
    missing = sorted(set(packets) - set(submitted))
    if missing:
        raise DecisionValidationError(
            "every review packet needs an explicit decision; missing: " + ", ".join(missing)
        )

    overrides = list(base_overrides or [])
    unresolved = []
    resolved = []
    for candidate_id in packets:
        decision = submitted[candidate_id]
        action = decision["decision"]
        trace = {
            "candidate_id": candidate_id,
            "reason_code": decision["reason_code"],
            "evidence_refs": decision["evidence_refs"],
        }
        if action == "needs_human":
            unresolved.append(trace)
            continue
        packet = packets[candidate_id]
        node_context = packet.get("node_context") or {}
        node_id = node_context.get("node_id")
        if not isinstance(node_id, str) or node_id != candidate_id:
            snapshot = _candidate_snapshot(packet)
            if action == "reject":
                overrides.append({
                    "operation": "reject_candidate",
                    **snapshot,
                    "review": trace,
                })
            elif action in {"accept", "change_level", "change_kind"}:
                level = decision.get("level", snapshot["suggested_level"])
                kind = decision.get("kind", "section")
                if kind == "toc_entry":
                    raise DecisionValidationError(
                        "a body text candidate cannot be promoted to toc_entry"
                    )
                overrides.append({
                    "operation": "promote_candidate",
                    **snapshot,
                    "level": level,
                    "kind": kind,
                    "review": trace,
                })
            elif action == "merge_with_previous":
                overrides.append({
                    "operation": "merge_candidate",
                    **snapshot,
                    "target_id": decision["merge_with_node_id"],
                    "review": trace,
                })
            else:
                raise DecisionValidationError(
                    f"unsupported non-node decision for {candidate_id}: {action}"
                )
            resolved.append(trace)
            continue
        record: Dict[str, Any] = {"operation": f"{action}_node", "node_id": node_id, "review": trace}
        if action == "change_level":
            record = {"operation": "update_node", "node_id": node_id, "set": {"level": decision["level"]}, "review": trace}
        elif action == "change_kind":
            record = {"operation": "update_node", "node_id": node_id, "set": {"kind": decision["kind"]}, "review": trace}
        elif action == "merge_with_previous":
            record = {
                "operation": "merge_nodes",
                "source_id": node_id,
                "target_id": decision["merge_with_node_id"],
                "review": trace,
            }
        overrides.append(record)
        resolved.append(trace)

    return {
        "schema_version": 1,
        "review_packets_sha256": _records_hash(review_packets),
        "decisions_sha256": _records_hash(decisions),
        "review_packet_count": len(review_packets),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "overrides": overrides,
        "resolved": resolved,
        "unresolved": unresolved,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise DecisionValidationError(f"JSONL line {line_number} must be an object")
        records.append(value)
    return records


def load_base_overrides(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("overrides") if isinstance(data, dict) else data
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise DecisionValidationError("base overrides must be a JSON list or compiled override bundle")
    return records


def compile_decision_files(
    review_path: Path,
    decision_path: Path,
    output_path: Path,
    base_override_path: Optional[Path] = None,
) -> Dict[str, Any]:
    bundle = compile_decisions(
        read_jsonl(review_path),
        read_jsonl(decision_path),
        load_base_overrides(base_override_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return bundle
