from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import resolve_path
from .io_utils import read_jsonl, write_json
from .migrate import HEADING, PLAIN_STRUCTURAL_HEADING, _structure_nodes, parse_existing_markdown


DECISIONS = {
    "accept_mapping",
    "reject_mapping",
    "link_existing_heading",
    "link_source_node",
    "leave_unmapped",
    "keep_unchanged",
    "demote_to_body",
    "needs_human",
}
ALLOWED_FIELDS = {"candidate_id", "decision", "reason", "existing_heading_id", "node_id", "target_level"}


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _decision_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    return [] if path is None else list(read_jsonl(path))


def _validate_decisions(
    reviews: List[Dict[str, Any]], decisions: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    review_by_id = {str(row["candidate_id"]): row for row in reviews}
    result: Dict[str, Dict[str, Any]] = {}
    for row in decisions:
        unknown_fields = set(row) - ALLOWED_FIELDS
        if unknown_fields:
            raise ValueError(f"Unknown migration decision fields: {sorted(unknown_fields)}")
        candidate_id = str(row.get("candidate_id") or "")
        decision = str(row.get("decision") or "")
        if candidate_id not in review_by_id:
            raise ValueError(f"Unknown migration candidate_id: {candidate_id}")
        if candidate_id in result:
            raise ValueError(f"Duplicate migration decision: {candidate_id}")
        if decision not in DECISIONS:
            raise ValueError(f"Unsupported migration decision: {decision}")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Migration decision requires a non-empty reason: {candidate_id}")
        review = review_by_id[candidate_id]
        allowed = set(review.get("allowed_decisions") or []) | {"needs_human"}
        if decision not in allowed:
            raise ValueError(f"Decision {decision} is not allowed for {candidate_id}")
        if decision == "link_existing_heading" and not row.get("existing_heading_id"):
            raise ValueError(f"link_existing_heading requires existing_heading_id: {candidate_id}")
        if decision == "link_source_node" and not row.get("node_id"):
            raise ValueError(f"link_source_node requires node_id: {candidate_id}")
        if decision != "link_existing_heading" and row.get("existing_heading_id"):
            raise ValueError(f"existing_heading_id is not allowed for {decision}: {candidate_id}")
        if decision != "link_source_node" and row.get("node_id"):
            raise ValueError(f"node_id is not allowed for {decision}: {candidate_id}")
        target_level = row.get("target_level")
        if target_level is not None:
            if decision != "accept_mapping":
                raise ValueError(f"target_level is only allowed for accept_mapping: {candidate_id}")
            if not isinstance(target_level, int) or isinstance(target_level, bool) or not 1 <= target_level <= 6:
                raise ValueError(f"target_level must be an integer from 1 to 6: {candidate_id}")
        result[candidate_id] = dict(row)
    return result


def _manual_links(
    reviews: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, Any]],
    nodes: Dict[str, Dict[str, Any]],
    headings: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], set[str]]:
    review_by_id = {str(row["candidate_id"]): row for row in reviews}
    source_candidates = {
        str(row["source_node"]["node_id"]): candidate_id
        for candidate_id, row in review_by_id.items()
        if row.get("reason") == "source_structure_node_unmatched"
    }
    heading_candidates = {
        str(row["existing_heading"]["heading_id"]): candidate_id
        for candidate_id, row in review_by_id.items()
        if row.get("reason") == "existing_heading_unmatched"
    }
    records: List[Dict[str, Any]] = []
    consumed: set[str] = set()
    for node_id, source_candidate_id in source_candidates.items():
        source_decision = decisions.get(source_candidate_id)
        if not source_decision or source_decision["decision"] != "link_existing_heading":
            continue
        heading_id = str(source_decision["existing_heading_id"])
        heading_candidate_id = heading_candidates.get(heading_id)
        if not heading_candidate_id:
            raise ValueError(f"Manual link target is not an unmatched heading: {heading_id}")
        heading_decision = decisions.get(heading_candidate_id)
        if not heading_decision or heading_decision.get("decision") != "link_source_node" or str(heading_decision.get("node_id")) != node_id:
            raise ValueError(
                f"Manual link must have reciprocal decisions for {node_id} and {heading_id}"
            )
        node = nodes.get(node_id)
        heading = headings.get(heading_id)
        if not node or not heading:
            raise ValueError(f"Manual link references unavailable evidence: {node_id}, {heading_id}")
        records.append({
            "node_id": node_id,
            "source_title": node.get("title_source"),
            "source_page_json": node.get("page_json"),
            "source_kind": node.get("kind"),
            "target_level": int(node.get("level", 2)),
            "existing_heading_id": heading_id,
            "existing_line": int(heading["line"]),
            "existing_page_json": heading.get("page_json"),
            "existing_title": heading.get("title"),
            "existing_level": int(heading["level"]),
            "action": "keep_level" if int(node.get("level", 2)) == int(heading["level"]) else "change_level",
            "score": None,
            "confidence": "human_confirmed",
            "evidence": ["reciprocal_human_link"],
        })
        consumed.update({source_candidate_id, heading_candidate_id})
    return records, consumed


def _replace_heading_level(raw_line: str, level: int) -> str:
    body = raw_line.rstrip("\r\n")
    ending = raw_line[len(body):]
    match = HEADING.match(body)
    if match:
        return "#" * level + body[len(match.group(1)):] + ending
    if PLAIN_STRUCTURAL_HEADING.match(body.strip()):
        return "#" * level + " " + body.strip() + ending
    raise ValueError("Migration patch target is no longer an approved heading candidate")


def _demote_heading(raw_line: str) -> str:
    body = raw_line.rstrip("\r\n")
    ending = raw_line[len(body):]
    match = HEADING.match(body)
    if not match:
        raise ValueError("Migration demotion target is no longer a Markdown heading")
    return match.group(2) + ending


def _protected_body_sha256(text: str, heading_lines: set[int]) -> str:
    protected = "".join(
        line
        for line_number, line in enumerate(text.splitlines(keepends=True), 1)
        if line_number not in heading_lines
    )
    return _sha256(protected.encode("utf-8"))


def create_migration_preview(config: Dict[str, Any], decisions_path: Optional[Path] = None) -> Dict[str, Any]:
    existing_value = config.get("existing_translation")
    if not existing_value:
        raise ValueError("Migration preview requires existing_translation")
    existing_path = Path(str(existing_value)).expanduser()
    if not existing_path.is_absolute():
        existing_path = (Path(config["_base_dir"]) / existing_path).resolve()
    output_dir = resolve_path(config, "output_dir")
    report_path = output_dir / "migration_report.json"
    alignment_path = output_dir / "migration_alignment.json"
    review_path = output_dir / "migration_review_packets.jsonl"
    for required in (report_path, alignment_path, review_path):
        if not required.exists():
            raise FileNotFoundError(f"Run audit-existing first: {required}")

    audit_report = json.loads(report_path.read_text(encoding="utf-8"))
    parsed = parse_existing_markdown(existing_path)
    if parsed["input_sha256"] != audit_report.get("input_sha256"):
        raise RuntimeError("Existing translation changed after migration audit; rerun audit-existing")
    structure_path = resolve_path(config, "structure_json")
    structure_bytes = structure_path.read_bytes()
    if _sha256(structure_bytes) != audit_report.get("structure_sha256"):
        raise RuntimeError("Structure changed after migration audit; rerun audit-existing")
    structure = json.loads(structure_bytes.decode("utf-8"))
    nodes = {str(node["id"]): node for node in _structure_nodes(structure)}
    headings = {str(row["heading_id"]): row for row in parsed["headings"]}
    records = json.loads(alignment_path.read_text(encoding="utf-8")).get("records", [])
    reviews = list(read_jsonl(review_path))
    decisions_path = decisions_path.expanduser().resolve() if decisions_path else None
    decisions = _validate_decisions(reviews, _decision_rows(decisions_path))

    selected: List[Dict[str, Any]] = [dict(row) for row in records if row.get("confidence") == "high"]
    review_by_id = {str(row["candidate_id"]): row for row in reviews}
    resolved: set[str] = set()
    demoted_heading_ids: set[str] = set()
    for candidate_id, decision in decisions.items():
        review = review_by_id[candidate_id]
        action = decision["decision"]
        if review.get("reason") == "migration_alignment_not_high_confidence":
            if action == "accept_mapping":
                accepted = dict(review["alignment"])
                if decision.get("target_level") is not None:
                    accepted["target_level"] = int(decision["target_level"])
                    accepted["action"] = (
                        "keep_level"
                        if int(accepted["existing_level"]) == int(accepted["target_level"])
                        else "change_level"
                    )
                accepted["confidence"] = "human_confirmed"
                accepted["evidence"] = list(accepted.get("evidence") or []) + [
                    "human_accept_mapping",
                    *( ["human_target_level_override"] if decision.get("target_level") is not None else [] ),
                ]
                selected.append(accepted)
            if action in {"accept_mapping", "reject_mapping"}:
                resolved.add(candidate_id)
        elif action in {"leave_unmapped", "keep_unchanged"}:
            resolved.add(candidate_id)
        elif action == "demote_to_body":
            demoted_heading_ids.add(str(review["existing_heading"]["heading_id"]))
            resolved.add(candidate_id)

    manual_records, linked_candidates = _manual_links(reviews, decisions, nodes, headings)
    selected.extend(manual_records)
    resolved.update(linked_candidates)

    selected_nodes = [str(row["node_id"]) for row in selected]
    selected_headings = [str(row["existing_heading_id"]) for row in selected]
    if len(selected_nodes) != len(set(selected_nodes)) or len(selected_headings) != len(set(selected_headings)):
        raise RuntimeError("Preview mappings are not one-to-one")
    if demoted_heading_ids.intersection(selected_headings):
        raise RuntimeError("A heading cannot be both mapped and demoted")

    source_bytes = existing_path.read_bytes()
    source_text = source_bytes.decode("utf-8")
    lines = source_text.splitlines(keepends=True)
    patches: List[Dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: int(item["existing_line"])):
        if row.get("action") != "change_level":
            continue
        line_number = int(row["existing_line"])
        before = lines[line_number - 1]
        after = _replace_heading_level(before, int(row["target_level"]))
        if before == after:
            continue
        lines[line_number - 1] = after
        patches.append({
            "operation": "change_heading_level",
            "line": line_number,
            "node_id": row["node_id"],
            "existing_heading_id": row["existing_heading_id"],
            "confidence": row["confidence"],
            "before": before,
            "after": after,
            "reverse": {"before": after, "after": before},
        })

    for heading_id in sorted(demoted_heading_ids, key=lambda value: int(headings[value]["line"])):
        heading = headings[heading_id]
        line_number = int(heading["line"])
        before = lines[line_number - 1]
        after = _demote_heading(before)
        lines[line_number - 1] = after
        patches.append({
            "operation": "demote_to_body",
            "line": line_number,
            "node_id": None,
            "existing_heading_id": heading_id,
            "confidence": "human_confirmed",
            "before": before,
            "after": after,
            "reverse": {"before": after, "after": before},
        })
    patches.sort(key=lambda item: int(item["line"]))

    preview_bytes = "".join(lines).encode("utf-8")
    preview_path = output_dir / "book_translated.preview.md"
    temporary = preview_path.with_suffix(preview_path.suffix + ".tmp")
    temporary.write_bytes(preview_bytes)
    os.replace(temporary, preview_path)

    preview_parsed = parse_existing_markdown(preview_path)
    source_titles = [
        row["title"] for row in parsed["headings"]
        if str(row["heading_id"]) not in demoted_heading_ids
    ]
    preview_titles = [row["title"] for row in preview_parsed["headings"]]
    if source_titles != preview_titles:
        raise RuntimeError("Preview changed existing heading text")
    protected_heading_lines = {int(row["line"]) for row in parsed["headings"]}
    protected_body_equal = (
        _protected_body_sha256(source_text, protected_heading_lines)
        == _protected_body_sha256(preview_bytes.decode("utf-8"), protected_heading_lines)
    )
    if not protected_body_equal:
        raise RuntimeError("Preview changed protected body content or page markers")
    if _sha256(existing_path.read_bytes()) != parsed["input_sha256"]:
        raise RuntimeError("Existing translation changed while writing preview")

    unresolved = [
        str(row["candidate_id"])
        for row in reviews
        if str(row["candidate_id"]) not in resolved
    ]
    upstream = dict(audit_report.get("upstream") or {})
    mapped_node_ids = {str(row["node_id"]) for row in selected}
    unrepresented_macro_nodes = [
        node_id
        for node_id, node in nodes.items()
        if node_id not in mapped_node_ids
        and (
            node.get("kind") in {"part", "chapter"}
            or node.get("identity_kind") in {"part", "chapter"}
        )
    ]
    ready = bool(
        upstream.get("validation_ok")
        and int(upstream.get("pending_reviews") or 0) == 0
        and not unresolved
        and not unrepresented_macro_nodes
    )
    patch_document = {
        "schema_version": 1,
        "source_sha256": parsed["input_sha256"],
        "preview_sha256": _sha256(preview_bytes),
        "reversible": True,
        "patches": patches,
    }
    write_json(output_dir / "migration_patch.json", patch_document)
    preview_report = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "preview_only": True,
        "existing_translation": str(existing_path),
        "preview": str(preview_path),
        "source_sha256": parsed["input_sha256"],
        "post_preview_source_sha256": _sha256(existing_path.read_bytes()),
        "preview_sha256": _sha256(preview_bytes),
        "structure_sha256": _sha256(structure_bytes),
        "non_heading_sha256_equal": protected_body_equal,
        "protected_body_sha256_equal": protected_body_equal,
        "heading_text_sequence_equal": source_titles == preview_titles,
        "selected_mappings": len(selected),
        "changed_heading_levels": sum(
            row.get("operation") == "change_heading_level" for row in patches
        ),
        "total_heading_patches": len(patches),
        "human_confirmed_mappings": sum(row.get("confidence") == "human_confirmed" for row in selected),
        "demoted_false_headings": len(demoted_heading_ids),
        "migration_review_packets": len(reviews),
        "resolved_review_packets": len(resolved),
        "unresolved_review_packets": len(unresolved),
        "unresolved_candidate_ids": unresolved,
        "unrepresented_macro_nodes": unrepresented_macro_nodes,
        "upstream": upstream,
        "ready_for_adoption": ready,
        "status": "ready_for_adoption" if ready else "partial_review_required",
    }
    write_json(output_dir / "migration_preview_report.json", preview_report)
    return preview_report
