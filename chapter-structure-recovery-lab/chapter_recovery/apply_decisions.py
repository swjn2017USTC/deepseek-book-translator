"""Deterministic v2 apply-decisions (plan P03 section 4.1; spec section 9).

This module is the chapter-side half of the LLM book structure resolver.  It
is fully deterministic and offline: it never imports or calls an LLM client,
never reads API keys, and never touches the network.  The LLM side
(translation repo) only writes two strict JSONL files; this module applies
them onto the deterministic ``recover_structure`` base tree and emits the v2
artifacts under ``<structure_dir>/v2/``.

Intake is fail-closed:

- decisions must start with a single ``_meta`` line (schema_version 2,
  book_id, evidence_sha256, prompts_version, ...) whose ``evidence_sha256``
  must match the live ``heading_candidates.jsonl`` + ``book_structure_evidence.json``
  bytes (staleness gate);
- every decision/record is validated against the strict schema of plan 3.2-3.5
  (``validate_decision_v2``): id regex + live existence, five-decision enum,
  ``kind``/``level`` only for ``heading``, level 1..6, confidence [0, 1],
  non-empty evidence_refs, repair_round 0..2, no free-text title keys, merge
  adjacency, page range;
- rescue ``blk-*`` ids must exist among the normalized OCR pages
  (``rescue_re_evidence``), which re-establishes full candidate evidence
  through the P02 candidate conventions (``make_candidate_id`` etc.).

Application follows plan amendment A1 (deterministic candidate -> node
mapping): a decided candidate maps to the first live base node that shares a
source block, then same-page+title equality, then same-page+numbering token;
when nothing matches and the candidate's blocks are free a new node is
created from the candidate (the only promotion path for rescued/text
candidates); when nothing matches but a block is occupied the run fails
closed.  Merge decisions fold the decided candidate into its adjacent
preceding candidate's node (title concatenation in block order).

Artifacts (atomic, byte-deterministic, no timestamps):

- ``<structure_dir>/v2/book_structure.json``      (schema_version 3)
- ``<structure_dir>/v2/validation.json``          (legacy + N1..N5 checks)
- ``<structure_dir>/v2/hierarchy_audit.json``
- ``<structure_dir>/v2/decision_provenance.json`` (per emitted node)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .confirmed import load_confirmed_toc, merge_confirmed_toc
from .decisions import DecisionValidationError, FORBIDDEN_TEXT_KEYS, KINDS
from .evidence_models import make_candidate_id
from .heading_text import canonical_heading_text, numbering, rejection_reason
from .models import Block, Node, TocEntry, stable_id
from .paddleocr import load_pages, normalize_blocks
from .page_map import PageMap, infer_page_map
from .recover import _candidate_parent, recover_structure
from .toc import find_toc_pages, parse_toc_entries
from .validate_v2 import validate_structure_v2
from .zones import infer_page_zones

# Re-exported so tests reference one vocabulary (plan 3.3 enums).
V2_DECISIONS = {"heading", "body", "caption", "apparatus", "merge"}
V2_HEADING = "heading"
V2_MERGE = "merge"
RESCUE = "rescue"

CANDIDATE_ID_PATTERN = re.compile(r"^cand-[0-9a-f]{16}$")
BLOCK_ID_PATTERN = re.compile(r"^blk-[0-9a-f]{16}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NODE_ID_PATTERN = re.compile(r"^node-[0-9a-f]{16}$")

APPARATUS_ZONES = {"notes", "bibliography", "index"}

META_ALLOWED_KEYS = {
    "schema_version", "book_id", "evidence_sha256", "decision_model",
    "grammar_model", "prompts_version", "generated_by", "generated_at",
    "repair_total",
}
DECISION_ALLOWED_KEYS = {
    "schema_version", "candidate_id", "decision", "kind", "level",
    "parent_candidate_id", "confidence", "evidence_refs", "decision_model",
    "repair_round", "prompts_version", "evidence_sha256",
}
RESCUE_ALLOWED_KEYS = {
    "block_id", "reason", "confidence", "evidence_refs", "rescue_model",
    "prompts_version",
}
# Free-text keys are rejected even when they overlap DECISION_ALLOWED_KEYS.
FORBIDDEN_TEXT_KEYS_V2 = FORBIDDEN_TEXT_KEYS | {"exact_text", "text"}

V2_HIERARCHY_AUDIT_CODES = {
    "parent_before_child", "sibling_numbering_monotonic",
    "duplicate_block_occupation", "zone_consistency", "orphan_section",
}


# ---------------------------------------------------------------------------
# Reading the strict JSONL inputs
# ---------------------------------------------------------------------------


def _read_json_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise DecisionValidationError(f"missing file: {path}")
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionValidationError(
                f"{path.name} line {line_number} is not valid JSON: {exc.msg}"
            )
        if not isinstance(value, dict):
            raise DecisionValidationError(
                f"{path.name} line {line_number} must be a JSON object"
            )
        records.append(value)
    return records


def _require_type(
    record: Dict[str, Any],
    key: str,
    kinds: Tuple[type, ...],
    context: str,
) -> None:
    value = record.get(key)
    if value is None or not isinstance(value, kinds) or (
        isinstance(value, bool) and int not in kinds and float not in kinds
    ):
        raise DecisionValidationError(
            f"{context}: field {key!r} must be one of "
            f"{[kind.__name__ for kind in kinds]}"
        )


def read_decisions_jsonl(path: Path) -> Dict[str, Any]:
    """Read a meta-first decisions JSONL (plan 3.2).

    Returns ``{"meta": {...}, "decisions": [...]}``.  The first non-empty
    line must be the single ``_meta`` object; every later line is one
    decision record.  Malformed structure raises ``DecisionValidationError``.
    """
    records = _read_json_lines(path)
    if not records:
        raise DecisionValidationError(f"{path.name} is empty: missing _meta line")
    first = records[0]
    if set(first) != {"_meta"}:
        raise DecisionValidationError(
            f"{path.name} line 1 must be a single-key _meta object"
        )
    meta = first["_meta"]
    if not isinstance(meta, dict):
        raise DecisionValidationError(f"{path.name}: _meta must be an object")
    unknown = set(meta) - META_ALLOWED_KEYS
    if unknown:
        raise DecisionValidationError(
            f"{path.name}: unknown _meta keys: {sorted(unknown)}"
        )
    _require_type(meta, "schema_version", (int,), "meta")
    if meta.get("schema_version") != 2:
        raise DecisionValidationError(
            f"{path.name}: meta schema_version must be 2, got "
            f"{meta.get('schema_version')!r}"
        )
    _require_type(meta, "book_id", (str,), "meta")
    if not str(meta.get("book_id") or "").strip():
        raise DecisionValidationError(f"{path.name}: meta book_id must be non-empty")
    _require_type(meta, "evidence_sha256", (str,), "meta")
    if not SHA256_PATTERN.match(str(meta.get("evidence_sha256") or "")):
        raise DecisionValidationError(
            f"{path.name}: meta evidence_sha256 must be a 64-hex sha256"
        )
    _require_type(meta, "prompts_version", (int,), "meta")
    for key in ("decision_model", "grammar_model", "generated_by", "generated_at"):
        if meta.get(key) is not None and not isinstance(meta.get(key), str):
            raise DecisionValidationError(
                f"{path.name}: meta {key!r} must be a string when present"
            )
    if meta.get("repair_total") is not None:
        _require_type(meta, "repair_total", (int,), "meta")

    decisions = []
    for line_number, record in enumerate(records[1:], 2):
        if "_meta" in record:
            raise DecisionValidationError(
                f"{path.name} line {line_number}: _meta must be the first line only"
            )
        decisions.append(record)
    return {"meta": meta, "decisions": decisions}


def read_rescue_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a rescue JSONL (plan 3.5; one object per line, no meta line)."""
    records = _read_json_lines(path)
    validated = []
    for line_number, record in enumerate(records, 1):
        unknown = set(record) - RESCUE_ALLOWED_KEYS
        if unknown:
            raise DecisionValidationError(
                f"{path.name} line {line_number}: unknown rescue keys: "
                f"{sorted(unknown)}"
            )
        _require_type(record, "block_id", (str,), f"rescue line {line_number}")
        if not BLOCK_ID_PATTERN.match(str(record.get("block_id") or "")):
            raise DecisionValidationError(
                f"{path.name} line {line_number}: block_id must match "
                f"^blk-[0-9a-f]{{16}}$"
            )
        _require_type(record, "reason", (str,), f"rescue line {line_number}")
        if not str(record.get("reason") or "").strip():
            raise DecisionValidationError(
                f"{path.name} line {line_number}: reason must be non-empty"
            )
        confidence = record.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise DecisionValidationError(
                f"{path.name} line {line_number}: confidence must be in [0, 1]"
            )
        evidence = record.get("evidence_refs")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            raise DecisionValidationError(
                f"{path.name} line {line_number}: evidence_refs must be a "
                f"non-empty string list"
            )
        _require_type(record, "rescue_model", (str,), f"rescue line {line_number}")
        if not str(record.get("rescue_model") or "").strip():
            raise DecisionValidationError(
                f"{path.name} line {line_number}: rescue_model must be non-empty"
            )
        _require_type(record, "prompts_version", (int,), f"rescue line {line_number}")
        validated.append(record)
    return validated


# ---------------------------------------------------------------------------
# Strict per-decision intake validation
# ---------------------------------------------------------------------------


def _add_error(errors: List[Dict[str, Any]], code: str, message: str) -> None:
    errors.append({"code": code, "message": message})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _adjacent_same_page(
    candidates_by_id: Dict[str, Dict[str, Any]],
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:
    """True when two block-backed candidates are consecutive on one page.

    Consecutive means no other block-backed candidate's primary bbox top lies
    between the two tops in reading order (ties broken by bbox left then id).
    """
    page = left.get("page_json")
    if page is None or page != right.get("page_json"):
        return False
    if not left.get("source_block_ids") or not right.get("source_block_ids"):
        return False
    left_bbox, right_bbox = left.get("bbox"), right.get("bbox")
    if not isinstance(left_bbox, list) or len(left_bbox) != 4:
        return False
    if not isinstance(right_bbox, list) or len(right_bbox) != 4:
        return False
    ordered = sorted(
        [
            candidate
            for candidate in candidates_by_id.values()
            if candidate.get("page_json") == page
            and candidate.get("source_block_ids")
            and isinstance(candidate.get("bbox"), list)
            and len(candidate["bbox"]) == 4
        ],
        key=lambda item: (item["bbox"][1], item["bbox"][0], item["candidate_id"]),
    )
    try:
        left_index = ordered.index(left)
        right_index = ordered.index(right)
    except ValueError:
        return False
    return abs(left_index - right_index) == 1


def merge_target_candidate(
    decision: Dict[str, Any],
    candidates_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the merge target of a ``merge`` decision (deterministic).

    The explicit ``parent_candidate_id`` wins when present; otherwise the
    target is the nearest block-backed candidate immediately above the
    decided candidate on the same page.  Returns None when no adjacent
    target exists.
    """
    candidate = candidates_by_id.get(str(decision.get("candidate_id") or ""))
    if candidate is None:
        return None
    page = candidate.get("page_json")
    if not isinstance(page, int) or isinstance(page, bool):
        return None
    bbox = candidate.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    explicit = decision.get("parent_candidate_id")
    if explicit is not None:
        target = candidates_by_id.get(str(explicit))
        if target is None or target.get("page_json") != page:
            return None
        return (
            target if _adjacent_same_page(candidates_by_id, target, candidate) else None
        )
    above = [
        other
        for other in candidates_by_id.values()
        if other.get("page_json") == page
        and other.get("source_block_ids")
        and isinstance(other.get("bbox"), list)
        and len(other["bbox"]) == 4
        and other["candidate_id"] != candidate["candidate_id"]
        and float(other["bbox"][1]) < float(bbox[1])
    ]
    if not above:
        return None
    nearest = max(
        above,
        key=lambda item: (item["bbox"][1], item["bbox"][0], item["candidate_id"]),
    )
    return (
        nearest if _adjacent_same_page(candidates_by_id, nearest, candidate) else None
    )


def validate_decision_v2(
    decision: Dict[str, Any],
    candidates_by_id: Dict[str, Dict[str, Any]],
    page_count: int,
    allowed_kinds: Set[str],
    evidence_sha256: str,
) -> List[Dict[str, Any]]:
    """Strict intake validation of one v2 decision record.

    Returns a list of error dicts ``{"code": ..., "message": ...}``; an
    empty list means the record is valid.  Raises nothing by itself.
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(decision, dict):
        _add_error(errors, "not_an_object", "decision record must be an object")
        return errors

    unknown = set(decision) - DECISION_ALLOWED_KEYS
    forbidden = unknown & FORBIDDEN_TEXT_KEYS_V2
    if forbidden:
        _add_error(
            errors,
            "forbidden_text_key",
            f"decisions must not supply title text; found: {sorted(forbidden)}",
        )
    if unknown - FORBIDDEN_TEXT_KEYS_V2:
        _add_error(
            errors,
            "unknown_top_level_key",
            f"unknown decision keys: {sorted(unknown - FORBIDDEN_TEXT_KEYS_V2)}",
        )

    for key in ("schema_version", "candidate_id", "decision", "repair_round"):
        if decision.get(key) is None:
            _add_error(errors, "missing_field", f"missing decision field: {key}")
    if "schema_version" in decision and (
        not isinstance(decision["schema_version"], int)
        or isinstance(decision["schema_version"], bool)
        or decision["schema_version"] != 2
    ):
        _add_error(
            errors,
            "schema_version_mismatch",
            f"schema_version must be 2, got {decision.get('schema_version')!r}",
        )

    candidate_id = decision.get("candidate_id")
    if candidate_id is not None and not isinstance(candidate_id, str):
        _add_error(
            errors,
            "bad_candidate_id",
            f"candidate_id must be a string matching "
            f"^cand-[0-9a-f]{{16}}$: {candidate_id!r}",
        )
    if isinstance(candidate_id, str) and not CANDIDATE_ID_PATTERN.match(candidate_id):
        _add_error(
            errors,
            "bad_candidate_id",
            f"candidate_id must match ^cand-[0-9a-f]{{16}}$: {candidate_id!r}",
        )
    candidate = candidates_by_id.get(candidate_id) if isinstance(candidate_id, str) else None
    if isinstance(candidate_id, str) and CANDIDATE_ID_PATTERN.match(candidate_id) and candidate is None:
        _add_error(
            errors,
            "unknown_candidate_id",
            f"candidate_id is not in the live candidate set: {candidate_id}",
        )
    if candidate is not None:
        page_json = candidate.get("page_json")
        if isinstance(page_json, int) and not (
            0 <= page_json < page_count
        ):
            _add_error(
                errors,
                "page_out_of_range",
                f"candidate {candidate_id} page_json {page_json} is outside "
                f"[0, {page_count})",
            )

    action = decision.get("decision")
    if action is None or not isinstance(action, str) or action not in V2_DECISIONS:
        _add_error(
            errors,
            "bad_decision",
            f"decision must be one of {sorted(V2_DECISIONS)}, got {action!r}",
        )

    parent_id = decision.get("parent_candidate_id")
    if parent_id is not None:
        if not isinstance(parent_id, str) or not CANDIDATE_ID_PATTERN.match(parent_id):
            _add_error(
                errors,
                "bad_parent_id",
                f"parent_candidate_id must be null or match "
                f"^cand-[0-9a-f]{{16}}$: {parent_id!r}",
            )
        elif parent_id not in candidates_by_id:
            _add_error(
                errors,
                "unknown_parent_candidate_id",
                f"parent_candidate_id is not in the live candidate set: {parent_id}",
            )
        elif (
            action == "heading"
            and parent_id == candidate_id
        ):
            _add_error(
                errors,
                "self_parent",
                "a heading cannot be its own parent",
            )

    if action == V2_HEADING:
        kind = decision.get("kind")
        if kind not in allowed_kinds:
            _add_error(
                errors,
                "bad_kind",
                f"kind must be one of {sorted(allowed_kinds)} when heading, "
                f"got {kind!r}",
            )
        level = decision.get("level")
        if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 6:
            _add_error(
                errors,
                "bad_level",
                f"level must be an integer 1..6 when heading, got {level!r}",
            )
    else:
        if isinstance(action, str) and action in V2_DECISIONS:
            if decision.get("kind") is not None:
                _add_error(
                    errors,
                    "unexpected_kind",
                    f"kind is only valid for heading decisions: {action}",
                )
            if decision.get("level") is not None:
                _add_error(
                    errors,
                    "unexpected_level",
                    f"level is only valid for heading decisions: {action}",
                )
            if parent_id is not None and action not in (V2_HEADING, V2_MERGE):
                _add_error(
                    errors,
                    "unexpected_parent",
                    f"parent_candidate_id is only meaningful for "
                    f"heading/merge decisions: {action}",
                )

    confidence = decision.get("confidence")
    if confidence is None or not _is_number(confidence) or not (
        0.0 <= float(confidence) <= 1.0
    ):
        _add_error(
            errors,
            "bad_confidence",
            f"confidence must be a number in [0, 1], got {confidence!r}",
        )
    evidence = decision.get("evidence_refs")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        _add_error(
            errors,
            "empty_evidence_refs",
            "evidence_refs must be a non-empty string list",
        )
    repair_round = decision.get("repair_round")
    if repair_round is None or not isinstance(repair_round, int) or isinstance(
        repair_round, bool
    ) or not 0 <= repair_round <= 2:
        _add_error(
            errors,
            "bad_repair_round",
            f"repair_round must be an integer 0..2, got {repair_round!r}",
        )
    prompts_version = decision.get("prompts_version")
    if prompts_version is not None and (
        not isinstance(prompts_version, int) or isinstance(prompts_version, bool)
    ):
        _add_error(
            errors,
            "bad_prompts_version",
            f"prompts_version must be an integer when present: {prompts_version!r}",
        )
    decision_model = decision.get("decision_model")
    if decision_model is not None and (
        not isinstance(decision_model, str) or not decision_model.strip()
    ):
        _add_error(
            errors,
            "bad_decision_model",
            "decision_model must be a non-empty string when present",
        )
    record_sha = decision.get("evidence_sha256")
    if record_sha is not None and (
        not isinstance(record_sha, str) or record_sha != evidence_sha256
    ):
        _add_error(
            errors,
            "stale_evidence_sha256",
            "evidence_sha256 does not match the live evidence bytes",
        )

    if action == V2_MERGE and not errors:
        if candidate is None:
            _add_error(
                errors,
                "unknown_candidate_id",
                f"merge candidate is not in the live candidate set: {candidate_id}",
            )
        elif not candidate.get("source_block_ids"):
            _add_error(
                errors,
                "merge_without_blocks",
                "merge requires a block-backed candidate",
            )
        elif merge_target_candidate(decision, candidates_by_id) is None:
            _add_error(
                errors,
                "merge_not_adjacent",
                f"merge candidate {candidate_id} has no adjacent preceding "
                f"candidate on the same page",
            )
    return errors


# ---------------------------------------------------------------------------
# Rescue re-evidence (plan 3.5 / 4.5): deterministic candidates from blocks
# ---------------------------------------------------------------------------


def _round_bbox(bbox: Sequence[float]) -> List[float]:
    return [round(float(value), 4) for value in bbox]


def rescue_re_evidence(
    block_ids: List[str],
    pages: List[List[Block]],
    page_map: PageMap,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Synthesize candidate records (cand-*) from existing OCR blocks.

    Every id must match ``^blk-[0-9a-f]{16}$`` AND exist among the
    normalized blocks of ``pages``; anything else raises
    ``DecisionValidationError`` (fail closed).  Candidates reuse the P02
    evidence conventions (``make_candidate_id``, canonical heading text,
    normalized rounded bbox, numbering, per-page zone, contexts).
    """
    blocks_by_id: Dict[str, Block] = {}
    for page in pages:
        for block in page:
            blocks_by_id[block.id] = block
    book_id = str(config.get("book_id") or "")
    if not book_id:
        raise DecisionValidationError("config is missing book_id for rescue re-evidence")
    offset = page_map.offset
    zones = infer_page_zones(pages)

    seen: Set[str] = set()
    output: List[Dict[str, Any]] = []
    missing = []
    for block_id in block_ids:
        if block_id in seen:
            continue
        seen.add(block_id)
        if not isinstance(block_id, str) or not BLOCK_ID_PATTERN.match(block_id):
            missing.append(f"{block_id!r} (malformed)")
            continue
        block = blocks_by_id.get(block_id)
        if block is None:
            missing.append(block_id)
            continue
        text = canonical_heading_text(block.clean_content)
        if not text:
            raise DecisionValidationError(
                f"rescue block has no canonical text: {block_id}"
            )
        zone = zones.get(block.page_index, "mainmatter")
        page_blocks = pages[block.page_index]
        index = page_blocks.index(block)
        before = page_blocks[index - 1].clean_content[:80] if index > 0 else None
        after = (
            page_blocks[index + 1].clean_content[:80]
            if index + 1 < len(page_blocks)
            else None
        )
        suppressions: List[str] = []
        reason = rejection_reason(text)
        if reason:
            suppressions.append(reason)
        if zone in APPARATUS_ZONES:
            suppressions.append("apparatus_zone")
        parsed = numbering(text)
        candidate_id = make_candidate_id(
            book_id, block.page_index, [block.id], text
        )
        output.append(
            {
                "candidate_id": candidate_id,
                "source_block_ids": [block.id],
                "exact_text": text,
                "page_json": block.page_index,
                "page_print": (
                    int(block.page_index + offset) if offset is not None else None
                ),
                "bbox": _round_bbox(block.normalized_bbox()),
                "zone": zone,
                "paddle_label": block.label,
                "paddle_relevel": None,
                "pdf_bookmark": None,
                "native_font": None,
                "numbering": (
                    {
                        "token": parsed.token,
                        "path": list(parsed.path),
                        "depth": parsed.depth,
                        "family": parsed.family,
                    }
                    if parsed is not None
                    else None
                ),
                "toc_matches": [],
                "repetition": {"count": 1, "nearby_count": 1},
                "before_context": before,
                "after_context": after,
                "deterministic_suppressions": sorted(set(suppressions)),
                "evidence_refs": [f"paddle:{block.id}:{block.label}"],
            }
        )
    if missing:
        raise DecisionValidationError(
            "rescue block ids must exist among the OCR pages; unknown: "
            + ", ".join(missing)
        )
    return output


# ---------------------------------------------------------------------------
# Deterministic candidate -> node mapping + decision application (plan A1)
# ---------------------------------------------------------------------------


def _occupied_block_ids(nodes: List[Node]) -> Set[str]:
    occupied: Set[str] = set()
    for node in nodes:
        if node.ignored:
            continue
        if node.block_id:
            occupied.add(node.block_id)
        occupied.update(node.source_block_ids)
    return occupied


def _map_candidate(
    candidate: Dict[str, Any], nodes: List[Node]
) -> Optional[Node]:
    """Plan A1: map a decided candidate onto a live base node.

    Priority order, first match wins (rule 1 across all nodes, then rule 2,
    then rule 3):
      1. source_block_ids overlap (node.block_id or source_block_ids);
      2. same page_json and casefold-canonical title equality;
      3. same page_json and equal numbering token.
    """
    blocks = set(candidate.get("source_block_ids") or [])
    page = candidate.get("page_json")
    candidate_text = canonical_heading_text(
        str(candidate.get("exact_text") or "")
    ).casefold()
    numbering_data = candidate.get("numbering")
    candidate_token = (
        numbering_data.get("token")
        if isinstance(numbering_data, dict)
        else None
    )
    ordered = sorted(
        nodes,
        key=lambda node: (
            node.page_json if node.page_json is not None else -1,
            node.bbox[1] if node.bbox is not None else -1.0,
            node.id,
        ),
    )
    if blocks:
        for node in ordered:
            if node.block_id in blocks or (
                blocks.intersection(node.source_block_ids)
            ):
                return node
    if page is not None and candidate_text:
        for node in ordered:
            if node.page_json != page:
                continue
            if (
                canonical_heading_text(node.title_source).casefold()
                == candidate_text
            ):
                return node
    if page is not None and candidate_token is not None:
        for node in ordered:
            if node.page_json != page:
                continue
            if node.ordinal == candidate_token:
                return node
    return None


def _new_node_id(book_id: str, candidate_id: str) -> str:
    return "node-" + stable_id(book_id, candidate_id, "v2-heading")


def _new_node_from_candidate(
    candidate: Dict[str, Any],
    book_id: str,
    book_node_id: str,
    page_map: PageMap,
    level: int,
    kind: str,
    parent_id: Optional[str],
    confidence: float,
    model: Optional[str],
    prompts_version: Optional[int],
    nodes: List[Node],
) -> Node:
    page_json = candidate.get("page_json")
    bbox = candidate.get("bbox")
    bbox_tuple = (
        tuple(float(value) for value in bbox)
        if isinstance(bbox, list) and len(bbox) == 4
        else None
    )
    page_print = candidate.get("page_print")
    if page_print is None and isinstance(page_json, int) and page_map.offset is not None:
        page_print = int(page_json + page_map.offset)
    source_blocks = list(candidate.get("source_block_ids") or [])
    parsed = numbering(candidate.get("exact_text") or "")
    if parent_id is None:
        parent_id = _candidate_parent(
            nodes,
            page_json if isinstance(page_json, int) else -1,
            bbox_tuple if bbox_tuple is not None else (0.0, 0.0, 0.0, 0.0),
            level,
            book_node_id,
        )
    paddle_label = candidate.get("paddle_label")
    if isinstance(paddle_label, str) and paddle_label.strip():
        source = paddle_label
    elif candidate.get("pdf_bookmark") is not None:
        source = "bookmark"
    elif candidate.get("native_font") is not None:
        source = "font"
    else:
        source = "evidence"
    evidence = [
        f"v2_source:{candidate.get('candidate_id')}",
    ]
    if model:
        evidence.append(f"v2_decision_model:{model}")
    if prompts_version is not None:
        evidence.append(f"v2_prompts_version:{prompts_version}")
    return Node(
        id=_new_node_id(book_id, candidate["candidate_id"]),
        kind=kind,
        level=level,
        title_source=str(candidate.get("exact_text") or "").strip(),
        page_json=page_json if isinstance(page_json, int) else None,
        page_print=page_print,
        parent_id=parent_id,
        block_id=source_blocks[0] if source_blocks else None,
        source=source,
        confidence=float(confidence),
        evidence=evidence,
        bbox=bbox_tuple,
        zone=str(candidate.get("zone") or "mainmatter"),
        ordinal=parsed.token if parsed else None,
        numbering_path=list(parsed.path) if parsed else [],
        source_block_ids=source_blocks,
    )


def _record_touch(
    touched: Dict[str, List[Dict[str, Any]]],
    node_id: str,
    decision: Dict[str, Any],
) -> None:
    touched.setdefault(node_id, []).append(dict(decision))


def _apply_heading(
    node: Node,
    decision: Dict[str, Any],
    candidates_by_id: Dict[str, Dict[str, Any]],
    nodes: List[Node],
    touched: Dict[str, List[Dict[str, Any]]],
    node_evidence: str,
) -> None:
    node.kind = decision["kind"]
    node.level = decision["level"]
    node.confidence = float(decision["confidence"])
    parent_id = decision.get("parent_candidate_id")
    if parent_id is not None:
        parent_candidate = candidates_by_id.get(parent_id)
        parent_node = (
            _map_candidate(parent_candidate, nodes)
            if parent_candidate is not None
            else None
        )
        if parent_node is not None and parent_node.id != node.id:
            node.parent_id = parent_node.id
        else:
            # Unresolved (or self-referential) parent: keep the current
            # deterministic parent and record the trace.
            node.evidence.append(f"v2_parent_unresolved:{parent_id}")
    if node.ignored:
        node.ignored = False
    node.evidence.append(node_evidence)
    _record_touch(touched, node.id, decision)


def _redirect_children(nodes: List[Node], old_parent_id: str, new_parent_id: str) -> None:
    for child in nodes:
        if child.parent_id == old_parent_id:
            child.parent_id = new_parent_id
            child.evidence.append("parent_redirected_after_v2_decision")


def _apply_decisions_to_base(
    nodes: List[Node],
    candidates_by_id: Dict[str, Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    page_map: PageMap,
    book_id: str,
    book_node_id: str,
) -> Dict[str, Any]:
    """Apply validated decisions onto the live base node list (in file order)."""
    touched: Dict[str, List[Dict[str, Any]]] = {}

    def _occupied() -> Set[str]:
        return _occupied_block_ids(nodes)

    for decision in decisions:
        action = decision["decision"]
        candidate = candidates_by_id[decision["candidate_id"]]
        node_evidence = f"v2_decision:{action}"
        if action != V2_MERGE:
            node = _map_candidate(candidate, nodes)
            if action == V2_HEADING:
                if node is None:
                    blocks = set(candidate.get("source_block_ids") or [])
                    if blocks.intersection(_occupied()):
                        raise DecisionValidationError(
                            f"candidate {decision['candidate_id']} has no base "
                            f"node and its source blocks are occupied "
                            f"(duplicate occupation)"
                        )
                    node = _new_node_from_candidate(
                        candidate,
                        book_id,
                        book_node_id,
                        page_map,
                        decision["level"],
                        decision["kind"],
                        None,
                        decision["confidence"],
                        decision.get("decision_model"),
                        decision.get("prompts_version"),
                        nodes,
                    )
                    nodes.append(node)
                _apply_heading(
                    node,
                    decision,
                    candidates_by_id,
                    nodes,
                    touched,
                    node_evidence,
                )
            else:  # body / caption / apparatus -> demote
                if node is not None:
                    if not node.ignored:
                        node.ignored = True
                        node.confidence = float(decision["confidence"])
                        _redirect_children(
                            nodes, node.id, node.parent_id or book_node_id
                        )
                        node.evidence.append(node_evidence)
                    _record_touch(touched, node.id, decision)
            continue

        # merge: fold the decided candidate into its adjacent preceding node.
        target_candidate = merge_target_candidate(decision, candidates_by_id)
        if target_candidate is None:
            raise DecisionValidationError(
                f"merge candidate {decision['candidate_id']} has no adjacent "
                f"preceding candidate on the same page"
            )
        target_node = _map_candidate(target_candidate, nodes)
        if target_node is None:
            raise DecisionValidationError(
                f"merge target candidate {target_candidate['candidate_id']} "
                f"has no base node"
            )
        source_node = _map_candidate(candidate, nodes)
        if source_node is not None and source_node.id == target_node.id:
            source_node = None  # already the target: nothing separate to fold
        source_text = (
            str(candidate.get("exact_text") or "").strip()
            if source_node is None
            else source_node.title_source.strip()
        )
        if not source_text:
            raise DecisionValidationError(
                f"merge candidate {decision['candidate_id']} has no text"
            )
        target_node.title_source = " ".join(
            value
            for value in (target_node.title_source.strip(), source_text)
            if value
        )
        added_blocks = list(candidate.get("source_block_ids") or [])
        if source_node is not None and source_node.block_id:
            added_blocks.append(source_node.block_id)
        target_node.source_block_ids = list(
            dict.fromkeys(
                [
                    *target_node.source_block_ids,
                    *([target_node.block_id] if target_node.block_id else []),
                    *added_blocks,
                ]
            )
        )
        candidate_bbox = candidate.get("bbox")
        if (
            target_node.page_json == candidate.get("page_json")
            and target_node.bbox is not None
            and isinstance(candidate_bbox, list)
            and len(candidate_bbox) == 4
        ):
            target_node.bbox = (
                min(target_node.bbox[0], float(candidate_bbox[0])),
                min(target_node.bbox[1], float(candidate_bbox[1])),
                max(target_node.bbox[2], float(candidate_bbox[2])),
                max(target_node.bbox[3], float(candidate_bbox[3])),
            )
        target_node.confidence = float(decision["confidence"])
        target_node.evidence.append(node_evidence)
        _record_touch(touched, target_node.id, decision)
        if source_node is not None:
            if not source_node.ignored:
                source_node.ignored = True
                _redirect_children(nodes, source_node.id, target_node.id)
                source_node.evidence.append("merged_by_v2_decision")
            _record_touch(touched, source_node.id, decision)
    return {"nodes": nodes, "touched": touched}


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def _json_text(value: Any, indent: Optional[int] = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent) + "\n"


def _atomic_write_json(path: Path, value: Any, indent: Optional[int] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json_text(value, indent=indent), encoding="utf-8")
    os.replace(tmp, path)


def _hierarchy_audit(validation: Dict[str, Any]) -> Dict[str, Any]:
    findings = [
        item
        for item in [
            *validation.get("errors", []),
            *validation.get("warnings", []),
        ]
        if str(item.get("code", "")).startswith(
            ("hierarchy_", "heading_level_", "book_not_", "active_book_title_")
        )
        or item.get("code") in V2_HIERARCHY_AUDIT_CODES
    ]
    return {
        "schema_version": 1,
        "heading_level_counts": validation.get("metrics", {}).get(
            "heading_level_counts", {}
        ),
        "book_title_duplicates_suppressed": validation.get("metrics", {}).get(
            "book_title_duplicates_suppressed", 0
        ),
        "hierarchy_findings": findings,
    }


def _decision_provenance(
    nodes: List[Node],
    touched: Dict[str, List[Dict[str, Any]]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    meta_model = meta.get("decision_model")
    meta_prompts = meta.get("prompts_version")
    records = []
    for node in nodes:
        touches = touched.get(node.id)
        if touches:
            last = touches[-1]
            records.append(
                {
                    "node_id": node.id,
                    "candidate_id": last.get("candidate_id"),
                    "decision_origin": "llm",
                    "decision_model": last.get("decision_model") or meta_model,
                    "repair_round": last.get("repair_round"),
                    "prompts_version": last.get("prompts_version")
                    if last.get("prompts_version") is not None
                    else meta_prompts,
                }
            )
        else:
            records.append(
                {
                    "node_id": node.id,
                    "candidate_id": None,
                    "decision_origin": "deterministic",
                    "decision_model": None,
                    "repair_round": 0,
                    "prompts_version": meta_prompts,
                }
            )
    return {
        "schema_version": 1,
        "book_id": meta.get("book_id"),
        "repair_total": meta.get("repair_total", 0),
        "nodes": records,
    }


def _sort_nodes(nodes: List[Node]) -> None:
    nodes.sort(
        key=lambda node: (
            node.page_json is not None,
            node.page_json if node.page_json is not None else -1,
            node.bbox[1] if node.bbox is not None else -1,
            node.level,
        )
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _evidence_sha256(evidence_dir: Path) -> str:
    candidates_path = evidence_dir / "heading_candidates.jsonl"
    packet_path = evidence_dir / "book_structure_evidence.json"
    if not candidates_path.is_file() or not packet_path.is_file():
        raise DecisionValidationError(
            f"evidence files missing under {evidence_dir}: "
            f"need heading_candidates.jsonl and book_structure_evidence.json"
        )
    digest = hashlib.sha256()
    digest.update(candidates_path.read_bytes())
    digest.update(packet_path.read_bytes())
    return digest.hexdigest()


def apply_decisions(
    config_path: Path,
    decisions_path: Path,
    rescue_path: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Deterministically apply v2 decisions onto one book's base structure.

    Reads the chapter config + OCR pages, rebuilds the deterministic base
    tree (``recover_structure``), loads the live evidence candidates, strict-
    intakes the decisions + rescues (fail closed), applies plan A1's
    deterministic mapping, validates with ``validate_structure_v2`` and
    writes the four v2 artifacts under ``<structure_dir>/v2/``.
    """
    config_path = config_path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    input_path = _resolve_path(base, str(config["input_json"]))
    structure_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else _resolve_path(base, str(config["output_dir"]))
    )
    override_path = (
        _resolve_path(base, str(config["overrides"])) if config.get("overrides") else None
    )
    book_id = str(config["book_id"])
    book_node_id = "book-" + stable_id(book_id)

    packet = read_decisions_jsonl(decisions_path.expanduser().resolve())
    meta = packet["meta"]
    decisions = packet["decisions"]
    rescues = read_rescue_jsonl(rescue_path.expanduser().resolve())

    evidence_sha = _evidence_sha256(structure_dir / "evidence")
    if meta.get("evidence_sha256") != evidence_sha:
        raise DecisionValidationError(
            "decisions are stale: evidence_sha256 in the decisions meta does "
            "not match the live evidence files under "
            f"{structure_dir / 'evidence'}"
        )
    if meta.get("book_id") != book_id:
        raise DecisionValidationError(
            f"decisions meta book_id {meta.get('book_id')!r} does not match "
            f"config book_id {book_id!r}"
        )

    raw_pages = load_pages(input_path)
    pages = normalize_blocks(raw_pages)
    page_map = infer_page_map(pages)
    page_count = len(pages)
    toc_pages = find_toc_pages(pages, config.get("toc_search_pages", [0, 30]))
    toc_entries = parse_toc_entries(pages, toc_pages)
    if config.get("confirmed_toc"):
        confirmed_path = _resolve_path(base, str(config["confirmed_toc"]))
        toc_entries = merge_confirmed_toc(
            toc_entries, load_confirmed_toc(confirmed_path)
        )

    # Deterministic base tree (analyze-equivalent; overrides honored as in
    # the legacy analyze flow).
    base_nodes, _text_candidates = recover_structure(
        pages, toc_entries, page_map, config, override_path
    )
    nodes = base_nodes

    # Live candidate index: on-disk evidence first, then rescue re-evidence.
    candidates_by_id: Dict[str, Dict[str, Any]] = {}
    candidates_path = structure_dir / "evidence" / "heading_candidates.jsonl"
    for record in _read_json_lines(candidates_path):
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not CANDIDATE_ID_PATTERN.match(candidate_id):
            raise DecisionValidationError(
                f"malformed heading_candidates.jsonl record: missing or bad "
                f"candidate_id"
            )
        if candidate_id in candidates_by_id:
            raise DecisionValidationError(
                f"duplicate candidate_id in heading_candidates.jsonl: {candidate_id}"
            )
        candidates_by_id[candidate_id] = record
    covered_blocks = {
        block_id
        for record in candidates_by_id.values()
        for block_id in (record.get("source_block_ids") or [])
    }
    rescued = rescue_re_evidence(
        [record["block_id"] for record in rescues], pages, page_map, config
    )
    added_rescue_records: List[Dict[str, Any]] = []
    for record in rescued:
        block_ids = set(record.get("source_block_ids") or [])
        if record["candidate_id"] in candidates_by_id or block_ids.intersection(
            covered_blocks
        ):
            continue  # already covered by live evidence: dedupe, never shadow
        candidates_by_id[record["candidate_id"]] = record
        covered_blocks.update(block_ids)
        added_rescue_records.append(record)

    # Recompute rescue-candidate repetition counts against the full index
    # (existing evidence records stay byte-untouched).
    text_pages: Dict[str, List[Optional[int]]] = {}
    for record in candidates_by_id.values():
        key = canonical_heading_text(str(record.get("exact_text") or "")).casefold()
        text_pages.setdefault(key, []).append(record.get("page_json"))
    window = int(config.get("repetition_window", 10))
    for record in added_rescue_records:
        key = canonical_heading_text(str(record.get("exact_text") or "")).casefold()
        pages_for_text = text_pages[key]
        nearby = 0
        for other_page in pages_for_text:
            if (
                other_page is not None
                and isinstance(record.get("page_json"), int)
                and abs(int(other_page) - int(record["page_json"])) <= window
            ):
                nearby += 1
        record["repetition"] = {"count": len(pages_for_text), "nearby_count": nearby}

    for record in decisions:
        intake_errors = validate_decision_v2(
            record,
            candidates_by_id,
            page_count,
            set(KINDS),
            evidence_sha,
        )
        if intake_errors:
            messages = "; ".join(
                f"{item['code']}: {item['message']}" for item in intake_errors
            )
            raise DecisionValidationError(
                f"invalid v2 decision for {record.get('candidate_id')!r}: {messages}"
            )

    applied = _apply_decisions_to_base(
        nodes,
        candidates_by_id,
        decisions,
        page_map,
        book_id,
        book_node_id,
    )

    _sort_nodes(nodes)
    validation = validate_structure_v2(nodes, toc_entries, config, meta)

    v2_dir = structure_dir / "v2"
    structure = {
        "schema_version": 3,
        "book_id": book_id,
        "book_title": str(config.get("book_title") or book_id),
        "page_map": page_map.to_dict(),
        "toc_entries": [entry.to_dict() for entry in toc_entries],
        "nodes": [node.to_dict() for node in nodes],
    }
    _atomic_write_json(v2_dir / "book_structure.json", structure, indent=2)
    _atomic_write_json(v2_dir / "validation.json", validation, indent=2)
    _atomic_write_json(v2_dir / "hierarchy_audit.json", _hierarchy_audit(validation), indent=2)
    _atomic_write_json(
        v2_dir / "decision_provenance.json",
        _decision_provenance(nodes, applied["touched"], meta),
        indent=2,
    )
    return {
        "output_dir": str(v2_dir),
        "book_id": book_id,
        "pages": page_count,
        "toc_entries": len(toc_entries),
        "candidates": len(candidates_by_id),
        "rescue_records": len(rescues),
        "rescue_candidates_added": len(added_rescue_records),
        "decisions": len(decisions),
        "decisions_applied": len(applied["touched"]),
        "nodes": len([node for node in nodes if not node.ignored]),
        "validation_ok": validation["ok"],
    }
