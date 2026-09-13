from __future__ import annotations

from hashlib import sha256
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import resolve_path
from .io_utils import write_json, write_jsonl


PAGE_MARKER = re.compile(r"^\s*<!--\s*PAGE(?:_JSON)?\s*:\s*(\d+)\s*-->\s*$", re.I)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PLAIN_STRUCTURAL_HEADING = re.compile(
    r"^(?:第\s*[0-9一二三四五六七八九十百〇零]+\s*(?:部分|部|卷|篇)(?:\s*$|[：:]\s*|\s+)|(?:part|book|volume|teil)\s+(?:[0-9ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b)",
    re.I,
)
MACRO_PREFIX = re.compile(
    r"^(?:part|chapter|book|volume|teil|kapitel)\s+([0-9ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b|^第\s*([0-9一二三四五六七八九十百〇零]+)\s*(部分|章|部|卷|篇)",
    re.I,
)
LEADING_NUMBER = re.compile(
    r"^([0-9]+(?:\.[0-9]+)*|[ivxlcdm]+|[A-Za-z]|[一二三四五六七八九十百〇零]+)(?:[.)、])?(?:\s|$)",
    re.I,
)
CHINESE_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
KEYWORD_GROUPS = (
    {"index", "索引", "register"},
    {"preface", "vorwort", "前言", "序言"},
    {"introduction", "einleitung", "导论", "引言"},
    {"notes", "anmerkungen", "注释"},
    {"bibliography", "references", "literatur", "参考文献", "书目"},
    {"further reading", "further readings", "延伸阅读", "推荐阅读"},
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _number(value: str) -> Optional[int]:
    text = value.strip().casefold()
    if text.isdigit():
        return int(text)
    if text in WORD_NUMBERS:
        return WORD_NUMBERS[text]
    if text and all(character in ROMAN for character in text):
        total = 0
        previous = 0
        for character in reversed(text):
            current = ROMAN[character]
            total += -current if current < previous else current
            previous = max(previous, current)
        return total
    if text and all(character in CHINESE_DIGITS or character in "十百" for character in text):
        if text == "十":
            return 10
        if "百" in text:
            left, right = text.split("百", 1)
            return CHINESE_DIGITS.get(left, 1) * 100 + (_number(right.lstrip("零〇")) or 0)
        if "十" in text:
            left, right = text.split("十", 1)
            return CHINESE_DIGITS.get(left, 1) * 10 + CHINESE_DIGITS.get(right, 0)
        if len(text) > 1:
            return int("".join(str(CHINESE_DIGITS[character]) for character in text))
        return CHINESE_DIGITS.get(text)
    return None


def _identity(title: str) -> Tuple[Optional[str], Optional[str]]:
    clean = " ".join(title.strip().split())
    macro = MACRO_PREFIX.search(clean)
    if macro:
        raw = macro.group(1) or macro.group(2)
        unit = (macro.group(3) or clean.split(None, 1)[0]).casefold()
        kind = "chapter" if unit in {"章", "chapter", "kapitel"} else "part"
        number = _number(raw)
        return kind, str(number) if number is not None else raw.casefold()
    leading = LEADING_NUMBER.match(clean)
    if not leading:
        return None, None
    token = leading.group(1)
    if "." in token and token[0].isdigit():
        return "numbered", token
    number = _number(token)
    return "numbered", str(number) if number is not None else token.casefold()


def parse_existing_markdown(path: Path) -> Dict[str, Any]:
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8")
    lines = text.splitlines(keepends=True)
    headings: List[Dict[str, Any]] = []
    page: Optional[int] = None
    non_heading_lines: List[str] = []
    for line_number, raw in enumerate(lines, 1):
        value = raw.rstrip("\r\n")
        marker = PAGE_MARKER.match(value)
        if marker:
            page = int(marker.group(1))
            non_heading_lines.append(raw)
            continue
        heading = HEADING.match(value)
        if heading:
            title = heading.group(2).strip()
            kind, ordinal = _identity(title)
            headings.append({
                "heading_id": f"existing-line-{line_number}",
                "line": line_number,
                "page_json": page,
                "level": len(heading.group(1)),
                "title": title,
                "identity_kind": kind,
                "identity_ordinal": ordinal,
                "was_markdown_heading": True,
            })
        elif PLAIN_STRUCTURAL_HEADING.match(value.strip()):
            title = value.strip()
            kind, ordinal = _identity(title)
            headings.append({
                "heading_id": f"existing-line-{line_number}",
                "line": line_number,
                "page_json": page,
                "level": 0,
                "title": title,
                "identity_kind": kind,
                "identity_ordinal": ordinal,
                "was_markdown_heading": False,
            })
        else:
            non_heading_lines.append(raw)
    return {
        "text": text,
        "headings": headings,
        "input_sha256": _digest_bytes(raw_bytes),
        "non_heading_sha256": _digest("".join(non_heading_lines)),
    }


def _structure_nodes(structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = [
        dict(node)
        for node in structure.get("nodes", [])
        if node.get("kind") != "book"
        and not node.get("ignored")
        and isinstance(node.get("page_json"), int)
    ]
    nodes.sort(key=lambda node: (
        int(node["page_json"]),
        float((node.get("bbox") or [0, -1])[1]),
        int(node.get("level", 2)),
    ))
    for node in nodes:
        kind, ordinal = _identity(str(node.get("title_source") or ""))
        if node.get("kind") in {"part", "chapter"}:
            kind = str(node["kind"])
        raw_ordinal = node.get("ordinal")
        if raw_ordinal not in {None, ""}:
            parsed = _number(str(raw_ordinal))
            ordinal = str(parsed) if parsed is not None else str(raw_ordinal).casefold()
        node["identity_kind"] = kind
        node["identity_ordinal"] = ordinal
    return nodes


def _keyword_match(left: str, right: str) -> bool:
    a = left.casefold()
    b = right.casefold()
    return any(any(term in a for term in group) and any(term in b for term in group) for group in KEYWORD_GROUPS)


def _preserve_frontmatter_metadata(heading: Dict[str, Any], first_structure_page: Optional[int]) -> bool:
    if heading.get("page_json") is None:
        return True
    title = str(heading.get("title") or "").strip().casefold()
    metadata_titles = {"contents", "目录", "preface", "前言", "序言"}
    return bool(
        title in metadata_titles
        and first_structure_page is not None
        and int(heading["page_json"]) < first_structure_page
    )


def _pair_score(node: Dict[str, Any], heading: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0.0
    evidence: List[str] = []
    source_page = int(node["page_json"])
    target_page = heading.get("page_json")
    if isinstance(target_page, int):
        distance = abs(source_page - target_page)
        if distance == 0:
            score += 6.0
            evidence.append("same_json_page")
        elif distance == 1:
            score += 3.0
            evidence.append("adjacent_json_page")
        elif distance <= 3:
            score += 1.0
            evidence.append("near_json_page")
        elif distance > 8:
            score -= 6.0
    else:
        score -= 1.0

    node_kind = node.get("identity_kind")
    heading_kind = heading.get("identity_kind")
    node_ordinal = node.get("identity_ordinal")
    heading_ordinal = heading.get("identity_ordinal")
    if node_ordinal is not None and heading_ordinal is not None:
        if node_ordinal == heading_ordinal:
            score += 3.0
            evidence.append("same_printed_ordinal")
        else:
            score -= 4.0
    if node_kind in {"part", "chapter"} and heading_kind in {"part", "chapter"}:
        if node_kind == heading_kind:
            score += 3.0
            evidence.append("same_macro_kind")
        else:
            score -= 5.0
    if (
        node_kind == "chapter"
        and heading_kind is None
        and int(heading.get("level", 0)) == 2
        and isinstance(target_page, int)
        and abs(source_page - target_page) <= 3
    ):
        score += 1.5
        evidence.append("nearby_h2_macro_shape")
    if _keyword_match(str(node.get("title_source") or ""), str(heading.get("title") or "")):
        score += 2.0
        evidence.append("cross_language_structural_keyword")
    if int(node.get("level", 2)) == int(heading.get("level", 1)):
        score += 0.5
        evidence.append("existing_level_agrees")
    return score, evidence


def align_existing_headings(
    nodes: Sequence[Dict[str, Any]], headings: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Globally align two ordered heading sequences with one-to-one DP."""
    n, m = len(nodes), len(headings)
    skip_node = -1.5
    skip_heading = -1.0
    scores = [[float("-inf")] * (m + 1) for _ in range(n + 1)]
    moves: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    scores[0][0] = 0.0
    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + skip_node
        moves[i][0] = "skip_node"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + skip_heading
        moves[0][j] = "skip_heading"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pair_score, _ = _pair_score(nodes[i - 1], headings[j - 1])
            options = [
                (scores[i - 1][j] + skip_node, "skip_node"),
                (scores[i][j - 1] + skip_heading, "skip_heading"),
            ]
            if pair_score >= 2.5:
                options.append((scores[i - 1][j - 1] + pair_score, "match"))
            scores[i][j], moves[i][j] = max(options, key=lambda item: item[0])

    pairs: List[Tuple[int, int]] = []
    unmatched_nodes: List[int] = []
    unmatched_headings: List[int] = []
    i, j = n, m
    while i or j:
        move = moves[i][j]
        if move == "match":
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "skip_node":
            unmatched_nodes.append(i - 1)
            i -= 1
        elif move == "skip_heading":
            unmatched_headings.append(j - 1)
            j -= 1
        else:
            raise RuntimeError("Migration alignment backtrack failed")
    pairs.reverse()
    unmatched_nodes.reverse()
    unmatched_headings.reverse()
    return {
        "pairs": pairs,
        "unmatched_nodes": unmatched_nodes,
        "unmatched_headings": unmatched_headings,
        "score": round(scores[n][m], 4),
    }


def _upstream_state(config: Dict[str, Any], structure_path: Path) -> Dict[str, Any]:
    quality = config.get("quality", {})
    validation_path = Path(
        quality.get("validation_json") or structure_path.with_name("validation.json")
    ).expanduser().resolve()
    review_path = Path(
        quality.get("review_packets") or structure_path.with_name("review_packets.jsonl")
    ).expanduser().resolve()
    validation_ok = False
    if validation_path.exists():
        validation_ok = bool(json.loads(validation_path.read_text(encoding="utf-8")).get("ok"))
    pending = 0
    if review_path.exists():
        pending = sum(bool(line.strip()) for line in review_path.read_text(encoding="utf-8").splitlines())
    return {
        "validation_json": str(validation_path),
        "validation_ok": validation_ok,
        "review_packets": str(review_path),
        "pending_reviews": pending,
    }


def audit_existing_translation(config: Dict[str, Any]) -> Dict[str, Any]:
    existing_value = config.get("existing_translation")
    if not existing_value:
        raise ValueError("Migration audit requires existing_translation")
    existing_path = Path(str(existing_value)).expanduser()
    if not existing_path.is_absolute():
        existing_path = (Path(config["_base_dir"]) / existing_path).resolve()
    if not existing_path.is_file():
        raise FileNotFoundError(existing_path)
    structure_path = resolve_path(config, "structure_json")
    output_dir = resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = parse_existing_markdown(existing_path)
    structure_bytes = structure_path.read_bytes()
    structure = json.loads(structure_bytes.decode("utf-8"))
    nodes = _structure_nodes(structure)
    headings = parsed["headings"]
    aligned = align_existing_headings(nodes, headings)
    node_page_counts = Counter(node["page_json"] for node in nodes)
    heading_page_counts = Counter(
        heading["page_json"] for heading in headings if isinstance(heading.get("page_json"), int)
    )
    exact_pair_counts = Counter(
        nodes[node_index]["page_json"]
        for node_index, heading_index in aligned["pairs"]
        if nodes[node_index]["page_json"] == headings[heading_index].get("page_json")
    )
    macro_node_page_counts = Counter(
        node["page_json"]
        for node in nodes
        if node.get("identity_kind") in {"part", "chapter"}
    )
    macro_heading_page_counts = Counter(
        heading["page_json"]
        for heading in headings
        if isinstance(heading.get("page_json"), int)
        and (
            heading.get("identity_kind") in {"part", "chapter"}
            or int(heading.get("level", 0)) == 2
        )
    )
    exact_macro_pair_counts = Counter(
        nodes[node_index]["page_json"]
        for node_index, heading_index in aligned["pairs"]
        if nodes[node_index]["page_json"] == headings[heading_index].get("page_json")
        and nodes[node_index].get("identity_kind") in {"part", "chapter"}
        and (
            headings[heading_index].get("identity_kind") in {"part", "chapter"}
            or int(headings[heading_index].get("level", 0)) == 2
        )
    )
    records: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    preserved_unmatched_headings: List[Dict[str, Any]] = []
    for node_index, heading_index in aligned["pairs"]:
        node = nodes[node_index]
        heading = headings[heading_index]
        score, evidence = _pair_score(node, heading)
        page_exact = node["page_json"] == heading.get("page_json")
        identity_exact = bool(
            node.get("identity_ordinal") is not None
            and node.get("identity_ordinal") == heading.get("identity_ordinal")
        )
        kind_exact = bool(
            node.get("identity_kind") in {"part", "chapter"}
            and node.get("identity_kind") == heading.get("identity_kind")
        )
        page_sequence_exact = bool(
            page_exact
            and node_page_counts[node["page_json"]] == heading_page_counts[node["page_json"]]
            and exact_pair_counts[node["page_json"]] == node_page_counts[node["page_json"]]
        )
        macro_page_sequence_exact = bool(
            page_exact
            and node.get("identity_kind") in {"part", "chapter"}
            and macro_node_page_counts[node["page_json"]]
            == macro_heading_page_counts[node["page_json"]]
            == exact_macro_pair_counts[node["page_json"]]
        )
        if page_sequence_exact:
            evidence.append("complete_same_page_sequence")
        if macro_page_sequence_exact and not page_sequence_exact:
            evidence.append("complete_same_page_macro_sequence")
        confidence = "high" if page_exact and (
            (score >= 8.5 and (identity_exact or kind_exact))
            or (score >= 6.0 and page_sequence_exact)
            or (score >= 7.5 and macro_page_sequence_exact)
        ) else (
            "medium" if page_exact and score >= 6.0 else "low"
        )
        action = (
            "keep_level"
            if int(node.get("level", 2)) == int(heading["level"])
            else "change_level"
        )
        record = {
            "node_id": node["id"],
            "source_title": node.get("title_source"),
            "source_page_json": node["page_json"],
            "source_kind": node.get("kind"),
            "target_level": int(node.get("level", 2)),
            "existing_heading_id": heading["heading_id"],
            "existing_line": heading["line"],
            "existing_page_json": heading.get("page_json"),
            "existing_title": heading["title"],
            "existing_level": heading["level"],
            "action": action,
            "score": round(score, 4),
            "confidence": confidence,
            "evidence": evidence,
        }
        records.append(record)
        if confidence != "high":
            reviews.append({
                "candidate_id": f"migration-{node['id']}-{heading['line']}",
                "review_status": "needs_human",
                "reason": "migration_alignment_not_high_confidence",
                "alignment": record,
                "allowed_decisions": ["accept_mapping", "reject_mapping", "needs_human"],
                "guardrail": "Keep the existing Chinese title text; decide only whether this source node and existing heading are the same printed heading.",
            })
    for index in aligned["unmatched_nodes"]:
        node = nodes[index]
        reviews.append({
            "candidate_id": f"migration-unmatched-node-{node['id']}",
            "review_status": "needs_human",
            "reason": "source_structure_node_unmatched",
            "source_node": {
                "node_id": node["id"],
                "title_source": node.get("title_source"),
                "page_json": node["page_json"],
                "kind": node.get("kind"),
                "level": node.get("level"),
            },
            "allowed_decisions": ["link_existing_heading", "leave_unmapped", "needs_human"],
            "guardrail": "Do not translate or invent a Chinese heading.",
        })
    for index in aligned["unmatched_headings"]:
        heading = headings[index]
        first_structure_page = min((int(node["page_json"]) for node in nodes), default=None)
        if _preserve_frontmatter_metadata(heading, first_structure_page):
            preserved_unmatched_headings.append({
                **heading,
                "action": "keep_unchanged",
                "reason": "frontmatter_metadata_outside_source_structure",
            })
            continue
        reviews.append({
            "candidate_id": f"migration-unmatched-heading-{heading['line']}",
            "review_status": "needs_human",
            "reason": "existing_heading_unmatched",
            "existing_heading": heading,
            "allowed_decisions": ["link_source_node", "keep_unchanged", "demote_to_body", "needs_human"],
            "guardrail": "Preserve the existing Chinese text and all non-heading content; demotion may remove only its Markdown heading prefix.",
        })

    upstream = _upstream_state(config, structure_path)
    high = sum(record["confidence"] == "high" for record in records)
    changes = sum(record["action"] == "change_level" for record in records)
    macro_unmatched = sum(
        nodes[index].get("kind") in {"part", "chapter"}
        or nodes[index].get("identity_kind") in {"part", "chapter"}
        for index in aligned["unmatched_nodes"]
    )
    node_ids = [record["node_id"] for record in records]
    heading_ids = [record["existing_heading_id"] for record in records]
    if len(node_ids) != len(set(node_ids)) or len(heading_ids) != len(set(heading_ids)):
        raise RuntimeError("Migration alignment violated one-to-one assignment")
    post_input_sha256 = _digest_bytes(existing_path.read_bytes())
    source_unchanged = post_input_sha256 == parsed["input_sha256"]
    if not source_unchanged:
        raise RuntimeError("Existing translation changed during audit")
    report = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "audit_only": True,
        "source_library_unchanged": source_unchanged,
        "existing_translation": str(existing_path),
        "structure_json": str(structure_path),
        "structure_sha256": _digest_bytes(structure_bytes),
        "output_dir": str(output_dir),
        "input_sha256": parsed["input_sha256"],
        "post_audit_input_sha256": post_input_sha256,
        "non_heading_sha256": parsed["non_heading_sha256"],
        "upstream": upstream,
        "metrics": {
            "structure_nodes": len(nodes),
            "existing_headings": len(headings),
            "matched": len(records),
            "high_confidence": high,
            "medium_or_low": len(records) - high,
            "proposed_level_changes": changes,
            "unmatched_structure_nodes": len(aligned["unmatched_nodes"]),
            "unmatched_macro_nodes": macro_unmatched,
            "unmatched_existing_headings": len(aligned["unmatched_headings"]),
            "auto_preserved_frontmatter_headings": len(preserved_unmatched_headings),
            "migration_review_packets": len(reviews),
            "structure_match_rate": round(len(records) / len(nodes), 4) if nodes else 1.0,
            "high_confidence_rate": round(high / len(records), 4) if records else 1.0,
        },
        "ready_for_preview": bool(
            upstream["validation_ok"]
            and upstream["pending_reviews"] == 0
            and macro_unmatched == 0
            and not reviews
        ),
        "alignment_score": aligned["score"],
    }
    write_json(output_dir / "migration_report.json", report)
    write_json(output_dir / "migration_alignment.json", {
        "schema_version": 1,
        "records": records,
        "preserved_unmatched_headings": preserved_unmatched_headings,
    })
    write_jsonl(output_dir / "migration_review_packets.jsonl", reviews)
    return report
