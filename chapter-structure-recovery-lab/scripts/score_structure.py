#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re


def grouped(nodes):
    result = defaultdict(list)
    for node in nodes:
        page = node.get("page_json")
        if page is not None and node.get("kind") != "book" and not node.get("ignored"):
            result[int(page)].append(node)
    for values in result.values():
        values.sort(key=lambda node: (node.get("bbox") or [0, -1])[1])
    return result


def _gold_parents(nodes):
    parents = {}
    stack = []
    ordered = sorted(
        [node for node in nodes if node.get("page_json") is not None],
        key=lambda node: (node.get("line", 0), node.get("page_json", -1), node.get("order_on_page", 0)),
    )
    for node in ordered:
        level = int(node.get("level", 1))
        while stack and int(stack[-1].get("level", 1)) >= level:
            stack.pop()
        parents[id(node)] = id(stack[-1]) if stack else None
        stack.append(node)
    return parents


CHINESE_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
GOLD_MACRO = re.compile(r"^第\s*([0-9一二三四五六七八九十百〇零]+)\s*(部分|章|部|卷|篇)")


def _number(value):
    text = str(value or "").strip().casefold()
    if text.isdigit():
        return int(text)
    if text and all(character in ROMAN_VALUES for character in text):
        total = 0
        previous = 0
        for character in reversed(text):
            current = ROMAN_VALUES[character]
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


def _gold_macro_identity(node):
    match = GOLD_MACRO.match(str(node.get("title") or "").strip())
    if not match:
        return None
    unit = match.group(2)
    kind = "chapter" if unit == "章" else "part"
    ordinal = _number(match.group(1))
    return (kind, ordinal) if ordinal is not None else None


def _predicted_macro_identities(nodes):
    ordered = sorted(
        [
            node for node in nodes
            if node.get("kind") in {"part", "chapter"}
            and node.get("page_json") is not None
            and not node.get("ignored")
        ],
        key=lambda node: (
            int(node["page_json"]),
            (node.get("bbox") or [0, -1])[1],
        ),
    )
    positions = defaultdict(int)
    identities = {}
    for node in ordered:
        kind = node["kind"]
        positions[kind] += 1
        sequence_number = positions[kind]
        raw_ordinal = str(node.get("ordinal") or "")
        explicit = _number(raw_ordinal)
        if kind == "part":
            ordinal = explicit or sequence_number
        elif raw_ordinal.isdigit() or explicit == sequence_number:
            ordinal = explicit or sequence_number
        else:
            ordinal = sequence_number
        identities[(kind, ordinal)] = node
    return identities


def _explicit_identity_pairs(predicted_nodes, gold_nodes, identity_data):
    if not identity_data:
        return [], set(), set(), 0
    entries = identity_data.get("positive_headings") or identity_data.get("nodes") or []
    predicted_by_block = {
        node.get("block_id"): node
        for node in predicted_nodes
        if node.get("block_id") and not node.get("ignored")
    }
    gold_by_line = {node.get("line"): node for node in gold_nodes if node.get("line") is not None}
    pairs = []
    used_predicted = set()
    used_gold = set()
    expected = 0
    seen_blocks = set()
    seen_lines = set()
    for entry in entries:
        block_id = entry.get("block_id")
        gold_line = entry.get("manual_gold_line")
        if not block_id or gold_line is None:
            raise ValueError("Explicit identity gold requires block_id and manual_gold_line")
        if block_id in seen_blocks or gold_line in seen_lines:
            raise ValueError("Explicit identity gold contains a duplicate block or manual line")
        seen_blocks.add(block_id)
        seen_lines.add(gold_line)
        gold = gold_by_line.get(gold_line)
        if gold is None:
            raise ValueError(f"Explicit identity gold line not found: {gold_line}")
        # Macro identities are already evaluated by kind+ordinal and must not
        # be counted a second time as local title identities.
        if _gold_macro_identity(gold) is not None:
            continue
        expected += 1
        predicted = predicted_by_block.get(block_id)
        if predicted is None or predicted.get("kind") in {"part", "chapter"}:
            continue
        pairs.append((predicted, gold, "explicit_block_gold_line"))
        used_predicted.add(id(predicted))
        used_gold.add(id(gold))
    return pairs, used_predicted, used_gold, expected


def _hybrid_pairs(predicted_nodes, gold_nodes, identity_data=None):
    predicted_macros = _predicted_macro_identities(predicted_nodes)
    gold_macros = {
        identity: node
        for node in gold_nodes
        if (identity := _gold_macro_identity(node)) is not None
    }
    macro_pairs = [
        (predicted_macros[identity], gold, "macro_identity")
        for identity, gold in gold_macros.items()
        if identity in predicted_macros
    ]

    identity_pairs, used_predicted, used_gold, identity_expected = _explicit_identity_pairs(
        predicted_nodes, gold_nodes, identity_data
    )
    predicted_local = grouped([
        node for node in predicted_nodes
        if node.get("kind") not in {"part", "chapter"}
        and id(node) not in used_predicted
    ])
    gold_local = grouped([
        node for node in gold_nodes
        if _gold_macro_identity(node) is None and id(node) not in used_gold
    ])
    local_pairs = []
    for page in set(predicted_local) | set(gold_local):
        local_pairs.extend(
            (pred, gold, "local_page_order")
            for pred, gold in zip(
                predicted_local.get(page, []), gold_local.get(page, [])
            )
        )
    return (
        macro_pairs + identity_pairs + local_pairs,
        len(predicted_macros),
        len(gold_macros),
        len(identity_pairs),
        identity_expected,
    )


def _parent_metrics(pairs, predicted_nodes, gold_nodes):
    book_ids = {node.get("id") for node in predicted_nodes if node.get("kind") == "book"}
    gold_parents = _gold_parents(gold_nodes)
    matched_gold_to_pred = {id(gold): pred.get("id") for pred, gold, *_ in pairs}
    comparable = 0
    matches = 0
    for pred, gold, *_ in pairs:
        gold_parent = gold_parents.get(id(gold))
        predicted_parent = pred.get("parent_id")
        if gold_parent is None:
            comparable += 1
            matches += predicted_parent in book_ids or predicted_parent is None
            continue
        expected_parent = matched_gold_to_pred.get(gold_parent)
        if expected_parent is None:
            continue
        comparable += 1
        matches += predicted_parent == expected_parent
    return comparable, matches


def score_structure(predicted_data, gold_data, identity_data=None):
    predicted_nodes = predicted_data["nodes"]
    gold_nodes = gold_data["nodes"]
    predicted = grouped(predicted_nodes)
    expected = grouped(gold_nodes)

    pred_total = sum(len(values) for values in predicted.values())
    gold_total = sum(len(values) for values in expected.values())
    pairs = []
    for page in set(predicted) | set(expected):
        pairs.extend(zip(predicted.get(page, []), expected.get(page, [])))

    level_matches = sum(pred.get("level") == gold.get("level") for pred, gold in pairs)
    precision = len(pairs) / pred_total if pred_total else 0.0
    recall = len(pairs) / gold_total if gold_total else 0.0

    legacy_pairs = [(pred, gold, "legacy_page_order") for pred, gold in pairs]
    parent_comparable, parent_matches = _parent_metrics(
        legacy_pairs, predicted_nodes, gold_nodes
    )

    (
        hybrid_pairs,
        predicted_macro_total,
        gold_macro_total,
        explicit_identity_matched,
        explicit_identity_gold,
    ) = _hybrid_pairs(
        predicted_nodes, gold_nodes, identity_data
    )
    hybrid_level_matches = sum(
        pred.get("kind") == (_gold_macro_identity(gold) or (None, None))[0]
        if mode == "macro_identity"
        else pred.get("level") == gold.get("level")
        for pred, gold, mode in hybrid_pairs
    )
    hybrid_raw_level_matches = sum(
        pred.get("level") == gold.get("level")
        for pred, gold, _ in hybrid_pairs
    )
    hybrid_parent_comparable, hybrid_parent_matches = _parent_metrics(
        hybrid_pairs, predicted_nodes, gold_nodes
    )
    macro_matched = sum(mode == "macro_identity" for _, _, mode in hybrid_pairs)
    local_matched = len(hybrid_pairs) - macro_matched - explicit_identity_matched

    return {
        "predicted": pred_total,
        "gold": gold_total,
        "matched_by_page_and_order": len(pairs),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "level_matches": level_matches,
        "level_accuracy_on_matched": round(level_matches / len(pairs), 4) if pairs else 0.0,
        "parent_comparable": parent_comparable,
        "parent_matches": parent_matches,
        "parent_accuracy_on_comparable": round(parent_matches / parent_comparable, 4)
        if parent_comparable
        else 0.0,
        "hybrid": {
            "strategy": (
                "macro_identity_plus_explicit_block_gold_line_plus_local_page_order_v2"
                if identity_data else "macro_kind_ordinal_plus_local_page_order_v1"
            ),
            "predicted": pred_total,
            "gold": gold_total,
            "matched": len(hybrid_pairs),
            "macro_identity_matched": macro_matched,
            "macro_predicted": predicted_macro_total,
            "macro_gold": gold_macro_total,
            "explicit_identity_matched": explicit_identity_matched,
            "explicit_identity_gold": explicit_identity_gold,
            "local_page_order_matched": local_matched,
            "precision": round(len(hybrid_pairs) / pred_total, 4) if pred_total else 0.0,
            "recall": round(len(hybrid_pairs) / gold_total, 4) if gold_total else 0.0,
            "semantic_level_matches": hybrid_level_matches,
            "semantic_level_accuracy": round(hybrid_level_matches / len(hybrid_pairs), 4)
            if hybrid_pairs else 0.0,
            "raw_markdown_level_matches": hybrid_raw_level_matches,
            "raw_markdown_level_accuracy": round(hybrid_raw_level_matches / len(hybrid_pairs), 4)
            if hybrid_pairs else 0.0,
            "parent_comparable": hybrid_parent_comparable,
            "parent_matches": hybrid_parent_matches,
            "parent_accuracy": round(hybrid_parent_matches / hybrid_parent_comparable, 4)
            if hybrid_parent_comparable else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score recovered headings against page-aware gold")
    parser.add_argument("structure", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("--identity-gold", type=Path)
    args = parser.parse_args()
    predicted = json.loads(args.structure.read_text(encoding="utf-8"))
    expected = json.loads(args.gold.read_text(encoding="utf-8"))
    identity = (
        json.loads(args.identity_gold.read_text(encoding="utf-8"))
        if args.identity_gold else None
    )
    result = score_structure(predicted, expected, identity)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
