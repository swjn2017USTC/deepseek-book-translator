import json

import pytest

from chapter_recovery.decisions import (
    DecisionValidationError,
    compile_decision_files,
    compile_decisions,
)
from chapter_recovery.models import Node
from chapter_recovery.page_map import PageMap
from chapter_recovery.recover import (
    _apply_compiled_candidate_override,
    _apply_compiled_override,
)


def packet(candidate_id):
    return {
        "candidate_id": candidate_id,
        "allowed_decisions": [
            "accept", "reject", "change_level", "change_kind",
            "merge_with_previous", "needs_human",
        ],
        "node_context": {"node_id": candidate_id},
    }


def decision(candidate_id, action, **values):
    return {
        "candidate_id": candidate_id,
        "decision": action,
        "reason_code": "page_evidence" if action != "needs_human" else "insufficient_evidence",
        "evidence_refs": ["page_context.blocks[0]"],
        **values,
    }


def text_packet(candidate_id="candidate-text"):
    return {
        "candidate_id": candidate_id,
        "block_id": "blk-text",
        "page_json": 10,
        "source_text": "Existing source heading",
        "suggested_level": 3,
        "confidence": 0.6,
        "bbox": [10, 20, 200, 40],
        "evidence": ["short_text", "surrounding_whitespace"],
        "allowed_decisions": [
            "accept", "reject", "change_level", "change_kind",
            "merge_with_previous", "needs_human",
        ],
    }


def test_compiles_complete_decisions_and_preserves_needs_human():
    packets = [packet("node-one"), packet("node-two"), packet("node-three")]
    decisions = [
        decision("node-one", "accept"),
        decision("node-two", "change_level", level=4),
        decision("node-three", "needs_human"),
    ]

    bundle = compile_decisions(packets, decisions, [{"match": {"id": "existing"}, "set": {"ignored": True}}])

    assert bundle["review_packet_count"] == 3
    assert bundle["resolved_count"] == 2
    assert bundle["unresolved_count"] == 1
    assert bundle["overrides"][0]["match"]["id"] == "existing"
    assert bundle["overrides"][1]["operation"] == "accept_node"
    assert bundle["overrides"][2]["set"] == {"level": 4}
    assert bundle["unresolved"][0]["candidate_id"] == "node-three"
    assert len(bundle["review_packets_sha256"]) == 64


def test_decisions_fail_closed_on_missing_packets_and_title_rewrites():
    packets = [packet("node-one"), packet("node-two")]
    with pytest.raises(DecisionValidationError, match="every review packet"):
        compile_decisions(packets, [decision("node-one", "accept")])

    rewritten = decision("node-one", "accept", title_source="invented")
    with pytest.raises(DecisionValidationError, match="must not supply"):
        compile_decisions([packet("node-one")], [rewritten])


def test_non_node_candidates_compile_without_copying_decision_text():
    source_packet = text_packet()
    accepted = compile_decisions(
        [source_packet], [decision("candidate-text", "accept")]
    )
    assert accepted["overrides"][0]["operation"] == "promote_candidate"
    assert accepted["overrides"][0]["source_text"] == source_packet["source_text"]
    rejected = compile_decisions(
        [source_packet], [decision("candidate-text", "reject")]
    )
    assert rejected["overrides"][0]["operation"] == "reject_candidate"
    bundle = compile_decisions(
        [source_packet], [decision("candidate-text", "needs_human")]
    )
    assert bundle["unresolved_count"] == 1

    with pytest.raises(DecisionValidationError, match="cannot be promoted"):
        compile_decisions(
            [source_packet],
            [decision("candidate-text", "change_kind", kind="toc_entry")],
        )


def test_non_node_promotion_rechecks_live_source_and_preserves_provenance():
    source_packet = text_packet()
    bundle = compile_decisions(
        [source_packet], [decision("candidate-text", "change_level", level=4)]
    )
    override = bundle["overrides"][0]
    book = Node("book", "book", 1, "Book", None, None, None, None, "config", 1.0)
    chapter = Node(
        "chapter", "chapter", 2, "Chapter", 9, 11, "book", "blk-chapter",
        "doc_title", 1.0, bbox=(0, 10, 300, 30),
    )
    nodes = [book, chapter]
    candidates = [dict(source_packet)]
    _apply_compiled_candidate_override(
        nodes, candidates, override,
        PageMap(2, 1.0, 1, 1, {}), "test-book", "book",
    )
    promoted = nodes[-1]
    assert promoted.title_source == source_packet["source_text"]
    assert promoted.level == 4
    assert promoted.parent_id == chapter.id
    assert promoted.page_print == 12
    assert promoted.source_block_ids == [source_packet["block_id"]]
    assert candidates == []

    stale = [dict(source_packet, source_text="Changed OCR")]
    with pytest.raises(ValueError, match="stale or altered"):
        _apply_compiled_candidate_override(
            [book, chapter], stale, override,
            PageMap(2, 1.0, 1, 1, {}), "test-book", "book",
        )

    rejected_override = compile_decisions(
        [source_packet], [decision("candidate-text", "reject")]
    )["overrides"][0]
    rejected_candidates = [dict(source_packet)]
    _apply_compiled_candidate_override(
        [book, chapter], rejected_candidates, rejected_override,
        PageMap(2, 1.0, 1, 1, {}), "test-book", "book",
    )
    assert rejected_candidates[0]["suppression_reason"] == "review_rejected_heading"

    merge_override = compile_decisions(
        [source_packet],
        [
            decision(
                "candidate-text", "merge_with_previous",
                merge_with_node_id="node-target",
            )
        ],
    )["overrides"][0]
    target = Node(
        "node-target", "section", 3, "Left:", 9, 11, "chapter", "blk-left",
        "paragraph_title", 0.8, bbox=(10, 20, 200, 40),
        source_block_ids=["blk-left"],
    )
    merge_candidates = [dict(source_packet)]
    _apply_compiled_candidate_override(
        [book, chapter, target], merge_candidates, merge_override,
        PageMap(2, 1.0, 1, 1, {}), "test-book", "book",
    )
    assert target.title_source == "Left: Existing source heading"
    assert target.source_block_ids == ["blk-left", "blk-text"]
    assert merge_candidates == []


def test_compiled_override_operations_preserve_provenance_and_parent_links():
    book = Node("book", "book", 1, "Book", None, None, None, None, "config", 1.0)
    target = Node(
        "node-target", "section", 3, "Left:", 10, 1, "book", "block-left",
        "paragraph_title", 0.7, bbox=(0, 0, 10, 10), source_block_ids=["block-left"],
    )
    source = Node(
        "node-source", "section", 3, "Right", 10, 1, "book", "block-right",
        "paragraph_title", 0.6, bbox=(0, 11, 10, 20), source_block_ids=["block-right"],
    )
    child = Node(
        "node-child", "section", 4, "Child", 11, 2, "node-source", "block-child",
        "paragraph_title", 0.9,
    )
    nodes = [book, target, source, child]

    _apply_compiled_override(nodes, {
        "operation": "merge_nodes", "source_id": "node-source", "target_id": "node-target",
        "review": {"reason_code": "split_title"},
    })

    assert target.title_source == "Left: Right"
    assert target.source_block_ids == ["block-left", "block-right"]
    assert target.bbox == (0, 0, 10, 20)
    assert target.confidence == 1.0
    assert source.ignored is True
    assert child.parent_id == "node-target"


def test_compile_decision_files_writes_a_reusable_bundle(tmp_path):
    review_path = tmp_path / "reviews.jsonl"
    decision_path = tmp_path / "decisions.jsonl"
    output_path = tmp_path / "compiled.json"
    review_path.write_text(json.dumps(packet("node-one")) + "\n", encoding="utf-8")
    decision_path.write_text(json.dumps(decision("node-one", "reject")) + "\n", encoding="utf-8")

    result = compile_decision_files(review_path, decision_path, output_path)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["resolved_count"] == 1
    assert saved["overrides"][0]["operation"] == "reject_node"
