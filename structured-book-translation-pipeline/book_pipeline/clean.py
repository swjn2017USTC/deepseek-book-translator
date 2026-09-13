from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import resolve_path
from .io_utils import write_json, write_jsonl
from .models import Segment, digest
from .provenance import write_clean_manifest


BLOCK_PATH = ("prunedResult", "parsing_res_list")
BODY_LABELS = {"text", "abstract", "reference_content", "list", "formula", "table", "figure_title", "table_title"}
TITLE_LABELS = {"doc_title", "paragraph_title"}
FOOTNOTE_LABELS = {"footnote", "vision_footnote"}
IMAGE_LABELS = {"image", "chart"}
DROP_LABELS = {"header", "footer", "number", "content", "header_image", "footer_image", "aside_text", "algorithm"}
SENTENCE_END = re.compile(r"[.!?。！？»”’\"']\s*$")
FOOTNOTE_START = re.compile(r"^\s*(?:\$?\s*\^\{?(\d{1,4})\}?\s*\$?|[\[(]?([0-9]{1,4})[\]).．、\s])\s*(.*)$", re.S)
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def raw_blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    value: Any = page
    for key in BLOCK_PATH:
        if not isinstance(value, dict):
            return []
        value = value.get(key, {})
    return value if isinstance(value, list) else []


def block_text(block: Dict[str, Any]) -> str:
    raw = str(block.get("block_content") or "").replace("\u00a0", " ").strip()
    if str(block.get("block_label") or "") == "table":
        return "\n".join(" ".join(line.split()) for line in raw.splitlines() if line.strip())
    return " ".join(raw.split()).strip()


def block_id(page: int, position: int, block: Dict[str, Any]) -> str:
    # Must match chapter_recovery.models.Block so structure provenance joins exactly.
    raw = "\x1f".join(str(part) for part in (page, block.get("block_id"), position, str(block.get("block_label") or "unknown"), str(block.get("block_content") or "")))
    return "blk-" + sha1(raw.encode("utf-8")).hexdigest()[:16]


def bbox(block: Dict[str, Any]) -> List[float]:
    value = block.get("block_bbox") or [0, 0, 0, 0]
    return [float(item) for item in value] if isinstance(value, list) and len(value) == 4 else [0.0, 0.0, 0.0, 0.0]


def require_structure_gate(config: Dict[str, Any]) -> Dict[str, Any]:
    structure_path = resolve_path(config, "structure_json")
    quality = config.get("quality", {})
    validation_path = Path(quality.get("validation_json") or structure_path.with_name("validation.json")).expanduser().resolve()
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing structure validation: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("ok"):
        raise RuntimeError(f"Chapter structure quality gate failed: {validation_path}")
    review_path = Path(quality.get("review_packets") or structure_path.with_name("review_packets.jsonl")).expanduser().resolve()
    pending = 0
    if review_path.exists():
        pending = sum(1 for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if pending and not quality.get("allow_pending_review", False):
        raise RuntimeError(f"{pending} unresolved review packets; resolve them before translation")
    return {"validation_path": str(validation_path), "review_path": str(review_path), "pending_reviews": pending}


def _footnote_parts(text: str) -> Tuple[Optional[int], str]:
    match = FOOTNOTE_START.match(text)
    if not match:
        return None, text
    number = match.group(1) or match.group(2)
    return int(number), match.group(3).strip()


def _replace_marker(text: str, number: int, reference: str) -> Tuple[str, bool]:
    if 1 <= number <= len(CIRCLED) and CIRCLED[number - 1] in text:
        return text.replace(CIRCLED[number - 1], reference, 1), True
    for match in re.finditer(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]{1,4}", text):
        if int(match.group().translate(SUPERSCRIPTS)) == number:
            return text[:match.start()] + reference + text[match.end():], True
    for pattern in (rf"\$\s*\^\{{{number}\}}\s*\$", rf"\^\{{{number}\}}"):
        replaced, count = re.subn(pattern, lambda _: reference, text, count=1)
        if count:
            return replaced, True
    return text, False


def _segment(book_id: str, kind: str, text: str, page: Optional[int], zone: str, parent: Optional[str], block_ids: List[str], level: Optional[int] = None, translatable: bool = True, metadata: Optional[Dict[str, Any]] = None, stable_hint: str = "") -> Segment:
    source_hash = digest(text)
    segment_id = f"seg-{digest(book_id, stable_hint or '|'.join(block_ids), kind)[:20]}"
    return Segment(segment_id, source_hash, kind, text, page, zone, parent, level, translatable, block_ids, metadata or {})


def clean_book(config: Dict[str, Any]) -> Dict[str, Any]:
    gate = require_structure_gate(config)
    input_path = resolve_path(config, "input_json")
    structure_path = resolve_path(config, "structure_json")
    output_dir = resolve_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = json.loads(input_path.read_text(encoding="utf-8"))
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    all_nodes = list(structure.get("nodes", []))
    nodes = [node for node in all_nodes if not node.get("ignored") and node.get("kind") != "book" and node.get("page_json") is not None]
    nodes.sort(key=lambda node: (int(node["page_json"]), (node.get("bbox") or [0, -1])[1], int(node.get("level", 2))))
    main_start = min((int(node["page_json"]) for node in nodes if node.get("zone") == "mainmatter"), default=0)
    headings_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    accepted_title_blocks = {
        identifier
        for node in all_nodes
        for identifier in (node.get("source_block_ids") or ([node["block_id"]] if node.get("block_id") else []))
    }
    for node in nodes:
        headings_by_page[int(node["page_json"])].append(node)

    raw_footnotes: Dict[int, List[Tuple[int, str, List[str]]]] = defaultdict(list)
    normalized_pages: Dict[int, List[Dict[str, Any]]] = {}
    unresolved_title_blocks = []
    for page_index, page in enumerate(pages):
        values = []
        for position, block in enumerate(raw_blocks(page)):
            label = str(block.get("block_label") or "unknown")
            text = block_text(block)
            identifier = block_id(page_index, position, block)
            item = {"id": identifier, "label": label, "text": text, "bbox": bbox(block), "position": position}
            values.append(item)
            if label in FOOTNOTE_LABELS and text:
                number, body = _footnote_parts(text)
                if number is not None:
                    raw_footnotes[page_index].append((number, body, [identifier]))
                elif raw_footnotes[page_index]:
                    old_number, old_body, old_ids = raw_footnotes[page_index][-1]
                    raw_footnotes[page_index][-1] = (old_number, f"{old_body} {text}".strip(), old_ids + [identifier])
            if label in TITLE_LABELS and identifier not in accepted_title_blocks and text and page_index >= main_start and text.strip("# ").casefold() not in {"contents", "table of contents", "目录"}:
                unresolved_title_blocks.append({"page_json": page_index, "block_id": identifier, "text": text})
        normalized_pages[page_index] = sorted(values, key=lambda item: (item["bbox"][1], item["bbox"][0], item["position"]))

    segments: List[Segment] = []
    current_parent: Optional[str] = None
    current_zone = "frontmatter"
    footnotes_matched = 0
    matched_footnotes = set()
    footnotes_total = sum(len(items) for items in raw_footnotes.values())
    translate_zones = set(config.get("translate_zones", ["frontmatter", "mainmatter", "backmatter", "notes", "bibliography"]))

    for page_index in range(len(pages)):
        events: List[Tuple[float, int, str, Any]] = []
        for order, node in enumerate(headings_by_page.get(page_index, [])):
            y = float((node.get("bbox") or [0, -1])[1])
            events.append((y, order, "heading", node))
        for item in normalized_pages[page_index]:
            events.append((float(item["bbox"][1]), item["position"] + 10000, "block", item))
        events.sort(key=lambda event: (event[0], event[1]))
        for _, _, event_type, value in events:
            if event_type == "heading":
                node = value
                current_parent = node["id"]
                current_zone = _specific_zone(node, current_zone)
                ids = list(node.get("source_block_ids") or ([node["block_id"]] if node.get("block_id") else []))
                segments.append(_segment(config["book_id"], "heading", str(node["title_source"]), page_index, current_zone, node.get("parent_id"), ids, int(node.get("level", 2)), current_zone in translate_zones, {"node_id": node["id"], "node_kind": node.get("kind")}, node["id"]))
                continue
            item = value
            label, text, identifier = item["label"], item["text"], item["id"]
            if identifier in accepted_title_blocks or label in TITLE_LABELS or label in DROP_LABELS or label in FOOTNOTE_LABELS:
                continue
            if label in IMAGE_LABELS:
                segments.append(_segment(config["book_id"], "image", "", page_index, current_zone, current_parent, [identifier], translatable=False, metadata={"bbox": item["bbox"], "label": label}))
                continue
            if not text:
                continue
            if label not in BODY_LABELS:
                continue
            for number, _, _ in raw_footnotes.get(page_index, []):
                if (page_index, number) in matched_footnotes:
                    continue
                reference = f"[^fn-p{page_index:04d}-{number}]"
                text, matched = _replace_marker(text, number, reference)
                footnotes_matched += int(matched)
                if matched:
                    matched_footnotes.add((page_index, number))
            kind = "table" if label == "table" else ("caption" if label in {"figure_title", "table_title"} else "paragraph")
            segments.append(_segment(config["book_id"], kind, text, page_index, current_zone, current_parent, [identifier], translatable=current_zone in translate_zones, metadata={"label": label, "bbox": item["bbox"], "last_page": page_index}))
        for number, body, identifiers in raw_footnotes.get(page_index, []):
            reference = f"fn-p{page_index:04d}-{number}"
            segments.append(_segment(config["book_id"], "footnote", body, page_index, current_zone, current_parent, identifiers, translatable=current_zone in translate_zones, metadata={"reference": reference, "number": number}, stable_hint=reference))

    segments = _stitch_cross_page(segments, config["book_id"])
    report = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "input_json": str(input_path),
        "structure_json": str(structure_path),
        "structure_gate": gate,
        "metrics": {
            "pages": len(pages),
            "segments": len(segments),
            "kinds": dict(Counter(segment.kind for segment in segments)),
            "zones": dict(Counter(segment.zone for segment in segments)),
            "unresolved_title_blocks": len(unresolved_title_blocks),
            "footnotes_total": footnotes_total,
            "footnote_markers_matched": footnotes_matched,
        },
        "warnings": [{"code": "unresolved_title_block", **item} for item in unresolved_title_blocks[:200]],
    }
    segments_path = output_dir / "cleaned_segments.jsonl"
    report_path = output_dir / "cleaning_report.json"
    write_jsonl(segments_path, [segment.to_dict() for segment in segments])
    write_json(report_path, report)
    write_clean_manifest(config, segments_path, report_path)
    return report


def _specific_zone(node: Dict[str, Any], fallback: str) -> str:
    zone = str(node.get("zone") or fallback)
    if node.get("kind") != "backmatter" and zone != "backmatter":
        return zone
    title = re.sub(r"[^\w]+", " ", str(node.get("title_source") or "").casefold()).strip()
    if title in {"notes", "endnotes", "注释"}:
        return "notes"
    if title in {"references", "bibliography", "works cited", "参考文献"}:
        return "bibliography"
    if title in {"index", "name index", "subject index", "索引"}:
        return "index"
    return zone


def _stitch_cross_page(segments: List[Segment], book_id: str) -> List[Segment]:
    output: List[Segment] = []
    for segment in segments:
        if output and segment.kind == "paragraph" and output[-1].kind == "paragraph" and segment.parent_node_id == output[-1].parent_node_id and segment.page_json != output[-1].metadata.get("last_page") and not SENTENCE_END.search(output[-1].source_text):
            previous = output[-1]
            separator = ""
            if previous.source_text.endswith("-"):
                previous.source_text = previous.source_text[:-1]
            else:
                separator = " "
            previous.source_text += separator + segment.source_text
            previous.source_block_ids.extend(segment.source_block_ids)
            previous.source_hash = digest(previous.source_text)
            previous.id = f"seg-{digest(book_id, '|'.join(previous.source_block_ids), 'paragraph')[:20]}"
            previous.metadata["last_page"] = segment.page_json
            previous.metadata["cross_page_stitch"] = True
        else:
            output.append(segment)
    return output
