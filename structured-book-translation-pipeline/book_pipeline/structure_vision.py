from __future__ import annotations

"""DeepSeek Vision evidence collector for OCR chapter recovery.

This module performs the network/image work. The sibling chapter_recovery
package consumes only the resulting versioned JSON sidecar, keeping the final
tree construction deterministic and auditable.
"""

import base64
from hashlib import sha256
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from chapter_recovery.heading_text import numbering
from chapter_recovery.page_map import infer_page_map
from chapter_recovery.paddleocr import load_pages, normalize_blocks
from chapter_recovery.pdf_evidence import extract_book_pdf_evidence
from chapter_recovery.toc import find_toc_pages

from .llm_client import ChatClient, OpenAICompatibleClient, aggregate_usage


FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)
TERMINAL = (".", "?", "!", ";", "。", "？", "！", "；")
ALLOWED_KINDS = {"part", "chapter", "section", "frontmatter", "backmatter", "toc_entry"}


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _vision_settings(chapter_config: Dict[str, Any]) -> Dict[str, Any]:
    settings = dict(chapter_config.get("vision") or {})
    settings.setdefault("model", "deepseek-flash")
    settings.setdefault("toc_scan_pages", chapter_config.get("toc_search_pages", [0, 30]))
    settings.setdefault("toc_batch_size", 4)
    settings.setdefault("body_batch_size", 4)
    settings.setdefault("max_body_pages", 120)
    settings.setdefault("scan_all_body_pages", False)
    settings.setdefault("allow_remote_input_images", False)
    settings.setdefault("render_dpi", 120)
    settings.setdefault("max_tokens", 4096)
    settings.setdefault("temperature", 0.0)
    settings.setdefault("structured_retries", 3)
    settings.setdefault("toc_page_min_confidence", 0.70)
    settings.setdefault("toc_entry_min_confidence", 0.72)
    settings.setdefault("heading_min_confidence", 0.72)
    return settings


def _client(provider: Dict[str, Any], settings: Dict[str, Any]) -> OpenAICompatibleClient:
    value = dict(provider)
    value["model"] = str(settings.get("model") or "deepseek-flash")
    value["model_env"] = "DEEPSEEK_VISION_MODEL"
    value["thinking_mode"] = "disabled"
    value["thinking_env"] = "DEEPSEEK_VISION_THINKING_MODE"
    value["native_json_mode"] = True
    value["max_tokens"] = int(settings.get("max_tokens", 4096))
    value["temperature"] = float(settings.get("temperature", 0.0))
    return OpenAICompatibleClient(value)


def _mime(path: Path) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    return guessed if guessed in {"image/jpeg", "image/png", "image/gif", "image/webp"} else "image/png"


def _data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_mime(path)};base64,{payload}"


class PageImageProvider:
    def __init__(
        self,
        raw_pages: Sequence[Dict[str, Any]],
        input_json: Path,
        pdf_path: Optional[Path],
        *,
        dpi: int,
        allow_remote_urls: bool = False,
    ):
        self.raw_pages = raw_pages
        self.input_json = input_json
        self.pdf_path = pdf_path if pdf_path and pdf_path.is_file() else None
        self.dpi = max(72, min(200, int(dpi)))
        self.allow_remote_urls = bool(allow_remote_urls)
        self._document: Any = None
        self.sources: Dict[int, str] = {}

    def close(self) -> None:
        if self._document is not None:
            self._document.close()
            self._document = None

    def __enter__(self) -> "PageImageProvider":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _pdf_image(self, page_index: int) -> Optional[str]:
        if self.pdf_path is None:
            return None
        try:
            import fitz
        except ImportError:
            return None
        if self._document is None:
            self._document = fitz.open(self.pdf_path)
        if not (0 <= page_index < self._document.page_count):
            return None
        page = self._document.load_page(page_index)
        scale = float(self.dpi) / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        try:
            payload = pixmap.tobytes("jpeg")
            mime = "image/jpeg"
        except Exception:
            payload = pixmap.tobytes("png")
            mime = "image/png"
        self.sources[page_index] = "pdf_render"
        return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"

    def _ocr_image(self, page_index: int) -> Optional[str]:
        if not (0 <= page_index < len(self.raw_pages)):
            return None
        raw = self.raw_pages[page_index]
        value = raw.get("inputImage")
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip()
        if value.startswith("data:image/"):
            self.sources[page_index] = "ocr_data_url"
            return value
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            if not self.allow_remote_urls:
                return None
            self.sources[page_index] = "ocr_remote_url"
            return value
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (self.input_json.parent / path).resolve()
        if not path.is_file():
            return None
        self.sources[page_index] = "ocr_local_image"
        return _data_url(path)

    def get(self, page_index: int) -> Optional[str]:
        return self._pdf_image(page_index) or self._ocr_image(page_index)


def _ocr_hint(page: Sequence[Any]) -> str:
    rows = []
    for block in page[:30]:
        text = " ".join(block.clean_content.split())
        if not text:
            continue
        rows.append(f"[{block.label}] {text[:240]}")
    return "\n".join(rows[:24])


def _blocks_for_pages(
    page_indices: Sequence[int],
    pages: Sequence[Sequence[Any]],
    provider: PageImageProvider,
    instruction: str,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": instruction}]
    available: List[int] = []
    for page_index in page_indices:
        image_url = provider.get(page_index)
        if not image_url:
            continue
        available.append(page_index)
        content.append({
            "type": "text",
            "text": f"PAGE_JSON={page_index}\nOCR_HINTS:\n{_ocr_hint(pages[page_index])}",
        })
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content, available


def _batches(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    size = max(1, int(size))
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _json_completion(
    client: ChatClient,
    system: str,
    content: List[Dict[str, Any]],
    retries: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    method = getattr(client, "complete_multimodal_json_text", None)
    if not callable(method):
        raise TypeError("Vision client must support complete_multimodal_json_text")
    usages: List[Dict[str, Any]] = []
    last_error: Optional[BaseException] = None
    for _attempt in range(max(1, min(5, int(retries)))):
        text, usage = method(system, content)
        usages.append(dict(usage or {}))
        try:
            cleaned = FENCE.sub("", text.strip()).strip()
            value = json.loads(cleaned)
            if not isinstance(value, dict):
                raise ValueError("Vision response must be one JSON object")
            total = aggregate_usage(usages)
            total["request_count"] = sum(int(row.get("request_count") or 1) for row in usages)
            return value, total
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError("Vision JSON response failed validation after retries") from last_error


def _normalize_toc_response(
    value: Dict[str, Any],
    allowed_pages: Sequence[int],
) -> List[Dict[str, Any]]:
    allowed = set(allowed_pages)
    output: List[Dict[str, Any]] = []
    seen = set()
    for row in value.get("pages") or []:
        if not isinstance(row, dict):
            continue
        try:
            page_index = int(row.get("page_json"))
            confidence = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if page_index not in allowed or page_index in seen or not 0.0 <= confidence <= 1.0:
            continue
        seen.add(page_index)
        entries = []
        for item in row.get("entries") or []:
            if not isinstance(item, dict):
                continue
            title = " ".join(str(item.get("title") or "").split()).strip()
            if not title or len(title) > 240:
                continue
            try:
                item_confidence = float(item.get("confidence", confidence))
            except (TypeError, ValueError):
                item_confidence = confidence
            if not 0.0 <= item_confidence <= 1.0:
                continue
            kind = str(item.get("kind") or "toc_entry").strip().casefold()
            if kind not in ALLOWED_KINDS:
                kind = "toc_entry"
            try:
                level = min(6, max(1, int(item.get("level") or 1)))
            except (TypeError, ValueError):
                level = 1
            entries.append({
                "title": title,
                "page_label": str(item.get("page_label") or "").strip(),
                "kind": kind,
                "level": level,
                "confidence": round(item_confidence, 4),
            })
        output.append({
            "page_json": page_index,
            "is_toc": bool(row.get("is_toc")),
            "confidence": round(confidence, 4),
            "entries": entries,
        })
    return output


def _normalize_heading_response(
    value: Dict[str, Any],
    allowed_pages: Sequence[int],
) -> List[Dict[str, Any]]:
    allowed = set(allowed_pages)
    headings: List[Dict[str, Any]] = []
    seen = set()
    for page in value.get("pages") or []:
        if not isinstance(page, dict):
            continue
        try:
            page_index = int(page.get("page_json"))
        except (TypeError, ValueError):
            continue
        if page_index not in allowed:
            continue
        for item in page.get("headings") or []:
            if not isinstance(item, dict):
                continue
            title = " ".join(str(item.get("title") or "").split()).strip()
            if not title or len(title) > 240:
                continue
            try:
                confidence = float(item.get("confidence") or 0.0)
                level = min(6, max(2, int(item.get("level") or 3)))
            except (TypeError, ValueError):
                continue
            if not 0.0 <= confidence <= 1.0:
                continue
            kind = str(item.get("kind") or "section").strip().casefold()
            if kind not in ALLOWED_KINDS:
                kind = "section"
            bbox = item.get("bbox")
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in bbox)
            ):
                bbox = None
            key = (page_index, title.casefold(), level)
            if key in seen:
                continue
            seen.add(key)
            headings.append({
                "page_json": page_index,
                "title": title,
                "kind": kind,
                "level": level,
                "bbox": [round(float(v), 4) for v in bbox] if bbox else None,
                "confidence": round(confidence, 4),
            })
    return headings


def _toc_system() -> str:
    return """You inspect photographed or rendered book pages to find the real table of contents.
Only report text visibly present on each supplied page. Do not infer missing entries.
A TOC page usually contains multiple section/chapter titles associated with printed page labels.
Preserve source-language titles exactly as visible except obvious whitespace normalization.
For each entry classify kind as part, chapter, section, frontmatter, backmatter, or toc_entry.
level is the visible TOC hierarchy depth, starting at 1.
Return one strict JSON object:
{"pages":[{"page_json":0,"is_toc":true,"confidence":0.0,"entries":[{"title":"","page_label":"","kind":"chapter","level":1,"confidence":0.0}]}]}
Cover every supplied PAGE_JSON exactly once."""


def _heading_system(toc_titles: Sequence[str]) -> str:
    toc_context = "\n".join(f"- {title}" for title in toc_titles[:120])
    return f"""You inspect rendered body pages of a book and identify genuine structural headings.
The objective is to recover chapters and especially secondary sections that the printed TOC may omit.
Only return titles visibly printed as headings on the supplied page. Never turn running headers,
captions, figure labels, quotations, author bylines, page numbers, or ordinary sentences into headings.
Use layout, whitespace, typography, numbering and relation to surrounding body text.
level is the best book hierarchy depth from 2 to 6. kind is part, chapter, section, frontmatter,
or backmatter. bbox is normalized [x1,y1,x2,y2] in page coordinates when reasonably estimable.
Known TOC titles are context only; do not hallucinate them onto a page:
{toc_context}
Return one strict JSON object:
{{"pages":[{{"page_json":0,"headings":[{{"title":"","kind":"section","level":3,"bbox":[0,0,1,1],"confidence":0.0}}]}}]}}
Cover every supplied PAGE_JSON exactly once."""


def _suspect_body_pages(
    pages: Sequence[Sequence[Any]],
    toc_pages: Sequence[int],
    page_map: Any,
    toc_rows: Sequence[Dict[str, Any]],
    pdf_path: Optional[Path],
    settings: Dict[str, Any],
) -> List[int]:
    excluded = set(toc_pages)
    scores: Dict[int, float] = {}
    start = max(excluded) + 1 if excluded else 0
    for page_index in range(start, len(pages)):
        if page_index in excluded:
            continue
        score = 0.0
        for block in pages[page_index]:
            text = block.clean_content
            if block.label in {"doc_title", "paragraph_title"}:
                score += 5.0
            elif block.label == "header":
                score += 1.0
            if not text or len(text) > 160 or text.endswith(TERMINAL):
                continue
            words = text.split()
            parsed = numbering(text)
            if parsed is not None:
                score += 3.0
            if 1 <= len(words) <= 14:
                x1, y1, x2, y2 = block.normalized_bbox()
                if 0.06 <= y1 <= 0.92 and (y2 - y1) <= 0.08:
                    score += 1.0
        if score > 0:
            scores[page_index] = score

    if pdf_path is not None and pdf_path.is_file():
        pdf_result = extract_book_pdf_evidence(pdf_path, len(pages))
        if pdf_result.ok:
            for page in pdf_result.pages:
                if page.page_index < start or page.page_index in excluded:
                    continue
                if page.large_lines:
                    scores[page.page_index] = scores.get(page.page_index, 0.0) + min(6.0, len(page.large_lines) * 1.5)

    for toc_page in toc_rows:
        for item in toc_page.get("entries") or []:
            label = str(item.get("page_label") or "")
            if not label.isdigit():
                continue
            target = page_map.json_page(int(label))
            if target is None:
                continue
            for offset, bonus in ((0, 8.0), (-1, 2.0), (1, 2.0)):
                candidate = target + offset
                if start <= candidate < len(pages) and candidate not in excluded:
                    scores[candidate] = scores.get(candidate, 0.0) + bonus

    if bool(settings.get("scan_all_body_pages")):
        for page_index in range(start, len(pages)):
            if page_index not in excluded:
                scores.setdefault(page_index, 0.01)

    limit = max(1, int(settings.get("max_body_pages", 120)))
    ranked = sorted(scores, key=lambda page: (-scores[page], page))
    selected = ranked[:limit]
    return sorted(selected)


def collect_structure_vision(
    chapter_config: Dict[str, Any],
    provider: Dict[str, Any],
    *,
    config_base: Path,
    pdf_override: Optional[Path] = None,
    client: Optional[ChatClient] = None,
) -> Dict[str, Any]:
    settings = _vision_settings(chapter_config)
    input_json = _resolve(config_base, str(chapter_config["input_json"]))
    raw_pages = load_pages(input_json)
    pages = normalize_blocks(raw_pages)
    page_map = infer_page_map(pages)

    pdf_path = pdf_override.expanduser().resolve() if pdf_override is not None else None
    if pdf_path is None and chapter_config.get("pdf_path"):
        pdf_path = _resolve(config_base, str(chapter_config["pdf_path"]))
    if pdf_path is None:
        candidate = input_json.with_suffix(".pdf")
        pdf_path = candidate if candidate.is_file() else None

    review_client = client or _client(provider, settings)
    retries = int(settings.get("structured_retries", 3))
    usage_rows: List[Dict[str, Any]] = []

    scan_range = settings.get("toc_scan_pages") or [0, 30]
    start = max(0, int(scan_range[0]))
    end = min(len(pages) - 1, int(scan_range[1])) if pages else -1
    toc_candidates = list(range(start, end + 1)) if end >= start else []

    vision_toc: List[Dict[str, Any]] = []
    scanned_toc: List[int] = []
    scanned_body: List[int] = []
    with PageImageProvider(
        raw_pages,
        input_json,
        pdf_path,
        dpi=int(settings.get("render_dpi", 120)),
        allow_remote_urls=bool(settings.get("allow_remote_input_images", False)),
    ) as images:
        for batch in _batches(toc_candidates, int(settings.get("toc_batch_size", 4))):
            content, available = _blocks_for_pages(
                batch,
                pages,
                images,
                "Identify which supplied pages are table-of-contents pages and transcribe their visible entries.",
            )
            if not available:
                continue
            value, usage = _json_completion(review_client, _toc_system(), content, retries)
            usage_rows.append(usage)
            scanned_toc.extend(available)
            vision_toc.extend(_normalize_toc_response(value, available))

        toc_page_minimum = float(settings.get("toc_page_min_confidence", 0.70))
        accepted_toc_pages = sorted({
            int(row["page_json"])
            for row in vision_toc
            if row.get("is_toc") and float(row.get("confidence") or 0.0) >= toc_page_minimum
        })
        deterministic_toc = find_toc_pages(pages, chapter_config.get("toc_search_pages", [0, 30]))
        all_toc_pages = sorted(set(deterministic_toc) | set(accepted_toc_pages))
        body_candidates = _suspect_body_pages(
            pages, all_toc_pages, page_map, vision_toc, pdf_path, settings
        )
        toc_titles = [
            str(item.get("title") or "")
            for page in vision_toc
            for item in (page.get("entries") or [])
            if str(item.get("title") or "").strip()
        ]
        headings: List[Dict[str, Any]] = []
        for batch in _batches(body_candidates, int(settings.get("body_batch_size", 4))):
            content, available = _blocks_for_pages(
                batch,
                pages,
                images,
                "Find genuine structural headings visible on these body pages, including secondary sections omitted from the TOC.",
            )
            if not available:
                continue
            value, usage = _json_completion(review_client, _heading_system(toc_titles), content, retries)
            usage_rows.append(usage)
            scanned_body.extend(available)
            headings.extend(_normalize_heading_response(value, available))

        source_counts: Dict[str, int] = {}
        for source in images.sources.values():
            source_counts[source] = source_counts.get(source, 0) + 1

    total_usage = aggregate_usage(usage_rows)
    total_usage["request_count"] = sum(int(row.get("request_count") or 0) for row in usage_rows)
    input_sha = sha256(input_json.read_bytes()).hexdigest()
    pdf_sha = sha256(pdf_path.read_bytes()).hexdigest() if pdf_path and pdf_path.is_file() else None
    return {
        "schema_version": 1,
        "generated_by": "deepseek_vision_structure_v1",
        "model": getattr(review_client, "model", str(settings.get("model") or "deepseek-flash")),
        "input_json": str(input_json),
        "input_sha256": input_sha,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "pdf_sha256": pdf_sha,
        "toc_pages": sorted(vision_toc, key=lambda row: int(row["page_json"])),
        "headings": sorted(headings, key=lambda row: (int(row["page_json"]), float((row.get("bbox") or [0, 0])[1]), row["title"])),
        "scan": {
            "toc_pages_requested": toc_candidates,
            "toc_pages_scanned": sorted(set(scanned_toc)),
            "body_pages_scanned": sorted(set(scanned_body)),
            "image_sources": source_counts,
            "settings": settings,
        },
        "usage": total_usage,
    }
