"""Strict single-target response parser for contextual translation v2.

``parse_target_only`` accepts a response that references EXACTLY the target
segment id and nothing else: context segment ids, rewrites, duplicates, extra
items, missing ids and empty translations are all rejected with ``ValueError``.
It deliberately does not reuse v1's lenient singleton branch (which tolerates
ID rewrites), so a leaky model response can never be accepted as a valid
translation.

Structural-token preservation (footnote refs, HTML comments, URLs, formulas)
and table delimiter counts are enforced with the same rules v1 applies
(translate.parse_translation), sharing only the module-level ``STRUCTURAL_TOKEN``
regex — v1 code is not modified.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..models import Segment
from ..translate import STRUCTURAL_TOKEN  # read-only regex reuse; v1 untouched

# Provider tool-call framing noise.  A provider's JSON response sometimes
# appends DeepSeek tool-call close tags after an otherwise complete JSON
# object (observed: '…"}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>').
# They carry no payload and make ``json.loads`` fail with "Extra data", so
# every retry of an affected segment failed (live regression:
# the-dictators-dilemma segments seg-9a43660c9c72e1707a92 and siblings).
# Stripping these tags never makes an incomplete response parse: a truncated
# object still raises JSONDecodeError and is retried.
_TOOL_MARKUP = re.compile(r"</?\s*[｜|]+\s*DSML\s*[｜|]+[^>]*>")


def parse_target_only(content: str, target: Segment) -> str:
    """Parse a strict single-target response and return the translated text.

    Raises:
        ValueError: missing/multiple/wrong id, leaked context ids, empty
            translation, structural-token loss or table delimiter drift.
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    if "DSML" in cleaned:
        cleaned = _TOOL_MARKUP.sub("", cleaned).strip()
    data = json.loads(cleaned)  # JSONDecodeError propagates to the retry loop
    items = _items(data)
    if items is None or len(items) != 1 or not isinstance(items[0], dict):
        raise ValueError("Translation response must contain exactly one target object")
    item = items[0]
    identifiers = [
        str(item[key])
        for key in ("id", "segment_id")
        if item.get(key) is not None
    ]
    if len(identifiers) != 1 or identifiers[0] != target.id:
        raise ValueError(
            f"Translation response must reference exactly the target id {target.id!r} "
            f"(got {identifiers!r}); context ids must never be echoed"
        )
    text = str(
        item.get("translated_text")
        if item.get("translated_text") is not None
        else item.get("translation")
        if item.get("translation") is not None
        else item.get("text")
        if item.get("text") is not None
        else ""
    )
    if not text.strip():
        raise ValueError(f"Empty translation for {target.id}")
    return _verify_structural_tokens(target, text)


def _items(data: Any) -> Any:
    """Normalize accepted shapes to a list of items.

    Accepted: a single object carrying id/text keys; ``{"translations": [...]}``;
    ``{"segments": [...]}``; a JSON list of exactly one object.  Any container
    form that could smuggle extra ids is preserved as-is so the caller's length
    check rejects it.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "segments" in data:
            return data["segments"]
        if "translations" in data:
            return data["translations"]
        if any(key in data for key in ("id", "segment_id", "translated_text", "translation", "text")):
            return [data]
    return None


def _verify_structural_tokens(target: Segment, translated_text: str) -> None:
    def _norm_ws(token: str) -> str:
        """Whitespace-only normalization: used for the verbatim re-write below,
        which must stay limited to tokens that are semantically the SAME token
        (footnote whitespace), never to tokens that merely survive comparison."""
        return re.sub(r"\s+", "", token)

    def _norm(token: str) -> str:
        # P09 live-run fix: footnote/formula tokens may differ ONLY by internal
        # whitespace ("$ ^{9} $" vs "$^{9}$") and are semantically identical;
        # collapse all whitespace before comparing (punctuation-stripping kept
        # for leading/trailing separators).
        #
        # Closing brackets are separators too: the URL class in
        # STRUCTURAL_TOKEN excludes the full-width "）" but not the ASCII ")"
        # (which is also legal inside a URL), so a source citation written as
        # "(http://x.example)" and a Chinese rendering "(http://x.example）"
        # produced different tokens and could never parse.  Stripping a
        # trailing bracket from BOTH sides keeps the URL text itself required
        # verbatim while ignoring the citation punctuation around it.
        return _norm_ws(token.rstrip(".,;:!?。；！？)]}）】》」』"))

    source_tokens = STRUCTURAL_TOKEN.findall(target.source_text)
    translated_tokens = STRUCTURAL_TOKEN.findall(translated_text)
    if len(source_tokens) == len(translated_tokens):
        for source_token, translated_token in zip(source_tokens, translated_tokens):
            if (_norm_ws(source_token) == _norm_ws(translated_token)
                    and source_token != translated_token):
                translated_text = translated_text.replace(translated_token, source_token)
    required = [_norm(token) for token in source_tokens]
    produced = [_norm(token) for token in STRUCTURAL_TOKEN.findall(translated_text)]
    if required != produced:
        raise ValueError(f"Structural token mismatch for {target.id}")
    if target.kind == "table" and translated_text.count("|") != target.source_text.count("|"):
        raise ValueError(f"Markdown table delimiter mismatch for {target.id}")
    return translated_text
