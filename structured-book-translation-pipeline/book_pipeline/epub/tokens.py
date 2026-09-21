from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Tuple


TOKEN = re.compile(r"\[\[EPUB:(\d+):(OPEN|CLOSE):([A-Za-z0-9_-]+)\]\]")


def strip_tokens(text: str) -> str:
    """The logical prose of a segment, with protected tokens removed.

    A native EPUB segment interleaves its text with ``[[EPUB:n:OPEN:tag]]``
    boundaries. Anything that reasons about the *words* - terminology
    extraction, glossary matching, QA keywords - must read this form instead of
    the raw slot template: over the raw text a phrase scanner sees ``EPUB``,
    ``OPEN``/``CLOSE`` and fragments like ``a]][[EPUB`` as source words and
    proposes them as candidate terms.

    A token becomes a single space, so words on either side stay separate
    instead of being glued into a word that never appeared in the book.
    """
    if "[[EPUB:" not in text:
        return text
    return re.sub(r"\s+", " ", TOKEN.sub(" ", text)).strip()


def local_name(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1].casefold() if isinstance(tag, str) else "comment"


def _small_caps_lead(value: str) -> Tuple[str, str]:
    """Split a slot ending in the capital that introduces a small-caps run.

    The book sets small caps as ``V<small>ERBS AND ...</small>``: the leading
    capital lives in the *parent's* text, the rest inside ``<small>``. With one
    slot per DOM node that capital shares a slot with the separator (``". V"``),
    so a translation that moves the whole word into the small-caps span empties
    a slot that had content and is rejected -- the only surviving option is to
    echo the English verbatim. Returning the split lets the capital be compiled
    as part of the small-caps element instead, making the run one cohesive unit.

    Only a lone ``A-Z`` preceded by whitespace or a non-word character is taken,
    so an ordinary capitalised word, an all-caps acronym, or a digit does not
    qualify. An empty or ``None`` remainder is returned as ``""``.
    """
    if not value:
        return "", ""
    match = re.search(r"(?<![A-Za-z0-9])([A-Z])$", value)
    if not match:
        return value, ""
    return value[: match.start(1)], match.group(1)


def compile_inline(element: Any, paths: Dict[int, str], opaque: Callable[[Any], bool]) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """Compile a block without flattening child structure or editable opaque nodes."""
    slots: List[Dict[str, str]] = []
    template: List[str] = []
    token_index = 0

    def token(token_id: int, tag: str, direction: str) -> None:
        template.append(f"[[EPUB:{token_id}:{direction}:{tag}]]")

    def add_slot(value: str, kind: str, path: str) -> None:
        if not value:
            return
        slot_id = f"s{len(slots)}"
        slots.append({"id": slot_id, "kind": kind, "path": path, "attribute": "", "value": value})
        template.append(slot_id)

    def sink_lead(parent: Any, child: Any) -> str:
        """The stranded small-caps capital to compile inside ``child``.

        The book sets small caps as ``V<small>ERBS AND ...</small>``, so the
        leading capital sits *before* ``<small>`` -- in the parent's own text
        (``<p>V<small>ERBS``) or, far more often, in the tail of an inline
        sibling (``</em>. V<small>ERBS``). Compiling that capital into the
        small-caps element makes the run one cohesive translatable unit; left
        where it is, it shares a slot with the separator (``". V"``) and any
        translation that moves the whole word into the span empties a populated
        slot and is rejected.

        Only a lone capital at the very end of the preceding text is taken, and
        only when something precedes it, so an ordinary capitalised word or an
        all-caps acronym is left alone.
        """
        if local_name(child.tag) != "small" or len(child):
            return ""
        siblings = list(parent)
        index = siblings.index(child)
        if index:
            previous = siblings[index - 1]
            tail, lead = _small_caps_lead(str(previous.tail or ""))
            if lead:
                previous.tail = tail
                return lead
            return ""
        text, lead = _small_caps_lead(str(parent.text or ""))
        if not lead:
            return ""
        parent.text = text
        return lead

    def walk(node: Any, *, is_root: bool, lead: str = "") -> None:
        nonlocal token_index
        tag = local_name(node.tag)
        token_id = token_index
        if not is_root:
            token_index += 1
            token(token_id, tag, "OPEN")
            if opaque(node):
                token(token_id, tag, "CLOSE")
                return
        children = list(node)
        # Resolve every stranded capital first: it must leave the parent's text
        # or a preceding sibling's tail *before* either is compiled into a slot.
        leads = {id(child): sink_lead(node, child) for child in children}
        # The lead belongs to this element's own text slot, not to a second one:
        # ``apply_segment_slots`` writes each slot to the same ``element.text``,
        # so two text slots on one element would silently overwrite each other.
        add_slot(lead + str(node.text or ""), "text", paths[id(node)])
        for child in children:
            walk(child, is_root=False, lead=leads[id(child)])
            add_slot(str(child.tail or ""), "tail", paths[id(child)])
        if not is_root:
            token(token_id, tag, "CLOSE")
    walk(element, is_root=True)
    values = {slot["id"]: slot["value"] for slot in slots}
    return "".join(values.get(piece, piece) for piece in template), slots, template


def validate_tokens(source_text: str, candidate: str) -> None:
    source_tokens = [(match.group(1), match.group(2), match.group(3)) for match in TOKEN.finditer(source_text)]
    candidate_tokens = [(match.group(1), match.group(2), match.group(3)) for match in TOKEN.finditer(candidate)]
    if source_tokens != candidate_tokens:
        raise ValueError("EPUB protected tokens changed, reordered, or omitted")
    stack: List[Tuple[str, str]] = []
    for token_id, direction, tag in candidate_tokens:
        if direction == "OPEN":
            stack.append((token_id, tag))
        elif not stack or stack.pop() != (token_id, tag):
            raise ValueError("EPUB protected tokens are not legally nested")
    if stack:
        raise ValueError("EPUB protected tokens are not balanced")


def reconstruct_slots(template: Iterable[str], slots: Iterable[Dict[str, str]], candidate: str) -> Dict[str, str]:
    pieces = list(template)
    token_positions = [index for index, piece in enumerate(pieces) if TOKEN.fullmatch(piece)]
    boundaries = [-1] + token_positions + [len(pieces)]
    token_matches = list(TOKEN.finditer(candidate))
    if len(token_matches) != len(token_positions):
        raise ValueError("EPUB protected-token reconstruction mismatch")
    groups: List[List[str]] = [pieces[left + 1:right] for left, right in zip(boundaries, boundaries[1:])]
    text_parts: List[str] = []
    cursor = 0
    for match in token_matches:
        text_parts.append(candidate[cursor:match.start()])
        cursor = match.end()
    text_parts.append(candidate[cursor:])
    values: Dict[str, str] = {}
    for slot_ids, text in zip(groups, text_parts):
        if not slot_ids:
            if text:
                raise ValueError("EPUB protected-token reconstruction mismatch")
            continue
        if len(slot_ids) != 1:
            raise ValueError("EPUB slot template is ambiguous")
        values[slot_ids[0]] = text
    return values
