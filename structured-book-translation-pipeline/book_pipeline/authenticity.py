"""Translation authenticity guard (user-reported P09 defect).

The P08 rethinking export was produced from the demo-fake corpus: model
``demo-fake-contextual`` rows that merely prefix ``译：`` to the English source,
so the exported book was English text with stray 译 markers. Demo/fake
translations are valid only as pipeline-mechanics fixtures; they must NEVER
reach the product path (render/export) silently.

``render_book`` and ``export_book`` fail closed unless the caller explicitly
opts in with ``config["allow_demo_translations"] = true`` (smoke/tests only).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

# Models carrying demo/fake markers (our fixture clients use fake-model /
# demo-fake-contextual / demo-fake-translator / offline-fake...).
DEMO_MODEL = re.compile(r"(?:demo|fake)[-_]", re.I)

# Metadata markers a real provider path may set (future-proofing).
DEMO_METADATA_KEYS = ("demo_only", "is_demo")


def is_demo_row(row: Dict[str, Any]) -> bool:
    """True when a translation record clearly originates from a demo/fake
    client (by model name or an explicit metadata marker)."""
    model = str(row.get("model") or "")
    if DEMO_MODEL.search(model):
        return True
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in DEMO_METADATA_KEYS:
            if metadata.get(key):
                return True
    return False


def current_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Latest row per segment_id (last-write-wins, mirroring
    ``translate._completed`` semantics)."""
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[str(row.get("segment_id"))] = row
    return latest


def demo_current_segments(rows: Iterable[Dict[str, Any]]) -> list:
    """segment_ids whose CURRENT (latest) translation record is demo/fake."""
    return [
        segment_id
        for segment_id, row in current_rows(rows).items()
        if is_demo_row(row)
    ]


def reject_demo_translations(config: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> None:
    """Fail closed: demo/fake translations may not be rendered/exported unless
    ``config["allow_demo_translations"]`` is explicitly true (smoke/tests)."""
    if config.get("allow_demo_translations"):
        return
    demo = demo_current_segments(rows)
    if not demo:
        return
    shown = ", ".join(segment[:16] for segment in demo[:5])
    raise RuntimeError(
        "refusing to render/export demo or fake translations "
        f"({len(demo)} current demo rows; e.g. {shown}). Set "
        "config['allow_demo_translations']=true only for smoke/tests."
    )
