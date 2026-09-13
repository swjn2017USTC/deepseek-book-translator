"""Contextual translation v2 orchestrator (spec §12, P06).

Single-target loop with per-chapter context: every LLM call translates exactly
one pending segment carrying a deterministic chapter brief, bounded
previous/next source windows from the same chapter, a bounded previous
translation and per-segment concept dependencies (fine cache invalidation).

The v2 path is purely additive: it reuses the v1 gates
(``require_structure_gate`` / ``require_fresh_cleaning`` /
``require_ready_glossary`` / ``glossary_cache_sha``), the private loaders
``translate._segments`` / ``translate._completed`` and the 10-attempt parse
retry with ``last_invalid_response.txt``.  v1 ``translate.py`` / ``models.py``
/ ``glossary.py`` are read-only dependencies — no edits.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from ..clean import require_structure_gate
from ..config import resolve_path
from ..glossary import glossary_cache_sha, require_ready_glossary
from ..io_utils import append_jsonl, write_json, write_jsonl
from ..llm_client import ChatClient, OpenAICompatibleClient
from ..models import Segment
from ..provenance import require_fresh_cleaning
from ..translate import SENTENCE_FINISH, _completed, _segments, translation_status
from .briefs import (
    brief_payload,
    build_chapter_briefs,
    nodes_index,
    write_chapter_briefs,
)
from .deps import (
    glossary_subset,
    load_concept_glossary,
    resolve_concept_deps,
)
from .invalidation import (
    brief_key_for_segment,
    current_briefs,
    current_concept_versions,
    is_translation_current_v2,
)
from .parser import parse_target_only
from .prompts import PROMPTS_VERSION, build_payload, system_prompt
from .records import record_row
from .windows import chapter_order, previous_translation_text, window


# Serializes all appends to shared progress/usage JSONL files. Multi-chapter
# parallel translation runs threads that each hold their own append-mode file
# handle; the lock guarantees one writer at a time so a long record line can
# never be interleaved with another thread's line.
_APPEND_LOCK = threading.Lock()


def require_contextual_mode(config: Dict[str, Any]) -> None:
    """Gate: translate-v2 requires ``translation.context_mode == "contextual_v2"``.

    A config missing the block or set to ``legacy`` belongs to the v1 pipeline;
    the error points at ``translate --config`` instead of silently re-running
    v1 semantics under a v2 command (D1 / §16).
    """
    translation = config.get("translation") or {}
    mode = translation.get("context_mode")
    if mode == "contextual_v2":
        return
    if mode == "legacy":
        raise ValueError(
            "config translation.context_mode is 'legacy': this book runs the v1 "
            "pipeline — use 'translate --config <config>' (not translate-v2)"
        )
    raise ValueError(
        'config must set "translation": {"context_mode": "contextual_v2", ...} to '
        'run translate-v2; books without the block keep using "translate --config <config>"'
    )


def _v2_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    translation = config.get("translation") or {}
    if (translation.get("chapter_brief") or {}).get("llm_enrich"):
        raise ValueError(
            "chapter_brief.llm_enrich=true is not supported by this version of the "
            "contextual pipeline (P06 ships deterministic structural briefs only); set it to false"
        )
    previous_segments = int(translation.get("previous_segments", 2))
    next_segments = int(translation.get("next_segments", 2))
    previous_translation_max_chars = int(translation.get("previous_translation_max_chars", 500))
    if previous_segments < 0 or next_segments < 0:
        raise ValueError("translation.previous_segments/next_segments must be non-negative")
    if previous_translation_max_chars < 0:
        raise ValueError("translation.previous_translation_max_chars must be non-negative")
    return {
        "previous_segments": previous_segments,
        "next_segments": next_segments,
        "previous_translation_max_chars": previous_translation_max_chars,
        # P09 unattended-run fix: when true, a segment that fails all parse
        # attempts is recorded as an explicit status:'error' row and SKIPPED so
        # the rest of the book still translates; the next run retries it.
        "skip_failed_segments": bool(translation.get("skip_failed_segments", False)),
    }


def _concept_path(config: Dict[str, Any]) -> Optional[Path]:
    settings = config.get("glossary_dependencies") or {}
    if not settings.get("enabled"):
        return None
    value = settings.get("concept_glossary_path")
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_base_dir"]) / path).resolve()


def _prepare(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gate + load + build every cache artifact once for a v2 run."""
    require_contextual_mode(config)
    settings = _v2_settings(config)
    require_structure_gate(config)
    require_fresh_cleaning(config)
    require_ready_glossary(config)
    glossary_sha256 = glossary_cache_sha(config)
    output_dir = resolve_path(config, "output_dir")
    segments_path = output_dir / "cleaned_segments.jsonl"
    if not segments_path.exists():
        raise FileNotFoundError("Run clean before translate-v2")
    progress_path = output_dir / "translations.jsonl"
    usage_path = output_dir / "usage.jsonl"
    structure = json.loads(resolve_path(config, "structure_json").read_text(encoding="utf-8"))
    nodes_by_id = nodes_index(structure)
    segments = _segments(segments_path)
    briefs = build_chapter_briefs(structure, segments)
    write_chapter_briefs(output_dir, briefs, book_id=config["book_id"])
    brief_hashes = current_briefs(briefs)
    brief_payloads = {chapter_id: brief_payload(brief) for chapter_id, brief in briefs.items()}
    completed = _completed(progress_path)
    concepts: List[Dict[str, Any]] = []
    concept_path = _concept_path(config)
    if concept_path is not None:
        concepts = load_concept_glossary(concept_path)
        rows = [
            {
                "segment_id": segment.id,
                "glossary_dependencies": resolve_concept_deps(segment, concepts),
            }
            for segment in segments
            if segment.translatable
        ]
        write_jsonl(output_dir / "contextual_deps.jsonl", rows)
    concept_versions = current_concept_versions(concepts)
    chapter_of: Dict[str, str] = {}
    for segment in segments:
        if segment.translatable or segment.source_text.strip():
            chapter_of[segment.id] = brief_key_for_segment(segment, nodes_by_id)
    requests_per_minute = float(config.get("provider", {}).get("requests_per_minute", 20))
    delay = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0

    def current(segment: Segment) -> bool:
        return is_translation_current_v2(
            segment,
            completed.get(segment.id),
            brief_hashes=brief_hashes,
            concept_versions=concept_versions,
            nodes_by_id=nodes_by_id,
        )

    return {
        "config": config,
        "settings": settings,
        "output_dir": output_dir,
        "progress_path": progress_path,
        "usage_path": usage_path,
        "segments": segments,
        "completed": completed,
        "concepts": concepts,
        "concept_versions": concept_versions,
        "nodes_by_id": nodes_by_id,
        "chapter_of": chapter_of,
        "briefs": briefs,
        "brief_hashes": brief_hashes,
        "brief_payloads": brief_payloads,
        "glossary_sha256": glossary_sha256,
        "delay": delay,
        "current": current,
        "_contexts": {},
    }


def _context(prepared: Dict[str, Any], chapter_id: str) -> Dict[str, Any]:
    contexts = prepared["_contexts"]
    if chapter_id in contexts:
        return contexts[chapter_id]
    ordered = chapter_order([
        segment
        for segment in prepared["segments"]
        if prepared["chapter_of"].get(segment.id) == chapter_id
        and segment.kind != "image"
        and segment.source_text.strip()
    ])
    position = {segment.id: index for index, segment in enumerate(ordered)}
    rows: List[Optional[Dict[str, Any]]] = []
    for segment in ordered:
        row = prepared["completed"].get(segment.id)
        rows.append(row if (row and prepared["current"](segment)) else None)
    context = {"members": ordered, "position": position, "rows": rows}
    contexts[chapter_id] = context
    return context


def _translate_one(
    segment: Segment,
    prepared: Dict[str, Any],
    *,
    batch_index: int,
) -> Dict[str, Any]:
    """Translate one pending segment with full context; 10-attempt parse retry.

    Returns ``{"record": ..., "usage": ..., "attempts": n}``.  Raises
    RuntimeError when all 10 attempts fail (mirroring the v1 loop); every
    failed attempt writes ``last_invalid_response.txt``.
    """
    config = prepared["config"]
    output_dir = prepared["output_dir"]
    client = prepared["client"]
    settings = prepared["settings"]
    chapter_id = prepared["chapter_of"].get(segment.id)
    if not chapter_id:
        raise RuntimeError(f"Segment {segment.id} has no chapter brief; cannot translate")
    brief = prepared["briefs"][chapter_id]
    context = _context(prepared, chapter_id)
    index = context["position"].get(segment.id)
    if index is None:
        raise RuntimeError(f"Segment {segment.id} missing from chapter context")
    previous_sources, next_sources = window(
        context["members"],
        index,
        prev=settings["previous_segments"],
        next_=settings["next_segments"],
    )
    previous_translation = previous_translation_text(
        context["rows"][:index],
        max_chars=settings["previous_translation_max_chars"],
        sentence_finish=SENTENCE_FINISH,
    )
    dependencies = resolve_concept_deps(segment, prepared["concepts"])
    glossary = glossary_subset(segment, prepared["concepts"], dependencies)
    prompt = system_prompt(config)
    payload = build_payload(
        chapter_brief=prepared["brief_payloads"][chapter_id],
        previous_source=previous_sources,
        target=segment,
        next_source=next_sources,
        previous_translation=previous_translation,
        glossary=glossary,
    )
    last_error: Optional[Exception] = None
    translated_text = ""
    usage: Dict[str, Any] = {}
    for attempt in range(1, 11):
        content, usage = client.complete(prompt, json.dumps(payload, ensure_ascii=False))
        try:
            translated_text = parse_target_only(content, segment)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            (output_dir / "last_invalid_response.txt").write_text(content, encoding="utf-8")
    if not translated_text:
        raise RuntimeError(f"Segment {segment.id} failed response validation") from last_error
    record = record_row(
        segment,
        translated_text,
        client.model,
        prepared["glossary_sha256"],
        chapter_id=chapter_id,
        brief_hash=brief.brief_hash,
        prompt_version=PROMPTS_VERSION,
        deps=dependencies,
    )
    return {"record": record, "usage": usage, "attempts": attempt, "batch_index": batch_index}


def _commit_target(
    prepared: Dict[str, Any],
    segment: Segment,
    outcome: Dict[str, Any],
) -> None:
    """Append the record + usage row and refresh in-memory currency state."""
    record = outcome["record"]
    with _APPEND_LOCK:
        append_jsonl(prepared["progress_path"], [record])
    now = record["timestamp"]
    usage_row = {
        "timestamp": now,
        "batch": outcome["batch_index"],
        "model": record["model"],
        "segment_ids": [segment.id],
        "usage": outcome["usage"],
        "prompt_version": record["metadata"]["prompt_version"],
        "context_mode": record["metadata"]["context_mode"],
    }
    with _APPEND_LOCK:
        append_jsonl(prepared["usage_path"], [usage_row])
    prepared["completed"][segment.id] = record
    chapter_id = prepared["chapter_of"].get(segment.id)
    if chapter_id and chapter_id in prepared["_contexts"]:
        context = prepared["_contexts"][chapter_id]
        position = context["position"].get(segment.id)
        if position is not None:
            context["rows"][position] = record


def _v2_report(prepared: Dict[str, Any], translated_this_run: int) -> Dict[str, Any]:
    segments = prepared["segments"]
    current = [segment for segment in segments if segment.translatable and prepared["current"](segment)]
    required = [segment for segment in segments if segment.translatable]
    glossary_ready = True
    config = prepared["config"]
    try:
        prepared["glossary_sha256"] = glossary_cache_sha(config)
    except (FileNotFoundError, RuntimeError, ValueError):
        glossary_ready = False
    output_dir = prepared["output_dir"]
    artifact_names = ["chapter_briefs.json"]
    deps_path = output_dir / "contextual_deps.jsonl"
    if deps_path.exists():
        artifact_names.append("contextual_deps.jsonl")
    return {
        "schema_version": 1,
        "book_id": config["book_id"],
        "pipeline": "contextual_v2",
        "required_segments": len(required),
        "completed_segments": len(current),
        "remaining_segments": len(required) - len(current),
        "coverage": round(len(current) / len(required), 6) if required else 1.0,
        "ready_to_render": glossary_ready and len(current) == len(required),
        "glossary_ready": glossary_ready,
        "glossary_error": None,
        "glossary_sha256": prepared["glossary_sha256"] if glossary_ready else None,
        "translated_this_run": translated_this_run,
        "artifacts": {name: str(output_dir / name) for name in artifact_names},
        "translation_file": str(prepared["progress_path"]),
        "usage_file": str(prepared["usage_path"]),
    }


def _translate_item(
    prepared: Dict[str, Any],
    segment: Segment,
    *,
    batch_index: int,
    error_segments: List[str],
) -> bool:
    """Translate + persist one segment (shared by serial and parallel paths).

    Returns True on a completed commit, False when a segment was recorded as
    ``status:'error'`` under ``skip_failed_segments``.  Raises when
    ``skip_failed_segments`` is false (fail-closed), so a caller can abort.
    """
    try:
        outcome = _translate_one(segment, prepared, batch_index=batch_index)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        if not prepared["settings"].get("skip_failed_segments"):
            raise
        _record_failed_segment(prepared, segment, exc)
        error_segments.append(segment.id)
        return False
    _commit_target(prepared, segment, outcome)
    return True


def _translate_serial(
    prepared: Dict[str, Any],
    pending: List[Segment],
    error_segments: List[str],
) -> int:
    """Existing single-threaded loop (max_workers<=1): unchanged behaviour."""
    translated_this_run = 0
    for batch_index, segment in enumerate(pending, 1):
        if _translate_item(prepared, segment, batch_index=batch_index,
                           error_segments=error_segments):
            translated_this_run += 1
        if prepared["delay"]:
            time.sleep(prepared["delay"])
    return translated_this_run


class _Pacer:
    """Global request pacer shared by all parallel workers.

    Enforces at most one request start every ``interval`` seconds across the
    whole pool (``requests_per_minute`` stays an aggregate rate, not a per-
    worker multiplier, so N workers cannot exceed the provider's rate budget).
    ``interval <= 0`` (rpm<=0 in tests) disables pacing entirely.
    """

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(self._next, now)
            self._next = start + self._interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


def _translate_parallel(
    prepared: Dict[str, Any],
    pending: List[Segment],
    error_segments: List[str],
    *,
    max_workers: int,
) -> int:
    """Parallel translate across CHAPTER lanes; serial within a chapter.

    Segments of the same chapter are processed one at a time because
    ``_translate_one`` feeds the immediately-preceding committed translation
    back as context.  Different chapters are fully independent, so up to
    ``max_workers`` chapters translate concurrently.  Concurrency is bounded
    by ``min(max_workers, len(pending))`` threads in ONE process — no child
    processes are spawned, so nothing can become an orphaned worker holding a
    provider slot after the run ends.  Aggregate fire rate is globally paced
    by ``requests_per_minute`` (``_Pacer``); a worker stuck inside a bounded
    retry chain releases its slot automatically (error row under
    ``skip_failed_segments`` frees the lane for the next run).
    """
    if max_workers <= 1 or len(pending) <= 1:
        return _translate_serial(prepared, pending, error_segments)
    # Group pending by chapter, preserving source order within a chapter.
    lanes: Dict[str, List[tuple]] = {}
    for batch_index, segment in enumerate(pending, 1):
        chapter = prepared["chapter_of"].get(segment.id) or ""
        lanes.setdefault(chapter, []).append((batch_index, segment))
    pacer = _Pacer(prepared["delay"])
    lock = threading.Condition()
    ready: List[str] = []            # chapter keys whose head segment is free
    position: Dict[str, int] = {}    # chapter -> next pending index to dispatch
    for chapter in lanes:
        ready.append(chapter)
        position[chapter] = 0
    stop = threading.Event()
    translated = 0
    inflight = 0
    first_error: List[BaseException] = []

    def worker() -> None:
        nonlocal translated, inflight
        while not stop.is_set():
            with lock:
                while not ready and not stop.is_set():
                    if inflight == 0:
                        lock.notify_all()
                        return
                    lock.wait()
                if stop.is_set():
                    return
                chapter = ready.pop(0)
                idx = position[chapter]
                position[chapter] = idx + 1
                inflight += 1
            batch_index, segment = lanes[chapter][idx]
            try:
                pacer.wait()
                ok = _translate_item(
                    prepared, segment, batch_index=batch_index,
                    error_segments=error_segments,
                )
            except BaseException as exc:  # noqa: BLE001 - propagate first error
                with lock:
                    inflight -= 1
                    if not first_error:
                        first_error.append(exc)
                    stop.set()
                    lock.notify_all()
                return
            with lock:
                inflight -= 1
                if ok:
                    translated += 1
                if position[chapter] < len(lanes[chapter]):
                    ready.append(chapter)
                lock.notify_all()

    workers = [threading.Thread(target=worker, daemon=True)
               for _ in range(min(max_workers, len(lanes)))]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if first_error:
        raise first_error[0]
    return translated


def translate_v2(
    config: Dict[str, Any],
    client: Optional[ChatClient] = None,
    *,
    limit: Optional[int] = None,
    target_completed: Optional[int] = None,
    ladder: Optional[str] = None,
) -> Dict[str, Any]:
    """Contextual v2 translation with fine invalidation (P06 entry point).

    ``limit`` caps how many currently-pending segments to translate this run;
    ``target_completed`` is the absolute completed cap (resume-safe, v1
    semantics); ``ladder`` delegates to the smoke ladder (10/100/500/all).
    """
    if ladder is not None:
        from .ladder import run_ladder

        if client is None:
            client = OpenAICompatibleClient(dict(config.get("provider") or {}))
        return run_ladder(config, client, ladder)
    prepared = _prepare(config)
    if limit is not None and target_completed is not None:
        raise ValueError("Use either limit or target_completed, not both")
    pending = [segment for segment in prepared["segments"] if segment.translatable and not prepared["current"](segment)]
    if target_completed is not None:
        if target_completed < 0:
            raise ValueError("target_completed must be non-negative")
        current_completed = sum(
            segment.translatable and prepared["current"](segment)
            for segment in prepared["segments"]
        )
        pending = pending[:max(0, target_completed - current_completed)]
    elif limit is not None:
        pending = pending[:limit]
    if not pending:
        report = _v2_report(prepared, 0)
        _write_status_file(prepared)
        return report
    prepared["client"] = client or OpenAICompatibleClient(dict(config.get("provider") or {}))
    max_workers = int((config.get("translation") or {}).get("max_workers", 1))
    if max_workers < 1:
        raise ValueError("translation.max_workers must be >= 1")
    error_segments: List[str] = []
    if max_workers == 1 or len(pending) <= 1:
        translated_this_run = _translate_serial(prepared, pending, error_segments)
    else:
        translated_this_run = _translate_parallel(
            prepared, pending, error_segments, max_workers=max_workers,
        )
    report = _v2_report(prepared, translated_this_run)
    report["error_segment_count"] = len(error_segments)
    report["error_segment_ids"] = error_segments
    report["max_workers"] = max_workers
    _write_status_file(prepared)
    return report



def _record_failed_segment(prepared: Dict[str, Any], segment: Any, exc: BaseException) -> None:
    """Append an explicit status:'error' row (never a fabricated translation)
    when a segment exhausts its parse attempts under skip_failed_segments.
    Currency checks treat non-'completed' rows as pending, so the segment is
    retried on the next run."""
    from .records import record_row
    from datetime import datetime, timezone

    row = {
        "segment_id": segment.id,
        "source_hash": segment.source_hash,
        "translated_text": "",
        "model": str(prepared.get("client").model if prepared.get("client") else ""),
        "status": "error",
        "attempt": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "glossary_sha256": prepared.get("glossary_sha256") or "",
        "metadata": {
            "chapter_id": prepared["chapter_of"].get(segment.id),
            "context_mode": "contextual_v2",
            "prompt_version": prepared.get("prompt_version", 1),
            "error": f"{type(exc).__name__}: {exc}",
        },
    }
    with _APPEND_LOCK:
        append_jsonl(prepared["progress_path"], [row])

def _write_status_file(prepared: Dict[str, Any]) -> None:
    """v1-semantics status file (shared artifact; v2 rows are v1-current)."""
    config = prepared["config"]
    report = translation_status(config)
    write_json(prepared["output_dir"] / "translation_status.json", report)
