from __future__ import annotations

"""Terminology resolver v2 orchestrator (spec §11, plan §2/§4/§5).

State machine::

    deterministic mining -> window assignment
      -> (live) per-window termhood (flash) with wire-cap accounting
      -> accepted-candidate summaries + master provenance hints
      -> book-level consolidation (pro)
      -> deterministic compiler (constraint clamp D4, polysemy/dup check,
         master provenance join) -> concept_glossary.json
    any provider outage / budget violation / deterministic rejection
      -> terminology_llm_unavailable (fail closed, artifacts kept)
    offline (--offline) -> deterministic mine + metrics only, zero LLM calls

Artifacts under ``<work_dir>/`` follow plan §4; every LLM request is appended
to ``terminology_v2_usage.jsonl`` as ``{timestamp, role, model, kind, status,
wire_attempts, usage}`` with per-role wire budgets from ``spec.wire_budget``
(defaults: term_miner 25, term_consolidator 4) counted via
``client.wire_attempts``.  ``term_mining_progress.json`` (keyed by window) makes
re-runs idempotent: completed windows and an ``ok`` consolidation are resumed,
never re-mined.
"""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import _reject_embedded_secrets
from ..glossary import _master_variants, _normal
from ..io_utils import append_jsonl, read_jsonl, write_json, write_jsonl
from ..llm_client import OpenAICompatibleClient, resolve_model
from ..provenance import file_sha256
from . import evaluate
from .consolidator import run_consolidation
from .fallback import (
    NEEDS_HUMAN_REVIEW,
    TERMINOLOGY_LLM_UNAVAILABLE,
    TERMINOLOGY_V2_OK,
    write_term_status,
)
from .miner import mine_candidates, select_queue
from .prompts import PROMPTS_VERSION
from .schemas import (
    ConceptConsolidation,
    ConceptRecord,
    GlossaryConstraint,
    PROPER_NAME_CATEGORIES,
    TermhoodJudgment,
    TermProgress,
)
from .termhood import run_termhood_window

ROLE_MINER = "term_miner"
ROLE_CONSOLIDATOR = "term_consolidator"
DEFAULT_WIRE_BUDGET = {ROLE_MINER: 25, ROLE_CONSOLIDATOR: 4}

WINDOW_CANDIDATE_LIMIT = 80
MAX_EVIDENCE_SEGMENTS = 12
MAX_CONTEXTS = 3

# Master glossary categories -> canonical concept categories (provenance hints).
MASTER_CATEGORY_MAP = {
    "concept": "concept",
    "term": "term",
    "person": "person",
    "人物": "person",
    "place": "place",
    "地名": "place",
    "org": "org",
    "organization": "org",
    "机构": "org",
    "event": "event",
    "work": "work",
    "school": "org",
    "哲学概念": "concept",
    "哲学": "concept",
    "哲学术语": "term",
    "逻辑学概念": "concept",
    "修辞学/哲学": "concept",
}
MIN_MASTER_NORM_LEN = 4


class ProviderUnavailable(RuntimeError):
    """Raised for any outage the resolver must fail closed on."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ helpers
def _node_title(node: Dict[str, Any]) -> str:
    return " ".join(str(node.get("title_source") or "").split())[:160]


def _join_norm(text: str) -> str:
    """Normalized surface with 2-letter words dropped (see _load_master)."""
    return " ".join(word for word in _normal(str(text)).split() if len(word) > 2)


def _zone_group(zone: str) -> str:
    zone = str(zone or "").casefold()
    if zone == "frontmatter":
        return "frontmatter"
    if zone == "mainmatter":
        return "mainmatter"
    return "backmatter"


def _load_structure_nodes(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    structure = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return {str(node["id"]): node for node in structure.get("nodes", []) if node.get("id")}


def build_windows(
    segments: Sequence[Dict[str, Any]],
    node_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Deterministic windowing in cleaned-segment document order.

    With structure nodes: a window starts at every *top-level* mainmatter unit
    (its parent is the book root / absent) plus one window per non-mainmatter
    zone group (frontmatter / backmatter+notes+bibliography+index) when such
    segments exist — chapter-level termhood windows (plan §11.3, D5 "few
    windows").  Without structure nodes the fallback is segment grouping: any
    mainmatter heading at node level <= 2 starts a chapter window, zone-group
    crossings start zone windows.
    """
    node_by_id = node_by_id or {}
    root_ids = {
        node_id for node_id, node in node_by_id.items()
        if str(node.get("kind") or "") == "book" or not node.get("parent_id")
    }

    def top_anchor(node_id: str) -> Optional[str]:
        current = node_by_id.get(node_id)
        if current is None:
            return None
        guard = 0
        while guard < 64:
            guard += 1
            parent_id = current.get("parent_id")
            if not parent_id or parent_id in root_ids or parent_id not in node_by_id:
                return str(current.get("id"))
            current = node_by_id[parent_id]
        return str(current.get("id"))

    def heading_boundary(segment: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Return (window key, title) when this heading opens a window."""
        if segment.get("kind") != "heading":
            return None
        zone_group = _zone_group(segment.get("zone"))
        node_id = str((segment.get("metadata") or {}).get("node_id") or "")
        node = node_by_id.get(node_id)
        if node is not None:
            if str(node.get("zone") or "").casefold() != "mainmatter":
                return (f"zone:{zone_group}", _node_title(node) or zone_group)
            anchor = top_anchor(node_id)
            if anchor is None:
                return None
            anchor_node = node_by_id.get(anchor) or node
            if str(anchor) == str(node_id) or _zone_group(anchor_node.get("zone")) == "mainmatter":
                return (f"node:{anchor}", _node_title(anchor_node) or zone_group)
            return None
        # Fallback: no structure nodes — chapter-like headings only.
        try:
            level = int(segment.get("level") or 0)
        except (TypeError, ValueError):
            level = 0
        if zone_group != "mainmatter":
            if zone_group == "backmatter":
                return (f"zone:{zone_group}", zone_group)
            return (f"zone:{zone_group}", zone_group)
        if level <= 2:
            text = " ".join(str(segment.get("source_text") or "").split())
            return (f"node:{segment.get('id')}", text[:160] or zone_group)
        return None

    windows: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    current_key: Optional[str] = None
    current_group = ""
    for segment in segments:
        zone_group = _zone_group(segment.get("zone"))
        key = None
        title = ""
        boundary = heading_boundary(segment)
        if boundary is not None:
            key, title = boundary
            group = key.split(":", 1)[0]
            if group == "node":
                # chapter window: its zone group is the anchor node's zone.
                anchor = node_by_id.get(key.split(":", 1)[1]) if node_by_id else None
                group = _zone_group(anchor.get("zone")) if anchor is not None else zone_group
            boundary_group = group
        else:
            # Body/other segments: stay in the current window unless the zone
            # group changed (mainmatter -> notes/bibliography/backmatter etc.).
            boundary_group = zone_group
        if key is not None and key != current_key:
            current_key = key
            current_group = boundary_group
        elif current_key is None or boundary_group != current_group:
            current_key = f"zone:{boundary_group}"
            current_group = boundary_group
            title = title or boundary_group
        if current_key not in index_by_key:
            index_by_key[current_key] = len(windows)
            windows.append({
                "window_index": len(windows),
                "key": current_key,
                "label": current_key.split(":", 1)[1],
                "title": title,
                "segment_ids": [],
            })
        windows[index_by_key[current_key]]["segment_ids"].append(str(segment.get("id") or ""))
    return windows


# --------------------------------------------------------------- orchestrator
class TerminologyResolver:
    """Runs the whole v2 terminology flow for one spec document."""

    def __init__(
        self,
        spec: Dict[str, Any],
        *,
        offline: bool = False,
        max_wire: Optional[int] = None,
        client_factory: Optional[Callable[[str, str], Any]] = None,
    ):
        self.spec = dict(spec)
        _reject_embedded_secrets(self.spec)
        self.offline = bool(offline)
        self.client_factory = client_factory or self._default_client_factory
        budgets = dict(self.spec.get("wire_budget") or {})
        if max_wire is not None and int(max_wire) > 0:
            budgets = {ROLE_MINER: int(max_wire), ROLE_CONSOLIDATOR: int(max_wire)}
        self._budgets = {
            ROLE_MINER: int(budgets.get(ROLE_MINER) or DEFAULT_WIRE_BUDGET[ROLE_MINER]),
            ROLE_CONSOLIDATOR: int(budgets.get(ROLE_CONSOLIDATOR) or DEFAULT_WIRE_BUDGET[ROLE_CONSOLIDATOR]),
        }
        self._wire_used: Dict[str, int] = {ROLE_MINER: 0, ROLE_CONSOLIDATOR: 0}
        self._wire_baselines: Dict[int, int] = {}
        self._segments: List[Dict[str, Any]] = []
        self._candidates: List[TermCandidate] = []
        self._candidate_rows: List[Dict[str, Any]] = []
        self._queue: List[Any] = []
        self._windows: List[Dict[str, Any]] = []
        self._window_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self._decisions: List[Dict[str, Any]] = []
        self._master_index: Dict[str, Dict[str, Any]] = {}
        self._master_sha256: Optional[str] = None
        self._master_path: Optional[Path] = None
        self._concepts: List[ConceptRecord] = []
        self._concept_rows: List[Dict[str, Any]] = []
        self._started = 0.0

    # ------------------------------------------------------------------ paths
    @property
    def book_id(self) -> str:
        return str(self.spec["book_id"])

    @property
    def work_dir(self) -> Path:
        return Path(self.spec["work_dir"]).expanduser().resolve()

    @property
    def cleaned_segments_path(self) -> Path:
        return Path(self.spec["cleaned_segments"]).expanduser().resolve()

    @property
    def structure_json_path(self) -> Optional[Path]:
        value = self.spec.get("structure_json")
        return Path(str(value)).expanduser().resolve() if value else None

    @property
    def mining(self) -> Dict[str, Any]:
        return dict(self.spec.get("mining") or {})

    @property
    def candidates_path(self) -> Path:
        return self.work_dir / "term_candidates_v2.jsonl"

    @property
    def termhood_path(self) -> Path:
        return self.work_dir / "termhood_decisions.jsonl"

    @property
    def concept_candidates_path(self) -> Path:
        return self.work_dir / "concept_candidates.jsonl"

    @property
    def glossary_path(self) -> Path:
        return self.work_dir / "concept_glossary.json"

    @property
    def decisions_path(self) -> Path:
        return self.work_dir / "concept_decisions.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.work_dir / "concept_glossary_manifest.json"

    @property
    def progress_path(self) -> Path:
        return self.work_dir / "term_mining_progress.json"

    @property
    def usage_path(self) -> Path:
        return self.work_dir / "terminology_v2_usage.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.work_dir / "terminology_metrics.json"

    # ------------------------------------------------------------------ config
    def provider(self) -> Dict[str, Any]:
        provider = dict(self.spec.get("provider") or {})
        provider.setdefault("api_url", "https://api.deepseek.com/chat/completions")
        provider.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        provider.setdefault("model", "deepseek-v4-flash")
        provider.setdefault("auth_header", "Authorization")
        provider.setdefault("auth_scheme", "Bearer")
        provider.setdefault("requests_per_minute", 20)
        provider.setdefault("timeout_seconds", 1200)
        provider.setdefault("retries", 8)
        provider.setdefault("max_tokens", 8192)
        provider.setdefault("temperature", 0.2)
        return provider

    def llm_roles(self) -> Dict[str, str]:
        roles = dict(self.spec.get("llm_roles") or {})
        roles.setdefault(ROLE_MINER, "deepseek-v4-flash")
        roles.setdefault(ROLE_CONSOLIDATOR, "deepseek-v4-flash")
        return roles

    def _model(self, role: str) -> str:
        merged = dict(self.spec)
        merged["provider"] = self.provider()
        merged["llm_roles"] = self.llm_roles()
        return resolve_model(merged, role)

    def _default_client_factory(self, role: str, model: str) -> Any:
        provider = dict(self.provider())
        provider["model"] = model
        return OpenAICompatibleClient(provider)

    def _client(self, role: str) -> Tuple[Any, str]:
        model = self._model(role)
        return self.client_factory(role, model), model

    # ------------------------------------------------------------- wire meter
    def _call_wire(self, client: Any) -> int:
        """True HTTP-POST attempts of this logical call.

        The default client factory builds a fresh client per call, so
        ``client.wire_attempts`` is already per-call; shared test doubles
        accumulate across calls, so a per-client baseline converts their
        counter into per-call deltas.
        """
        attempts = int(getattr(client, "wire_attempts", 1) or 1)
        key = id(client)
        previous = self._wire_baselines.get(key, 0)
        if attempts >= previous:
            delta = attempts - previous
        else:  # counter was reset underneath us
            delta = attempts
        self._wire_baselines[key] = attempts
        return max(delta, 1) if attempts else max(delta, 0)

    def _meter(self, role: str, model: str, kind: str, status: str, wire: int,
               usage: Optional[Dict[str, Any]] = None, error: str = "") -> None:
        self._wire_used[role] += max(int(wire or 0), 1)
        row: Dict[str, Any] = {
            "timestamp": _utc_now(),
            "role": role,
            "model": model,
            "kind": kind,
            "status": status,
            "wire_attempts": max(int(wire or 0), 1),
        }
        if usage is not None:
            row["usage"] = dict(usage)
        if error:
            row["error"] = error
        append_jsonl(self.usage_path, [row])
        budget = self._budgets.get(role, 0)
        if budget > 0 and self._wire_used[role] > budget:
            raise ProviderUnavailable(
                f"wire request budget exceeded for {role}: "
                f"{self._wire_used[role]} > {budget}"
            )

    def _llm(self, role: str, kind: str, call: Callable[[Any, str], Any]) -> Any:
        """Run one LLM step with usage logging and per-role wire caps."""
        client, model = self._client(role)
        try:
            result = call(client, model)
        except BaseException as exc:
            wire = self._call_wire(client)
            self._meter(role, model, kind, "error", wire, error=f"{type(exc).__name__}: {exc}")
            raise
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            value, usage = result
            wire = self._call_wire(client)
            self._meter(role, model, kind, "ok", wire, usage=dict(usage))
            return value
        return result

    # ------------------------------------------------------------------ stages
    def _load_segments(self) -> None:
        if not self.cleaned_segments_path.is_file():
            raise FileNotFoundError(f"cleaned segments not found: {self.cleaned_segments_path}")
        segments = list(read_jsonl(self.cleaned_segments_path))
        if not segments:
            raise RuntimeError(f"cleaned segments file is empty: {self.cleaned_segments_path}")
        self._segments = segments

    def _load_master(self) -> None:
        path_value = str(self.mining.get("master_json") or "").strip()
        if not path_value:
            return
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"master glossary not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("master glossary must be a JSON list")
        index: Dict[str, Dict[str, Any]] = {}
        join_index: Dict[str, Dict[str, Any]] = {}
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term") or "").strip()
            translation = str(entry.get("translation") or "").strip()
            if len(_normal(term)) < MIN_MASTER_NORM_LEN or not translation:
                continue
            for variant in _master_variants(entry):
                norm = _normal(variant)
                if len(norm) < MIN_MASTER_NORM_LEN:
                    continue
                index.setdefault(norm, entry)
                # v1's WORD tokenizer drops 2-letter tokens ("of"/"in"), so a
                # mined surface can read "rule law" for the master's "rule of
                # law".  Join-norm (2-letter words removed) bridges that gap
                # for provenance joins only — surfaces are never rewritten.
                join = _join_norm(variant)
                if len(join) >= MIN_MASTER_NORM_LEN:
                    join_index.setdefault(join, entry)
        self._master_index = index
        self._master_join_index = join_index
        self._master_sha256 = file_sha256(path)
        self._master_path = path

    def _master_match(self, surface: str) -> Optional[Dict[str, Any]]:
        if not self._master_index:
            return None
        norm = _normal(surface)
        if not norm:
            return None
        for candidate_norm in (norm,):
            entry = self._master_index.get(candidate_norm)
            if entry is not None and _normal(str(entry.get("term") or "")) == candidate_norm:
                return entry
        entry = self._master_index.get(norm)
        if entry is not None:
            return entry
        if self._master_join_index:
            join = _join_norm(surface)
            if len(join) >= MIN_MASTER_NORM_LEN:
                return self._master_join_index.get(join)
        return None

    def _master_hint(self, surface: str) -> Optional[Dict[str, Any]]:
        entry = self._master_match(surface)
        if entry is None:
            return None
        aliases = [
            str(alias).strip() for alias in _master_variants(entry)
            if str(alias).strip() and _normal(str(alias)) != _normal(surface)
            and _join_norm(str(alias)) != _join_norm(surface)
        ]
        category = MASTER_CATEGORY_MAP.get(str(entry.get("category") or "").casefold())
        return {
            "term": str(entry.get("term") or ""),
            "translation": str(entry.get("translation") or ""),
            "category": category,
            "aliases": list(dict.fromkeys(aliases)),
        }

    def _mine(self) -> None:
        mining_config = {
            "book_title": str(self.mining.get("book_title") or "").strip(),
            "seed_terms": list(self.mining.get("seed_terms") or []),
            "use_yake": bool(self.mining.get("use_yake", False)),
        }
        # Full discovery record (never capped) is persisted and used for
        # mining metrics + known-term recall. The termhood LLM queue is a
        # protected-source superset of this set (select_queue); default limit
        # 0 = full queue (per-window WINDOW_CANDIDATE_LIMIT + wire budgets cap
        # LLM cost, so a global queue cap is not required).
        self._candidates = mine_candidates(mining_config, self._segments)
        self._candidate_rows = [candidate.model_dump() for candidate in self._candidates]
        write_jsonl(self.candidates_path, self._candidate_rows)
        queue_limit = int(self.mining.get("candidate_limit") or 0)
        self._queue = select_queue(self._candidates, queue_limit)

    def _assign_candidates(self) -> None:
        """Primary window of each candidate = window holding most of its
        evidence segments; ties resolve to the earlier window."""
        window_of_segment: Dict[str, str] = {}
        for window in self._windows:
            for segment_id in window["segment_ids"]:
                window_of_segment.setdefault(segment_id, str(window["window_index"]))
        assigned: Dict[str, List[Dict[str, Any]]] = {}
        for candidate in self._queue or self._candidates:
            counts: Dict[str, int] = {}
            for segment_id in candidate.evidence_segment_ids:
                index = window_of_segment.get(segment_id)
                if index is not None:
                    counts[index] = counts.get(index, 0) + 1
            best = sorted(counts.items(), key=lambda pair: (-pair[1], int(pair[0])))
            primary = best[0][0] if best else str(self._windows[0]["window_index"])
            row = candidate.model_dump()
            row["primary_window"] = primary
            assigned.setdefault(primary, []).append(row)
        # Deterministic splits: windows with more than the LLM limit are cut
        # into "<index>-<k>" subwindows preserving global candidate order.
        self._window_candidates = {}
        for window in self._windows:
            base_key = str(window["window_index"])
            rows = assigned.get(base_key, [])
            for start in range(0, len(rows), WINDOW_CANDIDATE_LIMIT):
                chunk = rows[start:start + WINDOW_CANDIDATE_LIMIT]
                key = base_key if start == 0 else f"{base_key}-{start // WINDOW_CANDIDATE_LIMIT}"
                self._window_candidates[key] = chunk
        if not self._window_candidates:
            for window in self._windows:
                self._window_candidates.setdefault(str(window["window_index"]), [])

    def _window_payload(self, key: str) -> Dict[str, Any]:
        base = self._windows[int(key.split("-", 1)[0])]
        return {
            "book_id": self.book_id,
            "window_index": key,
            "label": base["label"],
            "title": base["title"],
            "segment_count": len(base["segment_ids"]),
            "candidates": self._window_candidates[key],
        }

    # ------------------------------------------------------------------ resume
    def _load_progress(self, candidates_sha: str) -> TermProgress:
        segments_sha = file_sha256(self.cleaned_segments_path)
        progress = TermProgress(
            book_id=self.book_id,
            windows={key: "pending" for key in self._window_candidates},
            consolidator_state="pending",
            segments_sha256=segments_sha,
            term_candidates_sha256=candidates_sha,
        )
        if not self.progress_path.is_file():
            return progress
        try:
            raw = json.loads(self.progress_path.read_text(encoding="utf-8"))
            old = TermProgress(**raw)
        except Exception:
            return progress
        if old.book_id != self.book_id or old.segments_sha256 != segments_sha \
                or old.term_candidates_sha256 != candidates_sha:
            return progress
        merged = dict(progress.windows)
        merged.update({key: state for key, state in old.windows.items() if key in merged})
        progress.windows = merged
        if old.consolidator_state in {"ok", "error", "skipped_offline"}:
            progress.consolidator_state = old.consolidator_state
        return progress

    def _write_progress(self, progress: TermProgress) -> None:
        write_json(self.progress_path, progress.model_dump())

    def _window_expected_ids(self, key: str) -> List[str]:
        return [row["candidate_id"] for row in self._window_candidates[key]]

    def _rows_for_window(self, key: str) -> List[Dict[str, Any]]:
        return [row for row in self._decisions if str(row.get("window_index")) == key]

    def _load_previous_decisions(self) -> None:
        if self.termhood_path.is_file():
            self._decisions = list(read_jsonl(self.termhood_path))

    def _rewrite_decisions(self) -> None:
        write_jsonl(self.termhood_path, self._decisions)

    # ---------------------------------------------------------------- termhood
    def _termhood_pass(self, progress: TermProgress) -> bool:
        """Judge every pending window.  Returns True on full success."""
        role = ROLE_MINER
        keys = sorted(self._window_candidates, key=lambda key: (int(key.split("-", 1)[0]), key))
        for key in keys:
            if progress.windows.get(key) == "termhood_ok":
                expected = set(self._window_expected_ids(key))
                rows = self._rows_for_window(key)
                if {str(row.get("candidate_id")) for row in rows} == expected:
                    continue  # resumed (empty windows resume with zero rows)
                progress.windows[key] = "pending"  # stale rows -> re-judge
            expected = set(self._window_expected_ids(key))
            if not expected:
                # No candidates assigned to this window: nothing to judge.
                progress.windows[key] = "termhood_ok"
                self._write_progress(progress)
                continue
            window = self._window_payload(key)

            def _decide(client: Any, model: str) -> Any:
                judgments = run_termhood_window(client, model, window)
                rows = self._validate_termhood(window, judgments)
                self._decisions = [row for row in self._decisions if str(row.get("window_index")) != key]
                self._decisions.extend(rows)
                return len(rows), {}

            try:
                self._llm(role, "termhood", _decide)
            except BaseException:
                progress.windows[key] = "termhood_error"
                self._write_progress(progress)
                raise
            progress.windows[key] = "termhood_ok"
            self._rewrite_decisions()
            self._write_progress(progress)
        return True

    def _validate_termhood(
        self,
        window: Dict[str, Any],
        judgments: Sequence[TermhoodJudgment],
    ) -> List[Dict[str, Any]]:
        expected = {row["candidate_id"] for row in window["candidates"]}
        seen_ids = [judgment.candidate_id for judgment in judgments]
        if len(seen_ids) != len(set(seen_ids)):
            raise ValueError(f"duplicate judgments in window {window['window_index']}")
        if set(seen_ids) != expected:
            missing = sorted(expected - set(seen_ids))
            invented = sorted(set(seen_ids) - expected)
            raise ValueError(
                f"termhood returned judgments outside window {window['window_index']}: "
                f"missing={missing} invented={invented}"
            )
        rows: List[Dict[str, Any]] = []
        by_candidate = {row["candidate_id"]: row for row in window["candidates"]}
        for judgment in judgments:
            verdict = (
                "reject" if judgment.is_term is False
                else ("accept" if judgment.confidence >= evaluate.HUMAN_CONFIDENCE_THRESHOLD
                      else "needs_human")
            )
            row = {
                "schema_version": 1,
                "book_id": self.book_id,
                "window_index": window["window_index"],
                "decision_model": self._model(ROLE_MINER),
                "prompts_version": PROMPTS_VERSION,
                "verdict": verdict,
                **judgment.model_dump(),
            }
            rows.append(row)
        # Deterministic order: candidate order inside the window.
        order = {candidate_id: index for index, candidate_id in enumerate(by_candidate)}
        rows.sort(key=lambda row: order[row["candidate_id"]])
        return rows

    # ------------------------------------------------------------ consolidation
    def _accepted_ids(self) -> List[str]:
        return [str(row.get("candidate_id")) for row in self._decisions
                if row.get("verdict") in {"accept", "needs_human"} and row.get("is_term")]

    def _summaries(self) -> List[Dict[str, Any]]:
        candidate_by_id = {row["candidate_id"]: row for row in self._candidate_rows}
        summaries: List[Dict[str, Any]] = []
        accepted = self._accepted_ids()
        for row in self._decisions:
            if row.get("verdict") not in {"accept", "needs_human"}:
                continue
            candidate = candidate_by_id.get(str(row.get("candidate_id")))
            if candidate is None:
                continue
            surface = str(row.get("canonical_surface") or candidate.get("surface") or "")
            summaries.append({
                "book_id": self.book_id,
                "candidate_id": row.get("candidate_id"),
                "surface": surface,
                "category": row.get("category"),
                "canonical_surface": row.get("canonical_surface"),
                "definition_in_book": row.get("definition_in_book"),
                "translation_sensitive": bool(row.get("translation_sensitive")),
                "possible_senses": int(row.get("possible_senses") or 1),
                "confidence": float(row.get("confidence") or 0.0),
                "contexts": list(row.get("contexts") or candidate.get("context") or [])[:MAX_CONTEXTS],
                "evidence_segment_ids": list(candidate.get("evidence_segment_ids") or [])[:MAX_EVIDENCE_SEGMENTS],
                "master": self._master_hint(surface) or self._master_hint(str(candidate.get("surface") or "")),
                "verdict": row.get("verdict"),
            })
        summaries.sort(key=lambda summary: accepted.index(summary["candidate_id"]))
        return summaries

    def _consolidation_pass(self, progress: TermProgress) -> bool:
        """Run the pro-model consolidation when accepted terms exist."""
        summaries = self._summaries()
        if not summaries:
            progress.consolidator_state = "ok"
            self._write_progress(progress)
            return True
        if progress.consolidator_state == "ok" and self.concept_candidates_path.is_file():
            rows = list(read_jsonl(self.concept_candidates_path))
            if rows:
                return True
            progress.consolidator_state = "pending"
        role = ROLE_CONSOLIDATOR

        def _consolidate(client: Any, model: str) -> Any:
            consolidation = run_consolidation(client, model, summaries)
            self._write_consolidation(consolidation)
            return len(consolidation.concepts), {}

        try:
            self._llm(role, "consolidation", _consolidate)
        except BaseException:
            progress.consolidator_state = "error"
            self._write_progress(progress)
            raise
        progress.consolidator_state = "ok"
        self._write_progress(progress)
        return True

    # --------------------------------------------------------------- compiler
    def _write_consolidation(self, consolidation: ConceptConsolidation) -> None:
        rows = [{
            "schema_version": 1,
            "book_id": self.book_id,
            "model": self._model(ROLE_CONSOLIDATOR),
            "prompts_version": PROMPTS_VERSION,
            "consolidation": consolidation.model_dump(),
        }]
        write_jsonl(self.concept_candidates_path, rows)

    def _load_consolidation(self) -> Optional[ConceptConsolidation]:
        if not self.concept_candidates_path.is_file():
            return None
        for row in read_jsonl(self.concept_candidates_path):
            if row.get("consolidation"):
                return ConceptConsolidation(**row["consolidation"])
        return None

    def _compile(self, consolidation: ConceptConsolidation) -> List[ConceptRecord]:
        """Deterministic compiler: alias merges, deterministic concept-id
        rewrite, provenance join, constraint clamp (D4), duplicate-concept
        hard fail.

        The consolidator LLM cannot compute sha256, so its draft concept ids
        are structural handles only: merge ops must reference a concept the
        model actually emitted (unknown targets are rejected as
        hallucinated), and every emitted concept id is rewritten to the
        deterministic ``concept-<sha256(_normal(canonical_source))[:12]>``
        before anything is written.
        """
        # Draft handles from the consolidator are structural only and NOT
        # reliable identity (live finding: one draft id reused for different
        # canonicals). Identity = normalized canonical_source:
        #  * same canonical under one draft id -> merge surfaces+senses (polysemy);
        #  * different canonicals under one draft id -> keep as SEPARATE drafts;
        #  * merges referencing an ambiguous (multi-canonical) draft id degrade
        #    to a recorded warning, never a guess.
        self._merge_ambiguity_warnings: List[str] = []
        entries: List[ConceptRecord] = []
        by_draft: Dict[str, List[ConceptRecord]] = {}
        for concept in consolidation.concepts:
            prior = next((e for e in entries
                          if _normal(e.canonical_source) == _normal(concept.canonical_source)), None)
            if prior is None:
                entries.append(concept)
                by_draft.setdefault(concept.concept_id, []).append(concept)
                continue
            # same canonical duplicate -> merge surfaces + distinct senses
            for surface in concept.surface_forms:
                if not any(_normal(s) == _normal(surface) for s in prior.surface_forms):
                    prior.surface_forms.append(surface)
            seen_senses = {_normal(s.translation) for s in prior.senses}
            for sense in concept.senses:
                if _normal(sense.translation) in seen_senses:
                    continue
                seen_senses.add(_normal(sense.translation))
                prior.senses.append(sense)
            if len(prior.senses) == 1:
                prior.senses[0].sense_id = "default"
            else:
                for index, sense in enumerate(prior.senses, start=1):
                    sense.sense_id = f"sense-{index}"
        for merge in consolidation.merges:
            candidates = by_draft.get(merge.concept_id) or []
            if not candidates:
                raise ValueError(
                    f"consolidator merge references unknown concept id {merge.concept_id!r} "
                    "(hallucinated merge target)"
                )
            if len(candidates) > 1:
                self._merge_ambiguity_warnings.append(
                    f"merge for ambiguous draft id {merge.concept_id!r} skipped "
                    f"({len(candidates)} distinct canonicals)"
                )
                continue
            target = candidates[0]
            norms = {_normal(surface) for surface in target.surface_forms}
            if _normal(merge.alias_surface) and _normal(merge.alias_surface) not in norms:
                target.surface_forms.append(merge.alias_surface)

        compiled: List[ConceptRecord] = []
        for concept in entries:
            concept.category = str(concept.category).casefold()
            # canonical_source must be listed among the surface forms
            canonical_norm = _normal(concept.canonical_source)
            if not any(_normal(surface) == canonical_norm for surface in concept.surface_forms):
                concept.surface_forms.insert(0, concept.canonical_source)
            concept.concept_id = self._concept_id(concept.canonical_source)
            self._apply_provenance(concept)
            self._apply_constraint_clamp(concept)
            compiled.append(concept)
        compiled.sort(key=lambda concept: (_normal(concept.canonical_source), concept.concept_id))
        errors = evaluate.duplicate_concept_errors(compiled)
        if errors:
            raise RuntimeError(
                "duplicate concept merge error: the same normalized surface maps to "
                f"multiple concepts: {json.dumps(errors[:5], ensure_ascii=False)}"
            )
        return compiled

    @staticmethod
    def _concept_id(canonical_source: str) -> str:
        return "concept-" + sha256(_normal(canonical_source).encode("utf-8")).hexdigest()[:12]

    def _apply_provenance(self, concept: ConceptRecord) -> None:
        entry = None
        for surface in [concept.canonical_source] + list(concept.surface_forms):
            entry = entry or self._master_match(str(surface))
        if entry is None:
            concept.master_source = None
            concept.master_aliases = []
            concept.master_sha256 = None
            return
        concept.master_source = str(entry.get("translation") or "")
        aliases = [
            str(alias).strip() for alias in _master_variants(entry)
            if str(alias).strip()
            and _normal(str(alias)) not in {_normal(concept.canonical_source)}
            and not any(_normal(str(alias)) == _normal(surface) for surface in concept.surface_forms)
        ]
        concept.master_aliases = list(dict.fromkeys(aliases))
        concept.master_sha256 = self._master_sha256

    def _apply_constraint_clamp(self, concept: ConceptRecord) -> None:
        """D4: ``hard`` survives only for proper-name categories or a master
        precedent with in-book evidence; otherwise downgrade to ``preferred``.
        ``sense_constrained`` requires >1 sense.  Downgrades are recorded in
        ``constraint_origin`` (llm|master|deterministic)."""
        category = str(concept.category).casefold()
        master_backed = bool(concept.master_source)
        downgraded = False
        for sense in concept.senses:
            constraint = sense.constraint
            if constraint == GlossaryConstraint.SENSE_CONSTRAINED and len(concept.senses) == 1:
                sense.constraint = GlossaryConstraint.PREFERRED
                downgraded = True
                continue
            if constraint != GlossaryConstraint.HARD:
                continue
            allowed_by_category = category in PROPER_NAME_CATEGORIES
            allowed_by_master = master_backed and bool(sense.evidence_segments)
            if allowed_by_category or allowed_by_master:
                continue
            sense.constraint = GlossaryConstraint.PREFERRED
            downgraded = True
        if downgraded:
            concept.constraint_origin = "deterministic"
        elif master_backed and any(
            sense.constraint == GlossaryConstraint.HARD for sense in concept.senses
        ):
            concept.constraint_origin = "master"
        else:
            concept.constraint_origin = "llm"

    # ------------------------------------------------------------------ metrics
    def _assemble_metrics(
        self,
        status: str,
        detail: str = "",
        offline: bool = False,
    ) -> Dict[str, Any]:
        mining = evaluate.compute_mining_metrics(self._candidates, self._segments)
        mining["yake_enabled"] = bool(self.mining.get("use_yake", False))
        mining["queue_size"] = len(self._queue)
        mining["candidate_limit"] = int(self.mining.get("candidate_limit") or 0)
        known_terms = list((self.spec.get("sampling") or {}).get("known_terms") or [])
        if known_terms:
            recall = evaluate.known_term_recall(
                self._candidates,
                self._accepted_ids() if not offline else None,
                known_terms,
                queue_ids=[c.candidate_id for c in self._queue],
            )
        else:
            recall = {"known_term_count": 0, "note": "no sampling.known_terms configured"}
        judged = [row for row in self._decisions if row.get("verdict")]
        termhood = evaluate.termhood_metrics(judged)
        if offline:
            termhood["note"] = "termhood not run (offline): deterministic mine + metrics only"
        concepts = self._concepts
        polysemy = evaluate.polysemy_check(concepts)
        if not concepts:
            constraints = {
                "note": "no concepts compiled yet (offline or no accepted terms): "
                        "constraint distribution empty until termhood/consolidation",
                "sense_count": 0,
                "constraint_distribution": {},
                "constraint_origin_distribution": {},
                "hard_sense_rate": 0.0,
            }
        else:
            constraints = evaluate.constraint_distribution(concepts)
        provenance = evaluate.master_provenance_rate(concepts)
        provenance["master_sha256"] = self._master_sha256
        provenance["master_path"] = str(self._master_path) if self._master_path else None
        return {
            "schema_version": 1,
            "book_id": self.book_id,
            "status": status,
            "offline": offline,
            "detail": detail,
            "generated_at": _utc_now(),
            "prompts_version": PROMPTS_VERSION,
            "mining": mining,
            "known_term_recall": recall,
            "termhood": termhood,
            "consolidation": polysemy,
            "constraints": constraints,
            "master_provenance": provenance,
            "wire": {
                ROLE_MINER: {"budget": self._budgets[ROLE_MINER], "used": self._wire_used[ROLE_MINER]},
                ROLE_CONSOLIDATOR: {"budget": self._budgets[ROLE_CONSOLIDATOR], "used": self._wire_used[ROLE_CONSOLIDATOR]},
            },
            "wall_seconds": round(time.monotonic() - self._started, 3),
        }

    # ------------------------------------------------------------------- files
    def _write_glossary(self, concepts: List[ConceptRecord]) -> str:
        payload = {
            "schema_version": 2,
            "book_id": self.book_id,
            "concepts": [concept.model_dump() for concept in concepts],
        }
        write_json(self.glossary_path, payload)
        return file_sha256(self.glossary_path)

    def _write_decisions_template(self, concepts: List[ConceptRecord]) -> None:
        rows = evaluate.needs_human_rows(concepts, self._needs_human_surfaces())
        write_jsonl(self.decisions_path, rows)

    def _needs_human_surfaces(self) -> List[str]:
        candidate_by_id = {row["candidate_id"]: row for row in self._candidate_rows}
        surfaces: List[str] = []
        for row in self._decisions:
            if row.get("verdict") != "needs_human":
                continue
            candidate = candidate_by_id.get(str(row.get("candidate_id")))
            surface = str(row.get("canonical_surface")
                          or (candidate or {}).get("surface") or "")
            if surface:
                surfaces.append(surface)
        return surfaces

    def _write_manifest(
        self,
        glossary_sha: Optional[str],
        status: str,
    ) -> None:
        counts = {
            "candidate_terms": len(self._candidates),
            "termhood_judged": len(self._decisions),
            "termhood_accepted": sum(
                1 for row in self._decisions if row.get("verdict") in {"accept", "needs_human"}
            ),
            "termhood_rejected": sum(1 for row in self._decisions if row.get("verdict") == "reject"),
            "needs_human_count": sum(1 for row in self._decisions if row.get("verdict") == "needs_human"),
            "concept_count": len(self._concepts),
            "sense_count": sum(len(concept.senses) for concept in self._concepts),
        }
        manifest = {
            "schema_version": 1,
            "book_id": self.book_id,
            "status": status,
            "offline": self.offline,
            "cleaned_segments_sha256": file_sha256(self.cleaned_segments_path),
            "cleaned_segments_path": str(self.cleaned_segments_path),
            "master_sha256": self._master_sha256,
            "master_path": str(self._master_path) if self._master_path else None,
            "term_candidates_sha256": file_sha256(self.candidates_path),
            "termhood_decisions_sha256": file_sha256(self.termhood_path) if self.termhood_path.is_file() else None,
            "concept_glossary_sha256": glossary_sha,
            "concept_glossary_path": str(self.glossary_path),
            "concept_decisions_path": str(self.decisions_path),
            "yake_enabled": bool(self.mining.get("use_yake", False)),
            "prompts_version": PROMPTS_VERSION,
            "generated_at": _utc_now(),
            "counts": counts,
        }
        write_json(self.manifest_path, manifest)

    def _counts(self) -> Dict[str, Any]:
        polysemy = evaluate.polysemy_check(self._concepts)
        return {
            "candidate_terms": len(self._candidates),
            "termhood_judged": len(self._decisions),
            "termhood_accepted": sum(
                1 for row in self._decisions if row.get("verdict") in {"accept", "needs_human"}
            ),
            "termhood_rejected": sum(1 for row in self._decisions if row.get("verdict") == "reject"),
            "needs_human_candidates": sum(
                1 for row in self._decisions if row.get("verdict") == "needs_human"
            ),
            "concept_count": len(self._concepts),
            "sense_count": sum(len(concept.senses) for concept in self._concepts),
            "duplicate_concept_merge_error": polysemy["duplicate_concept_merge_error"],
            "windows": len(self._windows),
            "window_units": len(self._window_candidates),
        }

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        self._started = time.monotonic()
        try:
            return self._run_inner()
        except (ProviderUnavailable, RuntimeError, ValueError, OSError) as exc:
            return self._fail_closed(f"{type(exc).__name__}: {exc}")

    def _run_inner(self) -> Dict[str, Any]:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._load_segments()
        self._load_master()
        self._mine()
        nodes = _load_structure_nodes(self.structure_json_path)
        self._windows = build_windows(self._segments, nodes)
        self._assign_candidates()
        progress = self._load_progress(file_sha256(self.candidates_path))
        self._write_progress(progress)
        self._load_previous_decisions()

        if self.offline:
            progress.consolidator_state = "skipped_offline"
            self._write_progress(progress)
            self._write_metrics_and_status(
                TERMINOLOGY_V2_OK,
                detail="deterministic mine + metrics only (--offline); "
                       "termhood/consolidation skipped; rerun without --offline to resume",
                offline=True,
            )
            return self._result_payload(TERMINOLOGY_V2_OK)

        self._termhood_pass(progress)
        self._consolidation_pass(progress)

        consolidation = self._load_consolidation()
        if consolidation is not None:
            self._concepts = self._compile(consolidation)
        glossary_sha = self._write_glossary(self._concepts)
        self._write_decisions_template(self._concepts)
        needs_human = bool(evaluate.needs_human_rows(self._concepts, self._needs_human_surfaces()))
        status = NEEDS_HUMAN_REVIEW if needs_human else TERMINOLOGY_V2_OK
        self._write_metrics_and_status(
            status,
            detail=(
                "terminology v2 complete: mined candidates judged per window and "
                "consolidated into a concept glossary"
                if status == TERMINOLOGY_V2_OK
                else "terminology v2 complete with gray-zone concepts; "
                     "concept_decisions.jsonl written for the human gate (§11.6)"
            ),
            offline=False,
        )
        self._write_manifest(glossary_sha, status)
        return self._result_payload(status)

    def _write_metrics_and_status(self, status: str, *, detail: str, offline: bool) -> None:
        metrics = self._assemble_metrics(status, detail=detail, offline=offline)
        write_json(self.metrics_path, metrics)
        write_term_status(
            self.work_dir,
            status,
            book_id=self.book_id,
            detail=detail,
            counts=self._counts(),
        )

    def _result_payload(self, status: str) -> Dict[str, Any]:
        payload = {
            "status": status,
            "offline": self.offline,
            "book_id": self.book_id,
            "work_dir": str(self.work_dir),
            "counts": self._counts(),
            "wire": {
                ROLE_MINER: {"budget": self._budgets[ROLE_MINER], "used": self._wire_used[ROLE_MINER]},
                ROLE_CONSOLIDATOR: {"budget": self._budgets[ROLE_CONSOLIDATOR], "used": self._wire_used[ROLE_CONSOLIDATOR]},
            },
            "wall_seconds": round(time.monotonic() - self._started, 3),
        }
        return payload

    def _fail_closed(self, detail: str) -> Dict[str, Any]:
        try:
            self._write_metrics_and_status(
                TERMINOLOGY_LLM_UNAVAILABLE,
                detail="LLM/provider unavailable or deterministic rejection; "
                       "artifacts already flushed are kept; no concept glossary "
                       "claimed as approved (blocked, not faked)",
                offline=self.offline,
            )
        except Exception:  # keep the sentinel honest even when writes fail
            write_term_status(
                self.work_dir,
                TERMINOLOGY_LLM_UNAVAILABLE,
                book_id=self.book_id,
                detail="fail-closed sentinel (metrics write failed)",
                counts=self._counts(),
                error=detail,
            )
        try:
            self._write_manifest(None, TERMINOLOGY_LLM_UNAVAILABLE)
        except Exception:
            pass
        payload = self._result_payload(TERMINOLOGY_LLM_UNAVAILABLE)
        payload["error"] = detail
        return payload


def run_terminology_v2(
    spec: Dict[str, Any],
    *,
    offline: bool = False,
    max_wire: Optional[int] = None,
    client_factory: Optional[Callable[[str, str], Any]] = None,
) -> Dict[str, Any]:
    """Entry point shared by the CLI and integration tests."""
    return TerminologyResolver(
        spec,
        offline=offline,
        max_wire=max_wire,
        client_factory=client_factory,
    ).run()


def load_terminology_spec(path: Any) -> Dict[str, Any]:
    """Load + validate a ``terminology-v2 --spec`` document (extra=forbid)."""
    from pydantic import ValidationError

    from .schemas import TerminologySpec

    spec_path = Path(path).expanduser().resolve()
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("terminology spec must be a JSON object")
    try:
        parsed = TerminologySpec(**raw)
    except ValidationError as exc:
        raise ValueError(f"invalid terminology spec: {exc}") from exc
    data = parsed.model_dump()
    data["_spec_path"] = str(spec_path)
    if not Path(str(data["cleaned_segments"])).expanduser().is_file():
        raise FileNotFoundError(
            f"terminology spec cleaned_segments path does not exist: {data['cleaned_segments']}"
        )
    structure = data.get("structure_json")
    if structure and not Path(str(structure)).expanduser().is_file():
        raise FileNotFoundError(
            f"terminology spec structure_json path does not exist: {structure}"
        )
    mining = dict(data.get("mining") or {})
    master = str(mining.get("master_json") or "").strip()
    if master and not Path(master).expanduser().is_file():
        raise FileNotFoundError(f"terminology spec mining.master_json path does not exist: {master}")
    return data
