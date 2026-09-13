from __future__ import annotations

"""V2 structure resolver orchestrator (plan §5.5).

State machine::

    evidence_ready -> pass_a -> pass_b -> pass_c -> decisions_written
      -> apply_decisions -> validation
        -> structure_v2_ok | (repair <= max_repair) | needs_human_review
    any LLM/provider outage -> structure_llm_unavailable (fail closed)

Hard caps (§6): Pass A <= 1 call, Pass B/C <= 10 windows each, repair <= 2
rounds.  Every LLM request is appended to ``<work_dir>/structure_v2_usage.jsonl``
as ``{timestamp, role, model, kind, usage}``.  Artifacts are written under
``<structure_dir>/v2/``: ``structure_decisions.jsonl`` (meta line first),
``structure_rescue.jsonl``, ``status.json``, ``human_review_packets.jsonl``.
The chapter subprocess pattern matches ``workflow._run_chapter``::

    python3 -m chapter_recovery apply-decisions --config <chapter_config> \
      --decisions <structure>/v2/structure_decisions.jsonl \
      --rescue <structure>/v2/structure_rescue.jsonl

run with ``cwd`` = the chapter root (from the spec, else the sibling repo).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import _reject_embedded_secrets
from ..io_utils import append_jsonl, write_jsonl
from ..llm_client import OpenAICompatibleClient, resolve_model
from .evidence_reader import BookEvidence, EvidenceError
from .fallback import (
    NEEDS_HUMAN_REVIEW,
    STRUCTURE_LLM_UNAVAILABLE,
    STRUCTURE_V2_OK,
    write_status,
)
from .grammar import run_grammar_pass
from .local_window import (
    MAX_WINDOWS,
    StructureParseError,
    build_windows,
    validate_decision_records,
    window_candidate_ids,
)
from .prompts import (
    local_window_system_prompt,
    local_window_user_prompt,
    PROMPTS_VERSION,
)
from .repair import run_repair
from .rescue import RescueParseError, run_rescue_pass, window_block_ids
from .schemas import (
    DECISION_SCHEMA_VERSION,
    DecisionsMeta,
    StructureDecision,
    V2TargetSpec,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAPTER_ROOT = ROOT.parent / "chapter-structure-recovery-lab"
MAX_REPAIR_ROUNDS = 2
MAX_DECODE_RETRIES_PER_PASS = 3
# Plan §6: per-run ceiling 1 (Pass A) + 10 (Pass B) + 10 (Pass C)
# + 2 repair rounds x 10 = 41 requests; exceeding it fails closed.
LLM_REQUEST_BUDGET = 1 + MAX_WINDOWS * 2 + MAX_REPAIR_ROUNDS * MAX_WINDOWS

_GR = "grammar"
_LOCAL = "local"
_RESCUE = "rescue"
_REPAIR = "repair"


class ProviderUnavailable(RuntimeError):
    """Raised by the orchestrator for any outage it must fail closed on."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StructureResolver:
    """Runs the whole v2 flow for one book spec (amendment A2)."""

    def __init__(
        self,
        spec: Dict[str, Any],
        *,
        window_size: int = 40,
        max_repair: int = MAX_REPAIR_ROUNDS,
        client_factory: Optional[Callable[[str, str], Any]] = None,
        run_chapter: Optional[Callable[[Path, Sequence[str]], None]] = None,
    ):
        self.spec = dict(spec)
        _reject_embedded_secrets(self.spec)
        self.window_size = int(window_size)
        self.max_repair = max(min(int(max_repair), MAX_REPAIR_ROUNDS), 0)
        self._max_windows = max(1, min(int(self.spec.get("max_windows") or MAX_WINDOWS), MAX_WINDOWS))
        self._page_range = (
            [int(v) for v in self.spec["page_range"]]
            if self.spec.get("page_range") is not None
            else None
        )
        self.client_factory = client_factory or self._default_client_factory
        self._run_chapter = run_chapter or self._default_run_chapter
        self._evidence: Optional[BookEvidence] = None
        self._grammar: Any = None
        self._window_decisions: Dict[int, List[Dict[str, Any]]] = {}
        self._rescue_records: List[Dict[str, Any]] = []
        self._repair_round = 0
        self._llm_calls = 0
        # True wire-request meter: sums per-client wire_attempts (every actual
        # HTTP POST incl. client-internal retries) across all logical steps.
        self._wire_used = 0

    # ------------------------------------------------------------------ paths
    @property
    def book_id(self) -> str:
        return str(self.spec["book_id"])

    @property
    def structure_dir(self) -> Path:
        return Path(self.spec["structure_dir"]).expanduser().resolve()

    @property
    def work_dir(self) -> Path:
        return Path(self.spec["work_dir"]).expanduser().resolve()

    @property
    def chapter_config(self) -> Path:
        return Path(self.spec["chapter_config"]).expanduser().resolve()

    @property
    def chapter_root(self) -> Path:
        value = self.spec.get("chapter_root")
        if value:
            return Path(str(value)).expanduser().resolve()
        return DEFAULT_CHAPTER_ROOT

    @property
    def v2_dir(self) -> Path:
        return self.structure_dir / "v2"

    @property
    def decisions_path(self) -> Path:
        return self.v2_dir / "structure_decisions.jsonl"

    @property
    def rescue_path(self) -> Path:
        return self.v2_dir / "structure_rescue.jsonl"

    @property
    def validation_path(self) -> Path:
        return self.v2_dir / "validation.json"

    @property
    def status_path(self) -> Path:
        return self.v2_dir / "status.json"

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
        roles.setdefault("structure_global", "deepseek-v4-flash")
        roles.setdefault("structure_local", "deepseek-v4-flash")
        roles.setdefault("structure_review", "deepseek-v4-flash")
        return roles

    # -------------------------------------------------------------- machinery
    def _model(self, role: str) -> str:
        merged = dict(self.spec)
        merged["provider"] = self.provider()
        merged["llm_roles"] = self.llm_roles()
        return resolve_model(merged, role)

    def _default_client_factory(self, role: str, model: str) -> Any:
        provider = dict(self.provider())
        provider["model"] = model
        return OpenAICompatibleClient(provider)

    def _client(self, role: str) -> Any:
        model = self._model(role)
        return self.client_factory(role, model), model

    def _llm(
        self,
        role: str,
        kind: str,
        call: Callable[[Any, str], Any],
    ) -> Any:
        """Run one LLM step with usage logging and the global request cap.

        Budget per book (plan §6, fail closed when exceeded): Pass A <= 1,
        Pass B <= 10 windows, Pass C <= 10 windows, repair <= 2 rounds of <= 10
        windows -> 1 + 10 + 10 + 2*10 = 41 hard ceiling.
        """
        self._llm_calls += 1
        if self._llm_calls > LLM_REQUEST_BUDGET:
            raise ProviderUnavailable(f"LLM request budget exceeded ({self._llm_calls})")
        client, model = self._client(role)
        try:
            result = call(client, model)
        except BaseException as exc:
            # Account every attempted request (success or failure) so the usage
            # log doubles as an attempt meter for the global live budget.
            wire = int(getattr(client, "wire_attempts", 1) or 1)
            self._wire_used += wire
            if self._wire_used > LLM_REQUEST_BUDGET:
                raise ProviderUnavailable(
                    f"wire request budget exceeded ({self._wire_used} > {LLM_REQUEST_BUDGET})"
                ) from exc
            append_jsonl(
                self.work_dir / "structure_v2_usage.jsonl",
                [{"timestamp": _utc_now(), "role": role, "model": model, "kind": kind,
                  "status": "error", "wire_attempts": wire,
                  "error": f"{type(exc).__name__}: {exc}"}],
            )
            raise
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            value, usage = result
            wire = int(getattr(client, "wire_attempts", 1) or 1)
            self._wire_used += wire
            if self._wire_used > LLM_REQUEST_BUDGET:
                raise ProviderUnavailable(
                    f"wire request budget exceeded ({self._wire_used} > {LLM_REQUEST_BUDGET})"
                )
            now = _utc_now()
            append_jsonl(
                self.work_dir / "structure_v2_usage.jsonl",
                [{"timestamp": now, "role": role, "model": model, "kind": kind,
                  "status": "ok", "wire_attempts": wire, "usage": dict(usage)}],
            )
            return value
        return result

    def _retry_step(
        self,
        role: str,
        kind: str,
        call: Callable[[Any, str], Any],
        attempts: int = MAX_DECODE_RETRIES_PER_PASS,
    ) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return self._llm(role, kind, call)
            except StructureParseError:
                raise  # deterministic rejection (hallucinated ids, forbidden text)
            except RescueParseError:
                raise  # deterministic rejection (hallucinated block ids)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"{kind} step failed after {attempts} retries") from last_error

    def _default_run_chapter(self, chapter_root: Path, arguments: Sequence[str]) -> None:
        if not (chapter_root / "chapter_recovery").is_dir():
            raise FileNotFoundError(f"Chapter recovery project not found: {chapter_root}")
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(
            [sys.executable, "-m", "chapter_recovery", *arguments],
            cwd=chapter_root,
            env=environment,
            check=True,
        )

    def _apply_decisions(self) -> None:
        arguments = [
            "apply-decisions",
            "--config", str(self.chapter_config),
            "--decisions", str(self.decisions_path),
            "--rescue", str(self.rescue_path),
        ]
        self._run_chapter(self.chapter_root, arguments)

    def _read_validation(self) -> Dict[str, Any]:
        if not self.validation_path.is_file():
            raise RuntimeError(f"apply-decisions produced no validation file: {self.validation_path}")
        return json.loads(self.validation_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------- evidence
    def _load_evidence(self) -> BookEvidence:
        evidence = BookEvidence(self.structure_dir)
        if not evidence.heading_candidates:
            raise EvidenceError(f"No heading candidates in {self.structure_dir}")
        self._evidence = evidence
        return evidence

    # ---------------------------------------------------------------- passes
    def _pass_a(self, evidence: BookEvidence) -> None:
        role = "structure_global"

        def _grammar_call(client: Any, model: str) -> Any:
            return run_grammar_pass(evidence, client, model)

        self._grammar = self._retry_step(role, _GR, _grammar_call)

    def _pass_b(self, evidence: BookEvidence) -> None:
        role = "structure_local"
        known: List[str] = []
        for window in self._windows(evidence):
            local = [str(candidate["candidate_id"]) for candidate in window["candidates"]]
            record: List[Dict[str, Any]] = []

            def _decide_window(client: Any, model: str) -> Any:
                nonlocal record
                system = local_window_system_prompt(evidence.book_id)
                user = local_window_user_prompt(self._grammar, window)
                parsed, usage = client.complete_json(system, user, List[StructureDecision])
                record = [item.model_dump() for item in parsed]
                return record, usage

            self._retry_step(role, _LOCAL, _decide_window)
            self._window_decisions[window["index"]] = validate_decision_records(
                window, record, known
            )
            known = list(dict.fromkeys(known + local))

    def _windows(self, evidence: BookEvidence) -> List[Dict[str, Any]]:
        return build_windows(
            evidence,
            window_size=self.window_size,
            window_limit=self._max_windows,
            page_range=self._page_range,
        )

    def _pass_c(self, evidence: BookEvidence) -> None:
        role = "structure_local"
        for window in self._windows(evidence):
            if not window_block_ids(window):
                continue

            def _rescue_window(client: Any, model: str) -> Any:
                return run_rescue_pass(evidence, window, client, model)

            records = self._retry_step(role, _RESCUE, _rescue_window)
            self._rescue_records.extend(records)

    # -------------------------------------------------------------- artifacts
    def _meta(self, repair_total: int) -> Dict[str, Any]:
        return DecisionsMeta(
            schema_version=DECISION_SCHEMA_VERSION,
            book_id=self.book_id,
            evidence_sha256=self._evidence.evidence_sha256,
            decision_model=self._model("structure_local"),
            grammar_model=self._model("structure_global"),
            prompts_version=PROMPTS_VERSION,
            generated_by="book_pipeline.structure_resolver",
            generated_at=_utc_now(),
            repair_total=repair_total,
        ).model_dump()

    def _current_decisions(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index in sorted(self._window_decisions):
            rows.extend(self._window_decisions[index])
        return rows

    def _write_decisions(self, repair_total: int) -> int:
        self.v2_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"_meta": self._meta(repair_total)}] + self._current_decisions()
        write_jsonl(self.decisions_path, rows)
        return len(rows) - 1

    def _write_rescue(self) -> int:
        rows: List[Dict[str, Any]] = []
        for item in self._rescue_records:
            record = dict(item)
            record["rescue_model"] = self._model("structure_local")
            record["prompts_version"] = PROMPTS_VERSION
            rows.append(record)
        self.v2_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.rescue_path, rows)
        return len(rows)

    # ------------------------------------------------------------------ repair
    def _run_repair_round(self, evidence: BookEvidence, validation: Dict[str, Any]) -> None:
        role = "structure_review"
        round_number = self._repair_round + 1
        for window in self._windows(evidence):
            prior = self._window_decisions.get(window["index"], [])
            if not prior:
                continue

            def _revise(client: Any, model: str) -> Any:
                revised, usage = run_repair(
                    evidence, window, prior, validation, client, model, round_number,
                )
                return revised, usage

            revised = self._retry_step(role, _REPAIR, _revise)
            candidates = set(window_candidate_ids(window))
            for item in revised:
                candidate_id = str(item.get("candidate_id") or "")
                if candidate_id not in candidates:
                    raise ValueError(
                        f"repair returned decision for {candidate_id!r} outside the "
                        f"review window {window['index']}"
                    )
                parsed = StructureDecision(**item)
                self._window_decisions[window["index"]] = [
                    record for record in self._window_decisions.get(window["index"], [])
                    if record.get("candidate_id") != candidate_id
                ]
                data = parsed.model_dump()
                data["decision_model"] = self._model(role)
                data["repair_round"] = round_number
                data["prompts_version"] = PROMPTS_VERSION
                self._window_decisions.setdefault(window["index"], []).append(data)
        self._repair_round = round_number

    def _human_review_packets(self, validation: Dict[str, Any]) -> None:
        self.v2_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for window_index in sorted(self._window_decisions):
            rows.append({
                "schema_version": 1,
                "book_id": self.book_id,
                "window_index": window_index,
                "repair_round": self._repair_round,
                "repair_total": self._repair_round,
                "generated_at": _utc_now(),
                "validator_errors": validation.get("errors") or [],
                "validator_gate_failures": validation.get("gate_failures") or [],
                "candidates": [
                    {
                        "candidate_id": decision.get("candidate_id"),
                        "decision": decision.get("decision"),
                        "kind": decision.get("kind"),
                        "level": decision.get("level"),
                        "parent_candidate_id": decision.get("parent_candidate_id"),
                    }
                    for decision in self._window_decisions.get(window_index, [])
                ],
            })
        append_jsonl(self.v2_dir / "human_review_packets.jsonl", rows)

    # -------------------------------------------------------------------- run
    def run(self) -> Dict[str, Any]:
        evidence = self._load_evidence()
        try:
            self._pass_a(evidence)
            self._pass_b(evidence)
            self._pass_c(evidence)
        except (ProviderUnavailable, RuntimeError, ValueError, EvidenceError) as exc:
            return self._fail_closed(str(exc))
        decision_count = self._write_decisions(0)
        rescue_count = self._write_rescue()
        try:
            self._apply_decisions()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            return self._fail_closed(f"chapter apply-decisions failed: {exc}")
        validation = self._read_validation()
        if not validation.get("gate_failures") and not validation.get("errors"):
            payload = write_status(
                self.structure_dir,
                STRUCTURE_V2_OK,
                book_id=self.book_id,
                detail="structure decisions applied and v2 validation passed",
                decision_count=decision_count,
                rescue_count=rescue_count,
            )
            return payload
        while self._repair_round < self.max_repair:
            try:
                self._run_repair_round(evidence, validation)
                decision_count = self._write_decisions(self._repair_round)
                self._apply_decisions()
                validation = self._read_validation()
            except (ProviderUnavailable, RuntimeError, ValueError, EvidenceError) as exc:
                return self._fail_closed(str(exc))
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
                return self._fail_closed(f"chapter apply-decisions failed: {exc}")
            if not validation.get("gate_failures") and not validation.get("errors"):
                payload = write_status(
                    self.structure_dir,
                    STRUCTURE_V2_OK,
                    book_id=self.book_id,
                    detail=f"structure decisions applied after {self._repair_round} repair round(s)",
                    repair_total=self._repair_round,
                    decision_count=decision_count,
                    rescue_count=rescue_count,
                )
                return payload
        self._human_review_packets(validation)
        payload = write_status(
            self.structure_dir,
            NEEDS_HUMAN_REVIEW,
            book_id=self.book_id,
            detail=f"validation still failing after {self.max_repair} repair round(s); "
                   "human review packets written",
            repair_total=self._repair_round,
            decision_count=decision_count,
            rescue_count=rescue_count,
            error=json.dumps(validation.get("gate_failures") or validation.get("errors") or [], ensure_ascii=False)[:500],
        )
        return payload

    def _fail_closed(self, detail: str) -> Dict[str, Any]:
        try:
            decision_count = self._write_decisions(self._repair_round)
            rescue_count = self._write_rescue()
        except Exception:  # keep the sentinel honest even when writes fail
            decision_count = len(self._current_decisions())
            rescue_count = len(self._rescue_records)
        payload = write_status(
            self.structure_dir,
            STRUCTURE_LLM_UNAVAILABLE,
            book_id=self.book_id,
            detail="LLM/provider unavailable; deterministic legacy path remains "
                   "the runnable pipeline; v2 not claimed",
            repair_total=self._repair_round,
            decision_count=decision_count,
            rescue_count=rescue_count,
            error=detail,
        )
        return payload


def run_structure_v2(
    spec: Dict[str, Any],
    *,
    window_size: int = 40,
    max_repair: int = MAX_REPAIR_ROUNDS,
    client_factory: Optional[Callable[[str, str], Any]] = None,
    run_chapter: Optional[Callable[[Path, Sequence[str]], None]] = None,
) -> Dict[str, Any]:
    """Entry point shared by the CLI and integration tests."""
    return StructureResolver(
        spec,
        window_size=window_size,
        max_repair=max_repair,
        client_factory=client_factory,
        run_chapter=run_chapter,
    ).run()


def load_v2_spec(path: Any) -> Dict[str, Any]:
    """Load + validate a v2 target spec document (amendment A2)."""
    spec_path = Path(path).expanduser().resolve()
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v2 target spec must be a JSON object")
    parsed = V2TargetSpec(**raw)
    data = parsed.model_dump()
    data["_spec_path"] = str(spec_path)
    required = ("book_id", "chapter_config", "structure_dir", "work_dir")
    for key in required:
        if not data.get(key):
            raise ValueError(f"Missing required v2 target spec field: {key}")
    for key in ("chapter_config", "structure_dir"):
        path = Path(str(data[key])).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"v2 target spec path does not exist: {path}")
    return data
