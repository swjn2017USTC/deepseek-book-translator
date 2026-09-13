from __future__ import annotations

"""Deterministic evidence loader for the v2 structure resolver (plan §5.4).

Reads the chapter-side evidence pack from ``<structure_dir>/evidence/``:

* ``heading_candidates.jsonl`` — full candidate records (one JSON object/line);
* ``book_structure_evidence.json`` — book-level summary (metadata, toc_entries,
  numbering_summary, style_clusters, zones, repeated_headers, page_density);
* ``pdf_evidence.json`` — optional native-PDF evidence (missing for no-pdf
  holdouts; the reader tolerates its absence).

The reader never mutates evidence and is fully deterministic: the same
directory yields the same in-memory structures and the same ``evidence_sha256``
(sha256 over the concatenated bytes of ``heading_candidates.jsonl`` +
``book_structure_evidence.json``, the staleness contract of plan §3.2/§4.5).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EvidenceError(ValueError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise EvidenceError(f"Missing evidence file: {path}")
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"Evidence file must contain a JSON object: {path}")
    return value


class BookEvidence:
    """Loaded, immutable view of one book's evidence pack."""

    def __init__(self, structure_dir: Path):
        structure_dir = Path(structure_dir).expanduser().resolve()
        evidence_dir = structure_dir / "evidence"
        candidates_path = evidence_dir / "heading_candidates.jsonl"
        summary_path = evidence_dir / "book_structure_evidence.json"
        if not candidates_path.is_file():
            raise EvidenceError(f"Missing evidence pack: {candidates_path}")
        if not summary_path.is_file():
            raise EvidenceError(f"Missing evidence pack: {summary_path}")
        self.structure_dir = structure_dir
        self.evidence_dir = evidence_dir
        self.heading_candidates: List[Dict[str, Any]] = []
        with candidates_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceError(f"Invalid JSONL at {candidates_path}:{line_number}") from exc
                if not isinstance(row, dict) or not row.get("candidate_id"):
                    raise EvidenceError(f"Malformed candidate row at {candidates_path}:{line_number}")
                self.heading_candidates.append(row)
        self.summary = _read_json(summary_path)
        self.metadata: Dict[str, Any] = dict(self.summary.get("metadata") or {})
        self.book_id: str = str(self.metadata.get("book_id") or self.summary.get("book_id") or "")
        self.book_title: str = str(self.metadata.get("book_title") or "")
        self.page_count: int = int(self.metadata.get("page_count") or 0)
        self.candidate_count: int = int(self.metadata.get("candidate_count") or len(self.heading_candidates))
        self.toc_entries: List[Dict[str, Any]] = list(self.summary.get("toc_entries") or [])
        self.numbering_summary: Dict[str, Any] = dict(self.summary.get("numbering_summary") or {})
        self.style_clusters: List[Dict[str, Any]] = list(self.summary.get("style_clusters") or [])
        self.repeated_headers: List[Dict[str, Any]] = list(self.summary.get("repeated_headers") or [])
        self.page_density: float = float(self.summary.get("page_density") or 0.0)
        raw_zones = self.summary.get("zones") or {}
        self.zones: Dict[int, str] = {}
        if isinstance(raw_zones, dict):
            for page, zone in raw_zones.items():
                try:
                    self.zones[int(page)] = str(zone)
                except (TypeError, ValueError):
                    continue
        self.candidates_by_id: Dict[str, Dict[str, Any]] = {
            str(row["candidate_id"]): row for row in self.heading_candidates
        }
        if len(self.candidates_by_id) != len(self.heading_candidates):
            raise EvidenceError("Duplicate candidate_id in heading_candidates.jsonl")
        self.candidate_ids: List[str] = [str(row["candidate_id"]) for row in self.heading_candidates]
        self.evidence_sha256: str = self._sha256_of(candidates_path, summary_path)
        # Optional native-PDF evidence (absent for no-pdf books).
        self.pdf_evidence: Optional[Dict[str, Any]] = None
        pdf_path = evidence_dir / "pdf_evidence.json"
        if pdf_path.is_file():
            self.pdf_evidence = _read_json(pdf_path)

    @staticmethod
    def _sha256_of(candidates_path: Path, summary_path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(candidates_path.read_bytes())
        digest.update(summary_path.read_bytes())
        return digest.hexdigest()

    def zone_of(self, page: Optional[int]) -> str:
        if page is None:
            return "mainmatter"
        return self.zones.get(page, "mainmatter")

    def page_of(self, candidate: Dict[str, Any]) -> Optional[int]:
        page_json = candidate.get("page_json")
        if page_json is None:
            return None
        try:
            return int(page_json)
        except (TypeError, ValueError):
            return None

    def print_page_of(self, candidate: Dict[str, Any]) -> Optional[int]:
        value = candidate.get("page_print")
        if value is None:
            return self.page_of(candidate)
        try:
            return int(value)
        except (TypeError, ValueError):
            return self.page_of(candidate)

    def ordered_candidates(self) -> List[Dict[str, Any]]:
        """Candidates in reading order: print page (fallback page_json) then id.

        Deterministic for identical input (holdout no-pdf books carry
        ``page_print=null`` and sort on ``page_json`` instead)."""
        return sorted(
            self.heading_candidates,
            key=lambda row: (
                self.print_page_of(row) if self.print_page_of(row) is not None else 10**9,
                self.page_of(row) if self.page_of(row) is not None else 10**9,
                str(row["candidate_id"]),
            ),
        )
