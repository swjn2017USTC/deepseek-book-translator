from __future__ import annotations

"""Pass B: deterministic window building + per-window local decisions.

Window rule (plan §5.6): candidates are chunked into contiguous print-page
windows of width ``window_size`` (default 40, allowed 20-40).  Windows prefer
breaking at zone boundaries and at deterministic chapter-boundary pages (a
candidate page whose candidates start a chapter/part family or carry a
doc-title label), and a candidate is never split across windows.  Identical
evidence input yields identical windows (dedicated determinism test).

A window covers the half-open page interval ``[start_page, end_page)``;
``end_page - start_page <= window_size`` always holds.
"""

from typing import Any, Dict, List, Optional, Sequence

MIN_WINDOW_SIZE = 20
MAX_WINDOW_SIZE = 40
MAX_WINDOWS = 10


def _candidate_page(candidate: Dict[str, Any], order: Sequence[str]) -> Optional[int]:
    for key in order:
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


class WindowBuilder:
    """Deterministic non-overlapping window builder over candidate pages."""

    def __init__(
        self,
        evidence: Any,
        window_size: int = MAX_WINDOW_SIZE,
        order: Sequence[str] = ("page_print", "page_json"),
        window_limit: int = MAX_WINDOWS,
        page_range: Optional[Sequence[int]] = None,
    ):
        if window_size < MIN_WINDOW_SIZE or window_size > MAX_WINDOW_SIZE:
            raise ValueError(f"window_size must be within {MIN_WINDOW_SIZE}..{MAX_WINDOW_SIZE}")
        if window_limit < 1:
            raise ValueError("window_limit must be positive")
        self.window_size = window_size
        self.order = tuple(order)
        self.window_limit = window_limit
        self.page_range = (
            (int(page_range[0]), int(page_range[1]))
            if page_range is not None and len(page_range) == 2
            else None
        )
        self._evidence = evidence
        self._candidates = evidence.ordered_candidates()

    def build(self) -> List[Dict[str, Any]]:
        """Chunk all candidates into deterministic non-overlapping windows.

        ``page_range`` (optional, inclusive print-page bounds) restricts the
        windowed candidates deterministically — used for bounded live runs
        (e.g. plan §6 hegel print 100-180) so natural window count stays under
        the hard ``window_limit`` guard.
        """
        lo, hi = self.page_range or (None, None)
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for candidate in self._candidates:
            page = _candidate_page(candidate, self.order)
            if page is None:
                if lo is None:
                    page = -1  # unplaceable candidates ride in the first window
                else:
                    continue  # page_range active: unplaceable candidates excluded
            if lo is not None and (page < lo or page > hi):
                continue
            buckets.setdefault(page, []).append(candidate)
        pages = sorted(buckets)
        zones = self._evidence.zones
        boundaries = set(self._boundary_pages(buckets))
        windows: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []
        current_start: Optional[int] = None
        for page in pages:
            if not current:
                current_start = page
                current.extend(buckets[page])
                continue
            width_if_extended = page - (current_start or page) + 1
            break_here = self._should_break(page, pages, boundaries, zones, current_start)
            if width_if_extended > self.window_size or break_here:
                windows.append(self._emit(current, current_start, page - 1, len(windows)))
                current = [candidate for candidate in buckets[page]]
                current_start = page
            else:
                current.extend(buckets[page])
        if current:
            last_page = pages[-1] if pages else (current_start or 0)
            windows.append(self._emit(current, current_start, last_page, len(windows)))
        if len(windows) > self.window_limit:
            raise ValueError(f"evidence spans {len(windows)} windows; hard cap is {self.window_limit}")
        return windows

    def _should_break(
        self,
        page: int,
        pages: Sequence[int],
        boundaries: set,
        zones: Dict[int, str],
        current_start: Optional[int],
    ) -> bool:
        """Break at a chapter/zone boundary when the window is already wide."""
        if page in boundaries and page - (current_start or page) + 1 >= MIN_WINDOW_SIZE:
            return True
        if zones.get(page) != zones.get(page - 1):
            return True
        return False

    def _boundary_pages(self, buckets: Dict[int, List[Dict[str, Any]]]) -> List[int]:
        """Pages where a part/chapter-level family starts (deterministic)."""
        boundaries: List[int] = []
        for page, candidates in sorted(buckets.items()):
            labels = {str(candidate.get("paddle_label") or "") for candidate in candidates}
            if labels & {"doc_title", "title", "subtitle"}:
                boundaries.append(page)
                continue
            families = set()
            for candidate in candidates:
                numbering = candidate.get("numbering")
                if isinstance(numbering, dict):
                    families.add(str(numbering.get("family") or ""))
            if families & {"chapter", "part", "upper_roman"}:
                boundaries.append(page)
        return boundaries

    def _emit(
        self,
        candidates: List[Dict[str, Any]],
        start: Optional[int],
        end: Optional[int],
        index: int,
    ) -> Dict[str, Any]:
        ordered = sorted(
            candidates,
            key=lambda row: (
                _candidate_page(row, ("page_print", "page_json"))
                if _candidate_page(row, ("page_print", "page_json")) is not None
                else 10**9,
                str(row.get("candidate_id") or ""),
            ),
        )
        start_page = start if start is not None else -1
        end_page = end if end is not None else start_page
        return {
            "index": index,
            "book_id": self._evidence.book_id,
            "start_page": start_page,
            "end_page": end_page + 1,
            "candidates": ordered,
        }


def build_windows(
    evidence: Any,
    window_size: int = MAX_WINDOW_SIZE,
    window_limit: int = MAX_WINDOWS,
    page_range: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    return WindowBuilder(
        evidence, window_size=window_size, window_limit=window_limit, page_range=page_range
    ).build()


class StructureParseError(RuntimeError):
    """A window's decisions failed intake against the live candidate set.

    A RuntimeError (not ValueError) so the orchestrator fails closed with the
    specific intake message instead of retrying on a deterministic rejection
    (hallucinated candidate ids / forbidden text fields can never be fixed by
    another identical retry loop)."""


def window_candidate_ids(window: Dict[str, Any]) -> List[str]:
    return [str(candidate["candidate_id"]) for candidate in window["candidates"]]


def validate_decision_records(
    window: Dict[str, Any],
    decisions: Sequence[Any],
    known_parents: Sequence[str],
) -> List[Dict[str, Any]]:
    """Intake a window's parsed decisions against live candidates.

    Rejects (via ``StructureParseError``) hallucinated candidate ids, unknown
    parents, and free-text title keys.  A window candidate left undecided is
    not an error here — the deterministic base tree already covers it.
    """
    local_ids = set(window_candidate_ids(window))
    parent_ids = set(known_parents)
    errors: List[str] = []
    records: List[Dict[str, Any]] = []
    for decision in decisions:
        data = dict(decision)
        candidate_id = str(data.get("candidate_id") or "")
        if candidate_id not in local_ids:
            errors.append(f"candidate_id {candidate_id!r} not in window candidate set")
            continue
        for forbidden in ("title_source", "exact_text", "title", "source_text", "new_title", "text"):
            if data.get(forbidden) not in (None, ""):
                errors.append(
                    f"decision for {candidate_id} carries forbidden free-text field {forbidden!r}"
                )
                break
        parent = data.get("parent_candidate_id")
        if parent is not None and parent not in local_ids and parent not in parent_ids:
            errors.append(f"parent_candidate_id {parent!r} not in any candidate set")
        records.append(data)
    if errors:
        raise StructureParseError("; ".join(errors))
    return records
