from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import resolve_path
from .io_utils import read_jsonl, write_json, write_jsonl


DEFAULT_MASTER = Path(__file__).resolve().parents[1] / "glossary_master.json"
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž][A-Za-zÀ-ÖØ-öø-ÿĀ-ž’'\-]{2,}")
PROPER = re.compile(r"\b(?:[A-ZÀ-ÖØ-ÞĀ-Ž][\wÀ-ÖØ-öø-ÿĀ-ž’'\-]+(?:\s+|$)){2,5}")
QUOTED = re.compile(r"[\"“‘]([^\"”’]{3,80})[\"”’]")
LEADING_HEADING = re.compile(
    r"^\s*(?:(?:part|chapter|section|book)\s+)?(?:\d+|[ivxlcdm]+)[.):-]?\s+",
    re.I,
)
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "into", "over", "under",
    "was", "were", "are", "has", "have", "had", "not", "but", "his", "her", "their",
    "its", "our", "who", "which", "when", "what", "where", "than", "then", "also", "only",
    "between", "through", "about", "after", "before", "these", "those", "there", "they", "them",
    "can", "could", "would", "should", "will", "may", "might", "one", "two", "first", "new",
}
GENERIC_HEADINGS = {
    "contents", "introduction", "conclusion", "notes", "bibliography", "references",
    "acknowledgements", "acknowledgments", "index", "preface", "foreword",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config.get("glossary_settings") or {})


def _path(config: Dict[str, Any], key: str, default_name: str) -> Path:
    settings = _settings(config)
    value = settings.get(key)
    if value:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (Path(config["_base_dir"]) / path).resolve()
    return resolve_path(config, "glossary").parent / default_name


def _master_path(config: Dict[str, Any]) -> Path:
    value = _settings(config).get("master_json")
    return Path(str(value)).expanduser().resolve() if value else DEFAULT_MASTER


def _normal(text: str) -> str:
    text = text.casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def _phrase_pattern(term: str) -> re.Pattern[str]:
    tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿĀ-ž]+(?:[’'][\wÀ-ÖØ-öø-ÿĀ-ž]+)?", term)
    body = r"(?:[\s\-–—,]+)".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.I)


def _matching_segments(term: str, segments: Sequence[Dict[str, Any]], case_sensitive: bool = False) -> List[str]:
    if not term.strip():
        return []
    pattern = _phrase_pattern(term)
    result = []
    for segment in segments:
        source = str(segment.get("source_text") or "")
        if case_sensitive:
            tokens = re.findall(r"[\wÀ-ÖØ-öø-ÿĀ-ž]+", term)
            body = r"(?:[\s\-–—,]+)".join(re.escape(token) for token in tokens)
            matched = re.search(rf"(?<!\w){body}(?!\w)", source)
        else:
            matched = pattern.search(source)
        if matched:
            result.append(str(segment.get("id") or ""))
    return result


def _master_variants(item: Dict[str, Any]) -> List[str]:
    term = str(item.get("term") or "").strip()
    variants = [term]
    if str(item.get("category") or "").casefold() in {"person", "人物"} and "," in term:
        surname, given = [part.strip() for part in term.split(",", 1)]
        variants.extend([f"{given} {surname}", surname])
    variants.extend(str(value).strip() for value in item.get("source_aliases", []) if str(value).strip())
    return list(dict.fromkeys(variants))


def _load_master(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Master glossary not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Master glossary must be a JSON list")
    return [dict(row) for row in value if isinstance(row, dict)]


def _master_subset(master: Sequence[Dict[str, Any]], segments: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], set[str]]:
    selected: List[Dict[str, Any]] = []
    covered: set[str] = set()
    for item in master:
        term = str(item.get("term") or "").strip()
        translation = str(item.get("translation") or "").strip()
        if len(_normal(term)) < 4 or not translation:
            continue
        category = str(item.get("category") or "").casefold()
        token_count = len(WORD.findall(term))
        proper_category = category in {"person", "place", "org", "organization", "人物", "地名", "机构"}
        single_proper = token_count == 1 and proper_category
        variants = _master_variants(item)
        hits = sorted({segment_id for variant in variants for segment_id in _matching_segments(variant, segments, case_sensitive=proper_category)})
        if not hits:
            continue
        # A historical master glossary contains many ordinary one-word entries.
        # They are useful in their source books but dangerously broad elsewhere.
        # Keep single-word concepts only when they are distinctive and repeated;
        # names remain case-sensitive, while multiword phrases use boundaries.
        if token_count == 1 and not single_proper and (len(term) < 9 or len(hits) < 2):
            continue
        normalized = _normal(term)
        if normalized in covered:
            continue
        row = dict(item)
        if category in {"person", "人物"} and "," in term:
            row["master_term"] = term
            row["term"] = variants[1]
            translated_parts = [part.strip() for part in translation.split(",", 1)]
            if len(translated_parts) == 2 and all(translated_parts):
                row["translation"] = f"{translated_parts[1]}·{translated_parts[0]}"
                normalized_alternatives = []
                for alternative in row.get("alternatives", []):
                    parts = [part.strip() for part in str(alternative).split(",", 1)]
                    normalized = f"{parts[1]}·{parts[0]}" if len(parts) == 2 and all(parts) else str(alternative)
                    if normalized and normalized != row["translation"]:
                        normalized_alternatives.append(normalized)
                row["alternatives"] = list(dict.fromkeys(normalized_alternatives))
        row["origin"] = "master"
        row["source_aliases"] = [variant for variant in variants if variant != row["term"]]
        row["matched_segments"] = hits[:20]
        selected.append(row)
        covered.add(normalized)
        covered.update(_normal(variant) for variant in variants if _normal(variant))
    selected.sort(key=lambda row: (_normal(str(row.get("term") or "")), str(row.get("translation") or "")))
    return selected, covered


def _candidate_sources(config: Dict[str, Any], segments: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    terms: Dict[str, Dict[str, Any]] = {}

    def add(term: str, reason: str, segment_id: str, context: str, weight: int = 1) -> None:
        cleaned = re.sub(r"\s+", " ", term).strip(" \t\n\r.,;:!?()[]{}")
        norm = _normal(cleaned)
        words = WORD.findall(cleaned)
        if len(norm) < 4 or len(words) > 8 or norm in GENERIC_HEADINGS:
            return
        item = terms.setdefault(norm, {"term": cleaned, "reasons": set(), "segment_ids": [], "contexts": [], "score": 0})
        if len(cleaned) < len(item["term"]):
            item["term"] = cleaned
        item["reasons"].add(reason)
        if segment_id and segment_id not in item["segment_ids"]:
            item["segment_ids"].append(segment_id)
        if context and context not in item["contexts"] and len(item["contexts"]) < 3:
            item["contexts"].append(context[:280])
        item["score"] += weight

    title = str(config.get("book_title") or "").strip()
    if title:
        for segment in segments:
            if _normal(title) in _normal(str(segment.get("source_text") or "")):
                add(title, "book_title_term", str(segment.get("id") or ""), str(segment.get("source_text") or ""), 100)
                break
    for seeded in _settings(config).get("seed_terms", []):
        term = str(seeded).strip()
        for segment in segments:
            source = str(segment.get("source_text") or "")
            if _normal(term) and _normal(term) in _normal(source):
                add(term, "manually_seeded_source_term", str(segment.get("id") or ""), source, 90)
                break

    ngrams: Counter[Tuple[str, ...]] = Counter()
    locations: Dict[Tuple[str, ...], List[Tuple[str, str]]] = defaultdict(list)
    for segment in segments:
        source = re.sub(r"\s+", " ", str(segment.get("source_text") or "")).strip()
        segment_id = str(segment.get("id") or "")
        if not source:
            continue
        if segment.get("kind") == "heading":
            heading = LEADING_HEADING.sub("", source)
            add(heading, "heading_phrase", segment_id, source, 60)
            for chunk in re.split(r"\s*[:;,]\s*|\s+and\s+", heading, flags=re.I):
                if 1 <= len(WORD.findall(chunk)) <= 5:
                    add(chunk, "heading_subphrase", segment_id, source, 35)
        for quoted in QUOTED.findall(source):
            if 1 <= len(WORD.findall(quoted)) <= 8:
                add(quoted, "quoted_phrase", segment_id, source, 20)
        for match in PROPER.finditer(source):
            add(match.group(0), "proper_name", segment_id, source, 15)
        tokens = [token.casefold() for token in WORD.findall(source)]
        for size in (2, 3):
            for index in range(len(tokens) - size + 1):
                phrase = tuple(tokens[index:index + size])
                if phrase[0] in STOPWORDS or phrase[-1] in STOPWORDS:
                    continue
                if all(token in STOPWORDS for token in phrase):
                    continue
                ngrams[phrase] += 1
                if len(locations[phrase]) < 3:
                    locations[phrase].append((segment_id, source))
        for token in WORD.findall(source):
            if ("-" in token or len(token) >= 12) and token.casefold() not in STOPWORDS:
                add(token, "distinctive_word", segment_id, source, 2)

    for phrase, count in ngrams.items():
        if count < 4:
            continue
        term = " ".join(phrase)
        for segment_id, context in locations[phrase]:
            add(term, "repeated_phrase", segment_id, context, min(30, count))
    return terms


def _candidates(config: Dict[str, Any], segments: Sequence[Dict[str, Any]], covered: set[str]) -> List[Dict[str, Any]]:
    cap = int(_settings(config).get("candidate_limit", 80))
    raw = _candidate_sources(config, segments)
    rows: List[Dict[str, Any]] = []
    for norm, item in raw.items():
        if norm in covered:
            continue
        item["reasons"] = sorted(item["reasons"])
        item["occurrences"] = len(item.pop("segment_ids"))
        item["sample_contexts"] = item.pop("contexts")
        item["normalized_term"] = norm
        item["candidate_id"] = "glo-" + sha256(norm.encode("utf-8")).hexdigest()[:12]
        item["status"] = "needs_review"
        rows.append(item)
    def tier(row: Dict[str, Any]) -> int:
        reasons = set(row["reasons"])
        if "book_title_term" in reasons:
            return 0
        if "manually_seeded_source_term" in reasons:
            return 0
        if reasons & {"heading_phrase", "heading_subphrase"}:
            return 1
        if "distinctive_word" in reasons and int(row["occurrences"]) >= 2:
            return 2
        if "quoted_phrase" in reasons:
            return 3
        if "proper_name" in reasons:
            return 4
        return 5
    rows.sort(key=lambda row: (tier(row), -int(row["score"]), -int(row["occurrences"]), str(row["normalized_term"])))
    return rows[:cap]


def _report(config: Dict[str, Any], manifest: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"# {config['book_title']} glossary report", "",
        f"- Status: `{manifest['status']}`",
        f"- Master entries matched: {manifest['master_entries']}",
        f"- Candidate terms: {manifest['candidate_terms']}",
        f"- Included book-specific terms: {manifest.get('included_terms', 0)}",
        f"- Rejected candidates: {manifest.get('rejected_terms', 0)}",
        f"- Unresolved candidates: {manifest.get('unresolved_terms', len(candidates))}", "",
        "Candidates are evidence packets, not approved terminology. Translation is blocked until every candidate is included or rejected.", "",
    ]
    for row in candidates[:30]:
        lines.append(f"- `{row['candidate_id']}` — {row['term']} ({', '.join(row['reasons'])}; score {row['score']})")
    return "\n".join(lines) + "\n"


def generate_glossary(config: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    settings = _settings(config)
    if not settings or settings.get("mode", "manual") == "manual":
        return {"status": "manual", "approved": True}
    segments_path = resolve_path(config, "output_dir") / "cleaned_segments.jsonl"
    if not segments_path.is_file():
        raise FileNotFoundError("Run clean before glossary generation")
    master_path = _master_path(config)
    glossary_path = resolve_path(config, "glossary")
    manifest_path = _path(config, "manifest", "glossary_manifest.json")
    candidates_path = _path(config, "candidates", "glossary_candidates.jsonl")
    decisions_template = _path(config, "decision_template", "glossary_decisions.template.jsonl")
    report_path = _path(config, "report", "GLOSSARY_REPORT.md")
    segments_sha = file_sha256(segments_path)
    master_sha = file_sha256(master_path)
    if not force and manifest_path.is_file() and glossary_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "approved" and old.get("segments_sha256") == segments_sha and old.get("master_sha256") == master_sha and old.get("glossary_sha256") == file_sha256(glossary_path):
            return old
    segments = list(read_jsonl(segments_path))
    master_entries, covered = _master_subset(_load_master(master_path), segments)
    existing_manual: List[Dict[str, Any]] = []
    if glossary_path.is_file():
        value = json.loads(glossary_path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict) and row.get("term") and row.get("translation") and row.get("origin") not in {"master", "book_specific"}:
                    kept = dict(row)
                    kept["origin"] = kept.get("origin") or "existing_manual"
                    existing_manual.append(kept)
                    covered.add(_normal(str(kept["term"])))
    candidates = _candidates(config, segments, covered)
    glossary = master_entries + existing_manual
    glossary.sort(key=lambda row: _normal(str(row.get("term") or "")))
    write_json(glossary_path, glossary)
    write_jsonl(candidates_path, candidates)
    write_jsonl(decisions_template, [
        {"candidate_id": row["candidate_id"], "decision": "needs_human", "translation": "", "category": "", "reason": "", "evidence": []}
        for row in candidates
    ])
    approved = not bool(candidates) or not bool(settings.get("require_approval", True))
    manifest = {
        "schema_version": 1, "book_id": config["book_id"],
        "status": "approved" if approved else "needs_review", "approved": approved,
        "segments_sha256": segments_sha, "master_path": str(master_path), "master_sha256": master_sha,
        "glossary_path": str(glossary_path), "glossary_sha256": file_sha256(glossary_path),
        "candidates_path": str(candidates_path), "decision_template": str(decisions_template),
        "master_entries": len(master_entries), "existing_manual_entries": len(existing_manual),
        "candidate_terms": len(candidates), "included_terms": 0, "rejected_terms": 0,
        "unresolved_terms": 0 if approved else len(candidates),
    }
    write_json(manifest_path, manifest)
    report_path.write_text(_report(config, manifest, candidates), encoding="utf-8")
    return manifest


def compile_glossary_decisions(config: Dict[str, Any], decisions_path: Path) -> Dict[str, Any]:
    manifest_path = _path(config, "manifest", "glossary_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("Run glossary generation before compiling decisions")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("segments_sha256") != file_sha256(resolve_path(config, "output_dir") / "cleaned_segments.jsonl"):
        raise RuntimeError("Glossary candidates are stale; regenerate them")
    if manifest.get("master_sha256") != file_sha256(_master_path(config)):
        raise RuntimeError("Master glossary changed; regenerate candidates")
    if manifest.get("glossary_sha256") != file_sha256(resolve_path(config, "glossary")):
        raise RuntimeError("Glossary changed after candidate generation; regenerate candidates")
    candidates = list(read_jsonl(Path(manifest["candidates_path"])))
    by_id = {row["candidate_id"]: row for row in candidates}
    decisions = list(read_jsonl(decisions_path.expanduser().resolve()))
    decision_ids = [str(row.get("candidate_id") or "") for row in decisions]
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(by_id):
        raise ValueError("Glossary decisions must cover every candidate exactly once")
    additions: List[Dict[str, Any]] = []
    rejected = 0
    unresolved = 0
    for decision in decisions:
        candidate = by_id[decision["candidate_id"]]
        action = str(decision.get("decision") or "needs_human")
        if action == "include":
            translation = str(decision.get("translation") or "").strip()
            evidence = decision.get("evidence") or []
            if not translation or not isinstance(evidence, list) or not evidence:
                raise ValueError(f"Included term requires translation and evidence: {decision['candidate_id']}")
            additions.append({
                "term": candidate["term"], "translation": translation,
                "category": str(decision.get("category") or "book_specific"),
                "alternatives": list(decision.get("alternatives") or []),
                "note": str(decision.get("note") or ""), "evidence": evidence,
                "origin": "book_specific", "candidate_id": decision["candidate_id"],
            })
        elif action == "reject":
            if not str(decision.get("reason") or "").strip():
                raise ValueError(f"Rejected term requires a reason: {decision['candidate_id']}")
            rejected += 1
        elif action == "needs_human":
            unresolved += 1
        else:
            raise ValueError(f"Unknown glossary decision: {action}")
    glossary_path = resolve_path(config, "glossary")
    base = [row for row in json.loads(glossary_path.read_text(encoding="utf-8")) if row.get("origin") != "book_specific"]
    merged = base + additions
    merged.sort(key=lambda row: _normal(str(row.get("term") or "")))
    write_json(_path(config, "additions", "glossary_additions.json"), additions)
    write_json(glossary_path, merged)
    manifest.update({
        "status": "approved" if unresolved == 0 else "needs_review", "approved": unresolved == 0,
        "glossary_sha256": file_sha256(glossary_path), "decisions_path": str(decisions_path.expanduser().resolve()),
        "decisions_sha256": file_sha256(decisions_path.expanduser().resolve()),
        "included_terms": len(additions), "rejected_terms": rejected, "unresolved_terms": unresolved,
    })
    write_json(manifest_path, manifest)
    _path(config, "report", "GLOSSARY_REPORT.md").write_text(_report(config, manifest, candidates), encoding="utf-8")
    return manifest


def require_ready_glossary(config: Dict[str, Any]) -> str:
    settings = _settings(config)
    glossary_path = resolve_path(config, "glossary") if config.get("glossary") else None
    if glossary_path and not glossary_path.is_file():
        raise FileNotFoundError(f"Glossary not found: {glossary_path}")
    if not settings or settings.get("mode", "manual") == "manual":
        return file_sha256(glossary_path) if glossary_path else ""
    manifest_path = _path(config, "manifest", "glossary_manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError("Glossary gate failed: run glossary generation and review")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if settings.get("require_approval", True) and manifest.get("status") != "approved":
        raise RuntimeError(f"Glossary gate failed: {manifest.get('unresolved_terms', '?')} unresolved candidate terms")
    checks = {
        "segments_sha256": resolve_path(config, "output_dir") / "cleaned_segments.jsonl",
        "master_sha256": _master_path(config), "glossary_sha256": glossary_path,
    }
    for key, path in checks.items():
        if path is None or not path.is_file() or manifest.get(key) != file_sha256(path):
            raise RuntimeError(f"Glossary gate failed: stale {key.replace('_sha256', '')}")
    return str(manifest["glossary_sha256"])


def glossary_cache_sha(config: Dict[str, Any]) -> str:
    settings = _settings(config)
    if not settings or not settings.get("bind_translation_cache", False):
        return ""
    return require_ready_glossary(config)
