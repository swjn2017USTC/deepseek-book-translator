from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .clean import clean_book
from .config import load_config
from .migrate import audit_existing_translation
from .migration_preview import create_migration_preview
from .render import render_book
from .structure_resolver.orchestrator import (
    MAX_REPAIR_ROUNDS,
    load_v2_spec,
    run_structure_v2,
)
from .terminology.orchestrator import load_terminology_spec, run_terminology_v2
from .translate import translate_book, translation_status


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Clean and translate a book from an approved chapter structure")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("clean", "status", "audit-existing"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    preview = commands.add_parser("preview-existing")
    preview.add_argument("--config", type=Path, required=True)
    preview.add_argument("--decisions", type=Path)
    translate = commands.add_parser("translate")
    translate.add_argument("--config", type=Path, required=True)
    translate_limit = translate.add_mutually_exclusive_group()
    translate_limit.add_argument("--limit", type=int)
    translate_limit.add_argument("--target-completed", type=int)
    translate_v2 = commands.add_parser(
        "translate-v2",
        help="run the contextual v2 translator (spec §12): single-target calls "
        "with chapter briefs + bounded prev/next context and fine per-concept "
        "cache invalidation; requires translation.context_mode=contextual_v2 "
        "in the config (legacy books keep using 'translate')",
    )
    translate_v2.add_argument("--config", type=Path, required=True)
    translate_v2_limit = translate_v2.add_mutually_exclusive_group()
    translate_v2_limit.add_argument("--limit", type=int)
    translate_v2_limit.add_argument("--target-completed", type=int)
    translate_v2.add_argument(
        "--ladder",
        choices=("10", "100", "500", "all"),
        help="run the smoke ladder (spec §12.7) instead of a plain pass: "
        "provider-probe gated, BLOCKED record when the provider is down",
    )
    render = commands.add_parser("render")
    render.add_argument("--config", type=Path, required=True)
    render.add_argument("--allow-partial", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run_limit = run.add_mutually_exclusive_group()
    run_limit.add_argument("--limit", type=int)
    run_limit.add_argument("--target-completed", type=int)
    run.add_argument("--allow-partial", action="store_true")
    structure = commands.add_parser(
        "structure-v2",
        help="run the v2 LLM book structure resolver against a v2 target spec "
        "(amendment A2); decoupled from translation-project artifacts",
    )
    structure_source = structure.add_mutually_exclusive_group(required=True)
    structure_source.add_argument("--spec", type=Path)
    structure_source.add_argument(
        "--project",
        type=Path,
        help="optional project mode: translate an initialized project.json "
        "manifest into the same internal v2 target spec",
    )
    structure.add_argument("--window-size", type=int, default=40)
    structure.add_argument("--max-repair", type=int, default=MAX_REPAIR_ROUNDS)
    terminology = commands.add_parser(
        "terminology-v2",
        help="run the v2 terminology resolver (deterministic mine -> per-window "
        "termhood -> book-level consolidation) against a terminology spec; "
        "--offline runs the deterministic mine + metrics only (spec §11, D8)",
    )
    terminology.add_argument("--spec", type=Path, required=True)
    terminology.add_argument("--offline", action="store_true")
    terminology.add_argument(
        "--max-wire",
        type=int,
        help="override both per-role wire budgets (term_miner/term_consolidator) "
        "with a uniform cap N (defaults come from spec.wire_budget: 25/4)",
    )
    qa = commands.add_parser(
        "qa",
        help="run the QA cascade (spec §13): deterministic QA-1/2/3 invariants "
        "over current translations, then the probe-gated QA-4 critic over the "
        "flagged segments only; read-only (writes qa_report.json / "
        "qa_audit.jsonl / qa_usage.jsonl and extends translation_status.json)",
    )
    qa.add_argument("--config", type=Path, required=True)
    qa.add_argument(
        "--only",
        choices=("qa-1", "qa-2", "qa-3", "qa-4"),
        help="run a single stage (qa-4 reads flags persisted by a prior full run)",
    )
    qa.add_argument(
        "--limit",
        type=int,
        help="cap how many flagged segments the QA-4 critic judges this run",
    )
    retranslate = commands.add_parser(
        "qa-retranslate",
        help="rewrite persisted likely_error segments through the contextual v2 "
        "path, appending revision rows (the only QA writer); blocks without a "
        "live provider; --rollback is out of scope",
    )
    retranslate.add_argument("--config", type=Path, required=True)
    retranslate.add_argument("--segment-id", type=str)
    retranslate.add_argument("--limit", type=int)
    return root


def _project_to_spec(project_path: Path) -> dict:
    """Optional project mode (A2): translate a new-book project.json manifest
    into the same internal v2 target spec the ``--spec`` mode consumes.

    Not required for P03 verification; wired for initialized books so the two
    entry points share one code path.
    """
    path = project_path.expanduser().resolve()
    if path.is_dir():
        path = path / "project.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "book_id": manifest["book_id"],
        "chapter_config": manifest["chapter_config"],
        "chapter_root": manifest.get("chapter_root"),
        "structure_dir": manifest["structure_dir"],
        "work_dir": manifest.get("work_dir") or manifest["structure_dir"],
        "provider": {},
        "llm_roles": {},
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "structure-v2":
        if args.project is not None:
            spec = _project_to_spec(args.project)
        else:
            spec = load_v2_spec(args.spec)
        result = run_structure_v2(spec, window_size=args.window_size, max_repair=args.max_repair)
    elif args.command == "terminology-v2":
        spec = load_terminology_spec(args.spec)
        result = run_terminology_v2(spec, offline=args.offline, max_wire=args.max_wire)
    else:
        config = load_config(args.config)
        if args.command == "clean":
            result = clean_book(config)
        elif args.command == "audit-existing":
            result = audit_existing_translation(config)
        elif args.command == "preview-existing":
            result = create_migration_preview(config, args.decisions)
        elif args.command == "translate":
            result = translate_book(
                config,
                limit=args.limit,
                target_completed=args.target_completed,
            )
        elif args.command == "translate-v2":
            from .contextual import translate_v2

            result = translate_v2(
                config,
                limit=args.limit,
                target_completed=args.target_completed,
                ladder=args.ladder,
            )
        elif args.command == "render":
            result = render_book(config, args.allow_partial)
        elif args.command == "qa":
            from .qa import run_qa

            result = run_qa(config, only=args.only, limit=args.limit)
        elif args.command == "qa-retranslate":
            from .qa import retranslate_likely_errors

            result = retranslate_likely_errors(
                config,
                segment_id=args.segment_id,
                limit=args.limit,
            )
        elif args.command == "status":
            result = translation_status(config)
        else:
            clean_book(config)
            translate_book(
                config,
                limit=args.limit,
                target_completed=args.target_completed,
            )
            result = render_book(config, args.allow_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
