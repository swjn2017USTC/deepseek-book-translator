from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .paddleocr import load_pages, normalize_blocks
from .confirmed import load_confirmed_toc, merge_confirmed_toc
from .apply_decisions import apply_decisions as run_apply_decisions
from .decisions import compile_decision_files
from .evidence_fusion import run_evidence_pipeline
from .page_map import infer_page_map
from .recover import recover_structure
from .report import write_outputs
from .toc import find_toc_pages, parse_toc_entries
from .validate import validate_structure


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def run_analyze(config_path: Path, output_override: Optional[Path] = None) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    input_path = _resolve(base, config["input_json"])
    output_dir = output_override or _resolve(base, config["output_dir"])
    override_path = _resolve(base, config["overrides"]) if config.get("overrides") else None

    raw_pages = load_pages(input_path)
    pages = normalize_blocks(raw_pages)
    page_map = infer_page_map(pages)
    toc_pages = find_toc_pages(pages, config.get("toc_search_pages", [0, 30]))
    toc_entries = parse_toc_entries(pages, toc_pages)
    if config.get("confirmed_toc"):
        confirmed_path = _resolve(base, config["confirmed_toc"])
        toc_entries = merge_confirmed_toc(toc_entries, load_confirmed_toc(confirmed_path))
    nodes, text_candidates = recover_structure(
        pages, toc_entries, page_map, config, override_path
    )
    validation = validate_structure(nodes, toc_entries, config)
    write_outputs(
        output_dir,
        config,
        nodes,
        toc_entries,
        page_map,
        text_candidates,
        validation,
        pages=pages,
        raw_pages=raw_pages,
        input_path=input_path,
    )
    result = {
        "output_dir": str(output_dir),
        "pages": len(pages),
        "toc_pages": toc_pages,
        "toc_entries": len(toc_entries),
        "nodes": len([node for node in nodes if not node.ignored]),
        "review_candidates": len(
            [
                node
                for node in nodes
                if node.kind != "book"
                and not node.ignored
                and node.confidence < float(config.get("review_threshold", 0.9))
            ]
        )
        + len([item for item in text_candidates if not item.get("suppressed")]),
        "validation_ok": validation["ok"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_evidence(
    config_path: Path,
    output_override: Optional[Path] = None,
    pdf_override: Optional[Path] = None,
    emit_timestamp: bool = False,
) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    input_path = _resolve(base, config["input_json"])
    # Artifacts land in <output_dir>/evidence/ (write_evidence_artifacts
    # appends the evidence/ segment); the default parent is the config's
    # output_dir so artifacts end up at runs/<book>/evidence/.
    output_dir = output_override or _resolve(base, config["output_dir"])
    pdf_path: Optional[Path] = None
    if pdf_override is not None:
        pdf_path = pdf_override.expanduser().resolve()
    elif config.get("pdf_path"):
        pdf_path = _resolve(base, config["pdf_path"])
    result = run_evidence_pipeline(
        config, input_path, output_dir, pdf_path, config_base=base,
        emit_timestamp=emit_timestamp,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recover a book heading structure")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze one PaddleOCR JSON")
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    compile_decisions = subparsers.add_parser(
        "compile-decisions", help="validate review decisions and compile auditable overrides"
    )
    compile_decisions.add_argument("--review-packets", type=Path, required=True)
    compile_decisions.add_argument("--decisions", type=Path, required=True)
    compile_decisions.add_argument("--output", type=Path, required=True)
    compile_decisions.add_argument("--base-overrides", type=Path)
    evidence = subparsers.add_parser(
        "evidence", help="extract deterministic structure evidence (P02, additive)"
    )
    evidence.add_argument("--config", type=Path, required=True)
    evidence.add_argument("--output", type=Path)
    evidence.add_argument("--pdf", type=Path)
    evidence.add_argument("--timestamp", action="store_true")
    apply_decisions = subparsers.add_parser(
        "apply-decisions",
        help="apply strict v2 LLM/human structure decisions onto the "
        "deterministic base tree (P03, additive)",
    )
    apply_decisions.add_argument("--config", type=Path, required=True)
    apply_decisions.add_argument("--decisions", type=Path, required=True)
    apply_decisions.add_argument("--rescue", type=Path, required=True)
    apply_decisions.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        run_analyze(args.config, args.output)
    elif args.command == "compile-decisions":
        result = compile_decision_files(
            args.review_packets.expanduser().resolve(),
            args.decisions.expanduser().resolve(),
            args.output.expanduser().resolve(),
            args.base_overrides.expanduser().resolve() if args.base_overrides else None,
        )
        print(json.dumps({
            "output": str(args.output.expanduser().resolve()),
            "review_packets": result["review_packet_count"],
            "resolved": result["resolved_count"],
            "needs_human": result["unresolved_count"],
        }, ensure_ascii=False, indent=2))
    elif args.command == "evidence":
        run_evidence(args.config, args.output, args.pdf, args.timestamp)
    elif args.command == "apply-decisions":
        result = run_apply_decisions(
            args.config, args.decisions, args.rescue, args.output
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
