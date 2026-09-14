#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
for package_root in (
    ROOT / "structured-book-translation-pipeline",
    ROOT / "chapter-structure-recovery-lab",
):
    value = str(package_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def main() -> int:
    from book_pipeline.gui import main as gui_main, self_test

    if "--offline-smoke" in sys.argv:
        from book_pipeline.offline_smoke import run_offline_smoke

        try:
            result = run_offline_smoke()
            exit_code = 0
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1
        report_path = os.environ.get("DEEPSEEK_OFFLINE_SMOKE_REPORT")
        if report_path:
            Path(report_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if getattr(sys, "stdout", None) is not None:
            print(json.dumps(result, ensure_ascii=False))
        return exit_code
    if "--self-test" in sys.argv:
        try:
            result = self_test()
            exit_code = 0
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            exit_code = 1
        report_path = os.environ.get("DEEPSEEK_SELF_TEST_REPORT")
        if report_path:
            Path(report_path).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if getattr(sys, "stdout", None) is not None:
            print(json.dumps(result, ensure_ascii=False))
        return exit_code
    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
