# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


# PyInstaller exposes SPECPATH as the directory containing this spec.  The
# spec lives in <repo>/windows, so its parent is the repository root.
repo = Path(SPECPATH).resolve().parent
translation = repo / "structured-book-translation-pipeline"
chapter = repo / "chapter-structure-recovery-lab"

hiddenimports = collect_submodules("book_pipeline") + collect_submodules("chapter_recovery")
datas = [
    (str(translation / "schemas"), "schemas"),
    (str(translation / "pandoc_lua"), "pandoc_lua"),
]

a = Analysis(
    [str(repo / "deepseek_book_translator_gui.py")],
    pathex=[str(repo), str(translation), str(chapter)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="DeepSeekBookTranslator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
