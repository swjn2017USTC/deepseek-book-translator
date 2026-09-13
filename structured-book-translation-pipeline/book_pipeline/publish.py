from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import glob
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Optional
import zipfile

from .authenticity import reject_demo_translations
from .clean import require_structure_gate
from .config import resolve_path
from .io_utils import read_jsonl, write_json
from .provenance import require_fresh_cleaning
from .translate import translation_status


ALLOWED_COVER_RIGHTS = {
    "generated",
    "public_domain",
    "licensed",
    "official_product_cover",
    "user_provided",
    "user_approved_personal_use",
}
IMAGE_MAGIC = {
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
}

UNICODE_MATH = {
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
    "ε": r"$\varepsilon$", "θ": r"$\theta$", "λ": r"$\lambda$", "μ": r"$\mu$",
    "π": r"$\pi$", "σ": r"$\sigma$", "φ": r"$\varphi$", "ω": r"$\omega$",
    "Γ": r"$\Gamma$", "Δ": r"$\Delta$", "Σ": r"$\Sigma$", "Ω": r"$\Omega$",
    "∀": r"$\forall$", "∃": r"$\exists$", "∈": r"$\in$", "∉": r"$\notin$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$", "≈": r"$\approx$",
    "∞": r"$\infty$", "→": r"$\rightarrow$", "⇒": r"$\Rightarrow$",
    "¬": r"$\neg$", "∧": r"$\wedge$", "∨": r"$\vee$", "⊢": r"$\vdash$",
    "×": r"$\times$", "÷": r"$\div$", "±": r"$\pm$",
    "√": r"$\sqrt{}$", "∑": r"$\sum$", "∏": r"$\prod$", "∫": r"$\int$",
    "∂": r"$\partial$", "′": r"$\prime$", "″": r"$\prime\prime$",
}
SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

# TeX heading -> PDF TOC/bookmark depth (\chapter = 0 ... \subparagraph = 5).
TEX_TOC_DEPTH = {
    "chapter": 0,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "subparagraph": 5,
}

# Additive publish{} settings (plan decision 5). Every call site reads its
# knobs only through _publish_settings, so missing publish{} sections fall
# back to these defaults and no config file has to change.
PUBLISH_SETTINGS_DEFAULTS: Dict[str, Any] = {
    "lua_filters": False,
    "toc_depth": 4,
    "epub_chapter_level": 2,
    "pdf": {"tocdepth": 3},
    "epubcheck": {"enabled": True, "require": False, "path": ""},
}


def _publish_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the additive publish{} block onto the documented defaults."""
    settings = json.loads(json.dumps(PUBLISH_SETTINGS_DEFAULTS))
    publish = config.get("publish")
    if not isinstance(publish, dict):
        return settings
    for key in ("lua_filters", "toc_depth", "epub_chapter_level"):
        if key in publish:
            settings[key] = publish[key]
    pdf = publish.get("pdf")
    if isinstance(pdf, dict) and "tocdepth" in pdf:
        settings["pdf"]["tocdepth"] = pdf["tocdepth"]
    epubcheck = publish.get("epubcheck")
    if isinstance(epubcheck, dict):
        for key in ("enabled", "require", "path"):
            if key in epubcheck:
                settings["epubcheck"][key] = epubcheck[key]
    settings["lua_filters"] = bool(settings["lua_filters"])
    settings["toc_depth"] = int(settings["toc_depth"])
    settings["epub_chapter_level"] = int(settings["epub_chapter_level"])
    settings["pdf"]["tocdepth"] = int(settings["pdf"]["tocdepth"])
    settings["epubcheck"]["enabled"] = bool(settings["epubcheck"]["enabled"])
    settings["epubcheck"]["require"] = bool(settings["epubcheck"]["require"])
    settings["epubcheck"]["path"] = str(settings["epubcheck"]["path"])
    return settings


def _filters_dir() -> Path:
    """Repo-root pandoc_lua/ asset bundle (no wheel packaging this phase)."""
    return Path(__file__).resolve().parents[1] / "pandoc_lua"


def _required_toc_depth(prepared_markdown: str) -> int:
    """Deepest TeX TOC/bookmark depth the prepared structure needs.

    Depth model: \chapter=0 ... \subparagraph=5. A promoted H1 heading
    (chapter) needs depth 0, H2 needs depth 1, etc. Empty or title-only
    documents clamp to 0 because there is nothing below a chapter to reach.
    """
    levels = {
        len(match.group(1))
        for match in re.finditer(r"^(#{1,6})\s", prepared_markdown, re.M)
    }
    return max(levels, default=1) - 1


def _verify_toc_depth(
    toc_text: str, *, required_depth: int, heading_count: int
) -> Dict[str, Any]:
    """Check B (plan decision 2): \contentsline depth + coverage in book.toc."""
    entries = re.findall(
        r"^\\contentsline\s*\{(chapter|section|subsection|subsubsection|paragraph|subparagraph)\}",
        toc_text,
        re.M,
    )
    observed = max((TEX_TOC_DEPTH[name] for name in entries), default=-1)
    if observed < required_depth:
        raise RuntimeError(
            "PDF TOC depth does not reach the deepest structure level: "
            f"need >= {required_depth}, found {observed}"
        )
    if len(entries) < max(0, heading_count - 1):
        raise RuntimeError(
            "PDF TOC is missing headings: need at least "
            f"{max(0, heading_count - 1)} entries, found {len(entries)}"
        )
    return {"toc_depth": observed, "toc_entries": len(entries)}


def _verify_bookmark_depth(
    bookmark_text: str, *, required_depth: int
) -> Dict[str, Any]:
    """Check C (plan decision 2): \BOOKMARK depth + contiguity in book.out."""
    levels = [
        int(match.group(1))
        for match in re.finditer(r"^\\BOOKMARK\s*\[(\d+)\]\[-\]", bookmark_text, re.M)
    ]
    observed = max(levels, default=-1)
    if observed < required_depth:
        raise RuntimeError(
            "PDF bookmark depth does not reach the deepest structure level: "
            f"need >= {required_depth}, found {observed}"
        )
    for previous, current in zip(levels, levels[1:]):
        if current > previous + 1:
            raise RuntimeError(
                "PDF bookmark tree skips levels: level jumps from "
                f"{previous} to {current}"
            )
    return {"bookmark_max_depth": observed, "bookmark_entries": len(levels)}


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cover_image(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix not in IMAGE_MAGIC:
        raise ValueError("Cover must be a JPEG or PNG file")
    header = path.read_bytes()[:16]
    if not any(header.startswith(signature) for signature in IMAGE_MAGIC[suffix]):
        raise ValueError("Cover extension does not match its file content")
    return suffix


def register_cover(
    project_dir: Path,
    image_path: Path,
    *,
    rights_status: str,
    source_page_url: str = "",
    source_image_url: str = "",
    license_name: str = "",
    selected_by: str = "user",
    edition_evidence: str = "",
) -> Dict[str, Any]:
    """Copy an approved cover into a project and create a provenance record."""
    project = project_dir.expanduser().resolve()
    image = image_path.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    if rights_status not in ALLOWED_COVER_RIGHTS:
        raise ValueError(f"Unsupported rights_status: {rights_status}")
    if rights_status == "official_product_cover" and not source_page_url:
        raise ValueError("An official product cover requires its official source page URL")
    if rights_status in {"licensed", "public_domain"} and not license_name:
        raise ValueError("Licensed/public-domain covers require a license or rights statement")
    suffix = _validate_cover_image(image)
    digest = _hash(image)
    cover_dir = project / "cover"
    cover_dir.mkdir(parents=True, exist_ok=True)
    destination = cover_dir / f"cover-{digest[:12]}{suffix}"
    if not destination.exists():
        shutil.copy2(image, destination)
    manifest = {
        "schema_version": 1,
        "image_path": str(destination),
        "sha256": digest,
        "rights_status": rights_status,
        "source_page_url": source_page_url,
        "source_image_url": source_image_url,
        "license": license_name,
        "selected_by": selected_by,
        "edition_evidence": edition_evidence,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(cover_dir / "cover.json", manifest)
    return manifest


def load_registered_cover(project_dir: Path) -> Dict[str, Any]:
    manifest_path = project_dir.expanduser().resolve() / "cover" / "cover.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "EPUB export requires cover/cover.json; run generate-cover or set-cover"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image = Path(str(manifest.get("image_path") or "")).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Registered cover image is missing: {image}")
    _validate_cover_image(image)
    if manifest.get("rights_status") not in ALLOWED_COVER_RIGHTS:
        raise RuntimeError("Registered cover has no accepted rights/provenance status")
    if manifest.get("sha256") != _hash(image):
        raise RuntimeError("Registered cover changed after approval")
    return {**manifest, "image_path": str(image)}


def _pandoc() -> str:
    executable = shutil.which("pandoc")
    if not executable:
        raise RuntimeError("pandoc is required for PDF/EPUB export")
    return executable


def _xelatex() -> str:
    executable = shutil.which("xelatex")
    if not executable:
        raise RuntimeError("xelatex is required for TeX -> PDF compilation")
    return executable


def _run(command: List[str], cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(f"Publication conversion failed ({completed.returncode}): {detail}")


def _ensure_heading_blank_lines(text: str) -> str:
    lines = text.splitlines()
    output: List[str] = []
    for index, line in enumerate(lines):
        if index > 0 and re.match(r"^#{1,6}\s", line):
            previous = lines[index - 1].strip()
            if previous and not previous.startswith("<!--"):
                output.append("")
        output.append(line)
    return "\n".join(output)


def _strip_rendered_title_page(text: str, title: str, author: str) -> str:
    """Remove render.py's H1/author because book.tex owns the title page."""
    lines = text.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is not None and lines[first].strip() == f"# {title}".strip():
        del lines[first]
        while first < len(lines) and not lines[first].strip():
            del lines[first]
        if author and first < len(lines) and lines[first].strip() == author.strip():
            del lines[first]
            while first < len(lines) and not lines[first].strip():
                del lines[first]
    return "\n".join(lines)


def _promote_headings_markdown(text: str, title: str, author: str) -> str:
    """Heading-stage Markdown preparation shared by both TeX paths.

    Drop render.py's title page (the removed H1 was a wrapper; book.tex owns
    the title page), then promote all remaining headings so H2 chapters
    become TeX chapters and H3 sections become TeX sections. The Lua-filter
    path performs the same two transforms in pandoc_lua/ so the depth
    measurement is path-independent.
    """
    text = _strip_rendered_title_page(text, title, author)
    return re.sub(
        r"^(#{2,6})(\s+)",
        lambda match: match.group(1)[1:] + match.group(2),
        text,
        flags=re.MULTILINE,
    )


def _escape_table_cell_list_markers(text: str) -> str:
    """Escape Markdown list markers that start a raw-HTML table cell.

    PaddleOCR tables are emitted as raw HTML, and Pandoc still parses Markdown
    inside a raw HTML block.  A cell written as ``<td …>1. text</td>`` therefore
    became a nested ``<ol><li>`` whose end tags the raw ``<td>`` never closes →
    EPUBCheck FATAL RSC-016 with cascading RSC-012 (live regression:
    the-dictators-dilemma ch005/ch007/ch010, 118 errors + 3 fatals).

    EPUB-only on purpose: the TeX path passes raw HTML through to LaTeX, where
    a Markdown backslash escape would surface as an undefined control sequence
    (observed as ``\\2.`` → "Undefined control sequence" plus CJK rendered in
    the Latin font).
    """
    return re.sub(
        r"(<t[dh]\b[^>]*>)(\s*)(\d{1,3}|[-*+])([.)]?)(\s+)",
        lambda match: (
            f"{match.group(1)}{match.group(2)}\\{match.group(3)}"
            f"{match.group(4)}{match.group(5)}"
        ),
        text,
    )


def _apply_text_level_clean(text: str) -> str:
    """Non-AST text cleanup shared by the legacy and Lua-filter TeX paths.

    These transforms either inject raw TeX ($...$) that Pandoc must parse as
    math (superscripts, UNICODE_MATH), remove OCR long-numeric runs, or
    normalize blank lines/comments that are not AST shape changes. They run
    byte-identically on both preparation paths (plan decision 1).
    """
    text = _ensure_heading_blank_lines(text)
    text = re.sub(r"<!--\s*PAGE(?:_JSON)?:\s*\d+\s*-->\n?", "", text)
    text = re.sub(r"<!--\s*NOTRANSLATE:BEGIN[^>]*-->", "", text)
    text = re.sub(r"<!--\s*NOTRANSLATE:END\s*-->", "", text)
    # Source emphasis markers of the form ">phrase<" (the German commentary
    # convention, preserved verbatim by the translator) are not valid HTML;
    # left alone, Pandoc parses "<" as a raw tag and emits invalid xhtml
    # (EPUB RSC-005/016) / drops content. Normalize the PAIR to Markdown
    # emphasis. The guards keep "-->" comments, "<sup>"/"</sup>" tags and
    # blockquote "> " lines untouched.
    text = re.sub(
        r"(?<![A-Za-z0-9_/\-])>([^\s<>][^<>\n]{0,119}?)<(?![A-Za-z/])",
        r"*\1*",
        text,
    )
    text = re.sub(r"\d{100,}", "", text)
    text = re.sub(
        r"\$\s*\^\{(\d+(?:[,-]\d+)*)\}\s*\$",
        lambda match: f"$^{{{match.group(1)}}}$",
        text,
    )
    text = re.sub(r"\$\^(\d+)\$", lambda match: f"$^{{{match.group(1)}}}$", text)
    text = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda match: f"$^{{{match.group(0).translate(SUPERSCRIPT_DIGITS)}}}$",
        text,
    )
    for character, math in UNICODE_MATH.items():
        text = text.replace(character, math)
    return re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"


def clean_for_latex(text: str, title: str, author: str = "") -> str:
    """Apply the proven Cambridge-series Markdown cleanup before Pandoc.

    Legacy path: title-page drop + heading promotion happen on the Markdown
    text itself; the AST-level equivalent lives in pandoc_lua/ for the
    Lua-filter path. Behavior is byte-identical to the historical pipeline.
    """
    return _apply_text_level_clean(_promote_headings_markdown(text, title, author))


def _pandoc_args_for_tex(
    config: Dict[str, Any], markdown: str, title: str, author: str
) -> tuple[str, List[str]]:
    """Route Markdown -> TeX preparation (plan decision 1, two-path design).

    Returns ``(prepared_markdown_text, pandoc --lua-filter args)``. Legacy
    (publish.lua_filters=false, the default) keeps the proven regex text
    cleanup; the filter path delegates the three AST categories (title-page
    drop, heading promotion, page/NOTRANSLATE comment removal) to Lua
    filters in fixed order drop_title_page -> promote_headings ->
    strip_comments and shares the text-level cleanup.
    """
    settings = _publish_settings(config)
    if not settings["lua_filters"]:
        return clean_for_latex(markdown, title, author), []
    filters = [
        str(_filters_dir() / f"{name}.lua")
        for name in ("drop_title_page", "promote_headings", "strip_comments")
    ]
    return _apply_text_level_clean(markdown), [f"--lua-filter={path}" for path in filters]


def _add_section_pagebreaks(body: str) -> str:
    """Keep sections flowing continuously within a chapter (pilot styling).

    Page breaks are only desired BETWEEN chapters. The ctexbook document
    class already starts each ``\\chapter`` on a fresh page, so this stage no
    longer injects ``\\clearpage`` before ``\\section`` — a section must not
    force its own page. The function is kept as an idempotent normalization
    pass so the surrounding TeX pipeline stays stable.
    """
    return body


def postprocess_body_tex(body: str) -> str:
    for command in ("chapter", "section", "subsection", "subsubsection", "paragraph"):
        body = re.sub(
            rf"(\\{command}\*?)\{{([^}}]*\n[^}}]*)\}}",
            lambda match: match.group(1) + "{" + re.sub(r"\s*\n\s*", " ", match.group(2)) + "}",
            body,
        )
    body = _add_section_pagebreaks(body)
    return re.sub(r"(\s)\^\{(\d+(?:[,-]\d+)*)\}", r"\1\\textsuperscript{\2}", body)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def build_book_tex(title: str, author: str, tocdepth: int = 3) -> str:
    escaped_title = _latex_escape(title.replace("（", "(").replace("）", ")"))
    escaped_author = _latex_escape(author)
    return rf"""\documentclass[UTF8,oneside,12pt,fontset=none]{{ctexbook}}
\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{Heiti SC}}
\setCJKmonofont{{Menlo}}
\usepackage{{amsmath,amssymb}}
\usepackage{{longtable,booktabs,array,calc}}
\usepackage{{graphicx}}
\newcounter{{none}}
\usepackage[hmargin=2.5cm,vmargin=2.5cm]{{geometry}}
\usepackage[unicode=true,hidelinks]{{hyperref}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\setcounter{{secnumdepth}}{{-1}}
\setcounter{{tocdepth}}{{{tocdepth}}}
\hypersetup{{pdftitle={{{escaped_title}}},pdfauthor={{{escaped_author}}},bookmarksdepth={{{tocdepth}}}}}
\title{{{escaped_title}}}
\author{{{escaped_author}}}
\date{{}}
\begin{{document}}
\maketitle
\tableofcontents
\input{{body.tex}}
\end{{document}}
"""


def _build_pdf_from_tex(
    *, config: Dict[str, Any], markdown: Path, exports_dir: Path, book_id: str,
    title: str, author: str, pandoc: str,
) -> Dict[str, Any]:
    """Create persistent TeX sources first, then compile them twice."""
    settings = _publish_settings(config)
    tex_dir = exports_dir / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)
    raw_markdown = markdown.read_text(encoding="utf-8")
    # Structure measurement is path-independent: both preparation paths drop
    # the render title page and promote headings, so derive the required
    # depth once from that canonical TeX-facing heading set (decision 2).
    structure_markdown = _promote_headings_markdown(raw_markdown, title, author)
    required_depth = _required_toc_depth(structure_markdown)
    heading_count = len(re.findall(r"^(#{1,6})\s", structure_markdown, re.M))
    if required_depth > int(settings["pdf"]["tocdepth"]):
        raise RuntimeError(
            "publish.pdf.tocdepth too shallow for structure: need depth "
            f"{required_depth}, configured {settings['pdf']['tocdepth']}"
        )
    prepared_markdown = tex_dir / "book_for_tex.md"
    body_tex = tex_dir / "body.tex"
    book_tex = tex_dir / "book.tex"
    prepared_text, filter_args = _pandoc_args_for_tex(config, raw_markdown, title, author)
    prepared_markdown.write_text(prepared_text, encoding="utf-8")
    command = [
        pandoc, str(prepared_markdown),
        "-f", "markdown-raw_tex-yaml_metadata_block", "-t", "latex",
        "-o", str(body_tex), "--top-level-division=chapter",
    ]
    if filter_args:
        # The Lua title-page filter matches --metadata title/author; without
        # -s/--standalone these metadata values never reach the LaTeX body.
        command += ["--metadata", f"title={title}"]
        if author:
            command += ["--metadata", f"author={author}"]
        command += filter_args
    _run(command, tex_dir)
    body_tex.write_text(
        postprocess_body_tex(body_tex.read_text(encoding="utf-8")), encoding="utf-8"
    )
    book_tex.write_text(
        build_book_tex(title, author, tocdepth=int(settings["pdf"]["tocdepth"])),
        encoding="utf-8",
    )

    for filename in ("book.pdf", "book.aux", "book.log", "book.out", "book.toc"):
        stale = tex_dir / filename
        if stale.exists():
            stale.unlink()
    command = [
        _xelatex(), "-interaction=nonstopmode", "-halt-on-error",
        "-jobname=book", "book.tex",
    ]
    _run(command, tex_dir)
    _run(command, tex_dir)
    compiled_pdf = tex_dir / "book.pdf"
    verification = _verify_pdf(compiled_pdf)
    toc_file = tex_dir / "book.toc"
    bookmark_file = tex_dir / "book.out"
    toc_text = (
        toc_file.read_text(encoding="utf-8", errors="ignore")
        if toc_file.is_file() else ""
    )
    bookmark_text = (
        bookmark_file.read_text(encoding="utf-8", errors="ignore")
        if bookmark_file.is_file() else ""
    )
    if not toc_text.strip() or not bookmark_text.strip():
        raise RuntimeError("PDF verification failed: TeX table of contents or clickable bookmarks missing")
    verification.update(_verify_toc_depth(
        toc_text, required_depth=required_depth, heading_count=heading_count,
    ))
    verification.update(_verify_bookmark_depth(
        bookmark_text, required_depth=required_depth,
    ))
    verification.update({
        "toc_present": True, "clickable_bookmarks_present": True,
        "xelatex_passes": 2,
    })
    destination = exports_dir / f"{book_id}.pdf"
    temporary_destination = exports_dir / f".{book_id}.pdf.tmp"
    shutil.copy2(compiled_pdf, temporary_destination)
    os.replace(temporary_destination, destination)
    return {
        "path": str(destination), "sha256": _hash(destination),
        "bytes": destination.stat().st_size,
        "verification": {
            **verification, "toc_present": True, "clickable_bookmarks_present": True,
            "xelatex_passes": 2,
        },
        "tex": {
            "directory": str(tex_dir),
            "book_for_tex": str(prepared_markdown),
            "body_tex": str(body_tex),
            "book_tex": str(book_tex),
            "book_tex_sha256": _hash(book_tex),
            "body_tex_sha256": _hash(body_tex),
        },
    }


def _verify_epub(path: Path) -> Dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise RuntimeError("EPUB verification failed: output is not a ZIP container")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "mimetype" not in names or archive.read("mimetype") != b"application/epub+zip":
            raise RuntimeError("EPUB verification failed: invalid mimetype")
        if not any("cover" in name.casefold() for name in names):
            raise RuntimeError("EPUB verification failed: cover asset/metadata not found")
    return {"container_entries": len(names), "cover_present": True}


def _epubcheck_discover(settings: Dict[str, Any]) -> Optional[str]:
    """Locate an epubcheck executable or jar, or return None (decision 3).

    Discovery order: explicit publish.epubcheck.path -> PATH executable ->
    common Homebrew jar locations. Absence is reported honestly as
    validation_incomplete downstream, never as a fabricated pass.
    """
    epubcheck = settings["epubcheck"]
    explicit = str(epubcheck.get("path") or "").strip()
    if explicit:
        return explicit
    executable = shutil.which("epubcheck")
    if executable:
        return executable
    jar_candidates = [
        "/opt/homebrew/opt/epubcheck/libexec/epubcheck.jar",
        "/usr/local/opt/epubcheck/libexec/epubcheck.jar",
        "/opt/homebrew/Cellar/epubcheck/*/libexec/epubcheck.jar",
    ]
    for pattern in jar_candidates:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _run_epubcheck(epub_path: Path, tool: str, cwd: Path) -> Dict[str, Any]:
    """Run EPUBCheck (jar via ``java -jar``, or an executable) once."""
    if tool.casefold().endswith(".jar"):
        command = [
            shutil.which("java") or "java", "-jar", tool,
            str(epub_path), "--quiet",
        ]
    else:
        command = [tool, str(epub_path)]
    completed = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    detail = ((completed.stderr or "") + (completed.stdout or "")).strip()[-4000:]
    return {"tool": tool, "exit_code": completed.returncode, "detail": detail}


def _run_epub_validation(
    epub_path: Path, settings: Dict[str, Any], cwd: Path
) -> Dict[str, Any]:
    """Apply publish.epubcheck settings to the temporary EPUB (decision 3).

    Status contract recorded in outputs.epub.verification.validation:
    absent tool + require=false -> "validation_incomplete" (artifact kept);
    absent/present-failing + require=true -> RuntimeError (fail closed);
    tool exit 0 -> "valid"; tool exit != 0 + require=false -> "invalid"
    with detail. enabled=false -> "skipped". Never fabricates a pass.
    """
    epubcheck = settings["epubcheck"]
    if not epubcheck["enabled"]:
        return {"validation": "skipped"}
    tool = _epubcheck_discover(settings)
    if tool is None:
        if epubcheck["require"]:
            raise RuntimeError(
                "EPUBCheck validation is required (publish.epubcheck.require=true) "
                "but no epubcheck executable or jar was found"
            )
        return {"validation": "validation_incomplete"}
    result = _run_epubcheck(epub_path, tool, cwd)
    if result["exit_code"] == 0:
        return {"validation": "valid", "epubcheck": result}
    if epubcheck["require"]:
        raise RuntimeError(
            "EPUBCheck validation failed (exit "
            f"{result['exit_code']}): {result['detail']}"
        )
    return {"validation": "invalid", "epubcheck": result}


def _verify_pdf(path: Path) -> Dict[str, Any]:
    if not path.is_file() or not path.read_bytes()[:5] == b"%PDF-":
        raise RuntimeError("PDF verification failed: invalid PDF signature")
    page_count: Optional[int] = None
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        completed = subprocess.run(
            [pdfinfo, str(path)], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if line.startswith("Pages:"):
                    page_count = int(line.split(":", 1)[1].strip())
                    break
    if page_count == 0:
        raise RuntimeError("PDF verification failed: document has no pages")
    return {"page_count": page_count}


def export_book(
    config: Dict[str, Any],
    project_dir: Path,
    formats: Iterable[str] = ("pdf", "epub"),
) -> Dict[str, Any]:
    """Export a fully rendered Markdown book to verified PDF and/or EPUB."""
    require_structure_gate(config)
    require_fresh_cleaning(config)
    status = translation_status(config)
    reject_demo_translations(config, list(read_jsonl(Path(config["output_dir"]) / "translations.jsonl"))) if (Path(config["output_dir"]) / "translations.jsonl").is_file() else None
    if not status["ready_to_render"]:
        raise RuntimeError("Publication export requires 100% completed translation")
    output_dir = resolve_path(config, "output_dir")
    markdown = output_dir / "book_translated.md"
    render_report_path = output_dir / "render_report.json"
    if not markdown.is_file() or not render_report_path.is_file():
        raise FileNotFoundError("Run new_book.py render before publication export")
    render_report = json.loads(render_report_path.read_text(encoding="utf-8"))
    if render_report.get("partial"):
        raise RuntimeError("Partial Markdown previews cannot be exported as deliverables")

    requested = list(dict.fromkeys(str(item).casefold() for item in formats))
    if not requested or any(item not in {"pdf", "epub"} for item in requested):
        raise ValueError("formats must contain pdf and/or epub")
    cover = load_registered_cover(project_dir) if "epub" in requested else None
    exports_dir = project_dir.expanduser().resolve() / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    settings = _publish_settings(config)
    pandoc = _pandoc()
    title = str(config["book_title_zh"])
    author = str(config.get("author") or "")
    common = [
        pandoc, str(markdown), "--standalone", "--toc",
        f"--toc-depth={settings['toc_depth']}",
        "--metadata", f"title={title}", "--resource-path", f"{markdown.parent}:{project_dir}",
    ]
    if author:
        common += ["--metadata", f"author={author}"]

    results: Dict[str, Any] = {}
    if "epub" in requested:
        with tempfile.TemporaryDirectory(prefix="book-epub-") as temporary:
            temp_dir = Path(temporary)
            temporary_epub = temp_dir / "book.epub"
            # EPUB consumes a text-level-cleaned copy of the rendered markdown
            # (page/NOTRANSLATE comments stripped, ">phrase<" normalized) so
            # Pandoc never sees raw HTML fragments; resource-path keeps the
            # original dir for any referenced assets.
            epub_markdown = temp_dir / "book_for_epub.md"
            epub_markdown.write_text(
                _escape_table_cell_list_markers(
                    _apply_text_level_clean(markdown.read_text(encoding="utf-8"))
                ),
                encoding="utf-8",
            )
            epub_common = list(common)
            epub_common[1] = str(epub_markdown)
            command = epub_common + [
                "--to=epub3",
                f"--epub-chapter-level={settings['epub_chapter_level']}",
                f"--epub-cover-image={cover['image_path']}",
                "--output", str(temporary_epub),
            ]
            _run(command, temp_dir)
            verification = _verify_epub(temporary_epub)
            verification.update(_run_epub_validation(temporary_epub, settings, temp_dir))
            destination = exports_dir / f"{config['book_id']}.epub"
            os.replace(temporary_epub, destination)
            results["epub"] = {
                "path": str(destination), "sha256": _hash(destination),
                "bytes": destination.stat().st_size, "verification": verification,
            }
    if "pdf" in requested:
        results["pdf"] = _build_pdf_from_tex(
            config=config,
            markdown=markdown,
            exports_dir=exports_dir,
            book_id=str(config["book_id"]),
            title=title,
            author=author,
            pandoc=pandoc,
        )

    report = {
        "schema_version": 1,
        "book_id": config["book_id"],
        "source_markdown": str(markdown),
        "source_markdown_sha256": _hash(markdown),
        "cover": cover,
        "outputs": results,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(exports_dir / "export_report.json", report)
    return report
