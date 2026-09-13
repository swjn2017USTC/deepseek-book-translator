from __future__ import annotations

"""Deterministic, locally generated typographic book covers."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .io_utils import write_json
from .publish import register_cover


WIDTH = 1600
HEIGHT = 2400
THEMES: Dict[str, Tuple[str, str, str, str]] = {
    "ink": ("#111827", "#26324A", "#F2E9D8", "#D9A441"),
    "ocean": ("#082F49", "#0E7490", "#ECFEFF", "#67E8F9"),
    "ember": ("#431407", "#9A3412", "#FFF7ED", "#FDBA74"),
    "forest": ("#052E16", "#166534", "#F0FDF4", "#86EFAC"),
    "plum": ("#2E1065", "#6B21A8", "#FAF5FF", "#D8B4FE"),
}


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - installation error
        raise RuntimeError("Pillow is required for generated covers") from exc
    return Image, ImageDraw, ImageFont


def _font_candidates() -> Iterable[Path]:
    configured = os.environ.get("BOOK_COVER_FONT")
    if configured:
        yield Path(configured).expanduser()
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for path in (
        windows / "msyh.ttc",
        windows / "msyhbd.ttc",
        windows / "simhei.ttf",
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        yield path


def _font(size: int):
    _, _, ImageFont = _pillow()
    for candidate in _font_candidates():
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _units(text: str) -> List[str]:
    result: List[str] = []
    latin = ""
    for character in text.strip():
        if character.isspace():
            if latin:
                result.append(latin)
                latin = ""
            if result and result[-1] != " ":
                result.append(" ")
        elif ord(character) < 256:
            latin += character
        else:
            if latin:
                result.append(latin)
                latin = ""
            result.append(character)
    if latin:
        result.append(latin)
    return result


def _text_width(draw: Any, text: str, font: Any) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0])


def _wrap(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    lines: List[str] = []
    current = ""
    for unit in _units(text):
        candidate = current + unit
        if current and _text_width(draw, candidate, font) > max_width:
            lines.append(current.strip())
            current = unit.lstrip()
        else:
            current = candidate
    if current.strip():
        lines.append(current.strip())
    return lines or [""]


def _fit(draw: Any, text: str, max_width: int, max_lines: int, start: int, floor: int):
    for size in range(start, floor - 1, -4):
        font = _font(size)
        lines = _wrap(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return font, lines
    font = _font(floor)
    lines = _wrap(draw, text, font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip("…") + "…"
    return font, lines


def _centered_lines(
    draw: Any,
    lines: Sequence[str],
    font: Any,
    *,
    center_y: int,
    fill: str,
    spacing: int,
) -> None:
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [int(box[3] - box[1]) for box in boxes]
    total = sum(heights) + spacing * max(0, len(lines) - 1)
    y = center_y - total // 2
    for line, box, line_height in zip(lines, boxes, heights):
        width = int(box[2] - box[0])
        draw.text(((WIDTH - width) // 2, y), line, font=font, fill=fill)
        y += line_height + spacing


def _theme(name: str, seed: str) -> Tuple[str, Tuple[str, str, str, str]]:
    if name != "auto":
        if name not in THEMES:
            raise ValueError(f"Unknown cover theme: {name}")
        return name, THEMES[name]
    names = sorted(THEMES)
    selected = names[int(sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % len(names)]
    return selected, THEMES[selected]


def render_typographic_cover(
    destination: Path,
    *,
    book_id: str,
    title: str,
    original_title: str = "",
    author: str = "",
    theme: str = "auto",
) -> Dict[str, Any]:
    """Render a 2:3 PNG using only metadata and local vector shapes."""
    Image, ImageDraw, _ = _pillow()
    selected_theme, palette = _theme(theme, f"{book_id}\n{title}\n{author}")
    background, panel, foreground, accent = palette
    image = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(image)

    # A deterministic geometric field provides visual identity without using
    # any third-party art or network image service.
    draw.rectangle((0, 0, WIDTH, 52), fill=accent)
    draw.rectangle((105, 160, WIDTH - 105, HEIGHT - 160), fill=panel)
    draw.ellipse((WIDTH - 610, 105, WIDTH + 180, 895), outline=accent, width=24)
    draw.ellipse((-260, HEIGHT - 770, 530, HEIGHT + 20), outline=accent, width=12)
    draw.line((155, 530, WIDTH - 155, 530), fill=accent, width=6)
    draw.line((155, HEIGHT - 465, WIDTH - 155, HEIGHT - 465), fill=accent, width=6)

    title_font, title_lines = _fit(draw, title or original_title, WIDTH - 330, 6, 154, 76)
    _centered_lines(draw, title_lines, title_font, center_y=1130, fill=foreground, spacing=34)

    if original_title and original_title.casefold() != title.casefold():
        original_font, original_lines = _fit(draw, original_title, WIDTH - 390, 4, 56, 34)
        _centered_lines(draw, original_lines, original_font, center_y=1630, fill=foreground, spacing=18)

    if author:
        author_font, author_lines = _fit(draw, author, WIDTH - 420, 2, 54, 34)
        _centered_lines(draw, author_lines, author_font, center_y=350, fill=foreground, spacing=16)

    footer = "AI-ASSISTED TRANSLATION · DEEPSEEK"
    footer_font = _font(31)
    footer_width = _text_width(draw, footer, footer_font)
    draw.text(((WIDTH - footer_width) // 2, HEIGHT - 330), footer, font=footer_font, fill=accent)

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)
    return {
        "generator": "book_pipeline.cover_generator",
        "generator_version": 1,
        "theme": selected_theme,
        "width": WIDTH,
        "height": HEIGHT,
        "uses_third_party_art": False,
    }


def generate_typographic_cover(project_dir: Path, *, theme: str = "auto") -> Dict[str, Any]:
    """Generate and register a cover for an initialized book project."""
    project = project_dir.expanduser().resolve()
    manifest_path = project / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Project manifest not found: {manifest_path}")
    project_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = Path(project_manifest["translation_config"]).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    title = str(config.get("book_title_zh") or config.get("book_title") or "").strip()
    original_title = str(config.get("book_title") or "").strip()
    author = str(config.get("author") or "").strip()
    if not title:
        raise ValueError("A title is required to generate a cover")

    with tempfile.TemporaryDirectory(prefix="book-cover-") as temp_dir:
        temp_image = Path(temp_dir) / "cover.png"
        generation = render_typographic_cover(
            temp_image,
            book_id=str(config.get("book_id") or project_manifest.get("book_id") or "book"),
            title=title,
            original_title=original_title,
            author=author,
            theme=theme,
        )
        registered = register_cover(
            project,
            temp_image,
            rights_status="generated",
            selected_by="local_generator",
            edition_evidence=(
                f"Typographic cover generated locally from project metadata: "
                f"title={title!r}, original_title={original_title!r}, author={author!r}. "
                "No third-party artwork was used."
            ),
        )
    result = {**registered, **generation}
    write_json(project / "cover" / "cover.json", result)
    write_json(project / "cover" / "cover_generation.json", {
        "schema_version": 1,
        "book_id": config.get("book_id"),
        "title": title,
        "original_title": original_title,
        "author": author,
        **generation,
        "image_path": result["image_path"],
        "sha256": result["sha256"],
    })
    return result
