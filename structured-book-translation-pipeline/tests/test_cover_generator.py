import json
from pathlib import Path

from PIL import Image

from book_pipeline.cover_generator import generate_typographic_cover
from book_pipeline.publish import load_registered_cover


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "book"
    project.mkdir()
    config = project / "translation_config.json"
    config.write_text(json.dumps({
        "book_id": "fixture-cover",
        "book_title": "The Structure of a Fixture",
        "book_title_zh": "样例结构研究",
        "author": "Example Author",
    }, ensure_ascii=False), encoding="utf-8")
    (project / "project.json").write_text(json.dumps({
        "book_id": "fixture-cover",
        "translation_config": str(config),
    }), encoding="utf-8")
    return project


def test_generated_cover_is_registered_and_deterministic(tmp_path):
    project = _project(tmp_path)
    first = generate_typographic_cover(project, theme="ocean")
    image_path = Path(first["image_path"])

    assert first["rights_status"] == "generated"
    assert first["selected_by"] == "local_generator"
    assert first["uses_third_party_art"] is False
    assert first["theme"] == "ocean"
    assert image_path.is_file()
    with Image.open(image_path) as image:
        assert image.size == (1600, 2400)
        assert image.format == "PNG"

    second = generate_typographic_cover(project, theme="ocean")
    assert second["sha256"] == first["sha256"]
    assert load_registered_cover(project)["sha256"] == first["sha256"]
    generation = json.loads(
        (project / "cover" / "cover_generation.json").read_text(encoding="utf-8")
    )
    assert generation["uses_third_party_art"] is False
