from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


SECRET_KEY = re.compile(r"(?:api[_-]?key|token|secret|password)$", re.I)


def load_config(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    config["_base_dir"] = str(path.parent)
    _reject_embedded_secrets(config)
    for required in ("book_id", "book_title", "book_title_zh", "structure_json", "output_dir"):
        if not config.get(required):
            raise ValueError(f"Missing required config field: {required}")
    source_keys = [key for key in ("input_json", "input_epub") if config.get(key)]
    if len(source_keys) != 1:
        raise ValueError("Config must define exactly one source: input_json or input_epub")
    input_path = resolve_path(config, source_keys[0])
    structure_path = resolve_path(config, "structure_json")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not structure_path.exists():
        raise FileNotFoundError(structure_path)
    output = resolve_path(config, "output_dir")
    source_library = config.get("source_library_root")
    if source_library:
        source_root = Path(source_library).expanduser().resolve()
        if output == source_root or source_root in output.parents:
            raise ValueError("output_dir must not be inside the read-only source library")
    return config


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    value = Path(str(config[key])).expanduser()
    return value.resolve() if value.is_absolute() else (Path(config["_base_dir"]) / value).resolve()


def _reject_embedded_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if SECRET_KEY.search(str(key)) and str(key) not in {"api_key_env"} and child:
                raise ValueError(f"Secret-like value is forbidden in {path}.{key}; use an environment variable name")
            _reject_embedded_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_embedded_secrets(child, f"{path}[{index}]")


def source_input_key(config: Dict[str, Any]) -> str:
    keys = [key for key in ("input_json", "input_epub") if config.get(key)]
    if len(keys) != 1:
        raise ValueError("Config must define exactly one source: input_json or input_epub")
    return keys[0]


def resolve_source_input(config: Dict[str, Any]) -> Path:
    return resolve_path(config, source_input_key(config))
