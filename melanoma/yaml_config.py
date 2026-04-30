from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from melanoma.config import PROJECT_ROOT


def load_yaml_file(path: Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing YAML config: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return data


def load_yaml_section(config_path: Path, section: str) -> dict[str, Any]:
    data = load_yaml_file(config_path)
    shared = data.get("shared", {})
    if shared is None:
        shared = {}
    if not isinstance(shared, dict):
        raise ValueError(f"'shared' section must be a mapping: {config_path}")
    section_data = data.get(section, {})
    if section_data is None:
        section_data = {}
    if not isinstance(section_data, dict):
        raise ValueError(f"'{section}' section must be a mapping: {config_path}")
    return {**shared, **section_data}


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
