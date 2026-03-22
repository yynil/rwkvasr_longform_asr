from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def _to_yaml_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return _to_yaml_primitive(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_yaml_primitive(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_yaml_primitive(item) for item in value]
    if isinstance(value, list):
        return [_to_yaml_primitive(item) for item in value]
    return value


def save_yaml(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _to_yaml_primitive(data),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data)}")
    return data
