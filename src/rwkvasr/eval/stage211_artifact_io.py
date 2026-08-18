from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _assert_identical(path: Path, payload: bytes, *, label: str) -> None:
    if not path.is_file() or path.read_bytes() != payload:
        raise ValueError(f"Refusing to overwrite a different {label}: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    """Publish complete immutable bytes without exposing a partial final path."""
    path = path.expanduser().resolve()
    if path.exists():
        _assert_identical(path, payload, label=label)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _assert_identical(path, payload, label=label)
        else:
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_immutable_text(path: Path, text: str, *, label: str) -> None:
    write_immutable_bytes(path, text.encode("utf-8"), label=label)


def write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    rendered = json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    write_immutable_text(path, rendered, label=label)
