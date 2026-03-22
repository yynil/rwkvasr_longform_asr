from __future__ import annotations

from pathlib import Path
from typing import Any

from rwkvasr.config import save_yaml


def save_epoch_metrics(
    output_dir: str | Path,
    *,
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> Path:
    output_dir = Path(output_dir)
    return save_yaml(
        output_dir / "epoch_metrics.yaml",
        {
            "epochs": history,
            "best": best or {},
        },
    )

