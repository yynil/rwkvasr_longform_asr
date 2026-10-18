from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch

from rwkvasr.eval.stage211_gate import (
    sha256_file,
    validate_stage211_runtime_epoch_coverage,
)


def audit_stage211_runtime_epoch_coverage(
    *,
    run_dir: Path,
    epochs: int,
    steps_per_epoch: int,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    records: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        checkpoint_path = run_dir / f"epoch-{epoch}.pt"
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
            raise ValueError(
                f"Stage211 runtime epoch checkpoint is missing or empty: "
                f"{checkpoint_path}"
            )
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        try:
            extra = payload.get("extra")
            if not isinstance(extra, dict):
                raise ValueError(
                    f"Stage211 runtime epoch checkpoint lacks extra state: "
                    f"{checkpoint_path}"
                )
            records.append(
                {
                    "epoch": int(extra.get("epoch", -1)),
                    "step": int(payload.get("step", -1)),
                    "epoch_batch_offset": int(
                        extra.get("epoch_batch_offset", -1)
                    ),
                    "completed_epoch_batch_count": int(
                        extra.get("completed_epoch_batch_count", -1)
                    ),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
        finally:
            del payload
            gc.collect()
    coverage = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "runtime_epoch_coverage",
        "complete": True,
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(epochs) * int(steps_per_epoch),
        "records": records,
    }
    return validate_stage211_runtime_epoch_coverage(
        coverage,
        epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        label=str(run_dir),
    )
