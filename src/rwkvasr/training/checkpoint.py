from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "step": int(step),
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        if optimizer is not None:
            raise ValueError("Optimizer state cannot be restored from a safetensors inference checkpoint.")
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
        model.load_state_dict(state_dict)
        return {
            "step": 0,
            "extra": {},
        }

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return {
        "step": int(checkpoint.get("step", 0)),
        "extra": checkpoint.get("extra", {}),
    }


def export_checkpoint_to_safetensors(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a model state dict in {checkpoint_path}, got {type(state_dict)}")

    tensors = {
        str(name): tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if isinstance(tensor, torch.Tensor)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output_path),
        metadata={
            "source_checkpoint": str(checkpoint_path),
            "step": str(int(checkpoint.get("step", 0))),
        },
    )
    return {
        "step": int(checkpoint.get("step", 0)),
        "extra": checkpoint.get("extra", {}),
        "num_tensors": len(tensors),
    }
