from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from rwkvasr.config import load_yaml, save_yaml


LATEST_CHECKPOINT_FILENAME = "latest_checkpoint.yaml"


def latest_checkpoint_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / LATEST_CHECKPOINT_FILENAME


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


def load_latest_checkpoint_state(output_dir: str | Path) -> dict[str, Any]:
    pointer = latest_checkpoint_path(output_dir)
    if not pointer.exists():
        return {}
    data = load_yaml(pointer)
    if isinstance(data, dict):
        return data
    return {}


def write_latest_checkpoint_state(output_dir: str | Path, payload: dict[str, Any]) -> None:
    latest = dict(payload)
    latest_checkpoint_file = latest_checkpoint_path(output_dir)
    latest_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    save_yaml(latest_checkpoint_file, latest)


def resolve_latest_checkpoint_path(output_dir: str | Path) -> str | None:
    payload = load_latest_checkpoint_state(output_dir)
    checkpoint_path = payload.get("checkpoint_path")
    if isinstance(checkpoint_path, str) and checkpoint_path:
        return checkpoint_path
    deepspeed_dir = payload.get("deepspeed_checkpoint_dir")
    if isinstance(deepspeed_dir, str) and deepspeed_dir:
        return deepspeed_dir
    return None


def extract_epoch_batch_offset(payload: Any) -> int:
    try:
        value = int(payload)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, value)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        if optimizer is not None:
            raise ValueError("Optimizer state cannot be restored from a safetensors inference checkpoint.")
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
        load_result = model.load_state_dict(state_dict, strict=strict)
        return {
            "step": 0,
            "extra": {
                "missing_keys": list(load_result.missing_keys),
                "unexpected_keys": list(load_result.unexpected_keys),
            },
        }

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    load_result = model.load_state_dict(checkpoint["model"], strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    extra = dict(checkpoint.get("extra", {}))
    extra["missing_keys"] = list(load_result.missing_keys)
    extra["unexpected_keys"] = list(load_result.unexpected_keys)
    return {
        "step": int(checkpoint.get("step", 0)),
        "extra": extra,
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
