#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an inference checkpoint by weighted-averaging CTC model tensors. "
            "Only floating-point tensors are averaged; non-floating tensors must match."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("LABEL", "WEIGHT", "PATH"),
        required=True,
        help="Checkpoint component. Repeat as: --checkpoint stage15 0.75 /path/to/ckpt.pt",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--anchor-label",
        default=None,
        help="Component label whose non-model metadata/config files should be preserved. Defaults to first input.",
    )
    parser.add_argument(
        "--no-copy-configs",
        action="store_true",
        help="Do not copy model_config.yaml and tokenizer_config.yaml from the anchor checkpoint directory.",
    )
    return parser


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(checkpoint)}")
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise TypeError(f"Expected checkpoint['model'] dict at {path}")
    return checkpoint


def _parse_components(raw: list[list[str]]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for label, weight_text, path_text in raw:
        weight = float(weight_text)
        if weight < 0.0:
            raise ValueError(f"Negative soup weight for {label}: {weight}")
        path = Path(path_text)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint missing or empty for {label}: {path}")
        components.append({"label": label, "weight": weight, "path": path})
    total = sum(float(item["weight"]) for item in components)
    if total <= 0.0:
        raise ValueError("At least one checkpoint weight must be positive.")
    for item in components:
        item["normalized_weight"] = float(item["weight"]) / total
    labels = [str(item["label"]) for item in components]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate checkpoint labels are not allowed: {labels}")
    return components


def _validate_compatible(anchor_model: dict[str, Any], model: dict[str, Any], label: str) -> None:
    anchor_keys = set(anchor_model)
    model_keys = set(model)
    if anchor_keys != model_keys:
        missing = sorted(anchor_keys - model_keys)[:20]
        extra = sorted(model_keys - anchor_keys)[:20]
        raise ValueError(f"Model key mismatch for {label}: missing={missing}, extra={extra}")

    for key, anchor_value in anchor_model.items():
        value = model[key]
        if not isinstance(anchor_value, torch.Tensor) or not isinstance(value, torch.Tensor):
            if type(anchor_value) is not type(value):
                raise TypeError(f"Non-tensor type mismatch for {label}:{key}")
            continue
        if anchor_value.shape != value.shape:
            raise ValueError(
                f"Tensor shape mismatch for {label}:{key}: {tuple(anchor_value.shape)} vs {tuple(value.shape)}"
            )
        if anchor_value.dtype != value.dtype:
            raise ValueError(f"Tensor dtype mismatch for {label}:{key}: {anchor_value.dtype} vs {value.dtype}")


def _average_models(components: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_model = components[0]["checkpoint"]["model"]
    for component in components[1:]:
        _validate_compatible(anchor_model, component["checkpoint"]["model"], str(component["label"]))

    output: dict[str, Any] = {}
    for key, anchor_value in anchor_model.items():
        if not isinstance(anchor_value, torch.Tensor):
            output[key] = copy.deepcopy(anchor_value)
            continue

        if anchor_value.is_floating_point() or anchor_value.is_complex():
            acc = None
            for component in components:
                tensor = component["checkpoint"]["model"][key].detach().cpu()
                weighted = tensor.to(torch.float32) * float(component["normalized_weight"])
                acc = weighted if acc is None else acc + weighted
            output[key] = acc.to(dtype=anchor_value.dtype).contiguous()
            continue

        for component in components[1:]:
            tensor = component["checkpoint"]["model"][key]
            if not torch.equal(anchor_value, tensor):
                raise ValueError(f"Non-floating tensor differs for {component['label']}:{key}")
        output[key] = anchor_value.detach().cpu().clone().contiguous()
    return output


def _copy_configs(anchor_path: Path, output_path: Path) -> None:
    output_dir = output_path.parent
    for name in ("model_config.yaml", "tokenizer_config.yaml"):
        source = anchor_path.resolve().parent / name
        if source.exists():
            target = output_dir / name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)


def main() -> None:
    args = build_parser().parse_args()
    components = _parse_components(args.checkpoint)
    anchor_label = args.anchor_label or str(components[0]["label"])
    anchor_matches = [item for item in components if item["label"] == anchor_label]
    if len(anchor_matches) != 1:
        raise ValueError(f"anchor-label must match exactly one component: {anchor_label}")
    anchor = anchor_matches[0]

    for component in components:
        component["checkpoint"] = _load_checkpoint(component["path"])
    if components[0] is not anchor:
        components = [anchor] + [item for item in components if item is not anchor]

    model = _average_models(components)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extra = copy.deepcopy(anchor["checkpoint"].get("extra", {}))
    extra["checkpoint_soup"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "anchor_label": str(anchor["label"]),
        "components": [
            {
                "label": str(component["label"]),
                "weight": float(component["weight"]),
                "normalized_weight": float(component["normalized_weight"]),
                "path": str(component["path"]),
                "step": int(component["checkpoint"].get("step", 0)),
            }
            for component in components
        ],
    }
    output = {
        "model": model,
        "step": int(anchor["checkpoint"].get("step", 0)),
        "extra": extra,
    }
    torch.save(output, output_path)
    if not args.no_copy_configs:
        _copy_configs(Path(anchor["path"]), output_path)

    metadata_path = output_path.with_suffix(output_path.suffix + ".soup.json")
    metadata_path.write_text(
        json.dumps(extra["checkpoint_soup"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "create_ctc_checkpoint_soup "
        f"output={output_path} tensors={len(model)} anchor={anchor['label']} "
        f"components={','.join(str(component['label']) for component in components)}"
    )


if __name__ == "__main__":
    main()
