from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import sha256_file


LAYER_IDS = tuple(range(70))


def _layer_metrics(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    components = report.get("layer_components") or report.get("layer_component_metrics")
    if isinstance(components, dict) and isinstance(components.get("mixer"), dict):
        layers = components["mixer"]
    else:
        layers = report.get("layers") or report.get("layer_metrics")
    if not isinstance(layers, dict) or set(layers) != {str(layer_id) for layer_id in LAYER_IDS}:
        raise ValueError("Calibration hidden report must contain exactly 70 mixer layers.")
    return layers


def _candidate_record(
    *,
    baseline_layers: dict[str, dict[str, float]],
    record: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(str(record.get("checkpoint_path") or "")).resolve()
    layer_report = Path(str(record.get("layer_metrics_path") or "")).resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ValueError(f"Calibration candidate checkpoint is unavailable: {checkpoint}")
    if not layer_report.is_file() or layer_report.stat().st_size <= 0:
        raise ValueError(f"Calibration candidate hidden report is unavailable: {layer_report}")
    report = load_yaml(layer_report)
    layers = _layer_metrics(report)
    eval_loss = float(report.get("eval_loss", float("nan")))
    if not math.isfinite(eval_loss):
        raise ValueError(f"Calibration candidate has non-finite eval loss: {layer_report}")
    loss_improved_layers = sum(
        float(layers[str(layer_id)]["loss"]) < float(baseline_layers[str(layer_id)]["loss"])
        for layer_id in LAYER_IDS
    )
    cosine_improved_layers = sum(
        float(layers[str(layer_id)]["cosine"]) > float(baseline_layers[str(layer_id)]["cosine"])
        for layer_id in LAYER_IDS
    )
    mean_mixer_loss = sum(float(layers[str(layer_id)]["loss"]) for layer_id in LAYER_IDS) / len(
        LAYER_IDS
    )
    mean_mixer_cosine = sum(float(layers[str(layer_id)]["cosine"]) for layer_id in LAYER_IDS) / len(
        LAYER_IDS
    )
    return {
        "step": int(record.get("step", report.get("step", 0))),
        "eval_loss": eval_loss,
        "eval_samples": int(report.get("eval_samples", 0)),
        "mean_mixer_loss": mean_mixer_loss,
        "mean_mixer_cosine": mean_mixer_cosine,
        "loss_improved_layers": loss_improved_layers,
        "cosine_improved_layers": cosine_improved_layers,
        "eligible": (
            int(report.get("eval_samples", 0)) == 256
            and loss_improved_layers == len(LAYER_IDS)
            and cosine_improved_layers == len(LAYER_IDS)
        ),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "layer_report_path": str(layer_report),
        "layer_report_sha256": sha256_file(layer_report),
    }


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def select_checkpoint(
    run_dir: Path,
    *,
    required_completion_step: int | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    baseline_path = run_dir / "step_eval_baseline.yaml"
    metrics_path = run_dir / "step_checkpoint_metrics.yaml"
    if not baseline_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"Calibration run lacks baseline or checkpoint metrics: {run_dir}")
    baseline_report = load_yaml(baseline_path)
    baseline_layers = _layer_metrics(baseline_report)
    metrics = load_yaml(metrics_path)
    raw_records = metrics.get("step_checkpoints")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Calibration run has no fixed-eval checkpoint records.")
    if required_completion_step is not None:
        completion_checkpoint = run_dir / f"step-{required_completion_step}.pt"
        recorded_steps = {
            int(record.get("step", 0)) for record in raw_records if isinstance(record, dict)
        }
        if (
            not completion_checkpoint.is_file()
            or _checkpoint_step(completion_checkpoint) != required_completion_step
            or required_completion_step not in recorded_steps
        ):
            raise ValueError(
                "Calibration run has not reached its required fixed-eval completion: "
                f"step={required_completion_step}"
            )
    candidates = [
        _candidate_record(
            baseline_layers=baseline_layers,
            record=record,
        )
        for record in raw_records
        if isinstance(record, dict)
    ]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise ValueError("No calibration checkpoint improves loss and cosine on all 70 layers.")
    selected = min(
        eligible,
        key=lambda candidate: (
            float(candidate["eval_loss"]),
            float(candidate["mean_mixer_loss"]),
            -float(candidate["mean_mixer_cosine"]),
            -int(candidate["step"]),
        ),
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "calibration_checkpoint_selection",
        "run_dir": str(run_dir),
        "baseline_report_path": str(baseline_path),
        "baseline_report_sha256": sha256_file(baseline_path),
        "selection_rule": (
            "all 70 mixer layers improve loss and cosine, then minimum fixed eval loss"
        ),
        "required_completion_step": required_completion_step,
        "selected": selected,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the fixed-hidden best Stage211A calibration checkpoint."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-path-output", type=Path, default=None)
    parser.add_argument("--required-completion-step", type=int, default=None)
    args = parser.parse_args()

    if args.required_completion_step is not None and args.required_completion_step <= 0:
        parser.error("--required-completion-step must be positive")
    report = select_checkpoint(
        args.run_dir,
        required_completion_step=args.required_completion_step,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different selection: {output}")
    output.write_text(rendered, encoding="utf-8")
    selected = report["selected"]
    if args.selected_path_output is not None:
        selected_path_output = args.selected_path_output.resolve()
        selected_path_output.parent.mkdir(parents=True, exist_ok=True)
        selected_path_output.write_text(
            str(selected["checkpoint_path"]) + "\n",
            encoding="utf-8",
        )
    print(
        f"selected_checkpoint={selected['checkpoint_path']} "
        f"step={selected['step']} eval_loss={selected['eval_loss']:.6f} "
        f"sha256={selected['checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
