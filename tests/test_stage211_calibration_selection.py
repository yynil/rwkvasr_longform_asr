from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from rwkvasr.config import save_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
selector = importlib.import_module("scripts.select_stage211_calibration_checkpoint")


def _layers(*, loss: float, cosine: float) -> dict[str, dict[str, float]]:
    return {
        str(layer_id): {
            "loss": loss,
            "cosine": cosine,
            "rms_ratio": 0.9,
        }
        for layer_id in range(70)
    }


def test_calibration_selection_uses_best_all_layer_candidate(
    tmp_path: Path,
) -> None:
    save_yaml(
        tmp_path / "step_eval_baseline.yaml",
        {
            "eval_loss": 1.0,
            "eval_samples": 256,
            "layer_component_metrics": {"mixer": _layers(loss=1.0, cosine=0.5)},
        },
    )
    records = []
    for step, eval_loss, mixer_loss, cosine in (
        (10_000, 0.4, 0.8, 0.6),
        (20_000, 0.3, 0.7, 0.7),
        (30_000, 0.2, 0.6, 0.4),
    ):
        checkpoint = tmp_path / f"step-{step}.pt"
        torch.save({"step": step}, checkpoint)
        layer_report = tmp_path / f"step_eval_layers_step-{step}.yaml"
        save_yaml(
            layer_report,
            {
                "step": step,
                "eval_loss": eval_loss,
                "eval_samples": 256,
                "layer_components": {"mixer": _layers(loss=mixer_loss, cosine=cosine)},
            },
        )
        records.append(
            {
                "step": step,
                "eval_loss": eval_loss,
                "checkpoint_path": str(checkpoint),
                "layer_metrics_path": str(layer_report),
            }
        )
    save_yaml(
        tmp_path / "step_checkpoint_metrics.yaml",
        {"step_checkpoints": records},
    )

    report = selector.select_checkpoint(
        tmp_path,
        required_completion_step=30_000,
    )

    assert report["selected"]["step"] == 20_000
    by_step = {candidate["step"]: candidate for candidate in report["candidates"]}
    assert by_step[10_000]["eligible"] is True
    assert by_step[20_000]["eligible"] is True
    assert by_step[30_000]["eligible"] is False
    assert report["required_completion_step"] == 30_000
