from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210g = importlib.import_module("scripts.run_stage210g_conditional_hard_ctc_distill")


def test_stage210g_adds_blank_isolated_hard_token_target(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210f.pt"
    checkpoint.touch()
    segment = stage210g._segments("admission", smoke=False)[0]
    config = stage210g._config(
        phase="admission",
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["max_steps"] == 2_000
    assert config["save_every"] == 500
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_conditional_nonblank_hard_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.25)
    assert config["step_eval_cache_batches"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True


def test_stage210g_direct_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210f.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210g_conditional_hard_ctc_distill.py"),
            "--dry-run",
            "--smoke",
            "--only-next",
            "--init-checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "run"),
            "--config-dir",
            str(tmp_path / "config"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "phase=admission" in result.stdout
    assert "target_step=2" in result.stdout
