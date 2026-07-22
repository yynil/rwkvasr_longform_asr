from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210d = importlib.import_module("scripts.run_stage210d_balanced_full_ctc_distill")


def test_stage210d_admission_uses_balanced_all_frame_kl(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210b.pt"
    checkpoint.touch()
    config = stage210d._config(
        phase="admission",
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["max_steps"] == 2_000
    assert config["save_every"] == 1_000
    assert config["step_eval_every"] == 1_000
    assert config["init_checkpoint_path"] == str(checkpoint)
    assert config["ctc_teacher_online_full_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_full_frame_filter"] == "all"
    assert config["ctc_teacher_online_full_nonblank_weight"] == pytest.approx(4.0)
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.125)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(0.50)
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["lr"] == pytest.approx(3.0e-7)


def test_stage210d_easy_uses_complete_easy_epoch(tmp_path: Path) -> None:
    config = stage210d._config(
        phase="easy",
        output_dir=tmp_path / "run",
        init_checkpoint=tmp_path / "stage210d-admission.pt",
        resume=False,
        smoke=False,
    )

    assert config["max_steps"] == 30_064
    assert config["save_every"] == 10_000
    assert config["step_eval_every"] == 10_000


def test_stage210d_resume_preserves_optimizer_state(tmp_path: Path) -> None:
    config = stage210d._config(
        phase="admission",
        output_dir=tmp_path / "run",
        init_checkpoint=tmp_path / "unused.pt",
        resume=True,
        smoke=True,
    )

    assert config["init_checkpoint_path"] is None
    assert config["resume_from"] == "latest"


def test_stage210d_direct_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210b.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210d_balanced_full_ctc_distill.py"),
            "--dry-run",
            "--smoke",
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
