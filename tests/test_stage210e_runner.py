from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210e = importlib.import_module("scripts.run_stage210e_path_state_ctc_distill")


def test_stage210e_admission_aligns_complete_frozen_ctc_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "incumbent.pt"
    checkpoint.touch()
    segment = stage210e._segments("admission", smoke=False)[0]
    config = stage210e._config(
        phase="admission",
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["max_steps"] == 2_000
    assert config["save_every"] == 1_000
    assert config["step_eval_every"] == 1_000
    assert config["init_checkpoint_path"] == str(checkpoint)
    assert config["weight_decay"] == 0.0
    assert config["ctc_teacher_online_full_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_full_frame_filter"] == "all"
    assert config["ctc_teacher_online_full_nonblank_weight"] == pytest.approx(4.0)
    assert config["ctc_teacher_online_full_frame_weight_mode"] == "posterior_nonblank"
    assert config["ctc_teacher_online_encoder_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.125)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.0625)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(0.25)
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True


def test_stage210e_curriculum_covers_all_data_for_three_epochs() -> None:
    segments = stage210e._segments("curriculum", smoke=False)

    assert len(segments) == 12
    assert [segment["split"].name for segment in segments[:4]] == [
        "easy",
        "medium",
        "hard",
        "long",
    ]
    assert [segment["epoch"] for segment in segments[::4]] == [1, 2, 3]
    assert all(
        int(right["target_step"]) > int(left["target_step"])
        for left, right in zip(segments, segments[1:], strict=False)
    )


def test_stage210e_resume_preserves_optimizer_state(tmp_path: Path) -> None:
    segment = stage210e._segments("admission", smoke=True)[0]
    config = stage210e._config(
        phase="admission",
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=tmp_path / "unused.pt",
        resume=True,
        smoke=True,
    )

    assert config["init_checkpoint_path"] is None
    assert config["resume_from"] == "latest"


def test_stage210e_direct_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "incumbent.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210e_path_state_ctc_distill.py"),
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
