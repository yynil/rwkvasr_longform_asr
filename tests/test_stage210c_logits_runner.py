from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210c = importlib.import_module("scripts.run_stage210c_ctc_logits_distill")
PHASES = stage210c.PHASES
_phase_config = stage210c._phase_config
_segments = stage210c._segments


def test_stage210c_nonblank_keeps_nano_output_path_frozen(tmp_path: Path) -> None:
    phase = PHASES["nonblank"]
    segment = _segments(phase, smoke=True)[0]
    checkpoint = tmp_path / "stage210b.pt"
    checkpoint.touch()
    config = _phase_config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["feature_extractor_type"] == "funasr_wav_frontend"
    assert config["dropout"] == 0.0
    assert config["batch_size"] == 12
    assert config["length_bucket_frame_budget"] == 8_000
    assert config["init_checkpoint_path"] == str(checkpoint)
    assert config["funasr_nano_ctc_init_checkpoint_path"] is None
    assert config["funasr_nano_ctc_init_load_rwkv_encoder_from_qkv"] is False
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["ctc_teacher_online_keep_full_log_probs_on_device"] is True
    assert config["ctc_teacher_online_keep_layer_hiddens_on_device"] is True
    assert config["ctc_teacher_online_full_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_full_frame_filter"] == "nonblank_neighbors"
    assert config["ctc_teacher_online_blank_loss_weight"] > 0.0
    assert config["ctc_teacher_online_mass_loss_weight"] > 0.0
    assert config["ctc_teacher_online_layer_input_mode"] == "stacked"
    assert config["ctc_teacher_online_layer_block_loss_weight"] > 0.0
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_window_loss_weight"] == 0.0


def test_stage210c_full_uses_three_epoch_all_split_curriculum(tmp_path: Path) -> None:
    phase = PHASES["full"]
    segments = _segments(phase, smoke=False)
    checkpoint = tmp_path / "stage210c-nonblank.pt"
    checkpoint.touch()
    config = _phase_config(
        phase=phase,
        segment=segments[0],
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=False,
        smoke=False,
    )

    assert len(segments) == 12
    assert [segment["split"].name for segment in segments[:4]] == ["easy", "medium", "hard", "long"]
    assert [segment["epoch"] for segment in segments[::4]] == [1, 2, 3]
    assert all(
        int(right["target_step"]) > int(left["target_step"])
        for left, right in zip(segments, segments[1:], strict=False)
    )
    assert config["ctc_teacher_online_full_frame_filter"] == "all"
    assert config["ctc_teacher_online_blank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_mass_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_block_loss_weight"] > 0.0
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True


def test_stage210c_resume_preserves_checkpoint_state(tmp_path: Path) -> None:
    phase = PHASES["nonblank"]
    segment = _segments(phase, smoke=True)[0]
    config = _phase_config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=tmp_path / "unused.pt",
        resume=True,
        smoke=True,
    )

    assert config["init_checkpoint_path"] is None
    assert config["resume_from"] == "latest"
    assert config["funasr_nano_ctc_init_checkpoint_path"] is None


def test_stage210c_direct_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210b.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210c_ctc_logits_distill.py"),
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
    assert "phase=nonblank" in result.stdout
    assert "target_step=2" in result.stdout
