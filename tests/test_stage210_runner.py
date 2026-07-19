from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210 = importlib.import_module("scripts.run_stage210_hierarchical_encoder_distill")
PHASES = stage210.PHASES
_phase_config = stage210._phase_config


def test_stage210_subblock_uses_exact_features_and_nano_initialization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage209.pt"
    nano_checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    nano_checkpoint.touch()
    config = _phase_config(
        phase=PHASES["subblock"],
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        nano_checkpoint=nano_checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["feature_extractor_type"] == "funasr_wav_frontend"
    assert config["frontend_type"] == "sensevoice_rwkv"
    assert config["num_layers"] == 70
    assert config["dropout"] == 0.0
    assert config["specaugment_enabled"] is False
    assert config["ctc_teacher_online_use_batch_features"] is True
    assert config["ctc_teacher_online_keep_layer_hiddens_on_device"] is True
    assert config["ctc_teacher_online_layer_frame_tolerance"] == 0
    assert config["ctc_teacher_online_layer_input_mode"] == "teacher_forced"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_energy_mse_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_log_rms_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_raw_mse_weight"] == 0.0
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_loss_weight"] == 0.0
    assert config["funasr_nano_ctc_init_checkpoint_path"] == str(nano_checkpoint)
    assert config["funasr_nano_ctc_init_load_rwkv_encoder_from_qkv"] is True
    assert config["funasr_nano_ctc_init_load_decoder"] is True
    assert config["funasr_nano_ctc_init_load_head"] is True
    assert config["ctc_decoder_type"] == "funasr_nano_transformer"
    assert config["ctc_decoder_num_layers"] == 5
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["epochs"] == 1
    assert config["max_steps"] is None
    assert config["save_every"] == 10_000
    assert config["step_eval_every"] == 10_000
    assert config["step_eval_shuffle"] is False
    assert config["step_eval_at_start"] is True
    assert config["wandb_enabled"] is True
    assert config["deepspeed"]["gradient_accumulation_steps"] == 1
    assert "stage179a_easy" in config["webdataset_length_index_path"]


def test_stage210_block_requires_a_passing_checkpoint_without_reinitializing(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210a.pt"
    nano_checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    nano_checkpoint.touch()
    config = _phase_config(
        phase=PHASES["block"],
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        nano_checkpoint=nano_checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["init_checkpoint_path"] == str(checkpoint)
    assert config["funasr_nano_ctc_init_checkpoint_path"] is None
    assert config["funasr_nano_ctc_init_load_rwkv_encoder_from_qkv"] is False
    assert config["freeze_encoder_except_time_mixer"] is False
    assert config["ctc_teacher_online_layer_input_mode"] == "stacked"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_layer_energy_mse_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_log_rms_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_raw_mse_weight"] == 0.0


def test_stage210_resume_uses_deepspeed_state_without_reapplying_nano(tmp_path: Path) -> None:
    config = _phase_config(
        phase=PHASES["subblock"],
        output_dir=tmp_path / "run",
        init_checkpoint=tmp_path / "unused.pt",
        nano_checkpoint=tmp_path / "model.pt",
        resume=True,
        smoke=False,
    )

    assert config["init_checkpoint_path"] is None
    assert config["resume_from"] == "latest"
    assert config["funasr_nano_ctc_init_checkpoint_path"] is None
    assert config["funasr_nano_ctc_init_load_rwkv_encoder_from_qkv"] is False


def test_stage210_direct_script_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage209.pt"
    nano_checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    nano_checkpoint.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210_hierarchical_encoder_distill.py"),
            "--dry-run",
            "--smoke",
            "--init-checkpoint",
            str(checkpoint),
            "--nano-checkpoint",
            str(nano_checkpoint),
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
    assert "phase=subblock" in result.stdout
    assert "estimated_steps=2" in result.stdout
