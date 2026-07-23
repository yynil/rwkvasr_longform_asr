from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210k = importlib.import_module("scripts.run_stage210k_timemixer_hidden_realign")


def test_stage210k_isolates_teacher_forced_timemixer_alignment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    segment = stage210k._segments(smoke=False)[0]
    config = stage210k._config(
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        bucket_manifest=manifest,
        resume=False,
        smoke=False,
    )

    assert config["max_steps"] == 1_000
    assert config["save_every"] == 250
    assert config["step_eval_every"] == 250
    assert config["webdataset_bucket_manifest_path"] == str(manifest)
    assert config["bucket_source_interleave"] is True
    assert config["lr"] == pytest.approx(3.0e-6)
    assert config["weight_decay"] == 0.0
    assert config["freeze_encoder"] is False
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["ctc_teacher_online_layer_input_mode"] == "teacher_forced"
    assert config["ctc_teacher_online_layer_sample_count"] == 8
    assert config["ctc_teacher_online_layer_include_boundaries"] is False
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == 0.0
    assert config["ctc_teacher_online_hidden_frame_balance_mode"] == "all"
    assert config["ctc_teacher_online_blank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_conditional_nonblank_hard_loss_weight"] == 0.0
    assert config["ctc_teacher_online_encoder_loss_weight"] == 0.0
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["step_eval_cache_batches"] is True


def test_stage210k_direct_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210k_timemixer_hidden_realign.py"),
            "--dry-run",
            "--smoke",
            "--init-checkpoint",
            str(checkpoint),
            "--bucket-manifest",
            str(manifest),
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
    assert "phase=timemixer_hidden" in result.stdout
    assert "target_step=2" in result.stdout
    assert f"bucket_manifest={manifest}" in result.stdout
