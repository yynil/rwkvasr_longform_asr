from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage210j = importlib.import_module("scripts.run_stage210j_decoupled_hidden_ctc_distill")


def test_stage210j_decouples_blank_and_hidden_alignment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210h.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    segment = stage210j._segments(smoke=False)[0]
    config = stage210j._config(
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        bucket_manifest=manifest,
        resume=False,
        smoke=False,
    )

    assert config["max_steps"] == 2_000
    assert config["save_every"] == 500
    assert config["webdataset_bucket_manifest_path"] == str(manifest)
    assert config["bucket_source_interleave"] is True
    assert config["lr"] == pytest.approx(2.0e-7)
    assert config["ctc_teacher_online_frame_balance_mode"] == "all"
    assert config["ctc_teacher_online_blank_frame_balance_mode"] == "all"
    assert config["ctc_teacher_online_hidden_frame_balance_mode"] == (
        "teacher_top1_balanced"
    )
    assert config["ctc_teacher_online_encoder_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_layer_sample_count"] == 24
    assert config["ctc_teacher_online_layer_include_boundaries"] is True
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True


def test_stage210j_direct_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage210h.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage210j_decoupled_hidden_ctc_distill.py"),
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
    assert "phase=admission" in result.stdout
    assert "target_step=2" in result.stdout
    assert f"bucket_manifest={manifest}" in result.stdout
