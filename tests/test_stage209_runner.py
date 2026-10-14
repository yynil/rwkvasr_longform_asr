from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from rwkvasr.config import save_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage209 = importlib.import_module("scripts.run_stage209_emission_margin_online_ctc_distill")
RAMP_PHASES = stage209.RAMP_PHASES
_latest_exported_checkpoint = stage209._latest_exported_checkpoint
_phase_config = stage209._phase_config


def test_stage209_ramp_has_long_monotonic_phases() -> None:
    assert [phase.target_step for phase in RAMP_PHASES] == [5_000, 10_000, 15_000, 20_000]
    for field in ("sequence_weight", "nonblank_window_margin_weight"):
        values = [float(getattr(phase, field)) for phase in RAMP_PHASES]
        assert values == sorted(values)
        assert values[0] > 0.0
    assert RAMP_PHASES[-1].sequence_weight == pytest.approx(1.0)
    assert RAMP_PHASES[-1].nonblank_window_margin_weight == pytest.approx(0.5)


def test_stage209_phase_config_targets_local_blank_competition(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-2000.pt"
    checkpoint.touch()
    config = _phase_config(
        phase=RAMP_PHASES[-1],
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=False,
        smoke=False,
    )

    assert config["init_checkpoint_path"] == str(checkpoint)
    assert config["resume_from"] is None
    assert config["ctc_teacher_online_loss_weight"] == pytest.approx(2.0)
    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.02)
    assert config["ctc_teacher_online_mass_loss_weight"] == pytest.approx(0.02)
    assert config["ctc_teacher_online_sequence_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_nonblank_margin"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_nonblank_window_topk_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_nonblank_window_margin_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_nonblank_window_radius"] == 2
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_encoder_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_presence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_window_loss_weight"] == 0.0
    assert config["batch_token_budget"] == 8000
    assert config["length_bucket_frame_budget"] == 8000
    assert config["max_steps"] == 20_000
    assert config["save_every"] == 5_000
    assert config["step_eval_every"] == 5_000
    assert config["step_eval_samples"] == 512
    assert config["epochs"] is None
    assert "stage179b_medium" in config["webdataset_length_index_path"]

    # Stage209 changes the objective and duration, not the BiRWKV architecture.
    assert config["frontend_type"] == "sensevoice_rwkv"
    assert config["num_layers"] == 70
    assert config["head_size"] == 64
    assert config["sensevoice_tp_blocks"] == 20


def test_stage209_resume_uses_deepspeed_state_not_init_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-2000.pt"
    checkpoint.touch()
    config = _phase_config(
        phase=RAMP_PHASES[1],
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        resume=True,
        smoke=False,
    )

    assert config["init_checkpoint_path"] is None
    assert config["resume_from"] == "latest"
    assert config["max_steps"] == 10_000


def test_stage209_resolves_stage208_exported_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-2000.pt"
    checkpoint.touch()
    save_yaml(
        tmp_path / "latest_checkpoint.yaml",
        {"step": 2000, "checkpoint_path": str(checkpoint)},
    )

    assert _latest_exported_checkpoint(tmp_path) == checkpoint


def test_stage209_direct_script_entrypoint_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-2000.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage209_emission_margin_online_ctc_distill.py"),
            "--dry-run",
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
    assert "target_step=5000" in result.stdout
