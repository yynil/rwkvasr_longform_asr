from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rwkvasr.config import save_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage208 = importlib.import_module("scripts.run_stage208_stronger_online_ctc_distill")
RAMP_PHASES = stage208.RAMP_PHASES
_latest_exported_checkpoint = stage208._latest_exported_checkpoint
_phase_config = stage208._phase_config


def test_stage208_ramp_is_monotonic_and_bounded() -> None:
    assert [phase.target_step for phase in RAMP_PHASES] == [500, 1000, 1500, 2000]
    for field in (
        "sequence_weight",
        "nonblank_window_weight",
        "nonblank_window_topk_weight",
        "nonblank_window_margin_weight",
    ):
        values = [float(getattr(phase, field)) for phase in RAMP_PHASES]
        assert values == sorted(values)
        assert values[0] > 0.0
        assert values[-1] <= 0.5


def test_stage208_phase_config_keeps_safe_base_objective(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-710000.pt"
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
    assert config["ctc_teacher_online_sequence_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_nonblank_window_topk_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_nonblank_window_margin_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_presence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_window_loss_weight"] == 0.0
    assert config["batch_token_budget"] == 8000
    assert config["length_bucket_frame_budget"] == 8000
    assert config["max_steps"] == 2000
    assert config["save_every"] == 500
    assert "stage179b_medium" in config["webdataset_length_index_path"]


def test_stage208_resume_uses_deepspeed_state_not_init_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-710000.pt"
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
    assert config["max_steps"] == 1000


def test_stage208_resolves_stage206_exported_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-710000.pt"
    checkpoint.touch()
    save_yaml(
        tmp_path / "latest_checkpoint.yaml",
        {"step": 710000, "checkpoint_path": str(checkpoint)},
    )

    assert _latest_exported_checkpoint(tmp_path) == checkpoint
