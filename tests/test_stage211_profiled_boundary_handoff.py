from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from rwkvasr.config import save_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
handoff = importlib.import_module("scripts.run_stage211_profiled_boundary_handoff")


def test_profiled_controller_binds_only_fresh_supplemental_admission(
    tmp_path: Path,
) -> None:
    admission = tmp_path / "admission.json"
    command = handoff._profiled_controller_command(
        initial_checkpoint=tmp_path / "initial.pt",
        output_root=tmp_path / "runs",
        config_root=tmp_path / "configs",
        metadata_root=tmp_path / "metadata",
        easy_manifest=tmp_path / "easy.json",
        nano_checkpoint=tmp_path / "model.pt",
        inventory=tmp_path / "inventory.json",
        profile_receipt=tmp_path / "profile.json",
        admission_path=admission,
        master_port=29631,
    )

    assert command[command.index("--batch-profile-admission") + 1] == (
        f"supplemental_natural={admission}"
    )
    assert command[command.index("--supplemental-inventory") + 1] == str(
        tmp_path / "inventory.json"
    )
    assert "--promotion-receipt" not in command


def test_boundary_preflight_keeps_all_three_default_profiles(tmp_path: Path) -> None:
    command = handoff._preflight_command(
        base_config=tmp_path / "base.yaml",
        init_checkpoint=tmp_path / "long.pt",
        output_root=tmp_path / "probe",
        master_port=29731,
    )

    assert "--profile" not in command
    assert command[command.index("--phase") + 1] == "mixer"
    assert command[command.index("--max-peak-memory-gib") + 1] == "22.0"
    assert command[command.index("--min-improvement-ratio") + 1] == "0.10"
    assert command[command.index("--max-loss-regression-ratio") + 1] == "0.05"
    assert command[command.index("--max-cosine-regression") + 1] == "0.005"


def test_template_config_selection_requires_one_exact_step_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mixer" / "full_supplemental_natural"
    first = root / "first.yaml"
    second = root / "second.yaml"
    save_yaml(first, {"max_steps": 123})
    save_yaml(second, {"max_steps": 456})

    assert handoff._find_template_config(tmp_path, expected_steps=456) == second.resolve()

    save_yaml(first, {"max_steps": 456})
    with pytest.raises(ValueError, match="ambiguous"):
        handoff._find_template_config(tmp_path, expected_steps=456)


def test_supervisor_replacement_resumes_after_mixer(tmp_path: Path) -> None:
    command = handoff._supervisor_command(
        session="stage211-supervisor",
        supervisor_log=tmp_path / "supervisor.log",
    )

    assert "START_STAGE=post_mixer" in command
    assert "REUSE_COMPLETED_CALIBRATION_EVAL=1" in command
    assert command[command.index("-s") + 1] == "stage211-supervisor"


def test_long_receipt_binds_exact_hard_to_long_chain(tmp_path: Path) -> None:
    command = handoff._long_receipt_command(
        phase_root=tmp_path / "phase",
        long_manifest=tmp_path / "long-manifest.json",
    )

    hard_steps = handoff.STAGE211_AUDIO_CURRICULUM["hard"]["steps"]
    long_steps = handoff.STAGE211_AUDIO_CURRICULUM["long"]["steps"]
    assert command[command.index("--init-checkpoint") + 1].endswith(
        f"hard/step-{hard_steps}.pt"
    )
    assert command[command.index("--completion-checkpoint") + 1].endswith(
        f"long/step-{long_steps}.pt"
    )
