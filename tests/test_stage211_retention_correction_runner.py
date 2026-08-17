from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
runner = importlib.import_module("scripts.run_stage211_retention_correction")


def _batch_profile(
    tmp_path: Path,
    *,
    batch_size: int = 36,
    frame_budget: int = 24_000,
    gradient_checkpointing: bool = False,
):
    report = tmp_path / "batch-profile.json"
    report.write_bytes(b"batch-profile")
    return {
        "report_path": str(report.resolve()),
        "report_sha256": runner.sha256_file(report),
        "selected_profile_name": "selected",
        "selected_profile_row": {
            "profile": {
                "name": "selected",
                "batch_size": batch_size,
                "frame_budget": frame_budget,
                "num_workers": 8,
                "gradient_checkpointing": gradient_checkpointing,
            }
        },
    }


def test_stage211_correction_segment_is_one_complete_low_lr_round() -> None:
    segment = runner._correction_segment(round_index=2, steps_per_epoch=1234)

    assert segment["name"] == "mixer_retention_round2_1234steps"
    assert segment["difficulty"] == "retention_round_02"
    assert segment["target_step"] == 1234
    assert segment["split_steps"] == 1234


@pytest.mark.parametrize(
    ("phase", "expected_lr"),
    (("block", 1.0e-6), ("logits", 1.5e-7)),
)
def test_stage211_phase_correction_uses_phase_objective_and_lr(
    phase: str,
    expected_lr: float,
) -> None:
    segment = runner._correction_segment(
        round_index=1,
        steps_per_epoch=321,
        phase=phase,
    )

    assert segment["name"] == f"{phase}_correction_round1_321steps"
    assert segment["target_step"] == 321
    provenance_lr = runner.stage211_post_coverage_correction_lr(phase)
    assert provenance_lr == expected_lr


def test_stage211_correction_admission_supports_mandatory_early_pass_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_path = tmp_path / "gate.json"
    checkpoint = tmp_path / "checkpoint.pt"
    gate_path.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    teacher_sha256 = "a" * 64
    gate = {
        "gate_passed": False,
        "full_data_coverage": {
            "segments": [{"nano_teacher_checkpoint_sha256": teacher_sha256}],
            "post_coverage_corrections": [{"round": 1}],
        },
    }
    monkeypatch.setattr(
        runner,
        "validate_stage211_phase_gate_report",
        lambda *args, **kwargs: gate,
    )

    admitted, admitted_teacher, admission_mode = runner._admit_gate(
        admission_gate_path=gate_path,
        init_checkpoint=checkpoint,
        round_index=2,
    )
    assert admitted is gate
    assert admitted_teacher == teacher_sha256
    assert admission_mode == "failed_gate"

    gate["gate_passed"] = True
    admitted, admitted_teacher, admission_mode = runner._admit_gate(
        admission_gate_path=gate_path,
        init_checkpoint=checkpoint,
        round_index=2,
    )
    assert admitted is gate
    assert admitted_teacher == teacher_sha256
    assert admission_mode == "early_pass_mandatory_continuation"

    gate["full_data_coverage"]["post_coverage_corrections"] = [
        {"round": 1},
        {"round": 2},
    ]
    _, _, admission_mode = runner._admit_gate(
        admission_gate_path=gate_path,
        init_checkpoint=checkpoint,
        round_index=3,
    )
    assert admission_mode == "early_pass_mandatory_continuation"

    gate["full_data_coverage"]["post_coverage_corrections"].append({"round": 3})
    with pytest.raises(ValueError, match="failed gate or an early passing gate"):
        runner._admit_gate(
            admission_gate_path=gate_path,
            init_checkpoint=checkpoint,
            round_index=4,
        )

    gate["full_data_coverage"]["post_coverage_corrections"] = []
    with pytest.raises(ValueError, match="failed gate or an early passing gate"):
        runner._admit_gate(
            admission_gate_path=gate_path,
            init_checkpoint=checkpoint,
            round_index=1,
        )

    gate["gate_passed"] = False
    gate["full_data_coverage"]["post_coverage_corrections"] = []
    with pytest.raises(ValueError, match="immediately follow"):
        runner._admit_gate(
            admission_gate_path=gate_path,
            init_checkpoint=checkpoint,
            round_index=2,
        )


def test_stage211_correction_provenance_binds_all_admission_inputs(
    tmp_path: Path,
) -> None:
    replay_receipt = tmp_path / "replay.json"
    replay_manifest = tmp_path / "manifest.json"
    gate = tmp_path / "gate.json"
    init_checkpoint = tmp_path / "init.pt"
    nano_checkpoint = tmp_path / "model.pt"
    smoke_marker = tmp_path / "smoke.json"
    layer_focus = tmp_path / "focus.json"
    for path in (
        replay_receipt,
        replay_manifest,
        gate,
        init_checkpoint,
        nano_checkpoint,
        smoke_marker,
        layer_focus,
    ):
        path.write_bytes(path.name.encode())
    run_dir = tmp_path / "run"
    batch_profile = _batch_profile(tmp_path)

    provenance = runner._provenance_payload(
        round_index=1,
        run_dir=run_dir,
        replay_receipt=replay_receipt,
        replay_manifest=replay_manifest,
        admission_gate=gate,
        admission_mode="failed_gate",
        init_checkpoint=init_checkpoint,
        nano_checkpoint=nano_checkpoint,
        smoke_marker=smoke_marker,
        layer_focus=layer_focus,
        layer_rotation_offset=0,
        batch_profile_preflight=batch_profile,
        batch_profile_admission=None,
        extension_decision=None,
        steps_per_epoch=99,
    )

    assert provenance["artifact"] == "retention_correction_run"
    assert provenance["schema_version"] == 2
    assert provenance["round"] == 1
    assert provenance["steps_per_epoch"] == 99
    assert provenance["learning_rate"] == runner.CORRECTION_LR
    assert provenance["trainable_boundary"] == "mixer_only"
    assert provenance["early_stopping"] is False
    assert provenance["admission_mode"] == "failed_gate"
    assert len(provenance["admission_gate_sha256"]) == 64
    assert provenance["smoke_marker_sha256"] == runner.sha256_file(smoke_marker)
    assert provenance["layer_focus_sha256"] == runner.sha256_file(layer_focus)
    assert provenance["layer_rotation_offset"] == 0
    assert provenance["batch_profile_preflight_sha256"] == batch_profile["report_sha256"]
    assert provenance["batch_size"] == 36
    assert provenance["frame_budget"] == 24_000
    assert provenance["num_workers"] == 8
    assert provenance["gradient_checkpointing"] is False
    assert provenance["correction_extension_decision_path"] is None
    assert provenance["correction_extension_decision_sha256"] is None


@pytest.mark.parametrize(
    ("phase", "expected_lr"),
    (("block", 1.0e-6), ("logits", 1.5e-7)),
)
def test_stage211_downstream_correction_provenance_records_phase_objective_with_frozen_nano_path(
    tmp_path: Path,
    phase: str,
    expected_lr: float,
) -> None:
    inputs = {
        name: tmp_path / name
        for name in (
            "replay.json",
            "manifest.json",
            "gate.json",
            "init.pt",
            "model.pt",
            "smoke.json",
            "focus.json",
        )
    }
    for path in inputs.values():
        path.write_bytes(path.name.encode())
    batch_profile = _batch_profile(tmp_path, gradient_checkpointing=True)

    provenance = runner._provenance_payload(
        round_index=2,
        run_dir=tmp_path / "run",
        replay_receipt=inputs["replay.json"],
        replay_manifest=inputs["manifest.json"],
        admission_gate=inputs["gate.json"],
        admission_mode="early_pass_mandatory_continuation",
        init_checkpoint=inputs["init.pt"],
        nano_checkpoint=inputs["model.pt"],
        smoke_marker=inputs["smoke.json"],
        layer_focus=inputs["focus.json"],
        layer_rotation_offset=24,
        batch_profile_preflight=batch_profile,
        batch_profile_admission=None,
        extension_decision=None,
        steps_per_epoch=321,
        phase=phase,
    )

    assert provenance["phase"] == phase
    assert provenance["learning_rate"] == expected_lr
    assert provenance["trainable_boundary"] == "mixer_only"
    assert provenance["early_stopping"] is False
    assert provenance["admission_mode"] == "early_pass_mandatory_continuation"
    assert provenance["gradient_checkpointing"] is True
