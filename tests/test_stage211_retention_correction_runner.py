from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
runner = importlib.import_module("scripts.run_stage211_retention_correction")


def test_stage211_correction_segment_is_one_complete_low_lr_round() -> None:
    segment = runner._correction_segment(round_index=2, steps_per_epoch=1234)

    assert segment["name"] == "mixer_retention_round2_1234steps"
    assert segment["difficulty"] == "retention_round_02"
    assert segment["target_step"] == 1234
    assert segment["split_steps"] == 1234


def test_stage211_correction_admission_requires_immediate_failed_gate(
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

    admitted, admitted_teacher = runner._admit_failed_gate(
        admission_gate_path=gate_path,
        init_checkpoint=checkpoint,
        round_index=2,
    )
    assert admitted is gate
    assert admitted_teacher == teacher_sha256

    gate["gate_passed"] = True
    with pytest.raises(ValueError, match="explicitly failed Mixer gate"):
        runner._admit_failed_gate(
            admission_gate_path=gate_path,
            init_checkpoint=checkpoint,
            round_index=2,
        )
    gate["gate_passed"] = False
    gate["full_data_coverage"]["post_coverage_corrections"] = []
    with pytest.raises(ValueError, match="immediately follow"):
        runner._admit_failed_gate(
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
    for path in (
        replay_receipt,
        replay_manifest,
        gate,
        init_checkpoint,
        nano_checkpoint,
        smoke_marker,
    ):
        path.write_bytes(path.name.encode())
    run_dir = tmp_path / "run"

    provenance = runner._provenance_payload(
        round_index=1,
        run_dir=run_dir,
        replay_receipt=replay_receipt,
        replay_manifest=replay_manifest,
        admission_gate=gate,
        init_checkpoint=init_checkpoint,
        nano_checkpoint=nano_checkpoint,
        smoke_marker=smoke_marker,
        steps_per_epoch=99,
    )

    assert provenance["artifact"] == "retention_correction_run"
    assert provenance["round"] == 1
    assert provenance["steps_per_epoch"] == 99
    assert provenance["learning_rate"] == runner.CORRECTION_LR
    assert provenance["trainable_boundary"] == "mixer_only"
    assert provenance["early_stopping"] is False
    assert len(provenance["admission_gate_sha256"]) == 64
    assert provenance["smoke_marker_sha256"] == runner.sha256_file(smoke_marker)
