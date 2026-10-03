from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

from rwkvasr.config import save_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
handoff = importlib.import_module("scripts.run_stage211_profiled_boundary_handoff")


def test_profiled_handoff_preserves_virtualenv_python() -> None:
    assert handoff.PYTHON == Path(sys.executable)


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


def test_profiled_controller_omits_admission_for_retained_legacy_baseline(
    tmp_path: Path,
) -> None:
    command = handoff._profiled_controller_command(
        initial_checkpoint=tmp_path / "initial.pt",
        output_root=tmp_path / "runs",
        config_root=tmp_path / "configs",
        metadata_root=tmp_path / "metadata",
        easy_manifest=tmp_path / "easy.json",
        nano_checkpoint=tmp_path / "model.pt",
        inventory=tmp_path / "inventory.json",
        profile_receipt=tmp_path / "profile.json",
        admission_path=None,
        master_port=29631,
    )

    assert "--batch-profile-admission" not in command
    assert command[command.index("--master-port") + 1] == "29631"


def test_supplemental_profile_admission_routing_is_fail_closed() -> None:
    retained_baseline = {
        "selection_decision": "keep_baseline",
        "selected_profile_row": {
            "profile": {
                "name": "baseline",
                "batch_size": 36,
                "frame_budget": 24_000,
                "num_workers": 8,
                "gradient_checkpointing": False,
            }
        },
    }
    admitted_candidate = {
        **retained_baseline,
        "selection_decision": "admit_candidate",
    }
    wrong_retained_baseline = {
        **retained_baseline,
        "selected_profile_row": {
            "profile": {
                "name": "baseline",
                "batch_size": 12,
                "frame_budget": 8_000,
                "num_workers": 8,
                "gradient_checkpointing": False,
            }
        },
    }
    wrong_retained_workers = {
        **retained_baseline,
        "selected_profile_row": {
            "profile": {
                "name": "baseline",
                "batch_size": 36,
                "frame_budget": 24_000,
                "num_workers": 2,
                "gradient_checkpointing": False,
            }
        },
    }
    wrong_retained_checkpointing = {
        **retained_baseline,
        "selected_profile_row": {
            "profile": {
                "name": "baseline",
                "batch_size": 36,
                "frame_budget": 24_000,
                "num_workers": 8,
                "gradient_checkpointing": True,
            }
        },
    }

    assert not handoff._supplemental_profile_requires_admission(retained_baseline)
    assert handoff._supplemental_profile_requires_admission(admitted_candidate)
    with pytest.raises(ValueError, match="differs from the formal default"):
        handoff._supplemental_profile_requires_admission(wrong_retained_baseline)
    with pytest.raises(ValueError, match="differs from the formal default"):
        handoff._supplemental_profile_requires_admission(wrong_retained_workers)
    with pytest.raises(ValueError, match="differs from the formal default"):
        handoff._supplemental_profile_requires_admission(wrong_retained_checkpointing)


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


def test_completed_boundary_reuses_existing_template_without_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_root = tmp_path / "runs" / "phase"
    provenance = phase_root / "supplemental_natural" / "stage211_provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}\n", encoding="utf-8")
    template = tmp_path / "configs" / "mixer" / "full_supplemental_natural" / "base.yaml"
    save_yaml(template, {"max_steps": 456})
    monkeypatch.setattr(
        handoff,
        "_run",
        lambda *args, **kwargs: pytest.fail("completed boundary must not rerun strict template"),
    )

    selected = handoff._ensure_template_config(
        phase_root=phase_root,
        config_root=tmp_path / "configs",
        manifest=tmp_path / "manifest.json",
        inventory=tmp_path / "inventory.json",
        nano_checkpoint=tmp_path / "nano.pt",
        master_port=29631,
        expected_steps=456,
        log_path=tmp_path / "handoff.log",
    )

    assert selected == template.resolve()


def test_completed_boundary_rejects_missing_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_root = tmp_path / "runs" / "phase"
    provenance = phase_root / "supplemental_natural" / "stage211_provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        handoff,
        "_run",
        lambda *args, **kwargs: pytest.fail("recovery must fail before strict template runner"),
    )

    with pytest.raises(ValueError, match="missing after formal provenance"):
        handoff._ensure_template_config(
            phase_root=phase_root,
            config_root=tmp_path / "configs",
            manifest=tmp_path / "manifest.json",
            inventory=tmp_path / "inventory.json",
            nano_checkpoint=tmp_path / "nano.pt",
            master_port=29631,
            expected_steps=456,
            log_path=tmp_path / "handoff.log",
        )


def test_fresh_boundary_generates_missing_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase_root = tmp_path / "runs" / "phase"
    config_root = tmp_path / "configs"
    template = config_root / "mixer" / "full_supplemental_natural" / "base.yaml"
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, log_path: Path | None = None) -> None:
        commands.append(command)
        save_yaml(template, {"max_steps": 456})

    monkeypatch.setattr(handoff, "_run", fake_run)

    selected = handoff._ensure_template_config(
        phase_root=phase_root,
        config_root=config_root,
        manifest=tmp_path / "manifest.json",
        inventory=tmp_path / "inventory.json",
        nano_checkpoint=tmp_path / "nano.pt",
        master_port=29631,
        expected_steps=456,
        log_path=tmp_path / "handoff.log",
    )

    assert selected == template.resolve()
    assert len(commands) == 1
    assert str(handoff.STRICT_RUNNER) in commands[0]


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
    assert command[command.index("--init-checkpoint") + 1].endswith(f"hard/step-{hard_steps}.pt")
    assert command[command.index("--completion-checkpoint") + 1].endswith(
        f"long/step-{long_steps}.pt"
    )


def test_retention_barrier_deep_validates_both_receipts_before_preflight(
    tmp_path: Path,
) -> None:
    stratified = tmp_path / "stratified.json"
    replay = tmp_path / "replay.json"
    commands = handoff._retention_validation_commands(
        stratified_hidden_receipt=stratified,
        retention_replay_receipt=replay,
    )

    assert commands[0][-2:] == ["--stratified-receipt", str(stratified)]
    assert commands[1][-2:] == ["--receipt", str(replay)]
    assert str(handoff.SUPPLEMENTAL_RETENTION_VALIDATOR) in commands[0]
    assert str(handoff.RETENTION_REPLAY_VALIDATOR) in commands[1]

    source = inspect.getsource(handoff.main)
    wait_index = source.index("(stratified_hidden_receipt, retention_replay_receipt)")
    validation_index = source.index("_retention_validation_commands(")
    template_index = source.index("_ensure_template_config(")
    preflight_index = source.index("_preflight_command(")
    assert wait_index < validation_index < template_index < preflight_index
