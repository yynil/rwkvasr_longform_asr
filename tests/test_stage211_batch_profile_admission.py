from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_batch_profile import (
    build_stage211_batch_profile_admission,
    validate_stage211_batch_profile_admission,
    validate_stage211_batch_profile_preflight,
)
from rwkvasr.eval.stage211_gate import (
    _validate_stage211_segment_batch_profile,
    build_stage211_full_data_coverage,
    sha256_file,
    stage211_phase_train_config_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage211_full_phase = importlib.import_module(
    "scripts.run_stage211_full_phase_curriculum"
)


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path.resolve()


def _profile_row(
    root: Path,
    *,
    name: str,
    batch_size: int,
    frame_budget: int,
    steps_per_epoch: int,
    projected_seconds: float,
    loss: float,
    cosine: float,
    phase: str,
    init_checkpoint: Path,
    manifest: Path,
) -> dict[str, object]:
    profile_root = root / name
    config_path = profile_root / "train_config.yaml"
    config = {
        **stage211_phase_train_config_contract(phase),
        "stage211_batch_profile_probe_phase": phase,
        "max_steps": 120,
        "batch_size": batch_size,
        "batch_token_budget": frame_budget,
        "length_bucket_frame_budget": frame_budget,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "init_checkpoint_path": str(init_checkpoint),
        "webdataset_bucket_manifest_path": str(manifest),
        "resume_from": None,
        "resume_tag": None,
        "wandb_enabled": False,
        "step_eval_every": None,
        "save_deepspeed_sharded_checkpoints": False,
        "deepspeed": {
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": batch_size,
            "train_batch_size": batch_size * 4,
        },
    }
    save_yaml(config_path, config)
    log_path = _write(profile_root / "train.log", "measured\n")
    full_steps = steps_per_epoch * 3
    rate = full_steps / projected_seconds
    required = ["online_layer_match"]
    summary = {
        "process_ok": True,
        "complete_steps": True,
        "measurement_steps": 100,
        "steps_per_second": rate,
        "mean_loss": loss,
        "mean_cosine": cosine,
        "required_match_fields": required,
        "missing_required_match_fields": {},
        "incomplete_required_match_fields": {},
        "complete_teacher_matches": True,
        "missing_total": 0,
        "max_abs_frame_delta": 0,
        "peak_memory_gib": 12.0,
        "memory_monitor_ok": True,
        "projected_full_coverage_seconds": projected_seconds,
        "safety_pass": True,
    }
    return {
        "profile": {
            "name": name,
            "batch_size": batch_size,
            "frame_budget": frame_budget,
        },
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "return_code": 0,
        "required_match_fields": required,
        "coverage": {
            "steps_per_epoch": steps_per_epoch,
            "formal_epochs": 3,
            "full_coverage_steps": full_steps,
            "tail_padding_samples_per_epoch": 7,
        },
        "summary": summary,
    }


def _report(tmp_path: Path, *, phase: str = "mixer") -> Path:
    init_checkpoint = _write(tmp_path / "init.pt", "checkpoint")
    manifest = _write(tmp_path / "manifest.json", "{}\n")
    benchmark = _write(tmp_path / "benchmark.py", "# benchmark\n")
    base_config_path = tmp_path / "base.yaml"
    save_yaml(
        base_config_path,
        {
            **stage211_phase_train_config_contract(phase),
            "webdataset_bucket_manifest_path": str(manifest),
        },
    )
    baseline = _profile_row(
        tmp_path,
        name="baseline",
        batch_size=36,
        frame_budget=24_000,
        steps_per_epoch=1000,
        projected_seconds=3000.0,
        loss=0.10,
        cosine=0.970,
        phase=phase,
        init_checkpoint=init_checkpoint,
        manifest=manifest,
    )
    candidate = _profile_row(
        tmp_path,
        name="batch48_frames42k",
        batch_size=48,
        frame_budget=42_000,
        steps_per_epoch=600,
        projected_seconds=2100.0,
        loss=0.102,
        cosine=0.969,
        phase=phase,
        init_checkpoint=init_checkpoint,
        manifest=manifest,
    )
    comparison = {
        "profile": "batch48_frames42k",
        "projected_full_coverage_seconds": 2100.0,
        "improvement_ratio": 0.3,
        "mean_loss": 0.102,
        "loss_regression_ratio": 0.01999999999999988,
        "mean_cosine": 0.969,
        "cosine_regression": 0.0010000000000000009,
        "quality_pass": True,
        "admissible": True,
    }
    report = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "batch_throughput_preflight",
        "phase": phase,
        "complete": True,
        "formal_admission": False,
        "dry_run": False,
        "git_commit": "a" * 40,
        "git_worktree_clean": True,
        "git_worktree_changes": [],
        "script_path": str(benchmark),
        "script_sha256": sha256_file(benchmark),
        "base_config_path": str(base_config_path.resolve()),
        "base_config_sha256": sha256_file(base_config_path),
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "bucket_manifest_path": str(manifest),
        "bucket_manifest_sha256": sha256_file(manifest),
        "warmup_steps": 20,
        "measure_steps": 100,
        "formal_epochs": 3,
        "world_size": 4,
        "gpu_indices": [0, 1, 2, 3],
        "max_peak_memory_gib": 22.0,
        "min_improvement_ratio": 0.10,
        "max_loss_regression_ratio": 0.05,
        "max_cosine_regression": 0.005,
        "profiles": [baseline, candidate],
        "selection": {
            "decision": "candidate_recommended",
            "baseline_profile": "baseline",
            "recommended_profile": "batch48_frames42k",
            "recommended_improvement_ratio": 0.3,
            "min_improvement_ratio": 0.10,
            "baseline_mean_loss": 0.10,
            "baseline_mean_cosine": 0.970,
            "max_loss_regression_ratio": 0.05,
            "max_cosine_regression": 0.005,
            "comparisons": [comparison],
            "formal_admission": False,
        },
    }
    path = tmp_path / "batch_throughput_preflight.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def test_measured_phase_specific_profile_can_be_admitted(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    measured = validate_stage211_batch_profile_preflight(report_path, phase="mixer")
    assert measured["selected_profile_name"] == "batch48_frames42k"

    receipt = build_stage211_batch_profile_admission(
        report_path,
        phase="mixer",
        admitted_by="test",
        reason="quality-equivalent measured speedup",
    )
    receipt_path = tmp_path / "admission.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    validated = validate_stage211_batch_profile_admission(
        receipt_path,
        phase="mixer",
        expected_init_checkpoint=tmp_path / "init.pt",
        expected_bucket_manifest=tmp_path / "manifest.json",
    )
    assert validated["selected_profile"] == {
        "name": "batch48_frames42k",
        "batch_size": 48,
        "frame_budget": 42_000,
    }


def test_dry_run_or_wrong_phase_cannot_be_admitted(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["dry_run"] = True
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formal measured report"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")

    report_path = _report(tmp_path / "fresh")
    with pytest.raises(ValueError, match="formal measured report"):
        validate_stage211_batch_profile_preflight(report_path, phase="block")


def test_incomplete_objective_match_rejects_selected_profile(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["profiles"][1]["summary"]["complete_teacher_matches"] = False
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="safety gates"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


def test_receipt_profile_tamper_is_rejected(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    receipt = build_stage211_batch_profile_admission(
        report_path,
        phase="mixer",
        admitted_by="test",
        reason="quality-equivalent measured speedup",
    )
    tampered = copy.deepcopy(receipt)
    tampered["selected_profile"]["frame_budget"] = 56_000
    receipt_path = tmp_path / "admission.json"
    receipt_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its measured report"):
        validate_stage211_batch_profile_admission(receipt_path, phase="mixer")


def test_controller_threads_an_explicit_segment_admission(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    parsed = stage211_full_phase._parse_batch_profile_admissions(
        [f"long={admission}"]
    )
    assert parsed == {"long": admission.resolve()}

    runner = stage211_full_phase._runner_command(
        phase="mixer",
        difficulty="long",
        output_dir=tmp_path / "run",
        config_dir=tmp_path / "config",
        manifest_path=tmp_path / "manifest.json",
        nano_checkpoint=tmp_path / "model.pt",
        master_port=29500,
        init_checkpoint=tmp_path / "init.pt",
        curriculum_receipt=tmp_path / "hard.json",
        promotion_receipt=None,
        smoke=False,
        dry_run=False,
        batch_profile_admission=admission,
    )
    assert runner[runner.index("--batch-profile-admission") + 1] == str(admission)

    receipt = stage211_full_phase._receipt_command(
        phase="mixer",
        difficulty="long",
        run_dir=tmp_path / "run",
        manifest_path=tmp_path / "manifest.json",
        init_checkpoint=tmp_path / "init.pt",
        completion_checkpoint=tmp_path / "step.pt",
        output=tmp_path / "long.json",
        batch_profile_admission=admission,
    )
    assert receipt[receipt.index("--batch-profile-admission") + 1] == str(admission)

    with pytest.raises(ValueError, match="supplied twice"):
        stage211_full_phase._parse_batch_profile_admissions(
            [f"long={admission}", f"long={admission}"]
        )


def test_schema2_profile_and_mixed_coverage_keep_dynamic_exposures(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    admission = build_stage211_batch_profile_admission(
        report_path,
        phase="mixer",
        admitted_by="test",
        reason="quality-equivalent measured speedup",
    )
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission) + "\n", encoding="utf-8")
    validated_admission = validate_stage211_batch_profile_admission(
        admission_path,
        phase="mixer",
    )
    selected = validated_admission["selected_coverage"]
    segment = {
        "schema_version": 2,
        "batch_profile_admission_path": str(admission_path.resolve()),
        "batch_profile_admission_sha256": sha256_file(admission_path),
        "batch_profile_name": "batch48_frames42k",
        "batch_size": 48,
        "world_size": 4,
        "frame_budget": 42_000,
        "steps_per_epoch": selected["steps_per_epoch"],
        "steps": selected["full_coverage_steps"],
        "tail_padding_samples_per_epoch": selected[
            "tail_padding_samples_per_epoch"
        ],
        "init_checkpoint_path": admission["init_checkpoint_path"],
        "bucket_manifest_path": admission["bucket_manifest_path"],
    }
    runtime = _validate_stage211_segment_batch_profile(
        segment,
        phase="mixer",
        difficulty="long",
    )
    assert runtime["schema_version"] == 2
    assert runtime["steps"] == selected["full_coverage_steps"]

    legacy = {
        "tail_padding_sample_exposures": 9,
        "executed_sample_exposures": 99,
    }
    admitted = {
        "tail_padding_sample_exposures": 21,
        "executed_sample_exposures": 121,
    }
    completion = tmp_path / "completion.pt"
    completion.write_bytes(b"checkpoint")
    coverage = build_stage211_full_data_coverage(
        phase="mixer",
        segments=[legacy, admitted],
        checkpoint_path=completion,
    )
    assert coverage["original_total_tail_padding_sample_exposures"] == 30
    assert coverage["original_total_executed_sample_exposures"] == 220
