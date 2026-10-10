from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.eval.stage211_batch_profile import (
    CandidateDominancePolicy,
    STAGE211_BATCH_PROFILE_PREFLIGHT_SCHEMA_VERSION,
    STAGE211_PROBE_ARTIFACT_CLEANUP_SCHEMA_VERSION,
    STAGE211_PROBE_FIXED_EVAL_CAPTURE_SCHEMA_VERSION,
    build_stage211_batch_profile_admission,
    candidate_dominance_evidence,
    _validate_git_bound_source,
    validate_stage211_batch_profile_fixed_eval,
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
stage211_full_phase = importlib.import_module("scripts.run_stage211_full_phase_curriculum")


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path.resolve()


def test_git_bound_source_accepts_only_matching_ancestor_blob(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Stage211 Test")
    _git(repository, "config", "user.email", "stage211@example.invalid")
    script = _write(repository / "scripts" / "probe.py", "old benchmark\n")
    _git(repository, "add", "scripts/probe.py")
    _git(repository, "commit", "-m", "old benchmark")
    old_commit = _git(repository, "rev-parse", "HEAD")
    old_sha256 = hashlib.sha256(b"old benchmark\n").hexdigest()

    script.write_text("new benchmark\n", encoding="utf-8")
    _git(repository, "add", "scripts/probe.py")
    _git(repository, "commit", "-m", "new benchmark")
    record = {
        "script_path": str(script),
        "script_sha256": old_sha256,
        "git_commit": old_commit,
    }

    assert _validate_git_bound_source(
        record,
        path_key="script_path",
        sha256_key="script_sha256",
        git_commit_key="git_commit",
        label="test source",
        repository_root=repository,
    ) == script

    mismatched = {**record, "script_sha256": "0" * 64}
    with pytest.raises(ValueError, match="historical Git blob SHA-256 mismatch"):
        _validate_git_bound_source(
            mismatched,
            path_key="script_path",
            sha256_key="script_sha256",
            git_commit_key="git_commit",
            label="test source",
            repository_root=repository,
        )

    tree = _git(repository, "write-tree")
    unrelated_commit = _git(repository, "commit-tree", tree, "-m", "unrelated")
    non_ancestor = {**record, "git_commit": unrelated_commit}
    with pytest.raises(ValueError, match="unavailable or not an ancestor"):
        _validate_git_bound_source(
            non_ancestor,
            path_key="script_path",
            sha256_key="script_sha256",
            git_commit_key="git_commit",
            label="test source",
            repository_root=repository,
        )

    outside = _write(tmp_path / "outside.py", "new benchmark\n")
    escaped = {**record, "script_path": str(outside)}
    with pytest.raises(ValueError, match="outside the bound repository"):
        _validate_git_bound_source(
            escaped,
            path_key="script_path",
            sha256_key="script_sha256",
            git_commit_key="git_commit",
            label="test source",
            repository_root=repository,
        )


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
    eval_part: Path,
    num_workers: int = 8,
    gradient_checkpointing: bool | None = None,
) -> dict[str, object]:
    if gradient_checkpointing is None:
        gradient_checkpointing = phase in {"block", "logits"}
    profile_root = root / name
    config_path = profile_root / "train_config.yaml"
    config = {
        **stage211_phase_train_config_contract(phase),
        "stage211_batch_profile_probe_phase": phase,
        "stage211_batch_profile_probe_epoch_batch_offset": 0,
        "max_steps": 120,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "gradient_checkpointing": gradient_checkpointing,
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
        "step_eval_at_start": False,
        "step_eval_every": 120,
        "step_eval_samples": 256,
        "step_eval_split": "eval",
        "step_eval_shuffle": False,
        "step_eval_cache_batches": True,
        "step_eval_feature_seed": 0,
        "save_deepspeed_sharded_checkpoints": False,
        "output_dir": str((profile_root / "run").resolve()),
        "deepspeed": {
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": batch_size,
            "train_batch_size": batch_size * 4,
        },
    }
    save_yaml(config_path, config)
    log_path = _write(profile_root / "train.log", "measured\n")
    fixed_eval_path = profile_root / "fixed_eval.yaml"
    components = {
        component: {
            str(layer_id): {
                "loss": loss,
                "cosine": cosine,
                "rms_ratio": 1.0,
            }
            for layer_id in range(70)
        }
        for component in (("mixer",) if phase == "mixer" else ("mixer", "ffn", "block"))
    }
    save_yaml(
        fixed_eval_path,
        {
            "step": 120,
            "eval_loss": loss,
            "eval_samples": 256,
            "eval_provenance": {
                "schema_version": 1,
                "split": "eval",
                "requested_samples": 256,
                "feature_seed": 0,
                "bucket_manifest_path": str(manifest),
                "bucket_manifest_sha256": sha256_file(manifest),
                "split_samples": 256,
                "parts": [
                    {
                        "path": str(eval_part),
                        "sha256": sha256_file(eval_part),
                        "num_samples": 256,
                    }
                ],
            },
            "layer_components": components,
            "logit_metrics": (
                {
                    name: (
                        256.0
                        if name == "matched_utterances"
                        else 0.0
                        if name in {"missing_utterances", "mean_frame_delta"}
                        else 0.1
                    )
                    for name in (
                        "full_kl",
                        "conditional_nonblank_kl",
                        "conditional_nonblank_hard_ce",
                        "blank_binary_kl",
                        "selected_top1_agreement",
                        "all_top1_agreement",
                        "active_top1_agreement",
                        "blank_prob_mae",
                        "teacher_nonblank_rate",
                        "student_nonblank_rate",
                        "nonblank_rate_ratio",
                        "ctc_token_error_rate",
                        "ctc_token_deletion_rate",
                        "collapsed_length_ratio",
                        "sequence_exact_rate",
                        "mean_frame_delta",
                        "matched_utterances",
                        "missing_utterances",
                    )
                }
                if phase == "logits"
                else {}
            ),
            "decoder_hidden_metrics": ({} if phase == "mixer" else {"loss": loss}),
        },
    )
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
        "training_mean_loss": loss,
        "training_mean_cosine": cosine,
        "training_mean_aggregation": "primary_match_sample_weighted",
        "alignment_mean_aggregation": (
            "fixed_eval_total_objective_and_unweighted_70_layer_primary_component"
        ),
        "quality_source": "terminal_fixed_eval_256",
        "fixed_eval_complete": True,
        "fixed_eval_samples": 256,
        "fixed_eval_primary_component": "mixer" if phase == "mixer" else "block",
        "fixed_eval_provenance": {
            "schema_version": 1,
            "split": "eval",
            "requested_samples": 256,
            "feature_seed": 0,
            "bucket_manifest_path": str(manifest),
            "bucket_manifest_sha256": sha256_file(manifest),
            "split_samples": 256,
            "parts": [
                {
                    "path": str(eval_part),
                    "sha256": sha256_file(eval_part),
                    "num_samples": 256,
                }
            ],
        },
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
            "num_workers": num_workers,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "probe_epoch_batch_offset": 0,
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
        "fixed_eval_capture": {
            "schema_version": STAGE211_PROBE_FIXED_EVAL_CAPTURE_SCHEMA_VERSION,
            "artifact": "probe_fixed_eval_capture",
            "source_name": "step_eval_layers_step-120.yaml",
            "report_path": str(fixed_eval_path.resolve()),
            "captured": True,
            "report_sha256": sha256_file(fixed_eval_path),
        },
        "probe_artifact_cleanup": {
            "schema_version": STAGE211_PROBE_ARTIFACT_CLEANUP_SCHEMA_VERSION,
            "artifact": "probe_artifact_cleanup",
            "complete": True,
            "run_dir": str((profile_root / "run").resolve()),
            "existed_before": True,
            "files_removed": 3,
            "directories_removed": 1,
            "bytes_removed": 123,
            "exists_after": False,
        },
    }


def _report(
    tmp_path: Path,
    *,
    phase: str = "mixer",
    baseline_batch_size: int = 36,
    baseline_frame_budget: int = 24_000,
) -> Path:
    init_checkpoint = _write(tmp_path / "init.pt", "checkpoint")
    manifest = _write(tmp_path / "manifest.json", "{}\n")
    eval_part = _write(tmp_path / "fixed_eval.jsonl", "{}\n" * 256)
    benchmark = _write(tmp_path / "benchmark.py", "# benchmark\n")
    base_config_path = tmp_path / "base.yaml"
    save_yaml(
        base_config_path,
        {
            **stage211_phase_train_config_contract(phase),
            "webdataset_bucket_manifest_path": str(manifest),
            "num_workers": 8,
        },
    )
    baseline = _profile_row(
        tmp_path,
        name="baseline",
        batch_size=baseline_batch_size,
        frame_budget=baseline_frame_budget,
        steps_per_epoch=1000,
        projected_seconds=3000.0,
        loss=0.10,
        cosine=0.970,
        phase=phase,
        init_checkpoint=init_checkpoint,
        manifest=manifest,
        eval_part=eval_part,
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
        eval_part=eval_part,
    )
    comparison = {
        "profile": "batch48_frames42k",
        "projected_full_coverage_seconds": 2100.0,
        "improvement_ratio": 0.3,
        "mean_loss": 0.102,
        "loss_regression_ratio": 0.01999999999999988,
        "mean_cosine": 0.969,
        "cosine_regression": 0.0010000000000000009,
        "fixed_eval_provenance_match": True,
        "quality_pass": True,
        "admissible": True,
    }
    report = {
        "schema_version": STAGE211_BATCH_PROFILE_PREFLIGHT_SCHEMA_VERSION,
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
        "probe_depth_fraction": 0.0,
        "formal_epochs": 3,
        "world_size": 4,
        "gpu_indices": [0, 1, 2, 3],
        "max_peak_memory_gib": 22.0,
        "min_improvement_ratio": 0.10,
        "max_loss_regression_ratio": 0.05,
        "max_cosine_regression": 0.005,
        "candidate_dominance": {
            "enabled_for_capacity_candidates_only": True,
            "rejection_ratio": 4.0,
            "min_points": 12,
            "excluded_profiles": ["baseline"],
        },
        "loader_worker_search": {
            "schema_version": 1,
            "mode": "explicit_profiles",
            "logical_cpus": 16,
            "physical_cores": 8,
            "topology_source": "linux_sysfs_affinity",
            "world_size": 4,
            "configured_num_workers": 8,
            "balanced_num_workers": 2,
            "enabled": False,
            "baseline_profile": "baseline",
            "candidate_profile": None,
            "selection_decision": "explicit_profiles",
            "selected_profile": None,
            "selected_num_workers": None,
        },
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


def _worker_only_report(tmp_path: Path) -> Path:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = report["profiles"][1]
    candidate_name = "baseline_workers2"
    candidate["profile"].update(
        {
            "name": candidate_name,
            "batch_size": 36,
            "frame_budget": 24_000,
            "num_workers": 2,
        }
    )
    config_path = Path(candidate["config_path"])
    config = load_yaml(config_path)
    config.update(
        {
            "batch_size": 36,
            "num_workers": 2,
            "batch_token_budget": 24_000,
            "length_bucket_frame_budget": 24_000,
        }
    )
    config["deepspeed"].update(
        {
            "train_micro_batch_size_per_gpu": 36,
            "train_batch_size": 144,
        }
    )
    save_yaml(config_path, config)
    candidate["config_sha256"] = sha256_file(config_path)
    candidate["coverage"].update(
        {
            "steps_per_epoch": 1000,
            "full_coverage_steps": 3000,
        }
    )
    candidate["summary"]["steps_per_second"] = 3000.0 / 2100.0
    comparison = report["selection"]["comparisons"][0]
    comparison["profile"] = candidate_name
    report["selection"]["recommended_profile"] = candidate_name
    report["loader_worker_search"] = {
        "schema_version": 1,
        "mode": "automatic_balanced_candidate",
        "logical_cpus": 16,
        "physical_cores": 8,
        "topology_source": "linux_sysfs_affinity",
        "world_size": 4,
        "configured_num_workers": 8,
        "balanced_num_workers": 2,
        "enabled": True,
        "baseline_profile": "baseline",
        "candidate_profile": candidate_name,
        "selection_decision": "balanced_workers_selected",
        "selected_profile": candidate_name,
        "selected_num_workers": 2,
    }
    report["candidate_dominance"]["excluded_profiles"] = ["baseline", candidate_name]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


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
        "num_workers": 8,
        "gradient_checkpointing": False,
    }


def test_representative_depth_profile_rejects_offset_tampering(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["probe_depth_fraction"] = 0.5
    for row in report["profiles"]:
        offset = int(row["coverage"]["steps_per_epoch"] * 0.5)
        row["probe_epoch_batch_offset"] = offset
        config_path = Path(row["config_path"])
        config = load_yaml(config_path)
        config["stage211_batch_profile_probe_epoch_batch_offset"] = offset
        save_yaml(config_path, config)
        row["config_sha256"] = sha256_file(config_path)
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    measured = validate_stage211_batch_profile_preflight(report_path, phase="mixer")
    assert measured["selected_profile_name"] == "batch48_frames42k"

    report["profiles"][0]["probe_epoch_batch_offset"] += 1
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="probe offset changed"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


def test_dominated_capacity_candidate_is_replayed_and_tamper_evident(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = report["profiles"][0]
    candidate = report["profiles"][1]
    telemetry = [
        {
            "step": step,
            "elapsed_seconds": step * 10.0,
            "loss": 0.1,
            "cosine": 0.97,
            "match_count": 8,
            "match_total": 8,
            "missing_total": 0,
            "max_abs_frame_delta": 0,
            "match_fields": {"online_layer_match": [8, 8]},
        }
        for step in range(1, 13)
    ]
    policy = CandidateDominancePolicy(
        full_coverage_steps=int(candidate["coverage"]["full_coverage_steps"]),
        baseline_projected_full_coverage_seconds=float(
            baseline["summary"]["projected_full_coverage_seconds"]
        ),
        min_improvement_ratio=float(report["min_improvement_ratio"]),
        rejection_ratio=float(report["candidate_dominance"]["rejection_ratio"]),
        min_points=int(report["candidate_dominance"]["min_points"]),
    )
    termination = candidate_dominance_evidence(telemetry, policy)
    assert termination is not None
    fixed_eval_path = Path(candidate["fixed_eval_capture"]["report_path"])
    fixed_eval_path.unlink()
    candidate.update(
        {
            "return_code": -15,
            "telemetry": telemetry,
            "early_termination": termination,
        }
    )
    candidate["fixed_eval_capture"].update({"captured": False, "report_sha256": None})
    candidate["summary"].update(
        {
            "process_ok": False,
            "complete_steps": False,
            "safety_pass": False,
            "fixed_eval_complete": False,
            "fixed_eval_provenance": None,
            "mean_loss": None,
            "mean_cosine": None,
            "projected_full_coverage_seconds": None,
        }
    )
    report["selection"].update(
        {
            "decision": "keep_baseline",
            "recommended_profile": "baseline",
            "recommended_improvement_ratio": 0.0,
        }
    )
    report["selection"]["comparisons"][0].update(
        {
            "projected_full_coverage_seconds": None,
            "improvement_ratio": None,
            "mean_loss": None,
            "loss_regression_ratio": None,
            "mean_cosine": None,
            "cosine_regression": None,
            "fixed_eval_provenance_match": False,
            "quality_pass": False,
            "admissible": False,
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    validated = validate_stage211_batch_profile_preflight(
        report_path,
        phase="mixer",
        require_candidate=False,
    )
    assert validated["selected_profile_name"] == "baseline"

    report["profiles"][1]["early_termination"]["window_last_step"] = 11
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dominance evidence changed"):
        validate_stage211_batch_profile_preflight(
            report_path,
            phase="mixer",
            require_candidate=False,
        )


def test_worker_only_profile_is_measured_admitted_and_tamper_evident(
    tmp_path: Path,
) -> None:
    report_path = _worker_only_report(tmp_path)
    measured = validate_stage211_batch_profile_preflight(report_path, phase="mixer")
    assert measured["selected_profile_row"]["profile"]["num_workers"] == 2

    receipt = build_stage211_batch_profile_admission(
        report_path,
        phase="mixer",
        admitted_by="test",
        reason="same-capacity loader worker calibration exceeded ten percent",
    )
    receipt_path = tmp_path / "worker-admission.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    validated = validate_stage211_batch_profile_admission(receipt_path, phase="mixer")
    assert validated["selected_profile"] == {
        "name": "baseline_workers2",
        "batch_size": 36,
        "frame_budget": 24_000,
        "num_workers": 2,
        "gradient_checkpointing": False,
    }

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["loader_worker_search"]["selected_num_workers"] = 8
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration decision changed"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


@pytest.mark.parametrize("phase", ("block", "logits"))
def test_stacked_preflight_accepts_safe_and_legacy_baselines(
    tmp_path: Path,
    phase: str,
) -> None:
    safe_report = _report(
        tmp_path / f"safe-{phase}",
        phase=phase,
        baseline_batch_size=4,
        baseline_frame_budget=4_000,
    )
    legacy_report = _report(tmp_path / f"legacy-{phase}", phase=phase)

    safe = validate_stage211_batch_profile_preflight(safe_report, phase=phase)
    legacy = validate_stage211_batch_profile_preflight(legacy_report, phase=phase)

    safe_baseline = next(
        row["profile"] for row in safe["profiles"] if row["profile"]["name"] == "baseline"
    )
    legacy_baseline = next(
        row["profile"] for row in legacy["profiles"] if row["profile"]["name"] == "baseline"
    )
    assert safe_baseline["batch_size"] == 4
    assert safe_baseline["frame_budget"] == 4_000
    assert legacy_baseline["batch_size"] == 36
    assert legacy_baseline["frame_budget"] == 24_000


def test_logits_retained_safe_baseline_can_be_formally_admitted(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        phase="logits",
        baseline_batch_size=4,
        baseline_frame_budget=4_000,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = report["profiles"][1]
    full_steps = int(candidate["coverage"]["full_coverage_steps"])
    candidate["summary"]["projected_full_coverage_seconds"] = 3_150.0
    candidate["summary"]["steps_per_second"] = full_steps / 3_150.0
    comparison = report["selection"]["comparisons"][0]
    comparison["projected_full_coverage_seconds"] = 3_150.0
    comparison["improvement_ratio"] = -0.05
    comparison["admissible"] = False
    report["selection"].update(
        {
            "decision": "keep_baseline",
            "recommended_profile": "baseline",
            "recommended_improvement_ratio": 0.0,
        }
    )
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    receipt = build_stage211_batch_profile_admission(
        report_path,
        phase="logits",
        admitted_by="test",
        reason="memory-safe measured baseline",
    )
    assert receipt["selected_profile"] == {
        "name": "baseline",
        "batch_size": 4,
        "frame_budget": 4_000,
        "num_workers": 8,
        "gradient_checkpointing": True,
    }
    assert receipt["selected_comparison"] is None
    receipt_path = tmp_path / "admission.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    validated = validate_stage211_batch_profile_admission(
        receipt_path,
        phase="logits",
        expected_init_checkpoint=tmp_path / "init.pt",
        expected_bucket_manifest=tmp_path / "manifest.json",
    )
    assert validated["selected_profile"] == receipt["selected_profile"]
    assert validated["selected_comparison"] is None


def test_automatic_profile_routes_nonlegacy_retained_baseline_to_admission(
    tmp_path: Path,
) -> None:
    base_config = tmp_path / "base.yaml"
    save_yaml(
        base_config,
        {
            "batch_size": 36,
            "length_bucket_frame_budget": 24_000,
            "num_workers": 8,
            "gradient_checkpointing": False,
        },
    )
    retained_safe_logits = {
        "phase": "logits",
        "selection_decision": "keep_baseline",
        "base_config_path": str(base_config),
        "selected_profile_row": {
            "profile": {
                "name": "baseline",
                "batch_size": 4,
                "frame_budget": 4_000,
                "num_workers": 8,
                "gradient_checkpointing": True,
            }
        },
    }
    retained_legacy = copy.deepcopy(retained_safe_logits)
    retained_legacy["phase"] = "mixer"
    retained_legacy["selected_profile_row"]["profile"].update(
        {"batch_size": 36, "frame_budget": 24_000, "gradient_checkpointing": False}
    )
    admitted_candidate = copy.deepcopy(retained_legacy)
    admitted_candidate["selection_decision"] = "admit_candidate"
    admitted_candidate["selected_profile_row"]["profile"].update(
        {"batch_size": 48, "frame_budget": 42_000}
    )

    assert stage211_full_phase._automatic_profile_requires_admission(retained_safe_logits)
    assert not stage211_full_phase._automatic_profile_requires_admission(retained_legacy)
    assert stage211_full_phase._automatic_profile_requires_admission(admitted_candidate)


def test_automatic_profile_treats_checkpoint_mode_change_as_admission_required(
    tmp_path: Path,
) -> None:
    base_config = tmp_path / "base.yaml"
    save_yaml(
        base_config,
        {
            "batch_size": 4,
            "length_bucket_frame_budget": 4_000,
            "num_workers": 8,
            "gradient_checkpointing": True,
        },
    )
    measured = {
        "phase": "block",
        "selection_decision": "admit_candidate",
        "base_config_path": str(base_config),
        "selected_profile_row": {
            "profile": {
                "name": "no_ckpt_batch4_frames4k",
                "batch_size": 4,
                "frame_budget": 4_000,
                "num_workers": 8,
                "gradient_checkpointing": False,
            }
        },
    }

    assert stage211_full_phase._automatic_profile_requires_admission(measured)


def test_profile_config_rejects_checkpoint_mode_mismatch(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        phase="block",
        baseline_batch_size=4,
        baseline_frame_budget=4_000,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["profiles"][1]["profile"]["gradient_checkpointing"] = False
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gradient_checkpointing mismatch"):
        validate_stage211_batch_profile_preflight(report_path, phase="block")


def test_same_capacity_no_checkpoint_candidate_can_be_admitted(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        phase="block",
        baseline_batch_size=4,
        baseline_frame_budget=4_000,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline, candidate = report["profiles"]
    candidate_name = "no_ckpt_batch4_frames4k"
    candidate["profile"].update(
        {
            "name": candidate_name,
            "batch_size": 4,
            "frame_budget": 4_000,
            "gradient_checkpointing": False,
        }
    )
    config_path = Path(candidate["config_path"])
    config = load_yaml(config_path)
    config.update(
        {
            "batch_size": 4,
            "batch_token_budget": 4_000,
            "length_bucket_frame_budget": 4_000,
            "gradient_checkpointing": False,
        }
    )
    config["deepspeed"].update(
        {
            "train_micro_batch_size_per_gpu": 4,
            "train_batch_size": 16,
        }
    )
    save_yaml(config_path, config)
    candidate["config_sha256"] = sha256_file(config_path)
    candidate["coverage"] = copy.deepcopy(baseline["coverage"])
    full_steps = int(candidate["coverage"]["full_coverage_steps"])
    candidate["summary"]["steps_per_second"] = full_steps / 2_100.0
    report["selection"]["recommended_profile"] = candidate_name
    report["selection"]["comparisons"][0]["profile"] = candidate_name
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    receipt = build_stage211_batch_profile_admission(
        report_path,
        phase="block",
        admitted_by="test",
        reason="same-capacity no-checkpoint candidate exceeded ten percent",
    )
    receipt_path = tmp_path / "no-checkpoint-admission.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    validated = validate_stage211_batch_profile_admission(receipt_path, phase="block")

    assert validated["selected_profile"] == candidate["profile"]


def test_mixer_preflight_rejects_stacked_safe_baseline(tmp_path: Path) -> None:
    report_path = _report(
        tmp_path,
        phase="mixer",
        baseline_batch_size=4,
        baseline_frame_budget=4_000,
    )

    with pytest.raises(ValueError, match="baseline is unsupported for this phase"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


@pytest.mark.parametrize(
    ("phase", "primary_component"),
    (("mixer", "mixer"), ("block", "block"), ("logits", "block")),
)
def test_fixed_eval_requires_phase_complete_same_distribution_metrics(
    tmp_path: Path,
    phase: str,
    primary_component: str,
) -> None:
    root = tmp_path / phase
    init_checkpoint = _write(root / "init.pt", "checkpoint")
    manifest = _write(root / "manifest.json", "{}\n")
    eval_part = _write(root / "fixed_eval.jsonl", "{}\n" * 256)
    row = _profile_row(
        root,
        name="baseline",
        batch_size=36,
        frame_budget=24_000,
        steps_per_epoch=10,
        projected_seconds=30.0,
        loss=0.1,
        cosine=0.97,
        phase=phase,
        init_checkpoint=init_checkpoint,
        manifest=manifest,
        eval_part=eval_part,
    )

    validated = validate_stage211_batch_profile_fixed_eval(
        row["fixed_eval_capture"]["report_path"],
        phase=phase,
        expected_step=120,
        expected_manifest=manifest,
    )
    assert validated["primary_component"] == primary_component
    assert validated["eval_samples"] == 256
    assert validated["mean_cosine"] == pytest.approx(0.97)

    if phase != "mixer":
        report_path = Path(row["fixed_eval_capture"]["report_path"])
        report = load_yaml(report_path)
        report["layer_components"].pop("ffn")
        save_yaml(report_path, report)
        with pytest.raises(ValueError, match="70-layer ffn coverage"):
            validate_stage211_batch_profile_fixed_eval(
                report_path,
                phase=phase,
                expected_step=120,
                expected_manifest=manifest,
            )


def test_complete_measurement_can_retain_legacy_baseline(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = report["profiles"][1]
    full_steps = int(candidate["coverage"]["full_coverage_steps"])
    candidate["summary"]["projected_full_coverage_seconds"] = 2_850.0
    candidate["summary"]["steps_per_second"] = full_steps / 2_850.0
    comparison = report["selection"]["comparisons"][0]
    comparison["projected_full_coverage_seconds"] = 2_850.0
    comparison["improvement_ratio"] = 0.05
    comparison["admissible"] = False
    report["selection"].update(
        {
            "decision": "keep_baseline",
            "recommended_profile": "baseline",
            "recommended_improvement_ratio": 0.0,
        }
    )
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    measured = validate_stage211_batch_profile_preflight(
        report_path,
        phase="mixer",
        require_candidate=False,
    )
    assert measured["selection_decision"] == "keep_baseline"
    assert measured["selected_profile_name"] == "baseline"
    with pytest.raises(ValueError, match="no admissible larger profile"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


def test_keep_baseline_measurement_rejects_hidden_admissible_candidate(
    tmp_path: Path,
) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["selection"].update(
        {
            "decision": "keep_baseline",
            "recommended_profile": "baseline",
            "recommended_improvement_ratio": 0.0,
        }
    )
    report["selection"]["comparisons"][0]["admissible"] = False
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="comparison decision"):
        validate_stage211_batch_profile_preflight(
            report_path,
            phase="mixer",
            require_candidate=False,
        )


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


def test_measured_profile_rejects_retained_probe_artifacts(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    retained = Path(report["profiles"][1]["probe_artifact_cleanup"]["run_dir"])
    retained.mkdir(parents=True)
    (retained / "step-120.pt").write_bytes(b"retained")

    with pytest.raises(ValueError, match="retained probe artifacts"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


def test_incomplete_objective_match_rejects_selected_profile(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["profiles"][1]["summary"]["complete_teacher_matches"] = False
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="safety gates"):
        validate_stage211_batch_profile_preflight(report_path, phase="mixer")


def test_candidate_on_different_fixed_eval_provenance_is_rejected(tmp_path: Path) -> None:
    report_path = _report(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = report["profiles"][1]
    fixed_eval_path = Path(candidate["fixed_eval_capture"]["report_path"])
    different_part = _write(tmp_path / "different_eval.jsonl", "{}\n" * 256)
    fixed_eval = load_yaml(fixed_eval_path)
    different_binding = {
        "path": str(different_part),
        "sha256": sha256_file(different_part),
        "num_samples": 256,
    }
    fixed_eval["eval_provenance"]["parts"] = [different_binding]
    save_yaml(fixed_eval_path, fixed_eval)
    candidate["fixed_eval_capture"]["report_sha256"] = sha256_file(fixed_eval_path)
    candidate["summary"]["fixed_eval_provenance"]["parts"] = [different_binding]
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="comparison decision"):
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

    tampered = copy.deepcopy(receipt)
    tampered["selected_profile"]["gradient_checkpointing"] = True
    receipt_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its measured report"):
        validate_stage211_batch_profile_admission(receipt_path, phase="mixer")


def test_controller_threads_an_explicit_segment_admission(tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    parsed = stage211_full_phase._parse_batch_profile_admissions([f"long={admission}"])
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


def test_controller_recovers_auto_admission_from_segment_provenance(
    tmp_path: Path,
) -> None:
    admission = _write(tmp_path / "admission.json", "{}\n")
    run_dir = tmp_path / "run"
    provenance = {
        "batch_profile_admission_path": str(admission),
        "batch_profile_admission_sha256": sha256_file(admission),
    }
    _write(run_dir / "stage211_provenance.json", json.dumps(provenance) + "\n")

    assert (
        stage211_full_phase._resolve_segment_batch_profile_admission(
            run_dir=run_dir,
            requested=None,
        )
        == admission
    )
    with pytest.raises(ValueError, match="differs from provenance"):
        stage211_full_phase._resolve_segment_batch_profile_admission(
            run_dir=run_dir,
            requested=tmp_path / "other.json",
        )


def test_controller_builds_phase_specific_automatic_preflight_command(
    tmp_path: Path,
) -> None:
    command = stage211_full_phase._profile_preflight_command(
        phase="block",
        base_config=tmp_path / "base.yaml",
        init_checkpoint=tmp_path / "init.pt",
        output_root=tmp_path / "probe",
        master_port=29741,
    )

    assert command[command.index("--phase") + 1] == "block"
    assert "--profile" not in command
    assert command[command.index("--max-peak-memory-gib") + 1] == "22.0"
    assert command[command.index("--min-improvement-ratio") + 1] == "0.10"
    assert command[command.index("--max-loss-regression-ratio") + 1] == "0.05"
    assert command[command.index("--max-cosine-regression") + 1] == "0.005"


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
        "num_workers": 8,
        "gradient_checkpointing": False,
        "steps_per_epoch": selected["steps_per_epoch"],
        "steps": selected["full_coverage_steps"],
        "tail_padding_samples_per_epoch": selected["tail_padding_samples_per_epoch"],
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
