from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
probe = importlib.import_module("scripts.benchmark_stage211_batch_profiles")


def test_default_profiles_cover_baseline_and_seven_larger_candidates() -> None:
    assert probe.DEFAULT_PROFILES == (
        probe.BatchProfile("baseline", 36, 24_000),
        probe.BatchProfile("batch48_frames42k", 48, 42_000),
        probe.BatchProfile("batch64_frames56k", 64, 56_000),
        probe.BatchProfile("batch80_frames70k", 80, 70_000),
        probe.BatchProfile("batch96_frames84k", 96, 84_000),
        probe.BatchProfile("batch128_frames112k", 128, 112_000),
        probe.BatchProfile("batch160_frames140k", 160, 140_000),
        probe.BatchProfile("batch192_frames168k", 192, 168_000),
    )


def test_logits_default_profiles_start_from_memory_safe_baseline() -> None:
    assert probe.default_profiles_for_phase("logits") == (
        probe.BatchProfile("baseline", 12, 8_000),
        probe.BatchProfile("batch24_frames16k", 24, 16_000),
        probe.BatchProfile("batch36_frames24k", 36, 24_000),
        *probe.DEFAULT_PROFILES[1:],
    )
    assert probe.default_profiles_for_phase("mixer") is probe.DEFAULT_PROFILES
    assert probe.default_profiles_for_phase("block") is probe.DEFAULT_PROFILES


def test_probe_artifact_cleanup_removes_only_generated_run_dir(tmp_path: Path) -> None:
    profile_root = tmp_path / "profile"
    run_dir = profile_root / "run"
    checkpoint = run_dir / "step-120.pt"
    nested = run_dir / "metadata" / "latest.yaml"
    config = profile_root / "train_config.yaml"
    log = profile_root / "train.log"
    checkpoint.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    nested.write_text("step: 120\n", encoding="utf-8")
    config.write_text("max_steps: 120\n", encoding="utf-8")
    log.write_text("measured\n", encoding="utf-8")

    receipt = probe._cleanup_probe_artifacts(
        run_dir=run_dir,
        profile_root=profile_root,
    )

    assert receipt == {
        "schema_version": probe.STAGE211_PROBE_ARTIFACT_CLEANUP_SCHEMA_VERSION,
        "artifact": "probe_artifact_cleanup",
        "complete": True,
        "run_dir": str(run_dir.resolve()),
        "existed_before": True,
        "files_removed": 2,
        "directories_removed": 1,
        "bytes_removed": len(b"checkpoint") + len("step: 120\n".encode()),
        "exists_after": False,
    }
    assert not run_dir.exists()
    assert config.read_text(encoding="utf-8") == "max_steps: 120\n"
    assert log.read_text(encoding="utf-8") == "measured\n"


def test_probe_artifact_cleanup_rejects_paths_outside_profile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpected probe directory"):
        probe._cleanup_probe_artifacts(
            run_dir=tmp_path / "other" / "run",
            profile_root=tmp_path / "profile",
        )


def test_probe_fixed_eval_is_preserved_before_run_cleanup(tmp_path: Path) -> None:
    profile_root = tmp_path / "profile"
    run_dir = profile_root / "run"
    source = run_dir / "step_eval_layers_step-120.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("step: 120\neval_samples: 256\n", encoding="utf-8")

    capture = probe._capture_probe_fixed_eval(
        run_dir=run_dir,
        profile_root=profile_root,
        max_steps=120,
    )
    cleanup = probe._cleanup_probe_artifacts(
        run_dir=run_dir,
        profile_root=profile_root,
    )

    preserved = profile_root / "fixed_eval.yaml"
    assert capture["captured"] is True
    assert capture["report_path"] == str(preserved.resolve())
    assert capture["report_sha256"] == probe.sha256_file(preserved)
    assert preserved.read_text(encoding="utf-8") == "step: 120\neval_samples: 256\n"
    assert cleanup["files_removed"] == 1
    assert not run_dir.exists()


def test_parse_profile_and_build_config_do_not_mutate_base(tmp_path: Path) -> None:
    profile = probe.parse_profile("larger:48:42000")
    checkpoint = tmp_path / "init.pt"
    checkpoint.touch()
    base = {
        "output_dir": "old",
        "resume_from": "latest",
        "max_steps": 999,
        "batch_size": 36,
        "batch_token_budget": 24_000,
        "length_bucket_frame_budget": 24_000,
        "wandb_enabled": True,
        "deepspeed": {
            "train_micro_batch_size_per_gpu": 36,
            "gradient_accumulation_steps": 2,
            "train_batch_size": 288,
        },
    }

    config = probe.build_probe_config(
        base,
        profile=profile,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        max_steps=120,
        world_size=4,
    )

    assert base["resume_from"] == "latest"
    assert base["batch_size"] == 36
    assert config["resume_from"] is None
    assert config["init_checkpoint_path"] == str(checkpoint.resolve())
    assert config["max_steps"] == 120
    assert config["step_eval_every"] == 120
    assert config["step_eval_samples"] == 256
    assert config["step_eval_split"] == "eval"
    assert config["step_eval_shuffle"] is False
    assert config["step_eval_cache_batches"] is True
    assert config["step_eval_feature_seed"] == 0
    assert config["wandb_enabled"] is False
    assert config["batch_size"] == 48
    assert config["batch_token_budget"] == 42_000
    assert config["length_bucket_frame_budget"] == 42_000
    assert config["deepspeed"]["train_micro_batch_size_per_gpu"] == 48
    assert config["deepspeed"]["train_batch_size"] == 384


def test_parse_train_telemetry_collects_alignment_safety_fields() -> None:
    line = (
        "[deepspeed-train] step=27 loss=0.0812 "
        "online_teacher_missing=0 online_layer_mixer_cosine=0.9712 "
        "online_layer_match=13/13 online_layer_missing=0 "
        "online_layer_frame_delta=0 online_decoder_hidden_frame_delta=0"
    )

    point = probe.parse_train_telemetry(line, elapsed_seconds=12.5)

    assert point is not None
    assert point.step == 27
    assert point.loss == 0.0812
    assert point.cosine == 0.9712
    assert point.match_count == point.match_total == 13
    assert point.match_fields["online_layer_match"] == (13, 13)
    assert point.missing_total == 0
    assert point.max_abs_frame_delta == 0


def test_required_online_match_fields_follow_enabled_phase_objectives() -> None:
    mixer = {"ctc_teacher_online_layer_mixer_loss_weight": 1.0}
    block = {
        "ctc_teacher_online_encoder_loss_weight": 0.5,
        "ctc_teacher_online_decoder_hidden_loss_weight": 0.5,
        "ctc_teacher_online_layer_mixer_loss_weight": 0.25,
        "ctc_teacher_online_layer_ffn_loss_weight": 0.25,
        "ctc_teacher_online_layer_block_loss_weight": 1.0,
    }
    logits = {
        **block,
        "ctc_teacher_online_blank_loss_weight": 0.25,
        "ctc_teacher_online_conditional_nonblank_loss_weight": 1.0,
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight": 0.125,
        "ctc_teacher_online_sequence_loss_weight": 0.2,
        "ctc_teacher_online_nonblank_window_loss_weight": 0.25,
    }

    assert probe.required_online_match_fields(mixer) == ("online_layer_match",)
    assert probe.required_online_match_fields(block) == (
        "online_encoder_match",
        "online_decoder_hidden_match",
        "online_layer_match",
    )
    assert probe.required_online_match_fields(logits) == (
        "online_blank_match",
        "online_conditional_nonblank_match",
        "online_conditional_nonblank_hard_match",
        "online_encoder_match",
        "online_decoder_hidden_match",
        "online_sequence_match",
        "online_nonblank_window_match",
        "online_layer_match",
    )


def test_missing_or_partial_enabled_match_rejects_fast_candidate() -> None:
    result = _result(name="candidate", seconds_per_step=0.5, peak_memory=8.0)
    for row in result["telemetry"]:
        row["match_fields"] = {
            "online_layer_match": (8, 8),
            "online_encoder_match": (7, 8),
        }
    summary = probe.summarize_profile(
        result,
        warmup_steps=2,
        max_steps=6,
        world_size=4,
        full_steps=1000,
        max_peak_memory_gib=22.0,
        required_match_fields=(
            "online_encoder_match",
            "online_decoder_hidden_match",
            "online_layer_match",
        ),
    )

    assert summary["complete_teacher_matches"] is False
    assert summary["safety_pass"] is False
    assert summary["missing_required_match_fields"] == {
        "3": ["online_decoder_hidden_match"],
        "4": ["online_decoder_hidden_match"],
        "5": ["online_decoder_hidden_match"],
        "6": ["online_decoder_hidden_match"],
    }
    assert summary["incomplete_required_match_fields"] == {
        "3": {"online_encoder_match": [7, 8]},
        "4": {"online_encoder_match": [7, 8]},
        "5": {"online_encoder_match": [7, 8]},
        "6": {"online_encoder_match": [7, 8]},
    }


def _result(
    *,
    name: str,
    seconds_per_step: float,
    peak_memory: float,
    missing: int = 0,
) -> dict[str, object]:
    telemetry = []
    for step in range(1, 7):
        telemetry.append(
            {
                "step": step,
                "elapsed_seconds": step * seconds_per_step,
                "loss": 0.1 - step * 0.001,
                "cosine": 0.96 + step * 0.001,
                "match_count": 8,
                "match_total": 8,
                "missing_total": missing if step == 6 else 0,
                "max_abs_frame_delta": 0,
            }
        )
    return {
        "profile": {"name": name, "batch_size": 36, "frame_budget": 24_000},
        "return_code": 0,
        "gpu_peak_memory_gib": {"0": peak_memory, "1": peak_memory},
        "gpu_memory_monitor_errors": [],
        "telemetry": telemetry,
        "fixed_eval_quality": {
            "complete": True,
            "eval_samples": 256,
            "eval_loss": 0.10,
            "primary_component": "mixer",
            "mean_cosine": 0.970,
            "provenance": {"fixed_eval": "same-256"},
        },
    }


def test_profile_summary_and_selection_use_projected_complete_coverage() -> None:
    baseline_result = _result(name="baseline", seconds_per_step=1.0, peak_memory=8.0)
    candidate_result = _result(name="candidate", seconds_per_step=1.4, peak_memory=18.0)
    unsafe_result = _result(name="unsafe", seconds_per_step=0.5, peak_memory=23.0)
    rows = []
    for result, full_steps in (
        (baseline_result, 1000),
        (candidate_result, 500),
        (unsafe_result, 400),
    ):
        summary = probe.summarize_profile(
            result,
            warmup_steps=2,
            max_steps=6,
            world_size=4,
            full_steps=full_steps,
            max_peak_memory_gib=22.0,
        )
        rows.append({"profile": result["profile"], "summary": summary})

    assert rows[0]["summary"]["safety_pass"] is True
    assert rows[1]["summary"]["safety_pass"] is True
    assert rows[2]["summary"]["safety_pass"] is False
    assert math.isclose(rows[0]["summary"]["steps_per_second"], 1.0)

    selection = probe.select_profile(
        rows,
        baseline_name="baseline",
        min_improvement_ratio=0.10,
    )

    assert selection["decision"] == "candidate_recommended"
    assert selection["recommended_profile"] == "candidate"
    assert selection["formal_admission"] is False


def test_missing_teacher_alignment_rejects_fast_candidate() -> None:
    baseline_result = _result(name="baseline", seconds_per_step=1.0, peak_memory=8.0)
    candidate_result = _result(
        name="candidate",
        seconds_per_step=0.5,
        peak_memory=8.0,
        missing=1,
    )
    rows = []
    for result in (baseline_result, candidate_result):
        summary = probe.summarize_profile(
            result,
            warmup_steps=2,
            max_steps=6,
            world_size=4,
            full_steps=1000,
            max_peak_memory_gib=22.0,
        )
        rows.append({"profile": result["profile"], "summary": summary})

    selection = probe.select_profile(
        rows,
        baseline_name="baseline",
        min_improvement_ratio=0.10,
    )

    assert rows[1]["summary"]["missing_total"] == 1
    assert rows[1]["summary"]["safety_pass"] is False
    assert selection["decision"] == "keep_baseline"
    assert selection["recommended_profile"] == "baseline"


def test_failed_larger_candidate_does_not_mask_safe_candidate() -> None:
    baseline_result = _result(name="baseline", seconds_per_step=1.0, peak_memory=8.0)
    safe_result = _result(name="safe", seconds_per_step=1.0, peak_memory=16.0)
    failed_result = _result(name="failed", seconds_per_step=0.5, peak_memory=20.0)
    failed_result["return_code"] = 1
    rows = []
    for result, full_steps in (
        (baseline_result, 1_000),
        (safe_result, 700),
        (failed_result, 400),
    ):
        rows.append(
            {
                "profile": result["profile"],
                "summary": probe.summarize_profile(
                    result,
                    warmup_steps=2,
                    max_steps=6,
                    world_size=4,
                    full_steps=full_steps,
                    max_peak_memory_gib=22.0,
                ),
            }
        )

    selection = probe.select_profile(
        rows,
        baseline_name="baseline",
        min_improvement_ratio=0.10,
    )

    assert rows[2]["summary"]["process_ok"] is False
    assert rows[2]["summary"]["safety_pass"] is False
    assert selection["decision"] == "candidate_recommended"
    assert selection["recommended_profile"] == "safe"


def test_alignment_quality_regression_rejects_faster_candidate() -> None:
    baseline_result = _result(name="baseline", seconds_per_step=1.0, peak_memory=8.0)
    candidate_result = _result(name="candidate", seconds_per_step=1.0, peak_memory=8.0)
    candidate_result["fixed_eval_quality"]["eval_loss"] = 0.12
    candidate_result["fixed_eval_quality"]["mean_cosine"] = 0.95
    rows = []
    for result, full_steps in ((baseline_result, 1000), (candidate_result, 500)):
        summary = probe.summarize_profile(
            result,
            warmup_steps=2,
            max_steps=6,
            world_size=4,
            full_steps=full_steps,
            max_peak_memory_gib=22.0,
        )
        rows.append({"profile": result["profile"], "summary": summary})

    selection = probe.select_profile(
        rows,
        baseline_name="baseline",
        min_improvement_ratio=0.10,
        max_loss_regression_ratio=0.05,
        max_cosine_regression=0.005,
    )

    comparison = selection["comparisons"][0]
    assert comparison["improvement_ratio"] == 0.5
    assert comparison["loss_regression_ratio"] > 0.05
    assert comparison["cosine_regression"] > 0.005
    assert comparison["quality_pass"] is False
    assert comparison["admissible"] is False
    assert selection["decision"] == "keep_baseline"
    assert selection["recommended_profile"] == "baseline"


def test_training_batch_quality_cannot_override_fixed_eval_regression() -> None:
    baseline_result = _result(name="baseline", seconds_per_step=1.0, peak_memory=8.0)
    candidate_result = _result(name="candidate", seconds_per_step=1.0, peak_memory=8.0)
    for row in candidate_result["telemetry"]:
        row["loss"] = 0.001
        row["cosine"] = 0.999
    candidate_result["fixed_eval_quality"]["eval_loss"] = 0.11
    candidate_result["fixed_eval_quality"]["mean_cosine"] = 0.96

    rows = []
    for result, full_steps in ((baseline_result, 1000), (candidate_result, 500)):
        summary = probe.summarize_profile(
            result,
            warmup_steps=2,
            max_steps=6,
            world_size=4,
            full_steps=full_steps,
            max_peak_memory_gib=22.0,
        )
        rows.append({"profile": result["profile"], "summary": summary})

    assert rows[1]["summary"]["training_mean_loss"] == pytest.approx(0.001)
    assert rows[1]["summary"]["mean_loss"] == pytest.approx(0.11)
    selection = probe.select_profile(
        rows,
        baseline_name="baseline",
        min_improvement_ratio=0.10,
    )
    assert selection["comparisons"][0]["quality_pass"] is False
    assert selection["recommended_profile"] == "baseline"


def test_git_worktree_changes_preserve_porcelain_evidence(
    monkeypatch: object,
) -> None:
    class Result:
        stdout = " M Spec.md\n?? new.py\n"

    def run(*args: object, **kwargs: object) -> Result:
        del args, kwargs
        return Result()

    monkeypatch.setattr(probe.subprocess, "run", run)

    assert probe._git_worktree_changes() == (" M Spec.md", "?? new.py")
