from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
probe = importlib.import_module("scripts.benchmark_stage211_batch_profiles")


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
    assert config["step_eval_every"] is None
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
    assert point.missing_total == 0
    assert point.max_abs_frame_delta == 0


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
