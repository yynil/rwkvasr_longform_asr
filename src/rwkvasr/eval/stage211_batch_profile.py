from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    sha256_file,
    validate_stage211_phase_train_config,
)


STAGE211_BATCH_PROFILE_PREFLIGHT_SCHEMA_VERSION = 3
STAGE211_BATCH_PROFILE_ADMISSION_SCHEMA_VERSION = 1
STAGE211_PROBE_ARTIFACT_CLEANUP_SCHEMA_VERSION = 1
STAGE211_BATCH_PROFILE_PHASES = ("mixer", "block", "logits")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FLOAT_TOLERANCE = 1.0e-12


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return payload


def _validate_bound_file(
    record: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> Path:
    path = Path(str(record.get(path_key) or "")).expanduser().resolve()
    fingerprint = record.get(sha256_key)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError(f"{label} has an invalid SHA-256 binding.")
    if sha256_file(path) != fingerprint:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _require_exact_float(actual: Any, expected: float, *, label: str) -> float:
    if not isinstance(actual, (float, int)) or isinstance(actual, bool):
        raise ValueError(f"{label} must be numeric.")
    value = float(actual)
    if not math.isfinite(value) or not math.isclose(
        value,
        float(expected),
        rel_tol=_FLOAT_TOLERANCE,
        abs_tol=_FLOAT_TOLERANCE,
    ):
        raise ValueError(f"{label} mismatch: actual={actual!r} expected={expected!r}")
    return value


def _require_optional_exact_float(
    actual: Any,
    expected: float | None,
    *,
    label: str,
) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{label} mismatch: actual={actual!r} expected=None")
        return
    _require_exact_float(actual, expected, label=label)


def _validate_profile_config(
    row: dict[str, Any],
    *,
    report: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    profile = row.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("Stage211 batch preflight profile record is invalid.")
    name = profile.get("name")
    batch_size = profile.get("batch_size")
    frame_budget = profile.get("frame_budget")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(frame_budget, int)
        or isinstance(frame_budget, bool)
        or frame_budget <= 0
    ):
        raise ValueError("Stage211 batch preflight profile values are invalid.")
    config_path = _validate_bound_file(
        row,
        path_key="config_path",
        sha256_key="config_sha256",
        label=f"Stage211 batch preflight {name} config",
    )
    config = load_yaml(config_path)
    validate_stage211_phase_train_config(config, phase=phase)
    world_size = int(report["world_size"])
    warmup_steps = int(report["warmup_steps"])
    measure_steps = int(report["measure_steps"])
    expected_fields = {
        "stage211_batch_profile_probe_phase": phase,
        "max_steps": warmup_steps + measure_steps,
        "batch_size": batch_size,
        "batch_token_budget": frame_budget,
        "length_bucket_frame_budget": frame_budget,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "init_checkpoint_path": str(Path(report["init_checkpoint_path"]).resolve()),
        "webdataset_bucket_manifest_path": str(
            Path(report["bucket_manifest_path"]).resolve()
        ),
        "resume_from": None,
        "resume_tag": None,
        "wandb_enabled": False,
        "step_eval_every": None,
        "save_deepspeed_sharded_checkpoints": False,
        "output_dir": str((config_path.parent / "run").resolve()),
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Stage211 batch preflight {name} config mismatch: "
                f"key={key} actual={config.get(key)!r} expected={expected!r}"
            )
    cleanup = row.get("probe_artifact_cleanup")
    expected_cleanup = {
        "schema_version": STAGE211_PROBE_ARTIFACT_CLEANUP_SCHEMA_VERSION,
        "artifact": "probe_artifact_cleanup",
        "complete": True,
        "run_dir": str((config_path.parent / "run").resolve()),
        "exists_after": False,
    }
    if not isinstance(cleanup, dict) or any(
        cleanup.get(key) != value for key, value in expected_cleanup.items()
    ):
        raise ValueError(f"Stage211 batch preflight {name} cleanup proof is invalid.")
    if not isinstance(cleanup.get("existed_before"), bool):
        raise ValueError(f"Stage211 batch preflight {name} cleanup state is invalid.")
    for key in ("files_removed", "directories_removed", "bytes_removed"):
        value = cleanup.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Stage211 batch preflight {name} cleanup count is invalid.")
    if cleanup["existed_before"] is False and any(
        int(cleanup[key]) != 0
        for key in ("files_removed", "directories_removed", "bytes_removed")
    ):
        raise ValueError(f"Stage211 batch preflight {name} cleanup counts are inconsistent.")
    if Path(str(cleanup["run_dir"])).exists():
        raise ValueError(f"Stage211 batch preflight {name} retained probe artifacts.")
    deepspeed = config.get("deepspeed")
    if not isinstance(deepspeed, dict):
        raise ValueError(f"Stage211 batch preflight {name} lacks a DeepSpeed config.")
    accumulation = int(deepspeed.get("gradient_accumulation_steps", 1))
    expected_deepspeed = {
        "train_micro_batch_size_per_gpu": batch_size,
        "train_batch_size": batch_size * world_size * accumulation,
    }
    for key, expected in expected_deepspeed.items():
        if deepspeed.get(key) != expected:
            raise ValueError(
                f"Stage211 batch preflight {name} DeepSpeed mismatch: "
                f"key={key} actual={deepspeed.get(key)!r} expected={expected!r}"
            )
    return profile


def _validate_safe_profile(
    row: dict[str, Any],
    *,
    report: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    profile = _validate_profile_config(row, report=report, phase=phase)
    name = str(profile["name"])
    _validate_bound_file(
        row,
        path_key="log_path",
        sha256_key="log_sha256",
        label=f"Stage211 batch preflight {name} log",
    )
    if int(row.get("return_code", -1)) != 0:
        raise ValueError(f"Stage211 batch preflight {name} process failed.")
    required_fields = row.get("required_match_fields")
    summary = row.get("summary")
    if not isinstance(required_fields, list) or not required_fields or not all(
        isinstance(value, str) and value for value in required_fields
    ):
        raise ValueError(f"Stage211 batch preflight {name} objective matches are invalid.")
    if not isinstance(summary, dict):
        raise ValueError(f"Stage211 batch preflight {name} summary is missing.")
    expected_true = (
        "process_ok",
        "complete_steps",
        "complete_teacher_matches",
        "memory_monitor_ok",
        "safety_pass",
    )
    if any(summary.get(key) is not True for key in expected_true):
        raise ValueError(f"Stage211 batch preflight {name} did not pass all safety gates.")
    if summary.get("required_match_fields") != required_fields:
        raise ValueError(f"Stage211 batch preflight {name} objective match binding changed.")
    if (
        summary.get("missing_required_match_fields") != {}
        or summary.get("incomplete_required_match_fields") != {}
        or int(summary.get("missing_total", -1)) != 0
        or int(summary.get("max_abs_frame_delta", -1)) != 0
        or int(summary.get("measurement_steps", -1)) != int(report["measure_steps"])
    ):
        raise ValueError(f"Stage211 batch preflight {name} alignment coverage is incomplete.")
    peak = float(summary.get("peak_memory_gib", float("nan")))
    memory_limit = float(report["max_peak_memory_gib"])
    rate = float(summary.get("steps_per_second", float("nan")))
    projected = float(summary.get("projected_full_coverage_seconds", float("nan")))
    loss = float(summary.get("mean_loss", float("nan")))
    cosine = float(summary.get("mean_cosine", float("nan")))
    if not all(math.isfinite(value) for value in (peak, rate, projected, loss, cosine)):
        raise ValueError(f"Stage211 batch preflight {name} has non-finite telemetry.")
    if peak > memory_limit or rate <= 0.0 or projected <= 0.0:
        raise ValueError(f"Stage211 batch preflight {name} telemetry is unsafe.")
    coverage = row.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError(f"Stage211 batch preflight {name} coverage is missing.")
    steps_per_epoch = int(coverage.get("steps_per_epoch", 0))
    epochs = int(coverage.get("formal_epochs", 0))
    if (
        steps_per_epoch <= 0
        or epochs != STAGE211_FULL_DATA_EPOCHS
        or int(coverage.get("full_coverage_steps", 0)) != steps_per_epoch * epochs
        or int(coverage.get("tail_padding_samples_per_epoch", -1)) < 0
    ):
        raise ValueError(f"Stage211 batch preflight {name} coverage accounting is invalid.")
    if not math.isclose(
        projected,
        int(coverage["full_coverage_steps"]) / rate,
        rel_tol=1.0e-9,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"Stage211 batch preflight {name} projected wall time is inconsistent.")
    return row


def validate_stage211_batch_profile_preflight(
    report_path: str | Path,
    *,
    phase: str,
    require_candidate: bool = True,
) -> dict[str, Any]:
    if phase not in STAGE211_BATCH_PROFILE_PHASES:
        raise ValueError(f"Unsupported Stage211 batch profile phase: {phase!r}")
    path = Path(report_path).expanduser().resolve()
    report = _load_json_object(path, label="Stage211 batch profile preflight")
    expected = {
        "schema_version": STAGE211_BATCH_PROFILE_PREFLIGHT_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "batch_throughput_preflight",
        "phase": phase,
        "complete": True,
        "formal_admission": False,
        "dry_run": False,
        "git_worktree_clean": True,
        "git_worktree_changes": [],
        "formal_epochs": STAGE211_FULL_DATA_EPOCHS,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 batch profile preflight is not a formal measured report.")
    if _GIT_COMMIT_PATTERN.fullmatch(str(report.get("git_commit") or "")) is None:
        raise ValueError("Stage211 batch profile preflight git commit is invalid.")
    for path_key, sha_key, label in (
        ("script_path", "script_sha256", "benchmark script"),
        ("base_config_path", "base_config_sha256", "base config"),
        ("init_checkpoint_path", "init_checkpoint_sha256", "initial checkpoint"),
        ("bucket_manifest_path", "bucket_manifest_sha256", "bucket manifest"),
    ):
        _validate_bound_file(
            report,
            path_key=path_key,
            sha256_key=sha_key,
            label=f"Stage211 batch preflight {label}",
        )
    base_config = load_yaml(Path(str(report["base_config_path"])))
    validate_stage211_phase_train_config(base_config, phase=phase)
    if Path(str(base_config.get("webdataset_bucket_manifest_path") or "")).resolve() != Path(
        str(report["bucket_manifest_path"])
    ).resolve():
        raise ValueError("Stage211 batch preflight base config uses another manifest.")
    for key in (
        "warmup_steps",
        "measure_steps",
        "max_peak_memory_gib",
        "min_improvement_ratio",
        "max_loss_regression_ratio",
        "max_cosine_regression",
    ):
        value = report.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Stage211 batch preflight {key} is invalid.")
    if int(report["warmup_steps"]) <= 0 or int(report["measure_steps"]) <= 0:
        raise ValueError("Stage211 batch preflight measurement window is empty.")
    if (
        float(report["max_peak_memory_gib"]) <= 0.0
        or not 0.0 <= float(report["min_improvement_ratio"]) < 1.0
        or float(report["max_loss_regression_ratio"]) < 0.0
        or float(report["max_cosine_regression"]) < 0.0
    ):
        raise ValueError("Stage211 batch preflight thresholds are invalid.")
    gpu_indices = report.get("gpu_indices")
    if not isinstance(gpu_indices, list) or len(gpu_indices) != STAGE211_FULL_DATA_WORLD_SIZE:
        raise ValueError("Stage211 batch preflight does not bind all four GPUs.")
    profiles = report.get("profiles")
    if not isinstance(profiles, list) or len(profiles) < 2:
        raise ValueError("Stage211 batch preflight requires baseline and candidate profiles.")
    by_name: dict[str, dict[str, Any]] = {}
    for row in profiles:
        if not isinstance(row, dict) or not isinstance(row.get("profile"), dict):
            raise ValueError("Stage211 batch preflight profile row is invalid.")
        name = row["profile"].get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("Stage211 batch preflight profile names are invalid or duplicated.")
        by_name[name] = row
    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Stage211 batch preflight selection is missing.")
    baseline_name = selection.get("baseline_profile")
    selected_name = selection.get("recommended_profile")
    decision = selection.get("decision")
    if (
        decision not in ("candidate_recommended", "keep_baseline")
        or selection.get("formal_admission") is not False
        or not isinstance(baseline_name, str)
        or not isinstance(selected_name, str)
        or baseline_name not in by_name
        or selected_name not in by_name
    ):
        raise ValueError("Stage211 batch preflight selection is not replayable.")
    if decision == "candidate_recommended" and selected_name == baseline_name:
        raise ValueError("Stage211 batch preflight candidate selection retained the baseline.")
    if decision == "keep_baseline" and selected_name != baseline_name:
        raise ValueError("Stage211 batch preflight baseline selection names a candidate.")
    baseline_profile = by_name[baseline_name]["profile"]
    if baseline_profile != {
        "name": baseline_name,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
    }:
        raise ValueError("Stage211 batch preflight baseline differs from the legacy profile.")
    baseline = _validate_safe_profile(by_name[baseline_name], report=report, phase=phase)
    baseline_summary = baseline["summary"]
    baseline_seconds = float(baseline_summary["projected_full_coverage_seconds"])
    baseline_loss = float(baseline_summary["mean_loss"])
    baseline_cosine = float(baseline_summary["mean_cosine"])
    comparisons = selection.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("Stage211 batch preflight candidate comparisons are missing.")
    comparison_by_name = {
        str(row.get("profile") or ""): row
        for row in comparisons
        if isinstance(row, dict)
    }
    candidate_names = set(by_name).difference((baseline_name,))
    if set(comparison_by_name) != candidate_names or len(comparisons) != len(candidate_names):
        raise ValueError("Stage211 batch preflight candidate comparison coverage is invalid.")
    replayed_admissible: list[tuple[float, str, float]] = []
    for candidate_name in sorted(candidate_names):
        candidate_row = by_name[candidate_name]
        _validate_profile_config(candidate_row, report=report, phase=phase)
        _validate_bound_file(
            candidate_row,
            path_key="log_path",
            sha256_key="log_sha256",
            label=f"Stage211 batch preflight {candidate_name} log",
        )
        summary = candidate_row.get("summary")
        if not isinstance(summary, dict):
            raise ValueError(
                f"Stage211 batch preflight {candidate_name} summary is missing."
            )
        if summary.get("safety_pass") is True:
            _validate_safe_profile(candidate_row, report=report, phase=phase)
        candidate_seconds_value = summary.get("projected_full_coverage_seconds")
        candidate_loss_value = summary.get("mean_loss")
        candidate_cosine_value = summary.get("mean_cosine")
        candidate_seconds = (
            float(candidate_seconds_value)
            if isinstance(candidate_seconds_value, (float, int))
            and not isinstance(candidate_seconds_value, bool)
            else None
        )
        candidate_loss = (
            float(candidate_loss_value)
            if isinstance(candidate_loss_value, (float, int))
            and not isinstance(candidate_loss_value, bool)
            else None
        )
        candidate_cosine = (
            float(candidate_cosine_value)
            if isinstance(candidate_cosine_value, (float, int))
            and not isinstance(candidate_cosine_value, bool)
            else None
        )
        improvement = (
            (baseline_seconds - candidate_seconds) / baseline_seconds
            if candidate_seconds is not None and baseline_seconds > 0.0
            else None
        )
        loss_regression = (
            (candidate_loss - baseline_loss) / max(abs(baseline_loss), 1.0e-12)
            if candidate_loss is not None
            else None
        )
        cosine_regression = (
            baseline_cosine - candidate_cosine if candidate_cosine is not None else None
        )
        quality_pass = (
            loss_regression is not None
            and loss_regression <= float(report["max_loss_regression_ratio"])
            and cosine_regression is not None
            and cosine_regression <= float(report["max_cosine_regression"])
        )
        admissible = (
            summary.get("safety_pass") is True
            and improvement is not None
            and improvement >= float(report["min_improvement_ratio"])
            and quality_pass
        )
        comparison_row = comparison_by_name[candidate_name]
        _require_optional_exact_float(
            comparison_row.get("projected_full_coverage_seconds"),
            candidate_seconds,
            label=f"{candidate_name} projected coverage",
        )
        _require_optional_exact_float(
            comparison_row.get("improvement_ratio"),
            improvement,
            label=f"{candidate_name} improvement",
        )
        _require_optional_exact_float(
            comparison_row.get("mean_loss"),
            candidate_loss,
            label=f"{candidate_name} mean loss",
        )
        _require_optional_exact_float(
            comparison_row.get("loss_regression_ratio"),
            loss_regression,
            label=f"{candidate_name} loss regression",
        )
        _require_optional_exact_float(
            comparison_row.get("mean_cosine"),
            candidate_cosine,
            label=f"{candidate_name} mean cosine",
        )
        _require_optional_exact_float(
            comparison_row.get("cosine_regression"),
            cosine_regression,
            label=f"{candidate_name} cosine regression",
        )
        if (
            comparison_row.get("quality_pass") is not quality_pass
            or comparison_row.get("admissible") is not admissible
        ):
            raise ValueError(
                f"Stage211 batch preflight {candidate_name} comparison decision changed."
            )
        if admissible:
            assert candidate_seconds is not None
            assert improvement is not None
            replayed_admissible.append((candidate_seconds, candidate_name, improvement))
    replayed_admissible.sort()
    for key in (
        "min_improvement_ratio",
        "max_loss_regression_ratio",
        "max_cosine_regression",
    ):
        _require_exact_float(selection.get(key), float(report[key]), label=f"selection {key}")
    _require_exact_float(
        selection.get("baseline_mean_loss"),
        float(baseline_summary["mean_loss"]),
        label="selection baseline mean loss",
    )
    _require_exact_float(
        selection.get("baseline_mean_cosine"),
        float(baseline_summary["mean_cosine"]),
        label="selection baseline mean cosine",
    )
    if decision == "keep_baseline":
        _require_exact_float(
            selection.get("recommended_improvement_ratio"),
            0.0,
            label="retained baseline improvement",
        )
        if replayed_admissible:
            raise ValueError(
                "Stage211 batch preflight retained the baseline despite an admissible candidate."
            )
        if require_candidate:
            raise ValueError("Stage211 batch preflight has no admissible larger profile.")
        return {
            **report,
            "report_path": str(path),
            "report_sha256": sha256_file(path),
            "selection_decision": decision,
            "selected_profile_name": baseline_name,
            "selected_profile_row": baseline,
            "selected_comparison": None,
        }

    selected = _validate_safe_profile(by_name[selected_name], report=report, phase=phase)
    if not replayed_admissible or replayed_admissible[0][1] != selected_name:
        raise ValueError("Stage211 batch preflight did not select the fastest admissible profile.")
    comparison = next(
        (
            row
            for row in comparisons
            if isinstance(row, dict) and row.get("profile") == selected_name
        ),
        None,
    )
    if (
        not isinstance(comparison, dict)
        or comparison.get("quality_pass") is not True
        or comparison.get("admissible") is not True
    ):
        raise ValueError("Stage211 selected batch profile failed its quality comparison.")
    selected_summary = selected["summary"]
    selected_seconds = float(selected_summary["projected_full_coverage_seconds"])
    improvement = (baseline_seconds - selected_seconds) / baseline_seconds
    selected_loss = float(selected_summary["mean_loss"])
    loss_regression = (selected_loss - baseline_loss) / max(abs(baseline_loss), 1.0e-12)
    selected_cosine = float(selected_summary["mean_cosine"])
    cosine_regression = baseline_cosine - selected_cosine
    _require_exact_float(
        comparison.get("improvement_ratio"), improvement, label="candidate improvement"
    )
    _require_exact_float(
        comparison.get("loss_regression_ratio"),
        loss_regression,
        label="candidate loss regression",
    )
    _require_exact_float(
        comparison.get("cosine_regression"),
        cosine_regression,
        label="candidate cosine regression",
    )
    _require_exact_float(
        selection.get("recommended_improvement_ratio"),
        improvement,
        label="recommended improvement",
    )
    if (
        improvement < float(report["min_improvement_ratio"])
        or loss_regression > float(report["max_loss_regression_ratio"])
        or cosine_regression > float(report["max_cosine_regression"])
    ):
        raise ValueError("Stage211 selected batch profile violates an admission threshold.")
    return {
        **report,
        "report_path": str(path),
        "report_sha256": sha256_file(path),
        "selection_decision": decision,
        "selected_profile_name": selected_name,
        "selected_profile_row": selected,
        "selected_comparison": comparison,
    }


def build_stage211_batch_profile_admission(
    report_path: str | Path,
    *,
    phase: str,
    admitted_by: str,
    reason: str,
) -> dict[str, Any]:
    actor = admitted_by.strip()
    rationale = reason.strip()
    if not actor or not rationale:
        raise ValueError("Stage211 batch profile admission requires an actor and reason.")
    report = validate_stage211_batch_profile_preflight(report_path, phase=phase)
    selected = report["selected_profile_row"]
    profile = selected["profile"]
    return {
        "schema_version": STAGE211_BATCH_PROFILE_ADMISSION_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "batch_profile_admission",
        "complete": True,
        "admitted": True,
        "phase": phase,
        "created_at": datetime.now(UTC).isoformat(),
        "admission_decision": "admit_recommended_profile",
        "admitted_by": actor,
        "reason": rationale,
        "preflight_report_path": report["report_path"],
        "preflight_report_sha256": report["report_sha256"],
        "preflight_git_commit": report["git_commit"],
        "preflight_script_path": report["script_path"],
        "preflight_script_sha256": report["script_sha256"],
        "base_config_path": report["base_config_path"],
        "base_config_sha256": report["base_config_sha256"],
        "init_checkpoint_path": report["init_checkpoint_path"],
        "init_checkpoint_sha256": report["init_checkpoint_sha256"],
        "bucket_manifest_path": report["bucket_manifest_path"],
        "bucket_manifest_sha256": report["bucket_manifest_sha256"],
        "world_size": report["world_size"],
        "formal_epochs": report["formal_epochs"],
        "selected_profile": dict(profile),
        "selected_coverage": dict(selected["coverage"]),
        "selected_summary": dict(selected["summary"]),
        "selected_comparison": dict(report["selected_comparison"]),
        "thresholds": {
            "max_peak_memory_gib": report["max_peak_memory_gib"],
            "min_improvement_ratio": report["min_improvement_ratio"],
            "max_loss_regression_ratio": report["max_loss_regression_ratio"],
            "max_cosine_regression": report["max_cosine_regression"],
        },
    }


def validate_stage211_batch_profile_admission(
    receipt_path: str | Path,
    *,
    phase: str,
    expected_init_checkpoint: str | Path | None = None,
    expected_bucket_manifest: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(receipt_path).expanduser().resolve()
    receipt = _load_json_object(path, label="Stage211 batch profile admission")
    expected = {
        "schema_version": STAGE211_BATCH_PROFILE_ADMISSION_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "batch_profile_admission",
        "complete": True,
        "admitted": True,
        "phase": phase,
        "admission_decision": "admit_recommended_profile",
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "formal_epochs": STAGE211_FULL_DATA_EPOCHS,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 batch profile admission contract mismatch.")
    if not str(receipt.get("admitted_by") or "").strip() or not str(
        receipt.get("reason") or ""
    ).strip():
        raise ValueError("Stage211 batch profile admission lacks explicit review metadata.")
    report_path = _validate_bound_file(
        receipt,
        path_key="preflight_report_path",
        sha256_key="preflight_report_sha256",
        label="Stage211 admitted batch preflight report",
    )
    report = validate_stage211_batch_profile_preflight(report_path, phase=phase)
    selected = report["selected_profile_row"]
    exact_fields = {
        "preflight_git_commit": report["git_commit"],
        "preflight_script_path": report["script_path"],
        "preflight_script_sha256": report["script_sha256"],
        "base_config_path": report["base_config_path"],
        "base_config_sha256": report["base_config_sha256"],
        "init_checkpoint_path": report["init_checkpoint_path"],
        "init_checkpoint_sha256": report["init_checkpoint_sha256"],
        "bucket_manifest_path": report["bucket_manifest_path"],
        "bucket_manifest_sha256": report["bucket_manifest_sha256"],
        "selected_profile": selected["profile"],
        "selected_coverage": selected["coverage"],
        "selected_summary": selected["summary"],
        "selected_comparison": report["selected_comparison"],
        "thresholds": {
            "max_peak_memory_gib": report["max_peak_memory_gib"],
            "min_improvement_ratio": report["min_improvement_ratio"],
            "max_loss_regression_ratio": report["max_loss_regression_ratio"],
            "max_cosine_regression": report["max_cosine_regression"],
        },
    }
    if any(receipt.get(key) != value for key, value in exact_fields.items()):
        raise ValueError("Stage211 batch profile admission differs from its measured report.")
    if expected_init_checkpoint is not None and Path(
        str(receipt["init_checkpoint_path"])
    ).resolve() != Path(expected_init_checkpoint).expanduser().resolve():
        raise ValueError("Stage211 batch profile admission uses another initial checkpoint.")
    if expected_bucket_manifest is not None and Path(
        str(receipt["bucket_manifest_path"])
    ).resolve() != Path(expected_bucket_manifest).expanduser().resolve():
        raise ValueError("Stage211 batch profile admission uses another bucket manifest.")
    profile = receipt["selected_profile"]
    if (
        int(profile["batch_size"]) <= STAGE211_FULL_DATA_BATCH_SIZE
        and int(profile["frame_budget"]) <= STAGE211_FULL_DATA_FRAME_BUDGET
    ):
        raise ValueError("Stage211 admitted profile is not larger than the legacy profile.")
    return {
        **receipt,
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
    }
