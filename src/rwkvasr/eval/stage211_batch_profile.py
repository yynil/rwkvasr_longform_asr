from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    STAGE211_STACKED_SAFE_BATCH_SIZE,
    STAGE211_STACKED_SAFE_FRAME_BUDGET,
    sha256_file,
    validate_stage211_phase_train_config,
)


STAGE211_BATCH_PROFILE_PREFLIGHT_SCHEMA_VERSION = 7
STAGE211_BATCH_PROFILE_ADMISSION_SCHEMA_VERSION = 3
STAGE211_PROBE_ARTIFACT_CLEANUP_SCHEMA_VERSION = 1
STAGE211_PROBE_FIXED_EVAL_CAPTURE_SCHEMA_VERSION = 1
STAGE211_BATCH_PROFILE_PHASES = ("mixer", "block", "logits")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_FLOAT_TOLERANCE = 1.0e-12
_STAGE211_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FIXED_EVAL_COMPONENTS = {
    "mixer": ("mixer",),
    "block": ("mixer", "ffn", "block"),
    "logits": ("mixer", "ffn", "block"),
}
_FIXED_EVAL_PRIMARY_COMPONENT = {
    "mixer": "mixer",
    "block": "block",
    "logits": "block",
}
_FIXED_EVAL_LOGIT_METRICS = (
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


@dataclass(frozen=True)
class CandidateDominancePolicy:
    full_coverage_steps: int
    baseline_projected_full_coverage_seconds: float
    min_improvement_ratio: float
    rejection_ratio: float = 4.0
    min_points: int = 12


def candidate_dominance_evidence(
    points: list[Mapping[str, Any]],
    policy: CandidateDominancePolicy,
) -> dict[str, Any] | None:
    if policy.full_coverage_steps <= 0:
        raise ValueError("candidate dominance requires positive full-coverage steps")
    if policy.baseline_projected_full_coverage_seconds <= 0.0:
        raise ValueError("candidate dominance requires positive baseline coverage time")
    if not 0.0 <= policy.min_improvement_ratio < 1.0:
        raise ValueError("candidate dominance improvement ratio must be in [0, 1)")
    if policy.rejection_ratio <= 1.0 or policy.min_points < 3:
        raise ValueError("candidate dominance requires rejection_ratio > 1 and min_points >= 3")
    normalized: dict[int, tuple[float, Mapping[str, Any]]] = {}
    for point in points:
        step = point.get("step")
        elapsed_seconds = point.get("elapsed_seconds")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step <= 0
            or not isinstance(elapsed_seconds, (float, int))
            or isinstance(elapsed_seconds, bool)
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) <= 0.0
        ):
            raise ValueError("candidate dominance telemetry is invalid")
        normalized[step] = (float(elapsed_seconds), point)
    ordered = [normalized[step] for step in sorted(normalized)]
    if len(ordered) < policy.min_points:
        return None
    window = ordered[-policy.min_points :]
    intervals = [
        (right[0] - left[0]) / (int(right[1]["step"]) - int(left[1]["step"]))
        for left, right in zip(window, window[1:])
        if int(right[1]["step"]) > int(left[1]["step"]) and right[0] > left[0]
    ]
    if len(intervals) != policy.min_points - 1:
        return None
    elapsed = window[-1][0] - window[0][0]
    completed_steps = int(window[-1][1]["step"]) - int(window[0][1]["step"])
    if elapsed <= 0.0 or completed_steps <= 0:
        return None
    window_rate = completed_steps / elapsed
    median_seconds_per_step = statistics.median(intervals)
    if window_rate <= 0.0 or median_seconds_per_step <= 0.0:
        return None
    median_rate = 1.0 / median_seconds_per_step
    admissible_seconds = policy.baseline_projected_full_coverage_seconds * (
        1.0 - policy.min_improvement_ratio
    )
    rejection_limit_seconds = admissible_seconds * policy.rejection_ratio
    window_projected_seconds = policy.full_coverage_steps / window_rate
    median_projected_seconds = policy.full_coverage_steps / median_rate
    if min(window_projected_seconds, median_projected_seconds) <= rejection_limit_seconds:
        return None
    return {
        "artifact": "candidate_dominance_termination",
        "complete": True,
        "reason": "mathematically_dominated_after_conservative_probe_window",
        "observed_points": len(ordered),
        "window_points": len(window),
        "window_first_step": int(window[0][1]["step"]),
        "window_last_step": int(window[-1][1]["step"]),
        "window_rate_steps_per_second": window_rate,
        "median_rate_steps_per_second": median_rate,
        "full_coverage_steps": policy.full_coverage_steps,
        "baseline_projected_full_coverage_seconds": (
            policy.baseline_projected_full_coverage_seconds
        ),
        "min_improvement_ratio": policy.min_improvement_ratio,
        "maximum_admissible_coverage_seconds": admissible_seconds,
        "rejection_ratio": policy.rejection_ratio,
        "rejection_limit_seconds": rejection_limit_seconds,
        "window_projected_full_coverage_seconds": window_projected_seconds,
        "median_projected_full_coverage_seconds": median_projected_seconds,
    }


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


def _validate_git_bound_source(
    record: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    git_commit_key: str,
    label: str,
    repository_root: Path = _STAGE211_REPOSITORY_ROOT,
) -> Path:
    path = Path(str(record.get(path_key) or "")).expanduser().resolve()
    fingerprint = record.get(sha256_key)
    if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise ValueError(f"{label} has an invalid SHA-256 binding.")
    if path.is_file() and path.stat().st_size > 0 and sha256_file(path) == fingerprint:
        return path

    commit = str(record.get(git_commit_key) or "")
    if _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"{label} has an invalid Git commit binding.")
    root = repository_root.expanduser().resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path is outside the bound repository: {path}") from error

    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "HEAD"),
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if ancestor.returncode != 0:
        detail = ancestor.stderr.decode("utf-8", errors="replace").strip()
        suffix = f" ({detail})" if detail else ""
        raise ValueError(
            f"{label} Git commit is unavailable or not an ancestor of HEAD: {commit}{suffix}"
        )
    historical = subprocess.run(
        ("git", "show", f"{commit}:{relative_path.as_posix()}"),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if historical.returncode != 0 or not historical.stdout:
        detail = historical.stderr.decode("utf-8", errors="replace").strip()
        suffix = f" ({detail})" if detail else ""
        raise ValueError(
            f"{label} historical Git blob is missing or empty: "
            f"{commit}:{relative_path.as_posix()}{suffix}"
        )
    if hashlib.sha256(historical.stdout).hexdigest() != fingerprint:
        raise ValueError(
            f"{label} historical Git blob SHA-256 mismatch: "
            f"{commit}:{relative_path.as_posix()}"
        )
    return path


def _fixed_eval_provenance(
    value: Any,
    *,
    expected_manifest: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Stage211 batch-profile fixed eval lacks provenance.")
    manifest = expected_manifest.expanduser().resolve()
    expected_manifest_sha256 = sha256_file(manifest)
    expected = {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
        "feature_seed": 0,
        "bucket_manifest_path": str(manifest),
        "bucket_manifest_sha256": expected_manifest_sha256,
        "split_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Stage211 batch-profile fixed-eval provenance changed.")
    parts = value.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Stage211 batch-profile fixed eval has no bound eval parts.")
    canonical_parts: list[dict[str, Any]] = []
    total_samples = 0
    seen_paths: set[Path] = set()
    for raw_part in parts:
        if not isinstance(raw_part, dict):
            raise ValueError("Stage211 batch-profile fixed-eval part is invalid.")
        part = Path(str(raw_part.get("path") or "")).expanduser().resolve()
        part_sha256 = raw_part.get("sha256")
        num_samples = raw_part.get("num_samples")
        if (
            part in seen_paths
            or not part.is_file()
            or part.stat().st_size <= 0
            or not isinstance(part_sha256, str)
            or _SHA256_PATTERN.fullmatch(part_sha256) is None
            or sha256_file(part) != part_sha256
            or not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError("Stage211 batch-profile fixed-eval part binding is invalid.")
        seen_paths.add(part)
        total_samples += num_samples
        canonical_parts.append(
            {
                "path": str(part),
                "sha256": part_sha256,
                "num_samples": num_samples,
            }
        )
    if total_samples != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES:
        raise ValueError("Stage211 batch-profile fixed-eval parts do not cover 256 samples.")
    return {**expected, "parts": canonical_parts}


def validate_stage211_batch_profile_fixed_eval(
    report_path: str | Path,
    *,
    phase: str,
    expected_step: int,
    expected_manifest: str | Path,
) -> dict[str, Any]:
    if phase not in STAGE211_BATCH_PROFILE_PHASES:
        raise ValueError(f"Unsupported Stage211 batch profile phase: {phase!r}")
    path = Path(report_path).expanduser().resolve()
    report = load_yaml(path)
    if not isinstance(report, dict):
        raise ValueError(f"Stage211 batch-profile fixed eval is invalid: {path}")
    if (
        int(report.get("step", -1)) != int(expected_step)
        or int(report.get("eval_samples", -1)) != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES
    ):
        raise ValueError("Stage211 batch-profile fixed eval has wrong step or sample count.")
    eval_loss = float(report.get("eval_loss", float("nan")))
    if not math.isfinite(eval_loss):
        raise ValueError("Stage211 batch-profile fixed-eval loss is non-finite.")
    provenance = _fixed_eval_provenance(
        report.get("eval_provenance"),
        expected_manifest=Path(expected_manifest),
    )
    components = report.get("layer_components")
    if not isinstance(components, dict):
        raise ValueError("Stage211 batch-profile fixed eval lacks layer components.")
    expected_layers = {str(layer_id) for layer_id in range(70)}
    validated_components: dict[str, dict[str, dict[str, float]]] = {}
    for component_name in _FIXED_EVAL_COMPONENTS[phase]:
        raw_layers = components.get(component_name)
        if not isinstance(raw_layers, dict) or set(raw_layers) != expected_layers:
            raise ValueError(
                f"Stage211 batch-profile fixed eval lacks exact 70-layer {component_name} coverage."
            )
        validated_layers: dict[str, dict[str, float]] = {}
        for layer_id, raw_metrics in raw_layers.items():
            if not isinstance(raw_metrics, dict):
                raise ValueError("Stage211 batch-profile fixed-eval layer is invalid.")
            metrics = {
                name: float(raw_metrics.get(name, float("nan")))
                for name in ("loss", "cosine", "rms_ratio")
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError("Stage211 batch-profile fixed-eval layer metrics are non-finite.")
            validated_layers[layer_id] = metrics
        validated_components[component_name] = validated_layers
    if phase in {"block", "logits"}:
        decoder_hidden = report.get("decoder_hidden_metrics")
        decoder_loss = (
            float(decoder_hidden.get("loss", float("nan")))
            if isinstance(decoder_hidden, dict)
            else float("nan")
        )
        if not math.isfinite(decoder_loss):
            raise ValueError("Stage211 batch-profile fixed eval lacks finite decoder-hidden loss.")
    if phase == "logits":
        raw_logits = report.get("logit_metrics")
        if not isinstance(raw_logits, dict):
            raise ValueError("Stage211 batch-profile fixed eval lacks Logits metrics.")
        logits = {
            name: float(raw_logits.get(name, float("nan"))) for name in _FIXED_EVAL_LOGIT_METRICS
        }
        if not all(math.isfinite(value) for value in logits.values()):
            raise ValueError("Stage211 batch-profile fixed-eval Logits metrics are non-finite.")
        if (
            int(round(logits["matched_utterances"])) != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES
            or logits["missing_utterances"] != 0.0
            or logits["mean_frame_delta"] != 0.0
        ):
            raise ValueError(
                "Stage211 batch-profile fixed-eval Logits coverage/frame parity failed."
            )
    primary_component = _FIXED_EVAL_PRIMARY_COMPONENT[phase]
    primary_layers = validated_components[primary_component]
    mean_cosine = sum(primary_layers[str(layer_id)]["cosine"] for layer_id in range(70)) / 70.0
    if not math.isfinite(mean_cosine):
        raise ValueError("Stage211 batch-profile fixed-eval mean cosine is non-finite.")
    return {
        "complete": True,
        "report_path": str(path),
        "report_sha256": sha256_file(path),
        "step": int(expected_step),
        "eval_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
        "eval_loss": eval_loss,
        "primary_component": primary_component,
        "mean_cosine": mean_cosine,
        "provenance": provenance,
    }


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
    num_workers = profile.get("num_workers")
    gradient_checkpointing = profile.get("gradient_checkpointing")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or not isinstance(frame_budget, int)
        or isinstance(frame_budget, bool)
        or frame_budget <= 0
        or not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers <= 0
        or type(gradient_checkpointing) is not bool
    ):
        raise ValueError("Stage211 batch preflight profile values are invalid.")
    config_path = _validate_bound_file(
        row,
        path_key="config_path",
        sha256_key="config_sha256",
        label=f"Stage211 batch preflight {name} config",
    )
    config = load_yaml(config_path)
    validate_stage211_phase_train_config(
        config,
        phase=phase,
        expected_gradient_checkpointing=gradient_checkpointing,
    )
    world_size = int(report["world_size"])
    warmup_steps = int(report["warmup_steps"])
    measure_steps = int(report["measure_steps"])
    expected_fields = {
        "stage211_batch_profile_probe_phase": phase,
        "max_steps": warmup_steps + measure_steps,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "gradient_checkpointing": gradient_checkpointing,
        "batch_token_budget": frame_budget,
        "length_bucket_frame_budget": frame_budget,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "init_checkpoint_path": str(Path(report["init_checkpoint_path"]).resolve()),
        "webdataset_bucket_manifest_path": str(Path(report["bucket_manifest_path"]).resolve()),
        "resume_from": None,
        "resume_tag": None,
        "wandb_enabled": False,
        "step_eval_at_start": False,
        "step_eval_every": warmup_steps + measure_steps,
        "step_eval_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
        "step_eval_split": "eval",
        "step_eval_shuffle": False,
        "step_eval_cache_batches": True,
        "step_eval_feature_seed": 0,
        "save_deepspeed_sharded_checkpoints": False,
        "output_dir": str((config_path.parent / "run").resolve()),
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Stage211 batch preflight {name} config mismatch: "
                f"key={key} actual={config.get(key)!r} expected={expected!r}"
            )
    fixed_eval_capture = row.get("fixed_eval_capture")
    expected_fixed_eval_path = (config_path.parent / "fixed_eval.yaml").resolve()
    if not isinstance(fixed_eval_capture, dict):
        raise ValueError(f"Stage211 batch preflight {name} fixed-eval capture is missing.")
    expected_capture = {
        "schema_version": STAGE211_PROBE_FIXED_EVAL_CAPTURE_SCHEMA_VERSION,
        "artifact": "probe_fixed_eval_capture",
        "source_name": f"step_eval_layers_step-{warmup_steps + measure_steps}.yaml",
        "report_path": str(expected_fixed_eval_path),
    }
    if any(
        fixed_eval_capture.get(key) != value for key, value in expected_capture.items()
    ) or not isinstance(fixed_eval_capture.get("captured"), bool):
        raise ValueError(f"Stage211 batch preflight {name} fixed-eval capture is invalid.")
    if fixed_eval_capture["captured"] is True:
        _validate_bound_file(
            fixed_eval_capture,
            path_key="report_path",
            sha256_key="report_sha256",
            label=f"Stage211 batch preflight {name} fixed eval",
        )
    else:
        if fixed_eval_capture.get("report_sha256") is not None:
            raise ValueError(f"Stage211 batch preflight {name} absent fixed eval has a SHA-256.")
        if expected_fixed_eval_path.exists():
            raise ValueError(f"Stage211 batch preflight {name} fixed-eval capture state is stale.")
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
        int(cleanup[key]) != 0 for key in ("files_removed", "directories_removed", "bytes_removed")
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
    if (
        not isinstance(required_fields, list)
        or not required_fields
        or not all(isinstance(value, str) and value for value in required_fields)
    ):
        raise ValueError(f"Stage211 batch preflight {name} objective matches are invalid.")
    if not isinstance(summary, dict):
        raise ValueError(f"Stage211 batch preflight {name} summary is missing.")
    expected_true = (
        "process_ok",
        "complete_steps",
        "complete_teacher_matches",
        "fixed_eval_complete",
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
    fixed_eval_capture = row["fixed_eval_capture"]
    if fixed_eval_capture.get("captured") is not True:
        raise ValueError(f"Stage211 batch preflight {name} lacks a captured fixed eval.")
    fixed_eval = validate_stage211_batch_profile_fixed_eval(
        fixed_eval_capture["report_path"],
        phase=phase,
        expected_step=int(report["warmup_steps"]) + int(report["measure_steps"]),
        expected_manifest=report["bucket_manifest_path"],
    )
    expected_fixed_summary = {
        "fixed_eval_samples": fixed_eval["eval_samples"],
        "fixed_eval_primary_component": fixed_eval["primary_component"],
        "fixed_eval_provenance": fixed_eval["provenance"],
        "quality_source": "terminal_fixed_eval_256",
        "alignment_mean_aggregation": (
            "fixed_eval_total_objective_and_unweighted_70_layer_primary_component"
        ),
    }
    if any(summary.get(key) != value for key, value in expected_fixed_summary.items()):
        raise ValueError(f"Stage211 batch preflight {name} fixed-eval summary changed.")
    _require_exact_float(
        summary.get("mean_loss"),
        float(fixed_eval["eval_loss"]),
        label=f"{name} fixed-eval loss",
    )
    _require_exact_float(
        summary.get("mean_cosine"),
        float(fixed_eval["mean_cosine"]),
        label=f"{name} fixed-eval mean cosine",
    )
    peak = float(summary.get("peak_memory_gib", float("nan")))
    memory_limit = float(report["max_peak_memory_gib"])
    rate = float(summary.get("steps_per_second", float("nan")))
    projected = float(summary.get("projected_full_coverage_seconds", float("nan")))
    loss = float(summary.get("mean_loss", float("nan")))
    cosine = float(summary.get("mean_cosine", float("nan")))
    training_loss = float(summary.get("training_mean_loss", float("nan")))
    training_cosine = float(summary.get("training_mean_cosine", float("nan")))
    if not all(
        math.isfinite(value)
        for value in (
            peak,
            rate,
            projected,
            loss,
            cosine,
            training_loss,
            training_cosine,
        )
    ):
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


def _validate_loader_worker_search(
    report: dict[str, Any],
    *,
    base_config: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    admissible_names: set[str],
) -> None:
    search = report.get("loader_worker_search")
    if not isinstance(search, dict):
        raise ValueError("Stage211 batch preflight loader-worker search is missing.")
    world_size = int(report["world_size"])
    configured_workers = int(base_config.get("num_workers", 0) or 0)
    physical_cores = search.get("physical_cores")
    logical_cpus = search.get("logical_cpus")
    balanced_workers = search.get("balanced_num_workers")
    common_expected = {
        "schema_version": 1,
        "world_size": world_size,
        "configured_num_workers": configured_workers,
    }
    if any(search.get(key) != value for key, value in common_expected.items()):
        raise ValueError("Stage211 batch preflight loader-worker search contract changed.")
    if (
        configured_workers <= 0
        or not isinstance(physical_cores, int)
        or isinstance(physical_cores, bool)
        or physical_cores <= 0
        or not isinstance(logical_cpus, int)
        or isinstance(logical_cpus, bool)
        or logical_cpus <= 0
        or physical_cores > logical_cpus
        or not isinstance(balanced_workers, int)
        or isinstance(balanced_workers, bool)
        or balanced_workers != max(1, physical_cores // world_size)
        or search.get("topology_source")
        not in ("linux_sysfs_affinity", "logical_affinity_fallback")
    ):
        raise ValueError("Stage211 batch preflight loader-worker topology is invalid.")
    mode = search.get("mode")
    baseline_name = search.get("baseline_profile")
    if not isinstance(baseline_name, str) or baseline_name not in by_name:
        raise ValueError("Stage211 batch preflight loader-worker baseline is invalid.")
    baseline_profile = by_name[baseline_name]["profile"]
    if int(baseline_profile["num_workers"]) != configured_workers:
        raise ValueError("Stage211 batch preflight baseline worker count changed.")
    if mode == "explicit_profiles":
        expected = {
            "enabled": False,
            "candidate_profile": None,
            "selection_decision": "explicit_profiles",
            "selected_profile": None,
            "selected_num_workers": None,
        }
        if any(search.get(key) != value for key, value in expected.items()):
            raise ValueError("Stage211 explicit loader-worker profile contract changed.")
        return
    if mode != "automatic_balanced_candidate":
        raise ValueError("Stage211 batch preflight loader-worker mode is unsupported.")
    enabled = balanced_workers != configured_workers
    if search.get("enabled") is not enabled:
        raise ValueError("Stage211 batch preflight loader-worker enablement changed.")
    if not enabled:
        expected = {
            "candidate_profile": None,
            "selection_decision": "configured_workers_already_balanced",
            "selected_profile": baseline_name,
            "selected_num_workers": configured_workers,
        }
        if any(search.get(key) != value for key, value in expected.items()):
            raise ValueError("Stage211 balanced loader-worker baseline changed.")
        if any(
            int(row["profile"]["num_workers"]) != configured_workers for row in by_name.values()
        ):
            raise ValueError("Stage211 profiles do not use the balanced configured workers.")
        return
    candidate_name = search.get("candidate_profile")
    if not isinstance(candidate_name, str) or candidate_name not in by_name:
        raise ValueError("Stage211 loader-worker calibration candidate is missing.")
    candidate_profile = by_name[candidate_name]["profile"]
    if any(
        (
            int(candidate_profile["batch_size"]) != int(baseline_profile["batch_size"]),
            int(candidate_profile["frame_budget"]) != int(baseline_profile["frame_budget"]),
            int(candidate_profile["num_workers"]) != balanced_workers,
        )
    ):
        raise ValueError("Stage211 loader-worker candidate changes more than workers.")
    candidate_selected = candidate_name in admissible_names
    selected_name = candidate_name if candidate_selected else baseline_name
    selected_workers = balanced_workers if candidate_selected else configured_workers
    expected = {
        "selection_decision": (
            "balanced_workers_selected" if candidate_selected else "configured_workers_retained"
        ),
        "selected_profile": selected_name,
        "selected_num_workers": selected_workers,
    }
    if any(search.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 loader-worker calibration decision changed.")
    for name, row in by_name.items():
        if name in (baseline_name, candidate_name):
            continue
        if int(row["profile"]["num_workers"]) != selected_workers:
            raise ValueError(
                "Stage211 larger batch profiles do not use the calibrated worker count."
            )


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
    _validate_git_bound_source(
        report,
        path_key="script_path",
        sha256_key="script_sha256",
        git_commit_key="git_commit",
        label="Stage211 batch preflight benchmark script",
    )
    for path_key, sha_key, label in (
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
    if (
        Path(str(base_config.get("webdataset_bucket_manifest_path") or "")).resolve()
        != Path(str(report["bucket_manifest_path"])).resolve()
    ):
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
    dominance = report.get("candidate_dominance")
    if not isinstance(dominance, dict):
        raise ValueError("Stage211 batch preflight candidate-dominance contract is missing.")
    rejection_ratio = dominance.get("rejection_ratio")
    minimum_points = dominance.get("min_points")
    excluded_profiles = dominance.get("excluded_profiles")
    if (
        dominance.get("enabled_for_capacity_candidates_only") is not True
        or not isinstance(rejection_ratio, (float, int))
        or isinstance(rejection_ratio, bool)
        or float(rejection_ratio) <= 1.0
        or not isinstance(minimum_points, int)
        or isinstance(minimum_points, bool)
        or minimum_points < 3
        or not isinstance(excluded_profiles, list)
        or not all(isinstance(name, str) and name for name in excluded_profiles)
    ):
        raise ValueError("Stage211 batch preflight candidate-dominance contract is invalid.")
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
    supported_baselines = [
        {
            "name": baseline_name,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "num_workers": int(base_config.get("num_workers", 0) or 0),
            "gradient_checkpointing": bool(base_config.get("gradient_checkpointing")),
        }
    ]
    if phase in {"block", "logits"}:
        supported_baselines.append(
            {
                "name": baseline_name,
                "batch_size": STAGE211_STACKED_SAFE_BATCH_SIZE,
                "frame_budget": STAGE211_STACKED_SAFE_FRAME_BUDGET,
                "num_workers": int(base_config.get("num_workers", 0) or 0),
                "gradient_checkpointing": bool(base_config.get("gradient_checkpointing")),
            }
        )
    if baseline_profile not in supported_baselines:
        raise ValueError("Stage211 batch preflight baseline is unsupported for this phase.")
    baseline = _validate_safe_profile(by_name[baseline_name], report=report, phase=phase)
    if baseline.get("early_termination") is not None:
        raise ValueError("Stage211 batch preflight baseline cannot be dominance-terminated.")
    baseline_summary = baseline["summary"]
    baseline_seconds = float(baseline_summary["projected_full_coverage_seconds"])
    baseline_loss = float(baseline_summary["mean_loss"])
    baseline_cosine = float(baseline_summary["mean_cosine"])
    baseline_fixed_eval_provenance = baseline_summary["fixed_eval_provenance"]
    worker_candidate_name = report["loader_worker_search"].get("candidate_profile")
    expected_excluded_profiles = [
        baseline_name,
        *(
            [worker_candidate_name]
            if isinstance(worker_candidate_name, str) and worker_candidate_name
            else []
        ),
    ]
    if excluded_profiles != expected_excluded_profiles:
        raise ValueError("Stage211 batch preflight dominance exclusions changed.")
    comparisons = selection.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("Stage211 batch preflight candidate comparisons are missing.")
    comparison_by_name = {
        str(row.get("profile") or ""): row for row in comparisons if isinstance(row, dict)
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
            raise ValueError(f"Stage211 batch preflight {candidate_name} summary is missing.")
        if summary.get("safety_pass") is True:
            _validate_safe_profile(candidate_row, report=report, phase=phase)
        early_termination = candidate_row.get("early_termination")
        if candidate_name in excluded_profiles:
            if early_termination is not None:
                raise ValueError(
                    f"Stage211 batch preflight {candidate_name} is excluded from dominance termination."
                )
        elif early_termination is not None:
            telemetry = candidate_row.get("telemetry")
            coverage = candidate_row.get("coverage")
            if not isinstance(telemetry, list) or not isinstance(coverage, dict):
                raise ValueError(
                    f"Stage211 batch preflight {candidate_name} dominance evidence is incomplete."
                )
            replayed_termination = candidate_dominance_evidence(
                telemetry,
                CandidateDominancePolicy(
                    full_coverage_steps=int(coverage.get("full_coverage_steps", 0)),
                    baseline_projected_full_coverage_seconds=baseline_seconds,
                    min_improvement_ratio=float(report["min_improvement_ratio"]),
                    rejection_ratio=float(rejection_ratio),
                    min_points=int(minimum_points),
                ),
            )
            if replayed_termination is None or early_termination != replayed_termination:
                raise ValueError(
                    f"Stage211 batch preflight {candidate_name} dominance evidence changed."
                )
            if (
                int(candidate_row.get("return_code", 0)) == 0
                or summary.get("safety_pass") is True
                or candidate_row["fixed_eval_capture"].get("captured") is True
            ):
                raise ValueError(
                    f"Stage211 batch preflight {candidate_name} dominance termination is inconsistent."
                )
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
        fixed_eval_provenance_match = (
            summary.get("fixed_eval_provenance") == baseline_fixed_eval_provenance
        )
        quality_pass = (
            loss_regression is not None
            and loss_regression <= float(report["max_loss_regression_ratio"])
            and cosine_regression is not None
            and cosine_regression <= float(report["max_cosine_regression"])
            and fixed_eval_provenance_match
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
            comparison_row.get("fixed_eval_provenance_match") is not fixed_eval_provenance_match
            or comparison_row.get("quality_pass") is not quality_pass
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
    _validate_loader_worker_search(
        report,
        base_config=base_config,
        by_name=by_name,
        admissible_names={name for _, name, _ in replayed_admissible},
    )
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
    report = validate_stage211_batch_profile_preflight(
        report_path,
        phase=phase,
        require_candidate=False,
    )
    selected = report["selected_profile_row"]
    profile = selected["profile"]
    selected_comparison = report["selected_comparison"]
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
        "selected_comparison": (
            dict(selected_comparison) if isinstance(selected_comparison, dict) else None
        ),
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
    if (
        not str(receipt.get("admitted_by") or "").strip()
        or not str(receipt.get("reason") or "").strip()
    ):
        raise ValueError("Stage211 batch profile admission lacks explicit review metadata.")
    report_path = _validate_bound_file(
        receipt,
        path_key="preflight_report_path",
        sha256_key="preflight_report_sha256",
        label="Stage211 admitted batch preflight report",
    )
    report = validate_stage211_batch_profile_preflight(
        report_path,
        phase=phase,
        require_candidate=False,
    )
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
    if (
        expected_init_checkpoint is not None
        and Path(str(receipt["init_checkpoint_path"])).resolve()
        != Path(expected_init_checkpoint).expanduser().resolve()
    ):
        raise ValueError("Stage211 batch profile admission uses another initial checkpoint.")
    if (
        expected_bucket_manifest is not None
        and Path(str(receipt["bucket_manifest_path"])).resolve()
        != Path(expected_bucket_manifest).expanduser().resolve()
    ):
        raise ValueError("Stage211 batch profile admission uses another bucket manifest.")
    profile = receipt["selected_profile"]
    base_config = load_yaml(Path(str(report["base_config_path"])))
    worker_count_changed = int(profile["num_workers"]) != int(
        base_config.get("num_workers", 0) or 0
    )
    checkpointing_changed = profile["gradient_checkpointing"] is not base_config.get(
        "gradient_checkpointing"
    )
    profile_is_noop = (
        int(profile["batch_size"]) == int(base_config.get("batch_size", 0) or 0)
        and int(profile["frame_budget"])
        == int(base_config.get("length_bucket_frame_budget", 0) or 0)
        and not worker_count_changed
        and not checkpointing_changed
    )
    if profile_is_noop:
        raise ValueError(
            "Stage211 admitted profile changes neither configured capacity nor loader workers."
        )
    return {
        **receipt,
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
    }
