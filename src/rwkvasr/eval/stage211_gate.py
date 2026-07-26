from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


STAGE211_PHASE_GATE_SCHEMA_VERSION = 1
STAGE211_FULL_DATA_EPOCHS = 3
STAGE211_FULL_DATA_BATCH_SIZE = 36
STAGE211_FULL_DATA_WORLD_SIZE = 4
STAGE211_FULL_DATA_FRAME_BUDGET = 24_000
STAGE211_ALLOWED_OPERATOR_KEY_MARKERS = (".time_mixer.", ".input_proj.")
STAGE211_AUDIO_CURRICULUM: dict[str, dict[str, float | int]] = {
    "easy": {
        "rows": 1_174_987,
        "hours": 1_490.119,
        "steps_per_epoch": 9_982,
        "steps": 29_946,
        "tail_padding_samples_per_epoch": 633,
    },
    "medium": {
        "rows": 42_247_508,
        "hours": 47_771.242,
        "steps_per_epoch": 334_571,
        "steps": 1_003_713,
        "tail_padding_samples_per_epoch": 744,
    },
    "hard": {
        "rows": 18_649_326,
        "hours": 69_200.368,
        "steps_per_epoch": 322_105,
        "steps": 966_315,
        "tail_padding_samples_per_epoch": 986,
    },
    "long": {
        "rows": 404,
        "hours": 3.432,
        "steps_per_epoch": 35,
        "steps": 105,
        "tail_padding_samples_per_epoch": 296,
    },
}
STAGE211_PUBLIC_BENCHMARKS: dict[str, dict[str, str | int]] = {
    "aishell1_test": {"language": "zh", "metric": "cer", "samples": 7_176},
    "librispeech_test_clean": {"language": "en", "metric": "wer", "samples": 2_620},
    "librispeech_test_other": {"language": "en", "metric": "wer", "samples": 2_939},
    "commonvoice_en_test": {"language": "en", "metric": "wer", "samples": 16_396},
    "wenetspeech_test_net": {"language": "zh", "metric": "cer", "samples": 24_774},
}
STAGE211_AUDIO_TOTAL_ROWS = sum(int(row["rows"]) for row in STAGE211_AUDIO_CURRICULUM.values())
STAGE211_AUDIO_TOTAL_HOURS = sum(
    float(row["hours"]) for row in STAGE211_AUDIO_CURRICULUM.values()
)
STAGE211_AUDIO_TOTAL_ROW_EXPOSURES = STAGE211_AUDIO_TOTAL_ROWS * STAGE211_FULL_DATA_EPOCHS
STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES = (
    STAGE211_AUDIO_TOTAL_HOURS * STAGE211_FULL_DATA_EPOCHS
)
STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES = sum(
    int(row["tail_padding_samples_per_epoch"]) * STAGE211_FULL_DATA_EPOCHS
    for row in STAGE211_AUDIO_CURRICULUM.values()
)
STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES = (
    STAGE211_AUDIO_TOTAL_ROW_EXPOSURES
    + STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
)


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_bound_file(
    record: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> Path:
    path = Path(str(record.get(path_key) or "")).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    expected_sha256 = str(record.get(sha256_key) or "")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _validate_parameter_delta_audit(
    segment: dict[str, Any],
    *,
    phase: str,
    difficulty: str,
) -> None:
    audit = segment.get("parameter_delta_audit")
    if not isinstance(audit, dict):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} lacks a checkpoint parameter-delta audit."
        )
    expected = {
        "schema_version": 1,
        "policy": "stage211_timemixer_and_input_projection_only",
        "complete": True,
        "allowed_key_markers": list(STAGE211_ALLOWED_OPERATOR_KEY_MARKERS),
        "forbidden_changed_tensors": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} checkpoint parameter-delta audit is invalid."
        )
    initial_tensor_count = int(audit.get("initial_tensor_count", -1))
    completion_tensor_count = int(audit.get("completion_tensor_count", -1))
    allowed_changed_tensors = int(audit.get("allowed_changed_tensors", -1))
    allowed_changed_numel = int(audit.get("allowed_changed_numel", -1))
    allowed_unchanged_tensors = int(audit.get("allowed_unchanged_tensors", -1))
    frozen_unchanged_tensors = int(audit.get("frozen_unchanged_tensors", -1))
    if (
        initial_tensor_count <= 0
        or completion_tensor_count != initial_tensor_count
        or allowed_changed_tensors <= 0
        or allowed_changed_numel <= 0
        or allowed_unchanged_tensors < 0
        or frozen_unchanged_tensors <= 0
        or (
            allowed_changed_tensors
            + allowed_unchanged_tensors
            + frozen_unchanged_tensors
            != initial_tensor_count
        )
    ):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} checkpoint parameter-delta counts are invalid."
        )


def validate_stage211_full_data_coverage(
    coverage: Any,
    *,
    phase: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 phase gate lacks a full_data_coverage object.")
    if coverage.get("complete") is not True:
        raise ValueError("Stage211 full-data coverage is not complete.")
    if coverage.get("phase") != phase:
        raise ValueError(
            "Stage211 full-data coverage phase mismatch: "
            f"expected={phase!r} actual={coverage.get('phase')!r}"
        )
    if int(coverage.get("total_unique_rows", -1)) != STAGE211_AUDIO_TOTAL_ROWS:
        raise ValueError("Stage211 full-data coverage row total mismatch.")
    total_hours = float(coverage.get("total_hours", float("nan")))
    if not math.isfinite(total_hours) or abs(total_hours - STAGE211_AUDIO_TOTAL_HOURS) > 0.002:
        raise ValueError("Stage211 full-data coverage hour total mismatch.")
    if (
        int(coverage.get("total_row_exposures", -1))
        != STAGE211_AUDIO_TOTAL_ROW_EXPOSURES
    ):
        raise ValueError("Stage211 full-data row-exposure total mismatch.")
    if (
        int(coverage.get("total_tail_padding_sample_exposures", -1))
        != STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
    ):
        raise ValueError("Stage211 full-data tail-padding exposure total mismatch.")
    if (
        int(coverage.get("total_executed_sample_exposures", -1))
        != STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
    ):
        raise ValueError("Stage211 full-data executed-sample exposure total mismatch.")
    total_hour_exposures = float(
        coverage.get("total_hour_exposures", float("nan"))
    )
    if (
        not math.isfinite(total_hour_exposures)
        or abs(total_hour_exposures - STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES) > 0.005
    ):
        raise ValueError("Stage211 full-data hour-exposure total mismatch.")

    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 full-data coverage segments must be a list.")
    by_difficulty = {
        str(segment.get("difficulty")): segment
        for segment in segments
        if isinstance(segment, dict)
    }
    if set(by_difficulty) != set(STAGE211_AUDIO_CURRICULUM):
        raise ValueError(
            "Stage211 full-data coverage must contain exactly easy, medium, hard, and long."
        )

    previous_checkpoint_sha256: str | None = None
    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        segment = by_difficulty[difficulty]
        expected_fields = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "curriculum_coverage",
            "phase": phase,
            "difficulty": difficulty,
            "complete": True,
            "full_data_profile": True,
            "epochs": STAGE211_FULL_DATA_EPOCHS,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
            "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
        }
        if any(segment.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"Stage211 {phase}/{difficulty} coverage is incomplete.")
        _validate_parameter_delta_audit(
            segment,
            phase=phase,
            difficulty=difficulty,
        )
        if int(segment.get("rows", -1)) != int(expected["rows"]):
            raise ValueError(f"Stage211 {phase}/{difficulty} row count mismatch.")
        if int(segment.get("row_exposures", -1)) != (
            int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} row exposures mismatch.")
        if int(segment.get("steps_per_epoch", -1)) != int(
            expected["steps_per_epoch"]
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} per-epoch steps mismatch.")
        if int(segment.get("steps", -1)) != int(expected["steps"]):
            raise ValueError(f"Stage211 {phase}/{difficulty} step count mismatch.")
        tail_padding_samples_per_epoch = int(
            expected["tail_padding_samples_per_epoch"]
        )
        if (
            int(segment.get("tail_padding_samples_per_epoch", -1))
            != tail_padding_samples_per_epoch
        ):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} tail-padding count mismatch."
            )
        if int(segment.get("tail_padding_sample_exposures", -1)) != (
            tail_padding_samples_per_epoch * STAGE211_FULL_DATA_EPOCHS
        ):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} tail-padding exposure mismatch."
            )
        if int(segment.get("executed_sample_exposures", -1)) != (
            int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS
            + tail_padding_samples_per_epoch * STAGE211_FULL_DATA_EPOCHS
        ):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} executed-sample exposure mismatch."
            )
        hours = float(segment.get("hours", float("nan")))
        if not math.isfinite(hours) or abs(hours - float(expected["hours"])) > 0.002:
            raise ValueError(f"Stage211 {phase}/{difficulty} hour count mismatch.")
        hour_exposures = float(segment.get("hour_exposures", float("nan")))
        expected_hour_exposures = float(expected["hours"]) * STAGE211_FULL_DATA_EPOCHS
        if (
            not math.isfinite(hour_exposures)
            or abs(hour_exposures - expected_hour_exposures) > 0.005
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} hour exposures mismatch.")
        _validate_bound_file(
            segment,
            path_key="receipt_path",
            sha256_key="receipt_sha256",
            label=f"Stage211 {phase}/{difficulty} coverage receipt",
        )
        _validate_bound_file(
            segment,
            path_key="provenance_path",
            sha256_key="provenance_sha256",
            label=f"Stage211 {phase}/{difficulty} provenance",
        )
        _validate_bound_file(
            segment,
            path_key="train_config_path",
            sha256_key="train_config_sha256",
            label=f"Stage211 {phase}/{difficulty} train config",
        )
        _validate_bound_file(
            segment,
            path_key="bucket_manifest_path",
            sha256_key="bucket_manifest_sha256",
            label=f"Stage211 {phase}/{difficulty} bucket manifest",
        )
        _validate_bound_file(
            segment,
            path_key="init_checkpoint_path",
            sha256_key="init_checkpoint_sha256",
            label=f"Stage211 {phase}/{difficulty} initial checkpoint",
        )
        _validate_bound_file(
            segment,
            path_key="completion_checkpoint_path",
            sha256_key="completion_checkpoint_sha256",
            label=f"Stage211 {phase}/{difficulty} completion checkpoint",
        )
        init_sha256 = str(segment.get("init_checkpoint_sha256") or "")
        if previous_checkpoint_sha256 is not None and init_sha256 != previous_checkpoint_sha256:
            raise ValueError(
                f"Stage211 {phase}/{difficulty} does not initialize from the preceding "
                "curriculum checkpoint."
            )
        previous_checkpoint_sha256 = str(segment["completion_checkpoint_sha256"])

    final_checkpoint = Path(str(coverage.get("final_checkpoint_path") or "")).resolve()
    if final_checkpoint != checkpoint_path.resolve():
        raise ValueError(
            "Stage211 full-data final checkpoint path does not match the selected checkpoint."
        )
    if str(coverage.get("final_checkpoint_sha256") or "") != sha256_file(checkpoint_path):
        raise ValueError(
            "Stage211 full-data final checkpoint SHA-256 does not match the selected checkpoint."
        )
    if previous_checkpoint_sha256 != str(coverage["final_checkpoint_sha256"]):
        raise ValueError("Stage211 long-segment checkpoint is not the full-data final checkpoint.")
    return dict(coverage)


def validate_stage211_public_benchmark(public_benchmark: Any) -> dict[str, Any]:
    if not isinstance(public_benchmark, dict):
        raise ValueError("Stage211 phase gate lacks a public_benchmark object.")
    if public_benchmark.get("decode") != "greedy_ctc":
        raise ValueError("Stage211 public benchmark must use direct greedy CTC.")
    if public_benchmark.get("normalization") != "ctc":
        raise ValueError("Stage211 public benchmark must use ctc normalization.")
    if public_benchmark.get("all_datasets_complete") is not True:
        raise ValueError("Stage211 public benchmark is not complete.")

    results = public_benchmark.get("results")
    if not isinstance(results, list):
        raise ValueError("Stage211 public benchmark results must be a list.")
    by_dataset = {
        str(result.get("dataset")): result
        for result in results
        if isinstance(result, dict)
    }
    if set(by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 public benchmark dataset set is incomplete or unexpected.")

    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        result = by_dataset[dataset]
        if result.get("language") != expected["language"]:
            raise ValueError(f"Stage211 public benchmark language mismatch for {dataset}.")
        if result.get("metric") != expected["metric"]:
            raise ValueError(f"Stage211 public benchmark metric mismatch for {dataset}.")
        if int(result.get("sample_count", -1)) != int(expected["samples"]):
            raise ValueError(f"Stage211 public benchmark sample count mismatch for {dataset}.")
        if (
            result.get("identical_utt_coverage") is not True
            or int(result.get("normalized_reference_mismatch_count", -1)) != 0
        ):
            raise ValueError(f"Stage211 public benchmark coverage mismatch for {dataset}.")
        for prefix in ("manifest", "nano_prediction", "student_prediction"):
            _validate_bound_file(
                result,
                path_key=f"{prefix}_path",
                sha256_key=f"{prefix}_sha256",
                label=f"Stage211 {dataset} {prefix.replace('_', ' ')}",
            )
        for metric_name in (
            "nano_error_rate",
            "student_error_rate",
            "absolute_gap_points",
            "relative_ratio",
            "nano_prediction_reference_unit_ratio",
            "student_prediction_reference_unit_ratio",
            "nano_deletion_rate",
            "student_deletion_rate",
        ):
            value = float(result.get(metric_name, float("nan")))
            if not math.isfinite(value):
                raise ValueError(
                    f"Stage211 public benchmark {dataset}/{metric_name} must be finite."
                )
    return dict(public_benchmark)


def validate_stage211_phase_gate_report(
    gate_report_path: str | Path,
    *,
    expected_phase: str,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    gate_report_path = Path(gate_report_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    report = _load_json_object(gate_report_path, label="Stage211 phase gate report")
    expected_fields = {
        "schema_version": STAGE211_PHASE_GATE_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "phase_gate",
        "phase": expected_phase,
    }
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(
                f"Stage211 phase gate {key} mismatch: "
                f"expected={expected!r} actual={report.get(key)!r}"
            )
    if report.get("gate_passed") is not True:
        raise ValueError("Stage211 phase gate report does not record a passing decision.")
    recorded_checkpoint = Path(str(report.get("checkpoint_path") or "")).resolve()
    if recorded_checkpoint != checkpoint_path:
        raise ValueError("Stage211 phase gate checkpoint path mismatch.")
    if str(report.get("checkpoint_sha256") or "") != sha256_file(checkpoint_path):
        raise ValueError("Stage211 phase gate checkpoint SHA-256 mismatch.")

    validate_stage211_full_data_coverage(
        report.get("full_data_coverage"),
        phase=expected_phase,
        checkpoint_path=checkpoint_path,
    )
    benchmark = validate_stage211_public_benchmark(report.get("public_benchmark"))
    if expected_phase in {"mixer", "block"}:
        if report.get("alignment_gate_passed") is not True:
            raise ValueError(f"Stage211 {expected_phase} alignment gate did not pass.")
        if report.get("public_progress_gate_passed") is not True:
            raise ValueError(
                f"Stage211 {expected_phase} public progress gate did not pass."
            )
    elif expected_phase == "logits":
        if benchmark.get("all_datasets_pass") is not True:
            raise ValueError(
                "Stage211 logits phase must pass the every-dataset Nano WER/CER gate."
            )
    else:
        raise ValueError(f"Stage211 phase {expected_phase!r} cannot promote.")
    return report
