from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.eval.text_metrics import (
    _normalize_text_for_error_tokens,
    edit_counts,
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)


STAGE211_PUBLIC_MAX_RELATIVE_RATIO = 1.20
STAGE211_PUBLIC_MAX_ABSOLUTE_GAP_POINTS = 3.0
STAGE211_PUBLIC_MAX_PROGRESS_REGRESSION = 0.03

_PUBLIC_LABELS = {
    "aishell1_test": "AISHELL-1 test",
    "librispeech_test_clean": "LibriSpeech test-clean",
    "librispeech_test_other": "LibriSpeech test-other",
    "commonvoice_en_test": "Common Voice 22 en test",
    "wenetspeech_test_net": "WenetSpeech TEST_NET",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 {label} is missing or empty: {path}")
    return path


def _jsonl_references(
    path: Path,
    *,
    language: str,
    reference_keys: tuple[str, ...],
) -> dict[str, str]:
    references: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Stage211 public row is invalid at {path}:{line_number}")
            utt_id = str(row.get("utt_id") or row.get("id") or row.get("key") or "")
            if not utt_id:
                raise ValueError(f"Stage211 public row lacks an ID at {path}:{line_number}")
            if utt_id in references:
                raise ValueError(
                    f"Stage211 public row duplicates {utt_id!r} at {path}:{line_number}"
                )
            reference = next(
                (str(row[key]) for key in reference_keys if row.get(key) is not None),
                None,
            )
            if reference is None:
                raise ValueError(
                    f"Stage211 public row lacks a reference for {utt_id!r} at {path}:{line_number}"
                )
            references[utt_id] = normalize_asr_text_for_metrics(
                reference,
                language=language,
                normalization="ctc",
            )
    if not references:
        raise ValueError(f"Stage211 public JSONL is empty: {path}")
    return references


def _relative_ratio(*, nano_rate: float, student_rate: float) -> float:
    if nano_rate > 0.0:
        return student_rate / nano_rate
    return 1.0 if student_rate <= 0.0 else math.inf


def _jsonl_predictions(
    path: Path,
    *,
    language: str,
) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Stage211 public row is invalid at {path}:{line_number}")
            utt_id = str(row.get("utt_id") or "")
            if not utt_id:
                raise ValueError(f"Stage211 public row lacks an ID at {path}:{line_number}")
            if utt_id in records:
                raise ValueError(
                    f"Stage211 public row duplicates {utt_id!r} at {path}:{line_number}"
                )
            if row.get("ref_text") is None or row.get("pred_text") is None:
                raise ValueError(
                    f"Stage211 public prediction row is incomplete at {path}:{line_number}"
                )
            reference = normalize_asr_text_for_metrics(
                str(row["ref_text"]),
                language=language,
                normalization="ctc",
            )
            prediction = normalize_asr_text_for_metrics(
                str(row["pred_text"]),
                language=language,
                normalization="ctc",
            )
            records[utt_id] = (reference, prediction)
    if not records:
        raise ValueError(f"Stage211 public prediction JSONL is empty: {path}")
    return records


def _score_prediction_records(
    records: dict[str, tuple[str, str]],
    *,
    metric: str,
) -> dict[str, Any]:
    if metric not in {"wer", "cer"}:
        raise ValueError(f"Unsupported Stage211 public metric: {metric!r}")
    total_wer_errors = 0
    total_cer_errors = 0
    total_ref_words = 0
    total_ref_chars = 0
    primary_insertions = 0
    primary_deletions = 0
    primary_substitutions = 0
    primary_reference_units = 0
    primary_prediction_units = 0
    per_sample_primary: dict[str, float] = {}
    normalized_predictions: dict[str, str] = {}
    for utt_id, (reference, prediction) in records.items():
        ref_words = tokenize_for_wer(reference)
        pred_words = tokenize_for_wer(prediction)
        ref_chars = tokenize_for_cer(reference)
        pred_chars = tokenize_for_cer(prediction)
        if metric == "wer":
            insertions, deletions, substitutions = edit_counts(ref_words, pred_words)
            wer_errors = insertions + deletions + substitutions
            cer_errors = edit_distance(pred_chars, ref_chars)
            ref_primary = ref_words
            pred_primary = pred_words
        else:
            insertions, deletions, substitutions = edit_counts(ref_chars, pred_chars)
            cer_errors = insertions + deletions + substitutions
            wer_errors = edit_distance(pred_words, ref_words)
            ref_primary = ref_chars
            pred_primary = pred_chars
        total_wer_errors += wer_errors
        total_cer_errors += cer_errors
        total_ref_words += len(ref_words)
        total_ref_chars += len(ref_chars)
        primary_insertions += insertions
        primary_deletions += deletions
        primary_substitutions += substitutions
        primary_reference_units += len(ref_primary)
        primary_prediction_units += len(pred_primary)
        per_sample_primary[utt_id] = float(insertions + deletions + substitutions) / float(
            max(1, len(ref_primary))
        )
        normalized_predictions[utt_id] = _normalize_text_for_error_tokens(prediction)
    primary_denominator = max(1, primary_reference_units)
    return {
        "sample_count": len(records),
        "avg_wer": float(total_wer_errors) / float(max(1, total_ref_words)),
        "avg_cer": float(total_cer_errors) / float(max(1, total_ref_chars)),
        "prediction_reference_unit_ratio": primary_prediction_units / primary_denominator,
        "insertion_rate": primary_insertions / primary_denominator,
        "deletion_rate": primary_deletions / primary_denominator,
        "substitution_rate": primary_substitutions / primary_denominator,
        "per_sample_primary": per_sample_primary,
        "normalized_predictions": normalized_predictions,
    }


def _recompute_dataset(
    *,
    dataset: str,
    language: str,
    metric: str,
    nano_path: Path,
    student_path: Path,
    nano_records: dict[str, tuple[str, str]],
    student_records: dict[str, tuple[str, str]],
    max_relative_ratio: float,
    max_absolute_gap_points: float,
) -> dict[str, Any]:
    nano_stats = _score_prediction_records(nano_records, metric=metric)
    student_stats = _score_prediction_records(student_records, metric=metric)
    metric_key = f"avg_{metric}"
    nano_rate = float(nano_stats[metric_key])
    student_rate = float(student_stats[metric_key])
    absolute_gap_points = (student_rate - nano_rate) * 100.0
    relative_ratio = _relative_ratio(
        nano_rate=nano_rate,
        student_rate=student_rate,
    )
    improved = 0
    worsened = 0
    unchanged = 0
    changed_predictions = 0
    for utt_id in nano_records:
        nano_error = float(nano_stats["per_sample_primary"][utt_id])
        student_error = float(student_stats["per_sample_primary"][utt_id])
        if student_error < nano_error - 1.0e-9:
            improved += 1
        elif student_error > nano_error + 1.0e-9:
            worsened += 1
        else:
            unchanged += 1
        if (
            nano_stats["normalized_predictions"][utt_id]
            != student_stats["normalized_predictions"][utt_id]
        ):
            changed_predictions += 1
    absolute_gate_pass = absolute_gap_points <= max_absolute_gap_points
    relative_gate_pass = relative_ratio <= max_relative_ratio
    return {
        "dataset": dataset,
        "label": _PUBLIC_LABELS[dataset],
        "language": language,
        "metric": metric,
        "sample_count": int(nano_stats["sample_count"]),
        "identical_utt_coverage": True,
        "normalized_reference_mismatch_count": 0,
        "nano_wer": float(nano_stats["avg_wer"]),
        "student_wer": float(student_stats["avg_wer"]),
        "nano_cer": float(nano_stats["avg_cer"]),
        "student_cer": float(student_stats["avg_cer"]),
        "nano_error_rate": nano_rate,
        "student_error_rate": student_rate,
        "absolute_gap_points": absolute_gap_points,
        "relative_ratio": relative_ratio,
        "absolute_gate_pass": absolute_gate_pass,
        "relative_gate_pass": relative_gate_pass,
        "gate_pass": absolute_gate_pass and relative_gate_pass,
        "nano_prediction_reference_unit_ratio": float(
            nano_stats["prediction_reference_unit_ratio"]
        ),
        "student_prediction_reference_unit_ratio": float(
            student_stats["prediction_reference_unit_ratio"]
        ),
        "nano_insertion_rate": float(nano_stats["insertion_rate"]),
        "student_insertion_rate": float(student_stats["insertion_rate"]),
        "nano_deletion_rate": float(nano_stats["deletion_rate"]),
        "student_deletion_rate": float(student_stats["deletion_rate"]),
        "nano_substitution_rate": float(nano_stats["substitution_rate"]),
        "student_substitution_rate": float(student_stats["substitution_rate"]),
        "changed_prediction_count": changed_predictions,
        "student_improved_count": improved,
        "student_worsened_count": worsened,
        "unchanged_count": unchanged,
        "nano_prediction_path": str(nano_path),
        "student_prediction_path": str(student_path),
    }


def _validate_recomputed_result(
    reported: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    dataset: str,
) -> None:
    for key, expected in recomputed.items():
        actual = reported.get(key)
        if key in {"nano_prediction_path", "student_prediction_path"}:
            if Path(str(actual or "")).expanduser().resolve() != Path(str(expected)).resolve():
                raise ValueError(f"Stage211 {dataset} replayed {key} path mismatch.")
        elif isinstance(expected, bool):
            if actual is not expected:
                raise ValueError(f"Stage211 {dataset} replayed {key} mismatch.")
        elif isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(actual),
                    expected,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"Stage211 {dataset} replayed {key} mismatch.")
        elif actual != expected:
            raise ValueError(f"Stage211 {dataset} replayed {key} mismatch.")


def replay_stage211_public_comparison(
    report: dict[str, Any],
    *,
    manifest_paths: dict[str, Path],
    benchmarks: dict[str, dict[str, str | int]],
    expected_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if report.get("decode") != "greedy_ctc" or report.get("normalization") != "ctc":
        raise ValueError("Stage211 public comparison must use greedy CTC and CTC normalization.")
    gate = report.get("gate")
    if (
        not isinstance(gate, dict)
        or gate.get("requires_every_dataset") is not True
        or not math.isclose(
            float(gate.get("max_relative_ratio", float("nan"))),
            STAGE211_PUBLIC_MAX_RELATIVE_RATIO,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(gate.get("max_absolute_gap_points", float("nan"))),
            STAGE211_PUBLIC_MAX_ABSOLUTE_GAP_POINTS,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("Stage211 public comparison metric-gate contract mismatch.")
    if set(manifest_paths) != set(benchmarks):
        raise ValueError("Stage211 public manifest map is incomplete or unexpected.")
    if expected_checkpoint is not None:
        expected_checkpoint = expected_checkpoint.expanduser().resolve()
        if Path(
            str(report.get("student_checkpoint_path") or "")
        ).expanduser().resolve() != expected_checkpoint or report.get(
            "student_checkpoint_sha256"
        ) != _sha256_file(expected_checkpoint):
            raise ValueError("Stage211 public comparison checkpoint binding mismatch.")
    raw_results = report.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(benchmarks):
        raise ValueError("Stage211 public comparison result coverage mismatch.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(benchmarks):
        raise ValueError("Stage211 public comparison dataset set is incomplete or unexpected.")

    replayed_results: list[dict[str, Any]] = []
    for dataset, expected in benchmarks.items():
        reported = by_dataset[dataset]
        language = str(expected["language"])
        metric = str(expected["metric"])
        manifest_path = _bound_file(manifest_paths[dataset], label=f"{dataset} manifest")
        nano_path = _bound_file(
            reported.get("nano_prediction_path"),
            label=f"{dataset} Nano predictions",
        )
        student_path = _bound_file(
            reported.get("student_prediction_path"),
            label=f"{dataset} student predictions",
        )
        manifest_references = _jsonl_references(
            manifest_path,
            language=language,
            reference_keys=("text", "transcript", "ref_text", "reference"),
        )
        nano_records = _jsonl_predictions(nano_path, language=language)
        student_records = _jsonl_predictions(student_path, language=language)
        nano_references = {utt_id: reference for utt_id, (reference, _) in nano_records.items()}
        student_references = {
            utt_id: reference for utt_id, (reference, _) in student_records.items()
        }
        expected_samples = int(expected["samples"])
        if (
            len(manifest_references) != expected_samples
            or set(nano_references) != set(manifest_references)
            or set(student_references) != set(manifest_references)
        ):
            raise ValueError(f"Stage211 {dataset} public utterance coverage mismatch.")
        if any(
            nano_references[utt_id] != reference or student_references[utt_id] != reference
            for utt_id, reference in manifest_references.items()
        ):
            raise ValueError(f"Stage211 {dataset} normalized public references mismatch.")
        recomputed = _recompute_dataset(
            dataset=dataset,
            language=language,
            metric=metric,
            nano_path=nano_path,
            student_path=student_path,
            nano_records=nano_records,
            student_records=student_records,
            max_relative_ratio=STAGE211_PUBLIC_MAX_RELATIVE_RATIO,
            max_absolute_gap_points=STAGE211_PUBLIC_MAX_ABSOLUTE_GAP_POINTS,
        )
        _validate_recomputed_result(reported, recomputed, dataset=dataset)
        replayed_results.append(
            {
                **reported,
                **recomputed,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "nano_prediction_path": str(nano_path),
                "nano_prediction_sha256": _sha256_file(nano_path),
                "student_prediction_path": str(student_path),
                "student_prediction_sha256": _sha256_file(student_path),
                "metric_source_recomputed": True,
            }
        )
    all_datasets_pass = all(bool(row["gate_pass"]) for row in replayed_results)
    if report.get("all_datasets_pass") is not all_datasets_pass:
        raise ValueError("Stage211 public comparison every-dataset decision mismatch.")
    return {
        **report,
        "all_datasets_complete": True,
        "all_datasets_pass": all_datasets_pass,
        "results": replayed_results,
    }


def build_stage211_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    benchmarks: dict[str, dict[str, str | int]],
    max_dataset_regression: float = STAGE211_PUBLIC_MAX_PROGRESS_REGRESSION,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(benchmarks) or set(candidate_results) != set(benchmarks):
        raise ValueError("Stage211 baseline/candidate public datasets differ.")
    rows: list[dict[str, Any]] = []
    for dataset in benchmarks:
        baseline_result = baseline_results[dataset]
        candidate_result = candidate_results[dataset]
        for hash_key in ("manifest_sha256", "nano_prediction_sha256"):
            if baseline_result.get(hash_key) != candidate_result.get(hash_key):
                raise ValueError(f"Stage211 {dataset} baseline/candidate {hash_key} differs.")
        baseline_error = float(baseline_result["student_error_rate"])
        candidate_error = float(candidate_result["student_error_rate"])
        baseline_deletion = float(baseline_result["student_deletion_rate"])
        candidate_deletion = float(candidate_result["student_deletion_rate"])
        rows.append(
            {
                "dataset": dataset,
                "baseline_error_rate": baseline_error,
                "candidate_error_rate": candidate_error,
                "absolute_change": candidate_error - baseline_error,
                "baseline_deletion_rate": baseline_deletion,
                "candidate_deletion_rate": candidate_deletion,
                "within_regression_limit": candidate_error
                <= baseline_error + max_dataset_regression,
                "improved": candidate_error < baseline_error,
            }
        )
    macro_baseline_error = sum(float(row["baseline_error_rate"]) for row in rows) / len(rows)
    macro_candidate_error = sum(float(row["candidate_error_rate"]) for row in rows) / len(rows)
    macro_baseline_deletion = sum(float(row["baseline_deletion_rate"]) for row in rows) / len(rows)
    macro_candidate_deletion = sum(float(row["candidate_deletion_rate"]) for row in rows) / len(
        rows
    )
    gate_passed = (
        all(bool(row["within_regression_limit"]) for row in rows)
        and macro_candidate_error < macro_baseline_error
        and macro_candidate_deletion < macro_baseline_deletion
        and any(bool(row["improved"]) for row in rows)
    )
    return {
        "gate_passed": gate_passed,
        "max_dataset_regression": max_dataset_regression,
        "macro_baseline_error_rate": macro_baseline_error,
        "macro_candidate_error_rate": macro_candidate_error,
        "macro_baseline_deletion_rate": macro_baseline_deletion,
        "macro_candidate_deletion_rate": macro_candidate_deletion,
        "improved_datasets": sum(bool(row["improved"]) for row in rows),
        "results": rows,
        "baseline_public_benchmark": baseline,
    }


def build_stage211_sft_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    benchmarks: dict[str, dict[str, str | int]],
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(benchmarks) or set(candidate_results) != set(benchmarks):
        raise ValueError("Stage211D baseline/candidate public datasets differ.")
    rows: list[dict[str, Any]] = []
    for dataset, expected in benchmarks.items():
        baseline_result = baseline_results[dataset]
        candidate_result = candidate_results[dataset]
        for hash_key in ("manifest_sha256", "nano_prediction_sha256"):
            if baseline_result.get(hash_key) != candidate_result.get(hash_key):
                raise ValueError(f"Stage211D {dataset} baseline/candidate {hash_key} differs.")
        baseline_error = float(baseline_result["student_error_rate"])
        candidate_error = float(candidate_result["student_error_rate"])
        if not math.isfinite(baseline_error) or not math.isfinite(candidate_error):
            raise ValueError(f"Stage211D {dataset} error rate must be finite.")
        rows.append(
            {
                "dataset": dataset,
                "metric": expected["metric"],
                "baseline_error_rate": baseline_error,
                "candidate_error_rate": candidate_error,
                "absolute_change": candidate_error - baseline_error,
                "no_regression": candidate_error <= baseline_error + tolerance,
                "improved": candidate_error < baseline_error - tolerance,
            }
        )
    macro_baseline = sum(float(row["baseline_error_rate"]) for row in rows) / len(rows)
    macro_candidate = sum(float(row["candidate_error_rate"]) for row in rows) / len(rows)
    no_regressions = all(bool(row["no_regression"]) for row in rows)
    improved_datasets = sum(bool(row["improved"]) for row in rows)
    macro_improved = macro_candidate < macro_baseline - tolerance
    gate_passed = no_regressions and macro_improved and improved_datasets > 0
    return {
        "gate_passed": gate_passed,
        "tolerance": tolerance,
        "no_dataset_regression": no_regressions,
        "macro_improved": macro_improved,
        "improved_datasets": improved_datasets,
        "macro_baseline_error_rate": macro_baseline,
        "macro_candidate_error_rate": macro_candidate,
        "results": rows,
    }


def _load_bound_comparison(
    report: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> dict[str, Any]:
    path = _bound_file(report.get(path_key), label=label)
    if report.get(sha256_key) != _sha256_file(path):
        raise ValueError(f"Stage211 {label} SHA-256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage211 {label} must be a JSON object.")
    return payload


def _validate_replayed_value(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"Stage211 {label} does not match replayed public evidence.")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Stage211 {label} field coverage mismatch.")
        for key, value in expected.items():
            _validate_replayed_value(actual[key], value, label=f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Stage211 {label} sequence mismatch.")
        for index, value in enumerate(expected):
            _validate_replayed_value(actual[index], value, label=f"{label}[{index}]")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise ValueError(f"Stage211 {label} does not match replayed public evidence.")
        return
    if isinstance(expected, float):
        try:
            matches = math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"Stage211 {label} does not match replayed public evidence.")
        return
    if actual != expected:
        raise ValueError(f"Stage211 {label} does not match replayed public evidence.")


def replay_stage211_sft_public_evidence(
    report: dict[str, Any],
    *,
    benchmarks: dict[str, dict[str, str | int]],
    expected_baseline_checkpoint: Path,
    expected_candidate_checkpoint: Path,
) -> dict[str, Any]:
    embedded_candidate = report.get("public_benchmark")
    if not isinstance(embedded_candidate, dict):
        raise ValueError("Stage211D report lacks its public benchmark.")
    raw_results = embedded_candidate.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Stage211D public benchmark results are invalid.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(benchmarks):
        raise ValueError("Stage211D public benchmark dataset coverage mismatch.")
    manifest_paths = {
        dataset: Path(str(result.get("manifest_path") or "")).expanduser().resolve()
        for dataset, result in by_dataset.items()
    }
    baseline_source = _load_bound_comparison(
        report,
        path_key="baseline_public_comparison_report_path",
        sha256_key="baseline_public_comparison_report_sha256",
        label="SFT baseline public comparison",
    )
    candidate_source = _load_bound_comparison(
        report,
        path_key="public_comparison_report_path",
        sha256_key="public_comparison_report_sha256",
        label="SFT candidate public comparison",
    )
    replayed_baseline = replay_stage211_public_comparison(
        baseline_source,
        manifest_paths=manifest_paths,
        benchmarks=benchmarks,
        expected_checkpoint=expected_baseline_checkpoint,
    )
    replayed_candidate = replay_stage211_public_comparison(
        candidate_source,
        manifest_paths=manifest_paths,
        benchmarks=benchmarks,
        expected_checkpoint=expected_candidate_checkpoint,
    )
    replayed_progress = build_stage211_sft_public_progress(
        baseline=replayed_baseline,
        candidate=replayed_candidate,
        benchmarks=benchmarks,
    )
    _validate_replayed_value(
        report.get("baseline_public_benchmark"),
        replayed_baseline,
        label="SFT baseline public benchmark",
    )
    _validate_replayed_value(
        embedded_candidate,
        replayed_candidate,
        label="SFT candidate public benchmark",
    )
    _validate_replayed_value(
        report.get("public_progress"),
        replayed_progress,
        label="SFT public progress",
    )
    return {
        "baseline_public_benchmark": replayed_baseline,
        "public_benchmark": replayed_candidate,
        "public_progress": replayed_progress,
    }
