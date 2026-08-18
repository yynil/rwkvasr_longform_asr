from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_artifact_io import write_immutable_text
from rwkvasr.eval.stage211_gate import (
    STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_PUBLIC_BENCHMARKS,
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT,
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256,
    STAGE211_SFT_STEP_EVAL_INTERVAL,
    STAGE211_STEP_EVAL_INTERVAL,
    build_stage211_correction_round_promotion_gate,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_phase_gate_report,
    validate_stage211_public_benchmark,
    validate_stage211_public_overlap_binding,
)
from rwkvasr.eval.stage211_initialization import (
    DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
    validate_stage211_initialization_receipt,
)
from rwkvasr.eval.stage211_public_metrics import (
    STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
    replay_stage211_sft_public_evidence,
)
from rwkvasr.eval.stage211_supplemental import STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER
from rwkvasr.eval.stage211_sft_public_clean import (
    validate_stage211_sft_public_clean_rebuild_receipt,
)
from rwkvasr.eval.stage211_sft_public_overlap import (
    validate_stage211_sft_public_overlap_receipt,
)

try:
    from scripts.install_stage211_unicode_metric_correction import (
        DEFAULT_CORRECTION_RECEIPT,
        DEFAULT_TOKENIZER_SOURCE,
        validate_completed_correction,
    )
    from scripts.create_stage211_labeled_profile_receipt import (
        validate_receipt as validate_labeled_profile_receipt,
    )
    from scripts.run_stage211_labeled_sft import (
        LABELED_EXPECTED,
        _validate_completion as _validate_sft_completion,
    )
    from scripts.run_stage211_strict_chained_alignment import (
        STAGE211_LABELED_LANGUAGE_COUNTS,
        STAGE211_LABELED_SOURCE_COUNTS,
        _validate_promotion_receipt as _validate_sft_promotion_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from install_stage211_unicode_metric_correction import (
        DEFAULT_CORRECTION_RECEIPT,
        DEFAULT_TOKENIZER_SOURCE,
        validate_completed_correction,
    )
    from create_stage211_labeled_profile_receipt import (  # type: ignore[no-redef]
        validate_receipt as validate_labeled_profile_receipt,
    )
    from run_stage211_labeled_sft import (
        LABELED_EXPECTED,
        _validate_completion as _validate_sft_completion,
    )
    from run_stage211_strict_chained_alignment import (
        STAGE211_LABELED_LANGUAGE_COUNTS,
        STAGE211_LABELED_SOURCE_COUNTS,
        _validate_promotion_receipt as _validate_sft_promotion_receipt,
    )


STAGE_ORDER = ("calibration", "mixer", "block", "logits", "sft")
REQUESTED_ALIGNMENT_STAGE_ORDER = ("rwkv_layer", "block", "logits", "sft")
REQUESTED_TO_INTERNAL_STAGE = {
    "rwkv_layer": "mixer",
    "block": "block",
    "logits": "logits",
    "sft": "sft",
}
REQUESTED_STAGE_OBJECTIVES = {
    "rwkv_layer": "hidden_states",
    "block": "hidden_states",
    "logits": "ctc_logits",
    "sft": "labeled_ctc_sft",
}
STAGE_LABELS = {
    "calibration": "Calibration",
    "mixer": "Layer / Mixer (A)",
    "block": "Block (B)",
    "logits": "Logits (C)",
    "sft": "Labeled CTC SFT (D)",
}
ALIGNMENT_CELLS = (
    "easy_en",
    "easy_zh",
    "medium_en",
    "medium_zh",
    "hard_en",
    "hard_zh",
    "long_zh",
    "supplemental_en",
    "supplemental_zh",
)
LOGITS_DISCLOSURE_METRICS = (
    "full_kl",
    "conditional_nonblank_kl",
    "ctc_token_error_rate",
    "selected_top1_agreement",
    "all_top1_agreement",
    "active_top1_agreement",
    "nonblank_rate_ratio",
    "collapsed_length_ratio",
)
DEFAULT_CALIBRATION_RECEIPT = (
    Path.home()
    / "rwkvasr_eval"
    / "stage211_calibration_selected_full"
    / "public"
    / "reuse_receipt.json"
)
DEFAULT_PHASE_GATE_ROOT = Path.home() / "rwkvasr_eval" / "stage211_phase_gates"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_cli_phase_gate(
    *,
    phase: str,
    requested_gate: Path | None,
    sft_final_report_path: Path,
) -> Path:
    if requested_gate is not None:
        return requested_gate.expanduser().resolve()
    sft_report = _load_json(
        sft_final_report_path,
        label="Stage211 SFT final report",
    )
    bound_gate = sft_report.get(f"{phase}_phase_gate_path")
    if bound_gate:
        return Path(str(bound_gate)).expanduser().resolve()
    return (DEFAULT_PHASE_GATE_ROOT / phase / "phase_gate.json").resolve()


def _resolve_cli_mixer_gate(
    *,
    requested_gate: Path | None,
    sft_final_report_path: Path,
) -> Path:
    return _resolve_cli_phase_gate(
        phase="mixer",
        requested_gate=requested_gate,
        sft_final_report_path=sft_final_report_path,
    )


def _validate_bound_file(
    record: dict[str, Any],
    *,
    path_key: str,
    sha_key: str,
    label: str,
) -> Path:
    path = Path(str(record.get(path_key) or "")).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    if record.get(sha_key) != sha256_file(path):
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _validate_calibration_receipt(path: Path) -> tuple[dict[str, Any], Path]:
    receipt = _load_json(path, label="Stage211 calibration reuse receipt")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "calibration_public_eval_reuse",
        "complete": True,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Stage211 calibration receipt {key} mismatch.")
    checkpoint = _validate_bound_file(
        receipt,
        path_key="checkpoint_path",
        sha_key="checkpoint_sha256",
        label="Stage211 calibration checkpoint",
    )
    for path_key, sha_key, label in (
        ("selection_report_path", "selection_report_sha256", "selection report"),
        ("comparison_report_path", "comparison_report_sha256", "comparison report"),
        ("metrics_path", "metrics_sha256", "metrics report"),
    ):
        _validate_bound_file(
            receipt,
            path_key=path_key,
            sha_key=sha_key,
            label=f"Stage211 calibration {label}",
        )
    benchmark = validate_stage211_public_benchmark(
        receipt.get("public_benchmark"),
        require_metric_source_recomputed=True,
    )
    validate_stage211_public_overlap_binding(
        receipt.get("public_overlap"),
        public_benchmark=benchmark,
    )
    return receipt, checkpoint


def _empty_sft_correction_coverage() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "applied": False,
        "rounds": 0,
        "unique_rows_per_round": 0,
        "row_exposures": 0,
        "hour_exposures": 0.0,
        "steps": 0,
        "tail_padding_sample_exposures": 0,
        "executed_sample_exposures": 0,
        "language_row_exposures": {},
        "source_row_exposures": {},
        "round_receipts": [],
    }


def _validate_sft_correction_coverage(
    report: dict[str, Any],
    *,
    full_completion_path: Path,
    full_checkpoint: Path,
    final_checkpoint: Path,
) -> dict[str, Any]:
    correction_keys = (
        "sft_correction_evaluation_path",
        "sft_correction_evaluation_sha256",
        "sft_correction_profile_path",
        "sft_correction_profile_sha256",
        "sft_correction_completion_receipts",
        "sft_correction_coverage",
        "sft_correction_public_progress",
    )
    if full_checkpoint == final_checkpoint:
        if any(key in report for key in correction_keys):
            raise ValueError("Uncorrected Stage211 SFT report contains correction evidence.")
        return _empty_sft_correction_coverage()

    try:
        from scripts.evaluate_stage211_sft_correction import (
            validate_correction_evaluation_report,
        )
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        from evaluate_stage211_sft_correction import (  # type: ignore[no-redef]
            validate_correction_evaluation_report,
        )

    evaluation_path = _validate_bound_file(
        report,
        path_key="sft_correction_evaluation_path",
        sha_key="sft_correction_evaluation_sha256",
        label="Stage211 SFT correction evaluation",
    )
    profile_path = _validate_bound_file(
        report,
        path_key="sft_correction_profile_path",
        sha_key="sft_correction_profile_sha256",
        label="Stage211 SFT correction profile",
    )
    evaluation = validate_correction_evaluation_report(
        evaluation_path,
        expected_full_completion_path=full_completion_path,
        expected_correction_profile_path=profile_path,
        require_passed=True,
    )
    if (
        Path(str(evaluation.get("checkpoint_path") or "")).resolve() != final_checkpoint
        or report.get("sft_correction_completion_receipts")
        != evaluation.get("correction_completion_receipts")
        or report.get("sft_correction_coverage") != evaluation.get("correction_coverage")
        or report.get("sft_correction_public_progress") != evaluation.get("public_progress")
    ):
        raise ValueError("Stage211 SFT correction evidence chain mismatch.")
    coverage = evaluation.get("correction_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("schema_version") != 1
        or coverage.get("applied") is not True
        or not 1 <= int(coverage.get("rounds", 0)) <= 3
        or int(coverage.get("rounds", 0)) != len(coverage.get("round_receipts") or [])
    ):
        raise ValueError("Stage211 SFT correction coverage is incomplete.")
    return dict(coverage)


def _validate_sft_report(
    path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    report = _load_json(path, label="Stage211 SFT final report")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "final_completion",
        "phase": "sft",
        "complete": True,
        "gate_passed": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Stage211 SFT final report {key} mismatch.")
    checkpoint = _validate_bound_file(
        report,
        path_key="checkpoint_path",
        sha_key="checkpoint_sha256",
        label="Stage211 SFT checkpoint",
    )
    coverage = report.get("labeled_data_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("phase") != "sft"
        or coverage.get("complete") is not True
    ):
        raise ValueError("Stage211 SFT labeled-data coverage is incomplete.")
    benchmark = validate_stage211_public_benchmark(
        report.get("public_benchmark"),
        require_metric_source_recomputed=True,
    )
    replayed_public = replay_stage211_sft_public_evidence(
        report,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_baseline_checkpoint=Path(
            str(coverage.get("init_checkpoint_path") or "")
        ).resolve(),
        expected_candidate_checkpoint=checkpoint,
    )
    benchmark = replayed_public["public_benchmark"]
    progress = replayed_public["public_progress"]
    if (
        progress.get("gate_passed") is not True
        or progress.get("no_dataset_regression") is not True
        or progress.get("macro_improved") is not True
        or int(progress.get("improved_datasets", 0)) <= 0
    ):
        raise ValueError("Stage211 SFT public-progress gate did not pass.")
    if benchmark.get("all_datasets_pass") is not True:
        raise ValueError("Stage211 SFT did not pass the every-dataset Nano WER/CER gate.")
    validate_stage211_public_overlap_binding(
        report.get("public_overlap"),
        public_benchmark=benchmark,
    )
    for path_key, sha_key, label in (
        ("sft_completion_path", "sft_completion_sha256", "completion report"),
        (
            "baseline_public_comparison_report_path",
            "baseline_public_comparison_report_sha256",
            "baseline comparison",
        ),
        (
            "public_comparison_report_path",
            "public_comparison_report_sha256",
            "public comparison",
        ),
        (
            "logits_promotion_receipt_path",
            "logits_promotion_receipt_sha256",
            "logits promotion receipt",
        ),
    ):
        _validate_bound_file(
            report,
            path_key=path_key,
            sha_key=sha_key,
            label=f"Stage211 SFT {label}",
        )
    _validate_bound_file(
        report,
        path_key="nano_teacher_checkpoint_path",
        sha_key="nano_teacher_checkpoint_sha256",
        label="Stage211 SFT Nano teacher checkpoint",
    )
    completion_path = Path(str(report["sft_completion_path"])).resolve()
    completion, completion_checkpoint = _validate_sft_completion(
        completion_path,
        require_full_profile=True,
    )
    if coverage != completion:
        raise ValueError(
            "Stage211 SFT final report differs from its validated completion coverage."
        )
    correction_coverage = _validate_sft_correction_coverage(
        report,
        full_completion_path=completion_path,
        full_checkpoint=completion_checkpoint,
        final_checkpoint=checkpoint,
    )
    for key in (
        "nano_teacher_checkpoint_path",
        "nano_teacher_checkpoint_sha256",
    ):
        if completion.get(key) != report.get(key):
            raise ValueError(f"Stage211 SFT {key} differs from its completion report.")
    nano_teacher_checkpoint = Path(str(report["nano_teacher_checkpoint_path"])).resolve()
    promotion = _validate_sft_promotion_receipt(
        receipt_path=Path(str(report["logits_promotion_receipt_path"])),
        target_phase="sft",
        checkpoint_path=Path(str(coverage["init_checkpoint_path"])),
        nano_checkpoint_path=nano_teacher_checkpoint,
    )
    if promotion.get("nano_teacher_checkpoint_sha256") != report.get(
        "nano_teacher_checkpoint_sha256"
    ):
        raise ValueError("Stage211 SFT Nano teacher differs from the logits promotion receipt.")
    if Path(str(report.get("logits_phase_gate_path") or "")).resolve() != Path(
        str(promotion.get("gate_report_path") or "")
    ).resolve() or report.get("logits_phase_gate_sha256") != promotion.get("gate_report_sha256"):
        raise ValueError("Stage211 SFT final report differs from its Logits promotion gate.")
    baseline_receipt = validate_stage211_nano_public_baseline_receipt(
        Path(str(report.get("nano_public_baseline_receipt_path") or "")),
        expected_receipt_sha256=str(report.get("nano_public_baseline_receipt_sha256") or ""),
        expected_nano_checkpoint_sha256=str(report.get("nano_teacher_checkpoint_sha256") or ""),
        public_benchmark=benchmark,
    )
    if report.get("nano_public_baseline_checkpoint_sha256") != baseline_receipt.get(
        "nano_checkpoint_sha256"
    ):
        raise ValueError("Stage211 SFT Nano public-baseline checkpoint binding mismatch.")
    if baseline_receipt.get("public_overlap") != report.get("public_overlap"):
        raise ValueError("Stage211 SFT and Nano baseline overlap bindings differ.")
    return report, checkpoint, correction_coverage


def _benchmark_rows(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validated = validate_stage211_public_benchmark(benchmark)
    return {
        str(result["dataset"]): dict(result)
        for result in validated["results"]
        if isinstance(result, dict)
    }


def _language_metric_summaries(
    dataset_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for name, language, metric in (
        ("english_wer", "en", "wer"),
        ("chinese_cer", "zh", "cer"),
    ):
        rows = [
            row
            for row in dataset_results
            if row.get("language") == language and row.get("metric") == metric
        ]
        expected_count = 3 if language == "en" else 2
        if len(rows) != expected_count:
            raise ValueError(f"Stage211 {name} macro requires exactly {expected_count} datasets.")
        stage_values = {
            stage: sum(float(row["stages"][stage]["student_error_rate"]) for row in rows)
            / len(rows)
            for stage in STAGE_ORDER
        }
        summaries.append(
            {
                "name": name,
                "language": language,
                "metric": metric,
                "aggregation": "unweighted_dataset_macro",
                "datasets": [str(row["dataset"]) for row in rows],
                "dataset_count": len(rows),
                "sample_count": sum(int(row["sample_count"]) for row in rows),
                "nano_error_rate": sum(float(row["nano_error_rate"]) for row in rows) / len(rows),
                "stages": stage_values,
            }
        )
    return summaries


def _requested_alignment_views(
    *,
    dataset_results: list[dict[str, Any]],
    language_metric_summaries: list[dict[str, Any]],
    stage_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_records_by_name = {str(row["stage"]): row for row in stage_records}
    summaries_by_name = {str(row["name"]): row for row in language_metric_summaries}
    english = summaries_by_name.get("english_wer")
    chinese = summaries_by_name.get("chinese_cer")
    if english is None or chinese is None:
        raise ValueError("Stage211 requested-stage view lacks English WER or Chinese CER.")
    nano_english_wer = _finite_float(english["nano_error_rate"], label="Nano English WER")
    nano_chinese_cer = _finite_float(chinese["nano_error_rate"], label="Nano Chinese CER")

    calibration_record = stage_records_by_name["calibration"]
    calibration_english_wer = _finite_float(
        english["stages"]["calibration"], label="calibration English WER"
    )
    calibration_chinese_cer = _finite_float(
        chinese["stages"]["calibration"], label="calibration Chinese CER"
    )
    initial_calibration_result = {
        "stage": "calibration",
        "role": "initial_baseline",
        "normalization": "ctc",
        "checkpoint_path": calibration_record["checkpoint_path"],
        "checkpoint_sha256": calibration_record["checkpoint_sha256"],
        "source_report_path": calibration_record["source_report_path"],
        "source_report_sha256": calibration_record["source_report_sha256"],
        "english_wer": calibration_english_wer,
        "nano_english_wer": nano_english_wer,
        "english_wer_gap_to_nano": calibration_english_wer - nano_english_wer,
        "chinese_cer": calibration_chinese_cer,
        "nano_chinese_cer": nano_chinese_cer,
        "chinese_cer_gap_to_nano": calibration_chinese_cer - nano_chinese_cer,
    }

    requested_alignment_results = []
    previous_stage = "calibration"
    previous_english_wer = calibration_english_wer
    previous_chinese_cer = calibration_chinese_cer
    for requested_stage in REQUESTED_ALIGNMENT_STAGE_ORDER:
        internal_stage = REQUESTED_TO_INTERNAL_STAGE[requested_stage]
        record = stage_records_by_name[internal_stage]
        english_wer = _finite_float(
            english["stages"][internal_stage],
            label=f"{requested_stage} English WER",
        )
        chinese_cer = _finite_float(
            chinese["stages"][internal_stage],
            label=f"{requested_stage} Chinese CER",
        )
        requested_alignment_results.append(
            {
                "stage": requested_stage,
                "internal_stage": internal_stage,
                "label": record["label"],
                "objective": REQUESTED_STAGE_OBJECTIVES[requested_stage],
                "normalization": "ctc",
                "previous_stage": previous_stage,
                "checkpoint_path": record["checkpoint_path"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "source_report_path": record["source_report_path"],
                "source_report_sha256": record["source_report_sha256"],
                "gate_passed": record["gate_passed"],
                "english_wer": english_wer,
                "previous_english_wer": previous_english_wer,
                "english_wer_delta_from_previous": english_wer - previous_english_wer,
                "nano_english_wer": nano_english_wer,
                "english_wer_gap_to_nano": english_wer - nano_english_wer,
                "english_wer_gap_reduction_from_previous": previous_english_wer - english_wer,
                "chinese_cer": chinese_cer,
                "previous_chinese_cer": previous_chinese_cer,
                "chinese_cer_delta_from_previous": chinese_cer - previous_chinese_cer,
                "nano_chinese_cer": nano_chinese_cer,
                "chinese_cer_gap_to_nano": chinese_cer - nano_chinese_cer,
                "chinese_cer_gap_reduction_from_previous": previous_chinese_cer - chinese_cer,
            }
        )
        previous_stage = requested_stage
        previous_english_wer = english_wer
        previous_chinese_cer = chinese_cer

    requested_language_metric_summaries = []
    for summary in language_metric_summaries:
        requested_language_metric_summaries.append(
            {
                **{key: value for key, value in summary.items() if key != "stages"},
                "initial_calibration_error_rate": _finite_float(
                    summary["stages"]["calibration"],
                    label=f"{summary['name']} calibration metric",
                ),
                "stages": {
                    requested_stage: _finite_float(
                        summary["stages"][REQUESTED_TO_INTERNAL_STAGE[requested_stage]],
                        label=f"{summary['name']} {requested_stage} metric",
                    )
                    for requested_stage in REQUESTED_ALIGNMENT_STAGE_ORDER
                },
            }
        )

    requested_dataset_results = []
    for row in dataset_results:
        requested_dataset_results.append(
            {
                **{key: value for key, value in row.items() if key != "stages"},
                "initial_calibration": dict(row["stages"]["calibration"]),
                "stages": {
                    requested_stage: dict(
                        row["stages"][REQUESTED_TO_INTERNAL_STAGE[requested_stage]]
                    )
                    for requested_stage in REQUESTED_ALIGNMENT_STAGE_ORDER
                },
            }
        )

    return (
        initial_calibration_result,
        requested_alignment_results,
        requested_language_metric_summaries,
        requested_dataset_results,
    )


def _phase_initial_checkpoint(report: dict[str, Any]) -> tuple[Path, str]:
    coverage = report["full_data_coverage"]
    segments = coverage.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Stage211 phase gate lacks curriculum segments.")
    first = segments[0]
    if not isinstance(first, dict):
        raise ValueError("Stage211 phase gate has an invalid first curriculum segment.")
    return (
        Path(str(first.get("init_checkpoint_path") or "")).resolve(),
        str(first.get("init_checkpoint_sha256") or ""),
    )


def _phase_nano_teacher_sha256(report: dict[str, Any]) -> str:
    coverage = report["full_data_coverage"]
    segments = coverage.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Stage211 phase gate lacks Nano teacher coverage.")
    supplemental = coverage.get("supplemental_natural")
    if not isinstance(supplemental, dict):
        raise ValueError("Stage211 phase gate lacks supplemental Nano teacher coverage.")
    values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "")
        for segment in [*segments, supplemental]
        if isinstance(segment, dict)
    }
    if len(values) != 1 or len(next(iter(values), "")) != 64:
        raise ValueError(
            "Stage211 Nano teacher checkpoint SHA-256 chain mismatch within phase coverage."
        )
    return next(iter(values))


def _nano_public_baseline_binding(report: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(report.get("nano_public_baseline_receipt_path") or ""),
        str(report.get("nano_public_baseline_receipt_sha256") or ""),
        str(report.get("nano_public_baseline_checkpoint_sha256") or ""),
    )


def _sft_initial_checkpoint(report: dict[str, Any]) -> tuple[Path, str]:
    coverage = report["labeled_data_coverage"]
    return (
        Path(str(coverage.get("init_checkpoint_path") or "")).resolve(),
        str(coverage.get("init_checkpoint_sha256") or ""),
    )


def _sft_ctc_label_proof(coverage: dict[str, Any]) -> dict[str, Any]:
    audit = coverage.get("labeled_data_audit")
    if not isinstance(audit, dict):
        raise ValueError("Stage211 SFT coverage lacks the full labeled-data audit.")
    profile_path_value = coverage.get("labeled_profile_receipt_path")
    profile_binding: dict[str, Any] = {}
    if profile_path_value is not None:
        profile_path = Path(str(profile_path_value)).resolve()
        if coverage.get("labeled_profile_receipt_sha256") != sha256_file(profile_path):
            raise ValueError("Stage211 SFT labeled profile receipt changed.")
        profile = validate_labeled_profile_receipt(
            profile_path,
            labeled_root=Path(str(coverage.get("labeled_webdataset_root") or "")),
            length_index=Path(str(coverage.get("length_index_path") or "")),
            bucket_manifest=Path(str(coverage.get("bucket_manifest_path") or "")),
        )
        if profile.get("labeled_data_audit") != audit:
            raise ValueError("Stage211 SFT audit differs from its full-labeled profile.")
        labeled_expected = dict(profile["expected"])
        expected_source_counts = labeled_expected["source_counts"]
        expected_language_counts = labeled_expected["language_counts"]
        profile_binding = {
            "labeled_profile_schema_version": int(profile["schema_version"]),
            "labeled_profile_receipt_path": str(profile_path),
            "labeled_profile_receipt_sha256": sha256_file(profile_path),
            "all_accepted_unique_rows_required": True,
            "source_language_interleave_required": True,
        }
        overlap_path = Path(
            str(coverage.get("sft_public_overlap_receipt_path") or "")
        ).resolve()
        if coverage.get("sft_public_overlap_receipt_sha256") != sha256_file(overlap_path):
            raise ValueError("Stage211 SFT public-audio overlap receipt changed.")
        overlap_receipt = validate_stage211_sft_public_overlap_receipt(
            overlap_path,
            expected_labeled_profile=profile_path,
        )
        overlap_coverage = dict(overlap_receipt["coverage"])
        overlap_result = dict(overlap_receipt["overlap"])
        overlap_profile = dict(overlap_receipt["labeled_profile"])
        profile_binding.update(
            {
                "public_audio_isolation_passed": True,
                "public_audio_overlap_receipt_path": str(overlap_path),
                "public_audio_overlap_receipt_sha256": sha256_file(overlap_path),
                "public_audio_comparison_mode": overlap_receipt["comparison_mode"],
                "public_audio_size_prefilter_lossless_for_exact_bytes": True,
                "public_audio_scanned_rows": int(overlap_coverage["scanned_rows"]),
                "public_audio_public_rows": int(
                    overlap_receipt["public_benchmark"]["public_rows"]
                ),
                "public_audio_training_overlap_rows": int(
                    overlap_result["training_rows"]
                ),
                "public_audio_internal_eval_overlap_rows": int(
                    overlap_result["internal_eval_rows"]
                ),
                "public_audio_normalized_pcm_complete": False,
                "public_audio_near_duplicate_complete": False,
            }
        )
        rebuild_path_value = overlap_profile.get("public_clean_rebuild_receipt_path")
        if rebuild_path_value is not None:
            rebuild_path = Path(str(rebuild_path_value)).resolve()
            if overlap_profile.get("public_clean_rebuild_receipt_sha256") != sha256_file(
                rebuild_path
            ):
                raise ValueError("Stage211 SFT public-clean rebuild receipt changed.")
            rebuild = validate_stage211_sft_public_clean_rebuild_receipt(rebuild_path)
            if (
                Path(str(rebuild["output_profile_path"])).resolve() != profile_path
                or rebuild["output_profile_sha256"] != sha256_file(profile_path)
            ):
                raise ValueError("Stage211 SFT public-clean rebuild profile changed.")
            profile_binding.update(
                {
                    "public_clean_rebuild_passed": True,
                    "public_clean_rebuild_receipt_path": str(rebuild_path),
                    "public_clean_rebuild_receipt_sha256": sha256_file(rebuild_path),
                    "public_clean_exclusion_reason": rebuild["exclusion_reason"],
                    "public_clean_exclusions_rows": int(rebuild["exclusions_rows"]),
                    "public_clean_rejected_training_overlap_rows": int(
                        rebuild["rejected_overlap_training_rows"]
                    ),
                    "public_clean_rejected_internal_eval_overlap_rows": int(
                        rebuild["rejected_overlap_internal_eval_rows"]
                    ),
                    "public_clean_source_profile_path": rebuild["source_profile_path"],
                    "public_clean_source_profile_sha256": rebuild["source_profile_sha256"],
                }
            )
    else:
        labeled_expected = dict(LABELED_EXPECTED)
        expected_source_counts = STAGE211_LABELED_SOURCE_COUNTS
        expected_language_counts = STAGE211_LABELED_LANGUAGE_COUNTS
    for key, expected in labeled_expected.items():
        audit_value = audit.get(key)
        coverage_value = coverage.get(key)
        if key == "total_hours":
            if not math.isclose(
                float(audit_value),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-5,
            ) or not math.isclose(
                float(coverage_value),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError(f"Stage211 SFT CTC label proof {key} mismatch.")
        elif audit_value != expected or coverage_value != expected:
            raise ValueError(f"Stage211 SFT CTC label proof {key} mismatch.")
    if coverage.get("ctc_suppress_non_pronunciation_tokens") is not True:
        raise ValueError("Stage211 SFT did not suppress non-pronunciation CTC logits.")
    expected_support = {
        "ctc_suppressed_token_ids_count": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT),
        "ctc_suppressed_token_ids_sha256": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256),
        "teacher_projection_support_matches_student": True,
    }
    for key, expected in expected_support.items():
        if coverage.get(key) != expected:
            raise ValueError(f"Stage211 SFT CTC pronunciation support {key} mismatch.")
    preparation = audit.get("label_preparation")
    if not isinstance(preparation, dict):
        raise ValueError("Stage211 SFT CTC label-preparation proof is missing.")
    expected_preparation = {
        "tokenizer_type": "sensevoice_tiktoken",
        "text_normalization": "ctc",
        "frontend_downsample": "sensevoice_lfr6",
        "drop_unk_token": True,
        "ctc_unk_tokens": 0,
        "non_pronunciation_target_policy": "ctc_normalization",
        "non_pronunciation_logit_policy": "tokenizer_special_tokens_suppressed",
        "ctc_suppressed_token_ids_count": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT),
        "ctc_suppressed_token_ids_sha256": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256),
        "source_counts": expected_source_counts,
        "language_counts": expected_language_counts,
    }
    for key, expected in expected_preparation.items():
        if preparation.get(key) != expected:
            raise ValueError(f"Stage211 SFT CTC label-preparation {key} mismatch.")
    for path_key, sha_key in (
        ("summary_path", "summary_sha256"),
        ("log_path", "log_sha256"),
        ("tokenizer_model_path", "tokenizer_model_sha256"),
    ):
        path = Path(str(preparation.get(path_key) or "")).resolve()
        if not path.is_file() or preparation.get(sha_key) != sha256_file(path):
            raise ValueError(
                f"Stage211 SFT CTC label-preparation artifact is unavailable or changed: {path}"
            )
    return {
        "full_length_index_audit_passed": True,
        "ctc_label_normalization_chain_passed": True,
        "ctc_suppress_non_pronunciation_tokens": True,
        **expected_support,
        "train_samples": int(audit["train_samples"]),
        "eval_samples": int(audit["eval_samples"]),
        "total_samples": int(audit["total_samples"]),
        "unique_utterance_ids": int(audit["unique_utterance_ids"]),
        "pronunciation_target_samples": int(audit["pronunciation_target_samples"]),
        "ctc_feasible_samples": int(audit["ctc_feasible_samples"]),
        "ctc_tokens": int(audit["ctc_tokens"]),
        "ctc_unk_tokens": int(audit["ctc_unk_tokens"]),
        "ctc_forbidden_tokens": int(audit.get("ctc_forbidden_tokens", 0)),
        **profile_binding,
        **expected_preparation,
        "summary_path": str(preparation["summary_path"]),
        "summary_sha256": str(preparation["summary_sha256"]),
        "log_path": str(preparation["log_path"]),
        "log_sha256": str(preparation["log_sha256"]),
        "tokenizer_model_path": str(preparation["tokenizer_model_path"]),
        "tokenizer_model_sha256": str(preparation["tokenizer_model_sha256"]),
    }


def _require_chain_link(
    *,
    source_stage: str,
    source_checkpoint: Path,
    source_sha256: str,
    target_stage: str,
    target_init: Path,
    target_init_sha256: str,
) -> dict[str, Any]:
    if target_init != source_checkpoint:
        raise ValueError(f"Stage211 checkpoint chain mismatch: {source_stage} -> {target_stage}.")
    if target_init_sha256 != source_sha256 or source_sha256 != sha256_file(source_checkpoint):
        raise ValueError(
            f"Stage211 checkpoint SHA-256 chain mismatch: {source_stage} -> {target_stage}."
        )
    return {
        "source_stage": source_stage,
        "target_stage": target_stage,
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": source_sha256,
    }


def _stage_record(
    *,
    stage: str,
    checkpoint: Path,
    source_report: Path,
    benchmark: dict[str, Any],
    data_coverage: dict[str, Any] | None,
    gate_passed: bool | None,
    nano_teacher_checkpoint_sha256: str | None,
    preflight_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": STAGE_LABELS[stage],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_report_path": str(source_report),
        "source_report_sha256": sha256_file(source_report),
        "gate_passed": gate_passed,
        "gate_status": "baseline" if gate_passed is None else "pass",
        "nano_teacher_checkpoint_sha256": nano_teacher_checkpoint_sha256,
        "preflight_smoke": preflight_smoke,
        "data_coverage": data_coverage,
        "public_benchmark": benchmark,
    }


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Stage211 {label} must be finite.")
    return result


def _alignment_component_result(
    raw: Any,
    *,
    label: str,
    include_cells: bool,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Stage211 {label} component summary is missing.")
    baseline_loss = _finite_float(raw.get("baseline_mean_loss"), label=f"{label} baseline loss")
    candidate_loss = _finite_float(raw.get("candidate_mean_loss"), label=f"{label} candidate loss")
    result: dict[str, Any] = {
        "baseline_mean_loss": baseline_loss,
        "candidate_mean_loss": candidate_loss,
        "relative_loss_reduction": (baseline_loss - candidate_loss)
        / max(abs(baseline_loss), 1.0e-12),
        "baseline_mean_cosine": _finite_float(
            raw.get("baseline_mean_cosine"), label=f"{label} baseline cosine"
        ),
        "candidate_mean_cosine": _finite_float(
            raw.get("candidate_mean_cosine"), label=f"{label} candidate cosine"
        ),
        "loss_improved_layers": int(raw.get("loss_improved_layers", -1)),
        "cosine_improved_layers": int(raw.get("cosine_improved_layers", -1)),
        "weak_bands": {},
    }
    if (
        not 0 <= result["loss_improved_layers"] <= 70
        or not 0 <= result["cosine_improved_layers"] <= 70
    ):
        raise ValueError(f"Stage211 {label} improved-layer count is invalid.")
    weak_bands = raw.get("weak_bands")
    if not isinstance(weak_bands, dict) or set(weak_bands) != {"10-19", "20-29"}:
        raise ValueError(f"Stage211 {label} weak-band evidence is incomplete.")
    for band_name, band in weak_bands.items():
        if not isinstance(band, dict):
            raise ValueError(f"Stage211 {label}/{band_name} weak-band evidence is invalid.")
        result["weak_bands"][band_name] = {
            key: _finite_float(band.get(key), label=f"{label}/{band_name} {key}")
            for key in (
                "baseline_loss",
                "candidate_loss",
                "baseline_cosine",
                "candidate_cosine",
            )
        }
    if include_cells:
        cells = raw.get("cells")
        if not isinstance(cells, dict) or set(cells) != set(ALIGNMENT_CELLS):
            raise ValueError(f"Stage211 {label} component cell coverage is incomplete.")
        result["cells"] = {
            cell_name: {
                key: (
                    int(cell.get(key, -1))
                    if key in {"layers_loss_improved", "layers_cosine_improved"}
                    else _finite_float(cell.get(key), label=f"{label}/{cell_name} {key}")
                )
                for key in (
                    "baseline_loss",
                    "candidate_loss",
                    "baseline_cosine",
                    "candidate_cosine",
                    "layers_loss_improved",
                    "layers_cosine_improved",
                )
            }
            for cell_name, cell in cells.items()
            if isinstance(cell, dict)
        }
        if set(result["cells"]) != set(ALIGNMENT_CELLS):
            raise ValueError(f"Stage211 {label} component cells are invalid.")
    return result


def _logits_metric_pairs(
    baseline: Any,
    candidate: Any,
    *,
    label: str,
) -> dict[str, dict[str, float]]:
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError(f"Stage211 {label} logits metrics are missing.")
    return {
        metric: {
            "baseline": _finite_float(baseline.get(metric), label=f"{label} baseline {metric}"),
            "candidate": _finite_float(candidate.get(metric), label=f"{label} candidate {metric}"),
        }
        for metric in LOGITS_DISCLOSURE_METRICS
    }


def _decoder_result(raw: Any, *, label: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Stage211 {label} decoder-hidden evidence is invalid.")
    baseline_loss = _finite_float(raw.get("baseline_loss"), label=f"{label} baseline loss")
    candidate_loss = _finite_float(raw.get("candidate_loss"), label=f"{label} candidate loss")
    return {
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "relative_change": (candidate_loss - baseline_loss) / max(abs(baseline_loss), 1.0e-12),
        **({"retained": bool(raw["retained"])} if "retained" in raw else {}),
    }


def _trajectory_retention_result(
    *,
    phase: str,
    phase_report: dict[str, Any],
) -> dict[str, Any]:
    trajectory = phase_report.get("trajectory_retention")
    if (
        not isinstance(trajectory, dict)
        or phase_report.get("trajectory_retention_gate_passed") is not True
        or trajectory.get("gate_passed") is not True
        or int(trajectory.get("fixed_eval_samples", -1)) != 256
        or float(trajectory.get("max_relative_regression_pct", float("nan"))) != 10.0
    ):
        raise ValueError(f"Stage211 {phase} trajectory-retention evidence did not pass.")
    source_order = trajectory.get("source_order")
    entries = trajectory.get("entries")
    required_prefix = ["easy", "medium", "hard", "long", "supplemental_natural"]
    if (
        not isinstance(source_order, list)
        or source_order[: len(required_prefix)] != required_prefix
        or not isinstance(entries, list)
        or len(entries) != len(source_order)
    ):
        raise ValueError(f"Stage211 {phase} trajectory-retention source order is invalid.")
    best_prior = trajectory.get("best_prior")
    candidate = trajectory.get("candidate")
    if not isinstance(best_prior, dict) or not isinstance(candidate, dict):
        raise ValueError(f"Stage211 {phase} trajectory-retention endpoints are invalid.")
    if candidate.get("checkpoint_sha256") != phase_report.get("checkpoint_sha256"):
        raise ValueError(f"Stage211 {phase} trajectory candidate checkpoint mismatch.")
    return {
        "gate_passed": True,
        "fixed_eval_samples": 256,
        "source_order": [str(value) for value in source_order],
        "terminal_entries": len(entries),
        "best_prior_source": str(best_prior.get("source_name") or ""),
        "best_prior_loss": _finite_float(
            trajectory.get("best_prior_loss"), label=f"{phase} trajectory best-prior loss"
        ),
        "candidate_source": str(candidate.get("source_name") or ""),
        "candidate_loss": _finite_float(
            trajectory.get("candidate_loss"), label=f"{phase} trajectory candidate loss"
        ),
        "relative_regression_pct": _finite_float(
            trajectory.get("relative_regression_pct"),
            label=f"{phase} trajectory relative regression",
        ),
        "max_relative_regression_pct": 10.0,
    }


def _step_eval_cadence_result(
    *,
    phase: str,
    phase_report: dict[str, Any],
) -> dict[str, Any]:
    expected_interval = (
        STAGE211_SFT_STEP_EVAL_INTERVAL if phase == "sft" else STAGE211_STEP_EVAL_INTERVAL
    )
    required_prefix = (
        ["labeled_sft"]
        if phase == "sft"
        else ["easy", "medium", "hard", "long", "supplemental_natural"]
    )
    cadence = phase_report.get("step_eval_cadence")
    if (
        not isinstance(cadence, dict)
        or cadence.get("pipeline") != "stage211"
        or cadence.get("artifact") != "step_eval_cadence"
        or cadence.get("phase") != phase
        or cadence.get("complete") is not True
        or int(cadence.get("interval_steps", -1)) != expected_interval
        or int(cadence.get("eval_samples", -1)) != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES
    ):
        raise ValueError(f"Stage211 {phase} periodic fixed-eval cadence is incomplete.")
    source_order = cadence.get("source_order")
    sources = cadence.get("sources")
    if (
        not isinstance(source_order, list)
        or source_order[: len(required_prefix)] != required_prefix
        or not isinstance(sources, list)
        or len(sources) != len(source_order)
        or int(cadence.get("source_count", -1)) != len(sources)
    ):
        raise ValueError(f"Stage211 {phase} periodic fixed-eval source order is invalid.")
    source_results = []
    total_reports = 0
    for order, (source_name, source) in enumerate(zip(source_order, sources, strict=True)):
        if (
            not isinstance(source, dict)
            or int(source.get("order", -1)) != order
            or source.get("source_name") != source_name
            or int(source.get("interval_steps", -1)) != expected_interval
            or int(source.get("expected_report_count", -1))
            != int(source.get("actual_report_count", -2))
        ):
            raise ValueError(f"Stage211 {phase}/{source_name} fixed-eval cadence is invalid.")
        report_count = int(source["actual_report_count"])
        total_reports += report_count
        source_results.append(
            {
                "source": str(source_name),
                "source_kind": str(source.get("source_kind") or ""),
                "terminal_step": int(source.get("terminal_step", -1)),
                "report_count": report_count,
                "eval_samples_per_report": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
            }
        )
    if total_reports != int(cadence.get("total_reports", -1)):
        raise ValueError(f"Stage211 {phase} fixed-eval report total is invalid.")
    return {
        "complete": True,
        "interval_steps": expected_interval,
        "eval_samples_per_report": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
        "source_order": [str(value) for value in source_order],
        "source_count": len(sources),
        "total_reports": total_reports,
        "sources": source_results,
    }


def _alignment_result(*, phase: str, phase_report: dict[str, Any]) -> dict[str, Any]:
    record = phase_report.get("alignment_report")
    if not isinstance(record, dict):
        raise ValueError(f"Stage211 {phase} phase report lacks alignment evidence.")
    source_path = Path(str(record.get("path") or "")).expanduser().resolve()
    if not source_path.is_file() or sha256_file(source_path) != record.get("sha256"):
        raise ValueError(f"Stage211 {phase} alignment report is unavailable or changed.")
    source = _load_json(source_path, label=f"Stage211 {phase} alignment report")
    expected_artifact = "logits_alignment_gate" if phase == "logits" else "hidden_alignment_gate"
    if (
        source.get("pipeline") != "stage211"
        or source.get("artifact") != expected_artifact
        or source.get("phase") != phase
        or source.get("gate_passed") is not True
        or phase_report.get("alignment_gate_passed") is not True
    ):
        raise ValueError(f"Stage211 {phase} alignment disclosure source did not pass.")
    stratified = source.get("stratified_summary")
    if not isinstance(stratified, dict) or set(stratified.get("cells", {})) != set(ALIGNMENT_CELLS):
        raise ValueError(f"Stage211 {phase} alignment disclosure lacks nine-cell evidence.")
    common = {
        "stage": phase,
        "label": STAGE_LABELS[phase],
        "objective": "ctc_logits" if phase == "logits" else "hidden_states",
        "gate_passed": True,
        "source_report_path": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "baseline_checkpoint_path": str(source["baseline_checkpoint_path"]),
        "baseline_checkpoint_sha256": str(source["baseline_checkpoint_sha256"]),
        "checkpoint_path": str(source["checkpoint_path"]),
        "checkpoint_sha256": str(source["checkpoint_sha256"]),
        "fixed_eval_samples": int(
            source.get("baseline_eval_provenance", {}).get("split_samples", -1)
        ),
        "stratified_gate_passed": source.get("stratified_gate_passed") is True,
        "stratified_summary_path": str(source.get("stratified_summary_path") or ""),
        "stratified_summary_sha256": str(source.get("stratified_summary_sha256") or ""),
        "stratified_cells": list(ALIGNMENT_CELLS),
        "stratified_samples": sum(
            int(cell.get("samples", -1))
            for cell in stratified["cells"].values()
            if isinstance(cell, dict)
        ),
        "trajectory_retention": _trajectory_retention_result(
            phase=phase,
            phase_report=phase_report,
        ),
        "step_eval_cadence": _step_eval_cadence_result(
            phase=phase,
            phase_report=phase_report,
        ),
    }
    if common["fixed_eval_samples"] != 256 or common["stratified_samples"] != 256 * len(
        ALIGNMENT_CELLS
    ):
        raise ValueError(f"Stage211 {phase} alignment disclosure sample coverage mismatch.")
    if phase in {"mixer", "block"}:
        components = source.get("component_summaries")
        stratified_components = stratified.get("component_summaries")
        required_components = ("mixer",) if phase == "mixer" else ("mixer", "ffn", "block")
        if (
            not isinstance(components, dict)
            or set(components) != set(required_components)
            or not isinstance(stratified_components, dict)
            or set(stratified_components) != set(required_components)
        ):
            raise ValueError(f"Stage211 {phase} alignment component coverage mismatch.")
        cells = {}
        for cell_name, cell in stratified["cells"].items():
            if not isinstance(cell, dict):
                raise ValueError(f"Stage211 {phase}/{cell_name} alignment cell is invalid.")
            cells[cell_name] = {
                "samples": int(cell.get("samples", -1)),
                "baseline_loss": _finite_float(
                    cell.get("baseline_loss"), label=f"{phase}/{cell_name} baseline loss"
                ),
                "candidate_loss": _finite_float(
                    cell.get("candidate_loss"), label=f"{phase}/{cell_name} candidate loss"
                ),
                "layers_loss_improved": int(cell.get("layers_loss_improved", -1)),
                "layers_cosine_improved": int(cell.get("layers_cosine_improved", -1)),
            }
        return {
            **common,
            "fixed": {
                "baseline_eval_loss": _finite_float(
                    source.get("baseline_eval_loss"), label=f"{phase} baseline eval loss"
                ),
                "candidate_eval_loss": _finite_float(
                    source.get("candidate_eval_loss"), label=f"{phase} candidate eval loss"
                ),
                "components": {
                    name: _alignment_component_result(
                        components[name], label=f"{phase}/fixed/{name}", include_cells=False
                    )
                    for name in required_components
                },
                "decoder_hidden": _decoder_result(
                    source.get("decoder_hidden"), label=f"{phase}/fixed decoder"
                ),
            },
            "stratified": {
                "macro": {
                    key: _finite_float(
                        stratified.get("macro", {}).get(key), label=f"{phase}/macro {key}"
                    )
                    for key in ("baseline_loss", "candidate_loss", "relative_change_pct")
                },
                "cells": cells,
                "components": {
                    name: _alignment_component_result(
                        stratified_components[name],
                        label=f"{phase}/stratified/{name}",
                        include_cells=True,
                    )
                    for name in required_components
                },
                "decoder_hidden": _decoder_result(
                    stratified.get("decoder_hidden"), label=f"{phase}/stratified decoder"
                ),
            },
        }

    hidden_components = source.get("hidden_component_summaries")
    stratified_hidden = stratified.get("hidden_component_summaries")
    required_hidden = ("mixer", "ffn", "block")
    if (
        not isinstance(hidden_components, dict)
        or set(hidden_components) != set(required_hidden)
        or not isinstance(stratified_hidden, dict)
        or set(stratified_hidden) != set(required_hidden)
    ):
        raise ValueError("Stage211 logits hidden-retention disclosure is incomplete.")
    logits_cells = {}
    for cell_name, cell in stratified["cells"].items():
        if not isinstance(cell, dict):
            raise ValueError(f"Stage211 logits/{cell_name} alignment cell is invalid.")
        logits_cells[cell_name] = {
            "samples": int(cell.get("samples", -1)),
            "metrics": _logits_metric_pairs(
                cell.get("baseline_metrics"),
                cell.get("candidate_metrics"),
                label=f"logits/{cell_name}",
            ),
        }
    return {
        **common,
        "fixed": {
            "metrics": _logits_metric_pairs(
                source.get("baseline_metrics"),
                source.get("candidate_metrics"),
                label="logits/fixed",
            ),
            "full_kl_relative_reduction": _finite_float(
                source.get("full_kl_relative_reduction"), label="logits fixed full-KL reduction"
            ),
            "conditional_nonblank_kl_relative_reduction": _finite_float(
                source.get("conditional_nonblank_kl_relative_reduction"),
                label="logits fixed conditional-KL reduction",
            ),
            "hidden_components": {
                name: _alignment_component_result(
                    hidden_components[name],
                    label=f"logits/fixed/{name}",
                    include_cells=False,
                )
                for name in required_hidden
            },
            "decoder_hidden": _decoder_result(
                source.get("decoder_hidden_retention"), label="logits/fixed decoder"
            ),
        },
        "stratified": {
            "metrics": _logits_metric_pairs(
                stratified.get("macro", {}).get("baseline_metrics"),
                stratified.get("macro", {}).get("candidate_metrics"),
                label="logits/stratified macro",
            ),
            "cells": logits_cells,
            "hidden_components": {
                name: _alignment_component_result(
                    stratified_hidden[name],
                    label=f"logits/stratified/{name}",
                    include_cells=True,
                )
                for name in required_hidden
            },
            "decoder_hidden": _decoder_result(
                stratified.get("decoder_hidden"), label="logits/stratified decoder"
            ),
        },
    }


def _coverage_record(
    *,
    stage: str,
    coverage: dict[str, Any],
    correction_round_promotion: dict[str, Any] | None = None,
    sft_correction_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage in {"mixer", "block", "logits"}:
        unique_rows = int(coverage["total_unique_rows"])
        row_exposures = int(coverage["total_row_exposures"])
        if unique_rows <= 0 or row_exposures % unique_rows != 0:
            raise ValueError(f"Stage211 {stage} coverage does not contain exact full epochs.")
        correction = coverage.get("post_coverage_correction_exposure")
        if not isinstance(correction, dict):
            correction = {
                "rounds": 0,
                "row_exposures": 0,
                "hour_exposures": 0.0,
                "steps": 0,
                "executed_sample_exposures": 0,
            }
        correction_rounds = int(correction.get("rounds", 0))
        expected_correction_round_promotion = build_stage211_correction_round_promotion_gate(
            correction_rounds
        )
        if (
            correction_round_promotion != expected_correction_round_promotion
            or expected_correction_round_promotion["gate_passed"] is not True
        ):
            raise ValueError(
                f"Stage211 {stage} final correction-round promotion evidence is invalid."
            )
        original_segments = coverage.get("segments")
        supplemental_segment = coverage.get("supplemental_natural")
        if not isinstance(original_segments, list) or not isinstance(supplemental_segment, dict):
            raise ValueError(f"Stage211 {stage} coverage segment proof is incomplete.")
        training_segments = []
        for segment in [*original_segments, supplemental_segment]:
            if not isinstance(segment, dict):
                raise ValueError(f"Stage211 {stage} contains an invalid coverage segment.")
            segment_epochs = int(segment.get("epochs", -1))
            if segment_epochs != STAGE211_FULL_DATA_EPOCHS:
                raise ValueError(
                    f"Stage211 {stage}/{segment.get('difficulty')} does not contain "
                    f"{STAGE211_FULL_DATA_EPOCHS} epochs."
                )
            training_segments.append(
                {
                    "difficulty": str(segment["difficulty"]),
                    "rows": int(segment["rows"]),
                    "hours": float(segment["hours"]),
                    "epochs": segment_epochs,
                    "steps_per_epoch": int(segment["steps_per_epoch"]),
                    "steps": int(segment["steps"]),
                    "row_exposures": int(segment["row_exposures"]),
                    "hour_exposures": float(segment["hour_exposures"]),
                    "tail_padding_sample_exposures": int(segment["tail_padding_sample_exposures"]),
                    "executed_sample_exposures": int(segment["executed_sample_exposures"]),
                }
            )
        expected_integer_totals = {
            "rows": unique_rows,
            "row_exposures": row_exposures,
            "tail_padding_sample_exposures": int(coverage["total_tail_padding_sample_exposures"]),
            "executed_sample_exposures": int(coverage["total_executed_sample_exposures"]),
        }
        for key, expected in expected_integer_totals.items():
            if sum(int(row[key]) for row in training_segments) != expected:
                raise ValueError(
                    f"Stage211 {stage} segment {key} total differs from combined coverage."
                )
        expected_float_totals = {
            "hours": float(coverage["total_hours"]),
            "hour_exposures": float(coverage["total_hour_exposures"]),
        }
        for key, expected in expected_float_totals.items():
            if not math.isclose(
                sum(float(row[key]) for row in training_segments),
                expected,
                rel_tol=0.0,
                abs_tol=0.005,
            ):
                raise ValueError(
                    f"Stage211 {stage} segment {key} total differs from combined coverage."
                )
        correction_hour_exposures = float(correction.get("hour_exposures", 0.0))
        return {
            "stage": stage,
            "label": STAGE_LABELS[stage],
            "objective": "online_nano_ctc_distillation",
            "unique_or_train_rows": unique_rows,
            "evaluation_rows": 0,
            "hours_per_epoch": float(coverage["total_hours"]),
            "epochs": row_exposures // unique_rows,
            "row_exposures": row_exposures,
            "hour_exposures": float(coverage["total_hour_exposures"]),
            "executed_sample_exposures": int(coverage["total_executed_sample_exposures"]),
            "correction_rounds": correction_rounds,
            "correction_round_promotion": expected_correction_round_promotion,
            "correction_row_exposures": int(correction.get("row_exposures", 0)),
            "correction_hour_exposures": correction_hour_exposures,
            "effective_hour_exposures": (
                float(coverage["total_hour_exposures"]) + correction_hour_exposures
            ),
            "original_unique_rows": int(coverage["original_total_unique_rows"]),
            "supplemental_unique_rows": int(supplemental_segment["rows"]),
            "supplemental_inventory_path": str(supplemental_segment["supplemental_inventory_path"]),
            "supplemental_inventory_sha256": str(
                supplemental_segment["supplemental_inventory_sha256"]
            ),
            "training_segments": training_segments,
        }
    if stage == "sft":
        epochs = int(coverage["epochs"])
        total_hours = float(coverage["total_hours"])
        label_proof = _sft_ctc_label_proof(coverage)
        correction = (
            dict(sft_correction_coverage)
            if sft_correction_coverage is not None
            else _empty_sft_correction_coverage()
        )
        correction_rounds = int(correction["rounds"])
        correction_rows = int(correction["row_exposures"])
        correction_hours = float(correction["hour_exposures"])
        correction_steps = int(correction["steps"])
        correction_tail = int(correction["tail_padding_sample_exposures"])
        correction_executed = int(correction["executed_sample_exposures"])
        base_rows = int(coverage["train_samples"]) * epochs
        base_hours = total_hours * epochs
        base_steps = int(coverage["estimated_train_steps"])
        base_tail = int(coverage["tail_padding_sample_exposures"])
        base_executed = int(coverage["executed_sample_exposures"])
        return {
            "stage": stage,
            "label": STAGE_LABELS[stage],
            "objective": "labeled_ctc_sft",
            "unique_or_train_rows": int(coverage["train_samples"]),
            "evaluation_rows": int(coverage["eval_samples"]),
            "hours_per_epoch": total_hours,
            "epochs": epochs,
            "row_exposures": base_rows,
            "hour_exposures": base_hours,
            "steps": base_steps,
            "tail_padding_sample_exposures": base_tail,
            "executed_sample_exposures": base_executed,
            "correction_rounds": correction_rounds,
            "correction_row_exposures": correction_rows,
            "correction_hour_exposures": correction_hours,
            "correction_steps": correction_steps,
            "correction_tail_padding_sample_exposures": correction_tail,
            "correction_executed_sample_exposures": correction_executed,
            "effective_row_exposures": base_rows + correction_rows,
            "effective_hour_exposures": base_hours + correction_hours,
            "effective_steps": base_steps + correction_steps,
            "effective_tail_padding_sample_exposures": base_tail + correction_tail,
            "effective_executed_sample_exposures": base_executed + correction_executed,
            "correction": correction,
            "ctc_tokens": int(coverage["ctc_tokens"]),
            "ctc_unk_tokens": int(coverage["ctc_unk_tokens"]),
            "ctc_label_proof": label_proof,
            "step_eval_cadence": _step_eval_cadence_result(
                phase="sft",
                phase_report=coverage,
            ),
        }
    raise ValueError(f"Stage211 stage has no training coverage: {stage!r}")


def _supplemental_dedupe_proof(inventory_path: Path) -> dict[str, Any]:
    inventory = _load_json(
        inventory_path,
        label="Stage211 supplemental natural inventory",
    )
    cross_pool = inventory.get("cross_pool_dedupe")
    if not isinstance(cross_pool, dict):
        raise ValueError("Stage211 supplemental inventory lacks cross-pool dedupe proof.")
    expected_exclusions = {"llaso_gigaspeech", "llaso_librispeech"}
    exclusions = set(cross_pool.get("known_overlap_exclusions") or [])
    if (
        inventory.get("schema_version") != 2
        or inventory.get("artifact") != "stage211_supplemental_combined_inventory"
        or cross_pool.get("mode") != "source_identity_plus_known_corpus_exclusion"
        or cross_pool.get("source_sets_disjoint") is not True
        or cross_pool.get("content_fingerprint_complete") is not False
        or cross_pool.get("base_public_overlap_normalized_pcm_exact_complete") is not True
        or int(cross_pool.get("base_public_overlap_rows", -1)) != 0
        or cross_pool.get("social_normalized_pcm_exact_complete") is not True
        or cross_pool.get("social_public_overlap_mode") != "normalized_pcm_exact"
        or cross_pool.get("archived_social_exact_duplicate_exclusion_complete") is not True
        or cross_pool.get("usb_top_level_classification_complete") is not True
        or cross_pool.get("usb_unresolved_natural_entries") != []
        or cross_pool.get("near_duplicate_complete") is not False
        or exclusions != expected_exclusions
    ):
        raise ValueError("Stage211 supplemental cross-pool dedupe contract mismatch.")
    supplemental_sources = cross_pool.get("supplemental_sources")
    if not isinstance(supplemental_sources, list) or not supplemental_sources:
        raise ValueError("Stage211 supplemental source-set proof is empty.")
    stage179 = cross_pool.get("stage179")
    if not isinstance(stage179, dict):
        raise ValueError("Stage211 supplemental inventory lacks Stage179 binding.")
    stage179_manifest = Path(str(stage179.get("manifest_path") or "")).resolve()
    stage179_manifest_sha256 = str(stage179.get("manifest_sha256") or "")
    if (
        not stage179_manifest.is_file()
        or sha256_file(stage179_manifest) != stage179_manifest_sha256
    ):
        raise ValueError("Stage211 supplemental Stage179 binding is missing or changed.")
    stage179_rows = int(stage179.get("total_unique_rows", -1))
    stage179_hours = float(stage179.get("total_unique_hours", float("nan")))
    if stage179_rows <= 0 or not math.isfinite(stage179_hours) or stage179_hours <= 0.0:
        raise ValueError("Stage211 supplemental Stage179 coverage binding is invalid.")
    components = inventory.get("component_inventories")
    if not isinstance(components, dict) or set(components) != {"base_natural", "social_vad"}:
        raise ValueError("Stage211 supplemental component inventory proof is incomplete.")
    base_component = components.get("base_natural")
    base_audit_record = inventory.get("base_public_overlap_audit")
    if not isinstance(base_component, dict) or not isinstance(base_audit_record, dict):
        raise ValueError("Stage211 supplemental base public-overlap proof is incomplete.")
    base_inventory_path = Path(str(base_component.get("inventory_path") or "")).resolve()
    base_audit_path = Path(str(base_audit_record.get("receipt_path") or "")).resolve()
    if (
        not base_inventory_path.is_file()
        or sha256_file(base_inventory_path) != base_component.get("inventory_sha256")
        or not base_audit_path.is_file()
        or sha256_file(base_audit_path) != base_audit_record.get("receipt_sha256")
    ):
        raise ValueError("Stage211 supplemental base public-overlap binding changed.")
    base_audit = _load_json(
        base_audit_path,
        label="Stage211 base public-overlap audit",
    )
    if (
        base_audit.get("artifact") != "stage211_base_public_pcm_overlap_audit"
        or base_audit.get("complete") is not True
        or base_audit.get("training_ready") is not True
        or base_audit.get("admission_state") != "normalized_pcm_exact_public_clear"
        or base_audit.get("comparison_mode") != "normalized_pcm_exact"
        or base_audit.get("scan_order") != STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER
        or base_audit.get("near_duplicate_complete") is not False
        or int(base_audit.get("decode_failures", -1)) != 0
        or int(base_audit.get("public_overlap_rows", -1)) != 0
        or base_audit.get("base_inventory_path") != str(base_inventory_path)
        or base_audit.get("base_inventory_sha256") != base_component.get("inventory_sha256")
        or int(base_audit.get("scanned_rows", -1)) != int(base_component.get("rows", -2))
        or base_audit_record.get("comparison_mode") != "normalized_pcm_exact"
        or base_audit_record.get("scan_order") != STAGE211_BASE_PUBLIC_PCM_SCAN_ORDER
        or int(base_audit_record.get("scanned_rows", -1)) != int(base_component.get("rows", -2))
        or int(base_audit_record.get("public_overlap_rows", -1)) != 0
        or base_audit_record.get("training_ready") is not True
    ):
        raise ValueError("Stage211 supplemental base public-overlap audit changed.")
    usb_record = inventory.get("usb_top_level_coverage")
    overlap_record = inventory.get("archived_social_exclusion")
    resolution = inventory.get("usb_natural_audio_resolution")
    if (
        not isinstance(usb_record, dict)
        or not isinstance(overlap_record, dict)
        or not isinstance(resolution, dict)
        or resolution.get("complete") is not True
        or resolution.get("unresolved_entries") != []
        or resolution.get("exact_duplicate_excluded") != ["new_video.tar"]
    ):
        raise ValueError("Stage211 supplemental USB-wide resolution proof is incomplete.")
    usb_path = Path(str(usb_record.get("receipt_path") or "")).resolve()
    overlap_path = Path(str(overlap_record.get("receipt_path") or "")).resolve()
    if (
        not usb_path.is_file()
        or sha256_file(usb_path) != usb_record.get("receipt_sha256")
        or not overlap_path.is_file()
        or sha256_file(overlap_path) != overlap_record.get("receipt_sha256")
    ):
        raise ValueError("Stage211 supplemental USB-wide receipt binding changed.")
    usb = _load_json(usb_path, label="Stage211 USB top-level coverage receipt")
    overlap = _load_json(overlap_path, label="Stage211 archived-social overlap receipt")
    if (
        usb.get("artifact") != "usb_top_level_coverage"
        or usb.get("classification_complete") is not True
        or overlap.get("artifact") != "archived_social_overlap"
        or overlap.get("complete") is not True
        or overlap.get("all_members_exact_existing_social_duplicates") is not True
        or overlap.get("archive_excluded_from_training_as_duplicate") is not True
        or int(overlap.get("unique_members", -1)) != 0
        or int(overlap.get("archive_audio_members", -1)) <= 0
        or int(overlap.get("exact_duplicate_members", -1))
        != int(overlap.get("archive_audio_members", -2))
        or overlap_record.get("archive_sha256") != overlap.get("archive_sha256")
        or int(overlap_record.get("audio_members", -1))
        != int(overlap.get("archive_audio_members", -2))
        or int(overlap_record.get("unique_members", -1)) != 0
    ):
        raise ValueError("Stage211 archived-social all-member exclusion proof changed.")
    return {
        "inventory_schema_version": int(inventory.get("schema_version", -1)),
        "inventory_artifact": str(inventory.get("artifact") or ""),
        "mode": cross_pool["mode"],
        "source_sets_disjoint": True,
        "content_fingerprint_complete": False,
        "known_overlap_exclusions": sorted(exclusions),
        "supplemental_sources": sorted(str(source) for source in supplemental_sources),
        "stage179_manifest_path": str(stage179_manifest),
        "stage179_manifest_sha256": stage179_manifest_sha256,
        "stage179_unique_rows": stage179_rows,
        "stage179_hours": stage179_hours,
        "base_public_overlap_normalized_pcm_exact_complete": True,
        "base_public_overlap_scan_order": base_audit["scan_order"],
        "base_public_overlap_rows": 0,
        "base_public_overlap_scanned_rows": int(base_audit["scanned_rows"]),
        "base_public_overlap_receipt_path": str(base_audit_path),
        "base_public_overlap_receipt_sha256": sha256_file(base_audit_path),
        "social_normalized_pcm_exact_complete": bool(
            cross_pool.get("social_normalized_pcm_exact_complete", False)
        ),
        "social_public_overlap_mode": cross_pool.get("social_public_overlap_mode"),
        "archived_social_exact_duplicate_exclusion_complete": True,
        "archived_social_overlap_receipt_path": str(overlap_path),
        "archived_social_overlap_receipt_sha256": sha256_file(overlap_path),
        "archived_social_archive_sha256": overlap["archive_sha256"],
        "archived_social_audio_members": int(overlap["archive_audio_members"]),
        "archived_social_unique_members": 0,
        "usb_top_level_classification_complete": True,
        "usb_top_level_coverage_receipt_path": str(usb_path),
        "usb_top_level_coverage_receipt_sha256": sha256_file(usb_path),
        "usb_natural_audio_resolution_complete": True,
        "usb_unresolved_natural_entries": [],
        "near_duplicate_complete": cross_pool.get("near_duplicate_complete"),
        "component_inventories": components,
    }


def build_stepwise_report(
    *,
    initialization_receipt_path: Path,
    calibration_receipt_path: Path,
    mixer_gate_path: Path,
    block_gate_path: Path,
    logits_gate_path: Path,
    sft_final_report_path: Path,
    public_metric_correction_receipt_path: Path = DEFAULT_CORRECTION_RECEIPT,
) -> dict[str, Any]:
    initialization_receipt_path = initialization_receipt_path.expanduser().resolve()
    calibration_receipt_path = calibration_receipt_path.expanduser().resolve()
    mixer_gate_path = mixer_gate_path.expanduser().resolve()
    block_gate_path = block_gate_path.expanduser().resolve()
    logits_gate_path = logits_gate_path.expanduser().resolve()
    sft_final_report_path = sft_final_report_path.expanduser().resolve()
    public_metric_correction_receipt_path = (
        public_metric_correction_receipt_path.expanduser().resolve()
    )

    calibration, calibration_checkpoint = _validate_calibration_receipt(calibration_receipt_path)
    phase_reports: dict[str, dict[str, Any]] = {}
    phase_checkpoints: dict[str, Path] = {}
    for phase, path in (
        ("mixer", mixer_gate_path),
        ("block", block_gate_path),
        ("logits", logits_gate_path),
    ):
        raw = _load_json(path, label=f"Stage211 {phase} phase gate")
        checkpoint = Path(str(raw.get("checkpoint_path") or "")).resolve()
        phase_reports[phase] = validate_stage211_phase_gate_report(
            path,
            expected_phase=phase,
            checkpoint_path=checkpoint,
        )
        phase_checkpoints[phase] = checkpoint
    sft, sft_checkpoint, sft_correction_coverage = _validate_sft_report(sft_final_report_path)
    if Path(str(sft.get("mixer_phase_gate_path") or "")).resolve() != mixer_gate_path or sft.get(
        "mixer_phase_gate_sha256"
    ) != sha256_file(mixer_gate_path):
        raise ValueError("Stage211 SFT final report does not bind the selected Mixer gate.")
    if Path(str(sft.get("block_phase_gate_path") or "")).resolve() != block_gate_path or sft.get(
        "block_phase_gate_sha256"
    ) != sha256_file(block_gate_path):
        raise ValueError("Stage211 SFT final report does not bind the selected Block gate.")
    if Path(str(sft.get("logits_phase_gate_path") or "")).resolve() != logits_gate_path or sft.get(
        "logits_phase_gate_sha256"
    ) != sha256_file(logits_gate_path):
        raise ValueError("Stage211 SFT final report does not bind the promoted Logits gate.")
    teacher_sha256_by_stage = {
        phase: _phase_nano_teacher_sha256(phase_reports[phase])
        for phase in ("mixer", "block", "logits")
    }
    teacher_sha256_by_stage["sft"] = str(sft.get("nano_teacher_checkpoint_sha256") or "")
    unique_teacher_sha256 = set(teacher_sha256_by_stage.values())
    if len(unique_teacher_sha256) != 1 or len(next(iter(unique_teacher_sha256), "")) != 64:
        raise ValueError("Stage211 A/B/C/D Nano teacher checkpoint SHA-256 chain mismatch.")
    nano_teacher_checkpoint_sha256 = next(iter(unique_teacher_sha256))
    initialization = validate_stage211_initialization_receipt(
        initialization_receipt_path,
        expected_calibration_checkpoint=calibration_checkpoint,
        expected_nano_checkpoint_sha256=nano_teacher_checkpoint_sha256,
    )
    metric_correction = validate_completed_correction(
        public_metric_correction_receipt_path,
        tokenizer_source=DEFAULT_TOKENIZER_SOURCE,
        expected_calibration_reuse_receipt=calibration_receipt_path,
        expected_initialization_receipt=initialization_receipt_path,
    )
    if metric_correction is None:
        raise ValueError("Stage211 Unicode metric-correction receipt is missing.")
    global_dedup_bindings = {
        (
            str(phase_reports[phase].get("global_dedup_manifest_path") or ""),
            str(phase_reports[phase].get("global_dedup_manifest_sha256") or ""),
        )
        for phase in ("mixer", "block", "logits")
    }
    if len(global_dedup_bindings) != 1:
        raise ValueError("Stage211 A/B/C global dedup provenance chain mismatch.")
    global_dedup_manifest_path, global_dedup_manifest_sha256 = next(iter(global_dedup_bindings))
    loaded_manifest_bindings = {
        (
            str(phase_reports[phase].get("loaded_manifest_receipt_path") or ""),
            str(phase_reports[phase].get("loaded_manifest_receipt_sha256") or ""),
        )
        for phase in ("mixer", "block", "logits")
    }
    if len(loaded_manifest_bindings) != 1:
        raise ValueError("Stage211 A/B/C loaded-manifest provenance chain mismatch.")
    loaded_manifest_receipt_path, loaded_manifest_receipt_sha256 = next(
        iter(loaded_manifest_bindings)
    )
    supplemental_inventory_bindings = {
        (
            str(
                phase_reports[phase]
                .get("full_data_coverage", {})
                .get("supplemental_natural", {})
                .get("supplemental_inventory_path", "")
            ),
            str(
                phase_reports[phase]
                .get("full_data_coverage", {})
                .get("supplemental_natural", {})
                .get("supplemental_inventory_sha256", "")
            ),
        )
        for phase in ("mixer", "block", "logits")
    }
    if len(supplemental_inventory_bindings) != 1:
        raise ValueError("Stage211 A/B/C supplemental inventory provenance chain mismatch.")
    supplemental_inventory_path, supplemental_inventory_sha256 = next(
        iter(supplemental_inventory_bindings)
    )
    supplemental_inventory = Path(supplemental_inventory_path).expanduser().resolve()
    if (
        not supplemental_inventory.is_file()
        or sha256_file(supplemental_inventory) != supplemental_inventory_sha256
    ):
        raise ValueError("Stage211 A/B/C supplemental inventory binding is invalid.")
    supplemental_dedupe_proof = _supplemental_dedupe_proof(supplemental_inventory)
    baseline_bindings = {
        _nano_public_baseline_binding(phase_reports[phase])
        for phase in ("mixer", "block", "logits")
    }
    baseline_bindings.add(_nano_public_baseline_binding(sft))
    if len(baseline_bindings) != 1:
        raise ValueError("Stage211 A/B/C/D Nano public-baseline provenance chain mismatch.")
    (
        nano_public_baseline_receipt_path,
        nano_public_baseline_receipt_sha256,
        nano_public_baseline_checkpoint_sha256,
    ) = next(iter(baseline_bindings))
    baseline_receipt = validate_stage211_nano_public_baseline_receipt(
        nano_public_baseline_receipt_path,
        expected_receipt_sha256=nano_public_baseline_receipt_sha256,
        expected_nano_checkpoint_sha256=nano_teacher_checkpoint_sha256,
    )
    if nano_public_baseline_checkpoint_sha256 != baseline_receipt["nano_checkpoint_sha256"]:
        raise ValueError("Stage211 Nano public-baseline checkpoint binding mismatch.")
    overlap_bindings = {
        (
            str(report.get("public_overlap", {}).get("receipt_path") or ""),
            str(report.get("public_overlap", {}).get("receipt_sha256") or ""),
        )
        for report in (
            calibration,
            phase_reports["mixer"],
            phase_reports["block"],
            phase_reports["logits"],
            sft,
            baseline_receipt,
        )
    }
    if len(overlap_bindings) != 1 or not next(iter(overlap_bindings))[0]:
        raise ValueError("Stage211 Calibration/A/B/C/D public-overlap chain mismatch.")
    public_overlap_receipt_path, public_overlap_receipt_sha256 = next(iter(overlap_bindings))

    checkpoints = {
        "calibration": calibration_checkpoint,
        **phase_checkpoints,
        "sft": sft_checkpoint,
    }
    checkpoint_hashes = {
        stage: sha256_file(checkpoint) for stage, checkpoint in checkpoints.items()
    }
    chain: list[dict[str, Any]] = []
    previous_stage = "calibration"
    for stage in ("mixer", "block", "logits"):
        init_path, init_sha256 = _phase_initial_checkpoint(phase_reports[stage])
        chain.append(
            _require_chain_link(
                source_stage=previous_stage,
                source_checkpoint=checkpoints[previous_stage],
                source_sha256=checkpoint_hashes[previous_stage],
                target_stage=stage,
                target_init=init_path,
                target_init_sha256=init_sha256,
            )
        )
        previous_stage = stage
    sft_init, sft_init_sha256 = _sft_initial_checkpoint(sft)
    chain.append(
        _require_chain_link(
            source_stage="logits",
            source_checkpoint=checkpoints["logits"],
            source_sha256=checkpoint_hashes["logits"],
            target_stage="sft",
            target_init=sft_init,
            target_init_sha256=sft_init_sha256,
        )
    )

    benchmarks = {
        "calibration": calibration["public_benchmark"],
        **{
            phase: phase_reports[phase]["public_benchmark"]
            for phase in ("mixer", "block", "logits")
        },
        "sft": sft["public_benchmark"],
    }
    rows_by_stage = {stage: _benchmark_rows(benchmark) for stage, benchmark in benchmarks.items()}
    dataset_results: list[dict[str, Any]] = []
    calibration_rows = rows_by_stage["calibration"]
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        reference = calibration_rows[dataset]
        stage_values: dict[str, Any] = {}
        for stage in STAGE_ORDER:
            result = rows_by_stage[stage][dataset]
            for hash_key in ("manifest_sha256", "nano_prediction_sha256"):
                if result.get(hash_key) != reference.get(hash_key):
                    raise ValueError(f"Stage211 {dataset} {hash_key} differs at stage {stage}.")
            if float(result["nano_error_rate"]) != float(reference["nano_error_rate"]):
                raise ValueError(f"Stage211 {dataset} Nano metric differs at stage {stage}.")
            stage_values[stage] = {
                "student_error_rate": float(result["student_error_rate"]),
                "student_deletion_rate": float(result["student_deletion_rate"]),
                "student_prediction_reference_unit_ratio": float(
                    result["student_prediction_reference_unit_ratio"]
                ),
                "absolute_gap_points": float(result["absolute_gap_points"]),
                "relative_ratio": float(result["relative_ratio"]),
            }
        dataset_results.append(
            {
                "dataset": dataset,
                "metric": expected["metric"],
                "language": expected["language"],
                "sample_count": int(expected["samples"]),
                "nano_error_rate": float(reference["nano_error_rate"]),
                "stages": stage_values,
            }
        )
    english_wer_datasets = [
        row["dataset"]
        for row in dataset_results
        if row["language"] == "en" and row["metric"] == "wer"
    ]
    chinese_cer_datasets = [
        row["dataset"]
        for row in dataset_results
        if row["language"] == "zh" and row["metric"] == "cer"
    ]
    if len(english_wer_datasets) != 3 or len(chinese_cer_datasets) != 2:
        raise ValueError("Stage211 final report lacks the required English WER/Chinese CER split.")
    if any(tuple(row["stages"]) != STAGE_ORDER for row in dataset_results):
        raise ValueError("Stage211 final report lacks a complete ordered public-metric stage map.")
    language_metric_summaries = _language_metric_summaries(dataset_results)

    stage_records = [
        _stage_record(
            stage="calibration",
            checkpoint=calibration_checkpoint,
            source_report=calibration_receipt_path,
            benchmark=benchmarks["calibration"],
            data_coverage=None,
            gate_passed=None,
            nano_teacher_checkpoint_sha256=None,
            preflight_smoke=None,
        ),
        *[
            _stage_record(
                stage=phase,
                checkpoint=phase_checkpoints[phase],
                source_report={
                    "mixer": mixer_gate_path,
                    "block": block_gate_path,
                    "logits": logits_gate_path,
                }[phase],
                benchmark=benchmarks[phase],
                data_coverage=phase_reports[phase]["full_data_coverage"],
                gate_passed=True,
                nano_teacher_checkpoint_sha256=teacher_sha256_by_stage[phase],
                preflight_smoke=phase_reports[phase]["preflight_smoke"],
            )
            for phase in ("mixer", "block", "logits")
        ],
        _stage_record(
            stage="sft",
            checkpoint=sft_checkpoint,
            source_report=sft_final_report_path,
            benchmark=benchmarks["sft"],
            data_coverage=sft["labeled_data_coverage"],
            gate_passed=True,
            nano_teacher_checkpoint_sha256=teacher_sha256_by_stage["sft"],
            preflight_smoke=None,
        ),
    ]
    (
        initial_calibration_result,
        requested_alignment_results,
        requested_alignment_language_metric_summaries,
        requested_alignment_dataset_results,
    ) = _requested_alignment_views(
        dataset_results=dataset_results,
        language_metric_summaries=language_metric_summaries,
        stage_records=stage_records,
    )
    coverage_results = [
        _coverage_record(
            stage=stage,
            coverage=(
                phase_reports[stage]["full_data_coverage"]
                if stage in {"mixer", "block", "logits"}
                else sft["labeled_data_coverage"]
            ),
            correction_round_promotion=(
                phase_reports[stage]["correction_round_promotion"]
                if stage in {"mixer", "block", "logits"}
                else None
            ),
            sft_correction_coverage=(sft_correction_coverage if stage == "sft" else None),
        )
        for stage in ("mixer", "block", "logits", "sft")
    ]
    alignment_results = [
        _alignment_result(phase=phase, phase_report=phase_reports[phase])
        for phase in ("mixer", "block", "logits")
    ]
    ctc_label_proof = _sft_ctc_label_proof(sft["labeled_data_coverage"])
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stepwise_final_results",
        "complete": True,
        "gate_passed": True,
        "strict_stage_order": list(STAGE_ORDER),
        "requested_alignment_stage_order": list(REQUESTED_ALIGNMENT_STAGE_ORDER),
        "requested_to_internal_stage": dict(REQUESTED_TO_INTERNAL_STAGE),
        "all_requested_alignment_metrics_complete": True,
        "public_metric_normalization": "ctc",
        "initial_calibration_result": initial_calibration_result,
        "requested_alignment_results": requested_alignment_results,
        "requested_alignment_language_metric_summaries": (
            requested_alignment_language_metric_summaries
        ),
        "requested_alignment_dataset_results": requested_alignment_dataset_results,
        "checkpoint_chain_passed": True,
        "nano_initialization_chain_passed": True,
        "nano_initialization_source_chain_passed": initialization["loader_source_chain_passed"],
        "ctc_label_normalization_chain_passed": True,
        "ctc_label_proof": ctc_label_proof,
        "sft_correction_evidence": sft_correction_coverage,
        "public_metric_definition_chain_passed": True,
        "public_metric_strip_language_confirmation": (STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION),
        "public_metric_correction_receipt_path": str(public_metric_correction_receipt_path),
        "public_metric_correction_receipt_sha256": sha256_file(
            public_metric_correction_receipt_path
        ),
        "public_metric_tokenizer_contract": metric_correction["tokenizer_contract"],
        "public_metric_tokenizer_source_sha256": metric_correction["tokenizer_source_sha256"],
        "initialization_receipt_path": str(initialization_receipt_path),
        "initialization_receipt_sha256": sha256_file(initialization_receipt_path),
        "initialization_proof": {
            "stage210_checkpoint_path": initialization["stage210_checkpoint_path"],
            "stage210_checkpoint_sha256": initialization["stage210_checkpoint_sha256"],
            "runtime_load_report": initialization["runtime_load_report"],
            "freeze_report": initialization["freeze_report"],
            "frozen_tensor_audit": initialization["frozen_tensor_audit"],
            "loader_source_validation": initialization["loader_source_validation"],
        },
        "nano_teacher_chain_passed": True,
        "nano_teacher_checkpoint_sha256": nano_teacher_checkpoint_sha256,
        "nano_public_baseline_provenance_passed": True,
        "nano_public_baseline_receipt_path": nano_public_baseline_receipt_path,
        "nano_public_baseline_receipt_sha256": (nano_public_baseline_receipt_sha256),
        "nano_public_baseline_checkpoint_sha256": (nano_public_baseline_checkpoint_sha256),
        "public_overlap_chain_passed": True,
        "public_overlap_receipt_path": public_overlap_receipt_path,
        "public_overlap_receipt_sha256": public_overlap_receipt_sha256,
        "global_dedup_manifest_path": global_dedup_manifest_path,
        "global_dedup_manifest_sha256": global_dedup_manifest_sha256,
        "loaded_manifest_receipt_path": loaded_manifest_receipt_path,
        "loaded_manifest_receipt_sha256": loaded_manifest_receipt_sha256,
        "supplemental_inventory_chain_passed": True,
        "supplemental_inventory_path": str(supplemental_inventory),
        "supplemental_inventory_sha256": supplemental_inventory_sha256,
        "supplemental_dedupe_proof": supplemental_dedupe_proof,
        "total_public_eval_samples_per_stage": sum(
            int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values()
        ),
        "public_metric_stage_order": list(STAGE_ORDER),
        "all_stage_public_metrics_complete": True,
        "all_stage_alignment_results_complete": True,
        "english_wer_datasets": english_wer_datasets,
        "chinese_cer_datasets": chinese_cer_datasets,
        "language_metric_summaries": language_metric_summaries,
        "checkpoint_chain": chain,
        "stages": stage_records,
        "coverage_results": coverage_results,
        "alignment_results": alignment_results,
        "dataset_results": dataset_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    label_proof = report["ctc_label_proof"]
    if label_proof.get("public_audio_isolation_passed") is True:
        public_audio_isolation = (
            "SFT/public exact encoded-audio isolation: `true`, "
            f"scanned `{int(label_proof['public_audio_scanned_rows']):,}` labeled rows "
            f"against `{int(label_proof['public_audio_public_rows']):,}` public rows, "
            "training overlap: `0` (normalized-PCM/near-duplicate completeness: "
            "`false/false`)"
        )
    else:
        public_audio_isolation = (
            "SFT/public exact encoded-audio isolation: `unavailable` "
            "(legacy labeled-profile evidence)"
        )
    lines = [
        "# Stage211 Stepwise Alignment Results",
        "",
        "Initial baseline: Calibration",
        "",
        "Requested alignment order: RWKV Layer -> Block -> Logits -> Labeled CTC SFT",
        "",
        f"Nano teacher SHA-256: `{report['nano_teacher_checkpoint_sha256']}`",
        "",
        f"Nano initialization proof: `{report['initialization_receipt_sha256']}`",
        "",
        f"Nano public baseline provenance: `{report['nano_public_baseline_receipt_sha256']}`",
        "",
        "Public metric definition proof: "
        f"`{report['public_metric_correction_receipt_sha256']}` "
        f"(`{report['public_metric_tokenizer_contract']}`, tokenizer "
        f"`{report['public_metric_tokenizer_source_sha256']}`, AR language-prefix "
        "stripping: `false`)",
        "",
        f"Supplemental inventory: `{report['supplemental_inventory_sha256']}`",
        "",
        "Supplemental cross-pool dedupe: "
        f"`{report['supplemental_dedupe_proof']['mode']}`, "
        "source sets disjoint: `true`, content fingerprint complete: `false`, "
        "known exclusions: `llaso_gigaspeech,llaso_librispeech`",
        "",
        "Supplemental components: "
        f"schema `{report['supplemental_dedupe_proof']['inventory_schema_version']}`, "
        "base public normalized-PCM exact audit: "
        f"`{str(report['supplemental_dedupe_proof']['base_public_overlap_normalized_pcm_exact_complete']).lower()}` "
        f"({int(report['supplemental_dedupe_proof']['base_public_overlap_scanned_rows']):,} rows, 0 overlap, "
        f"`{report['supplemental_dedupe_proof']['base_public_overlap_scan_order']}`), "
        "social normalized-PCM exact dedupe/public filtering: "
        f"`{str(report['supplemental_dedupe_proof']['social_normalized_pcm_exact_complete']).lower()}`/"
        f"`{report['supplemental_dedupe_proof']['social_public_overlap_mode']}`, "
        "acoustic near-duplicate coverage: `false`",
        "",
        "USB-wide natural-audio resolution: `true`, archived social exact duplicate "
        f"exclusion: `{str(report['supplemental_dedupe_proof']['archived_social_exact_duplicate_exclusion_complete']).lower()}` "
        f"({int(report['supplemental_dedupe_proof']['archived_social_audio_members'])} members), "
        "unresolved entries: `0`",
        "",
        "CTC label normalization: `ctc`, tokenizer: `sensevoice_tiktoken`, "
        f"unknown tokens: `{int(report['ctc_label_proof']['ctc_unk_tokens'])}`, "
        "non-pronunciation logits suppressed: `true`",
        "",
        public_audio_isolation,
        "",
        "Public WER/CER normalization: `ctc`.",
        "",
        "## Requested Alignment Results",
        "",
        "| Stage | Internal phase | Objective | English WER | Nano WER | WER gap | "
        "Chinese CER | Nano CER | CER gap | Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["requested_alignment_results"]:
        lines.append(
            f"| `{result['stage']}` | `{result['internal_stage']}` | "
            f"`{result['objective']}` | {float(result['english_wer']) * 100.0:.3f}% | "
            f"{float(result['nano_english_wer']) * 100.0:.3f}% | "
            f"{float(result['english_wer_gap_to_nano']) * 100.0:+.3f} pt | "
            f"{float(result['chinese_cer']) * 100.0:.3f}% | "
            f"{float(result['nano_chinese_cer']) * 100.0:.3f}% | "
            f"{float(result['chinese_cer_gap_to_nano']) * 100.0:+.3f} pt | "
            f"`{str(result['gate_passed']).lower()}` |"
        )
    lines.extend(
        (
            "",
            "## Incremental Stage Impact",
            "",
            "Negative error-rate delta and positive Nano-gap reduction indicate improvement.",
            "",
            "| Stage | Previous stage | English WER delta | English Nano-gap reduction | "
            "Chinese CER delta | Chinese Nano-gap reduction |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for result in report["requested_alignment_results"]:
        lines.append(
            f"| `{result['stage']}` | `{result['previous_stage']}` | "
            f"{float(result['english_wer_delta_from_previous']) * 100.0:+.3f} pt | "
            f"{float(result['english_wer_gap_reduction_from_previous']) * 100.0:+.3f} pt | "
            f"{float(result['chinese_cer_delta_from_previous']) * 100.0:+.3f} pt | "
            f"{float(result['chinese_cer_gap_reduction_from_previous']) * 100.0:+.3f} pt |"
        )
    lines.extend(
        (
            "",
            "## Language Macro Metrics",
            "",
            "Aggregation: `unweighted_dataset_macro`.",
            "",
            "| Language | Metric | Datasets | Samples | Nano | Calibration | Layer A | "
            "Block B | Logits C | SFT D |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for summary in report["language_metric_summaries"]:
        stages = summary["stages"]
        lines.append(
            f"| {str(summary['language']).upper()} | {str(summary['metric']).upper()} | "
            f"{int(summary['dataset_count'])} | {int(summary['sample_count']):,} | "
            f"{float(summary['nano_error_rate']) * 100.0:.3f}% | "
            f"{float(stages['calibration']) * 100.0:.3f}% | "
            f"{float(stages['mixer']) * 100.0:.3f}% | "
            f"{float(stages['block']) * 100.0:.3f}% | "
            f"{float(stages['logits']) * 100.0:.3f}% | "
            f"{float(stages['sft']) * 100.0:.3f}% |"
        )
    lines.extend(
        (
            "",
            "## Per-Dataset Metrics",
            "",
            "| Dataset | Metric | Samples | Nano | Calibration | Layer A | Block B | "
            "Logits C | SFT D | Final gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in report["dataset_results"]:
        stages = row["stages"]
        nano = float(row["nano_error_rate"]) * 100.0
        final = float(stages["sft"]["student_error_rate"]) * 100.0
        lines.append(
            f"| {row['dataset']} | {str(row['metric']).upper()} | "
            f"{int(row['sample_count']):,} | {nano:.3f}% | "
            f"{float(stages['calibration']['student_error_rate']) * 100.0:.3f}% | "
            f"{float(stages['mixer']['student_error_rate']) * 100.0:.3f}% | "
            f"{float(stages['block']['student_error_rate']) * 100.0:.3f}% | "
            f"{float(stages['logits']['student_error_rate']) * 100.0:.3f}% | "
            f"{final:.3f}% | {final - nano:+.3f} pt |"
        )
    lines.extend(
        (
            "",
            "## Hidden Alignment Metrics",
            "",
            "| Stage | Scope | Component | Baseline loss | Candidate loss | "
            "Baseline cosine | Candidate cosine | Loss improved layers | "
            "Cosine improved layers |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for alignment in report["alignment_results"]:
        if alignment["stage"] not in {"mixer", "block"}:
            continue
        for scope in ("fixed", "stratified"):
            for component_name, component in alignment[scope]["components"].items():
                lines.append(
                    f"| {alignment['label']} | {scope} | `{component_name}` | "
                    f"{float(component['baseline_mean_loss']):.6f} | "
                    f"{float(component['candidate_mean_loss']):.6f} | "
                    f"{float(component['baseline_mean_cosine']):.6f} | "
                    f"{float(component['candidate_mean_cosine']):.6f} | "
                    f"{int(component['loss_improved_layers'])}/70 | "
                    f"{int(component['cosine_improved_layers'])}/70 |"
                )
    lines.extend(
        (
            "",
            "## Nine-Cell Hidden Alignment",
            "",
            "| Stage | Cell | Samples | Baseline loss | Candidate loss | "
            "Loss improved layers | Cosine improved layers |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for alignment in report["alignment_results"]:
        if alignment["stage"] not in {"mixer", "block"}:
            continue
        for cell_name, cell in alignment["stratified"]["cells"].items():
            lines.append(
                f"| {alignment['label']} | `{cell_name}` | {int(cell['samples'])} | "
                f"{float(cell['baseline_loss']):.6f} | "
                f"{float(cell['candidate_loss']):.6f} | "
                f"{int(cell['layers_loss_improved'])}/70 | "
                f"{int(cell['layers_cosine_improved'])}/70 |"
            )
    logits_alignment = next(
        alignment for alignment in report["alignment_results"] if alignment["stage"] == "logits"
    )
    lines.extend(
        (
            "",
            "## Logits Alignment Metrics",
            "",
            "| Scope | Metric | Baseline | Candidate |",
            "|---|---|---:|---:|",
        )
    )
    for scope in ("fixed", "stratified"):
        for metric, values in logits_alignment[scope]["metrics"].items():
            lines.append(
                f"| {scope} | `{metric}` | {float(values['baseline']):.6f} | "
                f"{float(values['candidate']):.6f} |"
            )
    lines.extend(
        (
            "",
            "## Nine-Cell Logits Alignment",
            "",
            "| Cell | Samples | Full KL baseline | Full KL candidate | "
            "Conditional KL baseline | Conditional KL candidate | "
            "CTC token error baseline | CTC token error candidate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for cell_name, cell in logits_alignment["stratified"]["cells"].items():
        metrics = cell["metrics"]
        lines.append(
            f"| `{cell_name}` | {int(cell['samples'])} | "
            f"{float(metrics['full_kl']['baseline']):.6f} | "
            f"{float(metrics['full_kl']['candidate']):.6f} | "
            f"{float(metrics['conditional_nonblank_kl']['baseline']):.6f} | "
            f"{float(metrics['conditional_nonblank_kl']['candidate']):.6f} | "
            f"{float(metrics['ctc_token_error_rate']['baseline']):.6f} | "
            f"{float(metrics['ctc_token_error_rate']['candidate']):.6f} |"
        )
    lines.extend(
        (
            "",
            "## Logits Hidden Retention",
            "",
            "| Scope | Component | Baseline loss | Candidate loss | Baseline cosine | "
            "Candidate cosine |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for scope in ("fixed", "stratified"):
        for component_name, component in logits_alignment[scope]["hidden_components"].items():
            lines.append(
                f"| {scope} | `{component_name}` | "
                f"{float(component['baseline_mean_loss']):.6f} | "
                f"{float(component['candidate_mean_loss']):.6f} | "
                f"{float(component['baseline_mean_cosine']):.6f} | "
                f"{float(component['candidate_mean_cosine']):.6f} |"
            )
    lines.extend(
        (
            "",
            "## Decoder Hidden Alignment",
            "",
            "| Stage | Scope | Baseline loss | Candidate loss | Relative change |",
            "|---|---|---:|---:|---:|",
        )
    )
    for alignment in report["alignment_results"]:
        for scope in ("fixed", "stratified"):
            decoder = alignment[scope].get("decoder_hidden")
            if decoder is None:
                continue
            lines.append(
                f"| {alignment['label']} | {scope} | "
                f"{float(decoder['baseline_loss']):.6f} | "
                f"{float(decoder['candidate_loss']):.6f} | "
                f"{float(decoder['relative_change']) * 100.0:+.3f}% |"
            )
    lines.extend(
        (
            "",
            "## Alignment Evidence",
            "",
            "| Stage | Fixed samples | Stratified samples | Cells | Report SHA-256 | Gate |",
            "|---|---:|---:|---:|---|---:|",
        )
    )
    for alignment in report["alignment_results"]:
        lines.append(
            f"| {alignment['label']} | {int(alignment['fixed_eval_samples'])} | "
            f"{int(alignment['stratified_samples']):,} | "
            f"{len(alignment['stratified_cells'])} | "
            f"`{alignment['source_report_sha256']}` | pass |"
        )
    lines.extend(
        (
            "",
            "## Intra-Phase Retention",
            "",
            "| Stage | Terminal reports | Best prior segment | Best loss | "
            "Candidate segment | Candidate loss | Regression | Limit | Gate |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|",
        )
    )
    for alignment in report["alignment_results"]:
        trajectory = alignment["trajectory_retention"]
        lines.append(
            f"| {alignment['label']} | {int(trajectory['terminal_entries'])} | "
            f"`{trajectory['best_prior_source']}` | "
            f"{float(trajectory['best_prior_loss']):.6f} | "
            f"`{trajectory['candidate_source']}` | "
            f"{float(trajectory['candidate_loss']):.6f} | "
            f"{float(trajectory['relative_regression_pct']):+.3f}% | "
            f"{float(trajectory['max_relative_regression_pct']):.3f}% | pass |"
        )
    lines.extend(
        (
            "",
            "## Periodic Fixed Evaluation",
            "",
            "| Stage | Source | Terminal step | Interval | Reports | Samples/report | Gate |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for alignment in report["alignment_results"]:
        cadence = alignment["step_eval_cadence"]
        for source in cadence["sources"]:
            lines.append(
                f"| {alignment['label']} | `{source['source']}` | "
                f"{int(source['terminal_step']):,} | "
                f"{int(cadence['interval_steps']):,} | "
                f"{int(source['report_count']):,} | "
                f"{int(source['eval_samples_per_report'])} | pass |"
            )
    sft_coverage = next(
        coverage for coverage in report["coverage_results"] if coverage["stage"] == "sft"
    )
    sft_cadence = sft_coverage["step_eval_cadence"]
    for source in sft_cadence["sources"]:
        lines.append(
            f"| {sft_coverage['label']} | `{source['source']}` | "
            f"{int(source['terminal_step']):,} | "
            f"{int(sft_cadence['interval_steps']):,} | "
            f"{int(source['report_count']):,} | "
            f"{int(source['eval_samples_per_report'])} | pass |"
        )
    lines.extend(
        (
            "",
            "## Training Coverage",
            "",
            "| Stage | Objective | Train rows | Eval rows | Hours/epoch | "
            "Epochs | Executed exposures | Post-coverage correction | Promotion eligibility |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        )
    )
    for coverage in report["coverage_results"]:
        correction = (
            "none"
            if int(coverage["correction_rounds"]) == 0
            else (
                f"{int(coverage['correction_rounds'])} round(s), "
                f"{int(coverage['correction_row_exposures']):,} rows / "
                f"{float(coverage['correction_hour_exposures']):,.3f} h / "
                f"{int(coverage['correction_steps']):,} steps / "
                f"{int(coverage['correction_executed_sample_exposures']):,} "
                "executed"
            )
        )
        correction_policy = coverage.get("correction_round_promotion")
        if isinstance(correction_policy, dict):
            promotion_eligibility = (
                "not entered; eligible"
                if correction_policy["correction_started"] is not True
                else (
                    f"{int(correction_policy['completed_rounds'])}/"
                    f"{int(correction_policy['guaranteed_rounds'])} rounds; eligible"
                )
            )
        else:
            promotion_eligibility = "n/a"
        lines.append(
            f"| {coverage['label']} | `{coverage['objective']}` | "
            f"{int(coverage['unique_or_train_rows']):,} | "
            f"{int(coverage['evaluation_rows']):,} | "
            f"{float(coverage['hours_per_epoch']):,.3f} | "
            f"{int(coverage['epochs'])} | "
            f"{int(coverage['executed_sample_exposures']):,} | {correction} |"
            f" {promotion_eligibility} |"
        )
    correction_evidence = report["sft_correction_evidence"]
    lines.extend(("", "## SFT Correction Proof", ""))
    if correction_evidence["applied"] is not True:
        lines.append("No correction round was applied; all correction exposures are zero.")
    else:
        lines.extend(
            (
                "| Round | Rows | Hours | Steps | Tail padding | Executed exposures | "
                "Receipt SHA-256 |",
                "|---:|---:|---:|---:|---:|---:|---|",
            )
        )
        for receipt in correction_evidence["round_receipts"]:
            lines.append(
                f"| {int(receipt['round'])} | "
                f"{int(receipt['row_exposures']):,} | "
                f"{float(receipt['hour_exposures']):,.3f} | "
                f"{int(receipt['steps']):,} | "
                f"{int(receipt['tail_padding_sample_exposures']):,} | "
                f"{int(receipt['executed_sample_exposures']):,} | "
                f"`{receipt['receipt_sha256']}` |"
            )
    lines.extend(
        (
            "",
            "## Full Data Segment Proof",
            "",
            "| Stage | Segment | Rows | Hours/epoch | Epochs | Steps | Row exposures |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for coverage in report["coverage_results"]:
        for segment in coverage.get("training_segments", []):
            lines.append(
                f"| {coverage['label']} | `{segment['difficulty']}` | "
                f"{int(segment['rows']):,} | {float(segment['hours']):,.3f} | "
                f"{int(segment['epochs'])} | {int(segment['steps']):,} | "
                f"{int(segment['row_exposures']):,} |"
            )
    lines.extend(
        (
            "",
            "## Bound Checkpoints",
            "",
            "| Stage | Checkpoint | SHA-256 | Gate |",
            "|---|---|---|---:|",
        )
    )
    for stage in report["stages"]:
        lines.append(
            f"| {stage['label']} | `{stage['checkpoint_path']}` | "
            f"`{stage['checkpoint_sha256']}` | {stage['gate_status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_immutable(path: Path, text: str, *, label: str) -> None:
    write_immutable_text(path, text, label=label)


def create_stepwise_report(
    *,
    initialization_receipt_path: Path,
    calibration_receipt_path: Path,
    mixer_gate_path: Path,
    block_gate_path: Path,
    logits_gate_path: Path,
    sft_final_report_path: Path,
    output_json: Path,
    output_markdown: Path,
    public_metric_correction_receipt_path: Path = DEFAULT_CORRECTION_RECEIPT,
) -> dict[str, Any]:
    report = build_stepwise_report(
        initialization_receipt_path=initialization_receipt_path,
        calibration_receipt_path=calibration_receipt_path,
        mixer_gate_path=mixer_gate_path,
        block_gate_path=block_gate_path,
        logits_gate_path=logits_gate_path,
        sft_final_report_path=sft_final_report_path,
        public_metric_correction_receipt_path=public_metric_correction_receipt_path,
    )
    rendered_json = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    rendered_markdown = render_markdown(report)
    _write_immutable(output_json, rendered_json, label="Stage211 stepwise JSON report")
    _write_immutable(
        output_markdown,
        rendered_markdown,
        label="Stage211 stepwise Markdown report",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Create the immutable calibration/A/B/C/D Stage211 WER/CER result report.")
    )
    parser.add_argument(
        "--initialization-receipt",
        type=Path,
        default=DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
    )
    parser.add_argument(
        "--calibration-reuse-receipt",
        type=Path,
        default=DEFAULT_CALIBRATION_RECEIPT,
    )
    parser.add_argument(
        "--public-metric-correction-receipt",
        type=Path,
        default=DEFAULT_CORRECTION_RECEIPT,
    )
    parser.add_argument(
        "--mixer-phase-gate",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--block-phase-gate",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--logits-phase-gate",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--sft-final-report",
        type=Path,
        default=DEFAULT_PHASE_GATE_ROOT / "sft" / "stage211_complete.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_PHASE_GATE_ROOT / "sft" / "stage211_stepwise_results.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_PHASE_GATE_ROOT / "sft" / "stage211_stepwise_results.md",
    )
    args = parser.parse_args()

    mixer_phase_gate = _resolve_cli_mixer_gate(
        requested_gate=args.mixer_phase_gate,
        sft_final_report_path=args.sft_final_report,
    )
    block_phase_gate = _resolve_cli_phase_gate(
        phase="block",
        requested_gate=args.block_phase_gate,
        sft_final_report_path=args.sft_final_report,
    )
    logits_phase_gate = _resolve_cli_phase_gate(
        phase="logits",
        requested_gate=args.logits_phase_gate,
        sft_final_report_path=args.sft_final_report,
    )
    report = create_stepwise_report(
        initialization_receipt_path=args.initialization_receipt,
        calibration_receipt_path=args.calibration_reuse_receipt,
        mixer_gate_path=mixer_phase_gate,
        block_gate_path=block_phase_gate,
        logits_gate_path=logits_phase_gate,
        sft_final_report_path=args.sft_final_report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        public_metric_correction_receipt_path=(args.public_metric_correction_receipt),
    )
    print(
        f"stepwise_report={args.output_json.resolve()} "
        f"stages={len(report['stages'])} datasets={len(report['dataset_results'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
