from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_phase_gate_report,
    validate_stage211_public_benchmark,
)


STAGE_ORDER = ("calibration", "mixer", "block", "logits", "sft")
STAGE_LABELS = {
    "calibration": "Calibration",
    "mixer": "Layer / Mixer (A)",
    "block": "Block (B)",
    "logits": "Logits (C)",
    "sft": "Labeled CTC SFT (D)",
}
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
    validate_stage211_public_benchmark(receipt.get("public_benchmark"))
    return receipt, checkpoint


def _validate_sft_report(path: Path) -> tuple[dict[str, Any], Path]:
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
    progress = report.get("public_progress")
    if (
        not isinstance(progress, dict)
        or progress.get("gate_passed") is not True
        or progress.get("no_dataset_regression") is not True
        or progress.get("macro_improved") is not True
        or int(progress.get("improved_datasets", 0)) <= 0
    ):
        raise ValueError("Stage211 SFT public-progress gate did not pass.")
    coverage = report.get("labeled_data_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("phase") != "sft"
        or coverage.get("complete") is not True
    ):
        raise ValueError("Stage211 SFT labeled-data coverage is incomplete.")
    benchmark = validate_stage211_public_benchmark(report.get("public_benchmark"))
    if benchmark.get("all_datasets_pass") is not True:
        raise ValueError("Stage211 SFT did not pass the every-dataset Nano WER/CER gate.")
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
    return report, checkpoint


def _benchmark_rows(benchmark: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validated = validate_stage211_public_benchmark(benchmark)
    return {
        str(result["dataset"]): dict(result)
        for result in validated["results"]
        if isinstance(result, dict)
    }


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


def _sft_initial_checkpoint(report: dict[str, Any]) -> tuple[Path, str]:
    coverage = report["labeled_data_coverage"]
    return (
        Path(str(coverage.get("init_checkpoint_path") or "")).resolve(),
        str(coverage.get("init_checkpoint_sha256") or ""),
    )


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
        "data_coverage": data_coverage,
        "public_benchmark": benchmark,
    }


def build_stepwise_report(
    *,
    calibration_receipt_path: Path,
    mixer_gate_path: Path,
    block_gate_path: Path,
    logits_gate_path: Path,
    sft_final_report_path: Path,
) -> dict[str, Any]:
    calibration_receipt_path = calibration_receipt_path.expanduser().resolve()
    mixer_gate_path = mixer_gate_path.expanduser().resolve()
    block_gate_path = block_gate_path.expanduser().resolve()
    logits_gate_path = logits_gate_path.expanduser().resolve()
    sft_final_report_path = sft_final_report_path.expanduser().resolve()

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
    sft, sft_checkpoint = _validate_sft_report(sft_final_report_path)

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

    stage_records = [
        _stage_record(
            stage="calibration",
            checkpoint=calibration_checkpoint,
            source_report=calibration_receipt_path,
            benchmark=benchmarks["calibration"],
            data_coverage=None,
            gate_passed=None,
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
        ),
    ]
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stepwise_final_results",
        "complete": True,
        "gate_passed": True,
        "strict_stage_order": list(STAGE_ORDER),
        "checkpoint_chain_passed": True,
        "total_public_eval_samples_per_stage": sum(
            int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values()
        ),
        "checkpoint_chain": chain,
        "stages": stage_records,
        "dataset_results": dataset_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage211 Stepwise Alignment Results",
        "",
        "Strict order: Calibration -> Layer/Mixer (A) -> Block (B) -> "
        "Logits (C) -> Labeled CTC SFT (D)",
        "",
        "| Dataset | Metric | Samples | Nano | Calibration | Layer A | Block B | "
        "Logits C | SFT D | Final gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
    path = path.expanduser().resolve()
    if path.is_file() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Refusing to overwrite a different {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_stepwise_report(
    *,
    calibration_receipt_path: Path,
    mixer_gate_path: Path,
    block_gate_path: Path,
    logits_gate_path: Path,
    sft_final_report_path: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    report = build_stepwise_report(
        calibration_receipt_path=calibration_receipt_path,
        mixer_gate_path=mixer_gate_path,
        block_gate_path=block_gate_path,
        logits_gate_path=logits_gate_path,
        sft_final_report_path=sft_final_report_path,
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
        "--calibration-reuse-receipt",
        type=Path,
        default=DEFAULT_CALIBRATION_RECEIPT,
    )
    parser.add_argument(
        "--mixer-phase-gate",
        type=Path,
        default=DEFAULT_PHASE_GATE_ROOT / "mixer" / "phase_gate.json",
    )
    parser.add_argument(
        "--block-phase-gate",
        type=Path,
        default=DEFAULT_PHASE_GATE_ROOT / "block" / "phase_gate.json",
    )
    parser.add_argument(
        "--logits-phase-gate",
        type=Path,
        default=DEFAULT_PHASE_GATE_ROOT / "logits" / "phase_gate.json",
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

    report = create_stepwise_report(
        calibration_receipt_path=args.calibration_reuse_receipt,
        mixer_gate_path=args.mixer_phase_gate,
        block_gate_path=args.block_phase_gate,
        logits_gate_path=args.logits_phase_gate,
        sft_final_report_path=args.sft_final_report,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(
        f"stepwise_report={args.output_json.resolve()} "
        f"stages={len(report['stages'])} datasets={len(report['dataset_results'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
