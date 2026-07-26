from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_public_benchmark,
)

try:
    from scripts.create_stage211_stepwise_report import create_stepwise_report
    from scripts.create_stage211_phase_gate import _enrich_public_benchmark
    from scripts.finalize_stage211_phase import (
        DEFAULT_NANO_PREDICTION_DIR,
        DEFAULT_PUBLIC_MANIFEST_DIR,
        _run_nano_comparison,
        _run_public_eval,
    )
    from scripts.run_stage211_labeled_sft import _validate_completion
    from scripts.run_stage211_strict_chained_alignment import (
        _validate_promotion_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_stepwise_report import create_stepwise_report
    from create_stage211_phase_gate import _enrich_public_benchmark
    from finalize_stage211_phase import (
        DEFAULT_NANO_PREDICTION_DIR,
        DEFAULT_PUBLIC_MANIFEST_DIR,
        _run_nano_comparison,
        _run_public_eval,
    )
    from run_stage211_labeled_sft import _validate_completion
    from run_stage211_strict_chained_alignment import (
        _validate_promotion_receipt,
    )


DEFAULT_SFT_RUN_DIR = (
    Path.home() / "rwkvasr_runs" / "stage211_full_alignment" / "stage211d_labeled_ctc_sft_1ep"
)
DEFAULT_OUTPUT_DIR = Path.home() / "rwkvasr_eval" / "stage211_phase_gates" / "sft"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211 final report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _build_sft_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(candidate_results):
        raise ValueError("Stage211D baseline/candidate public datasets differ.")
    rows: list[dict[str, Any]] = []
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
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
                "metric": STAGE211_PUBLIC_BENCHMARKS[dataset]["metric"],
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


def _validate_final_report(
    path: Path,
    *,
    checkpoint: Path,
) -> dict[str, Any]:
    report = _load_json(path, label="Stage211 final completion report")
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
            raise ValueError(f"Stage211 final report {key} mismatch.")
    if Path(str(report.get("checkpoint_path") or "")).resolve() != checkpoint.resolve():
        raise ValueError("Stage211 final report checkpoint path mismatch.")
    if report.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("Stage211 final report checkpoint SHA-256 mismatch.")
    benchmark = validate_stage211_public_benchmark(report.get("public_benchmark"))
    if benchmark.get("all_datasets_pass") is not True:
        raise ValueError("Stage211D did not pass the every-dataset Nano WER/CER gate.")
    progress = report.get("public_progress")
    if (
        not isinstance(progress, dict)
        or progress.get("gate_passed") is not True
        or progress.get("no_dataset_regression") is not True
        or progress.get("macro_improved") is not True
        or int(progress.get("improved_datasets", 0)) <= 0
    ):
        raise ValueError("Stage211D public progress gate did not pass.")
    for path_key, sha_key in (
        ("sft_completion_path", "sft_completion_sha256"),
        (
            "baseline_public_comparison_report_path",
            "baseline_public_comparison_report_sha256",
        ),
        (
            "public_comparison_report_path",
            "public_comparison_report_sha256",
        ),
        (
            "logits_promotion_receipt_path",
            "logits_promotion_receipt_sha256",
        ),
        (
            "nano_teacher_checkpoint_path",
            "nano_teacher_checkpoint_sha256",
        ),
    ):
        artifact = Path(str(report.get(path_key) or "")).resolve()
        if not artifact.is_file() or sha256_file(artifact) != report.get(sha_key):
            raise ValueError(
                f"Stage211 final report artifact is unavailable or changed: {artifact}"
            )
    completion = _load_json(
        Path(str(report["sft_completion_path"])).resolve(),
        label="Stage211 SFT completion",
    )
    for key in (
        "nano_teacher_checkpoint_path",
        "nano_teacher_checkpoint_sha256",
    ):
        if report.get(key) != completion.get(key):
            raise ValueError(
                f"Stage211 final report {key} differs from SFT completion."
            )
    promotion_receipt = _load_json(
        Path(str(report["logits_promotion_receipt_path"])).resolve(),
        label="Stage211 logits promotion receipt",
    )
    if promotion_receipt.get("nano_teacher_checkpoint_sha256") != report.get(
        "nano_teacher_checkpoint_sha256"
    ):
        raise ValueError(
            "Stage211 final report Nano teacher differs from the logits promotion."
        )
    return report


def finalize_sft(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.expanduser().resolve()
    completion_path = (
        args.completion_report.expanduser().resolve()
        if args.completion_report is not None
        else (run_dir / "sft_complete.json").resolve()
    )
    completion, checkpoint = _validate_completion(completion_path)
    init_checkpoint = Path(str(completion["init_checkpoint_path"])).resolve()
    nano_teacher_checkpoint = Path(
        str(completion["nano_teacher_checkpoint_path"])
    ).resolve()
    promotion_receipt_path = Path(str(completion["logits_promotion_receipt_path"])).resolve()
    promotion_receipt = _validate_promotion_receipt(
        receipt_path=promotion_receipt_path,
        target_phase="sft",
        checkpoint_path=init_checkpoint,
        nano_checkpoint_path=nano_teacher_checkpoint,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_output = output_dir / "public"
    comparison_json = output_dir / "nano_comparison.json"
    comparison_md = output_dir / "nano_comparison.md"
    manifest_dir = args.public_manifest_dir.expanduser().resolve()
    nano_prediction_dir = args.nano_prediction_dir.expanduser().resolve()
    _run_public_eval(
        checkpoint=checkpoint,
        output_dir=public_output,
        manifest_dir=manifest_dir,
        devices=str(args.devices),
        dry_run=bool(args.dry_run),
    )
    _run_nano_comparison(
        checkpoint=checkpoint,
        public_output_dir=public_output,
        nano_prediction_dir=nano_prediction_dir,
        comparison_json=comparison_json,
        comparison_md=comparison_md,
        dry_run=bool(args.dry_run),
    )
    final_report_path = output_dir / "stage211_complete.json"
    if args.dry_run:
        print(f"[stage211-sft-finalize] dry-run report={final_report_path}", flush=True)
        return final_report_path

    baseline_report_path = args.baseline_public_comparison_report.expanduser().resolve()
    baseline_report = _load_json(
        baseline_report_path,
        label="Stage211C baseline public comparison",
    )
    if Path(
        str(baseline_report.get("student_checkpoint_path") or "")
    ).resolve() != init_checkpoint or baseline_report.get(
        "student_checkpoint_sha256"
    ) != sha256_file(init_checkpoint):
        raise ValueError(
            "Stage211D baseline public comparison does not bind the promoted Stage211C checkpoint."
        )
    candidate_report = _load_json(
        comparison_json,
        label="Stage211D public comparison",
    )
    baseline_benchmark = _enrich_public_benchmark(
        baseline_report,
        manifest_dir=manifest_dir,
    )
    candidate_benchmark = _enrich_public_benchmark(
        candidate_report,
        manifest_dir=manifest_dir,
    )
    progress = _build_sft_public_progress(
        baseline=baseline_benchmark,
        candidate=candidate_benchmark,
    )
    gate_passed = (
        progress["gate_passed"] is True and candidate_benchmark.get("all_datasets_pass") is True
    )
    report = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "final_completion",
        "phase": "sft",
        "complete": gate_passed,
        "gate_passed": gate_passed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sft_completion_path": str(completion_path),
        "sft_completion_sha256": sha256_file(completion_path),
        "labeled_data_coverage": completion,
        "logits_promotion_receipt_path": str(promotion_receipt_path),
        "logits_promotion_receipt_sha256": sha256_file(promotion_receipt_path),
        "logits_phase_gate_path": promotion_receipt["gate_report_path"],
        "logits_phase_gate_sha256": promotion_receipt["gate_report_sha256"],
        "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint),
        "nano_teacher_checkpoint_sha256": completion[
            "nano_teacher_checkpoint_sha256"
        ],
        "baseline_public_comparison_report_path": str(baseline_report_path),
        "baseline_public_comparison_report_sha256": sha256_file(baseline_report_path),
        "public_comparison_report_path": str(comparison_json),
        "public_comparison_report_sha256": sha256_file(comparison_json),
        "baseline_public_benchmark": baseline_benchmark,
        "public_benchmark": candidate_benchmark,
        "public_progress": progress,
    }
    _write_immutable_json(final_report_path, report)
    if not gate_passed:
        raise ValueError(
            "Stage211D did not improve the selected C checkpoint without "
            f"public-set regression; report={final_report_path}"
        )
    _validate_final_report(final_report_path, checkpoint=checkpoint)
    phase_gate_root = (
        args.phase_gate_root.expanduser().resolve()
        if args.phase_gate_root is not None
        else output_dir.parent
    )
    create_stepwise_report(
        calibration_receipt_path=args.calibration_reuse_receipt,
        mixer_gate_path=phase_gate_root / "mixer" / "phase_gate.json",
        block_gate_path=phase_gate_root / "block" / "phase_gate.json",
        logits_gate_path=phase_gate_root / "logits" / "phase_gate.json",
        sft_final_report_path=final_report_path,
        output_json=output_dir / "stage211_stepwise_results.json",
        output_markdown=output_dir / "stage211_stepwise_results.md",
    )
    print(
        f"[stage211-sft-finalize] complete checkpoint={checkpoint} "
        f"report={final_report_path} "
        f"stepwise_report={output_dir / 'stage211_stepwise_results.json'}",
        flush=True,
    )
    return final_report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete direct-CTC benchmark after Stage211D and emit the "
            "strict final A/B/C/D completion gate."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_SFT_RUN_DIR)
    parser.add_argument("--completion-report", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-public-comparison-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--public-manifest-dir",
        type=Path,
        default=DEFAULT_PUBLIC_MANIFEST_DIR,
    )
    parser.add_argument(
        "--nano-prediction-dir",
        type=Path,
        default=DEFAULT_NANO_PREDICTION_DIR,
    )
    parser.add_argument(
        "--calibration-reuse-receipt",
        type=Path,
        default=(
            Path.home()
            / "rwkvasr_eval"
            / "stage211_calibration_selected_full"
            / "public"
            / "reuse_receipt.json"
        ),
    )
    parser.add_argument("--phase-gate-root", type=Path, default=None)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    finalize_sft(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
