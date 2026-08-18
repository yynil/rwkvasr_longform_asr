from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_artifact_io import write_immutable_json
from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_phase_gate_report,
    validate_stage211_public_benchmark,
    validate_stage211_public_overlap_binding,
)
from rwkvasr.eval.stage211_initialization import DEFAULT_STAGE211_INITIALIZATION_RECEIPT
from rwkvasr.eval.stage211_public_metrics import (
    build_stage211_sft_public_progress,
    replay_stage211_sft_public_evidence,
)

try:
    from scripts.create_stage211_stepwise_report import create_stepwise_report
    from scripts.create_stage211_phase_gate import _enrich_public_benchmark
    from scripts.finalize_stage211_phase import (
        DEFAULT_NANO_PREDICTION_DIR,
        DEFAULT_PUBLIC_MANIFEST_DIR,
        _run_nano_comparison,
        _run_public_eval,
        _validate_public_eval_inputs,
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
        _validate_public_eval_inputs,
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
    write_immutable_json(path, payload, label="Stage211 final report")


def _resolve_phase_gate(
    *,
    phase: str,
    phase_gate_root: Path,
    selection_path: Path | None,
    nano_teacher_checkpoint: Path,
) -> tuple[Path, Path | None]:
    targets = {"mixer": "block", "block": "logits", "logits": "sft"}
    if phase not in targets:
        raise ValueError(f"Unsupported Stage211 phase selection: {phase!r}")
    phase_gate_root = phase_gate_root.expanduser().resolve()
    explicit_selection = selection_path is not None
    selected_path = (
        selection_path.expanduser().resolve()
        if selection_path is not None
        else (phase_gate_root / f"{phase}_selected.json").resolve()
    )
    if not selected_path.is_file():
        if explicit_selection:
            raise ValueError(f"Stage211 {phase} gate selection is missing: {selected_path}")
        return (phase_gate_root / phase / "phase_gate.json").resolve(), None

    selection = _load_json(selected_path, label=f"Stage211 {phase} gate selection")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "mixer_gate_selection" if phase == "mixer" else "phase_gate_selection",
        "phase": phase,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {phase} gate selection contract mismatch.")
    gate_dir = Path(str(selection.get("gate_dir") or "")).resolve()
    gate_path = Path(str(selection.get("gate_path") or "")).resolve()
    if gate_path != gate_dir / "phase_gate.json":
        raise ValueError(f"Stage211 {phase} gate selection path mismatch.")
    if not gate_path.is_file() or selection.get("gate_sha256") != sha256_file(gate_path):
        raise ValueError(f"Stage211 selected {phase} gate is missing or changed.")
    checkpoint = Path(str(selection.get("checkpoint_path") or "")).resolve()
    if not checkpoint.is_file() or selection.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError(f"Stage211 selected {phase} checkpoint is missing or changed.")
    gate = validate_stage211_phase_gate_report(
        gate_path,
        expected_phase=phase,
        checkpoint_path=checkpoint,
    )
    if gate.get("gate_passed") is not True:
        raise ValueError(f"Stage211 selected {phase} gate did not pass.")
    promotion = Path(str(selection.get("promotion_receipt_path") or "")).resolve()
    if not promotion.is_file() or selection.get("promotion_receipt_sha256") != sha256_file(
        promotion
    ):
        raise ValueError(f"Stage211 selected {phase} promotion is missing or changed.")
    _validate_promotion_receipt(
        receipt_path=promotion,
        target_phase=targets[phase],
        checkpoint_path=checkpoint,
        nano_checkpoint_path=nano_teacher_checkpoint,
    )
    return gate_path, selected_path


def _resolve_mixer_gate(
    *,
    phase_gate_root: Path,
    selection_path: Path | None,
    nano_teacher_checkpoint: Path,
) -> tuple[Path, Path | None]:
    return _resolve_phase_gate(
        phase="mixer",
        phase_gate_root=phase_gate_root,
        selection_path=selection_path,
        nano_teacher_checkpoint=nano_teacher_checkpoint,
    )


def _build_sft_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    return build_stage211_sft_public_progress(
        baseline=baseline,
        candidate=candidate,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        tolerance=tolerance,
    )


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
    benchmark = validate_stage211_public_benchmark(
        report.get("public_benchmark"),
        require_metric_source_recomputed=True,
    )
    labeled_coverage = report.get("labeled_data_coverage")
    if not isinstance(labeled_coverage, dict):
        raise ValueError("Stage211 final report lacks labeled-data coverage.")
    replayed_public = replay_stage211_sft_public_evidence(
        report,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_baseline_checkpoint=Path(
            str(labeled_coverage.get("init_checkpoint_path") or "")
        ).resolve(),
        expected_candidate_checkpoint=checkpoint,
    )
    benchmark = replayed_public["public_benchmark"]
    if benchmark.get("all_datasets_pass") is not True:
        raise ValueError("Stage211D did not pass the every-dataset Nano WER/CER gate.")
    validate_stage211_public_overlap_binding(
        report.get("public_overlap"),
        public_benchmark=benchmark,
    )
    progress = replayed_public["public_progress"]
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
        (
            "nano_public_baseline_receipt_path",
            "nano_public_baseline_receipt_sha256",
        ),
        ("mixer_phase_gate_path", "mixer_phase_gate_sha256"),
        ("block_phase_gate_path", "block_phase_gate_sha256"),
        ("logits_phase_gate_path", "logits_phase_gate_sha256"),
    ):
        artifact = Path(str(report.get(path_key) or "")).resolve()
        if not artifact.is_file() or sha256_file(artifact) != report.get(sha_key):
            raise ValueError(
                f"Stage211 final report artifact is unavailable or changed: {artifact}"
            )
    for phase in ("mixer", "block", "logits"):
        selection_value = report.get(f"{phase}_gate_selection_path")
        if selection_value is not None:
            selection = Path(str(selection_value)).resolve()
            if not selection.is_file() or sha256_file(selection) != report.get(
                f"{phase}_gate_selection_sha256"
            ):
                raise ValueError(
                    f"Stage211 final report {phase} gate selection is unavailable or changed."
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
            raise ValueError(f"Stage211 final report {key} differs from SFT completion.")
    completion_checkpoint_value = completion.get("completion_checkpoint_path")
    full_sft_checkpoint = (
        Path(str(completion_checkpoint_value)).resolve()
        if completion_checkpoint_value is not None
        else checkpoint.resolve()
    )
    correction_keys = (
        "sft_correction_evaluation_path",
        "sft_correction_evaluation_sha256",
        "sft_correction_profile_path",
        "sft_correction_profile_sha256",
        "sft_correction_completion_receipts",
        "sft_correction_coverage",
        "sft_correction_public_progress",
    )
    if checkpoint.resolve() != full_sft_checkpoint:
        try:
            from scripts.evaluate_stage211_sft_correction import (
                validate_correction_evaluation_report,
            )
        except ModuleNotFoundError as error:
            if error.name != "scripts":
                raise
            from evaluate_stage211_sft_correction import (
                validate_correction_evaluation_report,
            )

        correction_evaluation_path = Path(
            str(report.get("sft_correction_evaluation_path") or "")
        ).resolve()
        correction_profile_path = Path(
            str(report.get("sft_correction_profile_path") or "")
        ).resolve()
        for artifact, expected_sha256, label in (
            (
                correction_evaluation_path,
                report.get("sft_correction_evaluation_sha256"),
                "correction evaluation",
            ),
            (
                correction_profile_path,
                report.get("sft_correction_profile_sha256"),
                "correction profile",
            ),
        ):
            if not artifact.is_file() or expected_sha256 != sha256_file(artifact):
                raise ValueError(
                    f"Stage211 final report {label} is unavailable or changed: {artifact}"
                )
        correction_evaluation = validate_correction_evaluation_report(
            correction_evaluation_path,
            expected_full_completion_path=Path(str(report["sft_completion_path"])),
            expected_correction_profile_path=correction_profile_path,
            require_passed=True,
        )
        if (
            Path(str(correction_evaluation.get("checkpoint_path") or "")).resolve()
            != checkpoint.resolve()
            or report.get("sft_correction_completion_receipts")
            != correction_evaluation.get("correction_completion_receipts")
            or report.get("sft_correction_coverage")
            != correction_evaluation.get("correction_coverage")
            or report.get("sft_correction_public_progress")
            != correction_evaluation.get("public_progress")
        ):
            raise ValueError("Stage211 final report correction evidence chain mismatch.")
    elif any(key in report for key in correction_keys):
        raise ValueError("Uncorrected Stage211 final report contains correction evidence.")
    promotion_receipt = _load_json(
        Path(str(report["logits_promotion_receipt_path"])).resolve(),
        label="Stage211 logits promotion receipt",
    )
    if promotion_receipt.get("nano_teacher_checkpoint_sha256") != report.get(
        "nano_teacher_checkpoint_sha256"
    ):
        raise ValueError("Stage211 final report Nano teacher differs from the logits promotion.")
    baseline_receipt = validate_stage211_nano_public_baseline_receipt(
        Path(str(report["nano_public_baseline_receipt_path"])),
        expected_receipt_sha256=str(report["nano_public_baseline_receipt_sha256"]),
        expected_nano_checkpoint_sha256=str(report["nano_teacher_checkpoint_sha256"]),
        public_benchmark=benchmark,
    )
    if report.get("nano_public_baseline_checkpoint_sha256") != baseline_receipt.get(
        "nano_checkpoint_sha256"
    ):
        raise ValueError("Stage211 final report Nano public-baseline checkpoint binding mismatch.")
    if baseline_receipt.get("public_overlap") != report.get("public_overlap"):
        raise ValueError("Stage211 final report and Nano baseline overlap bindings differ.")
    logits_gate_path = Path(str(report.get("logits_phase_gate_path") or "")).resolve()
    logits_gate = validate_stage211_phase_gate_report(
        logits_gate_path,
        expected_phase="logits",
        checkpoint_path=Path(str(labeled_coverage["init_checkpoint_path"])).resolve(),
    )
    if logits_gate.get("public_overlap") != report.get("public_overlap"):
        raise ValueError("Stage211 final report and Logits overlap bindings differ.")
    mixer_gate_path = Path(str(report.get("mixer_phase_gate_path") or "")).resolve()
    mixer_gate = validate_stage211_phase_gate_report(
        mixer_gate_path,
        expected_phase="mixer",
        checkpoint_path=Path(
            str(_load_json(mixer_gate_path, label="Mixer gate")["checkpoint_path"])
        ),
    )
    if mixer_gate.get("public_overlap") != report.get("public_overlap"):
        raise ValueError("Stage211 final report and Mixer overlap bindings differ.")
    block_gate_path = Path(str(report.get("block_phase_gate_path") or "")).resolve()
    block_gate = validate_stage211_phase_gate_report(
        block_gate_path,
        expected_phase="block",
        checkpoint_path=Path(
            str(_load_json(block_gate_path, label="Block gate")["checkpoint_path"])
        ),
    )
    if block_gate.get("public_overlap") != report.get("public_overlap"):
        raise ValueError("Stage211 final report and Block overlap bindings differ.")
    return report


def finalize_sft(args: argparse.Namespace) -> Path:
    run_dir = args.run_dir.expanduser().resolve()
    completion_path = (
        args.completion_report.expanduser().resolve()
        if args.completion_report is not None
        else (run_dir / "sft_complete.json").resolve()
    )
    completion, checkpoint = _validate_completion(
        completion_path,
        require_full_profile=True,
    )
    init_checkpoint = Path(str(completion["init_checkpoint_path"])).resolve()
    nano_teacher_checkpoint = Path(str(completion["nano_teacher_checkpoint_path"])).resolve()
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
    nano_public_baseline_receipt_path = (
        args.nano_public_baseline_receipt.expanduser().resolve()
        if args.nano_public_baseline_receipt is not None
        else (nano_prediction_dir.parent / "provenance_receipt.json").resolve()
    )
    if not bool(args.dry_run):
        _validate_public_eval_inputs(
            manifest_dir=manifest_dir,
            nano_prediction_dir=nano_prediction_dir,
            nano_public_baseline_receipt=nano_public_baseline_receipt_path,
        )
    student_prediction_receipt = _run_public_eval(
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
        student_prediction_receipt=student_prediction_receipt,
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
        require_student_prediction_receipt=True,
    )
    public_overlap_receipt_path = (
        Path(getattr(args, "public_overlap_receipt", DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT))
        .expanduser()
        .resolve()
    )
    public_overlap_binding = {
        "receipt_path": str(public_overlap_receipt_path),
        "receipt_sha256": sha256_file(public_overlap_receipt_path),
    }
    validate_stage211_public_overlap_binding(
        public_overlap_binding,
        public_benchmark=candidate_benchmark,
    )
    nano_public_baseline = validate_stage211_nano_public_baseline_receipt(
        nano_public_baseline_receipt_path,
        expected_nano_checkpoint_sha256=str(completion["nano_teacher_checkpoint_sha256"]),
        public_benchmark=candidate_benchmark,
    )
    progress = _build_sft_public_progress(
        baseline=baseline_benchmark,
        candidate=candidate_benchmark,
    )
    gate_passed = (
        progress["gate_passed"] is True and candidate_benchmark.get("all_datasets_pass") is True
    )
    phase_gate_root_value = getattr(args, "phase_gate_root", None)
    phase_gate_root = (
        phase_gate_root_value.expanduser().resolve()
        if phase_gate_root_value is not None
        else output_dir.parent
    )
    mixer_gate_path: Path | None = None
    mixer_selection_path: Path | None = None
    block_gate_path: Path | None = None
    block_selection_path: Path | None = None
    logits_gate_path: Path | None = None
    logits_selection_path: Path | None = None
    if gate_passed:
        mixer_gate_path, mixer_selection_path = _resolve_mixer_gate(
            phase_gate_root=phase_gate_root,
            selection_path=getattr(args, "mixer_gate_selection", None),
            nano_teacher_checkpoint=nano_teacher_checkpoint,
        )
        block_gate_path, block_selection_path = _resolve_phase_gate(
            phase="block",
            phase_gate_root=phase_gate_root,
            selection_path=getattr(args, "block_gate_selection", None),
            nano_teacher_checkpoint=nano_teacher_checkpoint,
        )
        logits_gate_path, logits_selection_path = _resolve_phase_gate(
            phase="logits",
            phase_gate_root=phase_gate_root,
            selection_path=getattr(args, "logits_gate_selection", None),
            nano_teacher_checkpoint=nano_teacher_checkpoint,
        )
        if logits_gate_path != Path(str(promotion_receipt["gate_report_path"])).resolve():
            raise ValueError(
                "Stage211 selected Logits gate differs from the SFT promotion receipt."
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
        "nano_teacher_checkpoint_sha256": completion["nano_teacher_checkpoint_sha256"],
        "nano_public_baseline_receipt_path": str(nano_public_baseline_receipt_path),
        "nano_public_baseline_receipt_sha256": sha256_file(nano_public_baseline_receipt_path),
        "nano_public_baseline_checkpoint_sha256": nano_public_baseline["nano_checkpoint_sha256"],
        "public_overlap": public_overlap_binding,
        "baseline_public_comparison_report_path": str(baseline_report_path),
        "baseline_public_comparison_report_sha256": sha256_file(baseline_report_path),
        "public_comparison_report_path": str(comparison_json),
        "public_comparison_report_sha256": sha256_file(comparison_json),
        "baseline_public_benchmark": baseline_benchmark,
        "public_benchmark": candidate_benchmark,
        "public_progress": progress,
        "mixer_phase_gate_path": (str(mixer_gate_path) if mixer_gate_path is not None else None),
        "mixer_phase_gate_sha256": (
            sha256_file(mixer_gate_path) if mixer_gate_path is not None else None
        ),
        "mixer_gate_selection_path": (
            str(mixer_selection_path) if mixer_selection_path is not None else None
        ),
        "mixer_gate_selection_sha256": (
            sha256_file(mixer_selection_path) if mixer_selection_path is not None else None
        ),
        "block_phase_gate_path": (str(block_gate_path) if block_gate_path is not None else None),
        "block_phase_gate_sha256": (
            sha256_file(block_gate_path) if block_gate_path is not None else None
        ),
        "block_gate_selection_path": (
            str(block_selection_path) if block_selection_path is not None else None
        ),
        "block_gate_selection_sha256": (
            sha256_file(block_selection_path) if block_selection_path is not None else None
        ),
        "logits_gate_selection_path": (
            str(logits_selection_path) if logits_selection_path is not None else None
        ),
        "logits_gate_selection_sha256": (
            sha256_file(logits_selection_path) if logits_selection_path is not None else None
        ),
    }
    _write_immutable_json(final_report_path, report)
    if not gate_passed:
        raise ValueError(
            "Stage211D did not improve the selected C checkpoint without "
            f"public-set regression; report={final_report_path}"
        )
    _validate_final_report(final_report_path, checkpoint=checkpoint)
    if mixer_gate_path is None or block_gate_path is None or logits_gate_path is None:
        raise ValueError("Stage211 final report lacks a selected A/B/C gate.")
    create_stepwise_report(
        initialization_receipt_path=getattr(
            args,
            "initialization_receipt",
            DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
        ),
        calibration_receipt_path=args.calibration_reuse_receipt,
        mixer_gate_path=mixer_gate_path,
        block_gate_path=block_gate_path,
        logits_gate_path=logits_gate_path,
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
        "--nano-public-baseline-receipt",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--initialization-receipt",
        type=Path,
        default=DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
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
    parser.add_argument(
        "--public-overlap-receipt",
        type=Path,
        default=DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    )
    parser.add_argument("--phase-gate-root", type=Path, default=None)
    parser.add_argument("--mixer-gate-selection", type=Path, default=None)
    parser.add_argument("--block-gate-selection", type=Path, default=None)
    parser.add_argument("--logits-gate-selection", type=Path, default=None)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    finalize_sft(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
