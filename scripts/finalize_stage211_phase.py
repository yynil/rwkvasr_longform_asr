from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_PUBLIC_BENCHMARKS,
    build_stage211_full_data_coverage,
    load_stage211_post_coverage_correction_receipts,
    sha256_file,
    validate_stage211_full_data_coverage,
    validate_stage211_nano_public_baseline_receipt,
)
from rwkvasr.eval.stage211_public_metrics import (
    build_stage211_student_public_prediction_receipt,
    validate_stage211_student_public_prediction_receipt,
)

try:
    from scripts.validate_stage211_supplemental_retention import (
        validate_stratified_hidden_eval_v2,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from validate_stage211_supplemental_retention import (  # type: ignore[no-redef]
        validate_stratified_hidden_eval_v2,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PUBLIC_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_public_eval_benchmarks.sh"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_public_ctc_with_nano.py"
HIDDEN_GATE_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_hidden_alignment_gate.py"
LOGITS_GATE_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_logits_alignment_gate.py"
ALIGNMENT_PAIR_EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_stage211_alignment_pair.py"
STRATIFIED_SUMMARY_SCRIPT = REPO_ROOT / "scripts" / "summarize_stage211_stratified_hidden_eval.py"
STRATIFIED_LOGITS_SUMMARY_SCRIPT = (
    REPO_ROOT / "scripts" / "summarize_stage211_stratified_logits_eval.py"
)
PHASE_GATE_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_phase_gate.py"
PROMOTION_SCRIPT = REPO_ROOT / "scripts" / "create_stage211_promotion_receipt.py"
DEFAULT_PUBLIC_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_NANO_PREDICTION_DIR = (
    Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512" / "predictions"
)
DEFAULT_EVAL_ROOT = Path.home() / "rwkvasr_eval" / "stage211_phase_gates"
DEFAULT_STRATIFIED_HIDDEN_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "stratified_hidden_eval_v2"
    / "receipt.json"
)
STRATIFIED_HIDDEN_CELLS = (
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
DATASETS = (
    "aishell1_test",
    "librispeech_test_clean",
    "librispeech_test_other",
    "commonvoice_en_test",
    "wenetspeech_test_net",
)


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
        raise ValueError(f"Refusing to overwrite a different Stage211 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"[stage211-finalize] command={shlex.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def _resolve_curriculum(
    *,
    phase: str,
    phase_root: Path,
) -> tuple[dict[str, Any], Path, list[Path]]:
    summary_path = phase_root / "curriculum_complete.json"
    summary = _load_json(summary_path, label="Stage211 full-phase curriculum summary")
    if (
        summary.get("pipeline") != "stage211"
        or summary.get("artifact") != "full_phase_curriculum"
        or summary.get("phase") != phase
        or summary.get("complete") is not True
    ):
        raise ValueError(f"Invalid Stage211 curriculum summary: {summary_path}")
    coverage = summary.get("full_data_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 curriculum summary lacks full_data_coverage.")
    checkpoint = Path(str(coverage.get("final_checkpoint_path") or "")).resolve()
    validate_stage211_full_data_coverage(
        coverage,
        phase=phase,
        checkpoint_path=checkpoint,
    )
    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 curriculum summary lacks segment receipts.")
    by_difficulty = {
        str(segment.get("difficulty")): segment for segment in segments if isinstance(segment, dict)
    }
    receipt_paths = [
        Path(str(by_difficulty[difficulty]["receipt_path"])).resolve()
        for difficulty in STAGE211_AUDIO_CURRICULUM
    ]
    supplemental = coverage.get("supplemental_natural")
    if not isinstance(supplemental, dict):
        raise ValueError("Stage211 curriculum summary lacks supplemental_natural.")
    receipt_paths.append(Path(str(supplemental["receipt_path"])).resolve())
    return coverage, checkpoint, receipt_paths


def validate_curriculum_only(*, phase: str, phase_root: Path) -> Path:
    coverage, checkpoint, receipt_paths = _resolve_curriculum(
        phase=phase,
        phase_root=phase_root.expanduser().resolve(),
    )
    print(
        "[stage211-finalize] curriculum validation passed "
        f"phase={phase} segments={len(receipt_paths)} "
        f"rows={coverage['total_unique_rows']} checkpoint={checkpoint}",
        flush=True,
    )
    return checkpoint


def _run_public_eval(
    *,
    checkpoint: Path,
    output_dir: Path,
    manifest_dir: Path,
    devices: str,
    dry_run: bool,
) -> Path:
    env = {
        **os.environ,
        "CHECKPOINT_PATH": str(checkpoint),
        "OUTPUT_DIR": str(output_dir),
        "MANIFEST_DIR": str(manifest_dir),
        "PREPARE_MANIFESTS": "0",
        "DEVICES": devices,
        "CTC_BATCH_SIZE": "4",
        "CTC_NUM_WORKERS": "0",
        "CTC_SHARD_STAGE2": "1",
        "CTC_LIMIT": "0",
        "CTC_TEXT_NORMALIZATION": "ctc",
        "RUN_AR": "0",
        "METRIC_NORMALIZATION": "ctc",
    }
    _run(
        ["bash", str(PUBLIC_EVAL_SCRIPT)],
        dry_run=dry_run,
        env=env,
    )
    receipt_path = output_dir / "student_prediction_receipt.json"
    if dry_run:
        return receipt_path
    manifest_paths = {
        dataset: (manifest_dir / f"{dataset}.jsonl").resolve() for dataset in DATASETS
    }
    prediction_paths = {
        dataset: (output_dir / "predictions" / f"{dataset}.ctc.jsonl").resolve()
        for dataset in DATASETS
    }
    receipt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=checkpoint,
        manifest_paths=manifest_paths,
        prediction_paths=prediction_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    _write_immutable_json(receipt_path, receipt)
    validate_stage211_student_public_prediction_receipt(
        receipt_path,
        expected_checkpoint=checkpoint,
        expected_manifest_paths=manifest_paths,
        expected_prediction_paths=prediction_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    return receipt_path


def _validate_public_eval_inputs(
    *,
    manifest_dir: Path,
    nano_prediction_dir: Path,
    nano_public_baseline_receipt: Path,
) -> dict[str, Any]:
    receipt = validate_stage211_nano_public_baseline_receipt(
        nano_public_baseline_receipt,
    )
    raw_results = receipt.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(DATASETS):
        raise ValueError("Stage211 Nano public-baseline preflight dataset count mismatch.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if len(by_dataset) != len(DATASETS) or set(by_dataset) != set(DATASETS):
        raise ValueError("Stage211 Nano public-baseline preflight dataset set mismatch.")
    manifest_dir = manifest_dir.expanduser().resolve()
    nano_prediction_dir = nano_prediction_dir.expanduser().resolve()
    for dataset in DATASETS:
        result = by_dataset[dataset]
        expected_manifest = (manifest_dir / f"{dataset}.jsonl").resolve()
        recorded_manifest = Path(str(result.get("manifest_path") or "")).resolve()
        if recorded_manifest != expected_manifest:
            raise ValueError(
                f"Stage211 {dataset} public-eval manifest differs from the Nano baseline."
            )
        expected_prediction = (nano_prediction_dir / f"{dataset}.ctc.jsonl").resolve()
        recorded_prediction = Path(str(result.get("nano_prediction_path") or "")).resolve()
        if recorded_prediction != expected_prediction:
            raise ValueError(
                f"Stage211 {dataset} Nano prediction differs from the public baseline."
            )
    print(
        "[stage211-finalize] public input preflight passed "
        f"datasets={len(DATASETS)} samples={receipt['total_samples']} "
        f"receipt={nano_public_baseline_receipt}",
        flush=True,
    )
    return receipt


def _run_nano_comparison(
    *,
    checkpoint: Path,
    public_output_dir: Path,
    nano_prediction_dir: Path,
    comparison_json: Path,
    comparison_md: Path,
    student_prediction_receipt: Path,
    dry_run: bool,
) -> None:
    command = [
        str(PYTHON),
        str(COMPARE_SCRIPT),
        "--student-prediction-dir",
        str(public_output_dir / "predictions"),
        "--student-prediction-receipt",
        str(student_prediction_receipt),
    ]
    for dataset in DATASETS:
        command.extend(
            (
                "--nano-prediction",
                f"{dataset}={nano_prediction_dir / f'{dataset}.ctc.jsonl'}",
            )
        )
    command.extend(
        (
            "--student-checkpoint",
            str(checkpoint),
            "--output-json",
            str(comparison_json),
            "--output-md",
            str(comparison_md),
            "--normalization",
            "ctc",
            "--max-relative-ratio",
            "1.20",
            "--max-absolute-gap-points",
            "3.0",
        )
    )
    _run(command, dry_run=dry_run)


def _stratified_cell_manifests(receipt_path: Path) -> dict[str, Path]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = validate_stratified_hidden_eval_v2(receipt_path)
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(STRATIFIED_HIDDEN_CELLS):
        raise ValueError("Stage211 stratified hidden-eval cell coverage mismatch.")
    manifests = {}
    for cell_name in STRATIFIED_HIDDEN_CELLS:
        cell = cells[cell_name]
        if not isinstance(cell, dict) or int(cell.get("samples", -1)) != 256:
            raise ValueError(f"Invalid Stage211 stratified cell: {cell_name}")
        manifest_path = Path(str(cell.get("manifest_path") or "")).resolve()
        if not manifest_path.is_file() or sha256_file(manifest_path) != cell.get("manifest_sha256"):
            raise ValueError(
                f"Stage211 stratified cell manifest is missing or changed: {cell_name}"
            )
        manifests[cell_name] = manifest_path
    return manifests


def finalize_phase(args: argparse.Namespace) -> Path:
    phase = str(args.phase)
    phase_root = args.phase_root.expanduser().resolve()
    manifest_dir = args.public_manifest_dir.expanduser().resolve()
    nano_prediction_dir = args.nano_prediction_dir.expanduser().resolve()
    nano_public_baseline_receipt = (
        args.nano_public_baseline_receipt.expanduser().resolve()
        if args.nano_public_baseline_receipt is not None
        else (nano_prediction_dir.parent / "provenance_receipt.json").resolve()
    )
    if not bool(args.dry_run):
        _validate_public_eval_inputs(
            manifest_dir=manifest_dir,
            nano_prediction_dir=nano_prediction_dir,
            nano_public_baseline_receipt=nano_public_baseline_receipt,
        )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (DEFAULT_EVAL_ROOT / phase).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage, checkpoint, receipt_paths = _resolve_curriculum(
        phase=phase,
        phase_root=phase_root,
    )
    correction_receipt_paths = [
        path.expanduser().resolve()
        for path in getattr(args, "post_coverage_correction_receipt", [])
    ]
    correction_receipts = (
        load_stage211_post_coverage_correction_receipts(correction_receipt_paths)
        if phase == "mixer"
        else load_stage211_post_coverage_correction_receipts(
            correction_receipt_paths,
            phase=phase,
        )
    )
    if correction_receipts:
        checkpoint = Path(
            str(correction_receipts[-1].get("completion_checkpoint_path") or "")
        ).resolve()
        segments = coverage.get("segments")
        if not isinstance(segments, list):
            raise ValueError("Stage211 phase coverage lacks segment records.")
        coverage = build_stage211_full_data_coverage(
            phase=phase,
            segments=segments,
            checkpoint_path=checkpoint,
            post_coverage_corrections=correction_receipts,
            supplemental_segment=coverage.get("supplemental_natural"),
        )
        validate_stage211_full_data_coverage(
            coverage,
            phase=phase,
            checkpoint_path=checkpoint,
        )
    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 phase coverage lacks segment records.")
    easy_segment = next(
        (
            segment
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("difficulty") or "") == "easy"
        ),
        None,
    )
    if easy_segment is None:
        raise ValueError("Stage211 phase coverage lacks the easy initialization segment.")
    baseline_checkpoint = (
        Path(str(easy_segment.get("init_checkpoint_path") or "")).expanduser().resolve()
    )
    if not baseline_checkpoint.is_file() or sha256_file(baseline_checkpoint) != easy_segment.get(
        "init_checkpoint_sha256"
    ):
        raise ValueError("Stage211 phase initialization checkpoint is missing or changed.")

    alignment_eval_dir = output_dir / "alignment_pair"
    baseline_report = alignment_eval_dir / "baseline.json"
    candidate_report = alignment_eval_dir / "candidate.json"
    pair_eval_command = [
        str(PYTHON),
        str(ALIGNMENT_PAIR_EVAL_SCRIPT),
        "--phase",
        phase,
        "--train-config",
        str(phase_root / "easy" / "train_config.yaml"),
        "--model-config",
        str(phase_root / "easy" / "model_config.yaml"),
        "--baseline-checkpoint",
        str(baseline_checkpoint),
        "--candidate-checkpoint",
        str(checkpoint),
        "--baseline-output",
        str(baseline_report),
        "--candidate-output",
        str(candidate_report),
        "--samples",
        "256",
        "--feature-seed",
        "0",
        "--batch-size",
        str(getattr(args, "alignment_batch_size", 4)),
        "--num-workers",
        str(getattr(args, "alignment_num_workers", 4)),
        "--device",
        str(getattr(args, "alignment_device", "cuda:0")),
    ]
    alignment_teacher_device = getattr(args, "alignment_teacher_device", None)
    if alignment_teacher_device is not None:
        pair_eval_command.extend(("--teacher-device", str(alignment_teacher_device)))
    _run(pair_eval_command, dry_run=bool(args.dry_run))

    stratified_summary_path: Path | None = None
    if phase in {"mixer", "block", "logits"}:
        stratified_receipt_path = (
            Path(
                getattr(
                    args,
                    "stratified_hidden_receipt",
                    DEFAULT_STRATIFIED_HIDDEN_RECEIPT,
                )
                or DEFAULT_STRATIFIED_HIDDEN_RECEIPT
            )
            .expanduser()
            .resolve()
        )
        stratified_manifests = _stratified_cell_manifests(stratified_receipt_path)
        stratified_eval_dir = output_dir / "alignment_stratified"
        stratified_batch_size = int(getattr(args, "stratified_alignment_batch_size", 1))
        stratified_num_workers = int(getattr(args, "stratified_alignment_num_workers", 2))
        stratified_device = str(
            getattr(
                args,
                "stratified_alignment_device",
                getattr(args, "alignment_device", "cuda:0"),
            )
        )
        stratified_teacher_device = getattr(
            args,
            "stratified_alignment_teacher_device",
            getattr(args, "alignment_teacher_device", None),
        )
        for cell_name in STRATIFIED_HIDDEN_CELLS:
            cell_command = [
                str(PYTHON),
                str(ALIGNMENT_PAIR_EVAL_SCRIPT),
                "--phase",
                phase,
                "--train-config",
                str(phase_root / "easy" / "train_config.yaml"),
                "--model-config",
                str(phase_root / "easy" / "model_config.yaml"),
                "--baseline-checkpoint",
                str(baseline_checkpoint),
                "--candidate-checkpoint",
                str(checkpoint),
                "--baseline-output",
                str(stratified_eval_dir / f"{cell_name}_baseline.json"),
                "--candidate-output",
                str(stratified_eval_dir / f"{cell_name}_candidate.json"),
                "--eval-bucket-manifest",
                str(stratified_manifests[cell_name]),
                "--samples",
                "256",
                "--feature-seed",
                "0",
                "--batch-size",
                str(stratified_batch_size),
                "--num-workers",
                str(stratified_num_workers),
                "--device",
                stratified_device,
            ]
            if stratified_teacher_device is not None:
                cell_command.extend(("--teacher-device", str(stratified_teacher_device)))
            _run(cell_command, dry_run=bool(args.dry_run))
        stratified_summary_path = stratified_eval_dir / "summary.json"
        summary_script = (
            STRATIFIED_LOGITS_SUMMARY_SCRIPT if phase == "logits" else STRATIFIED_SUMMARY_SCRIPT
        )
        _run(
            [
                str(PYTHON),
                str(summary_script),
                "--receipt",
                str(stratified_receipt_path),
                "--eval-dir",
                str(stratified_eval_dir),
                "--output",
                str(stratified_summary_path),
            ],
            dry_run=bool(args.dry_run),
        )

    public_output = output_dir / "public"
    comparison_json = output_dir / "nano_comparison.json"
    comparison_md = output_dir / "nano_comparison.md"
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

    alignment_gate_path: Path | None = None
    if phase in {"mixer", "block"}:
        alignment_gate_path = output_dir / "hidden_gate.json"
        hidden_gate_command = [
            str(PYTHON),
            str(HIDDEN_GATE_SCRIPT),
            "--phase",
            phase,
            "--baseline-report",
            str(baseline_report),
            "--candidate-report",
            str(candidate_report),
            "--baseline-checkpoint",
            str(baseline_checkpoint),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(alignment_gate_path),
        ]
        if stratified_summary_path is not None:
            hidden_gate_command.extend(("--stratified-summary", str(stratified_summary_path)))
        _run(
            hidden_gate_command,
            dry_run=bool(args.dry_run),
        )
    elif phase == "logits":
        alignment_gate_path = output_dir / "logits_gate.json"
        logits_gate_command = [
            str(PYTHON),
            str(LOGITS_GATE_SCRIPT),
            "--baseline-report",
            str(baseline_report),
            "--candidate-report",
            str(candidate_report),
            "--baseline-checkpoint",
            str(baseline_checkpoint),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(alignment_gate_path),
        ]
        if stratified_summary_path is not None:
            logits_gate_command.extend(("--stratified-summary", str(stratified_summary_path)))
        _run(logits_gate_command, dry_run=bool(args.dry_run))

    phase_gate_path = output_dir / "phase_gate.json"
    phase_gate_command = [
        str(PYTHON),
        str(PHASE_GATE_SCRIPT),
        "--phase",
        phase,
        "--checkpoint",
        str(checkpoint),
        "--preflight-smoke-marker",
        str(phase_root / "full_profile_smoke_passed.json"),
        "--global-dedup-manifest",
        str(
            Path(
                getattr(
                    args,
                    "global_dedup_manifest",
                    DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
                )
            )
            .expanduser()
            .resolve()
        ),
        "--loaded-manifest-receipt",
        str(
            Path(
                getattr(
                    args,
                    "loaded_manifest_receipt",
                    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
                )
            )
            .expanduser()
            .resolve()
        ),
        "--public-comparison-report",
        str(comparison_json),
        "--manifest-dir",
        str(manifest_dir),
        "--nano-public-baseline-receipt",
        str(nano_public_baseline_receipt),
        "--output",
        str(phase_gate_path),
    ]
    for receipt_path in receipt_paths:
        phase_gate_command.extend(("--coverage-receipt", str(receipt_path)))
    for correction_receipt_path in correction_receipt_paths:
        phase_gate_command.extend(
            (
                "--post-coverage-correction-receipt",
                str(correction_receipt_path),
            )
        )
    if alignment_gate_path is not None:
        phase_gate_command.extend(("--alignment-report", str(alignment_gate_path)))
    if phase in {"mixer", "block"}:
        if args.baseline_public_comparison_report is None:
            raise ValueError(
                f"Stage211 {phase} finalization requires --baseline-public-comparison-report."
            )
        phase_gate_command.extend(
            (
                "--baseline-public-comparison-report",
                str(args.baseline_public_comparison_report.expanduser().resolve()),
            )
        )
    _run(phase_gate_command, dry_run=bool(args.dry_run))

    promotion_receipt = output_dir / f"{phase}_promotion_receipt.json"
    if not args.dry_run:
        report = _load_json(phase_gate_path, label="Stage211 phase gate")
        if report.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError(f"Stage211 {phase} phase gate checkpoint changed.")
        if report.get("gate_passed") is not True:
            raise ValueError(
                f"Stage211 {phase} phase gate did not pass; "
                f"failure evidence is preserved at {phase_gate_path}."
            )
    _run(
        [
            str(PYTHON),
            str(PROMOTION_SCRIPT),
            "--source-phase",
            phase,
            "--checkpoint",
            str(checkpoint),
            "--gate-report",
            str(phase_gate_path),
            "--output",
            str(promotion_receipt),
            "--confirm-gate-passed",
        ],
        dry_run=bool(args.dry_run),
    )
    print(
        f"[stage211-finalize] complete phase={phase} checkpoint={checkpoint} "
        f"gate={phase_gate_path} promotion={promotion_receipt}",
        flush=True,
    )
    return promotion_receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and gate one completed Stage211 A/B/C full-data phase on "
            "the complete normalized English WER and Chinese CER suite."
        )
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), required=True)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
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
        "--baseline-public-comparison-report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--post-coverage-correction-receipt",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument(
        "--global-dedup-manifest",
        type=Path,
        default=DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
    )
    parser.add_argument(
        "--loaded-manifest-receipt",
        type=Path,
        default=DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    )
    parser.add_argument("--alignment-device", default="cuda:0")
    parser.add_argument("--alignment-teacher-device", default=None)
    parser.add_argument("--alignment-batch-size", type=int, default=4)
    parser.add_argument("--alignment-num-workers", type=int, default=4)
    parser.add_argument(
        "--stratified-hidden-receipt",
        type=Path,
        default=DEFAULT_STRATIFIED_HIDDEN_RECEIPT,
    )
    parser.add_argument("--stratified-alignment-device", default="cuda:0")
    parser.add_argument(
        "--stratified-alignment-teacher-device",
        default=None,
    )
    parser.add_argument("--stratified-alignment-batch-size", type=int, default=1)
    parser.add_argument("--stratified-alignment-num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-curriculum-only",
        action="store_true",
        help=(
            "Deep-validate completed five-segment curriculum evidence without "
            "creating evaluation outputs or launching model evaluation."
        ),
    )
    args = parser.parse_args()

    if args.validate_curriculum_only:
        validate_curriculum_only(
            phase=str(args.phase),
            phase_root=args.phase_root,
        )
        return 0
    finalize_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
