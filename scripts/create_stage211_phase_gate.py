from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval import normalize_asr_text_for_metrics
from rwkvasr.eval.stage211_artifact_io import write_immutable_text
from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    STAGE211_PROMOTION_POLICY_COVERAGE_NON_DIVERGENT,
    STAGE211_PROMOTION_POLICY_STRICT,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_PHASE_GATE_SCHEMA_VERSION,
    STAGE211_PUBLIC_BENCHMARKS,
    build_stage211_alignment_loss_nondivergence,
    build_stage211_correction_round_promotion_gate,
    build_stage211_full_data_coverage,
    build_stage211_phase_baseline_public_provenance,
    build_stage211_step_eval_cadence,
    build_stage211_trajectory_retention_gate,
    load_stage211_post_coverage_correction_receipts,
    sha256_file,
    stage211_phase_gate_decision,
    validate_stage211_full_profile_smoke_binding,
    validate_stage211_global_dedup_manifest,
    validate_stage211_loaded_manifest_receipt,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_phase_gate_report,
    validate_stage211_public_overlap_binding,
)
from rwkvasr.eval.stage211_supplemental import STAGE211_SUPPLEMENTAL_DIFFICULTY
from rwkvasr.eval.stage211_public_metrics import (
    STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
    build_stage211_public_progress,
    replay_stage211_public_comparison,
)

try:
    from scripts.create_stage211_hidden_alignment_gate import (
        build_gate as build_hidden_alignment_gate,
    )
    from scripts.create_stage211_logits_alignment_gate import (
        build_gate as build_logits_alignment_gate,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_hidden_alignment_gate import (
        build_gate as build_hidden_alignment_gate,
    )
    from create_stage211_logits_alignment_gate import (
        build_gate as build_logits_alignment_gate,
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _parse_coverage_receipts(
    values: list[Path],
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for path in values:
        resolved = path.resolve()
        receipt = _load_json(resolved, label="Stage211 curriculum coverage receipt")
        difficulty = str(receipt.get("difficulty") or "")
        if (
            receipt.get("pipeline") != "stage211"
            or receipt.get("artifact") != "curriculum_coverage"
            or receipt.get("phase") != phase
            or receipt.get("complete") is not True
            or difficulty not in {*STAGE211_AUDIO_CURRICULUM, STAGE211_SUPPLEMENTAL_DIFFICULTY}
        ):
            raise ValueError(f"Invalid Stage211 curriculum coverage receipt: {resolved}")
        if difficulty in receipts:
            raise ValueError(f"Duplicate Stage211 coverage receipt for {difficulty}.")
        receipts[difficulty] = {
            **receipt,
            "receipt_path": str(resolved),
            "receipt_sha256": sha256_file(resolved),
        }
    required = {*STAGE211_AUDIO_CURRICULUM, STAGE211_SUPPLEMENTAL_DIFFICULTY}
    if set(receipts) != required:
        raise ValueError(
            "Stage211 phase gate requires easy, medium, hard, long, and "
            "supplemental_natural receipts."
        )
    return (
        [receipts[difficulty] for difficulty in STAGE211_AUDIO_CURRICULUM],
        receipts[STAGE211_SUPPLEMENTAL_DIFFICULTY],
    )


def _jsonl_records(
    path: Path,
    *,
    language: str,
    reference_keys: tuple[str, ...],
) -> dict[str, str]:
    records: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            utt_id = str(row.get("utt_id") or row.get("id") or row.get("key") or "")
            if not utt_id:
                raise ValueError(f"Missing utterance id at {path}:{line_number}")
            if utt_id in records:
                raise ValueError(f"Duplicate utterance id {utt_id!r} at {path}:{line_number}")
            reference = next(
                (str(row[key]) for key in reference_keys if row.get(key) is not None),
                None,
            )
            if reference is None:
                raise ValueError(f"Missing reference text for {utt_id!r} at {path}:{line_number}")
            records[utt_id] = normalize_asr_text_for_metrics(
                reference,
                language=language,
                normalization="ctc",
                strip_language_confirmation=STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
            )
    return records


def _enrich_public_benchmark(
    report: dict[str, Any],
    *,
    manifest_dir: Path,
    require_student_prediction_receipt: bool = False,
) -> dict[str, Any]:
    return replay_stage211_public_comparison(
        report,
        manifest_paths={
            dataset: (manifest_dir / f"{dataset}.jsonl").resolve()
            for dataset in STAGE211_PUBLIC_BENCHMARKS
        },
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        require_student_prediction_receipt=require_student_prediction_receipt,
    )


def _build_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    max_dataset_regression: float = 0.03,
) -> dict[str, Any]:
    return build_stage211_public_progress(
        baseline=baseline,
        candidate=candidate,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        max_dataset_regression=max_dataset_regression,
    )


def _rebuild_alignment_gate(
    *,
    phase: str,
    alignment_report: dict[str, Any],
    baseline_report_path: Path,
    candidate_report_path: Path,
    phase_init_checkpoint: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    stratified_summary_path: Path | None = None
    recorded_path = alignment_report.get("stratified_summary_path")
    recorded_sha256 = alignment_report.get("stratified_summary_sha256")
    if recorded_path is not None or recorded_sha256 is not None:
        stratified_summary_path = Path(str(recorded_path or "")).resolve()
        if (
            not stratified_summary_path.is_file()
            or sha256_file(stratified_summary_path) != recorded_sha256
        ):
            raise ValueError("Stage211 alignment stratified summary is missing or changed.")
    if phase == "logits":
        return build_logits_alignment_gate(
            baseline_report_path=baseline_report_path,
            candidate_report_path=candidate_report_path,
            baseline_checkpoint_path=phase_init_checkpoint,
            checkpoint_path=checkpoint_path,
            stratified_summary_path=stratified_summary_path,
        )
    return build_hidden_alignment_gate(
        phase=phase,
        baseline_report_path=baseline_report_path,
        candidate_report_path=candidate_report_path,
        baseline_checkpoint_path=phase_init_checkpoint,
        checkpoint_path=checkpoint_path,
        stratified_summary_path=stratified_summary_path,
    )


def build_phase_gate(
    *,
    phase: str,
    checkpoint_path: Path,
    public_comparison_report_path: Path,
    manifest_dir: Path,
    coverage_receipt_paths: list[Path],
    preflight_smoke_marker_path: Path,
    global_dedup_manifest_path: Path,
    loaded_manifest_receipt_path: Path,
    alignment_report_path: Path | None,
    post_coverage_correction_receipt_paths: list[Path] | None = None,
    baseline_public_comparison_report_path: Path | None = None,
    baseline_public_reuse_receipt_path: Path | None = None,
    initialization_receipt_path: Path | None = None,
    nano_public_baseline_receipt_path: Path = (DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT),
    public_overlap_receipt_path: Path = DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    promotion_policy: str = STAGE211_PROMOTION_POLICY_STRICT,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    public_comparison_report_path = public_comparison_report_path.resolve()
    public_report = _load_json(
        public_comparison_report_path,
        label="Stage211 public comparison report",
    )
    recorded_public_checkpoint = Path(
        str(public_report.get("student_checkpoint_path") or "")
    ).resolve()
    if recorded_public_checkpoint != checkpoint_path:
        raise ValueError("Stage211 public comparison checkpoint path mismatch.")
    if public_report.get("student_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Stage211 public comparison checkpoint SHA-256 mismatch.")
    coverage, supplemental_coverage = _parse_coverage_receipts(
        coverage_receipt_paths,
        phase=phase,
    )
    correction_receipts = load_stage211_post_coverage_correction_receipts(
        list(post_coverage_correction_receipt_paths or []),
        phase=phase,
    )
    phase_init_checkpoint = Path(str(coverage[0].get("init_checkpoint_path") or "")).resolve()
    if not phase_init_checkpoint.is_file() or sha256_file(phase_init_checkpoint) != coverage[0].get(
        "init_checkpoint_sha256"
    ):
        raise ValueError("Stage211 phase initialization checkpoint is missing or changed.")
    preflight_smoke_marker_path = preflight_smoke_marker_path.resolve()
    preflight_smoke = {
        "marker_path": str(preflight_smoke_marker_path),
        "marker_sha256": sha256_file(preflight_smoke_marker_path),
    }
    validate_stage211_full_profile_smoke_binding(
        preflight_smoke,
        phase=phase,
        init_checkpoint=phase_init_checkpoint,
        easy_manifest=Path(str(coverage[0].get("bucket_manifest_path") or "")).resolve(),
    )
    global_dedup_manifest_path = global_dedup_manifest_path.expanduser().resolve()
    validate_stage211_global_dedup_manifest(global_dedup_manifest_path)
    loaded_manifest_receipt_path = loaded_manifest_receipt_path.expanduser().resolve()
    loaded_manifest_receipt = validate_stage211_loaded_manifest_receipt(
        loaded_manifest_receipt_path,
        expected_global_dedup_manifest=global_dedup_manifest_path,
    )
    loaded_by_difficulty = {
        str(segment.get("difficulty") or ""): segment
        for segment in loaded_manifest_receipt["segments"]
    }
    for segment in coverage:
        difficulty = str(segment.get("difficulty") or "")
        loaded_segment = loaded_by_difficulty.get(difficulty)
        if not isinstance(loaded_segment, dict):
            raise ValueError(f"Stage211 phase coverage lacks {difficulty} loader provenance.")
        if Path(str(segment.get("bucket_manifest_path") or "")).expanduser().resolve() != Path(
            str(loaded_segment.get("runtime_manifest_path") or "")
        ).expanduser().resolve() or str(segment.get("bucket_manifest_sha256") or "") != str(
            loaded_segment.get("runtime_manifest_sha256") or ""
        ):
            raise ValueError(
                f"Stage211 {difficulty} phase coverage does not use the audited runtime manifest."
            )
    alignment_gate_passed = False
    alignment_record: dict[str, Any] | None = None
    if alignment_report_path is not None:
        alignment_report_path = alignment_report_path.resolve()
        alignment_report = _load_json(
            alignment_report_path,
            label="Stage211 alignment gate report",
        )
        alignment_decision = alignment_report.get("gate_passed")
        if not isinstance(alignment_decision, bool):
            raise ValueError("Stage211 alignment report lacks a boolean decision.")
        if alignment_report.get("phase") != phase:
            raise ValueError("Stage211 alignment report phase mismatch.")
        expected_artifact = (
            "logits_alignment_gate" if phase == "logits" else "hidden_alignment_gate"
        )
        if (
            alignment_report.get("schema_version") != 1
            or alignment_report.get("pipeline") != "stage211"
            or alignment_report.get("artifact") != expected_artifact
        ):
            raise ValueError("Stage211 alignment report artifact mismatch.")
        alignment_checkpoint = Path(str(alignment_report.get("checkpoint_path") or "")).resolve()
        if alignment_checkpoint != checkpoint_path:
            raise ValueError("Stage211 alignment report checkpoint path mismatch.")
        if alignment_report.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError("Stage211 alignment report checkpoint SHA-256 mismatch.")
        if Path(
            str(alignment_report.get("baseline_checkpoint_path") or "")
        ).resolve() != phase_init_checkpoint or alignment_report.get(
            "baseline_checkpoint_sha256"
        ) != sha256_file(phase_init_checkpoint):
            raise ValueError("Stage211 alignment report baseline checkpoint mismatch.")
        baseline_alignment_report_path = Path(
            str(alignment_report.get("baseline_report_path") or "")
        ).resolve()
        candidate_alignment_report_path = Path(
            str(alignment_report.get("candidate_report_path") or "")
        ).resolve()
        baseline_alignment_source = _load_json(
            baseline_alignment_report_path,
            label="Stage211 alignment baseline source report",
        )
        if (
            Path(str(baseline_alignment_source.get("train_config_path") or "")).resolve()
            != Path(str(coverage[0]["train_config_path"])).resolve()
            or baseline_alignment_source.get("train_config_sha256")
            != coverage[0]["train_config_sha256"]
        ):
            raise ValueError(
                "Stage211 alignment pair does not bind the easy-segment phase train config."
            )
        rebuilt_alignment_report = _rebuild_alignment_gate(
            phase=phase,
            alignment_report=alignment_report,
            baseline_report_path=baseline_alignment_report_path,
            candidate_report_path=candidate_alignment_report_path,
            phase_init_checkpoint=phase_init_checkpoint,
            checkpoint_path=checkpoint_path,
        )
        if rebuilt_alignment_report != alignment_report:
            raise ValueError("Stage211 alignment report does not match its bound source reports.")
        alignment_gate_passed = alignment_decision
        alignment_record = {
            "path": str(alignment_report_path),
            "sha256": sha256_file(alignment_report_path),
            "artifact": expected_artifact,
        }
    else:
        raise ValueError(f"Stage211 {phase} requires an independent alignment gate report.")

    benchmark = _enrich_public_benchmark(
        public_report,
        manifest_dir=manifest_dir.resolve(),
        require_student_prediction_receipt=True,
    )
    public_overlap_receipt_path = public_overlap_receipt_path.expanduser().resolve()
    public_overlap_binding = {
        "receipt_path": str(public_overlap_receipt_path),
        "receipt_sha256": sha256_file(public_overlap_receipt_path),
    }
    validate_stage211_public_overlap_binding(
        public_overlap_binding,
        public_benchmark=benchmark,
    )
    teacher_sha256_values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "") for segment in coverage
    }
    teacher_sha256_values.add(
        str(supplemental_coverage.get("nano_teacher_checkpoint_sha256") or "")
    )
    if len(teacher_sha256_values) != 1:
        raise ValueError(
            "Stage211 phase coverage does not bind one Nano teacher checkpoint SHA-256."
        )
    nano_teacher_checkpoint_sha256 = next(iter(teacher_sha256_values))
    nano_public_baseline_receipt_path = nano_public_baseline_receipt_path.expanduser().resolve()
    nano_public_baseline = validate_stage211_nano_public_baseline_receipt(
        nano_public_baseline_receipt_path,
        expected_nano_checkpoint_sha256=nano_teacher_checkpoint_sha256,
        public_benchmark=benchmark,
    )
    public_progress: dict[str, Any] | None = None
    baseline_public_record: dict[str, str] | None = None
    if baseline_public_comparison_report_path is None:
        raise ValueError(f"Stage211 {phase} requires a baseline public comparison report.")
    baseline_public_comparison_report_path = baseline_public_comparison_report_path.resolve()
    baseline_public_report = _load_json(
        baseline_public_comparison_report_path,
        label="Stage211 baseline public comparison report",
    )
    if Path(
        str(baseline_public_report.get("student_checkpoint_path") or "")
    ).resolve() != phase_init_checkpoint or baseline_public_report.get(
        "student_checkpoint_sha256"
    ) != sha256_file(phase_init_checkpoint):
        raise ValueError("Stage211 baseline public report does not bind the phase initialization.")
    baseline_benchmark = _enrich_public_benchmark(
        baseline_public_report,
        manifest_dir=manifest_dir.resolve(),
        require_student_prediction_receipt=phase in {"block", "logits"},
    )
    baseline_public_provenance = build_stage211_phase_baseline_public_provenance(
        phase=phase,
        baseline_public_report_path=baseline_public_comparison_report_path,
        baseline_public_report=baseline_public_report,
        baseline_public_benchmark=baseline_benchmark,
        phase_init_checkpoint=phase_init_checkpoint,
        nano_teacher_checkpoint_sha256=nano_teacher_checkpoint_sha256,
        calibration_reuse_receipt_path=baseline_public_reuse_receipt_path,
        initialization_receipt_path=initialization_receipt_path,
    )
    public_progress = _build_public_progress(
        baseline=baseline_benchmark,
        candidate=benchmark,
    )
    public_progress_gate_passed = bool(public_progress["gate_passed"])
    baseline_public_record = {
        "path": str(baseline_public_comparison_report_path),
        "sha256": sha256_file(baseline_public_comparison_report_path),
    }
    trajectory_retention = build_stage211_trajectory_retention_gate(
        phase=phase,
        segments=coverage,
        supplemental_segment=supplemental_coverage,
        post_coverage_corrections=correction_receipts,
        checkpoint_path=checkpoint_path,
    )
    trajectory_retention_gate_passed = bool(trajectory_retention["gate_passed"])
    step_eval_cadence = build_stage211_step_eval_cadence(
        phase=phase,
        segments=coverage,
        supplemental_segment=supplemental_coverage,
        post_coverage_corrections=correction_receipts,
    )
    strict_metric_gate_passed = stage211_phase_gate_decision(
        phase=phase,
        alignment_gate_passed=alignment_gate_passed,
        public_progress_gate_passed=public_progress_gate_passed,
        trajectory_retention_gate_passed=trajectory_retention_gate_passed,
        all_datasets_pass=benchmark.get("all_datasets_pass") is True,
    )
    alignment_loss_nondivergence = build_stage211_alignment_loss_nondivergence(
        baseline_source=baseline_alignment_source,
        candidate_source=_load_json(
            candidate_alignment_report_path,
            label="Stage211 alignment candidate source report",
        ),
    )
    if promotion_policy == STAGE211_PROMOTION_POLICY_STRICT:
        metric_gate_passed = strict_metric_gate_passed
    elif promotion_policy == STAGE211_PROMOTION_POLICY_COVERAGE_NON_DIVERGENT:
        if phase != "mixer":
            raise ValueError("Stage211 coverage/non-divergence promotion is restricted to Mixer.")
        if correction_receipts:
            raise ValueError(
                "Stage211 Mixer coverage/non-divergence promotion must precede corrections."
            )
        metric_gate_passed = bool(alignment_loss_nondivergence["gate_passed"])
    else:
        raise ValueError(f"Unsupported Stage211 promotion policy: {promotion_policy!r}")
    correction_round_promotion = build_stage211_correction_round_promotion_gate(
        len(correction_receipts)
    )
    gate_passed = metric_gate_passed and correction_round_promotion["gate_passed"]
    final_checkpoint_sha256 = sha256_file(checkpoint_path)
    report = {
        "schema_version": STAGE211_PHASE_GATE_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "phase_gate",
        "phase": phase,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": final_checkpoint_sha256,
        "gate_passed": gate_passed,
        "promotion_policy": promotion_policy,
        "metric_gate_passed": metric_gate_passed,
        "strict_metric_gate_passed": strict_metric_gate_passed,
        "alignment_loss_nondivergence": alignment_loss_nondivergence,
        "correction_round_promotion": correction_round_promotion,
        "alignment_gate_passed": alignment_gate_passed,
        "public_progress_gate_passed": public_progress_gate_passed,
        "trajectory_retention_gate_passed": trajectory_retention_gate_passed,
        "trajectory_retention": trajectory_retention,
        "step_eval_cadence": step_eval_cadence,
        "preflight_smoke": preflight_smoke,
        "global_dedup_manifest_path": str(global_dedup_manifest_path),
        "global_dedup_manifest_sha256": sha256_file(global_dedup_manifest_path),
        "loaded_manifest_receipt_path": str(loaded_manifest_receipt_path),
        "loaded_manifest_receipt_sha256": sha256_file(loaded_manifest_receipt_path),
        "alignment_report": alignment_record,
        "baseline_public_comparison_report": baseline_public_record,
        "baseline_public_provenance": baseline_public_provenance,
        "public_progress": public_progress,
        "full_data_coverage": build_stage211_full_data_coverage(
            phase=phase,
            segments=coverage,
            checkpoint_path=checkpoint_path,
            post_coverage_corrections=correction_receipts,
            supplemental_segment=supplemental_coverage,
        ),
        "public_comparison_report_path": str(public_comparison_report_path),
        "public_comparison_report_sha256": sha256_file(public_comparison_report_path),
        "public_overlap": public_overlap_binding,
        "nano_public_baseline_receipt_path": str(nano_public_baseline_receipt_path),
        "nano_public_baseline_receipt_sha256": sha256_file(nano_public_baseline_receipt_path),
        "nano_public_baseline_checkpoint_sha256": nano_public_baseline["nano_checkpoint_sha256"],
        "public_benchmark": benchmark,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a strict Stage211 full-data plus public-WER/CER phase gate."
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preflight-smoke-marker", type=Path, required=True)
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
    parser.add_argument("--public-comparison-report", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument(
        "--coverage-receipt",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--alignment-report", type=Path, default=None)
    parser.add_argument(
        "--post-coverage-correction-receipt",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--baseline-public-comparison-report",
        type=Path,
        default=None,
    )
    parser.add_argument("--baseline-public-reuse-receipt", type=Path, default=None)
    parser.add_argument("--initialization-receipt", type=Path, default=None)
    parser.add_argument(
        "--nano-public-baseline-receipt",
        type=Path,
        default=DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    )
    parser.add_argument(
        "--public-overlap-receipt",
        type=Path,
        default=DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--promotion-policy",
        choices=(
            STAGE211_PROMOTION_POLICY_STRICT,
            STAGE211_PROMOTION_POLICY_COVERAGE_NON_DIVERGENT,
        ),
        default=STAGE211_PROMOTION_POLICY_STRICT,
    )
    args = parser.parse_args()

    report = build_phase_gate(
        phase=str(args.phase),
        checkpoint_path=args.checkpoint,
        public_comparison_report_path=args.public_comparison_report,
        manifest_dir=args.manifest_dir,
        coverage_receipt_paths=list(args.coverage_receipt),
        preflight_smoke_marker_path=args.preflight_smoke_marker,
        global_dedup_manifest_path=args.global_dedup_manifest,
        loaded_manifest_receipt_path=args.loaded_manifest_receipt,
        post_coverage_correction_receipt_paths=list(args.post_coverage_correction_receipt),
        alignment_report_path=args.alignment_report,
        baseline_public_comparison_report_path=(args.baseline_public_comparison_report),
        baseline_public_reuse_receipt_path=args.baseline_public_reuse_receipt,
        initialization_receipt_path=args.initialization_receipt,
        nano_public_baseline_receipt_path=args.nano_public_baseline_receipt,
        public_overlap_receipt_path=args.public_overlap_receipt,
        promotion_policy=str(args.promotion_policy),
    )
    output_path = args.output.resolve()
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    write_immutable_text(output_path, rendered, label="phase gate")
    validate_stage211_phase_gate_report(
        output_path,
        expected_phase=str(args.phase),
        checkpoint_path=args.checkpoint,
        require_passed=False,
    )
    print(
        f"phase_gate={output_path} phase={args.phase} gate_passed={report['gate_passed']} "
        f"checkpoint_sha256={report['checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
