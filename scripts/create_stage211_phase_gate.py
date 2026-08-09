from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval import normalize_asr_text_for_metrics
from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES,
    STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
    STAGE211_AUDIO_TOTAL_HOURS,
    STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
    STAGE211_AUDIO_TOTAL_ROWS,
    STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES,
    STAGE211_PHASE_GATE_SCHEMA_VERSION,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_phase_gate_report,
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


def _parse_coverage_receipts(values: list[Path], *, phase: str) -> list[dict[str, Any]]:
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
            or difficulty not in STAGE211_AUDIO_CURRICULUM
        ):
            raise ValueError(f"Invalid Stage211 curriculum coverage receipt: {resolved}")
        if difficulty in receipts:
            raise ValueError(f"Duplicate Stage211 coverage receipt for {difficulty}.")
        receipts[difficulty] = {
            **receipt,
            "receipt_path": str(resolved),
            "receipt_sha256": sha256_file(resolved),
        }
    if set(receipts) != set(STAGE211_AUDIO_CURRICULUM):
        raise ValueError("Stage211 phase gate requires easy, medium, hard, and long receipts.")
    return [receipts[difficulty] for difficulty in STAGE211_AUDIO_CURRICULUM]


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
            )
    return records


def _enrich_public_benchmark(
    report: dict[str, Any],
    *,
    manifest_dir: Path,
) -> dict[str, Any]:
    if report.get("decode") != "greedy_ctc" or report.get("normalization") != "ctc":
        raise ValueError("Stage211 public comparison must use greedy_ctc and ctc normalization.")
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Stage211 public comparison results must be a list.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 public comparison dataset set is incomplete or unexpected.")

    results: list[dict[str, Any]] = []
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        result = dict(by_dataset[dataset])
        language = str(STAGE211_PUBLIC_BENCHMARKS[dataset]["language"])
        manifest_path = (manifest_dir / f"{dataset}.jsonl").resolve()
        nano_path = Path(str(result.pop("nano_prediction_path", "") or "")).resolve()
        student_path = Path(str(result.pop("student_prediction_path", "") or "")).resolve()
        for label, path in (
            ("manifest", manifest_path),
            ("Nano prediction", nano_path),
            ("student prediction", student_path),
        ):
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"Stage211 {dataset} {label} is missing or empty: {path}")
        manifest_records = _jsonl_records(
            manifest_path,
            language=language,
            reference_keys=("text", "transcript", "ref_text", "reference"),
        )
        nano_records = _jsonl_records(
            nano_path,
            language=language,
            reference_keys=("ref_text", "reference", "text", "transcript"),
        )
        student_records = _jsonl_records(
            student_path,
            language=language,
            reference_keys=("ref_text", "reference", "text", "transcript"),
        )
        expected_samples = int(STAGE211_PUBLIC_BENCHMARKS[dataset]["samples"])
        if (
            len(manifest_records) != expected_samples
            or set(nano_records) != set(manifest_records)
            or set(student_records) != set(manifest_records)
        ):
            raise ValueError(
                f"Stage211 {dataset} manifest/Nano/student utterance coverage mismatch: "
                f"manifest={len(manifest_records)} nano={len(nano_records)} "
                f"student={len(student_records)} expected={expected_samples}"
            )
        reference_mismatches = sum(
            nano_records[utt_id] != reference or student_records[utt_id] != reference
            for utt_id, reference in manifest_records.items()
        )
        if reference_mismatches:
            raise ValueError(
                f"Stage211 {dataset} normalized references differ from the manifest: "
                f"mismatches={reference_mismatches}"
            )
        results.append(
            {
                **result,
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "nano_prediction_path": str(nano_path),
                "nano_prediction_sha256": sha256_file(nano_path),
                "student_prediction_path": str(student_path),
                "student_prediction_sha256": sha256_file(student_path),
            }
        )
    return {
        **report,
        "all_datasets_complete": True,
        "results": results,
    }


def _build_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    max_dataset_regression: float = 0.03,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(candidate_results):
        raise ValueError("Stage211 baseline/candidate public datasets differ.")
    rows: list[dict[str, Any]] = []
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
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
                "within_regression_limit": (
                    candidate_error <= baseline_error + max_dataset_regression
                ),
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
            raise ValueError(
                "Stage211 alignment stratified summary is missing or changed."
            )
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
    alignment_report_path: Path | None,
    baseline_public_comparison_report_path: Path | None = None,
    nano_public_baseline_receipt_path: Path = (
        DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT
    ),
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
    coverage = _parse_coverage_receipts(coverage_receipt_paths, phase=phase)
    phase_init_checkpoint = Path(
        str(coverage[0].get("init_checkpoint_path") or "")
    ).resolve()
    if (
        not phase_init_checkpoint.is_file()
        or sha256_file(phase_init_checkpoint)
        != coverage[0].get("init_checkpoint_sha256")
    ):
        raise ValueError(
            "Stage211 phase initialization checkpoint is missing or changed."
        )
    alignment_gate_passed = False
    alignment_record: dict[str, Any] | None = None
    if alignment_report_path is not None:
        alignment_report_path = alignment_report_path.resolve()
        alignment_report = _load_json(
            alignment_report_path,
            label="Stage211 alignment gate report",
        )
        if alignment_report.get("gate_passed") is not True:
            raise ValueError("Stage211 alignment report does not record a passing decision.")
        if alignment_report.get("phase") != phase:
            raise ValueError("Stage211 alignment report phase mismatch.")
        expected_artifact = (
            "logits_alignment_gate"
            if phase == "logits"
            else "hidden_alignment_gate"
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
        if (
            Path(
                str(
                    alignment_report.get("baseline_checkpoint_path")
                    or ""
                )
            ).resolve()
            != phase_init_checkpoint
            or alignment_report.get("baseline_checkpoint_sha256")
            != sha256_file(phase_init_checkpoint)
        ):
            raise ValueError(
                "Stage211 alignment report baseline checkpoint mismatch."
            )
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
            Path(
                str(
                    baseline_alignment_source.get("train_config_path")
                    or ""
                )
            ).resolve()
            != Path(str(coverage[0]["train_config_path"])).resolve()
            or baseline_alignment_source.get("train_config_sha256")
            != coverage[0]["train_config_sha256"]
        ):
            raise ValueError(
                "Stage211 alignment pair does not bind the easy-segment "
                "phase train config."
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
            raise ValueError(
                "Stage211 alignment report does not match its bound source reports."
            )
        alignment_gate_passed = True
        alignment_record = {
            "path": str(alignment_report_path),
            "sha256": sha256_file(alignment_report_path),
            "artifact": expected_artifact,
        }
    else:
        raise ValueError(f"Stage211 {phase} requires an independent alignment gate report.")

    benchmark = _enrich_public_benchmark(public_report, manifest_dir=manifest_dir.resolve())
    teacher_sha256_values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "")
        for segment in coverage
    }
    if len(teacher_sha256_values) != 1:
        raise ValueError(
            "Stage211 phase coverage does not bind one Nano teacher checkpoint SHA-256."
        )
    nano_teacher_checkpoint_sha256 = next(iter(teacher_sha256_values))
    nano_public_baseline_receipt_path = (
        nano_public_baseline_receipt_path.expanduser().resolve()
    )
    nano_public_baseline = validate_stage211_nano_public_baseline_receipt(
        nano_public_baseline_receipt_path,
        expected_nano_checkpoint_sha256=nano_teacher_checkpoint_sha256,
        public_benchmark=benchmark,
    )
    public_progress: dict[str, Any] | None = None
    baseline_public_record: dict[str, str] | None = None
    public_progress_gate_passed = phase == "logits"
    if phase in {"mixer", "block"}:
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
            raise ValueError(
                "Stage211 baseline public report does not bind the phase initialization."
            )
        baseline_benchmark = _enrich_public_benchmark(
            baseline_public_report,
            manifest_dir=manifest_dir.resolve(),
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
    gate_passed = (
        alignment_gate_passed
        and public_progress_gate_passed
        and (phase != "logits" or benchmark.get("all_datasets_pass") is True)
    )
    final_checkpoint_sha256 = sha256_file(checkpoint_path)
    report = {
        "schema_version": STAGE211_PHASE_GATE_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "phase_gate",
        "phase": phase,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": final_checkpoint_sha256,
        "gate_passed": gate_passed,
        "alignment_gate_passed": alignment_gate_passed,
        "public_progress_gate_passed": public_progress_gate_passed,
        "alignment_report": alignment_record,
        "baseline_public_comparison_report": baseline_public_record,
        "public_progress": public_progress,
        "full_data_coverage": {
            "phase": phase,
            "complete": True,
            "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
            "total_hours": STAGE211_AUDIO_TOTAL_HOURS,
            "total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
            "total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
            "total_tail_padding_sample_exposures": (
                STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
            ),
            "total_executed_sample_exposures": (
                STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
            ),
            "segments": coverage,
            "final_checkpoint_path": str(checkpoint_path),
            "final_checkpoint_sha256": final_checkpoint_sha256,
        },
        "public_comparison_report_path": str(public_comparison_report_path),
        "public_comparison_report_sha256": sha256_file(public_comparison_report_path),
        "nano_public_baseline_receipt_path": str(
            nano_public_baseline_receipt_path
        ),
        "nano_public_baseline_receipt_sha256": sha256_file(
            nano_public_baseline_receipt_path
        ),
        "nano_public_baseline_checkpoint_sha256": nano_public_baseline[
            "nano_checkpoint_sha256"
        ],
        "public_benchmark": benchmark,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a strict Stage211 full-data plus public-WER/CER phase gate."
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
        "--baseline-public-comparison-report",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--nano-public-baseline-receipt",
        type=Path,
        default=DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_phase_gate(
        phase=str(args.phase),
        checkpoint_path=args.checkpoint,
        public_comparison_report_path=args.public_comparison_report,
        manifest_dir=args.manifest_dir,
        coverage_receipt_paths=list(args.coverage_receipt),
        alignment_report_path=args.alignment_report,
        baseline_public_comparison_report_path=(args.baseline_public_comparison_report),
        nano_public_baseline_receipt_path=args.nano_public_baseline_receipt,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output_path.is_file() and output_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different phase gate: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    validate_stage211_phase_gate_report(
        output_path,
        expected_phase=str(args.phase),
        checkpoint_path=args.checkpoint,
    )
    print(
        f"phase_gate={output_path} phase={args.phase} gate_passed={report['gate_passed']} "
        f"checkpoint_sha256={report['checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
