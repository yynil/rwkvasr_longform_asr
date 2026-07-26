from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval import normalize_asr_text_for_metrics
from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
    STAGE211_AUDIO_TOTAL_HOURS,
    STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
    STAGE211_AUDIO_TOTAL_ROWS,
    STAGE211_PHASE_GATE_SCHEMA_VERSION,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_phase_gate_report,
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
                (
                    str(row[key])
                    for key in reference_keys
                    if row.get(key) is not None
                ),
                None,
            )
            if reference is None:
                raise ValueError(
                    f"Missing reference text for {utt_id!r} at {path}:{line_number}"
                )
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
        str(result.get("dataset")): result
        for result in raw_results
        if isinstance(result, dict)
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
            nano_records[utt_id] != reference
            or student_records[utt_id] != reference
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


def build_phase_gate(
    *,
    phase: str,
    checkpoint_path: Path,
    public_comparison_report_path: Path,
    manifest_dir: Path,
    coverage_receipt_paths: list[Path],
    alignment_report_path: Path | None,
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
    alignment_gate_passed = phase == "logits"
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
        alignment_checkpoint = Path(
            str(alignment_report.get("checkpoint_path") or "")
        ).resolve()
        if alignment_checkpoint != checkpoint_path:
            raise ValueError("Stage211 alignment report checkpoint path mismatch.")
        if alignment_report.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise ValueError("Stage211 alignment report checkpoint SHA-256 mismatch.")
        alignment_gate_passed = True
        alignment_record = {
            "path": str(alignment_report_path),
            "sha256": sha256_file(alignment_report_path),
        }
    elif phase in {"mixer", "block"}:
        raise ValueError(f"Stage211 {phase} requires an independent alignment gate report.")

    benchmark = _enrich_public_benchmark(public_report, manifest_dir=manifest_dir.resolve())
    gate_passed = alignment_gate_passed and (
        phase in {"mixer", "block"} or benchmark.get("all_datasets_pass") is True
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
        "alignment_report": alignment_record,
        "full_data_coverage": {
            "phase": phase,
            "complete": True,
            "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
            "total_hours": STAGE211_AUDIO_TOTAL_HOURS,
            "total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
            "total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
            "segments": coverage,
            "final_checkpoint_path": str(checkpoint_path),
            "final_checkpoint_sha256": final_checkpoint_sha256,
        },
        "public_comparison_report_path": str(public_comparison_report_path),
        "public_comparison_report_sha256": sha256_file(public_comparison_report_path),
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_phase_gate(
        phase=str(args.phase),
        checkpoint_path=args.checkpoint,
        public_comparison_report_path=args.public_comparison_report,
        manifest_dir=args.manifest_dir,
        coverage_receipt_paths=list(args.coverage_receipt),
        alignment_report_path=args.alignment_report,
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
