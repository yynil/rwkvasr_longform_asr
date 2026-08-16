from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from rwkvasr.eval import (
    compute_text_error_stats,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)
from rwkvasr.eval.stage211_gate import STAGE211_PUBLIC_BENCHMARKS, sha256_file
from rwkvasr.eval.stage211_public_overlap import validate_stage211_public_overlap_receipt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAP_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "public_train_overlap_v2"
    / "receipt.json"
)
DEFAULT_NANO_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
DEFAULT_CALIBRATION_PUBLIC = (
    Path.home() / "rwkvasr_eval" / "stage211_calibration_selected_full" / "public"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_clean_v2"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} is not an object.")
            utt_id = str(row.get("utt_id") or "").strip()
            if not utt_id or utt_id in seen:
                raise ValueError(f"{label} has an invalid/duplicate utt_id={utt_id!r}.")
            seen.add(utt_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rendered = "".join(
        json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        for row in rows
    )
    _atomic_write_text(path, rendered)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _compact_metrics(path: Path, *, language: str) -> dict[str, float | int]:
    metrics = compute_text_error_stats(path, language=language, normalization="ctc")
    return {
        "avg_cer": float(metrics["avg_cer"]),
        "avg_wer": float(metrics["avg_wer"]),
        "sample_count": int(metrics["sample_count"]),
    }


def _nano_diagnostics(
    rows: list[dict[str, Any]],
    *,
    language: str,
) -> tuple[dict[str, float | int], float]:
    pred_units = 0
    ref_units = 0
    total_frames = 0
    blank_top1_sum = 0.0
    blank_probability_sum = 0.0
    total_audio_sec = 0.0
    tokenize = tokenize_for_wer if language == "en" else tokenize_for_cer
    for row in rows:
        normalized_pred = normalize_asr_text_for_metrics(
            str(row.get("pred_text") or ""),
            language=language,
            normalization="ctc",
        )
        normalized_ref = normalize_asr_text_for_metrics(
            str(row.get("ref_text") or ""),
            language=language,
            normalization="ctc",
        )
        debug = row.get("debug")
        if not isinstance(debug, dict):
            raise ValueError(f"Nano prediction lacks diagnostics: {row.get('utt_id')}")
        pred_units += len(tokenize(normalized_pred))
        ref_units += len(tokenize(normalized_ref))
        total_frames += int(debug.get("logit_length", 0))
        blank_top1_sum += float(debug.get("blank_top1_ratio", 0.0))
        blank_probability_sum += float(debug.get("avg_blank_prob", 0.0))
        total_audio_sec += float(row.get("duration_sec") or 0.0)
    sample_count = len(rows)
    return (
        {
            "mean_blank_probability": blank_probability_sum / sample_count,
            "mean_blank_top1_ratio": blank_top1_sum / sample_count,
            "mean_logit_frames": total_frames / sample_count,
            "pred_ref_unit_ratio": pred_units / max(1, ref_units),
            "pred_units": pred_units,
            "ref_units": ref_units,
        },
        total_audio_sec,
    )


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    allowed_ids: set[str],
    label: str,
) -> list[dict[str, Any]]:
    source_ids = {str(row["utt_id"]) for row in rows}
    missing = allowed_ids - source_ids
    if missing:
        raise ValueError(f"{label} lacks {len(missing)} required clean utterances.")
    filtered = [row for row in rows if str(row["utt_id"]) in allowed_ids]
    if len(filtered) != len(allowed_ids):
        raise ValueError(f"{label} clean coverage mismatch.")
    return filtered


def derive_clean_public_eval(
    *,
    overlap_receipt_path: Path,
    manifest_dir: Path,
    nano_root: Path,
    calibration_public_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    overlap_receipt_path = overlap_receipt_path.expanduser().resolve()
    manifest_dir = manifest_dir.expanduser().resolve()
    nano_root = nano_root.expanduser().resolve()
    calibration_public_dir = calibration_public_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    overlap = validate_stage211_public_overlap_receipt(overlap_receipt_path)
    clean_cv_manifest_path = Path(str(overlap["outputs"]["clean_manifest"]["path"])).resolve()
    clean_cv_rows = _load_jsonl(clean_cv_manifest_path, label="clean Common Voice manifest")
    clean_cv_ids = {str(row["utt_id"]) for row in clean_cv_rows}
    parent_cv_manifest = (manifest_dir / "commonvoice_en_test.jsonl").resolve()
    if sha256_file(parent_cv_manifest) != overlap["source_bindings"]["public_manifest_sha256"]:
        raise ValueError("The parent Common Voice manifest differs from the overlap source.")

    output_manifest_dir = output_root / "manifests"
    output_nano_prediction_dir = output_root / "nano_2512" / "predictions"
    output_nano_report_dir = output_root / "nano_2512" / "reports"
    output_calibration_prediction_dir = output_root / "calibration" / "predictions"
    results: list[dict[str, Any]] = []
    for dataset, benchmark in STAGE211_PUBLIC_BENCHMARKS.items():
        language = str(benchmark["language"])
        parent_manifest = (manifest_dir / f"{dataset}.jsonl").resolve()
        parent_nano_prediction = (nano_root / "predictions" / f"{dataset}.ctc.jsonl").resolve()
        parent_nano_report = (nano_root / "reports" / f"{dataset}.json").resolve()
        parent_calibration_prediction = (
            calibration_public_dir / "predictions" / f"{dataset}.ctc.jsonl"
        ).resolve()
        for source in (
            parent_manifest,
            parent_nano_prediction,
            parent_nano_report,
            parent_calibration_prediction,
        ):
            if not source.is_file():
                raise ValueError(f"Stage211 public parent artifact is unavailable: {source}")

        derived_manifest = output_manifest_dir / f"{dataset}.jsonl"
        derived_nano_prediction = output_nano_prediction_dir / f"{dataset}.ctc.jsonl"
        derived_nano_report = output_nano_report_dir / f"{dataset}.json"
        derived_calibration_prediction = output_calibration_prediction_dir / f"{dataset}.ctc.jsonl"
        if dataset == "commonvoice_en_test":
            manifest_rows = clean_cv_rows
            nano_rows = _filter_rows(
                _load_jsonl(parent_nano_prediction, label="parent Nano CV predictions"),
                allowed_ids=clean_cv_ids,
                label="parent Nano CV predictions",
            )
            calibration_rows = _filter_rows(
                _load_jsonl(
                    parent_calibration_prediction,
                    label="parent calibration CV predictions",
                ),
                allowed_ids=clean_cv_ids,
                label="parent calibration CV predictions",
            )
            excluded_rows = int(overlap["coverage"]["excluded_public_rows"])
        else:
            manifest_rows = _load_jsonl(parent_manifest, label=f"{dataset} manifest")
            nano_rows = _load_jsonl(
                parent_nano_prediction,
                label=f"{dataset} Nano predictions",
            )
            calibration_rows = _load_jsonl(
                parent_calibration_prediction,
                label=f"{dataset} calibration predictions",
            )
            excluded_rows = 0

        manifest_ids = {str(row["utt_id"]) for row in manifest_rows}
        if {str(row["utt_id"]) for row in nano_rows} != manifest_ids:
            raise ValueError(f"{dataset} derived Nano prediction coverage mismatch.")
        if {str(row["utt_id"]) for row in calibration_rows} != manifest_ids:
            raise ValueError(f"{dataset} derived calibration prediction coverage mismatch.")
        _write_jsonl(derived_manifest, manifest_rows)
        _write_jsonl(derived_nano_prediction, nano_rows)
        _write_jsonl(derived_calibration_prediction, calibration_rows)

        parent_report = _load_json(parent_nano_report, label=f"{dataset} Nano report")
        diagnostics, total_audio_sec = _nano_diagnostics(nano_rows, language=language)
        derived_report = dict(parent_report)
        derived_report.update(
            {
                "audio_hours": total_audio_sec / 3600.0,
                "derivation": {
                    "excluded_rows": excluded_rows,
                    "method": "filter_parent_predictions_by_clean_manifest_utt_id",
                    "overlap_receipt_path": str(overlap_receipt_path),
                    "overlap_receipt_sha256": sha256_file(overlap_receipt_path),
                    "parent_manifest_path": str(parent_manifest),
                    "parent_manifest_sha256": sha256_file(parent_manifest),
                    "parent_prediction_path": str(parent_nano_prediction),
                    "parent_prediction_sha256": sha256_file(parent_nano_prediction),
                    "parent_report_path": str(parent_nano_report),
                    "parent_report_sha256": sha256_file(parent_nano_report),
                    "without_model_reinference": True,
                },
                "diagnostics": diagnostics,
                "manifest_path": str(derived_manifest.resolve()),
                "metrics": _compact_metrics(derived_nano_prediction, language=language),
                "predictions_path": str(derived_nano_prediction.resolve()),
                "rtf": None,
                "sample_count": len(manifest_rows),
            }
        )
        _write_json(derived_nano_report, derived_report)
        results.append(
            {
                "dataset": dataset,
                "excluded_rows": excluded_rows,
                "language": language,
                "parent": {
                    "calibration_prediction_path": str(parent_calibration_prediction),
                    "calibration_prediction_sha256": sha256_file(parent_calibration_prediction),
                    "manifest_path": str(parent_manifest),
                    "manifest_sha256": sha256_file(parent_manifest),
                    "nano_prediction_path": str(parent_nano_prediction),
                    "nano_prediction_sha256": sha256_file(parent_nano_prediction),
                    "nano_report_path": str(parent_nano_report),
                    "nano_report_sha256": sha256_file(parent_nano_report),
                },
                "derived": {
                    "calibration_prediction_path": str(derived_calibration_prediction.resolve()),
                    "calibration_prediction_sha256": sha256_file(derived_calibration_prediction),
                    "manifest_path": str(derived_manifest.resolve()),
                    "manifest_sha256": sha256_file(derived_manifest),
                    "nano_prediction_path": str(derived_nano_prediction.resolve()),
                    "nano_prediction_sha256": sha256_file(derived_nano_prediction),
                    "nano_report_path": str(derived_nano_report.resolve()),
                    "nano_report_sha256": sha256_file(derived_nano_report),
                },
                "sample_count": len(manifest_rows),
            }
        )

    receipt = {
        "artifact": "stage211_clean_public_eval_derivation",
        "complete": True,
        "datasets": results,
        "overlap_receipt_path": str(overlap_receipt_path),
        "overlap_receipt_sha256": sha256_file(overlap_receipt_path),
        "pipeline": "stage211",
        "schema_version": 1,
        "total_samples": sum(int(row["sample_count"]) for row in results),
    }
    _write_json(output_root / "derivation_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the Stage211 clean public suite without model reinference."
    )
    parser.add_argument("--overlap-receipt", type=Path, default=DEFAULT_OVERLAP_RECEIPT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--nano-root", type=Path, default=DEFAULT_NANO_ROOT)
    parser.add_argument(
        "--calibration-public-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_PUBLIC,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = derive_clean_public_eval(
        overlap_receipt_path=args.overlap_receipt,
        manifest_dir=args.manifest_dir,
        nano_root=args.nano_root,
        calibration_public_dir=args.calibration_public_dir,
        output_root=args.output_root,
    )
    commonvoice = next(
        row for row in receipt["datasets"] if row["dataset"] == "commonvoice_en_test"
    )
    print(
        "[stage211-clean-public] "
        f"total_samples={receipt['total_samples']} "
        f"commonvoice_samples={commonvoice['sample_count']} "
        f"receipt={args.output_root.expanduser().resolve() / 'derivation_receipt.json'}"
    )


if __name__ == "__main__":
    main()
