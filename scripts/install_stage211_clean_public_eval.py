from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
)
from rwkvasr.eval.stage211_public_overlap import validate_stage211_public_overlap_receipt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEAN_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_clean_v2"
DEFAULT_NANO_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
DEFAULT_CALIBRATION_ROOT = Path.home() / "rwkvasr_eval" / "stage211_calibration_selected_full"
DEFAULT_ARCHIVE_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_pre_quote_repair_archive_v2"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _validate_derivation(clean_root: Path, overlap_receipt: Path) -> dict[str, Any]:
    receipt = _load_json(
        clean_root / "derivation_receipt.json",
        label="Stage211 clean public derivation receipt",
    )
    if (
        receipt.get("artifact") != "stage211_clean_public_eval_derivation"
        or receipt.get("pipeline") != "stage211"
        or receipt.get("complete") is not True
        or int(receipt.get("total_samples", -1))
        != sum(int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values())
        or Path(str(receipt.get("overlap_receipt_path") or "")).resolve() != overlap_receipt
        or receipt.get("overlap_receipt_sha256") != sha256_file(overlap_receipt)
    ):
        raise ValueError("Stage211 clean public derivation contract mismatch.")
    datasets = receipt.get("datasets")
    if not isinstance(datasets, list) or {
        str(row.get("dataset")) for row in datasets if isinstance(row, dict)
    } != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 clean public derivation dataset set mismatch.")
    for row in datasets:
        derived = row.get("derived")
        if not isinstance(derived, dict):
            raise ValueError("Stage211 clean public derivation lacks derived bindings.")
        for prefix in (
            "calibration_prediction",
            "manifest",
            "nano_prediction",
            "nano_report",
        ):
            path = Path(str(derived.get(f"{prefix}_path") or "")).resolve()
            if not path.is_file() or derived.get(f"{prefix}_sha256") != sha256_file(path):
                raise ValueError(f"Stage211 clean public derived {prefix} changed: {path}")
    return receipt


def _canonical_archive_map(
    *,
    manifest_dir: Path,
    nano_root: Path,
    calibration_root: Path,
    archive_root: Path,
) -> dict[Path, Path]:
    calibration_public = calibration_root / "public"
    return {
        manifest_dir / "commonvoice_en_test.jsonl": archive_root / "manifest.jsonl",
        nano_root / "predictions" / "commonvoice_en_test.ctc.jsonl": (
            archive_root / "nano" / "commonvoice_en_test.ctc.jsonl"
        ),
        nano_root / "reports" / "commonvoice_en_test.json": (
            archive_root / "nano" / "commonvoice_en_test.report.json"
        ),
        nano_root / "metrics.json": archive_root / "nano" / "metrics.json",
        nano_root / "metrics.md": archive_root / "nano" / "metrics.md",
        nano_root / "provenance_receipt.json": (archive_root / "nano" / "provenance_receipt.json"),
        calibration_public / "predictions" / "commonvoice_en_test.ctc.jsonl": (
            archive_root / "calibration" / "commonvoice_en_test.ctc.jsonl"
        ),
        calibration_public / "metrics.json": archive_root / "calibration" / "metrics.json",
        calibration_public / "metrics.md": archive_root / "calibration" / "metrics.md",
        calibration_public / "nano_comparison.json": (
            archive_root / "calibration" / "nano_comparison.json"
        ),
        calibration_public / "nano_comparison.md": (
            archive_root / "calibration" / "nano_comparison.md"
        ),
        calibration_public / "reuse_receipt.json": (
            archive_root / "calibration" / "reuse_receipt.json"
        ),
    }


def _archive_originals(mapping: dict[Path, Path], archive_root: Path) -> dict[str, Any]:
    records = []
    for source, archived in mapping.items():
        if not source.is_file():
            raise ValueError(f"Canonical Stage211 artifact is unavailable: {source}")
        source_sha256 = sha256_file(source)
        if archived.is_file():
            if sha256_file(archived) != source_sha256:
                raise ValueError(f"Stage211 contaminated archive conflicts with source: {archived}")
        else:
            _atomic_copy(source, archived)
        records.append(
            {
                "archive_path": str(archived.resolve()),
                "archive_sha256": sha256_file(archived),
                "canonical_path": str(source.resolve()),
                "canonical_sha256": source_sha256,
            }
        )
    receipt = {
        "artifact": "stage211_contaminated_public_archive",
        "complete": True,
        "files": records,
        "pipeline": "stage211",
        "schema_version": 1,
    }
    receipt_path = archive_root / "archive_receipt.json"
    if receipt_path.is_file() and _load_json(receipt_path, label="archive receipt") != receipt:
        raise ValueError("Stage211 contaminated archive receipt changed.")
    _write_json(receipt_path, receipt)
    return receipt


def _restore(mapping: dict[Path, Path]) -> None:
    for canonical, archived in mapping.items():
        _atomic_copy(archived, canonical)


def _validate_existing_install(receipt_path: Path) -> bool:
    if not receipt_path.is_file():
        return False
    receipt = _load_json(receipt_path, label="Stage211 clean public install receipt")
    if receipt.get("complete") is not True:
        return False
    files = receipt.get("installed_files")
    if not isinstance(files, list):
        return False
    return all(
        isinstance(row, dict)
        and (path := Path(str(row.get("path") or ""))).is_file()
        and row.get("sha256") == sha256_file(path)
        for row in files
    )


def install_clean_public_eval(
    *,
    clean_root: Path,
    manifest_dir: Path,
    nano_root: Path,
    calibration_root: Path,
    archive_root: Path,
    overlap_receipt: Path,
) -> dict[str, Any]:
    clean_root = clean_root.expanduser().resolve()
    manifest_dir = manifest_dir.expanduser().resolve()
    nano_root = nano_root.expanduser().resolve()
    calibration_root = calibration_root.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    overlap_receipt = overlap_receipt.expanduser().resolve()
    install_receipt_path = clean_root / "canonical_install_receipt.json"
    if _validate_existing_install(install_receipt_path):
        return _load_json(install_receipt_path, label="Stage211 clean public install receipt")

    validate_stage211_public_overlap_receipt(overlap_receipt)
    _validate_derivation(clean_root, overlap_receipt)
    mapping = _canonical_archive_map(
        manifest_dir=manifest_dir,
        nano_root=nano_root,
        calibration_root=calibration_root,
        archive_root=archive_root,
    )
    archive = _archive_originals(mapping, archive_root)
    calibration_public = calibration_root / "public"
    canonical_manifest = manifest_dir / "commonvoice_en_test.jsonl"
    canonical_nano_prediction = nano_root / "predictions" / "commonvoice_en_test.ctc.jsonl"
    canonical_nano_report = nano_root / "reports" / "commonvoice_en_test.json"
    canonical_calibration_prediction = (
        calibration_public / "predictions" / "commonvoice_en_test.ctc.jsonl"
    )
    try:
        _atomic_copy(clean_root / "manifests" / "commonvoice_en_test.jsonl", canonical_manifest)
        _atomic_copy(
            clean_root / "nano_2512" / "predictions" / "commonvoice_en_test.ctc.jsonl",
            canonical_nano_prediction,
        )
        _atomic_copy(
            clean_root / "calibration" / "predictions" / "commonvoice_en_test.ctc.jsonl",
            canonical_calibration_prediction,
        )
        nano_report = _load_json(
            clean_root / "nano_2512" / "reports" / "commonvoice_en_test.json",
            label="derived Common Voice Nano report",
        )
        nano_report["manifest_path"] = str(canonical_manifest.resolve())
        nano_report["predictions_path"] = str(canonical_nano_prediction.resolve())
        _write_json(canonical_nano_report, nano_report)

        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "summarize_asr_eval_results.py"),
                "--prediction-dir",
                str(nano_root / "predictions"),
                "--output-json",
                str(nano_root / "metrics.json"),
                "--output-md",
                str(nano_root / "metrics.md"),
                "--normalization",
                "ctc",
            ]
        )
        DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT.unlink(missing_ok=True)
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_stage211_nano_baseline_receipt.py"),
                "--nano-checkpoint",
                str(Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"),
                "--report-dir",
                str(nano_root / "reports"),
                "--prediction-dir",
                str(nano_root / "predictions"),
                "--manifest-dir",
                str(manifest_dir),
                "--output",
                str(DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT),
                "--public-overlap-receipt",
                str(overlap_receipt),
            ]
        )
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "summarize_asr_eval_results.py"),
                "--prediction-dir",
                str(calibration_public / "predictions"),
                "--output-json",
                str(calibration_public / "metrics.json"),
                "--output-md",
                str(calibration_public / "metrics.md"),
                "--normalization",
                "ctc",
            ]
        )
        selection = _load_json(
            calibration_root / "checkpoint_selection.json",
            label="Stage211 calibration selection",
        )
        calibration_checkpoint = Path(str(selection["selected"]["checkpoint_path"])).resolve()
        compare_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compare_public_ctc_with_nano.py"),
            "--student-prediction-dir",
            str(calibration_public / "predictions"),
        ]
        for dataset in STAGE211_PUBLIC_BENCHMARKS:
            compare_command.extend(
                [
                    "--nano-prediction",
                    f"{dataset}={nano_root / 'predictions' / f'{dataset}.ctc.jsonl'}",
                ]
            )
        compare_command.extend(
            [
                "--student-checkpoint",
                str(calibration_checkpoint),
                "--output-json",
                str(calibration_public / "nano_comparison.json"),
                "--output-md",
                str(calibration_public / "nano_comparison.md"),
                "--normalization",
                "ctc",
                "--max-relative-ratio",
                "1.20",
                "--max-absolute-gap-points",
                "3.0",
            ]
        )
        _run(compare_command)
        calibration_reuse_receipt = calibration_public / "reuse_receipt.json"
        calibration_reuse_receipt.unlink(missing_ok=True)
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "validate_stage211_calibration_eval.py"),
                "--selection-report",
                str(calibration_root / "checkpoint_selection.json"),
                "--comparison-report",
                str(calibration_public / "nano_comparison.json"),
                "--metrics",
                str(calibration_public / "metrics.json"),
                "--manifest-dir",
                str(manifest_dir),
                "--public-overlap-receipt",
                str(overlap_receipt),
                "--output",
                str(calibration_reuse_receipt),
            ]
        )
        validate_stage211_nano_public_baseline_receipt(
            DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT
        )
    except Exception:
        _restore(mapping)
        raise

    installed_paths = list(mapping)
    installed_files = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in installed_paths
    ]
    comparison = _load_json(
        calibration_public / "nano_comparison.json",
        label="installed calibration comparison",
    )
    commonvoice = next(
        row for row in comparison["results"] if row["dataset"] == "commonvoice_en_test"
    )
    receipt = {
        "archive_receipt_path": str((archive_root / "archive_receipt.json").resolve()),
        "archive_receipt_sha256": sha256_file(archive_root / "archive_receipt.json"),
        "artifact": "stage211_clean_public_canonical_install",
        "commonvoice_clean_metrics": {
            "nano_wer": commonvoice["nano_wer"],
            "sample_count": commonvoice["sample_count"],
            "student_wer": commonvoice["student_wer"],
        },
        "complete": True,
        "derivation_receipt_path": str((clean_root / "derivation_receipt.json").resolve()),
        "derivation_receipt_sha256": sha256_file(clean_root / "derivation_receipt.json"),
        "installed_files": installed_files,
        "overlap_receipt_path": str(overlap_receipt),
        "overlap_receipt_sha256": sha256_file(overlap_receipt),
        "pipeline": "stage211",
        "schema_version": 1,
        "source_archive": archive,
    }
    _write_json(install_receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the audited Stage211 clean public suite with rollback."
    )
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--nano-root", type=Path, default=DEFAULT_NANO_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument(
        "--overlap-receipt",
        type=Path,
        default=DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = install_clean_public_eval(
        clean_root=args.clean_root,
        manifest_dir=args.manifest_dir,
        nano_root=args.nano_root,
        calibration_root=args.calibration_root,
        archive_root=args.archive_root,
        overlap_receipt=args.overlap_receipt,
    )
    metrics = receipt["commonvoice_clean_metrics"]
    print(
        "[stage211-clean-public-install] "
        f"samples={metrics['sample_count']} nano_wer={metrics['nano_wer']:.6f} "
        f"student_wer={metrics['student_wer']:.6f}"
    )


if __name__ == "__main__":
    main()
