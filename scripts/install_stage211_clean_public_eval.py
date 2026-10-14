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
DEFAULT_ARCHIVE_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_pre_quote_repair_archive_v3"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"


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
    mapping = {
        manifest_dir / "commonvoice_en_test.jsonl": archive_root / "manifest.jsonl",
        nano_root / "predictions" / "commonvoice_en_test.ctc.jsonl": (
            archive_root / "nano" / "commonvoice_en_test.ctc.jsonl"
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
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        mapping[nano_root / "reports" / f"{dataset}.json"] = (
            archive_root / "nano" / "reports" / f"{dataset}.json"
        )
    return mapping


def _report_checkpoint_path(report: dict[str, Any]) -> Path:
    embedded = report.get("model_checkpoint_path")
    if embedded is not None and str(embedded).strip():
        return Path(str(embedded)).expanduser().resolve()
    model_path = Path(str(report.get("model_path") or "")).expanduser().resolve()
    return model_path if model_path.name == "model.pt" else model_path / "model.pt"


def _validate_nano_checkpoint(nano_checkpoint: Path) -> tuple[Path, str]:
    nano_checkpoint = nano_checkpoint.expanduser().resolve()
    if (
        not nano_checkpoint.is_file()
        or nano_checkpoint.stat().st_size <= 0
        or nano_checkpoint.name != "model.pt"
    ):
        raise ValueError(
            f"Stage211 Nano checkpoint must be a non-empty model.pt: {nano_checkpoint}"
        )
    return nano_checkpoint, sha256_file(nano_checkpoint)


def _bind_nano_report_checkpoint_identity(
    *,
    nano_root: Path,
    nano_checkpoint: Path,
) -> dict[str, Any]:
    nano_checkpoint, checkpoint_sha256 = _validate_nano_checkpoint(nano_checkpoint)
    prepared: list[tuple[str, Path, dict[str, Any]]] = []
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        report_path = (nano_root / "reports" / f"{dataset}.json").resolve()
        report = _load_json(report_path, label=f"Stage211 {dataset} Nano report")
        if _report_checkpoint_path(report) != nano_checkpoint:
            raise ValueError(f"Stage211 {dataset} Nano report refers to a different checkpoint.")
        embedded_path = report.get("model_checkpoint_path")
        embedded_sha256 = report.get("model_checkpoint_sha256")
        if (embedded_path is None) != (embedded_sha256 is None):
            raise ValueError(f"Stage211 {dataset} Nano report has partial checkpoint identity.")
        if embedded_path is not None and (
            Path(str(embedded_path)).expanduser().resolve() != nano_checkpoint
            or embedded_sha256 != checkpoint_sha256
        ):
            raise ValueError(f"Stage211 {dataset} Nano report checkpoint identity conflicts.")
        updated = dict(report)
        updated["model_checkpoint_path"] = str(nano_checkpoint)
        updated["model_checkpoint_sha256"] = checkpoint_sha256
        prepared.append((dataset, report_path, updated))

    for _, report_path, report in prepared:
        _write_json(report_path, report)

    return {
        "artifact": "stage211_nano_report_checkpoint_identity",
        "checkpoint_path": str(nano_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "complete": True,
        "mode": "embedded_checkpoint_sha256",
        "reports": [
            {
                "dataset": dataset,
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            }
            for dataset, report_path, _ in prepared
        ],
        "schema_version": 1,
    }


def _validate_nano_report_checkpoint_identity(
    proof: object,
    *,
    nano_root: Path,
    nano_checkpoint: Path,
) -> dict[str, Any]:
    nano_checkpoint, checkpoint_sha256 = _validate_nano_checkpoint(nano_checkpoint)
    if not isinstance(proof, dict):
        raise ValueError("Stage211 canonical install lacks Nano report checkpoint identity proof.")
    expected = {
        "artifact": "stage211_nano_report_checkpoint_identity",
        "checkpoint_path": str(nano_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "complete": True,
        "mode": "embedded_checkpoint_sha256",
        "schema_version": 1,
    }
    if any(proof.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 Nano report checkpoint identity proof mismatch.")
    reports = proof.get("reports")
    if not isinstance(reports, list) or len(reports) != len(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 Nano report checkpoint identity coverage mismatch.")
    by_dataset = {str(row.get("dataset")): row for row in reports if isinstance(row, dict)}
    if len(by_dataset) != len(reports) or set(by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 Nano report checkpoint identity dataset set mismatch.")
    for dataset, row in by_dataset.items():
        report_path = (nano_root / "reports" / f"{dataset}.json").resolve()
        if (
            Path(str(row.get("path") or "")).resolve() != report_path
            or not report_path.is_file()
            or row.get("sha256") != sha256_file(report_path)
        ):
            raise ValueError(f"Stage211 {dataset} Nano report identity binding changed.")
        report = _load_json(report_path, label=f"Stage211 {dataset} Nano report")
        if (
            Path(str(report.get("model_checkpoint_path") or "")).expanduser().resolve()
            != nano_checkpoint
            or report.get("model_checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError(f"Stage211 {dataset} Nano report embedded identity mismatch.")
    return proof


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


def _validate_existing_install(
    receipt_path: Path,
    *,
    clean_root: Path,
    manifest_dir: Path,
    nano_root: Path,
    calibration_root: Path,
    archive_root: Path,
    overlap_receipt: Path,
    nano_checkpoint: Path,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = _load_json(receipt_path, label="Stage211 clean public install receipt")
    expected_contract = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stage211_clean_public_canonical_install",
        "complete": True,
    }
    if any(receipt.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Stage211 clean public install receipt contract mismatch.")

    validate_stage211_public_overlap_receipt(overlap_receipt)
    derivation = _validate_derivation(clean_root, overlap_receipt)
    derivation_path = (clean_root / "derivation_receipt.json").resolve()
    if (
        receipt_path.resolve() != (clean_root / "canonical_install_receipt.json").resolve()
        or Path(str(receipt.get("overlap_receipt_path") or "")).resolve()
        != overlap_receipt.resolve()
        or receipt.get("overlap_receipt_sha256") != sha256_file(overlap_receipt)
        or Path(str(receipt.get("derivation_receipt_path") or "")).resolve() != derivation_path
        or receipt.get("derivation_receipt_sha256") != sha256_file(derivation_path)
        or int(derivation.get("total_samples", -1))
        != sum(int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values())
    ):
        raise ValueError("Stage211 clean public install provenance mismatch.")

    metrics = receipt.get("commonvoice_clean_metrics")
    if not isinstance(metrics, dict) or int(metrics.get("sample_count", -1)) != int(
        STAGE211_PUBLIC_BENCHMARKS["commonvoice_en_test"]["samples"]
    ):
        raise ValueError("Stage211 clean public install Common Voice coverage mismatch.")

    mapping = _canonical_archive_map(
        manifest_dir=manifest_dir,
        nano_root=nano_root,
        calibration_root=calibration_root,
        archive_root=archive_root,
    )
    resolved_mapping = {
        canonical.resolve(): archived.resolve() for canonical, archived in mapping.items()
    }
    expected_paths = set(resolved_mapping)
    files = receipt.get("installed_files")
    if not isinstance(files, list) or len(files) != len(expected_paths):
        raise ValueError("Stage211 clean public installed-file coverage mismatch.")
    installed_by_path: dict[Path, dict[str, Any]] = {}
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("Stage211 clean public installed-file record is invalid.")
        path = Path(str(row.get("path") or "")).resolve()
        if path in installed_by_path:
            raise ValueError("Stage211 clean public installed-file paths are duplicated.")
        if not path.is_file() or row.get("sha256") != sha256_file(path):
            raise ValueError(f"Stage211 clean public installed file changed: {path}")
        installed_by_path[path] = row
    if set(installed_by_path) != expected_paths:
        raise ValueError("Stage211 clean public canonical destination set mismatch.")

    archive_receipt_path = (archive_root / "archive_receipt.json").resolve()
    if (
        Path(str(receipt.get("archive_receipt_path") or "")).resolve() != archive_receipt_path
        or not archive_receipt_path.is_file()
        or receipt.get("archive_receipt_sha256") != sha256_file(archive_receipt_path)
    ):
        raise ValueError("Stage211 clean public archive-receipt binding mismatch.")
    archive = _load_json(
        archive_receipt_path,
        label="Stage211 contaminated public archive receipt",
    )
    if receipt.get("source_archive") != archive:
        raise ValueError("Stage211 clean public embedded archive receipt mismatch.")
    archive_files = archive.get("files")
    if (
        archive.get("schema_version") != 1
        or archive.get("pipeline") != "stage211"
        or archive.get("artifact") != "stage211_contaminated_public_archive"
        or archive.get("complete") is not True
        or not isinstance(archive_files, list)
        or len(archive_files) != len(mapping)
    ):
        raise ValueError("Stage211 clean public source archive contract mismatch.")
    archived_by_canonical: dict[Path, dict[str, Any]] = {}
    for row in archive_files:
        if not isinstance(row, dict):
            raise ValueError("Stage211 clean public source archive record is invalid.")
        canonical = Path(str(row.get("canonical_path") or "")).resolve()
        archived = Path(str(row.get("archive_path") or "")).resolve()
        if canonical in archived_by_canonical:
            raise ValueError("Stage211 clean public source archive paths are duplicated.")
        expected_archive = resolved_mapping.get(canonical)
        if (
            expected_archive is None
            or archived != expected_archive
            or not archived.is_file()
            or row.get("archive_sha256") != sha256_file(archived)
            or row.get("canonical_sha256") != row.get("archive_sha256")
        ):
            raise ValueError("Stage211 clean public source archive binding mismatch.")
        archived_by_canonical[canonical] = row
    if set(archived_by_canonical) != expected_paths:
        raise ValueError("Stage211 clean public source archive destination set mismatch.")
    _validate_nano_report_checkpoint_identity(
        receipt.get("nano_report_checkpoint_identity"),
        nano_root=nano_root,
        nano_checkpoint=nano_checkpoint,
    )
    nano_provenance = validate_stage211_nano_public_baseline_receipt(
        nano_root / "provenance_receipt.json",
        expected_nano_checkpoint_sha256=sha256_file(nano_checkpoint),
    )
    if nano_provenance.get("provenance_mode") != "embedded_checkpoint_sha256":
        raise ValueError("Stage211 canonical Nano baseline must use embedded checkpoint identity.")
    return receipt


def install_clean_public_eval(
    *,
    clean_root: Path,
    manifest_dir: Path,
    nano_root: Path,
    calibration_root: Path,
    archive_root: Path,
    overlap_receipt: Path,
    nano_checkpoint: Path,
) -> dict[str, Any]:
    clean_root = clean_root.expanduser().resolve()
    manifest_dir = manifest_dir.expanduser().resolve()
    nano_root = nano_root.expanduser().resolve()
    calibration_root = calibration_root.expanduser().resolve()
    archive_root = archive_root.expanduser().resolve()
    overlap_receipt = overlap_receipt.expanduser().resolve()
    nano_checkpoint, nano_checkpoint_sha256 = _validate_nano_checkpoint(nano_checkpoint)
    install_receipt_path = clean_root / "canonical_install_receipt.json"
    existing = _validate_existing_install(
        install_receipt_path,
        clean_root=clean_root,
        manifest_dir=manifest_dir,
        nano_root=nano_root,
        calibration_root=calibration_root,
        archive_root=archive_root,
        overlap_receipt=overlap_receipt,
        nano_checkpoint=nano_checkpoint,
    )
    if existing is not None:
        return existing

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
        nano_report_checkpoint_identity = _bind_nano_report_checkpoint_identity(
            nano_root=nano_root,
            nano_checkpoint=nano_checkpoint,
        )

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
        nano_baseline_receipt = nano_root / "provenance_receipt.json"
        nano_baseline_receipt.unlink(missing_ok=True)
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "create_stage211_nano_baseline_receipt.py"),
                "--nano-checkpoint",
                str(nano_checkpoint),
                "--report-dir",
                str(nano_root / "reports"),
                "--prediction-dir",
                str(nano_root / "predictions"),
                "--manifest-dir",
                str(manifest_dir),
                "--output",
                str(nano_baseline_receipt),
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
        nano_provenance = validate_stage211_nano_public_baseline_receipt(
            nano_baseline_receipt,
            expected_nano_checkpoint_sha256=nano_checkpoint_sha256,
        )
        if nano_provenance.get("provenance_mode") != "embedded_checkpoint_sha256":
            raise ValueError(
                "Stage211 canonical Nano baseline must use embedded checkpoint identity."
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
        "nano_report_checkpoint_identity": nano_report_checkpoint_identity,
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
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
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
        nano_checkpoint=args.nano_checkpoint,
    )
    metrics = receipt["commonvoice_clean_metrics"]
    print(
        "[stage211-clean-public-install] "
        f"samples={metrics['sample_count']} nano_wer={metrics['nano_wer']:.6f} "
        f"student_wer={metrics['student_wer']:.6f}"
    )


if __name__ == "__main__":
    main()
