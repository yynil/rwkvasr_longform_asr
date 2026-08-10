#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_public_benchmark,
)
from rwkvasr.eval.stage211_initialization import (
    DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
    validate_stage211_initialization_receipt,
)
from rwkvasr.eval.text_metrics import (
    _normalize_text_for_error_tokens,
    normalize_asr_text_for_metrics,
    tokenize_for_wer,
)

if __package__:
    from scripts.create_stage211_initialization_receipt import (
        build_receipt as build_initialization_receipt,
    )
    from scripts.validate_stage211_calibration_eval import build_reuse_receipt
else:
    from create_stage211_initialization_receipt import (
        build_receipt as build_initialization_receipt,
    )
    from validate_stage211_calibration_eval import build_reuse_receipt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_SOURCE = REPO_ROOT / "src" / "rwkvasr" / "eval" / "text_metrics.py"
DEFAULT_NANO_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
DEFAULT_CALIBRATION_ROOT = Path.home() / "rwkvasr_eval" / "stage211_calibration_selected_full"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_metric_unicode_v1"
DEFAULT_CORRECTION_RECEIPT = DEFAULT_OUTPUT_ROOT / "correction_receipt.json"
DEFAULT_PRIOR_INSTALL_RECEIPT = (
    Path.home() / "rwkvasr_eval" / "stage211_public_clean_v1" / "canonical_install_receipt.json"
)
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
EXPECTED_PRIOR_INSTALL_SHA256 = (
    "574314795de83ada77d5c9d14ededfcdad60474f3240a915f600deff9783ca91"
)
LEGACY_WER_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _render_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _affected_paths(
    *,
    nano_root: Path,
    calibration_root: Path,
    initialization_receipt: Path,
) -> dict[str, Path]:
    calibration_public = calibration_root / "public"
    return {
        "nano_metrics_json": nano_root / "metrics.json",
        "nano_metrics_markdown": nano_root / "metrics.md",
        "calibration_metrics_json": calibration_public / "metrics.json",
        "calibration_metrics_markdown": calibration_public / "metrics.md",
        "calibration_comparison_json": calibration_public / "nano_comparison.json",
        "calibration_comparison_markdown": calibration_public / "nano_comparison.md",
        "calibration_reuse_receipt": calibration_public / "reuse_receipt.json",
        "initialization_receipt": initialization_receipt,
    }


def validate_completed_correction(
    receipt_path: Path,
    *,
    tokenizer_source: Path,
    expected_calibration_reuse_receipt: Path | None = None,
    expected_initialization_receipt: Path | None = None,
) -> dict[str, Any] | None:
    if not receipt_path.is_file():
        return None
    receipt = _load_json(receipt_path, label="Stage211 Unicode metric correction receipt")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "unicode_wer_metric_correction",
        "complete": True,
        "tokenizer_contract": "unicode_alnum_words_basic_cjk_chars_v1",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 Unicode metric correction receipt contract mismatch.")
    if (
        Path(str(receipt.get("tokenizer_source_path") or "")).resolve()
        != tokenizer_source.resolve()
        or receipt.get("tokenizer_source_sha256") != sha256_file(tokenizer_source)
    ):
        raise ValueError("Stage211 Unicode metric tokenizer source changed.")
    installed = receipt.get("installed_files")
    if not isinstance(installed, list) or not installed:
        raise ValueError("Stage211 Unicode metric correction has no installed files.")
    installed_by_label: dict[str, dict[str, Any]] = {}
    for record in installed:
        if not isinstance(record, dict):
            raise ValueError("Stage211 Unicode metric installed-file record is invalid.")
        label = str(record.get("label") or "")
        if not label or label in installed_by_label:
            raise ValueError("Stage211 Unicode metric installed-file labels are invalid.")
        path = Path(str(record.get("path") or "")).resolve()
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            raise ValueError(f"Stage211 corrected metric artifact changed: {path}")
        installed_by_label[label] = record
    for label, stem, expected_path in (
        (
            "calibration_reuse_receipt",
            "calibration_reuse_receipt",
            expected_calibration_reuse_receipt,
        ),
        (
            "initialization_receipt",
            "initialization_receipt",
            expected_initialization_receipt,
        ),
    ):
        record = installed_by_label.get(label)
        if record is None:
            raise ValueError(f"Stage211 Unicode metric correction lacks {label}.")
        bound_path = Path(str(receipt.get(f"{stem}_path") or "")).resolve()
        bound_sha256 = str(receipt.get(f"{stem}_sha256") or "")
        if (
            bound_path != Path(str(record["path"])).resolve()
            or bound_sha256 != record["sha256"]
        ):
            raise ValueError(f"Stage211 Unicode metric {label} binding mismatch.")
        if expected_path is not None and bound_path != expected_path.expanduser().resolve():
            raise ValueError(f"Stage211 Unicode metric correction binds another {label}.")
    return receipt


def _validate_prior_install(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if sha256_file(path) != EXPECTED_PRIOR_INSTALL_SHA256:
        raise ValueError("Stage211 prior clean canonical-install receipt changed.")
    receipt = _load_json(path, label="Stage211 prior clean canonical-install receipt")
    if (
        receipt.get("pipeline") != "stage211"
        or receipt.get("artifact") != "stage211_clean_public_canonical_install"
        or receipt.get("complete") is not True
    ):
        raise ValueError("Stage211 prior clean canonical-install receipt is invalid.")
    for record in receipt.get("installed_files", []):
        artifact = Path(str(record.get("path") or "")).resolve()
        if not artifact.is_file() or record.get("sha256") != sha256_file(artifact):
            raise ValueError(f"Stage211 prior clean canonical artifact changed: {artifact}")
    return receipt


def _archive_originals(paths: dict[str, Path], archive_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, source in paths.items():
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(str(source))
        archived = archive_root / label / source.name
        source_sha256 = sha256_file(source)
        if archived.is_file():
            if sha256_file(archived) != source_sha256:
                raise ValueError(f"Stage211 Unicode metric archive conflicts: {archived}")
        else:
            _atomic_copy(source, archived)
        records.append(
            {
                "label": label,
                "canonical_path": str(source),
                "before_sha256": source_sha256,
                "archive_path": str(archived.resolve()),
                "archive_sha256": sha256_file(archived),
            }
        )
    return records


def _restore_archives(records: list[dict[str, Any]]) -> None:
    for record in records:
        _atomic_copy(
            Path(str(record["archive_path"])),
            Path(str(record["canonical_path"])),
        )


def _unicode_reference_audit(prediction_path: Path) -> list[dict[str, Any]]:
    affected: list[dict[str, Any]] = []
    with prediction_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            reference = normalize_asr_text_for_metrics(
                record.get("ref_text"),
                language="en",
                normalization="ctc",
            )
            legacy_tokens = LEGACY_WER_PATTERN.findall(
                _normalize_text_for_error_tokens(reference)
            )
            corrected_tokens = tokenize_for_wer(reference)
            if legacy_tokens != corrected_tokens:
                affected.append(
                    {
                        "line": line_number,
                        "utt_id": str(record.get("utt_id") or ""),
                        "normalized_reference": reference,
                        "legacy_tokens": legacy_tokens,
                        "corrected_tokens": corrected_tokens,
                    }
                )
    return affected


def _metric_row(metrics: dict[str, Any], dataset: str) -> dict[str, Any]:
    rows = metrics.get("results")
    if not isinstance(rows, list):
        raise ValueError("Stage211 metrics artifact has no results.")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("dataset") == dataset
        and row.get("branch") == "ctc"
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage211 metrics lacks one CTC row for {dataset}.")
    return dict(matches[0])


def _generate_derived_metrics(
    *,
    temporary_root: Path,
    nano_root: Path,
    calibration_root: Path,
) -> dict[str, Path]:
    outputs = {
        "nano_metrics_json": temporary_root / "nano" / "metrics.json",
        "nano_metrics_markdown": temporary_root / "nano" / "metrics.md",
        "calibration_metrics_json": temporary_root / "calibration" / "metrics.json",
        "calibration_metrics_markdown": temporary_root / "calibration" / "metrics.md",
        "calibration_comparison_json": temporary_root / "calibration" / "nano_comparison.json",
        "calibration_comparison_markdown": temporary_root
        / "calibration"
        / "nano_comparison.md",
    }
    summarize = REPO_ROOT / "scripts" / "summarize_asr_eval_results.py"
    for prediction_dir, json_key, markdown_key in (
        (nano_root / "predictions", "nano_metrics_json", "nano_metrics_markdown"),
        (
            calibration_root / "public" / "predictions",
            "calibration_metrics_json",
            "calibration_metrics_markdown",
        ),
    ):
        _run(
            [
                sys.executable,
                str(summarize),
                "--prediction-dir",
                str(prediction_dir),
                "--output-json",
                str(outputs[json_key]),
                "--output-md",
                str(outputs[markdown_key]),
                "--normalization",
                "ctc",
            ]
        )

    current_reuse = _load_json(
        calibration_root / "public" / "reuse_receipt.json",
        label="Stage211 current calibration reuse receipt",
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compare_public_ctc_with_nano.py"),
        "--student-prediction-dir",
        str(calibration_root / "public" / "predictions"),
        "--output-json",
        str(outputs["calibration_comparison_json"]),
        "--output-md",
        str(outputs["calibration_comparison_markdown"]),
        "--normalization",
        "ctc",
        "--max-relative-ratio",
        "1.20",
        "--max-absolute-gap-points",
        "3.0",
        "--student-checkpoint",
        str(current_reuse["checkpoint_path"]),
    ]
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        command.extend(
            (
                "--nano-prediction",
                f"{dataset}={nano_root / 'predictions' / f'{dataset}.ctc.jsonl'}",
            )
        )
    _run(command)
    return outputs


def install_correction(
    *,
    output_root: Path,
    prior_install_receipt: Path,
    nano_root: Path,
    calibration_root: Path,
    manifest_dir: Path,
    initialization_receipt: Path,
    nano_checkpoint: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    receipt_path = output_root / "correction_receipt.json"
    tokenizer_source = DEFAULT_TOKENIZER_SOURCE
    completed = validate_completed_correction(
        receipt_path,
        tokenizer_source=tokenizer_source,
    )
    if completed is not None:
        return completed

    prior_install = _validate_prior_install(prior_install_receipt)
    paths = _affected_paths(
        nano_root=nano_root,
        calibration_root=calibration_root,
        initialization_receipt=initialization_receipt,
    )
    old_comparison = _load_json(
        paths["calibration_comparison_json"],
        label="Stage211 superseded calibration comparison",
    )
    with tempfile.TemporaryDirectory(
        prefix=".stage211-unicode-metric-",
        dir=output_root.parent,
    ) as raw_temporary:
        generated = _generate_derived_metrics(
            temporary_root=Path(raw_temporary),
            nano_root=nano_root,
            calibration_root=calibration_root,
        )
        archive_records = _archive_originals(paths, output_root / "archive")
        try:
            for label, generated_path in generated.items():
                _atomic_copy(generated_path, paths[label])

            reuse = build_reuse_receipt(
                selection_report_path=calibration_root / "checkpoint_selection.json",
                comparison_report_path=paths["calibration_comparison_json"],
                metrics_path=paths["calibration_metrics_json"],
                manifest_dir=manifest_dir,
                public_overlap_receipt_path=DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT,
            )
            _atomic_write(paths["calibration_reuse_receipt"], _render_json(reuse))
            benchmark = validate_stage211_public_benchmark(reuse["public_benchmark"])
            nano_provenance = validate_stage211_nano_public_baseline_receipt(
                DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
                public_benchmark=benchmark,
            )

            initialization = build_initialization_receipt(
                calibration_reuse_receipt_path=paths["calibration_reuse_receipt"],
                nano_checkpoint=nano_checkpoint,
            )
            _atomic_write(paths["initialization_receipt"], _render_json(initialization))
            validate_stage211_initialization_receipt(
                paths["initialization_receipt"],
                expected_calibration_checkpoint=Path(
                    initialization["calibration_checkpoint_path"]
                ),
                expected_nano_checkpoint_sha256=initialization["nano_checkpoint_sha256"],
            )

            installed_files = [
                {
                    "label": label,
                    "path": str(path.expanduser().resolve()),
                    "sha256": sha256_file(path),
                }
                for label, path in paths.items()
            ]
            new_comparison = _load_json(
                paths["calibration_comparison_json"],
                label="Stage211 corrected calibration comparison",
            )
            old_by_dataset = {
                str(row["dataset"]): row for row in old_comparison["results"]
            }
            new_by_dataset = {
                str(row["dataset"]): row for row in new_comparison["results"]
            }
            cv_dataset = "commonvoice_en_test"
            affected = _unicode_reference_audit(
                calibration_root / "public" / "predictions" / f"{cv_dataset}.ctc.jsonl"
            )
            if len(affected) != 4:
                raise ValueError(
                    "Stage211 Unicode WER correction expected exactly four affected "
                    f"Common Voice references, got {len(affected)}."
                )
            receipt = {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "unicode_wer_metric_correction",
                "complete": True,
                "tokenizer_contract": "unicode_alnum_words_basic_cjk_chars_v1",
                "tokenizer_source_path": str(tokenizer_source.resolve()),
                "tokenizer_source_sha256": sha256_file(tokenizer_source),
                "prior_clean_install_receipt_path": str(prior_install_receipt.resolve()),
                "prior_clean_install_receipt_sha256": sha256_file(prior_install_receipt),
                "prior_clean_install_artifact": prior_install["artifact"],
                "predictions_changed": False,
                "manifests_changed": False,
                "decode_changed": False,
                "affected_references": affected,
                "before_commonvoice": old_by_dataset[cv_dataset],
                "after_commonvoice": new_by_dataset[cv_dataset],
                "nano_checkpoint_path": nano_provenance["nano_checkpoint_path"],
                "nano_checkpoint_sha256": nano_provenance["nano_checkpoint_sha256"],
                "calibration_reuse_receipt_path": str(
                    paths["calibration_reuse_receipt"].resolve()
                ),
                "calibration_reuse_receipt_sha256": sha256_file(
                    paths["calibration_reuse_receipt"]
                ),
                "initialization_receipt_path": str(paths["initialization_receipt"].resolve()),
                "initialization_receipt_sha256": sha256_file(
                    paths["initialization_receipt"]
                ),
                "archive_files": archive_records,
                "installed_files": installed_files,
                "corrected_nano_commonvoice_metrics": _metric_row(
                    _load_json(paths["nano_metrics_json"], label="Nano metrics"),
                    cv_dataset,
                ),
                "corrected_calibration_commonvoice_metrics": _metric_row(
                    _load_json(
                        paths["calibration_metrics_json"],
                        label="Calibration metrics",
                    ),
                    cv_dataset,
                ),
            }
            _atomic_write(receipt_path, _render_json(receipt))
            return validate_completed_correction(
                receipt_path,
                tokenizer_source=tokenizer_source,
            ) or receipt
        except BaseException:
            receipt_path.unlink(missing_ok=True)
            _restore_archives(archive_records)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically install and receipt the Stage211 Unicode-aware WER metric correction."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prior-install-receipt",
        type=Path,
        default=DEFAULT_PRIOR_INSTALL_RECEIPT,
    )
    parser.add_argument("--nano-root", type=Path, default=DEFAULT_NANO_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--initialization-receipt",
        type=Path,
        default=DEFAULT_STAGE211_INITIALIZATION_RECEIPT,
    )
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = install_correction(
        output_root=args.output_root,
        prior_install_receipt=args.prior_install_receipt,
        nano_root=args.nano_root.expanduser().resolve(),
        calibration_root=args.calibration_root.expanduser().resolve(),
        manifest_dir=args.manifest_dir.expanduser().resolve(),
        initialization_receipt=args.initialization_receipt.expanduser().resolve(),
        nano_checkpoint=args.nano_checkpoint.expanduser().resolve(),
    )
    print(
        "[stage211-unicode-metric] "
        f"receipt={args.output_root.expanduser().resolve() / 'correction_receipt.json'} "
        f"reuse_sha256={receipt['calibration_reuse_receipt_sha256']} "
        f"initialization_sha256={receipt['initialization_receipt_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
