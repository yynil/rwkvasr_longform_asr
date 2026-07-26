from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    STAGE211_NANO_PUBLIC_BASELINE_SCHEMA_VERSION,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    validate_stage211_nano_public_baseline_receipt,
)

try:
    from scripts.create_stage211_phase_gate import _jsonl_records
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_phase_gate import _jsonl_records


DEFAULT_NANO_BASELINE_ROOT = (
    Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
)
DEFAULT_NANO_CHECKPOINT = (
    Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
)
DEFAULT_PUBLIC_MANIFEST_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "eval_benchmarks"
    / "manifests"
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _report_checkpoint_path(report: dict[str, Any]) -> Path:
    embedded = report.get("model_checkpoint_path")
    if embedded is not None and str(embedded).strip():
        return Path(str(embedded)).expanduser().resolve()
    model_path = Path(str(report.get("model_path") or "")).expanduser().resolve()
    return model_path if model_path.name == "model.pt" else model_path / "model.pt"


def build_nano_baseline_receipt(
    *,
    nano_checkpoint_path: Path,
    report_dir: Path,
    prediction_dir: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    nano_checkpoint_path = nano_checkpoint_path.expanduser().resolve()
    if (
        not nano_checkpoint_path.is_file()
        or nano_checkpoint_path.stat().st_size <= 0
    ):
        raise ValueError(
            f"Stage211 Nano baseline checkpoint is missing or empty: "
            f"{nano_checkpoint_path}"
        )
    if nano_checkpoint_path.name != "model.pt":
        raise ValueError("Stage211 Nano baseline checkpoint must be model.pt.")
    nano_checkpoint_sha256 = sha256_file(nano_checkpoint_path)
    report_dir = report_dir.expanduser().resolve()
    prediction_dir = prediction_dir.expanduser().resolve()
    manifest_dir = manifest_dir.expanduser().resolve()

    results: list[dict[str, Any]] = []
    embedded_identity_count = 0
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        report_path = (report_dir / f"{dataset}.json").resolve()
        prediction_path = (
            prediction_dir / f"{dataset}.ctc.jsonl"
        ).resolve()
        manifest_path = (manifest_dir / f"{dataset}.jsonl").resolve()
        report = _load_json(
            report_path,
            label=f"Stage211 {dataset} Nano inference report",
        )
        expected_report = {
            "version": 1,
            "system": "FunASR-Nano-2512 direct CTC",
            "language": expected["language"],
            "normalization": "ctc",
            "decode": "greedy_ctc",
            "requested_limit": None,
            "sample_count": expected["samples"],
        }
        if any(report.get(key) != value for key, value in expected_report.items()):
            raise ValueError(
                f"Stage211 {dataset} Nano inference report contract mismatch."
            )
        if Path(str(report.get("manifest_path") or "")).resolve() != manifest_path:
            raise ValueError(
                f"Stage211 {dataset} Nano inference report manifest path mismatch."
            )
        if (
            Path(str(report.get("predictions_path") or "")).resolve()
            != prediction_path
        ):
            raise ValueError(
                f"Stage211 {dataset} Nano inference report prediction path mismatch."
            )
        if _report_checkpoint_path(report) != nano_checkpoint_path:
            raise ValueError(
                f"Stage211 {dataset} Nano inference report model path mismatch."
            )
        embedded_path = report.get("model_checkpoint_path")
        embedded_sha256 = report.get("model_checkpoint_sha256")
        if embedded_path is not None or embedded_sha256 is not None:
            if (
                Path(str(embedded_path or "")).expanduser().resolve()
                != nano_checkpoint_path
                or embedded_sha256 != nano_checkpoint_sha256
            ):
                raise ValueError(
                    f"Stage211 {dataset} Nano inference report checkpoint identity mismatch."
                )
            embedded_identity_count += 1

        language = str(expected["language"])
        manifest_records = _jsonl_records(
            manifest_path,
            language=language,
            reference_keys=("text", "transcript", "ref_text", "reference"),
        )
        prediction_records = _jsonl_records(
            prediction_path,
            language=language,
            reference_keys=("ref_text", "reference", "text", "transcript"),
        )
        expected_samples = int(expected["samples"])
        if (
            len(manifest_records) != expected_samples
            or set(prediction_records) != set(manifest_records)
        ):
            raise ValueError(
                f"Stage211 {dataset} Nano baseline coverage mismatch: "
                f"manifest={len(manifest_records)} "
                f"prediction={len(prediction_records)} expected={expected_samples}"
            )
        reference_mismatch_count = sum(
            prediction_records[utt_id] != reference
            for utt_id, reference in manifest_records.items()
        )
        if reference_mismatch_count:
            raise ValueError(
                f"Stage211 {dataset} Nano baseline normalized references differ "
                f"from the manifest: mismatches={reference_mismatch_count}"
            )
        results.append(
            {
                "dataset": dataset,
                "language": expected["language"],
                "metric": expected["metric"],
                "sample_count": expected_samples,
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "report_embeds_checkpoint_sha256": (
                    embedded_path is not None and embedded_sha256 is not None
                ),
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "nano_prediction_path": str(prediction_path),
                "nano_prediction_sha256": sha256_file(prediction_path),
            }
        )

    if embedded_identity_count not in {
        0,
        len(STAGE211_PUBLIC_BENCHMARKS),
    }:
        raise ValueError(
            "Stage211 Nano baseline mixes legacy and embedded checkpoint identity."
        )
    provenance_mode = (
        "embedded_checkpoint_sha256"
        if embedded_identity_count == len(STAGE211_PUBLIC_BENCHMARKS)
        else "legacy_report_attestation"
    )
    return {
        "schema_version": STAGE211_NANO_PUBLIC_BASELINE_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "nano_public_baseline_provenance",
        "complete": True,
        "provenance_mode": provenance_mode,
        "nano_checkpoint_path": str(nano_checkpoint_path),
        "nano_checkpoint_sha256": nano_checkpoint_sha256,
        "total_samples": sum(
            int(expected["samples"])
            for expected in STAGE211_PUBLIC_BENCHMARKS.values()
        ),
        "results": results,
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(
            f"Refusing to overwrite a different Nano baseline receipt: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the immutable Stage211 Nano direct-CTC public-baseline "
            "provenance receipt."
        )
    )
    parser.add_argument(
        "--nano-checkpoint",
        type=Path,
        default=DEFAULT_NANO_CHECKPOINT,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_NANO_BASELINE_ROOT / "reports",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_NANO_BASELINE_ROOT / "predictions",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_PUBLIC_MANIFEST_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT,
    )
    args = parser.parse_args()

    receipt = build_nano_baseline_receipt(
        nano_checkpoint_path=args.nano_checkpoint,
        report_dir=args.report_dir,
        prediction_dir=args.prediction_dir,
        manifest_dir=args.manifest_dir,
    )
    output_path = args.output.expanduser().resolve()
    _write_immutable_json(output_path, receipt)
    validate_stage211_nano_public_baseline_receipt(output_path)
    print(
        f"nano_baseline_receipt={output_path} "
        f"provenance_mode={receipt['provenance_mode']} "
        f"nano_checkpoint_sha256={receipt['nano_checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
