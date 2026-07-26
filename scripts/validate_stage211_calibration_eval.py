#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.eval.stage211_gate import STAGE211_PUBLIC_BENCHMARKS, sha256_file

if __package__:
    from scripts.create_stage211_phase_gate import _enrich_public_benchmark
else:
    from create_stage211_phase_gate import _enrich_public_benchmark


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _require_finite_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    tolerance: float = 1.0e-12,
) -> None:
    actual_value = float(actual)
    expected_value = float(expected)
    if (
        not math.isfinite(actual_value)
        or not math.isfinite(expected_value)
        or abs(actual_value - expected_value) > tolerance
    ):
        raise ValueError(f"{label} mismatch: actual={actual_value} expected={expected_value}")


def build_reuse_receipt(
    *,
    selection_report_path: Path,
    comparison_report_path: Path,
    metrics_path: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    selection_report_path = selection_report_path.resolve()
    comparison_report_path = comparison_report_path.resolve()
    metrics_path = metrics_path.resolve()
    manifest_dir = manifest_dir.resolve()

    selection = _load_json(
        selection_report_path,
        label="Stage211 calibration checkpoint selection",
    )
    if (
        selection.get("pipeline") != "stage211"
        or selection.get("artifact") != "calibration_checkpoint_selection"
        or int(selection.get("required_completion_step", -1)) != 30_064
    ):
        raise ValueError("Stage211 calibration selection contract mismatch.")
    selected = selection.get("selected")
    if not isinstance(selected, dict) or selected.get("eligible") is not True:
        raise ValueError("Stage211 calibration selection has no eligible checkpoint.")
    if (
        int(selected.get("loss_improved_layers", -1)) != 70
        or int(selected.get("cosine_improved_layers", -1)) != 70
    ):
        raise ValueError("Stage211 calibration checkpoint did not improve all 70 layers.")
    checkpoint_path = Path(str(selected.get("checkpoint_path") or "")).resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise ValueError(f"Selected calibration checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if selected.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Selected calibration checkpoint SHA-256 mismatch.")

    comparison = _load_json(
        comparison_report_path,
        label="Stage211 calibration public Nano comparison",
    )
    if comparison.get("decode") != "greedy_ctc" or comparison.get("normalization") != "ctc":
        raise ValueError("Stage211 calibration comparison decode contract mismatch.")
    recorded_checkpoint = Path(str(comparison.get("student_checkpoint_path") or "")).resolve()
    if recorded_checkpoint != checkpoint_path:
        raise ValueError("Stage211 calibration comparison checkpoint path mismatch.")
    if comparison.get("student_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Stage211 calibration comparison checkpoint SHA-256 mismatch.")
    benchmark = _enrich_public_benchmark(
        comparison,
        manifest_dir=manifest_dir,
    )

    metrics = _load_json(metrics_path, label="Stage211 calibration public metrics")
    metric_rows = metrics.get("results")
    if not isinstance(metric_rows, list):
        raise ValueError("Stage211 calibration metrics results must be a list.")
    metrics_by_dataset = {
        str(row.get("dataset")): row for row in metric_rows if isinstance(row, dict)
    }
    benchmark_rows = benchmark.get("results")
    if not isinstance(benchmark_rows, list):
        raise ValueError("Stage211 calibration benchmark results must be a list.")
    benchmark_by_dataset = {
        str(row.get("dataset")): row for row in benchmark_rows if isinstance(row, dict)
    }
    expected_datasets = set(STAGE211_PUBLIC_BENCHMARKS)
    if (
        set(metrics_by_dataset) != expected_datasets
        or set(benchmark_by_dataset) != expected_datasets
    ):
        raise ValueError("Stage211 calibration public dataset set mismatch.")

    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        metric_row = metrics_by_dataset[dataset]
        benchmark_row = benchmark_by_dataset[dataset]
        expected_samples = int(expected["samples"])
        if (
            metric_row.get("branch") != "ctc"
            or int(metric_row.get("samples", -1)) != expected_samples
            or int(benchmark_row.get("sample_count", -1)) != expected_samples
        ):
            raise ValueError(f"Stage211 calibration coverage mismatch for {dataset}.")
        _require_finite_close(
            metric_row.get("wer"),
            benchmark_row.get("student_wer"),
            label=f"{dataset} student WER",
        )
        _require_finite_close(
            metric_row.get("cer"),
            benchmark_row.get("student_cer"),
            label=f"{dataset} student CER",
        )

    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "calibration_public_eval_reuse",
        "complete": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "selection_report_path": str(selection_report_path),
        "selection_report_sha256": sha256_file(selection_report_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "comparison_report_path": str(comparison_report_path),
        "comparison_report_sha256": sha256_file(comparison_report_path),
        "public_benchmark": benchmark,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and bind a completed Stage211 calibration public evaluation "
            "before reusing it during supervisor recovery."
        )
    )
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = build_reuse_receipt(
        selection_report_path=args.selection_report,
        comparison_report_path=args.comparison_report,
        metrics_path=args.metrics,
        manifest_dir=args.manifest_dir,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output_path.is_file() and output_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different reuse receipt: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"reuse_receipt={output_path} checkpoint={receipt['checkpoint_path']} "
        f"checkpoint_sha256={receipt['checkpoint_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
