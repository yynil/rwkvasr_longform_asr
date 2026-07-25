#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.eval import (
    compare_prediction_text_sets,
    compute_text_error_stats,
    normalize_asr_text_for_metrics,
)


DATASETS: dict[str, dict[str, str]] = {
    "aishell1_test": {"language": "zh", "label": "AISHELL-1 test", "metric": "cer"},
    "librispeech_test_clean": {
        "language": "en",
        "label": "LibriSpeech test-clean",
        "metric": "wer",
    },
    "librispeech_test_other": {
        "language": "en",
        "label": "LibriSpeech test-other",
        "metric": "wer",
    },
    "commonvoice_en_test": {
        "language": "en",
        "label": "Common Voice 22 en test",
        "metric": "wer",
    },
    "wenetspeech_test_net": {
        "language": "zh",
        "label": "WenetSpeech TEST_NET",
        "metric": "cer",
    },
}


def _load_references(
    path: Path,
    *,
    language: str,
    normalization: str,
) -> dict[str, str]:
    references: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            utt_id = str(record.get("utt_id") or "").strip()
            if not utt_id:
                raise ValueError(f"Missing utt_id at {path}:{line_number}")
            if utt_id in references:
                raise ValueError(f"Duplicate utt_id={utt_id!r} at {path}:{line_number}")
            ref_text = record.get("ref_text")
            if ref_text is None:
                raise ValueError(f"Missing ref_text for utt_id={utt_id!r} at {path}:{line_number}")
            references[utt_id] = normalize_asr_text_for_metrics(
                str(ref_text),
                language=language,
                normalization=normalization,
            )
    if not references:
        raise ValueError(f"No prediction rows found: {path}")
    return references


def _relative_ratio(*, nano_rate: float, student_rate: float) -> float:
    if nano_rate > 0.0:
        return student_rate / nano_rate
    return 1.0 if student_rate <= 0.0 else math.inf


def compare_dataset(
    *,
    dataset: str,
    nano_path: Path,
    student_path: Path,
    normalization: str,
    max_relative_ratio: float,
    max_absolute_gap_points: float,
) -> dict[str, Any]:
    info = DATASETS[dataset]
    language = info["language"]
    metric = info["metric"]
    nano_references = _load_references(
        nano_path,
        language=language,
        normalization=normalization,
    )
    student_references = _load_references(
        student_path,
        language=language,
        normalization=normalization,
    )
    nano_ids = set(nano_references)
    student_ids = set(student_references)
    if nano_ids != student_ids:
        raise ValueError(
            f"{dataset}: prediction coverage differs: "
            f"nano_only={len(nano_ids - student_ids)} "
            f"student_only={len(student_ids - nano_ids)}"
        )
    reference_mismatches = [
        utt_id
        for utt_id in sorted(nano_ids)
        if nano_references[utt_id] != student_references[utt_id]
    ]
    if reference_mismatches:
        preview = ", ".join(reference_mismatches[:5])
        raise ValueError(
            f"{dataset}: normalized references differ for "
            f"{len(reference_mismatches)} utterances; first={preview}"
        )

    nano_stats = compute_text_error_stats(
        nano_path,
        language=language,
        normalization=normalization,
    )
    student_stats = compute_text_error_stats(
        student_path,
        language=language,
        normalization=normalization,
    )
    metric_key = f"avg_{metric}"
    nano_rate = float(nano_stats[metric_key])
    student_rate = float(student_stats[metric_key])
    absolute_gap_points = (student_rate - nano_rate) * 100.0
    relative_ratio = _relative_ratio(nano_rate=nano_rate, student_rate=student_rate)
    comparison = compare_prediction_text_sets(
        nano_path,
        student_path,
        baseline_label="FunASR-Nano CTC",
        candidate_label="BiRWKV CTC",
        language=language,
        normalization=normalization,
        metric=metric,
    )
    absolute_gate_pass = absolute_gap_points <= max_absolute_gap_points
    relative_gate_pass = relative_ratio <= max_relative_ratio
    return {
        "dataset": dataset,
        "label": info["label"],
        "language": language,
        "metric": metric,
        "sample_count": len(nano_ids),
        "identical_utt_coverage": True,
        "normalized_reference_mismatch_count": 0,
        "nano_wer": float(nano_stats["avg_wer"]),
        "student_wer": float(student_stats["avg_wer"]),
        "nano_cer": float(nano_stats["avg_cer"]),
        "student_cer": float(student_stats["avg_cer"]),
        "nano_error_rate": nano_rate,
        "student_error_rate": student_rate,
        "absolute_gap_points": absolute_gap_points,
        "relative_ratio": relative_ratio,
        "absolute_gate_pass": absolute_gate_pass,
        "relative_gate_pass": relative_gate_pass,
        "gate_pass": absolute_gate_pass and relative_gate_pass,
        "changed_prediction_count": int(comparison["changed_prediction_count"]),
        "student_improved_count": int(comparison["improved_count"]),
        "student_worsened_count": int(comparison["worsened_count"]),
        "unchanged_count": int(comparison["unchanged_count"]),
        "nano_prediction_path": str(nano_path),
        "student_prediction_path": str(student_path),
    }


def build_report(
    *,
    student_prediction_dir: Path,
    nano_predictions: dict[str, Path],
    normalization: str,
    max_relative_ratio: float,
    max_absolute_gap_points: float,
) -> dict[str, Any]:
    missing = sorted(set(DATASETS) - set(nano_predictions))
    extra = sorted(set(nano_predictions) - set(DATASETS))
    if missing or extra:
        raise ValueError(f"Nano prediction map mismatch: missing={missing} extra={extra}")
    if max_relative_ratio <= 0.0 or not math.isfinite(max_relative_ratio):
        raise ValueError("max_relative_ratio must be finite and positive")
    if max_absolute_gap_points < 0.0 or not math.isfinite(max_absolute_gap_points):
        raise ValueError("max_absolute_gap_points must be finite and non-negative")

    results: list[dict[str, Any]] = []
    for dataset in DATASETS:
        student_path = student_prediction_dir / f"{dataset}.ctc.jsonl"
        nano_path = nano_predictions[dataset]
        if not student_path.is_file():
            raise FileNotFoundError(f"Student predictions missing: {student_path}")
        if not nano_path.is_file():
            raise FileNotFoundError(f"Nano predictions missing: {nano_path}")
        results.append(
            compare_dataset(
                dataset=dataset,
                nano_path=nano_path,
                student_path=student_path,
                normalization=normalization,
                max_relative_ratio=max_relative_ratio,
                max_absolute_gap_points=max_absolute_gap_points,
            )
        )
    return {
        "version": 1,
        "systems": {
            "baseline": "FunASR-Nano-2512 direct CTC",
            "candidate": "BiRWKV CTC",
        },
        "decode": "greedy_ctc",
        "normalization": normalization,
        "gate": {
            "max_relative_ratio": max_relative_ratio,
            "max_absolute_gap_points": max_absolute_gap_points,
            "requires_every_dataset": True,
        },
        "all_datasets_pass": all(bool(row["gate_pass"]) for row in results),
        "results": results,
    }


def _parse_nano_predictions(values: list[str]) -> dict[str, Path]:
    predictions: dict[str, Path] = {}
    for value in values:
        dataset, separator, raw_path = value.partition("=")
        dataset = dataset.strip()
        raw_path = raw_path.strip()
        if not separator or not dataset or not raw_path:
            raise ValueError(f"Expected DATASET=PATH for --nano-prediction, got {value!r}")
        if dataset in predictions:
            raise ValueError(f"Duplicate --nano-prediction dataset: {dataset}")
        predictions[dataset] = Path(raw_path)
    return predictions


def _format_rate(value: float) -> str:
    return f"{value * 100.0:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# BiRWKV CTC vs FunASR-Nano CTC",
        "",
        (
            "Both systems use greedy CTC on identical real audio and references with "
            f"`{report['normalization']}` normalization."
        ),
        "",
        "| dataset | n | metric | Nano | BiRWKV | gap (pt) | ratio | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        lines.append(
            "| {label} | {sample_count} | {metric} | {nano} | {student} | "
            "{gap:+.3f} | {ratio:.3f}x | {gate_pass} |".format(
                label=row["label"],
                sample_count=row["sample_count"],
                metric=str(row["metric"]).upper(),
                nano=_format_rate(float(row["nano_error_rate"])),
                student=_format_rate(float(row["student_error_rate"])),
                gap=float(row["absolute_gap_points"]),
                ratio=float(row["relative_ratio"]),
                gate_pass="PASS" if row["gate_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            (
                "Gate: every dataset must have "
                f"`student/Nano <= {gate['max_relative_ratio']:.3f}` and "
                f"absolute gap `<= {gate['max_absolute_gap_points']:.3f}` points."
            ),
            "",
            f"Overall: **{'PASS' if report['all_datasets_pass'] else 'FAIL'}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare normalized public real-audio CTC metrics against cached Nano CTC."
    )
    parser.add_argument("--student-prediction-dir", required=True)
    parser.add_argument(
        "--nano-prediction",
        action="append",
        required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--normalization", default="ctc")
    parser.add_argument("--max-relative-ratio", type=float, default=1.20)
    parser.add_argument("--max-absolute-gap-points", type=float, default=3.0)
    args = parser.parse_args()

    report = build_report(
        student_prediction_dir=Path(args.student_prediction_dir),
        nano_predictions=_parse_nano_predictions(args.nano_prediction),
        normalization=args.normalization,
        max_relative_ratio=args.max_relative_ratio,
        max_absolute_gap_points=args.max_absolute_gap_points,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"saved_json={output_json}")
    print(f"saved_md={output_md}")
    print(f"all_datasets_pass={str(report['all_datasets_pass']).lower()}")


if __name__ == "__main__":
    main()
