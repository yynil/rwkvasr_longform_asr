#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.eval import compute_text_error_stats


DATASETS: dict[str, dict[str, str]] = {
    "aishell1_test": {"language": "zh", "condition": "clean", "label": "AISHELL-1 test"},
    "librispeech_test_clean": {"language": "en", "condition": "clean", "label": "LibriSpeech test-clean"},
    "librispeech_test_other": {"language": "en", "condition": "clean", "label": "LibriSpeech test-other"},
    "commonvoice_en_test": {"language": "en", "condition": "non_clean", "label": "Common Voice 22 en test"},
    "wenetspeech_test_net": {"language": "zh", "condition": "non_clean", "label": "WenetSpeech TEST_NET"},
}


def _format_rate(value: Any) -> str:
    return "n/a" if not isinstance(value, float) else f"{value * 100.0:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize normalized WER/CER from benchmark pred/ref JSONL files.")
    parser.add_argument("--prediction-dir", default="artifacts/eval_benchmarks/predictions")
    parser.add_argument("--output-json", default="artifacts/eval_benchmarks/metrics.json")
    parser.add_argument("--output-md", default="artifacts/eval_benchmarks/metrics.md")
    parser.add_argument("--normalization", default="ctc")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    rows: list[dict[str, Any]] = []
    for dataset_name, info in DATASETS.items():
        for branch in ("ctc", "ar"):
            path = prediction_dir / f"{dataset_name}.{branch}.jsonl"
            if not path.exists():
                continue
            stats = compute_text_error_stats(
                path,
                language=info["language"],
                normalization=args.normalization,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "label": info["label"],
                    "condition": info["condition"],
                    "language": info["language"],
                    "branch": branch,
                    "path": str(path),
                    "samples": stats.get("sample_count"),
                    "wer": stats.get("avg_wer"),
                    "cer": stats.get("avg_cer"),
                }
            )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ASR Eval Metrics",
        "",
        "WER/CER are normalized ASR content metrics; values below are percentages.",
        "",
        "| condition | language | dataset | branch | samples | WER | CER |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {language} | {label} | {branch} | {samples} | {wer} | {cer} |".format(
                condition=row["condition"],
                language=row["language"],
                label=row["label"],
                branch=row["branch"],
                samples=row.get("samples", "n/a"),
                wer=_format_rate(row.get("wer")),
                cer=_format_rate(row.get("cer")),
            )
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_json={output_json}")
    print(f"saved_md={output_md}")


if __name__ == "__main__":
    main()
