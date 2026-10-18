from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwkvasr.eval.funasr_nano_ctc import (
    FunASRNanoCTCManifestEvalConfig,
    evaluate_funasr_nano_ctc_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the direct FunASR-Nano CTC branch on a labeled audio manifest."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--language", required=True, choices=("en", "zh"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--normalization", default="ctc")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--progress-interval", default=100, type=int)
    parser.add_argument("--student-predictions-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictions_path = Path(args.predictions_path)
    report_path = (
        Path(args.report_path)
        if args.report_path is not None
        else predictions_path.with_suffix(".report.json")
    )
    report = evaluate_funasr_nano_ctc_manifest(
        FunASRNanoCTCManifestEvalConfig(
            manifest_path=args.manifest_path,
            model_path=args.model_path,
            predictions_path=str(predictions_path),
            report_path=str(report_path),
            language=args.language,
            device=args.device,
            normalization=args.normalization,
            limit=args.limit,
            progress_interval=args.progress_interval,
            student_predictions_path=args.student_predictions_path,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
