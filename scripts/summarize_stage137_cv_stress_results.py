#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rwkvasr.eval.text_metrics import (
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)


SPLITS = (
    ("cv_stress_guard", "all"),
    ("cv_hard", "hard"),
    ("cv_medium", "medium"),
    ("cv_easy", "easy"),
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _as_key(value: Any) -> str:
    return str(value or "").strip()


def _load_bucket_map(lengths_dir: Path) -> dict[str, str]:
    bucket_by_id: dict[str, str] = {}
    for path in sorted(lengths_dir.glob("cv_*.lengths.jsonl")):
        for row in _iter_jsonl(path):
            bucket = _as_key(row.get("_stage137_bucket"))
            if not bucket:
                if path.name == "cv_hard.lengths.jsonl":
                    bucket = "hard"
                elif path.name == "cv_medium.lengths.jsonl":
                    bucket = "medium"
                elif path.name == "cv_easy.lengths.jsonl":
                    bucket = "easy"
            if not bucket or bucket == "stress_guard":
                continue
            for key in (row.get("utt_id"), row.get("id"), row.get("key")):
                key_text = _as_key(key)
                if key_text:
                    bucket_by_id[key_text] = bucket
    return bucket_by_id


class Accumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.wer_errors = 0
        self.cer_errors = 0
        self.ref_words = 0
        self.ref_chars = 0

    def add(self, *, pred_text: str, ref_text: str, normalization: str) -> None:
        pred_norm = normalize_asr_text_for_metrics(
            pred_text,
            language="en",
            normalization=normalization,
        )
        ref_norm = normalize_asr_text_for_metrics(
            ref_text,
            language="en",
            normalization=normalization,
        )
        pred_words = tokenize_for_wer(pred_norm)
        ref_words = tokenize_for_wer(ref_norm)
        pred_chars = tokenize_for_cer(pred_norm)
        ref_chars = tokenize_for_cer(ref_norm)
        self.samples += 1
        self.wer_errors += edit_distance(pred_words, ref_words)
        self.cer_errors += edit_distance(pred_chars, ref_chars)
        self.ref_words += len(ref_words)
        self.ref_chars += len(ref_chars)

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "wer": float(self.wer_errors) / float(max(1, self.ref_words)),
            "cer": float(self.cer_errors) / float(max(1, self.ref_chars)),
            "wer_errors": self.wer_errors,
            "cer_errors": self.cer_errors,
            "ref_words": self.ref_words,
            "ref_chars": self.ref_chars,
        }


def _format_pct(value: Any) -> str:
    return "n/a" if not isinstance(value, float) else f"{value * 100.0:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage137 CommonVoice EN stress predictions.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--lengths-dir", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--normalization", default="ctc")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    rows: list[dict[str, Any]] = []
    combined_path = prediction_dir / "cv_stress_guard.ctc.jsonl"
    if args.lengths_dir is not None and combined_path.exists():
        bucket_by_id = _load_bucket_map(Path(args.lengths_dir))
        total = Accumulator()
        by_bucket: dict[str, Accumulator] = defaultdict(Accumulator)
        missing_bucket = 0
        for row in _iter_jsonl(combined_path):
            utt_id = _as_key(row.get("utt_id"))
            bucket = bucket_by_id.get(utt_id)
            if bucket is None:
                missing_bucket += 1
                bucket = "unknown"
            pred_text = _as_key(row.get("pred_text"))
            ref_text = _as_key(row.get("ref_text"))
            total.add(pred_text=pred_text, ref_text=ref_text, normalization=str(args.normalization))
            by_bucket[bucket].add(
                pred_text=pred_text,
                ref_text=ref_text,
                normalization=str(args.normalization),
            )
        rows.append({"split": "all", "path": str(combined_path), **total.as_dict()})
        for bucket in ("hard", "medium", "easy", "unknown"):
            acc = by_bucket.get(bucket)
            if acc is not None and acc.samples:
                rows.append({"split": bucket, "path": str(combined_path), **acc.as_dict()})
        output_extra = {"missing_bucket": missing_bucket}
    else:
        output_extra = {}

    if not rows:
        for stem, label in SPLITS:
            path = prediction_dir / f"{stem}.ctc.jsonl"
            if not path.exists():
                continue
            acc = Accumulator()
            for row in _iter_jsonl(path):
                acc.add(
                    pred_text=_as_key(row.get("pred_text")),
                    ref_text=_as_key(row.get("ref_text")),
                    normalization=str(args.normalization),
                )
            rows.append({"split": label, "path": str(path), **acc.as_dict()})

    output = {
        "prediction_dir": str(prediction_dir),
        "normalization": str(args.normalization),
        "results": rows,
        **output_extra,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage137 CV Stress Metrics",
        "",
        "| split | samples | WER | CER |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row.get('samples', 'n/a')} | "
            f"{_format_pct(row.get('wer'))} | {_format_pct(row.get('cer'))} |"
        )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_json={output_json}")
    print(f"saved_md={output_md}")


if __name__ == "__main__":
    main()
