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


SOURCE_LANGUAGE = {
    "aishell3": "zh",
    "commonvoice_cn": "zh",
    "wenetspeech": "zh",
    "commonvoice_en": "en",
    "gigaspeech": "en",
    "librispeech": "en",
}


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


def _load_source_map(paths: list[Path]) -> dict[str, str]:
    source_by_id: dict[str, str] = {}
    for path in paths:
        for row in _iter_jsonl(path):
            source = _as_key(row.get("_stage131_source") or row.get("source_dataset") or row.get("source"))
            if not source:
                continue
            for key in (
                row.get("utt_id"),
                row.get("id"),
                row.get("key"),
                row.get("audio_member"),
                row.get("json_member"),
            ):
                key_text = _as_key(key)
                if key_text:
                    source_by_id[key_text] = source
            audio_member = _as_key(row.get("audio_member"))
            if audio_member:
                stem = Path(audio_member).stem
                if stem:
                    source_by_id[stem] = source
    return source_by_id


class Accumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.wer_errors = 0
        self.cer_errors = 0
        self.ref_words = 0
        self.ref_chars = 0

    def add(self, *, pred_text: str, ref_text: str, language: str | None, normalization: str) -> None:
        pred_norm = normalize_asr_text_for_metrics(
            pred_text,
            language=language,
            normalization=normalization,
        )
        ref_norm = normalize_asr_text_for_metrics(
            ref_text,
            language=language,
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


def _summarize_file(
    *,
    name: str,
    path: Path,
    source_by_id: dict[str, str],
    normalization: str,
) -> dict[str, Any]:
    overall = Accumulator()
    by_source: dict[str, Accumulator] = defaultdict(Accumulator)
    missing_source = 0
    for row in _iter_jsonl(path):
        utt_id = _as_key(row.get("utt_id"))
        source = source_by_id.get(utt_id, "unknown")
        if source == "unknown":
            missing_source += 1
        language = SOURCE_LANGUAGE.get(source)
        pred_text = _as_key(row.get("pred_text"))
        ref_text = _as_key(row.get("ref_text"))
        overall.add(pred_text=pred_text, ref_text=ref_text, language=language, normalization=normalization)
        by_source[source].add(
            pred_text=pred_text,
            ref_text=ref_text,
            language=language,
            normalization=normalization,
        )
    return {
        "name": name,
        "path": str(path),
        "missing_source": missing_source,
        "overall": overall.as_dict(),
        "by_source": {source: acc.as_dict() for source, acc in sorted(by_source.items())},
    }


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage131 private guard predictions.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--clean-lengths", required=True)
    parser.add_argument("--hard-lengths", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--normalization", default="ctc")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    source_by_id = _load_source_map([Path(args.clean_lengths), Path(args.hard_lengths)])
    results = []
    for name in ("clean_guard", "hard_guard"):
        path = prediction_dir / f"{name}.ctc.jsonl"
        if not path.exists():
            continue
        results.append(
            _summarize_file(
                name=name,
                path=path,
                source_by_id=source_by_id,
                normalization=str(args.normalization),
            )
        )

    total = Accumulator()
    for result in results:
        overall = result["overall"]
        total.samples += int(overall["samples"])
        total.wer_errors += int(overall["wer_errors"])
        total.cer_errors += int(overall["cer_errors"])
        total.ref_words += int(overall["ref_words"])
        total.ref_chars += int(overall["ref_chars"])

    output = {
        "prediction_dir": str(prediction_dir),
        "normalization": str(args.normalization),
        "total": total.as_dict(),
        "results": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Private Guard Metrics",
        "",
        "| split | source | samples | WER | CER |",
        "|---|---:|---:|---:|---:|",
        f"| total | all | {total.samples} | {_format_pct(total.as_dict()['wer'])} | {_format_pct(total.as_dict()['cer'])} |",
    ]
    for result in results:
        overall = result["overall"]
        lines.append(
            f"| {result['name']} | all | {overall['samples']} | "
            f"{_format_pct(overall['wer'])} | {_format_pct(overall['cer'])} |"
        )
        for source, stats in result["by_source"].items():
            lines.append(
                f"| {result['name']} | {source} | {stats['samples']} | "
                f"{_format_pct(stats['wer'])} | {_format_pct(stats['cer'])} |"
            )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved_json={output_json}")
    print(f"saved_md={output_md}")


if __name__ == "__main__":
    main()
