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
                yield line_number, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def _key(value: Any) -> str:
    return str(value or "").strip()


def _tokens(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _debug_blank_prob(row: dict[str, Any]) -> float | None:
    debug = row.get("debug")
    if not isinstance(debug, dict):
        return None
    value = debug.get("avg_blank_prob")
    if value is None:
        return None
    return float(value)


def _debug_blank_top1(row: dict[str, Any]) -> float | None:
    debug = row.get("debug")
    if not isinstance(debug, dict):
        return None
    value = debug.get("blank_top1_ratio")
    if value is None:
        return None
    return float(value)


def _load_teacher(
    path: Path,
    *,
    teacher_text_key: str,
    teacher_token_key: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for _, row in _iter_jsonl(path):
        utt_id = _key(row.get("utt_id") or row.get("key"))
        if not utt_id:
            continue
        if row.get("funasr_ctc_error") not in (None, "", "null"):
            continue
        text = row.get(teacher_text_key)
        if text is None:
            text = row.get("funasr_ctc_text")
        if text is None:
            text = row.get("pred_text")
        tokens = row.get(teacher_token_key)
        if tokens is None:
            tokens = row.get("funasr_ctc_token_ids")
        if tokens is None:
            tokens = row.get("pred_token_ids")
        source = _key(row.get("source") or row.get("source_dataset") or "unknown").lower() or "unknown"
        language = _key(row.get("language")).lower() or SOURCE_LANGUAGE.get(source)
        rows[utt_id] = {
            "utt_id": utt_id,
            "text": "" if text is None else str(text),
            "tokens": _tokens(tokens),
            "source": source,
            "language": language,
            "avg_blank_prob": (
                None if row.get("funasr_ctc_avg_blank_prob") is None else float(row["funasr_ctc_avg_blank_prob"])
            ),
            "blank_top1_ratio": (
                None if row.get("funasr_ctc_blank_top1_ratio") is None else float(row["funasr_ctc_blank_top1_ratio"])
            ),
        }
    if not rows:
        raise ValueError(f"no usable Nano-CTC teacher rows found in {path}")
    return rows


class Accumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.text_exact = 0
        self.token_exact = 0
        self.wer_errors = 0
        self.cer_errors = 0
        self.ref_words = 0
        self.ref_chars = 0
        self.token_errors = 0
        self.ref_tokens = 0
        self.pred_tokens = 0
        self.length_ratio_sum = 0.0
        self.blank_prob_abs_sum = 0.0
        self.blank_prob_count = 0
        self.blank_top1_abs_sum = 0.0
        self.blank_top1_count = 0

    def add(
        self,
        *,
        pred_text: str,
        teacher_text: str,
        pred_tokens: list[int],
        teacher_tokens: list[int],
        language: str | None,
        normalization: str,
        student_blank_prob: float | None,
        teacher_blank_prob: float | None,
        student_blank_top1: float | None,
        teacher_blank_top1: float | None,
    ) -> None:
        pred_norm = normalize_asr_text_for_metrics(
            pred_text,
            language=language,
            normalization=normalization,
        )
        teacher_norm = normalize_asr_text_for_metrics(
            teacher_text,
            language=language,
            normalization=normalization,
        )
        pred_words = tokenize_for_wer(pred_norm)
        teacher_words = tokenize_for_wer(teacher_norm)
        pred_chars = tokenize_for_cer(pred_norm)
        teacher_chars = tokenize_for_cer(teacher_norm)

        self.samples += 1
        self.text_exact += int(pred_norm == teacher_norm)
        self.token_exact += int(pred_tokens == teacher_tokens)
        self.wer_errors += edit_distance(pred_words, teacher_words)
        self.cer_errors += edit_distance(pred_chars, teacher_chars)
        self.ref_words += len(teacher_words)
        self.ref_chars += len(teacher_chars)
        self.token_errors += edit_distance([str(v) for v in pred_tokens], [str(v) for v in teacher_tokens])
        self.ref_tokens += len(teacher_tokens)
        self.pred_tokens += len(pred_tokens)
        self.length_ratio_sum += float(len(pred_tokens)) / float(max(1, len(teacher_tokens)))
        if student_blank_prob is not None and teacher_blank_prob is not None:
            self.blank_prob_abs_sum += abs(float(student_blank_prob) - float(teacher_blank_prob))
            self.blank_prob_count += 1
        if student_blank_top1 is not None and teacher_blank_top1 is not None:
            self.blank_top1_abs_sum += abs(float(student_blank_top1) - float(teacher_blank_top1))
            self.blank_top1_count += 1

    def merge(self, other: "Accumulator") -> None:
        self.samples += other.samples
        self.text_exact += other.text_exact
        self.token_exact += other.token_exact
        self.wer_errors += other.wer_errors
        self.cer_errors += other.cer_errors
        self.ref_words += other.ref_words
        self.ref_chars += other.ref_chars
        self.token_errors += other.token_errors
        self.ref_tokens += other.ref_tokens
        self.pred_tokens += other.pred_tokens
        self.length_ratio_sum += other.length_ratio_sum
        self.blank_prob_abs_sum += other.blank_prob_abs_sum
        self.blank_prob_count += other.blank_prob_count
        self.blank_top1_abs_sum += other.blank_top1_abs_sum
        self.blank_top1_count += other.blank_top1_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "nano_wer": float(self.wer_errors) / float(max(1, self.ref_words)),
            "nano_cer": float(self.cer_errors) / float(max(1, self.ref_chars)),
            "nano_token_er": float(self.token_errors) / float(max(1, self.ref_tokens)),
            "text_exact_rate": float(self.text_exact) / float(max(1, self.samples)),
            "token_exact_rate": float(self.token_exact) / float(max(1, self.samples)),
            "wer_errors": self.wer_errors,
            "cer_errors": self.cer_errors,
            "token_errors": self.token_errors,
            "ref_words": self.ref_words,
            "ref_chars": self.ref_chars,
            "ref_tokens": self.ref_tokens,
            "pred_tokens": self.pred_tokens,
            "avg_token_length_ratio": self.length_ratio_sum / float(max(1, self.samples)),
            "avg_blank_prob_abs_diff": (
                None if self.blank_prob_count == 0 else self.blank_prob_abs_sum / float(self.blank_prob_count)
            ),
            "blank_prob_samples": self.blank_prob_count,
            "avg_blank_top1_abs_diff": (
                None if self.blank_top1_count == 0 else self.blank_top1_abs_sum / float(self.blank_top1_count)
            ),
            "blank_top1_samples": self.blank_top1_count,
        }


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty candidate label in {value!r}")
    return label, Path(raw_path)


def _summarize_candidate(
    *,
    label: str,
    path: Path,
    teacher: dict[str, dict[str, Any]],
    student_text_key: str,
    student_token_key: str,
    normalization: str,
) -> dict[str, Any]:
    overall = Accumulator()
    by_source: dict[str, Accumulator] = defaultdict(Accumulator)
    missing_teacher = 0
    duplicate_student = 0
    seen_student: set[str] = set()
    examples: list[dict[str, Any]] = []

    for _, row in _iter_jsonl(path):
        utt_id = _key(row.get("utt_id") or row.get("key"))
        if not utt_id:
            continue
        if utt_id in seen_student:
            duplicate_student += 1
            continue
        seen_student.add(utt_id)
        teacher_row = teacher.get(utt_id)
        if teacher_row is None:
            missing_teacher += 1
            continue
        pred_text = "" if row.get(student_text_key) is None else str(row.get(student_text_key))
        pred_tokens = _tokens(row.get(student_token_key))
        teacher_text = str(teacher_row["text"])
        teacher_tokens = list(teacher_row["tokens"])
        source = str(teacher_row["source"])
        language = teacher_row.get("language")
        if language is not None:
            language = str(language)
        overall.add(
            pred_text=pred_text,
            teacher_text=teacher_text,
            pred_tokens=pred_tokens,
            teacher_tokens=teacher_tokens,
            language=language,
            normalization=normalization,
            student_blank_prob=_debug_blank_prob(row),
            teacher_blank_prob=teacher_row.get("avg_blank_prob"),
            student_blank_top1=_debug_blank_top1(row),
            teacher_blank_top1=teacher_row.get("blank_top1_ratio"),
        )
        by_source[source].add(
            pred_text=pred_text,
            teacher_text=teacher_text,
            pred_tokens=pred_tokens,
            teacher_tokens=teacher_tokens,
            language=language,
            normalization=normalization,
            student_blank_prob=_debug_blank_prob(row),
            teacher_blank_prob=teacher_row.get("avg_blank_prob"),
            student_blank_top1=_debug_blank_top1(row),
            teacher_blank_top1=teacher_row.get("blank_top1_ratio"),
        )
        if len(examples) < 10:
            pred_norm = normalize_asr_text_for_metrics(pred_text, language=language, normalization=normalization)
            teacher_norm = normalize_asr_text_for_metrics(teacher_text, language=language, normalization=normalization)
            if pred_norm != teacher_norm:
                examples.append(
                    {
                        "utt_id": utt_id,
                        "source": source,
                        "student": pred_text,
                        "nano_ctc": teacher_text,
                        "student_token_ids": pred_tokens,
                        "nano_ctc_token_ids": teacher_tokens,
                    }
                )

    return {
        "label": label,
        "path": str(path),
        "missing_teacher": missing_teacher,
        "duplicate_student": duplicate_student,
        "overall": overall.as_dict(),
        "by_source": {source: acc.as_dict() for source, acc in sorted(by_source.items())},
        "examples": examples,
    }


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.4f}"


def _format_num(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Nano-CTC Imitation Metrics",
        "",
        f"Teacher: `{summary['teacher_jsonl']}`",
        "",
        "| candidate | samples | Nano WER | Nano CER | Nano token ER | text exact | token exact | len ratio | blank prob diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        stats = result["overall"]
        lines.append(
            "| {label} | {samples} | {wer} | {cer} | {ter} | {text_exact} | {token_exact} | {length_ratio} | {blank_diff} |".format(
                label=result["label"],
                samples=stats["samples"],
                wer=_format_pct(stats["nano_wer"]),
                cer=_format_pct(stats["nano_cer"]),
                ter=_format_pct(stats["nano_token_er"]),
                text_exact=_format_pct(stats["text_exact_rate"]),
                token_exact=_format_pct(stats["token_exact_rate"]),
                length_ratio=_format_num(stats["avg_token_length_ratio"]),
                blank_diff=_format_num(stats["avg_blank_prob_abs_diff"]),
            )
        )
    lines.append("")
    lines.append("## By Source")
    lines.append("")
    lines.append("| candidate | source | samples | Nano WER | Nano CER | Nano token ER | text exact | token exact |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for result in summary["results"]:
        for source, stats in result["by_source"].items():
            lines.append(
                "| {label} | {source} | {samples} | {wer} | {cer} | {ter} | {text_exact} | {token_exact} |".format(
                    label=result["label"],
                    source=source,
                    samples=stats["samples"],
                    wer=_format_pct(stats["nano_wer"]),
                    cer=_format_pct(stats["nano_cer"]),
                    ter=_format_pct(stats["nano_token_er"]),
                    text_exact=_format_pct(stats["text_exact_rate"]),
                    token_exact=_format_pct(stats["token_exact_rate"]),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare student CTC prediction JSONL files directly against exported FunASR-Nano CTC."
    )
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument(
        "--student-jsonl",
        action="append",
        required=True,
        help="Candidate as label=path or just path. May be repeated.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--normalization", default="ctc")
    parser.add_argument("--teacher-text-key", default="funasr_ctc_text")
    parser.add_argument("--teacher-token-key", default="funasr_ctc_token_ids")
    parser.add_argument("--student-text-key", default="pred_text")
    parser.add_argument("--student-token-key", default="pred_token_ids")
    args = parser.parse_args()

    teacher_path = Path(args.teacher_jsonl)
    teacher = _load_teacher(
        teacher_path,
        teacher_text_key=str(args.teacher_text_key),
        teacher_token_key=str(args.teacher_token_key),
    )
    results = []
    for candidate in args.student_jsonl:
        label, path = _parse_candidate(str(candidate))
        results.append(
            _summarize_candidate(
                label=label,
                path=path,
                teacher=teacher,
                student_text_key=str(args.student_text_key),
                student_token_key=str(args.student_token_key),
                normalization=str(args.normalization),
            )
        )

    summary = {
        "version": 1,
        "teacher_jsonl": str(teacher_path),
        "teacher_rows": len(teacher),
        "normalization": str(args.normalization),
        "student_text_key": str(args.student_text_key),
        "student_token_key": str(args.student_token_key),
        "teacher_text_key": str(args.teacher_text_key),
        "teacher_token_key": str(args.teacher_token_key),
        "results": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_json={output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        _write_markdown(output_md, summary)
        print(f"saved_md={output_md}")


if __name__ == "__main__":
    main()
