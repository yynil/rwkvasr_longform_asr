from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from rwkvasr.eval.text_metrics import (
    compare_prediction_text_sets,
    compute_text_error_stats,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)
from rwkvasr.training.funasr_online_teacher import (
    FunASRNanoCTCTopKOnlineTeacher,
    FunASROnlineCTCTeacherConfig,
)


@dataclass(frozen=True)
class FunASRNanoCTCManifestEvalConfig:
    manifest_path: str
    model_path: str
    predictions_path: str
    report_path: str
    language: str
    device: str = "cpu"
    normalization: str = "ctc"
    limit: int | None = None
    progress_interval: int = 100
    student_predictions_path: str | None = None


def _read_manifest_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            utt_id = str(row.get("utt_id") or row.get("id") or row.get("key") or "").strip()
            if not utt_id:
                raise ValueError(f"Manifest row has no utterance id at {path}:{line_number}")
            if utt_id in seen_ids:
                raise ValueError(f"Duplicate utt_id={utt_id!r} at {path}:{line_number}")
            if not str(row.get("audio_filepath") or row.get("audio_path") or "").strip():
                raise ValueError(f"Manifest row has no audio path at {path}:{line_number}")
            if row.get("text") is None and row.get("ref_text") is None:
                raise ValueError(f"Manifest row has no reference text at {path}:{line_number}")
            row = dict(row)
            row["utt_id"] = utt_id
            rows.append(row)
            seen_ids.add(utt_id)
            if limit is not None and len(rows) >= int(limit):
                break
    if not rows:
        raise ValueError(f"No usable rows found in manifest: {path}")
    return rows


def _teacher_source_for_language(language: str) -> str:
    if language == "en":
        return "librispeech"
    if language == "zh":
        return "aishell3"
    raise ValueError(f"language must be 'en' or 'zh', got {language!r}")


def _compact_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in stats.items()
        if key not in {"per_sample_wer", "per_sample_cer"}
    }


def _prediction_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                ids.add(str(json.loads(raw).get("utt_id") or ""))
    ids.discard("")
    return ids


def evaluate_funasr_nano_ctc_manifest(
    config: FunASRNanoCTCManifestEvalConfig,
    *,
    teacher: Any | None = None,
) -> dict[str, Any]:
    if config.limit is not None and int(config.limit) <= 0:
        raise ValueError("limit must be positive when provided.")
    if int(config.progress_interval) < 0:
        raise ValueError("progress_interval must be non-negative.")

    manifest_path = Path(config.manifest_path)
    rows = _read_manifest_rows(manifest_path, limit=config.limit)
    teacher_source = _teacher_source_for_language(config.language)
    if teacher is None:
        teacher = FunASRNanoCTCTopKOnlineTeacher(
            FunASROnlineCTCTeacherConfig(
                model_path=str(config.model_path),
                device=str(config.device),
                split="all",
                top_k=2,
            )
        )

    decode = teacher.model.ctc_tokenizer.decode
    predictions_path = Path(config.predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    total_pred_units = 0
    total_ref_units = 0
    total_frames = 0
    blank_top1_sum = 0.0
    blank_probability_sum = 0.0
    total_audio_sec = 0.0

    with predictions_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            teacher_row = dict(row)
            teacher_row["source_dataset"] = teacher_source
            utt_id = str(row["utt_id"])
            records = teacher.topk_records([utt_id], [teacher_row])
            record = records.get(utt_id)
            if record is None:
                raise RuntimeError(f"FunASR-Nano returned no CTC record for utt_id={utt_id!r}")

            token_ids = [int(value) for value in torch.as_tensor(record["argmax_token_ids"]).tolist()]
            pred_text = str(decode(token_ids))
            ref_text = str(row.get("text") if row.get("text") is not None else row.get("ref_text"))
            normalized_pred = normalize_asr_text_for_metrics(
                pred_text,
                language=config.language,
                normalization=config.normalization,
            )
            normalized_ref = normalize_asr_text_for_metrics(
                ref_text,
                language=config.language,
                normalization=config.normalization,
            )
            if config.language == "en":
                pred_units = tokenize_for_wer(normalized_pred)
                ref_units = tokenize_for_wer(normalized_ref)
            else:
                pred_units = tokenize_for_cer(normalized_pred)
                ref_units = tokenize_for_cer(normalized_ref)

            topk_ids = torch.as_tensor(record["topk_token_ids"])
            project_blank_id = int(record["project_blank_id"])
            blank_top1_ratio = (
                float(topk_ids[:, 0].eq(project_blank_id).float().mean().item())
                if topk_ids.numel() > 0
                else 0.0
            )
            blank_log_probs = torch.as_tensor(record["blank_log_probs"], dtype=torch.float32)
            avg_blank_probability = (
                float(blank_log_probs.exp().mean().item()) if blank_log_probs.numel() > 0 else 0.0
            )
            frame_count = int(record["num_frames"])
            duration_sec = float(row.get("duration_sec") or 0.0)
            output.write(
                json.dumps(
                    {
                        "utt_id": utt_id,
                        "pred_token_ids": token_ids,
                        "pred_text": pred_text,
                        "ref_text": ref_text,
                        "mode": "funasr_nano_ctc_greedy",
                        "duration_sec": duration_sec,
                        "debug": {
                            "logit_length": frame_count,
                            "pred_token_count": len(token_ids),
                            "blank_top1_ratio": blank_top1_ratio,
                            "avg_blank_prob": avg_blank_probability,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % 100 == 0:
                output.flush()

            total_pred_units += len(pred_units)
            total_ref_units += len(ref_units)
            total_frames += frame_count
            blank_top1_sum += blank_top1_ratio
            blank_probability_sum += avg_blank_probability
            total_audio_sec += duration_sec
            if config.progress_interval and index % int(config.progress_interval) == 0:
                elapsed = time.monotonic() - started_at
                print(
                    f"[funasr-nano-ctc-eval] samples={index}/{len(rows)} "
                    f"elapsed={elapsed:.1f}s rate={index / max(elapsed, 1e-9):.2f}/s",
                    file=sys.stderr,
                    flush=True,
                )
        output.flush()

    elapsed_sec = time.monotonic() - started_at
    metrics = _compact_metrics(
        compute_text_error_stats(
            predictions_path,
            language=config.language,
            normalization=config.normalization,
        )
    )
    report: dict[str, Any] = {
        "version": 1,
        "system": "FunASR-Nano-2512 direct CTC",
        "model_path": str(config.model_path),
        "manifest_path": str(manifest_path),
        "predictions_path": str(predictions_path),
        "language": config.language,
        "normalization": config.normalization,
        "decode": "greedy_ctc",
        "device": config.device,
        "requested_limit": config.limit,
        "sample_count": len(rows),
        "elapsed_sec": elapsed_sec,
        "audio_hours": total_audio_sec / 3600.0,
        "rtf": elapsed_sec / total_audio_sec if total_audio_sec > 0.0 else None,
        "metrics": metrics,
        "diagnostics": {
            "pred_units": total_pred_units,
            "ref_units": total_ref_units,
            "pred_ref_unit_ratio": total_pred_units / max(1, total_ref_units),
            "mean_logit_frames": total_frames / len(rows),
            "mean_blank_top1_ratio": blank_top1_sum / len(rows),
            "mean_blank_probability": blank_probability_sum / len(rows),
        },
    }

    if config.student_predictions_path is not None:
        student_path = Path(config.student_predictions_path)
        teacher_ids = _prediction_ids(predictions_path)
        student_ids = _prediction_ids(student_path)
        metric = "wer" if config.language == "en" else "cer"
        comparison = compare_prediction_text_sets(
            predictions_path,
            student_path,
            baseline_label="FunASR-Nano CTC",
            candidate_label="BiRWKV CTC",
            language=config.language,
            normalization=config.normalization,
            metric=metric,
        )
        comparison.update(
            {
                "teacher_sample_count": len(teacher_ids),
                "student_sample_count": len(student_ids),
                "identical_utt_coverage": teacher_ids == student_ids,
                "teacher_only_count": len(teacher_ids - student_ids),
                "student_only_count": len(student_ids - teacher_ids),
            }
        )
        report["student_predictions_path"] = str(student_path)
        report["comparison"] = comparison

    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
