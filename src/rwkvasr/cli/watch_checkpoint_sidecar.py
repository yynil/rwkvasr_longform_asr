from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.eval import (
    compare_prediction_text_sets,
    compute_text_error_stats,
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch exported step checkpoints and run small labeled prediction previews."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tmux-target", default="training:0")
    parser.add_argument("--output-subdir", default="sidecar_eval")
    parser.add_argument("--poll-seconds", default=3600.0, type=float)
    parser.add_argument("--checkpoint-stable-seconds", default=30.0, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--mode", default="bi", choices=["bi", "l2r", "r2l", "alt"])
    parser.add_argument("--beam-size", default=4, type=int)
    parser.add_argument("--token-prune-topk", default=32, type=int)
    parser.add_argument("--text-normalization", default="runtime")
    parser.add_argument(
        "--metric-normalization",
        default="ctc",
        help=(
            "Text normalization used only for WER/CER scoring. Keep this separate from "
            "--text-normalization so prediction previews can stay readable while metrics "
            "use ASR-content normalization."
        ),
    )
    parser.add_argument("--ar-max-new-tokens", default=None, type=int)
    parser.add_argument("--ar-max-new-tokens-factor", default=2.0, type=float)
    parser.add_argument("--ar-do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ar-temperature", default=0.8, type=float)
    parser.add_argument("--ar-top-k", default=5, type=int)
    parser.add_argument("--ar-top-p", default=0.8, type=float)
    parser.add_argument("--ar-seed", default=20260615, type=int)
    parser.add_argument("--ar-ctc-draft-fallback-max-cer", default=None, type=float)
    parser.add_argument("--ar-ctc-draft-fallback-min-length-ratio", default=0.75, type=float)
    parser.add_argument("--ar-ctc-draft-fallback-max-length-ratio", default=1.25, type=float)
    parser.add_argument(
        "--ar-ctc-draft-fallback-reject-repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ar-ctc-draft-fallback-metric-normalization", default="ctc")
    parser.add_argument("--limit", default=12, type=int)
    parser.add_argument("--preview-count", default=12, type=int)
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        metavar="LABEL:SHARD_GLOB:COUNT",
        help=(
            "Evaluate a fixed number of preview samples from one WebDataset source. "
            "Can be repeated, e.g. gigaspeech:GSXL-*.tar:6 and wenetspeech:WSL-*.tar:6."
        ),
    )
    parser.add_argument(
        "--webdataset-split",
        default="eval",
        choices=("train", "eval", "all"),
        help=(
            "WebDataset split used for sidecar preview sampling. Use train for sampled "
            "train-only sidecar datasets; keep eval for validation-style previews."
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save-debug-lengths", action="store_true", default=True)
    return parser


def _log(message: str) -> None:
    print(f"[rwkvasr-sidecar] {message}", flush=True)


def _capture_tmux_tail(tmux_target: str, output_path: Path) -> None:
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-pt", tmux_target],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        output_path.write_text(f"tmux capture failed for {tmux_target}: {exc}\n", encoding="utf-8")
        return
    lines = result.stdout.splitlines()
    tail = lines[-80:]
    output_path.write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")


def _load_train_config(run_dir: Path) -> dict[str, Any]:
    train_config_path = run_dir / "train_config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config.yaml not found under {run_dir}")
    return dict(load_yaml(train_config_path))


def _sidecar_paths(run_dir: Path, output_subdir: str) -> tuple[Path, Path, Path]:
    sidecar_dir = run_dir / output_subdir
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    state_path = sidecar_dir / "watch_state.yaml"
    tmux_tail_path = sidecar_dir / "tmux_tail.txt"
    return sidecar_dir, state_path, tmux_tail_path


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"evaluated": {}}
    raw = load_yaml(state_path)
    if not isinstance(raw, dict):
        return {"evaluated": {}}
    evaluated = raw.get("evaluated")
    if not isinstance(evaluated, dict):
        raw["evaluated"] = {}
    return raw


def _parse_step_number(name: str) -> int | None:
    stem = Path(name).stem
    if not stem.startswith("step-"):
        return None
    try:
        return int(stem.split("-")[-1])
    except ValueError:
        return None


def _list_step_checkpoints(run_dir: Path) -> list[Path]:
    def _step_key(path: Path) -> int:
        step = _parse_step_number(path.name)
        return -1 if step is None else step

    return sorted(run_dir.glob("step-*.pt"), key=_step_key)


def _is_stable(path: Path, *, stable_seconds: float) -> bool:
    age = time.time() - path.stat().st_mtime
    return age >= max(0.0, stable_seconds)


def _load_step_metrics(run_dir: Path) -> dict[int, dict[str, Any]]:
    metrics_path = run_dir / "step_checkpoint_metrics.yaml"
    if not metrics_path.exists():
        return {}
    raw = load_yaml(metrics_path)
    if not isinstance(raw, dict):
        return {}
    metrics = raw.get("step_checkpoints", [])
    if not isinstance(metrics, list):
        return {}
    by_step: dict[int, dict[str, Any]] = {}
    for item in metrics:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if not isinstance(step, int):
            continue
        by_step[step] = item
    return by_step


@dataclass(frozen=True)
class SourceQuota:
    label: str
    shard_pattern: str
    limit: int


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return label.strip("._-") or "source"


def _parse_source_quotas(raw_values: list[str]) -> list[SourceQuota]:
    quotas: list[SourceQuota] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        parts = raw_value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "--source-quota must use LABEL:SHARD_GLOB:COUNT, "
                f"got {raw_value!r}"
            )
        label, shard_pattern, count_raw = (part.strip() for part in parts)
        if not label:
            raise ValueError(f"--source-quota label must not be empty: {raw_value!r}")
        if not shard_pattern:
            raise ValueError(f"--source-quota shard glob must not be empty: {raw_value!r}")
        try:
            limit = int(count_raw)
        except ValueError as exc:
            raise ValueError(f"--source-quota COUNT must be an integer: {raw_value!r}") from exc
        if limit < 1:
            raise ValueError(f"--source-quota COUNT must be >= 1: {raw_value!r}")
        safe = _safe_label(label)
        if safe in seen:
            raise ValueError(f"Duplicate --source-quota label after sanitizing: {label!r}")
        seen.add(safe)
        quotas.append(SourceQuota(label=safe, shard_pattern=shard_pattern, limit=limit))
    return quotas


def _build_predict_command(
    *,
    module_name: str,
    checkpoint_path: Path,
    output_jsonl: Path,
    preview_path: Path,
    config_yaml_override: Path | None,
    train_config: dict[str, Any],
    args: argparse.Namespace,
    limit: int | None = None,
    preview_count: int | None = None,
    webdataset_shard_pattern: str | None = None,
    decoder_ctc_draft_cache_path: Path | None = None,
) -> list[str]:
    resolved_limit = args.limit if limit is None else int(limit)
    resolved_preview_count = args.preview_count if preview_count is None else int(preview_count)
    command = [
        sys.executable,
        "-m",
        module_name,
        "--checkpoint-path",
        str(checkpoint_path),
        "--output-path",
        str(output_jsonl),
        "--preview-path",
        str(preview_path),
        "--preview-count",
        str(resolved_preview_count),
        "--limit",
        str(resolved_limit),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--mode",
        args.mode,
    ]
    if module_name == "rwkvasr.cli.predict_ctc_labeled":
        command.extend(
            [
                "--num-workers",
                str(args.num_workers),
                "--beam-size",
                str(args.beam_size),
                "--token-prune-topk",
                str(args.token_prune_topk),
                "--decoder-ctc-draft-cache-path",
                "",
                "--decoder-ctc-draft-prompt-template",
                "",
                "--decoder-ctc-draft-missing-policy",
                "empty",
            ]
        )
    elif module_name == "rwkvasr.cli.predict_rwkv_decoder_labeled":
        command.extend(
            [
                "--num-workers",
                str(args.num_workers),
                "--max-new-tokens-factor",
                str(args.ar_max_new_tokens_factor),
                *(["--do-sample"] if args.ar_do_sample else []),
                "--temperature",
                str(args.ar_temperature),
                "--top-k",
                str(args.ar_top_k),
                "--top-p",
                str(args.ar_top_p),
                "--seed",
                str(args.ar_seed),
            ]
        )
        if args.ar_max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(args.ar_max_new_tokens)])
        if decoder_ctc_draft_cache_path is not None:
            command.extend(["--decoder-ctc-draft-cache-path", str(decoder_ctc_draft_cache_path)])
        if args.ar_ctc_draft_fallback_max_cer is not None:
            command.extend(
                [
                    "--ctc-draft-fallback-max-cer",
                    str(args.ar_ctc_draft_fallback_max_cer),
                    "--ctc-draft-fallback-min-length-ratio",
                    str(args.ar_ctc_draft_fallback_min_length_ratio),
                    "--ctc-draft-fallback-max-length-ratio",
                    str(args.ar_ctc_draft_fallback_max_length_ratio),
                    "--ctc-draft-fallback-metric-normalization",
                    str(args.ar_ctc_draft_fallback_metric_normalization),
                ]
            )
            if args.ar_ctc_draft_fallback_reject_repetition:
                command.append("--ctc-draft-fallback-reject-repetition")
            else:
                command.append("--no-ctc-draft-fallback-reject-repetition")
    else:
        raise ValueError(f"Unsupported prediction module: {module_name}")
    text_normalization = str(args.text_normalization or train_config.get("text_normalization", "runtime"))
    command.extend(["--text-normalization", text_normalization])
    if config_yaml_override is not None:
        command.extend(["--config-yaml", str(config_yaml_override)])
    if module_name == "rwkvasr.cli.predict_ctc_labeled" and args.save_debug_lengths:
        command.append("--save-debug-lengths")

    manifest_path = train_config.get("manifest_path")
    webdataset_root = train_config.get("webdataset_root")
    if manifest_path:
        command.extend(["--manifest-path", str(manifest_path)])
    elif webdataset_root:
        webdataset_split = str(args.webdataset_split)
        command.extend(
            [
                "--webdataset-root",
                str(webdataset_root),
                "--webdataset-split",
                webdataset_split,
                "--webdataset-shard-pattern",
                webdataset_shard_pattern or str(train_config.get("webdataset_shard_pattern", "*.tar")),
                "--webdataset-eval-ratio",
                str(train_config.get("webdataset_eval_ratio", 0.0)),
                "--webdataset-hash-seed",
                str(train_config.get("webdataset_hash_seed", 0)),
                "--webdataset-split-by",
                str(train_config.get("webdataset_split_by", "shard_name")),
                "--webdataset-utt-id-key",
                str(train_config.get("webdataset_utt_id_key", "sid")),
            ]
        )
        webdataset_length_index_path = train_config.get("webdataset_length_index_path")
        if webdataset_length_index_path and (
            webdataset_split == "all"
            or _length_index_contains_split(str(webdataset_length_index_path), webdataset_split)
        ):
            command.extend(["--webdataset-length-index-path", str(webdataset_length_index_path)])
        elif webdataset_length_index_path:
            _log(
                "Ignoring train_config webdataset_length_index_path for sidecar "
                f"{webdataset_split} preview because it has no {webdataset_split} split rows: "
                f"{webdataset_length_index_path}"
            )
    else:
        raise ValueError("train_config must contain manifest_path or webdataset_root")
    return command


@lru_cache(maxsize=32)
def _length_index_contains_split(length_index_path: str, split: str) -> bool:
    path = Path(length_index_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(raw.get("split")) == split:
                    return True
    except FileNotFoundError:
        return False
    return False


def _maybe_write_backend_override(
    *,
    run_dir: Path,
    sidecar_dir: Path,
    checkpoint_stem: str,
    device: str,
) -> Path | None:
    if str(device).startswith("cuda"):
        return None
    model_config_path = run_dir / "model_config.yaml"
    if not model_config_path.exists():
        return None
    model_config = dict(load_yaml(model_config_path))
    if str(model_config.get("backend", "native")) == "native":
        return None
    model_config["backend"] = "native"
    override_path = sidecar_dir / f"{checkpoint_stem}.cpu_model_config.yaml"
    save_yaml(override_path, model_config)
    return override_path


def _parse_preview_debug_stats(preview_path: Path) -> dict[str, Any]:
    if not preview_path.exists():
        return {}
    blank_top1_values: list[float] = []
    avg_blank_values: list[float] = []
    pred_ref_ratios: list[float] = []
    collapsed = 0
    total = 0
    pattern = re.compile(r"([a-zA-Z0-9_]+)=([0-9.]+)")
    for line in preview_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("  DEBUG:"):
            continue
        total += 1
        values = {match.group(1): float(match.group(2)) for match in pattern.finditer(line)}
        blank_top1 = values.get("blank_top1")
        avg_blank = values.get("avg_blank")
        pred_tok = values.get("pred_tok")
        ref_tok = values.get("ref_tok")
        if blank_top1 is not None:
            blank_top1_values.append(blank_top1)
        if avg_blank is not None:
            avg_blank_values.append(avg_blank)
        if pred_tok is not None and ref_tok is not None and ref_tok > 0:
            ratio = pred_tok / ref_tok
            pred_ref_ratios.append(ratio)
            if ratio <= 0.3:
                collapsed += 1
    if total == 0:
        return {}
    return {
        "sample_count": total,
        "blank_top1_mean": sum(blank_top1_values) / len(blank_top1_values)
        if blank_top1_values
        else None,
        "avg_blank_mean": sum(avg_blank_values) / len(avg_blank_values) if avg_blank_values else None,
        "pred_ref_ratio_mean": sum(pred_ref_ratios) / len(pred_ref_ratios)
        if pred_ref_ratios
        else None,
        "collapsed_ratio": collapsed / total,
    }


def _parse_ar_preview_debug_stats(preview_path: Path) -> dict[str, Any]:
    if not preview_path.exists():
        return {}
    pred_ref_ratios: list[float] = []
    avg_logprob_values: list[float] = []
    eos_values: list[float] = []
    collapsed = 0
    total = 0
    pattern = re.compile(r"([a-zA-Z0-9_]+)=(-?[0-9.]+)")
    for line in preview_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("  DEBUG:"):
            continue
        total += 1
        values = {match.group(1): float(match.group(2)) for match in pattern.finditer(line)}
        pred_tok = values.get("pred_tok")
        ref_tok = values.get("ref_tok")
        eos_value = values.get("eos")
        avg_logprob = values.get("avg_logprob")
        if pred_tok is not None and ref_tok is not None and ref_tok > 0:
            ratio = pred_tok / ref_tok
            pred_ref_ratios.append(ratio)
            if ratio <= 0.3:
                collapsed += 1
        if eos_value is not None:
            eos_values.append(eos_value)
        if avg_logprob is not None:
            avg_logprob_values.append(avg_logprob)
    if total == 0:
        return {}
    return {
        "sample_count": total,
        "pred_ref_ratio_mean": sum(pred_ref_ratios) / len(pred_ref_ratios) if pred_ref_ratios else None,
        "avg_logprob_mean": sum(avg_logprob_values) / len(avg_logprob_values) if avg_logprob_values else None,
        "eos_emitted_ratio": sum(eos_values) / len(eos_values) if eos_values else None,
        "collapsed_ratio": collapsed / total,
    }


def _format_optional_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_optional_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.{digits}f}%"


def _load_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _compare_prediction_sets(
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    *,
    baseline_label: str,
    candidate_label: str,
    metric_normalization: str = "ctc",
) -> dict[str, Any]:
    return compare_prediction_text_sets(
        baseline_jsonl,
        candidate_jsonl,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        normalization=metric_normalization,
    )


def _jsonl_text_error_stats(
    jsonl_path: Path | None,
    *,
    normalization: str = "ctc",
) -> dict[str, Any]:
    if jsonl_path is None or not jsonl_path.exists():
        return {}
    return compute_text_error_stats(jsonl_path, normalization=normalization)


def _short_text(value: Any, *, limit: int = 160) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _record_text_error(
    record: dict[str, Any],
    *,
    normalization: str = "runtime",
    language: str | None = None,
) -> tuple[float | None, float | None]:
    pred = record.get("pred_text")
    ref = record.get("ref_text")
    if not isinstance(pred, str) and not isinstance(ref, str):
        return None, None
    pred_norm = normalize_asr_text_for_metrics(pred if isinstance(pred, str) else "", language=language, normalization=normalization)
    ref_norm = normalize_asr_text_for_metrics(ref if isinstance(ref, str) else "", language=language, normalization=normalization)
    pred_wer_tokens = tokenize_for_wer(pred_norm)
    ref_wer_tokens = tokenize_for_wer(ref_norm)
    pred_cer_tokens = tokenize_for_cer(pred_norm)
    ref_cer_tokens = tokenize_for_cer(ref_norm)
    wer = float(edit_distance(pred_wer_tokens, ref_wer_tokens)) / float(max(1, len(ref_wer_tokens)))
    cer = float(edit_distance(pred_cer_tokens, ref_cer_tokens)) / float(max(1, len(ref_cer_tokens)))
    return wer, cer


def _record_length_ratio(
    record: dict[str, Any],
    *,
    normalization: str = "ctc",
    language: str | None = None,
) -> float | None:
    pred = record.get("pred_text")
    ref = record.get("ref_text")
    if not isinstance(pred, str) and not isinstance(ref, str):
        return None
    pred_norm = normalize_asr_text_for_metrics(
        pred if isinstance(pred, str) else "",
        language=language,
        normalization=normalization,
    )
    ref_norm = normalize_asr_text_for_metrics(
        ref if isinstance(ref, str) else "",
        language=language,
        normalization=normalization,
    )
    pred_tokens = tokenize_for_wer(pred_norm)
    ref_tokens = tokenize_for_wer(ref_norm)
    return float(len(pred_tokens)) / float(max(1, len(ref_tokens)))


def _jsonl_content_length_stats(
    jsonl_path: Path | None,
    *,
    normalization: str = "ctc",
) -> dict[str, Any]:
    if jsonl_path is None or not jsonl_path.exists():
        return {}
    ratios: list[float] = []
    collapsed = 0
    records = _load_prediction_jsonl(jsonl_path)
    for record in records:
        ratio = _record_length_ratio(record, normalization=normalization)
        if ratio is None:
            continue
        ratios.append(ratio)
        if ratio <= 0.3:
            collapsed += 1
    if not records or not ratios:
        return {}
    return {
        "sample_count": len(records),
        "pred_ref_ratio_mean": sum(ratios) / len(ratios),
        "collapsed_ratio": collapsed / len(ratios),
    }


def _build_sample_comment(
    *,
    ctc_wer: float | None,
    ar_wer: float | None,
    ctc_cer: float | None,
    ar_cer: float | None,
    ctc_ratio: float | None,
    ar_ratio: float | None,
    ar_record: dict[str, Any],
) -> str:
    parts: list[str] = []
    if ctc_wer is not None and ar_wer is not None:
        if ctc_wer + 1e-9 < ar_wer:
            parts.append("CTC closer on ASR-content normalized WER")
        elif ar_wer + 1e-9 < ctc_wer:
            parts.append("AR closer on ASR-content normalized WER")
        else:
            parts.append("CTC and AR tie on ASR-content normalized WER/CER")
    elif ctc_wer is not None:
        parts.append("only CTC WER/CER available")
    elif ar_wer is not None:
        parts.append("only AR WER/CER available")
    else:
        parts.append("WER/CER unavailable")
    if ctc_cer is not None and ar_cer is not None:
        if ctc_cer + 1e-9 < ar_cer:
            parts.append("CTC also wins on CER")
        elif ar_cer + 1e-9 < ctc_cer:
            parts.append("AR also wins on CER")

    if ctc_ratio is not None and ctc_ratio <= 0.5:
        parts.append("CTC is short")
    elif ctc_ratio is not None and ctc_ratio >= 1.5:
        parts.append("CTC is long")
    if ar_ratio is not None and ar_ratio <= 0.5:
        parts.append("AR is short")
    elif ar_ratio is not None and ar_ratio >= 1.5:
        parts.append("AR is long")

    debug = ar_record.get("debug")
    if isinstance(debug, dict) and debug.get("eos_emitted") is False:
        parts.append("AR did not emit EOS")
    return "; ".join(parts) + "."


def _build_prediction_examples(
    ctc_jsonl: Path | None,
    ar_jsonl: Path | None,
    *,
    metric_normalization: str = "ctc",
    limit: int = 5,
) -> list[dict[str, Any]]:
    if ctc_jsonl is None or ar_jsonl is None or not ctc_jsonl.exists() or not ar_jsonl.exists():
        return []
    ctc_records = _load_prediction_jsonl(ctc_jsonl)
    ar_records = {str(record.get("utt_id", "")): record for record in _load_prediction_jsonl(ar_jsonl)}
    examples: list[dict[str, Any]] = []
    for ctc_record in ctc_records:
        utt_id = str(ctc_record.get("utt_id", ""))
        ar_record = ar_records.get(utt_id)
        if ar_record is None:
            continue
        ctc_error = _record_text_error(
            ctc_record,
            normalization=metric_normalization,
        )
        ar_error = _record_text_error(
            ar_record,
            normalization=metric_normalization,
        )
        ctc_ratio = _record_length_ratio(
            ctc_record,
            normalization=metric_normalization,
        )
        ar_ratio = _record_length_ratio(
            ar_record,
            normalization=metric_normalization,
        )
        examples.append(
            {
                "utt_id": utt_id,
                "ref_text": ctc_record.get("ref_text") or ar_record.get("ref_text") or "",
                "ctc_pred_text": ctc_record.get("pred_text") or "",
                "ar_pred_text": ar_record.get("pred_text") or "",
                "ctc_wer": ctc_error[0],
                "ar_wer": ar_error[0],
                "ctc_cer": ctc_error[1],
                "ar_cer": ar_error[1],
                "ctc_pred_ref_ratio": ctc_ratio,
                "ar_pred_ref_ratio": ar_ratio,
                "comment": _build_sample_comment(
                    ctc_wer=ctc_error[0],
                    ar_wer=ar_error[0],
                    ctc_cer=ctc_error[1],
                    ar_cer=ar_error[1],
                    ctc_ratio=ctc_ratio,
                    ar_ratio=ar_ratio,
                    ar_record=ar_record,
                ),
            }
        )
        if len(examples) >= max(0, int(limit)):
            break
    return examples


def _build_comment(
    *,
    checkpoint_name: str,
    step: int | None,
    metrics_by_step: dict[int, dict[str, Any]],
    ctc_preview_stats: dict[str, Any],
    ar_preview_stats: dict[str, Any],
    ctc_error_stats: dict[str, Any] | None = None,
    ar_error_stats: dict[str, Any] | None = None,
    compare_stats: dict[str, Any] | None = None,
    examples: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    eval_loss = None
    prev_eval_loss = None
    best_eval_loss = None
    best_step = None
    eval_status = "preview-only"
    ctc_preview_status = "preview unavailable"
    ar_preview_status = "preview unavailable"
    overall_status = "unknown"

    if step is not None and step in metrics_by_step:
        current_metric = metrics_by_step[step]
        raw_eval_loss = current_metric.get("eval_loss")
        if isinstance(raw_eval_loss, (int, float)):
            eval_loss = float(raw_eval_loss)
        prior_steps = sorted(s for s in metrics_by_step if s < step)
        if prior_steps:
            prev_metric = metrics_by_step[prior_steps[-1]]
            raw_prev = prev_metric.get("eval_loss")
            if isinstance(raw_prev, (int, float)):
                prev_eval_loss = float(raw_prev)
        steps_so_far = sorted(s for s in metrics_by_step if s <= step)
        if steps_so_far:
            best_step = min(
                steps_so_far,
                key=lambda s: float(metrics_by_step[s].get("eval_loss", float("inf"))),
            )
            raw_best = metrics_by_step[best_step].get("eval_loss")
            if isinstance(raw_best, (int, float)):
                best_eval_loss = float(raw_best)

    blank_top1 = ctc_preview_stats.get("blank_top1_mean")
    ctc_pred_ref_ratio = ctc_preview_stats.get("pred_ref_ratio_mean")
    ctc_collapsed_ratio = float(ctc_preview_stats.get("collapsed_ratio", 0.0))
    if blank_top1 is None or ctc_pred_ref_ratio is None:
        ctc_preview_status = "preview unavailable"
    elif blank_top1 >= 0.95 or ctc_pred_ref_ratio <= 0.2:
        ctc_preview_status = "still blank-dominated; output collapse is obvious"
    elif blank_top1 >= 0.85 or ctc_pred_ref_ratio <= 0.45 or ctc_collapsed_ratio >= 0.5:
        ctc_preview_status = "transitioning out of blank collapse; outputs are still short"
    elif 0.65 <= ctc_pred_ref_ratio <= 1.25 and blank_top1 <= 0.82:
        ctc_preview_status = "output length looks broadly reasonable on the preview slice"
    else:
        ctc_preview_status = "sentence-level outputs exist, but deletion / insertion bias is still visible"

    ar_pred_ref_ratio = ar_preview_stats.get("pred_ref_ratio_mean")
    ar_collapsed_ratio = float(ar_preview_stats.get("collapsed_ratio", 0.0))
    ar_eos_ratio = ar_preview_stats.get("eos_emitted_ratio")
    ctc_error_stats = ctc_error_stats or {}
    ar_error_stats = ar_error_stats or {}
    ctc_avg_wer = ctc_error_stats.get("avg_wer")
    ctc_avg_cer = ctc_error_stats.get("avg_cer")
    ar_avg_wer = ar_error_stats.get("avg_wer")
    ar_avg_cer = ar_error_stats.get("avg_cer")
    ctc_sample_count = int(ctc_error_stats.get("sample_count", 0) or 0)
    ar_sample_count = int(ar_error_stats.get("sample_count", 0) or 0)
    if ar_pred_ref_ratio is None:
        ar_preview_status = "preview unavailable"
    elif ar_collapsed_ratio >= 0.5 or ar_pred_ref_ratio <= 0.35:
        ar_preview_status = "decoder AR is still collapsing into outputs that are too short"
    elif ar_eos_ratio is not None and ar_eos_ratio < 0.5:
        ar_preview_status = "decoder AR often fails to emit EOS inside the preview budget"
    elif 0.65 <= ar_pred_ref_ratio <= 1.35:
        ar_preview_status = "decoder AR output length is broadly reasonable on the preview slice"
    else:
        ar_preview_status = "decoder AR is producing sentence-level outputs, but length bias is still visible"

    if eval_loss is not None:
        if best_eval_loss is not None and abs(eval_loss - best_eval_loss) <= 1e-8:
            eval_status = f"sampled eval is the best seen so far at step {best_step}"
        elif prev_eval_loss is not None:
            delta = eval_loss - prev_eval_loss
            if delta <= -0.05:
                eval_status = f"sampled eval improved by {abs(delta):.4f} vs previous checkpoint"
            elif delta >= 0.15:
                eval_status = f"sampled eval regressed by {delta:.4f} vs previous checkpoint"
            else:
                eval_status = f"sampled eval moved by {delta:+.4f} vs previous checkpoint"
        else:
            eval_status = "first sampled eval point available"

    if eval_loss is None and ctc_avg_wer is not None and ctc_avg_cer is not None:
        overall_status = "preview-metrics"
    elif eval_loss is None:
        overall_status = "comment-only"
    elif best_eval_loss is not None and abs(eval_loss - best_eval_loss) <= 1e-8 and (
        blank_top1 is None or blank_top1 <= 0.9
    ):
        overall_status = "reasonable-best"
    elif prev_eval_loss is not None and eval_loss <= prev_eval_loss + 0.10 and (
        blank_top1 is None or blank_top1 <= 0.9
    ):
        overall_status = "reasonable"
    elif prev_eval_loss is not None and eval_loss > prev_eval_loss + 0.15:
        overall_status = "needs-attention"
    else:
        overall_status = "reasonable-with-noise"

    lines = [
        f"# {checkpoint_name}",
        "",
        f"- Overall: `{overall_status}`",
        f"- Sampled eval: {eval_status}",
        f"- CTC preview: {ctc_preview_status}",
        f"- RWKV decoder preview: {ar_preview_status}",
        f"- Eval loss: {_format_optional_float(eval_loss)}",
        f"- Previous eval loss: {_format_optional_float(prev_eval_loss)}",
        f"- Best eval loss so far: {_format_optional_float(best_eval_loss)}"
        + ("" if best_step is None else f" at step {best_step}"),
        f"- CTC normalized WER/CER: {_format_optional_percent(ctc_avg_wer)} / "
        f"{_format_optional_percent(ctc_avg_cer)}"
        + ("" if ctc_sample_count <= 0 else f" over {ctc_sample_count} samples"),
        f"- AR normalized WER/CER: {_format_optional_percent(ar_avg_wer)} / "
        f"{_format_optional_percent(ar_avg_cer)}"
        + ("" if ar_sample_count <= 0 else f" over {ar_sample_count} samples"),
        f"- Mean blank_top1: {_format_optional_float(blank_top1)}",
        f"- Mean avg_blank: {_format_optional_float(ctc_preview_stats.get('avg_blank_mean'))}",
        f"- CTC mean ASR-content pred/ref ratio: {_format_optional_float(ctc_pred_ref_ratio)}",
        f"- CTC collapsed preview ratio: {_format_optional_float(ctc_collapsed_ratio)}",
        f"- AR mean ASR-content pred/ref ratio: {_format_optional_float(ar_pred_ref_ratio)}",
        f"- AR EOS-emitted ratio: {_format_optional_float(ar_eos_ratio)}",
        f"- AR mean avg_logprob: {_format_optional_float(ar_preview_stats.get('avg_logprob_mean'))}",
        f"- AR collapsed preview ratio: {_format_optional_float(ar_collapsed_ratio)}",
    ]
    if compare_stats:
        lines.extend(
            [
                "",
                "## CTC vs RWKV Decoder",
                f"- Verdict: {compare_stats.get('verdict', 'n/a')}",
                "- Metric note: WER/CER strip punctuation, symbols, case, project-generated AR language-confirmation prefixes, and normalize Arabic numerals before scoring.",
                f"- {compare_stats.get('baseline_label', 'baseline')} avg WER/CER: "
                f"{_format_optional_float(compare_stats.get('baseline_avg_wer'))} / "
                f"{_format_optional_float(compare_stats.get('baseline_avg_cer'))}",
                f"- {compare_stats.get('candidate_label', 'candidate')} avg WER/CER: "
                f"{_format_optional_float(compare_stats.get('candidate_avg_wer'))} / "
                f"{_format_optional_float(compare_stats.get('candidate_avg_cer'))}",
                f"- Changed predictions: {int(compare_stats.get('changed_prediction_count', 0))}/{int(compare_stats.get('shared_sample_count', 0))}",
                f"- Improved / worsened / unchanged: {int(compare_stats.get('improved_count', 0))}/{int(compare_stats.get('worsened_count', 0))}/{int(compare_stats.get('unchanged_count', 0))}",
            ]
        )
    if examples:
        lines.extend(["", "## Text Examples"])
        for index, example in enumerate(examples, start=1):
            ctc_wer = example.get("ctc_wer")
            ar_wer = example.get("ar_wer")
            ctc_cer = example.get("ctc_cer")
            ar_cer = example.get("ar_cer")
            ctc_ratio = example.get("ctc_pred_ref_ratio")
            ar_ratio = example.get("ar_pred_ref_ratio")
            lines.extend(
                [
                    "",
                    f"### Example {index}: `{example.get('utt_id', '')}`",
                    f"- REF: {_short_text(example.get('ref_text'))}",
                    f"- CTC: {_short_text(example.get('ctc_pred_text'))}",
                    f"- AR: {_short_text(example.get('ar_pred_text'))}",
                    f"- WER CTC / AR: {_format_optional_float(ctc_wer)} / {_format_optional_float(ar_wer)}",
                    f"- CER CTC / AR: {_format_optional_float(ctc_cer)} / {_format_optional_float(ar_cer)}",
                    f"- ASR-content pred/ref length CTC / AR: {_format_optional_float(ctc_ratio)} / {_format_optional_float(ar_ratio)}",
                    f"- Comment: {example.get('comment', '')}",
                ]
            )
    summary = {
        "status": overall_status,
        "eval_status": eval_status,
        "ctc_preview_status": ctc_preview_status,
        "ar_preview_status": ar_preview_status,
        "eval_loss": eval_loss,
        "prev_eval_loss": prev_eval_loss,
        "best_eval_loss_so_far": best_eval_loss,
        "best_step_so_far": best_step,
        "ctc_preview_stats": ctc_preview_stats,
        "ar_preview_stats": ar_preview_stats,
        "ctc_error_stats": ctc_error_stats,
        "ar_error_stats": ar_error_stats,
        "branch_compare": compare_stats or {},
        "examples": examples or [],
    }
    return "\n".join(lines) + "\n", summary


def _write_comment_for_checkpoint(
    *,
    checkpoint_name: str,
    ctc_preview_path: Path,
    ar_preview_path: Path | None,
    sidecar_dir: Path,
    metrics_by_step: dict[int, dict[str, Any]],
    text_normalization: str = "runtime",
    compare_stats: dict[str, Any] | None = None,
    ctc_output_jsonl: Path | None = None,
    ar_output_jsonl: Path | None = None,
    metric_normalization: str = "ctc",
    example_count: int = 5,
) -> dict[str, Any]:
    step = _parse_step_number(checkpoint_name)
    comment_path = sidecar_dir / f"{Path(checkpoint_name).stem}.comment.md"
    ctc_preview_stats = _parse_preview_debug_stats(ctc_preview_path)
    ar_preview_stats = (
        _parse_ar_preview_debug_stats(ar_preview_path)
        if ar_preview_path is not None and ar_preview_path.exists()
        else {}
    )
    ctc_content_length_stats = _jsonl_content_length_stats(
        ctc_output_jsonl,
        normalization=metric_normalization,
    )
    if ctc_content_length_stats:
        ctc_preview_stats.update(ctc_content_length_stats)
    ar_content_length_stats = _jsonl_content_length_stats(
        ar_output_jsonl,
        normalization=metric_normalization,
    )
    if ar_content_length_stats:
        ar_preview_stats.update(ar_content_length_stats)
    ctc_error_stats = _jsonl_text_error_stats(
        ctc_output_jsonl,
        normalization=metric_normalization,
    )
    ar_error_stats = _jsonl_text_error_stats(
        ar_output_jsonl,
        normalization=metric_normalization,
    )
    examples = _build_prediction_examples(
        ctc_output_jsonl,
        ar_output_jsonl,
        metric_normalization=metric_normalization,
        limit=example_count,
    )
    comment_text, summary = _build_comment(
        checkpoint_name=checkpoint_name,
        step=step,
        metrics_by_step=metrics_by_step,
        ctc_preview_stats=ctc_preview_stats,
        ar_preview_stats=ar_preview_stats,
        ctc_error_stats=ctc_error_stats,
        ar_error_stats=ar_error_stats,
        compare_stats=compare_stats,
        examples=examples,
    )
    comment_path.write_text(comment_text, encoding="utf-8")
    summary["comment_path"] = str(comment_path)
    return summary


def _backfill_comments(
    *,
    state: dict[str, Any],
    sidecar_dir: Path,
    metrics_by_step: dict[int, dict[str, Any]],
    text_normalization: str = "runtime",
    metric_normalization: str = "ctc",
) -> bool:
    changed = False
    evaluated = state.get("evaluated", {})
    if not isinstance(evaluated, dict):
        return False
    for checkpoint_name, record in evaluated.items():
        if not isinstance(record, dict):
            continue
        ctc_preview_path_raw = record.get("ctc_preview_path") or record.get("preview_path")
        ar_preview_path_raw = record.get("ar_preview_path")
        if not isinstance(ctc_preview_path_raw, str):
            continue
        ctc_preview_path = Path(ctc_preview_path_raw)
        ar_preview_path = Path(ar_preview_path_raw) if isinstance(ar_preview_path_raw, str) else None
        if not ctc_preview_path.exists() or (
            ar_preview_path is not None and not ar_preview_path.exists()
        ):
            continue
        compare_stats = {}
        ctc_output_jsonl_raw = record.get("ctc_output_jsonl") or record.get("output_jsonl")
        ar_output_jsonl_raw = record.get("ar_output_jsonl")
        if isinstance(ctc_output_jsonl_raw, str) and isinstance(ar_output_jsonl_raw, str):
            ctc_output_jsonl_path = Path(ctc_output_jsonl_raw)
            ar_output_jsonl_path = Path(ar_output_jsonl_raw)
            if ctc_output_jsonl_path.exists() and ar_output_jsonl_path.exists():
                compare_stats = _compare_prediction_sets(
                    ctc_output_jsonl_path,
                    ar_output_jsonl_path,
                    baseline_label="ctc",
                    candidate_label="rwkv_decoder_ar",
                    metric_normalization=metric_normalization,
                )
        summary = _write_comment_for_checkpoint(
            checkpoint_name=checkpoint_name,
            ctc_preview_path=ctc_preview_path,
            ar_preview_path=ar_preview_path,
            sidecar_dir=sidecar_dir,
            metrics_by_step=metrics_by_step,
            text_normalization=text_normalization,
            metric_normalization=metric_normalization,
            compare_stats=compare_stats,
            ctc_output_jsonl=Path(ctc_output_jsonl_raw)
            if isinstance(ctc_output_jsonl_raw, str)
            else None,
            ar_output_jsonl=Path(ar_output_jsonl_raw) if isinstance(ar_output_jsonl_raw, str) else None,
        )
        record.update(summary)
        changed = True
    return changed


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_debug_preview_line(record: dict[str, Any], *, module_name: str) -> str | None:
    debug = record.get("debug")
    if not isinstance(debug, dict):
        return None
    if module_name == "rwkvasr.cli.predict_ctc_labeled":
        return (
            "  DEBUG: "
            f"feat={int(debug.get('feature_length', 0))} "
            f"logit={int(debug.get('logit_length', 0))} "
            f"pred_tok={int(debug.get('pred_token_count', 0))} "
            f"ref_tok={int(debug.get('ref_token_count', 0))} "
            f"blank_top1={float(debug.get('blank_top1_ratio', 0.0)):.3f} "
            f"avg_blank={float(debug.get('avg_blank_prob', 0.0)):.3f}"
        )
    if module_name == "rwkvasr.cli.predict_rwkv_decoder_labeled":
        return (
            "  DEBUG: "
            f"feat={int(debug.get('feature_length', 0))} "
            f"enc={int(debug.get('encoded_length', 0))} "
            f"pred_tok={int(debug.get('pred_token_count', 0))} "
            f"ref_tok={int(debug.get('ref_token_count', 0))} "
            f"eos={1 if bool(debug.get('eos_emitted', False)) else 0} "
            f"avg_logprob={float(debug.get('avg_logprob', 0.0)):.4f}"
        )
    return None


def _write_combined_preview(
    path: Path,
    records: list[dict[str, Any]],
    *,
    preview_count: int,
    module_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, record in enumerate(records[: max(0, int(preview_count))], start=1):
        source_label = record.get("sidecar_source")
        source_suffix = f" source={source_label}" if source_label else ""
        lines.append(f"[{index}] utt_id={record.get('utt_id', '')}{source_suffix}")
        lines.append(f"  REF : {record.get('ref_text') or ''}")
        lines.append(f"  PRED: {record.get('pred_text') or ''}")
        score = record.get("score")
        try:
            lines.append(f"  SCORE: {float(score):.4f}")
        except (TypeError, ValueError):
            lines.append("  SCORE: n/a")
        strategy = record.get("decode_strategy")
        if strategy and strategy != "ctc":
            lines.append(f"  STRATEGY: {strategy}")
        debug_line = _format_debug_preview_line(record, module_name=module_name)
        if debug_line is not None:
            lines.append(debug_line)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _run_prediction_command(
    *,
    command: list[str],
    cwd: Path,
    checkpoint_path: Path,
    meta_path: Path,
    meta: dict[str, Any],
) -> None:
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    duration_sec = time.time() - started_at
    meta.update(
        {
            "returncode": int(completed.returncode),
            "duration_sec": float(duration_sec),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Prediction preview failed for {checkpoint_path.name}; see {meta_path}"
        )


def _run_prediction_preview(
    *,
    module_name: str,
    checkpoint_path: Path,
    run_dir: Path,
    sidecar_dir: Path,
    train_config: dict[str, Any],
    args: argparse.Namespace,
    stem_suffix: str = "",
    decoder_ctc_draft_cache_path: Path | None = None,
) -> dict[str, Any]:
    stem = checkpoint_path.stem
    stem_with_suffix = f"{stem}{stem_suffix}"
    output_jsonl = sidecar_dir / f"{stem_with_suffix}.jsonl"
    preview_path = sidecar_dir / f"{stem_with_suffix}.preview.txt"
    config_yaml_override = _maybe_write_backend_override(
        run_dir=run_dir,
        sidecar_dir=sidecar_dir,
        checkpoint_stem=stem,
        device=args.device,
    )
    current_cwd = Path.cwd()
    if (current_cwd / "src").exists():
        cwd = current_cwd
    else:
        cwd = run_dir.parent.parent if (run_dir.parent.parent / "src").exists() else run_dir.parent
    source_quotas = _parse_source_quotas(args.source_quota)
    if source_quotas and not train_config.get("webdataset_root"):
        raise ValueError("--source-quota is only supported for WebDataset-backed sidecar evaluation.")

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "output_jsonl": str(output_jsonl),
        "preview_path": str(preview_path),
        "module_name": module_name,
        "config_yaml_override": None if config_yaml_override is None else str(config_yaml_override),
        "decoder_ctc_draft_cache_path": (
            None if decoder_ctc_draft_cache_path is None else str(decoder_ctc_draft_cache_path)
        ),
    }
    meta_path = sidecar_dir / f"{stem_with_suffix}.meta.json"

    if source_quotas:
        _log(
            f"Evaluating {checkpoint_path.name} -> {preview_path.name} "
            f"with source quotas "
            + ", ".join(f"{quota.label}:{quota.shard_pattern}:{quota.limit}" for quota in source_quotas)
        )
        all_records: list[dict[str, Any]] = []
        source_meta: list[dict[str, Any]] = []
        started_at = time.time()
        for quota in source_quotas:
            source_stem = f"{stem_with_suffix}.{quota.label}"
            source_jsonl = sidecar_dir / f"{source_stem}.jsonl"
            source_preview = sidecar_dir / f"{source_stem}.preview.txt"
            command = _build_predict_command(
                module_name=module_name,
                checkpoint_path=checkpoint_path,
                output_jsonl=source_jsonl,
                preview_path=source_preview,
                config_yaml_override=config_yaml_override,
                train_config=train_config,
                args=args,
                limit=quota.limit,
                preview_count=quota.limit,
                webdataset_shard_pattern=quota.shard_pattern,
                decoder_ctc_draft_cache_path=decoder_ctc_draft_cache_path,
            )
            source_meta_path = sidecar_dir / f"{source_stem}.meta.json"
            source_record = {
                "label": quota.label,
                "shard_pattern": quota.shard_pattern,
                "limit": quota.limit,
                "output_jsonl": str(source_jsonl),
                "preview_path": str(source_preview),
            }
            _run_prediction_command(
                command=command,
                cwd=cwd,
                checkpoint_path=checkpoint_path,
                meta_path=source_meta_path,
                meta={
                    **meta,
                    **source_record,
                    "output_jsonl": str(source_jsonl),
                    "preview_path": str(source_preview),
                },
            )
            records = _load_prediction_jsonl(source_jsonl)
            for record in records:
                record["sidecar_source"] = quota.label
            all_records.extend(records)
            source_record["num_predictions"] = len(records)
            source_meta.append(source_record)

        _write_jsonl_records(output_jsonl, all_records)
        _write_combined_preview(
            preview_path,
            all_records,
            preview_count=args.preview_count,
            module_name=module_name,
        )
        meta.update(
            {
                "returncode": 0,
                "duration_sec": float(time.time() - started_at),
                "stdout": "",
                "stderr": "",
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "source_quotas": [quota.__dict__ for quota in source_quotas],
                "source_outputs": source_meta,
                "num_predictions": len(all_records),
            }
        )
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return meta

    command = _build_predict_command(
        module_name=module_name,
        checkpoint_path=checkpoint_path,
        output_jsonl=output_jsonl,
        preview_path=preview_path,
        config_yaml_override=config_yaml_override,
        train_config=train_config,
        args=args,
        decoder_ctc_draft_cache_path=decoder_ctc_draft_cache_path,
    )
    _log(f"Evaluating {checkpoint_path.name} -> {preview_path.name}")
    _run_prediction_command(
        command=command,
        cwd=cwd,
        checkpoint_path=checkpoint_path,
        meta_path=meta_path,
        meta=meta,
    )
    return meta


def _evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    run_dir: Path,
    sidecar_dir: Path,
    train_config: dict[str, Any],
    args: argparse.Namespace,
    text_normalization: str,
    metric_normalization: str,
) -> dict[str, Any]:
    ctc_meta = _run_prediction_preview(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        sidecar_dir=sidecar_dir,
        train_config=train_config,
        args=args,
        module_name="rwkvasr.cli.predict_ctc_labeled",
        stem_suffix=".ctc",
    )
    if not bool(train_config.get("decoder_enabled", False)):
        return {
            "ctc": ctc_meta,
            "ar": None,
            "compare_stats": {},
        }
    ar_meta = _run_prediction_preview(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        sidecar_dir=sidecar_dir,
        train_config=train_config,
        args=args,
        module_name="rwkvasr.cli.predict_rwkv_decoder_labeled",
        stem_suffix=".ar",
        decoder_ctc_draft_cache_path=Path(ctc_meta["output_jsonl"]),
    )
    compare_stats = _compare_prediction_sets(
        Path(ctc_meta["output_jsonl"]),
        Path(ar_meta["output_jsonl"]),
        baseline_label="ctc",
        candidate_label="rwkv_decoder_ar",
        metric_normalization=metric_normalization,
    )
    return {
        "ctc": ctc_meta,
        "ar": ar_meta,
        "compare_stats": compare_stats,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    train_config_path = run_dir / "train_config.yaml"
    while not train_config_path.exists():
        if args.once:
            raise FileNotFoundError(f"train_config.yaml not found under {run_dir}")
        _log(f"Waiting for train_config.yaml under {run_dir}")
        time.sleep(min(30.0, max(1.0, float(args.poll_seconds))))
    train_config = _load_train_config(run_dir)
    text_normalization = str(args.text_normalization or train_config.get("text_normalization", "runtime"))
    metric_normalization = str(args.metric_normalization or "ctc")
    sidecar_dir, state_path, tmux_tail_path = _sidecar_paths(run_dir, args.output_subdir)
    state = _load_state(state_path)
    state["text_normalization"] = text_normalization
    state["metric_normalization"] = metric_normalization
    metrics_by_step = _load_step_metrics(run_dir)
    if _backfill_comments(
        state=state,
        sidecar_dir=sidecar_dir,
        metrics_by_step=metrics_by_step,
        text_normalization=text_normalization,
        metric_normalization=metric_normalization,
    ):
        save_yaml(state_path, state)

    while True:
        _capture_tmux_tail(args.tmux_target, tmux_tail_path)
        metrics_by_step = _load_step_metrics(run_dir)
        if _backfill_comments(
            state=state,
            sidecar_dir=sidecar_dir,
            metrics_by_step=metrics_by_step,
            text_normalization=text_normalization,
            metric_normalization=metric_normalization,
        ):
            save_yaml(state_path, state)
        checkpoints = _list_step_checkpoints(run_dir)
        new_work = False
        stable_unevaluated = [
            checkpoint_path
            for checkpoint_path in checkpoints
            if checkpoint_path.name not in state["evaluated"]
            and _is_stable(checkpoint_path, stable_seconds=args.checkpoint_stable_seconds)
        ]
        if stable_unevaluated:
            checkpoint_path = stable_unevaluated[-1]
            key = checkpoint_path.name
            meta = _evaluate_checkpoint(
                checkpoint_path,
                run_dir=run_dir,
                sidecar_dir=sidecar_dir,
                train_config=train_config,
                args=args,
                text_normalization=text_normalization,
                metric_normalization=metric_normalization,
            )
            ctc_meta = meta["ctc"]
            ar_meta = meta["ar"]
            compare_stats = meta.get("compare_stats", {})
            state_record = {
                "ctc_preview_path": ctc_meta["preview_path"],
                "ctc_output_jsonl": ctc_meta["output_jsonl"],
                "evaluated_at": ctc_meta["evaluated_at"],
                "ctc_duration_sec": ctc_meta["duration_sec"],
                "metric_normalization": metric_normalization,
                "text_normalization": text_normalization,
            }
            if isinstance(ar_meta, dict):
                state_record.update(
                    {
                        "ar_preview_path": ar_meta["preview_path"],
                        "ar_output_jsonl": ar_meta["output_jsonl"],
                        "ar_duration_sec": ar_meta["duration_sec"],
                    }
                )
            state["evaluated"][key] = state_record
            summary = _write_comment_for_checkpoint(
                checkpoint_name=key,
                ctc_preview_path=Path(ctc_meta["preview_path"]),
                ar_preview_path=Path(ar_meta["preview_path"]) if isinstance(ar_meta, dict) else None,
                sidecar_dir=sidecar_dir,
                metrics_by_step=metrics_by_step,
                text_normalization=text_normalization,
                metric_normalization=metric_normalization,
                compare_stats=compare_stats,
                ctc_output_jsonl=Path(ctc_meta["output_jsonl"]),
                ar_output_jsonl=Path(ar_meta["output_jsonl"]) if isinstance(ar_meta, dict) else None,
            )
            state["evaluated"][key].update(summary)
            save_yaml(state_path, state)
            _log(
                f"{checkpoint_path.name} comment={summary['status']} eval="
                f"{_format_optional_float(summary.get('eval_loss'))} "
                f"compare={compare_stats.get('verdict', 'n/a')}"
            )
            new_work = True
            if args.once:
                return
        if args.once:
            _log("No stable unevaluated checkpoint found.")
            return
        if not new_work:
            _log("No new stable checkpoints. Sleeping.")
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
