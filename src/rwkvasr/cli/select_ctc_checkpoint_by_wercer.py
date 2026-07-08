from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.eval import (
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CTC step checkpoints on a fixed labeled slice and select by normalized WER/CER."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--webdataset-root", required=True)
    parser.add_argument("--eval-length-index-path", required=True)
    parser.add_argument("--output-subdir", default="wercer_selector")
    parser.add_argument("--checkpoint-glob", default="step-*.pt")
    parser.add_argument("--poll-seconds", default=300.0, type=float)
    parser.add_argument("--checkpoint-stable-seconds", default=30.0, type=float)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--mode", default="bi", choices=["bi", "l2r", "r2l", "alt"])
    parser.add_argument("--beam-size", default=4, type=int)
    parser.add_argument("--token-prune-topk", default=16, type=int)
    parser.add_argument("--text-normalization", default="runtime")
    parser.add_argument("--metric-normalization", default="ctc")
    parser.add_argument("--preview-count", default=32, type=int)
    parser.add_argument("--progress-interval", default=64, type=int)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--selection-metric", default="wer", choices=["wer", "cer"])
    parser.add_argument("--webdataset-utt-id-key", default="id")
    parser.add_argument("--copy-best-name", default="wercer_best.pt")
    return parser


def _log(message: str) -> None:
    print(f"[rwkvasr-wercer-selector] {message}", flush=True)


def _parse_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step-"):
        return -1
    try:
        return int(stem.removeprefix("step-"))
    except ValueError:
        return -1


def _is_stable(path: Path, stable_seconds: float) -> bool:
    return (time.time() - path.stat().st_mtime) >= max(0.0, stable_seconds)


def _maybe_write_backend_override(
    *,
    run_dir: Path,
    output_dir: Path,
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
    override_path = output_dir / f"{checkpoint_stem}.cpu_model_config.yaml"
    save_yaml(override_path, model_config)
    return override_path


def _load_eval_meta(path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            utt_id = str(record.get("utt_id") or record.get("key") or "")
            if utt_id:
                meta[utt_id] = record
    return meta


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num) / float(max(1, den))


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wer_errors = sum(int(row["wer_edits"]) for row in rows)
    wer_ref = sum(int(row["wer_ref"]) for row in rows)
    cer_errors = sum(int(row["cer_edits"]) for row in rows)
    cer_ref = sum(int(row["cer_ref"]) for row in rows)
    return {
        "samples": len(rows),
        "wer": _safe_div(wer_errors, wer_ref),
        "cer": _safe_div(cer_errors, cer_ref),
        "exact": sum(1 for row in rows if bool(row["exact"])),
        "wer_errors": wer_errors,
        "wer_ref": wer_ref,
        "cer_errors": cer_errors,
        "cer_ref": cer_ref,
    }


def _compute_metrics(
    prediction_jsonl: Path,
    *,
    eval_meta: dict[str, dict[str, Any]],
    normalization: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with prediction_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            utt_id = str(record.get("utt_id") or "")
            meta = eval_meta.get(utt_id, {})
            language = meta.get("language")
            pred = normalize_asr_text_for_metrics(
                record.get("pred_text") or "",
                language=language,
                normalization=normalization,
            )
            ref = normalize_asr_text_for_metrics(
                record.get("ref_text") or "",
                language=language,
                normalization=normalization,
            )
            pred_wer = tokenize_for_wer(pred)
            ref_wer = tokenize_for_wer(ref)
            pred_cer = tokenize_for_cer(pred)
            ref_cer = tokenize_for_cer(ref)
            wer_edits = edit_distance(pred_wer, ref_wer)
            cer_edits = edit_distance(pred_cer, ref_cer)
            rows.append(
                {
                    "utt_id": utt_id,
                    "language": language or "unknown",
                    "source_dataset": meta.get("source_dataset")
                    or meta.get("source")
                    or meta.get("dataset")
                    or "unknown",
                    "difficulty_tier": meta.get("difficulty_tier") or "unknown",
                    "wer_edits": wer_edits,
                    "wer_ref": len(ref_wer),
                    "cer_edits": cer_edits,
                    "cer_ref": len(ref_cer),
                    "exact": pred_wer == ref_wer and pred_cer == ref_cer,
                }
            )

    metrics = {
        "overall": _summarize_rows(rows),
        "by_language": {},
        "by_source": {},
        "by_tier": {},
    }
    for language in sorted({str(row["language"]) for row in rows}):
        metrics["by_language"][language] = _summarize_rows(
            [row for row in rows if str(row["language"]) == language]
        )
    for source in sorted({str(row["source_dataset"]) for row in rows}):
        metrics["by_source"][source] = _summarize_rows(
            [row for row in rows if str(row["source_dataset"]) == source]
        )
    for tier in sorted({str(row["difficulty_tier"]) for row in rows}):
        metrics["by_tier"][tier] = _summarize_rows(
            [row for row in rows if str(row["difficulty_tier"]) == tier]
        )
    return metrics


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"checkpoints": [], "failures": []}
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        return {"checkpoints": [], "failures": []}
    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, list):
        raw["checkpoints"] = []
    failures = raw.get("failures")
    if not isinstance(failures, list):
        raw["failures"] = []
    return raw


def _record_sort_key(record: dict[str, Any], metric: str) -> tuple[float, float, int]:
    overall = record.get("metrics", {}).get("overall", {})
    primary = float(overall.get(metric, float("inf")))
    secondary_name = "cer" if metric == "wer" else "wer"
    secondary = float(overall.get(secondary_name, float("inf")))
    step = int(record.get("step", 0))
    return primary, secondary, step


def _checkpoint_exists(record: dict[str, Any]) -> bool:
    checkpoint_path = record.get("checkpoint_path")
    if not checkpoint_path:
        return False
    return Path(str(checkpoint_path)).exists()


def _write_state(
    path: Path,
    *,
    state: dict[str, Any],
    top_k: int,
    selection_metric: str,
) -> dict[str, Any] | None:
    records = [record for record in state.get("checkpoints", []) if isinstance(record, dict)]
    records = sorted(records, key=lambda item: int(item.get("step", 0)))
    selectable_records = [record for record in records if _checkpoint_exists(record)]
    ranked = sorted(selectable_records, key=lambda item: _record_sort_key(item, selection_metric))
    best = ranked[0] if ranked else None
    state["checkpoints"] = records
    state["ranked_top_k"] = ranked[: max(1, int(top_k))]
    state["best"] = best
    state["unavailable_checkpoint_count"] = len(records) - len(selectable_records)
    state["selection_metric"] = selection_metric
    save_yaml(path, state)
    return best


def _run_prediction(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_jsonl: Path,
    preview_path: Path,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "rwkvasr.cli.predict_ctc_labeled",
        "--webdataset-root",
        str(args.webdataset_root),
        "--webdataset-length-index-path",
        str(args.eval_length_index_path),
        "--webdataset-split",
        "all",
        "--webdataset-utt-id-key",
        str(args.webdataset_utt_id_key),
        "--checkpoint-path",
        str(checkpoint_path),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        str(args.device),
        "--mode",
        str(args.mode),
        "--beam-size",
        str(args.beam_size),
        "--token-prune-topk",
        str(args.token_prune_topk),
        "--text-normalization",
        str(args.text_normalization),
        "--output-path",
        str(output_jsonl),
        "--preview-path",
        str(preview_path),
        "--preview-count",
        str(args.preview_count),
        "--progress-interval",
        str(args.progress_interval),
        "--save-debug-lengths",
    ]
    config_yaml_override = _maybe_write_backend_override(
        run_dir=checkpoint_path.parent,
        output_dir=output_jsonl.parent,
        checkpoint_stem=checkpoint_path.stem,
        device=str(args.device),
    )
    if config_yaml_override is not None:
        command.extend(["--config-yaml", str(config_yaml_override)])
    env = dict(os.environ)
    venv_bin = Path.cwd() / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND: " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        log_handle.write(f"\nRETURN_CODE={completed.returncode}\n")
        log_handle.write(f"DURATION_SEC={time.time() - started:.3f}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"prediction failed for {checkpoint_path}; see {log_path}")


def _evaluate_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    eval_meta: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    step = _parse_step(checkpoint_path)
    stem = checkpoint_path.stem
    output_jsonl = output_dir / f"{stem}.ctc.jsonl"
    preview_path = output_dir / f"{stem}.preview.txt"
    log_path = output_dir / f"{stem}.predict.log"
    _log(f"evaluating {checkpoint_path.name}")
    _run_prediction(
        args=args,
        checkpoint_path=checkpoint_path,
        output_jsonl=output_jsonl,
        preview_path=preview_path,
        log_path=log_path,
    )
    metrics = _compute_metrics(
        output_jsonl,
        eval_meta=eval_meta,
        normalization=str(args.metric_normalization),
    )
    overall = metrics["overall"]
    _log(
        f"{checkpoint_path.name}: WER={overall['wer'] * 100.0:.2f}% "
        f"CER={overall['cer'] * 100.0:.2f}% exact={overall['exact']}/{overall['samples']}"
    )
    return {
        "step": step,
        "checkpoint_path": str(checkpoint_path),
        "output_jsonl": str(output_jsonl),
        "preview_path": str(preview_path),
        "predict_log_path": str(log_path),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "text_normalization": str(args.text_normalization),
        "metric_normalization": str(args.metric_normalization),
        "metrics": metrics,
    }


def _record_failure(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    exc: Exception,
    output_dir: Path,
) -> dict[str, Any]:
    step = _parse_step(checkpoint_path)
    log_path = output_dir / f"{checkpoint_path.stem}.predict.log"
    return {
        "step": step,
        "checkpoint_path": str(checkpoint_path),
        "predict_log_path": str(log_path),
        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "error": f"{type(exc).__name__}: {exc}",
        "device": str(args.device),
    }


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    output_dir = run_dir / str(args.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "wercer_checkpoint_metrics.yaml"
    best_meta_path = run_dir / "wercer_best_checkpoint.yaml"
    best_copy_path = run_dir / str(args.copy_best_name)
    eval_meta = _load_eval_meta(Path(args.eval_length_index_path))
    state = _load_state(state_path)

    while True:
        evaluated_paths = {
            str(record.get("checkpoint_path"))
            for record in state.get("checkpoints", [])
            if isinstance(record, dict)
        }
        candidates = sorted(run_dir.glob(str(args.checkpoint_glob)), key=_parse_step)
        new_records = 0
        for checkpoint_path in candidates:
            if _parse_step(checkpoint_path) < 0:
                continue
            if str(checkpoint_path) in evaluated_paths:
                continue
            if not _is_stable(checkpoint_path, float(args.checkpoint_stable_seconds)):
                continue
            try:
                record = _evaluate_checkpoint(
                    args=args,
                    checkpoint_path=checkpoint_path,
                    eval_meta=eval_meta,
                    output_dir=output_dir,
                )
            except Exception as exc:
                failure = _record_failure(
                    args=args,
                    checkpoint_path=checkpoint_path,
                    exc=exc,
                    output_dir=output_dir,
                )
                state.setdefault("failures", []).append(failure)
                save_yaml(state_path, state)
                _log(f"failed {checkpoint_path.name}: {failure['error']}")
                continue
            state.setdefault("checkpoints", []).append(record)
            evaluated_paths.add(str(checkpoint_path))
            new_records += 1
            best = _write_state(
                state_path,
                state=state,
                top_k=int(args.top_k),
                selection_metric=str(args.selection_metric),
            )
            if best is not None and best.get("checkpoint_path") == str(checkpoint_path):
                if checkpoint_path.exists():
                    shutil.copy2(checkpoint_path, best_copy_path)
                    best_payload = dict(best)
                    best_payload["copied_checkpoint_path"] = str(best_copy_path)
                    save_yaml(best_meta_path, best_payload)
                    _log(f"new WER/CER best: {checkpoint_path.name} -> {best_copy_path}")
                else:
                    _log(
                        f"new metric best {checkpoint_path.name} is no longer available; "
                        "kept metrics but skipped best checkpoint copy"
                    )

        if new_records == 0:
            _write_state(
                state_path,
                state=state,
                top_k=int(args.top_k),
                selection_metric=str(args.selection_metric),
            )
            _log("no new stable checkpoints")
        if args.once:
            return
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
