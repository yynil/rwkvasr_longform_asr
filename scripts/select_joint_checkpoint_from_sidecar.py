#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml, save_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a joint CTC+AR checkpoint by sidecar normalized WER/CER metrics."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sidecar-subdir", default="sidecar_eval")
    parser.add_argument("--state-name", default="watch_state.yaml")
    parser.add_argument("--output-name", default="joint_wercer_best.yaml")
    parser.add_argument("--copy-best-name", default="joint_wercer_best.pt")
    parser.add_argument("--metric", default="wer", choices=["wer", "cer"])
    parser.add_argument("--print-path-only", action="store_true")
    return parser.parse_args()


def _step_from_name(name: str) -> int:
    stem = Path(name).stem
    if not stem.startswith("step-"):
        return -1
    try:
        return int(stem.removeprefix("step-"))
    except ValueError:
        return -1


def _finite(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return math.inf
    value = float(value)
    if math.isnan(value):
        return math.inf
    return value


def _branch_score(record: dict[str, Any], metric: str) -> dict[str, Any] | None:
    compare = record.get("branch_compare")
    if not isinstance(compare, dict):
        return None

    baseline_label = str(compare.get("baseline_label") or "ctc")
    candidate_label = str(compare.get("candidate_label") or "rwkv_decoder_ar")
    baseline = {
        "branch": baseline_label,
        "wer": _finite(compare.get("baseline_avg_wer")),
        "cer": _finite(compare.get("baseline_avg_cer")),
    }
    candidate = {
        "branch": candidate_label,
        "wer": _finite(compare.get("candidate_avg_wer")),
        "cer": _finite(compare.get("candidate_avg_cer")),
    }
    if not math.isfinite(baseline["wer"]) and not math.isfinite(candidate["wer"]):
        return None
    secondary = "cer" if metric == "wer" else "wer"
    return min(
        (baseline, candidate),
        key=lambda item: (float(item[metric]), float(item[secondary])),
    )


def _rank_key(item: dict[str, Any], metric: str) -> tuple[float, float, int]:
    branch = item["selected_branch"]
    secondary = "cer" if metric == "wer" else "wer"
    return (
        float(branch[metric]),
        float(branch[secondary]),
        int(item.get("step", 0)),
    )


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    state_path = run_dir / args.sidecar_subdir / args.state_name
    if not state_path.exists():
        raise FileNotFoundError(f"sidecar state not found: {state_path}")

    state = load_yaml(state_path)
    evaluated = state.get("evaluated") if isinstance(state, dict) else None
    if not isinstance(evaluated, dict):
        raise ValueError(f"sidecar state has no evaluated checkpoints: {state_path}")

    candidates: list[dict[str, Any]] = []
    for checkpoint_name, record in evaluated.items():
        if not isinstance(record, dict):
            continue
        branch = _branch_score(record, args.metric)
        if branch is None:
            continue
        checkpoint_path = run_dir / str(checkpoint_name)
        if not checkpoint_path.exists():
            continue
        candidates.append(
            {
                "checkpoint_name": str(checkpoint_name),
                "checkpoint_path": str(checkpoint_path),
                "step": _step_from_name(str(checkpoint_name)),
                "selected_branch": branch,
                "status": record.get("status"),
                "eval_loss": record.get("eval_loss"),
                "comment_path": record.get("comment_path"),
            }
        )

    if not candidates:
        raise RuntimeError(f"no sidecar WER/CER candidates found in {state_path}")

    ranked = sorted(candidates, key=lambda item: _rank_key(item, args.metric))
    best = ranked[0]
    best_checkpoint = Path(best["checkpoint_path"])
    copied_path = run_dir / args.copy_best_name
    if copied_path != best_checkpoint:
        shutil.copy2(best_checkpoint, copied_path)

    report = {
        "version": 1,
        "run_dir": str(run_dir),
        "state_path": str(state_path),
        "selection_metric": str(args.metric),
        "metric_normalization": state.get("metric_normalization") if isinstance(state, dict) else None,
        "text_normalization": state.get("text_normalization") if isinstance(state, dict) else None,
        "best": {**best, "copied_checkpoint_path": str(copied_path)},
        "ranked_top_k": ranked[:10],
    }
    save_yaml(run_dir / args.output_name, report)

    if args.print_path_only:
        print(copied_path)
        return

    branch = best["selected_branch"]
    print(
        "selected "
        f"{best['checkpoint_name']} branch={branch['branch']} "
        f"wer={branch['wer']:.6f} cer={branch['cer']:.6f} "
        f"copied={copied_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
