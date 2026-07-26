from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import sha256_file

try:
    from scripts.create_stage211_hidden_alignment_gate import (
        FIXED_EVAL_SAMPLES,
        _eval_part_fingerprint,
        _validated_eval_provenance,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_hidden_alignment_gate import (
        FIXED_EVAL_SAMPLES,
        _eval_part_fingerprint,
        _validated_eval_provenance,
    )


MIN_KL_RELATIVE_REDUCTION = 0.05
RATIO_MIN = 0.90
RATIO_MAX = 1.10
TOLERANCE = 1.0e-12
REQUIRED_METRICS = (
    "full_kl",
    "conditional_nonblank_kl",
    "conditional_nonblank_hard_ce",
    "blank_binary_kl",
    "selected_top1_agreement",
    "all_top1_agreement",
    "active_top1_agreement",
    "blank_prob_mae",
    "teacher_nonblank_rate",
    "student_nonblank_rate",
    "nonblank_rate_ratio",
    "teacher_nonblank_to_blank_rate",
    "teacher_blank_to_nonblank_rate",
    "ctc_token_error_rate",
    "ctc_token_insertion_rate",
    "ctc_token_deletion_rate",
    "ctc_token_substitution_rate",
    "collapsed_length_ratio",
    "sequence_exact_rate",
    "mean_frame_delta",
    "selected_frames",
    "all_frames",
    "teacher_tokens",
    "student_tokens",
    "matched_utterances",
    "missing_utterances",
)
IDENTICAL_TEACHER_METRICS = (
    "teacher_nonblank_rate",
    "selected_frames",
    "all_frames",
    "teacher_tokens",
    "matched_utterances",
)


def _validated_metrics(
    report: dict[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    if int(report.get("eval_samples", -1)) != FIXED_EVAL_SAMPLES:
        raise ValueError(
            f"Stage211 {label} logits report eval sample count mismatch."
        )
    raw_metrics = report.get("logit_metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"Stage211 {label} report lacks logit_metrics.")
    metrics: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        value = float(raw_metrics.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(
                f"Stage211 {label} logits metric {key} must be finite."
            )
        metrics[key] = value
    if (
        metrics["selected_frames"] <= 0.0
        or metrics["all_frames"] <= 0.0
        or metrics["teacher_tokens"] <= 0.0
        or metrics["matched_utterances"] != FIXED_EVAL_SAMPLES
        or metrics["missing_utterances"] != 0.0
        or metrics["mean_frame_delta"] != 0.0
    ):
        raise ValueError(f"Stage211 {label} logits metric coverage is incomplete.")
    return metrics


def _relative_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1.0e-12)


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"step-([0-9]+)\.pt", path.name)
    if match is None:
        raise ValueError(
            f"Stage211 logits checkpoint must be named step-N.pt: {path}"
        )
    return int(match.group(1))


def build_gate(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    baseline_report_path = baseline_report_path.expanduser().resolve()
    candidate_report_path = candidate_report_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(str(checkpoint_path))
    baseline_report = load_yaml(baseline_report_path)
    candidate_report = load_yaml(candidate_report_path)
    if int(baseline_report.get("step", -1)) != 0:
        raise ValueError("Stage211 logits baseline report must be the step-0 report.")
    candidate_step = int(candidate_report.get("step", -1))
    if candidate_step <= 0 or candidate_step != _checkpoint_step(checkpoint_path):
        raise ValueError(
            "Stage211 logits candidate report step does not match the checkpoint."
        )

    baseline_provenance = _validated_eval_provenance(
        baseline_report,
        label="logits baseline",
    )
    candidate_provenance = _validated_eval_provenance(
        candidate_report,
        label="logits candidate",
    )
    baseline_feature_seed = baseline_provenance.get("feature_seed")
    candidate_feature_seed = candidate_provenance.get("feature_seed")
    if (
        not isinstance(baseline_feature_seed, int)
        or baseline_feature_seed < 0
        or candidate_feature_seed != baseline_feature_seed
    ):
        raise ValueError(
            "Stage211 logits baseline/candidate reports require the same "
            "non-negative fixed feature seed."
        )
    if _eval_part_fingerprint(baseline_provenance) != _eval_part_fingerprint(
        candidate_provenance
    ):
        raise ValueError(
            "Stage211 baseline and candidate logits reports use different eval samples."
        )

    baseline_metrics = _validated_metrics(
        baseline_report,
        label="baseline",
    )
    candidate_metrics = _validated_metrics(
        candidate_report,
        label="candidate",
    )
    for key in IDENTICAL_TEACHER_METRICS:
        if not math.isclose(
            baseline_metrics[key],
            candidate_metrics[key],
            rel_tol=0.0,
            abs_tol=TOLERANCE,
        ):
            raise ValueError(
                f"Stage211 logits baseline/candidate teacher metric differs: {key}."
            )

    full_kl_reduction = _relative_reduction(
        baseline_metrics["full_kl"],
        candidate_metrics["full_kl"],
    )
    conditional_kl_reduction = _relative_reduction(
        baseline_metrics["conditional_nonblank_kl"],
        candidate_metrics["conditional_nonblank_kl"],
    )
    checks = {
        "full_kl_materially_improved": (
            full_kl_reduction >= MIN_KL_RELATIVE_REDUCTION
        ),
        "conditional_nonblank_kl_materially_improved": (
            conditional_kl_reduction >= MIN_KL_RELATIVE_REDUCTION
        ),
        "conditional_nonblank_hard_ce_not_worse": (
            candidate_metrics["conditional_nonblank_hard_ce"]
            <= baseline_metrics["conditional_nonblank_hard_ce"] + TOLERANCE
        ),
        "blank_binary_kl_not_worse": (
            candidate_metrics["blank_binary_kl"]
            <= baseline_metrics["blank_binary_kl"] + TOLERANCE
        ),
        "blank_probability_mae_not_worse": (
            candidate_metrics["blank_prob_mae"]
            <= baseline_metrics["blank_prob_mae"] + TOLERANCE
        ),
        "selected_top1_improved": (
            candidate_metrics["selected_top1_agreement"]
            > baseline_metrics["selected_top1_agreement"] + TOLERANCE
        ),
        "all_top1_improved": (
            candidate_metrics["all_top1_agreement"]
            > baseline_metrics["all_top1_agreement"] + TOLERANCE
        ),
        "active_top1_improved": (
            candidate_metrics["active_top1_agreement"]
            > baseline_metrics["active_top1_agreement"] + TOLERANCE
        ),
        "ctc_token_error_rate_improved": (
            candidate_metrics["ctc_token_error_rate"]
            < baseline_metrics["ctc_token_error_rate"] - TOLERANCE
        ),
        "ctc_token_deletion_rate_not_worse": (
            candidate_metrics["ctc_token_deletion_rate"]
            <= baseline_metrics["ctc_token_deletion_rate"] + TOLERANCE
        ),
        "sequence_exact_rate_not_worse": (
            candidate_metrics["sequence_exact_rate"]
            >= baseline_metrics["sequence_exact_rate"] - TOLERANCE
        ),
        "nonblank_rate_ratio_in_range": (
            RATIO_MIN
            <= candidate_metrics["nonblank_rate_ratio"]
            <= RATIO_MAX
        ),
        "nonblank_rate_ratio_not_farther": (
            abs(candidate_metrics["nonblank_rate_ratio"] - 1.0)
            <= abs(baseline_metrics["nonblank_rate_ratio"] - 1.0)
            + TOLERANCE
        ),
        "collapsed_length_ratio_in_range": (
            RATIO_MIN
            <= candidate_metrics["collapsed_length_ratio"]
            <= RATIO_MAX
        ),
        "collapsed_length_ratio_not_farther": (
            abs(candidate_metrics["collapsed_length_ratio"] - 1.0)
            <= abs(baseline_metrics["collapsed_length_ratio"] - 1.0)
            + TOLERANCE
        ),
        "complete_exact_coverage": True,
    }
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "logits_alignment_gate",
        "phase": "logits",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "gate_passed": all(checks.values()),
        "thresholds": {
            "minimum_kl_relative_reduction": MIN_KL_RELATIVE_REDUCTION,
            "ratio_min": RATIO_MIN,
            "ratio_max": RATIO_MAX,
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "baseline_report_path": str(baseline_report_path),
        "baseline_report_sha256": sha256_file(baseline_report_path),
        "candidate_report_path": str(candidate_report_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "baseline_eval_provenance": baseline_provenance,
        "candidate_eval_provenance": candidate_provenance,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "full_kl_relative_reduction": full_kl_reduction,
        "conditional_nonblank_kl_relative_reduction": (
            conditional_kl_reduction
        ),
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(
            f"Refusing to overwrite a different logits gate: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the independent fixed-audio Stage211C logits-alignment gate."
        )
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_gate(
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        checkpoint_path=args.checkpoint,
    )
    output_path = args.output.expanduser().resolve()
    _write_immutable_json(output_path, report)
    print(
        f"logits_gate={output_path} "
        f"gate_passed={str(report['gate_passed']).lower()} "
        f"full_kl_reduction={report['full_kl_relative_reduction']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
