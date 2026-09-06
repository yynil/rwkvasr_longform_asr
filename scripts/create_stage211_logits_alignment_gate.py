from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_artifact_io import write_immutable_json
from rwkvasr.eval.stage211_gate import sha256_file

try:
    from scripts.create_stage211_hidden_alignment_gate import (
        FIXED_EVAL_SAMPLES,
        STRATIFIED_CELLS,
        STRATIFIED_EVAL_SAMPLES,
        WEAK_BANDS,
        _component_layers,
        _eval_part_fingerprint,
        _pair_shared_binding,
        _summary,
        _validated_pair_report,
        _validate_stratified_component_summary,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_hidden_alignment_gate import (
        FIXED_EVAL_SAMPLES,
        STRATIFIED_CELLS,
        STRATIFIED_EVAL_SAMPLES,
        WEAK_BANDS,
        _component_layers,
        _eval_part_fingerprint,
        _pair_shared_binding,
        _summary,
        _validated_pair_report,
        _validate_stratified_component_summary,
    )


MIN_KL_RELATIVE_REDUCTION = 0.05
RATIO_MIN = 0.90
RATIO_MAX = 1.10
TOLERANCE = 1.0e-12
MAX_CELL_TOKEN_ERROR_REGRESSION = 0.03
HIDDEN_COMPONENTS = ("mixer", "ffn", "block")
MAX_HIDDEN_LOSS_REGRESSION = 0.10
MAX_HIDDEN_COSINE_REGRESSION = 0.01
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
        raise ValueError(f"Stage211 {label} logits report eval sample count mismatch.")
    raw_metrics = report.get("logit_metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"Stage211 {label} report lacks logit_metrics.")
    metrics: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        value = float(raw_metrics.get(key, float("nan")))
        if not math.isfinite(value):
            raise ValueError(f"Stage211 {label} logits metric {key} must be finite.")
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


def _metric_checks(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> tuple[dict[str, bool], float, float]:
    full_kl_reduction = _relative_reduction(
        baseline_metrics["full_kl"],
        candidate_metrics["full_kl"],
    )
    conditional_kl_reduction = _relative_reduction(
        baseline_metrics["conditional_nonblank_kl"],
        candidate_metrics["conditional_nonblank_kl"],
    )
    checks = {
        "full_kl_materially_improved": (full_kl_reduction >= MIN_KL_RELATIVE_REDUCTION),
        "conditional_nonblank_kl_materially_improved": (
            conditional_kl_reduction >= MIN_KL_RELATIVE_REDUCTION
        ),
        "conditional_nonblank_hard_ce_not_worse": (
            candidate_metrics["conditional_nonblank_hard_ce"]
            <= baseline_metrics["conditional_nonblank_hard_ce"] + TOLERANCE
        ),
        "blank_binary_kl_not_worse": (
            candidate_metrics["blank_binary_kl"] <= baseline_metrics["blank_binary_kl"] + TOLERANCE
        ),
        "blank_probability_mae_not_worse": (
            candidate_metrics["blank_prob_mae"] <= baseline_metrics["blank_prob_mae"] + TOLERANCE
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
            RATIO_MIN <= candidate_metrics["nonblank_rate_ratio"] <= RATIO_MAX
        ),
        "nonblank_rate_ratio_not_farther": (
            abs(candidate_metrics["nonblank_rate_ratio"] - 1.0)
            <= abs(baseline_metrics["nonblank_rate_ratio"] - 1.0) + TOLERANCE
        ),
        "collapsed_length_ratio_in_range": (
            RATIO_MIN <= candidate_metrics["collapsed_length_ratio"] <= RATIO_MAX
        ),
        "collapsed_length_ratio_not_farther": (
            abs(candidate_metrics["collapsed_length_ratio"] - 1.0)
            <= abs(baseline_metrics["collapsed_length_ratio"] - 1.0) + TOLERANCE
        ),
        "complete_exact_coverage": True,
    }
    return checks, full_kl_reduction, conditional_kl_reduction


def _validated_stratified_summary(
    path: Path,
    *,
    baseline_checkpoint_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    summary = load_yaml(path)
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stratified_logits_eval_summary",
        "phase": "logits",
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 stratified logits summary binding mismatch.")
    receipt_path = Path(str(summary.get("receipt_path") or "")).resolve()
    if not receipt_path.is_file() or sha256_file(receipt_path) != summary.get("receipt_sha256"):
        raise ValueError("Stage211 stratified logits receipt is missing or changed.")
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Stage211 stratified logits summary lacks checkpoints.")
    for role, expected_path in (
        ("baseline", baseline_checkpoint_path),
        ("candidate", checkpoint_path),
    ):
        record = checkpoints.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 stratified logits lacks {role} checkpoint.")
        bound_path = Path(str(record.get("path") or "")).resolve()
        if (
            bound_path != expected_path
            or not bound_path.is_file()
            or sha256_file(bound_path) != record.get("sha256")
        ):
            raise ValueError(f"Stage211 stratified logits {role} checkpoint mismatch.")
    cells = summary.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(STRATIFIED_CELLS):
        raise ValueError("Stage211 stratified logits cell coverage mismatch.")
    for cell_name, cell in cells.items():
        if not isinstance(cell, dict) or int(cell.get("samples", -1)) != FIXED_EVAL_SAMPLES:
            raise ValueError(f"Stage211 stratified logits cell is invalid: {cell_name}")
        manifest_path = Path(str(cell.get("manifest_path") or "")).resolve()
        if not manifest_path.is_file() or sha256_file(manifest_path) != cell.get("manifest_sha256"):
            raise ValueError(f"Stage211 stratified logits manifest changed: {cell_name}")
        for role in ("baseline", "candidate"):
            raw_metrics = cell.get(f"{role}_metrics")
            if not isinstance(raw_metrics, dict):
                raise ValueError(
                    f"Stage211 stratified logits cell lacks {role} metrics: {cell_name}"
                )
            _validated_summary_metrics(
                raw_metrics,
                label=f"cell {cell_name}/{role}",
                expected_matched=FIXED_EVAL_SAMPLES,
            )
    macro = summary.get("macro")
    if (
        not isinstance(macro, dict)
        or int(macro.get("cells", -1)) != len(STRATIFIED_CELLS)
        or int(macro.get("samples", -1)) != STRATIFIED_EVAL_SAMPLES
    ):
        raise ValueError("Stage211 stratified logits macro coverage mismatch.")
    for role in ("baseline", "candidate"):
        raw_metrics = macro.get(f"{role}_metrics")
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"Stage211 stratified logits macro lacks {role} metrics.")
        _validated_summary_metrics(
            raw_metrics,
            label=f"macro/{role}",
            expected_matched=STRATIFIED_EVAL_SAMPLES,
        )
    hidden_component_summaries = summary.get("hidden_component_summaries")
    if not isinstance(hidden_component_summaries, dict) or set(hidden_component_summaries) != set(
        HIDDEN_COMPONENTS
    ):
        raise ValueError("Stage211 stratified logits hidden-component coverage mismatch.")
    for component_name in HIDDEN_COMPONENTS:
        _validate_stratified_component_summary(
            hidden_component_summaries[component_name],
            component_name=component_name,
        )
    decoder_hidden = summary.get("decoder_hidden")
    if (
        not isinstance(decoder_hidden, dict)
        or int(decoder_hidden.get("cells", -1)) != len(STRATIFIED_CELLS)
        or not all(
            math.isfinite(float(decoder_hidden.get(key, float("nan"))))
            for key in ("baseline_loss", "candidate_loss", "relative_change_pct")
        )
    ):
        raise ValueError("Stage211 stratified logits decoder-hidden summary is invalid.")
    decoder_cells = decoder_hidden.get("cell_results")
    if not isinstance(decoder_cells, dict) or set(decoder_cells) != set(STRATIFIED_CELLS):
        raise ValueError("Stage211 stratified logits decoder-hidden cell coverage mismatch.")
    for cell_name, row in decoder_cells.items():
        if not isinstance(row, dict) or not all(
            math.isfinite(float(row.get(key, float("nan"))))
            for key in ("baseline_loss", "candidate_loss", "relative_change_pct")
        ):
            raise ValueError(
                f"Stage211 stratified logits decoder-hidden cell is invalid: {cell_name}."
            )
    report_bindings = summary.get("reports")
    if not isinstance(report_bindings, dict) or set(report_bindings) != set(STRATIFIED_CELLS):
        raise ValueError("Stage211 stratified logits report coverage mismatch.")
    for cell_name, roles in report_bindings.items():
        if not isinstance(roles, dict) or set(roles) != {"baseline", "candidate"}:
            raise ValueError(f"Stage211 stratified logits report roles mismatch: {cell_name}")
        for role, binding in roles.items():
            if not isinstance(binding, dict):
                raise ValueError(f"Stage211 stratified logits report invalid: {cell_name}/{role}")
            report_path = Path(str(binding.get("path") or "")).resolve()
            if not report_path.is_file() or sha256_file(report_path) != binding.get("sha256"):
                raise ValueError(
                    f"Stage211 stratified logits report is missing or changed: {cell_name}/{role}"
                )
    return summary


def _validated_summary_metrics(
    raw_metrics: dict[str, Any],
    *,
    label: str,
    expected_matched: int,
) -> dict[str, float]:
    metrics = {key: float(raw_metrics.get(key, float("nan"))) for key in REQUIRED_METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Stage211 stratified logits {label} metrics are not finite.")
    if (
        int(round(metrics["matched_utterances"])) != expected_matched
        or metrics["missing_utterances"] != 0.0
        or metrics["mean_frame_delta"] != 0.0
    ):
        raise ValueError(f"Stage211 stratified logits {label} coverage is incomplete.")
    return metrics


def _stratified_gate_passed(summary: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    macro = summary["macro"]
    baseline = _validated_summary_metrics(
        macro["baseline_metrics"],
        label="macro/baseline",
        expected_matched=STRATIFIED_EVAL_SAMPLES,
    )
    candidate = _validated_summary_metrics(
        macro["candidate_metrics"],
        label="macro/candidate",
        expected_matched=STRATIFIED_EVAL_SAMPLES,
    )
    checks, _, _ = _metric_checks(baseline, candidate)
    cells = summary["cells"]
    hard_cells_improved = all(
        float(cells[cell_name]["candidate_metrics"]["full_kl"])
        < float(cells[cell_name]["baseline_metrics"]["full_kl"])
        and float(cells[cell_name]["candidate_metrics"]["conditional_nonblank_kl"])
        < float(cells[cell_name]["baseline_metrics"]["conditional_nonblank_kl"])
        and float(cells[cell_name]["candidate_metrics"]["ctc_token_error_rate"])
        < float(cells[cell_name]["baseline_metrics"]["ctc_token_error_rate"])
        for cell_name in ("hard_en", "hard_zh", "long_zh")
    )
    bounded_cell_token_regression = all(
        float(cell["candidate_metrics"]["ctc_token_error_rate"])
        <= float(cell["baseline_metrics"]["ctc_token_error_rate"]) + MAX_CELL_TOKEN_ERROR_REGRESSION
        for cell in cells.values()
    )
    component_retention: dict[str, bool] = {}
    for component_name in HIDDEN_COMPONENTS:
        component = summary["hidden_component_summaries"][component_name]
        macro_and_weak_retained = all(_hidden_retention_checks(component).values())
        cells_retained = all(
            float(row["candidate_loss"])
            <= float(row["baseline_loss"]) * (1.0 + MAX_HIDDEN_LOSS_REGRESSION) + TOLERANCE
            and float(row["candidate_cosine"])
            >= float(row["baseline_cosine"]) - MAX_HIDDEN_COSINE_REGRESSION - TOLERANCE
            for row in component["cells"].values()
        )
        component_retention[f"{component_name}_hidden_retained"] = (
            macro_and_weak_retained and cells_retained
        )
    decoder_hidden = summary["decoder_hidden"]
    decoder_hidden_retained = float(decoder_hidden["candidate_loss"]) <= float(
        decoder_hidden["baseline_loss"]
    ) * (1.0 + MAX_HIDDEN_LOSS_REGRESSION) + TOLERANCE and all(
        float(row["candidate_loss"])
        <= float(row["baseline_loss"]) * (1.0 + MAX_HIDDEN_LOSS_REGRESSION) + TOLERANCE
        for row in decoder_hidden["cell_results"].values()
    )
    representative_checks = {
        **checks,
        "hard_and_long_cells_improved": hard_cells_improved,
        "cell_token_error_regression_bounded": bounded_cell_token_regression,
        **component_retention,
        "decoder_hidden_retained": decoder_hidden_retained,
        "complete_stratified_coverage": True,
    }
    return all(representative_checks.values()), representative_checks


def _hidden_retention_checks(summary: dict[str, Any]) -> dict[str, bool]:
    required_scalars = (
        "baseline_mean_loss",
        "candidate_mean_loss",
        "baseline_mean_cosine",
        "candidate_mean_cosine",
    )
    if not all(math.isfinite(float(summary.get(key, float("nan")))) for key in required_scalars):
        raise ValueError("Stage211 logits hidden-component summary is not finite.")
    weak_bands = summary.get("weak_bands")
    if not isinstance(weak_bands, dict) or set(weak_bands) != set(WEAK_BANDS):
        raise ValueError("Stage211 logits hidden-component weak-band coverage mismatch.")
    for band_name, row in weak_bands.items():
        if not isinstance(row, dict) or not all(
            math.isfinite(float(row.get(key, float("nan"))))
            for key in (
                "baseline_loss",
                "candidate_loss",
                "baseline_cosine",
                "candidate_cosine",
            )
        ):
            raise ValueError(
                f"Stage211 logits hidden-component weak band is not finite: {band_name}."
            )
    return {
        "mean_loss_retained": float(summary["candidate_mean_loss"])
        <= float(summary["baseline_mean_loss"]) * (1.0 + MAX_HIDDEN_LOSS_REGRESSION) + TOLERANCE,
        "mean_cosine_retained": float(summary["candidate_mean_cosine"])
        >= float(summary["baseline_mean_cosine"]) - MAX_HIDDEN_COSINE_REGRESSION - TOLERANCE,
        "weak_band_loss_retained": all(
            float(row["candidate_loss"])
            <= float(row["baseline_loss"]) * (1.0 + MAX_HIDDEN_LOSS_REGRESSION) + TOLERANCE
            for row in weak_bands.values()
        ),
        "weak_band_cosine_retained": all(
            float(row["candidate_cosine"])
            >= float(row["baseline_cosine"]) - MAX_HIDDEN_COSINE_REGRESSION - TOLERANCE
            for row in weak_bands.values()
        ),
    }


def _decoder_hidden_retention(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = baseline_report.get("decoder_hidden_metrics") or {}
    candidate_metrics = candidate_report.get("decoder_hidden_metrics") or {}
    baseline_loss = float(baseline_metrics.get("loss", float("nan")))
    candidate_loss = float(candidate_metrics.get("loss", float("nan")))
    if not all(math.isfinite(value) for value in (baseline_loss, candidate_loss)):
        raise ValueError(
            "Stage211 logits gate requires finite decoder-hidden loss in both reports."
        )
    retained = candidate_loss <= baseline_loss * (1.0 + MAX_HIDDEN_LOSS_REGRESSION) + TOLERANCE
    return {
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "relative_change": (candidate_loss - baseline_loss) / max(baseline_loss, 1.0e-12),
        "retained": retained,
    }


def build_gate(
    *,
    baseline_report_path: Path,
    candidate_report_path: Path,
    baseline_checkpoint_path: Path,
    checkpoint_path: Path,
    stratified_summary_path: Path | None = None,
) -> dict[str, Any]:
    baseline_report_path = baseline_report_path.expanduser().resolve()
    candidate_report_path = candidate_report_path.expanduser().resolve()
    baseline_checkpoint_path = baseline_checkpoint_path.expanduser().resolve()
    checkpoint_path = checkpoint_path.expanduser().resolve()
    for path in (baseline_checkpoint_path, checkpoint_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(str(path))
    baseline_report = load_yaml(baseline_report_path)
    candidate_report = load_yaml(candidate_report_path)

    baseline_provenance = _validated_pair_report(
        baseline_report,
        phase="logits",
        role="baseline",
        checkpoint_path=baseline_checkpoint_path,
    )
    candidate_provenance = _validated_pair_report(
        candidate_report,
        phase="logits",
        role="candidate",
        checkpoint_path=checkpoint_path,
    )
    if _pair_shared_binding(baseline_report) != _pair_shared_binding(candidate_report):
        raise ValueError(
            "Stage211 baseline and candidate logits reports were not produced "
            "by the same paired evaluation."
        )
    if baseline_provenance != candidate_provenance:
        raise ValueError(
            "Stage211 baseline and candidate logits reports do not bind the "
            "same fixed-eval provenance."
        )
    if _eval_part_fingerprint(baseline_provenance) != _eval_part_fingerprint(candidate_provenance):
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
            raise ValueError(f"Stage211 logits baseline/candidate teacher metric differs: {key}.")

    checks, full_kl_reduction, conditional_kl_reduction = _metric_checks(
        baseline_metrics,
        candidate_metrics,
    )
    hidden_component_summaries = {
        component_name: _summary(
            _component_layers(
                baseline_report,
                phase="logits",
                component_name=component_name,
            ),
            _component_layers(
                candidate_report,
                phase="logits",
                component_name=component_name,
            ),
        )
        for component_name in HIDDEN_COMPONENTS
    }
    hidden_retention_checks = {
        component_name: _hidden_retention_checks(summary)
        for component_name, summary in hidden_component_summaries.items()
    }
    decoder_hidden_retention = _decoder_hidden_retention(
        baseline_report,
        candidate_report,
    )
    hidden_retention_passed = all(
        all(component_checks.values()) for component_checks in hidden_retention_checks.values()
    ) and bool(decoder_hidden_retention["retained"])
    legacy_gate_passed = all(checks.values())
    stratified_summary = None
    stratified_gate_passed = None
    stratified_checks = None
    if stratified_summary_path is not None:
        stratified_summary_path = stratified_summary_path.expanduser().resolve()
        stratified_summary = _validated_stratified_summary(
            stratified_summary_path,
            baseline_checkpoint_path=baseline_checkpoint_path,
            checkpoint_path=checkpoint_path,
        )
        stratified_gate_passed, stratified_checks = _stratified_gate_passed(stratified_summary)
    selected_logits_gate_passed = (
        bool(stratified_gate_passed) if stratified_gate_passed is not None else legacy_gate_passed
    )
    gate_passed = selected_logits_gate_passed and hidden_retention_passed
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "logits_alignment_gate",
        "phase": "logits",
        "baseline_checkpoint_path": str(baseline_checkpoint_path),
        "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "gate_passed": gate_passed,
        "legacy_gate_passed": legacy_gate_passed,
        "selected_logits_gate_passed": selected_logits_gate_passed,
        "stratified_gate_passed": stratified_gate_passed,
        "stratified_checks": stratified_checks,
        "stratified_summary_path": (
            str(stratified_summary_path) if stratified_summary_path is not None else None
        ),
        "stratified_summary_sha256": (
            sha256_file(stratified_summary_path) if stratified_summary_path is not None else None
        ),
        "stratified_summary": stratified_summary,
        "thresholds": {
            "minimum_kl_relative_reduction": MIN_KL_RELATIVE_REDUCTION,
            "ratio_min": RATIO_MIN,
            "ratio_max": RATIO_MAX,
            "max_cell_token_error_regression": (MAX_CELL_TOKEN_ERROR_REGRESSION),
            "max_hidden_loss_regression": MAX_HIDDEN_LOSS_REGRESSION,
            "max_hidden_cosine_regression": MAX_HIDDEN_COSINE_REGRESSION,
            "tolerance": TOLERANCE,
        },
        "checks": checks,
        "hidden_component_summaries": hidden_component_summaries,
        "hidden_retention_checks": hidden_retention_checks,
        "hidden_retention_passed": hidden_retention_passed,
        "decoder_hidden_retention": decoder_hidden_retention,
        "baseline_report_path": str(baseline_report_path),
        "baseline_report_sha256": sha256_file(baseline_report_path),
        "candidate_report_path": str(candidate_report_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "baseline_eval_provenance": baseline_provenance,
        "candidate_eval_provenance": candidate_provenance,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "full_kl_relative_reduction": full_kl_reduction,
        "conditional_nonblank_kl_relative_reduction": (conditional_kl_reduction),
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    write_immutable_json(path, payload, label="logits gate")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Create the independent fixed-audio Stage211C logits-alignment gate.")
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stratified-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_gate(
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        baseline_checkpoint_path=args.baseline_checkpoint,
        checkpoint_path=args.checkpoint,
        stratified_summary_path=args.stratified_summary,
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
