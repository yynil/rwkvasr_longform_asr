from __future__ import annotations

import argparse
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml


EXPECTED_LAYER_IDS = tuple(str(layer_id) for layer_id in range(70))
ALIGNMENT_TOLERANCE = 1.0e-12
MIN_IMPROVED_LAYERS = 68


def _normalize_layer_map(raw: Any, *, label: str) -> dict[str, Mapping[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} is not a mapping")
    normalized: dict[str, Mapping[str, Any]] = {}
    for raw_layer_id, raw_metrics in raw.items():
        layer_id = str(raw_layer_id)
        if layer_id in normalized:
            raise ValueError(f"{label} contains duplicate layer {layer_id}")
        if not isinstance(raw_metrics, Mapping):
            raise ValueError(f"{label} layer {layer_id} metrics are not a mapping")
        normalized[layer_id] = raw_metrics
    if set(normalized) != set(EXPECTED_LAYER_IDS):
        raise ValueError(f"{label} does not contain exactly 70 layers")
    for layer_id in EXPECTED_LAYER_IDS:
        for metric_name in ("loss", "cosine"):
            try:
                value = float(normalized[layer_id][metric_name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{label} layer {layer_id} {metric_name} is not numeric"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"{label} layer {layer_id} {metric_name} is not finite")
    return normalized


def _report_layer_map(
    report: Mapping[str, Any], *, role: str
) -> dict[str, Mapping[str, Any]] | None:
    raw = report.get("layer_metrics")
    if raw is None:
        raw = report.get("layers")
    return _normalize_layer_map(raw, label=f"{role} layers")


def _report_component_maps(
    report: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, dict[str, Mapping[str, Any]]] | None:
    raw = report.get("layer_component_metrics")
    if raw is None:
        raw = report.get("layer_components")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{role} layer components are not a mapping")
    components: dict[str, dict[str, Mapping[str, Any]]] = {}
    for raw_component, raw_layers in raw.items():
        component = str(raw_component)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", component):
            raise ValueError(f"{role} component name is invalid")
        normalized = _normalize_layer_map(
            raw_layers,
            label=f"{role} {component} layers",
        )
        assert normalized is not None
        components[component] = normalized
    return components


def summarize_layer_progress(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    loss_deltas = {
        layer_id: float(candidate[layer_id]["loss"]) - float(baseline[layer_id]["loss"])
        for layer_id in EXPECTED_LAYER_IDS
    }
    cosine_deltas = {
        layer_id: float(candidate[layer_id]["cosine"]) - float(baseline[layer_id]["cosine"])
        for layer_id in EXPECTED_LAYER_IDS
    }
    loss_improved = sum(delta < -ALIGNMENT_TOLERANCE for delta in loss_deltas.values())
    loss_regressed = sum(delta > ALIGNMENT_TOLERANCE for delta in loss_deltas.values())
    cosine_improved = sum(delta > ALIGNMENT_TOLERANCE for delta in cosine_deltas.values())
    cosine_regressed = sum(delta < -ALIGNMENT_TOLERANCE for delta in cosine_deltas.values())
    baseline_mean_loss = sum(
        float(baseline[layer_id]["loss"]) for layer_id in EXPECTED_LAYER_IDS
    ) / len(EXPECTED_LAYER_IDS)
    candidate_mean_loss = sum(
        float(candidate[layer_id]["loss"]) for layer_id in EXPECTED_LAYER_IDS
    ) / len(EXPECTED_LAYER_IDS)
    baseline_mean_cosine = sum(
        float(baseline[layer_id]["cosine"]) for layer_id in EXPECTED_LAYER_IDS
    ) / len(EXPECTED_LAYER_IDS)
    candidate_mean_cosine = sum(
        float(candidate[layer_id]["cosine"]) for layer_id in EXPECTED_LAYER_IDS
    ) / len(EXPECTED_LAYER_IDS)
    worst_loss_layer = max(EXPECTED_LAYER_IDS, key=lambda layer_id: loss_deltas[layer_id])
    worst_cosine_layer = min(EXPECTED_LAYER_IDS, key=lambda layer_id: cosine_deltas[layer_id])
    return {
        "layer_count": len(EXPECTED_LAYER_IDS),
        "baseline_mean_loss": baseline_mean_loss,
        "candidate_mean_loss": candidate_mean_loss,
        "mean_loss_delta": candidate_mean_loss - baseline_mean_loss,
        "baseline_mean_cosine": baseline_mean_cosine,
        "candidate_mean_cosine": candidate_mean_cosine,
        "mean_cosine_delta": candidate_mean_cosine - baseline_mean_cosine,
        "loss_improved": loss_improved,
        "loss_regressed": loss_regressed,
        "loss_flat": len(EXPECTED_LAYER_IDS) - loss_improved - loss_regressed,
        "cosine_improved": cosine_improved,
        "cosine_regressed": cosine_regressed,
        "cosine_flat": len(EXPECTED_LAYER_IDS) - cosine_improved - cosine_regressed,
        "worst_loss_layer": int(worst_loss_layer),
        "worst_loss_delta": loss_deltas[worst_loss_layer],
        "worst_cosine_layer": int(worst_cosine_layer),
        "worst_cosine_delta": cosine_deltas[worst_cosine_layer],
        "min_68_loss": loss_improved >= MIN_IMPROVED_LAYERS,
        "min_68_cosine": cosine_improved >= MIN_IMPROVED_LAYERS,
    }


def build_fixed_eval_layer_progress(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_layers = _report_layer_map(baseline_report, role="baseline")
    candidate_layers = _report_layer_map(candidate_report, role="candidate")
    if baseline_layers is None or candidate_layers is None:
        return {
            "status": "unavailable",
            "reason": "missing_layer_metrics",
            "baseline_layers": "present" if baseline_layers is not None else "absent",
            "candidate_layers": "present" if candidate_layers is not None else "absent",
            "components": {},
        }
    baseline_components = _report_component_maps(baseline_report, role="baseline")
    candidate_components = _report_component_maps(candidate_report, role="candidate")
    if (baseline_components is None) != (candidate_components is None):
        raise ValueError("baseline and candidate component coverage differs")
    component_summaries: dict[str, dict[str, Any]] = {}
    if baseline_components is not None and candidate_components is not None:
        if set(baseline_components) != set(candidate_components):
            raise ValueError("baseline and candidate component sets differ")
        component_summaries = {
            component: summarize_layer_progress(
                baseline_components[component],
                candidate_components[component],
            )
            for component in sorted(baseline_components)
        }
    return {
        "status": "ok",
        "summary": summarize_layer_progress(baseline_layers, candidate_layers),
        "components": component_summaries,
    }


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _summary_line(prefix: str, summary: Mapping[str, Any], *, component: str | None) -> str:
    component_field = "" if component is None else f" component={component}"
    return (
        f"{prefix} status=ok{component_field} layer_count={summary['layer_count']} "
        f"baseline_mean_loss={float(summary['baseline_mean_loss']):.10f} "
        f"candidate_mean_loss={float(summary['candidate_mean_loss']):.10f} "
        f"mean_loss_delta={float(summary['mean_loss_delta']):+.10f} "
        f"baseline_mean_cosine={float(summary['baseline_mean_cosine']):.10f} "
        f"candidate_mean_cosine={float(summary['candidate_mean_cosine']):.10f} "
        f"mean_cosine_delta={float(summary['mean_cosine_delta']):+.10f} "
        f"loss_improved={summary['loss_improved']} loss_regressed={summary['loss_regressed']} "
        f"loss_flat={summary['loss_flat']} cosine_improved={summary['cosine_improved']} "
        f"cosine_regressed={summary['cosine_regressed']} cosine_flat={summary['cosine_flat']} "
        f"worst_loss_layer={summary['worst_loss_layer']} "
        f"worst_loss_delta={float(summary['worst_loss_delta']):+.10f} "
        f"worst_cosine_layer={summary['worst_cosine_layer']} "
        f"worst_cosine_delta={float(summary['worst_cosine_delta']):+.10f} "
        f"min_68_loss={_bool_text(bool(summary['min_68_loss']))} "
        f"min_68_cosine={_bool_text(bool(summary['min_68_cosine']))}"
    )


def render_fixed_eval_layer_progress(report: Mapping[str, Any]) -> list[str]:
    if report.get("status") == "unavailable":
        return [
            "fixed_eval_layer_progress status=unavailable "
            f"reason={report['reason']} baseline_layers={report['baseline_layers']} "
            f"candidate_layers={report['candidate_layers']}"
        ]
    lines = [
        _summary_line(
            "fixed_eval_layer_progress",
            report["summary"],
            component=None,
        )
    ]
    lines.extend(
        _summary_line(
            "fixed_eval_component_progress",
            summary,
            component=component,
        )
        for component, summary in report["components"].items()
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report per-layer progress between two Stage211 fixed-hidden evaluations."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_fixed_eval_layer_progress(
            load_yaml(args.baseline.expanduser().resolve()),
            load_yaml(args.candidate.expanduser().resolve()),
        )
    except (OSError, TypeError, ValueError) as error:
        reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(error)).strip("_") or "unknown"
        print(f"fixed_eval_layer_progress status=invalid reason={reason}", flush=True)
        return 1
    for line in render_fixed_eval_layer_progress(report):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
