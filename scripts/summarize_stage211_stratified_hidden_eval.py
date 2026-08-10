from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROLES = ("baseline", "candidate")
LAYER_IDS = tuple(range(70))
WEAK_BANDS = {"10-19": tuple(range(10, 20)), "20-29": tuple(range(20, 30))}
PHASE_COMPONENTS = {
    "mixer": ("mixer",),
    "block": ("mixer", "ffn", "block"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    rendered = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to replace a different summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _validate_cell_report(
    *,
    report: dict[str, Any],
    report_path: Path,
    role: str,
    phase: str,
    expected_samples: int,
    expected_manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if str(report.get("role")) != role or str(report.get("phase")) != phase:
        raise ValueError(f"Stage211 sidecar role/phase mismatch: {report_path}")
    if int(report.get("eval_samples", -1)) != expected_samples:
        raise ValueError(f"Stage211 sidecar sample mismatch: {report_path}")
    eval_loss = float(report.get("eval_loss", float("nan")))
    if not math.isfinite(eval_loss):
        raise ValueError(f"Stage211 sidecar loss is not finite: {report_path}")
    provenance = report.get("eval_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Stage211 sidecar lacks eval provenance: {report_path}")
    if (
        Path(str(provenance.get("bucket_manifest_path") or "")).resolve()
        != expected_manifest_path
        or str(provenance.get("bucket_manifest_sha256") or "")
        != expected_manifest_sha256
        or int(provenance.get("split_samples", -1)) != expected_samples
    ):
        raise ValueError(f"Stage211 sidecar manifest provenance mismatch: {report_path}")
    components = report.get("layer_components")
    if not isinstance(components, dict):
        raise ValueError(f"Stage211 sidecar lacks layer components: {report_path}")
    try:
        required_components = PHASE_COMPONENTS[phase]
    except KeyError as error:
        raise ValueError(f"Unsupported Stage211 hidden phase: {phase!r}") from error
    validated: dict[str, dict[str, dict[str, float]]] = {}
    expected_layers = {str(index) for index in LAYER_IDS}
    for component_name in required_components:
        component_layers = components.get(component_name)
        if not isinstance(component_layers, dict) or set(component_layers) != expected_layers:
            raise ValueError(
                "Stage211 sidecar lacks exact 70-layer "
                f"{component_name} coverage: {report_path}"
            )
        for layer_id, metrics in component_layers.items():
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"Stage211 sidecar {component_name} layer {layer_id} is invalid: "
                    f"{report_path}"
                )
            for name in ("loss", "cosine", "rms_ratio"):
                if not math.isfinite(float(metrics.get(name, float("nan")))):
                    raise ValueError(
                        f"Stage211 sidecar {component_name} layer {layer_id} "
                        f"{name} is not finite: {report_path}"
                    )
        validated[component_name] = component_layers
    return validated


def _mean_layer_metric(
    layers: dict[str, dict[str, float]],
    metric: str,
    layer_ids: tuple[int, ...] = LAYER_IDS,
) -> float:
    return sum(float(layers[str(layer_id)][metric]) for layer_id in layer_ids) / len(
        layer_ids
    )


def _component_summary(
    *,
    component_name: str,
    components_by_cell: dict[
        str,
        dict[str, dict[str, dict[str, dict[str, float]]]],
    ],
) -> dict[str, Any]:
    macro_layers: dict[str, dict[str, float]] = {}
    for layer_id in LAYER_IDS:
        key = str(layer_id)
        baseline_loss = sum(
            values["baseline"][component_name][key]["loss"]
            for values in components_by_cell.values()
        ) / len(components_by_cell)
        candidate_loss = sum(
            values["candidate"][component_name][key]["loss"]
            for values in components_by_cell.values()
        ) / len(components_by_cell)
        baseline_cosine = sum(
            values["baseline"][component_name][key]["cosine"]
            for values in components_by_cell.values()
        ) / len(components_by_cell)
        candidate_cosine = sum(
            values["candidate"][component_name][key]["cosine"]
            for values in components_by_cell.values()
        ) / len(components_by_cell)
        baseline_rms_ratio = sum(
            values["baseline"][component_name][key]["rms_ratio"]
            for values in components_by_cell.values()
        ) / len(components_by_cell)
        candidate_rms_ratio = sum(
            values["candidate"][component_name][key]["rms_ratio"]
            for values in components_by_cell.values()
        ) / len(components_by_cell)
        macro_layers[key] = {
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss,
            "relative_loss_reduction": (baseline_loss - candidate_loss)
            / max(baseline_loss, 1.0e-12),
            "baseline_cosine": baseline_cosine,
            "candidate_cosine": candidate_cosine,
            "baseline_rms_ratio": baseline_rms_ratio,
            "candidate_rms_ratio": candidate_rms_ratio,
        }
    weak_bands: dict[str, dict[str, float]] = {}
    for name, layer_ids in WEAK_BANDS.items():
        baseline_loss = sum(
            macro_layers[str(layer_id)]["baseline_loss"] for layer_id in layer_ids
        ) / len(layer_ids)
        candidate_loss = sum(
            macro_layers[str(layer_id)]["candidate_loss"] for layer_id in layer_ids
        ) / len(layer_ids)
        weak_bands[name] = {
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss,
            "relative_loss_reduction": (baseline_loss - candidate_loss)
            / max(baseline_loss, 1.0e-12),
            "baseline_cosine": sum(
                macro_layers[str(layer_id)]["baseline_cosine"] for layer_id in layer_ids
            )
            / len(layer_ids),
            "candidate_cosine": sum(
                macro_layers[str(layer_id)]["candidate_cosine"] for layer_id in layer_ids
            )
            / len(layer_ids),
        }
    cells: dict[str, dict[str, float | int]] = {}
    for cell_name, values in components_by_cell.items():
        baseline_layers = values["baseline"][component_name]
        candidate_layers = values["candidate"][component_name]
        baseline_loss = _mean_layer_metric(baseline_layers, "loss")
        candidate_loss = _mean_layer_metric(candidate_layers, "loss")
        cells[cell_name] = {
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss,
            "relative_change_pct": 100.0 * (candidate_loss / baseline_loss - 1.0),
            "baseline_cosine": _mean_layer_metric(baseline_layers, "cosine"),
            "candidate_cosine": _mean_layer_metric(candidate_layers, "cosine"),
            "layers_loss_improved": sum(
                float(candidate_layers[str(layer_id)]["loss"])
                < float(baseline_layers[str(layer_id)]["loss"])
                for layer_id in LAYER_IDS
            ),
            "layers_cosine_improved": sum(
                float(candidate_layers[str(layer_id)]["cosine"])
                > float(baseline_layers[str(layer_id)]["cosine"])
                for layer_id in LAYER_IDS
            ),
        }
    return {
        "baseline_mean_loss": sum(
            row["baseline_loss"] for row in macro_layers.values()
        )
        / len(macro_layers),
        "candidate_mean_loss": sum(
            row["candidate_loss"] for row in macro_layers.values()
        )
        / len(macro_layers),
        "baseline_mean_cosine": sum(
            row["baseline_cosine"] for row in macro_layers.values()
        )
        / len(macro_layers),
        "candidate_mean_cosine": sum(
            row["candidate_cosine"] for row in macro_layers.values()
        )
        / len(macro_layers),
        "loss_improved_layers": sum(
            row["candidate_loss"] < row["baseline_loss"] for row in macro_layers.values()
        ),
        "cosine_improved_layers": sum(
            row["candidate_cosine"] > row["baseline_cosine"]
            for row in macro_layers.values()
        ),
        "weak_bands": weak_bands,
        "cells": cells,
        "layers": macro_layers,
    }


def summarize(
    *,
    receipt_path: Path,
    eval_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    eval_dir = eval_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 stratified eval receipt")
    if str(receipt.get("artifact")) != "stratified_hidden_eval_manifest":
        raise ValueError("Unexpected Stage211 stratified eval receipt artifact.")
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("Stage211 stratified eval receipt has no cells.")

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    checkpoint_bindings: dict[str, set[tuple[str, str]]] = {
        role: set() for role in ROLES
    }
    phase_values: set[str] = set()
    cell_results: dict[str, dict[str, Any]] = {}
    difficulty_losses: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {role: [] for role in ROLES}
    )
    report_bindings: dict[str, dict[str, Any]] = {}
    components_by_cell: dict[
        str,
        dict[str, dict[str, dict[str, dict[str, float]]]],
    ] = {}
    decoder_losses: dict[str, list[float]] = {role: [] for role in ROLES}
    for cell_name, cell in sorted(cells.items()):
        if not isinstance(cell, dict):
            raise ValueError(f"Invalid Stage211 sidecar cell receipt: {cell_name}")
        samples = int(cell.get("samples", -1))
        manifest_path = Path(str(cell.get("manifest_path") or "")).resolve()
        manifest_sha256 = str(cell.get("manifest_sha256") or "")
        if (
            samples <= 0
            or not manifest_path.is_file()
            or sha256_file(manifest_path) != manifest_sha256
        ):
            raise ValueError(f"Stage211 sidecar cell manifest mismatch: {cell_name}")
        reports[cell_name] = {}
        report_bindings[cell_name] = {}
        components_by_role: dict[
            str,
            dict[str, dict[str, dict[str, float]]],
        ] = {}
        for role in ROLES:
            report_path = (eval_dir / f"{cell_name}_{role}.json").resolve()
            report = _load_json(
                report_path,
                label=f"Stage211 {cell_name} {role} report",
            )
            phase = str(report.get("phase") or "")
            phase_values.add(phase)
            components_by_role[role] = _validate_cell_report(
                report=report,
                report_path=report_path,
                role=role,
                phase=phase,
                expected_samples=samples,
                expected_manifest_path=manifest_path,
                expected_manifest_sha256=manifest_sha256,
            )
            checkpoint_bindings[role].add(
                (
                    str(report.get("checkpoint_path") or ""),
                    str(report.get("checkpoint_sha256") or ""),
                )
            )
            reports[cell_name][role] = report
            report_bindings[cell_name][role] = {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            }
        baseline = reports[cell_name]["baseline"]
        candidate = reports[cell_name]["candidate"]
        cell_phases = {str(report.get("phase") or "") for report in reports[cell_name].values()}
        if len(cell_phases) != 1 or "" in cell_phases:
            raise ValueError(f"Stage211 sidecar pair phase mismatch: {cell_name}")
        cell_phase = next(iter(cell_phases))
        layers_by_role = {
            role: components_by_role[role][cell_phase] for role in ROLES
        }
        components_by_cell[cell_name] = components_by_role
        baseline_loss = float(baseline["eval_loss"])
        candidate_loss = float(candidate["eval_loss"])
        difficulty = cell_name.split("_", 1)[0]
        difficulty_losses[difficulty]["baseline"].append(baseline_loss)
        difficulty_losses[difficulty]["candidate"].append(candidate_loss)
        cell_results[cell_name] = {
            "samples": samples,
            "baseline_loss": baseline_loss,
            "candidate_loss": candidate_loss,
            "relative_change_pct": 100.0 * (candidate_loss / baseline_loss - 1.0),
            "layers_loss_improved": sum(
                float(layers_by_role["candidate"][layer_id]["loss"])
                < float(layers_by_role["baseline"][layer_id]["loss"])
                for layer_id in layers_by_role["baseline"]
            ),
            "layers_cosine_improved": sum(
                float(layers_by_role["candidate"][layer_id]["cosine"])
                > float(layers_by_role["baseline"][layer_id]["cosine"])
                for layer_id in layers_by_role["baseline"]
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
        }
        baseline_decoder = baseline.get("decoder_hidden_metrics") or {}
        candidate_decoder = candidate.get("decoder_hidden_metrics") or {}
        if "loss" in baseline_decoder or "loss" in candidate_decoder:
            if "loss" not in baseline_decoder or "loss" not in candidate_decoder:
                raise ValueError(
                    f"Stage211 sidecar decoder-hidden pair is incomplete: {cell_name}"
                )
            baseline_decoder_loss = float(baseline_decoder["loss"])
            candidate_decoder_loss = float(candidate_decoder["loss"])
            if not all(
                math.isfinite(value)
                for value in (baseline_decoder_loss, candidate_decoder_loss)
            ):
                raise ValueError(
                    f"Stage211 sidecar decoder-hidden loss is not finite: {cell_name}"
                )
            decoder_losses["baseline"].append(baseline_decoder_loss)
            decoder_losses["candidate"].append(candidate_decoder_loss)
            cell_results[cell_name]["decoder_hidden"] = {
                "baseline_loss": baseline_decoder_loss,
                "candidate_loss": candidate_decoder_loss,
                "relative_change_pct": 100.0
                * (candidate_decoder_loss / baseline_decoder_loss - 1.0),
            }

    if len(phase_values) != 1 or "" in phase_values:
        raise ValueError(f"Stage211 sidecar reports disagree on phase: {phase_values}")
    for role, bindings in checkpoint_bindings.items():
        if len(bindings) != 1:
            raise ValueError(
                f"Stage211 sidecar {role} reports disagree on checkpoint: {bindings}"
            )
        path, digest = next(iter(bindings))
        checkpoint_path = Path(path).expanduser().resolve()
        if (
            len(digest) != 64
            or not checkpoint_path.is_file()
            or sha256_file(checkpoint_path) != digest
        ):
            raise ValueError(f"Stage211 sidecar {role} checkpoint binding mismatch.")

    macro_baseline = sum(
        result["baseline_loss"] for result in cell_results.values()
    ) / len(cell_results)
    macro_candidate = sum(
        result["candidate_loss"] for result in cell_results.values()
    ) / len(cell_results)
    phase = next(iter(phase_values))
    component_summaries = {
        component_name: _component_summary(
            component_name=component_name,
            components_by_cell=components_by_cell,
        )
        for component_name in PHASE_COMPONENTS[phase]
    }
    primary_component_summary = component_summaries[phase]
    layer_summary = {
        key: primary_component_summary[key]
        for key in (
            "loss_improved_layers",
            "cosine_improved_layers",
            "weak_bands",
            "layers",
        )
    }
    difficulty_results = {}
    for difficulty, values in sorted(difficulty_losses.items()):
        baseline = sum(values["baseline"]) / len(values["baseline"])
        candidate = sum(values["candidate"]) / len(values["candidate"])
        difficulty_results[difficulty] = {
            "cells": len(values["baseline"]),
            "baseline_loss": baseline,
            "candidate_loss": candidate,
            "relative_change_pct": 100.0 * (candidate / baseline - 1.0),
        }
    checkpoint_records = {
        role: {
            "path": next(iter(bindings))[0],
            "sha256": next(iter(bindings))[1],
        }
        for role, bindings in checkpoint_bindings.items()
    }
    summary = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stratified_hidden_eval_summary",
        "phase": phase,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checkpoints": checkpoint_records,
        "cells": cell_results,
        "difficulties": difficulty_results,
        "macro": {
            "cells": len(cell_results),
            "samples": sum(result["samples"] for result in cell_results.values()),
            "baseline_loss": macro_baseline,
            "candidate_loss": macro_candidate,
            "relative_change_pct": 100.0 * (macro_candidate / macro_baseline - 1.0),
        },
        "layer_summary": layer_summary,
        "component_summaries": component_summaries,
        "decoder_hidden": (
            {
                "cells": len(decoder_losses["baseline"]),
                "baseline_loss": sum(decoder_losses["baseline"])
                / len(decoder_losses["baseline"]),
                "candidate_loss": sum(decoder_losses["candidate"])
                / len(decoder_losses["candidate"]),
            }
            if decoder_losses["baseline"]
            else None
        ),
        "reports": report_bindings,
    }
    _write_immutable(output_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Stage211 stratified hidden pair reports."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = summarize(
        receipt_path=args.receipt,
        eval_dir=args.eval_dir,
        output_path=args.output,
    )
    macro = summary["macro"]
    print(
        "stage211_stratified_hidden_summary "
        f"phase={summary['phase']} cells={macro['cells']} samples={macro['samples']} "
        f"baseline={macro['baseline_loss']:.6f} "
        f"candidate={macro['candidate_loss']:.6f} "
        f"change={macro['relative_change_pct']:.3f}%",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
