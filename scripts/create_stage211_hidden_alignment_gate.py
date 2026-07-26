from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import sha256_file


LAYER_IDS = tuple(range(70))
WEAK_BANDS = {"10-19": tuple(range(10, 20)), "20-29": tuple(range(20, 30))}
FIXED_EVAL_SAMPLES = 256


def _layer_components(report: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    value = report.get("layer_components") or report.get("layer_component_metrics") or {}
    return value if isinstance(value, dict) else {}


def _phase_layers(report: dict[str, Any], *, phase: str) -> dict[str, dict[str, float]]:
    components = _layer_components(report)
    component_name = "mixer" if phase == "mixer" else "block"
    layers = components.get(component_name)
    if not isinstance(layers, dict):
        layers = report.get("layers") or report.get("layer_metrics")
    if not isinstance(layers, dict) or set(layers) != {str(index) for index in LAYER_IDS}:
        raise ValueError(f"Stage211 {phase} hidden report does not contain exactly 70 layers.")
    return layers


def _mean(
    layers: dict[str, dict[str, float]],
    key: str,
    layer_ids: tuple[int, ...],
) -> float:
    return sum(float(layers[str(layer_id)][key]) for layer_id in layer_ids) / len(layer_ids)


def _summary(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, Any]:
    baseline_loss = _mean(baseline, "loss", LAYER_IDS)
    candidate_loss = _mean(candidate, "loss", LAYER_IDS)
    result: dict[str, Any] = {
        "baseline_mean_loss": baseline_loss,
        "candidate_mean_loss": candidate_loss,
        "relative_loss_reduction": (baseline_loss - candidate_loss)
        / max(
            baseline_loss,
            1.0e-12,
        ),
        "baseline_mean_cosine": _mean(baseline, "cosine", LAYER_IDS),
        "candidate_mean_cosine": _mean(candidate, "cosine", LAYER_IDS),
        "baseline_mean_rms_ratio": _mean(baseline, "rms_ratio", LAYER_IDS),
        "candidate_mean_rms_ratio": _mean(candidate, "rms_ratio", LAYER_IDS),
        "loss_improved_layers": sum(
            float(candidate[str(index)]["loss"]) < float(baseline[str(index)]["loss"])
            for index in LAYER_IDS
        ),
        "cosine_improved_layers": sum(
            float(candidate[str(index)]["cosine"]) > float(baseline[str(index)]["cosine"])
            for index in LAYER_IDS
        ),
        "weak_bands": {},
    }
    for name, layer_ids in WEAK_BANDS.items():
        weak_baseline_loss = _mean(baseline, "loss", layer_ids)
        weak_candidate_loss = _mean(candidate, "loss", layer_ids)
        result["weak_bands"][name] = {
            "baseline_loss": weak_baseline_loss,
            "candidate_loss": weak_candidate_loss,
            "relative_loss_reduction": (
                (weak_baseline_loss - weak_candidate_loss) / max(weak_baseline_loss, 1.0e-12)
            ),
            "baseline_cosine": _mean(baseline, "cosine", layer_ids),
            "candidate_cosine": _mean(candidate, "cosine", layer_ids),
        }
    return result


def _validated_eval_provenance(
    report: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    provenance = report.get("eval_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Stage211 {label} report lacks eval provenance.")
    expected_fields = {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": FIXED_EVAL_SAMPLES,
        "split_samples": FIXED_EVAL_SAMPLES,
    }
    for key, expected in expected_fields.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"Stage211 {label} eval provenance {key} mismatch: "
                f"expected={expected!r} actual={provenance.get(key)!r}"
            )
    manifest_path = Path(str(provenance.get("bucket_manifest_path") or "")).resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != provenance.get(
        "bucket_manifest_sha256"
    ):
        raise ValueError(f"Stage211 {label} eval manifest is missing or changed.")
    raw_parts = provenance.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError(f"Stage211 {label} eval provenance has no bound parts.")
    parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            raise ValueError(f"Stage211 {label} eval part record is invalid.")
        path = Path(str(raw_part.get("path") or "")).resolve()
        sha256 = str(raw_part.get("sha256") or "")
        num_samples = int(raw_part.get("num_samples", -1))
        if not path.is_file() or sha256_file(path) != sha256 or num_samples <= 0:
            raise ValueError(f"Stage211 {label} eval part is missing or changed: {path}")
        parts.append(
            {
                "path": str(path),
                "sha256": sha256,
                "num_samples": num_samples,
            }
        )
    if sum(int(part["num_samples"]) for part in parts) != FIXED_EVAL_SAMPLES:
        raise ValueError(f"Stage211 {label} eval-part sample count mismatch.")
    return {
        **provenance,
        "bucket_manifest_path": str(manifest_path),
        "parts": parts,
    }


def _eval_part_fingerprint(provenance: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple((str(part["sha256"]), int(part["num_samples"])) for part in provenance["parts"])


def build_gate(
    *,
    phase: str,
    baseline_report_path: Path,
    candidate_report_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    baseline_report_path = baseline_report_path.resolve()
    candidate_report_path = candidate_report_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    baseline_report = load_yaml(baseline_report_path)
    candidate_report = load_yaml(candidate_report_path)
    baseline_eval_provenance = _validated_eval_provenance(
        baseline_report,
        label="baseline",
    )
    candidate_eval_provenance = _validated_eval_provenance(
        candidate_report,
        label="candidate",
    )
    if _eval_part_fingerprint(baseline_eval_provenance) != _eval_part_fingerprint(
        candidate_eval_provenance
    ):
        raise ValueError(
            "Stage211 baseline and candidate hidden reports use different eval samples."
        )
    baseline_layers = _phase_layers(baseline_report, phase=phase)
    candidate_layers = _phase_layers(candidate_report, phase=phase)
    summary = _summary(baseline_layers, candidate_layers)
    baseline_eval_loss = float(baseline_report["eval_loss"])
    candidate_eval_loss = float(candidate_report["eval_loss"])
    weak_bands_pass = all(
        float(row["relative_loss_reduction"]) > 0.0
        and float(row["candidate_cosine"]) > float(row["baseline_cosine"])
        for row in summary["weak_bands"].values()
    )
    decoder_hidden: dict[str, float] | None = None
    decoder_gate_passed = True
    if phase == "block":
        baseline_decoder = baseline_report.get("decoder_hidden_metrics") or {}
        candidate_decoder = candidate_report.get("decoder_hidden_metrics") or {}
        if "loss" not in baseline_decoder or "loss" not in candidate_decoder:
            raise ValueError("Stage211 block gate requires decoder-hidden loss in both reports.")
        decoder_hidden = {
            "baseline_loss": float(baseline_decoder["loss"]),
            "candidate_loss": float(candidate_decoder["loss"]),
        }
        decoder_gate_passed = decoder_hidden["candidate_loss"] < decoder_hidden["baseline_loss"]

    gate_passed = (
        candidate_eval_loss < baseline_eval_loss
        and float(summary["candidate_mean_loss"]) < float(summary["baseline_mean_loss"])
        and float(summary["candidate_mean_cosine"]) > float(summary["baseline_mean_cosine"])
        and int(summary["loss_improved_layers"]) == len(LAYER_IDS)
        and int(summary["cosine_improved_layers"]) == len(LAYER_IDS)
        and weak_bands_pass
        and decoder_gate_passed
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "hidden_alignment_gate",
        "phase": phase,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "gate_passed": gate_passed,
        "baseline_report_path": str(baseline_report_path),
        "baseline_report_sha256": sha256_file(baseline_report_path),
        "candidate_report_path": str(candidate_report_path),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "baseline_eval_provenance": baseline_eval_provenance,
        "candidate_eval_provenance": candidate_eval_provenance,
        "baseline_eval_loss": baseline_eval_loss,
        "candidate_eval_loss": candidate_eval_loss,
        "layer_summary": summary,
        "decoder_hidden": decoder_hidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an independently reproducible Stage211 mixer/block hidden gate."
    )
    parser.add_argument("--phase", choices=("mixer", "block"), required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_gate(
        phase=str(args.phase),
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        checkpoint_path=args.checkpoint,
    )
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output_path.is_file() and output_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different hidden gate: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"hidden_gate={output_path} phase={args.phase} "
        f"gate_passed={str(report['gate_passed']).lower()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
