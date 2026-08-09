from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT,
    sha256_file,
)


LAYER_IDS = tuple(range(70))
WEAK_BANDS = {"10-19": tuple(range(10, 20)), "20-29": tuple(range(20, 30))}
FIXED_EVAL_SAMPLES = 256
PAIR_SHARED_FIELDS = (
    "pair_eval_id",
    "train_config_path",
    "train_config_sha256",
    "model_config_path",
    "model_config_sha256",
    "nano_checkpoint_path",
    "nano_checkpoint_sha256",
    "feature_seed",
)
STRATIFIED_CELLS = (
    "easy_en",
    "easy_zh",
    "medium_en",
    "medium_zh",
    "hard_en",
    "hard_zh",
    "long_zh",
)
STRATIFIED_EVAL_SAMPLES = 256 * len(STRATIFIED_CELLS)
MAX_STRATIFIED_CELL_REGRESSION_PCT = 10.0
MIN_STRATIFIED_MACRO_IMPROVED_LAYERS = 68


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


def _checkpoint_step_from_name(path: Path) -> int:
    match = re.fullmatch(r"step-([0-9]+)\.pt", path.name)
    if match is None:
        raise ValueError(
            f"Stage211 candidate checkpoint must be named step-N.pt: {path}"
        )
    return int(match.group(1))


def _validate_bound_report_file(
    report: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> Path:
    path = Path(str(report.get(path_key) or "")).expanduser().resolve()
    if (
        not path.is_file()
        or path.stat().st_size <= 0
        or sha256_file(path) != report.get(sha256_key)
    ):
        raise ValueError(f"Stage211 {label} is missing or changed: {path}")
    return path


def _validated_pair_report(
    report: dict[str, Any],
    *,
    phase: str,
    role: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT,
        "phase": phase,
        "role": role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "eval_samples": FIXED_EVAL_SAMPLES,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Stage211 {phase} {role} alignment checkpoint report binding mismatch."
        )
    logical_step = int(report.get("step", -1))
    checkpoint_step = int(report.get("checkpoint_step", -1))
    if role == "baseline":
        if logical_step != 0 or checkpoint_step < 0:
            raise ValueError(
                "Stage211 alignment baseline report must use logical step 0 "
                "and record a non-negative checkpoint step."
            )
    elif role == "candidate":
        expected_step = _checkpoint_step_from_name(checkpoint_path)
        if logical_step != expected_step or checkpoint_step != expected_step:
            raise ValueError(
                "Stage211 alignment candidate report/checkpoint step mismatch."
            )
    else:
        raise ValueError(f"Unsupported Stage211 alignment report role: {role!r}")
    pair_eval_id = str(report.get("pair_eval_id") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", pair_eval_id):
        raise ValueError(
            f"Stage211 {phase} {role} alignment report has invalid pair_eval_id."
        )
    for prefix, label in (
        ("train_config", "pair train config"),
        ("model_config", "pair model config"),
        ("nano_checkpoint", "pair Nano checkpoint"),
    ):
        _validate_bound_report_file(
            report,
            path_key=f"{prefix}_path",
            sha256_key=f"{prefix}_sha256",
            label=f"{phase} {role} {label}",
        )
    provenance = _validated_eval_provenance(
        report,
        label=f"{phase} {role}",
    )
    feature_seed = report.get("feature_seed")
    if (
        feature_seed != 0
        or provenance.get("feature_seed") != feature_seed
    ):
        raise ValueError(
            f"Stage211 {phase} {role} report lacks a matching fixed feature seed."
        )
    return provenance


def _pair_shared_binding(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in PAIR_SHARED_FIELDS}


def _validated_stratified_summary(
    path: Path,
    *,
    phase: str,
    baseline_checkpoint_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    summary = load_yaml(path)
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stratified_hidden_eval_summary",
        "phase": phase,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 stratified hidden summary binding mismatch.")
    receipt_path = Path(str(summary.get("receipt_path") or "")).resolve()
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != summary.get("receipt_sha256")
    ):
        raise ValueError("Stage211 stratified hidden receipt is missing or changed.")
    checkpoint_records = summary.get("checkpoints")
    if not isinstance(checkpoint_records, dict):
        raise ValueError("Stage211 stratified hidden summary lacks checkpoints.")
    for role, expected_path in (
        ("baseline", baseline_checkpoint_path),
        ("candidate", checkpoint_path),
    ):
        record = checkpoint_records.get(role)
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 stratified summary lacks {role} checkpoint.")
        bound_path = Path(str(record.get("path") or "")).resolve()
        if (
            bound_path != expected_path
            or not bound_path.is_file()
            or sha256_file(bound_path) != record.get("sha256")
        ):
            raise ValueError(
                f"Stage211 stratified summary {role} checkpoint mismatch."
            )
    cells = summary.get("cells")
    if not isinstance(cells, dict) or set(cells) != set(STRATIFIED_CELLS):
        raise ValueError("Stage211 stratified summary cell coverage mismatch.")
    for cell_name, cell in cells.items():
        if (
            not isinstance(cell, dict)
            or int(cell.get("samples", -1)) != FIXED_EVAL_SAMPLES
            or not all(
                0 <= int(cell.get(key, -1)) <= len(LAYER_IDS)
                for key in ("layers_loss_improved", "layers_cosine_improved")
            )
            or not all(
                isinstance(cell.get(key), (int, float))
                for key in (
                    "baseline_loss",
                    "candidate_loss",
                    "relative_change_pct",
                )
            )
        ):
            raise ValueError(f"Stage211 stratified summary cell is invalid: {cell_name}")
        if not all(
            math.isfinite(float(cell[key]))
            for key in (
                "baseline_loss",
                "candidate_loss",
                "relative_change_pct",
            )
        ):
            raise ValueError(
                f"Stage211 stratified summary cell is not finite: {cell_name}"
            )
        manifest_path = Path(str(cell.get("manifest_path") or "")).resolve()
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path) != cell.get("manifest_sha256")
        ):
            raise ValueError(
                f"Stage211 stratified summary cell manifest changed: {cell_name}"
            )
    macro = summary.get("macro")
    if (
        not isinstance(macro, dict)
        or int(macro.get("cells", -1)) != len(STRATIFIED_CELLS)
        or int(macro.get("samples", -1)) != STRATIFIED_EVAL_SAMPLES
        or not all(
            math.isfinite(float(macro.get(key, float("nan"))))
            for key in ("baseline_loss", "candidate_loss", "relative_change_pct")
        )
    ):
        raise ValueError("Stage211 stratified summary macro coverage mismatch.")
    layer_summary = summary.get("layer_summary")
    if not isinstance(layer_summary, dict):
        raise ValueError("Stage211 stratified summary lacks macro layer metrics.")
    layers = layer_summary.get("layers")
    if not isinstance(layers, dict) or set(layers) != {
        str(index) for index in LAYER_IDS
    }:
        raise ValueError("Stage211 stratified summary lacks exact macro layer coverage.")
    for layer_id, row in layers.items():
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
                f"Stage211 stratified macro layer is invalid: {layer_id}"
            )
    weak_bands = layer_summary.get("weak_bands")
    if not isinstance(weak_bands, dict) or set(weak_bands) != set(WEAK_BANDS):
        raise ValueError("Stage211 stratified summary weak-band coverage mismatch.")
    report_bindings = summary.get("reports")
    if not isinstance(report_bindings, dict) or set(report_bindings) != set(
        STRATIFIED_CELLS
    ):
        raise ValueError("Stage211 stratified summary report coverage mismatch.")
    for cell_name, roles in report_bindings.items():
        if not isinstance(roles, dict) or set(roles) != {"baseline", "candidate"}:
            raise ValueError(
                f"Stage211 stratified summary report roles mismatch: {cell_name}"
            )
        for role, binding in roles.items():
            if not isinstance(binding, dict):
                raise ValueError(
                    f"Stage211 stratified summary report binding invalid: {cell_name}/{role}"
                )
            report_path = Path(str(binding.get("path") or "")).resolve()
            if (
                not report_path.is_file()
                or sha256_file(report_path) != binding.get("sha256")
            ):
                raise ValueError(
                    f"Stage211 stratified report is missing or changed: {cell_name}/{role}"
                )
    return summary


def _stratified_gate_passed(
    summary: dict[str, Any],
    *,
    phase: str,
) -> bool:
    cells = summary["cells"]
    macro = summary["macro"]
    layer_summary = summary["layer_summary"]
    weak_bands = layer_summary["weak_bands"]
    hard_cells_pass = all(
        float(cells[cell_name]["candidate_loss"])
        < float(cells[cell_name]["baseline_loss"])
        and int(cells[cell_name]["layers_loss_improved"]) == len(LAYER_IDS)
        and int(cells[cell_name]["layers_cosine_improved"]) == len(LAYER_IDS)
        for cell_name in ("hard_en", "hard_zh")
    )
    decoder_pass = True
    if phase == "block":
        decoder = summary.get("decoder_hidden")
        decoder_pass = (
            isinstance(decoder, dict)
            and int(decoder.get("cells", -1)) == len(STRATIFIED_CELLS)
            and float(decoder["candidate_loss"]) < float(decoder["baseline_loss"])
        )
    return (
        float(macro["candidate_loss"]) < float(macro["baseline_loss"])
        and int(layer_summary["loss_improved_layers"])
        >= MIN_STRATIFIED_MACRO_IMPROVED_LAYERS
        and int(layer_summary["cosine_improved_layers"])
        >= MIN_STRATIFIED_MACRO_IMPROVED_LAYERS
        and hard_cells_pass
        and float(cells["long_zh"]["candidate_loss"])
        < float(cells["long_zh"]["baseline_loss"])
        and all(
            float(cell["relative_change_pct"])
            <= MAX_STRATIFIED_CELL_REGRESSION_PCT
            for cell in cells.values()
        )
        and all(
            float(row["candidate_loss"]) < float(row["baseline_loss"])
            and float(row["candidate_cosine"]) > float(row["baseline_cosine"])
            for row in weak_bands.values()
        )
        and decoder_pass
    )


def build_gate(
    *,
    phase: str,
    baseline_report_path: Path,
    candidate_report_path: Path,
    baseline_checkpoint_path: Path,
    checkpoint_path: Path,
    stratified_summary_path: Path | None = None,
) -> dict[str, Any]:
    baseline_report_path = baseline_report_path.resolve()
    candidate_report_path = candidate_report_path.resolve()
    baseline_checkpoint_path = baseline_checkpoint_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    for path in (baseline_checkpoint_path, checkpoint_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(str(path))
    baseline_report = load_yaml(baseline_report_path)
    candidate_report = load_yaml(candidate_report_path)
    baseline_eval_provenance = _validated_pair_report(
        baseline_report,
        phase=phase,
        role="baseline",
        checkpoint_path=baseline_checkpoint_path,
    )
    candidate_eval_provenance = _validated_pair_report(
        candidate_report,
        phase=phase,
        role="candidate",
        checkpoint_path=checkpoint_path,
    )
    if _pair_shared_binding(baseline_report) != _pair_shared_binding(
        candidate_report
    ):
        raise ValueError(
            "Stage211 baseline and candidate hidden reports were not produced "
            "by the same paired evaluation."
        )
    if baseline_eval_provenance != candidate_eval_provenance:
        raise ValueError(
            "Stage211 baseline and candidate hidden reports do not bind the "
            "same fixed-eval provenance."
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

    legacy_gate_passed = (
        candidate_eval_loss < baseline_eval_loss
        and float(summary["candidate_mean_loss"]) < float(summary["baseline_mean_loss"])
        and float(summary["candidate_mean_cosine"]) > float(summary["baseline_mean_cosine"])
        and int(summary["loss_improved_layers"]) == len(LAYER_IDS)
        and int(summary["cosine_improved_layers"]) == len(LAYER_IDS)
        and weak_bands_pass
        and decoder_gate_passed
    )
    stratified_summary = None
    stratified_gate_passed = None
    if stratified_summary_path is not None:
        stratified_summary_path = stratified_summary_path.expanduser().resolve()
        stratified_summary = _validated_stratified_summary(
            stratified_summary_path,
            phase=phase,
            baseline_checkpoint_path=baseline_checkpoint_path,
            checkpoint_path=checkpoint_path,
        )
        stratified_gate_passed = _stratified_gate_passed(
            stratified_summary,
            phase=phase,
        )
    gate_passed = (
        bool(stratified_gate_passed)
        if stratified_gate_passed is not None
        else legacy_gate_passed
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "hidden_alignment_gate",
        "phase": phase,
        "baseline_checkpoint_path": str(baseline_checkpoint_path),
        "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "gate_passed": gate_passed,
        "legacy_gate_passed": legacy_gate_passed,
        "stratified_gate_passed": stratified_gate_passed,
        "stratified_summary_path": (
            str(stratified_summary_path)
            if stratified_summary_path is not None
            else None
        ),
        "stratified_summary_sha256": (
            sha256_file(stratified_summary_path)
            if stratified_summary_path is not None
            else None
        ),
        "stratified_summary": stratified_summary,
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
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stratified-summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = build_gate(
        phase=str(args.phase),
        baseline_report_path=args.baseline_report,
        candidate_report_path=args.candidate_report,
        baseline_checkpoint_path=args.baseline_checkpoint,
        checkpoint_path=args.checkpoint,
        stratified_summary_path=args.stratified_summary,
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
