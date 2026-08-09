from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.create_stage211_logits_alignment_gate import (
        IDENTICAL_TEACHER_METRICS,
        REQUIRED_METRICS,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_logits_alignment_gate import (
        IDENTICAL_TEACHER_METRICS,
        REQUIRED_METRICS,
    )


ROLES = ("baseline", "candidate")
SUM_METRICS = {
    "selected_frames",
    "all_frames",
    "teacher_tokens",
    "student_tokens",
    "matched_utterances",
    "missing_utterances",
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


def _validated_metrics(report: dict[str, Any], *, report_path: Path) -> dict[str, float]:
    raw = report.get("logit_metrics")
    if not isinstance(raw, dict):
        raise ValueError(f"Stage211 sidecar lacks logit metrics: {report_path}")
    metrics = {name: float(raw.get(name, float("nan"))) for name in REQUIRED_METRICS}
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(
            f"Stage211 sidecar has non-finite logit metrics {invalid}: {report_path}"
        )
    return metrics


def _validate_report(
    *,
    report: dict[str, Any],
    report_path: Path,
    role: str,
    samples: int,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, float]:
    if (
        report.get("pipeline") != "stage211"
        or report.get("artifact") != "alignment_checkpoint_eval"
        or report.get("phase") != "logits"
        or report.get("role") != role
        or int(report.get("eval_samples", -1)) != samples
    ):
        raise ValueError(f"Stage211 stratified logits report binding mismatch: {report_path}")
    provenance = report.get("eval_provenance")
    if (
        not isinstance(provenance, dict)
        or Path(str(provenance.get("bucket_manifest_path") or "")).resolve()
        != manifest_path
        or provenance.get("bucket_manifest_sha256") != manifest_sha256
        or int(provenance.get("split_samples", -1)) != samples
    ):
        raise ValueError(
            f"Stage211 stratified logits manifest provenance mismatch: {report_path}"
        )
    metrics = _validated_metrics(report, report_path=report_path)
    if (
        int(round(metrics["matched_utterances"])) != samples
        or metrics["missing_utterances"] != 0.0
        or metrics["mean_frame_delta"] != 0.0
    ):
        raise ValueError(
            f"Stage211 stratified logits report lacks exact coverage: {report_path}"
        )
    return metrics


def _aggregate_metrics(
    values: list[dict[str, float]],
) -> dict[str, float]:
    return {
        name: (
            sum(row[name] for row in values)
            if name in SUM_METRICS
            else sum(row[name] for row in values) / len(values)
        )
        for name in REQUIRED_METRICS
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
    if (
        receipt.get("pipeline") != "stage211"
        or receipt.get("artifact") != "stratified_hidden_eval_manifest"
    ):
        raise ValueError("Unexpected Stage211 stratified eval receipt artifact.")
    cells = receipt.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError("Stage211 stratified eval receipt has no cells.")

    checkpoint_bindings: dict[str, set[tuple[str, str]]] = {
        role: set() for role in ROLES
    }
    metrics_by_role: dict[str, list[dict[str, float]]] = {
        role: [] for role in ROLES
    }
    cell_results: dict[str, dict[str, Any]] = {}
    report_bindings: dict[str, dict[str, Any]] = {}
    for cell_name, cell in sorted(cells.items()):
        if not isinstance(cell, dict):
            raise ValueError(f"Invalid Stage211 stratified logits cell: {cell_name}")
        samples = int(cell.get("samples", -1))
        manifest_path = Path(str(cell.get("manifest_path") or "")).resolve()
        manifest_sha256 = str(cell.get("manifest_sha256") or "")
        if (
            samples <= 0
            or not manifest_path.is_file()
            or sha256_file(manifest_path) != manifest_sha256
        ):
            raise ValueError(
                f"Stage211 stratified logits manifest mismatch: {cell_name}"
            )
        cell_metrics: dict[str, dict[str, float]] = {}
        cell_reports: dict[str, dict[str, Any]] = {}
        report_bindings[cell_name] = {}
        pair_ids: set[str] = set()
        for role in ROLES:
            report_path = (eval_dir / f"{cell_name}_{role}.json").resolve()
            report = _load_json(
                report_path,
                label=f"Stage211 {cell_name} {role} logits report",
            )
            cell_reports[role] = report
            pair_ids.add(str(report.get("pair_eval_id") or ""))
            metrics = _validate_report(
                report=report,
                report_path=report_path,
                role=role,
                samples=samples,
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
            )
            cell_metrics[role] = metrics
            metrics_by_role[role].append(metrics)
            checkpoint_bindings[role].add(
                (
                    str(report.get("checkpoint_path") or ""),
                    str(report.get("checkpoint_sha256") or ""),
                )
            )
            report_bindings[cell_name][role] = {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            }
        if len(pair_ids) != 1 or "" in pair_ids:
            raise ValueError(f"Stage211 stratified logits pair mismatch: {cell_name}")
        for name in IDENTICAL_TEACHER_METRICS:
            if not math.isclose(
                cell_metrics["baseline"][name],
                cell_metrics["candidate"][name],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"Stage211 stratified logits teacher metric differs: {cell_name}/{name}"
                )
        baseline = cell_metrics["baseline"]
        candidate = cell_metrics["candidate"]
        cell_results[cell_name] = {
            "samples": samples,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "baseline_metrics": baseline,
            "candidate_metrics": candidate,
            "full_kl_relative_reduction": (
                baseline["full_kl"] - candidate["full_kl"]
            )
            / max(abs(baseline["full_kl"]), 1.0e-12),
            "conditional_nonblank_kl_relative_reduction": (
                baseline["conditional_nonblank_kl"]
                - candidate["conditional_nonblank_kl"]
            )
            / max(abs(baseline["conditional_nonblank_kl"]), 1.0e-12),
            "ctc_token_error_absolute_change": (
                candidate["ctc_token_error_rate"]
                - baseline["ctc_token_error_rate"]
            ),
        }

    checkpoint_records = {}
    for role, bindings in checkpoint_bindings.items():
        if len(bindings) != 1:
            raise ValueError(
                f"Stage211 stratified logits {role} checkpoint mismatch: {bindings}"
            )
        path_value, digest = next(iter(bindings))
        checkpoint_path = Path(path_value).expanduser().resolve()
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != digest:
            raise ValueError(
                f"Stage211 stratified logits {role} checkpoint is missing or changed."
            )
        checkpoint_records[role] = {
            "path": str(checkpoint_path),
            "sha256": digest,
        }
    macro_baseline = _aggregate_metrics(metrics_by_role["baseline"])
    macro_candidate = _aggregate_metrics(metrics_by_role["candidate"])
    summary = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stratified_logits_eval_summary",
        "phase": "logits",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "checkpoints": checkpoint_records,
        "cells": cell_results,
        "macro": {
            "cells": len(cell_results),
            "samples": sum(int(cell["samples"]) for cell in cell_results.values()),
            "baseline_metrics": macro_baseline,
            "candidate_metrics": macro_candidate,
            "full_kl_relative_reduction": (
                macro_baseline["full_kl"] - macro_candidate["full_kl"]
            )
            / max(abs(macro_baseline["full_kl"]), 1.0e-12),
            "conditional_nonblank_kl_relative_reduction": (
                macro_baseline["conditional_nonblank_kl"]
                - macro_candidate["conditional_nonblank_kl"]
            )
            / max(abs(macro_baseline["conditional_nonblank_kl"]), 1.0e-12),
        },
        "reports": report_bindings,
    }
    _write_immutable(output_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize Stage211 stratified logits pair reports."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        receipt_path=args.receipt,
        eval_dir=args.eval_dir,
        output_path=args.output,
    )
    macro = summary["macro"]
    print(
        "stage211_stratified_logits_summary "
        f"samples={macro['samples']} "
        f"full_kl_reduction={macro['full_kl_relative_reduction']:.6f} "
        "conditional_kl_reduction="
        f"{macro['conditional_nonblank_kl_relative_reduction']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
