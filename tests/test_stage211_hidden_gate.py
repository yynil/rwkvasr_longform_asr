from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
hidden_gate = importlib.import_module("scripts.create_stage211_hidden_alignment_gate")


def _eval_provenance(
    *,
    manifest: Path,
    part: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": 256,
        "feature_seed": 0,
        "bucket_manifest_path": str(manifest.resolve()),
        "bucket_manifest_sha256": sha256_file(manifest),
        "split_samples": 256,
        "parts": [
            {
                "path": str(part.resolve()),
                "sha256": sha256_file(part),
                "num_samples": 256,
            }
        ],
    }


def _write_report(
    path: Path,
    *,
    manifest: Path,
    part: Path,
    role: str,
    checkpoint: Path,
    loss: float,
    cosine: float,
    phase: str = "mixer",
    component_metrics: dict[str, tuple[float, float]] | None = None,
    decoder_loss: float | None = None,
) -> Path:
    train_config = path.parent / "pair-train.yaml"
    model_config = path.parent / "pair-model.yaml"
    nano_checkpoint = path.parent / "pair-nano.pt"
    if not train_config.exists():
        train_config.write_text("{}\n", encoding="utf-8")
        model_config.write_text("{}\n", encoding="utf-8")
        nano_checkpoint.write_bytes(b"nano")
    checkpoint_step = (
        int(checkpoint.stem.removeprefix("step-"))
        if role == "candidate"
        else 17
    )
    component_metrics = component_metrics or {phase: (loss, cosine)}
    layer_components = {
        component_name: {
            str(layer_id): {
                "loss": component_loss,
                "cosine": component_cosine,
                "rms_ratio": 0.9,
            }
            for layer_id in range(70)
        }
        for component_name, (component_loss, component_cosine) in component_metrics.items()
    }
    save_yaml(
        path,
        {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "alignment_checkpoint_eval",
            "phase": phase,
            "role": role,
            "pair_eval_id": "a" * 64,
            "step": 0 if role == "baseline" else checkpoint_step,
            "checkpoint_step": checkpoint_step,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "train_config_path": str(train_config.resolve()),
            "train_config_sha256": sha256_file(train_config),
            "model_config_path": str(model_config.resolve()),
            "model_config_sha256": sha256_file(model_config),
            "nano_checkpoint_path": str(nano_checkpoint.resolve()),
            "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
            "feature_seed": 0,
            "eval_loss": loss,
            "eval_samples": 256,
            "eval_provenance": _eval_provenance(
                manifest=manifest,
                part=part,
            ),
            "layer_component_metrics": layer_components,
            "decoder_hidden_metrics": (
                {"loss": decoder_loss} if decoder_loss is not None else {}
            ),
        },
    )
    return path


def _write_stratified_summary(
    root: Path,
    *,
    baseline_checkpoint: Path,
    checkpoint: Path,
) -> Path:
    receipt = root / "stratified-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    cells = {}
    reports = {}
    for cell_name in hidden_gate.STRATIFIED_CELLS:
        manifest = root / f"manifest_{cell_name}.json"
        manifest.write_text("{}\n", encoding="utf-8")
        cells[cell_name] = {
            "samples": 256,
            "baseline_loss": 1.0,
            "candidate_loss": 0.8,
            "relative_change_pct": -20.0,
            "layers_loss_improved": 70,
            "layers_cosine_improved": 70,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
        }
        reports[cell_name] = {}
        for role in ("baseline", "candidate"):
            report = root / f"{cell_name}_{role}.json"
            report.write_text(f'{{"role": "{role}"}}\n', encoding="utf-8")
            reports[cell_name][role] = {
                "path": str(report.resolve()),
                "sha256": sha256_file(report),
            }
    layers = {
        str(layer_id): {
            "baseline_loss": 1.0,
            "candidate_loss": 0.8,
            "baseline_cosine": 0.5,
            "candidate_cosine": 0.6,
            "baseline_rms_ratio": 1.0,
            "candidate_rms_ratio": 1.0,
        }
        for layer_id in hidden_gate.LAYER_IDS
    }
    weak_bands = {
        band: {
            "baseline_loss": 1.0,
            "candidate_loss": 0.8,
            "baseline_cosine": 0.5,
            "candidate_cosine": 0.6,
        }
        for band in hidden_gate.WEAK_BANDS
    }
    component_cells = {
        cell_name: {
            "baseline_loss": 1.0,
            "candidate_loss": 0.8,
            "relative_change_pct": -20.0,
            "baseline_cosine": 0.5,
            "candidate_cosine": 0.6,
            "layers_loss_improved": 70,
            "layers_cosine_improved": 70,
        }
        for cell_name in hidden_gate.STRATIFIED_CELLS
    }
    summary = root / "stratified-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "stratified_hidden_eval_summary",
                "phase": "mixer",
                "receipt_path": str(receipt.resolve()),
                "receipt_sha256": sha256_file(receipt),
                "checkpoints": {
                    "baseline": {
                        "path": str(baseline_checkpoint.resolve()),
                        "sha256": sha256_file(baseline_checkpoint),
                    },
                    "candidate": {
                        "path": str(checkpoint.resolve()),
                        "sha256": sha256_file(checkpoint),
                    },
                },
                "cells": cells,
                "macro": {
                    "cells": 7,
                    "samples": 1792,
                    "baseline_loss": 1.0,
                    "candidate_loss": 0.8,
                    "relative_change_pct": -20.0,
                },
                "layer_summary": {
                    "loss_improved_layers": 70,
                    "cosine_improved_layers": 70,
                    "weak_bands": weak_bands,
                    "layers": layers,
                },
                "component_summaries": {
                    "mixer": {
                        "baseline_mean_loss": 1.0,
                        "candidate_mean_loss": 0.8,
                        "baseline_mean_cosine": 0.5,
                        "candidate_mean_cosine": 0.6,
                        "loss_improved_layers": 70,
                        "cosine_improved_layers": 70,
                        "weak_bands": weak_bands,
                        "cells": component_cells,
                        "layers": layers,
                    }
                },
                "decoder_hidden": None,
                "reports": reports,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def test_stage211_hidden_gate_requires_identical_bound_eval_parts(
    tmp_path: Path,
) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    baseline_manifest = tmp_path / "easy_manifest.json"
    baseline_manifest.write_text("{}\n", encoding="utf-8")
    baseline_checkpoint = tmp_path / "init.pt"
    baseline_checkpoint.write_bytes(b"initial")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        manifest=baseline_manifest,
        part=eval_part,
        role="baseline",
        checkpoint=baseline_checkpoint,
        loss=1.0,
        cosine=0.5,
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        manifest=baseline_manifest,
        part=eval_part,
        role="candidate",
        checkpoint=checkpoint,
        loss=0.8,
        cosine=0.6,
    )

    report = hidden_gate.build_gate(
        phase="mixer",
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=checkpoint,
    )

    assert report["gate_passed"] is True
    assert (
        report["baseline_eval_provenance"]["parts"][0]["sha256"]
        == report["candidate_eval_provenance"]["parts"][0]["sha256"]
    )

    different_part = tmp_path / "different_eval.jsonl"
    different_part.write_text('{"utt_id": "different"}\n', encoding="utf-8")
    mismatched_candidate = _write_report(
        tmp_path / "mismatched_candidate.yaml",
        manifest=baseline_manifest,
        part=different_part,
        role="candidate",
        checkpoint=checkpoint,
        loss=0.8,
        cosine=0.6,
    )
    with pytest.raises(ValueError, match="same fixed-eval provenance"):
        hidden_gate.build_gate(
            phase="mixer",
            baseline_report_path=baseline,
            candidate_report_path=mismatched_candidate,
            baseline_checkpoint_path=baseline_checkpoint,
            checkpoint_path=checkpoint,
        )


def test_stage211_block_gate_requires_mixer_ffn_and_block_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    baseline_checkpoint = tmp_path / "init.pt"
    candidate_checkpoint = tmp_path / "step-105.pt"
    baseline_checkpoint.write_bytes(b"initial")
    candidate_checkpoint.write_bytes(b"candidate")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        manifest=manifest,
        part=eval_part,
        role="baseline",
        checkpoint=baseline_checkpoint,
        loss=1.0,
        cosine=0.5,
        phase="block",
        component_metrics={
            "mixer": (1.0, 0.5),
            "ffn": (1.0, 0.5),
            "block": (1.0, 0.5),
        },
        decoder_loss=1.0,
    )
    regressed_ffn = _write_report(
        tmp_path / "regressed_ffn.yaml",
        manifest=manifest,
        part=eval_part,
        role="candidate",
        checkpoint=candidate_checkpoint,
        loss=0.8,
        cosine=0.6,
        phase="block",
        component_metrics={
            "mixer": (0.8, 0.6),
            "ffn": (1.1, 0.4),
            "block": (0.8, 0.6),
        },
        decoder_loss=0.8,
    )

    rejected = hidden_gate.build_gate(
        phase="block",
        baseline_report_path=baseline,
        candidate_report_path=regressed_ffn,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=candidate_checkpoint,
    )

    assert rejected["component_gate_passed"] is False
    assert rejected["gate_passed"] is False
    assert rejected["component_summaries"]["ffn"]["candidate_mean_loss"] == pytest.approx(
        1.1
    )

    stratified_summary = tmp_path / "stratified-summary.json"
    stratified_summary.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        hidden_gate,
        "_validated_stratified_summary",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        hidden_gate,
        "_stratified_gate_passed",
        lambda *args, **kwargs: True,
    )
    still_rejected = hidden_gate.build_gate(
        phase="block",
        baseline_report_path=baseline,
        candidate_report_path=regressed_ffn,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=candidate_checkpoint,
        stratified_summary_path=stratified_summary,
    )
    assert still_rejected["stratified_gate_passed"] is True
    assert still_rejected["gate_passed"] is False

    aligned = _write_report(
        tmp_path / "aligned.yaml",
        manifest=manifest,
        part=eval_part,
        role="candidate",
        checkpoint=candidate_checkpoint,
        loss=0.8,
        cosine=0.6,
        phase="block",
        component_metrics={
            "mixer": (0.8, 0.6),
            "ffn": (0.8, 0.6),
            "block": (0.8, 0.6),
        },
        decoder_loss=0.8,
    )
    admitted = hidden_gate.build_gate(
        phase="block",
        baseline_report_path=baseline,
        candidate_report_path=aligned,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=candidate_checkpoint,
    )

    assert admitted["component_gate_passed"] is True
    assert admitted["gate_passed"] is True


def test_stage211_hidden_gate_rejects_mutated_eval_part(tmp_path: Path) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    baseline_checkpoint = tmp_path / "init.pt"
    baseline_checkpoint.write_bytes(b"initial")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        manifest=manifest,
        part=eval_part,
        role="baseline",
        checkpoint=baseline_checkpoint,
        loss=1.0,
        cosine=0.5,
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        manifest=manifest,
        part=eval_part,
        role="candidate",
        checkpoint=checkpoint,
        loss=0.8,
        cosine=0.6,
    )
    eval_part.write_text('{"utt_id": "mutated"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing or changed"):
        hidden_gate.build_gate(
            phase="mixer",
            baseline_report_path=baseline,
            candidate_report_path=candidate,
            baseline_checkpoint_path=baseline_checkpoint,
            checkpoint_path=checkpoint,
        )


def test_stage211_hidden_gate_uses_bound_stratified_result_for_promotion(
    tmp_path: Path,
) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    baseline_checkpoint = tmp_path / "init.pt"
    baseline_checkpoint.write_bytes(b"initial")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        manifest=manifest,
        part=eval_part,
        role="baseline",
        checkpoint=baseline_checkpoint,
        loss=1.0,
        cosine=0.6,
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        manifest=manifest,
        part=eval_part,
        role="candidate",
        checkpoint=checkpoint,
        loss=1.1,
        cosine=0.5,
    )
    stratified_summary = _write_stratified_summary(
        tmp_path,
        baseline_checkpoint=baseline_checkpoint,
        checkpoint=checkpoint,
    )

    report = hidden_gate.build_gate(
        phase="mixer",
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=checkpoint,
        stratified_summary_path=stratified_summary,
    )

    assert report["legacy_gate_passed"] is False
    assert report["stratified_gate_passed"] is True
    assert report["gate_passed"] is True
    assert report["stratified_summary_sha256"] == sha256_file(
        stratified_summary
    )

    bound_report = tmp_path / "easy_en_candidate.json"
    bound_report.write_text('{"mutated": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="report is missing or changed"):
        hidden_gate.build_gate(
            phase="mixer",
            baseline_report_path=baseline,
            candidate_report_path=candidate,
            baseline_checkpoint_path=baseline_checkpoint,
            checkpoint_path=checkpoint,
            stratified_summary_path=stratified_summary,
        )
