from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from rwkvasr.eval.stage211_gate import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
logits_gate = importlib.import_module(
    "scripts.create_stage211_logits_alignment_gate"
)
summarizer = importlib.import_module(
    "scripts.summarize_stage211_stratified_logits_eval"
)


def _baseline_metrics() -> dict[str, float]:
    return {
        "full_kl": 1.0,
        "conditional_nonblank_kl": 2.0,
        "conditional_nonblank_hard_ce": 3.0,
        "blank_binary_kl": 0.20,
        "selected_top1_agreement": 0.50,
        "all_top1_agreement": 0.50,
        "active_top1_agreement": 0.40,
        "blank_prob_mae": 0.10,
        "teacher_nonblank_rate": 0.20,
        "student_nonblank_rate": 0.16,
        "nonblank_rate_ratio": 0.80,
        "teacher_nonblank_to_blank_rate": 0.30,
        "teacher_blank_to_nonblank_rate": 0.10,
        "ctc_token_error_rate": 0.60,
        "ctc_token_insertion_rate": 0.10,
        "ctc_token_deletion_rate": 0.40,
        "ctc_token_substitution_rate": 0.10,
        "collapsed_length_ratio": 0.80,
        "sequence_exact_rate": 0.05,
        "mean_frame_delta": 0.0,
        "selected_frames": 1_000.0,
        "all_frames": 1_000.0,
        "teacher_tokens": 500.0,
        "student_tokens": 400.0,
        "matched_utterances": 256.0,
        "missing_utterances": 0.0,
    }


def _candidate_metrics() -> dict[str, float]:
    return {
        **_baseline_metrics(),
        "full_kl": 0.80,
        "conditional_nonblank_kl": 1.50,
        "conditional_nonblank_hard_ce": 2.50,
        "blank_binary_kl": 0.15,
        "selected_top1_agreement": 0.65,
        "all_top1_agreement": 0.65,
        "active_top1_agreement": 0.60,
        "blank_prob_mae": 0.08,
        "student_nonblank_rate": 0.20,
        "nonblank_rate_ratio": 1.0,
        "teacher_nonblank_to_blank_rate": 0.10,
        "teacher_blank_to_nonblank_rate": 0.05,
        "ctc_token_error_rate": 0.30,
        "ctc_token_insertion_rate": 0.05,
        "ctc_token_deletion_rate": 0.20,
        "ctc_token_substitution_rate": 0.05,
        "collapsed_length_ratio": 1.0,
        "sequence_exact_rate": 0.20,
        "student_tokens": 500.0,
    }


def _write_report(
    path: Path,
    *,
    role: str,
    checkpoint: Path,
    manifest: Path,
    part: Path,
    metrics: dict[str, float],
    pair_id: str,
    component_overrides: dict[str, tuple[float, float]] | None = None,
) -> Path:
    train_config = path.parent / "pair-train.yaml"
    model_config = path.parent / "pair-model.yaml"
    nano_checkpoint = path.parent / "pair-nano.pt"
    if not train_config.exists():
        train_config.write_text("{}\n", encoding="utf-8")
        model_config.write_text("{}\n", encoding="utf-8")
        nano_checkpoint.write_bytes(b"nano")
    step = 0 if role == "baseline" else 105
    checkpoint_step = 17 if role == "baseline" else 105
    component_metrics = {
        "mixer": (1.0, 0.8),
        "ffn": (1.0, 0.8),
        "block": (1.0, 0.8),
    }
    component_metrics.update(component_overrides or {})
    layer_components = {
        component_name: {
            str(layer_id): {
                "loss": component_loss,
                "cosine": component_cosine,
                "rms_ratio": 1.0,
            }
            for layer_id in range(70)
        }
        for component_name, (
            component_loss,
            component_cosine,
        ) in component_metrics.items()
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "alignment_checkpoint_eval",
        "phase": "logits",
        "role": role,
        "pair_eval_id": pair_id,
        "step": step,
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
        "eval_loss": metrics["full_kl"],
        "eval_samples": 256,
        "eval_provenance": {
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
        },
        "logit_metrics": metrics,
        "layer_components": layer_components,
        "decoder_hidden_metrics": {"loss": 1.0},
    }
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return path


def test_stage211_stratified_logits_summary_governs_gate(tmp_path: Path) -> None:
    baseline_checkpoint = tmp_path / "init.pt"
    baseline_checkpoint.write_bytes(b"initial")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"candidate")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    cells = {}
    for index, cell_name in enumerate(logits_gate.STRATIFIED_CELLS):
        manifest = tmp_path / f"manifest_{cell_name}.json"
        part = tmp_path / f"part_{cell_name}.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")
        part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
        cells[cell_name] = {
            "samples": 256,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
        }
        pair_id = f"{index + 1:064x}"
        _write_report(
            eval_dir / f"{cell_name}_baseline.json",
            role="baseline",
            checkpoint=baseline_checkpoint,
            manifest=manifest,
            part=part,
            metrics=_baseline_metrics(),
            pair_id=pair_id,
        )
        _write_report(
            eval_dir / f"{cell_name}_candidate.json",
            role="candidate",
            checkpoint=checkpoint,
            manifest=manifest,
            part=part,
            metrics=_candidate_metrics(),
            pair_id=pair_id,
        )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "pipeline": "stage211",
                "artifact": "stratified_hidden_eval_manifest",
                "cells": cells,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary = summarizer.summarize(
        receipt_path=receipt,
        eval_dir=eval_dir,
        output_path=summary_path,
    )

    assert summary["macro"]["samples"] == 1792
    assert summary["macro"]["full_kl_relative_reduction"] == pytest.approx(
        0.20
    )
    assert summary["macro"]["baseline_metrics"]["matched_utterances"] == 1792

    legacy_manifest = tmp_path / "legacy-manifest.json"
    legacy_part = tmp_path / "legacy-part.jsonl"
    legacy_manifest.write_text("{}\n", encoding="utf-8")
    legacy_part.write_text('{"utt_id": "legacy"}\n', encoding="utf-8")
    weak_metrics = _candidate_metrics()
    weak_metrics["full_kl"] = 0.97
    legacy_baseline = _write_report(
        tmp_path / "legacy-baseline.json",
        role="baseline",
        checkpoint=baseline_checkpoint,
        manifest=legacy_manifest,
        part=legacy_part,
        metrics=_baseline_metrics(),
        pair_id="a" * 64,
    )
    legacy_candidate = _write_report(
        tmp_path / "legacy-candidate.json",
        role="candidate",
        checkpoint=checkpoint,
        manifest=legacy_manifest,
        part=legacy_part,
        metrics=weak_metrics,
        pair_id="a" * 64,
    )
    gate = logits_gate.build_gate(
        baseline_report_path=legacy_baseline,
        candidate_report_path=legacy_candidate,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=checkpoint,
        stratified_summary_path=summary_path,
    )

    assert gate["legacy_gate_passed"] is False
    assert gate["stratified_gate_passed"] is True
    assert gate["gate_passed"] is True
    assert all(gate["stratified_checks"].values())

    regressed_candidate = _write_report(
        tmp_path / "legacy-regressed-candidate.json",
        role="candidate",
        checkpoint=checkpoint,
        manifest=legacy_manifest,
        part=legacy_part,
        metrics=weak_metrics,
        pair_id="a" * 64,
        component_overrides={"ffn": (1.2, 0.8)},
    )
    rejected = logits_gate.build_gate(
        baseline_report_path=legacy_baseline,
        candidate_report_path=regressed_candidate,
        baseline_checkpoint_path=baseline_checkpoint,
        checkpoint_path=checkpoint,
        stratified_summary_path=summary_path,
    )

    assert rejected["stratified_gate_passed"] is True
    assert rejected["selected_logits_gate_passed"] is True
    assert rejected["hidden_retention_passed"] is False
    assert rejected["gate_passed"] is False

    bound_report = eval_dir / "hard_en_candidate.json"
    bound_report.write_text('{"mutated": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="report is missing or changed"):
        logits_gate.build_gate(
            baseline_report_path=legacy_baseline,
            candidate_report_path=legacy_candidate,
            baseline_checkpoint_path=baseline_checkpoint,
            checkpoint_path=checkpoint,
            stratified_summary_path=summary_path,
        )
