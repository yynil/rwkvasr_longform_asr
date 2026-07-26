from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
logits_gate = importlib.import_module(
    "scripts.create_stage211_logits_alignment_gate"
)


def _eval_provenance(
    *,
    manifest: Path,
    part: Path,
    feature_seed: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": 256,
        "feature_seed": feature_seed,
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
    step: int,
    manifest: Path,
    part: Path,
    metrics: dict[str, float],
    feature_seed: int = 0,
) -> Path:
    save_yaml(
        path,
        {
            "step": step,
            "eval_loss": metrics["full_kl"],
            "eval_samples": 256,
            "eval_provenance": _eval_provenance(
                manifest=manifest,
                part=part,
                feature_seed=feature_seed,
            ),
            "logit_metrics": metrics,
        },
    )
    return path


def test_stage211_logits_gate_requires_distribution_and_decode_improvement(
    tmp_path: Path,
) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    baseline_manifest = tmp_path / "easy.json"
    baseline_manifest.write_text("{}\n", encoding="utf-8")
    candidate_manifest = tmp_path / "long.json"
    candidate_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        step=0,
        manifest=baseline_manifest,
        part=eval_part,
        metrics=_baseline_metrics(),
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        step=105,
        manifest=candidate_manifest,
        part=eval_part,
        metrics=_candidate_metrics(),
    )

    report = logits_gate.build_gate(
        baseline_report_path=baseline,
        candidate_report_path=candidate,
        checkpoint_path=checkpoint,
    )

    assert report["gate_passed"] is True
    assert report["full_kl_relative_reduction"] == pytest.approx(0.20)
    assert all(report["checks"].values())

    weak_candidate_metrics = _candidate_metrics()
    weak_candidate_metrics["full_kl"] = 0.97
    weak_candidate = _write_report(
        tmp_path / "weak-candidate.yaml",
        step=105,
        manifest=candidate_manifest,
        part=eval_part,
        metrics=weak_candidate_metrics,
    )
    rejected = logits_gate.build_gate(
        baseline_report_path=baseline,
        candidate_report_path=weak_candidate,
        checkpoint_path=checkpoint,
    )

    assert rejected["gate_passed"] is False
    assert rejected["checks"]["full_kl_materially_improved"] is False


def test_stage211_logits_gate_requires_identical_fixed_audio(
    tmp_path: Path,
) -> None:
    baseline_part = tmp_path / "baseline_eval.jsonl"
    baseline_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    candidate_part = tmp_path / "candidate_eval.jsonl"
    candidate_part.write_text('{"utt_id": "u2"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        step=0,
        manifest=manifest,
        part=baseline_part,
        metrics=_baseline_metrics(),
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        step=105,
        manifest=manifest,
        part=candidate_part,
        metrics=_candidate_metrics(),
    )

    with pytest.raises(ValueError, match="different eval samples"):
        logits_gate.build_gate(
            baseline_report_path=baseline,
            candidate_report_path=candidate,
            checkpoint_path=checkpoint,
        )


def test_stage211_logits_gate_requires_identical_fixed_feature_seed(
    tmp_path: Path,
) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        step=0,
        manifest=manifest,
        part=eval_part,
        metrics=_baseline_metrics(),
        feature_seed=0,
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        step=105,
        manifest=manifest,
        part=eval_part,
        metrics=_candidate_metrics(),
        feature_seed=1,
    )

    with pytest.raises(ValueError, match="fixed feature seed"):
        logits_gate.build_gate(
            baseline_report_path=baseline,
            candidate_report_path=candidate,
            checkpoint_path=checkpoint,
        )
