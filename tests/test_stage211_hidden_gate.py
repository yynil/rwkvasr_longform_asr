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
    loss: float,
    cosine: float,
) -> Path:
    layers = {
        str(layer_id): {
            "loss": loss,
            "cosine": cosine,
            "rms_ratio": 0.9,
        }
        for layer_id in range(70)
    }
    save_yaml(
        path,
        {
            "eval_loss": loss,
            "eval_samples": 256,
            "eval_provenance": _eval_provenance(
                manifest=manifest,
                part=part,
            ),
            "layer_component_metrics": {"mixer": layers},
            "decoder_hidden_metrics": {},
        },
    )
    return path


def test_stage211_hidden_gate_requires_identical_bound_eval_parts(
    tmp_path: Path,
) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    baseline_manifest = tmp_path / "easy_manifest.json"
    baseline_manifest.write_text("{}\n", encoding="utf-8")
    candidate_manifest = tmp_path / "long_manifest.json"
    candidate_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        manifest=baseline_manifest,
        part=eval_part,
        loss=1.0,
        cosine=0.5,
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        manifest=candidate_manifest,
        part=eval_part,
        loss=0.8,
        cosine=0.6,
    )

    report = hidden_gate.build_gate(
        phase="mixer",
        baseline_report_path=baseline,
        candidate_report_path=candidate,
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
        manifest=candidate_manifest,
        part=different_part,
        loss=0.8,
        cosine=0.6,
    )
    with pytest.raises(ValueError, match="different eval samples"):
        hidden_gate.build_gate(
            phase="mixer",
            baseline_report_path=baseline,
            candidate_report_path=mismatched_candidate,
            checkpoint_path=checkpoint,
        )


def test_stage211_hidden_gate_rejects_mutated_eval_part(tmp_path: Path) -> None:
    eval_part = tmp_path / "fixed_eval.jsonl"
    eval_part.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    baseline = _write_report(
        tmp_path / "baseline.yaml",
        manifest=manifest,
        part=eval_part,
        loss=1.0,
        cosine=0.5,
    )
    candidate = _write_report(
        tmp_path / "candidate.yaml",
        manifest=manifest,
        part=eval_part,
        loss=0.8,
        cosine=0.6,
    )
    eval_part.write_text('{"utt_id": "mutated"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing or changed"):
        hidden_gate.build_gate(
            phase="mixer",
            baseline_report_path=baseline,
            candidate_report_path=candidate,
            checkpoint_path=checkpoint,
        )
