from __future__ import annotations

import json
from pathlib import Path

import pytest

from rwkvasr.eval.stage211_gate import (
    build_stage211_correction_layer_focus,
    sha256_file,
    stage211_correction_layer_rotation_offset,
    stage211_post_coverage_train_config_contract,
    validate_stage211_correction_layer_focus,
)
from rwkvasr.training.deepspeed_loop import _select_layer_hidden_ids


CELLS = (
    "easy_en",
    "easy_zh",
    "hard_en",
    "hard_zh",
    "long_zh",
    "medium_en",
    "medium_zh",
    "supplemental_en",
    "supplemental_zh",
)
PHASE_COMPONENTS = {
    "mixer": ("mixer",),
    "block": ("mixer", "ffn", "block"),
    "logits": ("mixer", "ffn", "block"),
}


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _layers(*, failed: set[int], candidate: bool) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for layer_id in range(70):
        severity = 0.01 + layer_id / 1_000.0
        if not candidate:
            loss, cosine = 1.0, 0.5
        elif layer_id in failed:
            loss, cosine = 1.0 + severity, 0.5 - severity
        else:
            loss, cosine = 0.5, 0.8
        rows[str(layer_id)] = {"loss": loss, "cosine": cosine, "rms_ratio": 1.0}
    return rows


def _component_summary(*, failed: set[int]) -> dict[str, object]:
    baseline = _layers(failed=failed, candidate=False)
    candidate = _layers(failed=failed, candidate=True)
    layers = {
        str(layer_id): {
            "baseline_loss": baseline[str(layer_id)]["loss"],
            "candidate_loss": candidate[str(layer_id)]["loss"],
            "baseline_cosine": baseline[str(layer_id)]["cosine"],
            "candidate_cosine": candidate[str(layer_id)]["cosine"],
            "baseline_rms_ratio": 1.0,
            "candidate_rms_ratio": 1.0,
        }
        for layer_id in range(70)
    }
    weak = {
        band: {
            "baseline_loss": 1.0,
            "candidate_loss": 0.5,
            "baseline_cosine": 0.5,
            "candidate_cosine": 0.8,
        }
        for band in ("10-19", "20-29")
    }
    cells = {
        cell: {
            "baseline_loss": 1.0,
            "candidate_loss": 0.5,
            "relative_change_pct": -50.0,
            "baseline_cosine": 0.5,
            "candidate_cosine": 0.8,
            "layers_loss_improved": 70 - len(failed),
            "layers_cosine_improved": 70 - len(failed),
        }
        for cell in CELLS
    }
    return {
        "baseline_mean_loss": 1.0,
        "candidate_mean_loss": 0.5,
        "baseline_mean_cosine": 0.5,
        "candidate_mean_cosine": 0.8,
        "loss_improved_layers": 70 - len(failed),
        "cosine_improved_layers": 70 - len(failed),
        "layers": layers,
        "weak_bands": weak,
        "cells": cells,
    }


def _focus_fixture(
    tmp_path: Path,
    *,
    phase: str,
    failed: set[int],
    trajectory_failed: set[int] | None = None,
) -> tuple[Path, dict[str, object]]:
    components = PHASE_COMPONENTS[phase]
    reports: dict[str, dict[str, dict[str, str]]] = {}

    def source(*, candidate: bool, failed_layers: set[int] = failed) -> dict[str, object]:
        return {
            "layer_components": {
                component: _layers(failed=failed_layers, candidate=candidate)
                for component in components
            }
        }

    fixed_bindings: dict[str, dict[str, str]] = {}
    for role, candidate in (("baseline", False), ("candidate", True)):
        path = _write_json(tmp_path / f"fixed-{role}.json", source(candidate=candidate))
        fixed_bindings[role] = {"path": str(path), "sha256": sha256_file(path)}
    for cell in CELLS:
        reports[cell] = {}
        for role, candidate in (("baseline", False), ("candidate", True)):
            path = _write_json(
                tmp_path / f"{cell}-{role}.json",
                source(candidate=candidate),
            )
            reports[cell][role] = {"path": str(path), "sha256": sha256_file(path)}

    summaries = {component: _component_summary(failed=failed) for component in components}
    summary_key = "hidden_component_summaries" if phase == "logits" else "component_summaries"
    summary = {
        "phase": phase,
        summary_key: summaries,
        "reports": reports,
    }
    summary_path = _write_json(tmp_path / "stratified-summary.json", summary)
    alignment = {
        "phase": phase,
        "baseline_report_path": fixed_bindings["baseline"]["path"],
        "baseline_report_sha256": fixed_bindings["baseline"]["sha256"],
        "candidate_report_path": fixed_bindings["candidate"]["path"],
        "candidate_report_sha256": fixed_bindings["candidate"]["sha256"],
        "stratified_summary_path": str(summary_path),
        "stratified_summary_sha256": sha256_file(summary_path),
        "stratified_summary": summary,
    }
    alignment_path = _write_json(tmp_path / "alignment.json", alignment)
    trajectory_reports: dict[str, Path] = {}
    for role, candidate in (("best-prior", False), ("candidate", True)):
        trajectory_reports[role] = _write_json(
            tmp_path / f"trajectory-{role}.yaml",
            {
                "step": 1 if role == "best-prior" else 2,
                **source(
                    candidate=candidate,
                    failed_layers=trajectory_failed or set(),
                ),
            },
        )
    best_prior = {
        "source_name": "medium",
        "step": 1,
        "eval_report_path": str(trajectory_reports["best-prior"]),
        "eval_report_sha256": sha256_file(trajectory_reports["best-prior"]),
    }
    candidate = {
        "source_name": "supplemental_natural",
        "step": 2,
        "eval_report_path": str(trajectory_reports["candidate"]),
        "eval_report_sha256": sha256_file(trajectory_reports["candidate"]),
    }
    gate = {
        "gate_passed": False,
        "alignment_report": {
            "path": str(alignment_path),
            "sha256": sha256_file(alignment_path),
        },
        "trajectory_retention": {
            "phase": phase,
            "gate_passed": trajectory_failed is None,
            "entries": [best_prior, candidate],
            "best_prior_index": 0,
            "best_prior": best_prior,
            "candidate_index": 1,
            "candidate": candidate,
            "best_prior_loss": 1.0,
            "candidate_loss": 0.5 if trajectory_failed is None else 1.2,
        },
    }
    gate_path = _write_json(tmp_path / "phase-gate.json", gate)
    return gate_path, gate


@pytest.mark.parametrize("phase", ("mixer", "block", "logits"))
def test_stage211_correction_focus_uses_all_layers_and_keeps_ranking_diagnostic(
    tmp_path: Path,
    phase: str,
) -> None:
    trajectory_failed = {55, 68, 69}
    gate_path, gate = _focus_fixture(
        tmp_path,
        phase=phase,
        failed={40, 64, 65},
        trajectory_failed=trajectory_failed,
    )

    focus = build_stage211_correction_layer_focus(
        phase=phase,
        admission_gate_path=gate_path,
        admission_gate=gate,
    )

    assert focus["strategy"] == "simultaneous_all_layers"
    assert focus["sample_count"] == 70
    assert focus["optimization_layer_ids"] == list(range(70))
    assert focus["failure_ranking_diagnostic_only"] is True
    assert focus["selected_failure_layer_ids"] == []
    assert focus["boundary_layer_ids"] == []
    assert focus["rotating_slots"] == 70
    assert focus["all_failed_layer_ids"] == [40, 55, 64, 65, 68, 69]
    assert focus["trajectory_evidence"]["failure_signals_used"] is True
    assert any("trajectory_retention" in row["scopes"] for row in focus["ranking"])
    assert _select_layer_hidden_ids(
        step=123,
        num_layers=70,
        sample_count=focus["sample_count"],
        boundary_ids=focus["boundary_layer_ids"],
    ) == tuple(range(70))


def test_stage211_correction_focus_replay_rejects_modified_selection(tmp_path: Path) -> None:
    gate_path, gate = _focus_fixture(tmp_path, phase="mixer", failed={55, 68, 69})
    focus = build_stage211_correction_layer_focus(
        phase="mixer",
        admission_gate_path=gate_path,
        admission_gate=gate,
    )
    focus["boundary_layer_ids"] = [0]
    focus_path = _write_json(tmp_path / "focus.json", focus)

    with pytest.raises(ValueError, match="differs from failed-gate evidence"):
        validate_stage211_correction_layer_focus(
            focus_path,
            phase="mixer",
            admission_gate_path=gate_path,
            admission_gate=gate,
        )


def test_stage211_correction_config_contract_rejects_focus_and_rotation() -> None:
    with pytest.raises(ValueError, match="forbids layer focus"):
        stage211_post_coverage_train_config_contract(
            "mixer",
            boundary_layer_ids=[0],
        )
    with pytest.raises(ValueError, match="forbids layer rotation"):
        stage211_post_coverage_train_config_contract(
            "logits",
            boundary_layer_ids=[],
            rotation_offset=1,
        )


def test_stage211_correction_rounds_keep_zero_rotation_for_all_layers() -> None:
    offsets = [
        stage211_correction_layer_rotation_offset(
            round_index=round_index,
            boundary_layer_ids=[],
        )
        for round_index in range(1, 4)
    ]

    assert offsets == [0, 0, 0]


def test_layer_hidden_sampler_rejects_negative_cross_round_offset() -> None:
    with pytest.raises(ValueError, match="rotation_offset must be non-negative"):
        _select_layer_hidden_ids(
            step=0,
            num_layers=70,
            sample_count=8,
            boundary_ids=(),
            rotation_offset=-1,
        )
