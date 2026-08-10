from __future__ import annotations

import importlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_ALLOWED_OPERATOR_KEY_MARKERS,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_AUDIO_TOTAL_ROWS,
    STAGE211_AUDIO_TRAIN_PART_COUNTS,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    STAGE211_GLOBAL_DEDUP_TOTAL_HOURS,
    STAGE211_PUBLIC_BENCHMARKS,
    STAGE211_RETENTION_CORRECTION_EPOCHS,
    STAGE211_RETENTION_CORRECTION_LR,
    build_stage211_full_data_coverage,
    sha256_file,
    stage211_post_coverage_correction_exposure,
    stage211_phase_train_config_contract,
    validate_stage211_nano_public_baseline_receipt,
    validate_stage211_loaded_manifest_receipt,
    validate_stage211_phase_gate_report,
    validate_stage211_phase_train_config,
)
from rwkvasr.eval.stage211_supplemental import (
    STAGE211_SUPPLEMENTAL_DIFFICULTY,
    STAGE211_SUPPLEMENTAL_SOURCES,
    stage211_supplemental_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_DEDUP_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "stage211_global_dedup_manifest.json"
sys.path.insert(0, str(REPO_ROOT))
stage211 = importlib.import_module("scripts.run_stage211_strict_chained_alignment")
stage211_phase_gate = importlib.import_module("scripts.create_stage211_phase_gate")
stage211_full_phase = importlib.import_module("scripts.run_stage211_full_phase_curriculum")
stage211_phase_finalizer = importlib.import_module("scripts.finalize_stage211_phase")
stage211_calibration_eval = importlib.import_module("scripts.validate_stage211_calibration_eval")
stage211_supplemental_profile_receipt = importlib.import_module(
    "scripts.create_stage211_supplemental_profile_receipt"
)


def test_stage211_controllers_preserve_virtualenv_python() -> None:
    expected = Path(sys.executable)

    assert stage211_full_phase.PYTHON == expected
    assert stage211_phase_finalizer.PYTHON == expected


def test_stage211_full_phase_smoke_marker_rebuilds_source_evidence(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / "phase"
    smoke_run_dir = phase_root / "full_profile_smoke"
    log_dir = smoke_run_dir / "logs"
    log_dir.mkdir(parents=True)
    init_checkpoint = tmp_path / "init.pt"
    easy_manifest = tmp_path / "easy.json"
    marker_path = phase_root / "full_profile_smoke_passed.json"
    torch.save({"step": 0}, init_checkpoint)
    torch.save({"step": 2}, smoke_run_dir / "step-2.pt")
    easy_manifest.write_text("{}\n", encoding="utf-8")
    (log_dir / "block_smoke_2steps.log").write_text(
        "[rwkvasr] Distributed init complete.\n"
        "[deepspeed-train] step=1 loss=0.8 peak_reserved=5.50GiB\n"
        "[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB\n",
        encoding="utf-8",
    )
    marker = stage211_full_phase._audit_smoke(
        phase="block",
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=22.0,
    )
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")

    assert (
        stage211_full_phase._validate_smoke_marker(
            marker_path=marker_path,
            phase="block",
            smoke_run_dir=smoke_run_dir,
            init_checkpoint=init_checkpoint,
            easy_manifest=easy_manifest,
            max_peak_reserved_gib=22.0,
        )
        == marker
    )

    marker["peak_reserved_gib"] = 5.0
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rebuilt source evidence"):
        stage211_full_phase._validate_smoke_marker(
            marker_path=marker_path,
            phase="block",
            smoke_run_dir=smoke_run_dir,
            init_checkpoint=init_checkpoint,
            easy_manifest=easy_manifest,
            max_peak_reserved_gib=22.0,
        )


def test_stage211_full_phase_formal_progress_detection_is_fail_closed(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / "phase"
    phase_root.mkdir()
    assert stage211_full_phase._formal_phase_training_started(phase_root) is False

    hard_log_dir = phase_root / "hard" / "logs"
    hard_log_dir.mkdir(parents=True)
    training_log = hard_log_dir / "hard.log"
    training_log.write_text("[deepspeed-train] step=0 loss=1.0\n", encoding="utf-8")
    assert stage211_full_phase._formal_phase_training_started(phase_root) is False
    training_log.write_text("[deepspeed-train] step=1 loss=0.9\n", encoding="utf-8")
    assert stage211_full_phase._formal_phase_training_started(phase_root) is True


def test_stage211_legacy_curriculum_summary_migration_is_restart_safe(
    tmp_path: Path,
) -> None:
    phase_root = tmp_path / "phase"
    phase_root.mkdir()
    summary = phase_root / "curriculum_complete.json"
    legacy_payload = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "full_phase_curriculum",
        "phase": "mixer",
        "complete": True,
        "full_data_coverage": {"segments": []},
    }
    summary.write_text(json.dumps(legacy_payload) + "\n", encoding="utf-8")

    stage211_full_phase._migrate_legacy_curriculum_summary(
        phase_root=phase_root,
        phase="mixer",
    )

    archive = phase_root / "curriculum_complete.original_only.json"
    migration = phase_root / "curriculum_complete.original_only.migration.json"
    assert not summary.exists()
    assert archive.is_file()
    assert migration.is_file()

    summary.write_bytes(archive.read_bytes())
    stage211_full_phase._migrate_legacy_curriculum_summary(
        phase_root=phase_root,
        phase="mixer",
    )
    assert not summary.exists()
    assert archive.is_file()


def test_stage211_current_curriculum_summary_is_reused_immutably(tmp_path: Path) -> None:
    phase_root = tmp_path / "phase"
    phase_root.mkdir()
    summary = phase_root / "curriculum_complete.json"
    current_payload = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "full_phase_curriculum",
        "phase": "mixer",
        "complete": True,
        "full_data_coverage": {"supplemental_natural": {"complete": True}},
    }
    stage211_full_phase._write_immutable_json(summary, current_payload)

    stage211_full_phase._migrate_legacy_curriculum_summary(
        phase_root=phase_root,
        phase="mixer",
    )
    stage211_full_phase._write_immutable_json(summary, current_payload)

    assert json.loads(summary.read_text(encoding="utf-8")) == current_payload
    assert not (phase_root / "curriculum_complete.original_only.json").exists()


def test_stage211_public_eval_shards_large_second_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(
        command: list[str],
        *,
        dry_run: bool,
        env: dict[str, str] | None = None,
    ) -> None:
        assert dry_run is False
        calls.append((command, env))

    monkeypatch.setattr(stage211_phase_finalizer, "_run", fake_run)
    stage211_phase_finalizer._run_public_eval(
        checkpoint=tmp_path / "checkpoint.pt",
        output_dir=tmp_path / "output",
        manifest_dir=tmp_path / "manifests",
        devices="0,1,2,3",
        dry_run=False,
    )

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][1]["CTC_SHARD_STAGE2"] == "1"


def test_stage211_public_enrichment_recomputes_bound_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = "librispeech_test_clean"
    monkeypatch.setattr(
        stage211_phase_gate,
        "STAGE211_PUBLIC_BENCHMARKS",
        {dataset: {"language": "en", "metric": "wer", "samples": 2}},
    )
    manifest = tmp_path / f"{dataset}.jsonl"
    nano = tmp_path / f"{dataset}.nano.jsonl"
    student = tmp_path / f"{dataset}.student.jsonl"
    rows = [
        {"utt_id": "utt-1", "ref_text": "hello world", "pred_text": "hello world"},
        {"utt_id": "utt-2", "ref_text": "speech test", "pred_text": "speech test"},
    ]
    manifest.write_text(
        "".join(
            json.dumps({"utt_id": row["utt_id"], "text": row["ref_text"]}) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    nano.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    student_rows = [dict(row) for row in rows]
    student_rows[1]["pred_text"] = "speech"
    student.write_text(
        "".join(json.dumps(row) + "\n" for row in student_rows),
        encoding="utf-8",
    )
    report = {
        "decode": "greedy_ctc",
        "normalization": "ctc",
        "gate": {
            "max_relative_ratio": 2.0,
            "max_absolute_gap_points": 100.0,
            "requires_every_dataset": True,
        },
        "all_datasets_pass": False,
        "results": [
            stage211_phase_gate.compare_dataset(
                dataset=dataset,
                nano_path=nano,
                student_path=student,
                normalization="ctc",
                max_relative_ratio=2.0,
                max_absolute_gap_points=100.0,
            )
        ],
    }

    enriched = stage211_phase_gate._enrich_public_benchmark(
        report,
        manifest_dir=tmp_path,
    )

    assert enriched["all_datasets_complete"] is True
    assert enriched["results"][0]["metric_source_recomputed"] is True
    assert enriched["results"][0]["student_error_rate"] == pytest.approx(0.25)

    report["results"][0]["student_error_rate"] += 0.01
    with pytest.raises(ValueError, match="recomputed student_error_rate mismatch"):
        stage211_phase_gate._enrich_public_benchmark(
            report,
            manifest_dir=tmp_path,
        )
    report["results"][0]["student_error_rate"] -= 0.01
    report["all_datasets_pass"] = True
    with pytest.raises(ValueError, match="all_datasets_pass differs"):
        stage211_phase_gate._enrich_public_benchmark(
            report,
            manifest_dir=tmp_path,
        )


def test_stage211_mixer_finalizer_evaluates_latest_retention_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_checkpoint = tmp_path / "phase-init.pt"
    baseline_checkpoint.write_bytes(b"phase-init")
    long_checkpoint = tmp_path / "long-complete.pt"
    long_checkpoint.write_bytes(b"long-complete")
    corrected_checkpoint = tmp_path / "retention-complete.pt"
    corrected_checkpoint.write_bytes(b"retention-complete")
    curriculum_receipt = tmp_path / "long-receipt.json"
    curriculum_receipt.write_text("{}\n", encoding="utf-8")
    correction_receipt = tmp_path / "correction-receipt.json"
    correction_receipt.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "stratified.json"
    manifest.write_text("{}\n", encoding="utf-8")
    correction = {
        "completion_checkpoint_path": str(corrected_checkpoint.resolve()),
        "row_exposures": 8,
        "hour_exposures": 0.01,
        "steps": 1,
        "tail_padding_sample_exposures": 4,
        "executed_sample_exposures": 12,
    }
    commands: list[list[str]] = []
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_resolve_curriculum",
        lambda **kwargs: (
            {
                "segments": [
                    {
                        "difficulty": "easy",
                        "init_checkpoint_path": str(baseline_checkpoint.resolve()),
                        "init_checkpoint_sha256": sha256_file(baseline_checkpoint),
                    }
                ]
            },
            long_checkpoint,
            [curriculum_receipt],
        ),
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "load_stage211_post_coverage_correction_receipts",
        lambda paths: [correction],
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "validate_stage211_full_data_coverage",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_stratified_cell_manifests",
        lambda path: {cell: manifest for cell in stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS},
    )
    monkeypatch.setattr(stage211_phase_finalizer, "_run_public_eval", lambda **kwargs: None)
    monkeypatch.setattr(stage211_phase_finalizer, "_run_nano_comparison", lambda **kwargs: None)
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run",
        lambda command, *, dry_run, env=None: commands.append(command),
    )

    stage211_phase_finalizer.finalize_phase(
        SimpleNamespace(
            phase="mixer",
            phase_root=tmp_path / "phase",
            public_manifest_dir=tmp_path / "public-manifests",
            nano_prediction_dir=tmp_path / "nano" / "predictions",
            nano_public_baseline_receipt=None,
            output_dir=tmp_path / "eval",
            devices="0,1,2,3",
            dry_run=True,
            baseline_public_comparison_report=tmp_path / "baseline-public.json",
            post_coverage_correction_receipt=[correction_receipt],
            stratified_hidden_receipt=tmp_path / "stratified-receipt.json",
        )
    )

    pair_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.ALIGNMENT_PAIR_EVAL_SCRIPT) in command
    )
    assert pair_command[pair_command.index("--candidate-checkpoint") + 1] == str(
        corrected_checkpoint.resolve()
    )
    phase_gate_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.PHASE_GATE_SCRIPT) in command
    )
    assert phase_gate_command[
        phase_gate_command.index("--post-coverage-correction-receipt") + 1
    ] == str(correction_receipt.resolve())


def test_stage211_logits_finalizer_runs_independent_alignment_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_checkpoint = tmp_path / "init.pt"
    baseline_checkpoint.write_bytes(b"initial")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    receipt = tmp_path / "long-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    stratified_cells = {}
    for cell_name in stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS:
        manifest = tmp_path / f"logits-manifest-{cell_name}.json"
        manifest.write_text("{}\n", encoding="utf-8")
        stratified_cells[cell_name] = {
            "samples": 256,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
        }
    stratified_receipt = tmp_path / "logits-stratified-receipt.json"
    stratified_receipt.write_text(
        json.dumps(
            {
                "pipeline": "stage211",
                "artifact": "stratified_hidden_eval_manifest",
                "cells": stratified_cells,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_resolve_curriculum",
        lambda **kwargs: (
            {
                "segments": [
                    {
                        "difficulty": "easy",
                        "init_checkpoint_path": str(baseline_checkpoint.resolve()),
                        "init_checkpoint_sha256": sha256_file(baseline_checkpoint),
                    }
                ]
            },
            checkpoint,
            [receipt],
        ),
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run_public_eval",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run_nano_comparison",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run",
        lambda command, *, dry_run, env=None: commands.append(command),
    )
    phase_root = tmp_path / "phase"
    nano_prediction_dir = tmp_path / "nano" / "predictions"
    stage211_phase_finalizer.finalize_phase(
        SimpleNamespace(
            phase="logits",
            phase_root=phase_root,
            public_manifest_dir=tmp_path / "manifests",
            nano_prediction_dir=nano_prediction_dir,
            nano_public_baseline_receipt=None,
            output_dir=tmp_path / "eval",
            devices="0,1,2,3",
            dry_run=True,
            baseline_public_comparison_report=None,
            stratified_hidden_receipt=stratified_receipt,
        )
    )

    logits_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.LOGITS_GATE_SCRIPT) in command
    )
    phase_gate_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.PHASE_GATE_SCRIPT) in command
    )
    logits_gate_path = tmp_path / "eval" / "logits_gate.json"
    pair_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.ALIGNMENT_PAIR_EVAL_SCRIPT) in command
    )
    assert pair_command[pair_command.index("--baseline-checkpoint") + 1] == str(
        baseline_checkpoint.resolve()
    )
    assert pair_command[pair_command.index("--feature-seed") + 1] == "0"
    pair_baseline_report = pair_command[pair_command.index("--baseline-output") + 1]
    pair_candidate_report = pair_command[pair_command.index("--candidate-output") + 1]
    assert logits_command[logits_command.index("--baseline-report") + 1] == pair_baseline_report
    assert logits_command[logits_command.index("--candidate-report") + 1] == pair_candidate_report
    assert logits_command[logits_command.index("--output") + 1] == str(logits_gate_path)
    assert logits_command[logits_command.index("--baseline-checkpoint") + 1] == str(
        baseline_checkpoint.resolve()
    )
    stratified_summary = tmp_path / "eval" / "alignment_stratified" / "summary.json"
    assert logits_command[logits_command.index("--stratified-summary") + 1] == str(
        stratified_summary
    )
    pair_commands = [
        command
        for command in commands
        if str(stage211_phase_finalizer.ALIGNMENT_PAIR_EVAL_SCRIPT) in command
    ]
    assert len(pair_commands) == 8
    summary_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.STRATIFIED_LOGITS_SUMMARY_SCRIPT) in command
    )
    assert summary_command[summary_command.index("--output") + 1] == str(stratified_summary)
    assert phase_gate_command[phase_gate_command.index("--alignment-report") + 1] == str(
        logits_gate_path
    )


def test_stage211_mixer_finalizer_runs_stratified_hidden_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_checkpoint = tmp_path / "init.pt"
    baseline_checkpoint.write_bytes(b"initial")
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    curriculum_receipt = tmp_path / "long-receipt.json"
    curriculum_receipt.write_text("{}\n", encoding="utf-8")
    stratified_cells = {}
    for cell_name in stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS:
        manifest = tmp_path / f"manifest_{cell_name}.json"
        manifest.write_text("{}\n", encoding="utf-8")
        stratified_cells[cell_name] = {
            "samples": 256,
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
        }
    stratified_receipt = tmp_path / "stratified-receipt.json"
    stratified_receipt.write_text(
        json.dumps(
            {
                "pipeline": "stage211",
                "artifact": "stratified_hidden_eval_manifest",
                "cells": stratified_cells,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_resolve_curriculum",
        lambda **kwargs: (
            {
                "segments": [
                    {
                        "difficulty": "easy",
                        "init_checkpoint_path": str(baseline_checkpoint.resolve()),
                        "init_checkpoint_sha256": sha256_file(baseline_checkpoint),
                    }
                ]
            },
            checkpoint,
            [curriculum_receipt],
        ),
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run_public_eval",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run_nano_comparison",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run",
        lambda command, *, dry_run, env=None: commands.append(command),
    )
    output_dir = tmp_path / "eval"
    stage211_phase_finalizer.finalize_phase(
        SimpleNamespace(
            phase="mixer",
            phase_root=tmp_path / "phase",
            public_manifest_dir=tmp_path / "manifests",
            nano_prediction_dir=tmp_path / "nano" / "predictions",
            nano_public_baseline_receipt=None,
            output_dir=output_dir,
            devices="0,1,2,3",
            dry_run=True,
            baseline_public_comparison_report=tmp_path / "baseline-public.json",
            stratified_hidden_receipt=stratified_receipt,
        )
    )

    pair_commands = [
        command
        for command in commands
        if str(stage211_phase_finalizer.ALIGNMENT_PAIR_EVAL_SCRIPT) in command
    ]
    assert len(pair_commands) == 8
    sidecar_commands = [command for command in pair_commands if "--eval-bucket-manifest" in command]
    assert len(sidecar_commands) == 7
    assert {
        command[command.index("--eval-bucket-manifest") + 1] for command in sidecar_commands
    } == {str(Path(cell["manifest_path"]).resolve()) for cell in stratified_cells.values()}
    summary_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.STRATIFIED_SUMMARY_SCRIPT) in command
    )
    summary_path = output_dir / "alignment_stratified" / "summary.json"
    assert summary_command[summary_command.index("--output") + 1] == str(summary_path)
    hidden_gate_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.HIDDEN_GATE_SCRIPT) in command
    )
    assert hidden_gate_command[hidden_gate_command.index("--stratified-summary") + 1] == str(
        summary_path
    )


def test_stage211_phase_gate_rebuilds_with_bound_stratified_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = tmp_path / "stratified-summary.json"
    summary_path.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_alignment_gate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"rebuilt": True}

    monkeypatch.setattr(
        stage211_phase_gate,
        "build_hidden_alignment_gate",
        fake_alignment_gate,
    )
    monkeypatch.setattr(
        stage211_phase_gate,
        "build_logits_alignment_gate",
        fake_alignment_gate,
    )
    for phase in ("mixer", "logits"):
        captured.clear()
        report = stage211_phase_gate._rebuild_alignment_gate(
            phase=phase,
            alignment_report={
                "stratified_summary_path": str(summary_path.resolve()),
                "stratified_summary_sha256": sha256_file(summary_path),
            },
            baseline_report_path=tmp_path / "baseline.json",
            candidate_report_path=tmp_path / "candidate.json",
            phase_init_checkpoint=tmp_path / "init.pt",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )

        assert report == {"rebuilt": True}
        assert captured["stratified_summary_path"] == summary_path.resolve()

    summary_path.write_text('{"mutated": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stratified summary is missing or changed"):
        stage211_phase_gate._rebuild_alignment_gate(
            phase="mixer",
            alignment_report={
                "stratified_summary_path": str(summary_path.resolve()),
                "stratified_summary_sha256": "0" * 64,
            },
            baseline_report_path=tmp_path / "baseline.json",
            candidate_report_path=tmp_path / "candidate.json",
            phase_init_checkpoint=tmp_path / "init.pt",
            checkpoint_path=tmp_path / "checkpoint.pt",
        )


def test_stage211_calibration_eval_reuse_binds_checkpoint_and_complete_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "step-30000.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    selection_path = tmp_path / "checkpoint_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "pipeline": "stage211",
                "artifact": "calibration_checkpoint_selection",
                "required_completion_step": 30_064,
                "selected": {
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "eligible": True,
                    "loss_improved_layers": 70,
                    "cosine_improved_layers": 70,
                },
            }
        ),
        encoding="utf-8",
    )
    comparison_path = tmp_path / "nano_comparison.json"
    comparison = {
        "decode": "greedy_ctc",
        "normalization": "ctc",
        "student_checkpoint_path": str(checkpoint),
        "student_checkpoint_sha256": checkpoint_sha256,
    }
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    benchmark_results = []
    metric_results = []
    for index, (dataset, expected) in enumerate(STAGE211_PUBLIC_BENCHMARKS.items()):
        wer = 0.1 + index * 0.01
        cer = 0.2 + index * 0.01
        benchmark_results.append(
            {
                "dataset": dataset,
                "sample_count": expected["samples"],
                "student_wer": wer,
                "student_cer": cer,
            }
        )
        metric_results.append(
            {
                "dataset": dataset,
                "branch": "ctc",
                "samples": expected["samples"],
                "wer": wer,
                "cer": cer,
            }
        )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"results": metric_results}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stage211_calibration_eval,
        "_enrich_public_benchmark",
        lambda report, *, manifest_dir: {
            **report,
            "all_datasets_complete": True,
            "results": benchmark_results,
        },
    )
    monkeypatch.setattr(
        stage211_calibration_eval,
        "validate_stage211_public_overlap_binding",
        lambda binding, *, public_benchmark: {},
    )

    receipt = stage211_calibration_eval.build_reuse_receipt(
        selection_report_path=selection_path,
        comparison_report_path=comparison_path,
        metrics_path=metrics_path,
        manifest_dir=tmp_path,
    )

    assert receipt["complete"] is True
    assert receipt["checkpoint_path"] == str(checkpoint)
    assert receipt["checkpoint_sha256"] == checkpoint_sha256
    assert receipt["public_benchmark"]["all_datasets_complete"] is True

    comparison["student_checkpoint_sha256"] = "0" * 64
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    with pytest.raises(ValueError, match="comparison checkpoint SHA-256 mismatch"):
        stage211_calibration_eval.build_reuse_receipt(
            selection_report_path=selection_path,
            comparison_report_path=comparison_path,
            metrics_path=metrics_path,
            manifest_dir=tmp_path,
        )


def test_stage211_supervisor_bootstrap_supports_immutable_snapshot() -> None:
    script = (REPO_ROOT / "scripts" / "start_stage211_abcd_after_calibration.sh").read_text(
        encoding="utf-8"
    )

    assert "STAGE211_REPO_ROOT" in script
    assert "REUSE_COMPLETED_CALIBRATION_EVAL" in script
    assert "validate_stage211_calibration_eval.py" in script
    assert (
        "--baseline-public-comparison-report "
        '"${CALIBRATION_EVAL_DIR}/public/nano_comparison.json"' in script
    )
    assert '--baseline-public-comparison-report "${mixer_gate_dir}/nano_comparison.json"' in script
    assert (
        "--baseline-public-comparison-report "
        '"${logits_gate_dir}/nano_comparison.json"' in script
    )
    assert '--calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}"' in script
    assert "create_stage211_nano_baseline_receipt.py" in script
    assert '--output "${NANO_BASELINE_RECEIPT}"' in script
    assert 'START_STAGE="${START_STAGE:-full}"' in script
    assert "run_stage211_mixer_retention_loop.py" in script
    assert "MIXER_SELECTION" in script
    assert "BLOCK_SELECTION" in script
    assert "LOGITS_SELECTION" in script
    assert '--mixer-gate-selection "${MIXER_SELECTION}"' in script
    assert '--block-gate-selection "${BLOCK_SELECTION}"' in script
    assert '--logits-gate-selection "${LOGITS_SELECTION}"' in script
    assert "post_mixer)" in script
    assert 'SUPPLEMENTAL_POLL_SECONDS="${SUPPLEMENTAL_POLL_SECONDS:-3600}"' in script
    assert "wait_for_supplemental_training_data()" in script
    assert '--inventory "${SUPPLEMENTAL_INVENTORY}"' in script
    assert '--output "${SUPPLEMENTAL_PROFILE_RECEIPT}"' in script
    assert "supplemental inventory and immutable profile receipt validated" in script

    main_body = script[script.index("main() {") :]
    readiness_offset = main_body.index("wait_for_supplemental_training_data")
    case_offset = main_body.index('case "${START_STAGE}"')
    assert readiness_offset < case_offset
    full_branch = main_body[main_body.index("full)") : main_body.index("post_mixer)")]
    assert readiness_offset < main_body.index("build_fixed_manifests")
    assert "run_full_mixer_phase" in full_branch
    continuation = main_body[main_body.index("esac") :]
    phase_calls = (
        "run_full_block_phase",
        "run_full_logits_phase",
        "run_labeled_sft_phase",
    )
    phase_offsets = [continuation.index(call) for call in phase_calls]
    assert phase_offsets == sorted(phase_offsets)


@pytest.mark.parametrize(
    ("create_inputs", "expected_message"),
    (
        (False, "supplemental inventory unavailable"),
        (True, "supplemental readiness validation failed"),
    ),
)
def test_stage211_supervisor_waits_before_setup_when_supplemental_is_not_ready(
    tmp_path: Path,
    create_inputs: bool,
    expected_message: str,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sleep = fake_bin / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 73\n", encoding="utf-8")
    sleep.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text("#!/usr/bin/env bash\nexit 42\n", encoding="utf-8")
    uv.chmod(0o755)
    inventory = tmp_path / "supplemental_inventory.json"
    profile = tmp_path / "supplemental_profile_receipt.json"
    if create_inputs:
        inventory.write_text("{}\n", encoding="utf-8")
        profile.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "start_stage211_abcd_after_calibration.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "STAGE211_REPO_ROOT": str(REPO_ROOT),
            "SUPPLEMENTAL_INVENTORY": str(inventory),
            "SUPPLEMENTAL_PROFILE_RECEIPT": str(profile),
            "SUPPLEMENTAL_POLL_SECONDS": "3600",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 73
    assert expected_message in result.stdout
    assert "building and validating shared fixed-eval manifests" not in result.stdout
    assert "starting strict Stage211A full-data controller" not in result.stdout


def test_stage211_hourly_monitor_scopes_errors_to_latest_attempt(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    training_log = tmp_path / "train.log"
    training_log.write_text(
        "Traceback from an older failed attempt\n"
        "[rwkvasr] Distributed init complete. world_size=4\n"
        "[deepspeed-train] step=10 loss=0.2000 online_layer_missing=1\n"
        "[rwkvasr] Distributed init complete. world_size=4\n"
        "[deepspeed-train] step=20 loss=0.1000 "
        "online_layer_missing=0 online_layer_frame_delta=0\n",
        encoding="utf-8",
    )

    command = [
        "bash",
        "-c",
        'source "$1"; stage211_current_attempt_start "$2"; stage211_current_attempt_errors "$2"',
        "stage211-monitor-test",
        str(monitor_script),
        str(training_log),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "4"
    assert "Traceback" not in result.stdout
    assert "online_layer_missing=1" not in result.stdout

    with training_log.open("a", encoding="utf-8") as output:
        output.write("[deepspeed-train] step=30 online_layer_frame_delta=1\n")
    failed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert failed.returncode == 0, failed.stderr
    assert failed.stdout.splitlines()[0] == "4"
    assert "online_layer_frame_delta=1" in failed.stdout


def test_stage211_hourly_monitor_reports_bound_live_progress(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    (run_dir / "step-20.pt").write_bytes(b"checkpoint")
    (run_dir / "step-100.pt.tmp").write_bytes(b"incomplete")
    training_log = logs_dir / "train.log"
    training_log.write_text(
        "[deepspeed-train] step=21 loss=0.2000\n"
        "[deepspeed-train] step=25 loss=0.1000\n",
        encoding="utf-8",
    )
    config = tmp_path / "stage211_test.yaml"
    config.write_text(
        f"output_dir: {run_dir}\n"
        "wandb_run_name: stage211-live-test\n"
        "max_steps: 100\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_config_progress "$2"',
            "stage211-monitor-test",
            str(monitor_script),
            str(config),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"active_config={config}" in result.stdout
    assert f"run_name=stage211-live-test run_dir={run_dir}" in result.stdout
    assert "live_step=25 target_step=100 progress_pct=25.0000 persisted_step=20" in result.stdout
    assert result.stdout.count("[deepspeed-train] step=") == 1
    assert "step=25 loss=0.1000" in result.stdout


def test_stage211_hourly_monitor_resolves_one_config_from_four_ranks(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        "for rank in 0 1 2 3; do\n"
        "  printf '%s\\n' \"python3 -m rwkvasr.cli.train_ctc_deepspeed "
        "--config-yaml /tmp/stage211_active.yaml --rank ${rank}\"\n"
        "done\n"
        "printf '%s\\n' 'python3 -m unrelated --config-yaml /tmp/ignored.yaml'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_active_config_paths',
            "stage211-monitor-test",
            str(monitor_script),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["/tmp/stage211_active.yaml"]


def test_stage211_calibration_eval_validator_cli_loads() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_stage211_calibration_eval.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--selection-report" in result.stdout


def test_stage211_formal_defaults_use_persistent_storage_and_local_teacher() -> None:
    output_dir = stage211._default_output_dir(stage211.PHASES["mixer"]).resolve()

    assert not output_dir.is_relative_to(Path("/tmp"))
    assert not output_dir.is_relative_to(Path("/var/tmp"))
    assert not output_dir.is_relative_to(Path("/dev/shm"))
    assert output_dir.is_relative_to(Path.home() / "rwkvasr_runs")
    assert stage211.NANO_CHECKPOINT == (
        Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
    )
    assert stage211.NANO_CHECKPOINT != (REPO_ROOT / "assets" / "fun-asr-nano-2512" / "model.pt")


def test_stage211_phase_gate_normalizes_bound_references(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"utt_id": "utt-1", "text": "HELLO, WORLD!"}) + "\n",
        encoding="utf-8",
    )
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(
        json.dumps({"utt_id": "utt-1", "ref_text": "hello world"}) + "\n",
        encoding="utf-8",
    )

    manifest_records = stage211_phase_gate._jsonl_records(
        manifest,
        language="en",
        reference_keys=("text",),
    )
    prediction_records = stage211_phase_gate._jsonl_records(
        prediction,
        language="en",
        reference_keys=("ref_text",),
    )

    assert manifest_records == prediction_records


def test_stage211_public_progress_requires_macro_and_deletion_improvement() -> None:
    baseline_results = []
    candidate_results = []
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        shared = {
            "dataset": dataset,
            "manifest_sha256": f"manifest-{dataset}",
            "nano_prediction_sha256": f"nano-{dataset}",
        }
        baseline_results.append(
            {
                **shared,
                "student_error_rate": 0.8,
                "student_deletion_rate": 0.6,
            }
        )
        candidate_results.append(
            {
                **shared,
                "student_error_rate": 0.7,
                "student_deletion_rate": 0.5,
            }
        )

    progress = stage211_phase_gate._build_public_progress(
        baseline={"results": baseline_results},
        candidate={"results": candidate_results},
    )

    assert progress["gate_passed"] is True
    assert progress["improved_datasets"] == len(STAGE211_PUBLIC_BENCHMARKS)

    candidate_results[0]["student_error_rate"] = 0.84
    rejected = stage211_phase_gate._build_public_progress(
        baseline={"results": baseline_results},
        candidate={"results": candidate_results},
    )
    assert rejected["gate_passed"] is False
    assert rejected["results"][0]["within_regression_limit"] is False


def test_stage211_full_phase_dry_run_expands_smoke_and_all_curricula(
    tmp_path: Path,
) -> None:
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"init")
    nano_checkpoint = tmp_path / "nano.pt"
    nano_checkpoint.write_bytes(b"nano")
    manifests: dict[str, Path] = {}
    for difficulty in STAGE211_AUDIO_CURRICULUM:
        manifest = tmp_path / f"{difficulty}.json"
        manifest.write_text("{}\n", encoding="utf-8")
        manifests[difficulty] = manifest
    supplemental_inventory, supplemental_profile = _write_supplemental_inventory_fixture(
        tmp_path
    )
    supplemental_profile_receipt = tmp_path / "supplemental-profile-receipt.json"
    stage211_supplemental_profile_receipt.write_immutable_receipt(
        supplemental_profile_receipt,
        stage211_supplemental_profile_receipt.build_receipt(supplemental_inventory),
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_stage211_full_phase_curriculum.py"),
        "--phase",
        "mixer",
        "--init-checkpoint",
        str(init_checkpoint),
        "--nano-checkpoint",
        str(nano_checkpoint),
        "--output-root",
        str(tmp_path / "runs"),
        "--config-root",
        str(tmp_path / "configs"),
        "--easy-manifest",
        str(manifests["easy"]),
        "--supplemental-inventory",
        str(supplemental_inventory),
        "--supplemental-profile-receipt",
        str(supplemental_profile_receipt),
        "--dry-run",
    ]
    for difficulty in ("medium", "hard", "long"):
        command.extend(("--manifest", f"{difficulty}={manifests[difficulty]}"))

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[stage211-full-phase] command=") == 6
    assert "--smoke" in result.stdout
    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        assert f"/{difficulty}/step-{expected['steps']}.pt" in result.stdout
    assert (
        f"/{STAGE211_SUPPLEMENTAL_DIFFICULTY}/step-{supplemental_profile['steps']}.pt"
        in result.stdout
    )


def test_stage211_reuses_completed_receipt_without_rehashing_bound_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = "mixer"
    difficulty = "easy"
    expected = STAGE211_AUDIO_CURRICULUM[difficulty]
    run_dir = tmp_path / "easy"
    run_dir.mkdir()
    paths = {
        "provenance": run_dir / "stage211_provenance.json",
        "train_config": run_dir / "train_config.yaml",
        "nano_teacher_checkpoint": tmp_path / "model.pt",
        "bucket_manifest": tmp_path / "manifest.json",
        "init_checkpoint": tmp_path / "init.pt",
        "completion_checkpoint": run_dir / f"step-{expected['steps']}.pt",
    }
    for path in paths.values():
        path.write_bytes(b"bound")
    runtime_records = []
    for epoch in range(1, STAGE211_FULL_DATA_EPOCHS + 1):
        checkpoint = run_dir / f"epoch-{epoch}.pt"
        checkpoint.write_bytes(b"epoch")
        runtime_records.append(
            {
                "epoch": epoch,
                "step": epoch * int(expected["steps_per_epoch"]),
                "epoch_batch_offset": 0,
                "completed_epoch_batch_count": int(expected["steps_per_epoch"]),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": "a" * 64,
            }
        )
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "phase": phase,
        "difficulty": difficulty,
        "complete": True,
        "full_data_profile": True,
        "epochs": STAGE211_FULL_DATA_EPOCHS,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "rows": int(expected["rows"]),
        "row_exposures": int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS,
        "tail_padding_samples_per_epoch": int(expected["tail_padding_samples_per_epoch"]),
        "tail_padding_sample_exposures": int(expected["tail_padding_samples_per_epoch"])
        * STAGE211_FULL_DATA_EPOCHS,
        "executed_sample_exposures": (
            int(expected["rows"]) + int(expected["tail_padding_samples_per_epoch"])
        )
        * STAGE211_FULL_DATA_EPOCHS,
        "hours": float(expected["hours"]),
        "hour_exposures": float(expected["hours"]) * STAGE211_FULL_DATA_EPOCHS,
        "steps_per_epoch": int(expected["steps_per_epoch"]),
        "steps": int(expected["steps"]),
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "run_dir": str(run_dir.resolve()),
        "runtime_epoch_coverage": {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "runtime_epoch_coverage",
            "complete": True,
            "epochs": STAGE211_FULL_DATA_EPOCHS,
            "steps_per_epoch": int(expected["steps_per_epoch"]),
            "total_steps": int(expected["steps"]),
            "records": runtime_records,
        },
        "parameter_delta_audit": {
            "complete": True,
            "policy": "stage211_timemixer_and_input_projection_only",
            "forbidden_changed_tensors": 0,
            "allowed_changed_tensors": 1,
            "allowed_changed_numel": 1,
        },
    }
    for name, path in paths.items():
        receipt[f"{name}_path"] = str(path.resolve())
        receipt[f"{name}_sha256"] = "b" * 64
    receipt_path = tmp_path / "easy.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    real_sha256_file = sha256_file
    hash_calls: list[Path] = []

    def recording_sha256(path: Path) -> str:
        hash_calls.append(Path(path).resolve())
        return real_sha256_file(path)

    monkeypatch.setattr(stage211_full_phase, "sha256_file", recording_sha256)
    reused = stage211_full_phase._load_reusable_receipt(
        receipt_path,
        phase=phase,
        difficulty=difficulty,
        run_dir=run_dir,
        manifest_path=paths["bucket_manifest"],
        init_checkpoint=paths["init_checkpoint"],
        completion_checkpoint=paths["completion_checkpoint"],
    )

    assert reused["complete"] is True
    assert hash_calls == [receipt_path.resolve()]

    receipt["completion_checkpoint_path"] = str((tmp_path / "other.pt").resolve())
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the completed segment"):
        stage211_full_phase._load_reusable_receipt(
            receipt_path,
            phase=phase,
            difficulty=difficulty,
            run_dir=run_dir,
            manifest_path=paths["bucket_manifest"],
            init_checkpoint=paths["init_checkpoint"],
            completion_checkpoint=paths["completion_checkpoint"],
        )


def test_stage211_full_phase_smoke_audit_rejects_teacher_misses(
    tmp_path: Path,
) -> None:
    phase = "mixer"
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"init")
    easy_manifest = tmp_path / "easy.json"
    easy_manifest.write_text("{}\n", encoding="utf-8")
    smoke_run_dir = tmp_path / "smoke"
    log_dir = smoke_run_dir / "logs"
    log_dir.mkdir(parents=True)
    torch.save({"step": 2}, smoke_run_dir / "step-2.pt")
    log_path = log_dir / "mixer_smoke_2steps.log"
    valid_log = (
        "[rwkvasr] Batch stats peak_reserved=20.50GiB\n"
        "[deepspeed-train] step=2 loss=0.2000 "
        "online_layer_missing=0 online_layer_frame_delta=0\n"
    )
    log_path.write_text(valid_log, encoding="utf-8")

    report = stage211_full_phase._audit_smoke(
        phase=phase,
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=22.0,
    )

    assert report["complete"] is True
    assert report["peak_reserved_gib"] == 20.5

    log_path.write_text(
        "Traceback from an older failed attempt\n"
        "[rwkvasr] Distributed init complete. world_size=4\n" + valid_log,
        encoding="utf-8",
    )
    retried_report = stage211_full_phase._audit_smoke(
        phase=phase,
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=22.0,
    )
    assert retried_report["complete"] is True
    assert retried_report["peak_reserved_gib"] == 20.5

    log_path.write_text(
        "[rwkvasr] Distributed init complete. world_size=4\n"
        + valid_log
        + "[deepspeed-train] online_layer_missing=1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rejected condition"):
        stage211_full_phase._audit_smoke(
            phase=phase,
            smoke_run_dir=smoke_run_dir,
            init_checkpoint=init_checkpoint,
            easy_manifest=easy_manifest,
            max_peak_reserved_gib=22.0,
        )


def test_stage211_formal_rejects_volatile_output_but_smoke_allows_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="persistent storage"):
        stage211._validate_output_storage(
            output_dir=tmp_path / "formal",
            smoke=False,
            dry_run=False,
        )

    stage211._validate_output_storage(
        output_dir=tmp_path / "smoke",
        smoke=True,
        dry_run=False,
    )


def test_stage211_audio_bucket_storage_audit_binds_runtime_paths(
    tmp_path: Path,
) -> None:
    webdataset_root = tmp_path / "webdataset"
    webdataset_root.mkdir()
    length_index = tmp_path / "webdataset_lengths.jsonl"
    length_index.write_text("{}\n", encoding="utf-8")
    bucket_part = tmp_path / "bucket-0-part-0.jsonl"
    bucket_part.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "root": str(webdataset_root),
                "source_length_index_path": str(length_index),
                "bucket_width": 100,
                "entries_per_part": 1000,
                "splits": {
                    "train": {
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 1,
                                "parts": [
                                    {
                                        "path": bucket_part.name,
                                        "num_samples": 1,
                                    }
                                ],
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    audit = stage211._audit_audio_bucket_storage(manifest)

    assert audit["webdataset_root"] == str(webdataset_root.resolve())
    assert audit["length_index_path"] == str(length_index.resolve())
    assert audit["bucket_part_files"] == 1
    assert audit["split_samples"] == {"train": 1}

    phase = stage211.PHASES["mixer"]
    segment = stage211._segments(phase=phase, smoke=False)[0]
    config = stage211._config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=tmp_path / "selected.pt",
        bucket_manifest=manifest,
        resume=False,
        smoke=False,
        audio_data_audit=audit,
    )
    assert config["webdataset_root"] == str(webdataset_root.resolve())
    assert config["webdataset_length_index_path"] == str(length_index.resolve())


def _config_for_phase(
    tmp_path: Path,
    phase_name: str,
    *,
    smoke: bool = False,
) -> dict[str, object]:
    phase = stage211.PHASES[phase_name]
    formal_steps = 12_045 if phase.requires_labels and not smoke else None
    segment = stage211._segments(
        phase=phase,
        smoke=smoke,
        formal_steps=formal_steps,
    )[0]
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    labeled_root = tmp_path / "labeled"
    labeled_root.mkdir(exist_ok=True)
    labeled_length_index = labeled_root / "webdataset_lengths.jsonl"
    labeled_length_index.touch()
    return stage211._config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / phase_name,
        init_checkpoint=checkpoint,
        bucket_manifest=manifest,
        resume=False,
        smoke=smoke,
        labeled_webdataset_root=labeled_root if phase.requires_labels else None,
        labeled_length_index=labeled_length_index if phase.requires_labels else None,
    )


def test_stage211_config_uses_explicit_nano_checkpoint_parent(tmp_path: Path) -> None:
    phase = stage211.PHASES["mixer"]
    segment = stage211._segments(phase=phase, smoke=False)[0]
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    nano_dir = tmp_path / "persistent-nano"
    nano_dir.mkdir()
    nano_checkpoint = nano_dir / "model.pt"
    nano_checkpoint.touch()

    config = stage211._config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / "mixer",
        init_checkpoint=checkpoint,
        bucket_manifest=manifest,
        resume=False,
        smoke=False,
        nano_checkpoint=nano_checkpoint,
    )

    assert config["ctc_teacher_online_model_path"] == str(nano_dir.resolve())


@pytest.mark.parametrize("phase_name", tuple(stage211.PHASES))
def test_stage211_freezes_nano_non_attention_path_in_every_phase(
    tmp_path: Path,
    phase_name: str,
) -> None:
    config = _config_for_phase(tmp_path, phase_name)

    assert validate_stage211_phase_train_config(
        config,
        phase=phase_name,
    ) == stage211_phase_train_config_contract(phase_name)
    expected_steps = 12_045 if phase_name == "sft" else 30_064
    expected_interval = 2_000 if phase_name == "sft" else 10_000
    assert config["max_steps"] == expected_steps
    assert config["save_every"] == stage211.FORMAL_RESUME_SAVE_INTERVAL
    assert config["step_eval_every"] == expected_interval
    assert config["top_k_step_checkpoints"] == (7 if phase_name == "sft" else 4)
    assert config["periodic_checkpoint_keep_last"] == 2
    assert config["weight_decay"] == 0.0
    assert config["freeze_encoder"] is False
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["funasr_nano_ctc_init_checkpoint_path"] is None
    expected_ffn_weight = {
        "mixer": 0.0,
        "block": 0.25,
        "logits": 0.10,
        "sft": 0.05,
    }[phase_name]
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(
        expected_ffn_weight
    )
    assert config["step_eval_cache_batches"] is True
    assert config["step_eval_feature_seed"] == 0
    if phase_name == "sft":
        assert config["length_bucket_drop_last"] is False
        assert config["skip_oversized_samples"] is False
        assert config["webdataset_skip_decode_errors"] is False


def test_stage211_phase_config_contract_rejects_cross_phase_objective(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "block")
    config["ctc_teacher_online_layer_input_mode"] = "teacher_forced"

    with pytest.raises(
        ValueError,
        match="block train config ctc_teacher_online_layer_input_mode mismatch",
    ):
        validate_stage211_phase_train_config(config, phase="block")


def test_stage211_mixer_phase_has_only_teacher_forced_mixer_objective(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "mixer")

    assert config["lr"] == pytest.approx(3.0e-6)
    assert config["ctc_teacher_online_layer_input_mode"] == "teacher_forced"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_block_loss_weight"] == 0.0
    assert config["ctc_teacher_online_encoder_loss_weight"] == 0.0
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == 0.0
    assert config["ctc_teacher_online_blank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0


def test_stage211_block_phase_chains_stacked_block_without_logits(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "block")

    assert config["lr"] == pytest.approx(2.0e-6)
    assert config["ctc_teacher_online_layer_input_mode"] == "stacked"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_encoder_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_blank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0


def test_stage211_logits_phase_enables_outputs_after_hidden_anchors(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "logits")

    assert config["lr"] == pytest.approx(3.0e-7)
    assert config["ctc_teacher_online_layer_input_mode"] == "stacked"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_conditional_nonblank_hard_loss_weight"] == pytest.approx(
        0.125
    )
    assert config["ctc_teacher_online_sequence_loss_weight"] == pytest.approx(0.20)
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_include_boundaries"] is True
    assert config["ctc_teacher_online_layer_boundary_ids"] == list(stage211.HARD_LAYER_IDS)


def test_stage211_sft_phase_uses_labels_after_logits_with_low_teacher_anchors(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "sft")

    assert config["lr"] == pytest.approx(3.0e-7)
    assert config["allow_missing_targets"] is False
    assert config["ctc_loss_weight"] == pytest.approx(1.0)
    assert config["decoder_loss_weight"] == 0.0
    assert config["ctc_suppress_non_pronunciation_tokens"] is True
    assert config["step_eval_split"] == "eval"
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_encoder_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0


def test_stage211_nano_non_attention_mapping_covers_all_expected_tensors() -> None:
    pairs = stage211._nano_non_attention_pairs()

    assert len(pairs) == 564
    assert len(set(pairs)) == 564
    assert (
        "encoder.sensevoice_encoder.layers.0.feed_forward.w_1.weight",
        "audio_encoder.encoders0.0.feed_forward.w_1.weight",
    ) in pairs
    assert (
        "encoder.sensevoice_encoder.layers.69.norm2.bias",
        "audio_encoder.tp_encoders.19.norm2.bias",
    ) in pairs


def test_stage211_block_entrypoint_requires_explicit_preceding_checkpoint(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"),
            "--phase",
            "block",
            "--dry-run",
            "--skip-nano-weight-audit",
            "--bucket-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--config-dir",
            str(tmp_path / "config"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "explicitly selected checkpoint from the preceding phase" in result.stderr


def test_stage211_block_entrypoint_requires_passing_promotion_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"),
            "--phase",
            "block",
            "--dry-run",
            "--skip-nano-weight-audit",
            "--init-checkpoint",
            str(checkpoint),
            "--bucket-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--config-dir",
            str(tmp_path / "config"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires a passing promotion receipt" in result.stderr


def test_stage211_mixer_entrypoint_dry_run_is_runnable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"),
            "--phase",
            "mixer",
            "--dry-run",
            "--skip-nano-weight-audit",
            "--init-checkpoint",
            str(checkpoint),
            "--bucket-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--config-dir",
            str(tmp_path / "config"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "phase=mixer" in result.stdout
    assert "target_step=30064" in result.stdout


def _write_runtime_epoch_coverage(
    tmp_path: Path,
    *,
    prefix: str,
    epochs: int,
    steps_per_epoch: int,
) -> dict[str, object]:
    records = []
    for epoch in range(1, epochs + 1):
        checkpoint = tmp_path / f"{prefix}-epoch-{epoch}.pt"
        checkpoint.write_bytes(f"{prefix}-{epoch}".encode())
        records.append(
            {
                "epoch": epoch,
                "step": epoch * steps_per_epoch,
                "epoch_batch_offset": 0,
                "completed_epoch_batch_count": steps_per_epoch,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "runtime_epoch_coverage",
        "complete": True,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": epochs * steps_per_epoch,
        "records": records,
    }


def _write_supplemental_inventory_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "supplemental-inventory-fixture"
    inventory_path = root / "supplemental_inventory.json"
    if inventory_path.is_file():
        return inventory_path, stage211_supplemental_profile(
            inventory_path,
            epochs=STAGE211_FULL_DATA_EPOCHS,
            batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        )

    root.mkdir()
    train_part = root / "train.jsonl"
    train_part.write_text(
        "".join(json.dumps({"key": f"supplemental-{index}"}) + "\n" for index in range(5)),
        encoding="utf-8",
    )
    fixed_eval_part = root / "fixed_eval.jsonl"
    fixed_eval_part.write_text("{}\n" * 256, encoding="utf-8")
    fixed_eval_manifest = root / "fixed_eval_manifest.json"
    fixed_eval_manifest.write_text("{}\n", encoding="utf-8")
    stage179_manifest = root / "stage179_global_dedup.json"
    stage179_manifest.write_text("{}\n", encoding="utf-8")
    bucket_manifest = root / "manifest_stage211_fixed_eval.json"
    bucket_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(train_part.resolve()),
                "bucket_width": 80,
                "entries_per_part": 5,
                "splits": {
                    "train": {
                        "num_samples": 5,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 5,
                                "parts": [
                                    {
                                        "path": train_part.name,
                                        "num_samples": 5,
                                        "source_label": "supplemental_fixture",
                                    }
                                ],
                            }
                        ],
                    },
                    "eval": {
                        "num_samples": 256,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 256,
                                "parts": [
                                    {
                                        "path": fixed_eval_part.name,
                                        "num_samples": 256,
                                    }
                                ],
                            }
                        ],
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_seconds = 1.0
    selected_hours_by_source = {
        source: source_seconds / 3600.0 for source in STAGE211_SUPPLEMENTAL_SOURCES
    }
    inventory = {
        "schema_version": 1,
        "artifact": "stage211_supplemental_natural_inventory",
        "complete": True,
        "training_ready": True,
        "hash_archives": True,
        "require_production_layout": True,
        "language": "en",
        "uses_text_labels": False,
        "storage_kinds": ["parquet", "zip"],
        "layout_errors": [],
        "selected_rows": 5,
        "selected_hours": sum(selected_hours_by_source.values()),
        "selected_counts_by_source": {
            source: 1 for source in STAGE211_SUPPLEMENTAL_SOURCES
        },
        "selected_hours_by_source": selected_hours_by_source,
        "dedupe": {
            "algorithm": "blake2b16(corpus + NUL + source_identity)",
            "accepted_unique_rows": 5,
        },
        "cross_pool_dedupe": {
            "mode": "source_identity_plus_known_corpus_exclusion",
            "content_fingerprint_complete": False,
            "source_sets_disjoint": True,
            "supplemental_sources": sorted(STAGE211_SUPPLEMENTAL_SOURCES),
            "known_overlap_exclusions": ["llaso_gigaspeech", "llaso_librispeech"],
            "stage179": {
                "expected_source_set_match": True,
                "manifest_path": str(stage179_manifest.resolve()),
                "manifest_sha256": sha256_file(stage179_manifest),
            },
        },
        "fixed_eval": {
            "rows": 256,
            "manifest_path": str(fixed_eval_manifest.resolve()),
            "manifest_sha256": sha256_file(fixed_eval_manifest),
            "parts": [
                {
                    "path": str(fixed_eval_part.resolve()),
                    "sha256": sha256_file(fixed_eval_part),
                    "num_samples": 256,
                }
            ],
        },
        "bucket_manifest_path": str(bucket_manifest.resolve()),
        "bucket_manifest_sha256": sha256_file(bucket_manifest),
        "part_records": [
            {
                "path": str(train_part.resolve()),
                "num_samples": 5,
                "size_bytes": train_part.stat().st_size,
                "sha256": sha256_file(train_part),
            }
        ],
        "archive_records": [{"status": "absent"}],
    }
    inventory_path.write_text(json.dumps(inventory) + "\n", encoding="utf-8")
    return inventory_path, stage211_supplemental_profile(
        inventory_path,
        epochs=STAGE211_FULL_DATA_EPOCHS,
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
    )


def test_stage211_supplemental_profile_receipt_is_complete_and_immutable(
    tmp_path: Path,
) -> None:
    inventory_path, profile = _write_supplemental_inventory_fixture(tmp_path)
    receipt = stage211_supplemental_profile_receipt.build_receipt(inventory_path)

    assert receipt["complete"] is True
    assert receipt["inventory_sha256"] == sha256_file(inventory_path)
    assert receipt["rows"] == profile["rows"]
    assert receipt["hours"] == profile["hours"]
    assert receipt["epochs"] == STAGE211_FULL_DATA_EPOCHS
    assert receipt["steps"] == profile["steps"]
    assert receipt["train_part_count"] == 1
    assert receipt["archive_status_counts"] == {"unknown:absent": 1}

    output = tmp_path / "supplemental-profile-receipt.json"
    stage211_supplemental_profile_receipt.write_immutable_receipt(output, receipt)
    stage211_supplemental_profile_receipt.write_immutable_receipt(output, receipt)
    validated = stage211_full_phase._validate_supplemental_profile_receipt(
        output,
        inventory_path=inventory_path,
    )
    assert validated["sha256"] == sha256_file(output)
    changed = {**receipt, "steps": int(receipt["steps"]) + 1}
    with pytest.raises(ValueError, match="Refusing to replace a different"):
        stage211_supplemental_profile_receipt.write_immutable_receipt(output, changed)
    forged = tmp_path / "forged-supplemental-profile-receipt.json"
    forged.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the current inventory"):
        stage211_full_phase._validate_supplemental_profile_receipt(
            forged,
            inventory_path=inventory_path,
        )

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    train_part = Path(inventory["part_records"][0]["path"])
    payload = train_part.read_bytes()
    train_part.write_bytes(b"X" + payload[1:])
    with pytest.raises(ValueError, match="supplemental train part changed"):
        stage211_supplemental_profile_receipt.build_receipt(inventory_path)


def _write_supplemental_coverage_fixture(
    tmp_path: Path,
    *,
    phase: str,
    init_checkpoint: Path,
    completion_checkpoint: Path,
    nano_teacher_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    inventory_path, profile = _write_supplemental_inventory_fixture(tmp_path)
    provenance = tmp_path / "supplemental-provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    train_config = tmp_path / "supplemental-train-config.yaml"
    train_config_payload = stage211_phase_train_config_contract(phase)
    train_config_payload.update(
        {
            "ctc_teacher_online_model_path": str(nano_teacher_dir.resolve()),
            "max_steps": int(profile["steps"]),
            "webdataset_bucket_manifest_path": profile["bucket_manifest_path"],
        }
    )
    save_yaml(train_config, train_config_payload)
    runtime_epoch_coverage = _write_runtime_epoch_coverage(
        tmp_path,
        prefix=f"{phase}-{STAGE211_SUPPLEMENTAL_DIFFICULTY}",
        epochs=STAGE211_FULL_DATA_EPOCHS,
        steps_per_epoch=int(profile["steps_per_epoch"]),
    )
    nano_teacher_checkpoint = nano_teacher_dir / "model.pt"
    segment: dict[str, object] = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "phase": phase,
        "difficulty": STAGE211_SUPPLEMENTAL_DIFFICULTY,
        "complete": True,
        "full_data_profile": True,
        "epochs": STAGE211_FULL_DATA_EPOCHS,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "rows": profile["rows"],
        "row_exposures": profile["row_exposures"],
        "tail_padding_samples_per_epoch": profile["tail_padding_samples_per_epoch"],
        "tail_padding_sample_exposures": profile["tail_padding_sample_exposures"],
        "executed_sample_exposures": profile["executed_sample_exposures"],
        "hours": profile["hours"],
        "hour_exposures": profile["hour_exposures"],
        "steps_per_epoch": profile["steps_per_epoch"],
        "steps": profile["steps"],
        "supplemental_inventory_path": str(inventory_path.resolve()),
        "supplemental_inventory_sha256": sha256_file(inventory_path),
        "provenance_path": str(provenance.resolve()),
        "provenance_sha256": sha256_file(provenance),
        "train_config_path": str(train_config.resolve()),
        "train_config_sha256": sha256_file(train_config),
        "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_teacher_checkpoint),
        "bucket_manifest_path": profile["bucket_manifest_path"],
        "bucket_manifest_sha256": profile["bucket_manifest_sha256"],
        "init_checkpoint_path": str(init_checkpoint.resolve()),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "completion_checkpoint_path": str(completion_checkpoint.resolve()),
        "completion_checkpoint_sha256": sha256_file(completion_checkpoint),
        "runtime_epoch_coverage": runtime_epoch_coverage,
        "parameter_delta_audit": {
            "schema_version": 1,
            "policy": "stage211_timemixer_and_input_projection_only",
            "complete": True,
            "allowed_key_markers": list(STAGE211_ALLOWED_OPERATOR_KEY_MARKERS),
            "initial_tensor_count": 4,
            "completion_tensor_count": 4,
            "allowed_changed_tensors": 1,
            "allowed_changed_numel": 1,
            "allowed_unchanged_tensors": 1,
            "frozen_unchanged_tensors": 2,
            "forbidden_changed_tensors": 0,
        },
    }
    receipt = tmp_path / "supplemental-receipt.json"
    receipt.write_text(json.dumps(segment) + "\n", encoding="utf-8")
    segment["receipt_path"] = str(receipt.resolve())
    segment["receipt_sha256"] = sha256_file(receipt)
    return segment, profile


def test_stage211_medium_curriculum_requires_receipt_and_uses_full_steps(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "easy-complete.pt"
    checkpoint.write_bytes(b"easy")
    nano_dir = tmp_path / "nano"
    nano_dir.mkdir()
    nano_checkpoint = nano_dir / "model.pt"
    nano_checkpoint.write_bytes(b"nano")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"),
        "--phase",
        "mixer",
        "--difficulty",
        "medium",
        "--full-data-profile",
        "--dry-run",
        "--skip-nano-weight-audit",
        "--init-checkpoint",
        str(checkpoint),
        "--bucket-manifest",
        str(manifest),
        "--nano-checkpoint",
        str(nano_checkpoint),
        "--output-dir",
        str(tmp_path / "run"),
        "--config-dir",
        str(tmp_path / "config"),
    ]
    missing = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert missing.returncode != 0
    assert "preceding curriculum coverage receipt" in missing.stderr

    receipt = tmp_path / "easy-coverage.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "curriculum_coverage",
                "phase": "mixer",
                "difficulty": "easy",
                "complete": True,
                "full_data_profile": True,
                "completion_checkpoint_path": str(checkpoint.resolve()),
                "completion_checkpoint_sha256": sha256_file(checkpoint),
                "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
                "runtime_epoch_coverage": _write_runtime_epoch_coverage(
                    tmp_path,
                    prefix="medium-admission-easy",
                    epochs=STAGE211_FULL_DATA_EPOCHS,
                    steps_per_epoch=int(STAGE211_AUDIO_CURRICULUM["easy"]["steps_per_epoch"]),
                ),
            }
        ),
        encoding="utf-8",
    )
    admitted = subprocess.run(
        [*command, "--curriculum-receipt", str(receipt)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert admitted.returncode == 0, admitted.stderr
    assert "difficulty=medium" in admitted.stdout
    assert "full_data_profile=true" in admitted.stdout
    assert "target_step=1003713" in admitted.stdout


def test_stage211_medium_config_uses_fixed_eval_split(tmp_path: Path) -> None:
    phase = stage211.PHASES["mixer"]
    segment = stage211._segments(
        phase=phase,
        smoke=False,
        difficulty="medium",
        full_data_profile=True,
    )[0]
    checkpoint = tmp_path / "selected.pt"
    checkpoint.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    config = stage211._config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / "run",
        init_checkpoint=checkpoint,
        bucket_manifest=manifest,
        resume=False,
        smoke=False,
        full_data_profile=True,
    )

    assert config["max_steps"] == 1_003_713
    assert config["batch_size"] == 36
    assert config["batch_token_budget"] == 24_000
    assert config["length_bucket_drop_last"] is False
    assert config["skip_oversized_samples"] is False
    assert config["webdataset_skip_decode_errors"] is False
    assert config["length_bucket_frame_budget"] == 24_000
    assert config["deepspeed"]["train_micro_batch_size_per_gpu"] == 36
    assert config["deepspeed"]["train_batch_size"] == 144
    assert config["step_eval_split"] == "eval"
    assert config["top_k_step_checkpoints"] == 8
    assert config["periodic_checkpoint_keep_last"] == 2


def _write_minimal_labeled_data(tmp_path: Path) -> tuple[Path, Path, Path]:
    webdataset_root = tmp_path / "webdataset"
    webdataset_root.mkdir()
    length_index = tmp_path / "webdataset_lengths.jsonl"
    rows = []
    for split, utt_id in (("train", "train-1"), ("eval", "eval-1")):
        rows.append(
            {
                "key": utt_id,
                "utt_id": utt_id,
                "split": split,
                "num_frames": 100,
                "json_member": f"{utt_id}.json",
                "normalized_text_chars": 5,
                "ctc_num_tokens": 5,
                "ctc_unk_tokens": 0,
                "ctc_required_frames": 5,
                "ctc_logit_frames": 16,
            }
        )
    length_index.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": str(tmp_path),
                "source_length_index_path": str(length_index),
                "bucket_width": 100,
                "entries_per_part": 1000,
                "splits": {
                    split: {
                        "num_samples": 1,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 1,
                                "parts": [],
                            }
                        ],
                    }
                    for split in ("train", "eval")
                },
            }
        ),
        encoding="utf-8",
    )
    (length_index.parent / "webdataset_lengths.summary.json").write_text(
        json.dumps(
            {
                "version": 1,
                "output_dir": str(webdataset_root.resolve()),
                "length_index_path": str(length_index.resolve()),
                "tokenizer_type": "sensevoice_tiktoken",
                "tokenizer_model_path": str(
                    (REPO_ROOT / "assets/fun-asr-nano-2512/multilingual.tiktoken").resolve()
                ),
                "text_normalization": "ctc",
                "frontend_downsample": "sensevoice_lfr6",
                "drop_unk_token": True,
                "unk_token_id": None,
                "num_input_samples": 285_302,
                "num_kept_samples": 285_302,
                "num_dropped_samples": 0,
                "counts": {
                    "input_by_split": {"eval": 1_434, "train": 283_868},
                    "kept_by_split": {"eval": 1_434, "train": 283_868},
                    "kept_by_source": {"aishell3": 63_262, "librispeech": 222_040},
                    "kept_by_language": {"en": 222_040, "zh": 63_262},
                    "kept_by_split_source": {
                        "eval/aishell3": 310,
                        "eval/librispeech": 1_124,
                        "train/aishell3": 62_952,
                        "train/librispeech": 220_916,
                    },
                    "kept_by_split_language": {
                        "eval/en": 1_124,
                        "eval/zh": 310,
                        "train/en": 220_916,
                        "train/zh": 62_952,
                    },
                    "dropped_by_reason": {},
                    "dropped_by_source": {},
                    "dropped_by_language": {},
                    "dropped_unk_tokens_by_source": {},
                    "dropped_unk_tokens_by_language": {},
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (length_index.parent / "prepare_ctc_aligned.log").write_text(
        "tokenizer_type=sensevoice_tiktoken\n"
        "text_normalization=ctc\n"
        "frontend_downsample=sensevoice_lfr6\n"
        "drop_unk_token=1\n"
        "CTC-aligned clean preprocessing complete\n",
        encoding="utf-8",
    )
    return webdataset_root, length_index, manifest


def test_stage211_sft_labeled_data_audit_and_epoch_estimate(tmp_path: Path) -> None:
    webdataset_root, length_index, manifest = _write_minimal_labeled_data(tmp_path)

    audit = stage211._audit_labeled_data(
        webdataset_root=webdataset_root,
        length_index_path=length_index,
        bucket_manifest_path=manifest,
    )

    assert audit["train_samples"] == 1
    assert audit["eval_samples"] == 1
    assert audit["ctc_tokens"] == 10
    assert audit["ctc_unk_tokens"] == 0
    assert audit["unique_utterance_ids"] == 2
    assert audit["pronunciation_target_samples"] == 2
    assert audit["ctc_feasible_samples"] == 2
    assert audit["label_preparation"]["text_normalization"] == "ctc"
    assert audit["estimated_train_steps"] == 1
    assert audit["tail_padding_samples_per_epoch"] == 47
    assert audit["executed_sample_exposures"] == 48


def _canonical_json_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def _write_loaded_manifest_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "loaded-manifest-fixture"
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        return receipt_path, {
            difficulty: root / difficulty / "manifest_stage211_fixed_eval.json"
            for difficulty in STAGE211_AUDIO_CURRICULUM
        }
    root.mkdir()
    fixed_eval = root / "fixed-eval.jsonl"
    fixed_eval.write_text("{}\n", encoding="utf-8")
    global_manifest = json.loads(GLOBAL_DEDUP_FIXTURE.read_text(encoding="utf-8"))
    global_stages = global_manifest["stages"]
    segments = []
    runtime_manifests: dict[str, Path] = {}
    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        stage_name, global_stage = next(
            (
                (name, stage)
                for name, stage in global_stages.items()
                if stage["difficulty"] == difficulty
            )
        )
        segment_root = root / difficulty
        part_root = segment_root / "train"
        part_root.mkdir(parents=True)
        part_count = STAGE211_AUDIO_TRAIN_PART_COUNTS[difficulty]
        remaining_rows = int(expected["rows"])
        manifest_parts = []
        part_records = []
        for index in range(part_count):
            part_rows = remaining_rows - (part_count - index - 1) if index == 0 else 1
            remaining_rows -= part_rows
            part_path = part_root / f"part-{index:04d}.jsonl"
            part_path.write_text("{}\n", encoding="utf-8")
            stat = part_path.stat()
            manifest_parts.append(
                {
                    "path": str(part_path.relative_to(segment_root)),
                    "num_samples": part_rows,
                }
            )
            part_records.append(
                {
                    "path": str(part_path.resolve()),
                    "rows": part_rows,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(part_path),
                }
            )
        source_manifest = {
            "version": 1,
            "root": "/",
            "source_length_index_path": global_stage["length_index_path"],
            "splits": {
                "train": {
                    "num_samples": int(expected["rows"]),
                    "buckets": [
                        {
                            "bucket_id": 0,
                            "num_samples": int(expected["rows"]),
                            "parts": manifest_parts,
                        }
                    ],
                }
            },
        }
        source_manifest_path = segment_root / "manifest.json"
        source_manifest_path.write_text(json.dumps(source_manifest) + "\n", encoding="utf-8")
        runtime_manifest = json.loads(json.dumps(source_manifest))
        runtime_manifest["splits"]["eval"] = {
            "num_samples": 256,
            "buckets": [
                {
                    "bucket_id": 0,
                    "num_samples": 256,
                    "parts": [{"path": str(fixed_eval.resolve()), "num_samples": 256}],
                }
            ],
        }
        runtime_manifest_path = segment_root / "manifest_stage211_fixed_eval.json"
        runtime_manifest_path.write_text(json.dumps(runtime_manifest) + "\n", encoding="utf-8")
        runtime_manifests[difficulty] = runtime_manifest_path
        global_source_path = Path(global_stage["bucket_manifest_path"])
        global_source_sha256 = (
            sha256_file(global_source_path) if global_source_path.is_file() else "0" * 64
        )
        key_set_sha256 = sha256_file(part_records[0]["path"])
        segments.append(
            {
                "difficulty": difficulty,
                "global_stage_name": stage_name,
                "train_rows": int(expected["rows"]),
                "train_hours": float(global_stage["selected_hours"]),
                "train_parts": part_count,
                "steps_per_epoch": int(expected["steps_per_epoch"]),
                "epochs": STAGE211_FULL_DATA_EPOCHS,
                "total_steps": int(expected["steps"]),
                "tail_padding_samples_per_epoch": int(expected["tail_padding_samples_per_epoch"]),
                "source_length_index_path": global_stage["length_index_path"],
                "global_source_manifest_path": str(global_source_path),
                "global_source_manifest_sha256": global_source_sha256,
                "source_manifest_path": str(source_manifest_path.resolve()),
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "runtime_manifest_path": str(runtime_manifest_path.resolve()),
                "runtime_manifest_sha256": sha256_file(runtime_manifest_path),
                "train_split_sha256": _canonical_json_sha256(runtime_manifest["splits"]["train"]),
                "repartitioned": True,
                "loaded_audio_key_set_sha256": key_set_sha256,
                "global_audio_key_set_sha256": key_set_sha256,
                "part_records_sha256": _canonical_json_sha256(part_records),
                "part_records": part_records,
            }
        )
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "loaded_manifest_chain",
        "complete": True,
        "part_hash_algorithm": "sha256",
        "part_hash_audit_complete": True,
        "global_dedup_manifest_path": str(GLOBAL_DEDUP_FIXTURE.resolve()),
        "global_dedup_manifest_sha256": sha256_file(GLOBAL_DEDUP_FIXTURE),
        "dedupe_key": global_manifest["dedupe_key"],
        "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
        "total_hours": STAGE211_GLOBAL_DEDUP_TOTAL_HOURS,
        "total_train_parts": sum(STAGE211_AUDIO_TRAIN_PART_COUNTS.values()),
        "fixed_eval": {
            "path": str(fixed_eval.resolve()),
            "sha256": sha256_file(fixed_eval),
            "rows": 256,
        },
        "segments": segments,
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    return receipt_path, runtime_manifests


def test_stage211_loaded_manifest_receipt_binds_actual_runtime_manifests(
    tmp_path: Path,
) -> None:
    receipt_path, runtime_manifests = _write_loaded_manifest_fixture(tmp_path)

    receipt = validate_stage211_loaded_manifest_receipt(
        receipt_path,
        expected_global_dedup_manifest=GLOBAL_DEDUP_FIXTURE,
    )

    assert receipt["total_unique_rows"] == STAGE211_AUDIO_TOTAL_ROWS
    assert receipt["total_train_parts"] == 745
    assert {
        segment["difficulty"]: Path(segment["runtime_manifest_path"])
        for segment in receipt["segments"]
    } == runtime_manifests


def test_stage211_loaded_manifest_receipt_rejects_runtime_manifest_rewrite(
    tmp_path: Path,
) -> None:
    receipt_path, runtime_manifests = _write_loaded_manifest_fixture(tmp_path)
    runtime_manifests["hard"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hard runtime manifest changed"):
        validate_stage211_loaded_manifest_receipt(
            receipt_path,
            expected_global_dedup_manifest=GLOBAL_DEDUP_FIXTURE,
        )


def _write_public_overlap_fixture(tmp_path: Path) -> tuple[Path, Path]:
    clean_rows = [
        {"utt_id": f"clean-{index:05d}", "text": "clean"}
        for index in range(int(STAGE211_PUBLIC_BENCHMARKS["commonvoice_en_test"]["samples"]))
    ]
    leaked_id = "leaked-test-audio"
    original_manifest = tmp_path / "commonvoice-full-contaminated.jsonl"
    clean_manifest = tmp_path / "commonvoice-clean.jsonl"
    candidate_rows = tmp_path / "commonvoice-overlap-candidates.jsonl"
    exclusions = tmp_path / "commonvoice-exclusions.jsonl"
    original_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in [*clean_rows, {"utt_id": leaked_id}]),
        encoding="utf-8",
    )
    clean_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in clean_rows),
        encoding="utf-8",
    )
    candidate_rows.write_text(
        json.dumps({"byte_identical_public_ids": [leaked_id]}) + "\n",
        encoding="utf-8",
    )
    exclusions.write_text(
        json.dumps({"public_utt_id": leaked_id}) + "\n",
        encoding="utf-8",
    )
    source_files = {}
    for name in ("test.tsv", "converter.py", "stage178.jsonl"):
        source = tmp_path / name
        source.write_text("fixture\n", encoding="utf-8")
        source_files[name] = source
    receipt = tmp_path / "public-overlap-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "artifact": "stage211_public_train_overlap_audit",
                "audio_hash_algorithm": "sha256_exact_encoded_mp3_bytes",
                "complete": True,
                "coverage": {
                    "candidate_training_rows": 1,
                    "clean_public_rows": len(clean_rows),
                    "exact_byte_identical_training_rows": 1,
                    "excluded_public_rows": 1,
                    "public_rows": len(clean_rows) + 1,
                    "stage178_english_rows": 1,
                },
                "dataset": "commonvoice_en_test",
                "outputs": {
                    "candidate_rows": {
                        "path": str(candidate_rows.resolve()),
                        "rows": 1,
                        "sha256": sha256_file(candidate_rows),
                    },
                    "clean_manifest": {
                        "path": str(clean_manifest.resolve()),
                        "rows": len(clean_rows),
                        "sha256": sha256_file(clean_manifest),
                    },
                    "exclusions": {
                        "path": str(exclusions.resolve()),
                        "rows": 1,
                        "sha256": sha256_file(exclusions),
                    },
                },
                "pipeline": "stage211",
                "schema_version": 1,
                "source_bindings": {
                    "commonvoice_test_tsv_path": str(source_files["test.tsv"].resolve()),
                    "commonvoice_test_tsv_sha256": sha256_file(source_files["test.tsv"]),
                    "converter_source_path": str(source_files["converter.py"].resolve()),
                    "converter_source_sha256": sha256_file(source_files["converter.py"]),
                    "loaded_manifest": None,
                    "public_manifest_path": str(original_manifest.resolve()),
                    "public_manifest_sha256": sha256_file(original_manifest),
                    "stage178_index_path": str(source_files["stage178.jsonl"].resolve()),
                    "stage178_index_sha256": sha256_file(source_files["stage178.jsonl"]),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt, clean_manifest


def _write_valid_phase_gate(
    tmp_path: Path,
    *,
    phase: str,
    checkpoint: Path,
) -> Path:
    loaded_manifest_receipt, runtime_manifests = _write_loaded_manifest_fixture(tmp_path)
    public_overlap_receipt, clean_commonvoice_manifest = _write_public_overlap_fixture(tmp_path)
    segments = []
    nano_teacher_dir = tmp_path / "nano-teacher"
    nano_teacher_dir.mkdir()
    nano_teacher_checkpoint = nano_teacher_dir / "model.pt"
    nano_teacher_checkpoint.write_bytes(b"nano-teacher")
    previous_checkpoint = tmp_path / "phase-init.pt"
    previous_checkpoint.write_bytes(b"phase-init")
    for index, (difficulty, expected) in enumerate(STAGE211_AUDIO_CURRICULUM.items()):
        manifest = runtime_manifests[difficulty]
        provenance = tmp_path / f"{difficulty}-provenance.json"
        provenance.write_text("{}\n", encoding="utf-8")
        train_config = tmp_path / f"{difficulty}-train-config.yaml"
        train_config_payload = stage211_phase_train_config_contract(phase)
        train_config_payload["ctc_teacher_online_model_path"] = str(nano_teacher_dir.resolve())
        save_yaml(train_config, train_config_payload)
        completion = tmp_path / f"{difficulty}.pt"
        completion.write_bytes(f"checkpoint-{index}".encode())
        runtime_epoch_coverage = _write_runtime_epoch_coverage(
            tmp_path,
            prefix=f"{phase}-{difficulty}",
            epochs=STAGE211_FULL_DATA_EPOCHS,
            steps_per_epoch=int(expected["steps_per_epoch"]),
        )
        segment = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "curriculum_coverage",
            "phase": phase,
            "difficulty": difficulty,
            "complete": True,
            "full_data_profile": True,
            "epochs": STAGE211_FULL_DATA_EPOCHS,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
            "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
            "rows": expected["rows"],
            "row_exposures": expected["rows"] * STAGE211_FULL_DATA_EPOCHS,
            "tail_padding_samples_per_epoch": expected["tail_padding_samples_per_epoch"],
            "tail_padding_sample_exposures": (
                expected["tail_padding_samples_per_epoch"] * STAGE211_FULL_DATA_EPOCHS
            ),
            "executed_sample_exposures": (
                (expected["rows"] + expected["tail_padding_samples_per_epoch"])
                * STAGE211_FULL_DATA_EPOCHS
            ),
            "hours": expected["hours"],
            "hour_exposures": expected["hours"] * STAGE211_FULL_DATA_EPOCHS,
            "steps_per_epoch": expected["steps_per_epoch"],
            "steps": expected["steps"],
            "provenance_path": str(provenance.resolve()),
            "provenance_sha256": sha256_file(provenance),
            "train_config_path": str(train_config.resolve()),
            "train_config_sha256": sha256_file(train_config),
            "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
            "nano_teacher_checkpoint_sha256": sha256_file(nano_teacher_checkpoint),
            "bucket_manifest_path": str(manifest.resolve()),
            "bucket_manifest_sha256": sha256_file(manifest),
            "init_checkpoint_path": str(previous_checkpoint.resolve()),
            "init_checkpoint_sha256": sha256_file(previous_checkpoint),
            "completion_checkpoint_path": str(completion.resolve()),
            "completion_checkpoint_sha256": sha256_file(completion),
            "runtime_epoch_coverage": runtime_epoch_coverage,
            "parameter_delta_audit": {
                "schema_version": 1,
                "policy": "stage211_timemixer_and_input_projection_only",
                "complete": True,
                "allowed_key_markers": list(STAGE211_ALLOWED_OPERATOR_KEY_MARKERS),
                "initial_tensor_count": 4,
                "completion_tensor_count": 4,
                "allowed_changed_tensors": 1,
                "allowed_changed_numel": 1,
                "allowed_unchanged_tensors": 1,
                "frozen_unchanged_tensors": 2,
                "forbidden_changed_tensors": 0,
            },
        }
        receipt = tmp_path / f"{difficulty}-receipt.json"
        receipt.write_text(json.dumps(segment) + "\n", encoding="utf-8")
        segment["receipt_path"] = str(receipt.resolve())
        segment["receipt_sha256"] = sha256_file(receipt)
        segments.append(segment)
        previous_checkpoint = completion

    supplemental_segment, _ = _write_supplemental_coverage_fixture(
        tmp_path,
        phase=phase,
        init_checkpoint=previous_checkpoint,
        completion_checkpoint=checkpoint,
        nano_teacher_dir=nano_teacher_dir,
    )
    full_data_coverage = build_stage211_full_data_coverage(
        phase=phase,
        segments=segments,
        supplemental_segment=supplemental_segment,
        checkpoint_path=checkpoint,
    )

    public_results = []
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        manifest = (
            clean_commonvoice_manifest
            if dataset == "commonvoice_en_test"
            else tmp_path / f"{dataset}.jsonl"
        )
        nano = tmp_path / f"{dataset}.nano.jsonl"
        student = tmp_path / f"{dataset}.student.jsonl"
        for path in (nano, student):
            path.write_text("{}\n", encoding="utf-8")
        if dataset != "commonvoice_en_test":
            manifest.write_text("{}\n", encoding="utf-8")
        public_results.append(
            {
                "dataset": dataset,
                "language": expected["language"],
                "metric": expected["metric"],
                "sample_count": expected["samples"],
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "metric_source_recomputed": True,
                "nano_error_rate": 0.1,
                "student_error_rate": 0.11,
                "absolute_gap_points": 1.0,
                "relative_ratio": 1.1,
                "nano_prediction_reference_unit_ratio": 1.0,
                "student_prediction_reference_unit_ratio": 0.99,
                "nano_deletion_rate": 0.01,
                "student_deletion_rate": 0.02,
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": sha256_file(manifest),
                "nano_prediction_path": str(nano.resolve()),
                "nano_prediction_sha256": sha256_file(nano),
                "student_prediction_path": str(student.resolve()),
                "student_prediction_sha256": sha256_file(student),
            }
        )

    baseline_results = []
    for result in public_results:
        dataset = str(result["dataset"])
        report = tmp_path / f"{dataset}.nano-report.json"
        report.write_text(
            json.dumps(
                {
                    "version": 1,
                    "system": "FunASR-Nano-2512 direct CTC",
                    "model_path": str(nano_teacher_dir.resolve()),
                    "manifest_path": result["manifest_path"],
                    "predictions_path": result["nano_prediction_path"],
                    "language": result["language"],
                    "normalization": "ctc",
                    "decode": "greedy_ctc",
                    "requested_limit": None,
                    "sample_count": result["sample_count"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        baseline_results.append(
            {
                "dataset": dataset,
                "language": result["language"],
                "metric": result["metric"],
                "sample_count": result["sample_count"],
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "report_embeds_checkpoint_sha256": False,
                "report_path": str(report.resolve()),
                "report_sha256": sha256_file(report),
                "manifest_path": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
                "nano_prediction_path": result["nano_prediction_path"],
                "nano_prediction_sha256": result["nano_prediction_sha256"],
            }
        )
    baseline_receipt = tmp_path / "nano-baseline-receipt.json"
    baseline_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "nano_public_baseline_provenance",
                "complete": True,
                "provenance_mode": "legacy_report_attestation",
                "nano_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
                "nano_checkpoint_sha256": sha256_file(nano_teacher_checkpoint),
                "public_overlap": {
                    "receipt_path": str(public_overlap_receipt.resolve()),
                    "receipt_sha256": sha256_file(public_overlap_receipt),
                },
                "total_samples": sum(
                    int(expected["samples"]) for expected in STAGE211_PUBLIC_BENCHMARKS.values()
                ),
                "results": baseline_results,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    alignment_baseline_source = tmp_path / f"{phase}-alignment-baseline.yaml"
    alignment_candidate_source = tmp_path / f"{phase}-alignment-candidate.yaml"
    alignment_manifest = tmp_path / f"{phase}-alignment-manifest.json"
    alignment_part = tmp_path / f"{phase}-alignment-part.jsonl"
    alignment_model_config = tmp_path / f"{phase}-alignment-model.yaml"
    alignment_manifest.write_text("{}\n", encoding="utf-8")
    alignment_part.write_text("{}\n", encoding="utf-8")
    alignment_model_config.write_text("{}\n", encoding="utf-8")
    alignment_provenance = {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": 256,
        "feature_seed": 0,
        "bucket_manifest_path": str(alignment_manifest.resolve()),
        "bucket_manifest_sha256": sha256_file(alignment_manifest),
        "split_samples": 256,
        "parts": [
            {
                "path": str(alignment_part.resolve()),
                "sha256": sha256_file(alignment_part),
                "num_samples": 256,
            }
        ],
    }
    phase_init_checkpoint = Path(segments[0]["init_checkpoint_path"])
    alignment_train_config = Path(segments[0]["train_config_path"])
    pair_eval_id = "a" * 64
    shared_source = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "alignment_checkpoint_eval",
        "phase": phase,
        "pair_eval_id": pair_eval_id,
        "train_config_path": str(alignment_train_config.resolve()),
        "train_config_sha256": sha256_file(alignment_train_config),
        "model_config_path": str(alignment_model_config.resolve()),
        "model_config_sha256": sha256_file(alignment_model_config),
        "nano_checkpoint_path": str(nano_teacher_checkpoint.resolve()),
        "nano_checkpoint_sha256": sha256_file(nano_teacher_checkpoint),
        "feature_seed": 0,
        "eval_samples": 256,
        "eval_provenance": alignment_provenance,
    }
    alignment_baseline_source.write_text(
        json.dumps(
            {
                **shared_source,
                "role": "baseline",
                "step": 0,
                "checkpoint_step": 0,
                "checkpoint_path": str(phase_init_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(phase_init_checkpoint),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    alignment_candidate_source.write_text(
        json.dumps(
            {
                **shared_source,
                "role": "candidate",
                "step": int(STAGE211_AUDIO_CURRICULUM["long"]["steps"]),
                "checkpoint_step": int(STAGE211_AUDIO_CURRICULUM["long"]["steps"]),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    alignment_report = tmp_path / f"{phase}-alignment.json"
    alignment_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": (
                    "logits_alignment_gate" if phase == "logits" else "hidden_alignment_gate"
                ),
                "phase": phase,
                "baseline_checkpoint_path": str(phase_init_checkpoint.resolve()),
                "baseline_checkpoint_sha256": sha256_file(phase_init_checkpoint),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "gate_passed": True,
                "baseline_report_path": str(alignment_baseline_source.resolve()),
                "baseline_report_sha256": sha256_file(alignment_baseline_source),
                "candidate_report_path": str(alignment_candidate_source.resolve()),
                "candidate_report_sha256": sha256_file(alignment_candidate_source),
                "baseline_eval_provenance": alignment_provenance,
                "candidate_eval_provenance": alignment_provenance,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    smoke_dir = tmp_path / f"{phase}-full-profile-smoke"
    smoke_log_dir = smoke_dir / "logs"
    smoke_log_dir.mkdir(parents=True)
    smoke_checkpoint = smoke_dir / "step-2.pt"
    smoke_log = smoke_log_dir / f"{phase}_smoke_2steps.log"
    torch.save({"step": 2}, smoke_checkpoint)
    smoke_log.write_text(
        "[rwkvasr] Distributed init complete.\n"
        "[deepspeed-train] step=1 loss=0.8 peak_reserved=5.50GiB\n"
        "[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB\n",
        encoding="utf-8",
    )
    easy_manifest = Path(str(segments[0]["bucket_manifest_path"])).resolve()
    smoke_marker = tmp_path / f"{phase}-full-profile-smoke-passed.json"
    smoke_marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "full_profile_smoke",
                "phase": phase,
                "complete": True,
                "init_checkpoint_path": str(phase_init_checkpoint.resolve()),
                "init_checkpoint_sha256": sha256_file(phase_init_checkpoint),
                "easy_manifest_path": str(easy_manifest),
                "easy_manifest_sha256": sha256_file(easy_manifest),
                "smoke_checkpoint_path": str(smoke_checkpoint.resolve()),
                "smoke_checkpoint_sha256": sha256_file(smoke_checkpoint),
                "smoke_log_path": str(smoke_log.resolve()),
                "smoke_log_sha256": sha256_file(smoke_log),
                "peak_reserved_gib": 6.0,
                "max_peak_reserved_gib": 22.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    gate_report = tmp_path / "mixer_gate.json"
    gate_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "phase_gate",
                "phase": phase,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "gate_passed": True,
                "alignment_gate_passed": True,
                "public_progress_gate_passed": True,
                "global_dedup_manifest_path": str(GLOBAL_DEDUP_FIXTURE.resolve()),
                "global_dedup_manifest_sha256": sha256_file(GLOBAL_DEDUP_FIXTURE),
                "loaded_manifest_receipt_path": str(loaded_manifest_receipt.resolve()),
                "loaded_manifest_receipt_sha256": sha256_file(loaded_manifest_receipt),
                "preflight_smoke": {
                    "marker_path": str(smoke_marker.resolve()),
                    "marker_sha256": sha256_file(smoke_marker),
                },
                "alignment_report": {
                    "path": str(alignment_report.resolve()),
                    "sha256": sha256_file(alignment_report),
                    "artifact": (
                        "logits_alignment_gate" if phase == "logits" else "hidden_alignment_gate"
                    ),
                },
                "nano_public_baseline_receipt_path": str(baseline_receipt.resolve()),
                "nano_public_baseline_receipt_sha256": sha256_file(baseline_receipt),
                "nano_public_baseline_checkpoint_sha256": sha256_file(nano_teacher_checkpoint),
                "public_overlap": {
                    "receipt_path": str(public_overlap_receipt.resolve()),
                    "receipt_sha256": sha256_file(public_overlap_receipt),
                },
                "full_data_coverage": full_data_coverage,
                "public_benchmark": {
                    "decode": "greedy_ctc",
                    "normalization": "ctc",
                    "all_datasets_complete": True,
                    "all_datasets_pass": True,
                    "results": public_results,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return gate_report


def _write_failed_phase_gate(gate_report: Path) -> Path:
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(report["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment["gate_passed"] = False
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    report["alignment_report"]["sha256"] = sha256_file(alignment_path)
    report["alignment_gate_passed"] = False
    report["gate_passed"] = False
    failed_gate = gate_report.with_name("failed_phase_gate.json")
    failed_gate.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return failed_gate


def _write_retention_correction(
    tmp_path: Path,
    *,
    failed_gate: Path,
    completion_checkpoint: Path,
) -> tuple[dict[str, object], Path]:
    failed = json.loads(failed_gate.read_text(encoding="utf-8"))
    init_checkpoint = Path(failed["checkpoint_path"])
    nano_checkpoint = Path(
        failed["full_data_coverage"]["segments"][0]["nano_teacher_checkpoint_path"]
    )
    replay_root = tmp_path / "retention-replay"
    replay_root.mkdir()
    replay_manifest = replay_root / "manifest.json"
    replay_manifest.write_text("{}\n", encoding="utf-8")
    replay_part = replay_root / "part.jsonl"
    replay_part.write_text('{"key":"replay-1"}\n', encoding="utf-8")
    replay_builder = replay_root / "builder.py"
    replay_builder.write_text("# builder\n", encoding="utf-8")
    replay_preflight = replay_root / "capacity.json"
    replay_preflight.write_text("{}\n", encoding="utf-8")
    source_manifests = {}
    for difficulty in STAGE211_AUDIO_CURRICULUM:
        path = replay_root / f"{difficulty}.manifest.json"
        path.write_text("{}\n", encoding="utf-8")
        source_manifests[difficulty] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    exclusions = []
    for name in ("stratified", "fixed"):
        path = replay_root / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        exclusions.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    replay_receipt = replay_root / "receipt.json"
    replay_receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "retention_replay_manifest",
                "samples": 8,
                "unique_keys": 8,
                "total_hours": 0.01,
                "manifest_path": str(replay_manifest.resolve()),
                "manifest_sha256": sha256_file(replay_manifest),
                "builder": {
                    "path": str(replay_builder.resolve()),
                    "sha256": sha256_file(replay_builder),
                },
                "capacity_preflight_path": str(replay_preflight.resolve()),
                "capacity_preflight_sha256": sha256_file(replay_preflight),
                "source_manifests": source_manifests,
                "exclusions": exclusions,
                "output_parts": [
                    {
                        "path": str(replay_part.resolve()),
                        "sha256": sha256_file(replay_part),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    smoke_checkpoint = replay_root / "smoke-step-2.pt"
    smoke_log = replay_root / "smoke.log"
    smoke_checkpoint.write_bytes(b"smoke-checkpoint")
    smoke_log.write_bytes(b"smoke-log")
    smoke_marker = replay_root / "round-01-smoke-passed.json"
    smoke_marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "full_profile_smoke",
                "phase": "mixer",
                "complete": True,
                "correction_round": 1,
                "init_checkpoint_path": str(init_checkpoint.resolve()),
                "init_checkpoint_sha256": sha256_file(init_checkpoint),
                "easy_manifest_path": str(replay_manifest.resolve()),
                "easy_manifest_sha256": sha256_file(replay_manifest),
                "replay_receipt_path": str(replay_receipt.resolve()),
                "replay_receipt_sha256": sha256_file(replay_receipt),
                "admission_gate_path": str(failed_gate.resolve()),
                "admission_gate_sha256": sha256_file(failed_gate),
                "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
                "smoke_checkpoint_path": str(smoke_checkpoint.resolve()),
                "smoke_checkpoint_sha256": sha256_file(smoke_checkpoint),
                "smoke_log_path": str(smoke_log.resolve()),
                "smoke_log_sha256": sha256_file(smoke_log),
                "peak_reserved_gib": 6.0,
                "max_peak_reserved_gib": 22.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "retention-round-01"
    run_dir.mkdir()
    train_config = run_dir / "train_config.yaml"
    config = stage211_phase_train_config_contract("mixer")
    config.update(
        {
            "lr": STAGE211_RETENTION_CORRECTION_LR,
            "max_steps": 1,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "batch_token_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
            "webdataset_bucket_manifest_path": str(replay_manifest.resolve()),
            "webdataset_split": "train",
            "ctc_teacher_online_model_path": str(nano_checkpoint.parent.resolve()),
            "init_checkpoint_path": str(init_checkpoint.resolve()),
            "stage211_post_coverage_correction_round": 1,
            "stage211_post_coverage_replay_receipt_path": str(replay_receipt.resolve()),
            "stage211_post_coverage_admission_gate_path": str(failed_gate.resolve()),
            "stage211_post_coverage_original_coverage_unchanged": True,
            "stage211_post_coverage_smoke_marker_path": str(smoke_marker.resolve()),
            "stage211_post_coverage_smoke_marker_sha256": sha256_file(smoke_marker),
        }
    )
    save_yaml(train_config, config)
    provenance = run_dir / "stage211_correction_provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    epoch_checkpoint = run_dir / "epoch-1.pt"
    epoch_checkpoint.write_bytes(b"epoch-1")
    correction = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "post_coverage_correction",
        "phase": "mixer",
        "round": 1,
        "complete": True,
        "epochs": STAGE211_RETENTION_CORRECTION_EPOCHS,
        "learning_rate": STAGE211_RETENTION_CORRECTION_LR,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "rows": 8,
        "row_exposures": 8,
        "hours": 0.01,
        "hour_exposures": 0.01,
        "steps_per_epoch": 1,
        "steps": 1,
        "tail_padding_samples_per_epoch": 4,
        "tail_padding_sample_exposures": 4,
        "executed_sample_exposures": 12,
        "run_dir": str(run_dir.resolve()),
        "provenance_path": str(provenance.resolve()),
        "provenance_sha256": sha256_file(provenance),
        "train_config_path": str(train_config.resolve()),
        "train_config_sha256": sha256_file(train_config),
        "smoke_marker_path": str(smoke_marker.resolve()),
        "smoke_marker_sha256": sha256_file(smoke_marker),
        "replay_receipt_path": str(replay_receipt.resolve()),
        "replay_receipt_sha256": sha256_file(replay_receipt),
        "bucket_manifest_path": str(replay_manifest.resolve()),
        "bucket_manifest_sha256": sha256_file(replay_manifest),
        "admission_gate_path": str(failed_gate.resolve()),
        "admission_gate_sha256": sha256_file(failed_gate),
        "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "init_checkpoint_path": str(init_checkpoint.resolve()),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "completion_checkpoint_path": str(completion_checkpoint.resolve()),
        "completion_checkpoint_sha256": sha256_file(completion_checkpoint),
        "runtime_epoch_coverage": {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "runtime_epoch_coverage",
            "complete": True,
            "epochs": 1,
            "steps_per_epoch": 1,
            "total_steps": 1,
            "records": [
                {
                    "epoch": 1,
                    "step": 1,
                    "epoch_batch_offset": 0,
                    "completed_epoch_batch_count": 1,
                    "checkpoint_path": str(epoch_checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(epoch_checkpoint),
                }
            ],
        },
        "parameter_delta_audit": {
            "schema_version": 1,
            "policy": "stage211_timemixer_and_input_projection_only",
            "complete": True,
            "allowed_key_markers": list(STAGE211_ALLOWED_OPERATOR_KEY_MARKERS),
            "initial_tensor_count": 4,
            "completion_tensor_count": 4,
            "allowed_changed_tensors": 1,
            "allowed_changed_numel": 1,
            "allowed_unchanged_tensors": 1,
            "frozen_unchanged_tensors": 2,
            "forbidden_changed_tensors": 0,
        },
    }
    receipt_path = run_dir / "correction_receipt.json"
    receipt_path.write_text(json.dumps(correction) + "\n", encoding="utf-8")
    return {
        **correction,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
    }, replay_part


def _write_corrected_phase_gate(
    tmp_path: Path,
    *,
    failed_gate: Path,
    checkpoint: Path,
    correction: dict[str, object],
) -> Path:
    report = json.loads(failed_gate.read_text(encoding="utf-8"))
    failed_alignment_path = Path(report["alignment_report"]["path"])
    alignment = json.loads(failed_alignment_path.read_text(encoding="utf-8"))
    old_candidate_path = Path(alignment["candidate_report_path"])
    candidate_source = json.loads(old_candidate_path.read_text(encoding="utf-8"))
    candidate_source.update(
        {
            "step": 1,
            "checkpoint_step": 1,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    )
    candidate_source_path = tmp_path / "corrected-alignment-candidate.json"
    candidate_source_path.write_text(
        json.dumps(candidate_source) + "\n",
        encoding="utf-8",
    )
    alignment.update(
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "candidate_report_path": str(candidate_source_path.resolve()),
            "candidate_report_sha256": sha256_file(candidate_source_path),
            "gate_passed": True,
        }
    )
    alignment_path = tmp_path / "corrected-alignment-gate.json"
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    report.update(
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "gate_passed": True,
            "alignment_gate_passed": True,
        }
    )
    report["alignment_report"] = {
        "path": str(alignment_path.resolve()),
        "sha256": sha256_file(alignment_path),
        "artifact": "hidden_alignment_gate",
    }
    original_segments = report["full_data_coverage"]["segments"]
    supplemental_segment = report["full_data_coverage"]["supplemental_natural"]
    report["full_data_coverage"] = build_stage211_full_data_coverage(
        phase="mixer",
        segments=original_segments,
        supplemental_segment=supplemental_segment,
        checkpoint_path=checkpoint,
        post_coverage_corrections=[correction],
    )
    corrected_gate = tmp_path / "corrected_phase_gate.json"
    corrected_gate.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return corrected_gate


def test_stage211_phase_gate_validates_retention_correction_chain(
    tmp_path: Path,
) -> None:
    original_checkpoint = tmp_path / "long-complete.pt"
    original_checkpoint.write_bytes(b"long-complete")
    original_gate = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=original_checkpoint,
    )
    failed_gate = _write_failed_phase_gate(original_gate)
    corrected_checkpoint = tmp_path / "retention-complete.pt"
    corrected_checkpoint.write_bytes(b"retention-complete")
    correction, replay_part = _write_retention_correction(
        tmp_path,
        failed_gate=failed_gate,
        completion_checkpoint=corrected_checkpoint,
    )
    corrected_gate = _write_corrected_phase_gate(
        tmp_path,
        failed_gate=failed_gate,
        checkpoint=corrected_checkpoint,
        correction=correction,
    )

    validated = validate_stage211_phase_gate_report(
        corrected_gate,
        expected_phase="mixer",
        checkpoint_path=corrected_checkpoint,
    )
    assert (
        validated["full_data_coverage"]["segments"]
        == json.loads(failed_gate.read_text(encoding="utf-8"))["full_data_coverage"]["segments"]
    )
    assert validated["full_data_coverage"]["post_coverage_correction_exposure"] == (
        stage211_post_coverage_correction_exposure([correction])
    )

    replay_part.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output part 0 SHA-256 mismatch"):
        validate_stage211_phase_gate_report(
            corrected_gate,
            expected_phase="mixer",
            checkpoint_path=corrected_checkpoint,
        )


def test_stage211_phase_gate_rejects_correction_checkpoint_chain_break(
    tmp_path: Path,
) -> None:
    original_checkpoint = tmp_path / "long-complete.pt"
    original_checkpoint.write_bytes(b"long-complete")
    failed_gate = _write_failed_phase_gate(
        _write_valid_phase_gate(
            tmp_path,
            phase="mixer",
            checkpoint=original_checkpoint,
        )
    )
    corrected_checkpoint = tmp_path / "retention-complete.pt"
    corrected_checkpoint.write_bytes(b"retention-complete")
    correction, _ = _write_retention_correction(
        tmp_path,
        failed_gate=failed_gate,
        completion_checkpoint=corrected_checkpoint,
    )
    wrong_init = tmp_path / "wrong-init.pt"
    wrong_init.write_bytes(b"wrong-init")
    receipt_path = Path(str(correction["receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["init_checkpoint_path"] = str(wrong_init.resolve())
    receipt["init_checkpoint_sha256"] = sha256_file(wrong_init)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    correction = {
        **receipt,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
    }
    corrected_gate = _write_corrected_phase_gate(
        tmp_path,
        failed_gate=failed_gate,
        checkpoint=corrected_checkpoint,
        correction=correction,
    )

    with pytest.raises(ValueError, match="does not initialize from the preceding checkpoint"):
        validate_stage211_phase_gate_report(
            corrected_gate,
            expected_phase="mixer",
            checkpoint_path=corrected_checkpoint,
        )


def test_stage211_promotion_receipt_binds_checkpoint_and_gate_hashes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-30064.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    receipt_path = tmp_path / "mixer_to_block.json"
    receipt = stage211._build_promotion_receipt(
        source_phase="mixer",
        checkpoint_path=checkpoint,
        gate_report_path=gate_report,
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    gate_payload = json.loads(gate_report.read_text(encoding="utf-8"))
    nano_checkpoint = Path(
        gate_payload["full_data_coverage"]["segments"][0]["nano_teacher_checkpoint_path"]
    )
    assert receipt["nano_teacher_checkpoint_sha256"] == sha256_file(nano_checkpoint)

    validated = stage211._validate_promotion_receipt(
        receipt_path=receipt_path,
        target_phase="block",
        checkpoint_path=checkpoint,
        nano_checkpoint_path=nano_checkpoint,
    )
    assert validated["source_phase"] == "mixer"
    assert validated["target_phase"] == "block"

    alternate_nano = tmp_path / "alternate-nano.pt"
    alternate_nano.write_bytes(b"different-nano")
    with pytest.raises(
        ValueError,
        match="Nano teacher checkpoint SHA-256 differs from the preceding stage",
    ):
        stage211._validate_promotion_receipt(
            receipt_path=receipt_path,
            target_phase="block",
            checkpoint_path=checkpoint,
            nano_checkpoint_path=alternate_nano,
        )

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        stage211._validate_promotion_receipt(
            receipt_path=receipt_path,
            target_phase="block",
            checkpoint_path=checkpoint,
        )


def test_stage211_curriculum_receipt_requires_teacher_continuity(
    tmp_path: Path,
) -> None:
    final_checkpoint = tmp_path / "step-final.pt"
    final_checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=final_checkpoint,
    )
    gate_payload = json.loads(gate_report.read_text(encoding="utf-8"))
    easy = gate_payload["full_data_coverage"]["segments"][0]
    receipt_path = Path(easy["receipt_path"])
    checkpoint = Path(easy["completion_checkpoint_path"])
    nano_checkpoint = Path(easy["nano_teacher_checkpoint_path"])

    receipt = stage211._validate_curriculum_receipt(
        receipt_path=receipt_path,
        phase="mixer",
        target_difficulty="medium",
        checkpoint_path=checkpoint,
        nano_checkpoint_path=nano_checkpoint,
    )

    assert receipt["difficulty"] == "easy"
    alternate_nano = tmp_path / "alternate-curriculum-nano.pt"
    alternate_nano.write_bytes(b"different-nano")
    with pytest.raises(
        ValueError,
        match="Nano teacher checkpoint SHA-256 differs from the preceding stage",
    ):
        stage211._validate_curriculum_receipt(
            receipt_path=receipt_path,
            phase="mixer",
            target_difficulty="medium",
            checkpoint_path=checkpoint,
            nano_checkpoint_path=alternate_nano,
        )


def test_stage211_promotion_rejects_unverified_gate_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-30064.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = tmp_path / "mixer_gate.json"
    gate_report.write_text('{"passed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version mismatch"):
        stage211._build_promotion_receipt(
            source_phase="mixer",
            checkpoint_path=checkpoint,
            gate_report_path=gate_report,
        )


def test_stage211_phase_gate_rejects_mutated_coverage_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    (tmp_path / "easy-receipt.json").write_text(
        '{"mutated": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coverage receipt SHA-256 mismatch"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_alignment_report(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="logits",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(gate["alignment_report"]["path"])
    alignment_path.write_text('{"gate_passed": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="alignment report SHA-256 mismatch"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_preflight_smoke_log(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "long-complete.pt"
    checkpoint.write_bytes(b"long-complete")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    marker_path = Path(report["preflight_smoke"]["marker_path"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    smoke_log = Path(marker["smoke_log_path"])
    smoke_log.write_text(
        smoke_log.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="smoke log SHA-256 mismatch"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_false_step2_smoke_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "long-complete.pt"
    checkpoint.write_bytes(b"long-complete")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    marker_path = Path(report["preflight_smoke"]["marker_path"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    smoke_checkpoint = Path(marker["smoke_checkpoint_path"])
    torch.save({"step": 3}, smoke_checkpoint)
    marker["smoke_checkpoint_sha256"] = sha256_file(smoke_checkpoint)
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    report["preflight_smoke"]["marker_sha256"] = sha256_file(marker_path)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint is not step 2"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_smoke_sample_skipping(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "long-complete.pt"
    checkpoint.write_bytes(b"long-complete")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    marker_path = Path(report["preflight_smoke"]["marker_path"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    smoke_log = Path(marker["smoke_log_path"])
    smoke_log.write_text(
        smoke_log.read_text(encoding="utf-8") + "skipped_samples=1\n",
        encoding="utf-8",
    )
    marker["smoke_log_sha256"] = sha256_file(smoke_log)
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    report["preflight_smoke"]["marker_sha256"] = sha256_file(marker_path)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="smoke log failed validation"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_rewritten_global_dedup_manifest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "long-complete.pt"
    checkpoint.write_bytes(b"long-complete")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    rewritten_manifest = tmp_path / "rewritten-global-dedup.json"
    manifest = json.loads(GLOBAL_DEDUP_FIXTURE.read_text(encoding="utf-8"))
    manifest["total_unique_audio_rows"] -= 1
    rewritten_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    report["global_dedup_manifest_path"] = str(rewritten_manifest.resolve())
    report["global_dedup_manifest_sha256"] = sha256_file(rewritten_manifest)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="global dedup manifest SHA-256 mismatch"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_rewritten_loaded_manifest_receipt(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    receipt_path = Path(gate["loaded_manifest_receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["part_hash_audit_complete"] = False
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    gate["loaded_manifest_receipt_sha256"] = sha256_file(receipt_path)
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="loaded-manifest receipt contract mismatch"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_alignment_source(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment = json.loads(Path(gate["alignment_report"]["path"]).read_text(encoding="utf-8"))
    Path(alignment["candidate_report_path"]).write_text(
        "mutated\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="alignment candidate report SHA-256 mismatch",
    ):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_logits_phase_gate_rejects_missing_fixed_feature_seed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="logits",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(gate["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment["baseline_eval_provenance"].pop("feature_seed")
    alignment["candidate_eval_provenance"].pop("feature_seed")
    for prefix in ("baseline_report", "candidate_report"):
        source_path = Path(alignment[f"{prefix}_path"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["eval_provenance"].pop("feature_seed")
        source_path.write_text(
            json.dumps(source) + "\n",
            encoding="utf-8",
        )
        alignment[f"{prefix}_sha256"] = sha256_file(source_path)
    alignment_path.write_text(
        json.dumps(alignment) + "\n",
        encoding="utf-8",
    )
    gate["alignment_report"]["sha256"] = sha256_file(alignment_path)
    gate_report.write_text(
        json.dumps(gate) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed feature seed"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_nano_teacher(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    (tmp_path / "nano-teacher" / "model.pt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="Nano teacher checkpoint SHA-256 mismatch"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_nano_baseline_receipt_rejects_mutated_inference_report(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    receipt_path = Path(gate["nano_public_baseline_receipt_path"])
    receipt = validate_stage211_nano_public_baseline_receipt(receipt_path)
    report_path = Path(receipt["results"][0]["report_path"])
    inference_report = json.loads(report_path.read_text(encoding="utf-8"))
    inference_report["decode"] = "beam_search"
    report_path.write_text(json.dumps(inference_report) + "\n", encoding="utf-8")
    receipt["results"][0]["report_sha256"] = sha256_file(report_path)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inference report contract mismatch"):
        validate_stage211_nano_public_baseline_receipt(receipt_path)


def test_stage211_phase_gate_rejects_missing_public_overlap_binding(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    gate.pop("public_overlap")
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks the public/train overlap binding"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_clean_public_manifest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    overlap_receipt = json.loads(
        Path(gate["public_overlap"]["receipt_path"]).read_text(encoding="utf-8")
    )
    clean_manifest = Path(overlap_receipt["outputs"]["clean_manifest"]["path"])
    clean_manifest.write_text(
        clean_manifest.read_text(encoding="utf-8") + '{"utt_id":"injected"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_nano_baseline_prediction_substitution(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    replacement = tmp_path / "replacement-nano.jsonl"
    replacement.write_text("{}\n", encoding="utf-8")
    result = report["public_benchmark"]["results"][0]
    result["nano_prediction_path"] = str(replacement.resolve())
    result["nano_prediction_sha256"] = sha256_file(replacement)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="public-baseline provenance receipt"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_public_baseline_from_other_nano(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    receipt_path = Path(gate["nano_public_baseline_receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    alternate_dir = tmp_path / "alternate-public-nano"
    alternate_dir.mkdir()
    alternate_checkpoint = alternate_dir / "model.pt"
    alternate_checkpoint.write_bytes(b"alternate-public-nano")
    receipt["nano_checkpoint_path"] = str(alternate_checkpoint.resolve())
    receipt["nano_checkpoint_sha256"] = sha256_file(alternate_checkpoint)
    for result in receipt["results"]:
        report_path = Path(result["report_path"])
        inference_report = json.loads(report_path.read_text(encoding="utf-8"))
        inference_report["model_path"] = str(alternate_dir.resolve())
        report_path.write_text(
            json.dumps(inference_report) + "\n",
            encoding="utf-8",
        )
        result["report_sha256"] = sha256_file(report_path)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    gate["nano_public_baseline_receipt_sha256"] = sha256_file(receipt_path)
    gate["nano_public_baseline_checkpoint_sha256"] = sha256_file(alternate_checkpoint)
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="public baseline and online teacher checkpoint SHA-256 differ",
    ):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mixed_nano_teachers(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    alternate_dir = tmp_path / "alternate-nano"
    alternate_dir.mkdir()
    alternate_checkpoint = alternate_dir / "model.pt"
    alternate_checkpoint.write_bytes(b"alternate-nano-teacher")
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    medium = report["full_data_coverage"]["segments"][1]
    train_config = Path(medium["train_config_path"])
    train_config_payload = stage211_phase_train_config_contract("mixer")
    train_config_payload["ctc_teacher_online_model_path"] = str(alternate_dir.resolve())
    save_yaml(train_config, train_config_payload)
    medium["train_config_sha256"] = sha256_file(train_config)
    medium["nano_teacher_checkpoint_path"] = str(alternate_checkpoint.resolve())
    medium["nano_teacher_checkpoint_sha256"] = sha256_file(alternate_checkpoint)
    receipt_path = Path(medium["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["train_config_sha256"] = medium["train_config_sha256"]
    receipt["nano_teacher_checkpoint_path"] = medium["nano_teacher_checkpoint_path"]
    receipt["nano_teacher_checkpoint_sha256"] = medium["nano_teacher_checkpoint_sha256"]
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    medium["receipt_sha256"] = sha256_file(receipt_path)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="uses a different Nano teacher checkpoint"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_missing_supplemental_coverage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    del report["full_data_coverage"]["supplemental_natural"]
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks supplemental_natural"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_reordered_original_segments(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    segments = report["full_data_coverage"]["segments"]
    segments[0], segments[1] = segments[1], segments[0]
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly ordered easy, medium, hard, and long"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_supplemental_checkpoint_chain_break(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    wrong_init = tmp_path / "wrong-long-completion.pt"
    wrong_init.write_bytes(b"wrong-long-completion")
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    supplemental = report["full_data_coverage"]["supplemental_natural"]
    supplemental["init_checkpoint_path"] = str(wrong_init.resolve())
    supplemental["init_checkpoint_sha256"] = sha256_file(wrong_init)
    receipt_path = Path(supplemental["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["init_checkpoint_path"] = supplemental["init_checkpoint_path"]
    receipt["init_checkpoint_sha256"] = supplemental["init_checkpoint_sha256"]
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    supplemental["receipt_sha256"] = sha256_file(receipt_path)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not initialize from Long"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_supplemental_inventory(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    inventory_path = Path(
        report["full_data_coverage"]["supplemental_natural"][
            "supplemental_inventory_path"
        ]
    )
    inventory_path.write_text(
        inventory_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supplemental inventory SHA-256 mismatch"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_combined_coverage_total(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    report["full_data_coverage"]["total_unique_rows"] += 1
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="combined full-data coverage total_unique_rows"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_decode_skipping_coverage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    report["full_data_coverage"]["segments"][0]["webdataset_skip_decode_errors"] = True
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage is incomplete"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_partial_runtime_epoch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    report["full_data_coverage"]["segments"][0]["runtime_epoch_coverage"]["records"][0][
        "completed_epoch_batch_count"
    ] -= 1
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime epoch 1 completion mismatch"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_frozen_path_change(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    report["full_data_coverage"]["segments"][0]["parameter_delta_audit"][
        "forbidden_changed_tensors"
    ] = 1
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parameter-delta audit is invalid"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_preserves_a_strict_failed_decision(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(report["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment["gate_passed"] = False
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    report["alignment_report"]["sha256"] = sha256_file(alignment_path)
    report["alignment_gate_passed"] = False
    report["gate_passed"] = False
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    validated = stage211.validate_stage211_phase_gate_report(
        gate_report,
        expected_phase="mixer",
        checkpoint_path=checkpoint,
        require_passed=False,
    )
    assert validated["gate_passed"] is False

    with pytest.raises(ValueError, match="does not record a passing decision"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )

    report["gate_passed"] = True
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="decision is inconsistent"):
        stage211.validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
            require_passed=False,
        )


def test_stage211_continuation_watcher_is_hourly_and_restart_safe() -> None:
    script = (REPO_ROOT / "scripts" / "watch_stage211_strict_continuation.sh").read_text(
        encoding="utf-8"
    )

    assert 'POLL_SECONDS="${POLL_SECONDS:-3600}"' in script
    assert 'RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-3600}"' in script
    assert "while true; do" in script
    assert 'while tmux has-session -t "${SUPERVISOR_SESSION}"' in script
    assert 'sleep "${POLL_SECONDS}"' in script
    assert 'START_STAGE="${start_stage}"' in script
    assert "REUSE_COMPLETED_CALIBRATION_EVAL=1" in script
    assert "stage211_choose_start_stage" in script
    assert "stage211_final_proof_valid" in script
    assert "create_stage211_stepwise_report.py" in script
    assert "FINAL_STEPWISE_REPORT" in script
    assert ".checkpoint_chain_passed == true" in script
    assert ".nano_initialization_chain_passed == true" in script
    assert ".ctc_label_normalization_chain_passed == true" in script
    assert ".ctc_label_proof.ctc_unk_tokens == 0" in script
    assert ".public_metric_definition_chain_passed == true" in script
    assert ".public_metric_tokenizer_contract" in script
    assert "--public-metric-correction-receipt" in script
    assert ".nano_teacher_chain_passed == true" in script
    assert ".supplemental_inventory_chain_passed == true" in script
    assert ".supplemental_dedupe_proof.source_sets_disjoint == true" in script
    assert ".supplemental_dedupe_proof.content_fingerprint_complete == false" in script
    assert ".supplemental_dedupe_proof.inventory_schema_version == 2" in script
    assert (
        ".supplemental_dedupe_proof.base_public_overlap_normalized_pcm_exact_complete == true"
        in script
    )
    assert ".supplemental_dedupe_proof.base_public_overlap_rows == 0" in script
    assert (
        '.supplemental_dedupe_proof.base_public_overlap_scan_order == '
        '"manifest_location_index_archive_order_v1"'
        in script
    )
    assert ".supplemental_dedupe_proof.base_public_overlap_scanned_rows > 0" in script
    assert ".supplemental_dedupe_proof.base_public_overlap_receipt_sha256" in script
    assert ".supplemental_dedupe_proof.social_normalized_pcm_exact_complete == true" in script
    assert '.supplemental_dedupe_proof.social_public_overlap_mode == "normalized_pcm_exact"' in script
    assert (
        ".supplemental_dedupe_proof.archived_social_exact_duplicate_exclusion_complete == true"
        in script
    )
    assert ".supplemental_dedupe_proof.archived_social_unique_members == 0" in script
    assert ".supplemental_dedupe_proof.usb_natural_audio_resolution_complete == true" in script
    assert ".supplemental_dedupe_proof.usb_unresolved_natural_entries == []" in script
    assert ".all_stage_public_metrics_complete == true" in script
    assert 'select(.language == "en" and .metric == "wer")' in script
    assert 'select(.language == "zh" and .metric == "cer")' in script
    assert '.strict_stage_order == ["calibration", "mixer", "block", "logits", "sft"]' in script
    assert '.requested_alignment_stage_order == ["rwkv_layer", "block", "logits", "sft"]' in script
    assert "post_mixer" in script
    assert "tmux kill-session" not in script


def _run_stage211_continuation_watcher_fixture(
    tmp_path: Path,
    *,
    uv_mode: str,
    tmux_mode: str = "stateless",
    watch_once: str | None = "1",
    initial_final_report: bool = True,
) -> tuple[subprocess.CompletedProcess[str], str, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    tmux_calls = tmp_path / "tmux-calls.log"
    tmux = fake_bin / "tmux"
    tmux.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${TMUX_CALLS}"\n'
        'if [[ "${TMUX_MODE}" == multi_handoff ]]; then\n'
        '  if [[ "${1:-}" == has-session ]]; then\n'
        '    if [[ "$*" == *rwkvasr_stage211_abcd_hourly_monitor* ]]; then exit 0; fi\n'
        '    if [[ "$*" == *rwkvasr_stage211_abcd_strict_supervisor* '
        '&& -f "${TMUX_STATE}.active" ]]; then\n'
        '      rm -f "${TMUX_STATE}.active"\n'
        '      launches="$(cat "${TMUX_STATE}.launches")"\n'
        '      if ((launches >= 2)); then\n'
        '        mkdir -p "$(dirname "${FINAL_REPORT_PATH}")"\n'
        '        printf \'%s\\n\' '
        '\'{"pipeline":"stage211","artifact":"final_completion",'
        '"complete":true,"gate_passed":true}\' >"${FINAL_REPORT_PATH}"\n'
        '      fi\n'
        '      exit 0\n'
        '    fi\n'
        '    exit 1\n'
        '  fi\n'
        '  if [[ "${1:-}" == new-session '
        '&& "$*" == *rwkvasr_stage211_abcd_strict_supervisor* ]]; then\n'
        '    launches=0\n'
        '    [[ -f "${TMUX_STATE}.launches" ]] '
        '&& launches="$(cat "${TMUX_STATE}.launches")"\n'
        '    printf \'%s\\n\' "$((launches + 1))" >"${TMUX_STATE}.launches"\n'
        '    touch "${TMUX_STATE}.active"\n'
        '  fi\n'
        '  exit 0\n'
        'fi\n'
        'if [[ "${1:-}" == has-session ]]; then exit 1; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    tmux.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${UV_MODE}" == failure ]]; then exit 42; fi\n'
        "output_json=\n"
        "output_markdown=\n"
        "while (($#)); do\n"
        '  case "$1" in\n'
        '    --output-json) output_json="$2"; shift 2 ;;\n'
        '    --output-markdown) output_markdown="$2"; shift 2 ;;\n'
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'mkdir -p "$(dirname "${output_json}")"\n'
        'if [[ "${UV_MODE}" == malformed_metrics ]]; then\n'
        '  dataset_proof=\'"public_metric_stage_order":["calibration","mixer","block","logits","sft"],"all_stage_public_metrics_complete":true,"english_wer_datasets":["en1","en2","en3"],"chinese_cer_datasets":["zh1","zh2"],"dataset_results":[1,2,3,4,5]\'\n'
        "else\n"
        '  stages=\'{"calibration":{},"mixer":{},"block":{},"logits":{},"sft":{}}\'\n'
        '  dataset_proof=\'"public_metric_stage_order":["calibration","mixer","block","logits","sft"],"all_stage_public_metrics_complete":true,"english_wer_datasets":["en1","en2","en3"],"chinese_cer_datasets":["zh1","zh2"],"dataset_results":[{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"zh","metric":"cer","stages":\'"${stages}"\'},{"language":"zh","metric":"cer","stages":\'"${stages}"\'}]\'\n'
        "fi\n"
        'printf \'%s\\n\' \'{"pipeline":"stage211","artifact":"stepwise_final_results","complete":true,"gate_passed":true,"strict_stage_order":["calibration","mixer","block","logits","sft"],"requested_alignment_stage_order":["rwkv_layer","block","logits","sft"],"checkpoint_chain_passed":true,"nano_initialization_chain_passed":true,"ctc_label_normalization_chain_passed":true,"ctc_label_proof":{"full_length_index_audit_passed":true,"ctc_suppress_non_pronunciation_tokens":true,"ctc_unk_tokens":0},"public_metric_definition_chain_passed":true,"public_metric_tokenizer_contract":"unicode_alnum_words_basic_cjk_chars_v1","public_metric_correction_receipt_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","public_metric_tokenizer_source_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","nano_teacher_chain_passed":true,"nano_public_baseline_provenance_passed":true,"supplemental_inventory_chain_passed":true,"supplemental_dedupe_proof":{"inventory_schema_version":2,"inventory_artifact":"stage211_supplemental_combined_inventory","mode":"source_identity_plus_known_corpus_exclusion","source_sets_disjoint":true,"content_fingerprint_complete":false,"base_public_overlap_normalized_pcm_exact_complete":true,"base_public_overlap_scan_order":"manifest_location_index_archive_order_v1","base_public_overlap_rows":0,"base_public_overlap_scanned_rows":1,"base_public_overlap_receipt_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","social_normalized_pcm_exact_complete":true,"social_public_overlap_mode":"normalized_pcm_exact","archived_social_exact_duplicate_exclusion_complete":true,"archived_social_unique_members":0,"archived_social_overlap_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","usb_top_level_classification_complete":true,"usb_natural_audio_resolution_complete":true,"usb_unresolved_natural_entries":[],"usb_top_level_coverage_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","near_duplicate_complete":false,"component_inventories":{"base_natural":{},"social_vad":{}},"known_overlap_exclusions":["llaso_gigaspeech","llaso_librispeech"]},"coverage_results":[{"stage":"mixer","training_segments":[{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3}]},{"stage":"block","training_segments":[{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3}]},{"stage":"logits","training_segments":[{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3}]},{"stage":"sft"}],\'"${dataset_proof}"\'}\' >"${output_json}"\n'
        "printf '%s\\n' '# stepwise' >\"${output_markdown}\"\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    phase_gate_root = tmp_path / "gates"
    final_report = phase_gate_root / "sft" / "stage211_complete.json"
    final_report.parent.mkdir(parents=True)
    if initial_final_report:
        final_report.write_text(
            json.dumps(
                {
                    "pipeline": "stage211",
                    "artifact": "final_completion",
                    "complete": True,
                    "gate_passed": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    extra_env = {}
    if watch_once is not None:
        extra_env["WATCH_ONCE"] = watch_once
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "watch_stage211_strict_continuation.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TMUX_CALLS": str(tmux_calls),
            "TMUX_MODE": tmux_mode,
            "TMUX_STATE": str(tmp_path / "tmux-state"),
            "FINAL_REPORT_PATH": str(final_report),
            "UV_MODE": uv_mode,
            "PHASE_GATE_ROOT": str(phase_gate_root),
            "FULL_OUTPUT_ROOT": str(tmp_path / "runs"),
            "WATCH_LOG": str(tmp_path / "watch.log"),
            "POLL_SECONDS": "0",
            "RESTART_BACKOFF_SECONDS": "0",
            **extra_env,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    return result, tmux_calls.read_text(encoding="utf-8"), phase_gate_root


def test_stage211_continuation_watcher_accepts_only_deep_stepwise_proof(
    tmp_path: Path,
) -> None:
    result, tmux_calls, phase_gate_root = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="success",
    )

    assert result.returncode == 0, result.stderr
    assert "SFT and stepwise proofs pass" in result.stdout
    assert "new-session" not in tmux_calls
    assert (phase_gate_root / "sft" / "stage211_stepwise_results.json").is_file()


def test_stage211_continuation_watcher_restarts_after_stepwise_failure(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="failure",
    )

    assert result.returncode == 0, result.stderr
    assert "stepwise proof is missing or invalid" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_rejects_placeholder_bilingual_metrics(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="malformed_metrics",
    )

    assert result.returncode == 0, result.stderr
    assert "final Stage211 stepwise proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_survives_multiple_supervisor_handoffs(
    tmp_path: Path,
) -> None:
    result, tmux_calls, phase_gate_root = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="success",
        tmux_mode="multi_handoff",
        watch_once=None,
        initial_final_report=False,
    )

    assert result.returncode == 0, result.stderr
    assert tmux_calls.count("new-session") == 2
    assert result.stdout.count("replacement supervisor launched") == 2
    assert "SFT and stepwise proofs pass" in result.stdout
    assert (phase_gate_root / "sft" / "stage211_stepwise_results.json").is_file()
