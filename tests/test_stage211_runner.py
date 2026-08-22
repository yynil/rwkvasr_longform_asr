from __future__ import annotations

import ast
import importlib
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.data.webdataset_bucketed import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)
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
    STAGE211_PHASE_GATE_SCHEMA_VERSION,
    STAGE211_PROMOTION_POLICY_COVERAGE_NON_DIVERGENT,
    STAGE211_PUBLIC_BENCHMARKS,
    STAGE211_RETENTION_CORRECTION_EPOCHS,
    STAGE211_RETENTION_CORRECTION_LR,
    STAGE211_STACKED_SAFE_BATCH_SIZE,
    STAGE211_STACKED_SAFE_FRAME_BUDGET,
    build_stage211_alignment_loss_nondivergence,
    build_stage211_correction_round_promotion_gate,
    build_stage211_correction_layer_focus,
    build_stage211_full_data_coverage,
    build_stage211_step_eval_cadence,
    build_stage211_trajectory_retention_gate,
    sha256_file,
    stage211_post_coverage_correction_exposure,
    stage211_post_coverage_train_config_contract,
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
from rwkvasr.eval.stage211_public_metrics import (
    STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
    build_stage211_student_public_prediction_receipt,
    replay_stage211_public_comparison,
    validate_stage211_student_public_prediction_receipt,
)
from stage211_public_helpers import (
    stage211_test_student_ctc_context,
    stage211_test_student_ctc_row,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_DEDUP_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "stage211_global_dedup_manifest.json"
sys.path.insert(0, str(REPO_ROOT))
stage211 = importlib.import_module("scripts.run_stage211_strict_chained_alignment")
public_compare = importlib.import_module("scripts.compare_public_ctc_with_nano")
stage211_phase_gate = importlib.import_module("scripts.create_stage211_phase_gate")
stage211_full_phase = importlib.import_module("scripts.run_stage211_full_phase_curriculum")
stage211_phase_finalizer = importlib.import_module("scripts.finalize_stage211_phase")
stage211_stepwise_report = importlib.import_module("scripts.create_stage211_stepwise_report")
stage211_fixed_eval_layer_progress = importlib.import_module(
    "scripts.report_stage211_fixed_eval_layer_progress"
)
stage211_calibration_eval = importlib.import_module("scripts.validate_stage211_calibration_eval")
stage211_supplemental_profile_receipt = importlib.import_module(
    "scripts.create_stage211_supplemental_profile_receipt"
)
stage211_replay_validator = importlib.import_module("scripts.validate_stage211_retention_replay")
stage211_gate_module = importlib.import_module("rwkvasr.eval.stage211_gate")
stage211_batch_profile_test = importlib.import_module("tests.test_stage211_batch_profile_admission")


def _validate_synthetic_retention_replay(receipt_path: Path) -> dict[str, object]:
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for index, record in enumerate(receipt.get("supplemental_output_parts", [])):
        part_path = Path(str(record.get("path") or "")).resolve()
        if not part_path.is_file() or record.get("sha256") != sha256_file(part_path):
            raise ValueError(
                f"Stage211 retention replay output part {index} SHA-256 mismatch: {part_path}"
            )
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "validated_unique_keys": int(receipt["samples"]),
    }


def _stage211_smoke_runtime_fields(phase: str) -> str:
    matches = " ".join(
        f"{field}=8/8" for field in stage211_full_phase.SMOKE_RUNTIME_MATCH_FIELDS[phase]
    )
    losses = " ".join(
        f"{field}=0.1250" for field in stage211_full_phase.SMOKE_RUNTIME_LOSS_FIELDS[phase]
    )
    return f"{matches} {losses}"


def _write_stacked_smoke_config(path: Path, *, phase: str) -> Path:
    config = stage211_phase_train_config_contract(phase)
    config.update(
        {
            "batch_size": STAGE211_STACKED_SAFE_BATCH_SIZE,
            "batch_token_budget": STAGE211_STACKED_SAFE_FRAME_BUDGET,
            "length_bucket_frame_budget": STAGE211_STACKED_SAFE_FRAME_BUDGET,
            "deepspeed": {
                "train_micro_batch_size_per_gpu": STAGE211_STACKED_SAFE_BATCH_SIZE,
                "gradient_accumulation_steps": 1,
                "train_batch_size": (
                    STAGE211_STACKED_SAFE_BATCH_SIZE * STAGE211_FULL_DATA_WORLD_SIZE
                ),
            },
        }
    )
    save_yaml(path, config)
    return path.resolve()


@pytest.fixture(autouse=True)
def _compact_public_replay_for_phase_gate_tests(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase_gate_test = request.node.name.startswith(
        (
            "test_stage211_phase_gate_",
            "test_stage211_block_phase_gate_",
            "test_stage211_logits_phase_gate_",
            "test_stage211_nano_baseline_receipt_",
            "test_stage211_promotion_receipt_binds_",
            "test_stage211_curriculum_receipt_requires_",
            "test_stage211_stepwise_",
            "test_stage211_trajectory_",
        )
    )
    if not phase_gate_test:
        return
    if request.node.name in {
        "test_stage211_phase_gate_validates_retention_correction_chain",
        "test_stage211_phase_gate_rejects_mutated_correction_focus_summary",
        "test_stage211_phase_gate_rejects_spoofed_early_pass_admission_mode",
        "test_stage211_phase_gate_rejects_mutated_correction_layer_rotation_offset",
        "test_stage211_phase_gate_rejects_correction_checkpoint_chain_break",
    }:
        # These fixtures exercise the correction chain. Production replay deep-validation
        # is covered independently with a schema-v2 runtime-layout artifact.
        monkeypatch.setattr(
            stage211_replay_validator,
            "validate_retention_replay",
            _validate_synthetic_retention_replay,
        )
    compact = {
        dataset: {
            **expected,
            "samples": (int(expected["samples"]) if dataset == "commonvoice_en_test" else 2),
        }
        for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items()
    }
    monkeypatch.setattr(sys.modules[__name__], "STAGE211_PUBLIC_BENCHMARKS", compact)
    monkeypatch.setattr(stage211_gate_module, "STAGE211_PUBLIC_BENCHMARKS", compact)
    monkeypatch.setattr(stage211_phase_gate, "STAGE211_PUBLIC_BENCHMARKS", compact)
    compact_steps_per_epoch = {
        "easy": 3_334,
        "medium": 6_667,
        "hard": 10_001,
        "long": 35,
    }
    compact_curriculum = {
        difficulty: {
            **expected,
            "steps_per_epoch": compact_steps_per_epoch[difficulty],
            "steps": compact_steps_per_epoch[difficulty] * STAGE211_FULL_DATA_EPOCHS,
        }
        for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items()
    }
    monkeypatch.setattr(sys.modules[__name__], "STAGE211_AUDIO_CURRICULUM", compact_curriculum)
    monkeypatch.setattr(stage211_gate_module, "STAGE211_AUDIO_CURRICULUM", compact_curriculum)
    monkeypatch.setattr(stage211_phase_gate, "STAGE211_AUDIO_CURRICULUM", compact_curriculum)
    monkeypatch.setattr(stage211, "STAGE211_AUDIO_CURRICULUM", compact_curriculum)


def test_stage211_public_benchmark_contract_uses_full_real_sets() -> None:
    assert {
        dataset: int(expected["samples"])
        for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items()
    } == {
        "aishell1_test": 7_176,
        "librispeech_test_clean": 2_620,
        "librispeech_test_other": 2_939,
        "commonvoice_en_test": 14_927,
        "wenetspeech_test_net": 24_774,
    }


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
    _write_stacked_smoke_config(smoke_run_dir / "train_config.yaml", phase="block")
    (log_dir / "block_smoke_2steps.log").write_text(
        "[rwkvasr] Distributed init complete.\n"
        "[deepspeed-train] step=1 loss=0.8 peak_reserved=5.50GiB\n"
        "[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB "
        f"{_stage211_smoke_runtime_fields('block')}\n",
        encoding="utf-8",
    )
    marker = stage211_full_phase._audit_smoke(
        phase="block",
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=22.0,
    )
    assert marker["runtime_objective_evidence"] == {
        "schema_version": 1,
        "step": 2,
        "required_match_fields": {
            field: {"matched": 8, "total": 8}
            for field in stage211_full_phase.SMOKE_RUNTIME_MATCH_FIELDS["block"]
        },
        "active_loss_fields": {
            field: 0.125 for field in stage211_full_phase.SMOKE_RUNTIME_LOSS_FIELDS["block"]
        },
        "primary_loss_field": "online_layer_block",
    }
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
        assert dry_run is True
        calls.append((command, env))

    monkeypatch.setattr(stage211_phase_finalizer, "_run", fake_run)
    receipt_path = stage211_phase_finalizer._run_public_eval(
        checkpoint=tmp_path / "checkpoint.pt",
        output_dir=tmp_path / "output",
        manifest_dir=tmp_path / "manifests",
        devices="0,1,2,3",
        dry_run=True,
    )

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][1]["CTC_SHARD_STAGE2"] == "1"
    assert calls[0][1]["CTC_TEXT_NORMALIZATION"] == "ctc"
    assert calls[0][1]["METRIC_NORMALIZATION"] == "ctc"
    assert receipt_path == tmp_path / "output" / "student_prediction_receipt.json"


def test_stage211_stratified_alignment_uses_four_collocated_devices() -> None:
    student_devices, teacher_devices = (
        stage211_phase_finalizer._resolve_stratified_alignment_devices(
            SimpleNamespace(devices="0,1,2,3")
        )
    )

    assert student_devices == ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    assert teacher_devices == student_devices

    with pytest.raises(ValueError, match="Use only one"):
        stage211_phase_finalizer._resolve_stratified_alignment_devices(
            SimpleNamespace(
                devices="0,1,2,3",
                stratified_alignment_device="cuda:0",
                stratified_alignment_devices="0,1,2,3",
            )
        )
    with pytest.raises(ValueError, match="teacher-device count"):
        stage211_phase_finalizer._resolve_stratified_alignment_devices(
            SimpleNamespace(
                devices="0,1,2,3",
                stratified_alignment_teacher_devices="0,1",
            )
        )
    with pytest.raises(ValueError, match="exactly one device"):
        stage211_phase_finalizer._resolve_stratified_alignment_devices(
            SimpleNamespace(
                devices="0,1,2,3",
                stratified_alignment_device="0,1",
            )
        )
    with pytest.raises(ValueError, match="exactly one device"):
        stage211_phase_finalizer._resolve_stratified_alignment_devices(
            SimpleNamespace(
                devices="0",
                stratified_alignment_teacher_device="0,1",
            )
        )


def test_stage211_stratified_alignment_runs_deterministic_device_width_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = [["cell", str(index)] for index in range(9)]
    waves: list[tuple[list[list[str]], bool]] = []
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_run_parallel_wave",
        lambda wave, *, dry_run: waves.append((wave, dry_run)),
    )

    stage211_phase_finalizer._run_parallel_waves(
        commands,
        width=4,
        dry_run=False,
    )

    assert [len(wave) for wave, _ in waves] == [4, 4, 1]
    assert [command for wave, _ in waves for command in wave] == commands
    assert all(dry_run is False for _, dry_run in waves)


def test_stage211_parallel_alignment_failure_terminates_wave_peers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, return_code: int | None) -> None:
            self.return_code = return_code
            self.terminated = False

        def poll(self) -> int | None:
            return self.return_code

        def terminate(self) -> None:
            self.terminated = True
            self.return_code = -15

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is None or timeout == 10
            return int(self.return_code or 0)

        def kill(self) -> None:
            self.return_code = -9

    peer = FakeProcess(None)
    failure = FakeProcess(7)
    pending = [peer, failure]

    def fake_popen(command: list[str], *, cwd: Path) -> FakeProcess:
        assert cwd == stage211_phase_finalizer.REPO_ROOT
        assert command in (["peer"], ["failure"])
        return pending.pop(0)

    monkeypatch.setattr(stage211_phase_finalizer.subprocess, "Popen", fake_popen)

    with pytest.raises(subprocess.CalledProcessError) as error:
        stage211_phase_finalizer._run_parallel_wave(
            [["peer"], ["failure"]],
            dry_run=False,
        )

    assert error.value.returncode == 7
    assert peer.terminated is True


def test_stage211_public_eval_preflight_binds_canonical_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_dir = tmp_path / "manifests"
    prediction_dir = tmp_path / "nano" / "predictions"
    receipt_path = tmp_path / "provenance.json"
    results = [
        {
            "dataset": dataset,
            "manifest_path": str((manifest_dir / f"{dataset}.jsonl").resolve()),
            "nano_prediction_path": str((prediction_dir / f"{dataset}.ctc.jsonl").resolve()),
        }
        for dataset in stage211_phase_finalizer.DATASETS
    ]
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "validate_stage211_nano_public_baseline_receipt",
        lambda path: {"results": results, "total_samples": 52_436},
    )

    validated = stage211_phase_finalizer._validate_public_eval_inputs(
        manifest_dir=manifest_dir,
        nano_prediction_dir=prediction_dir,
        nano_public_baseline_receipt=receipt_path,
    )

    assert validated["total_samples"] == 52_436
    with pytest.raises(ValueError, match="public-eval manifest differs"):
        stage211_phase_finalizer._validate_public_eval_inputs(
            manifest_dir=tmp_path / "different-manifests",
            nano_prediction_dir=prediction_dir,
            nano_public_baseline_receipt=receipt_path,
        )
    with pytest.raises(ValueError, match="Nano prediction differs"):
        stage211_phase_finalizer._validate_public_eval_inputs(
            manifest_dir=manifest_dir,
            nano_prediction_dir=tmp_path / "different-predictions",
            nano_public_baseline_receipt=receipt_path,
        )


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
        "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
        "gate": {
            "max_relative_ratio": 1.2,
            "max_absolute_gap_points": 3.0,
            "requires_every_dataset": True,
        },
        "all_datasets_pass": False,
        "results": [
            public_compare.compare_dataset(
                dataset=dataset,
                nano_path=nano,
                student_path=student,
                normalization="ctc",
                max_relative_ratio=1.2,
                max_absolute_gap_points=3.0,
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
    with pytest.raises(ValueError, match="replayed student_error_rate mismatch"):
        stage211_phase_gate._enrich_public_benchmark(
            report,
            manifest_dir=tmp_path,
        )
    report["results"][0]["student_error_rate"] -= 0.01
    report["all_datasets_pass"] = True
    with pytest.raises(ValueError, match="every-dataset decision mismatch"):
        stage211_phase_gate._enrich_public_benchmark(
            report,
            manifest_dir=tmp_path,
        )


def test_stage211_public_replay_matches_canonical_metric_implementation(
    tmp_path: Path,
) -> None:
    dataset = "librispeech_test_clean"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
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
    nano.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    student_rows = [dict(row) for row in rows]
    student_rows[1]["pred_text"] = "speech"
    student.write_text(
        "".join(json.dumps(row) + "\n" for row in student_rows),
        encoding="utf-8",
    )
    canonical = public_compare.compare_dataset(
        dataset=dataset,
        nano_path=nano,
        student_path=student,
        normalization="ctc",
        max_relative_ratio=1.20,
        max_absolute_gap_points=3.0,
    )
    source = {
        "version": 1,
        "decode": "greedy_ctc",
        "normalization": "ctc",
        "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
        "gate": {
            "max_relative_ratio": 1.20,
            "max_absolute_gap_points": 3.0,
            "requires_every_dataset": True,
        },
        "all_datasets_pass": False,
        "student_checkpoint_path": str(checkpoint.resolve()),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "results": [canonical],
    }

    replayed = replay_stage211_public_comparison(
        source,
        manifest_paths={dataset: manifest},
        benchmarks={dataset: {"language": "en", "metric": "wer", "samples": 2}},
        expected_checkpoint=checkpoint,
    )

    assert replayed["results"][0]["student_error_rate"] == pytest.approx(0.25)
    for key, expected in canonical.items():
        assert replayed["results"][0][key] == expected

    missing_contract = dict(source)
    missing_contract.pop("strip_language_confirmation")
    with pytest.raises(ValueError, match="no AR language-prefix stripping"):
        replay_stage211_public_comparison(
            missing_contract,
            manifest_paths={dataset: manifest},
            benchmarks={dataset: {"language": "en", "metric": "wer", "samples": 2}},
            expected_checkpoint=checkpoint,
        )
    with pytest.raises(ValueError, match="no AR language-prefix stripping"):
        replay_stage211_public_comparison(
            {**source, "strip_language_confirmation": True},
            manifest_paths={dataset: manifest},
            benchmarks={dataset: {"language": "en", "metric": "wer", "samples": 2}},
            expected_checkpoint=checkpoint,
        )


def test_stage211_direct_ctc_language_prefix_is_scored_as_recognition_output(
    tmp_path: Path,
) -> None:
    dataset = "librispeech_test_clean"
    nano = tmp_path / "nano.jsonl"
    student = tmp_path / "student.jsonl"
    nano.write_text(
        '{"utt_id":"utt-1","ref_text":"wrong","pred_text":"wrong"}\n',
        encoding="utf-8",
    )
    student.write_text(
        '{"utt_id":"utt-1","ref_text":"wrong","pred_text":"this is English text wrong"}\n',
        encoding="utf-8",
    )

    result = public_compare.compare_dataset(
        dataset=dataset,
        nano_path=nano,
        student_path=student,
        normalization="ctc",
        max_relative_ratio=1.20,
        max_absolute_gap_points=3.0,
    )

    assert result["nano_error_rate"] == 0.0
    assert result["student_error_rate"] == 4.0
    assert result["student_insertion_rate"] == 4.0


def test_stage211_public_prediction_receipt_binds_checkpoint_and_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = "librispeech_test_clean"
    benchmarks = {dataset: {"language": "en", "metric": "wer", "samples": 2}}
    checkpoint = tmp_path / "checkpoint.pt"
    other_checkpoint = tmp_path / "other.pt"
    checkpoint.write_bytes(b"checkpoint")
    other_checkpoint.write_bytes(b"other")
    manifest = tmp_path / f"{dataset}.jsonl"
    prediction = tmp_path / f"{dataset}.ctc.jsonl"
    manifest.write_text(
        '{"utt_id":"one","text":"hello world"}\n{"utt_id":"two","text":"speech test"}\n',
        encoding="utf-8",
    )
    provenance, tokenizer = stage211_test_student_ctc_context(checkpoint)
    prediction_rows = [
        stage211_test_student_ctc_row(
            utt_id="one",
            ref_text="hello world",
            pred_text="hello world",
            provenance=provenance,
            tokenizer=tokenizer,
        ),
        stage211_test_student_ctc_row(
            utt_id="two",
            ref_text="speech test",
            pred_text="speech",
            provenance=provenance,
            tokenizer=tokenizer,
        ),
    ]
    prediction.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    receipt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=checkpoint,
        manifest_paths={dataset: manifest},
        prediction_paths={dataset: prediction},
        benchmarks=benchmarks,
    )
    receipt_path = tmp_path / "student_prediction_receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    validated = validate_stage211_student_public_prediction_receipt(
        receipt_path,
        expected_checkpoint=checkpoint,
        expected_manifest_paths={dataset: manifest},
        expected_prediction_paths={dataset: prediction},
        benchmarks=benchmarks,
    )
    assert validated["schema_version"] == 2
    assert validated["total_samples"] == 2
    monkeypatch.setattr(
        public_compare,
        "DATASETS",
        {
            dataset: {
                "language": "en",
                "label": "LibriSpeech test-clean",
                "metric": "wer",
                "samples": 2,
            }
        },
    )
    comparison = public_compare.build_report(
        student_prediction_dir=tmp_path,
        nano_predictions={dataset: prediction},
        normalization="ctc",
        max_relative_ratio=1.20,
        max_absolute_gap_points=3.0,
        student_checkpoint=checkpoint,
        student_prediction_receipt=receipt_path,
    )
    assert comparison["student_prediction_receipt_path"] == str(receipt_path.resolve())
    assert comparison["student_prediction_receipt_sha256"] == sha256_file(receipt_path)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        validate_stage211_student_public_prediction_receipt(
            receipt_path,
            expected_checkpoint=other_checkpoint,
            expected_manifest_paths={dataset: manifest},
            expected_prediction_paths={dataset: prediction},
            benchmarks=benchmarks,
        )

    prediction_rows[1] = stage211_test_student_ctc_row(
        utt_id="two",
        ref_text="speech test",
        pred_text="speech test",
        provenance=provenance,
        tokenizer=tokenizer,
    )
    prediction.write_text(
        "".join(json.dumps(row) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match current files"):
        validate_stage211_student_public_prediction_receipt(
            receipt_path,
            expected_checkpoint=checkpoint,
            expected_manifest_paths={dataset: manifest},
            expected_prediction_paths={dataset: prediction},
            benchmarks=benchmarks,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_provenance", "lacks CTC execution provenance"),
        ("nano_row", "lacks CTC execution provenance"),
        ("wrong_checkpoint", "does not match current checkpoint/config files"),
        ("wrong_blank", "does not match current checkpoint/config files"),
        ("decoder_rescore", "not CTC-only greedy output"),
    ),
)
def test_stage211_public_prediction_receipt_rejects_unbound_or_non_ctc_rows(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    dataset = "librispeech_test_clean"
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"student")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"utt_id":"one","text":"hello world"}\n', encoding="utf-8")
    provenance, tokenizer = stage211_test_student_ctc_context(checkpoint)
    row = stage211_test_student_ctc_row(
        utt_id="one",
        ref_text="hello world",
        pred_text="hello world",
        provenance=provenance,
        tokenizer=tokenizer,
    )
    if mutation == "missing_provenance":
        row.pop("inference_provenance")
    elif mutation == "nano_row":
        row = {"utt_id": "one", "ref_text": "hello world", "pred_text": "hello world"}
    elif mutation == "wrong_checkpoint":
        other_checkpoint = tmp_path / "other.pt"
        other_checkpoint.write_bytes(b"other")
        row["inference_provenance"], _ = stage211_test_student_ctc_context(other_checkpoint)
    elif mutation == "wrong_blank":
        row["inference_provenance"] = {
            **provenance,
            "blank_id": 60_514,
        }
    elif mutation == "decoder_rescore":
        row["decoder_score"] = 0.0
    else:
        raise AssertionError(mutation)
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_stage211_student_public_prediction_receipt(
            checkpoint_path=checkpoint,
            manifest_paths={dataset: manifest},
            prediction_paths={dataset: prediction},
            benchmarks={dataset: {"language": "en", "metric": "wer", "samples": 1}},
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
    baseline_reuse_receipt = tmp_path / "calibration-reuse.json"
    baseline_reuse_receipt.write_text("{}\n", encoding="utf-8")
    initialization_receipt = tmp_path / "initialization.json"
    initialization_receipt.write_text("{}\n", encoding="utf-8")
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
            baseline_public_reuse_receipt=baseline_reuse_receipt,
            initialization_receipt=initialization_receipt,
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
    assert phase_gate_command[
        phase_gate_command.index("--baseline-public-reuse-receipt") + 1
    ] == str(baseline_reuse_receipt.resolve())
    assert phase_gate_command[phase_gate_command.index("--initialization-receipt") + 1] == str(
        initialization_receipt.resolve()
    )


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
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "validate_stratified_hidden_eval_v2",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
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
    baseline_public_report = tmp_path / "block-nano-comparison.json"
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
            baseline_public_comparison_report=baseline_public_report,
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
    assert len(pair_commands) == 1 + len(stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS)
    summary_command = next(
        command
        for command in commands
        if str(stage211_phase_finalizer.STRATIFIED_LOGITS_SUMMARY_SCRIPT) in command
    )
    assert summary_command[summary_command.index("--output") + 1] == str(stratified_summary)
    assert phase_gate_command[phase_gate_command.index("--alignment-report") + 1] == str(
        logits_gate_path
    )
    assert phase_gate_command[
        phase_gate_command.index("--baseline-public-comparison-report") + 1
    ] == str(baseline_public_report.resolve())


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
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "validate_stratified_hidden_eval_v2",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
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
    assert len(pair_commands) == 1 + len(stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS)
    sidecar_commands = [command for command in pair_commands if "--eval-bucket-manifest" in command]
    assert len(sidecar_commands) == len(stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS)
    assert {
        command[command.index("--eval-bucket-manifest") + 1] for command in sidecar_commands
    } == {str(Path(cell["manifest_path"]).resolve()) for cell in stratified_cells.values()}
    assert [command[command.index("--device") + 1] for command in sidecar_commands] == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:0",
        "cuda:1",
        "cuda:2",
        "cuda:3",
        "cuda:0",
    ]
    assert [command[command.index("--teacher-device") + 1] for command in sidecar_commands] == [
        command[command.index("--device") + 1] for command in sidecar_commands
    ]
    assert {
        Path(command[command.index("--audio-cache-dir") + 1]).name for command in sidecar_commands
    } == set(stage211_phase_finalizer.STRATIFIED_HIDDEN_CELLS)
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
        "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
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
    assert '--baseline-public-comparison-report "${block_gate_dir}/nano_comparison.json"' in script
    assert '--baseline-public-comparison-report "${logits_gate_dir}/nano_comparison.json"' in script
    assert '--calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}"' in script
    assert '--baseline-public-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}"' in script
    assert '--initialization-receipt "${INITIALIZATION_RECEIPT}"' in script
    assert "create_stage211_nano_baseline_receipt.py" in script
    assert '--output "${NANO_BASELINE_RECEIPT}"' in script
    assert 'START_STAGE="${START_STAGE:-full}"' in script
    assert "run_stage211_mixer_retention_loop.py" in script
    assert "create_stage211_stepwise_report.py" in script
    assert 'run_stepwise_report "${PHASE_GATE_ROOT}/sft/stage211_complete.json"' in script
    assert 'run_stepwise_report "${SFT_CORRECTED_FINAL_ROOT}/stage211_complete.json"' in script
    assert "MIXER_SELECTION" in script
    assert "BLOCK_SELECTION" in script
    assert "LOGITS_SELECTION" in script
    assert '--mixer-gate-selection "${MIXER_SELECTION}"' in script
    assert '--block-gate-selection "${BLOCK_SELECTION}"' in script
    assert '--logits-gate-selection "${LOGITS_SELECTION}"' in script
    assert (
        'LABELED_ROOT="${LABELED_ROOT:-${HOME}/rwkvasr_data/'
        'stage211_sft_full_labeled_v3_public_clean}"' in script
    )
    assert '--labeled-profile-receipt "${LABELED_PROFILE_RECEIPT}"' in script
    assert "post_mixer)" in script
    assert 'SUPPLEMENTAL_POLL_SECONDS="${SUPPLEMENTAL_POLL_SECONDS:-3600}"' in script
    assert "wait_for_supplemental_training_data()" in script
    assert '--inventory "${SUPPLEMENTAL_INVENTORY}"' in script
    assert '--output "${SUPPLEMENTAL_PROFILE_RECEIPT}"' in script
    assert "supplemental inventory, profile, nine-cell eval, and runtime replay validated" in script
    assert "CTC_TEXT_NORMALIZATION=ctc" in script

    main_body = script[script.index("main() {") :]
    readiness_offset = main_body.index("wait_for_supplemental_training_data")
    public_readiness_offset = main_body.index("validate_corrected_public_readiness")
    case_offset = main_body.index('case "${START_STAGE}"')
    assert readiness_offset < public_readiness_offset < case_offset
    assert "--validate-only" in script
    assert '--correction-receipt "${PUBLIC_METRIC_CORRECTION_RECEIPT}"' in script
    assert '--nano-baseline-receipt "${NANO_BASELINE_RECEIPT}"' in script
    full_branch = main_body[main_body.index("full)") : main_body.index("post_mixer)")]
    assert readiness_offset < main_body.index("build_fixed_manifests")
    assert all(
        call in full_branch
        for call in (
            "run_full_mixer_phase",
            "run_full_block_phase",
            "run_full_logits_phase",
            "run_labeled_sft_phase",
        )
    )
    post_mixer_branch = main_body[main_body.index("post_mixer)") : main_body.index("block)")]
    block_branch = main_body[main_body.index("block)") : main_body.index("logits)")]
    for branch in (post_mixer_branch, block_branch):
        offsets = [
            branch.index(call)
            for call in (
                "run_mixer_retention_loop",
                "run_full_block_phase",
                "run_full_logits_phase",
                "run_labeled_sft_phase",
            )
        ]
        assert offsets == sorted(offsets)
    logits_branch = main_body[main_body.index("logits)") : main_body.index("sft)")]
    assert "run_block_correction_loop" in logits_branch
    assert "run_full_block_phase" not in logits_branch
    assert "run_full_logits_phase" in logits_branch
    sft_branch = main_body[main_body.index("sft)") : main_body.index("*)")]
    assert "run_block_correction_loop" in sft_branch
    assert "run_logits_correction_loop" in sft_branch
    assert "run_full_block_phase" not in sft_branch
    assert "run_full_logits_phase" not in sft_branch


@pytest.mark.parametrize(
    ("start_stage", "expected_full_phase"),
    (("logits", "logits"), ("sft", None)),
)
def test_stage211_supervisor_narrow_restart_skips_completed_full_phase_controllers(
    tmp_path: Path,
    start_stage: str,
    expected_full_phase: str | None,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${CALL_LOG}"\n'
        'if [[ "$*" == *create_stage211_stepwise_report.py* ]]; then\n'
        "  output_json=\n"
        "  output_markdown=\n"
        "  while (($#)); do\n"
        '    case "$1" in\n'
        '      --output-json) output_json="$2"; shift 2 ;;\n'
        '      --output-markdown) output_markdown="$2"; shift 2 ;;\n'
        "      *) shift ;;\n"
        "    esac\n"
        "  done\n"
        '  mkdir -p "$(dirname "${output_json}")"\n'
        "  printf '{}\\n' >\"${output_json}\"\n"
        "  printf '# stepwise\\n' >\"${output_markdown}\"\n"
        "fi\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    jq = fake_bin / "jq"
    jq.write_text(
        "#!/usr/bin/env bash\n"
        'name="$(basename "$3" _selected.json)"\n'
        'case "$2" in\n'
        "  .checkpoint_path) printf '/tmp/%s-checkpoint.pt\\n' \"$name\" ;;\n"
        "  .promotion_receipt_path) printf '/tmp/%s-promotion.json\\n' \"$name\" ;;\n"
        "  .gate_dir) printf '/tmp/%s-gate\\n' \"$name\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    jq.chmod(0o755)
    inventory = tmp_path / "supplemental_inventory.json"
    profile = tmp_path / "supplemental_profile_receipt.json"
    stratified = tmp_path / "stratified_hidden_eval_v2.json"
    replay = tmp_path / "retention_replay_v2.json"
    inventory.write_text("{}\n", encoding="utf-8")
    profile.write_text("{}\n", encoding="utf-8")
    stratified.write_text("{}\n", encoding="utf-8")
    replay.write_text("{}\n", encoding="utf-8")
    phase_gate_root = tmp_path / "phase-gates"
    final_report = phase_gate_root / "sft" / "stage211_complete.json"
    final_report.parent.mkdir(parents=True)
    final_report.write_text("{}\n", encoding="utf-8")

    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "start_stage211_abcd_after_calibration.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CALL_LOG": str(call_log),
            "START_STAGE": start_stage,
            "STAGE211_REPO_ROOT": str(REPO_ROOT),
            "SUPPLEMENTAL_INVENTORY": str(inventory),
            "SUPPLEMENTAL_PROFILE_RECEIPT": str(profile),
            "STRATIFIED_HIDDEN_RECEIPT": str(stratified),
            "RETENTION_REPLAY_RECEIPT": str(replay),
            "PHASE_GATE_ROOT": str(phase_gate_root),
            "MIXER_SELECTION": str(tmp_path / "mixer_selected.json"),
            "BLOCK_SELECTION": str(tmp_path / "block_selected.json"),
            "LOGITS_SELECTION": str(tmp_path / "logits_selected.json"),
        },
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )

    calls = call_log.read_text(encoding="utf-8").splitlines()
    full_phase_calls = [call for call in calls if "run_stage211_full_phase_curriculum.py" in call]
    if expected_full_phase is None:
        assert full_phase_calls == []
    else:
        assert len(full_phase_calls) == 1
        assert f"--phase {expected_full_phase}" in full_phase_calls[0]
    assert not any(
        "run_stage211_full_phase_curriculum.py" in call
        and ("--phase mixer" in call or "--phase block" in call)
        for call in calls
    )
    correction_calls = [call for call in calls if "run_stage211_mixer_retention_loop.py" in call]
    assert len(correction_calls) == 3
    assert "--phase block" in correction_calls[1]
    assert "--phase logits" in correction_calls[2]
    assert any("create_stage211_stepwise_report.py" in call for call in calls)


def test_stage211_supervisor_corrected_sft_builds_stepwise_report(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "calls.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${CALL_LOG}"\n'
        'if [[ "$*" == *finalize_stage211_labeled_sft.py* ]]; then\n'
        '  mkdir -p "${PHASE_GATE_ROOT}/sft" "${SFT_OUTPUT_DIR}"\n'
        "  printf '%s\\n' "
        '\'{"pipeline":"stage211","artifact":"final_completion",'
        '"complete":true,"gate_passed":false}\' '
        '>"${PHASE_GATE_ROOT}/sft/stage211_complete.json"\n'
        "  printf '{}\\n' >\"${SFT_OUTPUT_DIR}/sft_complete.json\"\n"
        "  exit 1\n"
        "fi\n"
        'if [[ "$*" == *run_stage211_sft_correction_loop.py* ]]; then\n'
        '  mkdir -p "${SFT_CORRECTED_FINAL_ROOT}"\n'
        "  printf '%s\\n' "
        '\'{"pipeline":"stage211","artifact":"final_completion",'
        '"complete":true,"gate_passed":true}\' '
        '>"${SFT_CORRECTED_FINAL_ROOT}/stage211_complete.json"\n'
        "fi\n"
        'if [[ "$*" == *create_stage211_stepwise_report.py* ]]; then\n'
        "  output_json=\n"
        "  output_markdown=\n"
        "  while (($#)); do\n"
        '    case "$1" in\n'
        '      --output-json) output_json="$2"; shift 2 ;;\n'
        '      --output-markdown) output_markdown="$2"; shift 2 ;;\n'
        "      *) shift ;;\n"
        "    esac\n"
        "  done\n"
        '  mkdir -p "$(dirname "${output_json}")"\n'
        "  printf '{}\\n' >\"${output_json}\"\n"
        "  printf '# stepwise\\n' >\"${output_markdown}\"\n"
        "fi\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    jq = fake_bin / "jq"
    jq.write_text(
        "#!/usr/bin/env bash\n"
        'name="$(basename "$3" _selected.json)"\n'
        'case "$2" in\n'
        "  .checkpoint_path) printf '/tmp/%s-checkpoint.pt\\n' \"$name\" ;;\n"
        "  .promotion_receipt_path) printf '/tmp/%s-promotion.json\\n' \"$name\" ;;\n"
        "  .gate_dir) printf '/tmp/%s-gate\\n' \"$name\" ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    jq.chmod(0o755)

    inventory = tmp_path / "supplemental_inventory.json"
    profile = tmp_path / "supplemental_profile_receipt.json"
    stratified = tmp_path / "stratified_hidden_eval_v2.json"
    replay = tmp_path / "retention_replay_v2.json"
    for path in (inventory, profile, stratified, replay):
        path.write_text("{}\n", encoding="utf-8")
    phase_gate_root = tmp_path / "phase-gates"
    sft_output_dir = tmp_path / "sft-run"
    corrected_root = phase_gate_root / "sft_corrected"

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "start_stage211_abcd_after_calibration.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "CALL_LOG": str(call_log),
            "START_STAGE": "sft",
            "STAGE211_REPO_ROOT": str(REPO_ROOT),
            "SUPPLEMENTAL_INVENTORY": str(inventory),
            "SUPPLEMENTAL_PROFILE_RECEIPT": str(profile),
            "STRATIFIED_HIDDEN_RECEIPT": str(stratified),
            "RETENTION_REPLAY_RECEIPT": str(replay),
            "PHASE_GATE_ROOT": str(phase_gate_root),
            "SFT_OUTPUT_DIR": str(sft_output_dir),
            "SFT_CORRECTED_FINAL_ROOT": str(corrected_root),
            "MIXER_SELECTION": str(tmp_path / "mixer_selected.json"),
            "BLOCK_SELECTION": str(tmp_path / "block_selected.json"),
            "LOGITS_SELECTION": str(tmp_path / "logits_selected.json"),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    assert (corrected_root / "stage211_stepwise_results.json").is_file()
    assert (corrected_root / "stage211_stepwise_results.md").is_file()
    calls = call_log.read_text(encoding="utf-8").splitlines()
    finalizer_index = next(
        index for index, call in enumerate(calls) if "finalize_stage211_labeled_sft.py" in call
    )
    correction_index = next(
        index for index, call in enumerate(calls) if "run_stage211_sft_correction_loop.py" in call
    )
    stepwise_index = next(
        index for index, call in enumerate(calls) if "create_stage211_stepwise_report.py" in call
    )
    assert finalizer_index < correction_index < stepwise_index
    assert str(corrected_root / "stage211_complete.json") in calls[stepwise_index]


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
    stratified = tmp_path / "stratified_hidden_eval_v2.json"
    replay = tmp_path / "retention_replay_v2.json"
    if create_inputs:
        inventory.write_text("{}\n", encoding="utf-8")
        profile.write_text("{}\n", encoding="utf-8")
        stratified.write_text("{}\n", encoding="utf-8")
        replay.write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "start_stage211_abcd_after_calibration.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "STAGE211_REPO_ROOT": str(REPO_ROOT),
            "SUPPLEMENTAL_INVENTORY": str(inventory),
            "SUPPLEMENTAL_PROFILE_RECEIPT": str(profile),
            "STRATIFIED_HIDDEN_RECEIPT": str(stratified),
            "RETENTION_REPLAY_RECEIPT": str(replay),
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


def test_stage211_hourly_monitor_classifies_only_formal_training_logs(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    formal_log = tmp_path / "formal.log"
    formal_log.write_text(
        "[rwkvasr] Distributed init complete. world_size=4\n"
        "[deepspeed-train] step=10 loss=0.1000\n",
        encoding="utf-8",
    )
    supplemental_log = tmp_path / "supplemental.log"
    supplemental_log.write_text("Traceback from a controlled audit restart\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; while IFS= read -r -d "" path; do basename "$path"; done '
            '< <(stage211_recent_formal_training_logs "$2")',
            "stage211-monitor-test",
            str(monitor_script),
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["formal.log"]


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
        "[deepspeed-train] step=21 loss=0.2000\n[deepspeed-train] step=25 loss=0.1000\n",
        encoding="utf-8",
    )
    config = tmp_path / "stage211_test.yaml"
    config.write_text(
        f"output_dir: {run_dir}\nwandb_run_name: stage211-live-test\nmax_steps: 100\n",
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


def test_stage211_hourly_monitor_reports_latest_fixed_eval_without_calling_it_public(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "step_eval_baseline.yaml").write_text(
        "step: 0\neval_loss: 0.1000000000\neval_samples: 256\n",
        encoding="utf-8",
    )
    (run_dir / "step_eval_layers_step-10.yaml").write_text(
        "step: 10\neval_loss: 0.0900000000\neval_samples: 256\n",
        encoding="utf-8",
    )
    latest = run_dir / "step_eval_layers_step-20.yaml"
    latest.write_text(
        "step: 20\neval_loss: 0.1750000000\neval_samples: 256\n",
        encoding="utf-8",
    )
    (run_dir / "step_eval_layers_step-999.yaml.tmp").write_text(
        "step: 999\neval_loss: 0.001\neval_samples: 1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_fixed_step_eval_progress "$2"',
            "stage211-monitor-test",
            str(monitor_script),
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed_eval_scope=fixed_hidden_not_public_wer_cer status=ok" in result.stdout
    assert (
        "sentinel_cell=unknown sentinel_language=unknown sentinel_sources=unknown" in result.stdout
    )
    assert "baseline_loss=0.1000000000 latest_step=20 latest_loss=0.1750000000" in result.stdout
    assert "delta_abs=+0.0750000000 delta_pct=+75.0000 trend=regressed" in result.stdout
    assert "eval_samples=256" in result.stdout
    assert f"report={latest}" in result.stdout
    assert "step-999" not in result.stdout
    assert (
        "fixed_eval_layer_progress status=unavailable reason=missing_layer_metrics "
        "baseline_layers=absent candidate_layers=absent"
    ) in result.stdout


def _stage211_monitor_layer_metrics(
    *,
    loss: float,
    cosine: float,
) -> dict[str, dict[str, float]]:
    return {
        str(layer_id): {
            "loss": loss,
            "cosine": cosine,
        }
        for layer_id in range(70)
    }


def test_stage211_hourly_monitor_reports_gate_like_per_layer_progress(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    baseline_layers = _stage211_monitor_layer_metrics(loss=1.0, cosine=0.5)
    candidate_layers = _stage211_monitor_layer_metrics(loss=0.8, cosine=0.7)
    candidate_layers["69"] = {"loss": 1.2, "cosine": 0.4}
    baseline_component = _stage211_monitor_layer_metrics(loss=0.5, cosine=0.6)
    candidate_component = _stage211_monitor_layer_metrics(loss=0.4, cosine=0.8)
    save_yaml(
        run_dir / "step_eval_baseline.yaml",
        {
            "step": 0,
            "eval_loss": 1.0,
            "eval_samples": 256,
            "layer_metrics": baseline_layers,
            "layer_component_metrics": {"mixer": baseline_component},
        },
    )
    latest = run_dir / "step_eval_layers_step-10000.yaml"
    save_yaml(
        latest,
        {
            "step": 10000,
            "eval_loss": 0.8,
            "eval_samples": 256,
            "layers": candidate_layers,
            "layer_components": {"mixer": candidate_component},
        },
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_fixed_step_eval_progress "$2"',
            "stage211-monitor-test",
            str(monitor_script),
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed_eval_layer_progress status=ok layer_count=70" in result.stdout
    assert "loss_improved=69 loss_regressed=1 loss_flat=0" in result.stdout
    assert "cosine_improved=69 cosine_regressed=1 cosine_flat=0" in result.stdout
    assert "worst_loss_layer=69 worst_loss_delta=+0.2000000000" in result.stdout
    assert "worst_cosine_layer=69 worst_cosine_delta=-0.1000000000" in result.stdout
    assert "min_68_loss=true min_68_cosine=true" in result.stdout
    assert "fixed_eval_component_progress status=ok component=mixer layer_count=70" in result.stdout
    assert "loss_improved=70 loss_regressed=0 loss_flat=0" in result.stdout
    assert "cosine_improved=70 cosine_regressed=0 cosine_flat=0" in result.stdout


def test_stage211_fixed_eval_layer_reporter_rejects_partial_formal_maps() -> None:
    baseline = {
        "layer_metrics": _stage211_monitor_layer_metrics(loss=1.0, cosine=0.5),
    }
    candidate = {
        "layers": _stage211_monitor_layer_metrics(loss=0.8, cosine=0.7),
    }
    candidate["layers"].pop("69")

    with pytest.raises(ValueError, match="does not contain exactly 70 layers"):
        stage211_fixed_eval_layer_progress.build_fixed_eval_layer_progress(
            baseline,
            candidate,
        )


def test_stage211_hourly_monitor_identifies_sha_bound_easy_zh_sentinel(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fixed_part = tmp_path / "part_000000.jsonl"
    fixed_part.write_text('{"key":"fixed-easy-zh"}\n', encoding="utf-8")
    fixed_sha = sha256_file(fixed_part)
    provenance = (
        "eval_provenance:\n"
        "  parts:\n"
        f"  - path: {fixed_part}\n"
        f"    sha256: {fixed_sha}\n"
        "    num_samples: 256\n"
    )
    (run_dir / "step_eval_baseline.yaml").write_text(
        "step: 0\neval_loss: 0.1000000000\neval_samples: 256\n" + provenance,
        encoding="utf-8",
    )
    (run_dir / "step_eval_layers_step-10.yaml").write_text(
        "step: 10\neval_loss: 0.0900000000\neval_samples: 256\n" + provenance,
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_fixed_step_eval_progress "$2"',
            "stage211-monitor-test",
            str(monitor_script),
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "FIXED_EVAL_CANONICAL_PART": str(fixed_part),
            "FIXED_EVAL_CANONICAL_PART_SHA256": fixed_sha,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "sentinel_cell=easy_zh sentinel_language=zh "
        "sentinel_sources=aishell3,commonvoice_cn" in result.stdout
    )

    fixed_part.write_text('{"key":"changed-after-binding"}\n', encoding="utf-8")
    changed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_fixed_step_eval_progress "$2"',
            "stage211-monitor-test",
            str(monitor_script),
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "FIXED_EVAL_CANONICAL_PART": str(fixed_part),
            "FIXED_EVAL_CANONICAL_PART_SHA256": fixed_sha,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert changed.returncode == 0, changed.stderr
    assert (
        "sentinel_cell=unknown sentinel_language=unknown sentinel_sources=unknown" in changed.stdout
    )


def test_stage211_hourly_monitor_rejects_fixed_eval_step_filename_mismatch(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "step_eval_baseline.yaml").write_text(
        "step: 0\neval_loss: 0.1000000000\neval_samples: 256\n",
        encoding="utf-8",
    )
    report = run_dir / "step_eval_layers_step-20.yaml"
    report.write_text(
        "step: 19\neval_loss: 0.0900000000\neval_samples: 256\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_fixed_step_eval_progress "$2"',
            "stage211-monitor-test",
            str(monitor_script),
            str(run_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed_eval_scope=fixed_hidden_not_public_wer_cer status=invalid" in result.stdout
    assert "recorded_step=19 filename_step=20" in result.stdout


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
        '--config-yaml /tmp/stage211_active.yaml --rank ${rank}"\n'
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


def test_stage211_hourly_monitor_invokes_social_pcm_progress_reporter(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    inventory = tmp_path / "materialized_inventory.json"
    inventory.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "filtered"
    output_root.mkdir()
    reporter = tmp_path / "reporter.py"
    reporter.write_text(
        "import sys\nprint('reporter_args=' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_social_pcm_progress',
            "stage211-monitor-test",
            str(monitor_script),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "MONITOR_PYTHON": sys.executable,
            "SOCIAL_PCM_INVENTORY": str(inventory),
            "SOCIAL_PCM_OUTPUT_ROOT": str(output_root),
            "SOCIAL_PCM_PROGRESS_REPORTER": str(reporter),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        f"reporter_args=--materialized-inventory|{inventory}|--output-root|{output_root}"
    )


def test_stage211_hourly_monitor_distinguishes_social_source_and_v2_rebase(
    tmp_path: Path,
) -> None:
    monitor_script = REPO_ROOT / "scripts" / "monitor_stage211_abcd.sh"
    inventory = tmp_path / "materialized_inventory.json"
    inventory.write_text("{}\n", encoding="utf-8")
    source_root = tmp_path / "filtered-v1"
    rebase_root = tmp_path / "filtered-v2"
    source_root.mkdir()
    rebase_root.mkdir()
    reporter = tmp_path / "reporter.py"
    reporter.write_text(
        "import sys\nprint('reporter_args=' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; stage211_emit_social_pcm_readiness',
            "stage211-monitor-test",
            str(monitor_script),
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "MONITOR_PYTHON": sys.executable,
            "SOCIAL_PCM_INVENTORY": str(inventory),
            "SOCIAL_PCM_SOURCE_OUTPUT_ROOT": str(source_root),
            "SOCIAL_PCM_OUTPUT_ROOT": str(rebase_root),
            "SOCIAL_PCM_PROGRESS_REPORTER": str(reporter),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "-- social source exact PCM fingerprints --",
        f"reporter_args=--materialized-inventory|{inventory}|--output-root|{source_root}",
        "-- social corrected-public v2 rebase --",
        f"reporter_args=--materialized-inventory|{inventory}|--output-root|{rebase_root}",
    ]


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
    assert progress["language_summaries"]["en"]["gate_passed"] is True
    assert progress["language_summaries"]["zh"]["gate_passed"] is True

    for result in candidate_results:
        dataset = str(result["dataset"])
        if STAGE211_PUBLIC_BENCHMARKS[dataset]["language"] == "en":
            result["student_error_rate"] = 0.8
            result["student_deletion_rate"] = 0.6
    one_language_only = stage211_phase_gate._build_public_progress(
        baseline={"results": baseline_results},
        candidate={"results": candidate_results},
    )
    assert one_language_only["gate_passed"] is False
    assert one_language_only["language_summaries"]["en"]["gate_passed"] is False
    assert one_language_only["language_summaries"]["zh"]["gate_passed"] is True

    for result in candidate_results:
        result["student_error_rate"] = 0.7
        result["student_deletion_rate"] = 0.5
    candidate_results[0]["student_error_rate"] = 0.84
    rejected = stage211_phase_gate._build_public_progress(
        baseline={"results": baseline_results},
        candidate={"results": candidate_results},
    )
    assert rejected["gate_passed"] is False
    assert rejected["results"][0]["within_regression_limit"] is False


@pytest.mark.parametrize("phase", ("mixer", "block"))
def test_stage211_hidden_phases_require_every_dataset_nano_proximity(
    phase: str,
) -> None:
    assert (
        stage211_gate_module.stage211_phase_gate_decision(
            phase=phase,
            alignment_gate_passed=True,
            public_progress_gate_passed=True,
            trajectory_retention_gate_passed=True,
            all_datasets_pass=False,
        )
        is False
    )


@pytest.mark.parametrize("phase", ("mixer", "block", "logits"))
def test_stage211_every_phase_requires_progress_and_nano_proximity(phase: str) -> None:
    assert (
        stage211_gate_module.stage211_phase_gate_decision(
            phase=phase,
            alignment_gate_passed=True,
            public_progress_gate_passed=True,
            trajectory_retention_gate_passed=True,
            all_datasets_pass=False,
        )
        is False
    )
    assert (
        stage211_gate_module.stage211_phase_gate_decision(
            phase=phase,
            alignment_gate_passed=True,
            public_progress_gate_passed=False,
            trajectory_retention_gate_passed=True,
            all_datasets_pass=True,
        )
        is False
    )
    assert (
        stage211_gate_module.stage211_phase_gate_decision(
            phase=phase,
            alignment_gate_passed=True,
            public_progress_gate_passed=True,
            trajectory_retention_gate_passed=True,
            all_datasets_pass=True,
        )
        is True
    )


@pytest.mark.parametrize(
    ("completed_rounds", "correction_started", "gate_passed"),
    (
        (0, False, True),
        (1, True, False),
        (2, True, False),
        (3, True, True),
        (32, True, True),
    ),
)
def test_stage211_correction_round_promotion_requires_three_complete_rounds(
    completed_rounds: int,
    correction_started: bool,
    gate_passed: bool,
) -> None:
    result = build_stage211_correction_round_promotion_gate(completed_rounds)

    assert result["completed_rounds"] == completed_rounds
    assert result["correction_started"] is correction_started
    assert result["guaranteed_rounds"] == 3
    assert result["gate_passed"] is gate_passed


@pytest.mark.parametrize("completed_rounds", (-1, 33))
def test_stage211_correction_round_promotion_rejects_invalid_counts(
    completed_rounds: int,
) -> None:
    with pytest.raises(ValueError, match="correction-round count"):
        build_stage211_correction_round_promotion_gate(completed_rounds)


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
    supplemental_inventory, supplemental_profile = _write_supplemental_inventory_fixture(tmp_path)
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

    log_path.write_text(
        valid_log.replace("step=2", "step=20"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not execute two training steps"):
        stage211_full_phase._audit_smoke(
            phase=phase,
            smoke_run_dir=smoke_run_dir,
            init_checkpoint=init_checkpoint,
            easy_manifest=easy_manifest,
            max_peak_reserved_gib=22.0,
        )


def test_stage211_logits_smoke_requires_complete_full_logit_match(
    tmp_path: Path,
) -> None:
    phase = "logits"
    init_checkpoint = tmp_path / "init.pt"
    init_checkpoint.write_bytes(b"init")
    easy_manifest = tmp_path / "easy.json"
    easy_manifest.write_text("{}\n", encoding="utf-8")
    smoke_run_dir = tmp_path / "smoke"
    log_dir = smoke_run_dir / "logs"
    log_dir.mkdir(parents=True)
    torch.save({"step": 2}, smoke_run_dir / "step-2.pt")
    _write_stacked_smoke_config(smoke_run_dir / "train_config.yaml", phase=phase)
    log_path = log_dir / "logits_smoke_2steps.log"

    def audit(match_field: str) -> dict[str, object]:
        runtime_fields = _stage211_smoke_runtime_fields(phase)
        runtime_fields = runtime_fields.replace("online_full_match=8/8", match_field)
        log_path.write_text(
            "[rwkvasr] Distributed init complete. world_size=4\n"
            "[deepspeed-train] step=2 loss=0.2000 peak_reserved=20.50GiB "
            f"{runtime_fields} online_full_missing=0\n",
            encoding="utf-8",
        )
        return stage211_full_phase._audit_smoke(
            phase=phase,
            smoke_run_dir=smoke_run_dir,
            init_checkpoint=init_checkpoint,
            easy_manifest=easy_manifest,
            max_peak_reserved_gib=22.0,
        )

    assert audit("online_full_match=8/8")["complete"] is True
    with pytest.raises(ValueError, match="lacks step-2 online_full_match evidence"):
        audit("")
    with pytest.raises(ValueError, match="incomplete step-2 online_full_match: 7/8"):
        audit("online_full_match=7/8")
    with pytest.raises(ValueError, match="incomplete step-2 online_full_match: 0/0"):
        audit("online_full_match=0/0")


def test_stage211_block_smoke_requires_each_runtime_objective() -> None:
    valid = _stage211_smoke_runtime_fields("block")
    evidence = stage211_full_phase._smoke_runtime_objective_evidence(
        phase="block",
        step_two_line=f"[deepspeed-train] step=2 {valid}",
    )
    assert evidence["primary_loss_field"] == "online_layer_block"

    missing_ffn = valid.replace("online_layer_ffn=0.1250", "")
    with pytest.raises(ValueError, match="lacks step-2 online_layer_ffn telemetry"):
        stage211_full_phase._smoke_runtime_objective_evidence(
            phase="block",
            step_two_line=f"[deepspeed-train] step=2 {missing_ffn}",
        )

    zero_primary = valid.replace("online_layer_block=0.1250", "online_layer_block=0.0000")
    with pytest.raises(ValueError, match="online_layer_block must be positive"):
        stage211_full_phase._smoke_runtime_objective_evidence(
            phase="block",
            step_two_line=f"[deepspeed-train] step=2 {zero_primary}",
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
    assert config["webdataset_bucket_manifest_path"] == str(manifest)
    assert config["webdataset_split"] == "train"

    changed = dict(config)
    changed["webdataset_split"] = "easy"
    with pytest.raises(ValueError, match="webdataset_split=train"):
        stage211._validate_runtime_data_binding(
            config=changed,
            phase=phase,
            bucket_manifest=manifest,
            audio_data_audit=audit,
            labeled_webdataset_root=None,
            labeled_length_index=None,
        )

    changed = dict(config)
    changed["webdataset_bucket_manifest_path"] = str(tmp_path / "old-easy-manifest.json")
    with pytest.raises(ValueError, match="differs from the admitted manifest"):
        stage211._validate_runtime_data_binding(
            config=changed,
            phase=phase,
            bucket_manifest=manifest,
            audio_data_audit=audit,
            labeled_webdataset_root=None,
            labeled_length_index=None,
        )


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


@pytest.mark.parametrize("phase_name", ("block", "logits"))
def test_stage211_stacked_full_profile_smoke_uses_safe_capacity(
    tmp_path: Path,
    phase_name: str,
) -> None:
    phase = stage211.PHASES[phase_name]
    segment = stage211._segments(phase=phase, smoke=True)[0]
    config = stage211._config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / phase_name,
        init_checkpoint=tmp_path / "selected.pt",
        bucket_manifest=tmp_path / "manifest.json",
        resume=False,
        smoke=True,
        full_data_profile=True,
    )

    assert config["batch_size"] == STAGE211_STACKED_SAFE_BATCH_SIZE
    assert config["batch_token_budget"] == STAGE211_STACKED_SAFE_FRAME_BUDGET
    assert config["length_bucket_frame_budget"] == STAGE211_STACKED_SAFE_FRAME_BUDGET
    assert config["deepspeed"]["train_micro_batch_size_per_gpu"] == (
        STAGE211_STACKED_SAFE_BATCH_SIZE
    )
    assert config["deepspeed"]["train_batch_size"] == (
        STAGE211_STACKED_SAFE_BATCH_SIZE * STAGE211_FULL_DATA_WORLD_SIZE
    )


def test_stage211_admitted_block_profile_can_disable_checkpointing(
    tmp_path: Path,
) -> None:
    phase = stage211.PHASES["block"]
    segment = stage211._segments(phase=phase, smoke=False)[0]
    admission = {
        "receipt_path": str((tmp_path / "admission.json").resolve()),
        "receipt_sha256": "a" * 64,
        "selected_profile": {
            "name": "no_ckpt_batch12_frames8k",
            "batch_size": 12,
            "frame_budget": 8_000,
            "num_workers": 8,
            "gradient_checkpointing": False,
        },
    }

    config = stage211._config(
        phase=phase,
        segment=segment,
        output_dir=tmp_path / "block",
        init_checkpoint=tmp_path / "selected.pt",
        bucket_manifest=tmp_path / "manifest.json",
        resume=False,
        smoke=False,
        full_data_profile=True,
        batch_profile_admission=admission,
    )

    assert config["gradient_checkpointing"] is False
    assert config["batch_size"] == 12
    assert config["stage211_batch_profile_gradient_checkpointing"] is False
    validated = validate_stage211_phase_train_config(config, phase="block")
    assert validated["gradient_checkpointing"] is False

    config["stage211_batch_profile_gradient_checkpointing"] = True
    with pytest.raises(ValueError, match="gradient_checkpointing mismatch"):
        validate_stage211_phase_train_config(config, phase="block")


@pytest.mark.parametrize("phase_name", tuple(stage211.PHASES))
def test_stage211_config_preflights_phase_objective_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase_name: str,
) -> None:
    calls: list[str] = []
    original = stage211.validate_stage211_phase_train_config

    def recording_validator(config: dict[str, object], *, phase: str) -> dict[str, object]:
        calls.append(phase)
        return original(config, phase=phase)

    monkeypatch.setattr(
        stage211,
        "validate_stage211_phase_train_config",
        recording_validator,
    )

    _config_for_phase(tmp_path, phase_name)

    assert calls == [phase_name]


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
def test_stage211_formal_runs_enable_segment_scoped_wandb(
    tmp_path: Path,
    phase_name: str,
) -> None:
    config = _config_for_phase(tmp_path, phase_name)
    phase = stage211.PHASES[phase_name]
    segment = stage211._segments(
        phase=phase,
        smoke=False,
        formal_steps=12_045 if phase.requires_labels else None,
    )[0]

    assert config["wandb_enabled"] is True
    assert config["wandb_project"] == stage211.STAGE211_WANDB_PROJECT
    assert config["wandb_run_name"] == f"{phase_name}_{segment['name']}"


@pytest.mark.parametrize("phase_name", tuple(stage211.PHASES))
def test_stage211_smoke_runs_do_not_publish_wandb(
    tmp_path: Path,
    phase_name: str,
) -> None:
    config = _config_for_phase(tmp_path, phase_name, smoke=True)

    assert config["wandb_enabled"] is False
    assert config["wandb_project"] == stage211.STAGE211_WANDB_PROJECT


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
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(expected_ffn_weight)
    assert config["ctc_teacher_online_keep_layer_hiddens_on_device"] is True
    assert config["ctc_teacher_online_compute_ctc_outputs"] is (phase_name in {"logits", "sft"})
    assert config["ctc_teacher_online_capture_layer_inputs"] is (phase_name == "mixer")
    assert config["ctc_student_compute_ctc_logits"] is (phase_name in {"logits", "sft"})
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


def test_stage211_legacy_mixer_projection_config_requires_exact_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_for_phase(tmp_path, "mixer")
    for key in stage211_gate_module._STAGE211_PROJECTION_ELISION_CONFIG_KEYS:
        config.pop(key)
    normalized_sha256 = hashlib.sha256(
        json.dumps(
            config,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="compute_ctc_outputs mismatch"):
        validate_stage211_phase_train_config(config, phase="mixer")

    monkeypatch.setattr(
        stage211_gate_module,
        "_STAGE211_LEGACY_MIXER_CONFIG_SHA256",
        frozenset({normalized_sha256}),
    )
    assert validate_stage211_phase_train_config(config, phase="mixer") == (
        stage211_phase_train_config_contract("mixer")
    )

    config["lr"] = 9.0e-6
    with pytest.raises(ValueError, match="mismatch"):
        validate_stage211_phase_train_config(config, phase="mixer")


def test_stage211_mixer_phase_has_only_teacher_forced_mixer_objective(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "mixer")

    assert config["lr"] == pytest.approx(3.0e-6)
    assert config["gradient_checkpointing"] is False
    assert config["ctc_teacher_online_layer_input_mode"] == "teacher_forced"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_block_loss_weight"] == 0.0
    assert config["ctc_teacher_online_encoder_loss_weight"] == 0.0
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == 0.0
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_blank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_sample_count"] == 70
    assert config["ctc_teacher_online_layer_boundary_ids"] == []
    assert config["ctc_teacher_online_layer_include_boundaries"] is False


def test_stage211_block_phase_chains_stacked_block_without_logits(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "block")

    assert config["lr"] == pytest.approx(2.0e-6)
    assert config["gradient_checkpointing"] is True
    assert config["ctc_teacher_online_layer_input_mode"] == "stacked"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_encoder_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == pytest.approx(0.5)
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_blank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == 0.0
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_sample_count"] == 70
    assert config["ctc_teacher_online_layer_boundary_ids"] == []
    assert config["ctc_teacher_online_layer_include_boundaries"] is False


def test_stage211_logits_phase_enables_outputs_after_hidden_anchors(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "logits")

    assert config["lr"] == pytest.approx(3.0e-7)
    assert config["gradient_checkpointing"] is True
    assert config["ctc_teacher_online_layer_input_mode"] == "stacked"
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_full_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.25)
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == pytest.approx(1.0)
    assert config["ctc_teacher_online_conditional_nonblank_hard_loss_weight"] == pytest.approx(
        0.125
    )
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_sample_count"] == 70
    assert config["ctc_teacher_online_layer_include_boundaries"] is False
    assert config["ctc_teacher_online_layer_boundary_ids"] == []


def test_stage211_sft_phase_uses_labels_after_logits_with_low_teacher_anchors(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "sft")
    suppressed_token_ids = list(stage211.stage211_sft_ctc_suppressed_token_ids())

    assert config["lr"] == pytest.approx(3.0e-7)
    assert config["gradient_checkpointing"] is True
    assert config["allow_missing_targets"] is False
    assert config["ctc_loss_weight"] == pytest.approx(1.0)
    assert config["decoder_loss_weight"] == 0.0
    assert config["ctc_suppress_non_pronunciation_tokens"] is True
    assert len(suppressed_token_ids) == 2_114
    assert 60_514 in suppressed_token_ids
    assert 60_515 not in suppressed_token_ids
    assert config["ctc_suppressed_token_ids"] == suppressed_token_ids
    assert config["ctc_teacher_online_project_ignored_token_ids"] == suppressed_token_ids
    assert config["step_eval_split"] == "eval"
    assert config["step_eval_every"] == 2_000
    assert config["step_eval_samples"] == 256
    assert config["step_eval_shuffle"] is False
    assert config["step_eval_feature_seed"] == 0
    assert config["freeze_encoder_except_time_mixer"] is True
    assert config["freeze_ctc_decoder"] is True
    assert config["freeze_ctc_head"] is True
    assert config["ctc_teacher_online_full_loss_weight"] == 0.0
    assert config["ctc_teacher_online_blank_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_conditional_nonblank_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_layer_mixer_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == pytest.approx(0.05)
    assert config["ctc_teacher_online_layer_block_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_encoder_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_decoder_hidden_loss_weight"] == pytest.approx(0.10)
    assert config["ctc_teacher_online_sequence_loss_weight"] == 0.0
    assert config["ctc_teacher_online_nonblank_window_loss_weight"] == 0.0
    assert config["ctc_teacher_online_layer_sample_count"] == 70
    assert config["ctc_teacher_online_layer_boundary_ids"] == []
    assert config["ctc_teacher_online_layer_include_boundaries"] is False


def test_stage211_sft_phase_contract_rejects_teacher_student_support_mismatch(
    tmp_path: Path,
) -> None:
    config = _config_for_phase(tmp_path, "sft")
    config["ctc_teacher_online_project_ignored_token_ids"] = config[
        "ctc_teacher_online_project_ignored_token_ids"
    ][:-1]

    with pytest.raises(
        ValueError,
        match="sft train config ctc_teacher_online_project_ignored_token_ids mismatch",
    ):
        validate_stage211_phase_train_config(config, phase="sft")


@pytest.mark.parametrize("phase_name", ("mixer", "block", "logits"))
def test_stage211_pre_sft_phases_keep_standard_nano_blank_projection_ignore(
    tmp_path: Path,
    phase_name: str,
) -> None:
    config = _config_for_phase(tmp_path, phase_name)

    assert config["ctc_teacher_online_project_ignored_token_ids"] == [60_514]
    assert "ctc_suppressed_token_ids" not in config


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
        "selected_counts_by_source": {source: 1 for source in STAGE211_SUPPLEMENTAL_SOURCES},
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
    run_dir = tmp_path / f"{phase}-supplemental-run"
    run_dir.mkdir(exist_ok=True)
    provenance = tmp_path / "supplemental-provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    train_config = tmp_path / "supplemental-train-config.yaml"
    train_config_payload = stage211_phase_train_config_contract(phase)
    train_config_payload.update(
        {
            "ctc_teacher_online_model_path": str(nano_teacher_dir.resolve()),
            "max_steps": int(profile["steps"]),
            "webdataset_bucket_manifest_path": profile["bucket_manifest_path"],
            "step_eval_every": 10_000,
            "step_eval_samples": 256,
            "step_eval_split": "eval",
            "step_eval_shuffle": False,
            "step_eval_feature_seed": 0,
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
        "run_dir": str(run_dir.resolve()),
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
    assert "loader_split=train" in admitted.stdout
    assert "split=easy" not in admitted.stdout
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
    assert config["length_bucket_schedule_block_size"] == 64
    assert config["bucket_source_interleave_block_size"] == 64
    assert config["bucket_serialize_reads"] is True
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


def test_stage211_sft_labeled_data_audit_and_epoch_estimate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webdataset_root, length_index, manifest = _write_minimal_labeled_data(tmp_path)
    preparation = stage211._label_preparation_proof(
        webdataset_root=webdataset_root,
        length_index_path=length_index,
    )
    preparation.update(
        {
            "source_counts": {"unknown": 2},
            "language_counts": {"unknown": 2},
        }
    )
    monkeypatch.setattr(stage211, "_label_preparation_proof", lambda **_kwargs: preparation)

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


def test_stage211_full_phase_admits_only_receipt_bound_difficulty_manifests(
    tmp_path: Path,
) -> None:
    receipt_path, runtime_manifests = _write_loaded_manifest_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    stage211_full_phase._require_loaded_manifest_bindings(
        manifests=runtime_manifests,
        receipt=receipt,
    )

    swapped = dict(runtime_manifests)
    swapped["hard"] = runtime_manifests["easy"]
    with pytest.raises(ValueError, match="hard manifest differs"):
        stage211_full_phase._require_loaded_manifest_bindings(
            manifests=swapped,
            receipt=receipt,
        )

    runtime_manifests["hard"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hard manifest SHA-256 differs"):
        stage211_full_phase._require_loaded_manifest_bindings(
            manifests=runtime_manifests,
            receipt=receipt,
        )


def test_stage211_full_phase_revalidates_manifests_before_every_segment_runner() -> None:
    tree = ast.parse(inspect.getsource(stage211_full_phase.run_phase))
    call_lines: dict[str, list[int]] = {
        "_require_loaded_manifest_bindings": [],
        "_runner_command": [],
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in call_lines:
            call_lines[node.func.id].append(node.lineno)

    revalidation_lines = sorted(call_lines["_require_loaded_manifest_bindings"])
    runner_lines = sorted(call_lines["_runner_command"])
    assert len(revalidation_lines) == 2
    assert len(runner_lines) == 4
    assert revalidation_lines[0] < runner_lines[0] < runner_lines[1]
    assert runner_lines[1] < revalidation_lines[1] < runner_lines[2] < runner_lines[3]


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
    clean_manifest = tmp_path / "commonvoice_en_test.jsonl"
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


_ALIGNMENT_LAYER_IDS = tuple(range(70))
_ALIGNMENT_CELLS = (
    "easy_en",
    "easy_zh",
    "medium_en",
    "medium_zh",
    "hard_en",
    "hard_zh",
    "long_zh",
    "supplemental_en",
    "supplemental_zh",
)
_ALIGNMENT_PHASE_COMPONENTS = {
    "mixer": ("mixer",),
    "block": ("mixer", "ffn", "block"),
    "logits": ("mixer", "ffn", "block"),
}


def _alignment_layers(*, candidate: bool) -> dict[str, dict[str, float]]:
    return {
        str(layer_id): {
            "loss": 0.5 if candidate else 1.0,
            "cosine": 0.8 if candidate else 0.5,
            "rms_ratio": 1.0,
        }
        for layer_id in _ALIGNMENT_LAYER_IDS
    }


def _alignment_fixed_component_summary() -> dict[str, object]:
    weak_row = {
        "baseline_loss": 1.0,
        "candidate_loss": 0.5,
        "relative_loss_reduction": 0.5,
        "baseline_cosine": 0.5,
        "candidate_cosine": 0.8,
    }
    return {
        "baseline_mean_loss": 1.0,
        "candidate_mean_loss": 0.5,
        "relative_loss_reduction": 0.5,
        "baseline_mean_cosine": 0.5,
        "candidate_mean_cosine": 0.8,
        "baseline_mean_rms_ratio": 1.0,
        "candidate_mean_rms_ratio": 1.0,
        "loss_improved_layers": 70,
        "cosine_improved_layers": 70,
        "weak_bands": {
            "10-19": dict(weak_row),
            "20-29": dict(weak_row),
        },
    }


def _alignment_stratified_component_summary() -> dict[str, object]:
    weak_row = {
        "baseline_loss": 1.0,
        "candidate_loss": 0.5,
        "relative_loss_reduction": 0.5,
        "baseline_cosine": 0.5,
        "candidate_cosine": 0.8,
    }
    layer_row = {
        "baseline_loss": 1.0,
        "candidate_loss": 0.5,
        "relative_loss_reduction": 0.5,
        "baseline_cosine": 0.5,
        "candidate_cosine": 0.8,
        "baseline_rms_ratio": 1.0,
        "candidate_rms_ratio": 1.0,
    }
    cell_row = {
        "baseline_loss": 1.0,
        "candidate_loss": 0.5,
        "relative_change_pct": -50.0,
        "baseline_cosine": 0.5,
        "candidate_cosine": 0.8,
        "layers_loss_improved": 70,
        "layers_cosine_improved": 70,
    }
    return {
        "baseline_mean_loss": 1.0,
        "candidate_mean_loss": 0.5,
        "baseline_mean_cosine": 0.5,
        "candidate_mean_cosine": 0.8,
        "loss_improved_layers": 70,
        "cosine_improved_layers": 70,
        "layers": {str(layer_id): dict(layer_row) for layer_id in _ALIGNMENT_LAYER_IDS},
        "weak_bands": {
            "10-19": dict(weak_row),
            "20-29": dict(weak_row),
        },
        "cells": {cell_name: dict(cell_row) for cell_name in _ALIGNMENT_CELLS},
    }


def _alignment_logits_metrics(
    *,
    candidate: bool,
    matched_utterances: int,
) -> dict[str, float]:
    return {
        "full_kl": 0.8 if candidate else 1.0,
        "conditional_nonblank_kl": 0.8 if candidate else 1.0,
        "conditional_nonblank_hard_ce": 0.8 if candidate else 1.0,
        "blank_binary_kl": 0.8 if candidate else 1.0,
        "selected_top1_agreement": 0.7 if candidate else 0.5,
        "all_top1_agreement": 0.7 if candidate else 0.5,
        "active_top1_agreement": 0.7 if candidate else 0.5,
        "blank_prob_mae": 0.1 if candidate else 0.2,
        "teacher_nonblank_rate": 0.2,
        "student_nonblank_rate": 0.2 if candidate else 0.16,
        "nonblank_rate_ratio": 1.0 if candidate else 0.8,
        "teacher_nonblank_to_blank_rate": 0.1 if candidate else 0.2,
        "teacher_blank_to_nonblank_rate": 0.1 if candidate else 0.2,
        "ctc_token_error_rate": 0.2 if candidate else 0.4,
        "ctc_token_insertion_rate": 0.05,
        "ctc_token_deletion_rate": 0.1 if candidate else 0.2,
        "ctc_token_substitution_rate": 0.05 if candidate else 0.15,
        "collapsed_length_ratio": 1.0 if candidate else 0.8,
        "sequence_exact_rate": 0.5 if candidate else 0.4,
        "mean_frame_delta": 0.0,
        "selected_frames": float(matched_utterances * 10),
        "all_frames": float(matched_utterances * 20),
        "teacher_tokens": float(matched_utterances * 2),
        "student_tokens": float(matched_utterances * 2),
        "matched_utterances": float(matched_utterances),
        "missing_utterances": 0.0,
    }


def _alignment_logits_checks() -> dict[str, bool]:
    return {
        "full_kl_materially_improved": True,
        "conditional_nonblank_kl_materially_improved": True,
        "conditional_nonblank_hard_ce_not_worse": True,
        "blank_binary_kl_not_worse": True,
        "blank_probability_mae_not_worse": True,
        "selected_top1_improved": True,
        "all_top1_improved": True,
        "active_top1_improved": True,
        "ctc_token_error_rate_improved": True,
        "ctc_token_deletion_rate_not_worse": True,
        "sequence_exact_rate_not_worse": True,
        "nonblank_rate_ratio_in_range": True,
        "nonblank_rate_ratio_not_farther": True,
        "collapsed_length_ratio_in_range": True,
        "collapsed_length_ratio_not_farther": True,
        "complete_exact_coverage": True,
    }


def _write_alignment_evidence_fixture(
    tmp_path: Path,
    *,
    phase: str,
    baseline_checkpoint: Path,
    checkpoint: Path,
    train_config: Path,
    nano_checkpoint: Path,
) -> Path:
    components = _ALIGNMENT_PHASE_COMPONENTS[phase]
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
    shared_source = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "alignment_checkpoint_eval",
        "phase": phase,
        "pair_eval_id": "a" * 64,
        "train_config_path": str(train_config.resolve()),
        "train_config_sha256": sha256_file(train_config),
        "model_config_path": str(alignment_model_config.resolve()),
        "model_config_sha256": sha256_file(alignment_model_config),
        "nano_checkpoint_path": str(nano_checkpoint.resolve()),
        "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
        "feature_seed": 0,
        "eval_samples": 256,
        "eval_provenance": alignment_provenance,
    }
    source_paths: dict[str, Path] = {}
    for role, source_checkpoint, candidate in (
        ("baseline", baseline_checkpoint, False),
        ("candidate", checkpoint, True),
    ):
        source = {
            **shared_source,
            "role": role,
            "step": 105 if candidate else 0,
            "checkpoint_step": 105 if candidate else 0,
            "checkpoint_path": str(source_checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(source_checkpoint),
            "eval_loss": 0.5 if candidate else 1.0,
            "layer_components": {
                component: _alignment_layers(candidate=candidate) for component in components
            },
        }
        if phase in {"block", "logits"}:
            source["decoder_hidden_metrics"] = {"loss": 0.5 if candidate else 1.0}
        if phase == "logits":
            source["logit_metrics"] = _alignment_logits_metrics(
                candidate=candidate,
                matched_utterances=256,
            )
        source_path = tmp_path / f"{phase}-alignment-{role}.json"
        source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
        source_paths[role] = source_path

    receipt = tmp_path / f"{phase}-stratified-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    report_bindings: dict[str, dict[str, dict[str, str]]] = {}
    cell_manifests: dict[str, Path] = {}
    for cell_name in _ALIGNMENT_CELLS:
        manifest = tmp_path / f"{phase}-{cell_name}-manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        cell_manifests[cell_name] = manifest
        report_bindings[cell_name] = {}
        for role in ("baseline", "candidate"):
            report_path = tmp_path / f"{phase}-{cell_name}-{role}.json"
            candidate = role == "candidate"
            report_path.write_text(
                json.dumps(
                    {
                        "phase": phase,
                        "role": role,
                        "layer_components": {
                            component: _alignment_layers(candidate=candidate)
                            for component in components
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report_bindings[cell_name][role] = {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            }
    checkpoint_records = {
        "baseline": {
            "path": str(baseline_checkpoint.resolve()),
            "sha256": sha256_file(baseline_checkpoint),
        },
        "candidate": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
        },
    }
    stratified_components = {
        component: _alignment_stratified_component_summary() for component in components
    }
    if phase == "logits":
        cell_results = {
            cell_name: {
                "samples": 256,
                "manifest_path": str(cell_manifests[cell_name].resolve()),
                "manifest_sha256": sha256_file(cell_manifests[cell_name]),
                "baseline_metrics": _alignment_logits_metrics(
                    candidate=False,
                    matched_utterances=256,
                ),
                "candidate_metrics": _alignment_logits_metrics(
                    candidate=True,
                    matched_utterances=256,
                ),
            }
            for cell_name in _ALIGNMENT_CELLS
        }
        decoder_cells = {
            cell_name: {
                "baseline_loss": 1.0,
                "candidate_loss": 0.5,
                "relative_change_pct": -50.0,
            }
            for cell_name in _ALIGNMENT_CELLS
        }
        stratified_summary = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "stratified_logits_eval_summary",
            "phase": "logits",
            "receipt_path": str(receipt.resolve()),
            "receipt_sha256": sha256_file(receipt),
            "checkpoints": checkpoint_records,
            "cells": cell_results,
            "macro": {
                "cells": len(_ALIGNMENT_CELLS),
                "samples": 256 * len(_ALIGNMENT_CELLS),
                "baseline_metrics": _alignment_logits_metrics(
                    candidate=False,
                    matched_utterances=256 * len(_ALIGNMENT_CELLS),
                ),
                "candidate_metrics": _alignment_logits_metrics(
                    candidate=True,
                    matched_utterances=256 * len(_ALIGNMENT_CELLS),
                ),
                "full_kl_relative_reduction": 0.2,
                "conditional_nonblank_kl_relative_reduction": 0.2,
            },
            "hidden_component_summaries": stratified_components,
            "decoder_hidden": {
                "cells": len(_ALIGNMENT_CELLS),
                "baseline_loss": 1.0,
                "candidate_loss": 0.5,
                "relative_change_pct": -50.0,
                "cell_results": decoder_cells,
            },
            "reports": report_bindings,
        }
    else:
        primary = stratified_components[phase]
        cell_results = {
            cell_name: {
                "samples": 256,
                "manifest_path": str(cell_manifests[cell_name].resolve()),
                "manifest_sha256": sha256_file(cell_manifests[cell_name]),
                "baseline_loss": 1.0,
                "candidate_loss": 0.5,
                "relative_change_pct": -50.0,
                "layers_loss_improved": 70,
                "layers_cosine_improved": 70,
            }
            for cell_name in _ALIGNMENT_CELLS
        }
        stratified_summary = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "stratified_hidden_eval_summary",
            "phase": phase,
            "receipt_path": str(receipt.resolve()),
            "receipt_sha256": sha256_file(receipt),
            "checkpoints": checkpoint_records,
            "cells": cell_results,
            "macro": {
                "cells": len(_ALIGNMENT_CELLS),
                "samples": 256 * len(_ALIGNMENT_CELLS),
                "baseline_loss": 1.0,
                "candidate_loss": 0.5,
                "relative_change_pct": -50.0,
            },
            "layer_summary": {
                key: primary[key]
                for key in (
                    "loss_improved_layers",
                    "cosine_improved_layers",
                    "weak_bands",
                    "layers",
                )
            },
            "component_summaries": stratified_components,
            "decoder_hidden": (
                {
                    "cells": len(_ALIGNMENT_CELLS),
                    "baseline_loss": 1.0,
                    "candidate_loss": 0.5,
                }
                if phase == "block"
                else None
            ),
            "reports": report_bindings,
        }
    stratified_path = tmp_path / f"{phase}-stratified-summary.json"
    stratified_path.write_text(
        json.dumps(stratified_summary) + "\n",
        encoding="utf-8",
    )

    fixed_components = {component: _alignment_fixed_component_summary() for component in components}
    alignment = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": ("logits_alignment_gate" if phase == "logits" else "hidden_alignment_gate"),
        "phase": phase,
        "baseline_checkpoint_path": str(baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "gate_passed": True,
        "legacy_gate_passed": True,
        "stratified_gate_passed": True,
        "stratified_summary_path": str(stratified_path.resolve()),
        "stratified_summary_sha256": sha256_file(stratified_path),
        "stratified_summary": stratified_summary,
        "baseline_report_path": str(source_paths["baseline"].resolve()),
        "baseline_report_sha256": sha256_file(source_paths["baseline"]),
        "candidate_report_path": str(source_paths["candidate"].resolve()),
        "candidate_report_sha256": sha256_file(source_paths["candidate"]),
        "baseline_eval_provenance": alignment_provenance,
        "candidate_eval_provenance": alignment_provenance,
    }
    if phase == "logits":
        checks = _alignment_logits_checks()
        retention_checks = {
            component: {
                "mean_loss_retained": True,
                "mean_cosine_retained": True,
                "weak_band_loss_retained": True,
                "weak_band_cosine_retained": True,
            }
            for component in components
        }
        alignment.update(
            {
                "selected_logits_gate_passed": True,
                "stratified_checks": {
                    **checks,
                    "hard_and_long_cells_improved": True,
                    "cell_token_error_regression_bounded": True,
                    "mixer_hidden_retained": True,
                    "ffn_hidden_retained": True,
                    "block_hidden_retained": True,
                    "decoder_hidden_retained": True,
                    "complete_stratified_coverage": True,
                },
                "thresholds": {
                    "minimum_kl_relative_reduction": 0.05,
                    "ratio_min": 0.9,
                    "ratio_max": 1.1,
                    "max_cell_token_error_regression": 0.03,
                    "max_hidden_loss_regression": 0.1,
                    "max_hidden_cosine_regression": 0.01,
                    "tolerance": 1.0e-12,
                },
                "checks": checks,
                "hidden_component_summaries": fixed_components,
                "hidden_retention_checks": retention_checks,
                "hidden_retention_passed": True,
                "decoder_hidden_retention": {
                    "baseline_loss": 1.0,
                    "candidate_loss": 0.5,
                    "relative_change": -0.5,
                    "retained": True,
                },
                "baseline_metrics": _alignment_logits_metrics(
                    candidate=False,
                    matched_utterances=256,
                ),
                "candidate_metrics": _alignment_logits_metrics(
                    candidate=True,
                    matched_utterances=256,
                ),
                "full_kl_relative_reduction": 0.2,
                "conditional_nonblank_kl_relative_reduction": 0.2,
            }
        )
    else:
        alignment.update(
            {
                "baseline_eval_loss": 1.0,
                "candidate_eval_loss": 0.5,
                "layer_summary": fixed_components[phase],
                "component_summaries": fixed_components,
                "component_gate_passed": True,
                "decoder_hidden": (
                    {"baseline_loss": 1.0, "candidate_loss": 0.5} if phase == "block" else None
                ),
            }
        )
    alignment_path = tmp_path / f"{phase}-alignment.json"
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    return alignment_path


def _write_public_comparison_evidence_fixture(
    tmp_path: Path,
    *,
    checkpoint: Path,
    baseline_checkpoint: Path,
    clean_commonvoice_manifest: Path,
) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    candidate_provenance, candidate_tokenizer = stage211_test_student_ctc_context(checkpoint)
    baseline_provenance, baseline_tokenizer = stage211_test_student_ctc_context(baseline_checkpoint)
    candidate_results: list[dict[str, object]] = []
    baseline_results: list[dict[str, object]] = []
    manifest_paths: dict[str, Path] = {}
    candidate_prediction_paths: dict[str, Path] = {}
    baseline_prediction_paths: dict[str, Path] = {}
    public_labels = {
        "aishell1_test": "AISHELL-1 test",
        "librispeech_test_clean": "LibriSpeech test-clean",
        "librispeech_test_other": "LibriSpeech test-other",
        "commonvoice_en_test": "Common Voice 22 en test",
        "wenetspeech_test_net": "WenetSpeech TEST_NET",
    }
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        manifest_path = (
            clean_commonvoice_manifest
            if dataset == "commonvoice_en_test"
            else tmp_path / f"{dataset}.jsonl"
        )
        nano_path = tmp_path / f"{dataset}.nano.jsonl"
        candidate_path = tmp_path / f"{dataset}.candidate.jsonl"
        baseline_path = tmp_path / f"{dataset}.baseline.jsonl"
        language = str(expected["language"])
        sample_count = int(expected["samples"])
        if dataset == "commonvoice_en_test":
            reference = "clean"
            baseline_prediction = ""
            utt_ids = (f"clean-{index:05d}" for index in range(sample_count))
        else:
            reference = "alpha beta" if language == "en" else "你好"
            baseline_prediction = "alpha" if language == "en" else "你"
            utt_ids = (f"{dataset}-{index:05d}" for index in range(sample_count))

        manifest_rows: list[str] = []
        nano_rows: list[str] = []
        candidate_rows: list[str] = []
        baseline_rows: list[str] = []
        for row_index, utt_id in enumerate(utt_ids):
            if dataset != "commonvoice_en_test":
                manifest_rows.append(
                    json.dumps({"utt_id": utt_id, "text": reference}, ensure_ascii=True) + "\n"
                )
            nano_rows.append(
                json.dumps(
                    {
                        "utt_id": utt_id,
                        "ref_text": reference,
                        "pred_text": baseline_prediction if row_index == 0 else reference,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            candidate_rows.append(
                json.dumps(
                    stage211_test_student_ctc_row(
                        utt_id=utt_id,
                        ref_text=reference,
                        pred_text=(baseline_prediction if row_index == 0 else reference),
                        provenance=candidate_provenance,
                        tokenizer=candidate_tokenizer,
                    ),
                    ensure_ascii=True,
                )
                + "\n"
            )
            baseline_rows.append(
                json.dumps(
                    stage211_test_student_ctc_row(
                        utt_id=utt_id,
                        ref_text=reference,
                        pred_text=baseline_prediction,
                        provenance=baseline_provenance,
                        tokenizer=baseline_tokenizer,
                    ),
                    ensure_ascii=True,
                )
                + "\n"
            )
        if manifest_rows:
            manifest_path.write_text("".join(manifest_rows), encoding="utf-8")
        nano_path.write_text("".join(nano_rows), encoding="utf-8")
        candidate_path.write_text("".join(candidate_rows), encoding="utf-8")
        baseline_path.write_text("".join(baseline_rows), encoding="utf-8")
        manifest_paths[dataset] = manifest_path.resolve()
        candidate_prediction_paths[dataset] = candidate_path.resolve()
        baseline_prediction_paths[dataset] = baseline_path.resolve()

        def fixture_result(*, student_path: Path, baseline: bool) -> dict[str, object]:
            nano_wer = (
                1.0 / sample_count
                if dataset == "commonvoice_en_test"
                else 1.0 / (2.0 * sample_count)
            )
            nano_cer = (
                1.0 / sample_count
                if dataset == "commonvoice_en_test"
                else 4.0 / (9.0 * sample_count)
                if language == "en"
                else 1.0 / (2.0 * sample_count)
            )
            nano_error_rate = nano_wer if expected["metric"] == "wer" else nano_cer
            baseline_wer = 1.0 if dataset == "commonvoice_en_test" else 0.5
            baseline_cer = (
                1.0 if dataset == "commonvoice_en_test" else 4.0 / 9.0 if language == "en" else 0.5
            )
            student_wer = baseline_wer if baseline else nano_wer
            student_cer = baseline_cer if baseline else nano_cer
            student_error_rate = student_wer if expected["metric"] == "wer" else student_cer
            nano_unit_ratio = (
                (sample_count - 1.0) / sample_count
                if dataset == "commonvoice_en_test"
                else (2.0 * sample_count - 1.0) / (2.0 * sample_count)
            )
            return {
                "dataset": dataset,
                "label": public_labels[dataset],
                "language": language,
                "metric": expected["metric"],
                "sample_count": sample_count,
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "nano_wer": nano_wer,
                "student_wer": student_wer,
                "nano_cer": nano_cer,
                "student_cer": student_cer,
                "nano_error_rate": nano_error_rate,
                "student_error_rate": student_error_rate,
                "absolute_gap_points": (student_error_rate - nano_error_rate) * 100.0,
                "relative_ratio": student_error_rate / nano_error_rate,
                "absolute_gate_pass": not baseline,
                "relative_gate_pass": not baseline,
                "gate_pass": not baseline,
                "nano_prediction_reference_unit_ratio": nano_unit_ratio,
                "student_prediction_reference_unit_ratio": (
                    0.0
                    if baseline and dataset == "commonvoice_en_test"
                    else 0.5
                    if baseline
                    else nano_unit_ratio
                ),
                "nano_insertion_rate": 0.0,
                "student_insertion_rate": 0.0,
                "nano_deletion_rate": nano_error_rate,
                "student_deletion_rate": student_error_rate,
                "nano_substitution_rate": 0.0,
                "student_substitution_rate": 0.0,
                "changed_prediction_count": sample_count - 1 if baseline else 0,
                "student_improved_count": 0,
                "student_worsened_count": sample_count - 1 if baseline else 0,
                "unchanged_count": 1 if baseline else sample_count,
                "nano_prediction_path": str(nano_path.resolve()),
                "student_prediction_path": str(student_path.resolve()),
            }

        candidate_results.append(fixture_result(student_path=candidate_path, baseline=False))
        baseline_results.append(fixture_result(student_path=baseline_path, baseline=True))

    def write_report(
        *,
        path: Path,
        student_checkpoint: Path,
        results: list[dict[str, object]],
    ) -> dict[str, object]:
        report: dict[str, object] = {
            "version": 1,
            "systems": {
                "baseline": "FunASR-Nano-2512 direct CTC",
                "candidate": "BiRWKV CTC",
            },
            "decode": "greedy_ctc",
            "normalization": "ctc",
            "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
            "gate": {
                "max_relative_ratio": 1.20,
                "max_absolute_gap_points": 3.0,
                "requires_every_dataset": True,
            },
            "all_datasets_pass": all(bool(result["gate_pass"]) for result in results),
            "student_checkpoint_path": str(student_checkpoint.resolve()),
            "student_checkpoint_sha256": sha256_file(student_checkpoint),
            "results": results,
        }
        path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return report

    candidate_report_path = tmp_path / "candidate-public-comparison.json"
    candidate_report = write_report(
        path=candidate_report_path,
        student_checkpoint=checkpoint,
        results=candidate_results,
    )
    candidate_receipt_path = tmp_path / "candidate-public-prediction-receipt.json"
    candidate_receipt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=checkpoint,
        manifest_paths=manifest_paths,
        prediction_paths=candidate_prediction_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    candidate_receipt_path.write_text(
        json.dumps(candidate_receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_report["student_prediction_receipt_path"] = str(candidate_receipt_path.resolve())
    candidate_report["student_prediction_receipt_sha256"] = sha256_file(candidate_receipt_path)
    candidate_report_path.write_text(
        json.dumps(candidate_report) + "\n",
        encoding="utf-8",
    )
    baseline_report_path = tmp_path / "baseline-public-comparison.json"
    baseline_report = write_report(
        path=baseline_report_path,
        student_checkpoint=baseline_checkpoint,
        results=baseline_results,
    )
    baseline_receipt_path = tmp_path / "baseline-public-prediction-receipt.json"
    baseline_receipt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=baseline_checkpoint,
        manifest_paths=manifest_paths,
        prediction_paths=baseline_prediction_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    baseline_receipt_path.write_text(
        json.dumps(baseline_receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    baseline_report["student_prediction_receipt_path"] = str(baseline_receipt_path.resolve())
    baseline_report["student_prediction_receipt_sha256"] = sha256_file(baseline_receipt_path)
    baseline_report_path.write_text(
        json.dumps(baseline_report) + "\n",
        encoding="utf-8",
    )

    def enrich_report(report: dict[str, object]) -> dict[str, object]:
        enriched_results = []
        for result in report["results"]:
            assert isinstance(result, dict)
            dataset = str(result["dataset"])
            manifest_path = manifest_paths[dataset]
            nano_path = Path(str(result["nano_prediction_path"]))
            student_path = Path(str(result["student_prediction_path"]))
            enriched_results.append(
                {
                    **result,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "nano_prediction_path": str(nano_path),
                    "nano_prediction_sha256": sha256_file(nano_path),
                    "student_prediction_path": str(student_path),
                    "student_prediction_sha256": sha256_file(student_path),
                    "metric_source_recomputed": True,
                }
            )
        return {
            **report,
            "all_datasets_complete": True,
            "results": enriched_results,
        }

    candidate_benchmark = enrich_report(candidate_report)
    baseline_benchmark = enrich_report(baseline_report)
    progress = stage211_phase_gate._build_public_progress(
        baseline=baseline_benchmark,
        candidate=candidate_benchmark,
    )
    assert candidate_benchmark["all_datasets_pass"] is True
    assert progress["gate_passed"] is True
    return (
        candidate_report_path,
        candidate_benchmark,
        baseline_report_path,
        progress,
    )


def _write_trajectory_eval_fixture(
    *,
    phase: str,
    record: dict[str, object],
    eval_loss: float,
    eval_parts: list[dict[str, object]] | None = None,
) -> Path:
    run_dir = Path(str(record["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(record["bucket_manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_split = manifest["splits"]["eval"]
    parts = list(eval_parts or [])
    if not parts:
        for bucket in eval_split["buckets"]:
            for raw_part in bucket["parts"]:
                part_path = Path(str(raw_part["path"]))
                if not part_path.is_absolute():
                    manifest_relative = manifest_path.parent / part_path
                    part_path = (
                        manifest_relative
                        if manifest_relative.is_file()
                        else Path(str(manifest.get("root") or manifest_path.parent)) / part_path
                    )
                part_path = part_path.resolve()
                parts.append(
                    {
                        "path": str(part_path),
                        "sha256": sha256_file(part_path),
                        "num_samples": int(raw_part["num_samples"]),
                    }
                )
    components = _ALIGNMENT_PHASE_COMPONENTS[phase]
    report: dict[str, object] = {
        "step": int(record["steps"]),
        "eval_loss": eval_loss,
        "eval_samples": 256,
        "eval_provenance": {
            "schema_version": 1,
            "split": "eval",
            "requested_samples": 256,
            "feature_seed": 0,
            "bucket_manifest_path": str(manifest_path.resolve()),
            "bucket_manifest_sha256": sha256_file(manifest_path),
            "split_samples": 256,
            "parts": parts,
        },
        "layers": _alignment_layers(candidate=True),
        "layer_components": {
            component: _alignment_layers(candidate=True) for component in components
        },
        "decoder_hidden_metrics": {},
        "logit_metrics": {},
    }
    if phase in {"block", "logits"}:
        report["decoder_hidden_metrics"] = {"loss": eval_loss}
    if phase == "logits":
        report["logit_metrics"] = _alignment_logits_metrics(
            candidate=True,
            matched_utterances=256,
        )
    output = run_dir / f"step_eval_layers_step-{int(record['steps'])}.yaml"
    save_yaml(output, report)
    return output


def _write_step_eval_cadence_source_fixture(record: dict[str, object]) -> None:
    run_dir = Path(str(record["run_dir"]))
    terminal_step = int(record["steps"])
    steps_per_epoch = int(record["steps_per_epoch"])
    epochs = int(record["epochs"])
    expected_steps = list(range(10_000, terminal_step + 1, 10_000))
    if not expected_steps or expected_steps[-1] != terminal_step:
        expected_steps.append(terminal_step)
    terminal_report_path = run_dir / f"step_eval_layers_step-{terminal_step}.yaml"
    terminal_report = load_yaml(terminal_report_path)
    records = []
    for step in expected_steps:
        report_path = run_dir / f"step_eval_layers_step-{step}.yaml"
        if report_path.is_file():
            report = load_yaml(report_path)
        else:
            report = {
                "step": step,
                "eval_loss": float(terminal_report["eval_loss"]) + 1.0 / (step + 1),
                "eval_samples": 256,
                "eval_provenance": terminal_report["eval_provenance"],
            }
            save_yaml(report_path, report)
        records.append(
            {
                "step": step,
                "epoch": min(epochs, ((step - 1) // steps_per_epoch) + 1),
                "eval_loss": float(report["eval_loss"]),
                "eval_samples": 256,
                "checkpoint_path": str((run_dir / f"step-{step}.pt").resolve()),
                "deepspeed_checkpoint_dir": str(
                    (run_dir / "ds_checkpoints" / f"step-{step}").resolve()
                ),
                "resume_tag": f"step-{step}",
                "layer_metrics_path": str(report_path.resolve()),
            }
        )
    save_yaml(
        run_dir / "step_checkpoint_metrics.yaml",
        {
            "step_checkpoints": records,
            "best": records[: min(8, len(records))],
            "keep_top_k": min(8, len(records)),
        },
    )


def _write_step_eval_cadence_fixture(
    *,
    phase: str,
    segments: list[dict[str, object]],
    supplemental_segment: dict[str, object],
    corrections: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    records = [*segments, supplemental_segment, *(corrections or [])]
    for record in records:
        _write_step_eval_cadence_source_fixture(record)
    return build_stage211_step_eval_cadence(
        phase=phase,
        segments=segments,
        supplemental_segment=supplemental_segment,
        post_coverage_corrections=corrections or [],
    )


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
        run_dir = tmp_path / f"{phase}-{difficulty}-run"
        run_dir.mkdir()
        manifest = runtime_manifests[difficulty]
        provenance = run_dir / "stage211_provenance.json"
        provenance.write_text("{}\n", encoding="utf-8")
        train_config = run_dir / "train_config.yaml"
        train_config_payload = stage211_phase_train_config_contract(phase)
        train_config_payload.update(
            {
                "ctc_teacher_online_model_path": str(nano_teacher_dir.resolve()),
                "max_steps": int(expected["steps"]),
                "step_eval_every": 10_000,
                "step_eval_samples": 256,
                "step_eval_split": "eval",
                "step_eval_shuffle": False,
                "step_eval_feature_seed": 0,
            }
        )
        save_yaml(train_config, train_config_payload)
        completion = run_dir / f"step-{int(expected['steps'])}.pt"
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
            "run_dir": str(run_dir.resolve()),
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
        _write_trajectory_eval_fixture(
            phase=phase,
            record=segment,
            eval_loss=1.0 - index * 0.1,
        )
        segments.append(segment)
        previous_checkpoint = completion

    supplemental_segment, _ = _write_supplemental_coverage_fixture(
        tmp_path,
        phase=phase,
        init_checkpoint=previous_checkpoint,
        completion_checkpoint=checkpoint,
        nano_teacher_dir=nano_teacher_dir,
    )
    _write_trajectory_eval_fixture(
        phase=phase,
        record=supplemental_segment,
        eval_loss=0.5,
        eval_parts=list(
            load_yaml(
                Path(str(segments[0]["run_dir"]))
                / f"step_eval_layers_step-{int(segments[0]['steps'])}.yaml"
            )["eval_provenance"]["parts"]
        ),
    )
    step_eval_cadence = _write_step_eval_cadence_fixture(
        phase=phase,
        segments=segments,
        supplemental_segment=supplemental_segment,
    )
    full_data_coverage = build_stage211_full_data_coverage(
        phase=phase,
        segments=segments,
        supplemental_segment=supplemental_segment,
        checkpoint_path=checkpoint,
    )
    trajectory_retention = build_stage211_trajectory_retention_gate(
        phase=phase,
        segments=segments,
        supplemental_segment=supplemental_segment,
        post_coverage_corrections=[],
        checkpoint_path=checkpoint,
    )

    phase_init_checkpoint = Path(segments[0]["init_checkpoint_path"])
    (
        public_comparison_report,
        public_benchmark,
        baseline_public_comparison_report,
        public_progress,
    ) = _write_public_comparison_evidence_fixture(
        tmp_path,
        checkpoint=checkpoint,
        baseline_checkpoint=phase_init_checkpoint,
        clean_commonvoice_manifest=clean_commonvoice_manifest,
    )
    baseline_public_source = json.loads(
        baseline_public_comparison_report.read_text(encoding="utf-8")
    )
    public_results = public_benchmark["results"]

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
                    "strip_language_confirmation": (STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION),
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
    alignment_train_config = Path(segments[0]["train_config_path"])
    alignment_report = _write_alignment_evidence_fixture(
        tmp_path,
        phase=phase,
        baseline_checkpoint=phase_init_checkpoint,
        checkpoint=checkpoint,
        train_config=alignment_train_config,
        nano_checkpoint=nano_teacher_checkpoint,
    )

    smoke_dir = tmp_path / f"{phase}-full-profile-smoke"
    smoke_log_dir = smoke_dir / "logs"
    smoke_log_dir.mkdir(parents=True)
    smoke_checkpoint = smoke_dir / "step-2.pt"
    smoke_log = smoke_log_dir / f"{phase}_smoke_2steps.log"
    torch.save({"step": 2}, smoke_checkpoint)
    smoke_config = None
    smoke_runtime_fields = ""
    if phase in {"block", "logits"}:
        smoke_config = _write_stacked_smoke_config(
            smoke_dir / "train_config.yaml",
            phase=phase,
        )
        smoke_runtime_fields = f" {_stage211_smoke_runtime_fields(phase)}"
    smoke_log.write_text(
        "[rwkvasr] Distributed init complete.\n"
        "[deepspeed-train] step=1 loss=0.8 peak_reserved=5.50GiB\n"
        f"[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB{smoke_runtime_fields}\n",
        encoding="utf-8",
    )
    easy_manifest = Path(str(segments[0]["bucket_manifest_path"])).resolve()
    smoke_marker = tmp_path / f"{phase}-full-profile-smoke-passed.json"
    smoke_marker_payload = {
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
    if smoke_config is not None:
        runtime_objective_evidence = stage211_full_phase._smoke_runtime_objective_evidence(
            phase=phase,
            step_two_line=(
                f"[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB{smoke_runtime_fields}"
            ),
        )
        smoke_marker_payload.update(
            {
                "smoke_config_path": str(smoke_config),
                "smoke_config_sha256": sha256_file(smoke_config),
                "runtime_objective_evidence": runtime_objective_evidence,
                "smoke_runtime_profile": {
                    "batch_size": STAGE211_STACKED_SAFE_BATCH_SIZE,
                    "frame_budget": STAGE211_STACKED_SAFE_FRAME_BUDGET,
                    "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
                    "train_batch_size": (
                        STAGE211_STACKED_SAFE_BATCH_SIZE * STAGE211_FULL_DATA_WORLD_SIZE
                    ),
                },
            }
        )
    smoke_marker.write_text(
        json.dumps(smoke_marker_payload) + "\n",
        encoding="utf-8",
    )

    gate_report = tmp_path / "mixer_gate.json"
    gate_report.write_text(
        json.dumps(
            {
                "schema_version": STAGE211_PHASE_GATE_SCHEMA_VERSION,
                "pipeline": "stage211",
                "artifact": "phase_gate",
                "phase": phase,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "gate_passed": True,
                "metric_gate_passed": True,
                "correction_round_promotion": (build_stage211_correction_round_promotion_gate(0)),
                "alignment_gate_passed": True,
                "public_progress_gate_passed": True,
                "trajectory_retention_gate_passed": True,
                "trajectory_retention": trajectory_retention,
                "step_eval_cadence": step_eval_cadence,
                "public_comparison_report_path": str(public_comparison_report.resolve()),
                "public_comparison_report_sha256": sha256_file(public_comparison_report),
                "baseline_public_comparison_report": {
                    "path": str(baseline_public_comparison_report.resolve()),
                    "sha256": sha256_file(baseline_public_comparison_report),
                },
                "baseline_public_provenance": {
                    "mode": "student_prediction_receipt",
                    "student_prediction_receipt_path": baseline_public_source[
                        "student_prediction_receipt_path"
                    ],
                    "student_prediction_receipt_sha256": baseline_public_source[
                        "student_prediction_receipt_sha256"
                    ],
                },
                "public_progress": public_progress,
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
                "public_benchmark": public_benchmark,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return gate_report


@pytest.mark.parametrize("phase", ("mixer", "block", "logits"))
def test_stage211_stepwise_alignment_disclosure_accepts_deep_phase_evidence(
    tmp_path: Path,
    phase: str,
) -> None:
    checkpoint = tmp_path / "step-10.pt"
    checkpoint.write_bytes(f"{phase}-checkpoint".encode())
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase=phase,
        checkpoint=checkpoint,
    )
    gate = validate_stage211_phase_gate_report(
        gate_path,
        expected_phase=phase,
        checkpoint_path=checkpoint,
    )

    disclosure = stage211_stepwise_report._alignment_result(
        phase=phase,
        phase_report=gate,
    )

    assert disclosure["gate_passed"] is True
    assert disclosure["fixed_eval_samples"] == 256
    assert disclosure["stratified_samples"] == 2_304
    assert disclosure["stratified_cells"] == list(stage211_stepwise_report.ALIGNMENT_CELLS)
    assert disclosure["trajectory_retention"] == {
        "gate_passed": True,
        "fixed_eval_samples": 256,
        "source_order": [
            "easy",
            "medium",
            "hard",
            "long",
            STAGE211_SUPPLEMENTAL_DIFFICULTY,
        ],
        "terminal_entries": 5,
        "best_prior_source": "long",
        "best_prior_loss": pytest.approx(0.7),
        "candidate_source": STAGE211_SUPPLEMENTAL_DIFFICULTY,
        "candidate_loss": pytest.approx(0.5),
        "relative_regression_pct": pytest.approx(-28.5714285714),
        "max_relative_regression_pct": 10.0,
    }
    if phase == "logits":
        assert disclosure["fixed"]["metrics"]["full_kl"]["candidate"] == pytest.approx(0.8)
        assert set(disclosure["fixed"]["hidden_components"]) == {"mixer", "ffn", "block"}
    else:
        expected_components = {"mixer"} if phase == "mixer" else {"mixer", "ffn", "block"}
        assert set(disclosure["fixed"]["components"]) == expected_components


def test_stage211_stepwise_coverage_discloses_correction_promotion_gate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = validate_stage211_phase_gate_report(
        gate_path,
        expected_phase="mixer",
        checkpoint_path=checkpoint,
    )

    coverage = stage211_stepwise_report._coverage_record(
        stage="mixer",
        coverage=gate["full_data_coverage"],
        correction_round_promotion=gate["correction_round_promotion"],
    )

    assert coverage["correction_rounds"] == 0
    assert coverage["correction_round_promotion"] == (
        build_stage211_correction_round_promotion_gate(0)
    )
    tampered = dict(gate["correction_round_promotion"])
    tampered["completed_rounds"] = 3
    with pytest.raises(ValueError, match="final correction-round promotion evidence"):
        stage211_stepwise_report._coverage_record(
            stage="mixer",
            coverage=gate["full_data_coverage"],
            correction_round_promotion=tampered,
        )


def test_stage211_phase_gate_fails_trajectory_regression_from_best_prior(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    coverage = gate["full_data_coverage"]
    candidate_report_path = Path(gate["trajectory_retention"]["candidate"]["eval_report_path"])
    candidate_report = load_yaml(candidate_report_path)
    candidate_report["eval_loss"] = 0.8
    save_yaml(candidate_report_path, candidate_report)
    candidate_metrics_path = candidate_report_path.parent / "step_checkpoint_metrics.yaml"
    candidate_metrics = load_yaml(candidate_metrics_path)
    candidate_metrics["step_checkpoints"][-1]["eval_loss"] = 0.8
    save_yaml(candidate_metrics_path, candidate_metrics)
    trajectory = build_stage211_trajectory_retention_gate(
        phase="mixer",
        segments=coverage["segments"],
        supplemental_segment=coverage["supplemental_natural"],
        post_coverage_corrections=[],
        checkpoint_path=checkpoint,
    )
    assert trajectory["best_prior"]["source_name"] == "long"
    assert trajectory["best_prior_loss"] == pytest.approx(0.7)
    assert trajectory["candidate_loss"] == pytest.approx(0.8)
    assert trajectory["relative_regression_pct"] == pytest.approx(14.2857142857)
    assert trajectory["gate_passed"] is False

    gate["trajectory_retention"] = trajectory
    gate["trajectory_retention_gate_passed"] = False
    gate["metric_gate_passed"] = False
    gate["gate_passed"] = False
    gate["step_eval_cadence"] = build_stage211_step_eval_cadence(
        phase="mixer",
        segments=coverage["segments"],
        supplemental_segment=coverage["supplemental_natural"],
        post_coverage_corrections=[],
    )
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    validated = validate_stage211_phase_gate_report(
        gate_path,
        expected_phase="mixer",
        checkpoint_path=checkpoint,
        require_passed=False,
    )
    assert validated["alignment_gate_passed"] is True
    assert validated["public_progress_gate_passed"] is True
    assert validated["trajectory_retention_gate_passed"] is False
    assert validated["gate_passed"] is False


def test_stage211_phase_gate_builder_emits_trajectory_retention(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-105.pt"
    checkpoint.write_bytes(b"checkpoint")
    fixture_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    coverage = fixture["full_data_coverage"]
    report = stage211_phase_gate.build_phase_gate(
        phase="mixer",
        checkpoint_path=checkpoint,
        public_comparison_report_path=Path(fixture["public_comparison_report_path"]),
        manifest_dir=tmp_path,
        coverage_receipt_paths=[
            Path(segment["receipt_path"])
            for segment in [*coverage["segments"], coverage["supplemental_natural"]]
        ],
        preflight_smoke_marker_path=Path(fixture["preflight_smoke"]["marker_path"]),
        global_dedup_manifest_path=GLOBAL_DEDUP_FIXTURE,
        loaded_manifest_receipt_path=Path(fixture["loaded_manifest_receipt_path"]),
        alignment_report_path=Path(fixture["alignment_report"]["path"]),
        baseline_public_comparison_report_path=Path(
            fixture["baseline_public_comparison_report"]["path"]
        ),
        nano_public_baseline_receipt_path=Path(fixture["nano_public_baseline_receipt_path"]),
        public_overlap_receipt_path=Path(fixture["public_overlap"]["receipt_path"]),
    )
    assert report["schema_version"] == STAGE211_PHASE_GATE_SCHEMA_VERSION
    assert report["trajectory_retention_gate_passed"] is True
    assert report["trajectory_retention"]["source_order"] == [
        "easy",
        "medium",
        "hard",
        "long",
        STAGE211_SUPPLEMENTAL_DIFFICULTY,
    ]
    assert report["step_eval_cadence"]["complete"] is True
    assert report["step_eval_cadence"]["interval_steps"] == 10_000
    assert report["metric_gate_passed"] is True
    assert report["correction_round_promotion"] == (
        build_stage211_correction_round_promotion_gate(0)
    )
    assert report["gate_passed"] is True


def test_stage211_phase_gate_rejects_tampered_correction_round_promotion(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["correction_round_promotion"]["gate_passed"] = False
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="guaranteed correction-round promotion gate"):
        validate_stage211_phase_gate_report(
            gate_path,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_incomplete_or_inconsistent_step_eval_cadence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    hard = next(
        source for source in gate["step_eval_cadence"]["sources"] if source["source_name"] == "hard"
    )
    metrics_path = Path(hard["metrics_path"])
    original_metrics = metrics_path.read_text(encoding="utf-8")
    metrics = load_yaml(metrics_path)
    metrics["step_checkpoints"].pop(0)
    save_yaml(metrics_path, metrics)
    with pytest.raises(ValueError, match="step-eval cadence mismatch"):
        validate_stage211_phase_gate_report(
            gate_path,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )
    metrics_path.write_text(original_metrics, encoding="utf-8")

    report_path = Path(hard["reports"][0]["report_path"])
    hidden_path = report_path.with_suffix(".hidden")
    report_path.rename(hidden_path)
    with pytest.raises(ValueError, match="report files do not exactly cover"):
        validate_stage211_phase_gate_report(
            gate_path,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )
    hidden_path.rename(report_path)

    stale_path = report_path.parent / "step_eval_layers_step-123.yaml"
    stale_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="report files do not exactly cover"):
        validate_stage211_phase_gate_report(
            gate_path,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )
    stale_path.unlink()

    report = load_yaml(report_path)
    report["eval_loss"] = float(report["eval_loss"]) + 0.25
    save_yaml(report_path, report)
    with pytest.raises(ValueError, match="report metadata mismatch"):
        validate_stage211_phase_gate_report(
            gate_path,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_trajectory_terminal_report(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    candidate_report_path = Path(gate["trajectory_retention"]["candidate"]["eval_report_path"])
    candidate_report = load_yaml(candidate_report_path)
    candidate_report["eval_loss"] = 0.6
    save_yaml(candidate_report_path, candidate_report)

    with pytest.raises(ValueError, match="trajectory retention gate"):
        validate_stage211_phase_gate_report(
            gate_path,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


def test_stage211_trajectory_allows_only_legacy_mixer_easy_missing_seed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    coverage = gate["full_data_coverage"]
    easy_report_path = Path(gate["trajectory_retention"]["entries"][0]["eval_report_path"])
    easy_report = load_yaml(easy_report_path)
    easy_report["eval_provenance"].pop("feature_seed")
    save_yaml(easy_report_path, easy_report)

    trajectory = build_stage211_trajectory_retention_gate(
        phase="mixer",
        segments=coverage["segments"],
        supplemental_segment=coverage["supplemental_natural"],
        post_coverage_corrections=[],
        checkpoint_path=checkpoint,
    )
    assert trajectory["entries"][0]["legacy_missing_feature_seed"] is True
    medium_report_path = Path(trajectory["entries"][1]["eval_report_path"])
    medium_report = load_yaml(medium_report_path)
    medium_report["eval_provenance"].pop("feature_seed")
    save_yaml(medium_report_path, medium_report)
    with pytest.raises(ValueError, match="fixed feature seed mismatch"):
        build_stage211_trajectory_retention_gate(
            phase="mixer",
            segments=coverage["segments"],
            supplemental_segment=coverage["supplemental_natural"],
            post_coverage_corrections=[],
            checkpoint_path=checkpoint,
        )


def _write_failed_phase_gate(gate_report: Path) -> Path:
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(report["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    summary_path = Path(alignment["stratified_summary_path"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["cells"]["easy_en"]["candidate_loss"] = 1.2
    summary["cells"]["easy_en"]["relative_change_pct"] = 20.0
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    alignment["stratified_summary"] = summary
    alignment["stratified_summary_sha256"] = sha256_file(summary_path)
    alignment["stratified_gate_passed"] = False
    alignment["gate_passed"] = False
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    report["alignment_report"]["sha256"] = sha256_file(alignment_path)
    report["alignment_gate_passed"] = False
    report["metric_gate_passed"] = False
    report["gate_passed"] = False
    failed_gate = gate_report.with_name("failed_phase_gate.json")
    failed_gate.write_text(json.dumps(report) + "\n", encoding="utf-8")
    return failed_gate


def _rewrite_phase_alignment(
    gate_report: Path,
    alignment: dict[str, object],
) -> None:
    phase_gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(phase_gate["alignment_report"]["path"])
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    phase_gate["alignment_report"]["sha256"] = sha256_file(alignment_path)
    gate_report.write_text(json.dumps(phase_gate) + "\n", encoding="utf-8")


def _write_retention_correction_batch_profile(
    root: Path,
    *,
    init_checkpoint: Path,
    manifest: Path,
    eval_part: Path,
) -> tuple[dict[str, object], Path]:
    profile_root = root / "batch-profile"
    profile_root.mkdir()
    benchmark = profile_root / "benchmark.py"
    benchmark.write_text("# benchmark\n", encoding="utf-8")
    base_config = profile_root / "base.yaml"
    save_yaml(
        base_config,
        {
            **stage211_phase_train_config_contract("mixer"),
            "webdataset_bucket_manifest_path": str(manifest.resolve()),
            "num_workers": 8,
        },
    )
    loaded_manifest = load_webdataset_bucket_manifest(manifest)
    steps_per_epoch = estimate_bucket_manifest_steps(
        loaded_manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        drop_last=False,
    )
    tail_padding = estimate_bucket_manifest_tail_padding_samples(
        loaded_manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
    )
    baseline = stage211_batch_profile_test._profile_row(
        profile_root,
        name="baseline",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        steps_per_epoch=steps_per_epoch,
        projected_seconds=3.0,
        loss=0.10,
        cosine=0.970,
        phase="mixer",
        init_checkpoint=init_checkpoint.resolve(),
        manifest=manifest.resolve(),
        eval_part=eval_part.resolve(),
    )
    baseline["coverage"]["tail_padding_samples_per_epoch"] = tail_padding
    candidate = stage211_batch_profile_test._profile_row(
        profile_root,
        name="batch48_frames42k",
        batch_size=48,
        frame_budget=42_000,
        steps_per_epoch=steps_per_epoch,
        projected_seconds=6.0,
        loss=0.10,
        cosine=0.970,
        phase="mixer",
        init_checkpoint=init_checkpoint.resolve(),
        manifest=manifest.resolve(),
        eval_part=eval_part.resolve(),
    )
    candidate["coverage"]["tail_padding_samples_per_epoch"] = (
        estimate_bucket_manifest_tail_padding_samples(
            loaded_manifest,
            split="train",
            batch_size=48,
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=42_000,
        )
    )
    comparison = {
        "profile": "batch48_frames42k",
        "projected_full_coverage_seconds": 6.0,
        "improvement_ratio": -1.0,
        "mean_loss": 0.10,
        "loss_regression_ratio": 0.0,
        "mean_cosine": 0.970,
        "cosine_regression": 0.0,
        "fixed_eval_provenance_match": True,
        "quality_pass": True,
        "admissible": False,
    }
    report = {
        "schema_version": stage211_batch_profile_test.STAGE211_BATCH_PROFILE_PREFLIGHT_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "batch_throughput_preflight",
        "phase": "mixer",
        "complete": True,
        "formal_admission": False,
        "dry_run": False,
        "git_commit": "a" * 40,
        "git_worktree_clean": True,
        "git_worktree_changes": [],
        "script_path": str(benchmark.resolve()),
        "script_sha256": sha256_file(benchmark),
        "base_config_path": str(base_config.resolve()),
        "base_config_sha256": sha256_file(base_config),
        "init_checkpoint_path": str(init_checkpoint.resolve()),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "bucket_manifest_path": str(manifest.resolve()),
        "bucket_manifest_sha256": sha256_file(manifest),
        "warmup_steps": 20,
        "measure_steps": 100,
        "probe_depth_fraction": 0.0,
        "formal_epochs": STAGE211_FULL_DATA_EPOCHS,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "gpu_indices": [0, 1, 2, 3],
        "max_peak_memory_gib": 22.0,
        "min_improvement_ratio": 0.10,
        "max_loss_regression_ratio": 0.05,
        "max_cosine_regression": 0.005,
        "candidate_dominance": {
            "enabled_for_capacity_candidates_only": True,
            "rejection_ratio": 4.0,
            "min_points": 12,
            "excluded_profiles": ["baseline"],
        },
        "loader_worker_search": {
            "schema_version": 1,
            "mode": "explicit_profiles",
            "logical_cpus": 16,
            "physical_cores": 8,
            "topology_source": "linux_sysfs_affinity",
            "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
            "configured_num_workers": 8,
            "balanced_num_workers": 2,
            "enabled": False,
            "baseline_profile": "baseline",
            "candidate_profile": None,
            "selection_decision": "explicit_profiles",
            "selected_profile": None,
            "selected_num_workers": None,
        },
        "profiles": [baseline, candidate],
        "selection": {
            "decision": "keep_baseline",
            "baseline_profile": "baseline",
            "recommended_profile": "baseline",
            "recommended_improvement_ratio": 0.0,
            "min_improvement_ratio": 0.10,
            "baseline_mean_loss": 0.10,
            "baseline_mean_cosine": 0.970,
            "max_loss_regression_ratio": 0.05,
            "max_cosine_regression": 0.005,
            "comparisons": [comparison],
            "formal_admission": False,
        },
    }
    report_path = profile_root / "batch_throughput_preflight.json"
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    validated = stage211_batch_profile_test.validate_stage211_batch_profile_preflight(
        report_path,
        phase="mixer",
        require_candidate=False,
    )
    return validated, report_path


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
    replay_part = replay_root / "part.jsonl"
    replay_part.write_text('{"key":"replay-1"}\n', encoding="utf-8")
    prior_eval_report = Path(str(failed["trajectory_retention"]["entries"][0]["eval_report_path"]))
    prior_eval_provenance = load_yaml(prior_eval_report)["eval_provenance"]
    replay_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(replay_part.resolve()),
                "bucket_width": 100,
                "entries_per_part": 256,
                "splits": {
                    "train": {
                        "num_samples": 8,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 8,
                                "parts": [{"path": str(replay_part.resolve()), "num_samples": 8}],
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
                                        "path": part["path"],
                                        "num_samples": part["num_samples"],
                                    }
                                    for part in prior_eval_provenance["parts"]
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
    replay_builder = replay_root / "builder.py"
    replay_builder.write_text("# builder\n", encoding="utf-8")
    replay_preflight = replay_root / "capacity.json"
    replay_preflight.write_text("{}\n", encoding="utf-8")
    bindings = {}
    for name in (
        "base-replay",
        "stratified-v2",
        "supplemental-inventory",
        "supplemental-profile",
        "supplemental-manifest",
    ):
        path = replay_root / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        bindings[name] = path
    replay_receipt = replay_root / "receipt.json"
    replay_receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
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
                "base_replay_receipt": {
                    "path": str(bindings["base-replay"].resolve()),
                    "sha256": sha256_file(bindings["base-replay"]),
                },
                "stratified_hidden_eval": {
                    "path": str(bindings["stratified-v2"].resolve()),
                    "sha256": sha256_file(bindings["stratified-v2"]),
                },
                "supplemental_inputs": {
                    "inventory_path": str(bindings["supplemental-inventory"].resolve()),
                    "inventory_sha256": sha256_file(bindings["supplemental-inventory"]),
                    "profile_receipt_path": str(bindings["supplemental-profile"].resolve()),
                    "profile_receipt_sha256": sha256_file(bindings["supplemental-profile"]),
                    "manifest_path": str(bindings["supplemental-manifest"].resolve()),
                    "manifest_sha256": sha256_file(bindings["supplemental-manifest"]),
                },
                "source_manifest": {
                    "path": str(bindings["supplemental-manifest"].resolve()),
                    "sha256": sha256_file(bindings["supplemental-manifest"]),
                },
                "supplemental_output_parts": [
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
    layer_focus_payload = build_stage211_correction_layer_focus(
        phase="mixer",
        admission_gate_path=failed_gate,
        admission_gate=failed,
    )
    layer_focus = replay_root / "layer-focus.json"
    layer_focus.write_text(
        json.dumps(layer_focus_payload) + "\n",
        encoding="utf-8",
    )
    batch_profile, batch_profile_path = _write_retention_correction_batch_profile(
        replay_root,
        init_checkpoint=init_checkpoint,
        manifest=replay_manifest,
        eval_part=Path(str(prior_eval_provenance["parts"][0]["path"])),
    )
    selected_profile = batch_profile["selected_profile_row"]["profile"]
    selected_coverage = batch_profile["selected_profile_row"]["coverage"]
    steps_per_epoch = int(selected_coverage["steps_per_epoch"])
    tail_padding = int(selected_coverage["tail_padding_samples_per_epoch"])

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
                "admission_mode": "failed_gate",
                "layer_focus_path": str(layer_focus.resolve()),
                "layer_focus_sha256": sha256_file(layer_focus),
                "layer_rotation_offset": 0,
                "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
                "batch_profile_preflight_path": str(batch_profile_path.resolve()),
                "batch_profile_preflight_sha256": sha256_file(batch_profile_path),
                "batch_profile_admission_path": None,
                "batch_profile_admission_sha256": None,
                "batch_profile_name": selected_profile["name"],
                "batch_size": int(selected_profile["batch_size"]),
                "frame_budget": int(selected_profile["frame_budget"]),
                "num_workers": int(selected_profile["num_workers"]),
                "gradient_checkpointing": selected_profile["gradient_checkpointing"],
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
    config = stage211_post_coverage_train_config_contract(
        "mixer",
        boundary_layer_ids=layer_focus_payload["boundary_layer_ids"],
        rotation_offset=0,
    )
    config.update(
        {
            "lr": STAGE211_RETENTION_CORRECTION_LR,
            "max_steps": steps_per_epoch,
            "batch_size": int(selected_profile["batch_size"]),
            "num_workers": int(selected_profile["num_workers"]),
            "batch_token_budget": int(selected_profile["frame_budget"]),
            "length_bucket_frame_budget": int(selected_profile["frame_budget"]),
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
            "webdataset_bucket_manifest_path": str(replay_manifest.resolve()),
            "webdataset_split": "train",
            "ctc_teacher_online_model_path": str(nano_checkpoint.parent.resolve()),
            "init_checkpoint_path": str(init_checkpoint.resolve()),
            "stage211_post_coverage_correction_round": 1,
            "stage211_post_coverage_correction_phase": "mixer",
            "stage211_post_coverage_replay_receipt_path": str(replay_receipt.resolve()),
            "stage211_post_coverage_admission_gate_path": str(failed_gate.resolve()),
            "stage211_post_coverage_admission_mode": "failed_gate",
            "stage211_post_coverage_layer_focus_path": str(layer_focus.resolve()),
            "stage211_post_coverage_layer_focus_sha256": sha256_file(layer_focus),
            "stage211_post_coverage_layer_rotation_offset": 0,
            "stage211_post_coverage_batch_profile_preflight_path": str(
                batch_profile_path.resolve()
            ),
            "stage211_post_coverage_batch_profile_preflight_sha256": sha256_file(
                batch_profile_path
            ),
            "stage211_post_coverage_batch_profile_admission_path": None,
            "stage211_post_coverage_batch_profile_admission_sha256": None,
            "stage211_post_coverage_batch_profile_name": selected_profile["name"],
            "stage211_post_coverage_batch_size": int(selected_profile["batch_size"]),
            "stage211_post_coverage_frame_budget": int(selected_profile["frame_budget"]),
            "stage211_post_coverage_num_workers": int(selected_profile["num_workers"]),
            "stage211_post_coverage_gradient_checkpointing": selected_profile[
                "gradient_checkpointing"
            ],
            "stage211_post_coverage_original_coverage_unchanged": True,
            "stage211_post_coverage_smoke_marker_path": str(smoke_marker.resolve()),
            "stage211_post_coverage_smoke_marker_sha256": sha256_file(smoke_marker),
            "step_eval_every": 10_000,
            "step_eval_samples": 256,
            "step_eval_split": "eval",
            "step_eval_shuffle": False,
            "step_eval_feature_seed": 0,
            "stage211_batch_profile_admission_path": None,
            "stage211_batch_profile_admission_sha256": None,
            "stage211_batch_profile_name": None,
            "deepspeed": {
                "gradient_accumulation_steps": 1,
                "train_micro_batch_size_per_gpu": int(selected_profile["batch_size"]),
                "train_batch_size": (
                    int(selected_profile["batch_size"]) * STAGE211_FULL_DATA_WORLD_SIZE
                ),
            },
        }
    )
    save_yaml(train_config, config)
    provenance = run_dir / "stage211_correction_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "pipeline": "stage211",
                "artifact": "retention_correction_run",
                "phase": "mixer",
                "round": 1,
                "run_dir": str(run_dir.resolve()),
                "replay_receipt_path": str(replay_receipt.resolve()),
                "replay_receipt_sha256": sha256_file(replay_receipt),
                "replay_manifest_path": str(replay_manifest.resolve()),
                "replay_manifest_sha256": sha256_file(replay_manifest),
                "admission_gate_path": str(failed_gate.resolve()),
                "admission_gate_sha256": sha256_file(failed_gate),
                "admission_mode": "failed_gate",
                "init_checkpoint_path": str(init_checkpoint.resolve()),
                "init_checkpoint_sha256": sha256_file(init_checkpoint),
                "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
                "smoke_marker_path": str(smoke_marker.resolve()),
                "smoke_marker_sha256": sha256_file(smoke_marker),
                "layer_focus_path": str(layer_focus.resolve()),
                "layer_focus_sha256": sha256_file(layer_focus),
                "layer_rotation_offset": 0,
                "batch_profile_preflight_path": str(batch_profile_path.resolve()),
                "batch_profile_preflight_sha256": sha256_file(batch_profile_path),
                "batch_profile_admission_path": None,
                "batch_profile_admission_sha256": None,
                "batch_profile_name": selected_profile["name"],
                "batch_size": int(selected_profile["batch_size"]),
                "frame_budget": int(selected_profile["frame_budget"]),
                "num_workers": int(selected_profile["num_workers"]),
                "gradient_checkpointing": selected_profile["gradient_checkpointing"],
                "correction_extension_decision_path": None,
                "correction_extension_decision_sha256": None,
                "epochs": STAGE211_RETENTION_CORRECTION_EPOCHS,
                "steps_per_epoch": steps_per_epoch,
                "learning_rate": STAGE211_RETENTION_CORRECTION_LR,
                "trainable_boundary": "mixer_only",
                "early_stopping": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    epoch_checkpoint = run_dir / "epoch-1.pt"
    epoch_checkpoint.write_bytes(b"epoch-1")
    correction = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "post_coverage_correction",
        "phase": "mixer",
        "round": 1,
        "complete": True,
        "epochs": STAGE211_RETENTION_CORRECTION_EPOCHS,
        "learning_rate": STAGE211_RETENTION_CORRECTION_LR,
        "batch_size": int(selected_profile["batch_size"]),
        "num_workers": int(selected_profile["num_workers"]),
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": int(selected_profile["frame_budget"]),
        "gradient_checkpointing": selected_profile["gradient_checkpointing"],
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "rows": 8,
        "row_exposures": 8,
        "hours": 0.01,
        "hour_exposures": 0.01,
        "steps_per_epoch": steps_per_epoch,
        "steps": steps_per_epoch,
        "tail_padding_samples_per_epoch": tail_padding,
        "tail_padding_sample_exposures": tail_padding,
        "executed_sample_exposures": 8 + tail_padding,
        "run_dir": str(run_dir.resolve()),
        "provenance_path": str(provenance.resolve()),
        "provenance_sha256": sha256_file(provenance),
        "train_config_path": str(train_config.resolve()),
        "train_config_sha256": sha256_file(train_config),
        "smoke_marker_path": str(smoke_marker.resolve()),
        "smoke_marker_sha256": sha256_file(smoke_marker),
        "layer_focus_path": str(layer_focus.resolve()),
        "layer_focus_sha256": sha256_file(layer_focus),
        "layer_rotation_offset": 0,
        "layer_focus_strategy": str(layer_focus_payload["strategy"]),
        "failed_layer_count": int(layer_focus_payload["failed_layer_count"]),
        "dynamic_layer_limit": int(layer_focus_payload["dynamic_layer_limit"]),
        "adaptive_dynamic_layer_limit": int(layer_focus_payload["adaptive_dynamic_layer_limit"]),
        "minimum_rotating_layer_slots": int(layer_focus_payload["minimum_rotating_slots"]),
        "adaptive_rotating_layer_slots_target": int(
            layer_focus_payload["adaptive_rotating_slots_target"]
        ),
        "boundary_layer_ids": list(layer_focus_payload["boundary_layer_ids"]),
        "selected_failure_layer_ids": list(layer_focus_payload["selected_failure_layer_ids"]),
        "all_failed_layer_ids": list(layer_focus_payload["all_failed_layer_ids"]),
        "rotating_layer_slots": int(layer_focus_payload["rotating_slots"]),
        "replay_receipt_path": str(replay_receipt.resolve()),
        "replay_receipt_sha256": sha256_file(replay_receipt),
        "bucket_manifest_path": str(replay_manifest.resolve()),
        "bucket_manifest_sha256": sha256_file(replay_manifest),
        "admission_gate_path": str(failed_gate.resolve()),
        "admission_gate_sha256": sha256_file(failed_gate),
        "admission_mode": "failed_gate",
        "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "batch_profile_preflight_path": str(batch_profile_path.resolve()),
        "batch_profile_preflight_sha256": sha256_file(batch_profile_path),
        "batch_profile_admission_path": None,
        "batch_profile_admission_sha256": None,
        "batch_profile_name": selected_profile["name"],
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
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch,
            "records": [
                {
                    "epoch": 1,
                    "step": steps_per_epoch,
                    "epoch_batch_offset": 0,
                    "completed_epoch_batch_count": steps_per_epoch,
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
    bound_correction = {
        **correction,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
    }
    _write_trajectory_eval_fixture(
        phase="mixer",
        record=bound_correction,
        eval_loss=0.45,
    )
    _write_step_eval_cadence_source_fixture(bound_correction)
    return bound_correction, replay_part


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
            "stratified_gate_passed": True,
            "gate_passed": True,
        }
    )
    failed_summary_path = Path(alignment["stratified_summary_path"])
    corrected_summary = json.loads(failed_summary_path.read_text(encoding="utf-8"))
    corrected_summary["checkpoints"]["candidate"] = {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
    }
    corrected_summary["cells"]["easy_en"]["candidate_loss"] = 0.5
    corrected_summary["cells"]["easy_en"]["relative_change_pct"] = -50.0
    corrected_summary_path = tmp_path / "corrected-stratified-summary.json"
    corrected_summary_path.write_text(
        json.dumps(corrected_summary) + "\n",
        encoding="utf-8",
    )
    alignment["stratified_summary_path"] = str(corrected_summary_path.resolve())
    alignment["stratified_summary_sha256"] = sha256_file(corrected_summary_path)
    alignment["stratified_summary"] = corrected_summary
    alignment_path = tmp_path / "corrected-alignment-gate.json"
    alignment_path.write_text(json.dumps(alignment) + "\n", encoding="utf-8")
    report.update(
        {
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "gate_passed": False,
            "metric_gate_passed": True,
            "correction_round_promotion": (build_stage211_correction_round_promotion_gate(1)),
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
    report["trajectory_retention"] = build_stage211_trajectory_retention_gate(
        phase="mixer",
        segments=original_segments,
        supplemental_segment=supplemental_segment,
        post_coverage_corrections=[correction],
        checkpoint_path=checkpoint,
    )
    report["trajectory_retention_gate_passed"] = bool(report["trajectory_retention"]["gate_passed"])
    report["step_eval_cadence"] = build_stage211_step_eval_cadence(
        phase="mixer",
        segments=original_segments,
        supplemental_segment=supplemental_segment,
        post_coverage_corrections=[correction],
    )
    public_source_path = Path(report["public_comparison_report_path"])
    public_source = json.loads(public_source_path.read_text(encoding="utf-8"))
    public_source["student_checkpoint_path"] = str(checkpoint.resolve())
    public_source["student_checkpoint_sha256"] = sha256_file(checkpoint)
    prior_receipt_path = Path(public_source["student_prediction_receipt_path"])
    prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
    receipt_results = {str(result["dataset"]): result for result in prior_receipt["results"]}
    public_results = {str(result["dataset"]): result for result in public_source["results"]}
    corrected_provenance, corrected_tokenizer = stage211_test_student_ctc_context(checkpoint)
    corrected_prediction_paths: dict[str, Path] = {}
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        prior_prediction_path = Path(str(public_results[dataset]["student_prediction_path"]))
        corrected_prediction_path = tmp_path / f"{dataset}.corrected.jsonl"
        corrected_rows = []
        for line in prior_prediction_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            corrected_rows.append(
                stage211_test_student_ctc_row(
                    utt_id=str(row["utt_id"]),
                    ref_text=str(row["ref_text"]),
                    pred_text=str(row["pred_text"]),
                    provenance=corrected_provenance,
                    tokenizer=corrected_tokenizer,
                )
            )
        corrected_prediction_path.write_text(
            "".join(json.dumps(row) + "\n" for row in corrected_rows),
            encoding="utf-8",
        )
        corrected_prediction_paths[dataset] = corrected_prediction_path.resolve()
        public_results[dataset]["student_prediction_path"] = str(
            corrected_prediction_path.resolve()
        )
    corrected_receipt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=checkpoint,
        manifest_paths={
            dataset: Path(str(receipt_results[dataset]["manifest_path"]))
            for dataset in STAGE211_PUBLIC_BENCHMARKS
        },
        prediction_paths=corrected_prediction_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
    )
    corrected_receipt_path = tmp_path / "corrected-public-prediction-receipt.json"
    corrected_receipt_path.write_text(
        json.dumps(corrected_receipt) + "\n",
        encoding="utf-8",
    )
    public_source["student_prediction_receipt_path"] = str(corrected_receipt_path.resolve())
    public_source["student_prediction_receipt_sha256"] = sha256_file(corrected_receipt_path)
    report["public_benchmark"] = replay_stage211_public_comparison(
        public_source,
        manifest_paths={
            dataset: Path(str(receipt_results[dataset]["manifest_path"]))
            for dataset in STAGE211_PUBLIC_BENCHMARKS
        },
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_checkpoint=checkpoint,
    )
    corrected_public_source_path = tmp_path / "corrected-public-comparison.json"
    corrected_public_source_path.write_text(
        json.dumps(public_source) + "\n",
        encoding="utf-8",
    )
    report["public_comparison_report_path"] = str(corrected_public_source_path.resolve())
    report["public_comparison_report_sha256"] = sha256_file(corrected_public_source_path)
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
        require_passed=False,
    )
    assert validated["metric_gate_passed"] is True
    assert validated["correction_round_promotion"]["gate_passed"] is False
    assert validated["gate_passed"] is False
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
            require_passed=False,
        )


def test_stage211_phase_gate_rejects_mutated_correction_focus_summary(
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
    receipt_path = Path(str(correction["receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["failed_layer_count"] = int(receipt["failed_layer_count"]) + 1
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

    with pytest.raises(ValueError, match="layer-focus summary mismatch"):
        validate_stage211_phase_gate_report(
            corrected_gate,
            expected_phase="mixer",
            checkpoint_path=corrected_checkpoint,
            require_passed=False,
        )


def test_stage211_phase_gate_rejects_spoofed_early_pass_admission_mode(
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
    spoofed_mode = "early_pass_mandatory_continuation"

    smoke_path = Path(str(correction["smoke_marker_path"]))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["admission_mode"] = spoofed_mode
    smoke_path.write_text(json.dumps(smoke) + "\n", encoding="utf-8")

    config_path = Path(str(correction["train_config_path"]))
    config = load_yaml(config_path)
    config["stage211_post_coverage_admission_mode"] = spoofed_mode
    config["stage211_post_coverage_smoke_marker_sha256"] = sha256_file(smoke_path)
    save_yaml(config_path, config)

    provenance_path = Path(str(correction["provenance_path"]))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["admission_mode"] = spoofed_mode
    provenance["smoke_marker_sha256"] = sha256_file(smoke_path)
    provenance_path.write_text(json.dumps(provenance) + "\n", encoding="utf-8")

    receipt_path = Path(str(correction["receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "admission_mode": spoofed_mode,
            "smoke_marker_sha256": sha256_file(smoke_path),
            "train_config_sha256": sha256_file(config_path),
            "provenance_sha256": sha256_file(provenance_path),
        }
    )
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

    with pytest.raises(ValueError, match="admission mode mismatch"):
        validate_stage211_phase_gate_report(
            corrected_gate,
            expected_phase="mixer",
            checkpoint_path=corrected_checkpoint,
            require_passed=False,
        )


def test_stage211_phase_gate_rejects_mutated_correction_layer_rotation_offset(
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
    receipt_path = Path(str(correction["receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["layer_rotation_offset"] = 1
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

    with pytest.raises(ValueError, match="layer rotation offset mismatch"):
        validate_stage211_phase_gate_report(
            corrected_gate,
            expected_phase="mixer",
            checkpoint_path=corrected_checkpoint,
            require_passed=False,
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
            require_passed=False,
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


def test_stage211_phase_gate_replays_public_wer_cer_from_predictions(
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
    comparison_path = Path(gate["public_comparison_report_path"])
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["results"][0]["student_error_rate"] = 0.01
    comparison_path.write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    gate["public_comparison_report_sha256"] = sha256_file(comparison_path)
    gate["public_benchmark"]["results"][0]["student_error_rate"] = 0.01
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="replayed student_error_rate mismatch"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_missing_student_prediction_provenance(
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
    comparison_path = Path(gate["public_comparison_report_path"])
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison.pop("student_prediction_receipt_path")
    comparison.pop("student_prediction_receipt_sha256")
    comparison_path.write_text(json.dumps(comparison) + "\n", encoding="utf-8")
    gate["public_comparison_report_sha256"] = sha256_file(comparison_path)
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks student prediction provenance"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
            checkpoint_path=checkpoint,
        )


def test_stage211_phase_gate_rejects_mutated_baseline_prediction_receipt(
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
    receipt_path = Path(gate["baseline_public_provenance"]["student_prediction_receipt_path"])
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prediction receipt SHA-256 mismatch"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize("phase", ("block", "logits"))
def test_stage211_stacked_phase_gate_rejects_legacy_baseline_provenance(
    tmp_path: Path,
    phase: str,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase=phase,
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    gate["baseline_public_provenance"] = {
        "mode": "legacy_calibration_reuse",
        "calibration_reuse_receipt_path": str(tmp_path / "reuse.json"),
        "calibration_reuse_receipt_sha256": "0" * 64,
        "initialization_receipt_path": str(tmp_path / "initialization.json"),
        "initialization_receipt_sha256": "1" * 64,
    }
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="baseline must use intrinsic prediction provenance"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase=phase,
            checkpoint_path=checkpoint,
        )


def test_stage211_mixer_phase_gate_replays_legacy_calibration_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    baseline_path = Path(gate["baseline_public_comparison_report"]["path"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline.pop("student_prediction_receipt_path")
    baseline.pop("student_prediction_receipt_sha256")
    baseline_path.write_text(json.dumps(baseline) + "\n", encoding="utf-8")
    gate["baseline_public_comparison_report"]["sha256"] = sha256_file(baseline_path)

    phase_init_checkpoint = Path(gate["full_data_coverage"]["segments"][0]["init_checkpoint_path"])
    manifest_paths = {
        str(row["dataset"]): Path(row["manifest_path"])
        for row in gate["public_benchmark"]["results"]
    }
    baseline_benchmark = replay_stage211_public_comparison(
        baseline,
        manifest_paths=manifest_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_checkpoint=phase_init_checkpoint,
    )
    gate["public_progress"] = stage211_phase_gate._build_public_progress(
        baseline=baseline_benchmark,
        candidate=gate["public_benchmark"],
    )
    selection_path = tmp_path / "calibration-selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "pipeline": "stage211",
                "artifact": "calibration_checkpoint_selection",
                "required_completion_step": 30_064,
                "selected": {
                    "eligible": True,
                    "loss_improved_layers": 70,
                    "cosine_improved_layers": 70,
                    "checkpoint_path": str(phase_init_checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(phase_init_checkpoint),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics_path = tmp_path / "calibration-metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "dataset": row["dataset"],
                        "branch": "ctc",
                        "samples": row["sample_count"],
                        "wer": row["student_wer"],
                        "cer": row["student_cer"],
                    }
                    for row in baseline_benchmark["results"]
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    reuse_path = tmp_path / "calibration-reuse.json"
    reuse_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "calibration_public_eval_reuse",
                "complete": True,
                "checkpoint_path": str(phase_init_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(phase_init_checkpoint),
                "selection_report_path": str(selection_path.resolve()),
                "selection_report_sha256": sha256_file(selection_path),
                "metrics_path": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "comparison_report_path": str(baseline_path.resolve()),
                "comparison_report_sha256": sha256_file(baseline_path),
                "public_overlap": gate["public_overlap"],
                "public_benchmark": baseline_benchmark,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    initialization_path = tmp_path / "initialization.json"
    initialization_path.write_text("{}\n", encoding="utf-8")
    initialization_module = importlib.import_module("rwkvasr.eval.stage211_initialization")

    def fake_validate_initialization(
        receipt_path: Path,
        *,
        expected_calibration_checkpoint: Path,
        expected_nano_checkpoint_sha256: str,
    ) -> dict[str, object]:
        assert receipt_path == initialization_path.resolve()
        assert expected_calibration_checkpoint == phase_init_checkpoint.resolve()
        assert expected_nano_checkpoint_sha256 == gate["nano_public_baseline_checkpoint_sha256"]
        return {
            "calibration_reuse_receipt_path": str(reuse_path.resolve()),
            "calibration_reuse_receipt_sha256": sha256_file(reuse_path),
        }

    monkeypatch.setattr(
        initialization_module,
        "validate_stage211_initialization_receipt",
        fake_validate_initialization,
    )
    gate["baseline_public_provenance"] = {
        "mode": "legacy_calibration_reuse",
        "calibration_reuse_receipt_path": str(reuse_path.resolve()),
        "calibration_reuse_receipt_sha256": sha256_file(reuse_path),
        "initialization_receipt_path": str(initialization_path.resolve()),
        "initialization_receipt_sha256": sha256_file(initialization_path),
    }
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    validated = validate_stage211_phase_gate_report(
        gate_report,
        expected_phase="mixer",
        checkpoint_path=checkpoint,
    )
    assert validated["baseline_public_provenance"]["mode"] == ("legacy_calibration_reuse")

    reuse_path.write_text(
        reuse_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="baseline public provenance"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="mixer",
            checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize("phase", ("mixer", "block", "logits"))
def test_stage211_phase_gate_replays_public_progress_from_baseline(
    tmp_path: Path,
    phase: str,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase=phase,
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_report.read_text(encoding="utf-8"))
    gate["public_progress"]["macro_candidate_error_rate"] = 0.25
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="public progress gate"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase=phase,
            checkpoint_path=checkpoint,
        )


def test_stage211_logits_phase_gate_rejects_missing_block_public_baseline(
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
    gate["baseline_public_comparison_report"] = None
    gate["public_progress"] = None
    gate_report.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks its baseline public report"):
        validate_stage211_phase_gate_report(
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


def test_stage211_block_phase_gate_rejects_unsafe_smoke_capacity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "block-complete.pt"
    checkpoint.write_bytes(b"block-complete")
    gate_report = _write_valid_phase_gate(tmp_path, phase="block", checkpoint=checkpoint)
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    marker_path = Path(report["preflight_smoke"]["marker_path"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    smoke_config = Path(marker["smoke_config_path"])
    config = load_yaml(smoke_config)
    config["batch_size"] = 36
    save_yaml(smoke_config, config)
    marker["smoke_config_sha256"] = sha256_file(smoke_config)
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    report["preflight_smoke"]["marker_sha256"] = sha256_file(marker_path)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="memory-safe profile"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="block",
            checkpoint_path=checkpoint,
        )


def test_stage211_logits_phase_gate_replays_smoke_runtime_objectives(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "logits-complete.pt"
    checkpoint.write_bytes(b"logits-complete")
    gate_report = _write_valid_phase_gate(tmp_path, phase="logits", checkpoint=checkpoint)
    report = json.loads(gate_report.read_text(encoding="utf-8"))
    marker_path = Path(report["preflight_smoke"]["marker_path"])
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["runtime_objective_evidence"]["active_loss_fields"]["online_ctc_full"] = 0.5
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    report["preflight_smoke"]["marker_sha256"] = sha256_file(marker_path)
    gate_report.write_text(json.dumps(report) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime-objective evidence changed"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
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


def test_stage211_block_phase_gate_replays_three_component_evidence(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="block",
        checkpoint=checkpoint,
    )

    validated = validate_stage211_phase_gate_report(
        gate_report,
        expected_phase="block",
        checkpoint_path=checkpoint,
    )
    assert validated["gate_passed"] is True

    phase_gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(phase_gate["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    candidate_path = Path(alignment["candidate_report_path"])
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["layer_components"].pop("ffn")
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    alignment["candidate_report_sha256"] = sha256_file(candidate_path)
    _rewrite_phase_alignment(gate_report, alignment)

    with pytest.raises(ValueError, match="exactly 70 ffn layers"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="block",
            checkpoint_path=checkpoint,
        )


def test_stage211_logits_phase_gate_rejects_missing_hidden_retention_schema(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="logits",
        checkpoint=checkpoint,
    )
    phase_gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(phase_gate["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment.pop("hidden_retention_passed")
    _rewrite_phase_alignment(gate_report, alignment)

    with pytest.raises(ValueError, match="hidden_retention_passed"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
            checkpoint_path=checkpoint,
        )


def test_stage211_logits_phase_gate_replays_stratified_hidden_retention(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_report = _write_valid_phase_gate(
        tmp_path,
        phase="logits",
        checkpoint=checkpoint,
    )
    phase_gate = json.loads(gate_report.read_text(encoding="utf-8"))
    alignment_path = Path(phase_gate["alignment_report"]["path"])
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment["stratified_checks"]["ffn_hidden_retained"] = False
    _rewrite_phase_alignment(gate_report, alignment)

    with pytest.raises(ValueError, match="ffn_hidden_retained"):
        validate_stage211_phase_gate_report(
            gate_report,
            expected_phase="logits",
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

    with pytest.raises(ValueError, match="public WER/CER benchmark"):
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
        report["full_data_coverage"]["supplemental_natural"]["supplemental_inventory_path"]
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
    gate_report = _write_failed_phase_gate(gate_report)

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

    report = json.loads(gate_report.read_text(encoding="utf-8"))
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
    assert ".nano_initialization_source_chain_passed == true" in script
    assert ".ctc_label_normalization_chain_passed == true" in script
    assert ".ctc_label_proof.ctc_unk_tokens == 0" in script
    assert ".public_metric_definition_chain_passed == true" in script
    assert ".public_metric_strip_language_confirmation == false" in script
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
        ".supplemental_dedupe_proof.base_public_overlap_scan_order == "
        '"manifest_location_index_archive_order_v1"' in script
    )
    assert ".supplemental_dedupe_proof.base_public_overlap_scanned_rows > 0" in script
    assert ".supplemental_dedupe_proof.base_public_overlap_receipt_sha256" in script
    assert ".supplemental_dedupe_proof.social_normalized_pcm_exact_complete == true" in script
    assert (
        '.supplemental_dedupe_proof.social_public_overlap_mode == "normalized_pcm_exact"' in script
    )
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
    assert ".all_requested_alignment_metrics_complete == true" in script
    assert "[.requested_alignment_results[] | .stage]" in script
    assert "[.requested_alignment_results[] | .internal_stage]" in script
    assert ".requested_alignment_language_metric_summaries" in script
    assert ".requested_alignment_dataset_results" in script
    assert "stage211_trajectory_results_valid" in script
    assert (
        '.source_order[:5] == ["easy", "medium", "hard", "long", "supplemental_natural"]' in script
    )
    assert ".terminal_entries >= 5" in script
    assert ".max_relative_regression_pct == 10.0" in script
    assert "stage211_periodic_cadence_results_valid" in script
    assert ".interval_steps == 10000" in script
    assert ".interval_steps == 2000" in script
    assert '.source_order == ["labeled_sft"]' in script
    assert "stage211_full_labeled_profile_valid" in script
    assert "$labels.total_samples == ($labels.train_samples + $labels.eval_samples)" in script
    assert "$labels.public_clean_rebuild_passed == true" in script
    assert "$labels.public_clean_exclusions_rows == 21" in script
    assert "$labels.public_audio_scanned_rows == $labels.total_samples" in script
    assert "$sft.steps > 0" in script
    assert "$sft.step_eval_cadence.sources[0].terminal_step == $sft.steps" in script
    assert "stage211_requested_nano_gaps_valid" in script
    assert "english_wer_gap_to_nano" in script
    assert "chinese_cer_gap_to_nano" in script
    assert "stage211_sft_correction_coverage_valid" in script
    assert ".sft_correction_evidence" in script
    assert ".effective_executed_sample_exposures" in script
    assert "post_mixer" in script
    assert "tmux kill-session" not in script
    assert '[[ "${BASH_SOURCE[0]}" == "$0" ]]' in script
    assert "stage211_selection_valid" in script
    assert "--validate-selection-only" in script
    assert "stage211_mixer_curriculum_valid" in script
    assert "--validate-curriculum-only" in script


def test_stage211_curriculum_only_validation_skips_all_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "supplemental-complete.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[tuple[str, Path]] = []

    def fake_resolve_curriculum(*, phase: str, phase_root: Path):
        calls.append((phase, phase_root))
        receipts = [tmp_path / f"receipt-{index}" for index in range(5)]
        return {"total_unique_rows": 123}, checkpoint, receipts

    monkeypatch.setattr(
        stage211_phase_finalizer,
        "_resolve_curriculum",
        fake_resolve_curriculum,
    )
    monkeypatch.setattr(
        stage211_phase_finalizer,
        "finalize_phase",
        lambda args: pytest.fail("curriculum-only validation launched evaluation"),
    )
    phase_root = tmp_path / "phase"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(stage211_phase_finalizer.__file__),
            "--phase",
            "mixer",
            "--phase-root",
            str(phase_root),
            "--validate-curriculum-only",
        ],
    )

    assert stage211_phase_finalizer.main() == 0
    assert calls == [("mixer", phase_root.resolve())]
    assert "segments=5 rows=123" in capsys.readouterr().out


def _choose_stage211_continuation_stage(
    tmp_path: Path,
    summary: dict[str, object],
    *,
    curriculum_validator_exit_code: int = 42,
) -> str:
    output_root = tmp_path / "runs"
    summary_path = output_root / "stage211a_mixer_full_data_3ep" / "curriculum_complete.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${CURRICULUM_VALIDATOR_CALLS}"\n'
        'exit "${CURRICULUM_VALIDATOR_EXIT_CODE}"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    script = REPO_ROOT / "scripts" / "watch_stage211_strict_continuation.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; stage211_choose_start_stage', "_", str(script)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FULL_OUTPUT_ROOT": str(output_root),
            "PHASE_GATE_ROOT": str(tmp_path / "gates"),
            "WATCH_LOG": str(tmp_path / "watch.log"),
            "CURRICULUM_VALIDATOR_CALLS": str(tmp_path / "curriculum-validator-calls.log"),
            "CURRICULUM_VALIDATOR_EXIT_CODE": str(curriculum_validator_exit_code),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_stage211_continuation_watcher_routes_legacy_four_segment_summary_to_full(
    tmp_path: Path,
) -> None:
    assert (
        _choose_stage211_continuation_stage(
            tmp_path,
            {
                "pipeline": "stage211",
                "artifact": "full_phase_curriculum",
                "phase": "mixer",
                "complete": True,
                "full_data_coverage": {
                    "segments": [
                        {"difficulty": difficulty, "epochs": 3}
                        for difficulty in ("easy", "medium", "hard", "long")
                    ]
                },
            },
        )
        == "full"
    )


def test_stage211_continuation_watcher_routes_complete_supplemental_to_post_mixer(
    tmp_path: Path,
) -> None:
    assert (
        _choose_stage211_continuation_stage(
            tmp_path,
            {
                "pipeline": "stage211",
                "artifact": "full_phase_curriculum",
                "phase": "mixer",
                "complete": True,
                "full_data_coverage": {"supplemental_natural": {"complete": True, "epochs": 3}},
            },
            curriculum_validator_exit_code=0,
        )
        == "post_mixer"
    )


def test_stage211_continuation_watcher_rejects_shallow_supplemental_summary(
    tmp_path: Path,
) -> None:
    assert (
        _choose_stage211_continuation_stage(
            tmp_path,
            {
                "pipeline": "stage211",
                "artifact": "full_phase_curriculum",
                "phase": "mixer",
                "complete": True,
                "full_data_coverage": {"supplemental_natural": {"complete": True, "epochs": 3}},
            },
            curriculum_validator_exit_code=17,
        )
        == "full"
    )
    validator_command = (tmp_path / "curriculum-validator-calls.log").read_text(encoding="utf-8")
    assert str(stage211_phase_finalizer.__file__) in validator_command
    assert "--phase mixer" in validator_command
    assert "--validate-curriculum-only" in validator_command


@pytest.mark.parametrize(
    ("validator_exit_code", "expected_stage"),
    ((0, "sft"), (42, "full")),
)
def test_stage211_continuation_watcher_requires_deep_logits_selection_validation(
    tmp_path: Path,
    validator_exit_code: int,
    expected_stage: str,
) -> None:
    output_root = tmp_path / "runs"
    phase_gate_root = tmp_path / "gates"
    phase_gate_root.mkdir()
    (phase_gate_root / "logits_selected.json").write_text(
        '{"pipeline":"stage211","artifact":"phase_gate_selection","phase":"logits"}\n',
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    validator_calls = tmp_path / "validator-calls.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${VALIDATOR_CALLS}"\n'
        'exit "${VALIDATOR_EXIT_CODE}"\n',
        encoding="utf-8",
    )
    uv.chmod(0o755)
    script = REPO_ROOT / "scripts" / "watch_stage211_strict_continuation.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; stage211_choose_start_stage', "_", str(script)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FULL_OUTPUT_ROOT": str(output_root),
            "PHASE_GATE_ROOT": str(phase_gate_root),
            "WATCH_LOG": str(tmp_path / "watch.log"),
            "VALIDATOR_CALLS": str(validator_calls),
            "VALIDATOR_EXIT_CODE": str(validator_exit_code),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected_stage
    validator_command = validator_calls.read_text(encoding="utf-8")
    assert "--phase logits" in validator_command
    assert f"--selection {phase_gate_root / 'logits_selected.json'}" in validator_command
    assert "--nano-checkpoint" in validator_command
    assert "--validate-selection-only" in validator_command


def test_stage211_continuation_watcher_falls_back_to_nearest_deep_selection(
    tmp_path: Path,
) -> None:
    phase_gate_root = tmp_path / "gates"
    phase_gate_root.mkdir()
    for phase in ("logits", "block", "mixer"):
        (phase_gate_root / f"{phase}_selected.json").write_text("{}\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    validator_calls = tmp_path / "validator-calls.log"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >>"${VALIDATOR_CALLS}"\n'
        'if [[ "$*" == *"--phase logits"* ]]; then exit 42; fi\n'
        'if [[ "$*" == *"--phase block"* ]]; then exit 0; fi\n'
        "exit 99\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    script = REPO_ROOT / "scripts" / "watch_stage211_strict_continuation.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; stage211_choose_start_stage', "_", str(script)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FULL_OUTPUT_ROOT": str(tmp_path / "runs"),
            "PHASE_GATE_ROOT": str(phase_gate_root),
            "WATCH_LOG": str(tmp_path / "watch.log"),
            "VALIDATOR_CALLS": str(validator_calls),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "logits"
    calls = validator_calls.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--phase logits" in calls[0]
    assert "--phase block" in calls[1]
    assert all("--phase mixer" not in call for call in calls)


def _run_stage211_continuation_watcher_fixture(
    tmp_path: Path,
    *,
    uv_mode: str,
    tmux_mode: str = "stateless",
    watch_once: str | None = "1",
    initial_final_report: bool = True,
    corrected_final_report: bool = False,
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
        "      if ((launches >= 2)); then\n"
        '        mkdir -p "$(dirname "${FINAL_REPORT_PATH}")"\n'
        "        printf '%s\\n' "
        '\'{"pipeline":"stage211","artifact":"final_completion",'
        '"complete":true,"gate_passed":true}\' >"${FINAL_REPORT_PATH}"\n'
        "      fi\n"
        "      exit 0\n"
        "    fi\n"
        "    exit 1\n"
        "  fi\n"
        '  if [[ "${1:-}" == new-session '
        '&& "$*" == *rwkvasr_stage211_abcd_strict_supervisor* ]]; then\n'
        "    launches=0\n"
        '    [[ -f "${TMUX_STATE}.launches" ]] '
        '&& launches="$(cat "${TMUX_STATE}.launches")"\n'
        '    printf \'%s\\n\' "$((launches + 1))" >"${TMUX_STATE}.launches"\n'
        '    touch "${TMUX_STATE}.active"\n'
        "  fi\n"
        "  exit 0\n"
        "fi\n"
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
        'elif [[ "${UV_MODE}" == missing_language_macro ]]; then\n'
        '  stages=\'{"calibration":{},"mixer":{},"block":{},"logits":{},"sft":{}}\'\n'
        '  dataset_proof=\'"public_metric_stage_order":["calibration","mixer","block","logits","sft"],"all_stage_public_metrics_complete":true,"english_wer_datasets":["en1","en2","en3"],"chinese_cer_datasets":["zh1","zh2"],"dataset_results":[{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"zh","metric":"cer","stages":\'"${stages}"\'},{"language":"zh","metric":"cer","stages":\'"${stages}"\'}]\'\n'
        "else\n"
        '  stages=\'{"calibration":{},"mixer":{},"block":{},"logits":{},"sft":{}}\'\n'
        '  macro_stages=\'{"calibration":0.5,"mixer":0.4,"block":0.3,"logits":0.2,"sft":0.1}\'\n'
        '  dataset_proof=\'"public_metric_stage_order":["calibration","mixer","block","logits","sft"],"all_stage_public_metrics_complete":true,"english_wer_datasets":["en1","en2","en3"],"chinese_cer_datasets":["zh1","zh2"],"language_metric_summaries":[{"name":"english_wer","language":"en","metric":"wer","aggregation":"unweighted_dataset_macro","datasets":["en1","en2","en3"],"dataset_count":3,"sample_count":3,"nano_error_rate":0.1,"stages":\'"${macro_stages}"\'},{"name":"chinese_cer","language":"zh","metric":"cer","aggregation":"unweighted_dataset_macro","datasets":["zh1","zh2"],"dataset_count":2,"sample_count":2,"nano_error_rate":0.1,"stages":\'"${macro_stages}"\'}],"dataset_results":[{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"en","metric":"wer","stages":\'"${stages}"\'},{"language":"zh","metric":"cer","stages":\'"${stages}"\'},{"language":"zh","metric":"cer","stages":\'"${stages}"\'}]\'\n'
        "fi\n"
        '  requested_stage_metrics=\'{"rwkv_layer":{"student_error_rate":0.4},"block":{"student_error_rate":0.3},"logits":{"student_error_rate":0.2},"sft":{"student_error_rate":0.1}}\'\n'
        '  requested_proof=\'"all_requested_alignment_metrics_complete":true,"initial_calibration_result":{"role":"initial_baseline","checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","english_wer":0.5,"chinese_cer":0.5},"requested_alignment_results":[{"stage":"rwkv_layer","internal_stage":"mixer","gate_passed":true,"checkpoint_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","source_report_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","english_wer":0.4,"chinese_cer":0.4},{"stage":"block","internal_stage":"block","gate_passed":true,"checkpoint_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","source_report_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","english_wer":0.3,"chinese_cer":0.3},{"stage":"logits","internal_stage":"logits","gate_passed":true,"checkpoint_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","source_report_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","english_wer":0.2,"chinese_cer":0.2},{"stage":"sft","internal_stage":"sft","gate_passed":true,"checkpoint_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","source_report_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","english_wer":0.1,"chinese_cer":0.1}],"requested_alignment_language_metric_summaries":[{"name":"english_wer","initial_calibration_error_rate":0.5,"stages":{"rwkv_layer":0.4,"block":0.3,"logits":0.2,"sft":0.1}},{"name":"chinese_cer","initial_calibration_error_rate":0.5,"stages":{"rwkv_layer":0.4,"block":0.3,"logits":0.2,"sft":0.1}}],"requested_alignment_dataset_results":[{"initial_calibration":{"student_error_rate":0.5},"stages":\'"${requested_stage_metrics}"\'},{"initial_calibration":{"student_error_rate":0.5},"stages":\'"${requested_stage_metrics}"\'},{"initial_calibration":{"student_error_rate":0.5},"stages":\'"${requested_stage_metrics}"\'},{"initial_calibration":{"student_error_rate":0.5},"stages":\'"${requested_stage_metrics}"\'},{"initial_calibration":{"student_error_rate":0.5},"stages":\'"${requested_stage_metrics}"\'}]\'\n'
        'printf \'%s\\n\' \'{"pipeline":"stage211","artifact":"stepwise_final_results","complete":true,"gate_passed":true,"strict_stage_order":["calibration","mixer","block","logits","sft"],"requested_alignment_stage_order":["rwkv_layer","block","logits","sft"],"checkpoint_chain_passed":true,"nano_initialization_chain_passed":true,"nano_initialization_source_chain_passed":true,"ctc_label_normalization_chain_passed":true,"ctc_label_proof":{"full_length_index_audit_passed":true,"ctc_suppress_non_pronunciation_tokens":true,"ctc_suppressed_token_ids_count":2114,"ctc_suppressed_token_ids_sha256":"76a68d03bb2dc486c214fd44891f3e3e2c36286d79fea1e69c8e74d7767d5a09","teacher_projection_support_matches_student":true,"ctc_unk_tokens":0},"public_metric_definition_chain_passed":true,"public_metric_tokenizer_contract":"unicode_alnum_words_basic_cjk_chars_v1","public_metric_correction_receipt_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","public_metric_tokenizer_source_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","nano_teacher_chain_passed":true,"nano_public_baseline_provenance_passed":true,"supplemental_inventory_chain_passed":true,"supplemental_dedupe_proof":{"inventory_schema_version":2,"inventory_artifact":"stage211_supplemental_combined_inventory","mode":"source_identity_plus_known_corpus_exclusion","source_sets_disjoint":true,"content_fingerprint_complete":false,"base_public_overlap_normalized_pcm_exact_complete":true,"base_public_overlap_scan_order":"manifest_location_index_archive_order_v1","base_public_overlap_rows":0,"base_public_overlap_scanned_rows":1,"base_public_overlap_receipt_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","social_normalized_pcm_exact_complete":true,"social_public_overlap_mode":"normalized_pcm_exact","archived_social_exact_duplicate_exclusion_complete":true,"archived_social_unique_members":0,"archived_social_overlap_receipt_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","usb_top_level_classification_complete":true,"usb_natural_audio_resolution_complete":true,"usb_unresolved_natural_entries":[],"usb_top_level_coverage_receipt_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","near_duplicate_complete":false,"component_inventories":{"base_natural":{},"social_vad":{}},"known_overlap_exclusions":["llaso_gigaspeech","llaso_librispeech"]},"coverage_results":[{"stage":"mixer","training_segments":[{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3}]},{"stage":"block","training_segments":[{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3}]},{"stage":"logits","training_segments":[{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3},{"epochs":3}]},{"stage":"sft"}],"all_stage_alignment_results_complete":true,"alignment_results":[{"stage":"mixer","gate_passed":true,"stratified_gate_passed":true,"fixed_eval_samples":256,"stratified_samples":2304,"stratified_cells":["easy_en","easy_zh","medium_en","medium_zh","hard_en","hard_zh","long_zh","supplemental_en","supplemental_zh"],"source_report_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{"stage":"block","gate_passed":true,"stratified_gate_passed":true,"fixed_eval_samples":256,"stratified_samples":2304,"stratified_cells":["easy_en","easy_zh","medium_en","medium_zh","hard_en","hard_zh","long_zh","supplemental_en","supplemental_zh"],"source_report_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},{"stage":"logits","gate_passed":true,"stratified_gate_passed":true,"fixed_eval_samples":256,"stratified_samples":2304,"stratified_cells":["easy_en","easy_zh","medium_en","medium_zh","hard_en","hard_zh","long_zh","supplemental_en","supplemental_zh"],"source_report_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}],\'"${requested_proof}"\',\'"${dataset_proof}"\'}\' >"${output_json}"\n'
        'jq \'.public_metric_strip_language_confirmation = false\' "${output_json}" >"${output_json}.tmp"\n'
        'mv "${output_json}.tmp" "${output_json}"\n'
        'trajectory=\'{"gate_passed":true,"fixed_eval_samples":256,"source_order":["easy","medium","hard","long","supplemental_natural"],"terminal_entries":5,"best_prior_loss":0.1,"candidate_loss":0.105,"relative_regression_pct":5.0,"max_relative_regression_pct":10.0}\'\n'
        'phase_cadence=\'{"complete":true,"interval_steps":10000,"eval_samples_per_report":256,"source_order":["easy","medium","hard","long","supplemental_natural"],"source_count":5,"total_reports":5,"sources":[{"source":"easy","terminal_step":3,"report_count":1,"eval_samples_per_report":256},{"source":"medium","terminal_step":3,"report_count":1,"eval_samples_per_report":256},{"source":"hard","terminal_step":3,"report_count":1,"eval_samples_per_report":256},{"source":"long","terminal_step":3,"report_count":1,"eval_samples_per_report":256},{"source":"supplemental_natural","terminal_step":3,"report_count":1,"eval_samples_per_report":256}]}\'\n'
        'sft_cadence=\'{"complete":true,"interval_steps":2000,"eval_samples_per_report":256,"source_order":["labeled_sft"],"source_count":1,"total_reports":19,"sources":[{"source":"labeled_sft","terminal_step":37505,"report_count":19,"eval_samples_per_report":256}]}\'\n'
        'full_labeled_profile=\'{"labeled_profile_schema_version":2,'
        '"labeled_profile_receipt_path":"/proof/clean-profile.json",'
        '"labeled_profile_receipt_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",'
        '"all_accepted_unique_rows_required":true,"source_language_interleave_required":true,'
        '"train_samples":1418180,"eval_samples":7142,"total_samples":1425322,'
        '"unique_utterance_ids":1425322,"pronunciation_target_samples":1425322,'
        '"ctc_feasible_samples":1425322,"ctc_tokens":21711822,"ctc_unk_tokens":0,'
        '"ctc_forbidden_tokens":0,"source_counts":{"aishell3":63262,'
        '"commonvoice_cn":32712,"commonvoice_en":1109456,"librispeech":219892},'
        '"language_counts":{"en":1329348,"zh":95974},'
        '"public_clean_rebuild_passed":true,'
        '"public_clean_rebuild_receipt_sha256":"8888888888888888888888888888888888888888888888888888888888888888",'
        '"public_clean_source_profile_sha256":"7777777777777777777777777777777777777777777777777777777777777777",'
        '"public_clean_exclusion_reason":"public_evaluation_overlap_exact_encoded_audio",'
        '"public_clean_exclusions_rows":21,"public_clean_rejected_training_overlap_rows":21,'
        '"public_clean_rejected_internal_eval_overlap_rows":0,'
        '"public_audio_isolation_passed":true,'
        '"public_audio_overlap_receipt_sha256":"9999999999999999999999999999999999999999999999999999999999999999",'
        '"public_audio_comparison_mode":"exact_encoded_audio_bytes",'
        '"public_audio_size_prefilter_lossless_for_exact_bytes":true,'
        '"public_audio_scanned_rows":1425322,"public_audio_public_rows":52436,'
        '"public_audio_training_overlap_rows":0,"public_audio_internal_eval_overlap_rows":0,'
        '"public_audio_normalized_pcm_complete":false,'
        '"public_audio_near_duplicate_complete":false}\'\n'
        'sft_coverage=\'{"unique_or_train_rows":1418180,"evaluation_rows":7142,'
        '"hours_per_epoch":2421.973575,"epochs":1,"row_exposures":1418180,'
        '"hour_exposures":2421.973575,"steps":37505,'
        '"tail_padding_sample_exposures":552,"executed_sample_exposures":1418732}\'\n'
        'if [[ "${SFT_CORRECTION_MODE}" == corrected ]]; then\n'
        '  sft_correction=\'{"schema_version":1,"applied":true,"rounds":1,"unique_rows_per_round":191024,"row_exposures":191024,"hour_exposures":309.737461,"steps":5147,"tail_padding_sample_exposures":432,"executed_sample_exposures":191456,"language_row_exposures":{"en":95512,"zh":95512},"source_row_exposures":{"aishell3":62952,"commonvoice_cn":32560,"commonvoice_en":47756,"librispeech":47756},"full_sft_completion_path":"/proof/sft_complete.json","full_sft_completion_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","correction_profile_path":"/proof/correction_profile.json","correction_profile_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","initial_checkpoint_path":"/proof/full_sft.pt","initial_checkpoint_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","final_checkpoint_path":"/proof/corrected.pt","final_checkpoint_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","round_receipts":[{"round":1,"receipt_path":"/proof/round1.json","receipt_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","init_checkpoint_path":"/proof/full_sft.pt","init_checkpoint_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","completion_checkpoint_path":"/proof/corrected.pt","completion_checkpoint_sha256":"dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","row_exposures":191024,"hour_exposures":309.737461,"steps":5147,"tail_padding_sample_exposures":432,"executed_sample_exposures":191456}]}\'\n'
        "else\n"
        '  sft_correction=\'{"schema_version":1,"applied":false,"rounds":0,"unique_rows_per_round":0,"row_exposures":0,"hour_exposures":0,"steps":0,"tail_padding_sample_exposures":0,"executed_sample_exposures":0,"language_row_exposures":{},"source_row_exposures":{},"round_receipts":[]}\'\n'
        "fi\n"
        'jq -c --argjson trajectory "${trajectory}" --argjson phase_cadence "${phase_cadence}" --argjson sft_cadence "${sft_cadence}" --argjson full_labeled_profile "${full_labeled_profile}" --argjson sft_coverage "${sft_coverage}" --argjson sft_correction "${sft_correction}" \'.alignment_results |= map(. + {trajectory_retention: $trajectory, step_eval_cadence: $phase_cadence}) | (.coverage_results[] | select(.stage == "sft")).step_eval_cadence = $sft_cadence | (.coverage_results[] | select(.stage == "sft")) += ($sft_coverage + {correction_rounds:$sft_correction.rounds,correction_row_exposures:$sft_correction.row_exposures,correction_hour_exposures:$sft_correction.hour_exposures,correction_steps:$sft_correction.steps,correction_tail_padding_sample_exposures:$sft_correction.tail_padding_sample_exposures,correction_executed_sample_exposures:$sft_correction.executed_sample_exposures,effective_row_exposures:($sft_coverage.row_exposures + $sft_correction.row_exposures),effective_hour_exposures:($sft_coverage.hour_exposures + $sft_correction.hour_exposures),effective_steps:($sft_coverage.steps + $sft_correction.steps),effective_tail_padding_sample_exposures:($sft_coverage.tail_padding_sample_exposures + $sft_correction.tail_padding_sample_exposures),effective_executed_sample_exposures:($sft_coverage.executed_sample_exposures + $sft_correction.executed_sample_exposures),correction:$sft_correction}) | .sft_correction_evidence = $sft_correction | .ctc_label_proof += $full_labeled_profile | .requested_alignment_language_metric_summaries |= map(. + {nano_error_rate: 0.1}) | .initial_calibration_result += {nano_english_wer: 0.1, nano_chinese_cer: 0.1} | .initial_calibration_result.english_wer_gap_to_nano = (.initial_calibration_result.english_wer - .initial_calibration_result.nano_english_wer) | .initial_calibration_result.chinese_cer_gap_to_nano = (.initial_calibration_result.chinese_cer - .initial_calibration_result.nano_chinese_cer) | .requested_alignment_results |= map(. + {nano_english_wer: 0.1, nano_chinese_cer: 0.1} | .english_wer_gap_to_nano = (.english_wer - .nano_english_wer) | .chinese_cer_gap_to_nano = (.chinese_cer - .nano_chinese_cer))\' "${output_json}" >"${output_json}.tmp"\n'
        'mv "${output_json}.tmp" "${output_json}"\n'
        'jq -c \'(.coverage_results[] | select(.stage == "mixer" or .stage == "block" or .stage == "logits")) += {correction_rounds:3,correction_round_promotion:{schema_version:1,pipeline:"stage211",artifact:"correction_round_promotion_gate",correction_started:true,completed_rounds:3,guaranteed_rounds:3,gate_passed:true}}\' "${output_json}" >"${output_json}.tmp"\n'
        'mv "${output_json}.tmp" "${output_json}"\n'
        'if [[ "${UV_MODE}" == missing_alignment ]]; then\n'
        '  sed -i \'s/"all_stage_alignment_results_complete":true/"all_stage_alignment_results_complete":false/\' "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_trajectory ]]; then\n'
        '  jq \'del(.alignment_results[1].trajectory_retention)\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_periodic_cadence ]]; then\n'
        '  jq \'del(.coverage_results[] | select(.stage == "sft") | .step_eval_cadence)\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_phase_correction ]]; then\n'
        '  jq \'(.coverage_results[] | select(.stage == "block") | .correction_rounds) = 2\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_requested_nano_gap ]]; then\n'
        '  jq \'del(.requested_alignment_results[2].english_wer_gap_to_nano)\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_metric_prefix_contract ]]; then\n'
        '  jq \'del(.public_metric_strip_language_confirmation)\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == legacy_sft_labeled_profile ]]; then\n'
        '  jq \'.ctc_label_proof.total_samples = 285302\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_sft_public_isolation ]]; then\n'
        '  jq \'del(.ctc_label_proof.public_audio_isolation_passed)\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_sft_public_clean ]]; then\n'
        '  jq \'del(.ctc_label_proof.public_clean_rebuild_passed)\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_sft_correction_coverage ]]; then\n'
        '  jq \'.sft_correction_evidence = null\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == forged_sft_correction_exposure ]]; then\n'
        '  jq \'.sft_correction_evidence.row_exposures += 1\' "${output_json}" >"${output_json}.tmp"\n'
        '  mv "${output_json}.tmp" "${output_json}"\n'
        'elif [[ "${UV_MODE}" == missing_requested_alignment ]]; then\n'
        '  sed -i \'s/"all_requested_alignment_metrics_complete":true/"all_requested_alignment_metrics_complete":false/\' "${output_json}"\n'
        'elif [[ "${UV_MODE}" == reordered_requested_alignment ]]; then\n'
        '  sed -i \'s/"stage":"rwkv_layer","internal_stage":"mixer"/"stage":"block","internal_stage":"mixer"/\' "${output_json}"\n'
        'elif [[ "${UV_MODE}" == inconsistent_requested_alignment ]]; then\n'
        '  sed -i \'s/"english_wer":0.4,"chinese_cer":0.4/"english_wer":0.41,"chinese_cer":0.4/\' "${output_json}"\n'
        "fi\n"
        "printf '%s\\n' '# stepwise' >\"${output_markdown}\"\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)

    phase_gate_root = tmp_path / "gates"
    final_report_dir = "sft_corrected" if corrected_final_report else "sft"
    final_report = phase_gate_root / final_report_dir / "stage211_complete.json"
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
            "SFT_CORRECTION_MODE": ("corrected" if corrected_final_report else "uncorrected"),
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


def test_stage211_continuation_watcher_accepts_corrected_sft_proof(
    tmp_path: Path,
) -> None:
    result, tmux_calls, phase_gate_root = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="success",
        corrected_final_report=True,
    )

    assert result.returncode == 0, result.stderr
    assert "SFT and stepwise proofs pass" in result.stdout
    assert "new-session" not in tmux_calls
    assert (phase_gate_root / "sft_corrected" / "stage211_stepwise_results.json").is_file()


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


def test_stage211_continuation_watcher_requires_language_macro_metrics(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_language_macro",
    )

    assert result.returncode == 0, result.stderr
    assert "final Stage211 stepwise proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_direct_ctc_prefix_contract(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_metric_prefix_contract",
    )

    assert result.returncode == 0, result.stderr
    assert "final Stage211 stepwise proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_alignment_disclosure(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_alignment",
    )

    assert result.returncode == 0, result.stderr
    assert "final Stage211 stepwise proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_trajectory_retention(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_trajectory",
    )

    assert result.returncode == 0, result.stderr
    assert "trajectory-retention proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_periodic_fixed_eval(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_periodic_cadence",
    )

    assert result.returncode == 0, result.stderr
    assert "periodic fixed-eval proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_three_phase_correction_rounds(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_phase_correction",
    )

    assert result.returncode == 0, result.stderr
    assert "final Stage211 stepwise proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_requested_nano_gaps(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_requested_nano_gap",
    )

    assert result.returncode == 0, result.stderr
    assert "requested Nano-gap proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_rejects_legacy_sft_labeled_profile(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="legacy_sft_labeled_profile",
    )

    assert result.returncode == 0, result.stderr
    assert "full-labeled profile proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_sft_public_audio_isolation(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_sft_public_isolation",
    )

    assert result.returncode == 0, result.stderr
    assert "full-labeled profile proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


def test_stage211_continuation_watcher_requires_sft_public_clean_rebuild(
    tmp_path: Path,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode="missing_sft_public_clean",
    )

    assert result.returncode == 0, result.stderr
    assert "full-labeled profile proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


@pytest.mark.parametrize(
    "uv_mode",
    ("missing_sft_correction_coverage", "forged_sft_correction_exposure"),
)
def test_stage211_continuation_watcher_requires_exact_sft_correction_coverage(
    tmp_path: Path,
    uv_mode: str,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode=uv_mode,
        corrected_final_report=True,
    )

    assert result.returncode == 0, result.stderr
    assert "SFT correction-coverage proof failed validation" in result.stdout
    assert "START_STAGE=full" in result.stdout
    assert "new-session" in tmux_calls


@pytest.mark.parametrize(
    "uv_mode",
    (
        "missing_requested_alignment",
        "reordered_requested_alignment",
        "inconsistent_requested_alignment",
    ),
)
def test_stage211_continuation_watcher_requires_requested_stage_metrics(
    tmp_path: Path,
    uv_mode: str,
) -> None:
    result, tmux_calls, _ = _run_stage211_continuation_watcher_fixture(
        tmp_path,
        uv_mode=uv_mode,
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


def test_stage211_alignment_loss_nondivergence_uses_aggregate_fixed_eval_loss() -> None:
    improved = build_stage211_alignment_loss_nondivergence(
        baseline_source={"eval_loss": 0.325721},
        candidate_source={"eval_loss": 0.159513},
    )
    regressed = build_stage211_alignment_loss_nondivergence(
        baseline_source={"eval_loss": 1.0},
        candidate_source={"eval_loss": 1.21},
    )

    assert improved["candidate_to_baseline_ratio"] == pytest.approx(0.159513 / 0.325721)
    assert improved["maximum_ratio"] == pytest.approx(1.2)
    assert improved["gate_passed"] is True
    assert regressed["gate_passed"] is False


def test_stage211_mixer_phase_gate_accepts_coverage_nondivergence_policy(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step-final.pt"
    checkpoint.write_bytes(b"checkpoint")
    gate_path = _write_valid_phase_gate(
        tmp_path,
        phase="mixer",
        checkpoint=checkpoint,
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    alignment = json.loads(Path(gate["alignment_report"]["path"]).read_text(encoding="utf-8"))
    baseline = json.loads(Path(alignment["baseline_report_path"]).read_text(encoding="utf-8"))
    candidate = json.loads(Path(alignment["candidate_report_path"]).read_text(encoding="utf-8"))
    gate.update(
        {
            "promotion_policy": (STAGE211_PROMOTION_POLICY_COVERAGE_NON_DIVERGENT),
            "strict_metric_gate_passed": True,
            "alignment_loss_nondivergence": build_stage211_alignment_loss_nondivergence(
                baseline_source=baseline,
                candidate_source=candidate,
            ),
        }
    )
    gate_path.write_text(json.dumps(gate) + "\n", encoding="utf-8")

    validated = validate_stage211_phase_gate_report(
        gate_path,
        expected_phase="mixer",
        checkpoint_path=checkpoint,
    )

    assert validated["promotion_policy"] == (STAGE211_PROMOTION_POLICY_COVERAGE_NON_DIVERGENT)
    assert validated["alignment_loss_nondivergence"]["gate_passed"] is True


def test_stage211_finalizer_reuses_only_an_exact_immutable_alignment_pair(
    tmp_path: Path,
) -> None:
    baseline_checkpoint = tmp_path / "baseline.pt"
    candidate_checkpoint = tmp_path / "step-10.pt"
    train_config = tmp_path / "train.yaml"
    model_config = tmp_path / "model.yaml"
    nano_checkpoint = tmp_path / "nano.pt"
    manifest = tmp_path / "manifest.json"
    part = tmp_path / "part.jsonl"
    for path, payload in (
        (baseline_checkpoint, b"baseline"),
        (candidate_checkpoint, b"candidate"),
        (train_config, b"train"),
        (model_config, b"model"),
        (nano_checkpoint, b"nano"),
        (manifest, b"manifest"),
        (part, b"part"),
    ):
        path.write_bytes(payload)
    provenance = {
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
    pair_binding = {
        "schema_version": 1,
        "phase": "mixer",
        "baseline_checkpoint_path": str(baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
        "candidate_checkpoint_path": str(candidate_checkpoint.resolve()),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "train_config_path": str(train_config.resolve()),
        "train_config_sha256": sha256_file(train_config),
        "model_config_path": str(model_config.resolve()),
        "model_config_sha256": sha256_file(model_config),
        "nano_checkpoint_path": str(nano_checkpoint.resolve()),
        "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
        "eval_provenance": provenance,
        "feature_seed": 0,
        "samples": 256,
    }
    pair_id = hashlib.sha256(
        json.dumps(pair_binding, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    report_paths: dict[str, Path] = {}
    for role, checkpoint in (
        ("baseline", baseline_checkpoint),
        ("candidate", candidate_checkpoint),
    ):
        report = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "alignment_checkpoint_eval",
            "phase": "mixer",
            "role": role,
            "pair_eval_id": pair_id,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "train_config_path": str(train_config.resolve()),
            "train_config_sha256": sha256_file(train_config),
            "model_config_path": str(model_config.resolve()),
            "model_config_sha256": sha256_file(model_config),
            "nano_checkpoint_path": str(nano_checkpoint.resolve()),
            "nano_checkpoint_sha256": sha256_file(nano_checkpoint),
            "feature_seed": 0,
            "eval_samples": 256,
            "eval_loss": 1.0 if role == "baseline" else 0.5,
            "eval_provenance": provenance,
        }
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        report_paths[role] = path

    kwargs = {
        "phase": "mixer",
        "baseline_report_path": report_paths["baseline"],
        "candidate_report_path": report_paths["candidate"],
        "baseline_checkpoint": baseline_checkpoint,
        "candidate_checkpoint": candidate_checkpoint,
        "train_config_path": train_config,
        "model_config_path": model_config,
        "eval_bucket_manifest_path": manifest,
    }
    assert stage211_phase_finalizer._validate_existing_alignment_pair(**kwargs) is True

    candidate = json.loads(report_paths["candidate"].read_text(encoding="utf-8"))
    candidate["checkpoint_sha256"] = "0" * 64
    report_paths["candidate"].write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate report binding changed"):
        stage211_phase_finalizer._validate_existing_alignment_pair(**kwargs)
