from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_ALLOWED_OPERATOR_KEY_MARKERS,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES,
    STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
    STAGE211_AUDIO_TOTAL_HOURS,
    STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
    STAGE211_AUDIO_TOTAL_ROWS,
    STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
    stage211_phase_train_config_contract,
    validate_stage211_phase_train_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage211 = importlib.import_module("scripts.run_stage211_strict_chained_alignment")
stage211_phase_gate = importlib.import_module("scripts.create_stage211_phase_gate")
stage211_full_phase = importlib.import_module("scripts.run_stage211_full_phase_curriculum")
stage211_phase_finalizer = importlib.import_module("scripts.finalize_stage211_phase")
stage211_calibration_eval = importlib.import_module("scripts.validate_stage211_calibration_eval")


def test_stage211_controllers_preserve_virtualenv_python() -> None:
    expected = Path(sys.executable)

    assert stage211_full_phase.PYTHON == expected
    assert stage211_phase_finalizer.PYTHON == expected


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
        '--baseline-public-comparison-report '
        '"${CALIBRATION_EVAL_DIR}/public/nano_comparison.json"' in script
    )
    assert (
        '--baseline-public-comparison-report '
        '"${PHASE_GATE_ROOT}/mixer/nano_comparison.json"' in script
    )
    assert (
        '--baseline-public-comparison-report '
        '"${PHASE_GATE_ROOT}/logits/nano_comparison.json"' in script
    )
    assert '--calibration-reuse-receipt "${CALIBRATION_REUSE_RECEIPT}"' in script

    main_body = script[script.index("main() {") :]
    phase_calls = (
        "run_full_mixer_phase",
        "run_full_block_phase",
        "run_full_logits_phase",
        "run_labeled_sft_phase",
    )
    phase_offsets = [main_body.index(call) for call in phase_calls]
    assert phase_offsets == sorted(phase_offsets)


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
    assert stage211.NANO_CHECKPOINT == (REPO_ROOT / "assets" / "fun-asr-nano-2512" / "model.pt")
    assert stage211.NANO_CHECKPOINT.is_file()


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
    assert result.stdout.count("[stage211-full-phase] command=") == 5
    assert "--smoke" in result.stdout
    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        assert f"/{difficulty}/step-{expected['steps']}.pt" in result.stdout


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
        "[rwkvasr] Distributed init complete. world_size=4\n"
        + valid_log,
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
    assert config["ctc_teacher_online_layer_ffn_loss_weight"] == 0.0
    assert config["step_eval_cache_batches"] is True
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
                "nano_teacher_checkpoint_path": str(
                    nano_checkpoint.resolve()
                ),
                "nano_teacher_checkpoint_sha256": sha256_file(
                    nano_checkpoint
                ),
                "runtime_epoch_coverage": _write_runtime_epoch_coverage(
                    tmp_path,
                    prefix="medium-admission-easy",
                    epochs=STAGE211_FULL_DATA_EPOCHS,
                    steps_per_epoch=int(
                        STAGE211_AUDIO_CURRICULUM["easy"][
                            "steps_per_epoch"
                        ]
                    ),
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
    assert audit["estimated_train_steps"] == 1
    assert audit["tail_padding_samples_per_epoch"] == 47
    assert audit["executed_sample_exposures"] == 48


def _write_valid_phase_gate(
    tmp_path: Path,
    *,
    phase: str,
    checkpoint: Path,
) -> Path:
    segments = []
    nano_teacher_dir = tmp_path / "nano-teacher"
    nano_teacher_dir.mkdir()
    nano_teacher_checkpoint = nano_teacher_dir / "model.pt"
    nano_teacher_checkpoint.write_bytes(b"nano-teacher")
    previous_checkpoint = tmp_path / "phase-init.pt"
    previous_checkpoint.write_bytes(b"phase-init")
    for index, (difficulty, expected) in enumerate(STAGE211_AUDIO_CURRICULUM.items()):
        manifest = tmp_path / f"{difficulty}-manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        provenance = tmp_path / f"{difficulty}-provenance.json"
        provenance.write_text("{}\n", encoding="utf-8")
        train_config = tmp_path / f"{difficulty}-train-config.yaml"
        train_config_payload = stage211_phase_train_config_contract(phase)
        train_config_payload["ctc_teacher_online_model_path"] = str(
            nano_teacher_dir.resolve()
        )
        save_yaml(train_config, train_config_payload)
        completion = checkpoint if difficulty == "long" else tmp_path / f"{difficulty}.pt"
        if completion != checkpoint:
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

    public_results = []
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        manifest = tmp_path / f"{dataset}.jsonl"
        nano = tmp_path / f"{dataset}.nano.jsonl"
        student = tmp_path / f"{dataset}.student.jsonl"
        for path in (manifest, nano, student):
            path.write_text("{}\n", encoding="utf-8")
        public_results.append(
            {
                "dataset": dataset,
                "language": expected["language"],
                "metric": expected["metric"],
                "sample_count": expected["samples"],
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
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
                "full_data_coverage": {
                    "phase": phase,
                    "complete": True,
                    "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
                    "total_hours": STAGE211_AUDIO_TOTAL_HOURS,
                    "total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
                    "total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
                    "total_tail_padding_sample_exposures": (
                        STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
                    ),
                    "total_executed_sample_exposures": (
                        STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
                    ),
                    "segments": segments,
                    "final_checkpoint_path": str(checkpoint.resolve()),
                    "final_checkpoint_sha256": sha256_file(checkpoint),
                },
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
        gate_payload["full_data_coverage"]["segments"][0][
            "nano_teacher_checkpoint_path"
        ]
    )
    assert receipt["nano_teacher_checkpoint_sha256"] == sha256_file(
        nano_checkpoint
    )

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
    train_config_payload["ctc_teacher_online_model_path"] = str(
        alternate_dir.resolve()
    )
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
    report["full_data_coverage"]["segments"][0]["runtime_epoch_coverage"][
        "records"
    ][0]["completed_epoch_batch_count"] -= 1
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
