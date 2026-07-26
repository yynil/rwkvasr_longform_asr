from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
    STAGE211_AUDIO_TOTAL_HOURS,
    STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
    STAGE211_AUDIO_TOTAL_ROWS,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    STAGE211_PUBLIC_BENCHMARKS,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
stage211 = importlib.import_module("scripts.run_stage211_strict_chained_alignment")
stage211_phase_gate = importlib.import_module("scripts.create_stage211_phase_gate")


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
    formal_steps = 12_019 if phase.requires_labels and not smoke else None
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


@pytest.mark.parametrize("phase_name", tuple(stage211.PHASES))
def test_stage211_freezes_nano_non_attention_path_in_every_phase(
    tmp_path: Path,
    phase_name: str,
) -> None:
    config = _config_for_phase(tmp_path, phase_name)

    expected_steps = 12_019 if phase_name == "sft" else 30_064
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


def test_stage211_medium_curriculum_requires_receipt_and_uses_full_steps(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "easy-complete.pt"
    checkpoint.write_bytes(b"easy")
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
    assert "target_step=1003659" in admitted.stdout


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

    assert config["max_steps"] == 1_003_659
    assert config["batch_size"] == 36
    assert config["batch_token_budget"] == 24_000
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


def _write_valid_phase_gate(
    tmp_path: Path,
    *,
    phase: str,
    checkpoint: Path,
) -> Path:
    segments = []
    previous_checkpoint = tmp_path / "phase-init.pt"
    previous_checkpoint.write_bytes(b"phase-init")
    for index, (difficulty, expected) in enumerate(STAGE211_AUDIO_CURRICULUM.items()):
        manifest = tmp_path / f"{difficulty}-manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        provenance = tmp_path / f"{difficulty}-provenance.json"
        provenance.write_text("{}\n", encoding="utf-8")
        completion = checkpoint if difficulty == "long" else tmp_path / f"{difficulty}.pt"
        if completion != checkpoint:
            completion.write_bytes(f"checkpoint-{index}".encode())
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
            "rows": expected["rows"],
            "row_exposures": expected["rows"] * STAGE211_FULL_DATA_EPOCHS,
            "hours": expected["hours"],
            "hour_exposures": expected["hours"] * STAGE211_FULL_DATA_EPOCHS,
            "steps_per_epoch": expected["steps_per_epoch"],
            "steps": expected["steps"],
            "provenance_path": str(provenance.resolve()),
            "provenance_sha256": sha256_file(provenance),
            "bucket_manifest_path": str(manifest.resolve()),
            "bucket_manifest_sha256": sha256_file(manifest),
            "init_checkpoint_path": str(previous_checkpoint.resolve()),
            "init_checkpoint_sha256": sha256_file(previous_checkpoint),
            "completion_checkpoint_path": str(completion.resolve()),
            "completion_checkpoint_sha256": sha256_file(completion),
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
                "full_data_coverage": {
                    "phase": phase,
                    "complete": True,
                    "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
                    "total_hours": STAGE211_AUDIO_TOTAL_HOURS,
                    "total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
                    "total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
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

    validated = stage211._validate_promotion_receipt(
        receipt_path=receipt_path,
        target_phase="block",
        checkpoint_path=checkpoint,
    )
    assert validated["source_phase"] == "mixer"
    assert validated["target_phase"] == "block"

    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        stage211._validate_promotion_receipt(
            receipt_path=receipt_path,
            target_phase="block",
            checkpoint_path=checkpoint,
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
