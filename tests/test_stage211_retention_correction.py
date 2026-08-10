from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    stage211_phase_train_config_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
correction = importlib.import_module("scripts.create_stage211_retention_correction_receipt")
correction_runner = importlib.import_module("scripts.run_stage211_retention_correction")


def _write_manifest(root: Path) -> Path:
    train_part = root / "train.jsonl"
    eval_part = root / "eval.jsonl"
    root.mkdir(parents=True)
    train_part.write_text(
        "".join(
            json.dumps({"key": f"train-{index}", "num_frames": 100}) + "\n" for index in range(8)
        ),
        encoding="utf-8",
    )
    eval_part.write_text(
        "".join(
            json.dumps({"key": f"eval-{index}", "num_frames": 100}) + "\n" for index in range(256)
        ),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "root": "/",
                "source_length_index_path": "/dev/null",
                "bucket_width": 80,
                "entries_per_part": 100,
                "splits": {
                    "train": {
                        "num_samples": 8,
                        "buckets": [
                            {
                                "bucket_id": 1,
                                "num_samples": 8,
                                "parts": [{"path": str(train_part), "num_samples": 8}],
                            }
                        ],
                    },
                    "eval": {
                        "num_samples": 256,
                        "buckets": [
                            {
                                "bucket_id": 1,
                                "num_samples": 256,
                                "parts": [{"path": str(eval_part), "num_samples": 256}],
                            }
                        ],
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_retention_correction_smoke_marker_binds_round_inputs(
    tmp_path: Path,
) -> None:
    init_checkpoint = tmp_path / "init.pt"
    replay_receipt = tmp_path / "replay.json"
    replay_manifest = tmp_path / "manifest.json"
    admission_gate = tmp_path / "failed-gate.json"
    nano_checkpoint = tmp_path / "nano.pt"
    smoke_dir = tmp_path / "smoke"
    smoke_log_dir = smoke_dir / "logs"
    smoke_log_dir.mkdir(parents=True)
    smoke_checkpoint = smoke_dir / "step-2.pt"
    smoke_log = smoke_log_dir / "mixer_smoke_2steps.log"
    for path, content in (
        (init_checkpoint, b"init"),
        (replay_receipt, b"replay"),
        (replay_manifest, b"manifest"),
        (admission_gate, b"gate"),
        (nano_checkpoint, b"nano"),
    ):
        path.write_bytes(content)
    torch.save({"step": 2}, smoke_checkpoint)
    smoke_log.write_text(
        "[rwkvasr] Distributed init complete.\n"
        "[deepspeed-train] step=1 loss=0.8 peak_reserved=5.50GiB\n"
        "[deepspeed-train] step=2 loss=0.7 peak_reserved=6.00GiB\n",
        encoding="utf-8",
    )
    marker_path = tmp_path / "smoke-passed.json"
    marker_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "full_profile_smoke",
                "phase": "mixer",
                "complete": True,
                "init_checkpoint_path": str(init_checkpoint.resolve()),
                "init_checkpoint_sha256": correction.sha256_file(init_checkpoint),
                "easy_manifest_path": str(replay_manifest.resolve()),
                "easy_manifest_sha256": correction.sha256_file(replay_manifest),
                "smoke_checkpoint_path": str(smoke_checkpoint.resolve()),
                "smoke_checkpoint_sha256": correction.sha256_file(smoke_checkpoint),
                "smoke_log_path": str(smoke_log.resolve()),
                "smoke_log_sha256": correction.sha256_file(smoke_log),
                "peak_reserved_gib": 6.0,
                "max_peak_reserved_gib": 22.0,
                "correction_round": 1,
                "replay_receipt_path": str(replay_receipt.resolve()),
                "replay_receipt_sha256": correction.sha256_file(replay_receipt),
                "admission_gate_path": str(admission_gate.resolve()),
                "admission_gate_sha256": correction.sha256_file(admission_gate),
                "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": correction.sha256_file(nano_checkpoint),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    marker = correction_runner._validate_correction_smoke_marker(
        marker_path=marker_path,
        round_index=1,
        init_checkpoint=init_checkpoint,
        replay_receipt=replay_receipt,
        replay_manifest=replay_manifest,
        admission_gate=admission_gate,
        nano_checkpoint=nano_checkpoint,
    )
    assert marker["correction_round"] == 1

    admission_gate.write_bytes(b"changed-gate")
    with pytest.raises(ValueError, match="binding mismatch"):
        correction_runner._validate_correction_smoke_marker(
            marker_path=marker_path,
            round_index=1,
            init_checkpoint=init_checkpoint,
            replay_receipt=replay_receipt,
            replay_manifest=replay_manifest,
            admission_gate=admission_gate,
            nano_checkpoint=nano_checkpoint,
        )

    admission_gate.write_bytes(b"gate")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["peak_reserved_gib"] = 23.0
    marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rebuilt source evidence|memory contract mismatch"):
        correction_runner._validate_correction_smoke_marker(
            marker_path=marker_path,
            round_index=1,
            init_checkpoint=init_checkpoint,
            replay_receipt=replay_receipt,
            replay_manifest=replay_manifest,
            admission_gate=admission_gate,
            nano_checkpoint=nano_checkpoint,
        )


def test_retention_correction_resume_requires_smoke_marker(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="lacks its preflight smoke marker"):
        correction_runner._run_correction_smoke(
            round_index=1,
            run_dir=tmp_path / "formal",
            config_dir=tmp_path / "config",
            replay_receipt=tmp_path / "replay.json",
            replay_manifest=tmp_path / "manifest.json",
            admission_gate=tmp_path / "gate.json",
            init_checkpoint=tmp_path / "init.pt",
            nano_checkpoint=tmp_path / "nano.pt",
            audio_data_audit={},
            master_port=29641,
            max_peak_reserved_gib=22.0,
            formal_latest_step=1,
            dry_run=False,
        )


@pytest.mark.parametrize("phase", ("mixer", "block", "logits"))
def test_create_stage211_retention_correction_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    correction_lr = correction.stage211_post_coverage_correction_lr(phase)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _write_manifest(tmp_path / "replay")
    replay_receipt = tmp_path / "replay-receipt.json"
    replay_receipt.write_text("{}\n", encoding="utf-8")
    admission_gate = tmp_path / "failed-gate.json"
    admission_gate.write_text("{}\n", encoding="utf-8")
    init_checkpoint = tmp_path / "init.pt"
    completion_checkpoint = tmp_path / "step-1.pt"
    torch.save({"step": 0, "model": {"weight": torch.zeros(1)}}, init_checkpoint)
    torch.save({"step": 1, "model": {"weight": torch.ones(1)}}, completion_checkpoint)
    nano_dir = tmp_path / "nano"
    nano_dir.mkdir()
    nano_checkpoint = nano_dir / "model.pt"
    nano_checkpoint.write_bytes(b"nano")
    smoke_checkpoint = tmp_path / "smoke-step-2.pt"
    smoke_log = tmp_path / "smoke.log"
    smoke_checkpoint.write_bytes(b"smoke-checkpoint")
    smoke_log.write_bytes(b"smoke-log")
    smoke_marker = tmp_path / "round-01-smoke-passed.json"
    smoke_marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "full_profile_smoke",
                "phase": phase,
                "complete": True,
                "correction_round": 1,
                "init_checkpoint_path": str(init_checkpoint.resolve()),
                "init_checkpoint_sha256": correction.sha256_file(init_checkpoint),
                "easy_manifest_path": str(manifest.resolve()),
                "easy_manifest_sha256": correction.sha256_file(manifest),
                "replay_receipt_path": str(replay_receipt.resolve()),
                "replay_receipt_sha256": correction.sha256_file(replay_receipt),
                "admission_gate_path": str(admission_gate.resolve()),
                "admission_gate_sha256": correction.sha256_file(admission_gate),
                "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
                "nano_teacher_checkpoint_sha256": correction.sha256_file(nano_checkpoint),
                "smoke_checkpoint_path": str(smoke_checkpoint.resolve()),
                "smoke_checkpoint_sha256": correction.sha256_file(smoke_checkpoint),
                "smoke_log_path": str(smoke_log.resolve()),
                "smoke_log_sha256": correction.sha256_file(smoke_log),
                "peak_reserved_gib": 6.0,
                "max_peak_reserved_gib": 22.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = stage211_phase_train_config_contract(phase)
    config.update(
        {
            "lr": correction_lr,
            "max_steps": 1,
            "batch_size": 36,
            "batch_token_budget": 24_000,
            "length_bucket_frame_budget": 24_000,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
            "webdataset_bucket_manifest_path": str(manifest.resolve()),
            "webdataset_split": "train",
            "ctc_teacher_online_model_path": str(nano_dir.resolve()),
            "init_checkpoint_path": str(init_checkpoint.resolve()),
            "stage211_post_coverage_correction_round": 1,
            "stage211_post_coverage_correction_phase": phase,
            "stage211_post_coverage_replay_receipt_path": str(replay_receipt.resolve()),
            "stage211_post_coverage_admission_gate_path": str(admission_gate.resolve()),
            "stage211_post_coverage_original_coverage_unchanged": True,
            "stage211_post_coverage_smoke_marker_path": str(smoke_marker.resolve()),
            "stage211_post_coverage_smoke_marker_sha256": correction.sha256_file(smoke_marker),
        }
    )
    save_yaml(run_dir / "train_config.yaml", config)
    provenance = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_correction_run",
        "phase": phase,
        "round": 1,
        "run_dir": str(run_dir.resolve()),
        "replay_receipt_path": str(replay_receipt.resolve()),
        "replay_receipt_sha256": correction.sha256_file(replay_receipt),
        "admission_gate_path": str(admission_gate.resolve()),
        "admission_gate_sha256": correction.sha256_file(admission_gate),
        "init_checkpoint_path": str(init_checkpoint.resolve()),
        "init_checkpoint_sha256": correction.sha256_file(init_checkpoint),
        "replay_manifest_path": str(manifest.resolve()),
        "replay_manifest_sha256": correction.sha256_file(manifest),
        "epochs": correction.CORRECTION_EPOCHS,
        "steps_per_epoch": 1,
        "learning_rate": correction_lr,
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
        "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
        "nano_teacher_checkpoint_sha256": correction.sha256_file(nano_checkpoint),
        "smoke_marker_path": str(smoke_marker.resolve()),
        "smoke_marker_sha256": correction.sha256_file(smoke_marker),
    }
    (run_dir / "stage211_correction_provenance.json").write_text(
        json.dumps(provenance) + "\n",
        encoding="utf-8",
    )
    nano_sha256 = correction.sha256_file(nano_checkpoint)
    coverage_segments = [
        {"nano_teacher_checkpoint_sha256": nano_sha256} for _ in STAGE211_AUDIO_CURRICULUM
    ]
    gate_payload = {
        "gate_passed": False,
        "full_data_coverage": {"segments": coverage_segments},
    }
    monkeypatch.setattr(
        correction,
        "validate_retention_replay",
        lambda _: {
            "manifest_path": str(manifest.resolve()),
            "validated_unique_keys": 8,
            "total_hours": 0.01,
        },
    )
    monkeypatch.setattr(
        correction,
        "validate_stage211_phase_gate_report",
        lambda *args, **kwargs: gate_payload,
    )
    monkeypatch.setattr(
        correction,
        "audit_stage211_runtime_epoch_coverage",
        lambda **kwargs: {"complete": True, "epochs": kwargs["epochs"]},
    )
    monkeypatch.setattr(
        correction,
        "audit_stage211_checkpoint_delta",
        lambda **kwargs: {"complete": True},
    )

    receipt = correction.build_receipt(
        round_index=1,
        run_dir=run_dir,
        replay_receipt_path=replay_receipt,
        admission_gate_path=admission_gate,
        init_checkpoint_path=init_checkpoint,
        completion_checkpoint_path=completion_checkpoint,
        phase=phase,
    )

    assert receipt["artifact"] == "post_coverage_correction"
    assert receipt["round"] == 1
    assert receipt["rows"] == 8
    assert receipt["steps"] == 1
    assert receipt["phase"] == phase
    assert receipt["learning_rate"] == correction_lr
    assert receipt["nano_teacher_checkpoint_sha256"] == nano_sha256
    assert receipt["smoke_marker_sha256"] == correction.sha256_file(smoke_marker)

    config["init_checkpoint_path"] = None
    config["resume_from"] = "latest"
    save_yaml(run_dir / "train_config.yaml", config)
    resumed_receipt = correction.build_receipt(
        round_index=1,
        run_dir=run_dir,
        replay_receipt_path=replay_receipt,
        admission_gate_path=admission_gate,
        init_checkpoint_path=init_checkpoint,
        completion_checkpoint_path=completion_checkpoint,
        phase=phase,
    )
    assert resumed_receipt["train_config_sha256"] == correction.sha256_file(
        run_dir / "train_config.yaml"
    )

    gate_payload["gate_passed"] = True
    with pytest.raises(ValueError, match="explicitly failed admission gate"):
        correction.build_receipt(
            round_index=1,
            run_dir=run_dir,
            replay_receipt_path=replay_receipt,
            admission_gate_path=admission_gate,
            init_checkpoint_path=init_checkpoint,
            completion_checkpoint_path=completion_checkpoint,
            phase=phase,
        )
