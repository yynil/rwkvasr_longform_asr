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


def test_create_stage211_retention_correction_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    config = stage211_phase_train_config_contract("mixer")
    config.update(
        {
            "lr": correction.CORRECTION_LR,
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
            "stage211_post_coverage_replay_receipt_path": str(replay_receipt.resolve()),
            "stage211_post_coverage_admission_gate_path": str(admission_gate.resolve()),
            "stage211_post_coverage_original_coverage_unchanged": True,
        }
    )
    save_yaml(run_dir / "train_config.yaml", config)
    provenance = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_correction_run",
        "phase": "mixer",
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
        "learning_rate": correction.CORRECTION_LR,
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
        "nano_teacher_checkpoint_path": str(nano_checkpoint.resolve()),
        "nano_teacher_checkpoint_sha256": correction.sha256_file(nano_checkpoint),
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
    )

    assert receipt["artifact"] == "post_coverage_correction"
    assert receipt["round"] == 1
    assert receipt["rows"] == 8
    assert receipt["steps"] == 1
    assert receipt["learning_rate"] == correction.CORRECTION_LR
    assert receipt["nano_teacher_checkpoint_sha256"] == nano_sha256

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
        )
