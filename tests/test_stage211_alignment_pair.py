from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rwkvasr.cli import eval_online_ctc_distill
from rwkvasr.config import save_yaml
from rwkvasr.eval.stage211_gate import (
    stage211_phase_train_config_contract,
)
from rwkvasr.training.deepspeed_loop import DeepSpeedTrainConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
pair_eval = importlib.import_module(
    "scripts.evaluate_stage211_alignment_pair"
)


def test_stage211_pair_config_forces_one_deterministic_fixed_eval(
    tmp_path: Path,
) -> None:
    train_config = tmp_path / "train.yaml"
    payload = {
        **stage211_phase_train_config_contract("logits"),
        "output_dir": str(tmp_path / "run"),
        "deepspeed": {},
    }
    save_yaml(train_config, payload)

    config = pair_eval._load_phase_config(
        phase="logits",
        train_config_path=train_config,
        samples=256,
        batch_size=4,
        num_workers=2,
        feature_seed=7,
        device="cuda:0",
        teacher_device="cuda:0",
        audio_cache_dir=tmp_path / "cache",
    )

    assert config.step_eval_split == "eval"
    assert config.step_eval_samples == 256
    assert config.step_eval_shuffle is False
    assert config.step_eval_cache_batches is True
    assert config.step_eval_feature_seed == 7
    assert config.specaugment_enabled is False
    assert config.num_workers == 2


def test_stage211_pair_config_can_override_only_eval_manifest(
    tmp_path: Path,
) -> None:
    train_config = tmp_path / "train.yaml"
    original_manifest = tmp_path / "original.json"
    sidecar_manifest = tmp_path / "sidecar.json"
    payload = {
        **stage211_phase_train_config_contract("mixer"),
        "output_dir": str(tmp_path / "run"),
        "deepspeed": {},
        "webdataset_bucket_manifest_path": str(original_manifest),
        "webdataset_length_index_path": str(tmp_path / "lengths.jsonl"),
    }
    save_yaml(train_config, payload)

    config = pair_eval._load_phase_config(
        phase="mixer",
        train_config_path=train_config,
        samples=256,
        batch_size=4,
        num_workers=0,
        feature_seed=0,
        device="cpu",
        teacher_device="cpu",
        audio_cache_dir=tmp_path / "cache",
        eval_bucket_manifest_path=sidecar_manifest,
    )

    assert config.webdataset_bucket_manifest_path == str(sidecar_manifest.resolve())
    assert config.webdataset_length_index_path == payload["webdataset_length_index_path"]
    assert config.step_eval_split == "eval"


def test_stage211_pair_metric_validation_requires_complete_outputs() -> None:
    layers = {
        layer_id: {"loss": 0.5, "cosine": 0.8, "rms_ratio": 1.0}
        for layer_id in range(70)
    }
    pair_eval._validate_report_metrics(
        phase="block",
        eval_samples=256,
        layer_components={"block": layers},
        logit_metrics={},
        decoder_hidden_metrics={"loss": 0.2},
    )

    with pytest.raises(RuntimeError, match="exactly 70 layers"):
        pair_eval._validate_report_metrics(
            phase="mixer",
            eval_samples=256,
            layer_components={"mixer": {0: layers[0]}},
            logit_metrics={},
            decoder_hidden_metrics={},
        )


def test_offline_online_teacher_enables_requested_hidden_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_online_ctc_distill,
        "FunASRNanoCTCTopKOnlineTeacher",
        lambda config: config,
    )
    config = DeepSpeedTrainConfig(
        output_dir="unused",
        deepspeed={},
        ctc_teacher_online_model_path="/models/nano",
        ctc_teacher_online_layer_mixer_loss_weight=1.0,
        ctc_teacher_online_decoder_hidden_loss_weight=0.5,
        ctc_teacher_online_conditional_nonblank_loss_weight=1.0,
        ctc_teacher_online_keep_layer_hiddens_on_device=True,
        ctc_teacher_online_keep_full_log_probs_on_device=True,
    )

    teacher_config = eval_online_ctc_distill._build_online_teacher(
        config=config,
        model_config=SimpleNamespace(ctc_vocab_size=60_516),
        device=torch.device("cpu"),
        audio_cache_dir="/tmp/unused",
    )

    assert teacher_config.return_layer_hiddens is True
    assert teacher_config.return_ctc_decoder_hiddens is True
    assert teacher_config.return_full_log_probs is True
    assert teacher_config.keep_layer_hiddens_on_device is True
    assert teacher_config.keep_full_log_probs_on_device is True


def test_stage211_pair_eval_reuses_one_materialized_batch_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_config = tmp_path / "train.yaml"
    model_config = tmp_path / "model.yaml"
    manifest = tmp_path / "manifest.json"
    nano_checkpoint = tmp_path / "nano.pt"
    baseline_checkpoint = tmp_path / "init.pt"
    candidate_checkpoint = tmp_path / "step-5.pt"
    for path, payload in (
        (train_config, "{}\n"),
        (model_config, "{}\n"),
        (manifest, "{}\n"),
    ):
        path.write_text(payload, encoding="utf-8")
    nano_checkpoint.write_bytes(b"nano")
    baseline_checkpoint.write_bytes(b"baseline")
    candidate_checkpoint.write_bytes(b"candidate")

    config = SimpleNamespace(
        webdataset_bucket_manifest_path=str(manifest),
        eval_mode="bi",
    )
    cached_batches = [object()]
    observed_batches: list[list[object]] = []
    monkeypatch.setattr(pair_eval, "_load_phase_config", lambda **kwargs: config)
    monkeypatch.setattr(
        pair_eval,
        "RWKVCTCModelConfig",
        lambda **kwargs: SimpleNamespace(frontend_type="sensevoice_rwkv"),
    )
    monkeypatch.setattr(
        pair_eval,
        "_resolve_student_dtype",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pair_eval,
        "_build_eval_loader",
        lambda *args, **kwargs: (object(), None),
    )
    monkeypatch.setattr(
        pair_eval,
        "_materialize_step_eval_batches",
        lambda *args, **kwargs: (cached_batches, 256),
    )
    provenance = {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": 256,
        "feature_seed": 0,
        "bucket_manifest_path": str(manifest.resolve()),
        "bucket_manifest_sha256": pair_eval.sha256_file(manifest),
        "split_samples": 256,
        "parts": [],
    }
    monkeypatch.setattr(
        pair_eval,
        "_build_step_eval_provenance",
        lambda **kwargs: provenance,
    )
    monkeypatch.setattr(
        pair_eval,
        "resolve_stage211_nano_teacher_checkpoint",
        lambda config: nano_checkpoint.resolve(),
    )
    monkeypatch.setattr(
        pair_eval,
        "_build_online_teacher",
        lambda **kwargs: object(),
    )

    def fake_evaluate_checkpoint(**kwargs):
        observed_batches.append(kwargs["cached_batches"])
        role = str(kwargs["role"])
        return {
            "role": role,
            "pair_eval_id": pair_eval._pair_id(kwargs["pair_binding"]),
            "eval_samples": 256,
            "eval_loss": 1.0 if role == "baseline" else 0.5,
        }

    monkeypatch.setattr(
        pair_eval,
        "_evaluate_checkpoint",
        fake_evaluate_checkpoint,
    )
    baseline_output = tmp_path / "eval" / "baseline.json"
    candidate_output = tmp_path / "eval" / "candidate.json"
    args = SimpleNamespace(
        phase="mixer",
        samples=256,
        feature_seed=0,
        batch_size=4,
        num_workers=0,
        train_config=train_config,
        model_config=model_config,
        baseline_checkpoint=baseline_checkpoint,
        candidate_checkpoint=candidate_checkpoint,
        baseline_output=baseline_output,
        candidate_output=candidate_output,
        audio_cache_dir=None,
        device="cpu",
        teacher_device=None,
        student_dtype="fp32",
    )

    baseline, candidate = pair_eval.evaluate_pair(args)

    assert observed_batches == [cached_batches, cached_batches]
    assert observed_batches[0] is observed_batches[1]
    assert baseline["pair_eval_id"] == candidate["pair_eval_id"]
    assert baseline_output.is_file()
    assert candidate_output.is_file()


def test_stage211_pair_eval_rejects_nonzero_feature_seed_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pair_eval,
        "_load_phase_config",
        lambda **kwargs: pytest.fail("phase config must not load"),
    )
    args = SimpleNamespace(
        phase="mixer",
        samples=256,
        feature_seed=1,
        batch_size=4,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="feature-seed=0"):
        pair_eval.evaluate_pair(args)
