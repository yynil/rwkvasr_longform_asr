import json
import shutil
from pathlib import Path

import pytest
import torch

from rwkvasr.cli.train_ctc_deepspeed import _resolve_deepspeed_train_config, build_parser
from rwkvasr.config import load_yaml
from rwkvasr.training.deepspeed_loop import (
    DeepSpeedTrainConfig,
    _maybe_load_initial_model_checkpoint,
    _build_deepspeed_optimizer,
    _normalize_deepspeed_config,
    _prune_deepspeed_step_checkpoint_artifacts,
    _resolve_max_steps as resolve_deepspeed_max_steps,
    _sample_direction_mask_distributed,
    _step_checkpoint_record_is_retained,
    train_ctc_model_deepspeed,
)
from rwkvasr.modules import DirectionDropoutConfig, DirectionDropoutScheduler, RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.checkpoint import save_checkpoint


def _write_manifest(tmp_path: Path, num_examples: int = 3, base_frames: int = 48) -> Path:
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx in range(num_examples):
            features = torch.randn(base_frames + idx, 80)
            feat_path = tmp_path / f"feat-{idx}.pt"
            torch.save(features, feat_path)
            handle.write(
                json.dumps(
                    {
                        "utt_id": f"utt-{idx}",
                        "feature_path": feat_path.name,
                        "token_ids": [1, 2, 3, 4],
                    }
                )
                + "\n"
            )
    return manifest_path


def test_deepspeed_cli_config_can_be_loaded_from_yaml_and_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ds.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {tmp_path / 'out'}",
                f"manifest_path: {tmp_path / 'manifest.jsonl'}",
                "batch_size: 8",
                "max_steps: 100",
                "device: cpu",
                "webdataset_utt_id_key: id",
                "deepspeed:",
                "  train_micro_batch_size_per_gpu: 8",
                "  gradient_accumulation_steps: 1",
                "  zero_optimization:",
                "    stage: 0",
            ]
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--config-yaml",
            str(config_path),
            "--batch-size",
            "2",
            "--max-steps",
            "4",
            "--webdataset-utt-id-key",
            "id",
        ]
    )
    resolved = _resolve_deepspeed_train_config(args)

    assert resolved == DeepSpeedTrainConfig(
        output_dir=str(tmp_path / "out"),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        batch_size=2,
        max_steps=4,
        device="cpu",
        webdataset_utt_id_key="id",
        deepspeed={
            "train_micro_batch_size_per_gpu": 8,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 0},
        },
    )


def test_deepspeed_cli_accepts_init_checkpoint_override(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ds.yaml"
    init_path = tmp_path / "init.pt"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {tmp_path / 'out'}",
                f"manifest_path: {tmp_path / 'manifest.jsonl'}",
                "device: cpu",
                "deepspeed:",
                "  train_micro_batch_size_per_gpu: 8",
                "  gradient_accumulation_steps: 1",
                "  zero_optimization:",
                "    stage: 0",
            ]
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--config-yaml",
            str(config_path),
            "--init-checkpoint-path",
            str(init_path),
        ]
    )
    resolved = _resolve_deepspeed_train_config(args)

    assert resolved.init_checkpoint_path == str(init_path)


def test_deepspeed_resolve_max_steps_supports_custom_utt_id_key(tmp_path: Path) -> None:
    index_path = tmp_path / "webdataset_index.json"
    index_path.write_text(
        json.dumps(
            {
                "num_samples": 6,
                "split": {
                    "eval_ratio": 0.25,
                    "hash_seed": 5,
                    "split_by": "shard_name",
                    "utt_id_key": "id",
                },
                "splits": {
                    "train": {"num_samples": 6},
                    "eval": {"num_samples": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    resolved_max_steps, steps_per_epoch = resolve_deepspeed_max_steps(
        DeepSpeedTrainConfig(
            output_dir=str(tmp_path / "out"),
            webdataset_root=str(tmp_path / "webdataset"),
            webdataset_index_path=str(index_path),
            webdataset_split="train",
            webdataset_eval_ratio=0.25,
            webdataset_hash_seed=5,
            webdataset_split_by="shard_name",
            webdataset_utt_id_key="id",
            batch_size=2,
            epochs=2,
            device="cpu",
            deepspeed={
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {"stage": 2},
            },
        ),
        grad_accum=1,
    )

    assert steps_per_epoch == 3
    assert resolved_max_steps == 6


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_train_ctc_model_deepspeed_smoke_single_process(tmp_path: Path) -> None:
    pytest.importorskip("deepspeed")
    if shutil.which("ninja") is None:
        pytest.skip("ninja is required for this DeepSpeed smoke test environment")
    manifest = _write_manifest(tmp_path)
    out_dir = tmp_path / "out_ds"

    result = train_ctc_model_deepspeed(
        DeepSpeedTrainConfig(
            output_dir=str(out_dir),
            manifest_path=str(manifest),
            vocab_size=8,
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            max_steps=1,
            save_every=1,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
            log_every=1,
            deepspeed={
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 1,
                "train_batch_size": 2,
                "gradient_clipping": 1.0,
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True,
                    },
                },
            },
        )
    )

    assert result["steps"] == 1
    assert (out_dir / "global_cmvn.json").exists()
    assert (out_dir / "model_config.yaml").exists()
    assert (out_dir / "train_config.yaml").exists()
    assert (out_dir / "deepspeed_config.yaml").exists()
    assert (out_dir / "step-1.pt").exists()
    assert (out_dir / "ds_checkpoints" / "step-1").exists()
    deepspeed_yaml = load_yaml(out_dir / "deepspeed_config.yaml")
    assert deepspeed_yaml["train_micro_batch_size_per_gpu"] == 2
    assert deepspeed_yaml["zero_optimization"]["stage"] == 2
    assert deepspeed_yaml["zero_optimization"]["offload_optimizer"]["device"] == "cpu"


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
def test_train_ctc_model_deepspeed_keeps_top_k_step_checkpoints(tmp_path: Path) -> None:
    pytest.importorskip("deepspeed")
    if shutil.which("ninja") is None:
        pytest.skip("ninja is required for this DeepSpeed smoke test environment")
    manifest = _write_manifest(tmp_path, num_examples=4, base_frames=40)
    out_dir = tmp_path / "out_ds_step_topk"

    result = train_ctc_model_deepspeed(
        DeepSpeedTrainConfig(
            output_dir=str(out_dir),
            manifest_path=str(manifest),
            vocab_size=8,
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            batch_size=2,
            max_steps=2,
            save_every=1,
            step_eval_samples=2,
            top_k_step_checkpoints=1,
            device="cpu",
            p_start=0.0,
            p_max=0.0,
            log_every=1,
            deepspeed={
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 1,
                "train_batch_size": 2,
                "gradient_clipping": 1.0,
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True,
                    },
                },
            },
        )
    )

    assert result["steps"] == 2
    step_metrics = load_yaml(out_dir / "step_checkpoint_metrics.yaml")
    assert len(step_metrics["step_checkpoints"]) == 2
    assert len(step_metrics["best"]) == 1
    remaining = sorted(path.name for path in out_dir.glob("step-*.pt"))
    assert len(remaining) == 1


def test_normalize_deepspeed_config_does_not_force_cpu_offload() -> None:
    config = DeepSpeedTrainConfig(
        output_dir="out",
        manifest_path="manifest.jsonl",
        batch_size=4,
        max_steps=1,
        device="cuda",
        deepspeed={
            "train_micro_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
            },
        },
    )

    normalized = _normalize_deepspeed_config(config)

    assert normalized["train_micro_batch_size_per_gpu"] == 4
    assert normalized["zero_optimization"]["stage"] == 2
    assert "offload_optimizer" not in normalized["zero_optimization"]


def test_build_deepspeed_optimizer_uses_adamw_when_offload_disabled() -> None:
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=64,
            dim_att=64,
            dim_ff=128,
            num_layers=1,
            vocab_size=8,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )
    config = DeepSpeedTrainConfig(
        output_dir="out",
        manifest_path="manifest.jsonl",
        batch_size=2,
        max_steps=1,
        device="cuda",
        deepspeed={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {
                "stage": 2,
            },
        },
    )
    normalized = _normalize_deepspeed_config(config)

    optimizer, optimizer_name = _build_deepspeed_optimizer(model, config, normalized)

    assert optimizer_name == "AdamW"
    assert isinstance(optimizer, torch.optim.AdamW)


def test_maybe_load_initial_model_checkpoint_loads_weights_only(tmp_path: Path) -> None:
    model_config = RWKVCTCModelConfig(
        input_dim=80,
        n_embd=64,
        dim_att=64,
        dim_ff=128,
        num_layers=1,
        vocab_size=8,
        head_size=32,
        conv_kernel_size=5,
        dropout=0.0,
        frontend_type="linear",
    )
    source_model = RWKVCTCModel(model_config)
    checkpoint_path = tmp_path / "init.pt"
    save_checkpoint(checkpoint_path, model=source_model, step=123)

    target_model = RWKVCTCModel(model_config)
    for parameter in target_model.parameters():
        parameter.data.zero_()

    restored = _maybe_load_initial_model_checkpoint(
        target_model,
        DeepSpeedTrainConfig(
            output_dir=str(tmp_path / "out"),
            manifest_path=str(tmp_path / "manifest.jsonl"),
            device="cpu",
            init_checkpoint_path=str(checkpoint_path),
            deepspeed={
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {"stage": 2},
            },
        ),
    )

    assert restored is not None
    assert restored["step"] == 123
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters(), strict=True):
        assert torch.equal(source_param, target_param)


def test_sample_direction_mask_distributed_falls_back_without_dist() -> None:
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=4,
            variant="drop_both",
            p_start=0.2,
            p_max=0.2,
            warmup_steps=0,
            ramp_steps=0,
        )
    )

    mask = _sample_direction_mask_distributed(
        scheduler,
        step=0,
        device=torch.device("cpu"),
    )

    assert mask.forward.dtype == torch.bool
    assert mask.backward.dtype == torch.bool
    assert mask.forward.shape == (4,)
    assert mask.backward.shape == (4,)
    assert torch.all(mask.forward | mask.backward)


def test_prune_deepspeed_step_checkpoint_artifacts_ignores_missing_paths(tmp_path: Path) -> None:
    kept_dir = tmp_path / "ds_checkpoints" / "step-2"
    kept_dir.mkdir(parents=True)
    kept_file = tmp_path / "step-2.pt"
    kept_file.write_text("keep", encoding="utf-8")

    removed_dir = tmp_path / "ds_checkpoints" / "step-1"
    removed_dir.mkdir(parents=True)
    (removed_dir / "meta.txt").write_text("x", encoding="utf-8")
    removed_file = tmp_path / "step-1.pt"
    removed_file.write_text("drop", encoding="utf-8")

    saved_records = [
        {
            "step": 1,
            "checkpoint_path": str(removed_file),
            "deepspeed_checkpoint_dir": str(removed_dir),
        },
        {
            "step": 2,
            "checkpoint_path": str(kept_file),
            "deepspeed_checkpoint_dir": str(kept_dir),
        },
    ]
    top_records = [saved_records[1]]

    _prune_deepspeed_step_checkpoint_artifacts(
        top_records=top_records,
        saved_records=saved_records,
    )
    _prune_deepspeed_step_checkpoint_artifacts(
        top_records=top_records,
        saved_records=saved_records,
    )

    assert not removed_file.exists()
    assert not removed_dir.exists()
    assert kept_file.exists()
    assert kept_dir.exists()


def test_step_checkpoint_record_is_retained_matches_by_file_or_dir() -> None:
    top_records = [
        {
            "checkpoint_path": "/tmp/step-2.pt",
            "deepspeed_checkpoint_dir": "/tmp/ds_checkpoints/step-2",
        }
    ]

    assert _step_checkpoint_record_is_retained(
        record={"checkpoint_path": "/tmp/step-2.pt"},
        top_records=top_records,
    )
    assert _step_checkpoint_record_is_retained(
        record={"deepspeed_checkpoint_dir": "/tmp/ds_checkpoints/step-2"},
        top_records=top_records,
    )
    assert not _step_checkpoint_record_is_retained(
        record={"checkpoint_path": "/tmp/step-3.pt"},
        top_records=top_records,
    )
