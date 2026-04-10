import json
import shutil
from pathlib import Path

import pytest
import torch

from rwkvasr.cli.train_ctc_deepspeed import _resolve_deepspeed_train_config, build_parser
from rwkvasr.config import load_yaml
from rwkvasr.training.deepspeed_loop import (
    DeepSpeedTrainConfig,
    _build_deepspeed_optimizer,
    _normalize_deepspeed_config,
    _resolve_max_steps as resolve_deepspeed_max_steps,
    train_ctc_model_deepspeed,
)
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig


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
