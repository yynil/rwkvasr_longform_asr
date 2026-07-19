import json
import inspect
import shutil
from pathlib import Path

import pytest
import torch

from rwkvasr.cli.train_ctc_deepspeed import _resolve_deepspeed_train_config, build_parser
from rwkvasr.config import load_yaml
from rwkvasr.training.deepspeed_loop import (
    DeepSpeedTrainConfig,
    _accumulate_layer_eval_metrics,
    _ctc_teacher_layer_hidden_loss,
    _finalize_layer_eval_metrics,
    _maybe_load_initial_model_checkpoint,
    _build_deepspeed_optimizer,
    _normalize_deepspeed_config,
    _prune_deepspeed_step_checkpoint_artifacts,
    _resolve_ctc_teacher_online_device,
    _resolve_max_steps as resolve_deepspeed_max_steps,
    _save_export_checkpoints,
    _select_eval_layer_hidden_ids,
    _sample_direction_mask_distributed,
    _select_layer_hidden_ids,
    _step_checkpoint_record_is_retained,
    _teacher_forced_student_layer_hiddens,
    _teacher_layer_capture_ids,
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


def test_online_ctc_teacher_device_uses_cuda_zero_for_single_process_debug() -> None:
    assert _resolve_ctc_teacher_online_device(None, torch.device("cuda"), local_rank=-1) == "cuda:0"
    assert _resolve_ctc_teacher_online_device(None, torch.device("cuda", 2), local_rank=2) == "cuda:2"
    assert _resolve_ctc_teacher_online_device("cuda:1", torch.device("cuda"), local_rank=-1) == "cuda:1"
    assert _resolve_ctc_teacher_online_device(None, torch.device("cpu"), local_rank=-1) == "cpu"


def test_deepspeed_loop_leaves_gradient_accumulation_to_engine() -> None:
    source = inspect.getsource(train_ctc_model_deepspeed)

    assert "engine.zero_grad()" not in source


def test_layer_hidden_sampler_keeps_boundaries_and_covers_every_layer() -> None:
    selections = [
        _select_layer_hidden_ids(
            step=step,
            num_layers=70,
            sample_count=8,
            boundary_ids=(0, 49, 50, 69),
        )
        for step in range(70)
    ]
    boundary_selection = _select_layer_hidden_ids(
        step=0,
        num_layers=70,
        sample_count=8,
        boundary_ids=(0, 49, 50, 69),
        include_boundaries=True,
    )

    assert all(len(selection) == 8 for selection in selections)
    assert set().union(*map(set, selections)) == set(range(70))
    assert {0, 49, 50, 69}.issubset(boundary_selection)


def test_fixed_eval_layer_sampler_covers_layers_uniformly_across_ranks() -> None:
    selections = [
        _select_eval_layer_hidden_ids(
            batch_index=batch_index,
            num_layers=70,
            sample_count=8,
            rank=rank,
            world_size=4,
        )
        for batch_index in range(16)
        for rank in range(4)
    ]
    counts = {
        layer_id: sum(layer_id in selection for selection in selections)
        for layer_id in range(70)
    }

    assert all(len(selection) == 8 for selection in selections)
    assert set().union(*map(set, selections)) == set(range(70))
    assert min(counts.values()) >= 7
    assert max(counts.values()) <= 8


def test_fixed_eval_layer_sampler_rejects_invalid_distributed_coordinates() -> None:
    with pytest.raises(ValueError, match="world_size must be positive"):
        _select_eval_layer_hidden_ids(
            batch_index=0,
            num_layers=70,
            sample_count=8,
            rank=0,
            world_size=0,
        )
    with pytest.raises(ValueError, match="rank must be in"):
        _select_eval_layer_hidden_ids(
            batch_index=0,
            num_layers=70,
            sample_count=8,
            rank=4,
            world_size=4,
        )


def test_teacher_forced_layer_alignment_uses_teacher_inputs_and_layer_zero_v_first() -> None:
    torch.manual_seed(2703)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=64,
            encoder_output_dim=64,
            dim_att=64,
            dim_ff=128,
            num_layers=3,
            vocab_size=16,
            head_size=32,
            dropout=0.0,
            frontend_type="sensevoice_rwkv",
            sensevoice_tp_blocks=1,
        )
    )
    records: dict[str, dict[str, object]] = {}
    for utt_id, length in (("utt-a", 5), ("utt-b", 3)):
        records[utt_id] = {
            "encoder_layer_hiddens": {
                "0": {"input": torch.randn(length, 80)},
                "1": {"input": torch.randn(length, 64)},
                "2": {"input": torch.randn(length, 64)},
            }
        }

    selected = (1, 2)
    assert _teacher_layer_capture_ids(selected, input_mode="teacher_forced") == (0, 1, 2)
    outputs, lengths = _teacher_forced_student_layer_hiddens(
        model,
        records,
        ("utt-a", "utt-b"),
        layer_ids=selected,
        missing_policy="error",
    )

    assert torch.equal(lengths, torch.tensor([5, 3]))
    assert set(outputs) == {1, 2}
    for components in outputs.values():
        assert set(components) == {"mixer", "ffn", "block"}
        assert components["mixer"].shape == (2, 5, 64)
    sum(component.sum() for components in outputs.values() for component in components.values()).backward()
    encoder = model.encoder.sensevoice_encoder
    assert encoder.layers[0].time_mixer.forward_mixer.value.weight.grad is not None
    assert encoder.layers[1].time_mixer.forward_mixer.value.weight.grad is not None


def test_layer_hidden_loss_matches_identical_sampled_components() -> None:
    student_hiddens = {
        layer_id: {
            component: torch.randn(2, 4, 6, requires_grad=True)
            for component in ("mixer", "ffn", "block")
        }
        for layer_id in (0, 2)
    }
    records: dict[str, dict[str, object]] = {}
    for sample_idx, (utt_id, length) in enumerate((("utt-a", 4), ("utt-b", 3))):
        records[utt_id] = {
            "encoder_layer_hiddens": {
                str(layer_id): {
                    component: student_hiddens[layer_id][component][sample_idx, :length].detach().clone()
                    for component in ("mixer", "ffn", "block")
                }
                for layer_id in (0, 2)
            }
        }

    result = _ctc_teacher_layer_hidden_loss(
        student_hiddens,
        torch.tensor([4, 3]),
        ["utt-a", "utt-b"],
        records,
        layer_ids=(0, 2),
        component_weights={"mixer": 1.0, "ffn": 0.0, "block": 0.5},
        normalized_mse_weight=1.0,
        cosine_weight=0.25,
        energy_mse_weight=0.5,
        log_rms_weight=0.1,
        raw_mse_weight=0.1,
        frame_tolerance=0,
        missing_policy="error",
    )

    assert result.loss.item() == pytest.approx(0.0, abs=1e-6)
    assert result.matched_samples == 2
    assert result.missing_samples == 0
    assert result.events == 8
    assert result.max_frame_delta == 0
    result.loss.backward()
    assert student_hiddens[0]["mixer"].grad is not None


def test_layer_hidden_energy_mse_is_bounded_and_detects_scale_mismatch() -> None:
    teacher = torch.randn(1, 3, 8)
    student = (teacher * 10.0).requires_grad_()
    result = _ctc_teacher_layer_hidden_loss(
        {0: {"mixer": student}},
        torch.tensor([3]),
        ["utt-a"],
        {
            "utt-a": {
                "encoder_layer_hiddens": {
                    "0": {"mixer": teacher[0]},
                }
            }
        },
        layer_ids=(0,),
        component_weights={"mixer": 1.0},
        normalized_mse_weight=0.0,
        cosine_weight=0.0,
        energy_mse_weight=1.0,
        log_rms_weight=0.0,
        raw_mse_weight=0.0,
        frame_tolerance=0,
        missing_policy="error",
    )

    assert result.loss.item() == pytest.approx(162.0 / 101.0, rel=1.0e-4)
    assert 0.0 < result.component_energy_mse["mixer"] <= 4.0
    assert result.component_rms_ratio["mixer"] == pytest.approx(10.0, rel=1.0e-5)
    assert result.component_log_rms["mixer"] > 0.0
    assert result.component_cosine["mixer"] == pytest.approx(1.0, abs=1.0e-5)
    accumulator = torch.zeros((2, 8), dtype=torch.float64)
    _accumulate_layer_eval_metrics(accumulator, result)
    _accumulate_layer_eval_metrics(accumulator, result)
    layer_metrics = _finalize_layer_eval_metrics(accumulator, device=torch.device("cpu"))
    assert set(layer_metrics) == {0}
    assert layer_metrics[0]["loss"] == pytest.approx(result.layer_losses[0].item())
    assert layer_metrics[0]["energy_mse"] == pytest.approx(result.layer_energy_mse[0])
    assert layer_metrics[0]["cosine"] == pytest.approx(1.0, abs=1.0e-5)
    assert layer_metrics[0]["rms_ratio"] == pytest.approx(10.0, rel=1.0e-5)
    result.loss.backward()
    assert student.grad is not None


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


def test_deepspeed_cli_accepts_decoder_text_token_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "train_ds.yaml"
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
            "--decoder-text-token-budget",
            "4096",
        ]
    )
    resolved = _resolve_deepspeed_train_config(args)

    assert resolved.decoder_text_token_budget == 4096


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


def test_deepspeed_resolve_max_steps_uses_bucket_manifest_without_webdataset_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    bucket_manifest = tmp_path / "buckets" / "manifest.json"
    bucket_manifest.parent.mkdir(parents=True)
    bucket_manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": "/",
                "source_length_index_path": str(tmp_path / "lengths.jsonl"),
                "bucket_width": 200,
                "entries_per_part": 100,
                "splits": {
                    "train": {
                        "num_samples": 48,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "num_samples": 48,
                                "parts": [],
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    resolved_max_steps, steps_per_epoch = resolve_deepspeed_max_steps(
        DeepSpeedTrainConfig(
            output_dir=str(tmp_path / "out"),
            deepspeed={"gradient_accumulation_steps": 1},
            webdataset_root="/",
            webdataset_bucket_manifest_path=str(bucket_manifest),
            webdataset_length_index_path=str(tmp_path / "lengths.jsonl"),
            webdataset_split="train",
            batch_size=3,
            epochs=2,
        ),
        grad_accum=1,
    )

    assert steps_per_epoch == 4
    assert resolved_max_steps == 8


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
                tokenizer_type="synthetic",
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
                tokenizer_type="synthetic",
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


def test_normalize_deepspeed_config_respects_explicit_bf16_false() -> None:
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
                "stage": 1,
            },
            "bf16": {
                "enabled": False,
            },
        },
    )

    normalized = _normalize_deepspeed_config(config)

    assert normalized["bf16"] == {"enabled": False}
    assert normalized["fp16"] == {"enabled": False}


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


def test_save_export_checkpoints_can_skip_deepspeed_shards(tmp_path: Path) -> None:
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

    class FakeEngine:
        module = model

        def save_checkpoint(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            raise AssertionError("DeepSpeed sharded checkpoint should be skipped")

    saved = _save_export_checkpoints(
        engine=FakeEngine(),
        output_dir=tmp_path,
        tag="step-1",
        export_name="step-1.pt",
        step=1,
        zero_stage=1,
        extra_state={"epoch": 1, "epoch_batch_offset": 0},
        save_deepspeed_sharded=False,
    )

    assert saved["checkpoint_path"] == str(tmp_path / "step-1.pt")
    assert saved["deepspeed_checkpoint_dir"] is None
    assert (tmp_path / "step-1.pt").exists()
    assert not (tmp_path / "ds_checkpoints").exists()
    latest = load_yaml(tmp_path / "latest_checkpoint.yaml")
    assert latest["checkpoint_type"] == "export"
    assert latest["checkpoint_path"] == str(tmp_path / "step-1.pt")
    assert "deepspeed_checkpoint_dir" not in latest
