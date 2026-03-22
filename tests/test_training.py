import torch

from rwkvasr.modules import (
    DirectionDropoutConfig,
    DirectionDropoutScheduler,
    RWKVCTCModel,
    RWKVCTCModelConfig,
)
from rwkvasr.training import (
    CTCBatch,
    RWKVDualModeCTCTrainer,
    build_rwkv_param_groups,
)
from rwkvasr.training.synthetic import (
    SyntheticOverfitConfig,
    make_synthetic_ctc_batch,
    run_synthetic_overfit,
)


def _build_model() -> RWKVCTCModel:
    return RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=80,
            n_embd=128,
            dim_att=128,
            dim_ff=256,
            num_layers=2,
            vocab_size=24,
            head_size=32,
            conv_kernel_size=5,
            dropout=0.0,
            frontend_type="linear",
        )
    )


def test_rwkv_optimizer_param_groups_respect_w0_and_weight_decay() -> None:
    model = _build_model()
    groups = build_rwkv_param_groups(model, lr=1e-3, weight_decay=0.1)
    names = {group["name"]: set(group["param_names"]) for group in groups}

    w0_names = names["rwkv_2x"]
    assert any(name.endswith(".w0") for name in w0_names)
    assert "encoder.blocks.0.time_mixer.forward_mixer.w0" in w0_names

    decay_names = names["rwkv_decay"]
    assert "ctc_head.weight" in decay_names
    assert "encoder.blocks.0.ffn1_norm.weight" not in decay_names


def test_dual_mode_trainer_returns_valid_training_mask() -> None:
    model = _build_model()
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=model.config.num_layers,
            variant="drop_both",
            p_start=1.0,
            p_max=1.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )
    trainer = RWKVDualModeCTCTrainer(model, direction_scheduler=scheduler)
    batch = CTCBatch(
        features=torch.randn(2, 9, 80),
        feature_lengths=torch.tensor([9, 9], dtype=torch.long),
        targets=torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.long),
        target_lengths=torch.tensor([3, 3], dtype=torch.long),
    )

    loss, mask = trainer.training_loss(
        batch,
        step=0,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.isfinite(loss)
    assert mask.num_layers == model.config.num_layers
    assert torch.all(mask.forward | mask.backward)


def test_ctc_batch_to_can_cast_feature_dtype() -> None:
    batch = CTCBatch(
        features=torch.randn(2, 9, 80),
        feature_lengths=torch.tensor([9, 7], dtype=torch.long),
        targets=torch.tensor([1, 2, 3], dtype=torch.long),
        target_lengths=torch.tensor([2, 1], dtype=torch.long),
    )

    moved = batch.to("cpu", feature_dtype=torch.bfloat16)

    assert moved.features.dtype == torch.bfloat16
    assert moved.feature_lengths.dtype == torch.long
    assert moved.targets.dtype == torch.long


def test_make_synthetic_ctc_batch_shapes() -> None:
    batch = make_synthetic_ctc_batch(SyntheticOverfitConfig(batch_size=3, target_len=5))

    assert batch.features.shape[0] == 3
    assert batch.feature_lengths.shape == (3,)
    assert batch.target_lengths.shape == (3,)
    assert batch.targets.numel() == 15


def test_synthetic_overfit_reduces_loss() -> None:
    result = run_synthetic_overfit(
        SyntheticOverfitConfig(
            steps=8,
            lr=3e-3,
            p_start=0.0,
            p_max=0.0,
            warmup_steps=0,
            ramp_steps=0,
        )
    )

    assert result["final_loss"] < result["initial_loss"]
