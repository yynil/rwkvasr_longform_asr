from __future__ import annotations

from dataclasses import dataclass

import torch

from rwkvasr.modules import (
    DirectionDropoutConfig,
    DirectionDropoutScheduler,
    RWKVCTCModel,
    RWKVCTCModelConfig,
)

from .ctc_task import CTCBatch, RWKVDualModeCTCTrainer
from .optimizer import RWKVOptimizerConfig, build_rwkv_optimizer


@dataclass(frozen=True)
class SyntheticOverfitConfig:
    batch_size: int = 2
    input_dim: int = 80
    vocab_size: int = 16
    target_len: int = 4
    frames_per_token: int = 3
    noise_std: float = 0.01
    n_embd: int = 128
    dim_att: int = 128
    dim_ff: int = 256
    num_layers: int = 2
    head_size: int = 32
    conv_kernel_size: int = 5
    dropout: float = 0.0
    steps: int = 12
    lr: float = 2e-3
    weight_decay: float = 0.01
    seed: int = 42
    direction_variant: str = "drop_both"
    p_start: float = 0.0
    p_max: float = 0.2
    warmup_steps: int = 2
    ramp_steps: int = 6


def make_synthetic_ctc_batch(config: SyntheticOverfitConfig) -> CTCBatch:
    generator = torch.Generator().manual_seed(config.seed)
    torch.manual_seed(config.seed)

    token_prototypes = torch.randn(config.vocab_size, config.input_dim, generator=generator)
    target_sequences: list[torch.Tensor] = []
    feature_sequences: list[torch.Tensor] = []

    for _ in range(config.batch_size):
        tokens = torch.randint(
            low=1,
            high=config.vocab_size,
            size=(config.target_len,),
            generator=generator,
        )
        target_sequences.append(tokens)

        frames = token_prototypes[tokens].repeat_interleave(config.frames_per_token, dim=0)
        noise = config.noise_std * torch.randn(
            frames.shape,
            generator=generator,
            dtype=frames.dtype,
            device=frames.device,
        )
        feature_sequences.append(frames + noise)

    max_frames = max(frames.size(0) for frames in feature_sequences)
    padded_features = torch.zeros(config.batch_size, max_frames, config.input_dim)
    feature_lengths = torch.zeros(config.batch_size, dtype=torch.long)
    for idx, frames in enumerate(feature_sequences):
        length = frames.size(0)
        padded_features[idx, :length] = frames
        feature_lengths[idx] = length

    targets = torch.cat(target_sequences, dim=0)
    target_lengths = torch.full((config.batch_size,), config.target_len, dtype=torch.long)
    return CTCBatch(
        features=padded_features,
        feature_lengths=feature_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )


def run_synthetic_overfit(config: SyntheticOverfitConfig) -> dict[str, float]:
    torch.manual_seed(config.seed)
    model = RWKVCTCModel(
        RWKVCTCModelConfig(
            input_dim=config.input_dim,
            n_embd=config.n_embd,
            dim_att=config.dim_att,
            dim_ff=config.dim_ff,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            head_size=config.head_size,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
            frontend_type="linear",
        )
    )
    scheduler = DirectionDropoutScheduler(
        DirectionDropoutConfig(
            num_layers=config.num_layers,
            variant=config.direction_variant,  # type: ignore[arg-type]
            p_start=config.p_start,
            p_max=config.p_max,
            warmup_steps=config.warmup_steps,
            ramp_steps=config.ramp_steps,
        )
    )
    trainer = RWKVDualModeCTCTrainer(model, direction_scheduler=scheduler)
    optimizer = build_rwkv_optimizer(
        model,
        RWKVOptimizerConfig(lr=config.lr, weight_decay=config.weight_decay),
    )
    batch = make_synthetic_ctc_batch(config)

    history: list[float] = []
    gen = torch.Generator().manual_seed(config.seed + 1)
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = trainer.training_loss(batch, step=step, generator=gen)
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    return {
        "initial_loss": history[0],
        "final_loss": history[-1],
        "best_loss": min(history),
    }


def main() -> None:
    result = run_synthetic_overfit(SyntheticOverfitConfig())
    print(
        f"synthetic_overfit initial={result['initial_loss']:.4f} "
        f"final={result['final_loss']:.4f} best={result['best_loss']:.4f}"
    )
