from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .direction_dropout import DirectionMask, build_inference_direction_mask
from .rwkv_asr_ctc import RWKVCTCModelConfig, RWKVConformerEncoder


@dataclass(frozen=True)
class BestRQModelConfig:
    encoder: RWKVCTCModelConfig
    codebook_size: int = 8192
    projection_dim: int = 16
    mask_prob: float = 0.15
    mask_span_length: int = 10
    quantizer_seed: int = 20260609


class RandomProjectionQuantizer(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, codebook_size: int, *, seed: int):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        projection = torch.randn(int(input_dim), int(projection_dim), generator=generator) / math.sqrt(
            float(input_dim)
        )
        codebook = torch.randn(int(codebook_size), int(projection_dim), generator=generator)
        codebook = F.normalize(codebook, dim=-1)
        self.register_buffer("projection", projection, persistent=True)
        self.register_buffer("codebook", codebook, persistent=True)

    @torch.no_grad()
    def forward(self, features: Tensor) -> Tensor:
        projected = features.float().matmul(self.projection.float())
        projected = F.normalize(projected, dim=-1)
        scores = projected.matmul(self.codebook.float().transpose(0, 1))
        return scores.argmax(dim=-1)


class BestRQPretrainModel(nn.Module):
    def __init__(self, config: BestRQModelConfig):
        super().__init__()
        self.config = config
        self.encoder = RWKVConformerEncoder(config.encoder.to_encoder_config())
        input_dim = int(config.encoder.input_dim)
        encoder_dim = int(config.encoder.resolved_encoder_output_dim)
        self.mask_embedding = nn.Parameter(torch.zeros(input_dim))
        self.quantizer = RandomProjectionQuantizer(
            input_dim=input_dim,
            projection_dim=int(config.projection_dim),
            codebook_size=int(config.codebook_size),
            seed=int(config.quantizer_seed),
        )
        self.best_rq_head = nn.Linear(encoder_dim, int(config.codebook_size))

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.enable_gradient_checkpointing(enabled)

    def _sample_mask(self, lengths: Tensor, max_length: int) -> Tensor:
        device = lengths.device
        mask = torch.zeros((int(lengths.numel()), int(max_length)), dtype=torch.bool, device=device)
        span = max(1, int(self.config.mask_span_length))
        prob = max(0.0, min(float(self.config.mask_prob), 1.0))
        for sample_idx, raw_length in enumerate(lengths.tolist()):
            length = max(0, min(int(raw_length), int(max_length)))
            if length <= 0:
                continue
            num_spans = max(1, int(round(prob * length / span))) if prob > 0.0 else 0
            if num_spans <= 0:
                continue
            starts = torch.randint(0, max(length, 1), (num_spans,), device=device)
            for start in starts.tolist():
                end = min(int(start) + span, length)
                mask[sample_idx, int(start) : end] = True
        return mask

    @staticmethod
    def _valid_mask(lengths: Tensor, max_length: int) -> Tensor:
        positions = torch.arange(int(max_length), device=lengths.device).unsqueeze(0)
        return positions < lengths.to(device=lengths.device, dtype=torch.long).unsqueeze(1)

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor,
        *,
        direction_mask: DirectionMask | None = None,
    ) -> dict[str, Tensor]:
        if feature_lengths is None:
            feature_lengths = torch.full(
                (int(features.size(0)),),
                int(features.size(1)),
                dtype=torch.long,
                device=features.device,
            )
        feature_lengths = feature_lengths.to(device=features.device, dtype=torch.long).clamp_max(int(features.size(1)))
        targets = self.quantizer(features.detach())
        masked_positions = self._sample_mask(feature_lengths, int(features.size(1)))
        masked_features = features.clone()
        if masked_positions.any():
            masked_features[masked_positions] = self.mask_embedding.to(
                device=features.device,
                dtype=features.dtype,
            )
        if direction_mask is None:
            direction_mask = build_inference_direction_mask(
                int(self.config.encoder.num_layers),
                mode="bi",
                device=features.device,
            )
        encoded, encoded_lengths, _ = self.encoder(
            masked_features,
            lengths=feature_lengths,
            direction_mask=direction_mask,
        )
        logits = self.best_rq_head(encoded)
        if encoded_lengths is None:
            encoded_lengths = feature_lengths.clamp_max(int(encoded.size(1)))
        encoded_lengths = encoded_lengths.to(device=features.device, dtype=torch.long).clamp_max(int(encoded.size(1)))
        target_time = min(int(targets.size(1)), int(logits.size(1)))
        logits = logits[:, :target_time]
        targets = targets[:, :target_time]
        loss_mask = masked_positions[:, :target_time] & self._valid_mask(encoded_lengths, target_time)
        if not bool(loss_mask.any().item()):
            loss_mask = self._valid_mask(encoded_lengths, target_time)
        flat_logits = logits[loss_mask]
        flat_targets = targets[loss_mask]
        loss = F.cross_entropy(flat_logits.float(), flat_targets.long())
        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            accuracy = (predictions == flat_targets).float().mean() if flat_targets.numel() else loss.new_tensor(0.0)
        return {
            "loss": loss,
            "accuracy": accuracy.to(device=loss.device, dtype=loss.dtype),
            "masked_frames": loss_mask.sum().to(device=loss.device, dtype=torch.float32),
        }


def best_rq_model_config_from_training_config(config: Any) -> BestRQModelConfig:
    encoder_config = RWKVCTCModelConfig(
        feature_extractor_type=config.feature_extractor_type,
        input_dim=config.input_dim,
        n_embd=config.n_embd,
        dim_att=config.dim_att,
        dim_ff=config.dim_ff,
        num_layers=config.num_layers,
        vocab_size=int(getattr(config, "vocab_size", 1) or 1),
        head_size=config.head_size,
        backend=config.backend,
        conv_kernel_size=config.conv_kernel_size,
        dropout=config.dropout,
        blank_id=int(getattr(config, "blank_id", 0)),
        frontend_type=config.frontend_type,
        encoder_output_dim=config.encoder_output_dim,
        aut_downsample_hidden_size=config.aut_downsample_hidden_size,
        aut_activation_function=config.aut_activation_function,
        aut_activation_dropout=config.aut_activation_dropout,
        aut_max_source_positions=config.aut_max_source_positions,
        aut_scale_embedding=config.aut_scale_embedding,
        aut_conv_chunksize=config.aut_conv_chunksize,
        sensevoice_tp_blocks=config.sensevoice_tp_blocks,
        cmvn_file=getattr(config, "cmvn_file", None),
        cmvn_is_json=bool(getattr(config, "cmvn_is_json", True)),
    )
    return BestRQModelConfig(
        encoder=encoder_config,
        codebook_size=int(config.best_rq_codebook_size),
        projection_dim=int(config.best_rq_projection_dim),
        mask_prob=float(config.best_rq_mask_prob),
        mask_span_length=int(config.best_rq_mask_span_length),
        quantizer_seed=int(config.best_rq_quantizer_seed),
    )
