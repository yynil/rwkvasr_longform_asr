from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from rwkvasr.modules import (
    DirectionDropoutScheduler,
    DirectionMask,
    RWKVCTCModel,
    build_inference_direction_mask,
)


@dataclass
class CTCBatch:
    features: Tensor
    feature_lengths: Tensor
    targets: Tensor
    target_lengths: Tensor
    utt_ids: list[str] | None = None
    decoder_targets: Tensor | None = None
    decoder_target_lengths: Tensor | None = None
    decoder_prompt_before_audio: Tensor | None = None
    decoder_prompt_before_audio_lengths: Tensor | None = None
    ctc_teacher_audio_rows: list[dict[str, Any] | None] | None = None

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype | None = None,
    ) -> "CTCBatch":
        features = self.features.to(device)
        if feature_dtype is not None and features.is_floating_point():
            features = features.to(dtype=feature_dtype)
        return CTCBatch(
            features=features,
            feature_lengths=self.feature_lengths.to(device),
            targets=self.targets.to(device),
            target_lengths=self.target_lengths.to(device),
            utt_ids=self.utt_ids,
            decoder_targets=(
                self.decoder_targets.to(device) if self.decoder_targets is not None else None
            ),
            decoder_target_lengths=(
                self.decoder_target_lengths.to(device)
                if self.decoder_target_lengths is not None
                else None
            ),
            decoder_prompt_before_audio=(
                self.decoder_prompt_before_audio.to(device)
                if self.decoder_prompt_before_audio is not None
                else None
            ),
            decoder_prompt_before_audio_lengths=(
                self.decoder_prompt_before_audio_lengths.to(device)
                if self.decoder_prompt_before_audio_lengths is not None
                else None
            ),
            ctc_teacher_audio_rows=self.ctc_teacher_audio_rows,
        )


class RWKVDualModeCTCTrainer:
    def __init__(
        self,
        model: RWKVCTCModel,
        *,
        direction_scheduler: DirectionDropoutScheduler | None = None,
    ):
        self.model = model
        self.direction_scheduler = direction_scheduler

    @property
    def num_layers(self) -> int:
        return self.model.config.num_layers

    def training_direction_mask(
        self,
        step: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> DirectionMask:
        if self.direction_scheduler is None:
            return build_inference_direction_mask(self.num_layers, mode="bi", device=device)
        return self.direction_scheduler.sample_mask(step, device=device, generator=generator)

    def eval_direction_mask(
        self,
        mode: str,
        *,
        device: torch.device | str | None = None,
    ) -> DirectionMask:
        return build_inference_direction_mask(self.num_layers, mode=mode, device=device)

    def training_loss(
        self,
        batch: CTCBatch,
        *,
        step: int,
        generator: torch.Generator | None = None,
        direction_mask: DirectionMask | None = None,
    ) -> tuple[Tensor, DirectionMask]:
        mask = direction_mask
        if mask is None:
            mask = self.training_direction_mask(
                step,
                device=batch.features.device,
                generator=generator,
            )
        losses = self.model.joint_losses(
            batch.features,
            batch.feature_lengths,
            batch.targets,
            batch.target_lengths,
            decoder_targets=batch.decoder_targets,
            decoder_target_lengths=batch.decoder_target_lengths,
            decoder_prompt_before_audio=batch.decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=batch.decoder_prompt_before_audio_lengths,
            direction_mask=mask,
        )
        return losses["loss"], mask

    @torch.no_grad()
    def inference_logits(
        self,
        features: Tensor,
        feature_lengths: Tensor | None,
        *,
        mode: str,
    ) -> tuple[Tensor, Tensor | None]:
        mask = self.eval_direction_mask(mode, device=features.device)
        logits, logit_lengths, _ = self.model(
            features,
            feature_lengths,
            direction_mask=mask,
        )
        return logits, logit_lengths

    @torch.no_grad()
    def eval_loss(
        self,
        batch: CTCBatch,
        *,
        mode: str = "bi",
    ) -> Tensor:
        mask = self.eval_direction_mask(mode, device=batch.features.device)
        losses = self.model.joint_losses(
            batch.features,
            batch.feature_lengths,
            batch.targets,
            batch.target_lengths,
            decoder_targets=batch.decoder_targets,
            decoder_target_lengths=batch.decoder_target_lengths,
            decoder_prompt_before_audio=batch.decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=batch.decoder_prompt_before_audio_lengths,
            direction_mask=mask,
        )
        return losses["loss"]
