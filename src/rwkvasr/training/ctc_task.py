from __future__ import annotations

from dataclasses import dataclass

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
        logits, logit_lengths, _ = self.model(
            batch.features,
            batch.feature_lengths,
            direction_mask=mask,
        )
        if logit_lengths is None:
            raise ValueError("CTC training requires feature lengths.")
        loss = self.model.ctc_loss(logits, logit_lengths, batch.targets, batch.target_lengths)
        return loss, mask

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
        logits, logit_lengths, _ = self.model(
            batch.features,
            batch.feature_lengths,
            direction_mask=mask,
        )
        if logit_lengths is None:
            raise ValueError("CTC evaluation requires feature lengths.")
        return self.model.ctc_loss(logits, logit_lengths, batch.targets, batch.target_lengths)
