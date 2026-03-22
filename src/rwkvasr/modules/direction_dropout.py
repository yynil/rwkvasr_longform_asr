from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

DirectionDropoutVariant = Literal["drop_r2l_only", "drop_both"]
InferenceMode = Literal["bi", "l2r", "r2l", "alt"]


@dataclass(frozen=True)
class LayerDirectionMask:
    use_forward: bool = True
    use_backward: bool = True

    def __post_init__(self) -> None:
        if not (self.use_forward or self.use_backward):
            raise ValueError("At least one direction must be enabled.")


@dataclass(frozen=True)
class DirectionMask:
    forward: Tensor
    backward: Tensor

    def __post_init__(self) -> None:
        if self.forward.dtype != torch.bool or self.backward.dtype != torch.bool:
            raise TypeError("Direction masks must be boolean tensors.")
        if self.forward.shape != self.backward.shape:
            raise ValueError("Forward and backward masks must have the same shape.")
        if not torch.all(self.forward | self.backward):
            raise ValueError("Each layer must keep at least one direction active.")

    @property
    def num_layers(self) -> int:
        return int(self.forward.numel())

    def layer(self, layer_idx: int) -> LayerDirectionMask:
        return LayerDirectionMask(
            use_forward=bool(self.forward[layer_idx].item()),
            use_backward=bool(self.backward[layer_idx].item()),
        )


@dataclass(frozen=True)
class DirectionDropoutConfig:
    num_layers: int
    variant: DirectionDropoutVariant = "drop_both"
    p_start: float = 0.2
    p_max: float = 0.2
    warmup_steps: int = 0
    ramp_steps: int = 0

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= self.p_start <= 1.0:
            raise ValueError("p_start must be within [0, 1].")
        if not 0.0 <= self.p_max <= 1.0:
            raise ValueError("p_max must be within [0, 1].")
        if self.p_max < self.p_start:
            raise ValueError("p_max must be greater than or equal to p_start.")
        if self.warmup_steps < 0 or self.ramp_steps < 0:
            raise ValueError("warmup_steps and ramp_steps must be non-negative.")


def build_inference_direction_mask(
    num_layers: int,
    mode: InferenceMode,
    *,
    start_with_forward: bool = True,
    device: torch.device | str | None = None,
) -> DirectionMask:
    forward = torch.ones(num_layers, dtype=torch.bool, device=device)
    backward = torch.ones(num_layers, dtype=torch.bool, device=device)

    if mode == "bi":
        return DirectionMask(forward=forward, backward=backward)
    if mode == "l2r":
        return DirectionMask(forward=forward, backward=torch.zeros_like(backward))
    if mode == "r2l":
        return DirectionMask(forward=torch.zeros_like(forward), backward=backward)
    if mode == "alt":
        layer_ids = torch.arange(num_layers, device=device)
        use_forward = (layer_ids % 2 == 0) if start_with_forward else (layer_ids % 2 == 1)
        return DirectionMask(forward=use_forward, backward=~use_forward)
    raise ValueError(f"Unsupported inference mode: {mode}")


def build_last_n_bidirectional_mask(
    num_layers: int,
    n_bidirectional: int,
    *,
    device: torch.device | str | None = None,
) -> DirectionMask:
    if not 0 <= n_bidirectional <= num_layers:
        raise ValueError("n_bidirectional must be within [0, num_layers].")
    forward = torch.ones(num_layers, dtype=torch.bool, device=device)
    backward = torch.zeros(num_layers, dtype=torch.bool, device=device)
    if n_bidirectional > 0:
        backward[-n_bidirectional:] = True
    return DirectionMask(forward=forward, backward=backward)


class DirectionDropoutScheduler:
    def __init__(self, config: DirectionDropoutConfig):
        self.config = config

    def probability_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be non-negative.")
        if step < self.config.warmup_steps:
            return self.config.p_start
        if self.config.ramp_steps == 0:
            return self.config.p_max

        ramp_pos = min(step - self.config.warmup_steps, self.config.ramp_steps)
        alpha = ramp_pos / self.config.ramp_steps
        return self.config.p_start + alpha * (self.config.p_max - self.config.p_start)

    def sample_mask(
        self,
        step: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> DirectionMask:
        p_drop = self.probability_at(step)
        forward = torch.ones(self.config.num_layers, dtype=torch.bool, device=device)
        backward = torch.ones(self.config.num_layers, dtype=torch.bool, device=device)
        drop = torch.rand(self.config.num_layers, generator=generator, device=device) < p_drop

        if self.config.variant == "drop_r2l_only":
            backward = backward & ~drop
            return DirectionMask(forward=forward, backward=backward)

        if self.config.variant == "drop_both":
            choose_backward = torch.rand(
                self.config.num_layers, generator=generator, device=device
            ) < 0.5
            forward = forward & ~(drop & ~choose_backward)
            backward = backward & ~(drop & choose_backward)
            return DirectionMask(forward=forward, backward=backward)

        raise ValueError(f"Unsupported dropout variant: {self.config.variant}")
