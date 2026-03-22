from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .direction_dropout import LayerDirectionMask
from .rwkv7_bidirectional import (
    BidirectionalRWKVTimeMixer,
    BidirectionalTimeMixerState,
    BidirectionalVFirstState,
)
from .rwkv7_time_mixer import RWKV7TimeMixerConfig


@dataclass(frozen=True)
class RWKVConformerBlockConfig:
    n_embd: int
    dim_att: int
    dim_ff: int
    n_layer: int
    layer_id: int
    head_size: int = 64
    backend: str = "native"
    conv_kernel_size: int = 31
    dropout: float = 0.1

    def to_time_mixer_config(self) -> RWKV7TimeMixerConfig:
        return RWKV7TimeMixerConfig(
            n_embd=self.n_embd,
            dim_att=self.dim_att,
            n_layer=self.n_layer,
            layer_id=self.layer_id,
            head_size=self.head_size,
            backend=self.backend,
        )


@dataclass
class RWKVConformerBlockState:
    time_mixer: BidirectionalTimeMixerState | None = None
    conv_cache: Tensor | None = None


class ConformerFeedForward(nn.Module):
    def __init__(self, dim_model: int, dim_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class CausalConvolutionModule(nn.Module):
    def __init__(self, dim_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")

        self.dim_model = dim_model
        self.kernel_size = kernel_size
        self.pointwise_in = nn.Conv1d(dim_model, dim_model * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim_model,
            dim_model,
            kernel_size=kernel_size,
            groups=dim_model,
            padding=0,
        )
        self.pointwise_out = nn.Conv1d(dim_model, dim_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    @property
    def cache_len(self) -> int:
        return self.kernel_size - 1

    def init_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.zeros(batch_size, self.dim_model, self.cache_len, device=device, dtype=dtype)

    def forward(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x_conv = x.transpose(1, 2)
        x_conv = self.pointwise_in(x_conv)
        x_main, x_gate = x_conv.chunk(2, dim=1)
        x_conv = x_main * torch.sigmoid(x_gate)

        if self.cache_len == 0:
            conv_input = x_conv
            next_cache = x_conv.new_zeros(x_conv.size(0), x_conv.size(1), 0)
        else:
            if cache is None:
                cache = self.init_cache(x_conv.size(0), device=x_conv.device, dtype=x_conv.dtype)
            conv_input = torch.cat([cache, x_conv], dim=-1)
            next_cache = conv_input[:, :, -self.cache_len :].detach()

        x_conv = self.depthwise(conv_input)
        x_conv = torch.nn.functional.silu(x_conv)
        x_conv = self.pointwise_out(x_conv)
        x_conv = self.dropout(x_conv).transpose(1, 2)
        return x_conv, next_cache


class RWKVConformerBlock(nn.Module):
    def __init__(self, config: RWKVConformerBlockConfig):
        super().__init__()
        self.config = config
        self.ffn1_norm = nn.LayerNorm(config.n_embd)
        self.time_mixer_norm = nn.LayerNorm(config.n_embd)
        self.conv_norm = nn.LayerNorm(config.n_embd)
        self.ffn2_norm = nn.LayerNorm(config.n_embd)
        self.final_norm = nn.LayerNorm(config.n_embd)

        self.ffn1 = ConformerFeedForward(config.n_embd, config.dim_ff, config.dropout)
        self.time_mixer = BidirectionalRWKVTimeMixer(config.to_time_mixer_config())
        self.conv = CausalConvolutionModule(
            dim_model=config.n_embd,
            kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
        )
        self.ffn2 = ConformerFeedForward(config.n_embd, config.dim_ff, config.dropout)

    def forward(
        self,
        x: Tensor,
        *,
        v_first: BidirectionalVFirstState | None = None,
        state: RWKVConformerBlockState | None = None,
        layer_mask: LayerDirectionMask | None = None,
    ) -> tuple[Tensor, BidirectionalVFirstState, RWKVConformerBlockState]:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        time_state = None if state is None else state.time_mixer
        time_out, next_v_first, next_time_state = self.time_mixer(
            self.time_mixer_norm(x),
            v_first=v_first,
            state=time_state,
            layer_mask=layer_mask,
        )
        x = x + time_out

        conv_cache = None if state is None else state.conv_cache
        conv_out, next_conv_cache = self.conv(self.conv_norm(x), cache=conv_cache)
        x = x + conv_out

        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        x = self.final_norm(x)

        return x, next_v_first, RWKVConformerBlockState(
            time_mixer=next_time_state,
            conv_cache=next_conv_cache,
        )
