from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NanoCTCLayerNorm(nn.LayerNorm):
    """LayerNorm compatible with FunASR Transformer LayerNorm defaults."""

    def __init__(self, size: int):
        super().__init__(int(size), eps=1e-12)


class NanoCTCMultiHeadedAttention(nn.Module):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super().__init__()
        if int(n_feat) % int(n_head) != 0:
            raise ValueError(f"n_feat must be divisible by n_head, got {n_feat=} {n_head=}")
        self.d_k = int(n_feat) // int(n_head)
        self.h = int(n_head)
        self.linear_q = nn.Linear(int(n_feat), int(n_feat))
        self.linear_k = nn.Linear(int(n_feat), int(n_feat))
        self.linear_v = nn.Linear(int(n_feat), int(n_feat))
        self.linear_out = nn.Linear(int(n_feat), int(n_feat))
        self.attn = None
        self.dropout = nn.Dropout(float(dropout_rate))

    def forward_qkv(self, query: Tensor, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = int(query.size(0))
        q = self.linear_q(query).view(batch_size, -1, self.h, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.h, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.h, self.d_k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(self, value: Tensor, scores: Tensor, mask: Tensor | None) -> Tensor:
        batch_size = int(value.size(0))
        if mask is not None:
            invalid = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(invalid, float("-inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(invalid, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attn), value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.d_k))
        return self.forward_attention(v, scores, mask)


class NanoCTCPositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.w_1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.w_2 = nn.Linear(int(hidden_dim), int(input_dim))
        self.dropout = nn.Dropout(float(dropout_rate))
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class NanoCTCTransformerLayer(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        attention_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
    ):
        super().__init__()
        self.self_attn = NanoCTCMultiHeadedAttention(
            int(attention_heads),
            int(size),
            float(attention_dropout_rate),
        )
        self.feed_forward = NanoCTCPositionwiseFeedForward(
            int(size),
            int(size) // 4,
            float(dropout_rate),
        )
        self.norm1 = NanoCTCLayerNorm(int(size))
        self.norm2 = NanoCTCLayerNorm(int(size))
        self.dropout = nn.Dropout(float(dropout_rate))
        self.size = int(size)
        self.normalize_before = True
        self.concat_after = False
        self.stochastic_depth_rate = 0.0

    def forward(self, x: Tensor, mask: Tensor | None, cache: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        residual = x
        x = self.norm1(x)
        if cache is None:
            x_q = x
        else:
            if tuple(cache.shape) != (int(x.shape[0]), int(x.shape[1]) - 1, self.size):
                raise ValueError("Nano CTC decoder cache shape does not match current input.")
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]
        x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, mask


@dataclass(frozen=True)
class NanoCTCTransformerDecoderConfig:
    downsample_rate: int = 1
    encoder_dim: int = 512
    hidden_dim: int = 512
    ffn_dim: int = 2048
    num_layers: int = 5
    attention_heads: int = 8
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0


def _valid_mask_from_lengths(lengths: Tensor, max_len: int) -> Tensor:
    positions = torch.arange(int(max_len), device=lengths.device).unsqueeze(0)
    return positions < lengths.to(device=lengths.device, dtype=torch.long).unsqueeze(1)


class NanoCTCTransformerDecoder(nn.Module):
    """FunASR-Nano adaptor.Transformer used as the CTC decoder.

    The parameter names intentionally match FunASR's `ctc_decoder.*` subtree after
    removing that prefix, so Nano checkpoint weights can be loaded directly.
    """

    def __init__(self, config: NanoCTCTransformerDecoderConfig):
        super().__init__()
        self.config = config
        self.k = int(config.downsample_rate)
        if self.k < 1:
            raise ValueError("Nano CTC decoder downsample_rate must be >= 1.")
        self.encoder_dim = int(config.encoder_dim)
        self.llm_dim = int(config.hidden_dim)
        self.linear1 = nn.Linear(self.encoder_dim * self.k, int(config.ffn_dim))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int(config.ffn_dim), self.llm_dim)
        self.blocks = nn.ModuleList(
            [
                NanoCTCTransformerLayer(
                    self.llm_dim,
                    attention_heads=int(config.attention_heads),
                    dropout_rate=float(config.dropout_rate),
                    attention_dropout_rate=float(config.attention_dropout_rate),
                )
                for _ in range(int(config.num_layers))
            ]
        )

    def forward(self, x: Tensor, ilens: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        batch_size, seq_len, dim = x.size()
        if int(dim) != self.encoder_dim:
            raise ValueError(f"Nano CTC decoder expected encoder_dim={self.encoder_dim}, got {dim}.")
        chunk_num = (int(seq_len) - 1) // self.k + 1
        pad_num = chunk_num * self.k - int(seq_len)
        if pad_num > 0:
            x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        x = x.contiguous().view(batch_size, chunk_num, dim * self.k)
        x = self.linear2(self.relu(self.linear1(x)))

        olens = None
        masks = None
        if ilens is not None:
            ilens = ilens.to(device=x.device, dtype=torch.long).clamp_min(0)
            olens = torch.div(ilens + self.k - 1, self.k, rounding_mode="floor")
            olens = olens.clamp(max=int(x.size(1)))
            masks = _valid_mask_from_lengths(olens, int(x.size(1)))[:, None, :].to(device=x.device)

        for block in self.blocks:
            x, masks = block(x, masks)
        return x, olens
