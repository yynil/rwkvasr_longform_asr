from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .aurwkv_encoder import SinusoidsPositionEmbedding, _activation, aut_conv2d8_out_freq_bins, aut_conv2d8_out_lengths
from .wenet_frontend import GlobalCMVN, load_wenet_cmvn


@dataclass(frozen=True)
class Qwen3TransformerEncoderConfig:
    input_dim: int = 128
    n_embd: int = 896
    output_dim: int = 1024
    dim_att: int = 896
    dim_ff: int = 3584
    num_layers: int = 18
    head_size: int = 64
    dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    max_source_positions: int = 1500
    downsample_hidden_size: int = 480
    scale_embedding: bool = False
    conv_chunksize: int = 500
    cmvn_file: str | None = None
    cmvn_is_json: bool = True


class Qwen3TransformerSelfAttention(nn.Module):
    def __init__(self, config: Qwen3TransformerEncoderConfig):
        super().__init__()
        self.embed_dim = int(config.n_embd)
        self.head_dim = int(config.head_size)
        if self.embed_dim % self.head_dim != 0:
            raise ValueError(
                f"n_embd must be divisible by head_size for qwen3_transformer: {self.embed_dim} % {self.head_dim}"
            )
        self.num_heads = self.embed_dim // self.head_dim
        self.dropout = float(config.dropout)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _project(self, projection: nn.Linear, hidden_states: Tensor) -> Tensor:
        batch, time, _ = hidden_states.shape
        projected = projection(hidden_states)
        return projected.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: Tensor, lengths: Tensor | None = None) -> Tensor:
        query = self._project(self.q_proj, hidden_states)
        key = self._project(self.k_proj, hidden_states)
        value = self._project(self.v_proj, hidden_states)

        attention_mask = None
        if lengths is not None:
            valid = torch.arange(hidden_states.size(1), device=hidden_states.device)[None, :] < lengths[:, None]
            attention_mask = valid[:, None, None, :]

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(hidden_states.size(0), hidden_states.size(1), -1)
        return self.out_proj(attended)


class Qwen3TransformerEncoderLayer(nn.Module):
    """Qwen3-ASR audio Transformer encoder layer used as an attention control baseline."""

    def __init__(self, config: Qwen3TransformerEncoderConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(int(config.n_embd))
        self.self_attn = Qwen3TransformerSelfAttention(config)
        self.dropout = float(config.dropout)
        self.activation_fn = _activation(config.activation_function)
        self.activation_dropout = nn.Dropout(float(config.activation_dropout))
        self.fc1 = nn.Linear(int(config.n_embd), int(config.dim_ff))
        self.fc2 = nn.Linear(int(config.dim_ff), int(config.n_embd))
        self.final_layer_norm = nn.LayerNorm(int(config.n_embd))
        self.resid_dropout = nn.Dropout(float(config.dropout))

    def forward(self, hidden_states: Tensor, lengths: Tensor | None = None) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.self_attn_layer_norm(hidden_states), lengths=lengths)
        hidden_states = residual + self.resid_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + self.resid_dropout(hidden_states)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class Qwen3TransformerEncoder(nn.Module):
    """Qwen3-ASR audio encoder with the original Transformer self-attention blocks."""

    def __init__(self, config: Qwen3TransformerEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.global_cmvn: nn.Module | None = None
        if config.cmvn_file is not None:
            mean, istd = load_wenet_cmvn(config.cmvn_file, is_json=config.cmvn_is_json)
            self.global_cmvn = GlobalCMVN(mean, istd)

        embed_dim = int(config.n_embd)
        self.num_mel_bins = int(config.input_dim)
        self.max_source_positions = int(config.max_source_positions)
        self.embed_scale = math.sqrt(embed_dim) if bool(config.scale_embedding) else 1.0
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([Qwen3TransformerEncoderLayer(config) for _ in range(int(config.num_layers))])
        self.ln_post = nn.LayerNorm(embed_dim)
        self.conv2d1 = nn.Conv2d(1, int(config.downsample_hidden_size), 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            int(config.downsample_hidden_size),
            int(config.downsample_hidden_size),
            3,
            2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            int(config.downsample_hidden_size),
            int(config.downsample_hidden_size),
            3,
            2,
            padding=1,
        )
        self.conv_out = nn.Linear(
            int(config.downsample_hidden_size) * aut_conv2d8_out_freq_bins(int(config.input_dim)),
            embed_dim,
            bias=False,
        )
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.act = _activation(config.activation_function)
        self.proj2 = nn.Linear(embed_dim, int(config.output_dim))
        self.conv_chunksize = int(config.conv_chunksize)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    @staticmethod
    def _module_floating_dtype(module: nn.Module) -> torch.dtype | None:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_floating_point():
                return tensor.dtype
        return None

    def _encoder_compute_dtype(self) -> torch.dtype | None:
        return self._module_floating_dtype(self)

    def _frontend(self, x: Tensor, lengths: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.global_cmvn is not None:
            x = self.global_cmvn(x.float())
        compute_dtype = self._encoder_compute_dtype()
        if compute_dtype is not None and x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        x = x.transpose(1, 2).unsqueeze(1)
        embeds: list[Tensor] = []
        for chunk in x.split(max(1, self.conv_chunksize), dim=0):
            chunk = F.gelu(self.conv2d1(chunk))
            chunk = F.gelu(self.conv2d2(chunk))
            chunk = F.gelu(self.conv2d3(chunk))
            embeds.append(chunk)
        x = torch.cat(embeds, dim=0)
        batch, channels, freq_bins, time_steps = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, time_steps, channels * freq_bins)
        x = self.conv_out(x) * float(self.embed_scale)
        pos = self.positional_embedding(int(x.size(1)), dtype=x.dtype).unsqueeze(0)
        x = x + pos
        if lengths is not None:
            lengths = aut_conv2d8_out_lengths(lengths.to(device=x.device))
            lengths = lengths.clamp_max(int(x.size(1)))
        return x, lengths

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
        *,
        direction_mask: Any = None,
        state: Any = None,
    ) -> tuple[Tensor, Tensor | None, Any]:
        del direction_mask, state
        x, lengths = self._frontend(x, lengths)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = activation_checkpoint(layer, x, lengths, use_reentrant=False)
            else:
                x = layer(x, lengths)

        x = self.ln_post(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)

        from .rwkv_asr_ctc import RWKVConformerEncoderState

        return x, lengths, RWKVConformerEncoderState(block_states=[None] * len(self.layers))

    def load_qwen3_asr_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Any]:
        own_state = self.state_dict()
        prefixes = (
            "",
            "audio_tower.",
            "audio_encoder.",
            "thinker.audio_tower.",
            "thinker.audio_encoder.",
            "model.audio_tower.",
            "model.audio_encoder.",
        )
        loaded: list[str] = []
        skipped: list[str] = []
        for own_key, own_value in own_state.items():
            source_value = None
            for prefix in prefixes:
                candidate = prefix + own_key
                if candidate in state_dict:
                    source_value = state_dict[candidate]
                    break
            if source_value is None or tuple(source_value.shape) != tuple(own_value.shape):
                skipped.append(own_key)
                continue
            own_value.copy_(source_value.to(dtype=own_value.dtype))
            loaded.append(own_key)
        return {"loaded": loaded, "skipped": skipped}

    def load_qwen3_asr_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(path)
        if checkpoint_path.is_dir():
            loaded: list[str] = []
            skipped: list[str] = []
            shards = [
                *sorted(checkpoint_path.glob("*.safetensors")),
                *sorted(checkpoint_path.glob("pytorch_model*.bin")),
                *sorted(checkpoint_path.glob("*.pt")),
                *sorted(checkpoint_path.glob("*.pth")),
            ]
            if not shards:
                raise FileNotFoundError(f"No checkpoint shards found under {checkpoint_path}")
            for shard in shards:
                report = self.load_qwen3_asr_checkpoint(shard)
                loaded.extend(report["loaded"])
                skipped.extend(report["skipped"])
            loaded_set = set(loaded)
            return {"loaded": sorted(loaded_set), "skipped": sorted(set(skipped) - loaded_set)}
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(checkpoint_path), device="cpu")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        return self.load_qwen3_asr_state_dict(state_dict)
