from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .direction_dropout import DirectionMask, LayerDirectionMask
from .rwkv7_bidirectional import BidirectionalRWKVTimeMixer, BidirectionalTimeMixerState, BidirectionalVFirstState
from .rwkv7_time_mixer import RWKV7TimeMixerConfig
from .rwkv_conformer import RWKVConformerBlockState
from .wenet_frontend import GlobalCMVN, load_wenet_cmvn


def aut_conv2d8_out_lengths(lengths: Tensor) -> Tensor:
    """Length formula for Qwen3-ASR style 3x stride-2 Conv2D frontend."""

    lengths = lengths.to(dtype=torch.long)
    for _ in range(3):
        lengths = torch.div(lengths + 1, 2, rounding_mode="floor")
    return lengths


def aut_conv2d8_out_freq_bins(num_mel_bins: int) -> int:
    bins = int(num_mel_bins)
    for _ in range(3):
        bins = (bins + 1) // 2
    return bins


def _activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu_new":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"Unsupported AuRWKV activation_function: {name}")


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: int = 10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs an even channel count.")
        log_timescale_increment = math.log(float(max_timescale)) / float(channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2, dtype=torch.float32))
        scaled_time = torch.arange(int(length), dtype=torch.float32)[:, None] * inv_timescales[None, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int, *, dtype: torch.dtype) -> Tensor:
        return self.positional_embedding[:seqlen, :].to(dtype=dtype)


@dataclass(frozen=True)
class AuRWKVEncoderConfig:
    input_dim: int = 128
    n_embd: int = 896
    output_dim: int = 1024
    dim_att: int = 896
    dim_ff: int = 3584
    num_layers: int = 18
    head_size: int = 64
    backend: str = "native"
    dropout: float = 0.0
    activation_dropout: float = 0.0
    activation_function: str = "gelu"
    max_source_positions: int = 1500
    downsample_hidden_size: int = 480
    scale_embedding: bool = False
    conv_chunksize: int = 500
    cmvn_file: str | None = None
    cmvn_is_json: bool = True


class AuRWKVEncoderLayer(nn.Module):
    """Qwen3-ASR audio encoder layer with self-attention replaced by BiRWKV TimeMixer."""

    def __init__(self, config: AuRWKVEncoderConfig, *, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = int(layer_id)
        self.embed_dim = int(config.n_embd)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.time_mixer = BidirectionalRWKVTimeMixer(
            RWKV7TimeMixerConfig(
                n_embd=self.embed_dim,
                dim_att=int(config.dim_att),
                n_layer=int(config.num_layers),
                layer_id=int(layer_id),
                head_size=int(config.head_size),
                backend=str(config.backend),
            )
        )
        self.dropout = float(config.dropout)
        self.activation_fn = _activation(config.activation_function)
        self.activation_dropout = nn.Dropout(float(config.activation_dropout))
        self.fc1 = nn.Linear(self.embed_dim, int(config.dim_ff))
        self.fc2 = nn.Linear(int(config.dim_ff), self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.resid_dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        v_first: BidirectionalVFirstState | None = None,
        state: RWKVConformerBlockState | None = None,
        layer_mask: LayerDirectionMask | None = None,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, BidirectionalVFirstState, RWKVConformerBlockState]:
        residual = hidden_states
        time_state = None if state is None else state.time_mixer
        hidden_states, next_v_first, next_time_state = self.time_mixer(
            self.self_attn_layer_norm(hidden_states),
            v_first=v_first,
            state=time_state,
            layer_mask=layer_mask,
            lengths=lengths,
        )
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

        return hidden_states, next_v_first, RWKVConformerBlockState(
            time_mixer=next_time_state,
            conv_cache=None,
        )


class AuRWKVEncoder(nn.Module):
    """Qwen3-ASR AuT-style audio encoder with bidirectional RWKV-7 TimeMixer layers."""

    def __init__(self, config: AuRWKVEncoderConfig):
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
        self.layers = nn.ModuleList(
            [AuRWKVEncoderLayer(config, layer_id=layer_id) for layer_id in range(int(config.num_layers))]
        )
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

    @staticmethod
    def _pack_optional_tensor(tensor: Tensor | None, *, reference: Tensor) -> Tensor:
        if tensor is None:
            return reference.new_empty(0)
        return tensor

    @staticmethod
    def _unpack_optional_tensor(tensor: Tensor) -> Tensor | None:
        if tensor.numel() == 0:
            return None
        return tensor

    def _forward_layer_checkpointed(
        self,
        layer: AuRWKVEncoderLayer,
        hidden_states: Tensor,
        *,
        v_first: BidirectionalVFirstState | None,
        layer_mask: LayerDirectionMask,
        lengths: Tensor | None,
    ) -> tuple[Tensor, BidirectionalVFirstState]:
        forward_v_first = self._pack_optional_tensor(
            None if v_first is None else v_first.forward,
            reference=hidden_states,
        )
        backward_v_first = self._pack_optional_tensor(
            None if v_first is None else v_first.backward,
            reference=hidden_states,
        )

        def _custom_forward(x_in: Tensor, forward_in: Tensor, backward_in: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            current_v_first = BidirectionalVFirstState(
                forward=self._unpack_optional_tensor(forward_in),
                backward=self._unpack_optional_tensor(backward_in),
            )
            if current_v_first.forward is None and current_v_first.backward is None:
                current_v_first = None
            next_x, next_v_first, _ = layer(
                x_in,
                v_first=current_v_first,
                state=None,
                layer_mask=layer_mask,
                lengths=lengths,
            )
            return (
                next_x,
                self._pack_optional_tensor(next_v_first.forward, reference=next_x),
                self._pack_optional_tensor(next_v_first.backward, reference=next_x),
            )

        next_x, next_forward_v_first, next_backward_v_first = activation_checkpoint(
            _custom_forward,
            hidden_states,
            forward_v_first,
            backward_v_first,
            use_reentrant=False,
        )
        return next_x, BidirectionalVFirstState(
            forward=self._unpack_optional_tensor(next_forward_v_first),
            backward=self._unpack_optional_tensor(next_backward_v_first),
        )

    def _frontend(self, x: Tensor, lengths: Tensor | None) -> tuple[Tensor, Tensor | None]:
        if self.global_cmvn is not None:
            x = self.global_cmvn(x.float())
        compute_dtype = self._encoder_compute_dtype()
        if compute_dtype is not None and x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        x = x.transpose(1, 2).unsqueeze(1)
        embeds: list[Tensor] = []
        for chunk in x.split(max(1, self.conv_chunksize), dim=0):
            chunk = torch.nn.functional.gelu(self.conv2d1(chunk))
            chunk = torch.nn.functional.gelu(self.conv2d2(chunk))
            chunk = torch.nn.functional.gelu(self.conv2d3(chunk))
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
        direction_mask: DirectionMask | None = None,
        state: Any = None,
    ) -> tuple[Tensor, Tensor | None, Any]:
        x, lengths = self._frontend(x, lengths)

        if state is None or len(state.block_states) == 0:
            block_states: list[RWKVConformerBlockState | None] = [None] * len(self.layers)
        else:
            if len(state.block_states) != len(self.layers):
                raise ValueError("Encoder state block count does not match the number of AuRWKV layers.")
            block_states = state.block_states

        next_states: list[RWKVConformerBlockState | None] = []
        v_first: BidirectionalVFirstState | None = None
        for layer_idx, layer in enumerate(self.layers):
            layer_mask = LayerDirectionMask()
            if direction_mask is not None:
                layer_mask = direction_mask.layer(layer_idx)
            if self.gradient_checkpointing and self.training and block_states[layer_idx] is None:
                x, v_first = self._forward_layer_checkpointed(
                    layer,
                    x,
                    v_first=v_first,
                    layer_mask=layer_mask,
                    lengths=lengths,
                )
                next_states.append(None)
            else:
                x, v_first, next_state = layer(
                    x,
                    v_first=v_first,
                    state=block_states[layer_idx],
                    layer_mask=layer_mask,
                    lengths=lengths,
                )
                next_states.append(next_state)

        x = self.ln_post(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)

        from .rwkv_asr_ctc import RWKVConformerEncoderState

        return x, lengths, RWKVConformerEncoderState(block_states=next_states)

    def load_qwen3_asr_non_attention_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Any]:
        """Load Qwen3-ASR audio encoder weights whose shapes still match after attention replacement."""

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
            if ".time_mixer." in own_key:
                skipped.append(own_key)
                continue
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

    def load_qwen3_asr_non_attention_checkpoint(self, path: str | Path) -> dict[str, Any]:
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
                report = self.load_qwen3_asr_non_attention_checkpoint(shard)
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
        return self.load_qwen3_asr_non_attention_state_dict(state_dict)
