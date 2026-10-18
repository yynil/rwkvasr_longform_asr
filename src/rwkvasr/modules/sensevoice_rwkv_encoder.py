from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .direction_dropout import DirectionMask, LayerDirectionMask
from .rwkv7_bidirectional import BidirectionalRWKVTimeMixer, BidirectionalVFirstState
from .rwkv7_time_mixer import RWKV7TimeMixerConfig
from .rwkv_conformer import CausalConvolutionModule, RWKVConformerBlockState


class SenseVoiceLayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        output = torch.nn.functional.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(dtype=input.dtype)


class SenseVoiceSinusoidalPositionEncoder(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        _, timesteps, input_dim = x.size()
        if input_dim % 2 != 0:
            raise ValueError("SenseVoice sinusoidal position encoding requires an even input dimension.")
        half_dim = input_dim // 2
        positions = torch.arange(1, timesteps + 1, device=x.device, dtype=torch.float32)[None, :]
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=torch.float32, device=x.device)) / (
            half_dim - 1
        )
        inv_timescales = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * (-log_timescale_increment)
        )
        scaled_time = positions.reshape(1, -1, 1) * inv_timescales.reshape(1, 1, -1)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return x + encoding.to(dtype=x.dtype)


class SenseVoicePositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.w_2 = nn.Linear(int(hidden_dim), int(input_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


@dataclass(frozen=True)
class SenseVoiceRWKVEncoderConfig:
    input_dim: int = 560
    n_embd: int = 512
    output_dim: int = 512
    dim_att: int = 512
    dim_ff: int = 2048
    num_layers: int = 70
    tp_blocks: int = 20
    head_size: int = 64
    backend: str = "native"
    conv_kernel_size: int = 31
    dropout: float = 0.1

    @property
    def main_blocks(self) -> int:
        main_blocks = int(self.num_layers) - int(self.tp_blocks)
        if main_blocks < 1:
            raise ValueError("SenseVoiceRWKVEncoderConfig requires num_layers > tp_blocks.")
        return main_blocks


class SenseVoiceRWKVEncoderLayer(nn.Module):
    def __init__(
        self,
        config: SenseVoiceRWKVEncoderConfig,
        *,
        layer_id: int,
        input_dim: int,
    ):
        super().__init__()
        self.layer_id = int(layer_id)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(config.n_embd)
        self.norm1 = SenseVoiceLayerNorm(self.input_dim)
        self.input_proj: nn.Linear | None = None
        if self.input_dim != self.hidden_dim:
            self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.time_mixer = BidirectionalRWKVTimeMixer(
            RWKV7TimeMixerConfig(
                n_embd=self.hidden_dim,
                dim_att=int(config.dim_att),
                n_layer=int(config.num_layers),
                layer_id=int(layer_id),
                head_size=int(config.head_size),
                backend=str(config.backend),
            )
        )
        self.norm2 = SenseVoiceLayerNorm(self.hidden_dim)
        self.feed_forward = SenseVoicePositionwiseFeedForward(self.hidden_dim, int(config.dim_ff), float(config.dropout))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        x: Tensor,
        *,
        v_first: BidirectionalVFirstState | None = None,
        state: RWKVConformerBlockState | None = None,
        layer_mask: LayerDirectionMask | None = None,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, BidirectionalVFirstState, RWKVConformerBlockState]:
        residual = x
        time_state = None if state is None else state.time_mixer
        mixed_input = self.norm1(x)
        if self.input_proj is not None:
            mixed_input = self.input_proj(mixed_input)
        mixed, next_v_first, next_time_state = self.time_mixer(
            mixed_input,
            v_first=v_first,
            state=time_state,
            layer_mask=layer_mask,
            lengths=lengths,
        )
        if self.input_dim == self.hidden_dim:
            x = residual + self.dropout(mixed)
        else:
            x = self.dropout(mixed)

        residual = x
        x = residual + self.dropout(self.feed_forward(self.norm2(x)))
        if x.dtype == torch.float16:
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x, next_v_first, RWKVConformerBlockState(time_mixer=next_time_state, conv_cache=None)


class SenseVoiceConformerConvEncoderLayer(nn.Module):
    """SenseVoice-shaped non-attention Conformer control layer.

    The temporal mixer is only a depthwise-separable convolution module; this
    intentionally excludes SANM/MHSA/self-attention so the path can isolate
    whether a non-RWKV, non-attention Conformer-style layer is easier to
    distill from FunASR-Nano CTC logits.
    """

    def __init__(
        self,
        config: SenseVoiceRWKVEncoderConfig,
        *,
        layer_id: int,
        input_dim: int,
    ):
        super().__init__()
        del layer_id
        self.input_dim = int(input_dim)
        self.hidden_dim = int(config.n_embd)
        self.norm1 = SenseVoiceLayerNorm(self.input_dim)
        self.input_proj: nn.Linear | None = None
        if self.input_dim != self.hidden_dim:
            self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.conv = CausalConvolutionModule(
            dim_model=self.hidden_dim,
            kernel_size=int(config.conv_kernel_size),
            dropout=float(config.dropout),
        )
        self.norm2 = SenseVoiceLayerNorm(self.hidden_dim)
        self.feed_forward = SenseVoicePositionwiseFeedForward(self.hidden_dim, int(config.dim_ff), float(config.dropout))
        self.dropout = nn.Dropout(float(config.dropout))

    @staticmethod
    def _length_mask(x: Tensor, lengths: Tensor | None) -> Tensor | None:
        if lengths is None:
            return None
        steps = torch.arange(int(x.size(1)), device=x.device)
        return steps.unsqueeze(0) < lengths.to(device=x.device, dtype=torch.long).unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        *,
        v_first: BidirectionalVFirstState | None = None,
        state: RWKVConformerBlockState | None = None,
        layer_mask: LayerDirectionMask | None = None,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, BidirectionalVFirstState, RWKVConformerBlockState]:
        del layer_mask
        residual = x
        mixed_input = self.norm1(x)
        if self.input_proj is not None:
            mixed_input = self.input_proj(mixed_input)

        mask = self._length_mask(mixed_input, lengths)
        if mask is not None:
            mixed_input = mixed_input * mask.unsqueeze(-1).to(dtype=mixed_input.dtype)
        conv_cache = None if state is None else state.conv_cache
        mixed, next_conv_cache = self.conv(mixed_input, cache=conv_cache)
        if mask is not None:
            mixed = mixed * mask.unsqueeze(-1).to(dtype=mixed.dtype)

        if self.input_dim == self.hidden_dim:
            x = residual + self.dropout(mixed)
        else:
            x = self.dropout(mixed)

        residual = x
        x = residual + self.dropout(self.feed_forward(self.norm2(x)))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)
        if x.dtype == torch.float16:
            clamp_value = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)

        next_v_first = v_first or BidirectionalVFirstState(forward=None, backward=None)
        return x, next_v_first, RWKVConformerBlockState(time_mixer=None, conv_cache=next_conv_cache)


class SenseVoiceRWKVEncoder(nn.Module):
    """SenseVoiceEncoderSmall-shaped encoder with SANM attention replaced by BiRWKV TimeMixer."""

    def __init__(self, config: SenseVoiceRWKVEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed = SenseVoiceSinusoidalPositionEncoder()
        layers: list[SenseVoiceRWKVEncoderLayer] = []
        for layer_id in range(int(config.num_layers)):
            input_dim = int(config.input_dim) if layer_id == 0 else int(config.n_embd)
            layers.append(SenseVoiceRWKVEncoderLayer(config, layer_id=layer_id, input_dim=input_dim))
        self.layers = nn.ModuleList(layers)
        self.main_layer_count = int(config.main_blocks)
        self.after_norm = SenseVoiceLayerNorm(int(config.n_embd))
        self.tp_norm = SenseVoiceLayerNorm(int(config.n_embd))
        self.output_proj: nn.Module
        if int(config.output_dim) == int(config.n_embd):
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(int(config.n_embd), int(config.output_dim))

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
        layer: SenseVoiceRWKVEncoderLayer,
        x: Tensor,
        *,
        v_first: BidirectionalVFirstState | None,
        layer_mask: LayerDirectionMask,
        lengths: Tensor | None,
    ) -> tuple[Tensor, BidirectionalVFirstState]:
        forward_v_first = self._pack_optional_tensor(
            None if v_first is None else v_first.forward,
            reference=x,
        )
        backward_v_first = self._pack_optional_tensor(
            None if v_first is None else v_first.backward,
            reference=x,
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
            x,
            forward_v_first,
            backward_v_first,
            use_reentrant=False,
        )
        return next_x, BidirectionalVFirstState(
            forward=self._unpack_optional_tensor(next_forward_v_first),
            backward=self._unpack_optional_tensor(next_backward_v_first),
        )

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
        *,
        direction_mask: DirectionMask | None = None,
        state: Any = None,
    ) -> tuple[Tensor, Tensor | None, Any]:
        compute_dtype = self._encoder_compute_dtype()
        if compute_dtype is not None and x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        x = x * math.sqrt(float(self.config.n_embd))
        x = self.embed(x)
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp_max(int(x.size(1)))

        if state is None or len(state.block_states) == 0:
            block_states: list[RWKVConformerBlockState | None] = [None] * len(self.layers)
        else:
            if len(state.block_states) != len(self.layers):
                raise ValueError("Encoder state block count does not match the number of SenseVoiceRWKV layers.")
            block_states = state.block_states

        next_states: list[RWKVConformerBlockState | None] = []
        v_first: BidirectionalVFirstState | None = None
        for layer_idx, layer in enumerate(self.layers):
            layer_mask = LayerDirectionMask()
            if direction_mask is not None:
                layer_mask = direction_mask.layer(layer_idx)
            if layer_idx == self.main_layer_count:
                x = self.after_norm(x)
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

        x = self.tp_norm(x)
        x = self.output_proj(x)

        from .rwkv_asr_ctc import RWKVConformerEncoderState

        return x, lengths, RWKVConformerEncoderState(block_states=next_states)

    def _source_layer_prefix(self, layer_idx: int) -> str:
        if layer_idx == 0:
            return "encoders0.0"
        if layer_idx < self.main_layer_count:
            return f"encoders.{layer_idx - 1}"
        return f"tp_encoders.{layer_idx - self.main_layer_count}"

    def load_sensevoice_non_attention_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Any]:
        own_state = self.state_dict()
        source_roots = (
            "",
            "audio_encoder.",
            "encoder.",
            "model.encoder.",
            "model.audio_encoder.",
        )
        loaded: list[str] = []
        skipped: list[str] = []

        def copy_if_matches(own_key: str, source_key: str) -> None:
            own_value = own_state[own_key]
            source_value = None
            for root in source_roots:
                candidate = root + source_key
                if candidate in state_dict:
                    source_value = state_dict[candidate]
                    break
            if source_value is None or tuple(source_value.shape) != tuple(own_value.shape):
                skipped.append(own_key)
                return
            own_value.copy_(source_value.to(dtype=own_value.dtype))
            loaded.append(own_key)

        for layer_idx, layer in enumerate(self.layers):
            prefix = self._source_layer_prefix(layer_idx)
            layer_prefix = f"layers.{layer_idx}"
            suffixes = (
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "feed_forward.w_1.weight",
                "feed_forward.w_1.bias",
                "feed_forward.w_2.weight",
                "feed_forward.w_2.bias",
            )
            for suffix in suffixes:
                copy_if_matches(f"{layer_prefix}.{suffix}", f"{prefix}.{suffix}")
            skipped.extend(
                f"{layer_prefix}.{key}"
                for key in layer.state_dict()
                if key.startswith("time_mixer.") or key.startswith("input_proj.")
            )

        copy_if_matches("after_norm.weight", "after_norm.weight")
        copy_if_matches("after_norm.bias", "after_norm.bias")
        copy_if_matches("tp_norm.weight", "tp_norm.weight")
        copy_if_matches("tp_norm.bias", "tp_norm.bias")
        if isinstance(self.output_proj, nn.Linear):
            skipped.extend(["output_proj.weight", "output_proj.bias"])
        return {"loaded": sorted(set(loaded)), "skipped": sorted(set(skipped))}

    def load_sensevoice_non_attention_checkpoint(self, path: str | Path) -> dict[str, Any]:
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
                report = self.load_sensevoice_non_attention_checkpoint(shard)
                loaded.extend(report["loaded"])
                skipped.extend(report["skipped"])
            loaded_set = set(loaded)
            return {"loaded": sorted(loaded_set), "skipped": sorted(set(skipped) - loaded_set)}
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(checkpoint_path), device="cpu")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        return self.load_sensevoice_non_attention_state_dict(state_dict)

    def load_sensevoice_qkv_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        projection_scale_mode: str = "exact",
    ) -> dict[str, Any]:
        """Warm-start BiRWKV projections from mapped SenseVoice SANM Q/K/V weights."""

        projection_scale_mode = str(projection_scale_mode).lower().strip()
        if projection_scale_mode not in {"exact", "rwkv_norm"}:
            raise ValueError(
                "projection_scale_mode must be 'exact' or 'rwkv_norm', "
                f"got {projection_scale_mode!r}."
            )

        source_roots = (
            "",
            "audio_encoder.",
            "encoder.",
            "model.encoder.",
            "model.audio_encoder.",
        )
        loaded: list[str] = []
        skipped: list[str] = []
        reconstruction_errors: dict[str, float] = {}

        def source_tensor(source_key: str) -> Tensor | None:
            for root in source_roots:
                value = state_dict.get(root + source_key)
                if isinstance(value, Tensor):
                    return value
            return None

        def canonical_svd_basis(weight: Tensor, rank: int) -> Tensor:
            _, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)
            basis = vh[:rank].contiguous()
            max_indices = basis.abs().argmax(dim=1)
            row_indices = torch.arange(int(basis.size(0)), device=basis.device)
            signs = basis[row_indices, max_indices].sign()
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            return basis * signs.unsqueeze(1)

        def projection_for_rwkv(source: Tensor, name: str) -> Tensor:
            if projection_scale_mode == "exact":
                return source
            half_range = 0.05 if name == "key" else 0.5
            target_rms = half_range / math.sqrt(3.0 * float(self.config.n_embd))
            source_rms = source.float().square().mean().sqrt().clamp_min(1.0e-12)
            return source * (target_rms / source_rms)

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                prefix = self._source_layer_prefix(layer_idx)
                qkv_key = f"{prefix}.self_attn.linear_q_k_v.weight"
                output_key = f"{prefix}.self_attn.linear_out.weight"
                qkv_weight = source_tensor(qkv_key)
                output_weight = source_tensor(output_key)
                layer_prefix = f"layers.{layer_idx}.time_mixer"

                if qkv_weight is None or qkv_weight.ndim != 2 or int(qkv_weight.size(0)) != 3 * self.config.n_embd:
                    skipped.extend(
                        f"{layer_prefix}.{direction}.{name}.weight"
                        for direction in ("forward_mixer", "backward_mixer")
                        for name in ("receptance", "key", "value")
                    )
                    continue

                q_weight, k_weight, v_weight = qkv_weight.float().chunk(3, dim=0)
                if layer_idx == 0:
                    if layer.input_proj is None:
                        raise ValueError("The first SenseVoiceRWKV layer requires a 560 -> 512 input projection.")
                    stacked = torch.cat((q_weight, k_weight, v_weight), dim=0)
                    basis = canonical_svd_basis(stacked, int(self.config.n_embd))
                    if tuple(basis.shape) != tuple(layer.input_proj.weight.shape):
                        raise ValueError(
                            "First-layer Nano QKV SVD basis shape does not match the RWKV input projection: "
                            f"{tuple(basis.shape)} != {tuple(layer.input_proj.weight.shape)}"
                        )
                    layer.input_proj.weight.copy_(basis.to(dtype=layer.input_proj.weight.dtype))
                    loaded.append(f"layers.{layer_idx}.input_proj.weight")
                    projected_weights = tuple(weight @ basis.transpose(0, 1) for weight in (q_weight, k_weight, v_weight))
                    for name, source, projected in zip(
                        ("q", "k", "v"),
                        (q_weight, k_weight, v_weight),
                        projected_weights,
                        strict=True,
                    ):
                        reconstructed = projected @ basis
                        relative_error = (reconstructed - source).norm() / source.norm().clamp_min(1.0e-12)
                        reconstruction_errors[name] = float(relative_error.item())
                else:
                    projected_weights = (q_weight, k_weight, v_weight)
                transfer_weights = tuple(
                    projection_for_rwkv(weight, name)
                    for name, weight in zip(
                        ("receptance", "key", "value"),
                        projected_weights,
                        strict=True,
                    )
                )

                for direction_name, mixer in (
                    ("forward_mixer", layer.time_mixer.forward_mixer),
                    ("backward_mixer", layer.time_mixer.backward_mixer),
                ):
                    for target_name, target, source in (
                        ("receptance", mixer.receptance.weight, transfer_weights[0]),
                        ("key", mixer.key.weight, transfer_weights[1]),
                        ("value", mixer.value.weight, transfer_weights[2]),
                    ):
                        if tuple(target.shape) != tuple(source.shape):
                            skipped.append(f"{layer_prefix}.{direction_name}.{target_name}.weight")
                            continue
                        target.copy_(source.to(dtype=target.dtype))
                        loaded.append(f"{layer_prefix}.{direction_name}.{target_name}.weight")

                    output_name = f"{layer_prefix}.{direction_name}.output.weight"
                    if output_weight is None or tuple(mixer.output.weight.shape) != tuple(output_weight.shape):
                        skipped.append(output_name)
                    else:
                        mixer.output.weight.copy_(output_weight.to(dtype=mixer.output.weight.dtype))
                        loaded.append(output_name)

                skipped.extend(
                    (
                        f"{prefix}.self_attn.linear_q_k_v.bias",
                        f"{prefix}.self_attn.linear_out.bias",
                        f"{prefix}.self_attn.fsmn_block.weight",
                    )
                )

        if reconstruction_errors:
            for name, value in reconstruction_errors.items():
                if not math.isfinite(value) or value > 0.25:
                    raise ValueError(
                        f"First-layer Nano {name.upper()} reconstruction error is invalid or too large: {value:.6f}"
                    )
        return {
            "loaded": sorted(set(loaded)),
            "skipped": sorted(set(skipped)),
            "first_layer_reconstruction_errors": reconstruction_errors,
            "projection_scale_mode": projection_scale_mode,
        }


class SenseVoiceConformerConvEncoder(nn.Module):
    """SenseVoiceEncoderSmall-shaped non-attention Conformer-conv control encoder."""

    def __init__(self, config: SenseVoiceRWKVEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed = SenseVoiceSinusoidalPositionEncoder()
        layers: list[SenseVoiceConformerConvEncoderLayer] = []
        for layer_id in range(int(config.num_layers)):
            input_dim = int(config.input_dim) if layer_id == 0 else int(config.n_embd)
            layers.append(SenseVoiceConformerConvEncoderLayer(config, layer_id=layer_id, input_dim=input_dim))
        self.layers = nn.ModuleList(layers)
        self.main_layer_count = int(config.main_blocks)
        self.after_norm = SenseVoiceLayerNorm(int(config.n_embd))
        self.tp_norm = SenseVoiceLayerNorm(int(config.n_embd))
        self.output_proj: nn.Module
        if int(config.output_dim) == int(config.n_embd):
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(int(config.n_embd), int(config.output_dim))

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
        layer: SenseVoiceConformerConvEncoderLayer,
        x: Tensor,
        *,
        lengths: Tensor | None,
    ) -> Tensor:
        def _custom_forward(x_in: Tensor) -> Tensor:
            next_x, _, _ = layer(
                x_in,
                v_first=None,
                state=None,
                layer_mask=None,
                lengths=lengths,
            )
            return next_x

        return activation_checkpoint(
            _custom_forward,
            x,
            use_reentrant=False,
        )

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
        *,
        direction_mask: DirectionMask | None = None,
        state: Any = None,
    ) -> tuple[Tensor, Tensor | None, Any]:
        del direction_mask
        compute_dtype = self._encoder_compute_dtype()
        if compute_dtype is not None and x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)

        x = x * math.sqrt(float(self.config.n_embd))
        x = self.embed(x)
        if lengths is not None:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp_max(int(x.size(1)))

        if state is None or len(state.block_states) == 0:
            block_states: list[RWKVConformerBlockState | None] = [None] * len(self.layers)
        else:
            if len(state.block_states) != len(self.layers):
                raise ValueError("Encoder state block count does not match the number of SenseVoice Conformer-conv layers.")
            block_states = state.block_states

        next_states: list[RWKVConformerBlockState | None] = []
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == self.main_layer_count:
                x = self.after_norm(x)
            if self.gradient_checkpointing and self.training and block_states[layer_idx] is None:
                x = self._forward_layer_checkpointed(layer, x, lengths=lengths)
                next_states.append(None)
            else:
                x, _, next_state = layer(
                    x,
                    v_first=None,
                    state=block_states[layer_idx],
                    layer_mask=None,
                    lengths=lengths,
                )
                next_states.append(next_state)

        x = self.tp_norm(x)
        x = self.output_proj(x)

        from .rwkv_asr_ctc import RWKVConformerEncoderState

        return x, lengths, RWKVConformerEncoderState(block_states=next_states)

    def _source_layer_prefix(self, layer_idx: int) -> str:
        if layer_idx == 0:
            return "encoders0.0"
        if layer_idx < self.main_layer_count:
            return f"encoders.{layer_idx - 1}"
        return f"tp_encoders.{layer_idx - self.main_layer_count}"

    def load_sensevoice_non_attention_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, Any]:
        own_state = self.state_dict()
        source_roots = (
            "",
            "audio_encoder.",
            "encoder.",
            "model.encoder.",
            "model.audio_encoder.",
        )
        loaded: list[str] = []
        skipped: list[str] = []

        def copy_if_matches(own_key: str, source_key: str) -> None:
            own_value = own_state[own_key]
            source_value = None
            for root in source_roots:
                candidate = root + source_key
                if candidate in state_dict:
                    source_value = state_dict[candidate]
                    break
            if source_value is None or tuple(source_value.shape) != tuple(own_value.shape):
                skipped.append(own_key)
                return
            own_value.copy_(source_value.to(dtype=own_value.dtype))
            loaded.append(own_key)

        for layer_idx, layer in enumerate(self.layers):
            prefix = self._source_layer_prefix(layer_idx)
            layer_prefix = f"layers.{layer_idx}"
            suffixes = (
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "feed_forward.w_1.weight",
                "feed_forward.w_1.bias",
                "feed_forward.w_2.weight",
                "feed_forward.w_2.bias",
            )
            for suffix in suffixes:
                copy_if_matches(f"{layer_prefix}.{suffix}", f"{prefix}.{suffix}")
            skipped.extend(
                f"{layer_prefix}.{key}"
                for key in layer.state_dict()
                if key.startswith("conv.") or key.startswith("input_proj.")
            )

        copy_if_matches("after_norm.weight", "after_norm.weight")
        copy_if_matches("after_norm.bias", "after_norm.bias")
        copy_if_matches("tp_norm.weight", "tp_norm.weight")
        copy_if_matches("tp_norm.bias", "tp_norm.bias")
        if isinstance(self.output_proj, nn.Linear):
            skipped.extend(["output_proj.weight", "output_proj.bias"])
        return {"loaded": sorted(set(loaded)), "skipped": sorted(set(skipped))}

    def load_sensevoice_non_attention_checkpoint(self, path: str | Path) -> dict[str, Any]:
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
                report = self.load_sensevoice_non_attention_checkpoint(shard)
                loaded.extend(report["loaded"])
                skipped.extend(report["skipped"])
            loaded_set = set(loaded)
            return {"loaded": sorted(loaded_set), "skipped": sorted(set(skipped) - loaded_set)}
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(checkpoint_path), device="cpu")
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        return self.load_sensevoice_non_attention_state_dict(state_dict)
