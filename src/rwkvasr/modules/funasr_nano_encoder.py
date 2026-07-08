from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FunASRNanoEncoderConfig:
    input_dim: int = 560
    n_embd: int = 512
    output_dim: int = 512
    dim_ff: int = 2048
    num_layers: int = 70
    tp_blocks: int = 20
    attention_heads: int = 4
    dropout: float = 0.0
    attention_dropout: float = 0.0
    kernel_size: int = 11

    @property
    def main_blocks(self) -> int:
        main_blocks = int(self.num_layers) - int(self.tp_blocks)
        if main_blocks < 1:
            raise ValueError("FunASRNanoEncoderConfig requires num_layers > tp_blocks.")
        return main_blocks


class FunASRNanoEncoder(nn.Module):
    """Thin wrapper around FunASR-Nano's native SenseVoiceEncoderSmall audio encoder."""

    def __init__(self, config: FunASRNanoEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        try:
            from funasr.models.sense_voice.model import SenseVoiceEncoderSmall
        except ImportError as exc:  # pragma: no cover - dependency is present in the project env.
            raise ImportError("funasr is required for frontend_type='funasr_nano_encoder'.") from exc

        self.audio_encoder = SenseVoiceEncoderSmall(
            input_size=int(config.input_dim),
            output_size=int(config.n_embd),
            attention_heads=int(config.attention_heads),
            linear_units=int(config.dim_ff),
            num_blocks=int(config.main_blocks),
            tp_blocks=int(config.tp_blocks),
            dropout_rate=float(config.dropout),
            attention_dropout_rate=float(config.attention_dropout),
            kernel_size=int(config.kernel_size),
        )
        self.output_proj: nn.Module
        if int(config.output_dim) == int(config.n_embd):
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(int(config.n_embd), int(config.output_dim))

    @property
    def layers(self) -> list[nn.Module]:
        return [
            *list(self.audio_encoder.encoders0),
            *list(self.audio_encoder.encoders),
            *list(self.audio_encoder.tp_encoders),
        ]

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        # FunASR's native encoder does not expose layer checkpointing; keep the flag
        # visible so config plumbing and diagnostics can report it consistently.
        self.gradient_checkpointing = bool(enabled)

    @staticmethod
    def _module_floating_dtype(module: nn.Module) -> torch.dtype | None:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_floating_point():
                return tensor.dtype
        return None

    def _encoder_compute_dtype(self) -> torch.dtype | None:
        return self._module_floating_dtype(self)

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
        *,
        direction_mask: Any = None,
        state: Any = None,
    ) -> tuple[Tensor, Tensor | None, Any]:
        del direction_mask, state
        compute_dtype = self._encoder_compute_dtype()
        if compute_dtype is not None and x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)
        if lengths is None:
            lengths = torch.full(
                (int(x.size(0)),),
                int(x.size(1)),
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp_min(0).clamp_max(int(x.size(1)))

        # SenseVoiceEncoderSmall multiplies the input in-place before adding
        # sinusoidal positions; clone so callers can safely reuse feature tensors.
        encoded, out_lengths = self.audio_encoder(x.clone(), lengths)
        encoded = self.output_proj(encoded)

        from .rwkv_asr_ctc import RWKVConformerEncoderState

        return encoded, out_lengths.to(device=encoded.device, dtype=torch.long), RWKVConformerEncoderState(
            block_states=[None] * len(self.layers)
        )

    @staticmethod
    def _is_attention_state_key(key: str) -> bool:
        return ".self_attn." in key or key.startswith("self_attn.")

    def load_funasr_nano_encoder_state_dict(
        self,
        state_dict: dict[str, Tensor],
        *,
        load_attention: bool = True,
    ) -> dict[str, Any]:
        own_state = self.audio_encoder.state_dict()
        source_roots = (
            "",
            "audio_encoder.",
            "model.audio_encoder.",
        )
        loaded: list[str] = []
        skipped: list[str] = []
        attention_skipped: list[str] = []
        for own_key, own_value in own_state.items():
            if not load_attention and self._is_attention_state_key(own_key):
                attention_skipped.append(f"audio_encoder.{own_key}")
                continue
            source_value = None
            for root in source_roots:
                candidate = root + own_key
                if candidate in state_dict:
                    source_value = state_dict[candidate]
                    break
            if source_value is None or tuple(source_value.shape) != tuple(own_value.shape):
                skipped.append(f"audio_encoder.{own_key}")
                continue
            own_value.copy_(source_value.to(dtype=own_value.dtype))
            loaded.append(f"audio_encoder.{own_key}")
        if not isinstance(self.output_proj, nn.Identity):
            skipped.extend(f"output_proj.{key}" for key in self.output_proj.state_dict())
        return {
            "loaded": sorted(set(loaded)),
            "skipped": sorted(set(skipped)),
            "attention_skipped": sorted(set(attention_skipped)),
        }

    def load_funasr_nano_encoder_checkpoint(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported FunASR-Nano checkpoint payload: {type(checkpoint)!r}")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state_dict, dict):
            raise ValueError("FunASR-Nano checkpoint has no state dict.")
        return self.load_funasr_nano_encoder_state_dict(state_dict)
