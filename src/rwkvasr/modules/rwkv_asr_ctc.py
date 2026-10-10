from __future__ import annotations

from dataclasses import dataclass, field
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .direction_dropout import DirectionMask, LayerDirectionMask
from .aurwkv_encoder import AuRWKVEncoder, AuRWKVEncoderConfig
from .funasr_nano_encoder import FunASRNanoEncoder, FunASRNanoEncoderConfig
from .funasr_nano_ctc_decoder import NanoCTCTransformerDecoder, NanoCTCTransformerDecoderConfig
from .qwen3_transformer_encoder import Qwen3TransformerEncoder, Qwen3TransformerEncoderConfig
from .rwkv7_bidirectional import BidirectionalVFirstState
from .rwkv7_decoder import (
    RWKV7DecoderConfig,
    RWKV7DecoderLM,
    infer_rwkv7_decoder_config_from_checkpoint,
)
from .rwkv_conformer import (
    RWKVConformerBlock,
    RWKVConformerBlockConfig,
    RWKVConformerBlockState,
)
from .sensevoice_rwkv_encoder import (
    SenseVoiceConformerConvEncoder,
    SenseVoiceRWKVEncoder,
    SenseVoiceRWKVEncoderConfig,
)
from .wenet_frontend import GlobalCMVN, WenetConv2dSubsampling6, load_wenet_cmvn


@dataclass(frozen=True)
class RWKVConformerEncoderConfig:
    input_dim: int
    n_embd: int
    dim_att: int
    dim_ff: int
    num_layers: int
    head_size: int = 64
    backend: str = "native"
    conv_kernel_size: int = 31
    dropout: float = 0.1
    frontend_type: str = "conv2d6"
    output_dim: int | None = None
    aut_downsample_hidden_size: int = 480
    aut_activation_function: str = "gelu"
    aut_activation_dropout: float = 0.0
    aut_max_source_positions: int = 1500
    aut_scale_embedding: bool = False
    aut_conv_chunksize: int = 500
    sensevoice_tp_blocks: int = 20
    cmvn_file: str | None = None
    cmvn_is_json: bool = True

    @property
    def encoder_output_dim(self) -> int:
        return int(self.n_embd if self.output_dim is None else self.output_dim)

    def to_aurwkv_config(self) -> AuRWKVEncoderConfig:
        return AuRWKVEncoderConfig(
            input_dim=self.input_dim,
            n_embd=self.n_embd,
            output_dim=self.encoder_output_dim,
            dim_att=self.dim_att,
            dim_ff=self.dim_ff,
            num_layers=self.num_layers,
            head_size=self.head_size,
            backend=self.backend,
            dropout=self.dropout,
            activation_dropout=self.aut_activation_dropout,
            activation_function=self.aut_activation_function,
            max_source_positions=self.aut_max_source_positions,
            downsample_hidden_size=self.aut_downsample_hidden_size,
            scale_embedding=self.aut_scale_embedding,
            conv_chunksize=self.aut_conv_chunksize,
            cmvn_file=self.cmvn_file,
            cmvn_is_json=self.cmvn_is_json,
        )

    def to_qwen3_transformer_config(self) -> Qwen3TransformerEncoderConfig:
        return Qwen3TransformerEncoderConfig(
            input_dim=self.input_dim,
            n_embd=self.n_embd,
            output_dim=self.encoder_output_dim,
            dim_att=self.dim_att,
            dim_ff=self.dim_ff,
            num_layers=self.num_layers,
            head_size=self.head_size,
            dropout=self.dropout,
            activation_dropout=self.aut_activation_dropout,
            activation_function=self.aut_activation_function,
            max_source_positions=self.aut_max_source_positions,
            downsample_hidden_size=self.aut_downsample_hidden_size,
            scale_embedding=self.aut_scale_embedding,
            conv_chunksize=self.aut_conv_chunksize,
            cmvn_file=self.cmvn_file,
            cmvn_is_json=self.cmvn_is_json,
        )

    def to_sensevoice_rwkv_config(self) -> SenseVoiceRWKVEncoderConfig:
        return SenseVoiceRWKVEncoderConfig(
            input_dim=self.input_dim,
            n_embd=self.n_embd,
            output_dim=self.encoder_output_dim,
            dim_att=self.dim_att,
            dim_ff=self.dim_ff,
            num_layers=self.num_layers,
            tp_blocks=self.sensevoice_tp_blocks,
            head_size=self.head_size,
            backend=self.backend,
            conv_kernel_size=self.conv_kernel_size,
            dropout=self.dropout,
        )

    def to_funasr_nano_encoder_config(self) -> FunASRNanoEncoderConfig:
        return FunASRNanoEncoderConfig(
            input_dim=self.input_dim,
            n_embd=self.n_embd,
            output_dim=self.encoder_output_dim,
            dim_ff=self.dim_ff,
            num_layers=self.num_layers,
            tp_blocks=self.sensevoice_tp_blocks,
            dropout=self.dropout,
        )


@dataclass
class RWKVConformerEncoderState:
    block_states: list[RWKVConformerBlockState | None] = field(default_factory=list)


@dataclass(frozen=True)
class RWKVCTCModelConfig:
    input_dim: int
    n_embd: int
    dim_att: int
    dim_ff: int
    num_layers: int
    vocab_size: int
    feature_extractor_type: str = "wenet_fbank"
    head_size: int = 64
    backend: str = "native"
    conv_kernel_size: int = 31
    dropout: float = 0.1
    blank_id: int = 0
    frontend_type: str = "conv2d6"
    encoder_output_dim: int | None = None
    aut_downsample_hidden_size: int = 480
    aut_activation_function: str = "gelu"
    aut_activation_dropout: float = 0.0
    aut_max_source_positions: int = 1500
    aut_scale_embedding: bool = False
    aut_conv_chunksize: int = 500
    sensevoice_tp_blocks: int = 20
    cmvn_file: str | None = None
    cmvn_is_json: bool = True
    decoder_enabled: bool = False
    decoder_checkpoint_path: str | None = None
    decoder_num_layers: int | None = None
    decoder_n_embd: int | None = None
    decoder_ffn_hidden_size: int | None = None
    decoder_vocab_size: int | None = None
    decoder_head_size: int = 64
    decoder_audio_conditioning: str = "full"
    decoder_prefix_tokens: int = 32
    decoder_loss_chunk_size: int = 1024
    decoder_prompt_before_audio_token_ids: tuple[int, ...] | list[int] = ()
    decoder_prompt_after_audio_token_ids: tuple[int, ...] | list[int] = ()
    decoder_target_suffix_token_ids: tuple[int, ...] | list[int] = ()
    decoder_eos_token_id: int = 0
    ctc_loss_weight: float = 1.0
    decoder_loss_weight: float = 0.0
    ctc_decoder_type: str = "none"
    ctc_decoder_downsample_rate: int = 1
    ctc_decoder_dim: int | None = None
    ctc_decoder_ffn_dim: int = 2048
    ctc_decoder_num_layers: int = 5
    ctc_decoder_attention_heads: int = 8
    ctc_decoder_dropout: float = 0.0
    ctc_decoder_attention_dropout: float = 0.0
    ctc_bridge_type: str = "none"
    ctc_bridge_hidden_dim: int | None = None
    ctc_bridge_dropout: float = 0.0
    ctc_suppressed_token_ids: tuple[int, ...] | list[int] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decoder_prompt_before_audio_token_ids",
            tuple(int(token_id) for token_id in self.decoder_prompt_before_audio_token_ids),
        )
        object.__setattr__(
            self,
            "decoder_prompt_after_audio_token_ids",
            tuple(int(token_id) for token_id in self.decoder_prompt_after_audio_token_ids),
        )
        object.__setattr__(
            self,
            "decoder_target_suffix_token_ids",
            tuple(int(token_id) for token_id in self.decoder_target_suffix_token_ids),
        )
        object.__setattr__(
            self,
            "ctc_suppressed_token_ids",
            tuple(sorted({int(token_id) for token_id in self.ctc_suppressed_token_ids})),
        )

    @property
    def decoder_prompt_token_count(self) -> int:
        return (
            len(tuple(self.decoder_prompt_before_audio_token_ids))
            + len(tuple(self.decoder_prompt_after_audio_token_ids))
            + len(tuple(self.decoder_target_suffix_token_ids))
        )

    @property
    def decoder_text_tokens_per_sample_extra(self) -> int:
        # Audio conditioning length is already represented by the audio-frame
        # side of the batch budget. This only covers non-target text positions.
        return self.decoder_prompt_token_count + 1

    @property
    def ctc_vocab_size(self) -> int:
        return max(int(self.vocab_size), int(self.blank_id) + 1)

    @property
    def resolved_encoder_output_dim(self) -> int:
        return int(self.n_embd if self.encoder_output_dim is None else self.encoder_output_dim)

    def to_encoder_config(self) -> RWKVConformerEncoderConfig:
        return RWKVConformerEncoderConfig(
            input_dim=self.input_dim,
            n_embd=self.n_embd,
            dim_att=self.dim_att,
            dim_ff=self.dim_ff,
            num_layers=self.num_layers,
            head_size=self.head_size,
            backend=self.backend,
            conv_kernel_size=self.conv_kernel_size,
            dropout=self.dropout,
            frontend_type=self.frontend_type,
            output_dim=self.encoder_output_dim,
            aut_downsample_hidden_size=self.aut_downsample_hidden_size,
            aut_activation_function=self.aut_activation_function,
            aut_activation_dropout=self.aut_activation_dropout,
            aut_max_source_positions=self.aut_max_source_positions,
            aut_scale_embedding=self.aut_scale_embedding,
            aut_conv_chunksize=self.aut_conv_chunksize,
            sensevoice_tp_blocks=self.sensevoice_tp_blocks,
            cmvn_file=self.cmvn_file,
            cmvn_is_json=self.cmvn_is_json,
        )


class RWKVConformerEncoder(nn.Module):
    def __init__(self, config: RWKVConformerEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.global_cmvn: nn.Module | None = None
        if config.frontend_type != "aut_rwkv" and config.cmvn_file is not None:
            mean, istd = load_wenet_cmvn(config.cmvn_file, is_json=config.cmvn_is_json)
            self.global_cmvn = GlobalCMVN(mean, istd)
        elif config.frontend_type == "conv2d6":
            # Keep a no-op CMVN module so checkpoints with CMVN buffers load without an external file.
            self.global_cmvn = GlobalCMVN(
                torch.zeros(config.input_dim, dtype=torch.float32),
                torch.ones(config.input_dim, dtype=torch.float32),
            )

        if config.frontend_type == "aut_rwkv":
            self.aut_encoder = AuRWKVEncoder(config.to_aurwkv_config())
            self.frontend = self.aut_encoder
            self.blocks = self.aut_encoder.layers
            return

        if config.frontend_type == "qwen3_transformer":
            self.qwen3_transformer_encoder = Qwen3TransformerEncoder(config.to_qwen3_transformer_config())
            self.frontend = self.qwen3_transformer_encoder
            self.blocks = self.qwen3_transformer_encoder.layers
            return

        if config.frontend_type == "sensevoice_rwkv":
            self.sensevoice_encoder = SenseVoiceRWKVEncoder(config.to_sensevoice_rwkv_config())
            self.frontend = self.sensevoice_encoder
            self.blocks = self.sensevoice_encoder.layers
            return

        if config.frontend_type == "sensevoice_conformer_conv":
            self.sensevoice_conformer_encoder = SenseVoiceConformerConvEncoder(config.to_sensevoice_rwkv_config())
            self.frontend = self.sensevoice_conformer_encoder
            self.blocks = self.sensevoice_conformer_encoder.layers
            return

        if config.frontend_type == "funasr_nano_encoder":
            self.funasr_nano_encoder = FunASRNanoEncoder(config.to_funasr_nano_encoder_config())
            self.frontend = self.funasr_nano_encoder
            self.blocks = self.funasr_nano_encoder.layers
            return

        if config.frontend_type == "linear":
            self.frontend = (
                nn.Identity()
                if config.input_dim == config.n_embd
                else nn.Linear(config.input_dim, config.n_embd)
            )
        elif config.frontend_type == "conv2d6":
            self.frontend = WenetConv2dSubsampling6(config.input_dim, config.n_embd)
        else:
            raise ValueError(f"Unsupported frontend_type: {config.frontend_type}")
        self.blocks = nn.ModuleList(
            [
                RWKVConformerBlock(
                    RWKVConformerBlockConfig(
                        n_embd=config.n_embd,
                        dim_att=config.dim_att,
                        dim_ff=config.dim_ff,
                        n_layer=config.num_layers,
                        layer_id=layer_id,
                        head_size=config.head_size,
                        backend=config.backend,
                        conv_kernel_size=config.conv_kernel_size,
                        dropout=config.dropout,
                    )
                )
                for layer_id in range(config.num_layers)
            ]
        )

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)
        if self.config.frontend_type == "aut_rwkv":
            self.aut_encoder.enable_gradient_checkpointing(enabled)
        if self.config.frontend_type == "qwen3_transformer":
            self.qwen3_transformer_encoder.enable_gradient_checkpointing(enabled)
        if self.config.frontend_type == "sensevoice_rwkv":
            self.sensevoice_encoder.enable_gradient_checkpointing(enabled)
        if self.config.frontend_type == "sensevoice_conformer_conv":
            self.sensevoice_conformer_encoder.enable_gradient_checkpointing(enabled)
        if self.config.frontend_type == "funasr_nano_encoder":
            self.funasr_nano_encoder.enable_gradient_checkpointing(enabled)

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

    @staticmethod
    def _module_floating_dtype(module: nn.Module) -> torch.dtype | None:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.is_floating_point():
                return tensor.dtype
        return None

    def _encoder_compute_dtype(self) -> torch.dtype | None:
        frontend_dtype = self._module_floating_dtype(self.frontend)
        if frontend_dtype is not None:
            return frontend_dtype
        for block in self.blocks:
            block_dtype = self._module_floating_dtype(block)
            if block_dtype is not None:
                return block_dtype
        return None

    def _forward_block_checkpointed(
        self,
        block: RWKVConformerBlock,
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
            next_x, next_v_first, _ = block(
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
        state: RWKVConformerEncoderState | None = None,
    ) -> tuple[Tensor, Tensor | None, RWKVConformerEncoderState]:
        if self.config.frontend_type == "aut_rwkv":
            return self.aut_encoder(
                x,
                lengths,
                direction_mask=direction_mask,
                state=state,
            )
        if self.config.frontend_type == "qwen3_transformer":
            return self.qwen3_transformer_encoder(
                x,
                lengths,
                direction_mask=direction_mask,
                state=state,
            )
        if self.config.frontend_type == "sensevoice_rwkv":
            return self.sensevoice_encoder(
                x,
                lengths,
                direction_mask=direction_mask,
                state=state,
            )
        if self.config.frontend_type == "sensevoice_conformer_conv":
            return self.sensevoice_conformer_encoder(
                x,
                lengths,
                direction_mask=direction_mask,
                state=state,
            )
        if self.config.frontend_type == "funasr_nano_encoder":
            return self.funasr_nano_encoder(
                x,
                lengths,
                direction_mask=direction_mask,
                state=state,
            )

        if self.global_cmvn is not None:
            x = self.global_cmvn(x.float())
        compute_dtype = self._encoder_compute_dtype()
        if compute_dtype is not None and x.dtype != compute_dtype:
            x = x.to(dtype=compute_dtype)
        if self.config.frontend_type == "conv2d6":
            x, lengths = self.frontend(x, lengths)
        else:
            x = self.frontend(x)

        if state is None or len(state.block_states) == 0:
            block_states: list[RWKVConformerBlockState | None] = [None] * len(self.blocks)
        else:
            if len(state.block_states) != len(self.blocks):
                raise ValueError("Encoder state block count does not match the number of blocks.")
            block_states = state.block_states

        next_states: list[RWKVConformerBlockState | None] = []
        v_first: BidirectionalVFirstState | None = None

        for layer_idx, block in enumerate(self.blocks):
            layer_mask = LayerDirectionMask()
            if direction_mask is not None:
                layer_mask = direction_mask.layer(layer_idx)
            if self.gradient_checkpointing and self.training and block_states[layer_idx] is None:
                x, v_first = self._forward_block_checkpointed(
                    block,
                    x,
                    v_first=v_first,
                    layer_mask=layer_mask,
                    lengths=lengths,
                )
                next_states.append(None)
            else:
                x, v_first, next_block_state = block(
                    x,
                    v_first=v_first,
                    state=block_states[layer_idx],
                    layer_mask=layer_mask,
                    lengths=lengths,
                )
                next_states.append(next_block_state)

        return x, lengths, RWKVConformerEncoderState(block_states=next_states)


class CTCEncoderBridge(nn.Module):
    """Optional CTC-path adapter between the ASR encoder and CTC decoder/head."""

    def __init__(
        self,
        dim: int,
        *,
        bridge_type: str = "none",
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.bridge_type = str(bridge_type or "none").lower()
        self.kind = "identity"
        if self.bridge_type in {"", "none", "identity"}:
            self.kind = "identity"
            self.net = nn.Identity()
            return
        if self.bridge_type == "linear":
            self.kind = "linear"
            self.net = nn.Linear(self.dim, self.dim)
            self.reset_parameters()
            return
        if self.bridge_type in {"mlp", "residual_mlp"}:
            self.kind = "residual_mlp"
            hidden_dim = int(hidden_dim or self.dim * 4)
            if hidden_dim <= 0:
                raise ValueError("ctc_bridge_hidden_dim must be positive.")
            self.norm = nn.LayerNorm(self.dim)
            self.up = nn.Linear(self.dim, hidden_dim)
            self.down = nn.Linear(hidden_dim, self.dim)
            self.dropout = nn.Dropout(float(dropout))
            self.activation = nn.GELU()
            self.reset_parameters()
            return
        if self.bridge_type in {"context_residual_mlp", "temporal_residual_mlp", "conv_residual_mlp"}:
            self.kind = "context_residual_mlp"
            hidden_dim = int(hidden_dim or self.dim * 4)
            if hidden_dim <= 0:
                raise ValueError("ctc_bridge_hidden_dim must be positive.")
            self.temporal_norm = nn.LayerNorm(self.dim)
            self.temporal = nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=5,
                padding=2,
                groups=self.dim,
            )
            self.ffn_norm = nn.LayerNorm(self.dim)
            self.up = nn.Linear(self.dim, hidden_dim)
            self.down = nn.Linear(hidden_dim, self.dim)
            self.dropout = nn.Dropout(float(dropout))
            self.activation = nn.GELU()
            self.reset_parameters()
            return
        if self.bridge_type.startswith("nano_encoder_tail") or self.bridge_type.startswith("nano_tp_tail"):
            self.kind = "nano_encoder_tail"
            if self.dim <= 0:
                raise ValueError("nano encoder tail bridge requires a positive dim.")
            tail_layers = self._parse_nano_tail_layers(self.bridge_type, default=5)
            if tail_layers <= 0 or tail_layers > 20:
                raise ValueError("nano_encoder_tailN requires 1 <= N <= 20.")
            if self.dim % 4 != 0:
                raise ValueError("nano encoder tail bridge requires dim divisible by 4 attention heads.")
            self.tail_layers = int(tail_layers)
            try:
                from funasr.models.sense_voice.model import SenseVoiceEncoderSmall
            except ImportError as exc:  # pragma: no cover - dependency is present in the project env.
                raise ImportError("funasr is required for ctc_bridge_type='nano_encoder_tailN'.") from exc
            assistant = SenseVoiceEncoderSmall(
                input_size=self.dim,
                output_size=self.dim,
                attention_heads=4,
                linear_units=int(hidden_dim or self.dim * 4),
                num_blocks=1,
                tp_blocks=self.tail_layers,
                dropout_rate=float(dropout),
                attention_dropout_rate=float(dropout),
                kernel_size=11,
            )
            self.blocks = nn.ModuleList(list(assistant.tp_encoders))
            self.final_norm = assistant.tp_norm
            self.source_block_offset = 20 - self.tail_layers
            return
        raise ValueError(f"Unsupported ctc_bridge_type: {bridge_type}")

    @staticmethod
    def _parse_nano_tail_layers(bridge_type: str, *, default: int) -> int:
        match = re.search(r"(?:tail|tail_)(\d+)$", str(bridge_type))
        if match is None:
            return int(default)
        return int(match.group(1))

    def reset_parameters(self) -> None:
        if getattr(self, "kind", "") == "linear":
            linear = self.net
            if not isinstance(linear, nn.Linear):
                raise TypeError("linear bridge expected nn.Linear")
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)
            return
        if getattr(self, "kind", "") == "residual_mlp":
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)
            return
        if getattr(self, "kind", "") == "context_residual_mlp":
            nn.init.zeros_(self.temporal.weight)
            nn.init.zeros_(self.temporal.bias)
            nn.init.zeros_(self.down.weight)
            nn.init.zeros_(self.down.bias)

    @staticmethod
    def _length_mask(x: Tensor, lengths: Tensor | None) -> Tensor | None:
        if lengths is None:
            return None
        steps = torch.arange(int(x.size(1)), device=x.device)
        return (steps.unsqueeze(0) < lengths.to(device=x.device).unsqueeze(1)).unsqueeze(-1)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        if self.kind == "identity":
            return x
        if self.kind == "linear":
            return self.net(x)
        if self.kind == "context_residual_mlp":
            mask = self._length_mask(x, lengths)
            residual = x
            y = self.temporal_norm(x)
            if mask is not None:
                y = y * mask.to(dtype=y.dtype)
            y = self.temporal(y.transpose(1, 2)).transpose(1, 2)
            if mask is not None:
                y = y * mask.to(dtype=y.dtype)
            x = residual + y
            y = self.ffn_norm(x)
            y = self.up(y)
            y = self.activation(y)
            y = self.dropout(y)
            y = self.down(y)
            x = x + y
            if mask is not None:
                x = torch.where(mask, x, residual)
            return x
        if self.kind == "nano_encoder_tail":
            mask = None
            if lengths is not None:
                steps = torch.arange(int(x.size(1)), device=x.device)
                mask = steps.unsqueeze(0) < lengths.to(device=x.device, dtype=torch.long).unsqueeze(1)
                mask = mask[:, None, :]
            for block in self.blocks:
                block_out = block(x, mask)
                if not isinstance(block_out, tuple) or len(block_out) < 2:
                    raise RuntimeError("FunASR Nano encoder tail block returned an unexpected payload.")
                x = block_out[0]
                mask = block_out[1]
            x = self.final_norm(x)
            return x
        residual = x
        x = self.norm(x)
        x = self.up(x)
        x = self.activation(x)
        x = self.dropout(x)
        return residual + self.down(x)

    def load_funasr_nano_bridge_state_dict(self, state_dict: dict[str, Tensor]) -> dict[str, list[str]]:
        if self.kind != "nano_encoder_tail":
            return {"loaded": [], "skipped": []}
        loaded: list[str] = []
        skipped: list[str] = []
        source_roots = ("", "audio_encoder.", "model.audio_encoder.")
        for block_index, block in enumerate(self.blocks):
            source_index = int(self.source_block_offset) + int(block_index)
            own_state = block.state_dict()
            for own_key, own_value in own_state.items():
                source_value = None
                for root in source_roots:
                    candidate = f"{root}tp_encoders.{source_index}.{own_key}"
                    if candidate in state_dict:
                        source_value = state_dict[candidate]
                        break
                target_name = f"blocks.{block_index}.{own_key}"
                if source_value is None or tuple(source_value.shape) != tuple(own_value.shape):
                    skipped.append(target_name)
                    continue
                own_value.copy_(source_value.to(dtype=own_value.dtype))
                loaded.append(target_name)
        for own_key, own_value in self.final_norm.state_dict().items():
            source_value = None
            for root in source_roots:
                candidate = f"{root}tp_norm.{own_key}"
                if candidate in state_dict:
                    source_value = state_dict[candidate]
                    break
            target_name = f"final_norm.{own_key}"
            if source_value is None or tuple(source_value.shape) != tuple(own_value.shape):
                skipped.append(target_name)
                continue
            own_value.copy_(source_value.to(dtype=own_value.dtype))
            loaded.append(target_name)
        return {"loaded": sorted(set(loaded)), "skipped": sorted(set(skipped))}


class RWKVCTCModel(nn.Module):
    def __init__(self, config: RWKVCTCModelConfig):
        super().__init__()
        self.config = config
        self.encoder = RWKVConformerEncoder(config.to_encoder_config())
        encoder_output_dim = int(config.resolved_encoder_output_dim)
        self.ctc_bridge = CTCEncoderBridge(
            encoder_output_dim,
            bridge_type=config.ctc_bridge_type,
            hidden_dim=config.ctc_bridge_hidden_dim,
            dropout=config.ctc_bridge_dropout,
        )
        self.ctc_decoder: NanoCTCTransformerDecoder | None = None
        ctc_head_input_dim = encoder_output_dim
        ctc_decoder_type = str(config.ctc_decoder_type or "none").lower()
        if ctc_decoder_type not in {"", "none"}:
            if ctc_decoder_type not in {"funasr_nano_transformer", "nano_transformer"}:
                raise ValueError(f"Unsupported ctc_decoder_type: {config.ctc_decoder_type}")
            ctc_decoder_dim = int(config.ctc_decoder_dim or encoder_output_dim)
            self.ctc_decoder = NanoCTCTransformerDecoder(
                NanoCTCTransformerDecoderConfig(
                    downsample_rate=int(config.ctc_decoder_downsample_rate),
                    encoder_dim=encoder_output_dim,
                    hidden_dim=ctc_decoder_dim,
                    ffn_dim=int(config.ctc_decoder_ffn_dim),
                    num_layers=int(config.ctc_decoder_num_layers),
                    attention_heads=int(config.ctc_decoder_attention_heads),
                    dropout_rate=float(config.ctc_decoder_dropout),
                    attention_dropout_rate=float(config.ctc_decoder_attention_dropout),
                )
            )
            ctc_head_input_dim = ctc_decoder_dim
        self.ctc_head = nn.Linear(ctc_head_input_dim, config.ctc_vocab_size)
        suppressed_token_ids = [
            int(token_id)
            for token_id in config.ctc_suppressed_token_ids
            if 0 <= int(token_id) < int(config.ctc_vocab_size) and int(token_id) != int(config.blank_id)
        ]
        self.register_buffer(
            "ctc_suppressed_token_ids",
            torch.tensor(sorted(set(suppressed_token_ids)), dtype=torch.long),
            persistent=False,
        )
        self.decoder: RWKV7DecoderLM | None = None
        self.decoder_prefix_proj: nn.Module | None = None
        self.decoder_bos: nn.Parameter | None = None
        if config.decoder_enabled:
            if str(config.decoder_audio_conditioning) != "full":
                raise ValueError("Only decoder_audio_conditioning='full' is supported.")
            decoder_config = self._resolve_decoder_config(config)
            self.decoder = RWKV7DecoderLM(decoder_config)
            if config.decoder_checkpoint_path is not None:
                self.decoder.load_official_checkpoint(config.decoder_checkpoint_path)
            if encoder_output_dim == int(decoder_config.n_embd):
                self.decoder_prefix_proj = nn.Identity()
            else:
                self.decoder_prefix_proj = nn.Linear(encoder_output_dim, decoder_config.n_embd, bias=False)
            self.decoder_bos = nn.Parameter(torch.zeros(1, 1, decoder_config.n_embd))

    @staticmethod
    def _resolve_decoder_config(config: RWKVCTCModelConfig) -> RWKV7DecoderConfig:
        inferred: RWKV7DecoderConfig | None = None
        if config.decoder_checkpoint_path is not None:
            inferred = infer_rwkv7_decoder_config_from_checkpoint(config.decoder_checkpoint_path)
        return RWKV7DecoderConfig(
            vocab_size=int(
                config.decoder_vocab_size
                or (inferred.vocab_size if inferred is not None else config.vocab_size)
            ),
            n_embd=int(config.decoder_n_embd or (inferred.n_embd if inferred is not None else config.n_embd)),
            num_layers=int(config.decoder_num_layers or (inferred.num_layers if inferred is not None else config.num_layers)),
            head_size=int(config.decoder_head_size or (inferred.head_size if inferred is not None else config.head_size)),
            backend="native",
            ffn_hidden_size=int(
                config.decoder_ffn_hidden_size
                or (inferred.ffn_hidden_size if inferred is not None and inferred.ffn_hidden_size is not None else 0)
                or (config.decoder_n_embd or (inferred.n_embd if inferred is not None else config.n_embd)) * 4
            ),
        )

    def load_funasr_nano_ctc_checkpoint(
        self,
        path: str,
        *,
        load_ctc_decoder: bool = True,
        load_ctc_head: bool = True,
        load_encoder: bool = False,
        load_encoder_attention: bool = True,
        load_rwkv_encoder_from_qkv: bool = False,
        rwkv_qkv_projection_scale_mode: str = "exact",
        teacher_blank_id: int = 60514,
        project_ignored_token_ids: tuple[int, ...] | list[int] = (60514,),
        blank_bias_delta: float = 0.0,
    ) -> dict[str, object]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported FunASR-Nano checkpoint payload: {type(checkpoint)!r}")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state_dict, dict):
            raise ValueError("FunASR-Nano checkpoint has no state dict.")

        report: dict[str, object] = {
            "encoder_loaded": 0,
            "encoder_skipped": 0,
            "encoder_attention_skipped": 0,
            "rwkv_encoder_loaded": 0,
            "rwkv_encoder_skipped": 0,
            "rwkv_encoder_first_layer_reconstruction_errors": {},
            "rwkv_encoder_qkv_projection_scale_mode": str(rwkv_qkv_projection_scale_mode),
            "ctc_bridge_loaded": 0,
            "ctc_bridge_skipped": 0,
            "ctc_decoder_loaded": 0,
            "ctc_head_loaded_rows": 0,
            "ctc_head_ignored_rows": [],
            "ctc_head_blank_bias_delta": float(blank_bias_delta),
        }
        bridge_loader = getattr(self.ctc_bridge, "load_funasr_nano_bridge_state_dict", None)
        if callable(bridge_loader):
            bridge_report = bridge_loader(state_dict)
            loaded = bridge_report.get("loaded", []) if isinstance(bridge_report, dict) else []
            skipped = bridge_report.get("skipped", []) if isinstance(bridge_report, dict) else []
            report["ctc_bridge_loaded"] = len(loaded)
            report["ctc_bridge_skipped"] = len(skipped)
        if load_encoder:
            funasr_encoder = getattr(self.encoder, "funasr_nano_encoder", None)
            if funasr_encoder is None:
                raise ValueError("frontend_type='funasr_nano_encoder' is required to load Nano audio_encoder weights.")
            encoder_report = funasr_encoder.load_funasr_nano_encoder_state_dict(
                state_dict,
                load_attention=bool(load_encoder_attention),
            )
            report["encoder_loaded"] = len(encoder_report["loaded"])
            report["encoder_skipped"] = len(encoder_report["skipped"])
            report["encoder_attention_skipped"] = len(encoder_report["attention_skipped"])
        if load_rwkv_encoder_from_qkv:
            sensevoice_encoder = getattr(self.encoder, "sensevoice_encoder", None)
            if sensevoice_encoder is None:
                raise ValueError(
                    "frontend_type='sensevoice_rwkv' is required to initialize BiRWKV from Nano QKV weights."
                )
            non_attention_report = sensevoice_encoder.load_sensevoice_non_attention_state_dict(state_dict)
            qkv_report = sensevoice_encoder.load_sensevoice_qkv_state_dict(
                state_dict,
                projection_scale_mode=rwkv_qkv_projection_scale_mode,
            )
            report["rwkv_encoder_loaded"] = len(
                set(non_attention_report["loaded"]) | set(qkv_report["loaded"])
            )
            report["rwkv_encoder_skipped"] = len(
                set(non_attention_report["skipped"]) | set(qkv_report["skipped"])
            )
            report["rwkv_encoder_first_layer_reconstruction_errors"] = qkv_report[
                "first_layer_reconstruction_errors"
            ]
            report["rwkv_encoder_qkv_projection_scale_mode"] = qkv_report[
                "projection_scale_mode"
            ]
        if load_ctc_decoder:
            if self.ctc_decoder is None:
                raise ValueError("ctc_decoder_type must be enabled before loading Nano ctc_decoder weights.")
            decoder_state = {
                key.removeprefix("ctc_decoder."): value
                for key, value in state_dict.items()
                if isinstance(key, str) and key.startswith("ctc_decoder.")
            }
            if not decoder_state:
                raise ValueError(f"No ctc_decoder.* weights found in {path}")
            self.ctc_decoder.load_state_dict(decoder_state, strict=True)
            report["ctc_decoder_loaded"] = len(decoder_state)

        if load_ctc_head:
            teacher_weight = state_dict.get("ctc.ctc_lo.weight")
            teacher_bias = state_dict.get("ctc.ctc_lo.bias")
            if not isinstance(teacher_weight, Tensor) or not isinstance(teacher_bias, Tensor):
                raise ValueError(f"No ctc.ctc_lo weights found in {path}")
            if int(teacher_weight.size(1)) != int(self.ctc_head.weight.size(1)):
                raise ValueError(
                    "Nano CTC head input dim does not match project CTC head: "
                    f"{int(teacher_weight.size(1))} != {int(self.ctc_head.weight.size(1))}"
                )
            project_blank_id = int(self.config.blank_id)
            teacher_blank_id = int(teacher_blank_id)
            ignored_rows: list[int] = []
            with torch.no_grad():
                self.ctc_head.weight.zero_()
                self.ctc_head.bias.fill_(-1.0e4)
                loaded_rows = 0
                for teacher_id in range(int(teacher_weight.size(0))):
                    project_id = project_blank_id if teacher_id == teacher_blank_id else teacher_id
                    if not (0 <= project_id < int(self.ctc_head.weight.size(0))):
                        continue
                    self.ctc_head.weight[project_id].copy_(
                        teacher_weight[teacher_id].to(dtype=self.ctc_head.weight.dtype)
                    )
                    self.ctc_head.bias[project_id].copy_(
                        teacher_bias[teacher_id].to(dtype=self.ctc_head.bias.dtype)
                    )
                    loaded_rows += 1
                for ignored_id in project_ignored_token_ids:
                    ignored_id = int(ignored_id)
                    if ignored_id == project_blank_id:
                        continue
                    if 0 <= ignored_id < int(self.ctc_head.weight.size(0)):
                        self.ctc_head.weight[ignored_id].zero_()
                        self.ctc_head.bias[ignored_id].fill_(-1.0e4)
                        ignored_rows.append(ignored_id)
                if float(blank_bias_delta) != 0.0:
                    if not (0 <= project_blank_id < int(self.ctc_head.bias.size(0))):
                        raise ValueError(
                            f"Project blank id {project_blank_id} is outside CTC head bias size "
                            f"{int(self.ctc_head.bias.size(0))}"
                        )
                    self.ctc_head.bias[project_blank_id].add_(
                        self.ctc_head.bias.new_tensor(float(blank_bias_delta))
                    )
            report["ctc_head_loaded_rows"] = loaded_rows
            report["ctc_head_ignored_rows"] = ignored_rows
        return report

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.enable_gradient_checkpointing(enabled)

    def ctc_features_from_encoded(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        ctc_encoded, ctc_encoded_lengths = self.ctc_encoder_features_from_encoded(encoded, encoded_lengths)
        return self.ctc_features_from_ctc_encoded(ctc_encoded, ctc_encoded_lengths)

    def ctc_encoder_features_from_encoded(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        return self.ctc_bridge(encoded, encoded_lengths), encoded_lengths

    def ctc_features_from_ctc_encoded(
        self,
        ctc_encoded: Tensor,
        ctc_encoded_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if self.ctc_decoder is None:
            return ctc_encoded, ctc_encoded_lengths
        return self.ctc_decoder(ctc_encoded, ctc_encoded_lengths)

    def ctc_logits_from_encoded(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        ctc_features, ctc_lengths = self.ctc_features_from_encoded(encoded, encoded_lengths)
        return self.apply_ctc_logit_mask(self.ctc_head(ctc_features)), ctc_lengths

    def apply_ctc_logit_mask(self, logits: Tensor) -> Tensor:
        suppressed = self.ctc_suppressed_token_ids
        if int(suppressed.numel()) == 0:
            return logits
        masked = logits.clone()
        masked.index_fill_(-1, suppressed.to(device=masked.device), masked.new_tensor(-1.0e4))
        return masked

    def forward(
        self,
        features: Tensor,
        feature_lengths: Tensor | None = None,
        *,
        direction_mask: DirectionMask | None = None,
        state: RWKVConformerEncoderState | None = None,
    ) -> tuple[Tensor, Tensor | None, RWKVConformerEncoderState]:
        encoded, encoded_lengths, next_state = self.encoder(
            features,
            lengths=feature_lengths,
            direction_mask=direction_mask,
            state=state,
        )
        logits, logit_lengths = self.ctc_logits_from_encoded(encoded, encoded_lengths)
        return logits, logit_lengths, next_state

    @staticmethod
    def _packed_targets_to_padded(
        targets: Tensor,
        target_lengths: Tensor,
        *,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ) -> tuple[Tensor, Tensor]:
        lengths = target_lengths.to(dtype=torch.long)
        batch_size = int(lengths.numel())
        max_len = int(lengths.max().item()) if batch_size > 0 else 0
        input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long, device=targets.device)
        target_ids = torch.full((batch_size, max_len), int(ignore_index), dtype=torch.long, device=targets.device)
        offset = 0
        for sample_idx, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            sample = targets[offset : offset + length]
            target_ids[sample_idx, :length] = sample
            if length > 1:
                input_ids[sample_idx, 1:length] = sample[:-1]
            offset += length
        return input_ids, target_ids

    def _project_decoder_audio_context(
        self,
        encoded: Tensor,
    ) -> Tensor:
        if self.decoder is None or self.decoder_prefix_proj is None:
            raise RuntimeError("Decoder audio conditioning is only available when decoder_enabled=True.")
        return self.decoder_prefix_proj(encoded)

    @staticmethod
    def _int_token_tuple(value: tuple[int, ...] | list[int]) -> tuple[int, ...]:
        return tuple(int(token_id) for token_id in value)

    def _decoder_static_token_embeds(
        self,
        token_ids: tuple[int, ...],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder token embeddings requested but decoder is disabled.")
        hidden_size = int(self.decoder.hidden_size)
        if not token_ids:
            return torch.empty(batch_size, 0, hidden_size, device=device, dtype=dtype)
        ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.decoder.emb(ids).to(dtype=dtype)

    def _decoder_packed_token_embeds(
        self,
        token_ids: Tensor,
        token_lengths: Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if self.decoder is None:
            raise RuntimeError("Decoder token embeddings requested but decoder is disabled.")
        lengths = token_lengths.to(device=device, dtype=torch.long).clamp_min(0)
        if int(lengths.numel()) != int(batch_size):
            raise ValueError("Packed decoder prompt lengths must match batch size.")
        max_len = int(lengths.max().item()) if batch_size > 0 else 0
        hidden_size = int(self.decoder.hidden_size)
        if max_len <= 0:
            return torch.empty(batch_size, 0, hidden_size, device=device, dtype=dtype), lengths
        output = torch.zeros(batch_size, max_len, hidden_size, device=device, dtype=dtype)
        tokens = token_ids.to(device=device, dtype=torch.long)
        offset = 0
        for sample_idx, length in enumerate(lengths.tolist()):
            length = int(length)
            if length <= 0:
                continue
            sample_ids = tokens[offset : offset + length].unsqueeze(0)
            output[sample_idx, :length] = self.decoder.emb(sample_ids).squeeze(0).to(dtype=dtype)
            offset += length
        return output, lengths

    def _decoder_template_context_embeds(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        audio = self._project_decoder_audio_context(encoded)
        if encoded_lengths is None:
            encoded_lengths = torch.full(
                (encoded.size(0),),
                int(encoded.size(1)),
                dtype=torch.long,
                device=encoded.device,
            )
        audio_lengths = encoded_lengths.to(dtype=torch.long, device=encoded.device).clamp(
            min=1,
            max=int(encoded.size(1)),
        )
        after_ids = self._int_token_tuple(self.config.decoder_prompt_after_audio_token_ids)
        if decoder_prompt_before_audio is not None and decoder_prompt_before_audio_lengths is not None:
            before, before_lengths = self._decoder_packed_token_embeds(
                decoder_prompt_before_audio,
                decoder_prompt_before_audio_lengths,
                batch_size=int(audio.size(0)),
                device=audio.device,
                dtype=audio.dtype,
            )
        else:
            before_ids = self._int_token_tuple(self.config.decoder_prompt_before_audio_token_ids)
            before = self._decoder_static_token_embeds(
                before_ids,
                batch_size=int(audio.size(0)),
                device=audio.device,
                dtype=audio.dtype,
            )
            before_lengths = torch.full(
                (int(audio.size(0)),),
                int(before.size(1)),
                dtype=torch.long,
                device=audio.device,
            )
        after = self._decoder_static_token_embeds(
            after_ids,
            batch_size=int(audio.size(0)),
            device=audio.device,
            dtype=audio.dtype,
        )
        after_len = int(after.size(1))
        context_lengths = audio_lengths + before_lengths + after_len
        max_context_len = int(context_lengths.max().item())
        if max_context_len <= 0:
            raise ValueError("Decoder template context must contain prompt or audio positions.")

        context = audio.new_zeros(int(audio.size(0)), max_context_len, int(audio.size(-1)))
        for sample_idx, audio_len in enumerate(audio_lengths.tolist()):
            cursor = 0
            before_len = int(before_lengths[sample_idx].item())
            if before_len > 0:
                context[sample_idx, cursor : cursor + before_len] = before[sample_idx, :before_len]
                cursor += before_len
            context[sample_idx, cursor : cursor + audio_len] = audio[sample_idx, :audio_len]
            cursor += audio_len
            if after_len > 0:
                context[sample_idx, cursor : cursor + after_len] = after[sample_idx]
        return context, context_lengths

    def _normalize_decoder_target_sequence(self, sample: Tensor) -> list[int]:
        eos_token_id = int(self.config.decoder_eos_token_id)
        token_ids = [int(token_id) for token_id in sample.tolist()]
        if token_ids and token_ids[-1] == eos_token_id:
            token_ids = token_ids[:-1]
        token_ids.extend(self._int_token_tuple(self.config.decoder_target_suffix_token_ids))
        token_ids.append(eos_token_id)
        return token_ids

    def _packed_targets_to_decoder_sequences(
        self,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> list[list[int]]:
        lengths = target_lengths.to(dtype=torch.long)
        sequences: list[list[int]] = []
        offset = 0
        for length in lengths.tolist():
            sample = targets[offset : offset + length]
            sequences.append(self._normalize_decoder_target_sequence(sample))
            offset += length
        return sequences

    def _decoder_template_hidden_and_labels(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        token_sequences: list[list[int]],
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.decoder is None:
            raise RuntimeError("Decoder template path requested but decoder is disabled.")
        if encoded.size(0) != len(token_sequences):
            raise ValueError("Encoded batch size and token sequence count must match.")
        context, context_lengths = self._decoder_template_context_embeds(
            encoded,
            encoded_lengths,
            decoder_prompt_before_audio=decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=decoder_prompt_before_audio_lengths,
        )
        batch_size = int(context.size(0))
        max_target_len = max((len(sequence) for sequence in token_sequences), default=0)
        if max_target_len <= 0:
            raise ValueError("Decoder AR path received no target tokens.")

        input_ids = torch.zeros((batch_size, max_target_len), dtype=torch.long, device=encoded.device)
        for sequence_idx, sequence in enumerate(token_sequences):
            input_ids[sequence_idx, : len(sequence)] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=encoded.device,
            )
        target_embeds = self.decoder.emb(input_ids).to(dtype=context.dtype)

        max_seq_len = int((context_lengths + max_target_len).max().item())
        full_embeds = context.new_zeros(batch_size, max_seq_len, int(context.size(-1)))
        target_ids = torch.full(
            (batch_size, max_seq_len),
            -100,
            dtype=torch.long,
            device=encoded.device,
        )
        for sequence_idx, sequence in enumerate(token_sequences):
            context_len = int(context_lengths[sequence_idx].item())
            target_len = len(sequence)
            full_embeds[sequence_idx, :context_len] = context[sequence_idx, :context_len]
            full_embeds[sequence_idx, context_len : context_len + target_len] = target_embeds[
                sequence_idx,
                :target_len,
            ]
            target_ids[sequence_idx, context_len : context_len + target_len] = torch.tensor(
                sequence,
                dtype=torch.long,
                device=encoded.device,
            )
        hidden, _ = self.decoder.forward_hidden_embeds(full_embeds)
        return hidden, target_ids

    def _decoder_template_logits_and_labels(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        token_sequences: list[list[int]],
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.decoder is None:
            raise RuntimeError("Decoder template path requested but decoder is disabled.")
        hidden, target_ids = self._decoder_template_hidden_and_labels(
            encoded,
            encoded_lengths,
            token_sequences,
            decoder_prompt_before_audio=decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=decoder_prompt_before_audio_lengths,
        )
        return self.decoder.head(hidden), target_ids

    def decoder_sequence_scores(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        token_sequences: list[list[int]],
        *,
        normalize_by_length: bool = True,
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
    ) -> Tensor:
        if self.decoder is None or self.decoder_bos is None:
            raise RuntimeError("Decoder scoring requested but decoder is disabled.")
        if encoded.size(0) != 1:
            raise ValueError("decoder_sequence_scores currently expects a single encoded sample.")
        num_sequences = len(token_sequences)
        if num_sequences == 0:
            return torch.empty(0, device=encoded.device, dtype=encoded.dtype)

        lengths = [len(sequence) for sequence in token_sequences]
        max_len = max(lengths) if lengths else 0
        if max_len <= 0:
            return torch.zeros(num_sequences, device=encoded.device, dtype=torch.float32)
        expanded_encoded = encoded.expand(num_sequences, -1, -1).contiguous()
        expanded_lengths = None
        if encoded_lengths is not None:
            expanded_lengths = encoded_lengths.expand(num_sequences).contiguous()
        expanded_prompt = None
        expanded_prompt_lengths = None
        if decoder_prompt_before_audio is not None and decoder_prompt_before_audio_lengths is not None:
            prompt_len = int(decoder_prompt_before_audio_lengths[0].item())
            sample_prompt = decoder_prompt_before_audio[:prompt_len].to(device=encoded.device, dtype=torch.long)
            expanded_prompt = sample_prompt.repeat(num_sequences)
            expanded_prompt_lengths = torch.full(
                (num_sequences,),
                prompt_len,
                dtype=torch.long,
                device=encoded.device,
            )
        normalized_sequences = [
            self._normalize_decoder_target_sequence(
                torch.tensor(sequence, dtype=torch.long, device=encoded.device)
            )
            for sequence in token_sequences
        ]
        hidden, target_ids = self._decoder_template_hidden_and_labels(
            expanded_encoded,
            expanded_lengths,
            normalized_sequences,
            decoder_prompt_before_audio=expanded_prompt,
            decoder_prompt_before_audio_lengths=expanded_prompt_lengths,
        )
        shifted_hidden = hidden[:, :-1, :]
        shifted_targets = target_ids[:, 1:]
        log_probs = self.decoder.head(shifted_hidden).float().log_softmax(dim=-1)
        gather_index = shifted_targets.clamp_min(0).unsqueeze(-1)
        gathered = torch.gather(log_probs, dim=-1, index=gather_index).squeeze(-1)
        valid = shifted_targets.ne(-100)
        scores = (gathered * valid).sum(dim=1)
        if normalize_by_length:
            valid_count = valid.sum(dim=1).clamp_min(1)
            scores = scores / valid_count
        return scores

    @staticmethod
    def _filter_decoder_sampling_logits(
        logits: Tensor,
        *,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tensor:
        filtered = logits
        if top_k > 0:
            top_k_value = max(1, int(top_k))
            top_k_value = min(filtered.size(-1), top_k_value)
            cutoff = torch.topk(filtered, k=top_k_value, dim=-1).values[:, -1:]
            filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
        if top_p < 1.0:
            probs = filtered.float().softmax(dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            remove_sorted = torch.cumsum(sorted_probs, dim=-1) > float(top_p)
            remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
            remove_sorted[:, 0] = False
            remove_mask = torch.zeros_like(remove_sorted, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, remove_sorted)
            filtered = filtered.masked_fill(remove_mask, float("-inf"))
        return filtered

    def decoder_greedy_decode(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        *,
        eos_token_id: int = 0,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
    ) -> tuple[list[list[int]], Tensor, list[bool]]:
        if self.decoder is None or self.decoder_bos is None:
            raise RuntimeError("Decoder generation requested but decoder is disabled.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if do_sample:
            if not (temperature > 0):
                raise ValueError("temperature must be > 0 when do_sample=True.")
            if top_k < 0:
                raise ValueError("top_k must be >= 0.")
            if not (0.0 < top_p <= 1.0):
                raise ValueError("top_p must be in the interval (0.0, 1.0].")
        if encoded_lengths is None:
            encoded_lengths = torch.full(
                (encoded.size(0),),
                int(encoded.size(1)),
                dtype=torch.long,
                device=encoded.device,
            )

        outputs: list[list[int]] = []
        avg_scores: list[float] = []
        eos_emitted: list[bool] = []
        for sample_idx in range(encoded.size(0)):
            sample_length = max(1, min(int(encoded_lengths[sample_idx].item()), int(encoded.size(1))))
            sample_encoded = encoded[sample_idx : sample_idx + 1, :sample_length, :]
            sample_lengths = encoded_lengths[sample_idx : sample_idx + 1]
            sample_prompt = None
            sample_prompt_lengths = None
            if decoder_prompt_before_audio is not None and decoder_prompt_before_audio_lengths is not None:
                prompt_start = int(decoder_prompt_before_audio_lengths[:sample_idx].sum().item())
                prompt_len = int(decoder_prompt_before_audio_lengths[sample_idx].item())
                sample_prompt = decoder_prompt_before_audio[prompt_start : prompt_start + prompt_len]
                sample_prompt_lengths = decoder_prompt_before_audio_lengths[sample_idx : sample_idx + 1]
            priming_embeds, context_lengths = self._decoder_template_context_embeds(
                sample_encoded,
                sample_lengths,
                decoder_prompt_before_audio=sample_prompt,
                decoder_prompt_before_audio_lengths=sample_prompt_lengths,
            )
            hidden, state = self.decoder.forward_hidden_embeds(priming_embeds)
            next_logits = self.decoder.head(hidden[:, int(context_lengths[0].item()) - 1, :])

            generated: list[int] = []
            total_logprob = 0.0
            num_scored = 0
            emitted_eos = False
            for _ in range(int(max_new_tokens)):
                if do_sample:
                    sample_logits = next_logits.float() / float(temperature)
                    sample_logits = self._filter_decoder_sampling_logits(
                        sample_logits,
                        top_k=top_k,
                        top_p=top_p,
                    )
                    filtered_probs = sample_logits.float().softmax(dim=-1)
                    next_token_tensor = torch.multinomial(filtered_probs, num_samples=1)
                    next_token = int(next_token_tensor[0, 0].item())
                    total_logprob += float(filtered_probs[0, next_token].log().item())
                else:
                    log_probs = next_logits.float().log_softmax(dim=-1)
                    next_token = int(log_probs.argmax(dim=-1).item())
                    total_logprob += float(log_probs[0, next_token].item())
                num_scored += 1
                if next_token == int(eos_token_id):
                    emitted_eos = True
                    break
                generated.append(next_token)
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=encoded.device)
                token_logits, state = self.decoder.forward_tokens(next_token_tensor, state=state)
                next_logits = token_logits[:, -1, :]

            outputs.append(generated)
            avg_scores.append(total_logprob / float(max(1, num_scored)))
            eos_emitted.append(emitted_eos)
        return outputs, torch.tensor(avg_scores, device=encoded.device, dtype=torch.float32), eos_emitted

    def decoder_ar_loss(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        targets: Tensor,
        target_lengths: Tensor,
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
    ) -> Tensor:
        if self.decoder is None or self.decoder_bos is None:
            raise RuntimeError("Decoder loss requested but decoder is disabled.")
        token_sequences = self._packed_targets_to_decoder_sequences(targets, target_lengths)
        hidden, target_ids = self._decoder_template_hidden_and_labels(
            encoded,
            encoded_lengths,
            token_sequences,
            decoder_prompt_before_audio=decoder_prompt_before_audio,
            decoder_prompt_before_audio_lengths=decoder_prompt_before_audio_lengths,
        )
        batch_size = int(hidden.size(0))
        shifted_hidden = hidden[:, :-1, :]
        shifted_targets = target_ids[:, 1:]
        seq_len = int(shifted_hidden.size(1))
        positions_per_chunk = max(1, int(self.config.decoder_loss_chunk_size))
        time_chunk = max(1, positions_per_chunk // max(batch_size, 1))
        if seq_len <= time_chunk:
            return F.cross_entropy(
                self.decoder.head(shifted_hidden).transpose(1, 2).float(),
                shifted_targets,
                ignore_index=-100,
            )

        total_loss: Tensor | None = None
        total_count = 0
        for start in range(0, seq_len, time_chunk):
            end = min(start + time_chunk, seq_len)
            chunk_targets = shifted_targets[:, start:end]
            valid = chunk_targets.ne(-100)
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                continue
            chunk_logits = self.decoder.head(shifted_hidden[:, start:end, :])
            chunk_loss = F.cross_entropy(
                chunk_logits.transpose(1, 2).float(),
                chunk_targets,
                ignore_index=-100,
                reduction="sum",
            )
            total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
            total_count += valid_count
        if total_loss is None or total_count <= 0:
            raise ValueError("Decoder AR loss received a batch with no valid target tokens.")
        return total_loss / total_count

    def joint_losses(
        self,
        features: Tensor,
        feature_lengths: Tensor | None,
        targets: Tensor,
        target_lengths: Tensor,
        *,
        decoder_targets: Tensor | None = None,
        decoder_target_lengths: Tensor | None = None,
        decoder_prompt_before_audio: Tensor | None = None,
        decoder_prompt_before_audio_lengths: Tensor | None = None,
        direction_mask: DirectionMask | None = None,
        state: RWKVConformerEncoderState | None = None,
        compute_ctc_logits: bool = True,
    ) -> dict[str, Tensor | Tensor | RWKVConformerEncoderState | None]:
        encoded, encoded_lengths, next_state = self.encoder(
            features,
            lengths=feature_lengths,
            direction_mask=direction_mask,
            state=state,
        )
        ctc_encoded, ctc_encoded_lengths = self.ctc_encoder_features_from_encoded(encoded, encoded_lengths)
        ctc_features, logit_lengths = self.ctc_features_from_ctc_encoded(ctc_encoded, ctc_encoded_lengths)
        if logit_lengths is None:
            raise ValueError("CTC training/distillation requires feature lengths.")
        ctc_loss_weight = float(self.config.ctc_loss_weight)
        decoder_loss_weight = float(self.config.decoder_loss_weight)
        if not compute_ctc_logits and ctc_loss_weight > 0.0:
            raise ValueError("compute_ctc_logits=False requires ctc_loss_weight=0.")
        logits = (
            self.apply_ctc_logit_mask(self.ctc_head(ctc_features))
            if compute_ctc_logits
            else None
        )
        zero_source = logits if isinstance(logits, Tensor) else ctc_features
        zero_loss = zero_source.float().sum() * 0.0
        ctc_loss = (
            self.ctc_loss(logits, logit_lengths, targets, target_lengths)
            if ctc_loss_weight > 0.0
            else zero_loss
        )
        decoder_loss = zero_loss
        total_loss = ctc_loss * ctc_loss_weight
        if decoder_loss_weight > 0.0:
            if self.decoder is None:
                raise ValueError("decoder_loss_weight > 0 requires decoder_enabled=True.")
            decoder_loss = self.decoder_ar_loss(
                encoded,
                encoded_lengths,
                decoder_targets if decoder_targets is not None else targets,
                decoder_target_lengths if decoder_target_lengths is not None else target_lengths,
                decoder_prompt_before_audio=decoder_prompt_before_audio,
                decoder_prompt_before_audio_lengths=decoder_prompt_before_audio_lengths,
            )
            total_loss = total_loss + decoder_loss * decoder_loss_weight
        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss,
            "decoder_loss": decoder_loss,
            "encoded": encoded,
            "encoded_lengths": encoded_lengths,
            "ctc_encoded": ctc_encoded,
            "ctc_encoded_lengths": ctc_encoded_lengths,
            "logits": logits,
            "logit_lengths": logit_lengths,
            "state": next_state,
        }

    def ctc_loss(
        self,
        logits: Tensor,
        logit_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        self._validate_ctc_targets(targets, target_lengths)
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        return F.ctc_loss(
            log_probs,
            targets,
            logit_lengths,
            target_lengths,
            blank=self.config.blank_id,
            zero_infinity=True,
        )

    def _validate_ctc_targets(self, targets: Tensor, target_lengths: Tensor) -> None:
        if targets.dim() != 1 or target_lengths.dim() != 1:
            raise ValueError(
                "CTC training requires one-dimensional packed targets and target lengths."
            )
        if target_lengths.numel() and bool((target_lengths < 0).any().item()):
            raise ValueError("CTC target lengths must be non-negative.")
        packed_length = int(target_lengths.to(dtype=torch.long).sum().item())
        if packed_length != int(targets.numel()):
            raise ValueError(
                "Packed CTC target length mismatch: "
                f"target_lengths sum to {packed_length}, but targets contain {targets.numel()} tokens."
            )
        if targets.numel() == 0:
            return

        ctc_vocab_size = int(self.config.ctc_vocab_size)
        out_of_range = targets[(targets < 0) | (targets >= ctc_vocab_size)]
        if out_of_range.numel():
            token_ids = sorted({int(token_id) for token_id in out_of_range.detach().cpu().tolist()})
            raise ValueError(
                f"CTC targets contain token ids outside [0, {ctc_vocab_size}): {token_ids}"
            )

        blank_id = int(self.config.blank_id)
        if bool((targets == blank_id).any().item()):
            raise ValueError(f"CTC targets must not contain the blank token id {blank_id}.")

        suppressed = self.ctc_suppressed_token_ids
        if suppressed.numel() == 0:
            return
        suppressed_targets = targets[
            torch.isin(targets, suppressed.to(device=targets.device))
        ]
        if suppressed_targets.numel():
            token_ids = sorted(
                {int(token_id) for token_id in suppressed_targets.detach().cpu().tolist()}
            )
            raise ValueError(f"CTC targets contain suppressed non-pronunciation token ids: {token_ids}")
