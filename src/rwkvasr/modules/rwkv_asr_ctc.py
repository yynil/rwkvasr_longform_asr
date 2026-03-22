from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .direction_dropout import DirectionMask, LayerDirectionMask
from .rwkv7_bidirectional import BidirectionalVFirstState
from .rwkv_conformer import (
    RWKVConformerBlock,
    RWKVConformerBlockConfig,
    RWKVConformerBlockState,
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
    cmvn_file: str | None = None
    cmvn_is_json: bool = True


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
    head_size: int = 64
    backend: str = "native"
    conv_kernel_size: int = 31
    dropout: float = 0.1
    blank_id: int = 0
    frontend_type: str = "conv2d6"
    cmvn_file: str | None = None
    cmvn_is_json: bool = True

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
            cmvn_file=self.cmvn_file,
            cmvn_is_json=self.cmvn_is_json,
        )


class RWKVConformerEncoder(nn.Module):
    def __init__(self, config: RWKVConformerEncoderConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.global_cmvn: nn.Module | None = None
        if config.cmvn_file is not None:
            mean, istd = load_wenet_cmvn(config.cmvn_file, is_json=config.cmvn_is_json)
            self.global_cmvn = GlobalCMVN(mean, istd)
        elif config.frontend_type == "conv2d6":
            # Keep a no-op CMVN module so checkpoints with CMVN buffers load without an external file.
            self.global_cmvn = GlobalCMVN(
                torch.zeros(config.input_dim, dtype=torch.float32),
                torch.ones(config.input_dim, dtype=torch.float32),
            )

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
                )
                next_states.append(None)
            else:
                x, v_first, next_block_state = block(
                    x,
                    v_first=v_first,
                    state=block_states[layer_idx],
                    layer_mask=layer_mask,
                )
                next_states.append(next_block_state)

        return x, lengths, RWKVConformerEncoderState(block_states=next_states)


class RWKVCTCModel(nn.Module):
    def __init__(self, config: RWKVCTCModelConfig):
        super().__init__()
        self.config = config
        self.encoder = RWKVConformerEncoder(config.to_encoder_config())
        self.ctc_head = nn.Linear(config.n_embd, config.vocab_size)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.encoder.enable_gradient_checkpointing(enabled)

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
        logits = self.ctc_head(encoded)
        return logits, encoded_lengths, next_state

    def ctc_loss(
        self,
        logits: Tensor,
        logit_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        return F.ctc_loss(
            log_probs,
            targets,
            logit_lengths,
            target_lengths,
            blank=self.config.blank_id,
            zero_infinity=True,
        )
