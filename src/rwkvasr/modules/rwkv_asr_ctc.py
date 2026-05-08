from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .direction_dropout import DirectionMask, LayerDirectionMask
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
    decoder_enabled: bool = False
    decoder_checkpoint_path: str | None = None
    decoder_num_layers: int | None = None
    decoder_n_embd: int | None = None
    decoder_ffn_hidden_size: int | None = None
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
        self.ctc_head = nn.Linear(config.n_embd, config.ctc_vocab_size)
        self.decoder: RWKV7DecoderLM | None = None
        self.decoder_prefix_proj: nn.Module | None = None
        self.decoder_bos: nn.Parameter | None = None
        if config.decoder_enabled:
            if str(config.decoder_audio_conditioning) != "full":
                raise ValueError("Only decoder_audio_conditioning='full' is supported.")
            decoder_config = self._resolve_decoder_config(config)
            if decoder_config.vocab_size != int(config.vocab_size):
                raise ValueError(
                    "Decoder vocabulary size must match the shared text vocabulary size: "
                    f"{decoder_config.vocab_size} != {config.vocab_size}"
                )
            self.decoder = RWKV7DecoderLM(decoder_config)
            if config.decoder_checkpoint_path is not None:
                self.decoder.load_official_checkpoint(config.decoder_checkpoint_path)
            if int(config.n_embd) == int(decoder_config.n_embd):
                self.decoder_prefix_proj = nn.Identity()
            else:
                self.decoder_prefix_proj = nn.Linear(config.n_embd, decoder_config.n_embd, bias=False)
            self.decoder_bos = nn.Parameter(torch.zeros(1, 1, decoder_config.n_embd))

    @staticmethod
    def _resolve_decoder_config(config: RWKVCTCModelConfig) -> RWKV7DecoderConfig:
        inferred: RWKV7DecoderConfig | None = None
        if config.decoder_checkpoint_path is not None:
            inferred = infer_rwkv7_decoder_config_from_checkpoint(config.decoder_checkpoint_path)
        return RWKV7DecoderConfig(
            vocab_size=int(config.vocab_size),
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

    def _decoder_template_context_embeds(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
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
        before_ids = self._int_token_tuple(self.config.decoder_prompt_before_audio_token_ids)
        after_ids = self._int_token_tuple(self.config.decoder_prompt_after_audio_token_ids)
        before = self._decoder_static_token_embeds(
            before_ids,
            batch_size=int(audio.size(0)),
            device=audio.device,
            dtype=audio.dtype,
        )
        after = self._decoder_static_token_embeds(
            after_ids,
            batch_size=int(audio.size(0)),
            device=audio.device,
            dtype=audio.dtype,
        )
        before_len = int(before.size(1))
        after_len = int(after.size(1))
        context_lengths = audio_lengths + before_len + after_len
        max_context_len = int(context_lengths.max().item())
        if max_context_len <= 0:
            raise ValueError("Decoder template context must contain prompt or audio positions.")

        context = audio.new_zeros(int(audio.size(0)), max_context_len, int(audio.size(-1)))
        for sample_idx, audio_len in enumerate(audio_lengths.tolist()):
            cursor = 0
            if before_len > 0:
                context[sample_idx, cursor : cursor + before_len] = before[sample_idx]
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
    ) -> tuple[Tensor, Tensor]:
        if self.decoder is None:
            raise RuntimeError("Decoder template path requested but decoder is disabled.")
        if encoded.size(0) != len(token_sequences):
            raise ValueError("Encoded batch size and token sequence count must match.")
        context, context_lengths = self._decoder_template_context_embeds(encoded, encoded_lengths)
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
    ) -> tuple[Tensor, Tensor]:
        if self.decoder is None:
            raise RuntimeError("Decoder template path requested but decoder is disabled.")
        hidden, target_ids = self._decoder_template_hidden_and_labels(
            encoded,
            encoded_lengths,
            token_sequences,
        )
        return self.decoder.head(hidden), target_ids

    def decoder_sequence_scores(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        token_sequences: list[list[int]],
        *,
        normalize_by_length: bool = True,
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

    def decoder_greedy_decode(
        self,
        encoded: Tensor,
        encoded_lengths: Tensor | None,
        *,
        eos_token_id: int = 0,
        max_new_tokens: int = 256,
    ) -> tuple[list[list[int]], Tensor, list[bool]]:
        if self.decoder is None or self.decoder_bos is None:
            raise RuntimeError("Decoder generation requested but decoder is disabled.")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
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
            priming_embeds, context_lengths = self._decoder_template_context_embeds(sample_encoded, sample_lengths)
            hidden, state = self.decoder.forward_hidden_embeds(priming_embeds)
            next_logits = self.decoder.head(hidden[:, int(context_lengths[0].item()) - 1, :])

            generated: list[int] = []
            total_logprob = 0.0
            num_scored = 0
            emitted_eos = False
            for _ in range(int(max_new_tokens)):
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
    ) -> Tensor:
        if self.decoder is None or self.decoder_bos is None:
            raise RuntimeError("Decoder loss requested but decoder is disabled.")
        token_sequences = self._packed_targets_to_decoder_sequences(targets, target_lengths)
        hidden, target_ids = self._decoder_template_hidden_and_labels(
            encoded,
            encoded_lengths,
            token_sequences,
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
        direction_mask: DirectionMask | None = None,
        state: RWKVConformerEncoderState | None = None,
    ) -> dict[str, Tensor | Tensor | RWKVConformerEncoderState | None]:
        encoded, encoded_lengths, next_state = self.encoder(
            features,
            lengths=feature_lengths,
            direction_mask=direction_mask,
            state=state,
        )
        logits = self.ctc_head(encoded)
        if encoded_lengths is None:
            raise ValueError("CTC training requires feature lengths.")
        ctc_loss = self.ctc_loss(logits, encoded_lengths, targets, target_lengths)
        if self.decoder is None or self.config.decoder_loss_weight <= 0:
            decoder_loss = ctc_loss.new_zeros(())
            total_loss = ctc_loss * float(self.config.ctc_loss_weight)
        else:
            decoder_loss = self.decoder_ar_loss(encoded, encoded_lengths, targets, target_lengths)
            total_loss = (
                ctc_loss * float(self.config.ctc_loss_weight)
                + decoder_loss * float(self.config.decoder_loss_weight)
            )
        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss,
            "decoder_loss": decoder_loss,
            "logits": logits,
            "logit_lengths": encoded_lengths,
            "state": next_state,
        }

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
