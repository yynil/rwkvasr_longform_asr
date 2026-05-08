from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from .rwkv7_time_mixer import RWKV7TimeMixer, RWKV7TimeMixerConfig, RWKV7TimeMixerState


@dataclass(frozen=True)
class RWKV7DecoderConfig:
    vocab_size: int
    n_embd: int
    num_layers: int
    head_size: int = 64
    backend: str = "native"
    ffn_hidden_size: int | None = None

    @property
    def resolved_ffn_hidden_size(self) -> int:
        return int(self.ffn_hidden_size or (self.n_embd * 4))


@dataclass
class RWKV7ChannelMixState:
    last_x: Tensor


@dataclass
class RWKV7DecoderBlockState:
    time_mixer: RWKV7TimeMixerState | None = None
    channel_mixer: RWKV7ChannelMixState | None = None


@dataclass
class RWKV7DecoderState:
    block_states: list[RWKV7DecoderBlockState | None] = field(default_factory=list)


def _normalize_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".pth":
        checkpoint_path = checkpoint_path.with_suffix(".pth")
    return checkpoint_path


def infer_rwkv7_decoder_config_from_checkpoint(path: str | Path) -> RWKV7DecoderConfig:
    checkpoint_path = _normalize_checkpoint_path(path)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vocab_size, n_embd = state_dict["emb.weight"].shape
    num_layers = 0
    while f"blocks.{num_layers}.ln1.weight" in state_dict:
        num_layers += 1
    if num_layers <= 0:
        raise ValueError(f"Unable to infer RWKV decoder layer count from {checkpoint_path}")
    n_head, head_size = state_dict["blocks.0.att.r_k"].shape
    if int(n_head * head_size) != int(n_embd):
        raise ValueError(
            f"Checkpoint {checkpoint_path} is inconsistent: n_head={n_head} head_size={head_size} n_embd={n_embd}"
        )
    ffn_hidden_size = int(state_dict["blocks.0.ffn.key.weight"].shape[0])
    return RWKV7DecoderConfig(
        vocab_size=int(vocab_size),
        n_embd=int(n_embd),
        num_layers=int(num_layers),
        head_size=int(head_size),
        backend="native",
        ffn_hidden_size=ffn_hidden_size,
    )


class RWKV7ChannelMix(nn.Module):
    def __init__(self, n_embd: int, hidden_size: int):
        super().__init__()
        self.n_embd = int(n_embd)
        self.hidden_size = int(hidden_size)
        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.key = nn.Linear(n_embd, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, n_embd, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd
            self.x_k.copy_(1.0 - torch.pow(ddd, 0.7))
            self.key.weight.data.uniform_(-0.05 / (self.n_embd**0.5), 0.05 / (self.n_embd**0.5))
            self.value.weight.data.zero_()

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RWKV7ChannelMixState:
        return RWKV7ChannelMixState(last_x=torch.zeros(batch_size, 1, self.n_embd, device=device, dtype=dtype))

    def forward(
        self,
        x: Tensor,
        state: RWKV7ChannelMixState | None = None,
    ) -> tuple[Tensor, RWKV7ChannelMixState]:
        if state is None:
            prev = torch.zeros_like(x[:, :1])
        else:
            prev = state.last_x.to(dtype=x.dtype, device=x.device)
        xx = torch.cat([prev, x[:, :-1]], dim=1) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        out = self.value(k)
        return out, RWKV7ChannelMixState(last_x=x[:, -1:].detach())


class RWKV7DecoderBlock(nn.Module):
    def __init__(self, config: RWKV7DecoderConfig, layer_id: int):
        super().__init__()
        self.layer_id = int(layer_id)
        self.ln0 = nn.LayerNorm(config.n_embd) if layer_id == 0 else None
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.att = RWKV7TimeMixer(
            RWKV7TimeMixerConfig(
                n_embd=config.n_embd,
                dim_att=config.n_embd,
                n_layer=config.num_layers,
                layer_id=layer_id,
                head_size=config.head_size,
                backend=config.backend,
            )
        )
        self.ffn = RWKV7ChannelMix(config.n_embd, config.resolved_ffn_hidden_size)

    def forward(
        self,
        x: Tensor,
        *,
        v_first: Tensor | None = None,
        state: RWKV7DecoderBlockState | None = None,
    ) -> tuple[Tensor, Tensor, RWKV7DecoderBlockState]:
        time_state = None if state is None else state.time_mixer
        ffn_state = None if state is None else state.channel_mixer

        att_out, next_v_first, next_time_state = self.att(self.ln1(x), v_first=v_first, state=time_state)
        x = x + att_out
        ffn_out, next_ffn_state = self.ffn(self.ln2(x), state=ffn_state)
        x = x + ffn_out
        return x, next_v_first, RWKV7DecoderBlockState(
            time_mixer=next_time_state,
            channel_mixer=next_ffn_state,
        )


class RWKV7DecoderLM(nn.Module):
    def __init__(self, config: RWKV7DecoderConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([RWKV7DecoderBlock(config, layer_id=i) for i in range(config.num_layers)])
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def hidden_size(self) -> int:
        return int(self.config.n_embd)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RWKV7DecoderState:
        return RWKV7DecoderState(
            block_states=[
                RWKV7DecoderBlockState(
                    time_mixer=block.att.init_state(batch_size, device=device, dtype=dtype),
                    channel_mixer=block.ffn.init_state(batch_size, device=device, dtype=dtype),
                )
                for block in self.blocks
            ]
        )

    def forward_hidden_embeds(
        self,
        embeds: Tensor,
        *,
        state: RWKV7DecoderState | None = None,
    ) -> tuple[Tensor, RWKV7DecoderState]:
        x = embeds
        if len(self.blocks) > 0 and self.blocks[0].ln0 is not None:
            x = self.blocks[0].ln0(x)

        if state is None or len(state.block_states) == 0:
            block_states: list[RWKV7DecoderBlockState | None] = [None] * len(self.blocks)
        else:
            if len(state.block_states) != len(self.blocks):
                raise ValueError("Decoder state block count does not match the number of blocks.")
            block_states = state.block_states

        next_states: list[RWKV7DecoderBlockState | None] = []
        v_first: Tensor | None = None
        for layer_idx, block in enumerate(self.blocks):
            x, v_first, next_state = block(x, v_first=v_first, state=block_states[layer_idx])
            next_states.append(next_state)
        x = self.ln_out(x)
        return x, RWKV7DecoderState(block_states=next_states)

    def forward_embeds(
        self,
        embeds: Tensor,
        *,
        state: RWKV7DecoderState | None = None,
    ) -> tuple[Tensor, RWKV7DecoderState]:
        hidden, next_state = self.forward_hidden_embeds(embeds, state=state)
        return self.head(hidden), next_state

    def forward_tokens(
        self,
        token_ids: Tensor,
        *,
        state: RWKV7DecoderState | None = None,
    ) -> tuple[Tensor, RWKV7DecoderState]:
        return self.forward_embeds(self.emb(token_ids), state=state)

    def forward_with_prefix(
        self,
        prefix_embeds: Tensor,
        input_token_ids: Tensor,
        *,
        bos_embed: Tensor,
        state: RWKV7DecoderState | None = None,
    ) -> tuple[Tensor, RWKV7DecoderState]:
        token_embeds = self.emb(input_token_ids)
        token_embeds[:, :1, :] = bos_embed.to(dtype=token_embeds.dtype, device=token_embeds.device)
        full_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        full_logits, next_state = self.forward_embeds(full_embeds, state=state)
        prefix_len = int(prefix_embeds.size(1))
        return full_logits[:, prefix_len:, :], next_state

    def load_official_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = _normalize_checkpoint_path(path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Official RWKV decoder checkpoint load mismatch for {checkpoint_path}: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
