from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .direction_dropout import LayerDirectionMask
from .rwkv7_time_mixer import RWKV7TimeMixer, RWKV7TimeMixerConfig, RWKV7TimeMixerState


@dataclass
class BidirectionalVFirstState:
    forward: Tensor | None = None
    backward: Tensor | None = None


@dataclass
class BidirectionalTimeMixerState:
    forward: RWKV7TimeMixerState | None = None
    backward: RWKV7TimeMixerState | None = None


def reverse_time(x: Tensor) -> Tensor:
    return x.flip(dims=(1,))


class BidirectionalRWKVTimeMixer(nn.Module):
    def __init__(
        self,
        config: RWKV7TimeMixerConfig,
        *,
        merge_mode: str = "avg",
    ):
        super().__init__()
        if merge_mode not in {"avg", "sum"}:
            raise ValueError("merge_mode must be 'avg' or 'sum'.")

        self.config = config
        self.merge_mode = merge_mode
        self.forward_mixer = RWKV7TimeMixer(config)
        self.backward_mixer = RWKV7TimeMixer(config)

    def init_state(
        self,
        batch_size: int,
        device,
        dtype,
    ) -> BidirectionalTimeMixerState:
        return BidirectionalTimeMixerState(
            forward=self.forward_mixer.init_state(batch_size, device=device, dtype=dtype),
            backward=self.backward_mixer.init_state(batch_size, device=device, dtype=dtype),
        )

    def forward(
        self,
        x: Tensor,
        *,
        v_first: BidirectionalVFirstState | None = None,
        state: BidirectionalTimeMixerState | None = None,
        layer_mask: LayerDirectionMask | None = None,
    ) -> tuple[Tensor, BidirectionalVFirstState, BidirectionalTimeMixerState]:
        mask = layer_mask or LayerDirectionMask()
        outputs: list[Tensor] = []

        forward_v_first = None if v_first is None else v_first.forward
        backward_v_first = None if v_first is None else v_first.backward
        forward_state = None if state is None else state.forward
        backward_state = None if state is None else state.backward

        out_forward = None
        out_backward = None
        next_forward_state = None
        next_backward_state = None
        next_forward_v_first = None
        next_backward_v_first = None

        if mask.use_forward:
            out_forward, next_forward_v_first, next_forward_state = self.forward_mixer(
                x,
                v_first=forward_v_first,
                state=forward_state,
            )
            outputs.append(out_forward)

        if mask.use_backward:
            x_rev = reverse_time(x)
            backward_v_first_rev = None if backward_v_first is None else reverse_time(backward_v_first)
            out_backward_rev, next_backward_v_first_rev, next_backward_state = self.backward_mixer(
                x_rev,
                v_first=backward_v_first_rev,
                state=backward_state,
            )
            out_backward = reverse_time(out_backward_rev)
            next_backward_v_first = reverse_time(next_backward_v_first_rev)
            outputs.append(out_backward)

        if len(outputs) == 1:
            merged = outputs[0]
        elif self.merge_mode == "avg":
            merged = (outputs[0] + outputs[1]) * 0.5
        else:
            merged = outputs[0] + outputs[1]

        return (
            merged,
            BidirectionalVFirstState(
                forward=next_forward_v_first,
                backward=next_backward_v_first,
            ),
            BidirectionalTimeMixerState(
                forward=next_forward_state,
                backward=next_backward_state,
            ),
        )
