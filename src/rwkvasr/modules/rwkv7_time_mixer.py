from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rwkv7_cuda import fused_wkv7


@dataclass
class RWKV7TimeMixerConfig:
    n_embd: int
    dim_att: int
    n_layer: int
    layer_id: int
    head_size: int = 64
    backend: str = "native"
    chunk_len: int = 16


@dataclass
class RWKV7TimeMixerState:
    att_state: Tensor
    last_x: Tensor


def _ortho_init(x: Tensor, scale: float) -> Tensor:
    shape = x.shape
    if len(shape) == 2:
        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
        nn.init.orthogonal_(x, gain=gain * scale)
        return x
    raise ValueError(f"Unsupported shape for orthogonal init: {shape}")


def pad_to_chunk_length(x: Tensor, chunk_len: int) -> tuple[Tensor, int]:
    pad = (-x.size(1)) % chunk_len
    if pad == 0:
        return x, 0
    pad_tensor = torch.zeros(x.size(0), pad, x.size(2), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=1), pad


def _native_wkv7(
    r: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    bsz, tsz, n_head, head_size = r.shape
    if state is None:
        state = torch.zeros(
            bsz,
            n_head,
            head_size,
            head_size,
            dtype=torch.float32,
            device=r.device,
        )
    else:
        state = state.to(dtype=torch.float32)

    decay = torch.exp(-torch.exp(w.float()))
    y_out: list[Tensor] = []
    work_state = state

    for t in range(tsz):
        rt = r[:, t].float()
        wt = decay[:, t]
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = a[:, t].float()
        bt = b[:, t].float()

        sa = torch.einsum("bhij,bhj->bhi", work_state, at)
        work_state = work_state * wt.unsqueeze(-2)
        work_state = work_state + sa.unsqueeze(-1) * bt.unsqueeze(-2)
        work_state = work_state + vt.unsqueeze(-1) * kt.unsqueeze(-2)
        yt = torch.einsum("bhij,bhj->bhi", work_state, rt)
        y_out.append(yt.to(dtype=v.dtype))

    return torch.stack(y_out, dim=1), work_state


class RWKV7TimeMixer(nn.Module):
    def __init__(self, config: RWKV7TimeMixerConfig):
        super().__init__()
        self.config = config
        self.layer_id = config.layer_id
        self.head_size = config.head_size
        self.n_head = config.dim_att // config.head_size
        if config.dim_att % config.head_size != 0:
            raise ValueError("dim_att must be divisible by head_size")

        h = self.n_head
        n = self.head_size
        c = config.n_embd

        with torch.no_grad():
            ratio_0_to_1 = config.layer_id / max(config.n_layer - 1, 1)
            ratio_1_to_almost0 = 1.0 - (config.layer_id / config.n_layer)
            ddd = torch.ones(1, 1, c)
            for i in range(c):
                ddd[0, 0, i] = i / c

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            www = torch.zeros(c)
            zigzag = torch.zeros(c)
            linear = torch.zeros(c)
            for i in range(c):
                linear[i] = i / max(c - 1, 1) - 0.5
                zigzag[i] = ((i % n) - ((n - 1) / 2)) / ((n - 1) / 2)
                zigzag[i] = zigzag[i] * abs(zigzag[i])
                www[i] = -6 + 6 * (i / max(c - 1, 1)) ** (1 + ratio_0_to_1**0.3)

            d_decay = max(32, int(round((2.5 * (c**0.5)) / 32) * 32))
            d_aaa = max(32, int(round((2.5 * (c**0.5)) / 32) * 32))
            d_mv = max(32, int(round((1.7 * (c**0.5)) / 32) * 32))
            d_gate = max(32, int(round((5.0 * (c**0.5)) / 32) * 32))

            self.w1 = nn.Parameter(torch.zeros(c, d_decay))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(d_decay, c), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, c) + 0.5 + zigzag * 2.5)

            self.a1 = nn.Parameter(torch.zeros(c, d_aaa))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(d_aaa, c), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, c) - 0.19 + zigzag * 0.3 + linear * 0.4)

            self.v1 = nn.Parameter(torch.zeros(c, d_mv))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(d_mv, c), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, c) + 0.73 - linear * 0.4)

            self.g1 = nn.Parameter(torch.zeros(c, d_gate))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(d_gate, c), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, c) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, c) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(h, n) - 0.04)

            self.receptance = nn.Linear(c, c, bias=False)
            self.key = nn.Linear(c, c, bias=False)
            self.value = nn.Linear(c, c, bias=False)
            self.output = nn.Linear(c, c, bias=False)
            self.ln_x = nn.GroupNorm(h, c, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
            self.key.weight.data.uniform_(-0.05 / (c**0.5), 0.05 / (c**0.5))
            self.value.weight.data.uniform_(-0.5 / (c**0.5), 0.5 / (c**0.5))
            self.output.weight.data.zero_()

    def init_state(
        self, batch_size: int, device: torch.device | str, dtype: torch.dtype = torch.float32
    ) -> RWKV7TimeMixerState:
        return RWKV7TimeMixerState(
            att_state=torch.zeros(
                batch_size,
                self.n_head,
                self.head_size,
                self.head_size,
                device=device,
                dtype=torch.float32,
            ),
            last_x=torch.zeros(batch_size, 1, self.config.n_embd, device=device, dtype=dtype),
        )

    def _mixed_inputs(self, x: Tensor, state: RWKV7TimeMixerState | None) -> tuple[Tensor, Tensor]:
        if state is None:
            prev = torch.zeros_like(x[:, :1])
        else:
            prev = state.last_x.to(dtype=x.dtype, device=x.device)
        xx = torch.cat([prev, x[:, :-1]], dim=1) - x
        return xx, x[:, -1:].detach()

    def _backend(
        self,
        r: Tensor,
        w: Tensor,
        k: Tensor,
        v: Tensor,
        a: Tensor,
        b: Tensor,
        state: RWKV7TimeMixerState | None,
    ) -> tuple[Tensor, Tensor]:
        att_state = None if state is None else state.att_state
        r = r.view(r.size(0), r.size(1), self.n_head, self.head_size)
        w = w.view(w.size(0), w.size(1), self.n_head, self.head_size)
        k = k.view(k.size(0), k.size(1), self.n_head, self.head_size)
        v = v.view(v.size(0), v.size(1), self.n_head, self.head_size)
        a = a.view(a.size(0), a.size(1), self.n_head, self.head_size)
        b = b.view(b.size(0), b.size(1), self.n_head, self.head_size)

        backend = self.config.backend.lower()
        if backend == "native" or (backend in {"cuda", "fused"} and state is not None):
            y, new_att_state = _native_wkv7(r, w, k, v, a, b, state=att_state)
        elif backend in {"cuda", "fused"}:
            y = fused_wkv7(
                r.view(r.size(0), r.size(1), -1),
                w.view(w.size(0), w.size(1), -1),
                k.view(k.size(0), k.size(1), -1),
                v.view(v.size(0), v.size(1), -1),
                a.view(a.size(0), a.size(1), -1),
                b.view(b.size(0), b.size(1), -1),
                head_size=self.head_size,
                chunk_len=self.config.chunk_len,
            ).view(r.size(0), r.size(1), self.n_head, self.head_size)
            # The upstream fused training kernel does not accept or return an arbitrary
            # recurrent state. Training paths ignore this state; stateful streaming
            # calls are routed to the native backend above.
            new_att_state = torch.empty(
                r.size(0),
                self.n_head,
                self.head_size,
                self.head_size,
                dtype=torch.float32,
                device=r.device,
            )
        else:
            raise RuntimeError(f"Unsupported RWKV-7 TimeMixer backend: {self.config.backend}")
        return y.view(y.size(0), y.size(1), -1), new_att_state

    def forward(
        self,
        x: Tensor,
        v_first: Tensor | None = None,
        state: RWKV7TimeMixerState | None = None,
    ) -> tuple[Tensor, Tensor, RWKV7TimeMixerState]:
        bsz, tsz, channels = x.shape
        if channels != self.config.n_embd:
            raise ValueError(f"Expected hidden size {self.config.n_embd}, got {channels}")

        xx, last_x = self._mixed_inputs(x, state)
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.layer_id == 0 or v_first is None:
            v_first_out = v
        else:
            mix = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v = v + (v_first - v) * mix
            v_first_out = v_first

        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(bsz, tsz, self.n_head, -1), dim=-1, p=2.0).view(bsz, tsz, -1)
        k = k * (1 + (a - 1) * self.k_a)

        y, new_state = self._backend(r, w, k, v, -kk, kk * a, state)
        y = self.ln_x(y.view(bsz * tsz, channels)).view(bsz, tsz, channels)
        residual = (
            (
                r.view(bsz, tsz, self.n_head, -1)
                * k.view(bsz, tsz, self.n_head, -1)
                * self.r_k
            ).sum(dim=-1, keepdim=True)
            * v.view(bsz, tsz, self.n_head, -1)
        ).view(bsz, tsz, channels)
        y = self.output((y + residual) * g)

        return y, v_first_out, RWKV7TimeMixerState(
            att_state=new_state,
            last_x=last_x,
        )
