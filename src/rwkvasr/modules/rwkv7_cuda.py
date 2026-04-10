from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor
import torch.utils.cpp_extension as cpp_extension


_SUPPORTED_HEAD_SIZE = 64
_SUPPORTED_CHUNK_LEN = 16


def _kernel_source_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "kernels" / "rwkv7_wind_backstepping"


def _select_matching_cuda_home() -> None:
    torch_cuda_version = torch.version.cuda
    if not torch_cuda_version:
        return
    candidate = Path(f"/usr/local/cuda-{torch_cuda_version}")
    if not candidate.exists():
        return
    current = os.environ.get("CUDA_HOME")
    if current and Path(current).resolve() == candidate.resolve():
        return
    os.environ["CUDA_HOME"] = str(candidate)
    cpp_extension.CUDA_HOME = str(candidate)


@lru_cache(maxsize=1)
def _load_wind_backstepping() -> object:
    _select_matching_cuda_home()
    source_dir = _kernel_source_dir()
    flags = [
        "-res-usage",
        f"-D_C_={_SUPPORTED_HEAD_SIZE}",
        f"-D_CHUNK_LEN_={_SUPPORTED_CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas",
        "-O3",
        "--extra-device-vectorization",
    ]
    # Torch's JIT extension cache uses a file lock, so all DDP ranks can call this safely.
    cpp_extension.load(
        name=f"rwkvasr_wind_backstepping_h{_SUPPORTED_HEAD_SIZE}_c{_SUPPORTED_CHUNK_LEN}",
        sources=[str(source_dir / "wkv7_cuda.cu"), str(source_dir / "wkv7_op.cpp")],
        is_python_module=False,
        verbose=os.environ.get("RWKVASR_VERBOSE_KERNEL_BUILD", "0") == "1",
        extra_cuda_cflags=flags,
    )
    return torch.ops.rwkvasr_wind_backstepping


def is_fused_wkv7_supported(
    *,
    head_size: int,
    chunk_len: int,
    tensors: tuple[Tensor, ...],
) -> tuple[bool, str]:
    if head_size != _SUPPORTED_HEAD_SIZE:
        return False, f"head_size must be {_SUPPORTED_HEAD_SIZE}, got {head_size}"
    if chunk_len != _SUPPORTED_CHUNK_LEN:
        return False, f"chunk_len must be {_SUPPORTED_CHUNK_LEN}, got {chunk_len}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    for tensor in tensors:
        if not tensor.is_cuda:
            return False, "all inputs must be CUDA tensors"
        if tensor.dtype != torch.bfloat16:
            return False, f"all inputs must be bf16 tensors, got {tensor.dtype}"
    return True, ""


def _pad_time_to_chunk(x: Tensor, *, chunk_len: int) -> tuple[Tensor, int]:
    pad = (-x.size(1)) % chunk_len
    if pad == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[1] = pad
    pad_tensor = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=1), pad


class _WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor, q: Tensor, k: Tensor, v: Tensor, z: Tensor, a: Tensor) -> Tensor:
        ops = _load_wind_backstepping()
        bsz, tsz, n_head, head_size = w.shape
        if tsz % _SUPPORTED_CHUNK_LEN != 0:
            raise ValueError("fused RWKV-7 inputs must already be padded to chunk_len")
        if head_size != _SUPPORTED_HEAD_SIZE:
            raise ValueError(f"fused RWKV-7 requires head_size={_SUPPORTED_HEAD_SIZE}")
        y = torch.empty_like(v)
        s = torch.empty(
            bsz,
            n_head,
            tsz // _SUPPORTED_CHUNK_LEN,
            head_size,
            head_size,
            dtype=torch.float32,
            device=w.device,
        )
        sa = torch.empty(bsz, tsz, n_head, head_size, dtype=torch.float32, device=w.device)
        ops.forward(w, q, k, v, z, a, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, a, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        ops = _load_wind_backstepping()
        w, q, k, v, z, a, s, sa = ctx.saved_tensors
        dy = dy.contiguous()
        dw, dq, dk, dv, dz, da = [torch.empty_like(x) for x in (w, q, k, v, z, a)]
        ops.backward(w, q, k, v, z, a, dy, s, sa, dw, dq, dk, dv, dz, da)
        return dw, dq, dk, dv, dz, da


def fused_wkv7(
    q: Tensor,
    w: Tensor,
    k: Tensor,
    v: Tensor,
    z: Tensor,
    a: Tensor,
    *,
    head_size: int,
    chunk_len: int,
) -> Tensor:
    supported, reason = is_fused_wkv7_supported(
        head_size=head_size,
        chunk_len=chunk_len,
        tensors=(q, w, k, v, z, a),
    )
    if not supported:
        raise RuntimeError(f"fused RWKV-7 backend is unavailable: {reason}")
    bsz, tsz, hidden = q.shape
    n_head = hidden // head_size
    if hidden % head_size != 0:
        raise ValueError("hidden size must be divisible by head_size")

    q4, w4, k4, v4, z4, a4 = [
        x.view(bsz, tsz, n_head, head_size).contiguous() for x in (q, w, k, v, z, a)
    ]
    padded: list[Tensor] = []
    pad = 0
    for tensor in (q4, w4, k4, v4, z4, a4):
        padded_tensor, tensor_pad = _pad_time_to_chunk(tensor, chunk_len=chunk_len)
        if padded and tensor_pad != pad:
            raise RuntimeError("inconsistent chunk padding across fused RWKV-7 inputs")
        pad = tensor_pad
        padded.append(padded_tensor.contiguous())
    y = _WindBackstepping.apply(padded[1], padded[0], padded[2], padded[3], padded[4], padded[5])
    if pad:
        y = y[:, :-pad]
    return y.contiguous().view(bsz, tsz, hidden)
