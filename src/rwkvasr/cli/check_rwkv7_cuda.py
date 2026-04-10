from __future__ import annotations

import argparse

import torch

from rwkvasr.modules.rwkv7_cuda import fused_wkv7
from rwkvasr.modules.rwkv7_time_mixer import _native_wkv7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile and validate the fused RWKV-7 CUDA kernel.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=17)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.hidden_size % args.head_size != 0:
        raise ValueError("hidden-size must be divisible by head-size")

    torch.manual_seed(17)
    bsz = int(args.batch_size)
    tsz = int(args.seq_len)
    hidden = int(args.hidden_size)
    head_size = int(args.head_size)
    n_head = hidden // head_size
    tensors = [
        torch.randn(bsz, tsz, hidden, device="cuda", dtype=torch.bfloat16).contiguous()
        for _ in range(6)
    ]
    q, w, k, v, z, a = tensors

    native_y, _ = _native_wkv7(
        q.view(bsz, tsz, n_head, head_size),
        w.view(bsz, tsz, n_head, head_size),
        k.view(bsz, tsz, n_head, head_size),
        v.view(bsz, tsz, n_head, head_size),
        z.view(bsz, tsz, n_head, head_size),
        a.view(bsz, tsz, n_head, head_size),
    )
    fused_y = fused_wkv7(
        q,
        w,
        k,
        v,
        z,
        a,
        head_size=head_size,
        chunk_len=int(args.chunk_len),
    )
    torch.cuda.synchronize()
    max_abs = (native_y.view_as(fused_y).float() - fused_y.float()).abs().max().item()
    if not torch.allclose(
        fused_y.float(),
        native_y.view_as(fused_y).float(),
        atol=float(args.atol),
        rtol=float(args.rtol),
    ):
        raise RuntimeError(f"fused RWKV-7 check failed: max_abs_diff={max_abs:.6f}")
    print(
        "rwkv7_cuda_check ok "
        f"shape=({bsz},{tsz},{hidden}) head_size={head_size} "
        f"chunk_len={args.chunk_len} max_abs_diff={max_abs:.6f}"
    )


if __name__ == "__main__":
    main()
