from __future__ import annotations

import argparse

import torch

from rwkvasr.modules.rwkv7_cuda import fused_wkv7, fused_wkv7_clampw
from rwkvasr.modules.rwkv7_time_mixer import _native_wkv7, _soft_clamp_w


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile and validate the fused RWKV-7 CUDA kernel.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=17)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--backend", choices=("cuda", "cuda_clampw"), default="cuda")
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--check-backward", action="store_true")
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
        (torch.randn(bsz, tsz, hidden, device="cuda", dtype=torch.bfloat16) * float(args.input_scale))
        .detach()
        .contiguous()
        for _ in range(6)
    ]
    if args.check_backward:
        tensors = [tensor.requires_grad_(True) for tensor in tensors]
    q, w, k, v, z, a = tensors
    native_w = _soft_clamp_w(w) if args.backend == "cuda_clampw" else w

    native_y, _ = _native_wkv7(
        q.view(bsz, tsz, n_head, head_size),
        native_w.view(bsz, tsz, n_head, head_size),
        k.view(bsz, tsz, n_head, head_size),
        v.view(bsz, tsz, n_head, head_size),
        z.view(bsz, tsz, n_head, head_size),
        a.view(bsz, tsz, n_head, head_size),
    )
    if args.backend == "cuda_clampw":
        fused_y = fused_wkv7_clampw(
            q,
            w,
            k,
            v,
            z,
            a,
            head_size=head_size,
            chunk_len=int(args.chunk_len),
        )
    else:
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

    grad_max_abs = None
    if args.check_backward:
        grad = torch.randn_like(fused_y).contiguous()
        native_loss = (native_y.view_as(fused_y) * grad).float().sum()
        native_grads = torch.autograd.grad(native_loss, tensors, retain_graph=False)
        fused_loss = (fused_y * grad).float().sum()
        fused_grads = torch.autograd.grad(fused_loss, tensors, retain_graph=False)
        grad_max_abs = max(
            (native_grad.float() - fused_grad.float()).abs().max().item()
            for native_grad, fused_grad in zip(native_grads, fused_grads, strict=True)
        )
        if grad_max_abs > max(float(args.atol), max_abs * 8.0):
            raise RuntimeError(f"fused RWKV-7 backward check failed: grad_max_abs_diff={grad_max_abs:.6f}")
    print(
        "rwkv7_cuda_check ok "
        f"backend={args.backend} shape=({bsz},{tsz},{hidden}) head_size={head_size} "
        f"chunk_len={args.chunk_len} max_abs_diff={max_abs:.6f}"
        + ("" if grad_max_abs is None else f" grad_max_abs_diff={grad_max_abs:.6f}")
    )


if __name__ == "__main__":
    main()
