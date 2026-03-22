# Project Agents

## Objective
- Build a long-form ASR project that replaces encoder self-attention with `RWKV-7 TimeMixer`.
- Make `one checkpoint` serve both streaming and non-streaming ASR.
- Treat `bidirectional RWKV + direction dropout training` as the mainline design, not a later add-on.
- Keep the non-attention path Conformer-like: feed-forward / MLP remains standard, convolution module remains available unless a spec update removes it.
- Use a `CTC` head as the primary target for real-time and efficient on-device recognition.
- Manage the training environment with `uv`, targeting `4 x RTX 4090 + DeepSpeed + CUDA 13.0 + torch 2.10`.

## Mandatory Workflow
1. Read and update `Spec.md` before any substantial code change.
2. Keep implementation aligned with the latest upstream `RWKV-v7/train_temp` TimeMixer behavior and optimization rules.
3. Implement in thin vertical slices:
   - model primitives
   - encoder block
   - CTC model
   - data pipeline
   - training loop
   - streaming inference
   - validation and benchmarks
4. After each slice, add or update a runnable verification step.
5. Do not silently widen scope. Record design changes in `Spec.md` first.

## Upstream Rules To Preserve
- Reuse only `TimeMixer` from RWKV-7 for attention replacement.
- Do not replace the ASR feed-forward path with RWKV channel mixing unless `Spec.md` explicitly changes.
- Preserve RWKV-7 specific parameter handling where relevant:
  - `att.w0` uses higher LR scaling.
  - Weight decay applies only to large matrix weights.
  - Initialization should follow upstream intent, not generic Transformer defaults.
- Respect kernel constraints from upstream implementation when CUDA kernel integration is enabled:
  - `head_size = 64`
  - sequence length padded to kernel chunk length
  - bf16 contiguous tensors for the fused path

## Delivery Order
1. `Spec.md` analysis and design
2. project skeleton and `uv` environment definition
3. single-direction RWKV-7 TimeMixer module and tests
4. bidirectional RWKV wrapper and merge path
5. Direction Dropout scheduler and dual-mode inference switches
6. Conformer-style RWKV encoder block
7. CTC model and loss path
8. dataset and feature pipeline
9. DeepSpeed training entrypoint
10. streaming / chunked inference
11. reproducible validation scripts and benchmark reports

## Verification Gates
- Unit tests for tensor shapes, masking, state passing, and chunk padding
- Unit tests for direction dropping, merge behavior, and mode switching
- One-batch overfit sanity run
- Multi-GPU smoke training with DeepSpeed
- Offline CTC decode sanity
- Streaming decode sanity with cached states
- Same-checkpoint `uni / bi / alternating` decode comparison
- Throughput and latency measurements on target hardware

## Documentation Rules
- Record assumptions, tradeoffs, and open questions in `Spec.md`.
- Keep benchmark numbers dated and tied to exact configs.
- Keep upstream-derived code clearly marked so future updates can be rebased cleanly.
