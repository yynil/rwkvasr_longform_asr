# Architecture
![Architecture](ConvRWKV_CTC_RWKV_ASR_Framework_EN.png "ConvRWKV-CTC/RWKV ASR")
# ConvRWKV-CTC/RWKV ASR

Streaming and Long-Form ASR: Architecture and Three-Stage Training Framework.

ConvRWKV combines local convolutional acoustic modeling with RWKV-based long-context modeling. A `CTC` head provides fast base decoding and native time alignment, while a pretrained `RWKV-7` decoder head adds LM-aware autoregressive decoding for long-form and complex-text scenarios.

## 1. Project Goal

This project targets a practical ASR stack with these capabilities:

| Capability | Description |
|---|---|
| **Streaming ASR** | Real-time, continuous encoding via RWKV's RNN-like recurrent property |
| **Long Context** | RWKV models long-range context without chunking or context fragmentation |
| **Time Alignment** | Native token-level timestamp output from the CTC head |
| **Pretrained LM** | RWKV-7 0.4B decoder head for LM-aware, long-form autoregressive decoding |

Core design constraints:

- encoder: `ConvRWKV` — Conformer-inspired, combining a local convolutional module with `RWKV-7 TimeMixer` for long-context modeling
- decoder heads from the same encoder checkpoint:
  - `CTC Head`: base decoding + native time alignment
  - `RWKV Head`: pretrained RWKV-7 0.4B, LM-aware autoregressive decoding
- train through a **three-stage pipeline** that progresses from stable bidirectional pretraining to true streaming capability and finally dual-head joint training
- support `4 x RTX 4090 + DeepSpeed + bf16` training

## 2. Core Design

### Model

- encoder frontend: `global CMVN + WeNet-style conv2d6`
- encoder body: `ConvRWKV`
  - **Convolution** module: local acoustic modeling, minimal look-ahead
  - **RWKV** (`TimeMixer`): long-context modeling, RNN-like streaming; replaces self-attention only
  - FFN stays standard
- decoder heads (shared acoustic representation):
  - **CTC Head**: base decoding, native time alignment
  - **RWKV Head**: pretrained RWKV-7 0.4B, LM-aware autoregressive decoding
- inference modes from the same checkpoint:
  - **Streaming**: uni-directional RWKV encoding, low-latency continuous output
  - **Long-form**: full bidirectional context + RWKV LM decoder for highest accuracy

### Three-Stage Training Pipeline

Training progresses through three stages, each building on the previous checkpoint:

```
Stage 1                    Stage 2                       Stage 3
ConvRWKV-CTC          ─CK1─▶  Streaming Adaptation  ─CK2─▶  Dual-Head Training
BiRWKV + CTC                  Continue CTC training         Add pretrained RWKV head
Stable acoustic encoder       20% direction drop/step       Jointly optimize CTC + RWKV
Base encoder checkpoint       Streaming encoder ckpt        Final dual-head model
```

**Stage 1 — Acoustic Pretraining** (`ConvRWKV-CTC / BiRWKV + CTC`)
- *Objective*: establish a stable, reliable base speech encoder; learn the core acoustic mapping from speech to text
- *Method*: use Conv + BiRWKV as the encoder; train with CTC head only; leverage bidirectional context to stabilize acoustic modeling
- *Outcome*: base encoder checkpoint; strong initialization for streaming adaptation

**Stage 2 — Streaming Adaptation** (`ConvRWKV-CTC / Uni-directional transition`)
- *Objective*: reduce dependence on full bidirectional context; acquire uni-directional, real-time, continuous encoding
- *Method*: continue training from stage-1 checkpoint; randomly drop 20% of RWKV directions at every step; force the encoder to function when one direction is unavailable
- *Outcome*: requires only minimal Conv look-ahead; improves the latency/accuracy tradeoff over chunk-based ASR

**Stage 3 — Joint Dual-Head Training** (`ConvRWKV-CTC/RWKV / CTC + RWKV`)
- *Objective*: enhance final decoding while preserving streaming encoding; make both heads decode reliably
- *Method*: initialize from stage-2 checkpoint; keep CTC head and attach pretrained RWKV-7 0.4B decoder; jointly optimize CTC + RWKV while continuing 20% direction drop
- *Outcome*: CTC provides base decoding and native time alignment; RWKV provides stronger long-form and complex-text decoding; produces the final ConvRWKV-CTC/RWKV model

### Prediction

- Python:
  - full-model prediction from `.pt` or `.safetensors`
  - `CTC prefix beam search`
  - token-level `CTC forced alignment`
- Rust + Candle:
  - current scope is decode-stage inference
  - consumes exported `logits + lengths + utt_ids`
  - runs `CTC prefix beam search`
  - outputs token-level time alignment

## 3. Data Preparation

### 3.1 Data Preparation Architecture

```mermaid
flowchart LR
    A[Raw audio + metadata] --> B[WebDataset tar shards]
    B --> C[Rust length indexer]
    B --> D[Offline global CMVN]
    C --> E[webdataset_lengths.jsonl]
    C --> F[webdataset_lengths.summary.json]
    D --> G[global_cmvn.json]
    E --> H[Length-bucketed dataloader]
    G --> H
    B --> H
```

### 3.2 Data Preparation Flow

```mermaid
flowchart TD
    A[Collect wav + json samples] --> B[Pack into WebDataset shards]
    B --> C[Scan shard metadata with Rust]
    C --> D[Estimate frame lengths]
    D --> E[Stable train/eval split]
    E --> F[Write length index + summary]
    B --> G[Offline CMVN accumulation]
    G --> H[Write global_cmvn.json]
    F --> I[Training dataloader]
    H --> I
```

## 4. Training

### 4.1 Three-Stage Training Overview

```mermaid
flowchart LR
    S1["Stage 1\nConvRWKV-CTC\nBiRWKV + CTC\nStable acoustic encoder"] -->|CK1 base encoder checkpoint| S2
    S2["Stage 2\nStreaming Adaptation\nContinue CTC training\n20% direction drop/step"] -->|CK2 streaming encoder checkpoint| S3
    S3["Stage 3\nDual-Head Training\nAdd pretrained RWKV-7 head\nJointly optimize CTC + RWKV"]
```

### 4.2 Stage 1 — Acoustic Pretraining

```mermaid
flowchart LR
    A[WebDataset shards] --> B[Online wav decode]
    B --> C[Fbank 80-dim]
    C --> D[Global CMVN]
    D --> E[Conv2dSubsampling6]
    E --> F[Bidirectional RWKV-7 TimeMixer blocks]
    F --> G[CTC projection]
    G --> H[CTC loss]
    H --> I[DeepSpeed ZeRO-2 bf16 optimizer]
    I --> J[Base encoder checkpoint CK1]
```

### 4.3 Stage 2 — Streaming Adaptation

```mermaid
flowchart LR
    CK1[CK1 base encoder checkpoint] --> A
    A[Fbank + CMVN + conv2d6] --> B[Bidirectional RWKV-7 TimeMixer blocks]
    B --> C["Direction Dropout\n20% drop per step"]
    C --> D[CTC projection]
    D --> E[CTC loss]
    E --> F[DeepSpeed ZeRO-2 bf16 optimizer]
    F --> G[Streaming encoder checkpoint CK2]
```

### 4.4 Stage 3 — Joint Dual-Head Training

```mermaid
flowchart LR
    CK2[CK2 streaming encoder checkpoint] --> A
    A[Fbank + CMVN + conv2d6] --> B[ConvRWKV encoder with 20% direction drop]
    B --> C[Shared Acoustic Representation]
    C --> D[CTC Head\nBase decoding + time alignment]
    C --> E["RWKV Head\nPretrained RWKV-7 0.4B\nLM-aware autoregressive decoding"]
    D --> F[Joint CTC + RWKV loss]
    E --> F
    F --> G[DeepSpeed ZeRO-2 bf16 optimizer]
    G --> H[Final ConvRWKV-CTC/RWKV model]
```

### 4.5 Training Flow

```mermaid
flowchart TD
    A[Load length index] --> B[Length-bucketed batch assembly]
    B --> C[Online wav decode + fbank]
    C --> D[Apply CMVN + conv2d6]
    D --> E[Forward through ConvRWKV encoder]
    E --> F["Apply Direction Dropout mask\n(Stage 2+3: 20% per step)"]
    F --> G[Compute CTC logits]
    G --> H["CTC loss\n(+ RWKV decoder loss in Stage 3)"]
    H --> I[DeepSpeed backward + optimizer step]
    I --> J[Per-epoch eval]
    J --> K[Save epoch checkpoint + best checkpoint]
```

## 5. Prediction

### 5.1 Prediction Architecture

```mermaid
flowchart LR
    A["Checkpoint .pt / .safetensors"] --> B[ConvRWKV encoder forward]
    B --> C[Shared Acoustic Representation]
    C --> D["CTC Head\nPrefix beam search\nForced alignment\nToken timestamps"]
    C --> E["RWKV Head\nLM-aware autoregressive decoding\nLong-form + complex text"]
    D --> F[JSONL predictions with token timestamps]
    E --> F

    B --> G[Export logits + lengths + utt_ids]
    G --> H[Rust Candle predictor]
    H --> I[CTC prefix beam search]
    I --> J[CTC forced alignment]
    J --> K[JSONL predictions with token timestamps]
```

### 5.2 Prediction Flow

```mermaid
flowchart TD
    A[Load model config + checkpoint] --> B[Load audio/features]
    B --> C[Encoder forward]
    C --> D[Shared acoustic representation]
    D --> E{Decode mode}
    E -->|streaming / base| F[CTC prefix beam search]
    E -->|long-form / LM| G[RWKV autoregressive decode]
    F --> H[Best token sequence]
    G --> H
    H --> I[CTC forced alignment]
    I --> J[Project encoder steps to ms timestamps]
    J --> K[Write JSONL result]
```

## 6. Why This Design Suits Streaming ASR

| | Conventional Streaming ASR | ConvRWKV-CTC/RWKV |
|---|---|---|
| **Encoding** | Chunk-based, fragments context | RWKV's RNN-like recurrence: continuous, long-context, real-time |
| **Latency** | Clear latency/accuracy tradeoff due to chunking | Minimal Conv look-ahead only; no chunk boundary artifacts |
| **Output** | Typically text only | CTC native time alignment + LM-augmented decoding |
| **LM integration** | External re-scoring | Pretrained RWKV-7 0.4B decoder baked in |

> **ConvRWKV-CTC/RWKV** = streaming acoustic encoding + native time alignment + pretrained LM-augmented decoding

## 7. Implementation Status

### Completed

- `uv` project and training environment definition
- `RWKV-7 TimeMixer` PyTorch implementation
- bidirectional RWKV wrapper (`ConvRWKV` encoder)
- `Direction Dropout` (Stage 2+3: 20% per step)
- Conformer-style ConvRWKV encoder block
- `CTC` head and loss path (Stage 1–3)
- WeNet-style `fbank + CMVN + conv2d6` frontend
- WebDataset online decode loader
- Rust multithreaded WebDataset length indexer
- length-bucketed training batches
- DeepSpeed ZeRO-2 multi-GPU training entrypoint
- per-epoch eval and best-checkpoint selection
- checkpoint export to `.safetensors`
- Python full-model prediction (CTC head)
- Python token-level CTC time alignment
- Rust + Candle decode-stage predictor
- Rust token-level CTC time alignment
- Python export of `logits + lengths + utt_ids` for Rust decode

### In Progress

- Stage 1 → Stage 2 → Stage 3 pipeline execution on target hardware
- Stage 3 RWKV-7 0.4B decoder head integration and joint training
- prediction/export workflow hardening for real eval runs
- streaming vs. long-form decode comparison from the same checkpoint

### Planned

- Rust full-model RWKV forward path
  - likely via Candle custom ops for the RWKV recurrent core
- streaming validation and latency reporting
- CER / WER reporting on held-out eval sets
- hotword biasing via weighted finite-state context graph
- benchmark scripts for throughput and memory

Public roadmap and task breakdown live in [ROADMAP.md](./ROADMAP.md).

## 8. Current Training Summary

This repository does **not** include training artifacts, checkpoints, DeepSpeed states, or data shards. The numbers below document current progress only.

### Dataset

As of `2026-03-23`:

- WebDataset shards: `101`
- total samples: `985,531`
- train samples: `965,531`
- eval samples: `20,000`
- minimum frames: `8`
- maximum frames: `3329`
- split policy: stable hash split by `shard_name`

### Current public training summary

Run: `runs/paper_bi_baseline_4x4090`

| Epoch | Step | Train Loss | Eval Loss |
| --- | ---: | ---: | ---: |
| 1 | 1578 | 6.1032 | 3.8228 |
| 2 | 3156 | 3.0942 | 2.4147 |
| 3 | 4734 | 2.2993 | 1.9029 |
| 4 | 6312 | 1.9542 | 1.6913 |
| 5 | 7890 | 1.7451 | 1.5618 |
| 6 | 9468 | 1.6020 | 1.4483 |
| 7 | 11046 | 1.4964 | 1.4162 |
| 8 | 12624 | 1.4139 | 1.3562 |

Current best:

- best epoch: `8`
- best eval loss: `1.356161480373144`
- corresponding train loss: `1.413856222917177`

These are training-time selection metrics, not final `CER/WER`.

## 9. Repository Layout

```text
src/rwkvasr/
  data/          data pipeline, WebDataset, CMVN, tokenizers
  modules/       RWKV-CTC model, frontend, encoder blocks
  training/      optimizer, loops, checkpointing, DeepSpeed integration
  predict/       Python CTC prefix beam search and alignment
  cli/           train / eval / predict / export CLIs

tools/
  Rust preprocessing and Rust+Candle decode tools

configs/
  training configs for paper-style runs

scripts/
  launch helpers
```

## 10. Quick Start

### Python / training side

```bash
uv sync --extra dev
```

```bash
./scripts/train_paper_rwkv_asr.sh bi_baseline
```

### Export checkpoint weights

```bash
rwkvasr-export-safetensors \
  --checkpoint-path runs/paper_bi_baseline_4x4090/best.pt \
  --output-path runs/paper_bi_baseline_4x4090/best.safetensors \
  --copy-model-config
```

### Python full-model prediction

```bash
rwkvasr-predict-ctc \
  --checkpoint-path runs/paper_bi_baseline_4x4090/best.safetensors \
  --config-yaml runs/paper_bi_baseline_4x4090/model_config.yaml \
  --webdataset-root /path/to/webdataset \
  --webdataset-split eval \
  --device cuda \
  --mode bi \
  --beam-size 8 \
  --output-path runs/paper_bi_baseline_4x4090/preds.eval.jsonl
```

### Export Rust decode inputs

```bash
rwkvasr-export-ctc-logits \
  --checkpoint-path runs/paper_bi_baseline_4x4090/best.safetensors \
  --config-yaml runs/paper_bi_baseline_4x4090/model_config.yaml \
  --webdataset-root /path/to/webdataset \
  --webdataset-split eval \
  --device cuda \
  --mode bi \
  --output-dir runs/paper_bi_baseline_4x4090/rust_decode_inputs
```

### Rust decode-stage prediction

```bash
cargo run --release --manifest-path tools/Cargo.toml --bin predict_ctc -- \
  --tensors-path runs/paper_bi_baseline_4x4090/rust_decode_inputs/part-00000.safetensors \
  --utt-ids-path runs/paper_bi_baseline_4x4090/rust_decode_inputs/part-00000.utt_ids.txt \
  --beam-size 8 \
  --subsampling-rate 6 \
  --right-context 10 \
  --frame-shift-ms 10 \
  --output-path runs/paper_bi_baseline_4x4090/rust_decode_inputs/part-00000.predictions.jsonl
```

## 11. Notes

- This repository intentionally excludes:
  - checkpoints
  - DeepSpeed optimizer states
  - TensorBoard logs
  - training `runs/`
  - exported `.safetensors` artifacts
  - local datasets
- the current Rust predictor is a decode-stage tool, not yet a full RWKV model forward implementation
- project design details and rationale remain in [Spec.md](./Spec.md)
