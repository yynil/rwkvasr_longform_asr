# Roadmap

This file tracks the public milestone structure for `rwkvasr_longform_asr`.

The current focus is to move from:

- Python-first training and prediction
- Rust preprocessing and Rust decode-stage validation

to:

- full Rust inference parity
- streaming validation with one-checkpoint dual-mode ASR
- reproducible `CER/WER` reporting

## Status Snapshot

### Implemented

- RWKV-7 TimeMixer encoder replacement in PyTorch
- bidirectional RWKV + Direction Dropout training
- WeNet-style frontend: `fbank + global CMVN + conv2d6`
- WebDataset online decode
- Rust length indexing tools
- Python prediction with prefix beam search + token timestamps
- Rust decode-stage prediction with prefix beam search + token timestamps
- checkpoint export to `.safetensors`

### Current Gaps

- Rust does not yet run the full RWKV model forward
- streaming-mode validation is not yet reported as a public benchmark
- checkpoint quality is tracked by `eval_loss`, not yet by final `CER/WER`

## Milestones

### M1. Rust Full-Model Forward

Goal:
- make Rust inference independent of Python model forward

Deliverables:
- weight loading from exported checkpoint assets
- frontend parity for `global CMVN + conv2d6`
- full encoder forward path in Candle
- RWKV recurrent core implementation suitable for deployment
- parity report against Python outputs on fixed test inputs

Public tasks:
- [#1 Rust full-model forward v0: checkpoint loader, frontend parity, and module skeleton](https://github.com/yynil/rwkvasr_longform_asr/issues/1)
- [#2 Rust full-model forward v1: RWKV recurrent core custom op and end-to-end parity](https://github.com/yynil/rwkvasr_longform_asr/issues/2)

### M2. Streaming Validation

Goal:
- prove that the same checkpoint is usable in both offline and streaming modes

Deliverables:
- chunked `L2R` inference path validation
- cached-state parity checks
- same-checkpoint `Bi / L2R / Alt` comparison
- latency and throughput measurements

Public tasks:
- [#3 Streaming validation: chunk/state parity and same-checkpoint mode comparison](https://github.com/yynil/rwkvasr_longform_asr/issues/3)
- [#4 Streaming benchmark report: latency, throughput, and memory](https://github.com/yynil/rwkvasr_longform_asr/issues/4)

### M3. CER / WER Evaluation

Goal:
- move checkpoint selection from loss-only to recognition metrics

Deliverables:
- normalized text eval pipeline
- `CER/WER` computation on eval split
- per-checkpoint comparison report
- best-checkpoint selection summary

Public tasks:
- [#5 Evaluation pipeline: CER/WER on eval split and checkpoint selection report](https://github.com/yynil/rwkvasr_longform_asr/issues/5)

## Tracking Rules

- every public milestone should be represented by GitHub issues
- `README.md` should stay consistent with the current milestone state
- large tasks should be split when they mix:
  - model implementation
  - validation
  - reporting
- benchmark and metric issues should link exact configs and dates
