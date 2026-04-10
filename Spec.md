# RWKV Long-Form ASR Spec

## 1. Goal

Build an ASR system for long-form and real-time inference where the encoder replaces self-attention with the latest `RWKV-7 TimeMixer` design from `BlinkDL/RWKV-LM/RWKV-v7/train_temp`, while keeping the feed-forward path standard. The primary decoding head is `CTC`.

The primary product goal is not merely a fast causal ASR model. It is a `single-checkpoint dual-mode ASR system`:
- `bidirectional` inference for higher-accuracy offline / non-streaming ASR
- `unidirectional` inference from the same weights for streaming ASR
- optional `alternating` or partially-bidirectional inference for intermediate speed/accuracy trade-offs

Primary target:
- same weights for streaming and non-streaming ASR
- efficient streaming / chunked ASR
- strong long-form robustness
- practical multi-GPU training on `4 x 4090` with `DeepSpeed`

Non-goals for v0:
- RNN-T or AED decoder
- replacing MLP with RWKV channel mixing
- exact reproduction of the paper's WeNet + RNN-T setup

## 2. Paper Analysis

Paper: `2506.19761v1.pdf`

Title:
- `Accurate, fast, cheap: Choose three. Replacing Multi-Head-Attention with Bidirectional Recurrent Attention for Long-Form ASR`

What the paper actually shows:
- The authors keep the overall Conformer encoder structure and only swap the attention computation.
- For RWKV, they use only the time-mixing component as the replacement for MHA. They explicitly do not use RWKV channel mixing because they expect the existing feed-forward and convolution modules to cover local transformations.
- Bidirectional recurrent attention closes the accuracy gap to MHA on short-form ASR.
- On long-form ASR, RWKV-based encoders generalize better than full-context MHA when training utterances are short and decoding chunks are long.
- After long-form fine-tuning, bidirectional RWKV beats a strong limited-context attention baseline in accuracy while delivering about `44%` higher throughput; unidirectional RWKV matches that baseline with about `72%` higher throughput.
- Direction Dropout is the key mechanism that makes a single bidirectional-trained model usable in both bidirectional and unidirectional inference modes with limited accuracy loss.
- Alternating-direction decoding shows that the model can mix future and past information without requiring both directions to be active in every layer.

Numbers directly relevant to this project:
- Short-form table: unidirectional RWKV underperforms MHA, while bidirectional RWKV matches it closely.
- Long-form table: bidirectional RWKV reaches roughly the same or slightly better WER than MHA / LCA at materially higher throughput.
- Throughput table: unidirectional RWKV is fastest; bidirectional RWKV offers a better speed/accuracy point than limited-context MHA.

Implications for our design:
- The mainline model should be `bidirectional RWKV encoder + Direction Dropout training`, not a separate causal baseline.
- Unidirectional inference is a deployment mode of the same model, not a separate model family.
- The training recipe must explicitly optimize for `single-checkpoint dual-mode behavior`.
- Long-form training and chunked evaluation are not optional extras. They are core to model behavior.

## 3. Upstream RWKV-7 Constraints

Source analyzed:
- official `RWKV-LM` repo
- path: `RWKV-v7/train_temp`

Important implementation constraints extracted from upstream:
- The main reusable component is `RWKV_Tmix_x070`.
- It depends on:
  - `head_size`
  - per-layer mix parameters `x_r/x_w/x_k/x_v/x_a/x_g`
  - learned low-rank adapters for decay, value residual, in-context rate, and gate
  - `v_first` value residual passed from the first layer through the whole stack
- The fused CUDA path currently assumes:
  - `HEAD_SIZE == 64`
  - `CHUNK_LEN == 16`
  - bf16 tensors
  - contiguous layout
- The upstream module uses custom initialization and optimizer grouping:
  - `att.w0` has `2x` learning-rate scale
  - weight decay applies only to large matrix weights
  - several output projections are zero-initialized
- Upstream `train_temp` is not a reusable general-purpose layer package. We should port the TimeMixer logic and preserve the optimization rules, not embed the LM trainer unchanged.

## 4. Product Requirements

### Functional
- Train a CTC ASR encoder-decoder system with RWKV-7 TimeMixer replacing self-attention.
- Support offline full-utterance training and chunked streaming-style inference from the same checkpoint.
- Support variable-length batches and long-form utterances.
- Support Whisper multilingual BPE text targets.
- Support configurable text tokenizers:
  - Whisper multilingual BPE
  - SentencePiece Unigram via external `.model`
- Save and load model architecture via YAML config files for training and prediction workflows.
- Save and load tokenizer settings via a separate `tokenizer_config.yaml` so prediction defaults stay aligned with the training checkpoint.
- Export inference-friendly `safetensors` weights from `.pt` checkpoints for deployment-oriented prediction workflows.
- Export Rust-consumable `CTC logits + lengths + utt_ids` artifacts from Python model forward passes so the first Candle predictor can be validated before full RWKV Rust forward parity exists.
- Expose inference modes:
  - left-to-right streaming
  - bidirectional offline
  - alternating directions
  - configurable subset of bidirectional layers

### Performance
- Mixed precision `bf16`
- DeepSpeed multi-GPU training on 4 GPUs
- kernel-aware path for RWKV-7 when shapes and dtypes are compatible
- fallback pure PyTorch path for correctness and unit tests
- on CUDA, runtime activations should stay in `bf16` outside numerically sensitive reduction paths

### Research
- Establish the dual-mode single-checkpoint path first.
- Keep interfaces compatible with:
  - fixed-rate paper-reproduction Direction Dropout
  - optional non-default Direction Dropout schedules for ablations
  - unidirectional / bidirectional / alternating inference from one checkpoint
  - per-layer direction masks at inference

## 5. Proposed Architecture

### 5.1 Encoder

Base shape:
- input: `B x T_audio`
- features: 80-dim log-Mel filterbanks
- frontend default: `global_cmvn + conv2d6` over 80-dim fbank features
- subsampling: WeNet-style `conv2d6` subsampling
- encoder hidden size: initial default `d_model = 512`
- FFN size: initial default `d_ff = 2048`
- layers: initial default `12`
- RWKV head size: `64`
- RWKV attention dim: `512`

Encoder block:
1. macaron FFN
2. bidirectional RWKV-7 TimeMixer in place of self-attention
3. depthwise convolution module
4. post FFN
5. layer norms and residual paths in Conformer style

Rationale:
- This matches the paper's main modeling choice: change the attention computation, keep the rest of the encoder structure.
- It also keeps a strong ASR inductive bias for local acoustic modeling.

### 5.2 RWKV TimeMixer Adaptation

Adaptation rules:
- Input is acoustic hidden states, not token embeddings.
- We port the `RWKV_Tmix_x070` equations into an ASR module.
- We do not port `RWKV_CMix_x070` into the encoder block for v0.
- We preserve `v_first` propagation across encoder layers.
- We define two execution backends:
  - `native`: pure PyTorch reference implementation
  - `cuda`: upstream-compatible fused CUDA extension path using RWKV-7 `wind_backstepping`

CUDA backend rules:
- Source is vendored from `BlinkDL/RWKV-LM/RWKV-v7/train_temp/cuda` under Apache-2.0 with local attribution.
- The first CUDA backend target is training full-sequence `state=None` TimeMixer calls.
- Streaming / chunked inference with an incoming recurrent state remains on the native backend until a stateful fused kernel wrapper is added.
- Runtime requirements:
  - CUDA tensors
  - bf16 contiguous `w/q/k/v/z/a` tensors
  - `head_size = 64`
  - `chunk_len = 16`
- If the CUDA backend is requested but these requirements are not met, training must fail loudly or fall back only for explicitly unsupported stateful inference paths; it must not silently run the slow native path for the main training loop.
- Before large training runs, validate the active CUDA toolchain with `rwkvasr-check-rwkv7-cuda` or `python -m rwkvasr.cli.check_rwkv7_cuda`; this compiles the extension and compares fused output against the native reference on a small tensor.

Directional design:
- Each encoder block contains:
  - an `L2R` TimeMixer branch
  - an `R2L` TimeMixer branch operating on the reversed sequence
- The default merge is elementwise average after restoring the reversed branch to original order.
- Training-time direction masks can disable one branch for a block.
- Inference-time direction masks can select:
  - all bidirectional blocks
  - all unidirectional blocks
  - alternating directions
  - only the last `N` blocks bidirectional

Current implementation status:
- native TimeMixer core is implemented and is the correctness reference
- bidirectional wrapper, Direction Dropout, CTC training, WebDataset bucketing, and prediction are implemented
- fused CUDA TimeMixer is being added as the primary Emilia-scale training backend because native recurrence is too slow for multi-million-hour-step schedules

### 5.3 Direction Dropout

Primary training mechanism:
- During training, each bidirectional RWKV block can drop one direction according to a scheduler.
- For strict paper reproduction, Direction Dropout should use a fixed dropout rate during training rather than a progressive schedule.
- Supported variants:
  - `drop_r2l_only`: optimized for L2R streaming deployment
  - `drop_both`: randomly drop either direction for symmetric dual-mode behavior

Training-time outputs:
- same checkpoint should remain usable under `Bi`, `L2R`, and `Alt` inference
- we will log mode-specific dev WER during training

### 5.4 Positional / Temporal Handling

Decision:
- Do not add Transformer-style relative self-attention position bias, because TimeMixer already injects temporal mixing through shift-based recurrence.
- Keep convolution module for local temporal inductive bias.

Open point:
- We may still need lightweight positional scaling around subsampling or conv modules if convergence is unstable. This remains an ablation item.

### 5.5 CTC Head

Head:
- encoder output projection to vocabulary size plus blank
- CTC loss during training
- greedy decode for lightweight evaluation sanity checks
- standalone `CTC prefix beam search` for prediction and later hotword-bias integration
- beam size must be configurable from the prediction CLI and YAML-driven model reconstruction path
- prediction CLI should expose decode-time insertion / length bonus so CTC deletion bias can be inspected without retraining
- prediction outputs must support token-level CTC time alignment, not only token ids / text
- labeled prediction outputs should optionally include decode diagnostics:
  - input feature length
  - encoder / logit length
  - predicted token count
  - reference token count
  - blank-top1 ratio
  - average blank probability

Why CTC first:
- simpler than RNN-T
- lower latency
- easier streaming path
- easier profiling of encoder gains from RWKV replacement

### 5.6 Dual-Mode Inference

Target behavior:
- `Bi` mode for non-streaming ASR with both directions enabled
- `L2R` mode for streaming ASR using the same checkpoint with only the forward direction enabled
- `Alt` mode for intermediate cost / accuracy trade-off
- optional mixed mode where only selected layers are bidirectional

Design rule:
- Mode switching must not require weight conversion or checkpoint rewriting.

### 5.7 Streaming Inference

Target behavior:
- `L2R` mode consumes audio in chunks
- cache convolution states and RWKV recurrent states across chunks
- output partial CTC hypotheses incrementally

State interfaces needed:
- subsampling cache
- conv cache
- RWKV recurrent state per layer
- optional emitted-token post-processing state for CTC collapse
- token-alignment post-processing state for incremental / full-utterance CTC timestamps

Note:
- Upstream `train_temp` focuses on full-sequence LM training. We will need:
  - an ASR-specific recurrent-state API for chunked `L2R` inference
  - a separate non-streaming execution path for `Bi` / `Alt`

## 6. Training Design

### 6.1 Data

Expected pipeline:
- manifest-based dataset
- WebDataset tar-shard dataset
- audio loading via `torchaudio`
- on-the-fly fbank extraction
- optional SpecAugment
- Whisper multilingual tokenization
- optional SentencePiece Unigram tokenization with externally provided `.model`

WeNet `examples/gigaspeech/s0` alignment:
- resample to `16k`
- `fbank` with:
  - `num_mel_bins = 80`
  - `frame_length = 25ms`
  - `frame_shift = 10ms`
  - `dither = 1.0`
- optional `SpecAugment`
- `global_cmvn`
- encoder input layer uses `conv2d6` subsampling before the encoder blocks

Default frontend contract:
- project default frontend is `WeNet-style fbank -> global_cmvn -> conv2d6 -> RWKV encoder`
- `linear` frontend remains only as an explicit fallback for debugging, unit tests, and ablations
- train / eval CLI defaults must use `conv2d6`

Tokenizer contract:
- tokenizer choice must be explicit in training config through:
  - `tokenizer_type`
  - `tokenizer_model_path`
  - `tokenizer_language`
  - `tokenizer_task`
- `vocab_size` may be inferred from the tokenizer; for SentencePiece, an explicit `vocab_size` must match the model's actual vocab size
- training outputs must save `tokenizer_config.yaml` next to `model_config.yaml`
- prediction CLIs should automatically reuse `tokenizer_config.yaml` when explicit tokenizer flags are not passed
- multilingual SentencePiece candidate preparation should support a `FLORES-200 dev+devtest -> unigram 4k/8k` path for controlled tokenizer comparisons

Global CMVN policy:
- `global_cmvn` statistics are computed on the training split only
- stored in WeNet-compatible JSON with:
  - `mean_stat`
  - `var_stat`
  - `frame_num`
- runtime normalization is:
  - `mean = mean_stat / frame_num`
  - `var = var_stat / frame_num - mean^2`
  - `istd = 1 / sqrt(max(var, 1e-20))`
  - `x_norm = (x - mean) * istd`
- runtime dtype contract is:
  - audio features may be extracted in `fp32` on CPU
  - after transfer to CUDA, batch feature tensors should be cast to `bf16`
  - `global_cmvn` should execute from `fp32` inputs for numerical stability, then cast back to the frontend / encoder compute dtype
  - `conv2d6`, RWKV blocks, and the CTC projection should run in `bf16` on CUDA
  - `log_softmax` and `ctc_loss` should upcast logits to `fp32`
- if `frontend_type == conv2d6` and no `cmvn_file` is supplied, the training entrypoint should materialize a default `global_cmvn.json` under `output_dir` and reuse it for later evaluation

Current implementation status:
- lightweight WeNet-style `fbank + global_cmvn + conv2d6` frontend extracted into this repo
- current train/eval pipeline still accepts precomputed features too, to keep the project usable without full audio manifests
- default training and evaluation path should now assume `conv2d6 + global_cmvn`

WebDataset support:
- training data may live in a directory of `.tar` shards
- inspected dataset root: `/home/yueyulin/data/voxbox/wenetasr`
- current observed structure:
  - `102` shard files
  - total size about `82G`
  - each sample is a pair of:
    - `{key}.wav`
    - `{key}.json`
- observed json schema:
  - `text`
  - `language`
  - `sample_rate`
  - `format`
  - `begin_time`
  - `end_time`
  - `confidence`
  - `sid`
- loader requirements:
  - stream tar shards without depending on the WeNet framework
  - decode audio bytes on the fly during training
  - extract WeNet-style fbank features on the fly
  - tokenize from `json["text"]` unless `token_ids` are already present
  - use `sid` as utterance id when available
  - partition shards across distributed ranks and dataloader workers
  - training entrypoint must accept both manifest and WebDataset roots
- automatic `global_cmvn` computation should work for both data sources
- for long-form multi-GPU training, WebDataset should support an optional offline `length index` built from tar metadata without decoding full audio
- offline dataset-wide preprocessing should live under `tools/` and use `Rust + multi-threading` by default; Python implementations remain only as fallback utilities for tests and debugging
- preferred training path with a length index is:
  - offline scan tar metadata into per-sample `num_frames`
  - build a map-style dataset keyed by `(shard_name, sample_key)`
  - form global batches from length-sorted indices
  - shuffle the batch order, not the within-batch length ordering, so each epoch covers the full length range while consecutive steps still see different length bands
  - compute local batch size dynamically from offline `num_frames`, so long utterances get a smaller `B` and short utterances get a larger `B`
  - split each global batch evenly across ranks so all ranks see a similar length band on the same step
- runtime fallback `B -> B'` token-budget shrinking remains only as a safety valve for residual text-length variance or rare outliers; it must not be the primary batching mechanism
- one epoch must cover the whole train split across all supported lengths; it is invalid to dedicate an epoch to only one length band
- current Python in-memory length-bucketing path is only valid for moderate `webdataset_lengths.jsonl` sizes
- when the length index grows beyond practical in-memory limits, training must:
  - avoid full JSONL loads during startup
  - log an explicit warning
  - fall back to the plain streaming WebDataset loader
  - compute `steps_per_epoch` from split sample counts instead of the offline length index
- exact large-scale length bucketing for Emilia-sized corpora requires a future binary / external-memory sampler; do not pretend the current in-memory JSONL sampler scales to tens of millions of utterances
- large-scale replacement for the current in-memory sampler:
  - Rust preprocessing should build a `bucket manifest` from the large length index
  - bucket id is derived from `num_frames // bucket_width`
  - default `bucket_width = 80` frames
  - output layout should be:
    - one compact manifest JSON
    - many small bucket part files capped by `entries_per_part`
  - each part file should preserve source shard order so samples from the same tar stay contiguous as much as possible
  - training must read only:
    - the small manifest during startup
    - the current bucket part while iterating
  - training must not load all bucket entries into memory at once
  - per-step bucket policy:
    - all ranks use the same bucket on the same step
    - each rank consumes a disjoint slice from that bucket's next global batch
    - local batch size is derived from the bucket's upper frame bound and the configured frame budget
  - each epoch must still cover the full train split across all buckets, with shuffled bucket-step order rather than single-bucket epochs
  - runtime parallelism constraint:
    - bucket-manifest training must preserve strict step ordering across ranks
    - do not let asynchronous worker scheduling reorder bucket steps, or distributed collectives can desynchronize
  - bucket-manifest runtime parallelism should therefore be implemented as:
    - one deterministic per-rank step iterator
    - thread-parallel local sample decode inside the current step
    - a small bounded prefetch queue for upcoming decoded batches
    - decoded prefetch must include tar reads, audio decode, feature extraction, tokenization, and collate, not only JSONL entry lookup
    - optional thread-local tar readers so repeated reads from the same shard reuse file handles safely
  - `num_workers` on the bucket-manifest path is interpreted as local decode / prefetch worker count, not as a generic DataLoader worker pool
- inspected Emilia Chinese dataset root: `/media/usbhd/training_data/asr/emilia/Emilia/ZH/Emilia/ZH`
- Emilia current observed structure:
  - flat directory of `.tar` shards, total size about `1.2T`
  - member pairs are `json + mp3`
  - json metadata already includes:
    - `id`
    - `wav`
    - `text`
    - `duration`
    - `speaker`
    - `language`
    - `dnsmos`
- Emilia-specific read policy:
  - do not unpack the dataset
  - do not build the training path around Python `tarfile.extractfile()` random member lookups
  - preferred path is:
    - Rust multi-threaded offline scan of tar headers plus json metadata only
    - infer `num_frames` directly from metadata `duration`
    - record `audio_member`, `audio_format`, `audio_offset`, `audio_size`, `json_offset`, and `json_size`
    - at training time, open each tar shard as a binary file and read members by byte offset rather than by tar-name rescans
  - this preserves the current length-bucketing design while avoiding a full audio decode during indexing and avoiding Python tar random-access overhead during training
- Emilia multilingual mixing policy:
  - if language-specific shard names are already unique, prefer building a virtual mixed root from symlinks rather than adding a bespoke multi-root training path
  - current observed shard naming is safe for this approach:
    - Chinese: `ZH-B*.tar`
    - English: `EN-B*.tar`
  - the mixed root should contain symlinks to both language directories and then reuse the normal single-root WebDataset workflow:
    - `inspect_webdataset`
    - Rust `webdataset_lengths.jsonl`
    - `global_cmvn`
    - standard training / eval configs
  - this keeps split logic, length bucketing, and checkpoint bookkeeping identical to the monolingual path
  - preprocessing and training must use the same `utt_id_key`
  - Emilia datasets use metadata key `id`, so:
    - `inspect_webdataset`
    - `inspect_webdataset_lengths`
    - `compute_cmvn`
    - training / eval configs
    must all set `webdataset_utt_id_key: id`
  - it is invalid to let training fall back to the project default `sid` once an index was built with `id`
- WebDataset length-index schema should no longer assume `.wav` audio:
  - audio entries must be generic `audio_member`
  - current minimum supported audio formats are:
    - `wav`
    - `mp3`
    - `flac`
  - old indexes that only store `wav_member` should remain readable for backward compatibility
- on the target `4 x RTX 4090` setup, default DeepSpeed ZeRO-2 configs should not offload optimizer state to CPU unless explicitly requested for an ablation; current model sizes and observed bf16 activation footprints leave enough GPU headroom, and CPU offload reduces utilization
- for 4090 training configs, prefer:
  - larger `max_local_batch` upper bounds
  - offline length-bucket frame budgets as the primary limiter
  - runtime token budget only as a final safety valve
- training outputs must be epoch-centric in addition to step-centric:
  - compute mean `train_loss` for each epoch
  - run one `eval_loss` pass on the eval split at the end of each epoch
  - support `max_eval_samples` so large eval splits can use a capped subset on intermediate epochs
  - when `epochs` is specified and `max_eval_samples` is set, the final epoch must still run full eval
  - save `epoch-N` checkpoints
  - track and export the current `best` checkpoint based on lowest eval loss
  - persist epoch history so checkpoint selection can be done after training without parsing logs
- experiment tracking:
  - support optional `wandb` logging for both single-process and DeepSpeed training
  - DeepSpeed logging must be rank-0 only
  - tracked metrics should include:
    - periodic train loss / data time / step time / rate
    - progress-equivalent fields that mirror the terminal view closely enough for remote monitoring: progress fraction, ETA, effective batch size, token counts, and peak memory
    - epoch train/eval loss
    - step-checkpoint sampled eval loss
    - checkpoint selection metadata
  - `wandb` project name and run name must be configurable from YAML / CLI
  - `wandb` base URL and init timeout must be configurable from YAML / CLI so self-hosted and cloud deployments can both work
  - `wandb` initialization failure must not block training startup; log the failure and continue without experiment tracking unless the user explicitly chooses strict failure behavior in a later spec revision
- step-level checkpoint policy:
  - support a configurable `step_eval_samples` subset for periodic step checkpoints
  - support a configurable `step_eval_every` cadence; if unset, reuse the step checkpoint save cadence
  - periodic step eval must use a randomized eval subset rather than always the first `N` samples
  - randomized step eval must still preserve offline length bucketing when a bucket manifest or offline length index is available; do not fall back to an unbucketed eval loader just to shuffle sample order
  - eval and step-eval batch sizes should be independently configurable from the train batch size so CTC `log_softmax` does not OOM on long eval batches
  - keep only the best `top_k_step_checkpoints` according to sampled `eval_loss`
  - track step-checkpoint selection metadata in a dedicated YAML so later manual checkpoint inspection does not require parsing logs
  - epoch checkpoints and the final epoch-level `best` checkpoint remain separate from step-checkpoint retention
- tokenizer switching rule:
  - changing tokenizer type or vocabulary size changes the CTC target space and output head shape
  - training cannot resume across tokenizer changes; a new run must start from step 0
- large-scale multilingual CTC preference:
  - for Emilia-scale multilingual training, prefer `SentencePiece Unigram 8k` over Whisper multilingual BPE as the default efficiency-oriented tokenizer
  - rationale:
    - much smaller CTC head
    - lower logits / `log_softmax` memory pressure
    - lower eval OOM risk
    - fewer total training steps needed once the larger local batch becomes feasible
  - Whisper multilingual BPE remains a baseline / compatibility tokenizer, not the default efficiency choice
- continue-training support should prioritize epoch checkpoints:
  - single-process training may resume from exported `.pt` checkpoints
  - DeepSpeed training should resume from `ds_checkpoints/` plus `resume_tag`, typically `epoch-N` or `best`
  - checkpoint extra/client_state must persist `step`, `epoch`, epoch history, and current best metric state
- v0 Rust preprocessing scope:
  - `webdataset_lengths.jsonl` generation from tar metadata
  - multi-threaded per-shard scanning
  - record optional tar member offsets and sizes for zero-copy-style random-access reads from plain `.tar` shards
  - summary JSON with split counts and frame bucket histogram
  - future offline CMVN / manifest materialization may also move under `tools/`, but that is not required for the first slice
- v0 Rust prediction scope:
  - standalone `Candle` tool under `tools/`
  - load exported `safetensors` tensors for `logits` and `lengths`
  - run configurable `CTC prefix beam search`
  - write JSONL utterance-level hypotheses with token-level time alignment
  - full RWKV encoder forward parity in Rust is a later slice and must not be implied by the first predictor tool
- prediction/export workflow is explicitly split into two supported paths:
  - Python full-model prediction from `.pt` or `.safetensors` checkpoints
  - Python export of `CTC logits + lengths + utt_ids` followed by Rust/Candle decode + token-level time alignment
- Candle custom-op design constraints for the later RWKV Rust port:
  - user-land custom ops are exposed via `CustomOp1`, `CustomOp2`, and `CustomOp3`
  - inference-only fused kernels should prefer `apply_op*_no_bwd`, because backward support is unnecessary for deployment
  - each custom op backend must be implemented separately:
    - `cpu_fwd`
    - optional `cuda_fwd`
    - optional `metal_fwd`
  - Candle passes arbitrary storage + `Layout`, so CPU fallback code must handle non-contiguous tensors and not assume dense contiguous buffers
  - because custom-op arity stops at 3 tensors, the RWKV fused op should only own the recurrent core; pre/post projections and simple elementwise transforms should remain regular Candle graph ops
  - any fused RWKV op should consume prepacked runtime tensors plus scalar kernel config, rather than trying to embed the whole TimeMixer module as a single opaque op

WebDataset indexing and split policy:
- do not split `train / eval` by shard file because tar packing may not be randomized
- use `sample-level stable hash split` instead
- split key should be:
  - `json["sid"]` if present
  - otherwise tar sample key
- provide an `inspect_webdataset` command that scans shards and writes an index file with at least:
  - total shard count
  - total sample count
  - per-shard sample count
  - per-split sample count under a stable-hash split rule
- train / eval / CMVN must all reuse the same split rule
- when training from WebDataset by `epochs`, compute `steps_per_epoch` from the indexed split sample count, not from shard count

Tokenizer policy:
- default text tokenizer is `Whisper multilingual tokenizer`
- tokenizer source should be fixed upstream assets, not retrained on the current ASR acoustic dataset
- CTC target vocabulary uses only the Whisper `text token` range
- language / task / timestamp / other control tokens are excluded from the default CTC label set
- CTC `blank` is added by this project on top of the Whisper text-token vocabulary
- this gives:
  - no `<unk>` / OOV behavior for normal UTF-8 text
  - a tokenizer that is not tightly coupled to current training-set growth
  - a much smaller CTC head than Qwen / Gemma class LLM tokenizers

YAML config policy:
- training must emit at least:
  - `model_config.yaml`
  - `train_config.yaml`
- public repository packaging must:
  - keep `README.md` aligned with the current implemented slices and open gaps
  - keep `ROADMAP.md` aligned with the public GitHub task breakdown for remaining milestones
  - document latest public training summary without committing checkpoints or run artifacts
  - exclude `runs/`, DeepSpeed states, exported weights, and local datasets from version control
- training CLI should also accept a YAML config file as the primary launch interface
- when both YAML and CLI args are present, only explicitly provided CLI args may override YAML fields
- DeepSpeed multi-GPU training should use the same YAML-first interface
- training YAML may embed a `deepspeed` mapping directly instead of forcing a separate JSON file
- prediction / evaluation should prefer reconstructing model architecture from YAML rather than repeated CLI shape arguments
- standalone prediction should live under a separate `predict` module / CLI and must not depend on training-loop entrypoints
- repository should also provide a labeled prediction CLI for dev/eval inspection:
  - consume manifest or WebDataset eval samples with references
  - run configurable CTC prefix beam search
  - support decode-time insertion / length bonus ablations
  - save `pred_text`, `ref_text`, token ids, and token-level alignments to JSONL
  - optionally save per-utterance decode diagnostics so short-hypothesis failures can be distinguished from frontend truncation bugs
  - support a small-sample preview workflow before running full WER/CER
- Rust deployment tooling should live under `tools/`; the first `Candle` predictor may target exported `CTC logits + lengths` rather than full RWKV encoder parity
- exported inference checkpoints should support:
  - `.pt` for training resume
  - `.safetensors` for inference loading in Python and future Rust model loading
- decode-export artifacts should support:
  - batched `part-xxxxx.safetensors` files with `logits` and `lengths`
  - sidecar `part-xxxxx.utt_ids.txt`
  - an index JSON summarizing exported parts, mode, blank id, frontend type, and timestamp projection config
- manual CLI shape arguments remain only as a fallback path
- repository should ship minimal example YAML files for training and model architecture so configs can be exchanged directly

Training stages:
1. short/mid-form bidirectional base training
2. fixed-rate Direction Dropout training for dual-mode robustness
3. long-form fine-tuning with concatenated or naturally long utterances
4. streaming/chunk robustness validation in `L2R` mode
5. offline evaluation in `Bi` and `Alt` modes

### 6.2 Optimization

Baseline optimizer rules derived from upstream:
- AdamW or DeepSpeed fused Adam
- `att.w0` uses `2x` LR scale
- weight decay only on large matrix weights
- bf16 training
- gradient checkpointing optional

DeepSpeed mainline policy for target 4090 training:
- optimizer should use `DeepSpeedCPUAdam` in AdamW mode
- default ZeRO mode should be `ZeRO-2`
- optimizer state should use `cpu offload`
- encoder activations should enable gradient checkpointing by default to maximize usable context / batch
- rank0 training output should include live progress, ETA, and per-step timing because audio decode / fbank extraction is online and the first batch can be much slower than steady-state
- variable-length training should use a `batch token budget` policy rather than blindly fixing `batch_size`
- the practical budget metric should be `padded_audio_tokens = B * max_audio_frames`, plus `text_tokens = sum(target_lengths)`
- training should log the observed batch token count, peak memory, and a `22 GiB` target-budget estimate from real runs
- when a candidate loader batch exceeds the configured token budget, the safe default is to shrink the effective local batch from `B` to the largest budget-compliant prefix `B'` for that step
- under `DDP` / `DeepSpeed`, token-budget enforcement must happen before the model step at batch-construction time; per-rank variable counts of forward/backward calls inside one logical step are not safe
- do not execute the dropped tail samples from an oversized candidate batch inside the same logical step
- we should not truncate audio time `T` against a full transcript by default, because that breaks CTC alignment semantics unless transcript segmentation is also available
- pathological single samples larger than the token budget should be isolated or skipped with an explicit warning rather than causing an OOM crash

Initial defaults:
- `lr = 4e-4` to `6e-4` range for base runs
- warmup enabled
- ZeRO-2 first, ZeRO-3 only if memory forces it
- DeepSpeed training YAML should normalize to:
  - `zero_optimization.stage = 2`
  - `zero_optimization.offload_optimizer.device = cpu`
  - `zero_optimization.offload_optimizer.pin_memory = true`

Direction Dropout for paper reproduction:
- The paper describes DirDrop as dropping one recurrent direction with a set probability.
- The experiment section states `For DirDrop, we use a dropout rate of 20%.`
- The strict reproduction default should therefore be:
  - `direction_variant = drop_both`
  - `p_start = 0.2`
  - `p_max = 0.2`
  - `warmup_steps = 0`
  - `ramp_steps = 0`
- Scheduler-based ramping can remain in code only as a non-default ablation path.

Why not reuse upstream Lightning trainer:
- `train_temp` is tailored to LM training and old Lightning.
- This project needs ASR losses, variable-length audio batching, chunked validation, and simpler maintainability.
- We should reuse the optimization policy, not the full framework.

### 6.3 Hardware Plan

Target environment:
- `4 x RTX 4090`
- `torch 2.10`
- `CUDA 13.0`
- `DeepSpeed`
- managed by `uv`

Execution policy:
- keep all project dependencies in `pyproject.toml`
- use `uv` virtualenv and lockfile when environment sync becomes possible
- compile the fused CUDA extension against the active torch / CUDA toolchain
- default distributed launcher path is `deepspeed --num_gpus ... rwkvasr-train-ctc-deepspeed --config-yaml ...`
- when using `DeepSpeedCPUAdam` under a `torch cu128 + system CUDA 13.0` environment, allow `DS_SKIP_CUDA_CHECK=1` for the CPU optimizer build path
- training artifacts should include:
  - `model_config.yaml`
  - `train_config.yaml`
  - `deepspeed_config.yaml`
  - DeepSpeed checkpoint directories by step tag
  - a plain inference-friendly `step-<N>.pt` checkpoint when the active ZeRO stage allows direct export

## 7. Evaluation Plan

Required validation axes:
- CTC loss decreases on 1-batch overfit
- offline greedy decode sanity in `Bi` mode
- offline prefix beam search sanity with configurable `beam_size`
- Rust/Candle prefix beam search sanity from exported `safetensors logits`
- token-level CTC time alignment sanity in both Python and Rust predictors
- streaming chunked decode sanity in `L2R` mode
- same-checkpoint `Bi / L2R / Alt` decode sanity
- WER/CER on dev set
- latency and throughput under:
  - batch size 1
  - streaming chunk sizes
  - offline long-form chunks

Minimum benchmark outputs:
- tokens or frames per second during training
- audio seconds per second during inference
- peak GPU memory
- WER by chunk size
- WER delta between `Bi` and `L2R` from the same checkpoint
- WER delta between `Bi` and `Alt` from the same checkpoint

## 8. Implementation Plan

### Phase 0: Project Skeleton
- create `uv` project files
- define config layout
- create package skeleton
- add test harness

Status:
- completed in current iteration

### Phase 1: RWKV-7 Core
- port `RWKV_Tmix_x070` into `src/`
- implement reference path without fused CUDA
- add shape and numerical sanity tests

Status:
- in progress
- native recurrent reference path implemented
- basic shape / `v_first` / chunk-consistency checks implemented
- fused CUDA backend not wired into this project yet

### Phase 2: Bidirectional RWKV Wrapper
- implement per-block `L2R` + `R2L` wrapper
- implement reverse / restore sequence utilities
- implement merge strategy and inference mode masks
- add unit tests for `Bi`, `L2R`, and `Alt`

Status:
- in progress
- bidirectional wrapper, reverse/restore, and mode-mask helpers implemented
- module-level inline validation passed

### Phase 3: Direction Dropout
- implement per-layer direction-drop masks
- implement fixed-rate paper-reproduction drop probability
- support `drop_r2l_only` and `drop_both`
- log active-direction statistics during training

Status:
- in progress
- fixed-rate DirDrop and training mask sampling implemented
- scheduler ramp path retained only for non-default ablations
- logging and trainer integration not implemented yet

### Phase 4: ASR Encoder
- implement Conformer-style block with bidirectional RWKV TimeMixer replacing self-attention
- add causal convolution cache support for `L2R`
- add encoder stack state API

Status:
- in progress
- single-block Conformer-style RWKV block implemented
- causal convolution cache and `L2R` chunk-state consistency validated with inline checks
- encoder stack not implemented yet

### Phase 5: CTC Model
- add frontend, tokenizer integration, CTC projection, loss
- add greedy decode
- add standalone prefix beam search prediction path

Status:
- in progress
- minimal feature-to-encoder-to-CTC logits/loss chain implemented
- greedy decode and token-level evaluation implemented
- standalone prefix beam search prediction path not implemented yet

### Phase 6: Training System
- manifest dataset
- collator
- DeepSpeed training entrypoint
- checkpoint save/load

Status:
- in progress
- upstream-style optimizer parameter grouping implemented
- Direction Dropout integrated into a minimal CTC training wrapper
- synthetic one-batch overfit runner implemented and validated
- JSONL manifest dataset, collator, checkpoint save/load, and a minimal train CLI implemented
- DeepSpeed entrypoint not implemented yet

### Phase 7: Dual-Mode Validation
- chunked `L2R` streaming runner
- `Bi` and `Alt` offline runners
- benchmark and WER scripts
- same-checkpoint mode-comparison reports
- standalone prediction CLI with JSONL outputs

Status:
- in progress
- greedy CTC decoding and same-checkpoint `Bi/L2R/Alt` token-level evaluation implemented
- current metric is token error rate on manifest token ids, not final WER/CER with text normalization
- standalone prefix-beam prediction CLI and exported utterance-level hypotheses are the next slice

## 9. Immediate Next Steps

1. Finish `uv sync` so `pytest`, `ruff`, and runtime dependencies are available in `.venv`.
2. Implement the bidirectional TimeMixer wrapper and mode masks.
3. Implement fixed-rate Direction Dropout first; keep scheduler logic only for ablations.
4. Add optimizer parameter-group helpers that preserve upstream `att.w0` lr scaling and matrix-only weight decay.
5. Extend verification with runnable unit tests for:
   - forward shape
   - `v_first` propagation
   - direction masking
   - `Bi` vs chunked `L2R` mode semantics
   - chunk padding to fused-kernel constraints
   - causal chunk-by-chunk consistency in `L2R`

## 10. Open Questions

- Which dataset is first target: AISHELL, GigaSpeech, LibriSpeech, internal manifests, or mixed?
- Should we keep a byte-level tokenizer as an ablation against Whisper multilingual text-only CTC targets?
- What is the target `Bi -> L2R` WER delta that is acceptable for the first milestone?
- Should the fused CUDA kernel be mandatory for training, or optional behind a feature flag until the pure PyTorch path is verified?
