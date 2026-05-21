# RWKV ASR Tools

Offline dataset preprocessing and deployment-oriented Rust utilities live here.

Current tools:
- `rwkvasr-tools` binary: multithreaded Rust WebDataset length indexer that scans tar metadata without decoding audio
- `build_bucket_index` binary: Rust external-memory bucket manifest builder over a large `webdataset_lengths.jsonl`
- `convert_asr_corpus` binary: Rust multi-threaded converter from GigaSpeech parquet or WenetSpeech Lhotse cuts into canonical WebDataset tar shards
- `predict_ctc` binary: Rust + Candle CTC prefix-beam predictor over exported `safetensors logits + lengths`
- `rwkvasr-convert-asr-corpus` Python CLI: small fallback/debug converter only; large corpora should use the Rust binary through the scripts below

New ASR corpus preprocessing:

```bash
# GigaSpeech XL HuggingFace parquet train shards -> WebDataset + length buckets.
# Scripts resume by default. Use RESUME=0 OVERWRITE=1 only for a clean rebuild.
./scripts/prepare_gigaspeech_xl.sh \
  /media/usbhd/training_data/asr/speechcolab/gigaspeech/parquet-data/xl \
  /media/usbhd/training_data/asr/speechcolab/gigaspeech/webdataset_xl_train_runtime

# GigaSpeech defaults to TEXT_NORMALIZATION=runtime:
# - lowercases English
# - maps <COMMA>/<PERIOD>/<QUESTIONMARK> to real punctuation
# - preserves non-speech tags such as <NOISE>/<MUSIC>/<SIL>
# - stores the original transcript in source_text when it changes

# WenetSpeech Lhotse cuts_L shards -> WebDataset + length buckets.
./scripts/prepare_wenetspeech_l.sh \
  /media/usbhd/training_data/asr/wenet-e2e/wenetspeech/data \
  /media/usbhd/training_data/asr/wenet-e2e/wenetspeech/webdataset_l_train

# Optional mixed EN+ZH training root from the two converted roots.
./scripts/prepare_gigaspeech_wenetspeech_mix.sh \
  /media/usbhd/training_data/asr/speechcolab/gigaspeech/webdataset_xl_train_runtime \
  /media/usbhd/training_data/asr/wenet-e2e/wenetspeech/webdataset_l_train \
  /media/usbhd/training_data/asr/mix/gigaspeech_xl_runtime_wenetspeech_l_webdataset
```

The conversion roots use `id` as `webdataset_utt_id_key`, and the scripts build:
- `webdataset_index.json`
- `webdataset_lengths.jsonl`
- `webdataset_buckets/manifest.json`
- `global_cmvn.json` unless `COMPUTE_CMVN=0`

The default scripts call the Rust converter first, then reuse the existing Python index/CMVN CLIs plus Rust length/bucket tools. WenetSpeech conversion does not need Lhotse at runtime; it reads the existing `cuts_*.jsonl.gz` metadata and paired `cuts_*.tar.gz` cut-level wav archives.

For joint `CTC + RWKV AR` training, the bucket builder should not use audio length alone. The scripts default to combined-cost bucketing:

```text
bucket_cost = audio_frames + BUCKET_TEXT_COST_WEIGHT * estimated_text_tokens
```

`estimated_text_tokens` is read from the length index when available, otherwise it falls back to text char/byte counts or a bounded `json_size` proxy. Set `BUCKET_TEXT_COST_WEIGHT=0` to restore audio-only bucketing.

Conversion progress is displayed by the Rust converter with an `indicatif` progress bar showing shard progress, elapsed time, ETA, total samples, skipped samples, and the latest worker state.

Set `STAGING_ROOT=/path/on/nvme STAGING_MAX_GB=100` to write each completed tar part to NVMe first and then publish it to the final HDD WebDataset root. This keeps final tar files complete-only and bounded by the staging limit.

Conversion resume is input-shard based. `RESUME=1` is the script default and passes `--resume --adopt-existing`, so completed shards are skipped by marker and older pre-marker tar output is adopted only when output part count plus the final part sample count match the source shard. `OVERWRITE=1` is rejected unless `RESUME=0` is also set.

Recommended prediction/export workflow:

1. Export training checkpoint weights for Python inference:

```bash
rwkvasr-export-safetensors \
  --checkpoint-path runs/paper_bi_baseline_4x4090/best.pt \
  --output-path runs/paper_bi_baseline_4x4090/best.safetensors \
  --copy-model-config
```

2. Run Python full-model prediction directly from `.pt` or `.safetensors`:

```bash
rwkvasr-predict-ctc \
  --checkpoint-path runs/paper_bi_baseline_4x4090/best.safetensors \
  --config-yaml runs/paper_bi_baseline_4x4090/model_config.yaml \
  --webdataset-root /home/yueyulin/data/voxbox/wenetasr \
  --webdataset-split eval \
  --device cuda \
  --mode bi \
  --beam-size 8 \
  --output-path runs/paper_bi_baseline_4x4090/preds.eval.jsonl
```

3. Export Rust-consumable decode inputs from the Python model forward:

```bash
rwkvasr-export-ctc-logits \
  --checkpoint-path runs/paper_bi_baseline_4x4090/best.safetensors \
  --config-yaml runs/paper_bi_baseline_4x4090/model_config.yaml \
  --webdataset-root /home/yueyulin/data/voxbox/wenetasr \
  --webdataset-split eval \
  --device cuda \
  --mode bi \
  --output-dir runs/paper_bi_baseline_4x4090/rust_decode_inputs
```

Build and run:

```bash
cargo run --release --manifest-path tools/Cargo.toml --bin rwkvasr-tools -- \
  --webdataset-root /home/yueyulin/data/voxbox/wenetasr \
  --output-path /home/yueyulin/data/voxbox/wenetasr/webdataset_lengths.jsonl \
  --summary-path /home/yueyulin/data/voxbox/wenetasr/webdataset_lengths.summary.json \
  --split-by shard_name \
  --eval-ratio 0.01 \
  --hash-seed 0 \
  --utt-id-key sid
```

The output format matches the Python loader in `src/rwkvasr/data/webdataset_lengths.py`, so training can consume the generated `webdataset_lengths.jsonl` directly.

For very large corpora such as Emilia, the monolithic `webdataset_lengths.jsonl` is too large to load into Python memory. In that case, build a compact bucket manifest plus many small part files:

```bash
cargo run --release --manifest-path tools/Cargo.toml --bin build_bucket_index -- \
  --shard-root /media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH \
  --length-index-path /media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH/webdataset_lengths.jsonl \
  --output-dir /media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH/webdataset_buckets \
  --manifest-path /media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH/webdataset_buckets/manifest.json \
  --bucket-width 80 \
  --text-cost-source auto \
  --text-cost-weight 4 \
  --entries-per-part 100000
```

The Python training path will prefer `webdataset_bucket_manifest_path` over the old in-memory JSONL sampler. This keeps startup memory bounded while preserving same-bucket batching across ranks.

For Emilia-style `json + mp3` shards, the same tool works without unpacking the dataset. It infers `num_frames` from metadata `duration` and records tar byte offsets so training can read members directly by `offset + size`:

```bash
cargo run --release --manifest-path tools/Cargo.toml --bin rwkvasr-tools -- \
  --webdataset-root /media/usbhd/training_data/asr/emilia/Emilia/ZH/Emilia/ZH \
  --output-path /tmp/emilia_zh_lengths.jsonl \
  --summary-path /tmp/emilia_zh_lengths.summary.json \
  --split-by shard_name \
  --eval-ratio 0.01 \
  --hash-seed 0 \
  --utt-id-key id
```

CTC prefix-beam prediction:

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

Expected tensor layout inside the safetensors file:
- `logits`: `[B, T, V]`
- `lengths`: `[B]`

This first Rust/Candle predictor only handles the decode stage. Full RWKV encoder forward in Rust will require a later custom-op slice for the recurrent core.

The JSONL output includes:
- `utt_id`
- `token_ids`
- `score`
- `alignments`

Each alignment contains token-level encoder steps and projected time spans:
- `start_encoder_t`
- `end_encoder_t`
- `start_frame`
- `end_frame`
- `start_ms`
- `end_ms`

`rwkvasr-export-ctc-logits` also writes `export_index.json` alongside the per-part files so Python and Rust tooling can agree on tensor keys and timestamp projection config (`frontend_type`, `subsampling_rate`, `right_context`, `frame_shift_ms`).
