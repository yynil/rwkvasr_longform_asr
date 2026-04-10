# RWKV ASR Tools

Offline dataset preprocessing and deployment-oriented Rust utilities live here.

Current tools:
- `rwkvasr-tools` binary: multithreaded Rust WebDataset length indexer that scans tar metadata without decoding audio
- `build_bucket_index` binary: Rust external-memory bucket manifest builder over a large `webdataset_lengths.jsonl`
- `predict_ctc` binary: Rust + Candle CTC prefix-beam predictor over exported `safetensors logits + lengths`

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
