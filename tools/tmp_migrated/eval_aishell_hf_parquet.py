from __future__ import annotations
import argparse
from dataclasses import replace
import io
import json
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.insert(0, '/tmp/benchdeps')
import pyarrow.parquet as pq

from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig, WenetFbankConfig, build_inference_direction_mask, compute_wenet_fbank
from rwkvasr.predict.ctc import (
    PredictionConfig,
    _build_decode_debug,
    _load_prediction_model,
    build_token_alignments,
    ctc_prefix_beam_search,
)
from rwkvasr.data import build_text_tokenizer


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]
            if x == y:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def strip_punct_keep_space(text: str) -> str:
    out = []
    for ch in text:
        if unicodedata.category(ch).startswith('P'):
            continue
        out.append(ch)
    return ''.join(out)


def norm_zh(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = strip_punct_keep_space(text)
    text = ''.join(text.split())
    return text


def bucket_of(sec: float) -> str:
    if sec < 5:
        return '0-5s'
    if sec < 10:
        return '5-10s'
    if sec < 15:
        return '10-15s'
    if sec < 20:
        return '15-20s'
    return '20+s'


def compute_quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    vals = sorted(values)
    def pick(q: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
        return float(vals[idx])
    return {
        'mean': float(sum(vals) / len(vals)),
        'p50': pick(0.50),
        'p90': pick(0.90),
        'p95': pick(0.95),
        'p99': pick(0.99),
        'max': float(vals[-1]),
    }


def load_rows(parquet_root: Path):
    paths = sorted(parquet_root.glob('test-*.parquet'))
    if not paths:
        raise FileNotFoundError(f'no parquet shards under {parquet_root}')
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=64, columns=['name', 'text', 'audio']):
            for row in batch.to_pylist():
                yield row


def decode_audio_bytes(audio_bytes: bytes, fbank_cfg: WenetFbankConfig):
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=True, dtype='float32')
    waveform = torch.from_numpy(audio.T).to(torch.float32)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != fbank_cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, fbank_cfg.sample_rate)
        sample_rate = fbank_cfg.sample_rate
    duration_sec = waveform.size(1) / float(sample_rate)
    features = compute_wenet_fbank(waveform, sample_rate, fbank_cfg).float()
    return waveform, features, duration_sec


def flush_batch(samples, *, model, tokenizer, device, feature_dtype, pred_cfg, out_f, totals, by_bucket, progress, hotwords, hotword_prefix_scale):
    if not samples:
        return
    feats = [s['features'] for s in samples]
    feat_lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
    padded = padded.to(device=device, dtype=feature_dtype)
    feat_lens = feat_lens.to(device=device)
    mask = build_inference_direction_mask(model.config.num_layers, mode='bi', device=device)
    with torch.no_grad():
        logits, logit_lengths, _ = model(padded, feat_lens, direction_mask=mask)
    log_probs = logits.detach().float().log_softmax(dim=-1).cpu()
    lengths = logit_lengths.detach().to(dtype=torch.long, device='cpu') if logit_lengths is not None else torch.full((logits.size(0),), logits.size(1), dtype=torch.long)

    for i, sample in enumerate(samples):
        length = int(lengths[i].item())
        hyps = ctc_prefix_beam_search(
            log_probs[i, :length],
            blank_id=model.config.blank_id,
            beam_size=pred_cfg.beam_size,
            token_prune_topk=pred_cfg.token_prune_topk,
            hotwords=hotwords,
            hotword_prefix_scale=hotword_prefix_scale,
        )
        best = hyps[0]
        pred_ids = [int(x) for x in best.token_ids]
        pred_text = tokenizer.decode(pred_ids)
        ref_text = sample['text']
        norm_ref = norm_zh(ref_text)
        norm_hyp = norm_zh(pred_text)
        raw_cer_num = edit_distance(list(ref_text), list(pred_text))
        raw_cer_den = max(1, len(ref_text))
        norm_cer_num = edit_distance(list(norm_ref), list(norm_hyp))
        norm_cer_den = max(1, len(norm_ref))
        dur = sample['duration_sec']
        bucket = bucket_of(dur)
        debug = _build_decode_debug(
            log_probs[i, :length],
            blank_id=model.config.blank_id,
            feature_length=sample['features'].size(0),
            logit_length=length,
            pred_token_count=len(pred_ids),
            ref_token_count=len(tokenizer.encode(ref_text)),
        )
        alignments = build_token_alignments(
            log_probs[i, :length],
            pred_ids,
            blank_id=model.config.blank_id,
            frontend_type=model.config.frontend_type,
            frame_shift_ms=pred_cfg.frame_shift_ms,
            decode_fn=tokenizer.decode,
        )
        row = {
            'utt_id': sample['utt_id'],
            'ref_text': ref_text,
            'pred_text': pred_text,
            'normalized_ref_text': norm_ref,
            'normalized_pred_text': norm_hyp,
            'score': float(best.score),
            'duration_sec': dur,
            'raw_cer': raw_cer_num / raw_cer_den,
            'norm_cer': norm_cer_num / norm_cer_den,
            'debug': {
                'feature_length': debug.feature_length,
                'logit_length': debug.logit_length,
                'pred_token_count': debug.pred_token_count,
                'ref_token_count': debug.ref_token_count,
                'blank_top1_ratio': debug.blank_top1_ratio,
                'avg_blank_prob': debug.avg_blank_prob,
            },
            'alignments': [
                {'token_text': a.token_text, 'start_ms': a.start_ms, 'end_ms': a.end_ms}
                for a in alignments[:20]
            ],
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + '\n')
        for scope in (totals, by_bucket[bucket]):
            scope['num_utts'] += 1
            scope['audio_sec'] += dur
            scope['raw_cer_num'] += raw_cer_num
            scope['raw_cer_den'] += raw_cer_den
            scope['norm_cer_num'] += norm_cer_num
            scope['norm_cer_den'] += norm_cer_den
        progress['utts'] += 1
        progress['audio_sec'] += dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet-root', required=True)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--output-prefix', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--beam-size', type=int, default=8)
    ap.add_argument('--token-prune-topk', type=int, default=64)
    ap.add_argument('--hotwords-path', default=None)
    ap.add_argument('--hotword-weight', type=float, default=3.0)
    ap.add_argument('--hotword-prefix-scale', type=float, default=0.3)
    args = ap.parse_args()

    parquet_root = Path(args.parquet_root)
    ckpt_dir = Path(args.checkpoint_dir)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pred_path = prefix.with_suffix('.predictions.jsonl')
    report_path = prefix.with_suffix('.report.json')

    base_cfg = RWKVCTCModelConfig(**load_yaml(ckpt_dir / 'model_config.yaml'))
    model_cfg = replace(base_cfg, backend='native')
    tok_cfg = load_yaml(ckpt_dir / 'tokenizer_config.yaml')
    pred_cfg = PredictionConfig(
        checkpoint_path=str(ckpt_dir / 'best.pt'),
        batch_size=args.batch_size,
        model_config=model_cfg,
        device=args.device,
        mode='bi',
        beam_size=args.beam_size,
        token_prune_topk=args.token_prune_topk,
        tokenizer_type=tok_cfg.get('tokenizer_type', 'whisper_multilingual'),
        tokenizer_model_path=tok_cfg.get('tokenizer_model_path'),
        tokenizer_language=tok_cfg.get('tokenizer_language'),
        tokenizer_task=tok_cfg.get('tokenizer_task'),
        save_debug_lengths=True,
        num_workers=0,
        hotwords_path=args.hotwords_path,
        hotword_weight=args.hotword_weight,
        hotword_prefix_scale=args.hotword_prefix_scale,
    )
    tokenizer = build_text_tokenizer(
        pred_cfg.tokenizer_type,
        model_path=pred_cfg.tokenizer_model_path,
        language=pred_cfg.tokenizer_language,
        task=pred_cfg.tokenizer_task,
    )
    from rwkvasr.predict.ctc import load_hotwords
    hotwords = tuple(load_hotwords(args.hotwords_path, tokenizer=tokenizer, default_weight=args.hotword_weight)) if args.hotwords_path else ()
    device = torch.device(args.device)
    model, feature_dtype = _load_prediction_model(pred_cfg, device=device)
    fbank_cfg = WenetFbankConfig()

    totals = defaultdict(float)
    by_bucket = defaultdict(lambda: defaultdict(float))
    durations = []
    char_lengths = []
    progress = {'utts': 0, 'audio_sec': 0.0}
    start = time.time()
    pending = []

    with pred_path.open('w', encoding='utf-8') as out_f:
        for row in load_rows(parquet_root):
            _, features, duration_sec = decode_audio_bytes(row['audio']['bytes'], fbank_cfg)
            text = str(row['text'])
            pending.append({
                'utt_id': str(row['name']),
                'text': text,
                'features': features,
                'duration_sec': duration_sec,
            })
            durations.append(duration_sec)
            char_lengths.append(len(norm_zh(text)))
            if len(pending) >= args.batch_size:
                flush_batch(pending, model=model, tokenizer=tokenizer, device=device, feature_dtype=feature_dtype, pred_cfg=pred_cfg, out_f=out_f, totals=totals, by_bucket=by_bucket, progress=progress, hotwords=hotwords, hotword_prefix_scale=args.hotword_prefix_scale)
                pending.clear()
                if progress['utts'] % 200 == 0:
                    elapsed = time.time() - start
                    print(f"utts={progress['utts']} elapsed={elapsed:.1f}s audio_h={progress['audio_sec']/3600:.2f}", flush=True)
        if pending:
            flush_batch(pending, model=model, tokenizer=tokenizer, device=device, feature_dtype=feature_dtype, pred_cfg=pred_cfg, out_f=out_f, totals=totals, by_bucket=by_bucket, progress=progress, hotwords=hotwords, hotword_prefix_scale=args.hotword_prefix_scale)

    elapsed = time.time() - start
    report = {
        'language': 'zh',
        'dataset': 'AISHELL-1 test (HF parquet)',
        'checkpoint_dir': str(ckpt_dir),
        'parquet_root': str(parquet_root),
        'predictions_path': str(pred_path),
        'elapsed_sec': elapsed,
        'audio_hours': progress['audio_sec'] / 3600.0,
        'rtf': elapsed / max(1e-6, progress['audio_sec']),
        'settings': {
            'device': args.device,
            'batch_size': args.batch_size,
            'beam_size': args.beam_size,
            'token_prune_topk': args.token_prune_topk,
            'mode': 'bi',
        },
        'dataset_stats': {
            'num_utts': int(len(durations)),
            'duration_sec': compute_quantiles(durations),
            'normalized_char_length': compute_quantiles([float(x) for x in char_lengths]),
        },
        'overall': {
            'num_utts': int(totals['num_utts']),
            'raw_cer': totals['raw_cer_num'] / max(1.0, totals['raw_cer_den']),
            'norm_cer': totals['norm_cer_num'] / max(1.0, totals['norm_cer_den']),
        },
        'by_duration_bucket': {},
    }
    for bucket, scope in by_bucket.items():
        report['by_duration_bucket'][bucket] = {
            'num_utts': int(scope['num_utts']),
            'raw_cer': scope['raw_cer_num'] / max(1.0, scope['raw_cer_den']),
            'norm_cer': scope['norm_cer_num'] / max(1.0, scope['norm_cer_den']),
        }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
