from __future__ import annotations

import argparse
import json
import time
import unicodedata
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig, WenetFbankConfig, build_inference_direction_mask, compute_wenet_fbank
from rwkvasr.predict.ctc import (
    LabeledPredictionCollator,
    PredictionConfig,
    _build_decode_debug,
    _load_prediction_model,
    build_token_alignments,
    ctc_prefix_beam_search,
)
from rwkvasr.data import build_text_tokenizer

class ManifestAudioDataset(Dataset):
    def __init__(self, manifest_path: str | Path, tokenizer):
        self.manifest_path = Path(manifest_path)
        self.entries = [json.loads(line) for line in self.manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.root = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.fbank_cfg = WenetFbankConfig()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        path = Path(row['audio_filepath'])
        if not path.is_absolute():
            path = self.root / path
        audio, sample_rate = sf.read(str(path), always_2d=True, dtype='float32')
        waveform = torch.from_numpy(audio.T).to(torch.float32)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.fbank_cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.fbank_cfg.sample_rate)
            sample_rate = self.fbank_cfg.sample_rate
        features = compute_wenet_fbank(waveform, sample_rate, self.fbank_cfg).float()
        targets = torch.tensor(self.tokenizer.encode(row['text']), dtype=torch.long)
        return {
            'utt_id': str(row['utt_id']),
            'features': features,
            'feature_length': features.size(0),
            'targets': targets,
            'target_length': targets.numel(),
            'text': row['text'],
        }

def edit_distance(a, b):
    dp = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]
            if x == y:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[-1]

def strip_punct_keep_space(text):
    out=[]
    for ch in text:
        if unicodedata.category(ch).startswith('P'):
            continue
        out.append(ch)
    return ''.join(out)

def norm_en(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = strip_punct_keep_space(text)
    text = ' '.join(text.split())
    return text

def norm_zh(text):
    text = unicodedata.normalize('NFKC', text)
    text = strip_punct_keep_space(text)
    text = ''.join(text.split())
    return text

def bucket_of(sec):
    if sec < 5:
        return '0-5s'
    if sec < 10:
        return '5-10s'
    if sec < 15:
        return '10-15s'
    if sec < 20:
        return '15-20s'
    return '20+s'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-path', required=True)
    ap.add_argument('--checkpoint-dir', required=True)
    ap.add_argument('--output-prefix', required=True)
    ap.add_argument('--language', choices=['en','zh'], required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--beam-size', type=int, default=8)
    ap.add_argument('--token-prune-topk', type=int, default=64)
    ap.add_argument('--hotwords-path', default=None)
    ap.add_argument('--hotword-weight', type=float, default=3.0)
    ap.add_argument('--hotword-prefix-scale', type=float, default=0.3)
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    base_cfg = RWKVCTCModelConfig(**load_yaml(ckpt_dir / 'model_config.yaml'))
    model_cfg = replace(base_cfg, backend='native')
    tok_cfg = load_yaml(ckpt_dir / 'tokenizer_config.yaml')
    pred_cfg = PredictionConfig(
        checkpoint_path=str(ckpt_dir / 'best.pt'),
        batch_size=args.batch_size,
        model_config=model_cfg,
        manifest_path=args.manifest_path,
        device=args.device,
        mode='bi',
        beam_size=args.beam_size,
        token_prune_topk=args.token_prune_topk,
        tokenizer_type=tok_cfg.get('tokenizer_type', 'whisper_multilingual'),
        tokenizer_model_path=tok_cfg.get('tokenizer_model_path'),
        tokenizer_language=tok_cfg.get('tokenizer_language'),
        tokenizer_task=tok_cfg.get('tokenizer_task'),
        save_debug_lengths=True,
        num_workers=args.num_workers,
        hotwords_path=args.hotwords_path,
        hotword_weight=args.hotword_weight,
        hotword_prefix_scale=args.hotword_prefix_scale,
    )
    tokenizer = build_text_tokenizer(pred_cfg.tokenizer_type, model_path=pred_cfg.tokenizer_model_path, language=pred_cfg.tokenizer_language, task=pred_cfg.tokenizer_task)
    dataset = ManifestAudioDataset(args.manifest_path, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=LabeledPredictionCollator())
    device = torch.device(args.device)
    model, feature_dtype = _load_prediction_model(pred_cfg, device=device)

    meta = {}
    with Path(args.manifest_path).open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            meta[row['utt_id']] = {'duration_sec': float(row.get('duration_sec', 0.0)), 'text': row.get('text', '')}

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pred_path = prefix.with_suffix('.predictions.jsonl')
    report_path = prefix.with_suffix('.report.json')

    totals = defaultdict(float)
    by_bucket = defaultdict(lambda: defaultdict(float))
    start = time.time()
    num_done = 0
    total_audio_sec = 0.0

    with pred_path.open('w', encoding='utf-8') as out, torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device, feature_dtype=feature_dtype)
            mask = build_inference_direction_mask(model.config.num_layers, mode='bi', device=batch.features.device)
            logits, logit_lengths, _ = model(batch.features, batch.feature_lengths, direction_mask=mask)
            log_probs = logits.detach().float().log_softmax(dim=-1).cpu()
            lengths = logit_lengths.detach().to(dtype=torch.long, device='cpu') if logit_lengths is not None else torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            target_offset = 0
            for i, utt_id in enumerate(batch.utt_ids):
                length = int(lengths[i].item())
                hyps = ctc_prefix_beam_search(log_probs[i, :length], blank_id=model.config.blank_id, beam_size=args.beam_size, token_prune_topk=args.token_prune_topk)
                best = hyps[0]
                pred_ids = [int(x) for x in best.token_ids]
                tgt_len = int(batch.target_lengths[i].item())
                target_offset += tgt_len
                pred_text = tokenizer.decode(pred_ids)
                ref_text = meta[utt_id]['text']
                debug = _build_decode_debug(log_probs[i, :length], blank_id=model.config.blank_id, feature_length=int(batch.feature_lengths[i].item()), logit_length=length, pred_token_count=len(pred_ids), ref_token_count=tgt_len)
                alignments = build_token_alignments(log_probs[i, :length], pred_ids, blank_id=model.config.blank_id, frontend_type=model.config.frontend_type, frame_shift_ms=pred_cfg.frame_shift_ms, decode_fn=tokenizer.decode)
                row = {
                    'utt_id': utt_id,
                    'ref_text': ref_text,
                    'pred_text': pred_text,
                    'score': float(best.score),
                    'duration_sec': meta[utt_id]['duration_sec'],
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
                out.write(json.dumps(row, ensure_ascii=False) + '\n')

                dur = meta[utt_id]['duration_sec']
                total_audio_sec += dur
                b = bucket_of(dur)
                if args.language == 'en':
                    raw_ref = ref_text
                    raw_hyp = pred_text
                    norm_ref = norm_en(ref_text)
                    norm_hyp = norm_en(pred_text)
                    raw_cer_num = edit_distance(list(raw_ref), list(raw_hyp))
                    raw_cer_den = max(1, len(raw_ref))
                    raw_wer_num = edit_distance(raw_ref.split(), raw_hyp.split())
                    raw_wer_den = max(1, len(raw_ref.split()))
                    norm_cer_num = edit_distance(list(norm_ref), list(norm_hyp))
                    norm_cer_den = max(1, len(norm_ref))
                    norm_wer_num = edit_distance(norm_ref.split(), norm_hyp.split())
                    norm_wer_den = max(1, len(norm_ref.split()))
                    for scope in (totals, by_bucket[b]):
                        scope['num_utts'] += 1
                        scope['audio_sec'] += dur
                        scope['raw_cer_num'] += raw_cer_num
                        scope['raw_cer_den'] += raw_cer_den
                        scope['raw_wer_num'] += raw_wer_num
                        scope['raw_wer_den'] += raw_wer_den
                        scope['norm_cer_num'] += norm_cer_num
                        scope['norm_cer_den'] += norm_cer_den
                        scope['norm_wer_num'] += norm_wer_num
                        scope['norm_wer_den'] += norm_wer_den
                else:
                    raw_ref = ref_text
                    raw_hyp = pred_text
                    norm_ref = norm_zh(ref_text)
                    norm_hyp = norm_zh(pred_text)
                    raw_cer_num = edit_distance(list(raw_ref), list(raw_hyp))
                    raw_cer_den = max(1, len(raw_ref))
                    norm_cer_num = edit_distance(list(norm_ref), list(norm_hyp))
                    norm_cer_den = max(1, len(norm_ref))
                    for scope in (totals, by_bucket[b]):
                        scope['num_utts'] += 1
                        scope['audio_sec'] += dur
                        scope['raw_cer_num'] += raw_cer_num
                        scope['raw_cer_den'] += raw_cer_den
                        scope['norm_cer_num'] += norm_cer_num
                        scope['norm_cer_den'] += norm_cer_den
                num_done += 1
            if batch_idx % 20 == 0:
                elapsed = time.time() - start
                print(f'batch={batch_idx} utts={num_done} elapsed={elapsed:.1f}s audio_h={total_audio_sec/3600:.2f}', flush=True)

    elapsed = time.time() - start
    report = {
        'language': args.language,
        'checkpoint_dir': str(ckpt_dir),
        'manifest_path': args.manifest_path,
        'predictions_path': str(pred_path),
        'elapsed_sec': elapsed,
        'audio_hours': total_audio_sec/3600.0,
        'rtf': elapsed / max(1e-6, total_audio_sec),
        'settings': {
            'device': args.device,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'beam_size': args.beam_size,
            'token_prune_topk': args.token_prune_topk,
            'mode': 'bi',
        },
        'overall': {},
        'by_duration_bucket': {},
    }
    if args.language == 'en':
        report['overall'] = {
            'num_utts': int(totals['num_utts']),
            'raw_cer': totals['raw_cer_num']/max(1.0, totals['raw_cer_den']),
            'raw_wer': totals['raw_wer_num']/max(1.0, totals['raw_wer_den']),
            'norm_cer': totals['norm_cer_num']/max(1.0, totals['norm_cer_den']),
            'norm_wer': totals['norm_wer_num']/max(1.0, totals['norm_wer_den']),
        }
        for b, s in by_bucket.items():
            report['by_duration_bucket'][b] = {
                'num_utts': int(s['num_utts']),
                'raw_cer': s['raw_cer_num']/max(1.0, s['raw_cer_den']),
                'raw_wer': s['raw_wer_num']/max(1.0, s['raw_wer_den']),
                'norm_cer': s['norm_cer_num']/max(1.0, s['norm_cer_den']),
                'norm_wer': s['norm_wer_num']/max(1.0, s['norm_wer_den']),
            }
    else:
        report['overall'] = {
            'num_utts': int(totals['num_utts']),
            'raw_cer': totals['raw_cer_num']/max(1.0, totals['raw_cer_den']),
            'norm_cer': totals['norm_cer_num']/max(1.0, totals['norm_cer_den']),
        }
        for b, s in by_bucket.items():
            report['by_duration_bucket'][b] = {
                'num_utts': int(s['num_utts']),
                'raw_cer': s['raw_cer_num']/max(1.0, s['raw_cer_den']),
                'norm_cer': s['norm_cer_num']/max(1.0, s['norm_cer_den']),
            }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('saved_report', report_path)

if __name__ == '__main__':
    main()
