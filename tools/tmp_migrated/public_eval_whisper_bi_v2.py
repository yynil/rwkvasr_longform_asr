from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from rwkvasr.config import load_yaml
from rwkvasr.data.manifest import WenetFbankConfig, compute_wenet_fbank
from rwkvasr.modules import RWKVCTCModelConfig
from rwkvasr.predict import PredictionConfig, predict_ctc_labeled, write_labeled_predictions_jsonl


def edit_distance(a, b) -> int:
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            cur = dp[j]
            if x == y:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def main() -> None:
    root = Path('outputs/public_audio_eval')
    raw_manifest = [
        {
            'utt_id': 'jfk_whisper',
            'audio_filepath': root / 'jfk.flac',
            'text': 'And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.',
        },
        {
            'utt_id': 'funasr_en_short',
            'audio_filepath': root / 'asr_example_en.wav',
            'text': 'he tried to think how it could be',
        },
        {
            'utt_id': 'funasr_zh_short',
            'audio_filepath': root / 'asr_example_zh.wav',
            'text': '欢迎大家来到魔搭社区进行体验',
        },
    ]

    fbank_cfg = WenetFbankConfig()
    feature_manifest_path = root / 'public_eval_feature_manifest_whisper_bi.jsonl'
    with feature_manifest_path.open('w', encoding='utf-8') as handle:
        for item in raw_manifest:
            audio, sample_rate = sf.read(str(item['audio_filepath']), always_2d=True)
            waveform = torch.from_numpy(audio.T).to(torch.float32)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != fbank_cfg.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, fbank_cfg.sample_rate)
                sample_rate = fbank_cfg.sample_rate
            features = compute_wenet_fbank(waveform, sample_rate, fbank_cfg).float().cpu()
            feature_path = root / f"{item['utt_id']}.whisper_bi.pt"
            torch.save(features, feature_path)
            handle.write(json.dumps({'utt_id': item['utt_id'], 'feature_path': feature_path.name, 'text': item['text']}, ensure_ascii=False) + '\n')

    run_dir = Path('runs/emilia_en_zh_bi_baseline_whisper_4x4090')
    base_cfg = RWKVCTCModelConfig(**load_yaml(run_dir / 'model_config.yaml'))
    model_cfg = replace(base_cfg, backend='native')
    tok_cfg = load_yaml(run_dir / 'tokenizer_config.yaml')
    config = PredictionConfig(
        checkpoint_path=str(run_dir / 'best.pt'),
        batch_size=1,
        model_config=model_cfg,
        manifest_path=str(feature_manifest_path),
        device='cuda',
        mode='bi',
        beam_size=8,
        tokenizer_type=tok_cfg.get('tokenizer_type', 'whisper_multilingual'),
        tokenizer_model_path=tok_cfg.get('tokenizer_model_path'),
        tokenizer_language=tok_cfg.get('tokenizer_language'),
        tokenizer_task=tok_cfg.get('tokenizer_task'),
        save_debug_lengths=True,
    )
    print('predict_start', flush=True)
    predictions = predict_ctc_labeled(config)
    print('predict_done', len(predictions), flush=True)
    output_path = root / 'public_eval_predictions_whisper_bi.jsonl'
    write_labeled_predictions_jsonl(output_path, predictions)

    summary = []
    for prediction in predictions:
        ref = prediction.ref_text or ''
        hyp = prediction.pred_text or ''
        cer = edit_distance(list(ref), list(hyp)) / max(1, len(ref))
        wer = edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
        summary.append(
            {
                'utt_id': prediction.utt_id,
                'ref': ref,
                'pred': hyp,
                'score': prediction.score,
                'cer': cer,
                'wer': wer,
                'debug': None if prediction.debug is None else {
                    'feature_length': prediction.debug.feature_length,
                    'logit_length': prediction.debug.logit_length,
                    'pred_token_count': prediction.debug.pred_token_count,
                    'ref_token_count': prediction.debug.ref_token_count,
                    'blank_top1_ratio': prediction.debug.blank_top1_ratio,
                    'avg_blank_prob': prediction.debug.avg_blank_prob,
                },
                'alignments': [
                    {
                        'token_text': alignment.token_text,
                        'start_ms': alignment.start_ms,
                        'end_ms': alignment.end_ms,
                    }
                    for alignment in prediction.alignments[:20]
                ],
            }
        )

    summary_path = root / 'public_eval_summary_whisper_bi.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for item in summary:
        print('UTT', item['utt_id'])
        print('REF ', item['ref'])
        print('PRED', item['pred'])
        print('CER ', round(item['cer'], 4), 'WER', round(item['wer'], 4), 'SCORE', round(item['score'], 4))
        if item['debug'] is not None:
            print('DEBUG', item['debug'])
        print()
    print('saved_jsonl=', output_path)
    print('saved_summary=', summary_path)

if __name__ == '__main__':
    main()
