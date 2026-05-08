from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import time
from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig
from rwkvasr.predict.ctc import PredictionConfig, predict_ctc_labeled

run_dir = Path('runs/emilia_en_zh_bi_baseline_whisper_4x4090')
base_cfg = RWKVCTCModelConfig(**load_yaml(run_dir / 'model_config.yaml'))
model_cfg = replace(base_cfg, backend='native')
config = PredictionConfig(
    checkpoint_path=str(run_dir / 'best.pt'),
    batch_size=1,
    model_config=model_cfg,
    manifest_path='outputs/public_audio_eval/public_eval_feature_manifest_whisper_bi.jsonl',
    device='cuda',
    mode='bi',
    beam_size=8,
    token_prune_topk=64,
    tokenizer_type='whisper_multilingual',
    save_debug_lengths=True,
)
start = time.time()
preds = predict_ctc_labeled(config, limit=1)
print('done', len(preds), time.time() - start, preds[0].utt_id, preds[0].pred_text[:120])
