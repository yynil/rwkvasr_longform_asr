from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import torch
from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig
from rwkvasr.predict.ctc import PredictionConfig, _load_prediction_model, _build_labeled_prediction_loader, build_text_tokenizer, build_inference_direction_mask, ctc_prefix_beam_search

run_dir = Path('runs/emilia_en_zh_bi_baseline_whisper_4x4090')
feature_manifest_path = Path('outputs/public_audio_eval/public_eval_feature_manifest_whisper_bi.jsonl')
print('start', flush=True)
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
print('config_ready', flush=True)
device = torch.device(config.device)
model, feature_dtype = _load_prediction_model(config, device=device)
print('model_loaded', feature_dtype, flush=True)
tokenizer = build_text_tokenizer(
    config.tokenizer_type,
    model_path=config.tokenizer_model_path,
    language=config.tokenizer_language,
    task=config.tokenizer_task,
)
print('tokenizer_ready', type(tokenizer).__name__, flush=True)
loader = _build_labeled_prediction_loader(config)
print('loader_ready', flush=True)
for i, batch in enumerate(loader):
    print('batch_loaded', i, batch.utt_ids, batch.features.shape, flush=True)
    batch = batch.to(device, feature_dtype=feature_dtype)
    print('batch_to_device', i, flush=True)
    mask = build_inference_direction_mask(model.config.num_layers, mode=config.mode, device=batch.features.device)
    print('mask_ready', i, mask.shape, flush=True)
    logits, logit_lengths, _ = model(batch.features, batch.feature_lengths, direction_mask=mask)
    print('forward_done', i, logits.shape, logit_lengths, flush=True)
    log_probs = logits.detach().float().log_softmax(dim=-1).cpu()
    length = int(logit_lengths[0].item()) if logit_lengths is not None else logits.size(1)
    hypotheses = ctc_prefix_beam_search(
        log_probs[0, :length],
        blank_id=model.config.blank_id,
        beam_size=config.beam_size,
        token_prune_topk=config.token_prune_topk,
        length_bonus=config.length_bonus,
        insertion_bonus=config.insertion_bonus,
    )
    print('decode_done', i, len(hypotheses), len(hypotheses[0].token_ids) if hypotheses else 0, flush=True)
    break
print('done', flush=True)
