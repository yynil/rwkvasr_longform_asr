from __future__ import annotations

from pathlib import Path
from typing import Any

from rwkvasr.config import save_yaml
from rwkvasr.data.manifest import build_text_tokenizer
from rwkvasr.eval.stage211_public_metrics import (
    build_stage211_student_ctc_execution_provenance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_MODEL = (
    REPO_ROOT / "assets" / "fun-asr-nano-2512" / "multilingual.tiktoken"
).resolve()


def stage211_test_student_ctc_context(checkpoint: Path) -> tuple[dict[str, Any], Any]:
    checkpoint = checkpoint.resolve()
    model_config = checkpoint.parent / "model_config.yaml"
    tokenizer_config = checkpoint.parent / "tokenizer_config.yaml"
    save_yaml(
        model_config,
        {
            "vocab_size": 60_515,
            "blank_id": 60_515,
            "ctc_decoder_type": "funasr_nano_transformer",
            "ctc_decoder_num_layers": 5,
            "ctc_suppressed_token_ids": [],
            "decoder_enabled": False,
        },
    )
    save_yaml(
        tokenizer_config,
        {
            "tokenizer_type": "sensevoice_tiktoken",
            "tokenizer_model_path": str(TOKENIZER_MODEL),
            "tokenizer_language": None,
            "tokenizer_task": None,
            "vocab_size": 60_515,
        },
    )
    provenance = build_stage211_student_ctc_execution_provenance(
        checkpoint_path=checkpoint,
        model_config_path=model_config,
        tokenizer_config_path=tokenizer_config,
        mode="bi",
        beam_size=1,
        token_prune_topk=16,
        decoder_rescore_topk=0,
        blank_logit_bias=0.0,
        hotwords_path=None,
        text_normalization="ctc",
        save_debug_lengths=True,
    )
    tokenizer = build_text_tokenizer(
        "sensevoice_tiktoken",
        model_path=str(TOKENIZER_MODEL),
    )
    return provenance, tokenizer


def stage211_test_student_ctc_row(
    *,
    utt_id: str,
    ref_text: str,
    pred_text: str,
    provenance: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    pred_token_ids = [int(token_id) for token_id in tokenizer.encode(pred_text)]
    ref_token_ids = [int(token_id) for token_id in tokenizer.encode(ref_text)]
    return {
        "utt_id": utt_id,
        "pred_token_ids": pred_token_ids,
        "ref_token_ids": ref_token_ids,
        "pred_text": pred_text,
        "ref_text": ref_text,
        "score": 0.0,
        "decode_strategy": "ctc_greedy",
        "ctc_score": None,
        "decoder_score": None,
        "combined_score": None,
        "mode": "bi",
        "alignments": [{"token_id": token_id} for token_id in pred_token_ids],
        "debug": {
            "feature_length": max(1, len(ref_token_ids)),
            "logit_length": max(1, len(ref_token_ids)),
            "pred_token_count": len(pred_token_ids),
            "ref_token_count": len(ref_token_ids),
            "blank_top1_ratio": 0.5,
            "avg_blank_prob": 0.5,
        },
        "inference_provenance": provenance,
    }
