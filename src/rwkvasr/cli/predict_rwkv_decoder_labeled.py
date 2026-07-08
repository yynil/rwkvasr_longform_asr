from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig
from rwkvasr.predict import PredictionConfig
from rwkvasr.predict.rwkv_decoder import (
    predict_rwkv_decoder_labeled,
    write_rwkv_decoder_labeled_predictions_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict labeled eval samples with direct RWKV decoder AR generation."
    )
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--webdataset-root", default=None)
    parser.add_argument("--webdataset-length-index-path", default=None)
    parser.add_argument("--webdataset-split", default="all")
    parser.add_argument("--webdataset-shard-pattern", default="*.tar")
    parser.add_argument("--webdataset-eval-ratio", default=0.0, type=float)
    parser.add_argument("--webdataset-hash-seed", default=0, type=int)
    parser.add_argument("--webdataset-split-by", default="shard_name")
    parser.add_argument("--webdataset-utt-id-key", default="sid")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-yaml", default=None)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="bi", choices=["bi", "l2r", "r2l", "alt"])
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--preview-path", default=None)
    parser.add_argument("--preview-count", default=20, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--progress-interval", default=0, type=int)
    parser.add_argument("--tokenizer-type", default=None)
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--tokenizer-language", default=None)
    parser.add_argument("--tokenizer-task", default=None)
    parser.add_argument("--text-normalization", default="none")
    parser.add_argument("--decoder-ctc-draft-cache-path", default=None)
    parser.add_argument("--decoder-ctc-draft-prompt-template", default=None)
    parser.add_argument("--decoder-ctc-draft-text-key", default=None)
    parser.add_argument("--decoder-ctc-draft-missing-policy", default=None, choices=("empty", "error"))
    parser.add_argument("--decoder-ctc-draft-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-language-mismatch-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-dropout-seed", default=None, type=int)
    parser.add_argument("--frame-shift-ms", default=10.0, type=float)
    parser.add_argument("--feature-extractor-type", default="wenet_fbank")
    parser.add_argument("--max-new-tokens", default=None, type=int)
    parser.add_argument("--max-new-tokens-factor", default=2.0, type=float)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top-k", default=0, type=int)
    parser.add_argument("--top-p", default=1.0, type=float)
    parser.add_argument("--ctc-draft-fallback-max-cer", default=None, type=float)
    parser.add_argument("--ctc-draft-fallback-min-length-ratio", default=0.75, type=float)
    parser.add_argument("--ctc-draft-fallback-max-length-ratio", default=1.25, type=float)
    parser.add_argument(
        "--ctc-draft-fallback-reject-repetition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ctc-draft-fallback-metric-normalization", default="ctc")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--vocab-size", default=None, type=int)
    parser.add_argument("--input-dim", default=80, type=int)
    parser.add_argument("--n-embd", default=512, type=int)
    parser.add_argument("--dim-att", default=512, type=int)
    parser.add_argument("--dim-ff", default=2048, type=int)
    parser.add_argument("--num-layers", default=12, type=int)
    parser.add_argument("--head-size", default=64, type=int)
    parser.add_argument("--conv-kernel-size", default=31, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--blank-id", default=0, type=int)
    parser.add_argument("--frontend-type", default="conv2d6")
    parser.add_argument("--cmvn-file", default=None)
    parser.add_argument("--cmvn-is-json", action="store_true", default=True)
    return parser


def _resolve_model_config(args: argparse.Namespace) -> RWKVCTCModelConfig:
    config_yaml = args.config_yaml
    if config_yaml is None:
        default_yaml = Path(args.checkpoint_path).resolve().parent / "model_config.yaml"
        if default_yaml.exists():
            config_yaml = str(default_yaml)

    if config_yaml is not None:
        return RWKVCTCModelConfig(**load_yaml(config_yaml))

    if args.vocab_size is None:
        raise ValueError("Either --config-yaml or --vocab-size plus model shape arguments must be provided.")
    return RWKVCTCModelConfig(
        feature_extractor_type=args.feature_extractor_type,
        input_dim=args.input_dim,
        n_embd=args.n_embd,
        dim_att=args.dim_att,
        dim_ff=args.dim_ff,
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        head_size=args.head_size,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        blank_id=args.blank_id,
        frontend_type=args.frontend_type,
        cmvn_file=args.cmvn_file,
        cmvn_is_json=args.cmvn_is_json,
    )


def _resolve_tokenizer_config(args: argparse.Namespace) -> dict[str, object]:
    resolved = {
        "tokenizer_type": "whisper_multilingual",
        "tokenizer_model_path": None,
        "tokenizer_language": None,
        "tokenizer_task": None,
        "decoder_prompt_before_audio": "",
        "decoder_prompt_before_audio_use_language": False,
        "decoder_ctc_draft_cache_path": None,
        "decoder_ctc_draft_prompt_template": "",
        "decoder_ctc_draft_text_key": "pred_text",
        "decoder_ctc_draft_missing_policy": "empty",
        "decoder_ctc_draft_dropout_prob": 0.0,
        "decoder_ctc_draft_language_mismatch_dropout_prob": 0.0,
        "decoder_ctc_draft_dropout_seed": 0,
        "decoder_target_prefix": "",
        "decoder_target_prefix_use_language": False,
        "decoder_language_confirmation_en": "This is English text.",
        "decoder_language_confirmation_zh": "这是中文文字。",
    }
    default_yaml = Path(args.checkpoint_path).resolve().parent / "tokenizer_config.yaml"
    if default_yaml.exists():
        config_data = load_yaml(default_yaml)
        resolved.update(
            {
                "tokenizer_type": config_data.get(
                    "decoder_tokenizer_type",
                    config_data.get("tokenizer_type", resolved["tokenizer_type"]),
                ),
                "tokenizer_model_path": config_data.get(
                    "decoder_tokenizer_model_path",
                    config_data.get("tokenizer_model_path"),
                ),
                "tokenizer_language": config_data.get(
                    "decoder_tokenizer_language",
                    config_data.get("tokenizer_language"),
                ),
                "tokenizer_task": config_data.get(
                    "decoder_tokenizer_task",
                    config_data.get("tokenizer_task"),
                ),
                "decoder_prompt_before_audio": config_data.get("decoder_prompt_before_audio", ""),
                "decoder_prompt_before_audio_use_language": bool(
                    config_data.get("decoder_prompt_before_audio_use_language", False)
                ),
                "decoder_ctc_draft_cache_path": config_data.get("decoder_ctc_draft_cache_path"),
                "decoder_ctc_draft_prompt_template": config_data.get("decoder_ctc_draft_prompt_template", ""),
                "decoder_ctc_draft_text_key": config_data.get("decoder_ctc_draft_text_key", "pred_text"),
                "decoder_ctc_draft_missing_policy": config_data.get(
                    "decoder_ctc_draft_missing_policy",
                    "empty",
                ),
                # Random draft dropout is a training regularizer; prediction keeps it disabled
                # unless the caller explicitly requests an ablation override.
                "decoder_ctc_draft_dropout_prob": 0.0,
                "decoder_ctc_draft_language_mismatch_dropout_prob": config_data.get(
                    "decoder_ctc_draft_language_mismatch_dropout_prob",
                    0.0,
                ),
                "decoder_ctc_draft_dropout_seed": config_data.get(
                    "decoder_ctc_draft_dropout_seed",
                    0,
                ),
                "decoder_target_prefix": config_data.get("decoder_target_prefix", ""),
                "decoder_target_prefix_use_language": bool(
                    config_data.get("decoder_target_prefix_use_language", False)
                ),
                "decoder_language_confirmation_en": config_data.get(
                    "decoder_language_confirmation_en",
                    "This is English text.",
                ),
                "decoder_language_confirmation_zh": config_data.get(
                    "decoder_language_confirmation_zh",
                    "这是中文文字。",
                ),
            }
        )
    for key in ("tokenizer_type", "tokenizer_model_path", "tokenizer_language", "tokenizer_task"):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value
    for key in (
        "decoder_ctc_draft_cache_path",
        "decoder_ctc_draft_prompt_template",
        "decoder_ctc_draft_text_key",
        "decoder_ctc_draft_missing_policy",
        "decoder_ctc_draft_dropout_prob",
        "decoder_ctc_draft_language_mismatch_dropout_prob",
        "decoder_ctc_draft_dropout_seed",
    ):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value
    return resolved


def _write_preview(
    path: str | Path,
    predictions,
    debug_rows,
    *,
    preview_count: int,
) -> Path:
    preview_path = Path(path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, (prediction, debug) in enumerate(zip(predictions[:preview_count], debug_rows[:preview_count], strict=False), start=1):
        lines.append(f"[{index}] utt_id={prediction.utt_id}")
        lines.append(f"  REF : {prediction.ref_text or ''}")
        lines.append(f"  PRED: {prediction.pred_text or ''}")
        lines.append(f"  SCORE: {prediction.score:.4f}")
        lines.append("  STRATEGY: rwkv_decoder_ar")
        lines.append(
            "  DEBUG: "
            f"feat={debug.feature_length} "
            f"enc={debug.encoded_length} "
            f"pred_tok={debug.pred_token_count} "
            f"ref_tok={debug.ref_token_count if debug.ref_token_count is not None else -1} "
            f"eos={1 if debug.eos_emitted else 0} "
            f"avg_logprob={debug.avg_logprob:.4f}"
        )
        if debug.ctc_draft_fallback:
            lines.append(
                "  CTC_DRAFT_FALLBACK: "
                f"reason={debug.ctc_draft_fallback_reason or 'unknown'} "
                f"cer={debug.ctc_draft_cer_distance if debug.ctc_draft_cer_distance is not None else -1:.4f} "
                f"len_ratio={debug.ctc_draft_length_ratio if debug.ctc_draft_length_ratio is not None else -1:.4f}"
            )
            if debug.raw_pred_text:
                lines.append(f"  RAW_AR: {debug.raw_pred_text}")
    preview_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return preview_path


def main() -> None:
    args = build_parser().parse_args()
    if (args.manifest_path is None) == (args.webdataset_root is None):
        raise ValueError("Exactly one of --manifest-path or --webdataset-root must be provided.")
    if args.seed is not None:
        random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
    tokenizer_config = _resolve_tokenizer_config(args)
    predictions, debug_rows = predict_rwkv_decoder_labeled(
        PredictionConfig(
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            model_config=_resolve_model_config(args),
            manifest_path=args.manifest_path,
            webdataset_root=args.webdataset_root,
            webdataset_length_index_path=args.webdataset_length_index_path,
            webdataset_split=args.webdataset_split,
            webdataset_shard_pattern=args.webdataset_shard_pattern,
            webdataset_eval_ratio=args.webdataset_eval_ratio,
            webdataset_hash_seed=args.webdataset_hash_seed,
            webdataset_split_by=args.webdataset_split_by,
            webdataset_utt_id_key=args.webdataset_utt_id_key,
            device=args.device,
            mode=args.mode,
            tokenizer_type=str(tokenizer_config["tokenizer_type"]),
            tokenizer_model_path=tokenizer_config["tokenizer_model_path"],
            tokenizer_language=tokenizer_config["tokenizer_language"],
            tokenizer_task=tokenizer_config["tokenizer_task"],
            text_normalization=args.text_normalization,
            decoder_prompt_before_audio=str(tokenizer_config["decoder_prompt_before_audio"] or ""),
            decoder_prompt_before_audio_use_language=bool(
                tokenizer_config["decoder_prompt_before_audio_use_language"]
            ),
            decoder_ctc_draft_cache_path=(
                str(tokenizer_config["decoder_ctc_draft_cache_path"])
                if tokenizer_config["decoder_ctc_draft_cache_path"] is not None
                else None
            ),
            decoder_ctc_draft_prompt_template=str(
                tokenizer_config["decoder_ctc_draft_prompt_template"] or ""
            ),
            decoder_ctc_draft_text_key=str(tokenizer_config["decoder_ctc_draft_text_key"] or "pred_text"),
            decoder_ctc_draft_missing_policy=str(
                tokenizer_config["decoder_ctc_draft_missing_policy"] or "empty"
            ),
            decoder_ctc_draft_dropout_prob=float(
                tokenizer_config["decoder_ctc_draft_dropout_prob"] or 0.0
            ),
            decoder_ctc_draft_language_mismatch_dropout_prob=float(
                tokenizer_config["decoder_ctc_draft_language_mismatch_dropout_prob"] or 0.0
            ),
            decoder_ctc_draft_dropout_seed=int(tokenizer_config["decoder_ctc_draft_dropout_seed"] or 0),
            decoder_target_prefix=str(tokenizer_config["decoder_target_prefix"] or ""),
            decoder_target_prefix_use_language=bool(tokenizer_config["decoder_target_prefix_use_language"]),
            decoder_language_confirmation_en=str(tokenizer_config["decoder_language_confirmation_en"]),
            decoder_language_confirmation_zh=str(tokenizer_config["decoder_language_confirmation_zh"]),
            num_workers=args.num_workers,
            frame_shift_ms=args.frame_shift_ms,
            progress_interval=args.progress_interval,
        ),
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_factor=args.max_new_tokens_factor,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        ctc_draft_fallback_max_cer=args.ctc_draft_fallback_max_cer,
        ctc_draft_fallback_min_length_ratio=args.ctc_draft_fallback_min_length_ratio,
        ctc_draft_fallback_max_length_ratio=args.ctc_draft_fallback_max_length_ratio,
        ctc_draft_fallback_reject_repetition=args.ctc_draft_fallback_reject_repetition,
        ctc_draft_fallback_metric_normalization=args.ctc_draft_fallback_metric_normalization,
    )
    output_path = write_rwkv_decoder_labeled_predictions_jsonl(args.output_path, predictions, debug_rows)
    preview_path = None
    if args.preview_path is not None:
        preview_path = _write_preview(
            args.preview_path,
            predictions,
            debug_rows,
            preview_count=args.preview_count,
        )
    print(f"saved_jsonl={output_path}")
    if preview_path is not None:
        print(f"saved_preview={preview_path}")
    print(f"num_predictions={len(predictions)}")


if __name__ == "__main__":
    main()
