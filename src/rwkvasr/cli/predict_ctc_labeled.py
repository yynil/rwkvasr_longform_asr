from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from rwkvasr.config import load_yaml
from rwkvasr.modules import RWKVCTCModelConfig
from rwkvasr.predict import PredictionConfig, labeled_prediction_to_json_dict, predict_ctc_labeled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict labeled eval samples with a RWKV CTC checkpoint and save pred/ref pairs."
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
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="bi", choices=["bi", "l2r", "r2l", "alt"])
    parser.add_argument("--beam-size", default=8, type=int)
    parser.add_argument("--token-prune-topk", default=None, type=int)
    parser.add_argument("--decoder-rescore-topk", default=0, type=int)
    parser.add_argument("--decoder-rescore-weight", default=0.5, type=float)
    parser.add_argument("--decoder-rescore-length-normalize", action="store_true", default=True)
    parser.add_argument("--length-bonus", default=0.0, type=float)
    parser.add_argument("--insertion-bonus", default=0.0, type=float)
    parser.add_argument(
        "--blank-logit-bias",
        default=0.0,
        type=float,
        help="Add this value to the CTC blank logit before decode log_softmax; negative values penalize blank.",
    )
    parser.add_argument("--hotwords-path", default=None)
    parser.add_argument("--hotword-weight", default=3.0, type=float)
    parser.add_argument("--hotword-prefix-scale", default=0.3, type=float)
    parser.add_argument("--save-debug-lengths", action="store_true")
    parser.add_argument(
        "--skip-decode-errors",
        action="store_true",
        help="Skip corrupt WebDataset samples during large draft-cache generation.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append predictions to an existing JSONL output instead of replacing it.",
    )
    parser.add_argument("--preview-path", default=None)
    parser.add_argument("--preview-count", default=20, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--progress-interval", default=0, type=int)
    parser.add_argument("--tokenizer-type", default=None)
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--tokenizer-language", default=None)
    parser.add_argument("--tokenizer-task", default=None)
    parser.add_argument("--text-normalization", default="none")
    parser.add_argument("--decoder-prompt-before-audio", default=None)
    parser.add_argument("--decoder-prompt-before-audio-use-language", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--decoder-ctc-draft-cache-path", default=None)
    parser.add_argument("--decoder-ctc-draft-prompt-template", default=None)
    parser.add_argument("--decoder-ctc-draft-text-key", default=None)
    parser.add_argument("--decoder-ctc-draft-missing-policy", default=None, choices=("empty", "error"))
    parser.add_argument("--decoder-ctc-draft-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-language-mismatch-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-dropout-seed", default=None, type=int)
    parser.add_argument("--frame-shift-ms", default=10.0, type=float)
    parser.add_argument("--feature-extractor-type", default="wenet_fbank")
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
    }
    default_yaml = Path(args.checkpoint_path).resolve().parent / "tokenizer_config.yaml"
    if default_yaml.exists():
        config_data = load_yaml(default_yaml)
        resolved.update(
            {
                "tokenizer_type": config_data.get("tokenizer_type", resolved["tokenizer_type"]),
                "tokenizer_model_path": config_data.get("tokenizer_model_path"),
                "tokenizer_language": config_data.get("tokenizer_language"),
                "tokenizer_task": config_data.get("tokenizer_task"),
                "decoder_prompt_before_audio": config_data.get("decoder_prompt_before_audio", ""),
                "decoder_prompt_before_audio_use_language": bool(
                    config_data.get("decoder_prompt_before_audio_use_language", False)
                ),
                "decoder_ctc_draft_cache_path": config_data.get("decoder_ctc_draft_cache_path"),
                "decoder_ctc_draft_prompt_template": config_data.get("decoder_ctc_draft_prompt_template", ""),
                "decoder_ctc_draft_text_key": config_data.get("decoder_ctc_draft_text_key", "pred_text"),
                "decoder_ctc_draft_missing_policy": config_data.get("decoder_ctc_draft_missing_policy", "empty"),
                "decoder_ctc_draft_dropout_prob": 0.0,
                "decoder_ctc_draft_language_mismatch_dropout_prob": config_data.get(
                    "decoder_ctc_draft_language_mismatch_dropout_prob",
                    0.0,
                ),
                "decoder_ctc_draft_dropout_seed": config_data.get("decoder_ctc_draft_dropout_seed", 0),
            }
        )
    for key in tuple(resolved.keys()):
        value = getattr(args, key)
        if value is not None:
            resolved[key] = value
    return resolved


def _write_preview(
    path: str | Path,
    predictions,
    *,
    preview_count: int,
) -> Path:
    preview_path = Path(path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, prediction in enumerate(predictions[:preview_count], start=1):
        lines.append(f"[{index}] utt_id={prediction.utt_id}")
        lines.append(f"  REF : {prediction.ref_text or ''}")
        lines.append(f"  PRED: {prediction.pred_text or ''}")
        lines.append(f"  SCORE: {prediction.score:.4f}")
        if prediction.decode_strategy != "ctc":
            lines.append(f"  STRATEGY: {prediction.decode_strategy}")
        if (
            prediction.ctc_score is not None
            or prediction.decoder_score is not None
            or prediction.combined_score is not None
        ):
            lines.append(
                "  RESCORE: "
                f"ctc={prediction.ctc_score if prediction.ctc_score is not None else 'n/a'} "
                f"decoder={prediction.decoder_score if prediction.decoder_score is not None else 'n/a'} "
                f"combined={prediction.combined_score if prediction.combined_score is not None else 'n/a'}"
            )
        if prediction.debug is not None:
            lines.append(
                "  DEBUG: "
                f"feat={prediction.debug.feature_length} "
                f"logit={prediction.debug.logit_length} "
                f"pred_tok={prediction.debug.pred_token_count} "
                f"ref_tok={prediction.debug.ref_token_count} "
                f"blank_top1={prediction.debug.blank_top1_ratio:.3f} "
                f"avg_blank={prediction.debug.avg_blank_prob:.3f}"
            )
    preview_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return preview_path


def main() -> None:
    args = build_parser().parse_args()
    if (args.manifest_path is None) == (args.webdataset_root is None):
        raise ValueError("Exactly one of --manifest-path or --webdataset-root must be provided.")
    tokenizer_config = _resolve_tokenizer_config(args)

    config = PredictionConfig(
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
        beam_size=args.beam_size,
        token_prune_topk=args.token_prune_topk,
        decoder_rescore_topk=args.decoder_rescore_topk,
        decoder_rescore_weight=args.decoder_rescore_weight,
        decoder_rescore_length_normalize=args.decoder_rescore_length_normalize,
        length_bonus=args.length_bonus,
        insertion_bonus=args.insertion_bonus,
        blank_logit_bias=args.blank_logit_bias,
        hotwords_path=args.hotwords_path,
        hotword_weight=args.hotword_weight,
        hotword_prefix_scale=args.hotword_prefix_scale,
        save_debug_lengths=args.save_debug_lengths,
        skip_decode_errors=args.skip_decode_errors,
        tokenizer_type=str(tokenizer_config["tokenizer_type"]),
        tokenizer_model_path=tokenizer_config["tokenizer_model_path"],
        tokenizer_language=tokenizer_config["tokenizer_language"],
        tokenizer_task=tokenizer_config["tokenizer_task"],
        text_normalization=args.text_normalization,
        decoder_prompt_before_audio=str(tokenizer_config["decoder_prompt_before_audio"] or ""),
        decoder_prompt_before_audio_use_language=bool(tokenizer_config["decoder_prompt_before_audio_use_language"]),
        decoder_ctc_draft_cache_path=(
            str(tokenizer_config["decoder_ctc_draft_cache_path"])
            if tokenizer_config["decoder_ctc_draft_cache_path"] is not None
            else None
        ),
        decoder_ctc_draft_prompt_template=str(tokenizer_config["decoder_ctc_draft_prompt_template"] or ""),
        decoder_ctc_draft_text_key=str(tokenizer_config["decoder_ctc_draft_text_key"] or "pred_text"),
        decoder_ctc_draft_missing_policy=str(tokenizer_config["decoder_ctc_draft_missing_policy"] or "empty"),
        decoder_ctc_draft_dropout_prob=float(tokenizer_config["decoder_ctc_draft_dropout_prob"] or 0.0),
        decoder_ctc_draft_language_mismatch_dropout_prob=float(
            tokenizer_config["decoder_ctc_draft_language_mismatch_dropout_prob"] or 0.0
        ),
        decoder_ctc_draft_dropout_seed=int(tokenizer_config["decoder_ctc_draft_dropout_seed"] or 0),
        num_workers=args.num_workers,
        frame_shift_ms=args.frame_shift_ms,
        progress_interval=args.progress_interval,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_predictions = []
    num_predictions = 0
    output_mode = "a" if args.append_output else "w"
    with output_path.open(output_mode, encoding="utf-8") as handle:
        def on_prediction(prediction) -> None:
            nonlocal num_predictions
            handle.write(json.dumps(labeled_prediction_to_json_dict(prediction), ensure_ascii=False) + "\n")
            num_predictions += 1
            if args.preview_path is not None and len(preview_predictions) < int(args.preview_count):
                preview_predictions.append(prediction)
            if num_predictions % 100 == 0:
                handle.flush()

        predictions = predict_ctc_labeled(
            config,
            limit=args.limit,
            on_prediction=on_prediction,
            collect_predictions=False,
        )
        handle.flush()

    preview_path = None
    if args.preview_path is not None:
        preview_path = _write_preview(
            args.preview_path,
            preview_predictions,
            preview_count=args.preview_count,
        )

    if args.output_path is not None:
        print(f"saved_jsonl={output_path}")
        if preview_path is not None:
            print(f"saved_preview={preview_path}")
        print(f"num_predictions={num_predictions}")
        return

    for prediction in predictions:
        sys.stdout.write(
            json.dumps(
                {
                    "utt_id": prediction.utt_id,
                    "pred_token_ids": prediction.pred_token_ids,
                    "ref_token_ids": prediction.ref_token_ids,
                    "pred_text": prediction.pred_text,
                    "ref_text": prediction.ref_text,
                    "score": prediction.score,
                    "decode_strategy": prediction.decode_strategy,
                    "ctc_score": prediction.ctc_score,
                    "decoder_score": prediction.decoder_score,
                    "combined_score": prediction.combined_score,
                    "mode": prediction.mode,
                    "alignments": [asdict(alignment) for alignment in prediction.alignments],
                    "debug": None if prediction.debug is None else asdict(prediction.debug),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
