from __future__ import annotations

import argparse

from rwkvasr.config import load_yaml
from rwkvasr.training.train_loop import TrainConfig, train_ctc_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RWKV dual-mode CTC ASR from a manifest or WebDataset.")
    parser.add_argument("--config-yaml", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--webdataset-root", default=None)
    parser.add_argument("--webdataset-index-path", default=None)
    parser.add_argument("--webdataset-length-index-path", default=None)
    parser.add_argument("--webdataset-bucket-manifest-path", default=None)
    parser.add_argument("--webdataset-split", default=None)
    parser.add_argument("--webdataset-eval-ratio", default=None, type=float)
    parser.add_argument("--webdataset-hash-seed", default=None, type=int)
    parser.add_argument("--webdataset-split-by", default=None)
    parser.add_argument("--webdataset-utt-id-key", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--vocab-size", default=None, type=int)
    parser.add_argument("--tokenizer-type", default=None)
    parser.add_argument("--tokenizer-model-path", default=None)
    parser.add_argument("--tokenizer-language", default=None)
    parser.add_argument("--tokenizer-task", default=None)
    parser.add_argument("--tokenizer-append-eos", dest="tokenizer_append_eos", action="store_true", default=None)
    parser.add_argument("--no-tokenizer-append-eos", dest="tokenizer_append_eos", action="store_false")
    parser.add_argument("--text-normalization", default=None)
    parser.add_argument("--feature-extractor-type", default=None)
    parser.add_argument("--input-dim", default=None, type=int)
    parser.add_argument("--n-embd", default=None, type=int)
    parser.add_argument("--dim-att", default=None, type=int)
    parser.add_argument("--dim-ff", default=None, type=int)
    parser.add_argument("--num-layers", default=None, type=int)
    parser.add_argument("--head-size", default=None, type=int)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--conv-kernel-size", default=None, type=int)
    parser.add_argument("--dropout", default=None, type=float)
    parser.add_argument("--frontend-type", default=None)
    parser.add_argument("--encoder-output-dim", default=None, type=int)
    parser.add_argument("--aut-downsample-hidden-size", default=None, type=int)
    parser.add_argument("--aut-activation-function", default=None)
    parser.add_argument("--aut-activation-dropout", default=None, type=float)
    parser.add_argument("--aut-max-source-positions", default=None, type=int)
    parser.add_argument("--aut-scale-embedding", dest="aut_scale_embedding", action="store_true", default=None)
    parser.add_argument("--no-aut-scale-embedding", dest="aut_scale_embedding", action="store_false")
    parser.add_argument("--aut-conv-chunksize", default=None, type=int)
    parser.add_argument("--sensevoice-tp-blocks", default=None, type=int)
    parser.add_argument("--cmvn-file", default=None)
    parser.add_argument("--cmvn-is-json", dest="cmvn_is_json", action="store_true", default=None)
    parser.add_argument("--cmvn-is-bin", dest="cmvn_is_json", action="store_false")
    parser.add_argument("--blank-id", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--decoded-batch-prefetch", default=None, type=int)
    parser.add_argument("--max-open-shards-per-worker", default=None, type=int)
    parser.add_argument("--bucket-source-interleave", dest="bucket_source_interleave", action="store_true", default=None)
    parser.add_argument("--no-bucket-source-interleave", dest="bucket_source_interleave", action="store_false")
    parser.add_argument("--length-bucket-schedule-block-size", default=None, type=int)
    parser.add_argument("--bucket-source-interleave-block-size", default=None, type=int)
    parser.add_argument("--bucket-serialize-reads", dest="bucket_serialize_reads", action="store_true", default=None)
    parser.add_argument("--no-bucket-serialize-reads", dest="bucket_serialize_reads", action="store_false")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--beta1", default=None, type=float)
    parser.add_argument("--beta2", default=None, type=float)
    parser.add_argument("--eps", default=None, type=float)
    parser.add_argument("--decoder-enabled", dest="decoder_enabled", action="store_true", default=None)
    parser.add_argument("--no-decoder", dest="decoder_enabled", action="store_false")
    parser.add_argument("--decoder-checkpoint-path", default=None)
    parser.add_argument("--decoder-num-layers", default=None, type=int)
    parser.add_argument("--decoder-n-embd", default=None, type=int)
    parser.add_argument("--decoder-ffn-hidden-size", default=None, type=int)
    parser.add_argument("--decoder-head-size", default=None, type=int)
    parser.add_argument("--decoder-audio-conditioning", default=None)
    parser.add_argument("--decoder-prefix-tokens", default=None, type=int)
    parser.add_argument("--decoder-loss-chunk-size", default=None, type=int)
    parser.add_argument("--decoder-text-token-budget", default=None, type=int)
    parser.add_argument("--decoder-prompt-before-audio", default=None)
    parser.add_argument("--decoder-ctc-draft-cache-path", default=None)
    parser.add_argument("--decoder-ctc-draft-prompt-template", default=None)
    parser.add_argument("--decoder-ctc-draft-text-key", default=None)
    parser.add_argument("--decoder-ctc-draft-missing-policy", default=None, choices=("empty", "error"))
    parser.add_argument("--decoder-ctc-draft-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-language-mismatch-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-dropout-seed", default=None, type=int)
    parser.add_argument("--ctc-label-override-cache-path", default=None)
    parser.add_argument("--ctc-label-override-text-key", default=None)
    parser.add_argument("--decoder-prompt-after-audio", default=None)
    parser.add_argument("--decoder-target-suffix", default=None)
    parser.add_argument("--decoder-eos-token-id", default=None, type=int)
    parser.add_argument("--ctc-loss-weight", default=None, type=float)
    parser.add_argument("--decoder-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-decoder-type", default=None)
    parser.add_argument("--ctc-decoder-downsample-rate", default=None, type=int)
    parser.add_argument("--ctc-decoder-dim", default=None, type=int)
    parser.add_argument("--ctc-decoder-ffn-dim", default=None, type=int)
    parser.add_argument("--ctc-decoder-num-layers", default=None, type=int)
    parser.add_argument("--ctc-decoder-attention-heads", default=None, type=int)
    parser.add_argument("--ctc-decoder-dropout", default=None, type=float)
    parser.add_argument("--ctc-decoder-attention-dropout", default=None, type=float)
    parser.add_argument("--ctc-bridge-type", default=None, choices=("none", "identity", "linear", "mlp", "residual_mlp"))
    parser.add_argument("--ctc-bridge-hidden-dim", default=None, type=int)
    parser.add_argument("--ctc-bridge-dropout", default=None, type=float)
    parser.add_argument(
        "--ctc-suppress-non-pronunciation-tokens",
        dest="ctc_suppress_non_pronunciation_tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ctc-suppressed-token-id", dest="ctc_suppressed_token_ids", action="append", type=int, default=None)
    parser.add_argument("--funasr-nano-ctc-init-checkpoint-path", default=None)
    parser.add_argument("--funasr-nano-ctc-init-load-decoder", dest="funasr_nano_ctc_init_load_decoder", action="store_true", default=None)
    parser.add_argument("--no-funasr-nano-ctc-init-load-decoder", dest="funasr_nano_ctc_init_load_decoder", action="store_false")
    parser.add_argument("--funasr-nano-ctc-init-load-head", dest="funasr_nano_ctc_init_load_head", action="store_true", default=None)
    parser.add_argument("--no-funasr-nano-ctc-init-load-head", dest="funasr_nano_ctc_init_load_head", action="store_false")
    parser.add_argument("--funasr-nano-ctc-teacher-blank-id", default=None, type=int)
    parser.add_argument("--direction-variant", default=None)
    parser.add_argument("--p-start", default=None, type=float)
    parser.add_argument("--p-max", default=None, type=float)
    parser.add_argument("--warmup-steps", default=None, type=int)
    parser.add_argument("--ramp-steps", default=None, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder-init-checkpoint-path", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", default=None)
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--funasr-nano-ctc-init-blank-bias-delta", default=None, type=float)
    parser.add_argument("--freeze-ctc-decoder", dest="freeze_ctc_decoder", action="store_true", default=None)
    parser.add_argument("--no-freeze-ctc-decoder", dest="freeze_ctc_decoder", action="store_false")
    parser.add_argument("--freeze-ctc-head", dest="freeze_ctc_head", action="store_true", default=None)
    parser.add_argument("--no-freeze-ctc-head", dest="freeze_ctc_head", action="store_false")
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-base-url", default=None)
    parser.add_argument("--wandb-init-timeout-sec", default=None, type=float)
    parser.add_argument("--eval-mode", default=None)
    parser.add_argument("--max-eval-samples", default=None, type=int)
    parser.add_argument("--eval-batch-size", default=None, type=int)
    parser.add_argument("--step-eval-batch-size", default=None, type=int)
    parser.add_argument("--step-eval-every", default=None, type=int)
    parser.add_argument("--step-eval-samples", default=None, type=int)
    parser.add_argument("--top-k-step-checkpoints", default=None, type=int)
    parser.add_argument("--batch-token-budget", default=None, type=int)
    parser.add_argument("--length-bucket-frame-budget", default=None, type=int)
    parser.add_argument("--target-gpu-memory-gib", default=None, type=float)
    parser.add_argument("--skip-oversized-samples", dest="skip_oversized_samples", action="store_true", default=None)
    parser.add_argument(
        "--no-skip-oversized-samples",
        dest="skip_oversized_samples",
        action="store_false",
    )
    return parser


def _resolve_train_config(args: argparse.Namespace) -> TrainConfig:
    config_data: dict[str, object] = {}
    if args.config_yaml is not None:
        config_data.update(load_yaml(args.config_yaml))

    cli_keys = (
        "manifest_path",
        "webdataset_root",
        "webdataset_index_path",
        "webdataset_length_index_path",
        "webdataset_bucket_manifest_path",
        "webdataset_split",
        "webdataset_eval_ratio",
        "webdataset_hash_seed",
        "webdataset_split_by",
        "webdataset_utt_id_key",
        "output_dir",
        "vocab_size",
        "tokenizer_type",
        "tokenizer_model_path",
        "tokenizer_language",
        "tokenizer_task",
        "tokenizer_append_eos",
        "text_normalization",
        "feature_extractor_type",
        "input_dim",
        "n_embd",
        "dim_att",
        "dim_ff",
        "num_layers",
        "head_size",
        "backend",
        "conv_kernel_size",
        "dropout",
        "frontend_type",
        "encoder_output_dim",
        "aut_downsample_hidden_size",
        "aut_activation_function",
        "aut_activation_dropout",
        "aut_max_source_positions",
        "aut_scale_embedding",
        "aut_conv_chunksize",
        "sensevoice_tp_blocks",
        "cmvn_file",
        "cmvn_is_json",
        "blank_id",
        "batch_size",
        "max_steps",
        "epochs",
        "save_every",
        "num_workers",
        "decoded_batch_prefetch",
        "max_open_shards_per_worker",
        "bucket_source_interleave",
        "length_bucket_schedule_block_size",
        "bucket_source_interleave_block_size",
        "bucket_serialize_reads",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps",
        "decoder_enabled",
        "decoder_checkpoint_path",
        "decoder_num_layers",
        "decoder_n_embd",
        "decoder_ffn_hidden_size",
        "decoder_head_size",
        "decoder_audio_conditioning",
        "decoder_prefix_tokens",
        "decoder_loss_chunk_size",
        "decoder_text_token_budget",
        "decoder_prompt_before_audio",
        "decoder_ctc_draft_cache_path",
        "decoder_ctc_draft_prompt_template",
        "decoder_ctc_draft_text_key",
        "decoder_ctc_draft_missing_policy",
        "decoder_ctc_draft_dropout_prob",
        "decoder_ctc_draft_language_mismatch_dropout_prob",
        "decoder_ctc_draft_dropout_seed",
        "ctc_label_override_cache_path",
        "ctc_label_override_text_key",
        "decoder_prompt_after_audio",
        "decoder_target_suffix",
        "decoder_eos_token_id",
        "ctc_loss_weight",
        "decoder_loss_weight",
        "ctc_decoder_type",
        "ctc_decoder_downsample_rate",
        "ctc_decoder_dim",
        "ctc_decoder_ffn_dim",
        "ctc_decoder_num_layers",
        "ctc_decoder_attention_heads",
        "ctc_decoder_dropout",
        "ctc_decoder_attention_dropout",
        "ctc_bridge_type",
        "ctc_bridge_hidden_dim",
        "ctc_bridge_dropout",
        "ctc_suppress_non_pronunciation_tokens",
        "ctc_suppressed_token_ids",
        "funasr_nano_ctc_init_checkpoint_path",
        "funasr_nano_ctc_init_load_decoder",
        "funasr_nano_ctc_init_load_head",
        "funasr_nano_ctc_teacher_blank_id",
        "funasr_nano_ctc_init_blank_bias_delta",
        "freeze_encoder",
        "freeze_ctc_decoder",
        "freeze_ctc_head",
        "direction_variant",
        "p_start",
        "p_max",
        "warmup_steps",
        "ramp_steps",
        "device",
        "encoder_init_checkpoint_path",
        "resume_from",
        "wandb_enabled",
        "wandb_project",
        "wandb_run_name",
        "wandb_base_url",
        "wandb_init_timeout_sec",
        "eval_mode",
        "max_eval_samples",
        "eval_batch_size",
        "step_eval_batch_size",
        "step_eval_every",
        "step_eval_samples",
        "top_k_step_checkpoints",
        "batch_token_budget",
        "length_bucket_frame_budget",
        "target_gpu_memory_gib",
        "skip_oversized_samples",
    )
    for key in cli_keys:
        value = getattr(args, key)
        if value is not None:
            config_data[key] = value

    try:
        return TrainConfig(**config_data)
    except TypeError as exc:
        raise ValueError(
            "TrainConfig is incomplete. Provide required fields in --config-yaml or via CLI."
        ) from exc


def main() -> None:
    args = build_parser().parse_args()
    result = train_ctc_model(_resolve_train_config(args))
    print(f"train_ctc steps={result['steps']} final_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
