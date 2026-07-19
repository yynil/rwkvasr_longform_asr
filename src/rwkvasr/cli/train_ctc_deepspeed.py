from __future__ import annotations

import argparse
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.training.deepspeed_loop import DeepSpeedTrainConfig, train_ctc_model_deepspeed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RWKV dual-mode CTC ASR with DeepSpeed.")
    parser.add_argument("--config-yaml", required=True)
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
    parser.add_argument("--blank-id", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--encoder-output-dim", default=None, type=int)
    parser.add_argument("--aut-downsample-hidden-size", default=None, type=int)
    parser.add_argument("--aut-activation-function", default=None)
    parser.add_argument("--aut-activation-dropout", default=None, type=float)
    parser.add_argument("--aut-max-source-positions", default=None, type=int)
    parser.add_argument("--aut-scale-embedding", dest="aut_scale_embedding", action="store_true", default=None)
    parser.add_argument("--no-aut-scale-embedding", dest="aut_scale_embedding", action="store_false")
    parser.add_argument("--aut-conv-chunksize", default=None, type=int)
    parser.add_argument("--sensevoice-tp-blocks", default=None, type=int)
    parser.add_argument("--decoder-enabled", dest="decoder_enabled", action="store_true", default=None)
    parser.add_argument("--no-decoder", dest="decoder_enabled", action="store_false")
    parser.add_argument("--decoder-checkpoint-path", default=None)
    parser.add_argument("--decoder-num-layers", default=None, type=int)
    parser.add_argument("--decoder-n-embd", default=None, type=int)
    parser.add_argument("--decoder-ffn-hidden-size", default=None, type=int)
    parser.add_argument("--decoder-vocab-size", default=None, type=int)
    parser.add_argument("--decoder-tokenizer-type", default=None)
    parser.add_argument("--decoder-tokenizer-model-path", default=None)
    parser.add_argument("--decoder-tokenizer-language", default=None)
    parser.add_argument("--decoder-tokenizer-task", default=None)
    parser.add_argument("--decoder-tokenizer-append-eos", dest="decoder_tokenizer_append_eos", action="store_true", default=None)
    parser.add_argument("--no-decoder-tokenizer-append-eos", dest="decoder_tokenizer_append_eos", action="store_false")
    parser.add_argument("--decoder-text-normalization", default=None)
    parser.add_argument("--decoder-head-size", default=None, type=int)
    parser.add_argument("--decoder-audio-conditioning", default=None)
    parser.add_argument("--decoder-prefix-tokens", default=None, type=int)
    parser.add_argument("--decoder-loss-chunk-size", default=None, type=int)
    parser.add_argument("--decoder-text-token-budget", default=None, type=int)
    parser.add_argument("--decoder-prompt-before-audio", default=None)
    parser.add_argument("--decoder-prompt-before-audio-use-language", dest="decoder_prompt_before_audio_use_language", action="store_true", default=None)
    parser.add_argument("--no-decoder-prompt-before-audio-use-language", dest="decoder_prompt_before_audio_use_language", action="store_false")
    parser.add_argument("--decoder-ctc-draft-cache-path", default=None)
    parser.add_argument("--decoder-ctc-draft-prompt-template", default=None)
    parser.add_argument("--decoder-ctc-draft-text-key", default=None)
    parser.add_argument("--decoder-ctc-draft-missing-policy", default=None, choices=("empty", "error"))
    parser.add_argument("--decoder-ctc-draft-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-language-mismatch-dropout-prob", default=None, type=float)
    parser.add_argument("--decoder-ctc-draft-dropout-seed", default=None, type=int)
    parser.add_argument("--ctc-label-override-cache-path", default=None)
    parser.add_argument("--ctc-label-override-text-key", default=None)
    parser.add_argument("--allow-missing-targets", dest="allow_missing_targets", action="store_true", default=None)
    parser.add_argument("--no-allow-missing-targets", dest="allow_missing_targets", action="store_false")
    parser.add_argument("--decoder-prompt-after-audio", default=None)
    parser.add_argument("--decoder-target-prefix", default=None)
    parser.add_argument("--decoder-target-prefix-use-language", dest="decoder_target_prefix_use_language", action="store_true", default=None)
    parser.add_argument("--no-decoder-target-prefix-use-language", dest="decoder_target_prefix_use_language", action="store_false")
    parser.add_argument("--decoder-target-suffix", default=None)
    parser.add_argument("--decoder-language-confirmation-en", default=None)
    parser.add_argument("--decoder-language-confirmation-zh", default=None)
    parser.add_argument("--decoder-prompt-language-label-noise-prob", default=None, type=float)
    parser.add_argument("--decoder-prompt-language-label-noise-seed", default=None, type=int)
    parser.add_argument("--decoder-eos-token-id", default=None, type=int)
    parser.add_argument("--ctc-loss-weight", default=None, type=float)
    parser.add_argument("--decoder-loss-weight", default=None, type=float)
    parser.add_argument("--encoder-anchor-checkpoint-path", default=None)
    parser.add_argument("--encoder-anchor-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-logit-anchor-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-logit-anchor-chunk-frames", default=None, type=int)
    parser.add_argument("--ctc-teacher-topk-cache-path", default=None)
    parser.add_argument("--ctc-teacher-topk-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-topk-blank-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-topk-mass-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-topk-time-map", default=None, choices=("nearest", "linear"))
    parser.add_argument("--ctc-teacher-frame-filter", default=None, choices=("all", "nonblank", "nonblank_neighbors"))
    parser.add_argument("--ctc-teacher-frame-filter-neighbor-radius", default=None, type=int)
    parser.add_argument("--ctc-teacher-frame-filter-min-nonblank-prob", default=None, type=float)
    parser.add_argument("--ctc-teacher-topk-missing-policy", default=None, choices=("skip", "error"))
    parser.add_argument("--ctc-teacher-online-model-path", default=None)
    parser.add_argument("--ctc-teacher-online-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-blank-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-mass-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-full-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-full-temperature", default=None, type=float)
    parser.add_argument(
        "--ctc-teacher-online-full-frame-filter",
        default=None,
        choices=("all", "nonblank", "nonblank_neighbors"),
    )
    parser.add_argument("--ctc-teacher-online-encoder-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-sequence-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-sequence-presence-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-sequence-window-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-sequence-window-radius", default=None, type=int)
    parser.add_argument("--ctc-teacher-online-sequence-window-temperature", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-hard-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-margin-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-margin", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-window-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-window-margin-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-window-topk-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-nonblank-window-radius", default=None, type=int)
    parser.add_argument("--ctc-teacher-online-nonblank-window-temperature", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-top-k", default=None, type=int)
    parser.add_argument("--ctc-teacher-online-audio-index-path", default=None)
    parser.add_argument("--ctc-teacher-online-webdataset-index-path", default=None)
    parser.add_argument("--ctc-teacher-online-audio-cache-dir", default=None)
    parser.add_argument(
        "--ctc-teacher-online-keep-audio-cache",
        dest="ctc_teacher_online_keep_audio_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ctc-teacher-online-device", default=None)
    parser.add_argument(
        "--ctc-teacher-online-use-batch-features",
        dest="ctc_teacher_online_use_batch_features",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--ctc-teacher-online-keep-layer-hiddens-on-device",
        dest="ctc_teacher_online_keep_layer_hiddens_on_device",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ctc-teacher-online-layer-mixer-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-ffn-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-block-loss-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-normalized-mse-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-cosine-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-energy-mse-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-log-rms-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-raw-mse-weight", default=None, type=float)
    parser.add_argument("--ctc-teacher-online-layer-sample-count", default=None, type=int)
    parser.add_argument(
        "--ctc-teacher-online-layer-boundary-id",
        dest="ctc_teacher_online_layer_boundary_ids",
        action="append",
        type=int,
        default=None,
    )
    parser.add_argument("--ctc-teacher-online-layer-frame-tolerance", default=None, type=int)
    parser.add_argument(
        "--ctc-teacher-online-layer-input-mode",
        default=None,
        choices=("stacked", "teacher_forced"),
    )
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
    parser.add_argument("--funasr-nano-ctc-init-load-encoder", dest="funasr_nano_ctc_init_load_encoder", action="store_true", default=None)
    parser.add_argument("--no-funasr-nano-ctc-init-load-encoder", dest="funasr_nano_ctc_init_load_encoder", action="store_false")
    parser.add_argument(
        "--funasr-nano-ctc-init-load-encoder-attention",
        dest="funasr_nano_ctc_init_load_encoder_attention",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-funasr-nano-ctc-init-load-encoder-attention",
        dest="funasr_nano_ctc_init_load_encoder_attention",
        action="store_false",
    )
    parser.add_argument(
        "--funasr-nano-ctc-init-load-rwkv-encoder-from-qkv",
        dest="funasr_nano_ctc_init_load_rwkv_encoder_from_qkv",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--funasr-nano-ctc-init-load-decoder", dest="funasr_nano_ctc_init_load_decoder", action="store_true", default=None)
    parser.add_argument("--no-funasr-nano-ctc-init-load-decoder", dest="funasr_nano_ctc_init_load_decoder", action="store_false")
    parser.add_argument("--funasr-nano-ctc-init-load-head", dest="funasr_nano_ctc_init_load_head", action="store_true", default=None)
    parser.add_argument("--no-funasr-nano-ctc-init-load-head", dest="funasr_nano_ctc_init_load_head", action="store_false")
    parser.add_argument("--funasr-nano-ctc-teacher-blank-id", default=None, type=int)
    parser.add_argument("--funasr-nano-ctc-init-blank-bias-delta", default=None, type=float)
    parser.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", default=None)
    parser.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument(
        "--freeze-encoder-except-time-mixer",
        dest="freeze_encoder_except_time_mixer",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--freeze-ctc-decoder", dest="freeze_ctc_decoder", action="store_true", default=None)
    parser.add_argument("--no-freeze-ctc-decoder", dest="freeze_ctc_decoder", action="store_false")
    parser.add_argument("--freeze-ctc-head", dest="freeze_ctc_head", action="store_true", default=None)
    parser.add_argument("--no-freeze-ctc-head", dest="freeze_ctc_head", action="store_false")
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--save-every", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--decoded-batch-prefetch", default=None, type=int)
    parser.add_argument("--max-open-shards-per-worker", default=None, type=int)
    parser.add_argument("--bucket-source-interleave", dest="bucket_source_interleave", action="store_true", default=None)
    parser.add_argument("--no-bucket-source-interleave", dest="bucket_source_interleave", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder-init-checkpoint-path", default=None)
    parser.add_argument("--init-checkpoint-path", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-tag", default=None)
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
    parser.add_argument("--step-eval-split", default=None)
    parser.add_argument(
        "--step-eval-shuffle",
        dest="step_eval_shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--step-eval-at-start",
        dest="step_eval_at_start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--top-k-step-checkpoints", default=None, type=int)
    parser.add_argument(
        "--save-deepspeed-sharded-checkpoints",
        dest="save_deepspeed_sharded_checkpoints",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-save-deepspeed-sharded-checkpoints",
        dest="save_deepspeed_sharded_checkpoints",
        action="store_false",
    )
    parser.add_argument("--log-every", default=None, type=int)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", default=None, type=int)
    parser.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--batch-token-budget", default=None, type=int)
    parser.add_argument("--length-bucket-frame-budget", default=None, type=int)
    parser.add_argument("--target-gpu-memory-gib", default=None, type=float)
    parser.add_argument("--skip-oversized-samples", dest="skip_oversized_samples", action="store_true", default=None)
    parser.add_argument(
        "--no-skip-oversized-samples",
        dest="skip_oversized_samples",
        action="store_false",
    )
    parser.add_argument("--specaugment-enabled", dest="specaugment_enabled", action="store_true", default=None)
    parser.add_argument("--no-specaugment", dest="specaugment_enabled", action="store_false")
    parser.add_argument("--specaugment-time-masks", default=None, type=int)
    parser.add_argument("--specaugment-time-width", default=None, type=int)
    parser.add_argument("--specaugment-freq-masks", default=None, type=int)
    parser.add_argument("--specaugment-freq-width", default=None, type=int)
    return parser


def _resolve_deepspeed_train_config(args: argparse.Namespace) -> DeepSpeedTrainConfig:
    config_data: dict[str, Any] = load_yaml(args.config_yaml)
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
        "blank_id",
        "batch_size",
        "backend",
        "encoder_output_dim",
        "aut_downsample_hidden_size",
        "aut_activation_function",
        "aut_activation_dropout",
        "aut_max_source_positions",
        "aut_scale_embedding",
        "aut_conv_chunksize",
        "sensevoice_tp_blocks",
        "decoder_enabled",
        "decoder_checkpoint_path",
        "decoder_num_layers",
        "decoder_n_embd",
        "decoder_ffn_hidden_size",
        "decoder_vocab_size",
        "decoder_tokenizer_type",
        "decoder_tokenizer_model_path",
        "decoder_tokenizer_language",
        "decoder_tokenizer_task",
        "decoder_tokenizer_append_eos",
        "decoder_text_normalization",
        "decoder_head_size",
        "decoder_audio_conditioning",
        "decoder_prefix_tokens",
        "decoder_loss_chunk_size",
        "decoder_text_token_budget",
        "decoder_prompt_before_audio",
        "decoder_prompt_before_audio_use_language",
        "decoder_ctc_draft_cache_path",
        "decoder_ctc_draft_prompt_template",
        "decoder_ctc_draft_text_key",
        "decoder_ctc_draft_missing_policy",
        "decoder_ctc_draft_dropout_prob",
        "decoder_ctc_draft_language_mismatch_dropout_prob",
        "decoder_ctc_draft_dropout_seed",
        "ctc_label_override_cache_path",
        "ctc_label_override_text_key",
        "allow_missing_targets",
        "decoder_prompt_after_audio",
        "decoder_target_prefix",
        "decoder_target_prefix_use_language",
        "decoder_target_suffix",
        "decoder_language_confirmation_en",
        "decoder_language_confirmation_zh",
        "decoder_prompt_language_label_noise_prob",
        "decoder_prompt_language_label_noise_seed",
        "decoder_eos_token_id",
        "ctc_loss_weight",
        "decoder_loss_weight",
        "encoder_anchor_checkpoint_path",
        "encoder_anchor_loss_weight",
        "ctc_logit_anchor_loss_weight",
        "ctc_logit_anchor_chunk_frames",
        "ctc_teacher_topk_cache_path",
        "ctc_teacher_topk_loss_weight",
        "ctc_teacher_topk_blank_loss_weight",
        "ctc_teacher_topk_mass_loss_weight",
        "ctc_teacher_topk_time_map",
        "ctc_teacher_frame_filter",
        "ctc_teacher_frame_filter_neighbor_radius",
        "ctc_teacher_frame_filter_min_nonblank_prob",
        "ctc_teacher_topk_missing_policy",
        "ctc_teacher_online_model_path",
        "ctc_teacher_online_loss_weight",
        "ctc_teacher_online_blank_loss_weight",
        "ctc_teacher_online_mass_loss_weight",
        "ctc_teacher_online_full_loss_weight",
        "ctc_teacher_online_full_temperature",
        "ctc_teacher_online_full_frame_filter",
        "ctc_teacher_online_encoder_loss_weight",
        "ctc_teacher_online_sequence_loss_weight",
        "ctc_teacher_online_sequence_presence_loss_weight",
        "ctc_teacher_online_sequence_window_loss_weight",
        "ctc_teacher_online_sequence_window_radius",
        "ctc_teacher_online_sequence_window_temperature",
        "ctc_teacher_online_nonblank_hard_loss_weight",
        "ctc_teacher_online_nonblank_margin_loss_weight",
        "ctc_teacher_online_nonblank_margin",
        "ctc_teacher_online_nonblank_window_loss_weight",
        "ctc_teacher_online_nonblank_window_margin_loss_weight",
        "ctc_teacher_online_nonblank_window_topk_loss_weight",
        "ctc_teacher_online_nonblank_window_radius",
        "ctc_teacher_online_nonblank_window_temperature",
        "ctc_teacher_online_top_k",
        "ctc_teacher_online_audio_index_path",
        "ctc_teacher_online_webdataset_index_path",
        "ctc_teacher_online_audio_cache_dir",
        "ctc_teacher_online_keep_audio_cache",
        "ctc_teacher_online_device",
        "ctc_teacher_online_use_batch_features",
        "ctc_teacher_online_keep_layer_hiddens_on_device",
        "ctc_teacher_online_layer_mixer_loss_weight",
        "ctc_teacher_online_layer_ffn_loss_weight",
        "ctc_teacher_online_layer_block_loss_weight",
        "ctc_teacher_online_layer_normalized_mse_weight",
        "ctc_teacher_online_layer_cosine_weight",
        "ctc_teacher_online_layer_energy_mse_weight",
        "ctc_teacher_online_layer_log_rms_weight",
        "ctc_teacher_online_layer_raw_mse_weight",
        "ctc_teacher_online_layer_sample_count",
        "ctc_teacher_online_layer_boundary_ids",
        "ctc_teacher_online_layer_frame_tolerance",
        "ctc_teacher_online_layer_input_mode",
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
        "funasr_nano_ctc_init_load_encoder",
        "funasr_nano_ctc_init_load_encoder_attention",
        "funasr_nano_ctc_init_load_rwkv_encoder_from_qkv",
        "funasr_nano_ctc_init_load_decoder",
        "funasr_nano_ctc_init_load_head",
        "funasr_nano_ctc_teacher_blank_id",
        "funasr_nano_ctc_init_blank_bias_delta",
        "freeze_encoder",
        "freeze_encoder_except_time_mixer",
        "freeze_ctc_decoder",
        "freeze_ctc_head",
        "max_steps",
        "epochs",
        "save_every",
        "num_workers",
        "decoded_batch_prefetch",
        "max_open_shards_per_worker",
        "bucket_source_interleave",
        "device",
        "encoder_init_checkpoint_path",
        "init_checkpoint_path",
        "resume_from",
        "resume_tag",
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
        "step_eval_split",
        "step_eval_shuffle",
        "step_eval_at_start",
        "top_k_step_checkpoints",
        "save_deepspeed_sharded_checkpoints",
        "log_every",
        "local_rank",
        "gradient_checkpointing",
        "batch_token_budget",
        "length_bucket_frame_budget",
        "target_gpu_memory_gib",
        "skip_oversized_samples",
        "specaugment_enabled",
        "specaugment_time_masks",
        "specaugment_time_width",
        "specaugment_freq_masks",
        "specaugment_freq_width",
    )
    for key in cli_keys:
        value = getattr(args, key)
        if value is not None:
            config_data[key] = value

    if "deepspeed" not in config_data:
        raise ValueError("DeepSpeed YAML config must contain a top-level `deepspeed` mapping.")

    try:
        return DeepSpeedTrainConfig(**config_data)
    except TypeError as exc:
        raise ValueError(
            "DeepSpeedTrainConfig is incomplete or has unexpected fields. "
            "Provide required fields in --config-yaml."
        ) from exc


def main() -> None:
    args = build_parser().parse_args()
    result = train_ctc_model_deepspeed(_resolve_deepspeed_train_config(args))
    print(
        "train_ctc_deepspeed "
        f"steps={result['steps']} final_loss={result['final_loss']:.4f} zero_stage={result['zero_stage']}"
    )


if __name__ == "__main__":
    main()
