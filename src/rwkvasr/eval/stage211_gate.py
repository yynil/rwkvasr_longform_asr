from __future__ import annotations

import hashlib
import json
import math
import re
from functools import cache
from pathlib import Path
from typing import Any

import torch

from rwkvasr.config import load_yaml
from rwkvasr.data import (
    build_text_tokenizer,
    ctc_suppressed_token_ids_for_tokenizer,
)
from rwkvasr.eval.stage211_public_metrics import (
    build_stage211_public_progress,
    replay_stage211_public_comparison,
)
from rwkvasr.eval.stage211_supplemental import (
    STAGE211_SUPPLEMENTAL_DIFFICULTY,
    stage211_supplemental_profile,
)


STAGE211_PHASE_GATE_SCHEMA_VERSION = 1
STAGE211_NANO_PUBLIC_BASELINE_SCHEMA_VERSION = 1
STAGE211_FULL_DATA_EPOCHS = 3
STAGE211_FULL_DATA_BATCH_SIZE = 36
STAGE211_FULL_DATA_WORLD_SIZE = 4
STAGE211_FULL_DATA_FRAME_BUDGET = 24_000
STAGE211_RETENTION_CORRECTION_MAX_ROUNDS = 3
STAGE211_RETENTION_CORRECTION_EPOCHS = 1
STAGE211_RETENTION_CORRECTION_LR = 1.0e-6
STAGE211_POST_COVERAGE_CORRECTION_LRS = {
    "mixer": STAGE211_RETENTION_CORRECTION_LR,
    "block": 1.0e-6,
    "logits": 1.5e-7,
}
STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES = 256
STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT = "alignment_checkpoint_eval"
STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT = 2_114
STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256 = (
    "76a68d03bb2dc486c214fd44891f3e3e2c36286d79fea1e69c8e74d7767d5a09"
)
_STAGE211_TOKENIZER_MODEL_PATH = "assets/fun-asr-nano-2512/multilingual.tiktoken"
_STAGE211_STUDENT_BLANK_ID = 60_515
_STAGE211_NANO_BLANK_ID = 60_514
_STAGE211_ALIGNMENT_LAYER_IDS = tuple(range(70))
_STAGE211_ALIGNMENT_LAYER_KEYS = {str(index) for index in _STAGE211_ALIGNMENT_LAYER_IDS}
_STAGE211_ALIGNMENT_WEAK_BANDS = {
    "10-19": tuple(range(10, 20)),
    "20-29": tuple(range(20, 30)),
}
_STAGE211_ALIGNMENT_STRATIFIED_CELLS = {
    "easy_en",
    "easy_zh",
    "medium_en",
    "medium_zh",
    "hard_en",
    "hard_zh",
    "long_zh",
    "supplemental_en",
    "supplemental_zh",
}
_STAGE211_ALIGNMENT_STRATIFIED_SAMPLES = STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES * len(
    _STAGE211_ALIGNMENT_STRATIFIED_CELLS
)
_STAGE211_ALIGNMENT_PHASE_COMPONENTS = {
    "mixer": ("mixer",),
    "block": ("mixer", "ffn", "block"),
    "logits": ("mixer", "ffn", "block"),
}
_STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS = 68
_STAGE211_ALIGNMENT_MAX_CELL_REGRESSION_PCT = 10.0
_STAGE211_LOGITS_MIN_KL_RELATIVE_REDUCTION = 0.05
_STAGE211_LOGITS_RATIO_MIN = 0.90
_STAGE211_LOGITS_RATIO_MAX = 1.10
_STAGE211_LOGITS_MAX_CELL_TOKEN_ERROR_REGRESSION = 0.03
_STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION = 0.10
_STAGE211_LOGITS_MAX_HIDDEN_COSINE_REGRESSION = 0.01
_STAGE211_ALIGNMENT_TOLERANCE = 1.0e-12
_STAGE211_LOGITS_REQUIRED_METRICS = (
    "full_kl",
    "conditional_nonblank_kl",
    "conditional_nonblank_hard_ce",
    "blank_binary_kl",
    "selected_top1_agreement",
    "all_top1_agreement",
    "active_top1_agreement",
    "blank_prob_mae",
    "teacher_nonblank_rate",
    "student_nonblank_rate",
    "nonblank_rate_ratio",
    "teacher_nonblank_to_blank_rate",
    "teacher_blank_to_nonblank_rate",
    "ctc_token_error_rate",
    "ctc_token_insertion_rate",
    "ctc_token_deletion_rate",
    "ctc_token_substitution_rate",
    "collapsed_length_ratio",
    "sequence_exact_rate",
    "mean_frame_delta",
    "selected_frames",
    "all_frames",
    "teacher_tokens",
    "student_tokens",
    "matched_utterances",
    "missing_utterances",
)
_STAGE211_LOGITS_IDENTICAL_TEACHER_METRICS = (
    "teacher_nonblank_rate",
    "selected_frames",
    "all_frames",
    "teacher_tokens",
    "matched_utterances",
)
STAGE211_ALLOWED_OPERATOR_KEY_MARKERS = (".time_mixer.", ".input_proj.")
DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "stage179_usbhd_dedup_alignment_manifest.json"
)
DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "stage211_loaded_manifest_chain_receipt.json"
)
DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "public_train_overlap_v1"
    / "receipt.json"
)
STAGE211_LOADED_MANIFEST_RECEIPT_SHA256 = (
    "af7c38a72fd714148390de5dea7d4d32a062a78ed1e25143928e0c3db88561c1"
)
STAGE211_PUBLIC_OVERLAP_RECEIPT_SHA256 = (
    "af6a0034efe6d231649966337d29aadf43b33d658c0e479e48f0ce1fae8da6b4"
)
STAGE211_CLEAN_COMMONVOICE_MANIFEST_SHA256 = (
    "4cf4f3e171f888660b140e187d4848c9f0d26d77b9ed166d7f3e0e25c699e21b"
)
STAGE211_GLOBAL_DEDUP_MANIFEST_SHA256 = (
    "9228d24a8befe24d0e35debd63070f24a6a6313bf318308452e95656dd6708ba"
)
STAGE211_GLOBAL_DEDUP_TOTAL_HOURS = 118_465.16068055555
STAGE211_AUDIO_TRAIN_PART_COUNTS = {
    "easy": 70,
    "medium": 436,
    "hard": 218,
    "long": 21,
}
STAGE211_AUDIO_CURRICULUM: dict[str, dict[str, float | int]] = {
    "easy": {
        "rows": 1_174_987,
        "hours": 1_490.119,
        "steps_per_epoch": 9_982,
        "steps": 29_946,
        "tail_padding_samples_per_epoch": 633,
    },
    "medium": {
        "rows": 42_247_508,
        "hours": 47_771.242,
        "steps_per_epoch": 334_571,
        "steps": 1_003_713,
        "tail_padding_samples_per_epoch": 744,
    },
    "hard": {
        "rows": 18_649_326,
        "hours": 69_200.368,
        "steps_per_epoch": 322_105,
        "steps": 966_315,
        "tail_padding_samples_per_epoch": 986,
    },
    "long": {
        "rows": 404,
        "hours": 3.432,
        "steps_per_epoch": 35,
        "steps": 105,
        "tail_padding_samples_per_epoch": 296,
    },
}
STAGE211_PUBLIC_BENCHMARKS: dict[str, dict[str, str | int]] = {
    "aishell1_test": {"language": "zh", "metric": "cer", "samples": 7_176},
    "librispeech_test_clean": {"language": "en", "metric": "wer", "samples": 2_620},
    "librispeech_test_other": {"language": "en", "metric": "wer", "samples": 2_939},
    "commonvoice_en_test": {"language": "en", "metric": "wer", "samples": 14_922},
    "wenetspeech_test_net": {"language": "zh", "metric": "cer", "samples": 24_774},
}
DEFAULT_STAGE211_NANO_PUBLIC_BASELINE_RECEIPT = (
    Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512" / "provenance_receipt.json"
)
STAGE211_AUDIO_TOTAL_ROWS = sum(int(row["rows"]) for row in STAGE211_AUDIO_CURRICULUM.values())
STAGE211_AUDIO_TOTAL_HOURS = sum(float(row["hours"]) for row in STAGE211_AUDIO_CURRICULUM.values())
STAGE211_AUDIO_TOTAL_ROW_EXPOSURES = STAGE211_AUDIO_TOTAL_ROWS * STAGE211_FULL_DATA_EPOCHS
STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES = STAGE211_AUDIO_TOTAL_HOURS * STAGE211_FULL_DATA_EPOCHS
STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES = sum(
    int(row["tail_padding_samples_per_epoch"]) * STAGE211_FULL_DATA_EPOCHS
    for row in STAGE211_AUDIO_CURRICULUM.values()
)
STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES = (
    STAGE211_AUDIO_TOTAL_ROW_EXPOSURES + STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
)
_STAGE211_COMMON_PHASE_TRAIN_CONFIG: dict[str, Any] = {
    "vocab_size": 60_515,
    "blank_id": _STAGE211_STUDENT_BLANK_ID,
    "tokenizer_type": "sensevoice_tiktoken",
    "tokenizer_model_path": _STAGE211_TOKENIZER_MODEL_PATH,
    "tokenizer_append_eos": False,
    "text_normalization": "ctc",
    "weight_decay": 0.0,
    "freeze_encoder": False,
    "freeze_encoder_except_time_mixer": True,
    "freeze_ctc_decoder": True,
    "freeze_ctc_head": True,
    "decoder_loss_weight": 0.0,
    "funasr_nano_ctc_init_checkpoint_path": None,
    "funasr_nano_ctc_init_load_encoder": False,
    "funasr_nano_ctc_init_load_encoder_attention": False,
    "funasr_nano_ctc_init_load_rwkv_encoder_from_qkv": False,
    "funasr_nano_ctc_init_load_decoder": False,
    "funasr_nano_ctc_init_load_head": False,
    "ctc_teacher_frame_filter": "all",
    "ctc_teacher_online_use_batch_features": True,
    "ctc_teacher_online_layer_raw_mse_weight": 0.0,
    "ctc_teacher_online_layer_normalized_mse_weight": 1.0,
    "ctc_teacher_online_layer_cosine_weight": 0.25,
    "ctc_teacher_online_layer_energy_mse_weight": 0.25,
    "ctc_teacher_online_layer_log_rms_weight": 0.1,
    "ctc_teacher_online_layer_frame_tolerance": 0,
    "ctc_teacher_online_frame_balance_mode": "all",
    "ctc_teacher_online_blank_frame_balance_mode": "all",
    "ctc_teacher_online_hidden_frame_balance_mode": "all",
    "ctc_teacher_online_loss_weight": 0.0,
    "ctc_teacher_online_mass_loss_weight": 0.0,
    "ctc_teacher_online_full_loss_weight": 0.0,
    "ctc_teacher_online_sequence_presence_loss_weight": 0.0,
    "ctc_teacher_online_sequence_window_loss_weight": 0.0,
    "ctc_teacher_online_nonblank_hard_loss_weight": 0.0,
    "ctc_teacher_online_nonblank_margin_loss_weight": 0.0,
    "ctc_teacher_online_nonblank_window_margin_loss_weight": 0.0,
    "ctc_teacher_online_nonblank_window_topk_loss_weight": 0.0,
    "ctc_teacher_online_nonblank_window_loss_mode": "conditional_nonblank_hard",
    "ctc_teacher_online_nonblank_window_radius": 2,
    "ctc_teacher_online_nonblank_window_temperature": 0.2,
    "ctc_teacher_online_project_ignored_token_ids": [_STAGE211_NANO_BLANK_ID],
    "ctc_teacher_online_top_k": 32,
    "funasr_nano_ctc_teacher_blank_id": _STAGE211_NANO_BLANK_ID,
}
_STAGE211_PHASE_TRAIN_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "mixer": {
        "lr": 3.0e-6,
        "allow_missing_targets": True,
        "ctc_loss_weight": 0.0,
        "ctc_suppress_non_pronunciation_tokens": False,
        "ctc_teacher_online_layer_input_mode": "teacher_forced",
        "ctc_teacher_online_layer_mixer_loss_weight": 1.0,
        "ctc_teacher_online_layer_ffn_loss_weight": 0.0,
        "ctc_teacher_online_layer_block_loss_weight": 0.0,
        "ctc_teacher_online_encoder_loss_weight": 0.0,
        "ctc_teacher_online_decoder_hidden_loss_weight": 0.0,
        "ctc_teacher_online_blank_loss_weight": 0.0,
        "ctc_teacher_online_conditional_nonblank_loss_weight": 0.0,
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight": 0.0,
        "ctc_teacher_online_sequence_loss_weight": 0.0,
        "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
        "ctc_teacher_online_layer_sample_count": 8,
        "ctc_teacher_online_layer_boundary_ids": [],
        "ctc_teacher_online_layer_include_boundaries": False,
        "ctc_teacher_online_keep_full_log_probs_on_device": False,
    },
    "block": {
        "lr": 2.0e-6,
        "allow_missing_targets": True,
        "ctc_loss_weight": 0.0,
        "ctc_suppress_non_pronunciation_tokens": False,
        "ctc_teacher_online_layer_input_mode": "stacked",
        "ctc_teacher_online_layer_mixer_loss_weight": 0.25,
        "ctc_teacher_online_layer_ffn_loss_weight": 0.25,
        "ctc_teacher_online_layer_block_loss_weight": 1.0,
        "ctc_teacher_online_encoder_loss_weight": 0.5,
        "ctc_teacher_online_decoder_hidden_loss_weight": 0.5,
        "ctc_teacher_online_blank_loss_weight": 0.0,
        "ctc_teacher_online_conditional_nonblank_loss_weight": 0.0,
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight": 0.0,
        "ctc_teacher_online_sequence_loss_weight": 0.0,
        "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
        "ctc_teacher_online_layer_sample_count": 8,
        "ctc_teacher_online_layer_boundary_ids": [],
        "ctc_teacher_online_layer_include_boundaries": False,
        "ctc_teacher_online_keep_full_log_probs_on_device": False,
    },
    "logits": {
        "lr": 3.0e-7,
        "allow_missing_targets": True,
        "ctc_loss_weight": 0.0,
        "ctc_suppress_non_pronunciation_tokens": False,
        "ctc_teacher_online_layer_input_mode": "stacked",
        "ctc_teacher_online_layer_mixer_loss_weight": 0.1,
        "ctc_teacher_online_layer_ffn_loss_weight": 0.1,
        "ctc_teacher_online_layer_block_loss_weight": 0.25,
        "ctc_teacher_online_encoder_loss_weight": 0.25,
        "ctc_teacher_online_decoder_hidden_loss_weight": 0.25,
        "ctc_teacher_online_blank_loss_weight": 0.25,
        "ctc_teacher_online_conditional_nonblank_loss_weight": 1.0,
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight": 0.125,
        "ctc_teacher_online_sequence_loss_weight": 0.2,
        "ctc_teacher_online_nonblank_window_loss_weight": 0.25,
        "ctc_teacher_online_layer_sample_count": 12,
        "ctc_teacher_online_layer_boundary_ids": [0, 11, 12, 17, 20, 49, 50, 69],
        "ctc_teacher_online_layer_include_boundaries": True,
        "ctc_teacher_online_keep_full_log_probs_on_device": True,
    },
    "sft": {
        "lr": 3.0e-7,
        "allow_missing_targets": False,
        "ctc_loss_weight": 1.0,
        "ctc_suppress_non_pronunciation_tokens": True,
        "ctc_teacher_online_layer_input_mode": "stacked",
        "ctc_teacher_online_layer_mixer_loss_weight": 0.05,
        "ctc_teacher_online_layer_ffn_loss_weight": 0.05,
        "ctc_teacher_online_layer_block_loss_weight": 0.1,
        "ctc_teacher_online_encoder_loss_weight": 0.1,
        "ctc_teacher_online_decoder_hidden_loss_weight": 0.1,
        "ctc_teacher_online_blank_loss_weight": 0.05,
        "ctc_teacher_online_conditional_nonblank_loss_weight": 0.1,
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight": 0.0,
        "ctc_teacher_online_sequence_loss_weight": 0.0,
        "ctc_teacher_online_nonblank_window_loss_weight": 0.0,
        "ctc_teacher_online_layer_sample_count": 8,
        "ctc_teacher_online_layer_boundary_ids": [0, 11, 12, 17, 20, 49, 50, 69],
        "ctc_teacher_online_layer_include_boundaries": True,
        "ctc_teacher_online_keep_full_log_probs_on_device": False,
    },
}


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def stage211_sft_ctc_suppressed_token_ids() -> tuple[int, ...]:
    tokenizer_model_path = Path(__file__).resolve().parents[3] / _STAGE211_TOKENIZER_MODEL_PATH
    if not tokenizer_model_path.is_file():
        raise FileNotFoundError(f"Stage211 tokenizer model is unavailable: {tokenizer_model_path}")
    tokenizer = build_text_tokenizer(
        "sensevoice_tiktoken",
        model_path=str(tokenizer_model_path),
    )
    token_ids = tuple(
        sorted(
            set(
                ctc_suppressed_token_ids_for_tokenizer(
                    tokenizer,
                    blank_id=_STAGE211_STUDENT_BLANK_ID,
                )
            )
        )
    )
    fingerprint = hashlib.sha256(
        ",".join(str(token_id) for token_id in token_ids).encode("ascii")
    ).hexdigest()
    if (
        len(token_ids) != STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT
        or fingerprint != STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256
        or _STAGE211_NANO_BLANK_ID not in token_ids
        or _STAGE211_STUDENT_BLANK_ID in token_ids
    ):
        raise ValueError(
            "Stage211 SFT pronunciation-token support differs from its bound contract: "
            f"count={len(token_ids)} sha256={fingerprint} "
            f"nano_blank_suppressed={_STAGE211_NANO_BLANK_ID in token_ids} "
            f"student_blank_suppressed={_STAGE211_STUDENT_BLANK_ID in token_ids}"
        )
    return token_ids


def stage211_phase_train_config_contract(phase: str) -> dict[str, Any]:
    try:
        phase_values = _STAGE211_PHASE_TRAIN_CONFIG_OVERRIDES[phase]
    except KeyError as error:
        raise ValueError(f"Unsupported Stage211 phase objective: {phase!r}") from error
    contract = {**_STAGE211_COMMON_PHASE_TRAIN_CONFIG, **phase_values}
    if phase == "sft":
        suppressed_token_ids = list(stage211_sft_ctc_suppressed_token_ids())
        contract["ctc_suppressed_token_ids"] = suppressed_token_ids
        contract["ctc_teacher_online_project_ignored_token_ids"] = list(suppressed_token_ids)
    return {
        key: list(value) if isinstance(value, list) else value for key, value in contract.items()
    }


def stage211_post_coverage_correction_lr(phase: str) -> float:
    try:
        return float(STAGE211_POST_COVERAGE_CORRECTION_LRS[phase])
    except KeyError as error:
        raise ValueError(
            f"Unsupported Stage211 post-coverage correction phase: {phase!r}"
        ) from error


def validate_stage211_phase_train_config(
    train_config: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    contract = stage211_phase_train_config_contract(phase)
    for key, expected in contract.items():
        actual = train_config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Stage211 {phase} train config {key} mismatch: "
                f"actual={actual!r} expected={expected!r}"
            )
    return contract


def resolve_stage211_nano_teacher_checkpoint(train_config: dict[str, Any]) -> Path:
    model_dir_value = train_config.get("ctc_teacher_online_model_path")
    if not isinstance(model_dir_value, (str, Path)) or not str(model_dir_value).strip():
        raise ValueError("Stage211 train config lacks ctc_teacher_online_model_path.")
    checkpoint_path = Path(model_dir_value).expanduser().resolve() / "model.pt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise ValueError(f"Stage211 Nano teacher checkpoint is missing or empty: {checkpoint_path}")
    return checkpoint_path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_bound_file(
    record: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> Path:
    path = Path(str(record.get(path_key) or "")).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    expected_sha256 = str(record.get(sha256_key) or "")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def validate_stage211_full_profile_smoke_binding(
    binding: Any,
    *,
    phase: str,
    init_checkpoint: str | Path,
    easy_manifest: str | Path,
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError(f"Stage211 {phase} phase gate lacks a preflight smoke binding.")
    marker_path = _validate_bound_file(
        binding,
        path_key="marker_path",
        sha256_key="marker_sha256",
        label=f"Stage211 {phase} preflight smoke marker",
    )
    marker = _load_json_object(
        marker_path,
        label=f"Stage211 {phase} preflight smoke marker",
    )
    init_checkpoint = Path(init_checkpoint).resolve()
    easy_manifest = Path(easy_manifest).resolve()
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "full_profile_smoke",
        "phase": phase,
        "complete": True,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "easy_manifest_path": str(easy_manifest),
        "easy_manifest_sha256": sha256_file(easy_manifest),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {phase} preflight smoke marker contract mismatch.")
    checkpoint = _validate_bound_file(
        marker,
        path_key="smoke_checkpoint_path",
        sha256_key="smoke_checkpoint_sha256",
        label=f"Stage211 {phase} preflight smoke checkpoint",
    )
    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    try:
        checkpoint_step = int(checkpoint_payload.get("step", 0))
    finally:
        del checkpoint_payload
    if checkpoint.name != "step-2.pt" or checkpoint_step != 2:
        raise ValueError(f"Stage211 {phase} preflight smoke checkpoint is not step 2.")
    log_path = _validate_bound_file(
        marker,
        path_key="smoke_log_path",
        sha256_key="smoke_log_sha256",
        label=f"Stage211 {phase} preflight smoke log",
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    attempt_start = log_text.rfind("[rwkvasr] Distributed init complete.")
    if attempt_start >= 0:
        log_text = log_text[attempt_start:]
    rejected_patterns = (
        re.compile(r"Traceback"),
        re.compile(r"CUDA out of memory", re.IGNORECASE),
        re.compile(r"OutOfMemory"),
        re.compile(r"\bloss=(?:nan|inf)\b", re.IGNORECASE),
        re.compile(r"\bonline_[a-z0-9_]*missing=[1-9][0-9]*\b"),
        re.compile(r"\bonline_[a-z0-9_]*frame_delta=[1-9][0-9]*\b"),
        re.compile(r"\bdropped_tail(?:_samples)?=[1-9][0-9]*\b"),
        re.compile(r"\bskipped_samples=[1-9][0-9]*\b"),
    )
    if "[deepspeed-train] step=2" not in log_text or any(
        pattern.search(log_text) is not None for pattern in rejected_patterns
    ):
        raise ValueError(f"Stage211 {phase} preflight smoke log failed validation.")
    peak_values = [
        float(value)
        for value in re.findall(
            r"peak_reserved=([0-9]+(?:\.[0-9]+)?)GiB",
            log_text,
        )
    ]
    peak = float(marker.get("peak_reserved_gib", float("nan")))
    limit = float(marker.get("max_peak_reserved_gib", float("nan")))
    if (
        not peak_values
        or not math.isfinite(peak)
        or not math.isfinite(limit)
        or peak < 0.0
        or limit <= 0.0
        or peak > limit
        or not math.isclose(peak, max(peak_values), rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(f"Stage211 {phase} preflight smoke memory mismatch.")
    return dict(binding)


def validate_stage211_global_dedup_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str = STAGE211_GLOBAL_DEDUP_MANIFEST_SHA256,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise ValueError(f"Stage211 global dedup manifest SHA-256 mismatch: {manifest_path}")
    manifest = _load_json_object(
        manifest_path,
        label="Stage211 global dedup manifest",
    )
    expected = {
        "version": 1,
        "root": "/",
        "curriculum": "usbhd_dedup_audio_only_online_ctc_alignment",
        "split": "train",
        "dedupe_key": ("blake2b16(root_or_tar_path + audio_member + audio_offset + audio_size)"),
        "total_unique_audio_rows": STAGE211_AUDIO_TOTAL_ROWS,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 global dedup manifest contract mismatch.")
    total_hours = float(manifest.get("total_unique_hours", float("nan")))
    if not math.isfinite(total_hours) or not math.isclose(
        total_hours,
        STAGE211_GLOBAL_DEDUP_TOTAL_HOURS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Stage211 global dedup manifest total hours mismatch.")
    rows_by_difficulty = manifest.get("rows_by_difficulty")
    hours_by_difficulty = manifest.get("hours_by_difficulty")
    stages = manifest.get("stages")
    if not all(
        isinstance(value, dict) for value in (rows_by_difficulty, hours_by_difficulty, stages)
    ):
        raise ValueError("Stage211 global dedup manifest lacks difficulty partitions.")
    if set(rows_by_difficulty) != set(STAGE211_AUDIO_CURRICULUM) or set(hours_by_difficulty) != set(
        STAGE211_AUDIO_CURRICULUM
    ):
        raise ValueError("Stage211 global dedup manifest difficulty set mismatch.")
    for difficulty, expected_segment in STAGE211_AUDIO_CURRICULUM.items():
        expected_rows = int(expected_segment["rows"])
        expected_hours = float(expected_segment["hours"])
        if int(rows_by_difficulty[difficulty]) != expected_rows or not math.isclose(
            float(hours_by_difficulty[difficulty]),
            expected_hours,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Stage211 global dedup manifest {difficulty} partition mismatch.")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Stage211 global dedup manifest lacks source inputs.")
    accepted_rows = 0
    for source, raw_input in inputs.items():
        if not isinstance(raw_input, dict):
            raise ValueError(f"Stage211 global dedup input {source} is invalid.")
        if int(raw_input.get("duplicate_audio_rows", -1)) != 0:
            raise ValueError(f"Stage211 global dedup input {source} contains duplicates.")
        accepted_rows += int(raw_input.get("accepted_unique_audio_rows", -1))
    if accepted_rows != STAGE211_AUDIO_TOTAL_ROWS:
        raise ValueError("Stage211 global dedup source-row total mismatch.")
    return manifest


def _canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def _resolve_manifest_part_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _manifest_train_parts(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[tuple[Path, int]]:
    train = manifest.get("splits", {}).get("train")
    if not isinstance(train, dict) or not isinstance(train.get("buckets"), list):
        raise ValueError(f"Stage211 runtime manifest lacks train buckets: {manifest_path}")
    parts: list[tuple[Path, int]] = []
    for bucket in train["buckets"]:
        if not isinstance(bucket, dict) or not isinstance(bucket.get("parts"), list):
            raise ValueError(
                f"Stage211 runtime manifest has an invalid train bucket: {manifest_path}"
            )
        for part in bucket["parts"]:
            if not isinstance(part, dict):
                raise ValueError(
                    f"Stage211 runtime manifest has an invalid train part: {manifest_path}"
                )
            parts.append(
                (
                    _resolve_manifest_part_path(manifest_path, str(part.get("path") or "")),
                    int(part.get("num_samples", -1)),
                )
            )
    return parts


def validate_stage211_loaded_manifest_receipt(
    receipt_path: str | Path,
    *,
    expected_global_dedup_manifest: str | Path = DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST,
    expected_sha256: str | None = None,
    verify_part_sha256: bool = False,
) -> dict[str, Any]:
    receipt_path = Path(receipt_path).expanduser().resolve()
    expected_global_dedup_manifest = Path(expected_global_dedup_manifest).expanduser().resolve()
    if (
        expected_sha256 is None
        and expected_global_dedup_manifest == DEFAULT_STAGE211_GLOBAL_DEDUP_MANIFEST.resolve()
    ):
        expected_sha256 = STAGE211_LOADED_MANIFEST_RECEIPT_SHA256
    if expected_sha256 is not None and sha256_file(receipt_path) != expected_sha256:
        raise ValueError(f"Stage211 loaded-manifest receipt SHA-256 mismatch: {receipt_path}")
    receipt = _load_json_object(
        receipt_path,
        label="Stage211 loaded-manifest chain receipt",
    )
    expected_fields = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "loaded_manifest_chain",
        "complete": True,
        "part_hash_algorithm": "sha256",
        "part_hash_audit_complete": True,
        "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
    }
    if any(receipt.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Stage211 loaded-manifest receipt contract mismatch.")
    if receipt.get("dedupe_key") != (
        "blake2b16(root_or_tar_path + audio_member + audio_offset + audio_size)"
    ):
        raise ValueError("Stage211 loaded-manifest dedupe-key mismatch.")
    total_hours = float(receipt.get("total_hours", float("nan")))
    if not math.isfinite(total_hours) or not math.isclose(
        total_hours,
        STAGE211_GLOBAL_DEDUP_TOTAL_HOURS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Stage211 loaded-manifest aggregate hours mismatch.")

    global_manifest_path = (
        Path(str(receipt.get("global_dedup_manifest_path") or "")).expanduser().resolve()
    )
    if global_manifest_path != expected_global_dedup_manifest:
        raise ValueError("Stage211 loaded-manifest global-dedup path mismatch.")
    if str(receipt.get("global_dedup_manifest_sha256") or "") != sha256_file(global_manifest_path):
        raise ValueError("Stage211 loaded-manifest global-dedup SHA-256 mismatch.")
    global_manifest = validate_stage211_global_dedup_manifest(global_manifest_path)

    fixed_eval = receipt.get("fixed_eval")
    if not isinstance(fixed_eval, dict) or int(fixed_eval.get("rows", -1)) != 256:
        raise ValueError("Stage211 loaded-manifest receipt lacks the fixed eval binding.")
    fixed_eval_path = Path(str(fixed_eval.get("path") or "")).expanduser().resolve()
    if not fixed_eval_path.is_file() or str(fixed_eval.get("sha256") or "") != sha256_file(
        fixed_eval_path
    ):
        raise ValueError("Stage211 loaded-manifest fixed eval changed.")

    raw_segments = receipt.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Stage211 loaded-manifest receipt lacks segments.")
    segments = {
        str(segment.get("difficulty") or ""): segment
        for segment in raw_segments
        if isinstance(segment, dict)
    }
    if list(segments) != list(STAGE211_AUDIO_CURRICULUM):
        raise ValueError("Stage211 loaded-manifest segments are not in curriculum order.")

    total_rows = 0
    total_parts = 0
    global_stages = global_manifest.get("stages")
    if not isinstance(global_stages, dict):
        raise ValueError("Stage211 global dedup manifest lacks stage records.")
    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        segment = segments[difficulty]
        if any(
            segment.get(key) != value
            for key, value in {
                "difficulty": difficulty,
                "train_rows": int(expected["rows"]),
                "train_parts": STAGE211_AUDIO_TRAIN_PART_COUNTS[difficulty],
                "steps_per_epoch": int(expected["steps_per_epoch"]),
                "epochs": STAGE211_FULL_DATA_EPOCHS,
                "total_steps": int(expected["steps"]),
                "tail_padding_samples_per_epoch": int(expected["tail_padding_samples_per_epoch"]),
            }.items()
        ):
            raise ValueError(f"Stage211 {difficulty} loaded-manifest totals mismatch.")
        runtime_manifest_path = (
            Path(str(segment.get("runtime_manifest_path") or "")).expanduser().resolve()
        )
        source_manifest_path = (
            Path(str(segment.get("source_manifest_path") or "")).expanduser().resolve()
        )
        for path, key, label in (
            (runtime_manifest_path, "runtime_manifest_sha256", "runtime"),
            (source_manifest_path, "source_manifest_sha256", "source"),
        ):
            if not path.is_file() or str(segment.get(key) or "") != sha256_file(path):
                raise ValueError(f"Stage211 {difficulty} {label} manifest changed.")
        runtime_manifest = _load_json_object(
            runtime_manifest_path,
            label=f"Stage211 {difficulty} runtime manifest",
        )
        source_manifest = _load_json_object(
            source_manifest_path,
            label=f"Stage211 {difficulty} source manifest",
        )
        runtime_without_eval = json.loads(json.dumps(runtime_manifest))
        runtime_without_eval.get("splits", {}).pop("eval", None)
        if runtime_without_eval != source_manifest:
            raise ValueError(
                f"Stage211 {difficulty} runtime manifest changes more than the eval split."
            )
        runtime_train = runtime_manifest.get("splits", {}).get("train")
        source_train = source_manifest.get("splits", {}).get("train")
        train_split_sha256 = _canonical_json_sha256(runtime_train)
        if runtime_train != source_train or segment.get("train_split_sha256") != train_split_sha256:
            raise ValueError(f"Stage211 {difficulty} train split identity mismatch.")
        if str(runtime_manifest.get("source_length_index_path") or "") != str(
            segment.get("source_length_index_path") or ""
        ):
            raise ValueError(f"Stage211 {difficulty} length-index binding mismatch.")

        eval_parts = []
        eval_rows = 0
        eval_split = runtime_manifest.get("splits", {}).get("eval")
        if isinstance(eval_split, dict):
            eval_rows = int(eval_split.get("num_samples", -1))
            for bucket in eval_split.get("buckets", []):
                for part in bucket.get("parts", []):
                    eval_parts.append(
                        _resolve_manifest_part_path(
                            runtime_manifest_path,
                            str(part.get("path") or ""),
                        )
                    )
        if eval_rows != 256 or eval_parts != [fixed_eval_path]:
            raise ValueError(f"Stage211 {difficulty} fixed eval split mismatch.")

        runtime_parts = _manifest_train_parts(runtime_manifest_path, runtime_manifest)
        if len({path for path, _ in runtime_parts}) != len(runtime_parts):
            raise ValueError(f"Stage211 {difficulty} train part paths are not unique.")
        raw_part_records = segment.get("part_records")
        if not isinstance(raw_part_records, list) or len(raw_part_records) != len(runtime_parts):
            raise ValueError(f"Stage211 {difficulty} part receipt count mismatch.")
        part_rows = 0
        for (part_path, declared_rows), raw_record in zip(
            runtime_parts,
            raw_part_records,
            strict=True,
        ):
            if not isinstance(raw_record, dict):
                raise ValueError(f"Stage211 {difficulty} part receipt is invalid.")
            if Path(str(raw_record.get("path") or "")).expanduser().resolve() != part_path:
                raise ValueError(f"Stage211 {difficulty} train part path mismatch.")
            if int(raw_record.get("rows", -1)) != declared_rows:
                raise ValueError(f"Stage211 {difficulty} train part row count mismatch.")
            if not part_path.is_file():
                raise ValueError(f"Stage211 {difficulty} train part is missing: {part_path}")
            stat = part_path.stat()
            if (
                int(raw_record.get("size_bytes", -1)) != stat.st_size
                or int(raw_record.get("mtime_ns", -1)) != stat.st_mtime_ns
            ):
                raise ValueError(f"Stage211 {difficulty} train part metadata changed: {part_path}")
            recorded_part_sha256 = str(raw_record.get("sha256") or "")
            if len(recorded_part_sha256) != 64:
                raise ValueError(f"Stage211 {difficulty} train part digest is invalid: {part_path}")
            if verify_part_sha256 and recorded_part_sha256 != sha256_file(part_path):
                raise ValueError(f"Stage211 {difficulty} train part content changed: {part_path}")
            part_rows += declared_rows
        if part_rows != int(expected["rows"]):
            raise ValueError(f"Stage211 {difficulty} part rows do not cover the partition.")
        if segment.get("part_records_sha256") != _canonical_json_sha256(raw_part_records):
            raise ValueError(f"Stage211 {difficulty} part-record digest mismatch.")

        global_stage_name = str(segment.get("global_stage_name") or "")
        global_stage = global_stages.get(global_stage_name)
        global_source_manifest_path = (
            Path(str(segment.get("global_source_manifest_path") or "")).expanduser().resolve()
        )
        if (
            not isinstance(global_stage, dict)
            or global_stage.get("difficulty") != difficulty
            or int(global_stage.get("selected_rows", -1)) != int(expected["rows"])
            or str(global_stage.get("length_index_path") or "")
            != str(segment.get("source_length_index_path") or "")
            or str(global_stage.get("bucket_manifest_path") or "")
            != str(segment.get("global_source_manifest_path") or "")
        ):
            raise ValueError(f"Stage211 {difficulty} global stage binding mismatch.")
        train_hours = float(segment.get("train_hours", float("nan")))
        if not math.isfinite(train_hours) or not math.isclose(
            train_hours,
            float(global_stage.get("selected_hours", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Stage211 {difficulty} global hours mismatch.")
        global_source_sha256 = str(segment.get("global_source_manifest_sha256") or "")
        if len(global_source_sha256) != 64:
            raise ValueError(f"Stage211 {difficulty} global source digest is invalid.")
        if (
            global_source_manifest_path.is_file()
            and sha256_file(global_source_manifest_path) != global_source_sha256
        ):
            raise ValueError(f"Stage211 {difficulty} global source manifest changed.")
        if segment.get("repartitioned") is True:
            loaded_digest = str(segment.get("loaded_audio_key_set_sha256") or "")
            global_digest = str(segment.get("global_audio_key_set_sha256") or "")
            if len(loaded_digest) != 64 or loaded_digest != global_digest:
                raise ValueError(f"Stage211 {difficulty} repartitioned key-set proof mismatch.")
        elif segment.get("global_source_manifest_sha256") != segment.get("source_manifest_sha256"):
            raise ValueError(f"Stage211 {difficulty} source manifest identity mismatch.")
        total_rows += part_rows
        total_parts += len(runtime_parts)

    if total_rows != STAGE211_AUDIO_TOTAL_ROWS or total_parts != sum(
        STAGE211_AUDIO_TRAIN_PART_COUNTS.values()
    ):
        raise ValueError("Stage211 loaded-manifest aggregate coverage mismatch.")
    if int(receipt.get("total_train_parts", -1)) != total_parts:
        raise ValueError("Stage211 loaded-manifest aggregate part count mismatch.")
    return receipt


def _report_nano_checkpoint_path(report: dict[str, Any]) -> Path:
    checkpoint_value = report.get("model_checkpoint_path")
    if checkpoint_value is not None and str(checkpoint_value).strip():
        return Path(str(checkpoint_value)).expanduser().resolve()
    model_value = report.get("model_path")
    if model_value is None or not str(model_value).strip():
        raise ValueError("Stage211 Nano inference report lacks model_path.")
    model_path = Path(str(model_value)).expanduser().resolve()
    return model_path if model_path.name == "model.pt" else model_path / "model.pt"


def validate_stage211_public_overlap_binding(
    binding: Any,
    *,
    public_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(binding, dict):
        raise ValueError("Stage211 artifact lacks the public/train overlap binding.")
    receipt_path = _validate_bound_file(
        binding,
        path_key="receipt_path",
        sha256_key="receipt_sha256",
        label="Stage211 public/train overlap receipt",
    )
    receipt_sha256 = str(binding["receipt_sha256"])
    production_receipt = (
        receipt_path == DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT.expanduser().resolve()
    )
    if production_receipt and receipt_sha256 != STAGE211_PUBLIC_OVERLAP_RECEIPT_SHA256:
        raise ValueError("Stage211 production public/train overlap receipt changed.")

    # Imported lazily because the overlap module reuses Stage211 gate helpers.
    from rwkvasr.eval.stage211_public_overlap import (
        validate_stage211_public_overlap_receipt,
    )

    overlap = validate_stage211_public_overlap_receipt(receipt_path)
    coverage = overlap.get("coverage")
    outputs = overlap.get("outputs")
    if not isinstance(coverage, dict) or not isinstance(outputs, dict):
        raise ValueError("Stage211 public/train overlap receipt lacks coverage outputs.")
    clean_rows = int(coverage.get("clean_public_rows", -1))
    excluded_rows = int(coverage.get("excluded_public_rows", -1))
    public_rows = int(coverage.get("public_rows", -1))
    exact_rows = int(coverage.get("exact_byte_identical_training_rows", -1))
    candidate_rows = int(coverage.get("candidate_training_rows", -1))
    if (
        clean_rows != 14_922
        or excluded_rows < 0
        or public_rows != clean_rows + excluded_rows
        or exact_rows != candidate_rows
        or exact_rows < excluded_rows
    ):
        raise ValueError("Stage211 public/train overlap coverage is inconsistent.")
    if production_receipt:
        expected_coverage = {
            "candidate_training_rows": 1_474,
            "clean_public_rows": 14_922,
            "exact_byte_identical_training_rows": 1_474,
            "excluded_public_rows": 1_474,
            "public_rows": 16_396,
            "stage178_english_rows": 63_625,
        }
        if any(int(coverage.get(key, -1)) != value for key, value in expected_coverage.items()):
            raise ValueError("Stage211 public/train overlap production coverage changed.")
    clean_manifest = outputs.get("clean_manifest")
    exclusions = outputs.get("exclusions")
    if not isinstance(clean_manifest, dict) or not isinstance(exclusions, dict):
        raise ValueError("Stage211 public/train overlap receipt lacks clean/exclusion outputs.")
    if int(clean_manifest.get("rows", -1)) != 14_922:
        raise ValueError("Stage211 clean Common Voice manifest contract changed.")
    if production_receipt and (
        clean_manifest.get("sha256") != STAGE211_CLEAN_COMMONVOICE_MANIFEST_SHA256
        or int(exclusions.get("rows", -1)) != 1_474
    ):
        raise ValueError("Stage211 Common Voice exclusion coverage changed.")

    if public_benchmark is not None:
        raw_results = public_benchmark.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("Stage211 public benchmark lacks results for overlap validation.")
        commonvoice = next(
            (
                result
                for result in raw_results
                if isinstance(result, dict) and result.get("dataset") == "commonvoice_en_test"
            ),
            None,
        )
        if not isinstance(commonvoice, dict):
            raise ValueError("Stage211 public benchmark lacks Common Voice.")
        if (
            int(commonvoice.get("sample_count", -1)) != 14_922
            or commonvoice.get("manifest_sha256") != clean_manifest["sha256"]
        ):
            raise ValueError(
                "Stage211 public benchmark does not use the audited clean Common Voice subset."
            )
    return overlap


def validate_stage211_nano_public_baseline_receipt(
    receipt_path: str | Path,
    *,
    expected_receipt_sha256: str | None = None,
    expected_nano_checkpoint_sha256: str | None = None,
    public_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt_path = Path(receipt_path).expanduser().resolve()
    receipt = _load_json_object(
        receipt_path,
        label="Stage211 Nano public-baseline provenance receipt",
    )
    if expected_receipt_sha256 is not None and (
        len(expected_receipt_sha256) != 64 or sha256_file(receipt_path) != expected_receipt_sha256
    ):
        raise ValueError(f"Stage211 Nano public-baseline receipt SHA-256 mismatch: {receipt_path}")
    expected_fields = {
        "schema_version": STAGE211_NANO_PUBLIC_BASELINE_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "nano_public_baseline_provenance",
        "complete": True,
    }
    if any(receipt.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Stage211 Nano public-baseline provenance receipt is invalid.")
    provenance_mode = receipt.get("provenance_mode")
    if provenance_mode not in {
        "embedded_checkpoint_sha256",
        "legacy_report_attestation",
    }:
        raise ValueError("Stage211 Nano public-baseline provenance mode is invalid.")

    nano_checkpoint = _validate_bound_file(
        receipt,
        path_key="nano_checkpoint_path",
        sha256_key="nano_checkpoint_sha256",
        label="Stage211 Nano public-baseline checkpoint",
    )
    if nano_checkpoint.name != "model.pt":
        raise ValueError("Stage211 Nano public-baseline checkpoint must be model.pt.")
    nano_checkpoint_sha256 = str(receipt["nano_checkpoint_sha256"])
    if (
        expected_nano_checkpoint_sha256 is not None
        and nano_checkpoint_sha256 != expected_nano_checkpoint_sha256
    ):
        raise ValueError(
            "Stage211 Nano public baseline and online teacher checkpoint SHA-256 differ."
        )
    expected_total_samples = sum(
        int(expected["samples"]) for expected in STAGE211_PUBLIC_BENCHMARKS.values()
    )
    if int(receipt.get("total_samples", -1)) != expected_total_samples:
        raise ValueError("Stage211 Nano public-baseline sample total mismatch.")
    overlap = validate_stage211_public_overlap_binding(receipt.get("public_overlap"))
    clean_commonvoice = overlap["outputs"]["clean_manifest"]

    raw_results = receipt.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Stage211 Nano public-baseline results must be a list.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 Nano public-baseline dataset set is incomplete or unexpected.")

    embedded_checkpoint_reports = 0
    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        result = by_dataset[dataset]
        expected_result = {
            "language": expected["language"],
            "metric": expected["metric"],
            "sample_count": expected["samples"],
            "identical_utt_coverage": True,
            "normalized_reference_mismatch_count": 0,
        }
        if any(result.get(key) != value for key, value in expected_result.items()):
            raise ValueError(f"Stage211 Nano public-baseline metadata mismatch for {dataset}.")
        report_path = _validate_bound_file(
            result,
            path_key="report_path",
            sha256_key="report_sha256",
            label=f"Stage211 {dataset} Nano inference report",
        )
        manifest_path = _validate_bound_file(
            result,
            path_key="manifest_path",
            sha256_key="manifest_sha256",
            label=f"Stage211 {dataset} public manifest",
        )
        prediction_path = _validate_bound_file(
            result,
            path_key="nano_prediction_path",
            sha256_key="nano_prediction_sha256",
            label=f"Stage211 {dataset} Nano prediction",
        )
        if dataset == "commonvoice_en_test" and (
            result.get("manifest_sha256") != clean_commonvoice["sha256"]
            or int(result.get("sample_count", -1)) != int(clean_commonvoice["rows"])
        ):
            raise ValueError(
                "Stage211 Nano baseline does not use the audited clean Common Voice subset."
            )
        inference_report = _load_json_object(
            report_path,
            label=f"Stage211 {dataset} Nano inference report",
        )
        expected_report_fields = {
            "version": 1,
            "system": "FunASR-Nano-2512 direct CTC",
            "language": expected["language"],
            "normalization": "ctc",
            "decode": "greedy_ctc",
            "requested_limit": None,
            "sample_count": expected["samples"],
        }
        if any(inference_report.get(key) != value for key, value in expected_report_fields.items()):
            raise ValueError(f"Stage211 {dataset} Nano inference report contract mismatch.")
        if Path(str(inference_report.get("manifest_path") or "")).resolve() != manifest_path:
            raise ValueError(f"Stage211 {dataset} Nano inference report manifest path mismatch.")
        if Path(str(inference_report.get("predictions_path") or "")).resolve() != prediction_path:
            raise ValueError(f"Stage211 {dataset} Nano inference report prediction path mismatch.")
        if _report_nano_checkpoint_path(inference_report) != nano_checkpoint:
            raise ValueError(f"Stage211 {dataset} Nano inference report model path mismatch.")
        embedded_path = inference_report.get("model_checkpoint_path")
        embedded_sha256 = inference_report.get("model_checkpoint_sha256")
        report_embeds_checkpoint = embedded_path is not None and embedded_sha256 is not None
        if result.get("report_embeds_checkpoint_sha256") is not report_embeds_checkpoint:
            raise ValueError(f"Stage211 {dataset} Nano inference report identity flag mismatch.")
        if embedded_path is not None or embedded_sha256 is not None:
            if (
                Path(str(embedded_path or "")).expanduser().resolve() != nano_checkpoint
                or embedded_sha256 != nano_checkpoint_sha256
            ):
                raise ValueError(
                    f"Stage211 {dataset} Nano inference report checkpoint identity mismatch."
                )
            embedded_checkpoint_reports += 1

    if embedded_checkpoint_reports not in {
        0,
        len(STAGE211_PUBLIC_BENCHMARKS),
    }:
        raise ValueError(
            "Stage211 Nano public baseline mixes legacy and embedded checkpoint identity."
        )
    if provenance_mode == "embedded_checkpoint_sha256" and embedded_checkpoint_reports != len(
        STAGE211_PUBLIC_BENCHMARKS
    ):
        raise ValueError("Stage211 Nano public baseline lacks embedded checkpoint identity.")
    if provenance_mode == "legacy_report_attestation" and embedded_checkpoint_reports == len(
        STAGE211_PUBLIC_BENCHMARKS
    ):
        raise ValueError("Stage211 Nano public baseline incorrectly uses legacy provenance mode.")

    if public_benchmark is not None:
        benchmark_results = public_benchmark.get("results")
        if not isinstance(benchmark_results, list):
            raise ValueError("Stage211 public benchmark results must be a list.")
        benchmark_by_dataset = {
            str(result.get("dataset")): result
            for result in benchmark_results
            if isinstance(result, dict)
        }
        if set(benchmark_by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
            raise ValueError("Stage211 public benchmark dataset set is incomplete or unexpected.")
        for dataset in STAGE211_PUBLIC_BENCHMARKS:
            provenance = by_dataset[dataset]
            benchmark = benchmark_by_dataset[dataset]
            for prefix in ("manifest", "nano_prediction"):
                path_key = f"{prefix}_path"
                sha256_key = f"{prefix}_sha256"
                if Path(str(benchmark.get(path_key) or "")).resolve() != Path(
                    str(provenance.get(path_key) or "")
                ).resolve() or benchmark.get(sha256_key) != provenance.get(sha256_key):
                    raise ValueError(
                        f"Stage211 {dataset} {prefix} differs from the Nano "
                        "public-baseline provenance receipt."
                    )
    return receipt


def _validate_stage211_alignment_eval_provenance(
    provenance: Any,
    *,
    label: str,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(provenance, dict):
        raise ValueError(f"Stage211 {label} lacks fixed-eval provenance.")
    expected = {
        "schema_version": 1,
        "split": "eval",
        "requested_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
        "split_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {label} fixed-eval provenance is invalid.")
    _validate_bound_file(
        provenance,
        path_key="bucket_manifest_path",
        sha256_key="bucket_manifest_sha256",
        label=f"Stage211 {label} fixed-eval manifest",
    )
    raw_parts = provenance.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError(f"Stage211 {label} fixed-eval provenance has no parts.")
    fingerprint: list[tuple[str, int]] = []
    total_samples = 0
    for index, raw_part in enumerate(raw_parts):
        if not isinstance(raw_part, dict):
            raise ValueError(f"Stage211 {label} fixed-eval part {index} is invalid.")
        _validate_bound_file(
            raw_part,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 {label} fixed-eval part {index}",
        )
        num_samples = int(raw_part.get("num_samples", -1))
        if num_samples <= 0:
            raise ValueError(f"Stage211 {label} fixed-eval part {index} sample count is invalid.")
        total_samples += num_samples
        fingerprint.append((str(raw_part["sha256"]), num_samples))
    if total_samples != STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES:
        raise ValueError(f"Stage211 {label} fixed-eval sample count mismatch.")
    return tuple(fingerprint)


def _stage211_alignment_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Stage211 {label} is not numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"Stage211 {label} is not finite.")
    return result


def _stage211_alignment_component_layers(
    report: dict[str, Any],
    *,
    phase: str,
    role: str,
    component: str,
) -> dict[str, dict[str, Any]]:
    raw_components = report.get("layer_components") or report.get("layer_component_metrics")
    if not isinstance(raw_components, dict):
        raise ValueError(f"Stage211 {phase} {role} alignment source lacks layer components.")
    raw_layers = raw_components.get(component)
    if not isinstance(raw_layers, dict) or set(raw_layers) != _STAGE211_ALIGNMENT_LAYER_KEYS:
        raise ValueError(
            f"Stage211 {phase} {role} alignment source must contain exactly 70 {component} layers."
        )
    for layer_id, metrics in raw_layers.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"Stage211 {phase} {role} {component} layer {layer_id} is invalid.")
        for metric in ("loss", "cosine", "rms_ratio"):
            _stage211_alignment_float(
                metrics.get(metric),
                label=f"{phase} {role} {component} layer {layer_id} {metric}",
            )
    return raw_layers


def _stage211_alignment_mean(
    layers: dict[str, dict[str, Any]],
    metric: str,
    layer_ids: tuple[int, ...],
) -> float:
    return sum(float(layers[str(layer_id)][metric]) for layer_id in layer_ids) / len(layer_ids)


def _stage211_alignment_component_summary(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_loss = _stage211_alignment_mean(baseline, "loss", _STAGE211_ALIGNMENT_LAYER_IDS)
    candidate_loss = _stage211_alignment_mean(candidate, "loss", _STAGE211_ALIGNMENT_LAYER_IDS)
    summary: dict[str, Any] = {
        "baseline_mean_loss": baseline_loss,
        "candidate_mean_loss": candidate_loss,
        "relative_loss_reduction": (baseline_loss - candidate_loss) / max(baseline_loss, 1.0e-12),
        "baseline_mean_cosine": _stage211_alignment_mean(
            baseline, "cosine", _STAGE211_ALIGNMENT_LAYER_IDS
        ),
        "candidate_mean_cosine": _stage211_alignment_mean(
            candidate, "cosine", _STAGE211_ALIGNMENT_LAYER_IDS
        ),
        "baseline_mean_rms_ratio": _stage211_alignment_mean(
            baseline, "rms_ratio", _STAGE211_ALIGNMENT_LAYER_IDS
        ),
        "candidate_mean_rms_ratio": _stage211_alignment_mean(
            candidate, "rms_ratio", _STAGE211_ALIGNMENT_LAYER_IDS
        ),
        "loss_improved_layers": sum(
            float(candidate[str(layer_id)]["loss"]) < float(baseline[str(layer_id)]["loss"])
            for layer_id in _STAGE211_ALIGNMENT_LAYER_IDS
        ),
        "cosine_improved_layers": sum(
            float(candidate[str(layer_id)]["cosine"]) > float(baseline[str(layer_id)]["cosine"])
            for layer_id in _STAGE211_ALIGNMENT_LAYER_IDS
        ),
        "weak_bands": {},
    }
    for band_name, layer_ids in _STAGE211_ALIGNMENT_WEAK_BANDS.items():
        band_baseline_loss = _stage211_alignment_mean(baseline, "loss", layer_ids)
        band_candidate_loss = _stage211_alignment_mean(candidate, "loss", layer_ids)
        summary["weak_bands"][band_name] = {
            "baseline_loss": band_baseline_loss,
            "candidate_loss": band_candidate_loss,
            "relative_loss_reduction": (band_baseline_loss - band_candidate_loss)
            / max(band_baseline_loss, 1.0e-12),
            "baseline_cosine": _stage211_alignment_mean(baseline, "cosine", layer_ids),
            "candidate_cosine": _stage211_alignment_mean(candidate, "cosine", layer_ids),
        }
    return summary


def _stage211_fixed_component_gate_passed(summary: dict[str, Any]) -> bool:
    return (
        float(summary["candidate_mean_loss"]) < float(summary["baseline_mean_loss"])
        and float(summary["candidate_mean_cosine"]) > float(summary["baseline_mean_cosine"])
        and int(summary["loss_improved_layers"]) >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
        and int(summary["cosine_improved_layers"]) >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
        and all(
            float(row["candidate_loss"]) < float(row["baseline_loss"])
            and float(row["candidate_cosine"]) > float(row["baseline_cosine"])
            for row in summary["weak_bands"].values()
        )
    )


def _validate_stage211_replayed_value(
    actual: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"Stage211 {label} does not match replayed evidence.")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Stage211 {label} field coverage mismatch.")
        for key, value in expected.items():
            _validate_stage211_replayed_value(actual[key], value, label=f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Stage211 {label} sequence mismatch.")
        for index, value in enumerate(expected):
            _validate_stage211_replayed_value(actual[index], value, label=f"{label}[{index}]")
        return
    if isinstance(expected, (int, float)):
        actual_value = _stage211_alignment_float(actual, label=label)
        if isinstance(expected, int) and not isinstance(expected, bool):
            if isinstance(actual, bool) or actual_value != float(expected):
                raise ValueError(f"Stage211 {label} does not match replayed evidence.")
        elif not math.isclose(
            actual_value,
            float(expected),
            rel_tol=0.0,
            abs_tol=_STAGE211_ALIGNMENT_TOLERANCE,
        ):
            raise ValueError(f"Stage211 {label} does not match replayed evidence.")
        return
    if actual != expected:
        raise ValueError(f"Stage211 {label} does not match replayed evidence.")


def _validate_stage211_logits_metrics(
    raw_metrics: Any,
    *,
    label: str,
    expected_matched: int,
) -> dict[str, float]:
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"Stage211 {label} lacks CTC logit metrics.")
    metrics = {
        name: _stage211_alignment_float(raw_metrics.get(name), label=f"{label} logit metric {name}")
        for name in _STAGE211_LOGITS_REQUIRED_METRICS
    }
    if (
        metrics["selected_frames"] <= 0.0
        or metrics["all_frames"] <= 0.0
        or metrics["teacher_tokens"] <= 0.0
        or metrics["matched_utterances"] != float(expected_matched)
        or metrics["missing_utterances"] != 0.0
        or metrics["mean_frame_delta"] != 0.0
    ):
        raise ValueError(f"Stage211 {label} CTC logit coverage is incomplete.")
    return metrics


def _stage211_logits_metric_checks(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> tuple[dict[str, bool], float, float]:
    full_kl_reduction = (baseline["full_kl"] - candidate["full_kl"]) / max(
        abs(baseline["full_kl"]), 1.0e-12
    )
    conditional_kl_reduction = (
        baseline["conditional_nonblank_kl"] - candidate["conditional_nonblank_kl"]
    ) / max(abs(baseline["conditional_nonblank_kl"]), 1.0e-12)
    tolerance = _STAGE211_ALIGNMENT_TOLERANCE
    checks = {
        "full_kl_materially_improved": full_kl_reduction
        >= _STAGE211_LOGITS_MIN_KL_RELATIVE_REDUCTION,
        "conditional_nonblank_kl_materially_improved": conditional_kl_reduction
        >= _STAGE211_LOGITS_MIN_KL_RELATIVE_REDUCTION,
        "conditional_nonblank_hard_ce_not_worse": candidate["conditional_nonblank_hard_ce"]
        <= baseline["conditional_nonblank_hard_ce"] + tolerance,
        "blank_binary_kl_not_worse": candidate["blank_binary_kl"]
        <= baseline["blank_binary_kl"] + tolerance,
        "blank_probability_mae_not_worse": candidate["blank_prob_mae"]
        <= baseline["blank_prob_mae"] + tolerance,
        "selected_top1_improved": candidate["selected_top1_agreement"]
        > baseline["selected_top1_agreement"] + tolerance,
        "all_top1_improved": candidate["all_top1_agreement"]
        > baseline["all_top1_agreement"] + tolerance,
        "active_top1_improved": candidate["active_top1_agreement"]
        > baseline["active_top1_agreement"] + tolerance,
        "ctc_token_error_rate_improved": candidate["ctc_token_error_rate"]
        < baseline["ctc_token_error_rate"] - tolerance,
        "ctc_token_deletion_rate_not_worse": candidate["ctc_token_deletion_rate"]
        <= baseline["ctc_token_deletion_rate"] + tolerance,
        "sequence_exact_rate_not_worse": candidate["sequence_exact_rate"]
        >= baseline["sequence_exact_rate"] - tolerance,
        "nonblank_rate_ratio_in_range": _STAGE211_LOGITS_RATIO_MIN
        <= candidate["nonblank_rate_ratio"]
        <= _STAGE211_LOGITS_RATIO_MAX,
        "nonblank_rate_ratio_not_farther": abs(candidate["nonblank_rate_ratio"] - 1.0)
        <= abs(baseline["nonblank_rate_ratio"] - 1.0) + tolerance,
        "collapsed_length_ratio_in_range": _STAGE211_LOGITS_RATIO_MIN
        <= candidate["collapsed_length_ratio"]
        <= _STAGE211_LOGITS_RATIO_MAX,
        "collapsed_length_ratio_not_farther": abs(candidate["collapsed_length_ratio"] - 1.0)
        <= abs(baseline["collapsed_length_ratio"] - 1.0) + tolerance,
        "complete_exact_coverage": True,
    }
    return checks, full_kl_reduction, conditional_kl_reduction


def _validate_stage211_alignment_source_report(
    path: Path,
    *,
    phase: str,
    role: str,
    checkpoint_path: Path,
    nano_teacher_checkpoint_sha256: str,
) -> tuple[dict[str, Any], tuple[tuple[str, int], ...]]:
    report = _load_json_object(
        path,
        label=f"Stage211 {phase} {role} alignment source report",
    )
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT,
        "phase": phase,
        "role": role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "eval_samples": STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {phase} {role} alignment source binding mismatch.")
    logical_step = int(report.get("step", -1))
    checkpoint_step = int(report.get("checkpoint_step", -1))
    if role == "baseline":
        if logical_step != 0 or checkpoint_step < 0:
            raise ValueError("Stage211 alignment baseline source has invalid step metadata.")
    else:
        if logical_step <= 0 or checkpoint_step != logical_step:
            raise ValueError("Stage211 alignment candidate source has invalid step metadata.")
    pair_eval_id = str(report.get("pair_eval_id") or "")
    if len(pair_eval_id) != 64 or any(
        character not in "0123456789abcdef" for character in pair_eval_id
    ):
        raise ValueError(f"Stage211 {phase} {role} alignment source pair ID is invalid.")
    for prefix, label in (
        ("train_config", "train config"),
        ("model_config", "model config"),
        ("nano_checkpoint", "Nano checkpoint"),
    ):
        _validate_bound_file(
            report,
            path_key=f"{prefix}_path",
            sha256_key=f"{prefix}_sha256",
            label=f"Stage211 {phase} {role} alignment {label}",
        )
    train_config = load_yaml(Path(str(report["train_config_path"])).resolve())
    validate_stage211_phase_train_config(train_config, phase=phase)
    if report.get("nano_checkpoint_sha256") != nano_teacher_checkpoint_sha256:
        raise ValueError(f"Stage211 {phase} {role} alignment teacher checkpoint mismatch.")
    feature_seed = report.get("feature_seed")
    provenance = report.get("eval_provenance")
    if (
        feature_seed != 0
        or not isinstance(provenance, dict)
        or provenance.get("feature_seed") != feature_seed
    ):
        raise ValueError(f"Stage211 {phase} {role} alignment fixed feature seed mismatch.")
    fingerprint = _validate_stage211_alignment_eval_provenance(
        provenance,
        label=f"{phase} {role} alignment source",
    )
    try:
        required_components = _STAGE211_ALIGNMENT_PHASE_COMPONENTS[phase]
    except KeyError as error:
        raise ValueError(f"Unsupported Stage211 alignment phase: {phase!r}.") from error
    for component in required_components:
        _stage211_alignment_component_layers(
            report,
            phase=phase,
            role=role,
            component=component,
        )
    _stage211_alignment_float(report.get("eval_loss"), label=f"{phase} {role} fixed eval loss")
    if phase in {"block", "logits"}:
        decoder_hidden = report.get("decoder_hidden_metrics")
        if not isinstance(decoder_hidden, dict):
            raise ValueError(
                f"Stage211 {phase} {role} alignment source lacks decoder hidden metrics."
            )
        _stage211_alignment_float(
            decoder_hidden.get("loss"),
            label=f"{phase} {role} decoder hidden loss",
        )
    if phase == "logits":
        _validate_stage211_logits_metrics(
            report.get("logit_metrics"),
            label=f"{phase} {role} alignment source",
            expected_matched=STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
        )
    return report, fingerprint


def _validate_stage211_stratified_component_summary(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Stage211 {label} component summary is missing.")
    for key in (
        "baseline_mean_loss",
        "candidate_mean_loss",
        "baseline_mean_cosine",
        "candidate_mean_cosine",
    ):
        _stage211_alignment_float(value.get(key), label=f"{label} {key}")
    for key in ("loss_improved_layers", "cosine_improved_layers"):
        count = int(value.get(key, -1))
        if not 0 <= count <= len(_STAGE211_ALIGNMENT_LAYER_IDS):
            raise ValueError(f"Stage211 {label} {key} is invalid.")
    layers = value.get("layers")
    if not isinstance(layers, dict) or set(layers) != _STAGE211_ALIGNMENT_LAYER_KEYS:
        raise ValueError(f"Stage211 {label} lacks exact 70-layer macro coverage.")
    for layer_id, row in layers.items():
        if not isinstance(row, dict):
            raise ValueError(f"Stage211 {label} layer {layer_id} is invalid.")
        for key in (
            "baseline_loss",
            "candidate_loss",
            "baseline_cosine",
            "candidate_cosine",
            "baseline_rms_ratio",
            "candidate_rms_ratio",
        ):
            _stage211_alignment_float(row.get(key), label=f"{label} layer {layer_id} {key}")
    weak_bands = value.get("weak_bands")
    if not isinstance(weak_bands, dict) or set(weak_bands) != set(_STAGE211_ALIGNMENT_WEAK_BANDS):
        raise ValueError(f"Stage211 {label} weak-band coverage mismatch.")
    for band_name, row in weak_bands.items():
        if not isinstance(row, dict):
            raise ValueError(f"Stage211 {label} weak band {band_name} is invalid.")
        for key in (
            "baseline_loss",
            "candidate_loss",
            "baseline_cosine",
            "candidate_cosine",
        ):
            _stage211_alignment_float(row.get(key), label=f"{label} weak band {band_name} {key}")
    cells = value.get("cells")
    if not isinstance(cells, dict) or set(cells) != _STAGE211_ALIGNMENT_STRATIFIED_CELLS:
        raise ValueError(f"Stage211 {label} nine-cell coverage mismatch.")
    for cell_name, row in cells.items():
        if not isinstance(row, dict):
            raise ValueError(f"Stage211 {label} cell {cell_name} is invalid.")
        for key in (
            "baseline_loss",
            "candidate_loss",
            "relative_change_pct",
            "baseline_cosine",
            "candidate_cosine",
        ):
            _stage211_alignment_float(row.get(key), label=f"{label} cell {cell_name} {key}")
        for key in ("layers_loss_improved", "layers_cosine_improved"):
            count = int(row.get(key, -1))
            if not 0 <= count <= len(_STAGE211_ALIGNMENT_LAYER_IDS):
                raise ValueError(f"Stage211 {label} cell {cell_name} {key} is invalid.")
    return value


def _validate_stage211_stratified_report_bindings(
    summary: dict[str, Any],
    *,
    label: str,
) -> None:
    reports = summary.get("reports")
    if not isinstance(reports, dict) or set(reports) != _STAGE211_ALIGNMENT_STRATIFIED_CELLS:
        raise ValueError(f"Stage211 {label} report coverage mismatch.")
    for cell_name, roles in reports.items():
        if not isinstance(roles, dict) or set(roles) != {"baseline", "candidate"}:
            raise ValueError(f"Stage211 {label} report roles mismatch: {cell_name}.")
        for role, binding in roles.items():
            if not isinstance(binding, dict):
                raise ValueError(f"Stage211 {label} report binding is invalid: {cell_name}/{role}.")
            _validate_bound_file(
                binding,
                path_key="path",
                sha256_key="sha256",
                label=f"Stage211 {label} report {cell_name}/{role}",
            )


def _validate_stage211_stratified_summary_binding(
    alignment_report: dict[str, Any],
    *,
    phase: str,
    baseline_checkpoint_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    summary_path = _validate_bound_file(
        alignment_report,
        path_key="stratified_summary_path",
        sha256_key="stratified_summary_sha256",
        label=f"Stage211 {phase} stratified summary",
    )
    summary = _load_json_object(summary_path, label=f"Stage211 {phase} stratified summary")
    _validate_stage211_replayed_value(
        alignment_report.get("stratified_summary"),
        summary,
        label=f"{phase} embedded stratified summary",
    )
    expected_artifact = (
        "stratified_logits_eval_summary" if phase == "logits" else "stratified_hidden_eval_summary"
    )
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": expected_artifact,
        "phase": phase,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {phase} stratified summary binding mismatch.")
    _validate_bound_file(
        summary,
        path_key="receipt_path",
        sha256_key="receipt_sha256",
        label=f"Stage211 {phase} stratified receipt",
    )
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"baseline", "candidate"}:
        raise ValueError(f"Stage211 {phase} stratified checkpoint coverage mismatch.")
    for role, expected_path in (
        ("baseline", baseline_checkpoint_path),
        ("candidate", checkpoint_path),
    ):
        record = checkpoints[role]
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 {phase} stratified {role} checkpoint is invalid.")
        bound_path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            bound_path != expected_path
            or not bound_path.is_file()
            or record.get("sha256") != sha256_file(bound_path)
        ):
            raise ValueError(f"Stage211 {phase} stratified {role} checkpoint binding mismatch.")
    cells = summary.get("cells")
    if not isinstance(cells, dict) or set(cells) != _STAGE211_ALIGNMENT_STRATIFIED_CELLS:
        raise ValueError(f"Stage211 {phase} stratified cell coverage mismatch.")
    for cell_name, cell in cells.items():
        if not isinstance(cell, dict) or int(cell.get("samples", -1)) != (
            STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES
        ):
            raise ValueError(f"Stage211 {phase} stratified cell is invalid: {cell_name}.")
        _validate_bound_file(
            cell,
            path_key="manifest_path",
            sha256_key="manifest_sha256",
            label=f"Stage211 {phase} stratified manifest {cell_name}",
        )
    _validate_stage211_stratified_report_bindings(summary, label=f"{phase} stratified")
    required_components = _STAGE211_ALIGNMENT_PHASE_COMPONENTS[phase]
    if phase == "logits":
        for cell_name, cell in cells.items():
            for role in ("baseline", "candidate"):
                _validate_stage211_logits_metrics(
                    cell.get(f"{role}_metrics"),
                    label=f"logits stratified {cell_name}/{role}",
                    expected_matched=STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
                )
        macro = summary.get("macro")
        if (
            not isinstance(macro, dict)
            or int(macro.get("cells", -1)) != len(_STAGE211_ALIGNMENT_STRATIFIED_CELLS)
            or int(macro.get("samples", -1)) != _STAGE211_ALIGNMENT_STRATIFIED_SAMPLES
        ):
            raise ValueError("Stage211 logits stratified macro coverage mismatch.")
        for role in ("baseline", "candidate"):
            _validate_stage211_logits_metrics(
                macro.get(f"{role}_metrics"),
                label=f"logits stratified macro/{role}",
                expected_matched=_STAGE211_ALIGNMENT_STRATIFIED_SAMPLES,
            )
        component_summaries = summary.get("hidden_component_summaries")
        if not isinstance(component_summaries, dict) or set(component_summaries) != set(
            required_components
        ):
            raise ValueError("Stage211 logits stratified hidden-component coverage mismatch.")
        for component in required_components:
            _validate_stage211_stratified_component_summary(
                component_summaries[component],
                label=f"logits stratified {component}",
            )
        decoder = summary.get("decoder_hidden")
        if not isinstance(decoder, dict) or int(decoder.get("cells", -1)) != len(
            _STAGE211_ALIGNMENT_STRATIFIED_CELLS
        ):
            raise ValueError("Stage211 logits stratified decoder hidden is invalid.")
        for key in ("baseline_loss", "candidate_loss", "relative_change_pct"):
            _stage211_alignment_float(decoder.get(key), label=f"logits stratified decoder {key}")
        decoder_cells = decoder.get("cell_results")
        if (
            not isinstance(decoder_cells, dict)
            or set(decoder_cells) != _STAGE211_ALIGNMENT_STRATIFIED_CELLS
        ):
            raise ValueError("Stage211 logits stratified decoder cell coverage mismatch.")
        for cell_name, row in decoder_cells.items():
            if not isinstance(row, dict):
                raise ValueError(
                    f"Stage211 logits stratified decoder cell is invalid: {cell_name}."
                )
            for key in ("baseline_loss", "candidate_loss", "relative_change_pct"):
                _stage211_alignment_float(
                    row.get(key),
                    label=f"logits stratified decoder {cell_name} {key}",
                )
        return summary

    for cell_name, cell in cells.items():
        for key in ("baseline_loss", "candidate_loss", "relative_change_pct"):
            _stage211_alignment_float(cell.get(key), label=f"{phase} stratified {cell_name} {key}")
        for key in ("layers_loss_improved", "layers_cosine_improved"):
            count = int(cell.get(key, -1))
            if not 0 <= count <= len(_STAGE211_ALIGNMENT_LAYER_IDS):
                raise ValueError(f"Stage211 {phase} stratified {cell_name} {key} is invalid.")
    macro = summary.get("macro")
    if (
        not isinstance(macro, dict)
        or int(macro.get("cells", -1)) != len(_STAGE211_ALIGNMENT_STRATIFIED_CELLS)
        or int(macro.get("samples", -1)) != _STAGE211_ALIGNMENT_STRATIFIED_SAMPLES
    ):
        raise ValueError(f"Stage211 {phase} stratified macro coverage mismatch.")
    for key in ("baseline_loss", "candidate_loss", "relative_change_pct"):
        _stage211_alignment_float(macro.get(key), label=f"{phase} stratified macro {key}")
    component_summaries = summary.get("component_summaries")
    if not isinstance(component_summaries, dict) or set(component_summaries) != set(
        required_components
    ):
        raise ValueError(f"Stage211 {phase} stratified component coverage mismatch.")
    for component in required_components:
        _validate_stage211_stratified_component_summary(
            component_summaries[component],
            label=f"{phase} stratified {component}",
        )
    layer_summary = summary.get("layer_summary")
    if not isinstance(layer_summary, dict):
        raise ValueError(f"Stage211 {phase} stratified layer summary is missing.")
    primary = component_summaries[phase]
    expected_layer_summary = {
        key: primary[key]
        for key in (
            "loss_improved_layers",
            "cosine_improved_layers",
            "weak_bands",
            "layers",
        )
    }
    _validate_stage211_replayed_value(
        layer_summary,
        expected_layer_summary,
        label=f"{phase} stratified primary layer summary",
    )
    decoder = summary.get("decoder_hidden")
    if phase == "block":
        if not isinstance(decoder, dict) or int(decoder.get("cells", -1)) != len(
            _STAGE211_ALIGNMENT_STRATIFIED_CELLS
        ):
            raise ValueError("Stage211 block stratified decoder hidden is invalid.")
        for key in ("baseline_loss", "candidate_loss"):
            _stage211_alignment_float(decoder.get(key), label=f"block stratified decoder {key}")
    elif decoder is not None:
        raise ValueError("Stage211 mixer stratified summary unexpectedly has decoder hidden.")
    return summary


def _stage211_stratified_component_gate_passed(component: dict[str, Any]) -> bool:
    cells = component["cells"]
    return (
        float(component["candidate_mean_loss"]) < float(component["baseline_mean_loss"])
        and float(component["candidate_mean_cosine"]) > float(component["baseline_mean_cosine"])
        and int(component["loss_improved_layers"]) >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
        and int(component["cosine_improved_layers"]) >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
        and all(
            float(cells[cell_name]["candidate_loss"]) < float(cells[cell_name]["baseline_loss"])
            and int(cells[cell_name]["layers_loss_improved"])
            >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
            and int(cells[cell_name]["layers_cosine_improved"])
            >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
            for cell_name in ("hard_en", "hard_zh")
        )
        and all(
            float(cell["relative_change_pct"]) <= _STAGE211_ALIGNMENT_MAX_CELL_REGRESSION_PCT
            for cell in cells.values()
        )
        and all(
            float(row["candidate_loss"]) < float(row["baseline_loss"])
            and float(row["candidate_cosine"]) > float(row["baseline_cosine"])
            for row in component["weak_bands"].values()
        )
    )


def _stage211_hidden_stratified_gate_passed(summary: dict[str, Any], *, phase: str) -> bool:
    cells = summary["cells"]
    layer_summary = summary["layer_summary"]
    decoder_passed = True
    if phase == "block":
        decoder = summary["decoder_hidden"]
        decoder_passed = float(decoder["candidate_loss"]) < float(decoder["baseline_loss"])
    return (
        float(summary["macro"]["candidate_loss"]) < float(summary["macro"]["baseline_loss"])
        and int(layer_summary["loss_improved_layers"]) >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
        and int(layer_summary["cosine_improved_layers"]) >= _STAGE211_ALIGNMENT_MIN_IMPROVED_LAYERS
        and all(
            float(cells[cell_name]["candidate_loss"]) < float(cells[cell_name]["baseline_loss"])
            and int(cells[cell_name]["layers_loss_improved"]) == len(_STAGE211_ALIGNMENT_LAYER_IDS)
            and int(cells[cell_name]["layers_cosine_improved"])
            == len(_STAGE211_ALIGNMENT_LAYER_IDS)
            for cell_name in ("hard_en", "hard_zh")
        )
        and float(cells["long_zh"]["candidate_loss"]) < float(cells["long_zh"]["baseline_loss"])
        and all(
            float(cell["relative_change_pct"]) <= _STAGE211_ALIGNMENT_MAX_CELL_REGRESSION_PCT
            for cell in cells.values()
        )
        and all(
            float(row["candidate_loss"]) < float(row["baseline_loss"])
            and float(row["candidate_cosine"]) > float(row["baseline_cosine"])
            for row in layer_summary["weak_bands"].values()
        )
        and decoder_passed
        and all(
            _stage211_stratified_component_gate_passed(component)
            for component in summary["component_summaries"].values()
        )
    )


def _stage211_hidden_retention_checks(summary: dict[str, Any]) -> dict[str, bool]:
    tolerance = _STAGE211_ALIGNMENT_TOLERANCE
    return {
        "mean_loss_retained": float(summary["candidate_mean_loss"])
        <= float(summary["baseline_mean_loss"])
        * (1.0 + _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION)
        + tolerance,
        "mean_cosine_retained": float(summary["candidate_mean_cosine"])
        >= float(summary["baseline_mean_cosine"])
        - _STAGE211_LOGITS_MAX_HIDDEN_COSINE_REGRESSION
        - tolerance,
        "weak_band_loss_retained": all(
            float(row["candidate_loss"])
            <= float(row["baseline_loss"]) * (1.0 + _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION)
            + tolerance
            for row in summary["weak_bands"].values()
        ),
        "weak_band_cosine_retained": all(
            float(row["candidate_cosine"])
            >= float(row["baseline_cosine"])
            - _STAGE211_LOGITS_MAX_HIDDEN_COSINE_REGRESSION
            - tolerance
            for row in summary["weak_bands"].values()
        ),
    }


def _stage211_logits_stratified_gate(
    summary: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    baseline = _validate_stage211_logits_metrics(
        summary["macro"]["baseline_metrics"],
        label="logits stratified macro/baseline replay",
        expected_matched=_STAGE211_ALIGNMENT_STRATIFIED_SAMPLES,
    )
    candidate = _validate_stage211_logits_metrics(
        summary["macro"]["candidate_metrics"],
        label="logits stratified macro/candidate replay",
        expected_matched=_STAGE211_ALIGNMENT_STRATIFIED_SAMPLES,
    )
    checks, _, _ = _stage211_logits_metric_checks(baseline, candidate)
    hard_and_long_improved = all(
        float(summary["cells"][cell_name]["candidate_metrics"][metric])
        < float(summary["cells"][cell_name]["baseline_metrics"][metric])
        for cell_name in ("hard_en", "hard_zh", "long_zh")
        for metric in (
            "full_kl",
            "conditional_nonblank_kl",
            "ctc_token_error_rate",
        )
    )
    bounded_cell_token_regression = all(
        float(cell["candidate_metrics"]["ctc_token_error_rate"])
        <= float(cell["baseline_metrics"]["ctc_token_error_rate"])
        + _STAGE211_LOGITS_MAX_CELL_TOKEN_ERROR_REGRESSION
        for cell in summary["cells"].values()
    )
    component_retention: dict[str, bool] = {}
    for component_name, component in summary["hidden_component_summaries"].items():
        macro_and_weak_retained = all(_stage211_hidden_retention_checks(component).values())
        cells_retained = all(
            float(row["candidate_loss"])
            <= float(row["baseline_loss"]) * (1.0 + _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION)
            + _STAGE211_ALIGNMENT_TOLERANCE
            and float(row["candidate_cosine"])
            >= float(row["baseline_cosine"])
            - _STAGE211_LOGITS_MAX_HIDDEN_COSINE_REGRESSION
            - _STAGE211_ALIGNMENT_TOLERANCE
            for row in component["cells"].values()
        )
        component_retention[f"{component_name}_hidden_retained"] = (
            macro_and_weak_retained and cells_retained
        )
    decoder = summary["decoder_hidden"]
    decoder_retained = float(decoder["candidate_loss"]) <= float(decoder["baseline_loss"]) * (
        1.0 + _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION
    ) + _STAGE211_ALIGNMENT_TOLERANCE and all(
        float(row["candidate_loss"])
        <= float(row["baseline_loss"]) * (1.0 + _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION)
        + _STAGE211_ALIGNMENT_TOLERANCE
        for row in decoder["cell_results"].values()
    )
    replayed_checks = {
        **checks,
        "hard_and_long_cells_improved": hard_and_long_improved,
        "cell_token_error_regression_bounded": bounded_cell_token_regression,
        **component_retention,
        "decoder_hidden_retained": decoder_retained,
        "complete_stratified_coverage": True,
    }
    return all(replayed_checks.values()), replayed_checks


def _validate_stage211_alignment_gate_replay(
    alignment_report: dict[str, Any],
    *,
    phase: str,
    baseline_source: dict[str, Any],
    candidate_source: dict[str, Any],
    baseline_checkpoint_path: Path,
    checkpoint_path: Path,
) -> bool:
    required_components = _STAGE211_ALIGNMENT_PHASE_COMPONENTS[phase]
    component_summaries = {
        component: _stage211_alignment_component_summary(
            _stage211_alignment_component_layers(
                baseline_source,
                phase=phase,
                role="baseline",
                component=component,
            ),
            _stage211_alignment_component_layers(
                candidate_source,
                phase=phase,
                role="candidate",
                component=component,
            ),
        )
        for component in required_components
    }
    stratified_summary = _validate_stage211_stratified_summary_binding(
        alignment_report,
        phase=phase,
        baseline_checkpoint_path=baseline_checkpoint_path,
        checkpoint_path=checkpoint_path,
    )
    if phase in {"mixer", "block"}:
        primary = component_summaries[phase]
        component_gate_passed = all(
            _stage211_fixed_component_gate_passed(component)
            for component in component_summaries.values()
        )
        baseline_eval_loss = _stage211_alignment_float(
            baseline_source.get("eval_loss"), label=f"{phase} baseline eval loss replay"
        )
        candidate_eval_loss = _stage211_alignment_float(
            candidate_source.get("eval_loss"), label=f"{phase} candidate eval loss replay"
        )
        decoder_hidden = None
        decoder_gate_passed = True
        if phase == "block":
            baseline_decoder_loss = float(baseline_source["decoder_hidden_metrics"]["loss"])
            candidate_decoder_loss = float(candidate_source["decoder_hidden_metrics"]["loss"])
            decoder_hidden = {
                "baseline_loss": baseline_decoder_loss,
                "candidate_loss": candidate_decoder_loss,
            }
            decoder_gate_passed = candidate_decoder_loss < baseline_decoder_loss
        legacy_gate_passed = (
            candidate_eval_loss < baseline_eval_loss
            and float(primary["candidate_mean_loss"]) < float(primary["baseline_mean_loss"])
            and float(primary["candidate_mean_cosine"]) > float(primary["baseline_mean_cosine"])
            and int(primary["loss_improved_layers"]) == len(_STAGE211_ALIGNMENT_LAYER_IDS)
            and int(primary["cosine_improved_layers"]) == len(_STAGE211_ALIGNMENT_LAYER_IDS)
            and all(
                float(row["candidate_loss"]) < float(row["baseline_loss"])
                and float(row["candidate_cosine"]) > float(row["baseline_cosine"])
                for row in primary["weak_bands"].values()
            )
            and decoder_gate_passed
            and component_gate_passed
        )
        stratified_gate_passed = _stage211_hidden_stratified_gate_passed(
            stratified_summary, phase=phase
        )
        expected_gate_passed = stratified_gate_passed and (
            component_gate_passed if phase == "block" else True
        )
        replayed = {
            "legacy_gate_passed": legacy_gate_passed,
            "stratified_gate_passed": stratified_gate_passed,
            "baseline_eval_loss": baseline_eval_loss,
            "candidate_eval_loss": candidate_eval_loss,
            "layer_summary": primary,
            "component_summaries": component_summaries,
            "component_gate_passed": component_gate_passed,
            "decoder_hidden": decoder_hidden,
            "gate_passed": expected_gate_passed,
        }
        for key, value in replayed.items():
            _validate_stage211_replayed_value(
                alignment_report.get(key), value, label=f"{phase} alignment {key}"
            )
        return expected_gate_passed

    baseline_metrics = _validate_stage211_logits_metrics(
        baseline_source.get("logit_metrics"),
        label="logits baseline replay",
        expected_matched=STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    )
    candidate_metrics = _validate_stage211_logits_metrics(
        candidate_source.get("logit_metrics"),
        label="logits candidate replay",
        expected_matched=STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES,
    )
    for metric in _STAGE211_LOGITS_IDENTICAL_TEACHER_METRICS:
        if not math.isclose(
            baseline_metrics[metric],
            candidate_metrics[metric],
            rel_tol=0.0,
            abs_tol=_STAGE211_ALIGNMENT_TOLERANCE,
        ):
            raise ValueError(
                f"Stage211 logits teacher metric changed between pair reports: {metric}."
            )
    checks, full_kl_reduction, conditional_kl_reduction = _stage211_logits_metric_checks(
        baseline_metrics, candidate_metrics
    )
    hidden_retention_checks = {
        component: _stage211_hidden_retention_checks(summary)
        for component, summary in component_summaries.items()
    }
    baseline_decoder_loss = float(baseline_source["decoder_hidden_metrics"]["loss"])
    candidate_decoder_loss = float(candidate_source["decoder_hidden_metrics"]["loss"])
    decoder_retained = (
        candidate_decoder_loss
        <= baseline_decoder_loss * (1.0 + _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION)
        + _STAGE211_ALIGNMENT_TOLERANCE
    )
    decoder_hidden_retention = {
        "baseline_loss": baseline_decoder_loss,
        "candidate_loss": candidate_decoder_loss,
        "relative_change": (candidate_decoder_loss - baseline_decoder_loss)
        / max(baseline_decoder_loss, 1.0e-12),
        "retained": decoder_retained,
    }
    hidden_retention_passed = (
        all(all(component_checks.values()) for component_checks in hidden_retention_checks.values())
        and decoder_retained
    )
    legacy_gate_passed = all(checks.values())
    stratified_gate_passed, stratified_checks = _stage211_logits_stratified_gate(stratified_summary)
    selected_logits_gate_passed = stratified_gate_passed
    expected_gate_passed = selected_logits_gate_passed and hidden_retention_passed
    replayed = {
        "legacy_gate_passed": legacy_gate_passed,
        "selected_logits_gate_passed": selected_logits_gate_passed,
        "stratified_gate_passed": stratified_gate_passed,
        "stratified_checks": stratified_checks,
        "thresholds": {
            "minimum_kl_relative_reduction": _STAGE211_LOGITS_MIN_KL_RELATIVE_REDUCTION,
            "ratio_min": _STAGE211_LOGITS_RATIO_MIN,
            "ratio_max": _STAGE211_LOGITS_RATIO_MAX,
            "max_cell_token_error_regression": (_STAGE211_LOGITS_MAX_CELL_TOKEN_ERROR_REGRESSION),
            "max_hidden_loss_regression": _STAGE211_LOGITS_MAX_HIDDEN_LOSS_REGRESSION,
            "max_hidden_cosine_regression": (_STAGE211_LOGITS_MAX_HIDDEN_COSINE_REGRESSION),
            "tolerance": _STAGE211_ALIGNMENT_TOLERANCE,
        },
        "checks": checks,
        "hidden_component_summaries": component_summaries,
        "hidden_retention_checks": hidden_retention_checks,
        "hidden_retention_passed": hidden_retention_passed,
        "decoder_hidden_retention": decoder_hidden_retention,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "full_kl_relative_reduction": full_kl_reduction,
        "conditional_nonblank_kl_relative_reduction": conditional_kl_reduction,
        "gate_passed": expected_gate_passed,
    }
    for key, value in replayed.items():
        _validate_stage211_replayed_value(
            alignment_report.get(key), value, label=f"logits alignment {key}"
        )
    return expected_gate_passed


def validate_stage211_runtime_epoch_coverage(
    coverage: Any,
    *,
    epochs: int,
    steps_per_epoch: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        raise ValueError(f"Stage211 {label} lacks runtime epoch coverage.")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "runtime_epoch_coverage",
        "complete": True,
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(epochs) * int(steps_per_epoch),
    }
    if any(coverage.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {label} runtime epoch coverage is incomplete.")
    records = coverage.get("records")
    if not isinstance(records, list) or len(records) != int(epochs):
        raise ValueError(f"Stage211 {label} runtime epoch coverage record count mismatch.")
    by_epoch = {
        int(record.get("epoch", -1)): record for record in records if isinstance(record, dict)
    }
    if set(by_epoch) != set(range(1, int(epochs) + 1)):
        raise ValueError(f"Stage211 {label} runtime epoch coverage epoch set mismatch.")
    for epoch in range(1, int(epochs) + 1):
        record = by_epoch[epoch]
        if (
            int(record.get("step", -1)) != epoch * int(steps_per_epoch)
            or int(record.get("epoch_batch_offset", -1)) != 0
            or int(record.get("completed_epoch_batch_count", -1)) != int(steps_per_epoch)
        ):
            raise ValueError(f"Stage211 {label} runtime epoch {epoch} completion mismatch.")
        _validate_bound_file(
            record,
            path_key="checkpoint_path",
            sha256_key="checkpoint_sha256",
            label=f"Stage211 {label} epoch {epoch} checkpoint",
        )
    return dict(coverage)


def _validate_parameter_delta_audit(
    segment: dict[str, Any],
    *,
    phase: str,
    difficulty: str,
) -> None:
    audit = segment.get("parameter_delta_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"Stage211 {phase}/{difficulty} lacks a checkpoint parameter-delta audit.")
    expected = {
        "schema_version": 1,
        "policy": "stage211_timemixer_and_input_projection_only",
        "complete": True,
        "allowed_key_markers": list(STAGE211_ALLOWED_OPERATOR_KEY_MARKERS),
        "forbidden_changed_tensors": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} checkpoint parameter-delta audit is invalid."
        )
    initial_tensor_count = int(audit.get("initial_tensor_count", -1))
    completion_tensor_count = int(audit.get("completion_tensor_count", -1))
    allowed_changed_tensors = int(audit.get("allowed_changed_tensors", -1))
    allowed_changed_numel = int(audit.get("allowed_changed_numel", -1))
    allowed_unchanged_tensors = int(audit.get("allowed_unchanged_tensors", -1))
    frozen_unchanged_tensors = int(audit.get("frozen_unchanged_tensors", -1))
    if (
        initial_tensor_count <= 0
        or completion_tensor_count != initial_tensor_count
        or allowed_changed_tensors <= 0
        or allowed_changed_numel <= 0
        or allowed_unchanged_tensors < 0
        or frozen_unchanged_tensors <= 0
        or (
            allowed_changed_tensors + allowed_unchanged_tensors + frozen_unchanged_tensors
            != initial_tensor_count
        )
    ):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} checkpoint parameter-delta counts are invalid."
        )


def stage211_post_coverage_correction_exposure(
    corrections: list[dict[str, Any]],
) -> dict[str, int | float]:
    return {
        "rounds": len(corrections),
        "row_exposures": sum(int(row["row_exposures"]) for row in corrections),
        "hour_exposures": sum(float(row["hour_exposures"]) for row in corrections),
        "steps": sum(int(row["steps"]) for row in corrections),
        "tail_padding_sample_exposures": sum(
            int(row["tail_padding_sample_exposures"]) for row in corrections
        ),
        "executed_sample_exposures": sum(
            int(row["executed_sample_exposures"]) for row in corrections
        ),
    }


def load_stage211_post_coverage_correction_receipts(
    receipt_paths: list[Path],
    *,
    phase: str = "mixer",
) -> list[dict[str, Any]]:
    correction_lr = stage211_post_coverage_correction_lr(phase)
    if len(receipt_paths) > STAGE211_RETENTION_CORRECTION_MAX_ROUNDS:
        raise ValueError("Stage211 post-coverage correction round limit exceeded.")
    corrections: list[dict[str, Any]] = []
    for round_index, raw_path in enumerate(receipt_paths, start=1):
        path = raw_path.expanduser().resolve()
        receipt = _load_json_object(
            path,
            label=f"Stage211 retention correction round {round_index} receipt",
        )
        expected = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "post_coverage_correction",
            "phase": phase,
            "round": round_index,
            "complete": True,
            "learning_rate": correction_lr,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Invalid Stage211 {phase} correction receipt for round {round_index}: {path}"
            )
        corrections.append(
            {
                **receipt,
                "receipt_path": str(path),
                "receipt_sha256": sha256_file(path),
            }
        )
    return corrections


def build_stage211_full_data_coverage(
    *,
    phase: str,
    segments: list[dict[str, Any]],
    checkpoint_path: Path,
    post_coverage_corrections: list[dict[str, Any]] | None = None,
    supplemental_segment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    corrections = list(post_coverage_corrections or [])
    supplemental_rows = int(supplemental_segment.get("rows", 0)) if supplemental_segment else 0
    supplemental_hours = (
        float(supplemental_segment.get("hours", 0.0)) if supplemental_segment else 0.0
    )
    supplemental_row_exposures = (
        int(supplemental_segment.get("row_exposures", 0)) if supplemental_segment else 0
    )
    supplemental_hour_exposures = (
        float(supplemental_segment.get("hour_exposures", 0.0)) if supplemental_segment else 0.0
    )
    supplemental_tail_exposures = (
        int(supplemental_segment.get("tail_padding_sample_exposures", 0))
        if supplemental_segment
        else 0
    )
    supplemental_executed_exposures = (
        int(supplemental_segment.get("executed_sample_exposures", 0)) if supplemental_segment else 0
    )
    coverage: dict[str, Any] = {
        "phase": phase,
        "complete": True,
        "original_total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
        "original_total_hours": STAGE211_AUDIO_TOTAL_HOURS,
        "original_total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
        "original_total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
        "original_total_tail_padding_sample_exposures": (
            STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
        ),
        "original_total_executed_sample_exposures": (
            STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
        ),
        "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS + supplemental_rows,
        "total_hours": STAGE211_AUDIO_TOTAL_HOURS + supplemental_hours,
        "total_row_exposures": (STAGE211_AUDIO_TOTAL_ROW_EXPOSURES + supplemental_row_exposures),
        "total_hour_exposures": (STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES + supplemental_hour_exposures),
        "total_tail_padding_sample_exposures": (
            STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES + supplemental_tail_exposures
        ),
        "total_executed_sample_exposures": (
            STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES + supplemental_executed_exposures
        ),
        "segments": segments,
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
    }
    if supplemental_segment is not None:
        coverage["supplemental_natural"] = supplemental_segment
    if corrections:
        coverage["post_coverage_corrections"] = corrections
        coverage["post_coverage_correction_exposure"] = stage211_post_coverage_correction_exposure(
            corrections
        )
    return coverage


def _validate_stage211_retention_replay_binding(
    correction: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    replay_receipt_path = _validate_bound_file(
        correction,
        path_key="replay_receipt_path",
        sha256_key="replay_receipt_sha256",
        label="Stage211 retention replay receipt",
    )
    replay = _load_json_object(
        replay_receipt_path,
        label="Stage211 retention replay receipt",
    )
    expected = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "retention_replay_manifest",
        "unique_keys": replay.get("samples"),
    }
    if any(replay.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 retention replay receipt contract mismatch.")
    if int(replay.get("samples", -1)) != int(correction.get("rows", -2)):
        raise ValueError("Stage211 correction and replay row counts differ.")
    replay_hours = float(replay.get("total_hours", float("nan")))
    correction_hours = float(correction.get("hours", float("nan")))
    if (
        not math.isfinite(replay_hours)
        or not math.isfinite(correction_hours)
        or abs(replay_hours - correction_hours) > 1e-9
    ):
        raise ValueError("Stage211 correction and replay hour counts differ.")
    manifest_path = _validate_bound_file(
        replay,
        path_key="manifest_path",
        sha256_key="manifest_sha256",
        label="Stage211 retention replay bucket manifest",
    )
    if Path(
        str(correction.get("bucket_manifest_path") or "")
    ).resolve() != manifest_path or correction.get("bucket_manifest_sha256") != replay.get(
        "manifest_sha256"
    ):
        raise ValueError("Stage211 correction replay manifest binding mismatch.")
    builder = replay.get("builder")
    if not isinstance(builder, dict):
        raise ValueError("Stage211 retention replay lacks a builder binding.")
    _validate_bound_file(
        builder,
        path_key="path",
        sha256_key="sha256",
        label="Stage211 retention replay builder",
    )
    _validate_bound_file(
        replay,
        path_key="capacity_preflight_path",
        sha256_key="capacity_preflight_sha256",
        label="Stage211 retention replay capacity preflight",
    )
    for key, label in (
        ("base_replay_receipt", "base replay receipt"),
        ("stratified_hidden_eval", "nine-cell stratified receipt"),
    ):
        record = replay.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 retention replay {label} binding is invalid.")
        _validate_bound_file(
            record,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 retention replay {label}",
        )
    supplemental = replay.get("supplemental_inputs")
    if not isinstance(supplemental, dict):
        raise ValueError("Stage211 retention replay Supplemental binding is invalid.")
    for path_key, sha256_key, label in (
        ("inventory_path", "inventory_sha256", "Supplemental inventory"),
        ("profile_receipt_path", "profile_receipt_sha256", "Supplemental profile"),
        ("manifest_path", "manifest_sha256", "Supplemental source manifest"),
    ):
        _validate_bound_file(
            supplemental,
            path_key=path_key,
            sha256_key=sha256_key,
            label=f"Stage211 retention replay {label}",
        )
    source_manifest = replay.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("Stage211 retention replay source-manifest binding is invalid.")
    _validate_bound_file(
        source_manifest,
        path_key="path",
        sha256_key="sha256",
        label="Stage211 retention replay Supplemental source manifest",
    )
    output_parts = replay.get("supplemental_output_parts")
    if not isinstance(output_parts, list) or not output_parts:
        raise ValueError("Stage211 retention replay Supplemental output coverage is empty.")
    for index, record in enumerate(output_parts):
        if not isinstance(record, dict):
            raise ValueError("Stage211 retention replay output-part binding is invalid.")
        _validate_bound_file(
            record,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 retention replay output part {index}",
        )
    return replay, manifest_path


def _validate_stage211_correction_train_config(
    correction: dict[str, Any],
    *,
    manifest_path: Path,
    init_checkpoint: Path,
    nano_teacher_checkpoint: Path,
) -> None:
    phase = str(correction.get("phase") or "")
    train_config_path = _validate_bound_file(
        correction,
        path_key="train_config_path",
        sha256_key="train_config_sha256",
        label=f"Stage211 {phase} correction train config",
    )
    config = load_yaml(train_config_path)
    configured_phase = config.get("stage211_post_coverage_correction_phase", "mixer")
    if configured_phase != phase:
        raise ValueError(
            f"Stage211 {phase} correction train config phase mismatch: actual={configured_phase!r}"
        )
    contract = stage211_phase_train_config_contract(phase)
    contract["lr"] = stage211_post_coverage_correction_lr(phase)
    for key, expected in contract.items():
        actual = config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Stage211 {phase} correction train config contract mismatch: "
                f"key={key} actual={actual!r} expected={expected!r}"
            )
    expected_fields = {
        "max_steps": int(correction["steps_per_epoch"]),
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "batch_token_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "webdataset_bucket_manifest_path": str(manifest_path),
        "webdataset_split": "train",
        "stage211_post_coverage_correction_round": int(correction["round"]),
        "stage211_post_coverage_replay_receipt_path": str(
            Path(str(correction["replay_receipt_path"])).resolve()
        ),
        "stage211_post_coverage_admission_gate_path": str(
            Path(str(correction["admission_gate_path"])).resolve()
        ),
        "stage211_post_coverage_original_coverage_unchanged": True,
        "stage211_post_coverage_smoke_marker_path": str(
            Path(str(correction["smoke_marker_path"])).resolve()
        ),
        "stage211_post_coverage_smoke_marker_sha256": str(correction["smoke_marker_sha256"]),
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Stage211 {phase} correction train config mismatch: "
                f"key={key} actual={config.get(key)!r} expected={expected!r}"
            )
    init_path = config.get("init_checkpoint_path")
    resume_from = config.get("resume_from")
    if not (
        (init_path == str(init_checkpoint) and resume_from is None)
        or (init_path is None and resume_from == "latest")
    ):
        raise ValueError(
            f"Stage211 {phase} correction train config does not bind its initial "
            "checkpoint or an in-run latest resume."
        )
    if resolve_stage211_nano_teacher_checkpoint(config) != nano_teacher_checkpoint:
        raise ValueError(f"Stage211 {phase} correction train config uses another Nano teacher.")


def _validate_stage211_correction_smoke_marker(
    correction: dict[str, Any],
    *,
    replay_manifest: Path,
    init_checkpoint: Path,
    admission_gate: Path,
    nano_teacher_checkpoint: Path,
) -> None:
    phase = str(correction.get("phase") or "")
    marker_path = _validate_bound_file(
        correction,
        path_key="smoke_marker_path",
        sha256_key="smoke_marker_sha256",
        label=(f"Stage211 {phase} correction round {int(correction['round'])} smoke marker"),
    )
    marker = _load_json_object(
        marker_path,
        label=f"Stage211 {phase} correction smoke marker",
    )
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "full_profile_smoke",
        "phase": phase,
        "complete": True,
        "correction_round": int(correction["round"]),
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "easy_manifest_path": str(replay_manifest),
        "easy_manifest_sha256": sha256_file(replay_manifest),
        "replay_receipt_path": str(Path(str(correction["replay_receipt_path"])).resolve()),
        "replay_receipt_sha256": str(correction["replay_receipt_sha256"]),
        "admission_gate_path": str(admission_gate),
        "admission_gate_sha256": sha256_file(admission_gate),
        "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_teacher_checkpoint),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 {phase} correction smoke marker mismatch.")
    for path_key, sha_key in (
        ("smoke_checkpoint_path", "smoke_checkpoint_sha256"),
        ("smoke_log_path", "smoke_log_sha256"),
    ):
        _validate_bound_file(
            marker,
            path_key=path_key,
            sha256_key=sha_key,
            label=f"Stage211 {phase} correction smoke artifact",
        )
    peak = float(marker.get("peak_reserved_gib", float("nan")))
    limit = float(marker.get("max_peak_reserved_gib", float("nan")))
    if (
        not math.isfinite(peak)
        or not math.isfinite(limit)
        or peak < 0.0
        or limit <= 0.0
        or peak > limit
    ):
        raise ValueError(f"Stage211 {phase} correction smoke memory mismatch.")


def _validate_stage211_post_coverage_corrections(
    corrections: Any,
    *,
    coverage: dict[str, Any],
    original_segments: list[dict[str, Any]],
    supplemental_segment: dict[str, Any],
    original_completion_sha256: str,
    nano_teacher_checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    phase = str(coverage.get("phase") or "")
    correction_lr = stage211_post_coverage_correction_lr(phase)
    if corrections is None:
        corrections = []
    if not isinstance(corrections, list):
        raise ValueError("Stage211 post-coverage corrections must be a list.")
    if len(corrections) > STAGE211_RETENTION_CORRECTION_MAX_ROUNDS:
        raise ValueError("Stage211 post-coverage correction round limit exceeded.")

    previous_checkpoint_sha256 = original_completion_sha256
    replay_receipt_sha256: str | None = None
    validated: list[dict[str, Any]] = []
    for round_index, correction in enumerate(corrections, start=1):
        if not isinstance(correction, dict):
            raise ValueError("Stage211 post-coverage correction record must be an object.")
        expected_fields = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "post_coverage_correction",
            "phase": phase,
            "round": round_index,
            "complete": True,
            "epochs": STAGE211_RETENTION_CORRECTION_EPOCHS,
            "learning_rate": correction_lr,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
            "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
        }
        if any(correction.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"Stage211 {phase} correction round {round_index} contract mismatch.")
        rows = int(correction.get("rows", -1))
        steps_per_epoch = int(correction.get("steps_per_epoch", -1))
        tail_padding = int(correction.get("tail_padding_samples_per_epoch", -1))
        hours = float(correction.get("hours", float("nan")))
        if (
            rows <= 0
            or steps_per_epoch <= 0
            or tail_padding < 0
            or not math.isfinite(hours)
            or hours <= 0.0
            or int(correction.get("row_exposures", -1)) != rows
            or int(correction.get("steps", -1)) != steps_per_epoch
            or int(correction.get("tail_padding_sample_exposures", -1)) != tail_padding
            or int(correction.get("executed_sample_exposures", -1)) != rows + tail_padding
            or abs(float(correction.get("hour_exposures", float("nan"))) - hours) > 1e-9
        ):
            raise ValueError(f"Stage211 {phase} correction round {round_index} exposure mismatch.")
        receipt_path = _validate_bound_file(
            correction,
            path_key="receipt_path",
            sha256_key="receipt_sha256",
            label=f"Stage211 {phase} correction round {round_index} receipt",
        )
        receipt = _load_json_object(
            receipt_path,
            label=f"Stage211 {phase} correction round {round_index} receipt",
        )
        embedded_receipt = {
            key: value
            for key, value in correction.items()
            if key not in {"receipt_path", "receipt_sha256"}
        }
        if receipt != embedded_receipt:
            raise ValueError(
                f"Stage211 {phase} correction round {round_index} differs from its receipt."
            )
        validate_stage211_runtime_epoch_coverage(
            correction.get("runtime_epoch_coverage"),
            epochs=STAGE211_RETENTION_CORRECTION_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            label=f"{phase}/correction-round-{round_index}",
        )
        _validate_parameter_delta_audit(
            correction,
            phase=phase,
            difficulty=f"correction-round-{round_index}",
        )
        replay, replay_manifest_path = _validate_stage211_retention_replay_binding(correction)
        current_replay_sha256 = str(correction.get("replay_receipt_sha256") or "")
        if replay_receipt_sha256 is not None and current_replay_sha256 != replay_receipt_sha256:
            raise ValueError(f"Stage211 {phase} correction rounds use different replay data.")
        replay_receipt_sha256 = current_replay_sha256

        init_checkpoint = _validate_bound_file(
            correction,
            path_key="init_checkpoint_path",
            sha256_key="init_checkpoint_sha256",
            label=f"Stage211 {phase} correction round {round_index} initial checkpoint",
        )
        if correction.get("init_checkpoint_sha256") != previous_checkpoint_sha256:
            raise ValueError(
                f"Stage211 {phase} correction round {round_index} does not initialize "
                "from the preceding checkpoint."
            )
        _validate_bound_file(
            correction,
            path_key="completion_checkpoint_path",
            sha256_key="completion_checkpoint_sha256",
            label=f"Stage211 {phase} correction round {round_index} completion checkpoint",
        )
        nano_teacher_checkpoint = _validate_bound_file(
            correction,
            path_key="nano_teacher_checkpoint_path",
            sha256_key="nano_teacher_checkpoint_sha256",
            label=f"Stage211 {phase} correction round {round_index} Nano teacher",
        )
        if correction.get("nano_teacher_checkpoint_sha256") != nano_teacher_checkpoint_sha256:
            raise ValueError(
                f"Stage211 {phase} correction round {round_index} uses another Nano teacher."
            )
        _validate_bound_file(
            correction,
            path_key="provenance_path",
            sha256_key="provenance_sha256",
            label=f"Stage211 {phase} correction round {round_index} provenance",
        )
        _validate_stage211_correction_train_config(
            correction,
            manifest_path=replay_manifest_path,
            init_checkpoint=init_checkpoint,
            nano_teacher_checkpoint=nano_teacher_checkpoint,
        )

        admission_gate_path = _validate_bound_file(
            correction,
            path_key="admission_gate_path",
            sha256_key="admission_gate_sha256",
            label=f"Stage211 {phase} correction round {round_index} admission gate",
        )
        admission_gate = validate_stage211_phase_gate_report(
            admission_gate_path,
            expected_phase=phase,
            checkpoint_path=init_checkpoint,
            require_passed=False,
        )
        if admission_gate.get("gate_passed") is not False:
            raise ValueError(
                f"Stage211 {phase} correction round {round_index} admission gate passed."
            )
        admission_coverage = admission_gate.get("full_data_coverage")
        if not isinstance(admission_coverage, dict):
            raise ValueError(f"Stage211 {phase} correction admission coverage is missing.")
        if admission_coverage.get("segments") != original_segments:
            raise ValueError(f"Stage211 {phase} correction rewrites original coverage segments.")
        if admission_coverage.get("supplemental_natural") != supplemental_segment:
            raise ValueError(f"Stage211 {phase} correction rewrites supplemental coverage.")
        prior_corrections = admission_coverage.get("post_coverage_corrections", [])
        if prior_corrections != validated:
            raise ValueError(
                f"Stage211 {phase} correction round {round_index} admission chain mismatch."
            )
        _validate_stage211_correction_smoke_marker(
            correction,
            replay_manifest=replay_manifest_path,
            init_checkpoint=init_checkpoint,
            admission_gate=admission_gate_path,
            nano_teacher_checkpoint=nano_teacher_checkpoint,
        )
        if int(replay.get("samples", -1)) != rows:
            raise ValueError(f"Stage211 {phase} correction replay sample count mismatch.")
        previous_checkpoint_sha256 = str(correction["completion_checkpoint_sha256"])
        validated.append(dict(correction))

    aggregate = coverage.get("post_coverage_correction_exposure")
    if validated:
        expected_aggregate = stage211_post_coverage_correction_exposure(validated)
        if not isinstance(aggregate, dict):
            raise ValueError("Stage211 correction exposure summary is missing.")
        for key, expected in expected_aggregate.items():
            actual = aggregate.get(key)
            try:
                if isinstance(expected, float):
                    actual_number: int | float = float(actual)
                else:
                    actual_number = int(actual)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Stage211 correction exposure summary contains a non-numeric value."
                ) from error
            if isinstance(expected, float):
                if (
                    not math.isfinite(float(actual_number))
                    or abs(float(actual_number) - expected) > 1e-6
                ):
                    raise ValueError("Stage211 correction hour-exposure summary mismatch.")
            elif int(actual_number) != expected:
                raise ValueError("Stage211 correction exposure summary mismatch.")
    elif aggregate is not None:
        raise ValueError("Stage211 empty correction chain must not report correction exposure.")
    return validated, previous_checkpoint_sha256


def _validate_stage211_supplemental_coverage_segment(
    segment: Any,
    *,
    phase: str,
    preceding_checkpoint_sha256: str,
    nano_teacher_checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(segment, dict):
        raise ValueError("Stage211 full-data coverage lacks supplemental_natural.")
    expected_fields = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "phase": phase,
        "difficulty": STAGE211_SUPPLEMENTAL_DIFFICULTY,
        "complete": True,
        "full_data_profile": True,
        "epochs": STAGE211_FULL_DATA_EPOCHS,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
    }
    if any(segment.get(key) != value for key, value in expected_fields.items()):
        raise ValueError(f"Stage211 {phase}/supplemental_natural coverage is incomplete.")
    inventory_path = _validate_bound_file(
        segment,
        path_key="supplemental_inventory_path",
        sha256_key="supplemental_inventory_sha256",
        label=f"Stage211 {phase} supplemental inventory",
    )
    profile = stage211_supplemental_profile(
        inventory_path,
        epochs=STAGE211_FULL_DATA_EPOCHS,
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        require_training_ready=True,
        verify_part_sha256=False,
    )
    expected_numbers = {
        "rows": int(profile["rows"]),
        "row_exposures": int(profile["row_exposures"]),
        "steps_per_epoch": int(profile["steps_per_epoch"]),
        "steps": int(profile["steps"]),
        "tail_padding_samples_per_epoch": int(profile["tail_padding_samples_per_epoch"]),
        "tail_padding_sample_exposures": int(profile["tail_padding_sample_exposures"]),
        "executed_sample_exposures": int(profile["executed_sample_exposures"]),
    }
    if any(int(segment.get(key, -1)) != value for key, value in expected_numbers.items()):
        raise ValueError(f"Stage211 {phase}/supplemental_natural exposure mismatch.")
    for key in ("hours", "hour_exposures"):
        actual = float(segment.get(key, float("nan")))
        expected = float(profile[key])
        if not math.isfinite(actual) or not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=0.005,
        ):
            raise ValueError(f"Stage211 {phase}/supplemental_natural {key} mismatch.")
    validate_stage211_runtime_epoch_coverage(
        segment.get("runtime_epoch_coverage"),
        epochs=STAGE211_FULL_DATA_EPOCHS,
        steps_per_epoch=int(profile["steps_per_epoch"]),
        label=f"{phase}/supplemental_natural",
    )
    _validate_parameter_delta_audit(
        segment,
        phase=phase,
        difficulty=STAGE211_SUPPLEMENTAL_DIFFICULTY,
    )
    receipt_path = _validate_bound_file(
        segment,
        path_key="receipt_path",
        sha256_key="receipt_sha256",
        label=f"Stage211 {phase} supplemental coverage receipt",
    )
    receipt = _load_json_object(
        receipt_path,
        label=f"Stage211 {phase} supplemental coverage receipt",
    )
    embedded = {
        key: value
        for key, value in segment.items()
        if key not in {"receipt_path", "receipt_sha256"}
    }
    if receipt != embedded:
        raise ValueError("Stage211 supplemental segment differs from its immutable receipt.")
    for path_key, sha_key, label in (
        ("provenance_path", "provenance_sha256", "provenance"),
        ("train_config_path", "train_config_sha256", "train config"),
        ("bucket_manifest_path", "bucket_manifest_sha256", "bucket manifest"),
        ("init_checkpoint_path", "init_checkpoint_sha256", "initial checkpoint"),
        (
            "completion_checkpoint_path",
            "completion_checkpoint_sha256",
            "completion checkpoint",
        ),
        (
            "nano_teacher_checkpoint_path",
            "nano_teacher_checkpoint_sha256",
            "Nano teacher checkpoint",
        ),
    ):
        _validate_bound_file(
            segment,
            path_key=path_key,
            sha256_key=sha_key,
            label=f"Stage211 {phase} supplemental {label}",
        )
    if Path(str(segment["bucket_manifest_path"])).resolve() != Path(
        str(profile["bucket_manifest_path"])
    ).resolve() or str(segment["bucket_manifest_sha256"]) != str(profile["bucket_manifest_sha256"]):
        raise ValueError("Stage211 supplemental segment does not bind its inventory manifest.")
    if str(segment.get("init_checkpoint_sha256") or "") != preceding_checkpoint_sha256:
        raise ValueError("Stage211 supplemental segment does not initialize from Long.")
    if str(segment.get("nano_teacher_checkpoint_sha256") or "") != nano_teacher_checkpoint_sha256:
        raise ValueError("Stage211 supplemental segment uses another Nano teacher.")
    train_config_path = Path(str(segment["train_config_path"])).resolve()
    train_config = load_yaml(train_config_path)
    validate_stage211_phase_train_config(train_config, phase=phase)
    if int(train_config.get("max_steps", -1)) != int(profile["steps"]):
        raise ValueError("Stage211 supplemental train config step contract mismatch.")
    if (
        Path(str(train_config.get("webdataset_bucket_manifest_path") or "")).resolve()
        != Path(str(profile["bucket_manifest_path"])).resolve()
    ):
        raise ValueError("Stage211 supplemental train config manifest mismatch.")
    configured_teacher = resolve_stage211_nano_teacher_checkpoint(train_config)
    if sha256_file(configured_teacher) != nano_teacher_checkpoint_sha256:
        raise ValueError("Stage211 supplemental train config Nano teacher mismatch.")
    return dict(segment), profile


def validate_stage211_full_data_coverage(
    coverage: Any,
    *,
    phase: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 phase gate lacks a full_data_coverage object.")
    if coverage.get("complete") is not True:
        raise ValueError("Stage211 full-data coverage is not complete.")
    if coverage.get("phase") != phase:
        raise ValueError(
            "Stage211 full-data coverage phase mismatch: "
            f"expected={phase!r} actual={coverage.get('phase')!r}"
        )
    expected_original_totals: dict[str, int | float] = {
        "original_total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
        "original_total_hours": STAGE211_AUDIO_TOTAL_HOURS,
        "original_total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
        "original_total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
        "original_total_tail_padding_sample_exposures": (
            STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
        ),
        "original_total_executed_sample_exposures": (
            STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
        ),
    }
    for key, expected in expected_original_totals.items():
        actual = coverage.get(key)
        try:
            actual_number: int | float = (
                float(actual) if isinstance(expected, float) else int(actual)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Stage211 original full-data coverage {key} mismatch.") from error
        if isinstance(expected, float):
            if not math.isclose(float(actual_number), expected, rel_tol=0.0, abs_tol=0.005):
                raise ValueError(f"Stage211 original full-data coverage {key} mismatch.")
        elif int(actual_number) != expected:
            raise ValueError(f"Stage211 original full-data coverage {key} mismatch.")

    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 full-data coverage segments must be a list.")
    expected_difficulties = tuple(STAGE211_AUDIO_CURRICULUM)
    actual_difficulties = tuple(
        str(segment.get("difficulty")) if isinstance(segment, dict) else "" for segment in segments
    )
    if actual_difficulties != expected_difficulties:
        raise ValueError(
            "Stage211 full-data coverage must contain exactly ordered "
            "easy, medium, hard, and long segments."
        )
    by_difficulty = {
        str(segment.get("difficulty")): segment for segment in segments if isinstance(segment, dict)
    }

    previous_checkpoint_sha256: str | None = None
    nano_teacher_checkpoint_sha256: str | None = None
    for difficulty, expected in STAGE211_AUDIO_CURRICULUM.items():
        segment = by_difficulty[difficulty]
        expected_fields = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "curriculum_coverage",
            "phase": phase,
            "difficulty": difficulty,
            "complete": True,
            "full_data_profile": True,
            "epochs": STAGE211_FULL_DATA_EPOCHS,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
            "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
        }
        if any(segment.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"Stage211 {phase}/{difficulty} coverage is incomplete.")
        _validate_parameter_delta_audit(
            segment,
            phase=phase,
            difficulty=difficulty,
        )
        validate_stage211_runtime_epoch_coverage(
            segment.get("runtime_epoch_coverage"),
            epochs=STAGE211_FULL_DATA_EPOCHS,
            steps_per_epoch=int(expected["steps_per_epoch"]),
            label=f"{phase}/{difficulty}",
        )
        if int(segment.get("rows", -1)) != int(expected["rows"]):
            raise ValueError(f"Stage211 {phase}/{difficulty} row count mismatch.")
        if int(segment.get("row_exposures", -1)) != (
            int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} row exposures mismatch.")
        if int(segment.get("steps_per_epoch", -1)) != int(expected["steps_per_epoch"]):
            raise ValueError(f"Stage211 {phase}/{difficulty} per-epoch steps mismatch.")
        if int(segment.get("steps", -1)) != int(expected["steps"]):
            raise ValueError(f"Stage211 {phase}/{difficulty} step count mismatch.")
        tail_padding_samples_per_epoch = int(expected["tail_padding_samples_per_epoch"])
        if int(segment.get("tail_padding_samples_per_epoch", -1)) != tail_padding_samples_per_epoch:
            raise ValueError(f"Stage211 {phase}/{difficulty} tail-padding count mismatch.")
        if int(segment.get("tail_padding_sample_exposures", -1)) != (
            tail_padding_samples_per_epoch * STAGE211_FULL_DATA_EPOCHS
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} tail-padding exposure mismatch.")
        if int(segment.get("executed_sample_exposures", -1)) != (
            int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS
            + tail_padding_samples_per_epoch * STAGE211_FULL_DATA_EPOCHS
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} executed-sample exposure mismatch.")
        hours = float(segment.get("hours", float("nan")))
        if not math.isfinite(hours) or abs(hours - float(expected["hours"])) > 0.002:
            raise ValueError(f"Stage211 {phase}/{difficulty} hour count mismatch.")
        hour_exposures = float(segment.get("hour_exposures", float("nan")))
        expected_hour_exposures = float(expected["hours"]) * STAGE211_FULL_DATA_EPOCHS
        if (
            not math.isfinite(hour_exposures)
            or abs(hour_exposures - expected_hour_exposures) > 0.005
        ):
            raise ValueError(f"Stage211 {phase}/{difficulty} hour exposures mismatch.")
        receipt_path = _validate_bound_file(
            segment,
            path_key="receipt_path",
            sha256_key="receipt_sha256",
            label=f"Stage211 {phase}/{difficulty} coverage receipt",
        )
        receipt = _load_json_object(
            receipt_path,
            label=f"Stage211 {phase}/{difficulty} coverage receipt",
        )
        if segment.get("runtime_epoch_coverage") != receipt.get("runtime_epoch_coverage"):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} runtime epoch coverage differs "
                "from its coverage receipt."
            )
        _validate_bound_file(
            segment,
            path_key="provenance_path",
            sha256_key="provenance_sha256",
            label=f"Stage211 {phase}/{difficulty} provenance",
        )
        train_config_path = _validate_bound_file(
            segment,
            path_key="train_config_path",
            sha256_key="train_config_sha256",
            label=f"Stage211 {phase}/{difficulty} train config",
        )
        _validate_bound_file(
            segment,
            path_key="bucket_manifest_path",
            sha256_key="bucket_manifest_sha256",
            label=f"Stage211 {phase}/{difficulty} bucket manifest",
        )
        _validate_bound_file(
            segment,
            path_key="init_checkpoint_path",
            sha256_key="init_checkpoint_sha256",
            label=f"Stage211 {phase}/{difficulty} initial checkpoint",
        )
        _validate_bound_file(
            segment,
            path_key="completion_checkpoint_path",
            sha256_key="completion_checkpoint_sha256",
            label=f"Stage211 {phase}/{difficulty} completion checkpoint",
        )
        teacher_checkpoint = _validate_bound_file(
            segment,
            path_key="nano_teacher_checkpoint_path",
            sha256_key="nano_teacher_checkpoint_sha256",
            label=f"Stage211 {phase}/{difficulty} Nano teacher checkpoint",
        )
        if teacher_checkpoint.name != "model.pt":
            raise ValueError(
                f"Stage211 {phase}/{difficulty} Nano teacher checkpoint must be model.pt."
            )
        for key in (
            "nano_teacher_checkpoint_path",
            "nano_teacher_checkpoint_sha256",
        ):
            if segment.get(key) != receipt.get(key):
                raise ValueError(
                    f"Stage211 {phase}/{difficulty} Nano teacher binding differs "
                    "from its coverage receipt."
                )
        train_config = load_yaml(train_config_path)
        validate_stage211_phase_train_config(train_config, phase=phase)
        configured_teacher_checkpoint = resolve_stage211_nano_teacher_checkpoint(train_config)
        if configured_teacher_checkpoint != teacher_checkpoint:
            raise ValueError(
                f"Stage211 {phase}/{difficulty} Nano teacher checkpoint differs "
                "from its train config."
            )
        segment_teacher_sha256 = str(segment["nano_teacher_checkpoint_sha256"])
        if (
            nano_teacher_checkpoint_sha256 is not None
            and segment_teacher_sha256 != nano_teacher_checkpoint_sha256
        ):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} uses a different Nano teacher checkpoint."
            )
        nano_teacher_checkpoint_sha256 = segment_teacher_sha256
        init_sha256 = str(segment.get("init_checkpoint_sha256") or "")
        if previous_checkpoint_sha256 is not None and init_sha256 != previous_checkpoint_sha256:
            raise ValueError(
                f"Stage211 {phase}/{difficulty} does not initialize from the preceding "
                "curriculum checkpoint."
            )
        previous_checkpoint_sha256 = str(segment["completion_checkpoint_sha256"])

    supplemental_segment, supplemental_profile = _validate_stage211_supplemental_coverage_segment(
        coverage.get("supplemental_natural"),
        phase=phase,
        preceding_checkpoint_sha256=str(previous_checkpoint_sha256 or ""),
        nano_teacher_checkpoint_sha256=str(nano_teacher_checkpoint_sha256 or ""),
    )
    previous_checkpoint_sha256 = str(supplemental_segment["completion_checkpoint_sha256"])
    expected_combined_totals: dict[str, int | float] = {
        "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS + int(supplemental_profile["rows"]),
        "total_hours": STAGE211_AUDIO_TOTAL_HOURS + float(supplemental_profile["hours"]),
        "total_row_exposures": (
            STAGE211_AUDIO_TOTAL_ROW_EXPOSURES + int(supplemental_profile["row_exposures"])
        ),
        "total_hour_exposures": (
            STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES + float(supplemental_profile["hour_exposures"])
        ),
        "total_tail_padding_sample_exposures": (
            STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
            + int(supplemental_profile["tail_padding_sample_exposures"])
        ),
        "total_executed_sample_exposures": (
            STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
            + int(supplemental_profile["executed_sample_exposures"])
        ),
    }
    for key, expected in expected_combined_totals.items():
        actual = coverage.get(key)
        try:
            actual_number = float(actual) if isinstance(expected, float) else int(actual)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Stage211 combined full-data coverage {key} mismatch.") from error
        if isinstance(expected, float):
            if not math.isclose(float(actual_number), expected, rel_tol=0.0, abs_tol=0.005):
                raise ValueError(f"Stage211 combined full-data coverage {key} mismatch.")
        elif int(actual_number) != expected:
            raise ValueError(f"Stage211 combined full-data coverage {key} mismatch.")

    corrections, previous_checkpoint_sha256 = _validate_stage211_post_coverage_corrections(
        coverage.get("post_coverage_corrections", []),
        coverage=coverage,
        original_segments=[by_difficulty[name] for name in STAGE211_AUDIO_CURRICULUM],
        supplemental_segment=supplemental_segment,
        original_completion_sha256=str(previous_checkpoint_sha256 or ""),
        nano_teacher_checkpoint_sha256=str(nano_teacher_checkpoint_sha256 or ""),
    )
    final_checkpoint = Path(str(coverage.get("final_checkpoint_path") or "")).resolve()
    if final_checkpoint != checkpoint_path.resolve():
        raise ValueError(
            "Stage211 full-data final checkpoint path does not match the selected checkpoint."
        )
    if str(coverage.get("final_checkpoint_sha256") or "") != sha256_file(checkpoint_path):
        raise ValueError(
            "Stage211 full-data final checkpoint SHA-256 does not match the selected checkpoint."
        )
    if previous_checkpoint_sha256 != str(coverage["final_checkpoint_sha256"]):
        raise ValueError(
            "Stage211 latest curriculum/correction checkpoint is not the phase final checkpoint."
        )
    return dict(coverage)


def validate_stage211_public_benchmark(
    public_benchmark: Any,
    *,
    require_metric_source_recomputed: bool = False,
) -> dict[str, Any]:
    if not isinstance(public_benchmark, dict):
        raise ValueError("Stage211 phase gate lacks a public_benchmark object.")
    if public_benchmark.get("decode") != "greedy_ctc":
        raise ValueError("Stage211 public benchmark must use direct greedy CTC.")
    if public_benchmark.get("normalization") != "ctc":
        raise ValueError("Stage211 public benchmark must use ctc normalization.")
    if public_benchmark.get("all_datasets_complete") is not True:
        raise ValueError("Stage211 public benchmark is not complete.")

    results = public_benchmark.get("results")
    if not isinstance(results, list):
        raise ValueError("Stage211 public benchmark results must be a list.")
    by_dataset = {
        str(result.get("dataset")): result for result in results if isinstance(result, dict)
    }
    if set(by_dataset) != set(STAGE211_PUBLIC_BENCHMARKS):
        raise ValueError("Stage211 public benchmark dataset set is incomplete or unexpected.")

    for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items():
        result = by_dataset[dataset]
        if result.get("language") != expected["language"]:
            raise ValueError(f"Stage211 public benchmark language mismatch for {dataset}.")
        if result.get("metric") != expected["metric"]:
            raise ValueError(f"Stage211 public benchmark metric mismatch for {dataset}.")
        if int(result.get("sample_count", -1)) != int(expected["samples"]):
            raise ValueError(f"Stage211 public benchmark sample count mismatch for {dataset}.")
        if (
            result.get("identical_utt_coverage") is not True
            or int(result.get("normalized_reference_mismatch_count", -1)) != 0
        ):
            raise ValueError(f"Stage211 public benchmark coverage mismatch for {dataset}.")
        if require_metric_source_recomputed and result.get("metric_source_recomputed") is not True:
            raise ValueError(
                f"Stage211 public benchmark metrics were not source-recomputed for {dataset}."
            )
        for prefix in ("manifest", "nano_prediction", "student_prediction"):
            _validate_bound_file(
                result,
                path_key=f"{prefix}_path",
                sha256_key=f"{prefix}_sha256",
                label=f"Stage211 {dataset} {prefix.replace('_', ' ')}",
            )
        for metric_name in (
            "nano_error_rate",
            "student_error_rate",
            "absolute_gap_points",
            "relative_ratio",
            "nano_prediction_reference_unit_ratio",
            "student_prediction_reference_unit_ratio",
            "nano_deletion_rate",
            "student_deletion_rate",
        ):
            value = float(result.get(metric_name, float("nan")))
            if not math.isfinite(value):
                raise ValueError(
                    f"Stage211 public benchmark {dataset}/{metric_name} must be finite."
                )
    return dict(public_benchmark)


def validate_stage211_phase_gate_report(
    gate_report_path: str | Path,
    *,
    expected_phase: str,
    checkpoint_path: str | Path,
    require_passed: bool = True,
) -> dict[str, Any]:
    gate_report_path = Path(gate_report_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    report = _load_json_object(gate_report_path, label="Stage211 phase gate report")
    expected_fields = {
        "schema_version": STAGE211_PHASE_GATE_SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": "phase_gate",
        "phase": expected_phase,
    }
    for key, expected in expected_fields.items():
        if report.get(key) != expected:
            raise ValueError(
                f"Stage211 phase gate {key} mismatch: "
                f"expected={expected!r} actual={report.get(key)!r}"
            )
    gate_passed = report.get("gate_passed")
    if not isinstance(gate_passed, bool):
        raise ValueError("Stage211 phase gate report lacks a boolean decision.")
    if require_passed and not gate_passed:
        raise ValueError("Stage211 phase gate report does not record a passing decision.")
    recorded_checkpoint = Path(str(report.get("checkpoint_path") or "")).resolve()
    if recorded_checkpoint != checkpoint_path:
        raise ValueError("Stage211 phase gate checkpoint path mismatch.")
    if str(report.get("checkpoint_sha256") or "") != sha256_file(checkpoint_path):
        raise ValueError("Stage211 phase gate checkpoint SHA-256 mismatch.")

    coverage = validate_stage211_full_data_coverage(
        report.get("full_data_coverage"),
        phase=expected_phase,
        checkpoint_path=checkpoint_path,
    )
    global_dedup_manifest = _validate_bound_file(
        report,
        path_key="global_dedup_manifest_path",
        sha256_key="global_dedup_manifest_sha256",
        label="Stage211 global dedup manifest",
    )
    validate_stage211_global_dedup_manifest(global_dedup_manifest)
    loaded_manifest_receipt_path = _validate_bound_file(
        report,
        path_key="loaded_manifest_receipt_path",
        sha256_key="loaded_manifest_receipt_sha256",
        label="Stage211 loaded-manifest chain receipt",
    )
    loaded_manifest_receipt = validate_stage211_loaded_manifest_receipt(
        loaded_manifest_receipt_path,
        expected_global_dedup_manifest=global_dedup_manifest,
    )
    loaded_segments = {
        str(segment.get("difficulty") or ""): segment
        for segment in loaded_manifest_receipt["segments"]
    }
    phase_segments = coverage.get("segments")
    if not isinstance(phase_segments, list):
        raise ValueError("Stage211 phase gate lacks ordered curriculum segments.")
    easy_segment = next(
        (
            segment
            for segment in phase_segments
            if isinstance(segment, dict) and str(segment.get("difficulty") or "") == "easy"
        ),
        None,
    )
    if easy_segment is None:
        raise ValueError("Stage211 phase gate lacks the easy initialization segment.")
    for phase_segment in phase_segments:
        if not isinstance(phase_segment, dict):
            raise ValueError("Stage211 phase gate has an invalid curriculum segment.")
        difficulty = str(phase_segment.get("difficulty") or "")
        loaded_segment = loaded_segments.get(difficulty)
        if not isinstance(loaded_segment, dict):
            raise ValueError(f"Stage211 phase gate lacks {difficulty} loader provenance.")
        if Path(
            str(phase_segment.get("bucket_manifest_path") or "")
        ).expanduser().resolve() != Path(
            str(loaded_segment.get("runtime_manifest_path") or "")
        ).expanduser().resolve() or str(phase_segment.get("bucket_manifest_sha256") or "") != str(
            loaded_segment.get("runtime_manifest_sha256") or ""
        ):
            raise ValueError(
                f"Stage211 {difficulty} phase gate does not bind the audited runtime manifest."
            )
    phase_init_checkpoint = Path(str(easy_segment.get("init_checkpoint_path") or "")).resolve()
    if not phase_init_checkpoint.is_file() or sha256_file(
        phase_init_checkpoint
    ) != easy_segment.get("init_checkpoint_sha256"):
        raise ValueError("Stage211 phase initialization checkpoint is missing or changed.")
    validate_stage211_full_profile_smoke_binding(
        report.get("preflight_smoke"),
        phase=expected_phase,
        init_checkpoint=phase_init_checkpoint,
        easy_manifest=Path(str(easy_segment.get("bucket_manifest_path") or "")).resolve(),
    )
    benchmark = validate_stage211_public_benchmark(
        report.get("public_benchmark"),
        require_metric_source_recomputed=True,
    )
    benchmark_results = {str(result["dataset"]): result for result in benchmark["results"]}
    manifest_paths = {
        dataset: Path(str(result["manifest_path"])).expanduser().resolve()
        for dataset, result in benchmark_results.items()
    }
    public_comparison_path = _validate_bound_file(
        report,
        path_key="public_comparison_report_path",
        sha256_key="public_comparison_report_sha256",
        label=f"Stage211 {expected_phase} public comparison report",
    )
    public_comparison_source = _load_json_object(
        public_comparison_path,
        label=f"Stage211 {expected_phase} public comparison report",
    )
    replayed_benchmark = replay_stage211_public_comparison(
        public_comparison_source,
        manifest_paths=manifest_paths,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_checkpoint=checkpoint_path,
    )
    _validate_stage211_replayed_value(
        benchmark,
        replayed_benchmark,
        label=f"{expected_phase} public WER/CER benchmark",
    )
    replayed_public_progress_gate_passed = True
    if expected_phase in {"mixer", "block"}:
        baseline_public_record = report.get("baseline_public_comparison_report")
        if not isinstance(baseline_public_record, dict):
            raise ValueError(
                f"Stage211 {expected_phase} phase gate lacks its baseline public report."
            )
        baseline_public_path = _validate_bound_file(
            baseline_public_record,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 {expected_phase} baseline public comparison report",
        )
        baseline_public_source = _load_json_object(
            baseline_public_path,
            label=f"Stage211 {expected_phase} baseline public comparison report",
        )
        replayed_baseline_benchmark = replay_stage211_public_comparison(
            baseline_public_source,
            manifest_paths=manifest_paths,
            benchmarks=STAGE211_PUBLIC_BENCHMARKS,
            expected_checkpoint=phase_init_checkpoint,
        )
        replayed_public_progress = build_stage211_public_progress(
            baseline=replayed_baseline_benchmark,
            candidate=replayed_benchmark,
            benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        )
        _validate_stage211_replayed_value(
            report.get("public_progress"),
            replayed_public_progress,
            label=f"{expected_phase} public progress gate",
        )
        replayed_public_progress_gate_passed = bool(replayed_public_progress["gate_passed"])
    elif (
        report.get("baseline_public_comparison_report") is not None
        or report.get("public_progress") is not None
    ):
        raise ValueError(
            "Stage211 logits phase gate unexpectedly contains a baseline progress gate."
        )
    validate_stage211_public_overlap_binding(
        report.get("public_overlap"),
        public_benchmark=benchmark,
    )
    teacher_sha256_values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "")
        for segment in coverage["segments"]
        if isinstance(segment, dict)
    }
    if len(teacher_sha256_values) != 1:
        raise ValueError("Stage211 phase gate does not bind one Nano teacher checkpoint SHA-256.")
    nano_teacher_checkpoint_sha256 = next(iter(teacher_sha256_values))
    baseline_receipt_path = _validate_bound_file(
        report,
        path_key="nano_public_baseline_receipt_path",
        sha256_key="nano_public_baseline_receipt_sha256",
        label="Stage211 Nano public-baseline provenance receipt",
    )
    baseline_receipt = validate_stage211_nano_public_baseline_receipt(
        baseline_receipt_path,
        expected_receipt_sha256=str(report["nano_public_baseline_receipt_sha256"]),
        expected_nano_checkpoint_sha256=nano_teacher_checkpoint_sha256,
        public_benchmark=benchmark,
    )
    if report.get("nano_public_baseline_checkpoint_sha256") != baseline_receipt.get(
        "nano_checkpoint_sha256"
    ):
        raise ValueError("Stage211 phase gate Nano public-baseline checkpoint binding mismatch.")
    alignment_record = report.get("alignment_report")
    if not isinstance(alignment_record, dict):
        raise ValueError("Stage211 phase gate lacks an alignment report binding.")
    alignment_path = _validate_bound_file(
        alignment_record,
        path_key="path",
        sha256_key="sha256",
        label=f"Stage211 {expected_phase} alignment report",
    )
    alignment_report = _load_json_object(
        alignment_path,
        label=f"Stage211 {expected_phase} alignment report",
    )
    expected_alignment_artifact = (
        "logits_alignment_gate" if expected_phase == "logits" else "hidden_alignment_gate"
    )
    alignment_gate_passed = report.get("alignment_gate_passed")
    if not isinstance(alignment_gate_passed, bool):
        raise ValueError("Stage211 phase gate lacks a boolean alignment decision.")
    expected_alignment_fields = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": expected_alignment_artifact,
        "phase": expected_phase,
        "gate_passed": alignment_gate_passed,
        "baseline_checkpoint_path": str(phase_init_checkpoint),
        "baseline_checkpoint_sha256": sha256_file(phase_init_checkpoint),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    if any(alignment_report.get(key) != value for key, value in expected_alignment_fields.items()):
        raise ValueError(f"Stage211 {expected_phase} alignment report contract mismatch.")
    alignment_source_paths: dict[str, Path] = {}
    for prefix in ("baseline_report", "candidate_report"):
        alignment_source_paths[prefix] = _validate_bound_file(
            alignment_report,
            path_key=f"{prefix}_path",
            sha256_key=f"{prefix}_sha256",
            label=(f"Stage211 {expected_phase} alignment {prefix.replace('_', ' ')}"),
        )
    baseline_source, baseline_source_fingerprint = _validate_stage211_alignment_source_report(
        alignment_source_paths["baseline_report"],
        phase=expected_phase,
        role="baseline",
        checkpoint_path=phase_init_checkpoint,
        nano_teacher_checkpoint_sha256=nano_teacher_checkpoint_sha256,
    )
    candidate_source, candidate_source_fingerprint = _validate_stage211_alignment_source_report(
        alignment_source_paths["candidate_report"],
        phase=expected_phase,
        role="candidate",
        checkpoint_path=checkpoint_path,
        nano_teacher_checkpoint_sha256=nano_teacher_checkpoint_sha256,
    )
    pair_shared_fields = (
        "pair_eval_id",
        "train_config_path",
        "train_config_sha256",
        "model_config_path",
        "model_config_sha256",
        "nano_checkpoint_path",
        "nano_checkpoint_sha256",
        "feature_seed",
    )
    if any(baseline_source.get(key) != candidate_source.get(key) for key in pair_shared_fields):
        raise ValueError(f"Stage211 {expected_phase} alignment sources are not one eval pair.")
    if Path(str(baseline_source.get("train_config_path") or "")).resolve() != Path(
        str(easy_segment.get("train_config_path") or "")
    ).resolve() or baseline_source.get("train_config_sha256") != easy_segment.get(
        "train_config_sha256"
    ):
        raise ValueError(f"Stage211 {expected_phase} alignment pair train config mismatch.")
    if baseline_source.get("eval_provenance") != candidate_source.get("eval_provenance"):
        raise ValueError(
            f"Stage211 {expected_phase} alignment sources use different fixed-eval provenance."
        )
    baseline_fingerprint = _validate_stage211_alignment_eval_provenance(
        alignment_report.get("baseline_eval_provenance"),
        label=f"{expected_phase} alignment baseline",
    )
    candidate_fingerprint = _validate_stage211_alignment_eval_provenance(
        alignment_report.get("candidate_eval_provenance"),
        label=f"{expected_phase} alignment candidate",
    )
    if baseline_fingerprint != candidate_fingerprint:
        raise ValueError(f"Stage211 {expected_phase} alignment reports use different eval samples.")
    if (
        baseline_source_fingerprint != baseline_fingerprint
        or candidate_source_fingerprint != candidate_fingerprint
        or baseline_source.get("eval_provenance")
        != alignment_report.get("baseline_eval_provenance")
        or candidate_source.get("eval_provenance")
        != alignment_report.get("candidate_eval_provenance")
    ):
        raise ValueError(f"Stage211 {expected_phase} alignment source provenance mismatch.")
    if expected_phase == "logits":
        baseline_feature_seed = alignment_report["baseline_eval_provenance"].get("feature_seed")
        candidate_feature_seed = alignment_report["candidate_eval_provenance"].get("feature_seed")
        if (
            not isinstance(baseline_feature_seed, int)
            or baseline_feature_seed < 0
            or candidate_feature_seed != baseline_feature_seed
        ):
            raise ValueError(
                "Stage211 logits alignment reports require the same "
                "non-negative fixed feature seed."
            )
    replayed_alignment_gate_passed = _validate_stage211_alignment_gate_replay(
        alignment_report,
        phase=expected_phase,
        baseline_source=baseline_source,
        candidate_source=candidate_source,
        baseline_checkpoint_path=phase_init_checkpoint,
        checkpoint_path=checkpoint_path,
    )
    if replayed_alignment_gate_passed != alignment_gate_passed:
        raise ValueError(
            f"Stage211 {expected_phase} alignment decision does not match replayed evidence."
        )
    if alignment_record.get("artifact") != expected_alignment_artifact:
        raise ValueError(f"Stage211 {expected_phase} alignment report binding mismatch.")
    public_progress_gate_passed = report.get("public_progress_gate_passed")
    if not isinstance(public_progress_gate_passed, bool):
        raise ValueError("Stage211 phase gate lacks a boolean public-progress decision.")
    if public_progress_gate_passed != replayed_public_progress_gate_passed:
        raise ValueError(
            f"Stage211 {expected_phase} public-progress decision does not match "
            "replayed WER/CER evidence."
        )
    if expected_phase in {"mixer", "block"}:
        expected_gate_passed = alignment_gate_passed and public_progress_gate_passed
        if require_passed and not public_progress_gate_passed:
            raise ValueError(f"Stage211 {expected_phase} public progress gate did not pass.")
    elif expected_phase == "logits":
        datasets_passed = benchmark.get("all_datasets_pass") is True
        expected_gate_passed = alignment_gate_passed and datasets_passed
        if require_passed and not datasets_passed:
            raise ValueError("Stage211 logits phase must pass the every-dataset Nano WER/CER gate.")
    else:
        raise ValueError(f"Stage211 phase {expected_phase!r} cannot promote.")
    if gate_passed != expected_gate_passed:
        raise ValueError("Stage211 phase gate decision is inconsistent with its sub-gates.")
    return report
