from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml


STAGE211_PHASE_GATE_SCHEMA_VERSION = 1
STAGE211_NANO_PUBLIC_BASELINE_SCHEMA_VERSION = 1
STAGE211_FULL_DATA_EPOCHS = 3
STAGE211_FULL_DATA_BATCH_SIZE = 36
STAGE211_FULL_DATA_WORLD_SIZE = 4
STAGE211_FULL_DATA_FRAME_BUDGET = 24_000
STAGE211_RETENTION_CORRECTION_MAX_ROUNDS = 3
STAGE211_RETENTION_CORRECTION_EPOCHS = 1
STAGE211_RETENTION_CORRECTION_LR = 1.0e-6
STAGE211_FIXED_ALIGNMENT_EVAL_SAMPLES = 256
STAGE211_ALIGNMENT_CHECKPOINT_EVAL_ARTIFACT = "alignment_checkpoint_eval"
STAGE211_ALLOWED_OPERATOR_KEY_MARKERS = (".time_mixer.", ".input_proj.")
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
    "commonvoice_en_test": {"language": "en", "metric": "wer", "samples": 16_396},
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
    "blank_id": 60_515,
    "tokenizer_type": "sensevoice_tiktoken",
    "tokenizer_model_path": "assets/fun-asr-nano-2512/multilingual.tiktoken",
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
    "ctc_teacher_online_layer_ffn_loss_weight": 0.0,
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
    "ctc_teacher_online_project_ignored_token_ids": [60_514],
    "ctc_teacher_online_top_k": 32,
    "funasr_nano_ctc_teacher_blank_id": 60_514,
}
_STAGE211_PHASE_TRAIN_CONFIG_OVERRIDES: dict[str, dict[str, Any]] = {
    "mixer": {
        "lr": 3.0e-6,
        "allow_missing_targets": True,
        "ctc_loss_weight": 0.0,
        "ctc_suppress_non_pronunciation_tokens": False,
        "ctc_teacher_online_layer_input_mode": "teacher_forced",
        "ctc_teacher_online_layer_mixer_loss_weight": 1.0,
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


def stage211_phase_train_config_contract(phase: str) -> dict[str, Any]:
    try:
        phase_values = _STAGE211_PHASE_TRAIN_CONFIG_OVERRIDES[phase]
    except KeyError as error:
        raise ValueError(f"Unsupported Stage211 phase objective: {phase!r}") from error
    contract = {**_STAGE211_COMMON_PHASE_TRAIN_CONFIG, **phase_values}
    return {
        key: list(value) if isinstance(value, list) else value for key, value in contract.items()
    }


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


def _report_nano_checkpoint_path(report: dict[str, Any]) -> Path:
    checkpoint_value = report.get("model_checkpoint_path")
    if checkpoint_value is not None and str(checkpoint_value).strip():
        return Path(str(checkpoint_value)).expanduser().resolve()
    model_value = report.get("model_path")
    if model_value is None or not str(model_value).strip():
        raise ValueError("Stage211 Nano inference report lacks model_path.")
    model_path = Path(str(model_value)).expanduser().resolve()
    return model_path if model_path.name == "model.pt" else model_path / "model.pt"


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
    return report, fingerprint


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
) -> list[dict[str, Any]]:
    if len(receipt_paths) > STAGE211_RETENTION_CORRECTION_MAX_ROUNDS:
        raise ValueError("Stage211 retention correction round limit exceeded.")
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
            "phase": "mixer",
            "round": round_index,
            "complete": True,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"Invalid Stage211 retention correction receipt for round {round_index}: {path}"
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
) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    corrections = list(post_coverage_corrections or [])
    coverage: dict[str, Any] = {
        "phase": phase,
        "complete": True,
        "total_unique_rows": STAGE211_AUDIO_TOTAL_ROWS,
        "total_hours": STAGE211_AUDIO_TOTAL_HOURS,
        "total_row_exposures": STAGE211_AUDIO_TOTAL_ROW_EXPOSURES,
        "total_hour_exposures": STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES,
        "total_tail_padding_sample_exposures": (STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES),
        "total_executed_sample_exposures": (STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES),
        "segments": segments,
        "final_checkpoint_path": str(checkpoint_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
    }
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
        "schema_version": 1,
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
    source_manifests = replay.get("source_manifests")
    if not isinstance(source_manifests, dict) or set(source_manifests) != {
        "easy",
        "medium",
        "hard",
        "long",
    }:
        raise ValueError("Stage211 retention replay source coverage mismatch.")
    for difficulty, record in source_manifests.items():
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 retention replay {difficulty} binding is invalid.")
        _validate_bound_file(
            record,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 retention replay {difficulty} source manifest",
        )
    exclusions = replay.get("exclusions")
    if not isinstance(exclusions, list) or len(exclusions) != 2:
        raise ValueError("Stage211 retention replay exclusions are incomplete.")
    for index, record in enumerate(exclusions):
        if not isinstance(record, dict):
            raise ValueError("Stage211 retention replay exclusion binding is invalid.")
        _validate_bound_file(
            record,
            path_key="path",
            sha256_key="sha256",
            label=f"Stage211 retention replay exclusion {index}",
        )
    output_parts = replay.get("output_parts")
    if not isinstance(output_parts, list) or not output_parts:
        raise ValueError("Stage211 retention replay output-part coverage is empty.")
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
    train_config_path = _validate_bound_file(
        correction,
        path_key="train_config_path",
        sha256_key="train_config_sha256",
        label="Stage211 retention correction train config",
    )
    config = load_yaml(train_config_path)
    contract = stage211_phase_train_config_contract("mixer")
    contract["lr"] = STAGE211_RETENTION_CORRECTION_LR
    for key, expected in contract.items():
        actual = config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "Stage211 retention correction train config contract mismatch: "
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
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise ValueError(
                "Stage211 retention correction train config mismatch: "
                f"key={key} actual={config.get(key)!r} expected={expected!r}"
            )
    init_path = config.get("init_checkpoint_path")
    resume_from = config.get("resume_from")
    if not (
        (init_path == str(init_checkpoint) and resume_from is None)
        or (init_path is None and resume_from == "latest")
    ):
        raise ValueError(
            "Stage211 retention correction train config does not bind its initial "
            "checkpoint or an in-run latest resume."
        )
    if resolve_stage211_nano_teacher_checkpoint(config) != nano_teacher_checkpoint:
        raise ValueError("Stage211 retention correction train config uses another Nano teacher.")


def _validate_stage211_post_coverage_corrections(
    corrections: Any,
    *,
    coverage: dict[str, Any],
    original_segments: list[dict[str, Any]],
    original_completion_sha256: str,
    nano_teacher_checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    if corrections is None:
        corrections = []
    if not isinstance(corrections, list):
        raise ValueError("Stage211 post-coverage corrections must be a list.")
    if len(corrections) > STAGE211_RETENTION_CORRECTION_MAX_ROUNDS:
        raise ValueError("Stage211 retention correction round limit exceeded.")
    if corrections and coverage.get("phase") != "mixer":
        raise ValueError("Stage211 post-coverage correction is valid only for Mixer.")

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
            "phase": "mixer",
            "round": round_index,
            "complete": True,
            "epochs": STAGE211_RETENTION_CORRECTION_EPOCHS,
            "learning_rate": STAGE211_RETENTION_CORRECTION_LR,
            "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
            "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
            "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
        }
        if any(correction.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(
                f"Stage211 retention correction round {round_index} contract mismatch."
            )
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
            raise ValueError(
                f"Stage211 retention correction round {round_index} exposure mismatch."
            )
        receipt_path = _validate_bound_file(
            correction,
            path_key="receipt_path",
            sha256_key="receipt_sha256",
            label=f"Stage211 retention correction round {round_index} receipt",
        )
        receipt = _load_json_object(
            receipt_path,
            label=f"Stage211 retention correction round {round_index} receipt",
        )
        embedded_receipt = {
            key: value
            for key, value in correction.items()
            if key not in {"receipt_path", "receipt_sha256"}
        }
        if receipt != embedded_receipt:
            raise ValueError(
                f"Stage211 retention correction round {round_index} differs from its receipt."
            )
        validate_stage211_runtime_epoch_coverage(
            correction.get("runtime_epoch_coverage"),
            epochs=STAGE211_RETENTION_CORRECTION_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            label=f"mixer/retention-round-{round_index}",
        )
        _validate_parameter_delta_audit(
            correction,
            phase="mixer",
            difficulty=f"retention-round-{round_index}",
        )
        replay, replay_manifest_path = _validate_stage211_retention_replay_binding(correction)
        current_replay_sha256 = str(correction.get("replay_receipt_sha256") or "")
        if replay_receipt_sha256 is not None and current_replay_sha256 != replay_receipt_sha256:
            raise ValueError("Stage211 retention correction rounds use different replay data.")
        replay_receipt_sha256 = current_replay_sha256

        init_checkpoint = _validate_bound_file(
            correction,
            path_key="init_checkpoint_path",
            sha256_key="init_checkpoint_sha256",
            label=f"Stage211 retention correction round {round_index} initial checkpoint",
        )
        if correction.get("init_checkpoint_sha256") != previous_checkpoint_sha256:
            raise ValueError(
                f"Stage211 retention correction round {round_index} does not initialize "
                "from the preceding checkpoint."
            )
        _validate_bound_file(
            correction,
            path_key="completion_checkpoint_path",
            sha256_key="completion_checkpoint_sha256",
            label=f"Stage211 retention correction round {round_index} completion checkpoint",
        )
        nano_teacher_checkpoint = _validate_bound_file(
            correction,
            path_key="nano_teacher_checkpoint_path",
            sha256_key="nano_teacher_checkpoint_sha256",
            label=f"Stage211 retention correction round {round_index} Nano teacher",
        )
        if correction.get("nano_teacher_checkpoint_sha256") != nano_teacher_checkpoint_sha256:
            raise ValueError(
                f"Stage211 retention correction round {round_index} uses another Nano teacher."
            )
        _validate_bound_file(
            correction,
            path_key="provenance_path",
            sha256_key="provenance_sha256",
            label=f"Stage211 retention correction round {round_index} provenance",
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
            label=f"Stage211 retention correction round {round_index} admission gate",
        )
        admission_gate = validate_stage211_phase_gate_report(
            admission_gate_path,
            expected_phase="mixer",
            checkpoint_path=init_checkpoint,
            require_passed=False,
        )
        if admission_gate.get("gate_passed") is not False:
            raise ValueError(
                f"Stage211 retention correction round {round_index} admission gate passed."
            )
        admission_coverage = admission_gate.get("full_data_coverage")
        if not isinstance(admission_coverage, dict):
            raise ValueError("Stage211 retention correction admission coverage is missing.")
        if admission_coverage.get("segments") != original_segments:
            raise ValueError("Stage211 retention correction rewrites original coverage segments.")
        prior_corrections = admission_coverage.get("post_coverage_corrections", [])
        if prior_corrections != validated:
            raise ValueError(
                f"Stage211 retention correction round {round_index} admission chain mismatch."
            )
        if int(replay.get("samples", -1)) != rows:
            raise ValueError("Stage211 retention correction replay sample count mismatch.")
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
    if int(coverage.get("total_unique_rows", -1)) != STAGE211_AUDIO_TOTAL_ROWS:
        raise ValueError("Stage211 full-data coverage row total mismatch.")
    total_hours = float(coverage.get("total_hours", float("nan")))
    if not math.isfinite(total_hours) or abs(total_hours - STAGE211_AUDIO_TOTAL_HOURS) > 0.002:
        raise ValueError("Stage211 full-data coverage hour total mismatch.")
    if int(coverage.get("total_row_exposures", -1)) != STAGE211_AUDIO_TOTAL_ROW_EXPOSURES:
        raise ValueError("Stage211 full-data row-exposure total mismatch.")
    if (
        int(coverage.get("total_tail_padding_sample_exposures", -1))
        != STAGE211_AUDIO_TOTAL_TAIL_PADDING_SAMPLE_EXPOSURES
    ):
        raise ValueError("Stage211 full-data tail-padding exposure total mismatch.")
    if (
        int(coverage.get("total_executed_sample_exposures", -1))
        != STAGE211_AUDIO_TOTAL_EXECUTED_SAMPLE_EXPOSURES
    ):
        raise ValueError("Stage211 full-data executed-sample exposure total mismatch.")
    total_hour_exposures = float(coverage.get("total_hour_exposures", float("nan")))
    if (
        not math.isfinite(total_hour_exposures)
        or abs(total_hour_exposures - STAGE211_AUDIO_TOTAL_HOUR_EXPOSURES) > 0.005
    ):
        raise ValueError("Stage211 full-data hour-exposure total mismatch.")

    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 full-data coverage segments must be a list.")
    by_difficulty = {
        str(segment.get("difficulty")): segment for segment in segments if isinstance(segment, dict)
    }
    if set(by_difficulty) != set(STAGE211_AUDIO_CURRICULUM):
        raise ValueError(
            "Stage211 full-data coverage must contain exactly easy, medium, hard, and long."
        )

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

    corrections, previous_checkpoint_sha256 = _validate_stage211_post_coverage_corrections(
        coverage.get("post_coverage_corrections", []),
        coverage=coverage,
        original_segments=[by_difficulty[name] for name in STAGE211_AUDIO_CURRICULUM],
        original_completion_sha256=str(previous_checkpoint_sha256 or ""),
        nano_teacher_checkpoint_sha256=str(nano_teacher_checkpoint_sha256 or ""),
    )
    if corrections and coverage.get("phase") != "mixer":
        raise ValueError("Stage211 post-coverage corrections are Mixer-only.")

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


def validate_stage211_public_benchmark(public_benchmark: Any) -> dict[str, Any]:
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
    phase_init_checkpoint = Path(str(easy_segment.get("init_checkpoint_path") or "")).resolve()
    if not phase_init_checkpoint.is_file() or sha256_file(
        phase_init_checkpoint
    ) != easy_segment.get("init_checkpoint_sha256"):
        raise ValueError("Stage211 phase initialization checkpoint is missing or changed.")
    benchmark = validate_stage211_public_benchmark(report.get("public_benchmark"))
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
    if alignment_record.get("artifact") != expected_alignment_artifact:
        raise ValueError(f"Stage211 {expected_phase} alignment report binding mismatch.")
    public_progress_gate_passed = report.get("public_progress_gate_passed")
    if not isinstance(public_progress_gate_passed, bool):
        raise ValueError("Stage211 phase gate lacks a boolean public-progress decision.")
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
