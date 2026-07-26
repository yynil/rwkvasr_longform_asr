from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rwkvasr.config import save_yaml
from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)
from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    validate_stage211_phase_gate_report,
    validate_stage211_runtime_epoch_coverage,
)

try:
    from scripts.run_stage210n_full_easy_spike_tolerant_distill import (
        EVAL_INTERVAL,
        _config as _stage210n_config,
        _latest_step,
        _run,
        _segments as _stage210n_segments,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage210n_full_easy_spike_tolerant_distill import (
        EVAL_INTERVAL,
        _config as _stage210n_config,
        _latest_step,
        _run,
        _segments as _stage210n_segments,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "generated" / "stage211_strict_chained_alignment"
NANO_CHECKPOINT = REPO_ROOT / "assets" / "fun-asr-nano-2512" / "model.pt"
DEFAULT_STAGE211_OUTPUT_ROOT = Path(
    os.environ.get(
        "RWKVASR_STAGE211_OUTPUT_ROOT",
        str(Path.home() / "rwkvasr_runs"),
    )
)
VOLATILE_OUTPUT_ROOTS = tuple(Path(path).resolve() for path in ("/tmp", "/var/tmp", "/dev/shm"))
HARD_LAYER_IDS = (0, 11, 12, 17, 20, 49, 50, 69)
PHASE_SEQUENCE = ("mixer", "block", "logits", "sft")
CURRICULUM_SEQUENCE = tuple(STAGE211_AUDIO_CURRICULUM)
PROMOTION_RECEIPT_SCHEMA_VERSION = 1
TRAIN_BATCH_SIZE = 12
TRAIN_WORLD_SIZE = STAGE211_FULL_DATA_WORLD_SIZE
TRAIN_FRAME_BUDGET = 8_000
FORMAL_RESUME_SAVE_INTERVAL = 2_000
FIXED_HIDDEN_EVAL_SAMPLES = 256
LEGACY_CURRICULUM_STEPS = {
    "easy": 30_064,
    "medium": 1_010_185,
    "hard": 1_045_385,
    "long": 86,
}


@dataclass(frozen=True)
class AlignmentPhase:
    name: str
    lr: float
    input_mode: str
    ctc_weight: float
    mixer_weight: float
    block_weight: float
    encoder_weight: float
    decoder_hidden_weight: float
    blank_weight: float
    conditional_weight: float
    conditional_hard_weight: float
    sequence_weight: float
    window_weight: float
    layer_sample_count: int
    include_hard_layers: bool
    requires_labels: bool
    eval_interval: int


PHASES: dict[str, AlignmentPhase] = {
    "mixer": AlignmentPhase(
        name="mixer",
        lr=3.0e-6,
        input_mode="teacher_forced",
        ctc_weight=0.0,
        mixer_weight=1.0,
        block_weight=0.0,
        encoder_weight=0.0,
        decoder_hidden_weight=0.0,
        blank_weight=0.0,
        conditional_weight=0.0,
        conditional_hard_weight=0.0,
        sequence_weight=0.0,
        window_weight=0.0,
        layer_sample_count=8,
        include_hard_layers=False,
        requires_labels=False,
        eval_interval=EVAL_INTERVAL,
    ),
    "block": AlignmentPhase(
        name="block",
        lr=2.0e-6,
        input_mode="stacked",
        ctc_weight=0.0,
        mixer_weight=0.25,
        block_weight=1.0,
        encoder_weight=0.5,
        decoder_hidden_weight=0.5,
        blank_weight=0.0,
        conditional_weight=0.0,
        conditional_hard_weight=0.0,
        sequence_weight=0.0,
        window_weight=0.0,
        layer_sample_count=8,
        include_hard_layers=False,
        requires_labels=False,
        eval_interval=EVAL_INTERVAL,
    ),
    "logits": AlignmentPhase(
        name="logits",
        lr=3.0e-7,
        input_mode="stacked",
        ctc_weight=0.0,
        mixer_weight=0.10,
        block_weight=0.25,
        encoder_weight=0.25,
        decoder_hidden_weight=0.25,
        blank_weight=0.25,
        conditional_weight=1.0,
        conditional_hard_weight=0.125,
        sequence_weight=0.20,
        window_weight=0.25,
        layer_sample_count=12,
        include_hard_layers=True,
        requires_labels=False,
        eval_interval=EVAL_INTERVAL,
    ),
    "sft": AlignmentPhase(
        name="sft",
        lr=3.0e-7,
        input_mode="stacked",
        ctc_weight=1.0,
        mixer_weight=0.05,
        block_weight=0.10,
        encoder_weight=0.10,
        decoder_hidden_weight=0.10,
        blank_weight=0.05,
        conditional_weight=0.10,
        conditional_hard_weight=0.0,
        sequence_weight=0.0,
        window_weight=0.0,
        layer_sample_count=8,
        include_hard_layers=True,
        requires_labels=True,
        eval_interval=2_000,
    ),
}


def _default_output_dir(phase: AlignmentPhase) -> Path:
    suffixes = {
        "mixer": (
            "stage211a_stage210a30064_nanomlpfrozen_teacherforced_mixeronly_"
            "easy1490h_1ep_lr3e6_wd0_4x4090"
        ),
        "block": (
            "stage211b_stage211a_nanomlpfrozen_stacked_blockpath_easy1490h_1ep_lr2e6_wd0_4x4090"
        ),
        "logits": (
            "stage211c_stage211b_nanomlpfrozen_spiketolerant_logits_easy1490h_1ep_lr3e7_wd0_4x4090"
        ),
        "sft": (
            "stage211d_stage211c_nanomlpfrozen_groundtruth_ctc_sft_labeled1ep_lr3e7_wd0_4x4090"
        ),
    }
    return DEFAULT_STAGE211_OUTPUT_ROOT / f"sensevoice_rwkv_{suffixes[phase.name]}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_output_storage(
    *,
    output_dir: Path,
    smoke: bool,
    dry_run: bool,
) -> None:
    if smoke or dry_run:
        return
    resolved = output_dir.resolve()
    volatile_root = next(
        (root for root in VOLATILE_OUTPUT_ROOTS if _is_relative_to(resolved, root)),
        None,
    )
    if volatile_root is not None:
        raise ValueError(
            "Formal Stage211 output must use persistent storage, not "
            f"{volatile_root}: {resolved}. Use --output-dir under "
            f"{DEFAULT_STAGE211_OUTPUT_ROOT} or another persistent filesystem."
        )


def _segments(
    *,
    phase: AlignmentPhase,
    smoke: bool,
    formal_steps: int | None = None,
    difficulty: str = "easy",
    full_data_profile: bool = False,
) -> list[dict[str, Any]]:
    source = _stage210n_segments(smoke=bool(smoke))[0]
    if smoke:
        target_step = 2
    elif phase.requires_labels:
        if formal_steps is None or int(formal_steps) <= 0:
            raise ValueError("Stage211 SFT requires the estimated full labeled-epoch step count.")
        target_step = int(formal_steps)
    else:
        target_step = int(
            STAGE211_AUDIO_CURRICULUM[difficulty]["steps"]
            if full_data_profile
            else LEGACY_CURRICULUM_STEPS[difficulty]
        )
    formal_name = (
        f"{phase.name}_full_{target_step}steps"
        if difficulty == "easy"
        else f"{phase.name}_{difficulty}_full_{target_step}steps"
    )
    return [
        {
            **source,
            "name": (
                f"{phase.name}_smoke_2steps" if smoke else formal_name
            ),
            "difficulty": difficulty,
            "split_steps": target_step,
            "target_step": target_step,
        }
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _preceding_phase(phase_name: str) -> str | None:
    index = PHASE_SEQUENCE.index(str(phase_name))
    return None if index == 0 else PHASE_SEQUENCE[index - 1]


def _phase_gate_nano_teacher_sha256(gate_report: dict[str, Any]) -> str:
    coverage = gate_report.get("full_data_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 phase gate lacks full-data teacher coverage.")
    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 phase gate lacks curriculum teacher records.")
    teacher_sha256_values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "")
        for segment in segments
        if isinstance(segment, dict)
    }
    if (
        len(teacher_sha256_values) != 1
        or len(next(iter(teacher_sha256_values), "")) != 64
    ):
        raise ValueError(
            "Stage211 phase gate does not bind one Nano teacher checkpoint SHA-256."
        )
    return next(iter(teacher_sha256_values))


def _validate_target_nano_teacher_checkpoint(
    *,
    recorded_sha256: str,
    nano_checkpoint_path: Path,
    label: str,
) -> Path:
    nano_checkpoint_path = nano_checkpoint_path.expanduser().resolve()
    if not nano_checkpoint_path.is_file() or nano_checkpoint_path.stat().st_size <= 0:
        raise ValueError(
            f"Stage211 {label} Nano teacher checkpoint is missing or empty: "
            f"{nano_checkpoint_path}"
        )
    if _sha256_file(nano_checkpoint_path) != recorded_sha256:
        raise ValueError(
            f"Stage211 {label} Nano teacher checkpoint SHA-256 differs from "
            "the preceding stage."
        )
    return nano_checkpoint_path


def _build_promotion_receipt(
    *,
    source_phase: str,
    checkpoint_path: Path,
    gate_report_path: Path,
) -> dict[str, Any]:
    source_phase = str(source_phase)
    if source_phase not in PHASE_SEQUENCE[:-1]:
        raise ValueError(f"Stage211 phase {source_phase!r} cannot promote to another phase.")
    target_phase = PHASE_SEQUENCE[PHASE_SEQUENCE.index(source_phase) + 1]
    checkpoint_path = checkpoint_path.resolve()
    gate_report_path = gate_report_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))
    if not gate_report_path.is_file() or gate_report_path.stat().st_size <= 0:
        raise ValueError(f"Stage211 promotion gate report is missing or empty: {gate_report_path}")
    gate_report = validate_stage211_phase_gate_report(
        gate_report_path,
        expected_phase=source_phase,
        checkpoint_path=checkpoint_path,
    )
    nano_teacher_checkpoint_sha256 = _phase_gate_nano_teacher_sha256(
        gate_report
    )
    return {
        "schema_version": PROMOTION_RECEIPT_SCHEMA_VERSION,
        "pipeline": "stage211",
        "source_phase": source_phase,
        "target_phase": target_phase,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "gate_report_path": str(gate_report_path),
        "gate_report_sha256": _sha256_file(gate_report_path),
        "nano_teacher_checkpoint_sha256": nano_teacher_checkpoint_sha256,
        "gate_passed": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def _validate_promotion_receipt(
    *,
    receipt_path: Path,
    target_phase: str,
    checkpoint_path: Path,
    nano_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(str(receipt_path))
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Invalid Stage211 promotion receipt: {receipt_path}") from error
    if not isinstance(receipt, dict):
        raise ValueError("Stage211 promotion receipt must be a JSON object.")

    expected_source = _preceding_phase(target_phase)
    expected_fields = {
        "schema_version": PROMOTION_RECEIPT_SCHEMA_VERSION,
        "pipeline": "stage211",
        "source_phase": expected_source,
        "target_phase": target_phase,
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"Stage211 promotion receipt {key} mismatch: "
                f"expected={expected!r} actual={receipt.get(key)!r}"
            )
    if receipt.get("gate_passed") is not True:
        raise ValueError("Stage211 promotion receipt does not record a passing gate.")

    checkpoint_path = checkpoint_path.resolve()
    recorded_checkpoint = Path(str(receipt.get("checkpoint_path") or "")).resolve()
    if recorded_checkpoint != checkpoint_path:
        raise ValueError(
            "Stage211 promotion receipt checkpoint path mismatch: "
            f"expected={checkpoint_path} actual={recorded_checkpoint}"
        )
    actual_checkpoint_sha256 = _sha256_file(checkpoint_path)
    if receipt.get("checkpoint_sha256") != actual_checkpoint_sha256:
        raise ValueError("Stage211 promotion receipt checkpoint SHA-256 mismatch.")

    gate_report_path = Path(str(receipt.get("gate_report_path") or "")).resolve()
    if not gate_report_path.is_file() or gate_report_path.stat().st_size <= 0:
        raise ValueError(
            f"Stage211 promotion receipt gate report is missing or empty: {gate_report_path}"
        )
    if receipt.get("gate_report_sha256") != _sha256_file(gate_report_path):
        raise ValueError("Stage211 promotion receipt gate-report SHA-256 mismatch.")
    gate_report = validate_stage211_phase_gate_report(
        gate_report_path,
        expected_phase=str(expected_source),
        checkpoint_path=checkpoint_path,
    )
    nano_teacher_checkpoint_sha256 = _phase_gate_nano_teacher_sha256(
        gate_report
    )
    if (
        receipt.get("nano_teacher_checkpoint_sha256")
        != nano_teacher_checkpoint_sha256
    ):
        raise ValueError(
            "Stage211 promotion receipt Nano teacher checkpoint SHA-256 mismatch."
        )
    if nano_checkpoint_path is not None:
        _validate_target_nano_teacher_checkpoint(
            recorded_sha256=nano_teacher_checkpoint_sha256,
            nano_checkpoint_path=nano_checkpoint_path,
            label=f"{expected_source}->{target_phase}",
        )
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
    }


def _validate_curriculum_receipt(
    *,
    receipt_path: Path,
    phase: str,
    target_difficulty: str,
    checkpoint_path: Path,
    nano_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(str(receipt_path))
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Invalid Stage211 curriculum receipt: {receipt_path}") from error
    if not isinstance(receipt, dict):
        raise ValueError("Stage211 curriculum receipt must be a JSON object.")
    target_index = CURRICULUM_SEQUENCE.index(target_difficulty)
    if target_index <= 0:
        raise ValueError("Stage211 easy curriculum does not accept a predecessor receipt.")
    expected_difficulty = CURRICULUM_SEQUENCE[target_index - 1]
    expected_fields = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "phase": phase,
        "difficulty": expected_difficulty,
        "complete": True,
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            raise ValueError(
                f"Stage211 curriculum receipt {key} mismatch: "
                f"expected={expected!r} actual={receipt.get(key)!r}"
            )
    checkpoint_path = checkpoint_path.resolve()
    recorded_checkpoint = Path(
        str(receipt.get("completion_checkpoint_path") or "")
    ).resolve()
    if recorded_checkpoint != checkpoint_path:
        raise ValueError(
            "Stage211 curriculum receipt checkpoint path mismatch: "
            f"expected={checkpoint_path} actual={recorded_checkpoint}"
        )
    if receipt.get("completion_checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise ValueError("Stage211 curriculum receipt checkpoint SHA-256 mismatch.")
    validate_stage211_runtime_epoch_coverage(
        receipt.get("runtime_epoch_coverage"),
        epochs=STAGE211_FULL_DATA_EPOCHS,
        steps_per_epoch=int(
            STAGE211_AUDIO_CURRICULUM[expected_difficulty]["steps_per_epoch"]
        ),
        label=f"{phase}/{expected_difficulty}",
    )
    recorded_teacher_sha256 = str(
        receipt.get("nano_teacher_checkpoint_sha256") or ""
    )
    recorded_teacher_path = Path(
        str(receipt.get("nano_teacher_checkpoint_path") or "")
    ).resolve()
    if (
        len(recorded_teacher_sha256) != 64
        or not recorded_teacher_path.is_file()
        or recorded_teacher_path.stat().st_size <= 0
        or _sha256_file(recorded_teacher_path) != recorded_teacher_sha256
    ):
        raise ValueError(
            "Stage211 curriculum receipt Nano teacher checkpoint is unavailable "
            "or changed."
        )
    if nano_checkpoint_path is not None:
        _validate_target_nano_teacher_checkpoint(
            recorded_sha256=recorded_teacher_sha256,
            nano_checkpoint_path=nano_checkpoint_path,
            label=f"{phase}/{expected_difficulty}->{target_difficulty}",
        )
    return {
        **receipt,
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
    }


def _resolve_manifest_recorded_path(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _audit_audio_bucket_storage(bucket_manifest_path: Path) -> dict[str, Any]:
    bucket_manifest_path = bucket_manifest_path.resolve()
    manifest = load_webdataset_bucket_manifest(bucket_manifest_path)
    root = _resolve_manifest_recorded_path(bucket_manifest_path, manifest.root)
    length_index = _resolve_manifest_recorded_path(
        bucket_manifest_path,
        manifest.source_length_index_path,
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Stage211 audio WebDataset root is unavailable: {root}")
    if not length_index.is_file():
        raise FileNotFoundError(f"Stage211 audio length index is unavailable: {length_index}")

    part_paths: list[Path] = []
    split_samples: dict[str, int] = {}
    for split, buckets in manifest.splits.items():
        split_samples[split] = sum(bucket.num_samples for bucket in buckets)
        for bucket in buckets:
            part_paths.extend(
                _resolve_manifest_recorded_path(bucket_manifest_path, part.path)
                for part in bucket.parts
            )
    if not part_paths:
        raise ValueError(
            f"Stage211 audio bucket manifest has no bucket parts: {bucket_manifest_path}"
        )
    missing_parts = [path for path in part_paths if not path.is_file()]
    if missing_parts:
        raise FileNotFoundError(
            "Stage211 audio bucket manifest references missing part files: "
            f"missing={len(missing_parts)}/{len(part_paths)} "
            f"first={missing_parts[:3]}"
        )
    return {
        "bucket_manifest_path": str(bucket_manifest_path),
        "bucket_manifest_sha256": _sha256_file(bucket_manifest_path),
        "webdataset_root": str(root),
        "length_index_path": str(length_index),
        "length_index_size": length_index.stat().st_size,
        "bucket_part_files": len(part_paths),
        "split_samples": split_samples,
    }


def _audit_labeled_data(
    *,
    webdataset_root: Path,
    length_index_path: Path,
    bucket_manifest_path: Path,
) -> dict[str, Any]:
    webdataset_root = webdataset_root.resolve()
    length_index_path = length_index_path.resolve()
    bucket_manifest_path = bucket_manifest_path.resolve()
    if not webdataset_root.is_dir():
        raise FileNotFoundError(str(webdataset_root))
    if not length_index_path.is_file():
        raise FileNotFoundError(str(length_index_path))

    manifest = load_webdataset_bucket_manifest(bucket_manifest_path)
    recorded_index = Path(manifest.source_length_index_path).resolve()
    if recorded_index != length_index_path:
        raise ValueError(
            "Stage211 SFT bucket manifest and length index do not match: "
            f"manifest={recorded_index} requested={length_index_path}"
        )
    manifest_counts = {
        split: sum(bucket.num_samples for bucket in manifest.splits.get(split, ()))
        for split in ("train", "eval")
    }
    if any(count <= 0 for count in manifest_counts.values()):
        raise ValueError(
            f"Stage211 SFT requires non-empty train and eval bucket splits: {manifest_counts}"
        )

    split_counts = {"train": 0, "eval": 0}
    seen_ids: dict[str, str] = {}
    total_frames = 0
    total_tokens = 0
    required_fields = (
        "json_member",
        "normalized_text_chars",
        "ctc_num_tokens",
        "ctc_unk_tokens",
        "ctc_required_frames",
        "ctc_logit_frames",
    )
    with length_index_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid Stage211 SFT length-index JSON at line {line_number}"
                ) from error
            missing = [field for field in required_fields if field not in row]
            if missing:
                raise ValueError(
                    f"Stage211 SFT row {line_number} lacks CTC label metadata: {missing}"
                )
            split = str(row.get("split") or "")
            if split not in split_counts:
                raise ValueError(f"Stage211 SFT row {line_number} has unsupported split {split!r}")
            utt_id = str(row.get("utt_id") or row.get("key") or "")
            if not utt_id:
                raise ValueError(f"Stage211 SFT row {line_number} has no utterance id.")
            if utt_id in seen_ids:
                raise ValueError(
                    f"Stage211 SFT utterance {utt_id!r} is duplicated across rows/splits."
                )
            seen_ids[utt_id] = split
            normalized_chars = int(row["normalized_text_chars"])
            ctc_tokens = int(row["ctc_num_tokens"])
            ctc_unk_tokens = int(row["ctc_unk_tokens"])
            required_frames = int(row["ctc_required_frames"])
            logit_frames = int(row["ctc_logit_frames"])
            num_frames = int(row.get("num_frames") or 0)
            if (
                not str(row["json_member"])
                or normalized_chars <= 0
                or ctc_tokens <= 0
                or ctc_unk_tokens != 0
                or required_frames <= 0
                or logit_frames <= 0
                or required_frames > logit_frames
                or num_frames <= 0
            ):
                raise ValueError(
                    f"Stage211 SFT row {line_number} has an invalid pronunciation-only "
                    "CTC target or infeasible frame length."
                )
            split_counts[split] += 1
            total_frames += num_frames
            total_tokens += ctc_tokens

    if split_counts != manifest_counts:
        raise ValueError(
            "Stage211 SFT length-index and bucket-manifest counts differ: "
            f"index={split_counts} manifest={manifest_counts}"
        )
    estimated_steps = estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        world_size=TRAIN_WORLD_SIZE,
        frame_budget=TRAIN_FRAME_BUDGET,
        drop_last=False,
    )
    tail_padding_samples = estimate_bucket_manifest_tail_padding_samples(
        manifest,
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
        world_size=TRAIN_WORLD_SIZE,
        frame_budget=TRAIN_FRAME_BUDGET,
    )
    return {
        "webdataset_root": str(webdataset_root),
        "length_index_path": str(length_index_path),
        "bucket_manifest_path": str(bucket_manifest_path),
        "train_samples": split_counts["train"],
        "eval_samples": split_counts["eval"],
        "total_samples": sum(split_counts.values()),
        "total_hours": total_frames / 100.0 / 3600.0,
        "ctc_tokens": total_tokens,
        "estimated_train_steps": estimated_steps,
        "tail_padding_samples_per_epoch": tail_padding_samples,
        "tail_padding_sample_exposures": tail_padding_samples,
        "executed_sample_exposures": split_counts["train"] + tail_padding_samples,
    }


def _zero_output_weights(config: dict[str, Any]) -> None:
    for key in (
        "ctc_teacher_online_loss_weight",
        "ctc_teacher_online_blank_loss_weight",
        "ctc_teacher_online_mass_loss_weight",
        "ctc_teacher_online_full_loss_weight",
        "ctc_teacher_online_conditional_nonblank_loss_weight",
        "ctc_teacher_online_conditional_nonblank_hard_loss_weight",
        "ctc_teacher_online_sequence_loss_weight",
        "ctc_teacher_online_sequence_presence_loss_weight",
        "ctc_teacher_online_sequence_window_loss_weight",
        "ctc_teacher_online_nonblank_hard_loss_weight",
        "ctc_teacher_online_nonblank_margin_loss_weight",
        "ctc_teacher_online_nonblank_window_loss_weight",
        "ctc_teacher_online_nonblank_window_margin_loss_weight",
        "ctc_teacher_online_nonblank_window_topk_loss_weight",
    ):
        config[key] = 0.0


def _config(
    *,
    phase: AlignmentPhase,
    segment: dict[str, Any],
    output_dir: Path,
    init_checkpoint: Path,
    bucket_manifest: Path,
    resume: bool,
    smoke: bool,
    nano_checkpoint: Path = NANO_CHECKPOINT,
    labeled_webdataset_root: Path | None = None,
    labeled_length_index: Path | None = None,
    audio_data_audit: dict[str, Any] | None = None,
    full_data_profile: bool = False,
) -> dict[str, Any]:
    config = _stage210n_config(
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        bucket_manifest=bucket_manifest,
        resume=resume,
        smoke=smoke,
    )
    target_step = int(segment["target_step"])
    eval_interval = target_step if smoke else int(phase.eval_interval)
    save_interval = target_step if smoke else min(int(FORMAL_RESUME_SAVE_INTERVAL), target_step)
    _zero_output_weights(config)
    config.update(
        {
            "max_steps": target_step,
            "lr": float(phase.lr),
            "weight_decay": 0.0,
            "save_every": save_interval,
            "step_eval_every": eval_interval,
            "top_k_step_checkpoints": (
                1 if smoke else min(8, max(1, math.ceil(target_step / eval_interval)))
            ),
            "periodic_checkpoint_keep_last": None if smoke else 2,
            "freeze_encoder": False,
            "freeze_encoder_except_time_mixer": True,
            "freeze_ctc_decoder": True,
            "freeze_ctc_head": True,
            "allow_missing_targets": not phase.requires_labels,
            "ctc_loss_weight": float(phase.ctc_weight),
            "decoder_loss_weight": 0.0,
            "ctc_suppress_non_pronunciation_tokens": bool(phase.requires_labels),
            "ctc_teacher_online_model_path": str(
                nano_checkpoint.expanduser().resolve().parent
            ),
            "funasr_nano_ctc_init_checkpoint_path": None,
            "funasr_nano_ctc_init_load_encoder": False,
            "funasr_nano_ctc_init_load_encoder_attention": False,
            "funasr_nano_ctc_init_load_rwkv_encoder_from_qkv": False,
            "funasr_nano_ctc_init_load_decoder": False,
            "funasr_nano_ctc_init_load_head": False,
            "ctc_teacher_frame_filter": "all",
            "ctc_teacher_online_encoder_loss_weight": float(phase.encoder_weight),
            "ctc_teacher_online_decoder_hidden_loss_weight": float(phase.decoder_hidden_weight),
            "ctc_teacher_online_layer_mixer_loss_weight": float(phase.mixer_weight),
            "ctc_teacher_online_layer_ffn_loss_weight": 0.0,
            "ctc_teacher_online_layer_block_loss_weight": float(phase.block_weight),
            "ctc_teacher_online_layer_sample_count": int(phase.layer_sample_count),
            "ctc_teacher_online_layer_boundary_ids": (
                list(HARD_LAYER_IDS) if phase.include_hard_layers else []
            ),
            "ctc_teacher_online_layer_include_boundaries": bool(phase.include_hard_layers),
            "ctc_teacher_online_layer_input_mode": str(phase.input_mode),
            "ctc_teacher_online_frame_balance_mode": "all",
            "ctc_teacher_online_blank_frame_balance_mode": "all",
            "ctc_teacher_online_hidden_frame_balance_mode": "all",
            "ctc_teacher_online_keep_full_log_probs_on_device": phase.name == "logits",
            "ctc_teacher_online_blank_loss_weight": float(phase.blank_weight),
            "ctc_teacher_online_conditional_nonblank_loss_weight": float(phase.conditional_weight),
            "ctc_teacher_online_conditional_nonblank_hard_loss_weight": float(
                phase.conditional_hard_weight
            ),
            "ctc_teacher_online_sequence_loss_weight": float(phase.sequence_weight),
            "ctc_teacher_online_nonblank_window_loss_weight": float(phase.window_weight),
            "ctc_teacher_online_nonblank_window_loss_mode": ("conditional_nonblank_hard"),
            "ctc_teacher_online_nonblank_window_radius": 2,
            "ctc_teacher_online_nonblank_window_temperature": 0.20,
            "step_eval_split": (
                "eval"
                if phase.requires_labels
                or full_data_profile
                or str(segment.get("difficulty") or "easy") != "easy"
                else "train"
            ),
            "step_eval_cache_batches": True,
            "wandb_run_name": f"{output_dir.name}_{segment['name']}",
        }
    )
    if full_data_profile:
        config.update(
            {
                "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
                "batch_token_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
                "length_bucket_drop_last": False,
                "length_bucket_frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
                "skip_oversized_samples": False,
                "webdataset_skip_decode_errors": False,
            }
        )
        deepspeed_config = dict(config["deepspeed"])
        gradient_accumulation = int(
            deepspeed_config.get("gradient_accumulation_steps", 1)
        )
        deepspeed_config.update(
            {
                "train_micro_batch_size_per_gpu": STAGE211_FULL_DATA_BATCH_SIZE,
                "train_batch_size": (
                    STAGE211_FULL_DATA_BATCH_SIZE
                    * TRAIN_WORLD_SIZE
                    * gradient_accumulation
                ),
            }
        )
        config["deepspeed"] = deepspeed_config
    if phase.requires_labels:
        if labeled_webdataset_root is None or labeled_length_index is None:
            raise ValueError(
                "Stage211 SFT requires an explicit labeled WebDataset root and length index."
            )
        config.update(
            {
                "webdataset_root": str(labeled_webdataset_root),
                "webdataset_index_path": None,
                "webdataset_length_index_path": str(labeled_length_index),
                "webdataset_bucket_manifest_path": str(bucket_manifest),
                "webdataset_split": "train",
                "webdataset_eval_ratio": 0.0,
                "webdataset_split_by": "sample_id",
                "webdataset_utt_id_key": "id",
                "feature_extractor_type": "funasr_wav_frontend",
                "bucket_source_interleave": True,
                "length_bucket_drop_last": False,
                "skip_oversized_samples": False,
                "webdataset_skip_decode_errors": False,
            }
        )
    elif audio_data_audit is not None:
        config.update(
            {
                "webdataset_root": str(audio_data_audit["webdataset_root"]),
                "webdataset_length_index_path": str(audio_data_audit["length_index_path"]),
                "webdataset_bucket_manifest_path": str(bucket_manifest),
            }
        )
    return config


def _nano_non_attention_pairs() -> tuple[tuple[str, str], ...]:
    suffixes = (
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
        "feed_forward.w_1.weight",
        "feed_forward.w_1.bias",
        "feed_forward.w_2.weight",
        "feed_forward.w_2.bias",
    )
    pairs: list[tuple[str, str]] = []
    for layer_id in range(70):
        if layer_id == 0:
            teacher_prefix = "audio_encoder.encoders0.0"
        elif layer_id < 50:
            teacher_prefix = f"audio_encoder.encoders.{layer_id - 1}"
        else:
            teacher_prefix = f"audio_encoder.tp_encoders.{layer_id - 50}"
        student_prefix = f"encoder.sensevoice_encoder.layers.{layer_id}"
        pairs.extend(
            (f"{student_prefix}.{suffix}", f"{teacher_prefix}.{suffix}") for suffix in suffixes
        )
    pairs.extend(
        (
            (f"encoder.sensevoice_encoder.{name}.{suffix}", f"audio_encoder.{name}.{suffix}")
            for name in ("after_norm", "tp_norm")
            for suffix in ("weight", "bias")
        )
    )
    return tuple(pairs)


def _audit_nano_non_attention_exact(
    *,
    checkpoint_path: Path,
    nano_checkpoint_path: Path,
) -> dict[str, int]:
    import torch

    student_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    student_state = student_payload.get("model", student_payload)
    nano_payload = torch.load(
        nano_checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    nano_state = nano_payload.get("model", nano_payload.get("state_dict", nano_payload))
    pairs = _nano_non_attention_pairs()
    missing: list[str] = []
    mismatched: list[str] = []
    for student_key, nano_key in pairs:
        student_value = student_state.get(student_key)
        nano_value = nano_state.get(nano_key)
        if not isinstance(student_value, torch.Tensor) or not isinstance(nano_value, torch.Tensor):
            missing.append(f"{student_key} <- {nano_key}")
            continue
        expected = nano_value.to(dtype=student_value.dtype)
        if not torch.equal(student_value, expected):
            mismatched.append(student_key)
    del student_payload, student_state, nano_payload, nano_state
    gc.collect()
    if missing or mismatched:
        raise ValueError(
            "Stage211 requires bit-identical Nano non-attention encoder weights: "
            f"expected={len(pairs)} missing={len(missing)} mismatched={len(mismatched)} "
            f"first_missing={missing[:3]} first_mismatched={mismatched[:3]}"
        )
    return {"expected": len(pairs), "matched": len(pairs)}


def _write_config(
    *,
    phase: AlignmentPhase,
    segment: dict[str, Any],
    config: dict[str, Any],
    config_dir: Path,
    smoke: bool,
) -> Path:
    formal_name = (
        "labeled_sft"
        if phase.requires_labels
        else f"full_{str(segment.get('difficulty') or 'easy')}"
    )
    target_dir = config_dir / phase.name / ("smoke" if smoke else formal_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"stage211_{segment['name']}.yaml"
    save_yaml(path, config)
    return path


def _record_or_validate_provenance(
    *,
    output_dir: Path,
    phase: AlignmentPhase,
    bucket_manifest: Path,
    init_checkpoint: Path,
    promotion_receipt: dict[str, Any] | None,
    curriculum_difficulty: str | None,
    curriculum_receipt: dict[str, Any] | None,
    full_data_profile: bool,
    labeled_data_audit: dict[str, Any] | None,
    audio_data_audit: dict[str, Any] | None,
) -> Path:
    path = output_dir / "stage211_provenance.json"
    payload = {
        "schema_version": 1,
        "pipeline": "stage211",
        "phase": phase.name,
        "output_dir": str(output_dir.resolve()),
        "bucket_manifest_path": str(bucket_manifest.resolve()),
        "init_checkpoint_path": str(init_checkpoint.resolve()),
        "init_checkpoint_sha256": _sha256_file(init_checkpoint),
        "promotion_receipt_path": (
            promotion_receipt.get("receipt_path") if promotion_receipt is not None else None
        ),
        "promotion_receipt_sha256": (
            promotion_receipt.get("receipt_sha256") if promotion_receipt is not None else None
        ),
        "labeled_data_audit": labeled_data_audit,
        "audio_data_audit": audio_data_audit,
    }
    if curriculum_difficulty is not None:
        payload["curriculum_difficulty"] = curriculum_difficulty
        payload["curriculum_receipt_path"] = (
            curriculum_receipt.get("receipt_path")
            if curriculum_receipt is not None
            else None
        )
        payload["curriculum_receipt_sha256"] = (
            curriculum_receipt.get("receipt_sha256")
            if curriculum_receipt is not None
            else None
        )
    if full_data_profile:
        payload["full_data_profile"] = True
        payload["length_bucket_drop_last"] = False
        payload["skip_oversized_samples"] = False
        payload["webdataset_skip_decode_errors"] = False
    if phase.requires_labels:
        payload["length_bucket_drop_last"] = False
        payload["skip_oversized_samples"] = False
        payload["webdataset_skip_decode_errors"] = False
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Stage211 output provenance cannot be changed in place: {path}")
        return path
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def _validate_resume_provenance(
    *,
    output_dir: Path,
    phase: AlignmentPhase,
    bucket_manifest: Path,
    labeled_data_audit: dict[str, Any] | None,
    audio_data_audit: dict[str, Any] | None,
    curriculum_difficulty: str | None,
    full_data_profile: bool,
) -> dict[str, Any]:
    path = output_dir / "stage211_provenance.json"
    if not path.is_file():
        raise ValueError(f"Stage211 resume requires immutable provenance: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "pipeline": "stage211",
        "phase": phase.name,
        "output_dir": str(output_dir.resolve()),
        "bucket_manifest_path": str(bucket_manifest.resolve()),
        "labeled_data_audit": labeled_data_audit,
        "audio_data_audit": audio_data_audit,
    }
    if curriculum_difficulty is not None:
        expected["curriculum_difficulty"] = curriculum_difficulty
    if full_data_profile:
        expected["full_data_profile"] = True
        expected["length_bucket_drop_last"] = False
        expected["skip_oversized_samples"] = False
        expected["webdataset_skip_decode_errors"] = False
    if phase.requires_labels:
        expected["length_bucket_drop_last"] = False
        expected["skip_oversized_samples"] = False
        expected["webdataset_skip_decode_errors"] = False
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Stage211 resume provenance {key} mismatch: "
                f"expected={value!r} actual={payload.get(key)!r}"
            )
    curriculum_receipt_path_value = payload.get("curriculum_receipt_path")
    curriculum_receipt_sha256 = payload.get("curriculum_receipt_sha256")
    is_curriculum_continuation = (
        curriculum_difficulty is not None and curriculum_difficulty != "easy"
    )
    if is_curriculum_continuation:
        curriculum_receipt_path = Path(
            str(curriculum_receipt_path_value or "")
        ).resolve()
        if (
            not curriculum_receipt_path.is_file()
            or _sha256_file(curriculum_receipt_path) != curriculum_receipt_sha256
        ):
            raise ValueError(
                "Stage211 resume curriculum-receipt provenance is missing or changed."
            )
    elif (
        curriculum_receipt_path_value is not None
        or curriculum_receipt_sha256 is not None
    ):
        raise ValueError("Stage211 easy/SFT provenance must not contain a curriculum receipt.")

    receipt_path_value = payload.get("promotion_receipt_path")
    receipt_sha256 = payload.get("promotion_receipt_sha256")
    requires_promotion = (
        _preceding_phase(phase.name) is not None and not is_curriculum_continuation
    )
    if not requires_promotion:
        if receipt_path_value is not None or receipt_sha256 is not None:
            raise ValueError(
                "Stage211 mixer/curriculum-continuation provenance must not contain "
                "a phase-promotion receipt."
            )
    else:
        receipt_path = Path(str(receipt_path_value or "")).resolve()
        if not receipt_path.is_file() or _sha256_file(receipt_path) != receipt_sha256:
            raise ValueError("Stage211 resume promotion-receipt provenance is missing or changed.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one explicitly selected phase of strict chained Nano attention, "
            "block, CTC-logit alignment, or labeled CTC SFT."
        )
    )
    parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--bucket-manifest", type=Path, default=None)
    parser.add_argument("--difficulty", choices=CURRICULUM_SEQUENCE, default=None)
    parser.add_argument("--curriculum-receipt", type=Path, default=None)
    parser.add_argument("--full-data-profile", action="store_true")
    parser.add_argument("--labeled-webdataset-root", type=Path, default=None)
    parser.add_argument("--labeled-length-index", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--promotion-receipt", type=Path, default=None)
    parser.add_argument("--nano-checkpoint", type=Path, default=NANO_CHECKPOINT)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-nano-weight-audit",
        action="store_true",
        help="Testing-only escape hatch; formal runs must keep the default exact-weight audit.",
    )
    args = parser.parse_args()

    phase = PHASES[str(args.phase)]
    if phase.requires_labels:
        if (
            args.difficulty is not None
            or args.curriculum_receipt is not None
            or args.full_data_profile
        ):
            parser.error("Stage211 SFT does not accept audio curriculum arguments")
        difficulty = "easy"
    else:
        difficulty = str(args.difficulty or "easy")
        if args.full_data_profile and args.difficulty is None:
            parser.error("--full-data-profile requires an explicit --difficulty")
    if args.skip_nano_weight_audit and not args.dry_run:
        parser.error("--skip-nano-weight-audit is allowed only with --dry-run")
    bucket_manifest = args.bucket_manifest
    if bucket_manifest is None:
        parser.error(
            "Stage211 requires an explicit persistent --bucket-manifest; "
            "volatile defaults are not recoverable after a reboot"
        )
    if not bucket_manifest.is_file():
        raise FileNotFoundError(str(bucket_manifest))

    labeled_data_audit: dict[str, Any] | None = None
    audio_data_audit: dict[str, Any] | None = None
    formal_steps: int | None = None
    if phase.requires_labels:
        if args.labeled_webdataset_root is None or args.labeled_length_index is None:
            parser.error(
                "Stage211 SFT requires --labeled-webdataset-root and --labeled-length-index"
            )
        labeled_data_audit = _audit_labeled_data(
            webdataset_root=args.labeled_webdataset_root,
            length_index_path=args.labeled_length_index,
            bucket_manifest_path=bucket_manifest,
        )
        formal_steps = int(labeled_data_audit["estimated_train_steps"])
    elif args.labeled_webdataset_root is not None or args.labeled_length_index is not None:
        parser.error("labeled data arguments are valid only for --phase sft")
    elif not args.dry_run:
        audio_data_audit = _audit_audio_bucket_storage(bucket_manifest)
        manifest = load_webdataset_bucket_manifest(bucket_manifest)
        expected_coverage = STAGE211_AUDIO_CURRICULUM[difficulty]
        actual_rows = int(audio_data_audit["split_samples"].get("train", 0))
        audit_batch_size = (
            STAGE211_FULL_DATA_BATCH_SIZE
            if args.full_data_profile
            else TRAIN_BATCH_SIZE
        )
        audit_frame_budget = (
            STAGE211_FULL_DATA_FRAME_BUDGET
            if args.full_data_profile
            else TRAIN_FRAME_BUDGET
        )
        actual_steps_per_epoch = estimate_bucket_manifest_steps(
            manifest,
            split="train",
            batch_size=audit_batch_size,
            world_size=TRAIN_WORLD_SIZE,
            frame_budget=audit_frame_budget,
            drop_last=not args.full_data_profile,
        )
        expected_steps_per_epoch = int(
            expected_coverage["steps_per_epoch"]
            if args.full_data_profile
            else LEGACY_CURRICULUM_STEPS[difficulty]
        )
        if (
            actual_rows != int(expected_coverage["rows"])
            or actual_steps_per_epoch != expected_steps_per_epoch
        ):
            raise ValueError(
                f"Stage211 {difficulty} manifest coverage mismatch: "
                f"rows={actual_rows}/{expected_coverage['rows']} "
                f"steps_per_epoch={actual_steps_per_epoch}/{expected_steps_per_epoch}"
            )
        if args.full_data_profile:
            actual_tail_padding = estimate_bucket_manifest_tail_padding_samples(
                manifest,
                split="train",
                batch_size=audit_batch_size,
                world_size=TRAIN_WORLD_SIZE,
                frame_budget=audit_frame_budget,
            )
            expected_tail_padding = int(
                expected_coverage["tail_padding_samples_per_epoch"]
            )
            if actual_tail_padding != expected_tail_padding:
                raise ValueError(
                    f"Stage211 {difficulty} manifest tail-padding mismatch: "
                    f"actual={actual_tail_padding} expected={expected_tail_padding}"
                )
        if difficulty != "easy" or args.full_data_profile:
            eval_rows = int(audio_data_audit["split_samples"].get("eval", 0))
            if eval_rows != FIXED_HIDDEN_EVAL_SAMPLES:
                raise ValueError(
                    f"Stage211 {difficulty} manifest must bind the fixed hidden-eval split: "
                    f"actual={eval_rows} expected={FIXED_HIDDEN_EVAL_SAMPLES}"
                )

    base_output_dir = args.output_dir or _default_output_dir(phase)
    output_dir = Path(f"{base_output_dir}_smoke") if args.smoke else base_output_dir
    _validate_output_storage(
        output_dir=output_dir,
        smoke=bool(args.smoke),
        dry_run=bool(args.dry_run),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(output_dir)
    init_checkpoint = args.init_checkpoint
    promotion_receipt: dict[str, Any] | None = None
    curriculum_receipt: dict[str, Any] | None = None
    if latest_step <= 0:
        if init_checkpoint is None:
            if phase.name == "mixer":
                parser.error(
                    "a fresh Stage211 mixer run requires an explicitly selected "
                    "persistent initialization checkpoint"
                )
            parser.error(
                f"a fresh Stage211 {phase.name} run requires the explicitly selected "
                "checkpoint from the preceding phase on persistent storage"
            )
        if not init_checkpoint.is_file():
            raise FileNotFoundError(str(init_checkpoint))
        is_curriculum_continuation = not phase.requires_labels and difficulty != "easy"
        if is_curriculum_continuation:
            if args.curriculum_receipt is None:
                parser.error(
                    f"a fresh Stage211 {phase.name}/{difficulty} run requires the "
                    "preceding curriculum coverage receipt"
                )
            if args.promotion_receipt is not None:
                parser.error(
                    "a Stage211 curriculum continuation does not accept a phase-promotion receipt"
                )
            curriculum_receipt = _validate_curriculum_receipt(
                receipt_path=args.curriculum_receipt,
                phase=phase.name,
                target_difficulty=difficulty,
                checkpoint_path=init_checkpoint,
                nano_checkpoint_path=args.nano_checkpoint,
            )
            if args.full_data_profile and curriculum_receipt.get(
                "full_data_profile"
            ) is not True:
                raise ValueError(
                    "Stage211 full-data curriculum continuation requires a "
                    "full-data predecessor receipt."
                )
        else:
            if args.curriculum_receipt is not None:
                parser.error("Stage211 easy/SFT does not accept a curriculum receipt")
            preceding_phase = _preceding_phase(phase.name)
            if preceding_phase is None:
                if args.promotion_receipt is not None:
                    parser.error("Stage211 mixer phase does not accept a promotion receipt")
            else:
                if args.promotion_receipt is None:
                    parser.error(
                        f"a fresh Stage211 {phase.name} run requires a passing promotion "
                        f"receipt from Stage211 {preceding_phase}"
                    )
                promotion_receipt = _validate_promotion_receipt(
                    receipt_path=args.promotion_receipt,
                    target_phase=phase.name,
                    checkpoint_path=init_checkpoint,
                    nano_checkpoint_path=args.nano_checkpoint,
                )
                print(
                    "promotion_receipt="
                    f"{promotion_receipt['source_phase']}->{promotion_receipt['target_phase']} "
                    f"sha256={promotion_receipt['receipt_sha256']}",
                    flush=True,
                )
        if not args.skip_nano_weight_audit:
            if not args.nano_checkpoint.is_file():
                raise FileNotFoundError(str(args.nano_checkpoint))
            audit = _audit_nano_non_attention_exact(
                checkpoint_path=init_checkpoint,
                nano_checkpoint_path=args.nano_checkpoint,
            )
            print(
                f"nano_non_attention_audit={audit['matched']}/{audit['expected']} exact",
                flush=True,
            )
        if not args.dry_run:
            provenance_path = _record_or_validate_provenance(
                output_dir=output_dir,
                phase=phase,
                bucket_manifest=bucket_manifest,
                init_checkpoint=init_checkpoint,
                promotion_receipt=promotion_receipt,
                curriculum_difficulty=args.difficulty,
                curriculum_receipt=curriculum_receipt,
                full_data_profile=bool(args.full_data_profile),
                labeled_data_audit=labeled_data_audit,
                audio_data_audit=audio_data_audit,
            )
            print(f"provenance={provenance_path}", flush=True)
    else:
        if (
            init_checkpoint is not None
            or args.promotion_receipt is not None
            or args.curriculum_receipt is not None
        ):
            parser.error(
                "a Stage211 resume must use its recorded initialization and cannot "
                "accept checkpoint or receipt arguments"
            )
        _validate_resume_provenance(
            output_dir=output_dir,
            phase=phase,
            bucket_manifest=bucket_manifest,
            labeled_data_audit=labeled_data_audit,
            audio_data_audit=audio_data_audit,
            curriculum_difficulty=args.difficulty,
            full_data_profile=bool(args.full_data_profile),
        )
    if init_checkpoint is None:
        init_checkpoint = output_dir / "resume-placeholder.pt"

    print(f"phase={phase.name}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"bucket_manifest={bucket_manifest}", flush=True)
    if not phase.requires_labels:
        print(f"difficulty={difficulty}", flush=True)
        print(f"full_data_profile={str(bool(args.full_data_profile)).lower()}", flush=True)
    if labeled_data_audit is not None:
        print(
            "labeled_data_audit="
            f"train={labeled_data_audit['train_samples']} "
            f"eval={labeled_data_audit['eval_samples']} "
            f"hours={labeled_data_audit['total_hours']:.3f} "
            f"ctc_tokens={labeled_data_audit['ctc_tokens']} "
            f"steps={labeled_data_audit['estimated_train_steps']}",
            flush=True,
        )
    if audio_data_audit is not None:
        print(
            "audio_data_audit="
            f"root={audio_data_audit['webdataset_root']} "
            f"parts={audio_data_audit['bucket_part_files']} "
            f"splits={audio_data_audit['split_samples']}",
            flush=True,
        )
    print(f"latest_step={latest_step}", flush=True)
    print(f"init_checkpoint={init_checkpoint}", flush=True)
    for segment in _segments(
        phase=phase,
        smoke=bool(args.smoke),
        formal_steps=formal_steps,
        difficulty=difficulty,
        full_data_profile=bool(args.full_data_profile),
    ):
        target_step = int(segment["target_step"])
        print(
            f"segment={segment['name']} "
            f"split={'labeled_train' if phase.requires_labels else segment['split'].name} "
            f"target_step={target_step}",
            flush=True,
        )
        if latest_step >= target_step:
            print(f"skip {segment['name']} target_step={target_step}", flush=True)
            continue
        config = _config(
            phase=phase,
            segment=segment,
            output_dir=output_dir,
            init_checkpoint=init_checkpoint,
            bucket_manifest=bucket_manifest,
            resume=latest_step > 0,
            smoke=bool(args.smoke),
            nano_checkpoint=args.nano_checkpoint,
            labeled_webdataset_root=args.labeled_webdataset_root,
            labeled_length_index=args.labeled_length_index,
            audio_data_audit=audio_data_audit,
            full_data_profile=bool(args.full_data_profile),
        )
        config_path = _write_config(
            phase=phase,
            segment=segment,
            config=config,
            config_dir=args.config_dir,
            smoke=bool(args.smoke),
        )
        code = _run(
            config_path,
            output_dir / "logs" / f"{segment['name']}.log",
            dry_run=bool(args.dry_run),
            master_port=int(args.master_port),
        )
        if code != 0:
            return code
        latest_step = _latest_step(output_dir)
        if not args.dry_run and latest_step < target_step:
            raise RuntimeError(
                f"Stage211 {phase.name} exited before target_step={target_step}: "
                f"latest_step={latest_step}"
            )
    if not args.dry_run:
        print(f"complete phase={phase.name} latest_step={latest_step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
