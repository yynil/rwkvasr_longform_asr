#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from rwkvasr.config import load_yaml, save_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_PUBLIC_BENCHMARKS,
    resolve_stage211_nano_teacher_checkpoint,
    sha256_file,
    stage211_phase_train_config_contract,
    stage211_post_coverage_correction_lr,
    validate_stage211_public_benchmark,
    validate_stage211_public_overlap_binding,
)
from rwkvasr.eval.stage211_public_metrics import replay_stage211_sft_public_evidence
from rwkvasr.eval.stage211_runtime import audit_stage211_runtime_epoch_coverage

try:
    from scripts.build_stage211_sft_correction_profile import (
        TRAIN_BATCH_SIZE,
        TRAIN_FRAME_BUDGET,
        TRAIN_WORLD_SIZE,
        validate_correction_profile,
    )
    from scripts.create_stage211_curriculum_receipt import (
        audit_stage211_checkpoint_delta,
    )
    from scripts.run_stage211_labeled_sft import (
        _checkpoint_step,
        _validate_completion as validate_full_sft_completion,
    )
    from scripts.run_stage211_strict_chained_alignment import (
        PHASES,
        _audit_nano_non_attention_exact,
        _config as build_stage211_config,
        _latest_step,
        _run,
        _segments,
        _validate_output_storage,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from build_stage211_sft_correction_profile import (
        TRAIN_BATCH_SIZE,
        TRAIN_FRAME_BUDGET,
        TRAIN_WORLD_SIZE,
        validate_correction_profile,
    )
    from create_stage211_curriculum_receipt import audit_stage211_checkpoint_delta
    from run_stage211_labeled_sft import (
        _checkpoint_step,
        _validate_completion as validate_full_sft_completion,
    )
    from run_stage211_strict_chained_alignment import (
        PHASES,
        _audit_nano_non_attention_exact,
        _config as build_stage211_config,
        _latest_step,
        _run,
        _segments,
        _validate_output_storage,
    )


MAX_CORRECTION_ROUNDS = 3
DEFAULT_PROFILE = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_sft_source_balanced_correction_v1"
    / "stage211_sft_correction_profile.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "rwkvasr_runs" / "stage211_full_alignment" / "stage211d_sft_correction"
)
DEFAULT_CONFIG_ROOT = Path.home() / "rwkvasr_configs" / "stage211_full_alignment"
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
BAD_LOG_PATTERNS = (
    re.compile(r"Traceback"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemory"),
    re.compile(r"\bloss=(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"\bonline_[a-z0-9_]*missing=[1-9][0-9]*\b"),
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"Refusing to overwrite a different Stage211D artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _current_attempt_log(log_path: Path) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    attempt_start = text.rfind("[rwkvasr] Distributed init complete.")
    return text[attempt_start:] if attempt_start >= 0 else text


def _validate_log(log_path: Path, *, required_step: int) -> tuple[str, float]:
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise ValueError(f"Stage211D correction log is missing: {log_path}")
    text = _current_attempt_log(log_path)
    if f"[deepspeed-train] step={required_step}" not in text:
        raise ValueError(f"Stage211D correction log lacks optimizer step {required_step}.")
    for pattern in BAD_LOG_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            raise ValueError(
                f"Stage211D correction log contains a rejected condition: {match.group(0)}"
            )
    peak_values = [
        float(value)
        for value in re.findall(r"peak_reserved=([0-9]+(?:\.[0-9]+)?)GiB", text)
    ]
    if not peak_values:
        raise ValueError("Stage211D correction log lacks peak-reserved memory telemetry.")
    return text, max(peak_values)


def _validate_failed_full_sft_report(
    report_path: Path,
    *,
    full_completion_path: Path,
    full_completion: Mapping[str, Any],
    full_checkpoint: Path,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = _load_json(report_path, label="Stage211D failed full-SFT report")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "final_completion",
        "phase": "sft",
        "complete": False,
        "gate_passed": False,
        "checkpoint_path": str(full_checkpoint),
        "checkpoint_sha256": sha256_file(full_checkpoint),
        "sft_completion_path": str(full_completion_path),
        "sft_completion_sha256": sha256_file(full_completion_path),
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211D correction admission does not bind the failed full SFT.")
    logits_checkpoint = Path(str(full_completion.get("init_checkpoint_path") or "")).resolve()
    replayed = replay_stage211_sft_public_evidence(
        report,
        benchmarks=STAGE211_PUBLIC_BENCHMARKS,
        expected_baseline_checkpoint=logits_checkpoint,
        expected_candidate_checkpoint=full_checkpoint,
    )
    benchmark = validate_stage211_public_benchmark(
        replayed["public_benchmark"],
        require_metric_source_recomputed=True,
    )
    validate_stage211_public_overlap_binding(
        report.get("public_overlap"),
        public_benchmark=benchmark,
    )
    progress = replayed["public_progress"]
    if progress.get("gate_passed") is True and benchmark.get("all_datasets_pass") is True:
        raise ValueError("Stage211D correction cannot admit a full SFT that already passed.")
    return report


def validate_failed_admission(
    report_path: Path,
    *,
    round_index: int,
    full_completion_path: Path,
    full_completion: Mapping[str, Any],
    full_checkpoint: Path,
    init_checkpoint: Path,
    correction_profile_path: Path,
) -> dict[str, Any]:
    if round_index == 1:
        if init_checkpoint != full_checkpoint:
            raise ValueError("Stage211D correction round 1 must initialize from full SFT.")
        return _validate_failed_full_sft_report(
            report_path,
            full_completion_path=full_completion_path,
            full_completion=full_completion,
            full_checkpoint=full_checkpoint,
        )
    try:
        from scripts.evaluate_stage211_sft_correction import (
            validate_correction_evaluation_report,
        )
    except ModuleNotFoundError as error:
        if error.name != "scripts":
            raise
        from evaluate_stage211_sft_correction import (
            validate_correction_evaluation_report,
        )

    report = validate_correction_evaluation_report(
        report_path,
        expected_round=round_index - 1,
        expected_full_completion_path=full_completion_path,
        expected_correction_profile_path=correction_profile_path,
        require_passed=False,
    )
    if report.get("gate_passed") is not False:
        raise ValueError("Stage211D correction cannot continue after a passed correction gate.")
    previous_checkpoint = Path(str(report.get("checkpoint_path") or "")).resolve()
    if previous_checkpoint != init_checkpoint:
        raise ValueError("Stage211D correction does not initialize from the previous round.")
    return report


def _phase() -> Any:
    return replace(PHASES["sft"], lr=stage211_post_coverage_correction_lr("sft"))


def _segment(*, round_index: int, steps_per_epoch: int, smoke: bool) -> dict[str, Any]:
    segment = _segments(
        phase=_phase(),
        smoke=smoke,
        formal_steps=steps_per_epoch,
        difficulty="easy",
        full_data_profile=False,
    )[0]
    target = 2 if smoke else steps_per_epoch
    return {
        **segment,
        "name": f"sft_correction_round{round_index}_{target}steps",
        "difficulty": f"sft_correction_round_{round_index:02d}",
        "split_steps": target,
        "target_step": target,
    }


def build_correction_train_config(
    *,
    round_index: int,
    steps_per_epoch: int,
    output_dir: Path,
    init_checkpoint: Path,
    correction_profile_path: Path,
    correction_profile: Mapping[str, Any],
    full_completion_path: Path,
    admission_report_path: Path,
    nano_checkpoint: Path,
    resume: bool,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    segment = _segment(
        round_index=round_index,
        steps_per_epoch=steps_per_epoch,
        smoke=smoke,
    )
    full_binding = correction_profile["full_labeled_profile"]
    labeled_root = Path(str(full_binding["labeled_root"])).resolve()
    length_index = Path(str(correction_profile["length_index_path"])).resolve()
    manifest = Path(str(correction_profile["bucket_manifest_path"])).resolve()
    config = build_stage211_config(
        phase=_phase(),
        segment=segment,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        bucket_manifest=manifest,
        resume=resume,
        smoke=smoke,
        nano_checkpoint=nano_checkpoint,
        labeled_webdataset_root=labeled_root,
        labeled_length_index=length_index,
        full_data_profile=False,
        post_coverage_correction=True,
    )
    config.update(
        {
            "stage211_post_coverage_correction_phase": "sft",
            "stage211_post_coverage_correction_round": round_index,
            "stage211_sft_correction_profile_path": str(correction_profile_path),
            "stage211_sft_correction_profile_sha256": sha256_file(correction_profile_path),
            "stage211_full_sft_completion_path": str(full_completion_path),
            "stage211_full_sft_completion_sha256": sha256_file(full_completion_path),
            "stage211_post_coverage_admission_gate_path": str(admission_report_path),
            "stage211_post_coverage_admission_gate_sha256": sha256_file(
                admission_report_path
            ),
            "stage211_post_coverage_original_coverage_unchanged": True,
        }
    )
    return config, segment


def _validate_smoke_marker(
    marker_path: Path,
    *,
    round_index: int,
    init_checkpoint: Path,
    correction_profile_path: Path,
    full_completion_path: Path,
    admission_report_path: Path,
    nano_checkpoint: Path,
) -> dict[str, Any]:
    marker = _load_json(marker_path, label="Stage211D correction smoke marker")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "sft_correction_smoke",
        "phase": "sft",
        "complete": True,
        "round": round_index,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "correction_profile_path": str(correction_profile_path),
        "correction_profile_sha256": sha256_file(correction_profile_path),
        "full_sft_completion_path": str(full_completion_path),
        "full_sft_completion_sha256": sha256_file(full_completion_path),
        "admission_report_path": str(admission_report_path),
        "admission_report_sha256": sha256_file(admission_report_path),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211D correction smoke marker binding mismatch.")
    checkpoint = Path(str(marker.get("smoke_checkpoint_path") or "")).resolve()
    log_path = Path(str(marker.get("smoke_log_path") or "")).resolve()
    if (
        not checkpoint.is_file()
        or marker.get("smoke_checkpoint_sha256") != sha256_file(checkpoint)
        or _checkpoint_step(checkpoint) != 2
    ):
        raise ValueError("Stage211D correction smoke checkpoint is unavailable or changed.")
    _, peak = _validate_log(log_path, required_step=2)
    limit = float(marker.get("max_peak_reserved_gib", float("nan")))
    if (
        not math.isfinite(limit)
        or limit <= 0.0
        or peak > limit
        or float(marker.get("peak_reserved_gib", float("nan"))) != peak
        or marker.get("smoke_log_sha256") != sha256_file(log_path)
    ):
        raise ValueError("Stage211D correction smoke memory/log contract mismatch.")
    return marker


def _run_smoke(
    *,
    round_index: int,
    run_dir: Path,
    config_dir: Path,
    init_checkpoint: Path,
    correction_profile_path: Path,
    correction_profile: Mapping[str, Any],
    full_completion_path: Path,
    admission_report_path: Path,
    nano_checkpoint: Path,
    steps_per_epoch: int,
    master_port: int,
    max_peak_reserved_gib: float,
    formal_latest_step: int,
    dry_run: bool,
) -> Path:
    smoke_dir = Path(f"{run_dir}_smoke")
    marker_path = run_dir.parent / f"round_{round_index:02d}_smoke_passed.json"
    if marker_path.is_file() and not dry_run:
        _validate_smoke_marker(
            marker_path,
            round_index=round_index,
            init_checkpoint=init_checkpoint,
            correction_profile_path=correction_profile_path,
            full_completion_path=full_completion_path,
            admission_report_path=admission_report_path,
            nano_checkpoint=nano_checkpoint,
        )
        return marker_path
    if formal_latest_step > 0 and not dry_run:
        raise ValueError("Stage211D correction formal run lacks its preflight smoke marker.")
    config, segment = build_correction_train_config(
        round_index=round_index,
        steps_per_epoch=steps_per_epoch,
        output_dir=smoke_dir,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
        correction_profile=correction_profile,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        nano_checkpoint=nano_checkpoint,
        resume=_latest_step(smoke_dir) > 0,
        smoke=True,
    )
    smoke_config_dir = config_dir / "smoke"
    smoke_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = smoke_config_dir / f"stage211_{segment['name']}.yaml"
    save_yaml(config_path, config)
    if _latest_step(smoke_dir) < 2:
        code = _run(
            config_path,
            smoke_dir / "logs" / f"{segment['name']}.log",
            dry_run=dry_run,
            master_port=master_port,
        )
        if code != 0:
            raise RuntimeError(f"Stage211D correction smoke exited with code={code}.")
    if dry_run:
        return marker_path
    checkpoint = smoke_dir / "step-2.pt"
    log_path = smoke_dir / "logs" / f"{segment['name']}.log"
    if not checkpoint.is_file() or _checkpoint_step(checkpoint) != 2:
        raise ValueError("Stage211D correction smoke lacks its exact step-2 checkpoint.")
    _, peak = _validate_log(log_path, required_step=2)
    if peak > max_peak_reserved_gib:
        raise ValueError(
            f"Stage211D correction smoke peak {peak:.2f} GiB exceeds "
            f"{max_peak_reserved_gib:.2f} GiB."
        )
    marker = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "sft_correction_smoke",
        "phase": "sft",
        "complete": True,
        "round": round_index,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "correction_profile_path": str(correction_profile_path),
        "correction_profile_sha256": sha256_file(correction_profile_path),
        "full_sft_completion_path": str(full_completion_path),
        "full_sft_completion_sha256": sha256_file(full_completion_path),
        "admission_report_path": str(admission_report_path),
        "admission_report_sha256": sha256_file(admission_report_path),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "smoke_checkpoint_path": str(checkpoint),
        "smoke_checkpoint_sha256": sha256_file(checkpoint),
        "smoke_log_path": str(log_path),
        "smoke_log_sha256": sha256_file(log_path),
        "peak_reserved_gib": peak,
        "max_peak_reserved_gib": max_peak_reserved_gib,
    }
    _write_immutable_json(marker_path, marker)
    _validate_smoke_marker(
        marker_path,
        round_index=round_index,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        nano_checkpoint=nano_checkpoint,
    )
    return marker_path


def _provenance(
    *,
    round_index: int,
    run_dir: Path,
    init_checkpoint: Path,
    correction_profile_path: Path,
    full_completion_path: Path,
    admission_report_path: Path,
    nano_checkpoint: Path,
    smoke_marker_path: Path,
    steps_per_epoch: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "sft_correction_run",
        "phase": "sft",
        "round": round_index,
        "run_dir": str(run_dir),
        "epochs": 1,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": stage211_post_coverage_correction_lr("sft"),
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "correction_profile_path": str(correction_profile_path),
        "correction_profile_sha256": sha256_file(correction_profile_path),
        "full_sft_completion_path": str(full_completion_path),
        "full_sft_completion_sha256": sha256_file(full_completion_path),
        "admission_report_path": str(admission_report_path),
        "admission_report_sha256": sha256_file(admission_report_path),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "smoke_marker_path": str(smoke_marker_path),
        "smoke_marker_sha256": sha256_file(smoke_marker_path),
    }


def _validate_correction_train_config(
    config: Mapping[str, Any],
    *,
    round_index: int,
    profile: Mapping[str, Any],
    profile_path: Path,
    full_completion_path: Path,
    admission_report_path: Path,
    smoke_marker_path: Path,
    steps_per_epoch: int,
) -> None:
    contract = stage211_phase_train_config_contract("sft")
    contract["lr"] = stage211_post_coverage_correction_lr("sft")
    expected = {
        **contract,
        "max_steps": steps_per_epoch,
        "batch_size": TRAIN_BATCH_SIZE,
        "batch_token_budget": TRAIN_FRAME_BUDGET,
        "length_bucket_frame_budget": TRAIN_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "freeze_encoder": False,
        "freeze_encoder_except_time_mixer": True,
        "freeze_ctc_decoder": True,
        "freeze_ctc_head": True,
        "weight_decay": 0.0,
        "webdataset_root": profile["full_labeled_profile"]["labeled_root"],
        "webdataset_length_index_path": profile["length_index_path"],
        "webdataset_bucket_manifest_path": profile["bucket_manifest_path"],
        "webdataset_split": "train",
        "stage211_post_coverage_correction_phase": "sft",
        "stage211_post_coverage_correction_round": round_index,
        "stage211_sft_correction_profile_path": str(profile_path),
        "stage211_sft_correction_profile_sha256": sha256_file(profile_path),
        "stage211_full_sft_completion_path": str(full_completion_path),
        "stage211_full_sft_completion_sha256": sha256_file(full_completion_path),
        "stage211_post_coverage_admission_gate_path": str(admission_report_path),
        "stage211_post_coverage_admission_gate_sha256": sha256_file(admission_report_path),
        "stage211_post_coverage_original_coverage_unchanged": True,
        "stage211_post_coverage_smoke_marker_path": str(smoke_marker_path),
        "stage211_post_coverage_smoke_marker_sha256": sha256_file(smoke_marker_path),
    }
    for key, value in expected.items():
        if type(config.get(key)) is not type(value) or config.get(key) != value:
            raise ValueError(
                f"Stage211D correction train config {key} mismatch: "
                f"actual={config.get(key)!r} expected={value!r}"
            )


def build_completion_receipt(
    *,
    round_index: int,
    run_dir: Path,
    correction_profile_path: Path,
    full_completion_path: Path,
    admission_report_path: Path,
    init_checkpoint: Path,
    completion_checkpoint: Path,
) -> dict[str, Any]:
    correction_profile = validate_correction_profile(correction_profile_path)
    full_completion, full_checkpoint = validate_full_sft_completion(full_completion_path)
    validate_failed_admission(
        admission_report_path,
        round_index=round_index,
        full_completion_path=full_completion_path,
        full_completion=full_completion,
        full_checkpoint=full_checkpoint,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
    )
    steps_per_epoch = int(correction_profile["estimated_train_steps"])
    if _checkpoint_step(completion_checkpoint) != steps_per_epoch:
        raise ValueError("Stage211D correction completion checkpoint step mismatch.")
    provenance_path = run_dir / "stage211_sft_correction_provenance.json"
    provenance = _load_json(provenance_path, label="Stage211D correction provenance")
    smoke_marker_path = Path(str(provenance.get("smoke_marker_path") or "")).resolve()
    nano_checkpoint = Path(str(provenance.get("nano_teacher_checkpoint_path") or "")).resolve()
    expected_provenance = _provenance(
        round_index=round_index,
        run_dir=run_dir,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        nano_checkpoint=nano_checkpoint,
        smoke_marker_path=smoke_marker_path,
        steps_per_epoch=steps_per_epoch,
    )
    if provenance != expected_provenance:
        raise ValueError("Stage211D correction provenance differs from deep recomputation.")
    _validate_smoke_marker(
        smoke_marker_path,
        round_index=round_index,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        nano_checkpoint=nano_checkpoint,
    )
    train_config_path = run_dir / "train_config.yaml"
    train_config = load_yaml(train_config_path)
    _validate_correction_train_config(
        train_config,
        round_index=round_index,
        profile=correction_profile,
        profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        smoke_marker_path=smoke_marker_path,
        steps_per_epoch=steps_per_epoch,
    )
    configured_teacher = resolve_stage211_nano_teacher_checkpoint(dict(train_config))
    if configured_teacher != nano_checkpoint:
        raise ValueError("Stage211D correction train config uses a different Nano teacher.")
    segment = _segment(round_index=round_index, steps_per_epoch=steps_per_epoch, smoke=False)
    training_log_path = run_dir / "logs" / f"{segment['name']}.log"
    _validate_log(training_log_path, required_step=steps_per_epoch)
    runtime = audit_stage211_runtime_epoch_coverage(
        run_dir=run_dir,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
    )
    delta = audit_stage211_checkpoint_delta(
        init_checkpoint_path=init_checkpoint,
        completion_checkpoint_path=completion_checkpoint,
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "labeled_sft_correction_completion",
        "phase": "sft",
        "complete": True,
        "round": round_index,
        "epochs": 1,
        "learning_rate": stage211_post_coverage_correction_lr("sft"),
        "batch_size": TRAIN_BATCH_SIZE,
        "world_size": TRAIN_WORLD_SIZE,
        "frame_budget": TRAIN_FRAME_BUDGET,
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
        "rows": correction_profile["train_samples"],
        "row_exposures": correction_profile["train_samples"],
        "hours": correction_profile["total_train_hours"],
        "steps": steps_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "tail_padding_samples": correction_profile["tail_padding_samples"],
        "executed_sample_exposures": correction_profile["executed_sample_exposures"],
        "language_counts": correction_profile["language_counts"],
        "source_counts": correction_profile["source_counts"],
        "full_sft_completion_path": str(full_completion_path),
        "full_sft_completion_sha256": sha256_file(full_completion_path),
        "full_sft_checkpoint_path": str(full_checkpoint),
        "full_sft_checkpoint_sha256": sha256_file(full_checkpoint),
        "correction_profile_path": str(correction_profile_path),
        "correction_profile_sha256": sha256_file(correction_profile_path),
        "admission_report_path": str(admission_report_path),
        "admission_report_sha256": sha256_file(admission_report_path),
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "completion_checkpoint_path": str(completion_checkpoint),
        "completion_checkpoint_sha256": sha256_file(completion_checkpoint),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "train_config_path": str(train_config_path),
        "train_config_sha256": sha256_file(train_config_path),
        "training_log_path": str(training_log_path),
        "training_log_sha256": sha256_file(training_log_path),
        "smoke_marker_path": str(smoke_marker_path),
        "smoke_marker_sha256": sha256_file(smoke_marker_path),
        "runtime_epoch_coverage": runtime,
        "parameter_delta_audit": delta,
    }


def validate_completion_receipt(
    receipt_path: Path,
    *,
    expected_round: int | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    actual = _load_json(receipt_path, label="Stage211D correction completion")
    required = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "labeled_sft_correction_completion",
        "phase": "sft",
        "complete": True,
    }
    if any(actual.get(key) != value for key, value in required.items()):
        raise ValueError("Stage211D correction completion contract mismatch.")
    round_index = int(actual.get("round", -1))
    if expected_round is not None and round_index != expected_round:
        raise ValueError("Stage211D correction completion round mismatch.")
    expected = build_completion_receipt(
        round_index=round_index,
        run_dir=receipt_path.parent,
        correction_profile_path=Path(str(actual["correction_profile_path"])).resolve(),
        full_completion_path=Path(str(actual["full_sft_completion_path"])).resolve(),
        admission_report_path=Path(str(actual["admission_report_path"])).resolve(),
        init_checkpoint=Path(str(actual["init_checkpoint_path"])).resolve(),
        completion_checkpoint=Path(str(actual["completion_checkpoint_path"])).resolve(),
    )
    if actual != expected:
        raise ValueError("Stage211D correction completion differs from deep recomputation.")
    return actual


def run_correction(args: argparse.Namespace) -> Path | None:
    round_index = int(args.round)
    if not 1 <= round_index <= MAX_CORRECTION_ROUNDS:
        raise ValueError(f"Stage211D correction round must be 1..{MAX_CORRECTION_ROUNDS}.")
    correction_profile_path = args.correction_profile.expanduser().resolve()
    full_completion_path = args.full_sft_completion.expanduser().resolve()
    admission_report_path = args.admission_report.expanduser().resolve()
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    nano_checkpoint = args.nano_checkpoint.expanduser().resolve()
    for label, path in (
        ("correction profile", correction_profile_path),
        ("full SFT completion", full_completion_path),
        ("admission report", admission_report_path),
        ("initial checkpoint", init_checkpoint),
        ("Nano checkpoint", nano_checkpoint),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage211D correction {label} is unavailable: {path}")
    profile = validate_correction_profile(correction_profile_path)
    full_completion, full_checkpoint = validate_full_sft_completion(full_completion_path)
    profile_full_receipt = Path(
        str(profile["full_labeled_profile"]["receipt_path"])
    ).resolve()
    completion_full_receipt = Path(
        str(full_completion.get("labeled_profile_receipt_path") or "")
    ).resolve()
    if profile_full_receipt != completion_full_receipt:
        raise ValueError("Correction profile and mandatory full SFT use different labeled data.")
    validate_failed_admission(
        admission_report_path,
        round_index=round_index,
        full_completion_path=full_completion_path,
        full_completion=full_completion,
        full_checkpoint=full_checkpoint,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
    )
    if sha256_file(nano_checkpoint) != full_completion["nano_teacher_checkpoint_sha256"]:
        raise ValueError("Stage211D correction Nano teacher differs from full SFT.")
    steps_per_epoch = int(profile["estimated_train_steps"])
    run_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.output_root.expanduser().resolve() / f"round_{round_index:02d}"
    )
    _validate_output_storage(
        output_dir=run_dir,
        smoke=False,
        dry_run=bool(args.dry_run),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_step = _latest_step(run_dir)
    if latest_step <= 0 and not args.skip_nano_weight_audit:
        audit = _audit_nano_non_attention_exact(
            checkpoint_path=init_checkpoint,
            nano_checkpoint_path=nano_checkpoint,
        )
        print(f"nano_non_attention_audit={audit['matched']}/{audit['expected']} exact")
    config_dir = args.config_dir.expanduser().resolve() / "sft" / (
        f"correction_round_{round_index:02d}"
    )
    smoke_marker_path = _run_smoke(
        round_index=round_index,
        run_dir=run_dir,
        config_dir=config_dir,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
        correction_profile=profile,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        nano_checkpoint=nano_checkpoint,
        steps_per_epoch=steps_per_epoch,
        master_port=int(args.master_port),
        max_peak_reserved_gib=float(args.max_peak_reserved_gib),
        formal_latest_step=latest_step,
        dry_run=bool(args.dry_run),
    )
    config, segment = build_correction_train_config(
        round_index=round_index,
        steps_per_epoch=steps_per_epoch,
        output_dir=run_dir,
        init_checkpoint=init_checkpoint,
        correction_profile_path=correction_profile_path,
        correction_profile=profile,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        nano_checkpoint=nano_checkpoint,
        resume=latest_step > 0,
        smoke=False,
    )
    if not args.dry_run:
        config.update(
            {
                "stage211_post_coverage_smoke_marker_path": str(smoke_marker_path),
                "stage211_post_coverage_smoke_marker_sha256": sha256_file(smoke_marker_path),
            }
        )
        provenance_path = run_dir / "stage211_sft_correction_provenance.json"
        provenance = _provenance(
            round_index=round_index,
            run_dir=run_dir,
            init_checkpoint=init_checkpoint,
            correction_profile_path=correction_profile_path,
            full_completion_path=full_completion_path,
            admission_report_path=admission_report_path,
            nano_checkpoint=nano_checkpoint,
            smoke_marker_path=smoke_marker_path,
            steps_per_epoch=steps_per_epoch,
        )
        _write_immutable_json(provenance_path, provenance)
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"stage211_{segment['name']}.yaml"
    save_yaml(config_path, config)
    print(
        "[stage211-sft-correction] "
        f"round={round_index} rows={profile['train_samples']} steps={steps_per_epoch} "
        f"latest_step={latest_step} run_dir={run_dir}",
        flush=True,
    )
    if latest_step < steps_per_epoch:
        code = _run(
            config_path,
            run_dir / "logs" / f"{segment['name']}.log",
            dry_run=bool(args.dry_run),
            master_port=int(args.master_port),
        )
        if code != 0:
            raise RuntimeError(f"Stage211D correction training exited with code={code}.")
    if args.dry_run:
        return None
    completion_checkpoint = run_dir / f"step-{steps_per_epoch}.pt"
    receipt = build_completion_receipt(
        round_index=round_index,
        run_dir=run_dir,
        correction_profile_path=correction_profile_path,
        full_completion_path=full_completion_path,
        admission_report_path=admission_report_path,
        init_checkpoint=init_checkpoint,
        completion_checkpoint=completion_checkpoint,
    )
    receipt_path = run_dir / "sft_correction_complete.json"
    _write_immutable_json(receipt_path, receipt)
    validate_completion_receipt(receipt_path, expected_round=round_index)
    print(
        f"[stage211-sft-correction] complete round={round_index} "
        f"checkpoint={completion_checkpoint} receipt={receipt_path}",
        flush=True,
    )
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one strict balanced-label Stage211D correction epoch."
    )
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--correction-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--full-sft-completion", type=Path, required=True)
    parser.add_argument("--admission-report", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--master-port", type=int, default=29651)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=22.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-nano-weight-audit", action="store_true")
    args = parser.parse_args()
    if args.skip_nano_weight_audit and not args.dry_run:
        parser.error("--skip-nano-weight-audit is allowed only with --dry-run")
    if args.max_peak_reserved_gib <= 0:
        parser.error("--max-peak-reserved-gib must be positive")
    run_correction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
