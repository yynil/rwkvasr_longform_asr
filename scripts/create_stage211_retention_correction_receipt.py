from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from rwkvasr.config import load_yaml
from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)
from rwkvasr.eval.stage211_gate import (
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    STAGE211_RETENTION_CORRECTION_EPOCHS,
    STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS,
    STAGE211_RETENTION_CORRECTION_LR,
    STAGE211_RETENTION_CORRECTION_MAX_ROUNDS,
    resolve_stage211_nano_teacher_checkpoint,
    sha256_file,
    stage211_post_coverage_correction_lr,
    stage211_phase_train_config_contract,
    validate_stage211_correction_extension_decision,
    validate_stage211_phase_gate_report,
)
from rwkvasr.eval.stage211_runtime import audit_stage211_runtime_epoch_coverage

try:
    from scripts.create_stage211_curriculum_receipt import (
        audit_stage211_checkpoint_delta,
    )
    from scripts.validate_stage211_retention_replay import (
        validate_retention_replay,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_curriculum_receipt import (
        audit_stage211_checkpoint_delta,
    )
    from validate_stage211_retention_replay import validate_retention_replay


MAX_CORRECTION_ROUNDS = STAGE211_RETENTION_CORRECTION_MAX_ROUNDS
CORRECTION_EPOCHS = STAGE211_RETENTION_CORRECTION_EPOCHS
CORRECTION_LR = STAGE211_RETENTION_CORRECTION_LR


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_correction_smoke_marker(
    marker_path: Path,
    *,
    round_index: int,
    replay_receipt: Path,
    replay_manifest: Path,
    admission_gate: Path,
    init_checkpoint: Path,
    nano_checkpoint: Path,
    phase: str = "mixer",
) -> dict[str, Any]:
    marker = _load_json(marker_path, label="Stage211 correction smoke marker")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "full_profile_smoke",
        "phase": phase,
        "complete": True,
        "correction_round": round_index,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "easy_manifest_path": str(replay_manifest),
        "easy_manifest_sha256": sha256_file(replay_manifest),
        "replay_receipt_path": str(replay_receipt),
        "replay_receipt_sha256": sha256_file(replay_receipt),
        "admission_gate_path": str(admission_gate),
        "admission_gate_sha256": sha256_file(admission_gate),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 correction smoke marker contract mismatch.")
    for path_key, sha_key in (
        ("smoke_checkpoint_path", "smoke_checkpoint_sha256"),
        ("smoke_log_path", "smoke_log_sha256"),
    ):
        path = Path(str(marker.get(path_key) or "")).resolve()
        if not path.is_file() or marker.get(sha_key) != sha256_file(path):
            raise ValueError("Stage211 correction smoke artifact is unavailable or changed.")
    peak = float(marker.get("peak_reserved_gib", float("nan")))
    peak_limit = float(marker.get("max_peak_reserved_gib", float("nan")))
    if (
        not math.isfinite(peak)
        or not math.isfinite(peak_limit)
        or peak < 0.0
        or peak_limit <= 0.0
        or peak > peak_limit
    ):
        raise ValueError("Stage211 correction smoke memory contract mismatch.")
    return marker


def _validate_correction_train_config(
    config: dict[str, Any],
    *,
    round_index: int,
    steps_per_epoch: int,
    replay_manifest: Path,
    replay_receipt: Path,
    admission_gate: Path,
    init_checkpoint: Path,
    smoke_marker: Path,
    phase: str = "mixer",
) -> None:
    correction_lr = stage211_post_coverage_correction_lr(phase)
    configured_phase = config.get("stage211_post_coverage_correction_phase", "mixer")
    if configured_phase != phase:
        raise ValueError(
            "Stage211 correction train config phase mismatch: "
            f"actual={configured_phase!r} expected={phase!r}"
        )
    contract = stage211_phase_train_config_contract(phase)
    contract["lr"] = correction_lr
    for key, expected in contract.items():
        actual = config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Stage211 correction train config {key} mismatch: "
                f"actual={actual!r} expected={expected!r}"
            )
    expected_fields = {
        "max_steps": steps_per_epoch,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "batch_token_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "freeze_encoder": False,
        "freeze_encoder_except_time_mixer": True,
        "freeze_ctc_decoder": True,
        "freeze_ctc_head": True,
        "weight_decay": 0.0,
        "webdataset_bucket_manifest_path": str(replay_manifest),
        "webdataset_split": "train",
        "stage211_post_coverage_correction_round": round_index,
        "stage211_post_coverage_replay_receipt_path": str(replay_receipt),
        "stage211_post_coverage_admission_gate_path": str(admission_gate),
        "stage211_post_coverage_original_coverage_unchanged": True,
        "stage211_post_coverage_smoke_marker_path": str(smoke_marker),
        "stage211_post_coverage_smoke_marker_sha256": sha256_file(smoke_marker),
    }
    for key, expected in expected_fields.items():
        if config.get(key) != expected:
            raise ValueError(
                f"Stage211 correction train config {key} mismatch: "
                f"actual={config.get(key)!r} expected={expected!r}"
            )
    init_path = config.get("init_checkpoint_path")
    resume_from = config.get("resume_from")
    if not (
        (init_path == str(init_checkpoint) and resume_from is None)
        or (init_path is None and resume_from == "latest")
    ):
        raise ValueError(
            "Stage211 correction train config does not bind its initial checkpoint "
            "or an in-run latest resume."
        )


def _admission_teacher_sha256(admission_gate: dict[str, Any]) -> str:
    coverage = admission_gate.get("full_data_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 correction admission gate lacks full-data coverage.")
    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 correction admission gate lacks curriculum segments.")
    values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "")
        for segment in segments
        if isinstance(segment, dict)
    }
    if len(values) != 1 or len(next(iter(values), "")) != 64:
        raise ValueError("Stage211 correction admission gate has mixed Nano teachers.")
    return next(iter(values))


def build_receipt(
    *,
    round_index: int,
    run_dir: Path,
    replay_receipt_path: Path,
    admission_gate_path: Path,
    init_checkpoint_path: Path,
    completion_checkpoint_path: Path,
    phase: str = "mixer",
) -> dict[str, Any]:
    correction_lr = stage211_post_coverage_correction_lr(phase)
    if not 1 <= round_index <= MAX_CORRECTION_ROUNDS:
        raise ValueError(f"Stage211 correction round must be 1..{MAX_CORRECTION_ROUNDS}.")
    run_dir = run_dir.expanduser().resolve()
    replay_receipt_path = replay_receipt_path.expanduser().resolve()
    admission_gate_path = admission_gate_path.expanduser().resolve()
    init_checkpoint_path = init_checkpoint_path.expanduser().resolve()
    completion_checkpoint_path = completion_checkpoint_path.expanduser().resolve()
    for label, path, is_directory in (
        ("run directory", run_dir, True),
        ("replay receipt", replay_receipt_path, False),
        ("admission gate", admission_gate_path, False),
        ("initial checkpoint", init_checkpoint_path, False),
        ("completion checkpoint", completion_checkpoint_path, False),
    ):
        exists = path.is_dir() if is_directory else path.is_file()
        if not exists:
            raise FileNotFoundError(f"Stage211 correction {label} is unavailable: {path}")

    replay = validate_retention_replay(replay_receipt_path)
    replay_manifest = Path(str(replay.get("manifest_path") or "")).resolve()
    manifest = load_webdataset_bucket_manifest(replay_manifest)
    rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    eval_rows = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
    if rows != int(replay.get("validated_unique_keys", -1)) or eval_rows != 256:
        raise ValueError("Stage211 correction replay manifest coverage mismatch.")
    steps_per_epoch = estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        drop_last=False,
    )
    tail_padding_samples = estimate_bucket_manifest_tail_padding_samples(
        manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
    )
    if _checkpoint_step(completion_checkpoint_path) != steps_per_epoch:
        raise ValueError(
            "Stage211 correction completion checkpoint step mismatch: "
            f"actual={_checkpoint_step(completion_checkpoint_path)} "
            f"expected={steps_per_epoch}"
        )

    admission_gate = validate_stage211_phase_gate_report(
        admission_gate_path,
        expected_phase=phase,
        checkpoint_path=init_checkpoint_path,
        require_passed=False,
    )
    if admission_gate.get("gate_passed") is not False:
        raise ValueError("Stage211 correction requires an explicitly failed admission gate.")
    admission_coverage = admission_gate.get("full_data_coverage")
    prior_corrections = (
        admission_coverage.get("post_coverage_corrections", [])
        if isinstance(admission_coverage, dict)
        else []
    )
    if not isinstance(prior_corrections, list) or len(prior_corrections) != round_index - 1:
        raise ValueError(
            "Stage211 correction round does not immediately follow its admission gate."
        )

    extension_decision_path: Path | None = None
    if round_index > STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS:
        extension_decision_path = admission_gate_path.parent / "correction_extension_decision.json"
        validate_stage211_correction_extension_decision(
            extension_decision_path,
            phase=phase,
            next_round=round_index,
            admission_gate_path=admission_gate_path,
            admission_gate=admission_gate,
        )

    provenance_path = run_dir / "stage211_correction_provenance.json"
    provenance = _load_json(
        provenance_path,
        label="Stage211 correction provenance",
    )
    smoke_marker_path = Path(str(provenance.get("smoke_marker_path") or "")).resolve()
    if not smoke_marker_path.is_file() or provenance.get("smoke_marker_sha256") != sha256_file(
        smoke_marker_path
    ):
        raise ValueError("Stage211 correction provenance smoke marker is unavailable or changed.")
    expected_provenance = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_correction_run",
        "phase": phase,
        "round": round_index,
        "run_dir": str(run_dir),
        "replay_receipt_path": str(replay_receipt_path),
        "replay_receipt_sha256": sha256_file(replay_receipt_path),
        "admission_gate_path": str(admission_gate_path),
        "admission_gate_sha256": sha256_file(admission_gate_path),
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_checkpoint_sha256": sha256_file(init_checkpoint_path),
        "replay_manifest_path": str(replay_manifest),
        "replay_manifest_sha256": sha256_file(replay_manifest),
        "epochs": CORRECTION_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": correction_lr,
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
        "smoke_marker_path": str(smoke_marker_path),
        "smoke_marker_sha256": sha256_file(smoke_marker_path),
        "correction_extension_decision_path": (
            str(extension_decision_path) if extension_decision_path is not None else None
        ),
        "correction_extension_decision_sha256": (
            sha256_file(extension_decision_path)
            if extension_decision_path is not None
            else None
        ),
    }
    if any(provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Stage211 correction provenance contract mismatch.")

    train_config_path = run_dir / "train_config.yaml"
    if not train_config_path.is_file():
        raise ValueError(f"Stage211 correction train config is missing: {train_config_path}")
    train_config = load_yaml(train_config_path)
    _validate_correction_train_config(
        train_config,
        round_index=round_index,
        steps_per_epoch=steps_per_epoch,
        replay_manifest=replay_manifest,
        replay_receipt=replay_receipt_path,
        admission_gate=admission_gate_path,
        init_checkpoint=init_checkpoint_path,
        smoke_marker=smoke_marker_path,
        phase=phase,
    )
    nano_teacher_checkpoint = resolve_stage211_nano_teacher_checkpoint(train_config)
    nano_teacher_sha256 = sha256_file(nano_teacher_checkpoint)
    if nano_teacher_sha256 != _admission_teacher_sha256(admission_gate):
        raise ValueError("Stage211 correction Nano teacher differs from its admission gate.")
    if (
        provenance.get("nano_teacher_checkpoint_path") != str(nano_teacher_checkpoint)
        or provenance.get("nano_teacher_checkpoint_sha256") != nano_teacher_sha256
    ):
        raise ValueError("Stage211 correction provenance Nano teacher binding mismatch.")
    _validate_correction_smoke_marker(
        smoke_marker_path,
        round_index=round_index,
        replay_receipt=replay_receipt_path,
        replay_manifest=replay_manifest,
        admission_gate=admission_gate_path,
        init_checkpoint=init_checkpoint_path,
        nano_checkpoint=nano_teacher_checkpoint,
        phase=phase,
    )

    runtime_epoch_coverage = audit_stage211_runtime_epoch_coverage(
        run_dir=run_dir,
        epochs=CORRECTION_EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )
    parameter_delta_audit = audit_stage211_checkpoint_delta(
        init_checkpoint_path=init_checkpoint_path,
        completion_checkpoint_path=completion_checkpoint_path,
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "post_coverage_correction",
        "phase": phase,
        "round": round_index,
        "complete": True,
        "epochs": CORRECTION_EPOCHS,
        "learning_rate": correction_lr,
        "batch_size": STAGE211_FULL_DATA_BATCH_SIZE,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": STAGE211_FULL_DATA_FRAME_BUDGET,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "rows": rows,
        "row_exposures": rows,
        "hours": float(replay["total_hours"]),
        "hour_exposures": float(replay["total_hours"]),
        "steps_per_epoch": steps_per_epoch,
        "steps": steps_per_epoch,
        "tail_padding_samples_per_epoch": tail_padding_samples,
        "tail_padding_sample_exposures": tail_padding_samples,
        "executed_sample_exposures": rows + tail_padding_samples,
        "run_dir": str(run_dir),
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "train_config_path": str(train_config_path),
        "train_config_sha256": sha256_file(train_config_path),
        "smoke_marker_path": str(smoke_marker_path),
        "smoke_marker_sha256": sha256_file(smoke_marker_path),
        "replay_receipt_path": str(replay_receipt_path),
        "replay_receipt_sha256": sha256_file(replay_receipt_path),
        "bucket_manifest_path": str(replay_manifest),
        "bucket_manifest_sha256": sha256_file(replay_manifest),
        "admission_gate_path": str(admission_gate_path),
        "admission_gate_sha256": sha256_file(admission_gate_path),
        "correction_extension_decision_path": (
            str(extension_decision_path) if extension_decision_path is not None else None
        ),
        "correction_extension_decision_sha256": (
            sha256_file(extension_decision_path)
            if extension_decision_path is not None
            else None
        ),
        "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint),
        "nano_teacher_checkpoint_sha256": nano_teacher_sha256,
        "init_checkpoint_path": str(init_checkpoint_path),
        "init_checkpoint_sha256": sha256_file(init_checkpoint_path),
        "completion_checkpoint_path": str(completion_checkpoint_path),
        "completion_checkpoint_sha256": sha256_file(completion_checkpoint_path),
        "parameter_delta_audit": parameter_delta_audit,
        "runtime_epoch_coverage": runtime_epoch_coverage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an immutable Stage211 post-coverage correction receipt."
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), default="mixer")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--admission-gate", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--completion-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipt = build_receipt(
        round_index=int(args.round),
        run_dir=args.run_dir,
        replay_receipt_path=args.replay_receipt,
        admission_gate_path=args.admission_gate,
        init_checkpoint_path=args.init_checkpoint,
        completion_checkpoint_path=args.completion_checkpoint,
        phase=str(args.phase),
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if output_path.is_file() and output_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different correction receipt: {output_path}")
    output_path.write_text(rendered, encoding="utf-8")
    print(
        f"stage211_correction_receipt={output_path} round={receipt['round']} "
        f"rows={receipt['rows']} steps={receipt['steps']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
