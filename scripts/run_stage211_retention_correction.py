from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from rwkvasr.config import save_yaml
from rwkvasr.data import (
    estimate_bucket_manifest_steps,
    estimate_bucket_manifest_tail_padding_samples,
    load_webdataset_bucket_manifest,
)
from rwkvasr.eval.stage211_gate import (
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    sha256_file,
    validate_stage211_phase_gate_report,
)

try:
    from scripts.create_stage211_retention_correction_receipt import (
        CORRECTION_LR,
        MAX_CORRECTION_ROUNDS,
        build_receipt,
    )
    from scripts.run_stage211_strict_chained_alignment import (
        DEFAULT_CONFIG_DIR,
        NANO_CHECKPOINT,
        PHASES,
        _audit_nano_non_attention_exact,
        _config as _stage211_config,
        _latest_step,
        _run,
        _segments,
        _validate_output_storage,
        _validate_target_nano_teacher_checkpoint,
    )
    from scripts.validate_stage211_retention_replay import (
        validate_retention_replay,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_retention_correction_receipt import (
        CORRECTION_LR,
        MAX_CORRECTION_ROUNDS,
        build_receipt,
    )
    from run_stage211_strict_chained_alignment import (
        DEFAULT_CONFIG_DIR,
        NANO_CHECKPOINT,
        PHASES,
        _audit_nano_non_attention_exact,
        _config as _stage211_config,
        _latest_step,
        _run,
        _segments,
        _validate_output_storage,
        _validate_target_nano_teacher_checkpoint,
    )
    from validate_stage211_retention_replay import validate_retention_replay


DEFAULT_REPLAY_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "retention_replay_v1"
    / "receipt.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / "rwkvasr_runs"
    / "stage211_full_alignment"
    / "stage211a_mixer_retention_correction"
)


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _admit_failed_gate(
    *,
    admission_gate_path: Path,
    init_checkpoint: Path,
    round_index: int,
) -> tuple[dict[str, Any], str]:
    gate = validate_stage211_phase_gate_report(
        admission_gate_path,
        expected_phase="mixer",
        checkpoint_path=init_checkpoint,
        require_passed=False,
    )
    if gate.get("gate_passed") is not False:
        raise ValueError("Stage211 correction requires an explicitly failed Mixer gate.")
    coverage = gate.get("full_data_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Stage211 correction admission gate lacks full-data coverage.")
    corrections = coverage.get("post_coverage_corrections", [])
    if not isinstance(corrections, list) or len(corrections) != round_index - 1:
        raise ValueError(
            "Stage211 correction round does not immediately follow the admission gate."
        )
    segments = coverage.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Stage211 correction admission gate lacks curriculum segments.")
    teacher_values = {
        str(segment.get("nano_teacher_checkpoint_sha256") or "")
        for segment in segments
        if isinstance(segment, dict)
    }
    if len(teacher_values) != 1 or len(next(iter(teacher_values), "")) != 64:
        raise ValueError("Stage211 correction admission gate has mixed Nano teachers.")
    return gate, next(iter(teacher_values))


def _audit_replay_storage(replay: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(str(replay.get("manifest_path") or "")).resolve()
    if (
        not manifest_path.is_file()
        or replay.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("Stage211 correction replay manifest is missing or changed.")
    manifest = load_webdataset_bucket_manifest(manifest_path)
    part_paths = [
        Path(part.path).resolve()
        for buckets in manifest.splits.values()
        for bucket in buckets
        for part in bucket.parts
    ]
    missing = [path for path in part_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Stage211 correction replay parts are missing: {missing[:3]}"
        )
    split_samples = {
        split: sum(bucket.num_samples for bucket in buckets)
        for split, buckets in manifest.splits.items()
    }
    if split_samples.get("train") != int(replay.get("validated_unique_keys", -1)):
        raise ValueError("Stage211 correction replay train count mismatch.")
    if split_samples.get("eval") != 256:
        raise ValueError("Stage211 correction replay must bind 256 fixed eval rows.")
    return manifest_path, {
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": sha256_file(manifest_path),
        "webdataset_root": "/",
        "length_index_path": "/dev/null",
        "length_index_size": 0,
        "partitioned_length_index": True,
        "bucket_part_files": len(part_paths),
        "split_samples": split_samples,
    }


def _correction_segment(*, round_index: int, steps_per_epoch: int) -> dict[str, Any]:
    phase = replace(PHASES["mixer"], lr=CORRECTION_LR)
    source = _segments(
        phase=phase,
        smoke=False,
        difficulty="easy",
        full_data_profile=True,
    )[0]
    return {
        **source,
        "name": f"mixer_retention_round{round_index}_{steps_per_epoch}steps",
        "difficulty": f"retention_round_{round_index:02d}",
        "split_steps": steps_per_epoch,
        "target_step": steps_per_epoch,
    }


def _provenance_payload(
    *,
    round_index: int,
    run_dir: Path,
    replay_receipt: Path,
    replay_manifest: Path,
    admission_gate: Path,
    init_checkpoint: Path,
    nano_checkpoint: Path,
    steps_per_epoch: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_correction_run",
        "phase": "mixer",
        "round": round_index,
        "run_dir": str(run_dir),
        "replay_receipt_path": str(replay_receipt),
        "replay_receipt_sha256": sha256_file(replay_receipt),
        "replay_manifest_path": str(replay_manifest),
        "replay_manifest_sha256": sha256_file(replay_manifest),
        "admission_gate_path": str(admission_gate),
        "admission_gate_sha256": sha256_file(admission_gate),
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        "epochs": 1,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": CORRECTION_LR,
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
    }


def run_correction(args: argparse.Namespace) -> Path | None:
    round_index = int(args.round)
    if not 1 <= round_index <= MAX_CORRECTION_ROUNDS:
        raise ValueError(f"Stage211 correction round must be 1..{MAX_CORRECTION_ROUNDS}.")
    replay_receipt = args.replay_receipt.expanduser().resolve()
    admission_gate = args.admission_gate.expanduser().resolve()
    init_checkpoint = args.init_checkpoint.expanduser().resolve()
    nano_checkpoint = args.nano_checkpoint.expanduser().resolve()
    for label, path in (
        ("replay receipt", replay_receipt),
        ("admission gate", admission_gate),
        ("initial checkpoint", init_checkpoint),
        ("Nano checkpoint", nano_checkpoint),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage211 correction {label} is unavailable: {path}")

    replay = validate_retention_replay(replay_receipt)
    replay_manifest, audio_data_audit = _audit_replay_storage(replay)
    _, teacher_sha256 = _admit_failed_gate(
        admission_gate_path=admission_gate,
        init_checkpoint=init_checkpoint,
        round_index=round_index,
    )
    _validate_target_nano_teacher_checkpoint(
        recorded_sha256=teacher_sha256,
        nano_checkpoint_path=nano_checkpoint,
        label=f"mixer retention correction round {round_index}",
    )
    manifest = load_webdataset_bucket_manifest(replay_manifest)
    steps_per_epoch = estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        drop_last=False,
    )
    tail_padding = estimate_bucket_manifest_tail_padding_samples(
        manifest,
        split="train",
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
    )
    run_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (args.output_root.expanduser().resolve() / f"round_{round_index:02d}")
    )
    _validate_output_storage(
        output_dir=run_dir,
        smoke=False,
        dry_run=bool(args.dry_run),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = run_dir / "stage211_correction_provenance.json"
    provenance = _provenance_payload(
        round_index=round_index,
        run_dir=run_dir,
        replay_receipt=replay_receipt,
        replay_manifest=replay_manifest,
        admission_gate=admission_gate,
        init_checkpoint=init_checkpoint,
        nano_checkpoint=nano_checkpoint,
        steps_per_epoch=steps_per_epoch,
    )
    if provenance_path.is_file():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise ValueError(
                f"Stage211 correction provenance cannot change in place: {provenance_path}"
            )
    elif not args.dry_run:
        _write_immutable_json(provenance_path, provenance)

    latest_step = _latest_step(run_dir)
    if latest_step <= 0 and not args.skip_nano_weight_audit:
        audit = _audit_nano_non_attention_exact(
            checkpoint_path=init_checkpoint,
            nano_checkpoint_path=nano_checkpoint,
        )
        print(
            f"nano_non_attention_audit={audit['matched']}/{audit['expected']} exact",
            flush=True,
        )
    phase = replace(PHASES["mixer"], lr=CORRECTION_LR)
    segment = _correction_segment(
        round_index=round_index,
        steps_per_epoch=steps_per_epoch,
    )
    config = _stage211_config(
        phase=phase,
        segment=segment,
        output_dir=run_dir,
        init_checkpoint=init_checkpoint,
        bucket_manifest=replay_manifest,
        resume=latest_step > 0,
        smoke=False,
        nano_checkpoint=nano_checkpoint,
        audio_data_audit=audio_data_audit,
        full_data_profile=True,
    )
    config.update(
        {
            "stage211_post_coverage_correction_round": round_index,
            "stage211_post_coverage_replay_receipt_path": str(replay_receipt),
            "stage211_post_coverage_admission_gate_path": str(admission_gate),
            "stage211_post_coverage_original_coverage_unchanged": True,
        }
    )
    config_dir = (
        args.config_dir.expanduser().resolve()
        / "mixer"
        / f"retention_round_{round_index:02d}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"stage211_{segment['name']}.yaml"
    save_yaml(config_path, config)
    print(
        "stage211_retention_correction "
        f"round={round_index} rows={replay['validated_unique_keys']} "
        f"steps={steps_per_epoch} tail_padding={tail_padding} "
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
            raise RuntimeError(f"Stage211 correction training exited with code={code}.")
    if args.dry_run:
        return None

    completion_checkpoint = run_dir / f"step-{steps_per_epoch}.pt"
    if (
        not completion_checkpoint.is_file()
        or _checkpoint_step(completion_checkpoint) != steps_per_epoch
    ):
        raise ValueError(
            "Stage211 correction lacks the exact full-round completion checkpoint: "
            f"{completion_checkpoint}"
        )
    receipt = build_receipt(
        round_index=round_index,
        run_dir=run_dir,
        replay_receipt_path=replay_receipt,
        admission_gate_path=admission_gate,
        init_checkpoint_path=init_checkpoint,
        completion_checkpoint_path=completion_checkpoint,
    )
    receipt_path = run_dir / "correction_receipt.json"
    _write_immutable_json(receipt_path, receipt)
    print(
        f"stage211_retention_correction_complete round={round_index} "
        f"checkpoint={completion_checkpoint} receipt={receipt_path}",
        flush=True,
    )
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one strict Stage211A post-coverage retention correction round."
    )
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument(
        "--replay-receipt",
        type=Path,
        default=DEFAULT_REPLAY_RECEIPT,
    )
    parser.add_argument("--admission-gate", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--nano-checkpoint", type=Path, default=NANO_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-nano-weight-audit", action="store_true")
    args = parser.parse_args()
    if args.skip_nano_weight_audit and not args.dry_run:
        parser.error("--skip-nano-weight-audit is allowed only with --dry-run")
    run_correction(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
