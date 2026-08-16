from __future__ import annotations

import argparse
import json
import math
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
    STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS,
    build_stage211_correction_layer_focus,
    sha256_file,
    stage211_post_coverage_correction_lr,
    validate_stage211_correction_layer_focus,
    validate_stage211_correction_extension_decision,
    validate_stage211_phase_gate_report,
)

try:
    from scripts.create_stage211_retention_correction_receipt import (
        CORRECTION_LR as _MIXER_CORRECTION_LR,
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
    from scripts.run_stage211_full_phase_curriculum import (
        _audit_smoke as _audit_full_profile_smoke,
    )
    from scripts.run_stage211_full_phase_curriculum import (
        _validate_smoke_marker as _validate_full_profile_smoke_marker,
    )
    from scripts.validate_stage211_retention_replay import (
        validate_retention_replay,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_retention_correction_receipt import (
        CORRECTION_LR as _MIXER_CORRECTION_LR,
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
    from run_stage211_full_phase_curriculum import (
        _audit_smoke as _audit_full_profile_smoke,
    )
    from run_stage211_full_phase_curriculum import (
        _validate_smoke_marker as _validate_full_profile_smoke_marker,
    )
    from validate_stage211_retention_replay import validate_retention_replay


# Preserve the public constant used by existing Mixer correction tooling.
CORRECTION_LR = _MIXER_CORRECTION_LR


DEFAULT_REPLAY_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "retention_replay_v2"
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
    phase: str = "mixer",
) -> tuple[dict[str, Any], str]:
    gate = validate_stage211_phase_gate_report(
        admission_gate_path,
        expected_phase=phase,
        checkpoint_path=init_checkpoint,
        require_passed=False,
    )
    if gate.get("gate_passed") is not False:
        phase_label = "Mixer" if phase == "mixer" else phase.capitalize()
        raise ValueError(f"Stage211 correction requires an explicitly failed {phase_label} gate.")
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
    if not manifest_path.is_file() or replay.get("manifest_sha256") != sha256_file(manifest_path):
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
        raise FileNotFoundError(f"Stage211 correction replay parts are missing: {missing[:3]}")
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


def _correction_segment(
    *,
    round_index: int,
    steps_per_epoch: int,
    phase: str = "mixer",
) -> dict[str, Any]:
    phase_config = replace(PHASES[phase], lr=stage211_post_coverage_correction_lr(phase))
    source = _segments(
        phase=phase_config,
        smoke=False,
        difficulty="easy",
        full_data_profile=True,
    )[0]
    return {
        **source,
        "name": (
            f"mixer_retention_round{round_index}_{steps_per_epoch}steps"
            if phase == "mixer"
            else f"{phase}_correction_round{round_index}_{steps_per_epoch}steps"
        ),
        "difficulty": (
            f"retention_round_{round_index:02d}"
            if phase == "mixer"
            else f"correction_round_{round_index:02d}"
        ),
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
    smoke_marker: Path,
    layer_focus: Path,
    extension_decision: Path | None,
    steps_per_epoch: int,
    phase: str = "mixer",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "retention_correction_run",
        "phase": phase,
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
        "smoke_marker_path": str(smoke_marker),
        "smoke_marker_sha256": sha256_file(smoke_marker),
        "layer_focus_path": str(layer_focus),
        "layer_focus_sha256": sha256_file(layer_focus),
        "correction_extension_decision_path": (
            str(extension_decision) if extension_decision is not None else None
        ),
        "correction_extension_decision_sha256": (
            sha256_file(extension_decision) if extension_decision is not None else None
        ),
        "epochs": 1,
        "steps_per_epoch": steps_per_epoch,
        "learning_rate": stage211_post_coverage_correction_lr(phase),
        "trainable_boundary": "mixer_only",
        "early_stopping": False,
    }


def _correction_config_metadata(
    *,
    round_index: int,
    replay_receipt: Path,
    admission_gate: Path,
    layer_focus: Path,
    phase: str = "mixer",
) -> dict[str, Any]:
    return {
        "stage211_post_coverage_correction_phase": phase,
        "stage211_post_coverage_correction_round": round_index,
        "stage211_post_coverage_replay_receipt_path": str(replay_receipt),
        "stage211_post_coverage_admission_gate_path": str(admission_gate),
        "stage211_post_coverage_layer_focus_path": str(layer_focus),
        "stage211_post_coverage_layer_focus_sha256": sha256_file(layer_focus),
        "stage211_post_coverage_original_coverage_unchanged": True,
    }


def _validate_correction_smoke_marker(
    *,
    marker_path: Path,
    round_index: int,
    init_checkpoint: Path,
    replay_receipt: Path,
    replay_manifest: Path,
    admission_gate: Path,
    layer_focus: Path,
    nano_checkpoint: Path,
    phase: str = "mixer",
) -> dict[str, Any]:
    raw_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(raw_marker, dict):
        raise ValueError("Stage211 correction smoke marker must be a JSON object.")
    smoke_checkpoint = Path(str(raw_marker.get("smoke_checkpoint_path") or "")).resolve()
    marker = _validate_full_profile_smoke_marker(
        marker_path=marker_path,
        phase=phase,
        smoke_run_dir=smoke_checkpoint.parent,
        init_checkpoint=init_checkpoint,
        easy_manifest=replay_manifest,
        max_peak_reserved_gib=float(raw_marker.get("max_peak_reserved_gib", float("nan"))),
    )
    expected = {
        "schema_version": 1,
        "phase": phase,
        "correction_round": round_index,
        "replay_receipt_path": str(replay_receipt),
        "replay_receipt_sha256": sha256_file(replay_receipt),
        "admission_gate_path": str(admission_gate),
        "admission_gate_sha256": sha256_file(admission_gate),
        "layer_focus_path": str(layer_focus),
        "layer_focus_sha256": sha256_file(layer_focus),
        "nano_teacher_checkpoint_path": str(nano_checkpoint),
        "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 correction smoke marker binding mismatch.")
    peak = float(marker.get("peak_reserved_gib", float("nan")))
    peak_limit = float(marker.get("max_peak_reserved_gib", float("nan")))
    if (
        not math.isfinite(peak)
        or not math.isfinite(peak_limit)
        or peak < 0.0
        or peak_limit <= 0.0
        or peak > peak_limit
    ):
        raise ValueError("Stage211 correction smoke marker memory contract mismatch.")
    return marker


def _run_correction_smoke(
    *,
    round_index: int,
    run_dir: Path,
    config_dir: Path,
    replay_receipt: Path,
    replay_manifest: Path,
    admission_gate: Path,
    layer_focus: Path,
    layer_focus_payload: dict[str, Any],
    init_checkpoint: Path,
    nano_checkpoint: Path,
    audio_data_audit: dict[str, Any],
    master_port: int,
    max_peak_reserved_gib: float,
    formal_latest_step: int,
    dry_run: bool,
    phase: str = "mixer",
) -> Path:
    smoke_run_dir = Path(f"{run_dir}_smoke")
    marker_path = run_dir.parent / f"{run_dir.name}_smoke_passed.json"
    if marker_path.is_file() and not dry_run:
        _validate_correction_smoke_marker(
            marker_path=marker_path,
            round_index=round_index,
            init_checkpoint=init_checkpoint,
            replay_receipt=replay_receipt,
            replay_manifest=replay_manifest,
            admission_gate=admission_gate,
            layer_focus=layer_focus,
            nano_checkpoint=nano_checkpoint,
            phase=phase,
        )
        return marker_path
    if formal_latest_step > 0 and not dry_run:
        raise ValueError("Stage211 correction formal training lacks its preflight smoke marker.")

    phase_config = replace(PHASES[phase], lr=stage211_post_coverage_correction_lr(phase))
    segment = _segments(
        phase=phase_config,
        smoke=True,
        difficulty="easy",
        full_data_profile=True,
    )[0]
    smoke_latest_step = _latest_step(smoke_run_dir)
    config = _stage211_config(
        phase=phase_config,
        segment=segment,
        output_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        bucket_manifest=replay_manifest,
        resume=smoke_latest_step > 0,
        smoke=True,
        nano_checkpoint=nano_checkpoint,
        audio_data_audit=audio_data_audit,
        full_data_profile=True,
        post_coverage_correction=True,
        correction_layer_boundary_ids=layer_focus_payload["boundary_layer_ids"],
    )
    config.update(
        _correction_config_metadata(
            round_index=round_index,
            replay_receipt=replay_receipt,
            admission_gate=admission_gate,
            layer_focus=layer_focus,
            phase=phase,
        )
    )
    smoke_config_dir = config_dir / "smoke"
    smoke_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = smoke_config_dir / f"stage211_{segment['name']}.yaml"
    save_yaml(config_path, config)
    if smoke_latest_step < 2:
        code = _run(
            config_path,
            smoke_run_dir / "logs" / f"{segment['name']}.log",
            dry_run=dry_run,
            master_port=master_port,
        )
        if code != 0:
            raise RuntimeError(f"Stage211 correction smoke exited with code={code}.")
    if dry_run:
        return marker_path

    marker = _audit_full_profile_smoke(
        phase=phase,
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=replay_manifest,
        max_peak_reserved_gib=max_peak_reserved_gib,
    )
    marker.update(
        {
            "correction_round": round_index,
            "correction_phase": phase,
            "replay_receipt_path": str(replay_receipt),
            "replay_receipt_sha256": sha256_file(replay_receipt),
            "admission_gate_path": str(admission_gate),
            "admission_gate_sha256": sha256_file(admission_gate),
            "layer_focus_path": str(layer_focus),
            "layer_focus_sha256": sha256_file(layer_focus),
            "nano_teacher_checkpoint_path": str(nano_checkpoint),
            "nano_teacher_checkpoint_sha256": sha256_file(nano_checkpoint),
        }
    )
    _write_immutable_json(marker_path, marker)
    _validate_correction_smoke_marker(
        marker_path=marker_path,
        round_index=round_index,
        init_checkpoint=init_checkpoint,
        replay_receipt=replay_receipt,
        replay_manifest=replay_manifest,
        admission_gate=admission_gate,
        layer_focus=layer_focus,
        nano_checkpoint=nano_checkpoint,
        phase=phase,
    )
    return marker_path


def run_correction(args: argparse.Namespace) -> Path | None:
    phase_name = str(getattr(args, "phase", "mixer"))
    correction_lr = stage211_post_coverage_correction_lr(phase_name)
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
    admitted_gate, teacher_sha256 = _admit_failed_gate(
        admission_gate_path=admission_gate,
        init_checkpoint=init_checkpoint,
        round_index=round_index,
        phase=phase_name,
    )
    extension_decision: Path | None = None
    if round_index > STAGE211_RETENTION_CORRECTION_GUARANTEED_ROUNDS:
        extension_decision = admission_gate.parent / "correction_extension_decision.json"
        validate_stage211_correction_extension_decision(
            extension_decision,
            phase=phase_name,
            next_round=round_index,
            admission_gate_path=admission_gate,
            admission_gate=admitted_gate,
        )
    _validate_target_nano_teacher_checkpoint(
        recorded_sha256=teacher_sha256,
        nano_checkpoint_path=nano_checkpoint,
        label=f"{phase_name} correction round {round_index}",
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
    layer_focus_path = run_dir / "correction_layer_focus.json"
    layer_focus = build_stage211_correction_layer_focus(
        phase=phase_name,
        admission_gate_path=admission_gate,
        admission_gate=admitted_gate,
    )
    _write_immutable_json(layer_focus_path, layer_focus)
    layer_focus = validate_stage211_correction_layer_focus(
        layer_focus_path,
        phase=phase_name,
        admission_gate_path=admission_gate,
        admission_gate=admitted_gate,
    )
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
    phase_config = replace(PHASES[phase_name], lr=correction_lr)
    segment = _correction_segment(
        round_index=round_index,
        steps_per_epoch=steps_per_epoch,
        phase=phase_name,
    )
    config = _stage211_config(
        phase=phase_config,
        segment=segment,
        output_dir=run_dir,
        init_checkpoint=init_checkpoint,
        bucket_manifest=replay_manifest,
        resume=latest_step > 0,
        smoke=False,
        nano_checkpoint=nano_checkpoint,
        audio_data_audit=audio_data_audit,
        full_data_profile=True,
        post_coverage_correction=True,
        correction_layer_boundary_ids=layer_focus["boundary_layer_ids"],
    )
    config.update(
        _correction_config_metadata(
            round_index=round_index,
            replay_receipt=replay_receipt,
            admission_gate=admission_gate,
            layer_focus=layer_focus_path,
            phase=phase_name,
        )
    )
    config_dir = (
        args.config_dir.expanduser().resolve() / phase_name / f"correction_round_{round_index:02d}"
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    smoke_marker_path = _run_correction_smoke(
        round_index=round_index,
        run_dir=run_dir,
        config_dir=config_dir,
        replay_receipt=replay_receipt,
        replay_manifest=replay_manifest,
        admission_gate=admission_gate,
        layer_focus=layer_focus_path,
        layer_focus_payload=layer_focus,
        init_checkpoint=init_checkpoint,
        nano_checkpoint=nano_checkpoint,
        audio_data_audit=audio_data_audit,
        master_port=int(args.master_port),
        max_peak_reserved_gib=float(args.max_peak_reserved_gib),
        formal_latest_step=latest_step,
        dry_run=bool(args.dry_run),
        phase=phase_name,
    )
    print(
        f"stage211_post_coverage_correction_smoke phase={phase_name} marker={smoke_marker_path}",
        flush=True,
    )
    if not args.dry_run:
        config.update(
            {
                "stage211_post_coverage_smoke_marker_path": str(smoke_marker_path),
                "stage211_post_coverage_smoke_marker_sha256": sha256_file(smoke_marker_path),
            }
        )
        provenance_path = run_dir / "stage211_correction_provenance.json"
        provenance = _provenance_payload(
            round_index=round_index,
            run_dir=run_dir,
            replay_receipt=replay_receipt,
            replay_manifest=replay_manifest,
            admission_gate=admission_gate,
            init_checkpoint=init_checkpoint,
            nano_checkpoint=nano_checkpoint,
            smoke_marker=smoke_marker_path,
            layer_focus=layer_focus_path,
            extension_decision=extension_decision,
            steps_per_epoch=steps_per_epoch,
            phase=phase_name,
        )
        if provenance_path.is_file():
            if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
                raise ValueError(
                    f"Stage211 correction provenance cannot change in place: {provenance_path}"
                )
        else:
            _write_immutable_json(provenance_path, provenance)
    config_path = config_dir / f"stage211_{segment['name']}.yaml"
    save_yaml(config_path, config)
    print(
        "stage211_post_coverage_correction "
        f"phase={phase_name} round={round_index} rows={replay['validated_unique_keys']} "
        f"steps={steps_per_epoch} tail_padding={tail_padding} "
        f"focus_layers={','.join(str(value) for value in layer_focus['boundary_layer_ids']) or '-'} "
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
        phase=phase_name,
    )
    receipt_path = run_dir / "correction_receipt.json"
    _write_immutable_json(receipt_path, receipt)
    print(
        f"stage211_post_coverage_correction_complete phase={phase_name} round={round_index} "
        f"checkpoint={completion_checkpoint} receipt={receipt_path}",
        flush=True,
    )
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one strict Stage211 post-coverage correction round."
    )
    parser.add_argument("--phase", choices=("mixer", "block", "logits"), default="mixer")
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
