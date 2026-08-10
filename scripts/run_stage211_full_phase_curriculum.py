from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from rwkvasr.eval.stage211_gate import (
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    build_stage211_full_data_coverage,
    sha256_file,
    validate_stage211_full_data_coverage,
)
from rwkvasr.eval.stage211_supplemental import (
    DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY,
    STAGE211_SUPPLEMENTAL_DIFFICULTY,
    stage211_supplemental_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SINGLE_SEGMENT_RUNNER = REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"
RECEIPT_CREATOR = REPO_ROOT / "scripts" / "create_stage211_curriculum_receipt.py"
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_runs" / "stage211_full_alignment"
DEFAULT_CONFIG_ROOT = Path.home() / "rwkvasr_configs" / "stage211_full_alignment"
DEFAULT_METADATA_ROOT = Path.home() / "rwkvasr_data" / "stage211_full_curriculum"
DEFAULT_EASY_MANIFEST = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_easy_source_grouped_buckets"
    / "manifest_stage211_fixed_eval.json"
)
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
PHASE_DIR_NAMES = {
    "mixer": "stage211a_mixer_full_data_3ep",
    "block": "stage211b_block_full_data_3ep",
    "logits": "stage211c_logits_full_data_3ep",
}
DEFAULT_MANIFESTS = {
    "easy": DEFAULT_EASY_MANIFEST,
    "medium": (
        DEFAULT_METADATA_ROOT
        / "stage179b_medium_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest_stage211_fixed_eval.json"
    ),
    "hard": (
        DEFAULT_METADATA_ROOT
        / "stage179c_hard_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest_stage211_fixed_eval.json"
    ),
    "long": (
        DEFAULT_METADATA_ROOT
        / "stage179d_long_dedup_audio_only_online_ctc"
        / "webdataset_buckets_audio_text"
        / "manifest_stage211_fixed_eval.json"
    ),
}
BAD_SMOKE_PATTERNS = (
    re.compile(r"Traceback"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemory"),
    re.compile(r"\bloss=(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"\bonline_[a-z0-9_]*missing=[1-9][0-9]*\b"),
    re.compile(r"\bonline_[a-z0-9_]*frame_delta=[1-9][0-9]*\b"),
    re.compile(r"\bdropped_tail(?:_samples)?=[1-9][0-9]*\b"),
    re.compile(r"\bskipped_samples=[1-9][0-9]*\b"),
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _latest_step(run_dir: Path) -> int:
    latest = 0
    for checkpoint in run_dir.glob("step-*.pt"):
        match = re.fullmatch(r"step-([0-9]+)\.pt", checkpoint.name)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def _formal_phase_training_started(phase_root: Path) -> bool:
    if any((phase_root / "receipts").glob("*.json")):
        return True
    for difficulty in (*STAGE211_AUDIO_CURRICULUM, STAGE211_SUPPLEMENTAL_DIFFICULTY):
        run_dir = phase_root / difficulty
        if _latest_step(run_dir) > 0 or (run_dir / "latest_checkpoint.yaml").exists():
            return True
        deepspeed_root = run_dir / "ds_checkpoints"
        if deepspeed_root.is_dir() and any(deepspeed_root.glob("step-[1-9]*")):
            return True
        step_pattern = re.compile(r"\[deepspeed-train\] step=[1-9][0-9]*\b")
        for log_path in (run_dir / "logs").glob("*.log"):
            with log_path.open("r", encoding="utf-8", errors="replace") as source:
                if any(step_pattern.search(line) is not None for line in source):
                    return True
    return False


def _parse_manifest_overrides(
    values: list[str],
    *,
    metadata_root: Path,
    easy_manifest: Path,
) -> dict[str, Path]:
    manifests = {
        "easy": easy_manifest,
        "medium": (
            metadata_root
            / "stage179b_medium_dedup_audio_only_online_ctc"
            / "webdataset_buckets_audio_text"
            / "manifest_stage211_fixed_eval.json"
        ),
        "hard": (
            metadata_root
            / "stage179c_hard_dedup_audio_only_online_ctc"
            / "webdataset_buckets_audio_text"
            / "manifest_stage211_fixed_eval.json"
        ),
        "long": (
            metadata_root
            / "stage179d_long_dedup_audio_only_online_ctc"
            / "webdataset_buckets_audio_text"
            / "manifest_stage211_fixed_eval.json"
        ),
    }
    for value in values:
        difficulty, separator, raw_path = value.partition("=")
        if separator != "=" or difficulty not in STAGE211_AUDIO_CURRICULUM or not raw_path:
            raise ValueError(
                "--manifest must use difficulty=/absolute/path for easy, medium, hard, or long."
            )
        manifests[difficulty] = Path(raw_path)
    return {difficulty: path.expanduser().resolve() for difficulty, path in manifests.items()}


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"[stage211-full-phase] command={shlex.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _runner_command(
    *,
    phase: str,
    difficulty: str,
    output_dir: Path,
    config_dir: Path,
    manifest_path: Path,
    nano_checkpoint: Path,
    master_port: int,
    init_checkpoint: Path | None,
    curriculum_receipt: Path | None,
    promotion_receipt: Path | None,
    smoke: bool,
    dry_run: bool,
    supplemental_inventory: Path | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        str(SINGLE_SEGMENT_RUNNER),
        "--phase",
        phase,
        "--difficulty",
        difficulty,
        "--full-data-profile",
        "--output-dir",
        str(output_dir),
        "--config-dir",
        str(config_dir),
        "--bucket-manifest",
        str(manifest_path),
        "--nano-checkpoint",
        str(nano_checkpoint),
        "--master-port",
        str(master_port),
    ]
    if init_checkpoint is not None:
        command.extend(("--init-checkpoint", str(init_checkpoint)))
    if curriculum_receipt is not None:
        command.extend(("--curriculum-receipt", str(curriculum_receipt)))
    if promotion_receipt is not None:
        command.extend(("--promotion-receipt", str(promotion_receipt)))
    if supplemental_inventory is not None:
        command.extend(("--supplemental-inventory", str(supplemental_inventory)))
    if smoke:
        command.append("--smoke")
    if dry_run:
        command.extend(("--dry-run", "--skip-nano-weight-audit"))
    return command


def _receipt_command(
    *,
    phase: str,
    difficulty: str,
    run_dir: Path,
    manifest_path: Path,
    init_checkpoint: Path,
    completion_checkpoint: Path,
    output: Path,
    supplemental_inventory: Path | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        str(RECEIPT_CREATOR),
        "--phase",
        phase,
        "--difficulty",
        difficulty,
        "--run-dir",
        str(run_dir),
        "--bucket-manifest",
        str(manifest_path),
        "--init-checkpoint",
        str(init_checkpoint),
        "--completion-checkpoint",
        str(completion_checkpoint),
        "--output",
        str(output),
    ]
    if supplemental_inventory is not None:
        command.extend(("--supplemental-inventory", str(supplemental_inventory)))
    return command


def _resolve_initial_checkpoint(
    *,
    requested: Path | None,
    easy_run_dir: Path,
) -> Path:
    provenance_path = easy_run_dir / "stage211_provenance.json"
    if provenance_path.is_file():
        provenance = _load_json(provenance_path, label="Stage211 easy provenance")
        recorded = Path(str(provenance.get("init_checkpoint_path") or "")).resolve()
        if not recorded.is_file():
            raise ValueError(f"Recorded Stage211 phase initialization is unavailable: {recorded}")
        if provenance.get("init_checkpoint_sha256") != sha256_file(recorded):
            raise ValueError("Recorded Stage211 phase initialization SHA-256 changed.")
        if requested is not None and requested.resolve() != recorded:
            raise ValueError(
                "Requested initialization differs from the existing easy-run provenance."
            )
        return recorded
    if requested is None:
        raise ValueError("A fresh Stage211 full phase requires --init-checkpoint.")
    requested = requested.expanduser().resolve()
    if not requested.is_file() or requested.stat().st_size <= 0:
        raise FileNotFoundError(str(requested))
    return requested


def _audit_smoke(
    *,
    phase: str,
    smoke_run_dir: Path,
    init_checkpoint: Path,
    easy_manifest: Path,
    max_peak_reserved_gib: float,
) -> dict[str, Any]:
    checkpoint = smoke_run_dir / "step-2.pt"
    log_path = smoke_run_dir / "logs" / f"{phase}_smoke_2steps.log"
    if not checkpoint.is_file() or _checkpoint_step(checkpoint) != 2:
        raise ValueError(f"Stage211 {phase} smoke did not produce step-2.pt.")
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise ValueError(f"Stage211 {phase} smoke log is missing: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    attempt_marker = "[rwkvasr] Distributed init complete."
    latest_attempt_start = log_text.rfind(attempt_marker)
    if latest_attempt_start >= 0:
        log_text = log_text[latest_attempt_start:]
    if "[deepspeed-train] step=2" not in log_text:
        raise ValueError(f"Stage211 {phase} smoke did not execute two training steps.")
    for pattern in BAD_SMOKE_PATTERNS:
        match = pattern.search(log_text)
        if match is not None:
            raise ValueError(
                f"Stage211 {phase} smoke contains a rejected condition: {match.group(0)}"
            )
    peak_values = [
        float(value) for value in re.findall(r"peak_reserved=([0-9]+(?:\.[0-9]+)?)GiB", log_text)
    ]
    if not peak_values:
        raise ValueError(f"Stage211 {phase} smoke lacks peak-reserved memory telemetry.")
    peak_reserved_gib = max(peak_values)
    if peak_reserved_gib > max_peak_reserved_gib:
        raise ValueError(
            f"Stage211 {phase} smoke peak memory is unsafe: "
            f"{peak_reserved_gib:.2f} GiB > {max_peak_reserved_gib:.2f} GiB"
        )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "full_profile_smoke",
        "phase": phase,
        "complete": True,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "easy_manifest_path": str(easy_manifest),
        "easy_manifest_sha256": sha256_file(easy_manifest),
        "smoke_checkpoint_path": str(checkpoint),
        "smoke_checkpoint_sha256": sha256_file(checkpoint),
        "smoke_log_path": str(log_path),
        "smoke_log_sha256": sha256_file(log_path),
        "peak_reserved_gib": peak_reserved_gib,
        "max_peak_reserved_gib": max_peak_reserved_gib,
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _migrate_legacy_curriculum_summary(*, phase_root: Path, phase: str) -> None:
    summary_path = phase_root / "curriculum_complete.json"
    archive_path = phase_root / "curriculum_complete.original_only.json"
    migration_path = phase_root / "curriculum_complete.original_only.migration.json"
    if summary_path.is_file():
        summary = _load_json(
            summary_path,
            label="Stage211 existing curriculum summary",
        )
        coverage = summary.get("full_data_coverage")
        if not isinstance(coverage, dict):
            raise ValueError(f"Stage211 curriculum summary coverage is invalid: {summary_path}")
        if "supplemental_natural" in coverage:
            return
        if archive_path.is_file():
            if summary_path.read_bytes() != archive_path.read_bytes():
                raise ValueError(
                    "Stage211 legacy summary differs from its existing archive: "
                    f"{summary_path}"
                )
            summary_path.unlink()
        else:
            summary_path.replace(archive_path)
    if archive_path.is_file():
        _write_immutable_json(
            migration_path,
            {
                "schema_version": 1,
                "pipeline": "stage211",
                "artifact": "supplemental_summary_migration",
                "phase": phase,
                "archived_path": str(archive_path.resolve()),
                "archived_sha256": sha256_file(archive_path),
            },
        )


def _validate_smoke_marker(
    *,
    marker_path: Path,
    phase: str,
    smoke_run_dir: Path,
    init_checkpoint: Path,
    easy_manifest: Path,
    max_peak_reserved_gib: float,
) -> dict[str, Any]:
    marker = _load_json(marker_path, label="Stage211 full-profile smoke marker")
    rebuilt = _audit_smoke(
        phase=phase,
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=max_peak_reserved_gib,
    )
    if any(marker.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("Stage211 smoke marker differs from its rebuilt source evidence.")
    return marker


def _run_smoke(
    *,
    phase: str,
    phase_root: Path,
    config_root: Path,
    easy_manifest: Path,
    init_checkpoint: Path,
    promotion_receipt: Path | None,
    nano_checkpoint: Path,
    master_port: int,
    max_peak_reserved_gib: float,
    dry_run: bool,
) -> None:
    smoke_base = phase_root / "full_profile"
    smoke_run_dir = Path(f"{smoke_base}_smoke")
    marker_path = phase_root / "full_profile_smoke_passed.json"
    if marker_path.is_file() and not dry_run:
        marker = _validate_smoke_marker(
            marker_path=marker_path,
            phase=phase,
            smoke_run_dir=smoke_run_dir,
            init_checkpoint=init_checkpoint,
            easy_manifest=easy_manifest,
            max_peak_reserved_gib=max_peak_reserved_gib,
        )
        print(
            f"[stage211-full-phase] smoke already passed "
            f"peak_reserved_gib={marker['peak_reserved_gib']}",
            flush=True,
        )
        return
    if _formal_phase_training_started(phase_root) and not dry_run:
        raise ValueError(
            f"Stage211 {phase} formal training has progress but lacks its preflight "
            "smoke marker."
        )
    latest_step = _latest_step(smoke_run_dir)
    command = _runner_command(
        phase=phase,
        difficulty="easy",
        output_dir=smoke_base,
        config_dir=config_root,
        manifest_path=easy_manifest,
        nano_checkpoint=nano_checkpoint,
        master_port=master_port,
        init_checkpoint=init_checkpoint if latest_step <= 0 else None,
        curriculum_receipt=None,
        promotion_receipt=promotion_receipt if latest_step <= 0 else None,
        smoke=True,
        dry_run=dry_run,
    )
    _run_command(command, dry_run=dry_run)
    if dry_run:
        return
    marker = _audit_smoke(
        phase=phase,
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=max_peak_reserved_gib,
    )
    _write_immutable_json(marker_path, marker)
    _validate_smoke_marker(
        marker_path=marker_path,
        phase=phase,
        smoke_run_dir=smoke_run_dir,
        init_checkpoint=init_checkpoint,
        easy_manifest=easy_manifest,
        max_peak_reserved_gib=max_peak_reserved_gib,
    )
    print(
        f"[stage211-full-phase] smoke passed peak_reserved_gib={marker['peak_reserved_gib']}",
        flush=True,
    )


def _load_enriched_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path, label="Stage211 curriculum receipt")
    return {
        **receipt,
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256_file(path),
    }


def _validate_sha256_field(payload: dict[str, Any], key: str, *, label: str) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} has an invalid {key}.")


def _validate_reusable_runtime_coverage(
    coverage: Any,
    *,
    difficulty: str,
    steps_per_epoch: int,
) -> None:
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "runtime_epoch_coverage",
        "complete": True,
        "epochs": STAGE211_FULL_DATA_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": STAGE211_FULL_DATA_EPOCHS * steps_per_epoch,
    }
    if not isinstance(coverage, dict) or any(
        coverage.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"Stage211 {difficulty} reusable runtime coverage is invalid.")
    records = coverage.get("records")
    if not isinstance(records, list) or len(records) != STAGE211_FULL_DATA_EPOCHS:
        raise ValueError(f"Stage211 {difficulty} reusable runtime coverage is incomplete.")
    for epoch, record in enumerate(records, start=1):
        if not isinstance(record, dict) or any(
            (
                int(record.get("epoch", -1)) != epoch,
                int(record.get("step", -1)) != epoch * steps_per_epoch,
                int(record.get("epoch_batch_offset", -1)) != 0,
                int(record.get("completed_epoch_batch_count", -1)) != steps_per_epoch,
            )
        ):
            raise ValueError(
                f"Stage211 {difficulty} reusable epoch {epoch} coverage is invalid."
            )
        checkpoint = Path(str(record.get("checkpoint_path") or "")).resolve()
        if not checkpoint.is_file():
            raise ValueError(
                f"Stage211 {difficulty} reusable epoch checkpoint is unavailable: {checkpoint}"
            )
        _validate_sha256_field(
            record,
            "checkpoint_sha256",
            label=f"Stage211 {difficulty} reusable epoch {epoch}",
        )


def _load_reusable_receipt(
    receipt_path: Path,
    *,
    phase: str,
    difficulty: str,
    run_dir: Path,
    manifest_path: Path,
    init_checkpoint: Path,
    completion_checkpoint: Path,
    expected_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, label="Stage211 reusable curriculum receipt")
    expected = expected_profile or STAGE211_AUDIO_CURRICULUM[difficulty]
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
        "rows": int(expected["rows"]),
        "row_exposures": int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS,
        "tail_padding_sample_exposures": int(
            expected["tail_padding_samples_per_epoch"]
        )
        * STAGE211_FULL_DATA_EPOCHS,
        "executed_sample_exposures": (
            int(expected["rows"])
            + int(expected["tail_padding_samples_per_epoch"])
        )
        * STAGE211_FULL_DATA_EPOCHS,
        "hours": float(expected["hours"]),
        "hour_exposures": float(expected["hours"]) * STAGE211_FULL_DATA_EPOCHS,
        "steps_per_epoch": int(expected["steps_per_epoch"]),
        "steps": int(expected["steps"]),
        "tail_padding_samples_per_epoch": int(
            expected["tail_padding_samples_per_epoch"]
        ),
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "run_dir": str(run_dir.resolve()),
        "bucket_manifest_path": str(manifest_path.resolve()),
        "init_checkpoint_path": str(init_checkpoint.resolve()),
        "completion_checkpoint_path": str(completion_checkpoint.resolve()),
    }
    if any(receipt.get(key) != value for key, value in expected_fields.items()):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} reusable curriculum receipt does not match "
            "the completed segment."
        )
    for path_key, sha_key in (
        ("provenance_path", "provenance_sha256"),
        ("train_config_path", "train_config_sha256"),
        ("nano_teacher_checkpoint_path", "nano_teacher_checkpoint_sha256"),
        ("bucket_manifest_path", "bucket_manifest_sha256"),
        ("init_checkpoint_path", "init_checkpoint_sha256"),
        ("completion_checkpoint_path", "completion_checkpoint_sha256"),
    ):
        bound_path = Path(str(receipt.get(path_key) or "")).resolve()
        if not bound_path.is_file():
            raise ValueError(
                f"Stage211 {phase}/{difficulty} reusable artifact is unavailable: {bound_path}"
            )
        _validate_sha256_field(
            receipt,
            sha_key,
            label=f"Stage211 {phase}/{difficulty} reusable receipt",
        )
    audit = receipt.get("parameter_delta_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("complete") is not True
        or audit.get("policy") != "stage211_timemixer_and_input_projection_only"
        or int(audit.get("forbidden_changed_tensors", -1)) != 0
        or int(audit.get("allowed_changed_tensors", 0)) <= 0
        or int(audit.get("allowed_changed_numel", 0)) <= 0
    ):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} reusable parameter-delta audit is invalid."
        )
    _validate_reusable_runtime_coverage(
        receipt.get("runtime_epoch_coverage"),
        difficulty=difficulty,
        steps_per_epoch=int(expected["steps_per_epoch"]),
    )
    if expected_profile is not None:
        for key in ("supplemental_inventory_path", "supplemental_inventory_sha256"):
            expected_key = "inventory_path" if key.endswith("_path") else "inventory_sha256"
            if receipt.get(key) != expected_profile.get(expected_key):
                raise ValueError(
                    f"Stage211 {phase}/{difficulty} reusable supplemental inventory changed."
                )
    return {
        **receipt,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
    }


def run_phase(args: argparse.Namespace) -> Path | None:
    phase = str(args.phase)
    output_root = args.output_root.expanduser().resolve()
    config_root = args.config_root.expanduser().resolve()
    phase_root = output_root / PHASE_DIR_NAMES[phase]
    easy_run_dir = phase_root / "easy"
    manifests = _parse_manifest_overrides(
        list(args.manifest),
        metadata_root=args.metadata_root.expanduser().resolve(),
        easy_manifest=args.easy_manifest.expanduser().resolve(),
    )
    for difficulty, manifest in manifests.items():
        if not manifest.is_file() or manifest.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Stage211 {difficulty} fixed-eval manifest unavailable: {manifest}"
            )
    supplemental_inventory = args.supplemental_inventory.expanduser().resolve()
    supplemental_profile = stage211_supplemental_profile(
        supplemental_inventory,
        epochs=STAGE211_FULL_DATA_EPOCHS,
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        require_training_ready=not args.dry_run,
        verify_part_sha256=False,
    )
    supplemental_manifest = Path(str(supplemental_profile["bucket_manifest_path"])).resolve()
    nano_checkpoint = args.nano_checkpoint.expanduser().resolve()
    if not nano_checkpoint.is_file() or nano_checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(str(nano_checkpoint))
    init_checkpoint = _resolve_initial_checkpoint(
        requested=args.init_checkpoint,
        easy_run_dir=easy_run_dir,
    )
    promotion_receipt = (
        args.promotion_receipt.expanduser().resolve()
        if args.promotion_receipt is not None
        else None
    )
    if phase == "mixer" and promotion_receipt is not None:
        raise ValueError("Stage211 mixer does not accept --promotion-receipt.")
    if phase != "mixer":
        if promotion_receipt is None or not promotion_receipt.is_file():
            raise ValueError(f"Stage211 {phase} requires the preceding phase promotion receipt.")

    _run_smoke(
        phase=phase,
        phase_root=phase_root,
        config_root=config_root,
        easy_manifest=manifests["easy"],
        init_checkpoint=init_checkpoint,
        promotion_receipt=promotion_receipt,
        nano_checkpoint=nano_checkpoint,
        master_port=int(args.master_port),
        max_peak_reserved_gib=float(args.max_peak_reserved_gib),
        dry_run=bool(args.dry_run),
    )
    if args.smoke_only:
        return None

    receipts: list[dict[str, Any]] = []
    current_init = init_checkpoint
    preceding_receipt: Path | None = None
    for index, difficulty in enumerate(STAGE211_AUDIO_CURRICULUM):
        run_dir = phase_root / difficulty
        receipt_path = phase_root / "receipts" / f"{difficulty}.json"
        target_step = int(STAGE211_AUDIO_CURRICULUM[difficulty]["steps"])
        latest_step = _latest_step(run_dir)
        runner = _runner_command(
            phase=phase,
            difficulty=difficulty,
            output_dir=run_dir,
            config_dir=config_root,
            manifest_path=manifests[difficulty],
            nano_checkpoint=nano_checkpoint,
            master_port=int(args.master_port),
            init_checkpoint=current_init if latest_step <= 0 else None,
            curriculum_receipt=(
                preceding_receipt if latest_step <= 0 and difficulty != "easy" else None
            ),
            promotion_receipt=(
                promotion_receipt if latest_step <= 0 and difficulty == "easy" else None
            ),
            smoke=False,
            dry_run=bool(args.dry_run),
            supplemental_inventory=None,
        )
        _run_command(runner, dry_run=bool(args.dry_run))
        completion_checkpoint = run_dir / f"step-{target_step}.pt"
        if args.dry_run:
            print(
                f"[stage211-full-phase] dry-run completion={completion_checkpoint} "
                f"receipt={receipt_path}",
                flush=True,
            )
            current_init = completion_checkpoint
            preceding_receipt = receipt_path
            continue
        if (
            not completion_checkpoint.is_file()
            or _checkpoint_step(completion_checkpoint) != target_step
        ):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} lacks exact completion checkpoint "
                f"step={target_step}: {completion_checkpoint}"
            )
        if receipt_path.is_file():
            receipt = _load_reusable_receipt(
                receipt_path,
                phase=phase,
                difficulty=difficulty,
                run_dir=run_dir,
                manifest_path=manifests[difficulty],
                init_checkpoint=current_init,
                completion_checkpoint=completion_checkpoint,
            )
            print(
                f"[stage211-full-phase] reused immutable receipt "
                f"difficulty={difficulty} receipt={receipt_path}",
                flush=True,
            )
        else:
            receipt_command = _receipt_command(
                phase=phase,
                difficulty=difficulty,
                run_dir=run_dir,
                manifest_path=manifests[difficulty],
                init_checkpoint=current_init,
                completion_checkpoint=completion_checkpoint,
                output=receipt_path,
                supplemental_inventory=None,
            )
            _run_command(receipt_command, dry_run=False)
            receipt = _load_enriched_receipt(receipt_path)
        receipts.append(receipt)
        print(
            f"[stage211-full-phase] segment complete "
            f"difficulty={difficulty} step={target_step} "
            f"receipt_sha256={receipt['receipt_sha256']}",
            flush=True,
        )
        current_init = completion_checkpoint
        preceding_receipt = receipt_path
        if index + 1 < len(STAGE211_AUDIO_CURRICULUM):
            next_difficulty = tuple(STAGE211_AUDIO_CURRICULUM)[index + 1]
            if receipt.get("difficulty") != difficulty:
                raise ValueError(f"Stage211 receipt cannot admit {next_difficulty}: {receipt_path}")

    supplemental_run_dir = phase_root / STAGE211_SUPPLEMENTAL_DIFFICULTY
    supplemental_receipt_path = (
        phase_root / "receipts" / f"{STAGE211_SUPPLEMENTAL_DIFFICULTY}.json"
    )
    supplemental_target_step = int(supplemental_profile["steps"])
    supplemental_latest_step = _latest_step(supplemental_run_dir)
    supplemental_runner = _runner_command(
        phase=phase,
        difficulty=STAGE211_SUPPLEMENTAL_DIFFICULTY,
        output_dir=supplemental_run_dir,
        config_dir=config_root,
        manifest_path=supplemental_manifest,
        nano_checkpoint=nano_checkpoint,
        master_port=int(args.master_port),
        init_checkpoint=current_init if supplemental_latest_step <= 0 else None,
        curriculum_receipt=preceding_receipt if supplemental_latest_step <= 0 else None,
        promotion_receipt=None,
        smoke=False,
        dry_run=bool(args.dry_run),
        supplemental_inventory=supplemental_inventory,
    )
    _run_command(supplemental_runner, dry_run=bool(args.dry_run))
    supplemental_completion = supplemental_run_dir / f"step-{supplemental_target_step}.pt"
    if args.dry_run:
        print(
            f"[stage211-full-phase] dry-run completion={supplemental_completion} "
            f"receipt={supplemental_receipt_path}",
            flush=True,
        )
        return None
    if (
        not supplemental_completion.is_file()
        or _checkpoint_step(supplemental_completion) != supplemental_target_step
    ):
        raise ValueError(
            "Stage211 supplemental_natural lacks exact completion checkpoint "
            f"step={supplemental_target_step}: {supplemental_completion}"
        )
    if supplemental_receipt_path.is_file():
        supplemental_receipt = _load_reusable_receipt(
            supplemental_receipt_path,
            phase=phase,
            difficulty=STAGE211_SUPPLEMENTAL_DIFFICULTY,
            run_dir=supplemental_run_dir,
            manifest_path=supplemental_manifest,
            init_checkpoint=current_init,
            completion_checkpoint=supplemental_completion,
            expected_profile=supplemental_profile,
        )
    else:
        supplemental_receipt_command = _receipt_command(
            phase=phase,
            difficulty=STAGE211_SUPPLEMENTAL_DIFFICULTY,
            run_dir=supplemental_run_dir,
            manifest_path=supplemental_manifest,
            init_checkpoint=current_init,
            completion_checkpoint=supplemental_completion,
            output=supplemental_receipt_path,
            supplemental_inventory=supplemental_inventory,
        )
        _run_command(supplemental_receipt_command, dry_run=False)
        supplemental_receipt = _load_enriched_receipt(supplemental_receipt_path)
    print(
        "[stage211-full-phase] segment complete "
        f"difficulty={STAGE211_SUPPLEMENTAL_DIFFICULTY} "
        f"step={supplemental_target_step} "
        f"receipt_sha256={supplemental_receipt['receipt_sha256']}",
        flush=True,
    )
    current_init = supplemental_completion

    if args.dry_run:
        return None
    final_checkpoint = current_init.resolve()
    coverage = build_stage211_full_data_coverage(
        phase=phase,
        segments=receipts,
        supplemental_segment=supplemental_receipt,
        checkpoint_path=final_checkpoint,
    )
    validate_stage211_full_data_coverage(
        coverage,
        phase=phase,
        checkpoint_path=final_checkpoint,
    )
    summary_path = phase_root / "curriculum_complete.json"
    _migrate_legacy_curriculum_summary(phase_root=phase_root, phase=phase)
    _write_immutable_json(
        summary_path,
        {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "full_phase_curriculum",
            "phase": phase,
            "complete": True,
            "full_data_coverage": coverage,
        },
    )
    if args.final_checkpoint_path_output is not None:
        final_checkpoint_path_output = args.final_checkpoint_path_output.expanduser().resolve()
        final_checkpoint_path_output.parent.mkdir(parents=True, exist_ok=True)
        rendered_checkpoint_path = str(final_checkpoint) + "\n"
        if (
            final_checkpoint_path_output.is_file()
            and final_checkpoint_path_output.read_text(encoding="utf-8") != rendered_checkpoint_path
        ):
            raise ValueError(
                "Refusing to overwrite a different Stage211 final-checkpoint path: "
                f"{final_checkpoint_path_output}"
            )
        final_checkpoint_path_output.write_text(
            rendered_checkpoint_path,
            encoding="utf-8",
        )
    print(
        f"[stage211-full-phase] curriculum complete phase={phase} "
        f"checkpoint={final_checkpoint} summary={summary_path}",
        flush=True,
    )
    return final_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one strict Stage211 A/B/C phase over easy, medium, hard, and long "
            "plus supplemental natural audio with three full epochs per segment and "
            "immutable coverage receipts."
        )
    )
    parser.add_argument("--phase", choices=tuple(PHASE_DIR_NAMES), required=True)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--promotion-receipt", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--easy-manifest", type=Path, default=DEFAULT_EASY_MANIFEST)
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Override one manifest as difficulty=/absolute/path.",
    )
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument(
        "--supplemental-inventory",
        type=Path,
        default=DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY,
    )
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=22.0)
    parser.add_argument("--final-checkpoint-path-output", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_peak_reserved_gib <= 0.0:
        parser.error("--max-peak-reserved-gib must be positive")
    run_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
