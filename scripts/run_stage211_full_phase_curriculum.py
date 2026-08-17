from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_batch_profile import (
    validate_stage211_batch_profile_admission,
    validate_stage211_batch_profile_preflight,
)
from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    STAGE211_AUDIO_CURRICULUM,
    STAGE211_FULL_DATA_BATCH_SIZE,
    STAGE211_FULL_DATA_EPOCHS,
    STAGE211_FULL_DATA_FRAME_BUDGET,
    STAGE211_FULL_DATA_WORLD_SIZE,
    build_stage211_full_data_coverage,
    sha256_file,
    validate_stage211_full_data_coverage,
    validate_stage211_loaded_manifest_receipt,
)
from rwkvasr.eval.stage211_supplemental import (
    DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY,
    DEFAULT_STAGE211_SUPPLEMENTAL_ROOT,
    STAGE211_SUPPLEMENTAL_DIFFICULTY,
    stage211_supplemental_profile,
    validate_stage211_formal_supplemental_profile,
)
from rwkvasr.eval.stage211_storage import compact_stage211_completed_segment

try:
    from scripts.create_stage211_supplemental_profile_receipt import (
        build_receipt as build_supplemental_profile_receipt,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ rather than the repo on sys.path.
    from create_stage211_supplemental_profile_receipt import (
        build_receipt as build_supplemental_profile_receipt,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
SINGLE_SEGMENT_RUNNER = REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"
RECEIPT_CREATOR = REPO_ROOT / "scripts" / "create_stage211_curriculum_receipt.py"
PROFILE_BENCHMARK = REPO_ROOT / "scripts" / "benchmark_stage211_batch_profiles.py"
PROFILE_ADMISSION_CREATOR = REPO_ROOT / "scripts" / "create_stage211_batch_profile_admission.py"
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
SMOKE_RUNTIME_MATCH_FIELDS = {
    "block": (
        "online_encoder_match",
        "online_decoder_hidden_match",
        "online_layer_match",
    ),
    "logits": (
        "online_blank_match",
        "online_full_match",
        "online_conditional_nonblank_match",
        "online_conditional_nonblank_hard_match",
        "online_encoder_match",
        "online_decoder_hidden_match",
        "online_layer_match",
    ),
}
SMOKE_RUNTIME_LOSS_FIELDS = {
    "block": (
        "online_layer_mixer",
        "online_layer_ffn",
        "online_layer_block",
        "online_ctc_encoder",
        "online_ctc_decoder_hidden",
    ),
    "logits": (
        "online_ctc_blank",
        "online_ctc_full",
        "online_ctc_conditional_nonblank",
        "online_ctc_conditional_nonblank_hard",
        "online_ctc_encoder",
        "online_ctc_decoder_hidden",
        "online_layer_mixer",
        "online_layer_ffn",
        "online_layer_block",
    ),
}
SMOKE_RUNTIME_PRIMARY_LOSS_FIELD = {
    "block": "online_layer_block",
    "logits": "online_ctc_full",
}


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


def _smoke_runtime_objective_evidence(*, phase: str, step_two_line: str) -> dict[str, Any]:
    if phase not in SMOKE_RUNTIME_MATCH_FIELDS:
        raise ValueError(f"Stage211 {phase} has no strict runtime-objective smoke contract.")
    match_fields: dict[str, dict[str, int]] = {}
    for field in SMOKE_RUNTIME_MATCH_FIELDS[phase]:
        match = re.search(rf"\b{re.escape(field)}=([0-9]+)/([0-9]+)\b", step_two_line)
        if match is None:
            raise ValueError(f"Stage211 {phase} smoke lacks step-2 {field} evidence.")
        matched, total = (int(value) for value in match.groups())
        if total <= 0 or matched != total:
            raise ValueError(
                f"Stage211 {phase} smoke has incomplete step-2 {field}: {matched}/{total}"
            )
        match_fields[field] = {"matched": matched, "total": total}

    loss_fields: dict[str, float] = {}
    for field in SMOKE_RUNTIME_LOSS_FIELDS[phase]:
        match = re.search(rf"\b{re.escape(field)}=([^ ]+)", step_two_line)
        if match is None:
            raise ValueError(f"Stage211 {phase} smoke lacks step-2 {field} telemetry.")
        try:
            value = float(match.group(1))
        except ValueError as error:
            raise ValueError(
                f"Stage211 {phase} smoke has non-numeric step-2 {field} telemetry."
            ) from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Stage211 {phase} smoke has invalid step-2 {field} telemetry: {value}"
            )
        loss_fields[field] = value

    primary_field = SMOKE_RUNTIME_PRIMARY_LOSS_FIELD[phase]
    if loss_fields[primary_field] <= 0.0:
        raise ValueError(f"Stage211 {phase} smoke primary step-2 {primary_field} must be positive.")
    return {
        "schema_version": 1,
        "step": 2,
        "required_match_fields": match_fields,
        "active_loss_fields": loss_fields,
        "primary_loss_field": primary_field,
    }


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


def _parse_batch_profile_admissions(values: list[str]) -> dict[str, Path]:
    allowed = (*tuple(STAGE211_AUDIO_CURRICULUM), STAGE211_SUPPLEMENTAL_DIFFICULTY)
    admissions: dict[str, Path] = {}
    for value in values:
        difficulty, separator, raw_path = value.partition("=")
        if separator != "=" or difficulty not in allowed or not raw_path:
            raise ValueError(
                "--batch-profile-admission must use "
                "difficulty=/absolute/path for easy, medium, hard, long, or "
                f"{STAGE211_SUPPLEMENTAL_DIFFICULTY}."
            )
        if difficulty in admissions:
            raise ValueError(f"Stage211 batch-profile admission was supplied twice: {difficulty}")
        admissions[difficulty] = Path(raw_path).expanduser().resolve()
    return admissions


def _require_loaded_manifest_bindings(
    *,
    manifests: dict[str, Path],
    receipt: dict[str, Any],
) -> None:
    raw_segments = receipt.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Stage211 loaded-manifest receipt lacks difficulty records.")
    segments = {
        str(segment.get("difficulty") or ""): segment
        for segment in raw_segments
        if isinstance(segment, dict)
    }
    if set(manifests) != set(STAGE211_AUDIO_CURRICULUM) or set(segments) != set(
        STAGE211_AUDIO_CURRICULUM
    ):
        raise ValueError("Stage211 full-phase difficulty-manifest set mismatch.")
    for difficulty in STAGE211_AUDIO_CURRICULUM:
        manifest = manifests[difficulty].expanduser().resolve()
        segment = segments[difficulty]
        recorded_manifest = (
            Path(str(segment.get("runtime_manifest_path") or "")).expanduser().resolve()
        )
        if manifest != recorded_manifest:
            raise ValueError(
                f"Stage211 {difficulty} manifest differs from loaded-manifest provenance."
            )
        if str(segment.get("runtime_manifest_sha256") or "") != sha256_file(manifest):
            raise ValueError(
                f"Stage211 {difficulty} manifest SHA-256 differs from loaded-manifest provenance."
            )


def _validate_loaded_manifest_bindings(
    *,
    manifests: dict[str, Path],
    receipt_path: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = validate_stage211_loaded_manifest_receipt(receipt_path)
    _require_loaded_manifest_bindings(manifests=manifests, receipt=receipt)
    print(
        "[stage211-full-phase] loaded-manifest admission passed "
        f"difficulties={len(manifests)} receipt_sha256={sha256_file(receipt_path)}",
        flush=True,
    )
    return receipt


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
    batch_profile_admission: Path | None = None,
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
    if batch_profile_admission is not None:
        command.extend(("--batch-profile-admission", str(batch_profile_admission)))
    if smoke:
        command.append("--smoke")
    if dry_run:
        command.extend(("--dry-run", "--skip-nano-weight-audit"))
    return command


def _recorded_batch_profile_admission(run_dir: Path) -> Path | None:
    provenance_path = run_dir / "stage211_provenance.json"
    if not provenance_path.is_file():
        return None
    provenance = _load_json(provenance_path, label="Stage211 segment provenance")
    raw_path = provenance.get("batch_profile_admission_path")
    raw_sha256 = provenance.get("batch_profile_admission_sha256")
    if raw_path is None and raw_sha256 is None:
        return None
    if not isinstance(raw_path, str) or not raw_path or not isinstance(raw_sha256, str):
        raise ValueError("Stage211 segment provenance has an incomplete batch-profile binding.")
    admission_path = Path(raw_path).expanduser().resolve()
    if not admission_path.is_file() or sha256_file(admission_path) != raw_sha256:
        raise ValueError("Stage211 segment provenance batch-profile admission changed.")
    return admission_path


def _resolve_segment_batch_profile_admission(
    *,
    run_dir: Path,
    requested: Path | None,
) -> Path | None:
    recorded = _recorded_batch_profile_admission(run_dir)
    if requested is not None:
        requested = requested.expanduser().resolve()
    if recorded is not None and requested is not None and recorded != requested:
        raise ValueError(
            "Requested Stage211 segment batch-profile admission differs from provenance."
        )
    return requested or recorded


def _find_generated_base_config(
    *,
    config_root: Path,
    phase: str,
    difficulty: str,
    expected_steps: int,
    init_checkpoint: Path,
    manifest_path: Path,
) -> Path:
    target_root = config_root / phase / f"full_{difficulty}"
    matches: list[Path] = []
    for path in sorted(target_root.glob("*.yaml")):
        config = load_yaml(path)
        if (
            int(config.get("max_steps", -1)) == int(expected_steps)
            and Path(str(config.get("init_checkpoint_path") or "")).resolve()
            == init_checkpoint.resolve()
            and Path(str(config.get("webdataset_bucket_manifest_path") or "")).resolve()
            == manifest_path.resolve()
        ):
            matches.append(path.resolve())
    if len(matches) != 1:
        raise ValueError(
            "Stage211 automatic batch preflight baseline config is ambiguous: "
            f"phase={phase} difficulty={difficulty} expected_steps={expected_steps} "
            f"matches={matches}"
        )
    return matches[0]


def _profile_preflight_command(
    *,
    phase: str,
    base_config: Path,
    init_checkpoint: Path,
    output_root: Path,
    master_port: int,
) -> list[str]:
    return [
        str(PYTHON),
        str(PROFILE_BENCHMARK),
        "--phase",
        phase,
        "--base-config",
        str(base_config),
        "--init-checkpoint",
        str(init_checkpoint),
        "--output-root",
        str(output_root),
        "--master-port",
        str(master_port),
        "--max-peak-memory-gib",
        "22.0",
        "--min-improvement-ratio",
        "0.10",
        "--max-loss-regression-ratio",
        "0.05",
        "--max-cosine-regression",
        "0.005",
    ]


def _profile_admission_command(
    *,
    phase: str,
    report_path: Path,
    admission_path: Path,
    admitted_by: str = "stage211-auto-segment-boundary",
    reason: str = (
        "phase/checkpoint/manifest-specific four-GPU profile passed memory, "
        "objective-match, quality-equivalence, and wall-time gates"
    ),
) -> list[str]:
    return [
        str(PYTHON),
        str(PROFILE_ADMISSION_CREATOR),
        "--preflight-report",
        str(report_path),
        "--phase",
        phase,
        "--admitted-by",
        admitted_by,
        "--reason",
        reason,
        "--admit-recommended-profile",
        "--output",
        str(admission_path),
    ]


def _automatic_profile_requires_admission(measured: dict[str, Any]) -> bool:
    selected_profile = measured["selected_profile_row"]["profile"]
    selected_is_legacy = (
        int(selected_profile["batch_size"]) == STAGE211_FULL_DATA_BATCH_SIZE
        and int(selected_profile["frame_budget"]) == STAGE211_FULL_DATA_FRAME_BUDGET
    )
    return measured["selection_decision"] != "keep_baseline" or not selected_is_legacy


def _ensure_measured_batch_profile(
    *,
    phase: str,
    scope: str,
    phase_root: Path,
    base_config: Path,
    manifest_path: Path,
    init_checkpoint: Path,
    master_port: int,
    admitted_by: str = "stage211-auto-segment-boundary",
    reason: str = (
        "phase/checkpoint/manifest-specific four-GPU profile passed memory, "
        "objective-match, quality-equivalence, and wall-time gates"
    ),
) -> tuple[dict[str, Any], Path | None]:
    init_sha256 = sha256_file(init_checkpoint)
    preflight_root = phase_root / "batch_profile_preflight" / f"{scope}-{init_sha256[:16]}"
    report_path = preflight_root / "batch_throughput_preflight.json"
    if report_path.is_file():
        measured = validate_stage211_batch_profile_preflight(
            report_path,
            phase=phase,
            require_candidate=False,
        )
    else:
        if preflight_root.exists() and any(preflight_root.iterdir()):
            raise ValueError(
                f"Incomplete Stage211 automatic batch preflight exists: {preflight_root}"
            )
        _run_command(
            _profile_preflight_command(
                phase=phase,
                base_config=base_config,
                init_checkpoint=init_checkpoint,
                output_root=preflight_root,
                master_port=master_port,
            ),
            dry_run=False,
        )
        measured = validate_stage211_batch_profile_preflight(
            report_path,
            phase=phase,
            require_candidate=False,
        )
    if (
        Path(str(measured["init_checkpoint_path"])).resolve() != init_checkpoint.resolve()
        or Path(str(measured["bucket_manifest_path"])).resolve() != manifest_path.resolve()
        or Path(str(measured["base_config_path"])).resolve() != base_config.resolve()
    ):
        raise ValueError("Stage211 automatic batch preflight binds different segment inputs.")
    if not _automatic_profile_requires_admission(measured):
        print(
            "[stage211-full-phase] automatic batch preflight retained legacy profile "
            f"phase={phase} scope={scope} report={report_path}",
            flush=True,
        )
        return measured, None
    admission_path = (
        phase_root / "batch_profile_preflight" / f"{scope}-{init_sha256[:16]}-admission.json"
    )
    _run_command(
        _profile_admission_command(
            phase=phase,
            report_path=report_path,
            admission_path=admission_path,
            admitted_by=admitted_by,
            reason=reason,
        ),
        dry_run=False,
    )
    admission = validate_stage211_batch_profile_admission(
        admission_path,
        phase=phase,
        expected_init_checkpoint=init_checkpoint,
        expected_bucket_manifest=manifest_path,
    )
    print(
        "[stage211-full-phase] automatic batch profile admitted "
        f"phase={phase} scope={scope} "
        f"profile={admission['selected_profile']['name']} "
        f"selection={measured['selection_decision']} admission={admission_path}",
        flush=True,
    )
    return measured, admission_path


def _ensure_automatic_batch_profile(
    *,
    phase: str,
    difficulty: str,
    phase_root: Path,
    config_root: Path,
    manifest_path: Path,
    init_checkpoint: Path,
    expected_legacy_steps: int,
    template_command: list[str],
    master_port: int,
) -> Path | None:
    _run_command(template_command, dry_run=False)
    base_config = _find_generated_base_config(
        config_root=config_root,
        phase=phase,
        difficulty=difficulty,
        expected_steps=expected_legacy_steps,
        init_checkpoint=init_checkpoint,
        manifest_path=manifest_path,
    )
    _, admission_path = _ensure_measured_batch_profile(
        phase=phase,
        scope=difficulty,
        phase_root=phase_root,
        base_config=base_config,
        manifest_path=manifest_path,
        init_checkpoint=init_checkpoint,
        master_port=master_port,
    )
    return admission_path


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
    batch_profile_admission: Path | None = None,
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
    if batch_profile_admission is not None:
        command.extend(("--batch-profile-admission", str(batch_profile_admission)))
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
    runtime_objective_evidence: dict[str, Any] | None = None
    if phase in SMOKE_RUNTIME_MATCH_FIELDS:
        step_two_lines = [
            line
            for line in log_text.splitlines()
            if re.search(r"\[deepspeed-train\] step=2\b", line) is not None
        ]
        runtime_objective_evidence = _smoke_runtime_objective_evidence(
            phase=phase,
            step_two_line=step_two_lines[-1],
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
    report = {
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
    if runtime_objective_evidence is not None:
        report["runtime_objective_evidence"] = runtime_objective_evidence
    return report


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _validate_supplemental_profile_receipt(
    receipt_path: Path,
    *,
    inventory_path: Path,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    actual = _load_json(
        receipt_path,
        label="Stage211 supplemental profile receipt",
    )
    expected = build_supplemental_profile_receipt(inventory_path)
    if actual != expected:
        raise ValueError(
            "Stage211 supplemental profile receipt differs from the current inventory: "
            f"{receipt_path}"
        )
    return {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "receipt": actual,
    }


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
                    f"Stage211 legacy summary differs from its existing archive: {summary_path}"
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
            f"Stage211 {phase} formal training has progress but lacks its preflight smoke marker."
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
            raise ValueError(f"Stage211 {difficulty} reusable epoch {epoch} coverage is invalid.")
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
    batch_profile_admission_path: Path | None = None,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, label="Stage211 reusable curriculum receipt")
    expected = expected_profile or STAGE211_AUDIO_CURRICULUM[difficulty]
    batch_size = STAGE211_FULL_DATA_BATCH_SIZE
    frame_budget = STAGE211_FULL_DATA_FRAME_BUDGET
    steps_per_epoch = int(expected["steps_per_epoch"])
    tail_padding_samples_per_epoch = int(expected["tail_padding_samples_per_epoch"])
    schema_version = 1
    batch_profile_admission: dict[str, Any] | None = None
    if batch_profile_admission_path is not None:
        batch_profile_admission = validate_stage211_batch_profile_admission(
            batch_profile_admission_path,
            phase=phase,
            expected_init_checkpoint=init_checkpoint,
            expected_bucket_manifest=manifest_path,
        )
        profile = batch_profile_admission["selected_profile"]
        coverage = batch_profile_admission["selected_coverage"]
        batch_size = int(profile["batch_size"])
        frame_budget = int(profile["frame_budget"])
        steps_per_epoch = int(coverage["steps_per_epoch"])
        tail_padding_samples_per_epoch = int(coverage["tail_padding_samples_per_epoch"])
        schema_version = 2
    total_steps = steps_per_epoch * STAGE211_FULL_DATA_EPOCHS
    expected_fields = {
        "schema_version": schema_version,
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "phase": phase,
        "difficulty": difficulty,
        "complete": True,
        "full_data_profile": True,
        "epochs": STAGE211_FULL_DATA_EPOCHS,
        "batch_size": batch_size,
        "world_size": STAGE211_FULL_DATA_WORLD_SIZE,
        "frame_budget": frame_budget,
        "rows": int(expected["rows"]),
        "row_exposures": int(expected["rows"]) * STAGE211_FULL_DATA_EPOCHS,
        "tail_padding_sample_exposures": int(tail_padding_samples_per_epoch)
        * STAGE211_FULL_DATA_EPOCHS,
        "executed_sample_exposures": (int(expected["rows"]) + tail_padding_samples_per_epoch)
        * STAGE211_FULL_DATA_EPOCHS,
        "hours": float(expected["hours"]),
        "hour_exposures": float(expected["hours"]) * STAGE211_FULL_DATA_EPOCHS,
        "steps_per_epoch": steps_per_epoch,
        "steps": total_steps,
        "tail_padding_samples_per_epoch": tail_padding_samples_per_epoch,
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
    if batch_profile_admission is not None:
        admission_fields = {
            "batch_profile_admission_path": batch_profile_admission["receipt_path"],
            "batch_profile_admission_sha256": batch_profile_admission["receipt_sha256"],
            "batch_profile_name": batch_profile_admission["selected_profile"]["name"],
        }
        if any(receipt.get(key) != value for key, value in admission_fields.items()):
            raise ValueError(
                f"Stage211 {phase}/{difficulty} reusable batch-profile binding changed."
            )
    elif any(
        receipt.get(key) is not None
        for key in (
            "batch_profile_admission_path",
            "batch_profile_admission_sha256",
            "batch_profile_name",
        )
    ):
        raise ValueError(
            f"Stage211 {phase}/{difficulty} legacy receipt unexpectedly binds a batch profile."
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
        steps_per_epoch=steps_per_epoch,
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
    batch_profile_admissions = _parse_batch_profile_admissions(
        list(getattr(args, "batch_profile_admission", ()))
    )
    for difficulty, manifest in manifests.items():
        if not manifest.is_file() or manifest.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Stage211 {difficulty} fixed-eval manifest unavailable: {manifest}"
            )
    supplemental_inventory = args.supplemental_inventory.expanduser().resolve()
    supplemental_profile_receipt = _validate_supplemental_profile_receipt(
        args.supplemental_profile_receipt,
        inventory_path=supplemental_inventory,
    )
    supplemental_profile = stage211_supplemental_profile(
        supplemental_inventory,
        epochs=STAGE211_FULL_DATA_EPOCHS,
        batch_size=STAGE211_FULL_DATA_BATCH_SIZE,
        world_size=STAGE211_FULL_DATA_WORLD_SIZE,
        frame_budget=STAGE211_FULL_DATA_FRAME_BUDGET,
        require_training_ready=not args.dry_run,
        verify_part_sha256=False,
    )
    loaded_manifest_receipt: dict[str, Any] | None = None
    if not args.dry_run:
        validate_stage211_formal_supplemental_profile(supplemental_profile)
        loaded_manifest_receipt = _validate_loaded_manifest_bindings(
            manifests=manifests,
            receipt_path=Path(
                getattr(
                    args,
                    "loaded_manifest_receipt",
                    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
                )
            ),
        )
    supplemental_manifest = Path(str(supplemental_profile["bucket_manifest_path"])).resolve()
    print(
        "[stage211-full-phase] supplemental profile receipt "
        f"sha256={supplemental_profile_receipt['sha256']}",
        flush=True,
    )
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
        if loaded_manifest_receipt is not None:
            _require_loaded_manifest_bindings(
                manifests=manifests,
                receipt=loaded_manifest_receipt,
            )
        run_dir = phase_root / difficulty
        receipt_path = phase_root / "receipts" / f"{difficulty}.json"
        latest_step = _latest_step(run_dir)
        batch_profile_admission_path = _resolve_segment_batch_profile_admission(
            run_dir=run_dir,
            requested=batch_profile_admissions.get(difficulty),
        )
        if (
            batch_profile_admission_path is None
            and bool(getattr(args, "auto_batch_profile", False))
            and not args.dry_run
            and latest_step <= 0
            and not receipt_path.is_file()
            and not (run_dir / "stage211_provenance.json").exists()
        ):
            template = _runner_command(
                phase=phase,
                difficulty=difficulty,
                output_dir=run_dir,
                config_dir=config_root,
                manifest_path=manifests[difficulty],
                nano_checkpoint=nano_checkpoint,
                master_port=int(args.master_port),
                init_checkpoint=current_init,
                curriculum_receipt=(preceding_receipt if difficulty != "easy" else None),
                promotion_receipt=(promotion_receipt if difficulty == "easy" else None),
                smoke=False,
                dry_run=True,
                supplemental_inventory=None,
                batch_profile_admission=None,
            )
            batch_profile_admission_path = _ensure_automatic_batch_profile(
                phase=phase,
                difficulty=difficulty,
                phase_root=phase_root,
                config_root=config_root,
                manifest_path=manifests[difficulty],
                init_checkpoint=current_init,
                expected_legacy_steps=int(STAGE211_AUDIO_CURRICULUM[difficulty]["steps"]),
                template_command=template,
                master_port=int(args.batch_profile_master_port),
            )
        batch_profile_admission = (
            validate_stage211_batch_profile_admission(
                batch_profile_admission_path,
                phase=phase,
                expected_init_checkpoint=current_init,
                expected_bucket_manifest=manifests[difficulty],
            )
            if batch_profile_admission_path is not None
            else None
        )
        target_step = int(
            batch_profile_admission["selected_coverage"]["full_coverage_steps"]
            if batch_profile_admission is not None
            else STAGE211_AUDIO_CURRICULUM[difficulty]["steps"]
        )
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
            batch_profile_admission=batch_profile_admission_path,
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
                batch_profile_admission_path=batch_profile_admission_path,
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
                batch_profile_admission=batch_profile_admission_path,
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
        compaction = compact_stage211_completed_segment(
            curriculum_receipt_path=receipt_path,
            expected_receipt=receipt,
        )
        print(
            "[stage211-full-phase] completed-segment storage compacted "
            f"difficulty={difficulty} removed_tags={len(compaction['removed_tags'])} "
            f"bytes_planned={compaction['bytes_planned']} "
            f"receipt_sha256={compaction['receipt_sha256']}",
            flush=True,
        )
        current_init = completion_checkpoint
        preceding_receipt = receipt_path
        if index + 1 < len(STAGE211_AUDIO_CURRICULUM):
            next_difficulty = tuple(STAGE211_AUDIO_CURRICULUM)[index + 1]
            if receipt.get("difficulty") != difficulty:
                raise ValueError(f"Stage211 receipt cannot admit {next_difficulty}: {receipt_path}")

    supplemental_run_dir = phase_root / STAGE211_SUPPLEMENTAL_DIFFICULTY
    supplemental_receipt_path = phase_root / "receipts" / f"{STAGE211_SUPPLEMENTAL_DIFFICULTY}.json"
    if loaded_manifest_receipt is not None:
        _require_loaded_manifest_bindings(
            manifests=manifests,
            receipt=loaded_manifest_receipt,
        )
    supplemental_batch_profile_path = batch_profile_admissions.get(STAGE211_SUPPLEMENTAL_DIFFICULTY)
    supplemental_latest_step = _latest_step(supplemental_run_dir)
    supplemental_batch_profile_path = _resolve_segment_batch_profile_admission(
        run_dir=supplemental_run_dir,
        requested=supplemental_batch_profile_path,
    )
    if (
        supplemental_batch_profile_path is None
        and bool(getattr(args, "auto_batch_profile", False))
        and not args.dry_run
        and supplemental_latest_step <= 0
        and not supplemental_receipt_path.is_file()
        and not (supplemental_run_dir / "stage211_provenance.json").exists()
    ):
        template = _runner_command(
            phase=phase,
            difficulty=STAGE211_SUPPLEMENTAL_DIFFICULTY,
            output_dir=supplemental_run_dir,
            config_dir=config_root,
            manifest_path=supplemental_manifest,
            nano_checkpoint=nano_checkpoint,
            master_port=int(args.master_port),
            init_checkpoint=current_init,
            curriculum_receipt=preceding_receipt,
            promotion_receipt=None,
            smoke=False,
            dry_run=True,
            supplemental_inventory=supplemental_inventory,
            batch_profile_admission=None,
        )
        supplemental_batch_profile_path = _ensure_automatic_batch_profile(
            phase=phase,
            difficulty=STAGE211_SUPPLEMENTAL_DIFFICULTY,
            phase_root=phase_root,
            config_root=config_root,
            manifest_path=supplemental_manifest,
            init_checkpoint=current_init,
            expected_legacy_steps=int(supplemental_profile["steps"]),
            template_command=template,
            master_port=int(args.batch_profile_master_port),
        )
    supplemental_batch_profile = (
        validate_stage211_batch_profile_admission(
            supplemental_batch_profile_path,
            phase=phase,
            expected_init_checkpoint=current_init,
            expected_bucket_manifest=supplemental_manifest,
        )
        if supplemental_batch_profile_path is not None
        else None
    )
    if supplemental_batch_profile is not None:
        selected_profile = supplemental_batch_profile["selected_profile"]
        supplemental_profile = stage211_supplemental_profile(
            supplemental_inventory,
            epochs=STAGE211_FULL_DATA_EPOCHS,
            batch_size=int(selected_profile["batch_size"]),
            world_size=STAGE211_FULL_DATA_WORLD_SIZE,
            frame_budget=int(selected_profile["frame_budget"]),
            require_training_ready=not args.dry_run,
            verify_part_sha256=False,
        )
        if not args.dry_run:
            validate_stage211_formal_supplemental_profile(supplemental_profile)
    supplemental_target_step = int(supplemental_profile["steps"])
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
        batch_profile_admission=supplemental_batch_profile_path,
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
            batch_profile_admission_path=supplemental_batch_profile_path,
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
            batch_profile_admission=supplemental_batch_profile_path,
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
    supplemental_compaction = compact_stage211_completed_segment(
        curriculum_receipt_path=supplemental_receipt_path,
        expected_receipt=supplemental_receipt,
    )
    print(
        "[stage211-full-phase] completed-segment storage compacted "
        f"difficulty={STAGE211_SUPPLEMENTAL_DIFFICULTY} "
        f"removed_tags={len(supplemental_compaction['removed_tags'])} "
        f"bytes_planned={supplemental_compaction['bytes_planned']} "
        f"receipt_sha256={supplemental_compaction['receipt_sha256']}",
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
        "--loaded-manifest-receipt",
        type=Path,
        default=DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Override one manifest as difficulty=/absolute/path.",
    )
    parser.add_argument(
        "--batch-profile-admission",
        action="append",
        default=[],
        help=(
            "Bind one fresh segment to a measured profile as "
            "difficulty=/absolute/path; repeat only for admitted segments."
        ),
    )
    parser.add_argument(
        "--auto-batch-profile",
        action="store_true",
        help=(
            "Measure and admit a phase-specific profile at each fresh segment boundary; "
            "currently restricted to future Block and Logits phases."
        ),
    )
    parser.add_argument("--batch-profile-master-port", type=int, default=29741)
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument(
        "--supplemental-inventory",
        type=Path,
        default=DEFAULT_STAGE211_SUPPLEMENTAL_INVENTORY,
    )
    parser.add_argument(
        "--supplemental-profile-receipt",
        type=Path,
        default=(DEFAULT_STAGE211_SUPPLEMENTAL_ROOT / "supplemental_profile_receipt.json"),
    )
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=22.0)
    parser.add_argument("--final-checkpoint-path-output", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_peak_reserved_gib <= 0.0:
        parser.error("--max-peak-reserved-gib must be positive")
    if args.batch_profile_master_port <= 0:
        parser.error("--batch-profile-master-port must be positive")
    if args.auto_batch_profile and args.phase not in ("block", "logits"):
        parser.error("--auto-batch-profile is restricted to Block and Logits")
    run_phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
