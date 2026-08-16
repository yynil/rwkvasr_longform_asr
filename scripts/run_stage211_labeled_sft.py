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
import yaml

from rwkvasr.config import load_yaml
from rwkvasr.eval.stage211_gate import (
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT,
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256,
    resolve_stage211_nano_teacher_checkpoint,
    sha256_file,
    validate_stage211_phase_train_config,
    validate_stage211_runtime_epoch_coverage,
)
from rwkvasr.eval.stage211_runtime import (
    audit_stage211_runtime_epoch_coverage,
)

try:
    from scripts.run_stage211_strict_chained_alignment import (
        _audit_labeled_data,
        _label_preparation_proof,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from run_stage211_strict_chained_alignment import (
        _audit_labeled_data,
        _label_preparation_proof,
    )

try:
    from scripts.create_stage211_labeled_profile_receipt import (
        validate_receipt as validate_labeled_profile_receipt,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from create_stage211_labeled_profile_receipt import (  # type: ignore[no-redef]
        validate_receipt as validate_labeled_profile_receipt,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUNNER = REPO_ROOT / "scripts" / "run_stage211_strict_chained_alignment.py"
DEFAULT_LABELED_ROOT = Path.home() / "rwkvasr_data" / "stage211_sft_full_labeled_v2"
DEFAULT_LABELED_PROFILE_RECEIPT = DEFAULT_LABELED_ROOT / "stage211_labeled_profile_receipt.json"
DEFAULT_OUTPUT_DIR = (
    Path.home() / "rwkvasr_runs" / "stage211_full_alignment" / "stage211d_labeled_ctc_sft_1ep"
)
DEFAULT_CONFIG_DIR = Path.home() / "rwkvasr_configs" / "stage211_full_alignment"
DEFAULT_NANO_CHECKPOINT = Path.home() / "models" / "Fun-ASR-Nano-2512-modelscope" / "model.pt"
LABELED_EXPECTED = {
    "train_samples": 283_868,
    "eval_samples": 1_434,
    "total_samples": 285_302,
    "total_hours": 809.12755,
    "ctc_tokens": 8_792_460,
    "ctc_unk_tokens": 0,
    "unique_utterance_ids": 285_302,
    "pronunciation_target_samples": 285_302,
    "ctc_feasible_samples": 285_302,
    "estimated_train_steps": 12_045,
    "tail_padding_samples_per_epoch": 368,
    "tail_padding_sample_exposures": 368,
    "executed_sample_exposures": 284_236,
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


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Refusing to overwrite a different Stage211D artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _latest_step(run_dir: Path) -> int:
    latest = 0
    for checkpoint in run_dir.glob("step-*.pt"):
        match = re.fullmatch(r"step-([0-9]+)\.pt", checkpoint.name)
        if match:
            latest = max(latest, int(match.group(1)))
    return latest


def _formal_training_started(run_dir: Path) -> bool:
    if _latest_step(run_dir) > 0:
        return True
    latest_record = run_dir / "latest_checkpoint.yaml"
    if latest_record.is_file():
        try:
            record = load_yaml(latest_record)
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            return True
        if not isinstance(record, dict):
            return True
        try:
            if int(record.get("step", 0)) > 0:
                return True
        except (TypeError, ValueError):
            return True
    deepspeed_root = run_dir / "ds_checkpoints"
    if deepspeed_root.is_dir():
        for candidate in deepspeed_root.iterdir():
            match = re.fullmatch(r"step-([0-9]+)", candidate.name)
            if candidate.is_dir() and match and int(match.group(1)) > 0:
                return True
    step_pattern = re.compile(r"\[deepspeed-train\] step=[1-9][0-9]*\b")
    for log_path in (run_dir / "logs").glob("sft_full_*steps.log"):
        with log_path.open("r", encoding="utf-8", errors="replace") as source:
            if any(step_pattern.search(line) is not None for line in source):
                return True
    return False


def _checkpoint_step(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        return int(payload.get("step", 0))
    finally:
        del payload


def _validate_labeled_audit(
    audit: dict[str, Any],
    *,
    labeled_root: Path,
    length_index: Path,
    bucket_manifest: Path,
    expected_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_paths = {
        "webdataset_root": str(labeled_root.resolve()),
        "length_index_path": str(length_index.resolve()),
        "bucket_manifest_path": str(bucket_manifest.resolve()),
    }
    for key, expected in expected_paths.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"Stage211D labeled audit {key} mismatch: "
                f"expected={expected!r} actual={audit.get(key)!r}"
            )
    expected_metrics = LABELED_EXPECTED if expected_metrics is None else expected_metrics
    for key, expected in expected_metrics.items():
        actual = audit.get(key)
        if key == "total_hours":
            if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1e-5:
                raise ValueError(
                    f"Stage211D labeled audit {key} mismatch: expected={expected} actual={actual}"
                )
        elif isinstance(expected, dict):
            if actual != expected:
                raise ValueError(
                    f"Stage211D labeled audit {key} mismatch: expected={expected} actual={actual}"
                )
        elif int(actual) != int(expected):
            raise ValueError(
                f"Stage211D labeled audit {key} mismatch: expected={expected} actual={actual}"
            )
    preparation = _label_preparation_proof(
        webdataset_root=labeled_root,
        length_index_path=length_index,
    )
    if audit.get("label_preparation") != preparation:
        raise ValueError(
            "Stage211D labeled audit differs from its label-preparation summary/log proof."
        )
    return dict(audit)


def _resolve_chain_inputs(
    *,
    output_dir: Path,
    requested_checkpoint: Path | None,
    requested_receipt: Path | None,
) -> tuple[Path, Path]:
    provenance_path = output_dir / "stage211_provenance.json"
    if provenance_path.is_file():
        provenance = _load_json(provenance_path, label="Stage211D provenance")
        checkpoint = Path(str(provenance.get("init_checkpoint_path") or "")).resolve()
        receipt = Path(str(provenance.get("promotion_receipt_path") or "")).resolve()
        if not checkpoint.is_file() or provenance.get("init_checkpoint_sha256") != sha256_file(
            checkpoint
        ):
            raise ValueError("Recorded Stage211D initial checkpoint is unavailable or changed.")
        if not receipt.is_file() or provenance.get("promotion_receipt_sha256") != sha256_file(
            receipt
        ):
            raise ValueError(
                "Recorded Stage211D logits-promotion receipt is unavailable or changed."
            )
        if requested_checkpoint is not None and requested_checkpoint.resolve() != checkpoint:
            raise ValueError("Requested Stage211D initialization differs from recorded provenance.")
        if requested_receipt is not None and requested_receipt.resolve() != receipt:
            raise ValueError(
                "Requested Stage211D promotion receipt differs from recorded provenance."
            )
        return checkpoint, receipt
    if requested_checkpoint is None or requested_receipt is None:
        raise ValueError(
            "A fresh Stage211D run requires --init-checkpoint and --logits-promotion-receipt."
        )
    checkpoint = requested_checkpoint.expanduser().resolve()
    receipt = requested_receipt.expanduser().resolve()
    for label, path in (
        ("Stage211C checkpoint", checkpoint),
        ("Stage211C promotion receipt", receipt),
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{label} is unavailable: {path}")
    return checkpoint, receipt


def _runner_command(
    *,
    output_dir: Path,
    config_dir: Path,
    bucket_manifest: Path,
    labeled_root: Path,
    length_index: Path,
    nano_checkpoint: Path,
    master_port: int,
    init_checkpoint: Path | None,
    promotion_receipt: Path | None,
    smoke: bool,
    dry_run: bool,
) -> list[str]:
    command = [
        str(PYTHON),
        str(RUNNER),
        "--phase",
        "sft",
        "--output-dir",
        str(output_dir),
        "--config-dir",
        str(config_dir),
        "--bucket-manifest",
        str(bucket_manifest),
        "--labeled-webdataset-root",
        str(labeled_root),
        "--labeled-length-index",
        str(length_index),
        "--nano-checkpoint",
        str(nano_checkpoint),
        "--master-port",
        str(master_port),
    ]
    if init_checkpoint is not None:
        command.extend(("--init-checkpoint", str(init_checkpoint)))
    if promotion_receipt is not None:
        command.extend(("--promotion-receipt", str(promotion_receipt)))
    if smoke:
        command.append("--smoke")
    if dry_run:
        command.extend(("--dry-run", "--skip-nano-weight-audit"))
    return command


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"[stage211-sft] command={shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _audit_smoke(
    *,
    smoke_dir: Path,
    init_checkpoint: Path,
    promotion_receipt: Path,
    bucket_manifest: Path,
    length_index: Path,
    max_peak_reserved_gib: float,
) -> dict[str, Any]:
    checkpoint = smoke_dir / "step-2.pt"
    log_path = smoke_dir / "logs" / "sft_smoke_2steps.log"
    if not checkpoint.is_file() or _checkpoint_step(checkpoint) != 2:
        raise ValueError("Stage211D smoke did not produce an exact step-2 checkpoint.")
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise ValueError(f"Stage211D smoke log is missing: {log_path}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "[deepspeed-train] step=2" not in log_text:
        raise ValueError("Stage211D smoke did not execute two training steps.")
    for pattern in BAD_SMOKE_PATTERNS:
        match = pattern.search(log_text)
        if match is not None:
            raise ValueError(f"Stage211D smoke contains a rejected condition: {match.group(0)}")
    peak_values = [
        float(value)
        for value in re.findall(
            r"peak_reserved=([0-9]+(?:\.[0-9]+)?)GiB",
            log_text,
        )
    ]
    if not peak_values:
        raise ValueError("Stage211D smoke lacks peak-reserved memory telemetry.")
    peak_reserved_gib = max(peak_values)
    if peak_reserved_gib > max_peak_reserved_gib:
        raise ValueError(
            f"Stage211D smoke peak memory is unsafe: "
            f"{peak_reserved_gib:.2f} GiB > {max_peak_reserved_gib:.2f} GiB"
        )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "labeled_sft_smoke",
        "phase": "sft",
        "complete": True,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "logits_promotion_receipt_path": str(promotion_receipt),
        "logits_promotion_receipt_sha256": sha256_file(promotion_receipt),
        "bucket_manifest_path": str(bucket_manifest),
        "bucket_manifest_sha256": sha256_file(bucket_manifest),
        "length_index_path": str(length_index),
        "length_index_sha256": sha256_file(length_index),
        "smoke_checkpoint_path": str(checkpoint),
        "smoke_checkpoint_sha256": sha256_file(checkpoint),
        "smoke_log_path": str(log_path),
        "smoke_log_sha256": sha256_file(log_path),
        "peak_reserved_gib": peak_reserved_gib,
        "max_peak_reserved_gib": max_peak_reserved_gib,
    }


def _validate_smoke_marker(
    marker_path: Path,
    *,
    init_checkpoint: Path,
    promotion_receipt: Path,
    bucket_manifest: Path,
    length_index: Path,
) -> dict[str, Any]:
    marker = _load_json(marker_path, label="Stage211D smoke marker")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "labeled_sft_smoke",
        "phase": "sft",
        "complete": True,
        "init_checkpoint_path": str(init_checkpoint),
        "init_checkpoint_sha256": sha256_file(init_checkpoint),
        "logits_promotion_receipt_path": str(promotion_receipt),
        "logits_promotion_receipt_sha256": sha256_file(promotion_receipt),
        "bucket_manifest_path": str(bucket_manifest),
        "bucket_manifest_sha256": sha256_file(bucket_manifest),
        "length_index_path": str(length_index),
        "length_index_sha256": sha256_file(length_index),
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"Stage211D smoke marker {key} mismatch.")
    smoke_checkpoint = Path(str(marker.get("smoke_checkpoint_path") or "")).resolve()
    if (
        not smoke_checkpoint.is_file()
        or sha256_file(smoke_checkpoint) != marker.get("smoke_checkpoint_sha256")
        or _checkpoint_step(smoke_checkpoint) != 2
    ):
        raise ValueError(
            f"Stage211D smoke checkpoint is unavailable, changed, or not step 2: {smoke_checkpoint}"
        )
    smoke_log = Path(str(marker.get("smoke_log_path") or "")).resolve()
    if not smoke_log.is_file() or sha256_file(smoke_log) != marker.get("smoke_log_sha256"):
        raise ValueError(f"Stage211D smoke log is unavailable or changed: {smoke_log}")
    log_text = smoke_log.read_text(encoding="utf-8", errors="replace")
    if "[deepspeed-train] step=2" not in log_text:
        raise ValueError("Stage211D smoke marker log lacks the second optimizer step.")
    for pattern in BAD_SMOKE_PATTERNS:
        match = pattern.search(log_text)
        if match is not None:
            raise ValueError(f"Stage211D smoke marker log contains: {match.group(0)}")
    peak_values = [
        float(value)
        for value in re.findall(
            r"peak_reserved=([0-9]+(?:\.[0-9]+)?)GiB",
            log_text,
        )
    ]
    peak = float(marker.get("peak_reserved_gib", float("nan")))
    peak_limit = float(marker.get("max_peak_reserved_gib", float("nan")))
    if (
        not peak_values
        or not math.isfinite(peak)
        or not math.isfinite(peak_limit)
        or peak < 0.0
        or peak_limit <= 0.0
        or peak > peak_limit
        or abs(peak - max(peak_values)) > 1e-9
    ):
        raise ValueError("Stage211D smoke marker memory contract mismatch.")
    return marker


def _validate_completion(
    completion_path: Path,
    *,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    completion = _load_json(completion_path, label="Stage211D completion report")
    labeled_root = Path(str(completion.get("labeled_webdataset_root") or "")).resolve()
    length_index = Path(str(completion.get("length_index_path") or "")).resolve()
    bucket_manifest = Path(str(completion.get("bucket_manifest_path") or "")).resolve()
    profile_path_value = completion.get("labeled_profile_receipt_path")
    labeled_profile: dict[str, Any] | None = None
    if profile_path_value is not None:
        profile_path = Path(str(profile_path_value)).resolve()
        if not profile_path.is_file() or completion.get(
            "labeled_profile_receipt_sha256"
        ) != sha256_file(profile_path):
            raise ValueError("Stage211D labeled profile receipt is unavailable or changed.")
        labeled_profile = validate_labeled_profile_receipt(
            profile_path,
            labeled_root=labeled_root,
            length_index=length_index,
            bucket_manifest=bucket_manifest,
        )
        labeled_expected = dict(labeled_profile["expected"])
    else:
        labeled_expected = dict(LABELED_EXPECTED)
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "labeled_sft_completion",
        "phase": "sft",
        "complete": True,
        "epochs": 1,
        "batch_size": 12,
        "world_size": 4,
        "frame_budget": 8_000,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
        "ctc_suppress_non_pronunciation_tokens": True,
        "ctc_suppressed_token_ids_count": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT),
        "ctc_suppressed_token_ids_sha256": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256),
        "teacher_projection_support_matches_student": True,
        **labeled_expected,
    }
    for key, value in expected.items():
        actual = completion.get(key)
        if key == "total_hours":
            if not math.isfinite(float(actual)) or abs(float(actual) - float(value)) > 1e-5:
                raise ValueError(f"Stage211D completion {key} mismatch.")
        elif actual != value:
            raise ValueError(f"Stage211D completion {key} mismatch.")
    validate_stage211_runtime_epoch_coverage(
        completion.get("runtime_epoch_coverage"),
        epochs=1,
        steps_per_epoch=int(labeled_expected["estimated_train_steps"]),
        label="sft",
    )
    for path_key, sha_key in (
        ("bucket_manifest_path", "bucket_manifest_sha256"),
        ("length_index_path", "length_index_sha256"),
        ("provenance_path", "provenance_sha256"),
        ("train_config_path", "train_config_sha256"),
        ("nano_teacher_checkpoint_path", "nano_teacher_checkpoint_sha256"),
        ("init_checkpoint_path", "init_checkpoint_sha256"),
        (
            "logits_promotion_receipt_path",
            "logits_promotion_receipt_sha256",
        ),
        ("completion_checkpoint_path", "completion_checkpoint_sha256"),
        ("training_log_path", "training_log_sha256"),
        ("smoke_marker_path", "smoke_marker_sha256"),
    ):
        path = Path(str(completion.get(path_key) or "")).resolve()
        if not path.is_file() or sha256_file(path) != completion.get(sha_key):
            raise ValueError(f"Stage211D completion artifact is unavailable or changed: {path}")
    _validate_smoke_marker(
        Path(str(completion["smoke_marker_path"])).resolve(),
        init_checkpoint=Path(str(completion["init_checkpoint_path"])).resolve(),
        promotion_receipt=Path(str(completion["logits_promotion_receipt_path"])).resolve(),
        bucket_manifest=Path(str(completion["bucket_manifest_path"])).resolve(),
        length_index=Path(str(completion["length_index_path"])).resolve(),
    )
    train_config_path = Path(str(completion["train_config_path"])).resolve()
    train_config = load_yaml(train_config_path)
    validate_stage211_phase_train_config(train_config, phase="sft")
    labeled_audit = completion.get("labeled_data_audit")
    if not isinstance(labeled_audit, dict):
        raise ValueError("Stage211D completion lacks its full labeled-data audit.")
    validated_labeled_audit = _validate_labeled_audit(
        labeled_audit,
        labeled_root=labeled_root,
        length_index=length_index,
        bucket_manifest=bucket_manifest,
        expected_metrics=labeled_expected,
    )
    if (
        labeled_profile is not None
        and labeled_profile.get("labeled_data_audit") != validated_labeled_audit
    ):
        raise ValueError("Stage211D completion audit differs from its labeled profile receipt.")
    provenance = _load_json(
        Path(str(completion["provenance_path"])).resolve(),
        label="Stage211D provenance",
    )
    if provenance.get("labeled_data_audit") != validated_labeled_audit:
        raise ValueError(
            "Stage211D completion labeled-data audit differs from immutable provenance."
        )
    configured_teacher_checkpoint = resolve_stage211_nano_teacher_checkpoint(train_config)
    recorded_teacher_checkpoint = Path(str(completion["nano_teacher_checkpoint_path"])).resolve()
    if configured_teacher_checkpoint != recorded_teacher_checkpoint:
        raise ValueError("Stage211D Nano teacher checkpoint differs from its train config.")
    recorded_checkpoint = Path(str(completion["completion_checkpoint_path"])).resolve()
    if checkpoint_path is not None and recorded_checkpoint != checkpoint_path.resolve():
        raise ValueError("Stage211D completion checkpoint path mismatch.")
    if _checkpoint_step(recorded_checkpoint) != labeled_expected["estimated_train_steps"]:
        raise ValueError("Stage211D completion checkpoint step mismatch.")
    return completion, recorded_checkpoint


def run_sft(args: argparse.Namespace) -> Path | None:
    output_dir = args.output_dir.expanduser().resolve()
    config_dir = args.config_dir.expanduser().resolve()
    labeled_root = args.labeled_webdataset_root.expanduser().resolve()
    length_index = (
        args.labeled_length_index.expanduser().resolve()
        if args.labeled_length_index is not None
        else (labeled_root / "webdataset_lengths.jsonl").resolve()
    )
    bucket_manifest = (
        args.bucket_manifest.expanduser().resolve()
        if args.bucket_manifest is not None
        else (labeled_root / "webdataset_buckets_audio_text" / "manifest.json").resolve()
    )
    nano_checkpoint = args.nano_checkpoint.expanduser().resolve()
    for label, path in (
        ("labeled root", labeled_root),
        ("labeled length index", length_index),
        ("labeled bucket manifest", bucket_manifest),
        ("Nano checkpoint", nano_checkpoint),
    ):
        exists = path.is_dir() if label == "labeled root" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"Stage211D {label} is unavailable: {path}")
    requested_profile = getattr(args, "labeled_profile_receipt", None)
    if requested_profile is not None:
        labeled_profile_path = requested_profile.expanduser().resolve()
        labeled_profile = validate_labeled_profile_receipt(
            labeled_profile_path,
            labeled_root=labeled_root,
            length_index=length_index,
            bucket_manifest=bucket_manifest,
        )
        labeled_expected = dict(labeled_profile["expected"])
        audit = _validate_labeled_audit(
            dict(labeled_profile["labeled_data_audit"]),
            labeled_root=labeled_root,
            length_index=length_index,
            bucket_manifest=bucket_manifest,
            expected_metrics=labeled_expected,
        )
    else:
        labeled_profile_path = None
        labeled_expected = dict(LABELED_EXPECTED)
        audit = _validate_labeled_audit(
            _audit_labeled_data(
                webdataset_root=labeled_root,
                length_index_path=length_index,
                bucket_manifest_path=bucket_manifest,
            ),
            labeled_root=labeled_root,
            length_index=length_index,
            bucket_manifest=bucket_manifest,
            expected_metrics=labeled_expected,
        )
    init_checkpoint, promotion_receipt = _resolve_chain_inputs(
        output_dir=output_dir,
        requested_checkpoint=args.init_checkpoint,
        requested_receipt=args.logits_promotion_receipt,
    )

    smoke_dir = Path(f"{output_dir}_smoke")
    smoke_marker_path = output_dir.parent / f"{output_dir.name}_smoke_passed.json"
    formal_training_started = _formal_training_started(output_dir)
    if smoke_marker_path.is_file() and not args.dry_run:
        _validate_smoke_marker(
            smoke_marker_path,
            init_checkpoint=init_checkpoint,
            promotion_receipt=promotion_receipt,
            bucket_manifest=bucket_manifest,
            length_index=length_index,
        )
    else:
        if formal_training_started and not args.dry_run:
            raise ValueError(
                "Stage211D formal training has progress but lacks its preflight smoke marker."
            )
        smoke_latest_step = _latest_step(smoke_dir)
        _run_command(
            _runner_command(
                output_dir=output_dir,
                config_dir=config_dir,
                bucket_manifest=bucket_manifest,
                labeled_root=labeled_root,
                length_index=length_index,
                nano_checkpoint=nano_checkpoint,
                master_port=int(args.master_port),
                init_checkpoint=(init_checkpoint if smoke_latest_step <= 0 else None),
                promotion_receipt=(promotion_receipt if smoke_latest_step <= 0 else None),
                smoke=True,
                dry_run=bool(args.dry_run),
            ),
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            _write_immutable_json(
                smoke_marker_path,
                _audit_smoke(
                    smoke_dir=smoke_dir,
                    init_checkpoint=init_checkpoint,
                    promotion_receipt=promotion_receipt,
                    bucket_manifest=bucket_manifest,
                    length_index=length_index,
                    max_peak_reserved_gib=float(args.max_peak_reserved_gib),
                ),
            )
            _validate_smoke_marker(
                smoke_marker_path,
                init_checkpoint=init_checkpoint,
                promotion_receipt=promotion_receipt,
                bucket_manifest=bucket_manifest,
                length_index=length_index,
            )
    if args.smoke_only or args.dry_run:
        return None

    latest_step = _latest_step(output_dir)
    _run_command(
        _runner_command(
            output_dir=output_dir,
            config_dir=config_dir,
            bucket_manifest=bucket_manifest,
            labeled_root=labeled_root,
            length_index=length_index,
            nano_checkpoint=nano_checkpoint,
            master_port=int(args.master_port),
            init_checkpoint=init_checkpoint if latest_step <= 0 else None,
            promotion_receipt=promotion_receipt if latest_step <= 0 else None,
            smoke=False,
            dry_run=False,
        ),
        dry_run=False,
    )

    completion_step = int(labeled_expected["estimated_train_steps"])
    completion_checkpoint = output_dir / f"step-{completion_step}.pt"
    if (
        not completion_checkpoint.is_file()
        or _checkpoint_step(completion_checkpoint) != completion_step
    ):
        raise ValueError(
            f"Stage211D lacks exact completion checkpoint step={completion_step}: "
            f"{completion_checkpoint}"
        )
    provenance_path = output_dir / "stage211_provenance.json"
    provenance = _load_json(provenance_path, label="Stage211D provenance")
    if (
        provenance.get("init_checkpoint_path") != str(init_checkpoint)
        or provenance.get("init_checkpoint_sha256") != sha256_file(init_checkpoint)
        or provenance.get("promotion_receipt_path") != str(promotion_receipt)
        or provenance.get("promotion_receipt_sha256") != sha256_file(promotion_receipt)
        or provenance.get("labeled_data_audit") != audit
    ):
        raise ValueError("Stage211D final provenance differs from the audited chain.")
    training_log = output_dir / "logs" / f"sft_full_{completion_step}steps.log"
    if not training_log.is_file() or training_log.stat().st_size <= 0:
        raise ValueError(f"Stage211D formal training log is missing: {training_log}")
    train_config_path = output_dir / "train_config.yaml"
    if not train_config_path.is_file():
        raise ValueError(f"Stage211D formal train config is missing: {train_config_path}")
    train_config = load_yaml(train_config_path)
    expected_train_config = {
        "max_steps": completion_step,
        "batch_size": 12,
        "batch_token_budget": 8_000,
        "length_bucket_frame_budget": 8_000,
        "length_bucket_drop_last": False,
        "skip_oversized_samples": False,
        "webdataset_skip_decode_errors": False,
    }
    for key, value in expected_train_config.items():
        if train_config.get(key) != value:
            raise ValueError(
                f"Stage211D train config {key} mismatch: "
                f"actual={train_config.get(key)!r} expected={value!r}"
            )
    validate_stage211_phase_train_config(train_config, phase="sft")
    nano_teacher_checkpoint_path = resolve_stage211_nano_teacher_checkpoint(train_config)
    runtime_epoch_coverage = audit_stage211_runtime_epoch_coverage(
        run_dir=output_dir,
        epochs=1,
        steps_per_epoch=completion_step,
    )
    completion_path = output_dir / "sft_complete.json"
    _write_immutable_json(
        completion_path,
        {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": "labeled_sft_completion",
            "phase": "sft",
            "complete": True,
            "epochs": 1,
            "batch_size": 12,
            "world_size": 4,
            "frame_budget": 8_000,
            "length_bucket_drop_last": False,
            "skip_oversized_samples": False,
            "webdataset_skip_decode_errors": False,
            "ctc_suppress_non_pronunciation_tokens": True,
            "ctc_suppressed_token_ids_count": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT),
            "ctc_suppressed_token_ids_sha256": (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256),
            "teacher_projection_support_matches_student": True,
            **labeled_expected,
            "labeled_data_audit": audit,
            "labeled_webdataset_root": str(labeled_root),
            "bucket_manifest_path": str(bucket_manifest),
            "bucket_manifest_sha256": sha256_file(bucket_manifest),
            "length_index_path": str(length_index),
            "length_index_sha256": sha256_file(length_index),
            "provenance_path": str(provenance_path),
            "provenance_sha256": sha256_file(provenance_path),
            "train_config_path": str(train_config_path),
            "train_config_sha256": sha256_file(train_config_path),
            "nano_teacher_checkpoint_path": str(nano_teacher_checkpoint_path),
            "nano_teacher_checkpoint_sha256": sha256_file(nano_teacher_checkpoint_path),
            "init_checkpoint_path": str(init_checkpoint),
            "init_checkpoint_sha256": sha256_file(init_checkpoint),
            "logits_promotion_receipt_path": str(promotion_receipt),
            "logits_promotion_receipt_sha256": sha256_file(promotion_receipt),
            "completion_checkpoint_path": str(completion_checkpoint),
            "completion_checkpoint_sha256": sha256_file(completion_checkpoint),
            "training_log_path": str(training_log),
            "training_log_sha256": sha256_file(training_log),
            "smoke_marker_path": str(smoke_marker_path),
            "smoke_marker_sha256": sha256_file(smoke_marker_path),
            "runtime_epoch_coverage": runtime_epoch_coverage,
            **(
                {
                    "labeled_profile_receipt_path": str(labeled_profile_path),
                    "labeled_profile_receipt_sha256": sha256_file(labeled_profile_path),
                }
                if labeled_profile_path is not None
                else {}
            ),
        },
    )
    _validate_completion(completion_path, checkpoint_path=completion_checkpoint)
    if args.final_checkpoint_path_output is not None:
        path_output = args.final_checkpoint_path_output.expanduser().resolve()
        path_output.parent.mkdir(parents=True, exist_ok=True)
        rendered = str(completion_checkpoint.resolve()) + "\n"
        if path_output.is_file() and path_output.read_text(encoding="utf-8") != rendered:
            raise ValueError(
                f"Refusing to overwrite a different Stage211D checkpoint path: {path_output}"
            )
        path_output.write_text(rendered, encoding="utf-8")
    print(
        f"[stage211-sft] complete checkpoint={completion_checkpoint} report={completion_path}",
        flush=True,
    )
    return completion_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run restart-safe Stage211D labeled pronunciation-only CTC SFT from "
            "the strictly promoted Stage211C checkpoint."
        )
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--logits-promotion-receipt", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--labeled-webdataset-root",
        type=Path,
        default=DEFAULT_LABELED_ROOT,
    )
    parser.add_argument("--labeled-length-index", type=Path, default=None)
    parser.add_argument("--bucket-manifest", type=Path, default=None)
    parser.add_argument(
        "--labeled-profile-receipt",
        type=Path,
        default=DEFAULT_LABELED_PROFILE_RECEIPT,
    )
    parser.add_argument("--nano-checkpoint", type=Path, default=DEFAULT_NANO_CHECKPOINT)
    parser.add_argument("--master-port", type=int, default=29634)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=22.0)
    parser.add_argument("--final-checkpoint-path-output", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_peak_reserved_gib <= 0.0:
        parser.error("--max-peak-reserved-gib must be positive")
    run_sft(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
