from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from rwkvasr.eval.stage211_gate import (
    sha256_file,
    validate_stage211_runtime_epoch_coverage,
)


_PLAN_NAME = "stage211_storage_compaction_plan.json"
_RECEIPT_NAME = "stage211_storage_compaction_receipt.json"
_SAFE_TAG = re.compile(r"(?:best|epoch-[1-9][0-9]*|step-[1-9][0-9]*)\Z")
_PRESERVED_DS_FILES = {"latest", "zero_to_fp32.py"}
_CORRECTION_POLICY = "export_only_after_completed_correction_v1"


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 {label} is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage211 {label} must be a JSON object: {path}")
    return payload


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 {label} is unavailable: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage211 {label} must be a YAML mapping: {path}")
    return payload


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Refusing to overwrite changed Stage211 artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _yaml_bytes(payload: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(dict(payload), sort_keys=False).encode("utf-8")


def _replace_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _yaml_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_bound_file(payload: Mapping[str, Any], path_key: str, sha_key: str) -> Path:
    path = Path(str(payload.get(path_key) or "")).expanduser().resolve()
    expected_sha256 = str(payload.get(sha_key) or "")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 compaction bound file is unavailable: {path}")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError(f"Stage211 compaction bound file changed: {path}")
    return path


def _tree_stats(path: Path) -> tuple[int, int]:
    bytes_total = 0
    files_total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"Stage211 compaction refuses a symlink: {child}")
        if child.is_file():
            bytes_total += child.stat().st_size
            files_total += 1
    return bytes_total, files_total


def _safe_tag_path(ds_root: Path, tag: str) -> Path:
    if _SAFE_TAG.fullmatch(tag) is None:
        raise ValueError(f"Stage211 compaction encountered an unsafe checkpoint tag: {tag!r}")
    path = (ds_root / tag).resolve()
    if path.parent != ds_root or path.name != tag:
        raise ValueError(f"Stage211 compaction checkpoint path escapes its run: {path}")
    return path


def _checkpoint_tag(
    payload: Mapping[str, Any],
    *,
    ds_root: Path,
    label: str,
) -> str:
    tag = str(payload.get("resume_tag") or "")
    tag_path = _safe_tag_path(ds_root, tag)
    configured_root = Path(str(payload.get("deepspeed_checkpoint_dir") or "")).resolve()
    if configured_root not in {ds_root, tag_path}:
        raise ValueError(f"Stage211 {label} points outside its DeepSpeed directory.")
    if not tag_path.is_dir() or tag_path.is_symlink():
        raise ValueError(f"Stage211 {label} recovery tag is unavailable: {tag_path}")
    return tag


def _validate_receipt(
    receipt_path: Path,
    *,
    expected_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="curriculum coverage receipt")
    schema_version = receipt.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError(
            "Stage211 storage compaction requires a supported curriculum receipt schema."
        )
    expected = {
        "pipeline": "stage211",
        "artifact": "curriculum_coverage",
        "complete": True,
        "full_data_profile": True,
        "epochs": 3,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 storage compaction requires a completed three-epoch receipt.")
    if expected_receipt is not None:
        for key in (
            "phase",
            "difficulty",
            "run_dir",
            "steps",
            "steps_per_epoch",
            "completion_checkpoint_path",
            "completion_checkpoint_sha256",
        ):
            if receipt.get(key) != expected_receipt.get(key):
                raise ValueError(f"Stage211 storage compaction receipt binding changed: {key}")
    run_dir = Path(str(receipt.get("run_dir") or "")).expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Stage211 completed run directory is unavailable: {run_dir}")
    return receipt, run_dir


def _validate_exported_coverage(receipt: Mapping[str, Any]) -> None:
    steps_per_epoch = int(receipt.get("steps_per_epoch", 0))
    steps = int(receipt.get("steps", 0))
    if steps_per_epoch <= 0 or steps != 3 * steps_per_epoch:
        raise ValueError("Stage211 compaction receipt has invalid three-epoch step coverage.")
    validate_stage211_runtime_epoch_coverage(
        receipt.get("runtime_epoch_coverage"),
        epochs=3,
        steps_per_epoch=steps_per_epoch,
        label="storage compaction",
    )
    _validate_bound_file(
        receipt,
        "completion_checkpoint_path",
        "completion_checkpoint_sha256",
    )
    audit = receipt.get("parameter_delta_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("complete") is not True
        or audit.get("policy") != "stage211_timemixer_and_input_projection_only"
        or int(audit.get("forbidden_changed_tensors", -1)) != 0
    ):
        raise ValueError("Stage211 compaction parameter-delta proof is invalid.")


def _preserved_tags(run_dir: Path, receipt: Mapping[str, Any]) -> tuple[Path, list[str]]:
    ds_root = (run_dir / "ds_checkpoints").resolve()
    if not ds_root.is_dir() or ds_root.is_symlink() or ds_root.parent != run_dir:
        raise ValueError(f"Stage211 DeepSpeed checkpoint directory is invalid: {ds_root}")
    latest = _load_yaml(run_dir / "latest_checkpoint.yaml", label="latest checkpoint pointer")
    if (
        int(latest.get("step", -1)) != int(receipt["steps"])
        or int(latest.get("epoch", -1)) != 3
        or int(latest.get("epoch_batch_offset", -1)) != 0
    ):
        raise ValueError("Stage211 compaction refuses a nonterminal latest checkpoint.")
    latest_checkpoint = Path(str(latest.get("checkpoint_path") or "")).resolve()
    expected_epoch_checkpoint = run_dir / "epoch-3.pt"
    if latest_checkpoint != expected_epoch_checkpoint or not latest_checkpoint.is_file():
        raise ValueError("Stage211 compaction latest checkpoint is not the terminal epoch export.")
    best = _load_yaml(run_dir / "best_checkpoint.yaml", label="best checkpoint pointer")
    best_checkpoint = Path(str(best.get("checkpoint_path") or "")).resolve()
    if best_checkpoint != run_dir / "best.pt" or not best_checkpoint.is_file():
        raise ValueError("Stage211 compaction best checkpoint export is unavailable.")
    tags = {
        _checkpoint_tag(latest, ds_root=ds_root, label="latest checkpoint"),
        _checkpoint_tag(best, ds_root=ds_root, label="best checkpoint"),
    }
    latest_tag_file = ds_root / "latest"
    if not latest_tag_file.is_file() or latest_tag_file.read_text(encoding="utf-8").strip() != str(
        latest["resume_tag"]
    ):
        raise ValueError("Stage211 DeepSpeed latest tag does not match the terminal pointer.")
    unknown_files = sorted(
        child.name
        for child in ds_root.iterdir()
        if child.is_file() and child.name not in _PRESERVED_DS_FILES
    )
    if unknown_files:
        raise ValueError(f"Stage211 compaction found unknown DeepSpeed files: {unknown_files}")
    return ds_root, sorted(tags)


def _new_plan(
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    run_dir: Path,
    ds_root: Path,
    preserved_tags: list[str],
) -> dict[str, Any]:
    remove: list[dict[str, Any]] = []
    for child in sorted(ds_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        path = _safe_tag_path(ds_root, child.name)
        if path.is_symlink():
            raise ValueError(f"Stage211 compaction refuses a symlink: {path}")
        if child.name in preserved_tags:
            continue
        bytes_total, files_total = _tree_stats(path)
        remove.append(
            {
                "tag": child.name,
                "path": str(path),
                "bytes": bytes_total,
                "files": files_total,
            }
        )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "completed_segment_storage_compaction_plan",
        "policy": "preserve_latest_and_best_deepspeed_tags_v1",
        "phase": receipt["phase"],
        "difficulty": receipt["difficulty"],
        "run_dir": str(run_dir),
        "curriculum_receipt_path": str(receipt_path),
        "curriculum_receipt_sha256": sha256_file(receipt_path),
        "ds_checkpoint_root": str(ds_root),
        "preserved_tags": preserved_tags,
        "remove": remove,
        "bytes_planned": sum(int(record["bytes"]) for record in remove),
        "files_planned": sum(int(record["files"]) for record in remove),
        "complete": False,
    }


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    run_dir: Path,
    ds_root: Path,
    preserved_tags: list[str],
) -> list[Path]:
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "completed_segment_storage_compaction_plan",
        "policy": "preserve_latest_and_best_deepspeed_tags_v1",
        "phase": receipt["phase"],
        "difficulty": receipt["difficulty"],
        "run_dir": str(run_dir),
        "curriculum_receipt_path": str(receipt_path),
        "curriculum_receipt_sha256": sha256_file(receipt_path),
        "ds_checkpoint_root": str(ds_root),
        "preserved_tags": preserved_tags,
        "complete": False,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 storage compaction plan binding changed.")
    records = plan.get("remove")
    if not isinstance(records, list):
        raise ValueError("Stage211 storage compaction plan lacks removal records.")
    paths: list[Path] = []
    tags: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Stage211 storage compaction plan has an invalid record.")
        tag = str(record.get("tag") or "")
        path = _safe_tag_path(ds_root, tag)
        if str(path) != record.get("path") or tag in preserved_tags:
            raise ValueError("Stage211 storage compaction plan removal path changed.")
        if int(record.get("bytes", -1)) < 0 or int(record.get("files", -1)) < 0:
            raise ValueError("Stage211 storage compaction plan has invalid size evidence.")
        tags.append(tag)
        paths.append(path)
    if tags != sorted(set(tags)):
        raise ValueError("Stage211 storage compaction plan tags are not unique and sorted.")
    if int(plan.get("bytes_planned", -1)) != sum(int(record["bytes"]) for record in records) or int(
        plan.get("files_planned", -1)
    ) != sum(int(record["files"]) for record in records):
        raise ValueError("Stage211 storage compaction plan totals changed.")
    current_tags = {
        child.name
        for child in ds_root.iterdir()
        if child.is_dir() and child.name not in preserved_tags
    }
    if not current_tags.issubset(set(tags)):
        raise ValueError("Stage211 storage compaction found an unplanned checkpoint tag.")
    return paths


def _completion_receipt(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    records = list(plan["remove"])
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "completed_segment_storage_compaction",
        "policy": plan["policy"],
        "phase": plan["phase"],
        "difficulty": plan["difficulty"],
        "run_dir": plan["run_dir"],
        "curriculum_receipt_path": plan["curriculum_receipt_path"],
        "curriculum_receipt_sha256": plan["curriculum_receipt_sha256"],
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "preserved_tags": plan["preserved_tags"],
        "removed_tags": [record["tag"] for record in records],
        "bytes_planned": plan["bytes_planned"],
        "files_planned": plan["files_planned"],
        "complete": True,
    }


def compact_stage211_completed_segment(
    *,
    curriculum_receipt_path: Path,
    expected_receipt: Mapping[str, Any] | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    receipt_path = curriculum_receipt_path.expanduser().resolve()
    receipt, run_dir = _validate_receipt(
        receipt_path,
        expected_receipt=expected_receipt,
    )
    ds_root, preserved_tags = _preserved_tags(run_dir, receipt)
    plan_path = run_dir / _PLAN_NAME
    completion_path = run_dir / _RECEIPT_NAME

    if plan_path.exists():
        plan = _load_json(plan_path, label="storage compaction plan")
    else:
        _validate_exported_coverage(receipt)
        plan = _new_plan(
            receipt_path=receipt_path,
            receipt=receipt,
            run_dir=run_dir,
            ds_root=ds_root,
            preserved_tags=preserved_tags,
        )
        _write_immutable_json(plan_path, plan)
    removal_paths = _validate_plan(
        plan,
        receipt_path=receipt_path,
        receipt=receipt,
        run_dir=run_dir,
        ds_root=ds_root,
        preserved_tags=preserved_tags,
    )
    if not execute:
        return {
            **plan,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
        }
    expected_completion = _completion_receipt(plan_path, plan)
    if completion_path.exists():
        existing = _load_json(completion_path, label="storage compaction receipt")
        if existing != expected_completion:
            raise ValueError("Stage211 storage compaction completion receipt changed.")
    else:
        for path in removal_paths:
            if path.exists():
                if not path.is_dir() or path.is_symlink():
                    raise ValueError(f"Stage211 compaction refuses checkpoint path: {path}")
                shutil.rmtree(path)
        for path in removal_paths:
            if path.exists():
                raise ValueError(f"Stage211 compaction did not remove checkpoint tag: {path}")
        _write_immutable_json(completion_path, expected_completion)

    for path in removal_paths:
        if path.exists():
            raise ValueError(f"Stage211 compacted checkpoint tag reappeared: {path}")
    for tag in preserved_tags:
        if not _safe_tag_path(ds_root, tag).is_dir():
            raise ValueError(f"Stage211 compaction lost a preserved checkpoint tag: {tag}")
    return {
        **expected_completion,
        "receipt_path": str(completion_path.resolve()),
        "receipt_sha256": sha256_file(completion_path),
    }


def _validate_correction_receipt(
    receipt_path: Path,
    *,
    expected_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="post-coverage correction receipt")
    if expected_receipt is not None and receipt != dict(expected_receipt):
        raise ValueError("Stage211 correction compaction receipt binding changed.")
    if (
        receipt.get("schema_version") != 2
        or receipt.get("pipeline") != "stage211"
        or receipt.get("artifact") != "post_coverage_correction"
        or receipt.get("phase") not in {"mixer", "block", "logits"}
        or receipt.get("complete") is not True
        or int(receipt.get("round", 0)) <= 0
        or int(receipt.get("epochs", 0)) != 1
    ):
        raise ValueError("Stage211 correction compaction requires a completed correction receipt.")
    run_dir = Path(str(receipt.get("run_dir") or "")).expanduser().resolve()
    if not run_dir.is_dir() or receipt_path != run_dir / "correction_receipt.json":
        raise ValueError("Stage211 correction receipt is outside its run directory.")
    steps = int(receipt.get("steps", 0))
    hours = float(receipt.get("hours", 0.0))
    if (
        steps <= 0
        or int(receipt.get("steps_per_epoch", 0)) != steps
        or int(receipt.get("rows", 0)) <= 0
        or not math.isfinite(hours)
        or hours <= 0.0
    ):
        raise ValueError("Stage211 correction receipt has invalid one-epoch coverage.")
    validate_stage211_runtime_epoch_coverage(
        receipt.get("runtime_epoch_coverage"),
        epochs=1,
        steps_per_epoch=steps,
        label="correction storage compaction",
    )
    completion_checkpoint = _validate_bound_file(
        receipt,
        "completion_checkpoint_path",
        "completion_checkpoint_sha256",
    )
    if completion_checkpoint != run_dir / f"step-{steps}.pt":
        raise ValueError("Stage211 correction completion checkpoint path is not terminal.")
    for path_key, sha_key in (
        ("provenance_path", "provenance_sha256"),
        ("train_config_path", "train_config_sha256"),
        ("smoke_marker_path", "smoke_marker_sha256"),
        ("layer_focus_path", "layer_focus_sha256"),
    ):
        _validate_bound_file(receipt, path_key, sha_key)
    audit = receipt.get("parameter_delta_audit")
    if (
        not isinstance(audit, dict)
        or audit.get("complete") is not True
        or audit.get("policy") != "stage211_timemixer_and_input_projection_only"
        or int(audit.get("forbidden_changed_tensors", -1)) != 0
    ):
        raise ValueError("Stage211 correction compaction parameter-delta proof is invalid.")
    return receipt, run_dir


def _correction_export_latest_state(
    *,
    run_dir: Path,
    receipt: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    epoch_export = (run_dir / "epoch-1.pt").resolve()
    if not epoch_export.is_file() or epoch_export.stat().st_size <= 0:
        raise ValueError("Stage211 correction terminal epoch export is unavailable.")
    return epoch_export, {
        "checkpoint_type": "export",
        "step": int(receipt["steps"]),
        "epoch": 1,
        "epoch_batch_offset": 0,
        "checkpoint_path": str(epoch_export),
    }


def _validate_correction_deepspeed_state(
    *,
    run_dir: Path,
    receipt: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    ds_root = (run_dir / "ds_checkpoints").resolve()
    if not ds_root.is_dir() or ds_root.is_symlink() or ds_root.parent != run_dir:
        raise ValueError(f"Stage211 correction DeepSpeed directory is invalid: {ds_root}")
    unknown_files = sorted(
        child.name
        for child in ds_root.iterdir()
        if child.is_file() and child.name not in _PRESERVED_DS_FILES
    )
    if unknown_files:
        raise ValueError(
            f"Stage211 correction compaction found unknown DeepSpeed files: {unknown_files}"
        )
    for child in ds_root.iterdir():
        if child.is_symlink():
            raise ValueError(f"Stage211 correction compaction refuses a symlink: {child}")
        if child.is_dir():
            _safe_tag_path(ds_root, child.name)
    latest_path = (run_dir / "latest_checkpoint.yaml").resolve()
    _load_yaml(latest_path, label="correction latest checkpoint pointer")
    epoch_export, export_latest = _correction_export_latest_state(
        run_dir=run_dir,
        receipt=receipt,
    )
    return ds_root, latest_path, epoch_export, export_latest


def _new_correction_plan(
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    run_dir: Path,
    ds_root: Path,
    latest_path: Path,
    epoch_export: Path,
    export_latest: Mapping[str, Any],
) -> dict[str, Any]:
    latest = _load_yaml(latest_path, label="correction latest checkpoint pointer")
    expected_latest = {
        "checkpoint_type": "deepspeed",
        "step": int(receipt["steps"]),
        "epoch": 1,
        "epoch_batch_offset": 0,
        "checkpoint_path": str(epoch_export),
        "deepspeed_checkpoint_dir": str(ds_root / "epoch-1"),
        "resume_tag": "epoch-1",
    }
    if any(latest.get(key) != value for key, value in expected_latest.items()):
        raise ValueError("Stage211 correction compaction refuses a nonterminal latest checkpoint.")
    latest_tag_path = ds_root / "latest"
    if (
        not latest_tag_path.is_file()
        or latest_tag_path.read_text(encoding="utf-8").strip() != "epoch-1"
    ):
        raise ValueError("Stage211 correction DeepSpeed latest tag is not terminal.")
    remove: list[dict[str, Any]] = []
    for child in sorted(ds_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        path = _safe_tag_path(ds_root, child.name)
        if path.is_symlink():
            raise ValueError(f"Stage211 correction compaction refuses a symlink: {path}")
        bytes_total, files_total = _tree_stats(path)
        remove.append(
            {
                "tag": child.name,
                "path": str(path),
                "bytes": bytes_total,
                "files": files_total,
            }
        )
    if "epoch-1" not in {record["tag"] for record in remove}:
        raise ValueError("Stage211 correction terminal DeepSpeed tag is unavailable.")
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "completed_correction_storage_compaction_plan",
        "policy": _CORRECTION_POLICY,
        "phase": receipt["phase"],
        "round": int(receipt["round"]),
        "run_dir": str(run_dir),
        "correction_receipt_path": str(receipt_path),
        "correction_receipt_sha256": sha256_file(receipt_path),
        "ds_checkpoint_root": str(ds_root),
        "terminal_epoch_export_path": str(epoch_export),
        "terminal_epoch_export_sha256": sha256_file(epoch_export),
        "latest_checkpoint_path": str(latest_path),
        "latest_checkpoint_before_sha256": sha256_file(latest_path),
        "latest_checkpoint_after": dict(export_latest),
        "deepspeed_latest_path": str(latest_tag_path),
        "deepspeed_latest_sha256": sha256_file(latest_tag_path),
        "remove": remove,
        "bytes_planned": sum(int(record["bytes"]) for record in remove),
        "files_planned": sum(int(record["files"]) for record in remove),
        "complete": False,
    }


def _validate_correction_plan(
    plan: Mapping[str, Any],
    *,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    run_dir: Path,
    ds_root: Path,
    latest_path: Path,
    epoch_export: Path,
    export_latest: Mapping[str, Any],
    completion_exists: bool,
) -> list[Path]:
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "completed_correction_storage_compaction_plan",
        "policy": _CORRECTION_POLICY,
        "phase": receipt["phase"],
        "round": int(receipt["round"]),
        "run_dir": str(run_dir),
        "correction_receipt_path": str(receipt_path),
        "correction_receipt_sha256": sha256_file(receipt_path),
        "ds_checkpoint_root": str(ds_root),
        "terminal_epoch_export_path": str(epoch_export),
        "terminal_epoch_export_sha256": sha256_file(epoch_export),
        "latest_checkpoint_path": str(latest_path),
        "latest_checkpoint_after": dict(export_latest),
        "deepspeed_latest_path": str(ds_root / "latest"),
        "complete": False,
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 correction storage compaction plan binding changed.")
    latest_bytes = latest_path.read_bytes()
    latest_after_bytes = _yaml_bytes(export_latest)
    if (
        hashlib.sha256(latest_bytes).hexdigest() != plan.get("latest_checkpoint_before_sha256")
        and latest_bytes != latest_after_bytes
    ):
        raise ValueError("Stage211 correction latest checkpoint pointer changed.")
    latest_tag_path = ds_root / "latest"
    if latest_tag_path.exists() and sha256_file(latest_tag_path) != plan.get(
        "deepspeed_latest_sha256"
    ):
        raise ValueError("Stage211 correction DeepSpeed latest tag changed.")
    records = plan.get("remove")
    if not isinstance(records, list):
        raise ValueError("Stage211 correction compaction plan lacks removal records.")
    paths: list[Path] = []
    tags: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Stage211 correction compaction plan has an invalid record.")
        tag = str(record.get("tag") or "")
        path = _safe_tag_path(ds_root, tag)
        if str(path) != record.get("path"):
            raise ValueError("Stage211 correction compaction removal path changed.")
        if int(record.get("bytes", -1)) < 0 or int(record.get("files", -1)) < 0:
            raise ValueError("Stage211 correction compaction plan has invalid size evidence.")
        if path.exists():
            if completion_exists:
                raise ValueError(f"Stage211 compacted correction checkpoint tag reappeared: {path}")
            if not path.is_dir() or path.is_symlink():
                raise ValueError(f"Stage211 correction compaction refuses checkpoint path: {path}")
            current_bytes, current_files = _tree_stats(path)
            if current_bytes != int(record["bytes"]) or current_files != int(record["files"]):
                raise ValueError(
                    f"Stage211 correction checkpoint state changed after planning: {path}"
                )
        tags.append(tag)
        paths.append(path)
    if tags != sorted(set(tags)) or "epoch-1" not in tags:
        raise ValueError("Stage211 correction compaction tags are invalid.")
    if int(plan.get("bytes_planned", -1)) != sum(int(record["bytes"]) for record in records) or int(
        plan.get("files_planned", -1)
    ) != sum(int(record["files"]) for record in records):
        raise ValueError("Stage211 correction compaction plan totals changed.")
    current_tags = {child.name for child in ds_root.iterdir() if child.is_dir()}
    if not current_tags.issubset(set(tags)):
        raise ValueError("Stage211 correction compaction found an unplanned checkpoint tag.")
    return paths


def _correction_completion_receipt(plan_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "completed_correction_storage_compaction",
        "policy": plan["policy"],
        "phase": plan["phase"],
        "round": plan["round"],
        "run_dir": plan["run_dir"],
        "correction_receipt_path": plan["correction_receipt_path"],
        "correction_receipt_sha256": plan["correction_receipt_sha256"],
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "removed_tags": [record["tag"] for record in plan["remove"]],
        "bytes_planned": plan["bytes_planned"],
        "files_planned": plan["files_planned"],
        "latest_checkpoint_path": plan["latest_checkpoint_path"],
        "latest_checkpoint_sha256": hashlib.sha256(
            _yaml_bytes(plan["latest_checkpoint_after"])
        ).hexdigest(),
        "terminal_epoch_export_path": plan["terminal_epoch_export_path"],
        "terminal_epoch_export_sha256": plan["terminal_epoch_export_sha256"],
        "complete": True,
    }


def compact_stage211_completed_correction(
    *,
    correction_receipt_path: Path,
    expected_receipt: Mapping[str, Any] | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    receipt_path = correction_receipt_path.expanduser().resolve()
    receipt, run_dir = _validate_correction_receipt(
        receipt_path,
        expected_receipt=expected_receipt,
    )
    ds_root, latest_path, epoch_export, export_latest = _validate_correction_deepspeed_state(
        run_dir=run_dir,
        receipt=receipt,
    )
    plan_path = run_dir / _PLAN_NAME
    completion_path = run_dir / _RECEIPT_NAME
    if plan_path.exists():
        plan = _load_json(plan_path, label="correction storage compaction plan")
    else:
        plan = _new_correction_plan(
            receipt_path=receipt_path,
            receipt=receipt,
            run_dir=run_dir,
            ds_root=ds_root,
            latest_path=latest_path,
            epoch_export=epoch_export,
            export_latest=export_latest,
        )
        _write_immutable_json(plan_path, plan)
    removal_paths = _validate_correction_plan(
        plan,
        receipt_path=receipt_path,
        receipt=receipt,
        run_dir=run_dir,
        ds_root=ds_root,
        latest_path=latest_path,
        epoch_export=epoch_export,
        export_latest=export_latest,
        completion_exists=completion_path.exists(),
    )
    if not execute:
        return {
            **plan,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
        }
    expected_completion = _correction_completion_receipt(plan_path, plan)
    if completion_path.exists():
        existing = _load_json(completion_path, label="correction storage compaction receipt")
        if existing != expected_completion:
            raise ValueError("Stage211 correction storage compaction receipt changed.")
    else:
        for path in removal_paths:
            if path.exists():
                if not path.is_dir() or path.is_symlink():
                    raise ValueError(
                        f"Stage211 correction compaction refuses checkpoint path: {path}"
                    )
                shutil.rmtree(path)
        (ds_root / "latest").unlink(missing_ok=True)
        _replace_yaml(latest_path, export_latest)
        for path in removal_paths:
            if path.exists():
                raise ValueError(
                    f"Stage211 correction compaction did not remove checkpoint tag: {path}"
                )
        _write_immutable_json(completion_path, expected_completion)
    for path in removal_paths:
        if path.exists():
            raise ValueError(f"Stage211 compacted correction checkpoint tag reappeared: {path}")
    if (ds_root / "latest").exists():
        raise ValueError("Stage211 compacted correction DeepSpeed latest tag reappeared.")
    if latest_path.read_bytes() != _yaml_bytes(export_latest):
        raise ValueError("Stage211 compacted correction latest pointer is not export-only.")
    return {
        **expected_completion,
        "receipt_path": str(completion_path.resolve()),
        "receipt_sha256": sha256_file(completion_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compact restart-safe DeepSpeed state for a completed Stage211 segment or correction."
        )
    )
    receipt_group = parser.add_mutually_exclusive_group(required=True)
    receipt_group.add_argument("--curriculum-receipt", type=Path)
    receipt_group.add_argument("--correction-receipt", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.correction_receipt is not None:
        result = compact_stage211_completed_correction(
            correction_receipt_path=args.correction_receipt,
            execute=not args.plan_only,
        )
    else:
        result = compact_stage211_completed_segment(
            curriculum_receipt_path=args.curriculum_receipt,
            execute=not args.plan_only,
        )
    if args.plan_only:
        scope = (
            f"round={result['round']}"
            if args.correction_receipt is not None
            else f"difficulty={result['difficulty']}"
        )
        print(
            "stage211_storage_compaction_planned "
            f"phase={result['phase']} {scope} "
            f"remove_tags={len(result['remove'])} "
            f"bytes_planned={result['bytes_planned']} "
            f"plan_sha256={result['plan_sha256']}",
            flush=True,
        )
        return 0
    scope = (
        f"round={result['round']}"
        if args.correction_receipt is not None
        else f"difficulty={result['difficulty']}"
    )
    print(
        "stage211_storage_compaction_complete "
        f"phase={result['phase']} {scope} "
        f"removed_tags={len(result['removed_tags'])} "
        f"bytes_planned={result['bytes_planned']} "
        f"receipt_sha256={result['receipt_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
