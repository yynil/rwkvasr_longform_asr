#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from rwkvasr.data import load_webdataset_bucket_manifest

try:
    from scripts.build_stage211_social_vad_boundaries import (
        INVENTORY_ARTIFACT as BOUNDARY_INVENTORY_ARTIFACT,
        exact_duplicate_decisions,
        validate_source_inventory,
        validate_source_payload,
    )
    from scripts.build_stage211_supplemental_natural_manifest import (
        DEFAULT_FIXED_EVAL_MANIFEST,
        BucketManifestWriter,
        _load_fixed_eval_split,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from build_stage211_social_vad_boundaries import (
        INVENTORY_ARTIFACT as BOUNDARY_INVENTORY_ARTIFACT,
        exact_duplicate_decisions,
        validate_source_inventory,
        validate_source_payload,
    )
    from build_stage211_supplemental_natural_manifest import (
        DEFAULT_FIXED_EVAL_MANIFEST,
        BucketManifestWriter,
        _load_fixed_eval_split,
    )


DEFAULT_BOUNDARY_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_boundaries_v1/boundary_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_social_vad_materialized_v1"
SOURCE_ARTIFACT = "stage211_social_vad_materialized_source"
MATERIALIZED_INVENTORY_ARTIFACT = "stage211_social_vad_materialized_inventory"
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
BUCKET_WIDTH = 80
ENTRIES_PER_PART = 100_000


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if path.is_file():
        if path.read_bytes() != rendered:
            raise ValueError(f"Refusing to replace a different Stage211 materialization: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(rendered)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def validate_boundary_inventory(
    path: Path, *, verify_source_part_hashes: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.expanduser().resolve()
    boundary = _load_json(path, label="Stage211 social VAD boundary inventory")
    expected = {
        "schema_version": 1,
        "artifact": BOUNDARY_INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "vad_boundaries_only",
    }
    if any(boundary.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 social VAD boundary inventory contract mismatch.")
    source_inventory_path = Path(str(boundary.get("source_inventory_path") or ""))
    source_inventory = validate_source_inventory(
        source_inventory_path,
        verify_media_hashes=False,
    )
    if boundary.get("source_inventory_sha256") != source_inventory["_inventory_sha256"]:
        raise ValueError("Stage211 social VAD boundary source inventory changed.")
    part_records = boundary.get("part_records")
    if (
        not isinstance(part_records, list)
        or len(part_records) != len(source_inventory["source_records"])
        or int(boundary.get("source_files", -1)) != len(part_records)
    ):
        raise ValueError("Stage211 social VAD boundary source parts are incomplete.")
    duplicate_of = exact_duplicate_decisions(source_inventory)
    admitted_sources = 0
    duplicate_sources = 0
    no_speech_sources = 0
    admitted_segments = 0
    admitted_duration_ms = 0
    for source_index, record in enumerate(part_records):
        if not isinstance(record, dict) or int(record.get("source_index", -1)) != source_index:
            raise ValueError("Stage211 social VAD boundary part order is invalid.")
        part_path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            not part_path.is_file()
            or int(record.get("size_bytes", -1)) != part_path.stat().st_size
            or len(str(record.get("sha256") or "")) != 64
        ):
            raise ValueError(f"Stage211 social VAD boundary part changed: {part_path}")
        if verify_source_part_hashes and _sha256(part_path) != record["sha256"]:
            raise ValueError(f"Stage211 social VAD boundary part SHA-256 changed: {part_path}")
        payload = _load_json(part_path, label=f"Stage211 social VAD source {source_index}")
        source = source_inventory["source_records"][source_index]
        validate_source_payload(
            payload,
            inventory=source_inventory,
            source_index=source_index,
            duplicate_of=duplicate_of.get(str(source["path"])),
        )
        if payload["status"] == "admitted":
            admitted_sources += 1
            admitted_segments += int(payload["admitted_segments"])
            admitted_duration_ms += int(payload["admitted_duration_ms"])
        elif payload["status"] == "exact_duplicate_skipped":
            duplicate_sources += 1
        elif payload["status"] == "no_speech_rejected":
            no_speech_sources += 1
    if (
        admitted_sources != int(boundary.get("admitted_source_files", -1))
        or duplicate_sources != int(boundary.get("exact_duplicate_source_files", -1))
        or no_speech_sources != int(boundary.get("no_speech_source_files", -1))
        or admitted_segments != int(boundary.get("admitted_segments", -1))
        or not math.isclose(
            admitted_duration_ms / 3_600_000.0,
            float(boundary.get("admitted_hours", float("nan"))),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Stage211 social VAD boundary totals changed.")
    boundary["_inventory_path"] = str(path)
    boundary["_inventory_sha256"] = _sha256(path)
    return boundary, source_inventory


def _boundary_source_payload(
    boundary: dict[str, Any], source_index: int
) -> tuple[Path, dict[str, Any]]:
    record = boundary["part_records"][source_index]
    path = Path(str(record["path"])).expanduser().resolve()
    payload = _load_json(path, label=f"Stage211 social VAD source {source_index}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"Stage211 social VAD source SHA-256 changed: {path}")
    return path, payload


def _source_receipt_path(output_root: Path, source_index: int) -> Path:
    return output_root / "sources" / f"source_{source_index:06d}.json"


def _source_tar_path(output_root: Path, source_index: int) -> Path:
    return output_root / "shards" / f"source_{source_index:06d}.tar"


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = int(size)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _read_source_segment(
    source: Any,
    *,
    start_ms: int,
    end_ms: int,
) -> tuple[Any, int]:
    import numpy as np
    import torch
    import torch.nn.functional as torch_functional
    import torchaudio.functional as audio_functional

    source_rate = int(source.samplerate)
    start_frame = (int(start_ms) * source_rate + 500) // 1000
    end_frame = (int(end_ms) * source_rate + 500) // 1000
    end_frame = min(end_frame, int(source.frames))
    if end_frame <= start_frame:
        raise ValueError(f"Invalid source frame interval {start_frame}:{end_frame}/{source.frames}")
    source.seek(start_frame)
    waveform = source.read(end_frame - start_frame, dtype="float32", always_2d=True)
    if waveform.shape != (end_frame - start_frame, int(source.channels)):
        raise ValueError("Source audio segment short-read.")
    mono = np.mean(waveform, axis=1, dtype=np.float32)
    tensor = torch.from_numpy(mono)
    if source_rate != TARGET_SAMPLE_RATE:
        tensor = audio_functional.resample(tensor, source_rate, TARGET_SAMPLE_RATE)
    target_samples = (int(end_ms) - int(start_ms)) * (TARGET_SAMPLE_RATE // 1000)
    delta = target_samples - int(tensor.numel())
    if abs(delta) > 32:
        raise ValueError(
            f"Resampled segment length drift is too large: target={target_samples} "
            f"actual={tensor.numel()}"
        )
    if delta > 0:
        tensor = torch_functional.pad(tensor, (0, delta))
    elif delta < 0:
        tensor = tensor[:target_samples]
    tensor = tensor.clamp(-1.0, 1.0).contiguous()
    return tensor.numpy(), target_samples


def _encode_flac(waveform: Any) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(
        buffer,
        waveform,
        TARGET_SAMPLE_RATE,
        format="FLAC",
        subtype="PCM_16",
    )
    payload = buffer.getvalue()
    with sf.SoundFile(io.BytesIO(payload)) as decoded:
        if (
            decoded.samplerate != TARGET_SAMPLE_RATE
            or decoded.channels != TARGET_CHANNELS
            or decoded.frames != len(waveform)
        ):
            raise ValueError("Encoded FLAC preflight mismatch.")
    return payload


def _metadata_bytes(
    *, source: dict[str, Any], boundary_segment: dict[str, Any], utt_id: str
) -> bytes:
    metadata = {
        "id": utt_id,
        "utt_id": utt_id,
        "key": boundary_segment["key"],
        "split": "train",
        "language": "zh",
        "source_dataset": source["source_label"],
        "source_audio_path": source["path"],
        "source_audio_sha256": source["sha256"],
        "source_start_ms": int(boundary_segment["start_ms"]),
        "source_end_ms": int(boundary_segment["end_ms"]),
        "duration_ms": int(boundary_segment["duration_ms"]),
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": TARGET_CHANNELS,
        "uses_text_labels": False,
    }
    return (json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _build_admitted_source(
    *,
    boundary: dict[str, Any],
    source_inventory: dict[str, Any],
    source_index: int,
    boundary_part_path: Path,
    boundary_payload: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    import soundfile as sf

    source = source_inventory["source_records"][source_index]
    source_path = Path(str(source["path"]))
    final_tar = _source_tar_path(output_root, source_index).resolve()
    final_tar.parent.mkdir(parents=True, exist_ok=True)
    temporary_tar = final_tar.with_name(f"{final_tar.name}.tmp.{os.getpid()}")
    rows: list[dict[str, Any]] = []
    with sf.SoundFile(str(source_path)) as audio_source:
        probe = source["probe"]
        if (
            audio_source.samplerate != int(probe["sample_rate"])
            or audio_source.channels != int(probe["channels"])
            or audio_source.frames != int(probe["frames"])
        ):
            raise ValueError(f"Source audio probe changed: {source_path}")
        with tarfile.open(temporary_tar, "w", format=tarfile.USTAR_FORMAT) as archive:
            for segment in boundary_payload["segments"]:
                waveform, target_samples = _read_source_segment(
                    audio_source,
                    start_ms=int(segment["start_ms"]),
                    end_ms=int(segment["end_ms"]),
                )
                if len(waveform) != target_samples:
                    raise ValueError("Materialized waveform length mismatch.")
                audio_bytes = _encode_flac(waveform)
                key = str(segment["key"])
                utt_id = f"social-{key}"
                metadata_bytes = _metadata_bytes(
                    source=source,
                    boundary_segment=segment,
                    utt_id=utt_id,
                )
                audio_member = f"{key}.flac"
                json_member = f"{key}.json"
                audio_info = _tar_info(audio_member, len(audio_bytes))
                audio_offset = int(archive.offset) + tarfile.BLOCKSIZE
                archive.addfile(audio_info, io.BytesIO(audio_bytes))
                json_info = _tar_info(json_member, len(metadata_bytes))
                json_offset = int(archive.offset) + tarfile.BLOCKSIZE
                archive.addfile(json_info, io.BytesIO(metadata_bytes))
                rows.append(
                    {
                        "shard_name": str(final_tar),
                        "key": key,
                        "utt_id": utt_id,
                        "split": "train",
                        "num_frames": max(1, int(round(int(segment["duration_ms"]) / 10.0))),
                        "audio_member": audio_member,
                        "audio_format": "flac",
                        "audio_offset": audio_offset,
                        "audio_size": len(audio_bytes),
                        "json_member": json_member,
                        "json_offset": json_offset,
                        "json_size": len(metadata_bytes),
                        "storage_kind": "tar",
                        "source_dataset": source["source_label"],
                        "language": "zh",
                        "sample_rate": TARGET_SAMPLE_RATE,
                        "duration_ms": int(segment["duration_ms"]),
                        "uses_text_labels": False,
                        "source_index": source_index,
                        "source_audio_sha256": source["sha256"],
                        "source_start_ms": int(segment["start_ms"]),
                        "source_end_ms": int(segment["end_ms"]),
                    }
                )
    temporary_sha256 = _sha256(temporary_tar)
    if final_tar.is_file():
        if _sha256(final_tar) != temporary_sha256:
            raise ValueError(f"Existing generated social tar differs: {final_tar}")
        temporary_tar.unlink()
    else:
        temporary_tar.replace(final_tar)
    return {
        "schema_version": 1,
        "artifact": SOURCE_ARTIFACT,
        "complete": True,
        "status": "materialized",
        "boundary_inventory_path": boundary["_inventory_path"],
        "boundary_inventory_sha256": boundary["_inventory_sha256"],
        "boundary_source_path": str(boundary_part_path),
        "boundary_source_sha256": _sha256(boundary_part_path),
        "source_index": source_index,
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "source_label": source["source_label"],
        "tar_path": str(final_tar),
        "tar_size_bytes": final_tar.stat().st_size,
        "tar_sha256": temporary_sha256,
        "segments": len(rows),
        "duration_ms": sum(int(row["duration_ms"]) for row in rows),
        "rows": rows,
    }


def _build_skipped_source(
    *,
    boundary: dict[str, Any],
    source_inventory: dict[str, Any],
    source_index: int,
    boundary_part_path: Path,
    boundary_payload: dict[str, Any],
) -> dict[str, Any]:
    source = source_inventory["source_records"][source_index]
    return {
        "schema_version": 1,
        "artifact": SOURCE_ARTIFACT,
        "complete": True,
        "status": boundary_payload["status"],
        "boundary_inventory_path": boundary["_inventory_path"],
        "boundary_inventory_sha256": boundary["_inventory_sha256"],
        "boundary_source_path": str(boundary_part_path),
        "boundary_source_sha256": _sha256(boundary_part_path),
        "source_index": source_index,
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "source_label": source["source_label"],
        "segments": 0,
        "duration_ms": 0,
        "rows": [],
    }


def validate_materialized_source(
    payload: dict[str, Any],
    *,
    boundary: dict[str, Any],
    source_inventory: dict[str, Any],
    source_index: int,
    boundary_part_path: Path,
    boundary_payload: dict[str, Any],
    verify_tar_hash: bool,
    verify_members: bool,
) -> None:
    source = source_inventory["source_records"][source_index]
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != SOURCE_ARTIFACT
        or payload.get("complete") is not True
        or payload.get("boundary_inventory_path") != boundary["_inventory_path"]
        or payload.get("boundary_inventory_sha256") != boundary["_inventory_sha256"]
        or payload.get("boundary_source_path") != str(boundary_part_path)
        or payload.get("boundary_source_sha256") != _sha256(boundary_part_path)
        or int(payload.get("source_index", -1)) != source_index
        or payload.get("source_path") != source["path"]
        or payload.get("source_sha256") != source["sha256"]
        or payload.get("source_label") != source["source_label"]
    ):
        raise ValueError(f"Stage211 materialized source binding changed: index={source_index}")
    boundary_status = boundary_payload["status"]
    if boundary_status != "admitted":
        if (
            payload.get("status") != boundary_status
            or int(payload.get("segments", -1)) != 0
            or int(payload.get("duration_ms", -1)) != 0
            or payload.get("rows") != []
        ):
            raise ValueError(f"Stage211 skipped materialization is invalid: index={source_index}")
        return
    rows = payload.get("rows")
    if (
        payload.get("status") != "materialized"
        or not isinstance(rows, list)
        or len(rows) != len(boundary_payload["segments"])
        or int(payload.get("segments", -1)) != len(rows)
    ):
        raise ValueError(f"Stage211 materialized segment rows are invalid: index={source_index}")
    tar_path = Path(str(payload.get("tar_path") or "")).expanduser().resolve()
    if (
        not tar_path.is_file()
        or int(payload.get("tar_size_bytes", -1)) != tar_path.stat().st_size
        or len(str(payload.get("tar_sha256") or "")) != 64
    ):
        raise ValueError(f"Stage211 materialized tar changed: {tar_path}")
    if verify_tar_hash and _sha256(tar_path) != payload["tar_sha256"]:
        raise ValueError(f"Stage211 materialized tar SHA-256 changed: {tar_path}")
    total_duration = 0
    with tar_path.open("rb") if verify_members else io.BytesIO() as tar_handle:
        for segment, row in zip(boundary_payload["segments"], rows):
            key = str(segment["key"])
            if (
                row.get("shard_name") != str(tar_path)
                or row.get("key") != key
                or row.get("utt_id") != f"social-{key}"
                or row.get("split") != "train"
                or row.get("audio_member") != f"{key}.flac"
                or row.get("json_member") != f"{key}.json"
                or row.get("storage_kind") != "tar"
                or row.get("source_dataset") != source["source_label"]
                or int(row.get("duration_ms", -1)) != int(segment["duration_ms"])
                or int(row.get("source_start_ms", -1)) != int(segment["start_ms"])
                or int(row.get("source_end_ms", -1)) != int(segment["end_ms"])
                or int(row.get("audio_offset", -1)) < 0
                or int(row.get("audio_size", -1)) <= 0
                or int(row.get("json_offset", -1)) < 0
                or int(row.get("json_size", -1)) <= 0
            ):
                raise ValueError(f"Stage211 materialized row binding is invalid: key={key}")
            total_duration += int(segment["duration_ms"])
            if verify_members:
                tar_handle.seek(int(row["audio_offset"]))
                audio_bytes = tar_handle.read(int(row["audio_size"]))
                tar_handle.seek(int(row["json_offset"]))
                metadata_bytes = tar_handle.read(int(row["json_size"]))
                if len(audio_bytes) != int(row["audio_size"]) or len(metadata_bytes) != int(
                    row["json_size"]
                ):
                    raise ValueError(f"Stage211 materialized member short-read: key={key}")
                import soundfile as sf

                try:
                    with sf.SoundFile(io.BytesIO(audio_bytes)) as decoded:
                        expected_samples = int(segment["duration_ms"]) * (
                            TARGET_SAMPLE_RATE // 1000
                        )
                        if (
                            decoded.samplerate != TARGET_SAMPLE_RATE
                            or decoded.channels != TARGET_CHANNELS
                            or decoded.frames != expected_samples
                        ):
                            raise ValueError(f"Stage211 materialized FLAC mismatch: key={key}")
                except Exception as exc:
                    raise ValueError(f"Stage211 materialized FLAC mismatch: key={key}") from exc
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                if (
                    metadata.get("key") != key
                    or metadata.get("utt_id") != f"social-{key}"
                    or int(metadata.get("duration_ms", -1)) != int(segment["duration_ms"])
                    or metadata.get("source_audio_sha256") != source["sha256"]
                ):
                    raise ValueError(f"Stage211 materialized metadata mismatch: key={key}")
    if int(payload.get("duration_ms", -1)) != total_duration:
        raise ValueError(f"Stage211 materialized duration total changed: index={source_index}")


def run_worker(
    *,
    boundary: dict[str, Any],
    source_inventory: dict[str, Any],
    output_root: Path,
    worker_index: int,
    num_workers: int,
    max_sources: int | None,
) -> dict[str, int]:
    if num_workers <= 0 or worker_index < 0 or worker_index >= num_workers:
        raise ValueError("Invalid materialization worker index/count.")
    output_root = output_root.expanduser().resolve()
    assigned = [
        index
        for index in range(len(source_inventory["source_records"]))
        if index % num_workers == worker_index
    ]
    if max_sources is not None:
        if max_sources <= 0:
            raise ValueError("max_sources must be positive.")
        assigned = assigned[:max_sources]
    created = 0
    reused = 0
    for position, source_index in enumerate(assigned, 1):
        boundary_part_path, boundary_payload = _boundary_source_payload(boundary, source_index)
        receipt_path = _source_receipt_path(output_root, source_index)
        if receipt_path.is_file():
            payload = _load_json(
                receipt_path,
                label=f"Stage211 materialized source {source_index}",
            )
            validate_materialized_source(
                payload,
                boundary=boundary,
                source_inventory=source_inventory,
                source_index=source_index,
                boundary_part_path=boundary_part_path,
                boundary_payload=boundary_payload,
                verify_tar_hash=False,
                verify_members=False,
            )
            reused += 1
        else:
            if boundary_payload["status"] == "admitted":
                payload = _build_admitted_source(
                    boundary=boundary,
                    source_inventory=source_inventory,
                    source_index=source_index,
                    boundary_part_path=boundary_part_path,
                    boundary_payload=boundary_payload,
                    output_root=output_root,
                )
            else:
                payload = _build_skipped_source(
                    boundary=boundary,
                    source_inventory=source_inventory,
                    source_index=source_index,
                    boundary_part_path=boundary_part_path,
                    boundary_payload=boundary_payload,
                )
            _write_immutable_json(receipt_path, payload)
            created += 1
        if position % 25 == 0 or position == len(assigned):
            print(
                f"[stage211-social-materialize] worker={worker_index}/{num_workers} "
                f"sources={position}/{len(assigned)} created={created} reused={reused}",
                flush=True,
            )
    return {"assigned": len(assigned), "created": created, "reused": reused}


def _validate_existing_final_inventory(
    path: Path,
    *,
    boundary: dict[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path, label="Stage211 social materialized inventory")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != MATERIALIZED_INVENTORY_ARTIFACT
        or payload.get("complete") is not True
        or payload.get("training_ready") is not False
        or payload.get("admission_state") != "materialized_pending_overlap_and_merge"
        or payload.get("boundary_inventory_sha256") != boundary["_inventory_sha256"]
    ):
        raise ValueError("Stage211 existing social materialized inventory changed.")
    manifest_path = Path(str(payload.get("bucket_manifest_path") or ""))
    if not manifest_path.is_file() or _sha256(manifest_path) != payload.get(
        "bucket_manifest_sha256"
    ):
        raise ValueError("Stage211 existing social bucket manifest changed.")
    return payload


def finalize_materialization(
    *,
    boundary: dict[str, Any],
    source_inventory: dict[str, Any],
    output_root: Path,
    fixed_eval_manifest_path: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    inventory_path = output_root / "materialized_inventory.json"
    existing_inventory = (
        _validate_existing_final_inventory(inventory_path, boundary=boundary)
        if inventory_path.is_file()
        else None
    )
    final_manifest_root = output_root / "webdataset_buckets_audio_text"
    if existing_inventory is None and final_manifest_root.exists():
        raise ValueError(
            "Stage211 social manifest directory exists without its inventory: "
            f"{final_manifest_root}"
        )
    source_receipts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_segment_counts: Counter[str] = Counter()
    source_duration_ms: Counter[str] = Counter()
    for source_index in range(len(source_inventory["source_records"])):
        boundary_part_path, boundary_payload = _boundary_source_payload(boundary, source_index)
        receipt_path = _source_receipt_path(output_root, source_index)
        payload = _load_json(
            receipt_path,
            label=f"Stage211 materialized source {source_index}",
        )
        validate_materialized_source(
            payload,
            boundary=boundary,
            source_inventory=source_inventory,
            source_index=source_index,
            boundary_part_path=boundary_part_path,
            boundary_payload=boundary_payload,
            verify_tar_hash=True,
            verify_members=True,
        )
        status = str(payload["status"])
        status_counts[status] += 1
        label = str(payload["source_label"])
        source_segment_counts[label] += int(payload["segments"])
        source_duration_ms[label] += int(payload["duration_ms"])
        rows.extend(payload["rows"])
        source_receipts.append(
            {
                "path": str(receipt_path),
                "size_bytes": receipt_path.stat().st_size,
                "sha256": _sha256(receipt_path),
                "source_index": source_index,
                "status": status,
                "segments": int(payload["segments"]),
                "duration_ms": int(payload["duration_ms"]),
                "tar_path": payload.get("tar_path"),
                "tar_size_bytes": payload.get("tar_size_bytes"),
                "tar_sha256": payload.get("tar_sha256"),
            }
        )
    if len(rows) != int(boundary["admitted_segments"]):
        raise ValueError(
            f"Stage211 materialized row total changed: {len(rows)}/{boundary['admitted_segments']}"
        )
    total_duration_ms = sum(source_duration_ms.values())
    if not math.isclose(
        total_duration_ms / 3_600_000.0,
        float(boundary["admitted_hours"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Stage211 materialized duration total changed.")

    selected_counts = dict(sorted(source_segment_counts.items()))
    selected_hours = {key: value / 3_600_000.0 for key, value in sorted(source_duration_ms.items())}
    if existing_inventory is not None:
        if (
            int(existing_inventory.get("selected_rows", -1)) != len(rows)
            or not math.isclose(
                float(existing_inventory.get("selected_hours", float("nan"))),
                total_duration_ms / 3_600_000.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or existing_inventory.get("selected_counts_by_source") != selected_counts
            or existing_inventory.get("selected_hours_by_source") != selected_hours
            or existing_inventory.get("source_status_counts") != dict(sorted(status_counts.items()))
            or existing_inventory.get("source_receipts") != source_receipts
        ):
            raise ValueError("Stage211 existing materialized totals changed.")
        manifest = load_webdataset_bucket_manifest(existing_inventory["bucket_manifest_path"])
        train_rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
        eval_rows = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
        if train_rows != len(rows) or eval_rows != 256:
            raise ValueError("Stage211 existing materialized manifest totals changed.")
        part_records = existing_inventory.get("part_records")
        if not isinstance(part_records, list) or not part_records:
            raise ValueError("Stage211 existing materialized part records are missing.")
        manifest_root = Path(existing_inventory["bucket_manifest_path"]).parent
        if sum(int(record.get("num_samples", -1)) for record in part_records) != len(rows):
            raise ValueError("Stage211 existing materialized part row total changed.")
        for record in part_records:
            part_path = Path(str(record.get("path") or ""))
            if not part_path.is_absolute():
                part_path = manifest_root / part_path
            if (
                not part_path.is_file()
                or int(record.get("size_bytes", -1)) != part_path.stat().st_size
                or _sha256(part_path) != record.get("sha256")
            ):
                raise ValueError(f"Stage211 existing materialized part changed: {part_path}")
        return existing_inventory

    staging_root = output_root / f"webdataset_buckets_audio_text.tmp.{os.getpid()}"
    if staging_root.exists():
        raise ValueError(f"Stage211 social staging manifest already exists: {staging_root}")
    writer = BucketManifestWriter(
        staging_root,
        bucket_width=BUCKET_WIDTH,
        entries_per_part=ENTRIES_PER_PART,
    )
    for row in rows:
        writer.write(row)
    manifest, part_records = writer.finalize(source_length_index_path=inventory_path)
    fixed_eval_split, fixed_eval_binding = _load_fixed_eval_split(fixed_eval_manifest_path)
    manifest["splits"]["eval"] = fixed_eval_split
    staging_manifest_path = staging_root / "manifest.json"
    staging_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    staging_root.replace(final_manifest_root)
    final_manifest_path = final_manifest_root / "manifest.json"
    payload = {
        "schema_version": 1,
        "artifact": MATERIALIZED_INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "materialized_pending_overlap_and_merge",
        "boundary_inventory_path": boundary["_inventory_path"],
        "boundary_inventory_sha256": boundary["_inventory_sha256"],
        "source_inventory_path": source_inventory["_inventory_path"],
        "source_inventory_sha256": source_inventory["_inventory_sha256"],
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
        "selected_rows": len(rows),
        "selected_hours": total_duration_ms / 3_600_000.0,
        "selected_counts_by_source": selected_counts,
        "selected_hours_by_source": selected_hours,
        "source_status_counts": dict(sorted(status_counts.items())),
        "fixed_eval": fixed_eval_binding,
        "bucket_manifest_path": str(final_manifest_path),
        "bucket_manifest_sha256": _sha256(final_manifest_path),
        "part_records": part_records,
        "source_receipts": source_receipts,
        "required_before_training": [
            "segment-level exact-content and public-overlap validation",
            "atomic base-supplemental plus social inventory migration",
            "independent combined profile receipt",
        ],
    }
    _write_immutable_json(inventory_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize Stage211 social VAD boundaries as indexed FLAC tar shards."
    )
    parser.add_argument("--boundary-inventory", type=Path, default=DEFAULT_BOUNDARY_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-eval-manifest", type=Path, default=DEFAULT_FIXED_EVAL_MANIFEST)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    boundary, source_inventory = validate_boundary_inventory(
        args.boundary_inventory,
        verify_source_part_hashes=True,
    )
    if args.finalize_only:
        result = finalize_materialization(
            boundary=boundary,
            source_inventory=source_inventory,
            output_root=args.output_root,
            fixed_eval_manifest_path=args.fixed_eval_manifest,
        )
        print(
            f"materialized_inventory={args.output_root.expanduser().resolve() / 'materialized_inventory.json'} "
            f"rows={result['selected_rows']} hours={result['selected_hours']:.6f}",
            flush=True,
        )
        return 0
    run_worker(
        boundary=boundary,
        source_inventory=source_inventory,
        output_root=args.output_root,
        worker_index=int(args.worker_index),
        num_workers=int(args.num_workers),
        max_sources=args.max_sources,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
