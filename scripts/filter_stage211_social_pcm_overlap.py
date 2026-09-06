#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from rwkvasr.data import load_webdataset_bucket_manifest

try:
    from scripts.build_stage211_supplemental_natural_manifest import BucketManifestWriter
    from scripts.materialize_stage211_social_vad_segments import (
        MATERIALIZED_INVENTORY_ARTIFACT,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from build_stage211_supplemental_natural_manifest import BucketManifestWriter
    from materialize_stage211_social_vad_segments import MATERIALIZED_INVENTORY_ARTIFACT


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATERIALIZED_INVENTORY = (
    Path.home()
    / "rwkvasr_data/stage211_social_vad_materialized_v1/materialized_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_social_vad_filtered_v1"
DEFAULT_PUBLIC_MANIFESTS = {
    name: REPO_ROOT / f"artifacts/eval_benchmarks/manifests/{name}.jsonl"
    for name in (
        "aishell1_test",
        "commonvoice_en_test",
        "librispeech_test_clean",
        "librispeech_test_other",
        "wenetspeech_test_net",
    )
}
EXPECTED_PUBLIC_ROWS = {
    "aishell1_test": 7_176,
    "commonvoice_en_test": 14_927,
    "librispeech_test_clean": 2_620,
    "librispeech_test_other": 2_939,
    "wenetspeech_test_net": 24_774,
}

SCHEMA_VERSION = 1
SOURCE_FINGERPRINT_ARTIFACT = "stage211_social_pcm_source_fingerprints"
PUBLIC_FINGERPRINT_ARTIFACT = "stage211_public_pcm_fingerprints"
FILTERED_INVENTORY_ARTIFACT = "stage211_social_pcm_filtered_inventory"
FINGERPRINT_ALGORITHM = "sha256(le_u32_sr + le_u32_channels + le_u64_samples + pcm_s16le)"
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
BUCKET_WIDTH = 80
ENTRIES_PER_PART = 100_000


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_line(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _immutable_bytes(path: Path, payload: bytes) -> str:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace different Stage211 evidence: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def _immutable_json(path: Path, payload: dict[str, Any]) -> str:
    rendered = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return _immutable_bytes(path, rendered)


def _immutable_hardlink(source: Path, destination: Path) -> str:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Stage211 hardlink source is unavailable: {source}")
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise ValueError(f"Stage211 hardlink destination differs: {destination}")
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        os.link(source, temporary)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if not os.path.samefile(source, destination):
        raise ValueError(f"Stage211 hardlink installation failed: {destination}")
    return "created"


def _read_region(path: Path, *, offset: int, size: int) -> bytes:
    if offset < 0 or size <= 0:
        raise ValueError(f"Invalid indexed audio region offset={offset} size={size}: {path}")
    with path.open("rb") as source:
        source.seek(offset)
        payload = source.read(size)
    if len(payload) != size:
        raise ValueError(f"Indexed audio region short-read: {path} offset={offset} size={size}")
    return payload


def _pcm16_from_audio(source: str | Path | BinaryIO) -> Any:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as audio_functional

    waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if waveform.shape[0] <= 0 or sample_rate <= 0:
        raise ValueError("Audio contains no decodable samples.")
    mono = np.asarray(waveform.mean(axis=1, dtype=np.float32), dtype=np.float32)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        tensor = torch.from_numpy(mono).unsqueeze(0)
        mono = (
            audio_functional.resample(
                tensor,
                orig_freq=int(sample_rate),
                new_freq=TARGET_SAMPLE_RATE,
            )
            .squeeze(0)
            .contiguous()
            .numpy()
        )
    encoded = io.BytesIO()
    sf.write(
        encoded,
        mono,
        TARGET_SAMPLE_RATE,
        format="FLAC",
        subtype="PCM_16",
    )
    encoded.seek(0)
    pcm, canonical_rate = sf.read(encoded, dtype="int16", always_2d=True)
    if (
        int(canonical_rate) != TARGET_SAMPLE_RATE
        or pcm.shape[1] != TARGET_CHANNELS
        or pcm.shape[0] <= 0
    ):
        raise ValueError("Canonical PCM conversion failed.")
    return np.asarray(pcm[:, 0], dtype="<i2")


def canonical_pcm_fingerprint(source: str | Path | BinaryIO) -> tuple[str, int]:
    pcm = _pcm16_from_audio(source)
    sample_count = int(pcm.shape[0])
    digest = hashlib.sha256()
    digest.update(
        struct.pack(
            "<IIQ",
            TARGET_SAMPLE_RATE,
            TARGET_CHANNELS,
            sample_count,
        )
    )
    digest.update(pcm.tobytes(order="C"))
    return digest.hexdigest(), sample_count


def validate_materialized_inventory(path: str | Path) -> dict[str, Any]:
    inventory_path = Path(path).expanduser().resolve()
    inventory = _load_json(inventory_path, label="Stage211 social materialized inventory")
    expected = {
        "schema_version": 1,
        "artifact": MATERIALIZED_INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "materialized_pending_overlap_and_merge",
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
    }
    if any(inventory.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 social materialized inventory contract mismatch.")
    source_receipts = inventory.get("source_receipts")
    if not isinstance(source_receipts, list) or not source_receipts:
        raise ValueError("Stage211 social materialized source receipts are missing.")
    selected_rows = int(inventory.get("selected_rows", -1))
    if selected_rows < 0 or sum(int(row.get("segments", -1)) for row in source_receipts) != selected_rows:
        raise ValueError("Stage211 social materialized row count changed.")
    for source_index, record in enumerate(source_receipts):
        if not isinstance(record, dict) or int(record.get("source_index", -1)) != source_index:
            raise ValueError("Stage211 social source receipt order changed.")
        receipt_path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            not receipt_path.is_file()
            or receipt_path.stat().st_size != int(record.get("size_bytes", -1))
            or _sha256(receipt_path) != record.get("sha256")
        ):
            raise ValueError(f"Stage211 social source receipt changed: {receipt_path}")
    manifest_path = Path(str(inventory.get("bucket_manifest_path") or "")).resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != inventory.get(
        "bucket_manifest_sha256"
    ):
        raise ValueError("Stage211 social materialized bucket manifest changed.")
    manifest = load_webdataset_bucket_manifest(manifest_path)
    train_rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    eval_rows = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
    if train_rows != selected_rows or eval_rows != 256:
        raise ValueError("Stage211 social materialized manifest totals changed.")
    return {
        "inventory": inventory,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def _source_fingerprint_paths(output_root: Path, source_index: int) -> tuple[Path, Path]:
    base = output_root / "source_fingerprints"
    return (
        base / f"source_{source_index:06d}.jsonl",
        base / f"source_{source_index:06d}.receipt.json",
    )


def _validate_source_fingerprint_receipt(
    receipt_path: Path,
    *,
    source_record: dict[str, Any],
    source_index: int,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, label="Stage211 social source fingerprint receipt")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": SOURCE_FINGERPRINT_ARTIFACT,
        "complete": True,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "source_index": source_index,
        "source_receipt_path": str(Path(str(source_record["path"])).resolve()),
        "source_receipt_sha256": source_record["sha256"],
        "rows": int(source_record["segments"]),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 social source fingerprint receipt changed: {receipt_path}")
    part_path = Path(str(receipt.get("part_path") or "")).resolve()
    if (
        not part_path.is_file()
        or part_path.stat().st_size != int(receipt.get("part_size_bytes", -1))
        or _sha256(part_path) != receipt.get("part_sha256")
    ):
        raise ValueError(f"Stage211 social source fingerprint part changed: {part_path}")
    with part_path.open("rb") as source:
        rows = sum(chunk.count(b"\n") for chunk in iter(lambda: source.read(1 << 20), b""))
    if rows != int(source_record["segments"]):
        raise ValueError(f"Stage211 social source fingerprint row count changed: {part_path}")
    return receipt


def fingerprint_social_source(
    *,
    materialized: dict[str, Any],
    output_root: Path,
    source_index: int,
) -> str:
    inventory = materialized["inventory"]
    source_record = inventory["source_receipts"][source_index]
    part_path, receipt_path = _source_fingerprint_paths(output_root, source_index)
    if receipt_path.is_file():
        _validate_source_fingerprint_receipt(
            receipt_path,
            source_record=source_record,
            source_index=source_index,
        )
        return "reused"
    if part_path.exists():
        raise ValueError(f"Fingerprint part exists without its receipt: {part_path}")

    source_receipt_path = Path(str(source_record["path"])).resolve()
    source_receipt = _load_json(
        source_receipt_path,
        label=f"Stage211 materialized source {source_index}",
    )
    if (
        int(source_receipt.get("source_index", -1)) != source_index
        or int(source_receipt.get("segments", -1)) != int(source_record["segments"])
    ):
        raise ValueError(f"Stage211 materialized source identity changed: {source_receipt_path}")
    rows = source_receipt.get("rows")
    if not isinstance(rows, list) or len(rows) != int(source_record["segments"]):
        raise ValueError(f"Stage211 materialized source rows changed: {source_receipt_path}")

    tar_path: Path | None = None
    if rows:
        tar_path = Path(str(source_receipt.get("tar_path") or "")).resolve()
        if (
            not tar_path.is_file()
            or tar_path.stat().st_size != int(source_receipt.get("tar_size_bytes", -1))
            or _sha256(tar_path) != source_receipt.get("tar_sha256")
            or source_receipt.get("tar_sha256") != source_record.get("tar_sha256")
        ):
            raise ValueError(f"Stage211 materialized source tar changed: {tar_path}")

    rendered = bytearray()
    for row_index, row in enumerate(rows):
        audio_bytes = _read_region(
            tar_path,
            offset=int(row.get("audio_offset", -1)),
            size=int(row.get("audio_size", -1)),
        )
        fingerprint, samples = canonical_pcm_fingerprint(io.BytesIO(audio_bytes))
        expected_samples = int(row.get("duration_ms", -1)) * 16
        if samples != expected_samples:
            raise ValueError(
                f"Canonical social sample count changed source={source_index} row={row_index}: "
                f"{samples}/{expected_samples}"
            )
        rendered.extend(
            _canonical_line(
                {
                    "audio_member_sha256": _sha256_bytes(audio_bytes),
                    "canonical_pcm_sha256": fingerprint,
                    "duration_ms": int(row["duration_ms"]),
                    "key": str(row["key"]),
                    "pcm_samples": samples,
                    "row_index": row_index,
                    "source_dataset": str(row["source_dataset"]),
                    "source_index": source_index,
                    "utt_id": str(row["utt_id"]),
                }
            )
        )
    _immutable_bytes(part_path, bytes(rendered))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact": SOURCE_FINGERPRINT_ARTIFACT,
        "complete": True,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "source_index": source_index,
        "source_path": str(source_receipt.get("source_path") or ""),
        "source_sha256": str(source_receipt.get("source_sha256") or ""),
        "source_label": str(source_receipt.get("source_label") or ""),
        "source_receipt_path": str(source_receipt_path),
        "source_receipt_sha256": _sha256(source_receipt_path),
        "tar_path": str(tar_path) if tar_path is not None else None,
        "tar_sha256": source_receipt.get("tar_sha256"),
        "rows": len(rows),
        "duration_ms": sum(int(row["duration_ms"]) for row in rows),
        "part_path": str(part_path.resolve()),
        "part_size_bytes": part_path.stat().st_size,
        "part_sha256": _sha256(part_path),
    }
    _immutable_json(receipt_path, receipt)
    return "created"


def run_social_fingerprint_worker(
    *,
    materialized: dict[str, Any],
    output_root: Path,
    worker_index: int,
    num_workers: int,
    max_sources: int | None,
) -> dict[str, int]:
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("worker-index must satisfy 0 <= worker-index < num-workers")
    source_count = len(materialized["inventory"]["source_receipts"])
    assigned = [index for index in range(source_count) if index % num_workers == worker_index]
    if max_sources is not None:
        assigned = assigned[: int(max_sources)]
    counts: Counter[str] = Counter()
    for source_index in assigned:
        status = fingerprint_social_source(
            materialized=materialized,
            output_root=output_root,
            source_index=source_index,
        )
        counts[status] += 1
        if sum(counts.values()) % 100 == 0:
            print(
                f"[stage211-social-pcm] worker={worker_index}/{num_workers} "
                f"processed={sum(counts.values())}/{len(assigned)}",
                flush=True,
            )
    return {"assigned": len(assigned), **dict(sorted(counts.items()))}


def rebase_social_source_fingerprints(
    *,
    materialized: dict[str, Any],
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("Stage211 social PCM rebase source and destination must differ.")
    source_records = materialized["inventory"]["source_receipts"]
    statuses: Counter[str] = Counter()
    source_receipt_digest = hashlib.sha256()
    destination_receipt_digest = hashlib.sha256()
    for source_index, source_record in enumerate(source_records):
        source_part, source_receipt_path = _source_fingerprint_paths(source_root, source_index)
        source_receipt = _validate_source_fingerprint_receipt(
            source_receipt_path,
            source_record=source_record,
            source_index=source_index,
        )
        destination_part, destination_receipt_path = _source_fingerprint_paths(
            output_root,
            source_index,
        )
        statuses[_immutable_hardlink(source_part, destination_part)] += 1
        destination_receipt = {
            **source_receipt,
            "part_path": str(destination_part.resolve()),
        }
        _immutable_json(destination_receipt_path, destination_receipt)
        _validate_source_fingerprint_receipt(
            destination_receipt_path,
            source_record=source_record,
            source_index=source_index,
        )
        source_receipt_digest.update(bytes.fromhex(_sha256(source_receipt_path)))
        destination_receipt_digest.update(bytes.fromhex(_sha256(destination_receipt_path)))
    return {
        "source_count": len(source_records),
        "source_hardlink_status": dict(sorted(statuses.items())),
        "source_receipt_set_sha256": source_receipt_digest.hexdigest(),
        "destination_receipt_set_sha256": destination_receipt_digest.hexdigest(),
    }


def _public_fingerprint_paths(output_root: Path, dataset: str) -> tuple[Path, Path]:
    base = output_root / "public_fingerprints"
    return base / f"{dataset}.jsonl", base / f"{dataset}.receipt.json"


def _validate_public_fingerprint_receipt(
    receipt_path: Path,
    *,
    dataset: str,
    manifest_path: Path,
    expected_rows: int | None,
) -> dict[str, Any]:
    receipt = _load_json(receipt_path, label="Stage211 public PCM fingerprint receipt")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": PUBLIC_FINGERPRINT_ARTIFACT,
        "complete": True,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "dataset": dataset,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stage211 public PCM fingerprint receipt changed: {receipt_path}")
    if expected_rows is not None and int(receipt.get("rows", -1)) != expected_rows:
        raise ValueError(f"Stage211 public PCM row count changed: {dataset}")
    part_path = Path(str(receipt.get("part_path") or "")).resolve()
    if (
        not part_path.is_file()
        or part_path.stat().st_size != int(receipt.get("part_size_bytes", -1))
        or _sha256(part_path) != receipt.get("part_sha256")
    ):
        raise ValueError(f"Stage211 public PCM fingerprint part changed: {part_path}")
    return receipt


def fingerprint_public_manifest(
    *,
    dataset: str,
    manifest_path: Path,
    output_root: Path,
    expected_rows: int | None,
) -> str:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Public manifest is unavailable: {manifest_path}")
    part_path, receipt_path = _public_fingerprint_paths(output_root, dataset)
    if receipt_path.is_file():
        _validate_public_fingerprint_receipt(
            receipt_path,
            dataset=dataset,
            manifest_path=manifest_path,
            expected_rows=expected_rows,
        )
        return "reused"
    if part_path.exists():
        raise ValueError(f"Public fingerprint part exists without its receipt: {part_path}")

    rendered = bytearray()
    seen_ids: set[str] = set()
    row_count = 0
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Public manifest row is not an object: {manifest_path}:{line_number}")
            utt_id = str(row.get("utt_id") or "").strip()
            if not utt_id or utt_id in seen_ids:
                raise ValueError(f"Public manifest utt_id is invalid/duplicate: {dataset}:{utt_id}")
            seen_ids.add(utt_id)
            audio_path = Path(str(row.get("audio_filepath") or "")).expanduser().resolve()
            if not audio_path.is_file() or audio_path.stat().st_size <= 0:
                raise ValueError(f"Public audio is unavailable: {audio_path}")
            encoded_sha256 = _sha256(audio_path)
            fingerprint, samples = canonical_pcm_fingerprint(str(audio_path))
            rendered.extend(
                _canonical_line(
                    {
                        "audio_path": str(audio_path),
                        "audio_sha256": encoded_sha256,
                        "audio_size_bytes": audio_path.stat().st_size,
                        "canonical_pcm_sha256": fingerprint,
                        "dataset": dataset,
                        "manifest_line_number": line_number,
                        "pcm_samples": samples,
                        "utt_id": utt_id,
                    }
                )
            )
            row_count += 1
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(
            f"Public manifest row count mismatch {dataset}: {row_count}/{expected_rows}"
        )
    _immutable_bytes(part_path, bytes(rendered))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact": PUBLIC_FINGERPRINT_ARTIFACT,
        "complete": True,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "dataset": dataset,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "rows": row_count,
        "part_path": str(part_path.resolve()),
        "part_size_bytes": part_path.stat().st_size,
        "part_sha256": _sha256(part_path),
    }
    _immutable_json(receipt_path, receipt)
    return "created"


def build_public_fingerprints(
    *,
    public_manifests: dict[str, Path],
    output_root: Path,
    expected_rows: dict[str, int] | None,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for dataset, manifest_path in sorted(public_manifests.items()):
        statuses[dataset] = fingerprint_public_manifest(
            dataset=dataset,
            manifest_path=manifest_path,
            output_root=output_root,
            expected_rows=None if expected_rows is None else expected_rows.get(dataset),
        )
        print(
            f"[stage211-social-pcm] public={dataset} status={statuses[dataset]}",
            flush=True,
        )
    return statuses


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            yield row


def _winner_priority(source_receipt: dict[str, Any], row: dict[str, Any]) -> str:
    label = str(source_receipt.get("source_label") or "")
    rank = 0 if label == "social_clean_vocals" else 1 if label == "social_videos_extracted_wav" else 2
    return json.dumps(
        [
            rank,
            str(source_receipt.get("source_path") or ""),
            int(source_receipt["source_index"]),
            str(row["key"]),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _tree_records(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _install_identical_tree(staging: Path, final: Path) -> list[dict[str, Any]]:
    staging_records = _tree_records(staging)
    if final.exists():
        if not final.is_dir() or _tree_records(final) != staging_records:
            raise ValueError(f"Existing Stage211 filtered output differs: {final}")
        shutil.rmtree(staging)
        return staging_records
    staging.replace(final)
    return staging_records


def _write_jsonl_handle(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(_canonical_line(payload))


def _source_receipt_payload(record: dict[str, Any]) -> dict[str, Any]:
    return _load_json(
        Path(str(record["path"])).resolve(),
        label=f"Stage211 materialized source {record['source_index']}",
    )


def _materialized_row(
    source_receipts: list[dict[str, Any]],
    cache: dict[int, dict[str, Any]],
    *,
    source_index: int,
    row_index: int,
) -> dict[str, Any]:
    if source_index not in cache:
        cache.clear()
        cache[source_index] = _source_receipt_payload(source_receipts[source_index])
    rows = cache[source_index].get("rows")
    if not isinstance(rows, list) or not 0 <= row_index < len(rows):
        raise ValueError(f"Materialized row index changed: {source_index}/{row_index}")
    return rows[row_index]


def _load_public_fingerprint_map(
    *,
    public_manifests: dict[str, Path],
    output_root: Path,
    expected_rows: dict[str, int] | None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    by_fingerprint: dict[str, list[dict[str, str]]] = defaultdict(list)
    receipts: list[dict[str, Any]] = []
    for dataset, manifest_path in sorted(public_manifests.items()):
        _, receipt_path = _public_fingerprint_paths(output_root, dataset)
        receipt = _validate_public_fingerprint_receipt(
            receipt_path,
            dataset=dataset,
            manifest_path=manifest_path.resolve(),
            expected_rows=None if expected_rows is None else expected_rows.get(dataset),
        )
        for row in _iter_jsonl(Path(str(receipt["part_path"]))):
            by_fingerprint[str(row["canonical_pcm_sha256"])].append(
                {"dataset": dataset, "utt_id": str(row["utt_id"])}
            )
        receipts.append(
            {
                "path": str(receipt_path.resolve()),
                "sha256": _sha256(receipt_path),
                **receipt,
            }
        )
    for matches in by_fingerprint.values():
        matches.sort(key=lambda row: (row["dataset"], row["utt_id"]))
    return dict(by_fingerprint), receipts


def validate_filtered_inventory(
    path: str | Path,
    *,
    verify_part_sha256: bool = True,
) -> dict[str, Any]:
    inventory_path = Path(path).expanduser().resolve()
    inventory = _load_json(inventory_path, label="Stage211 filtered social inventory")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": FILTERED_INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": True,
        "admission_state": "normalized_pcm_exact_filtered",
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
    }
    if any(inventory.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 filtered social inventory contract mismatch.")
    materialized_path = Path(str(inventory.get("materialized_inventory_path") or "")).resolve()
    if (
        not materialized_path.is_file()
        or _sha256(materialized_path) != inventory.get("materialized_inventory_sha256")
    ):
        raise ValueError("Stage211 filtered social materialized binding changed.")
    for collection in ("source_fingerprint_receipts", "public_fingerprint_receipts"):
        records = inventory.get(collection)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Stage211 filtered social {collection} is missing.")
        for record in records:
            receipt_path = Path(str(record.get("path") or "")).resolve()
            if not receipt_path.is_file() or _sha256(receipt_path) != record.get("sha256"):
                raise ValueError(f"Stage211 filtered social receipt changed: {receipt_path}")
            fingerprint_part = Path(str(record.get("part_path") or "")).resolve()
            if (
                not fingerprint_part.is_file()
                or fingerprint_part.stat().st_size
                != int(record.get("part_size_bytes", -1))
                or _sha256(fingerprint_part) != record.get("part_sha256")
            ):
                raise ValueError(
                    f"Stage211 filtered social fingerprint receipt changed: {fingerprint_part}"
                )
    manifest_path = Path(str(inventory.get("bucket_manifest_path") or "")).resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != inventory.get(
        "bucket_manifest_sha256"
    ):
        raise ValueError("Stage211 filtered social bucket manifest changed.")
    manifest = load_webdataset_bucket_manifest(manifest_path)
    train_rows = sum(bucket.num_samples for bucket in manifest.splits.get("train", ()))
    eval_rows = sum(bucket.num_samples for bucket in manifest.splits.get("eval", ()))
    selected_rows = int(inventory.get("selected_rows", -1))
    if train_rows != selected_rows or eval_rows != 256:
        raise ValueError("Stage211 filtered social manifest totals changed.")
    part_records = inventory.get("part_records")
    if not isinstance(part_records, list) or (selected_rows > 0 and not part_records):
        raise ValueError("Stage211 filtered social part records are missing.")
    recorded_rows = 0
    for record in part_records:
        part_path = Path(str(record.get("path") or ""))
        if not part_path.is_absolute():
            part_path = manifest_path.parent / part_path
        if (
            not part_path.is_file()
            or part_path.stat().st_size != int(record.get("size_bytes", -1))
            or (verify_part_sha256 and _sha256(part_path) != record.get("sha256"))
        ):
            raise ValueError(f"Stage211 filtered social train part changed: {part_path}")
        recorded_rows += int(record.get("num_samples", -1))
    if recorded_rows != selected_rows:
        raise ValueError("Stage211 filtered social train part row total changed.")
    for name in ("duplicate_exclusions", "public_overlap_exclusions"):
        record = inventory.get("outputs", {}).get(name)
        output_path = Path(str((record or {}).get("path") or "")).resolve()
        if not output_path.is_file() or _sha256(output_path) != (record or {}).get("sha256"):
            raise ValueError(f"Stage211 filtered social exclusion evidence changed: {output_path}")
    prefilter_rows = int(inventory.get("prefilter_rows", -1))
    duplicate_rows = int(inventory.get("dedupe", {}).get("rejected_rows", -1))
    public_rows = int(inventory.get("public_overlap", {}).get("rejected_rows", -1))
    if selected_rows + duplicate_rows + public_rows != prefilter_rows:
        raise ValueError("Stage211 filtered social row partition changed.")
    return {
        "inventory": inventory,
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "bucket_manifest_path": str(manifest_path),
        "bucket_manifest_sha256": _sha256(manifest_path),
        "rows": selected_rows,
        "hours": float(inventory.get("selected_hours", 0.0)),
    }


def finalize_filtered_inventory(
    *,
    materialized: dict[str, Any],
    public_manifests: dict[str, Path],
    output_root: Path,
    expected_public_rows: dict[str, int] | None,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = output_root / "filtered_inventory.json"
    if inventory_path.is_file():
        return validate_filtered_inventory(inventory_path, verify_part_sha256=True)["inventory"]

    inventory = materialized["inventory"]
    source_records = inventory["source_receipts"]
    source_fingerprint_receipts: list[dict[str, Any]] = []
    for source_index, source_record in enumerate(source_records):
        _, receipt_path = _source_fingerprint_paths(output_root, source_index)
        receipt = _validate_source_fingerprint_receipt(
            receipt_path,
            source_record=source_record,
            source_index=source_index,
        )
        source_fingerprint_receipts.append(
            {
                "path": str(receipt_path.resolve()),
                "sha256": _sha256(receipt_path),
                **receipt,
            }
        )
    public_by_fingerprint, public_fingerprint_receipts = _load_public_fingerprint_map(
        public_manifests=public_manifests,
        output_root=output_root,
        expected_rows=expected_public_rows,
    )

    database_path = output_root / f"dedupe.sqlite.tmp.{os.getpid()}"
    database_path.unlink(missing_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(
            """
            CREATE TABLE segments (
                fingerprint TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                priority TEXT NOT NULL,
                utt_id TEXT NOT NULL,
                key TEXT NOT NULL,
                PRIMARY KEY (source_index, row_index)
            );
            CREATE INDEX segments_fingerprint ON segments(fingerprint);
            CREATE TABLE winners (
                fingerprint TEXT PRIMARY KEY,
                priority TEXT NOT NULL,
                source_index INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                utt_id TEXT NOT NULL,
                key TEXT NOT NULL
            );
            """
        )
        for receipt in source_fingerprint_receipts:
            for row in _iter_jsonl(Path(str(receipt["part_path"]))):
                priority = _winner_priority(receipt, row)
                values = (
                    str(row["canonical_pcm_sha256"]),
                    int(row["source_index"]),
                    int(row["row_index"]),
                    priority,
                    str(row["utt_id"]),
                    str(row["key"]),
                )
                connection.execute("INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?)", values)
                connection.execute(
                    """
                    INSERT INTO winners VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(fingerprint) DO UPDATE SET
                        priority=excluded.priority,
                        source_index=excluded.source_index,
                        row_index=excluded.row_index,
                        utt_id=excluded.utt_id,
                        key=excluded.key
                    WHERE excluded.priority < winners.priority
                    """,
                    (values[0], values[3], values[1], values[2], values[4], values[5]),
                )
        connection.commit()
        segment_count = int(connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
        if segment_count != int(inventory["selected_rows"]):
            raise ValueError(
                f"Stage211 social fingerprint coverage changed: {segment_count}/"
                f"{inventory['selected_rows']}"
            )
        duplicate_groups = int(
            connection.execute(
                "SELECT COUNT(*) FROM (SELECT fingerprint FROM segments GROUP BY fingerprint "
                "HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        )

        staging_root = output_root / f"webdataset_buckets_audio_text.tmp.{os.getpid()}"
        if staging_root.exists():
            raise ValueError(f"Stage211 filtered staging path already exists: {staging_root}")
        writer = BucketManifestWriter(
            staging_root,
            bucket_width=BUCKET_WIDTH,
            entries_per_part=ENTRIES_PER_PART,
        )
        duplicate_path = staging_root / "duplicate_exclusions.jsonl"
        public_path = staging_root / "public_overlap_exclusions.jsonl"
        duplicate_path.parent.mkdir(parents=True, exist_ok=True)
        selected_counts: Counter[str] = Counter()
        selected_duration_ms: Counter[str] = Counter()
        duplicate_rejected = 0
        public_rejected = 0
        public_match_counts: Counter[str] = Counter()
        source_cache: dict[int, dict[str, Any]] = {}
        query = connection.execute(
            """
            SELECT s.fingerprint, s.source_index, s.row_index, s.utt_id, s.key,
                   w.source_index, w.row_index, w.utt_id, w.key
            FROM segments AS s
            JOIN winners AS w ON w.fingerprint = s.fingerprint
            ORDER BY s.source_index, s.row_index
            """
        )
        with duplicate_path.open("wb") as duplicate_output, public_path.open(
            "wb"
        ) as public_output:
            for (
                fingerprint,
                source_index,
                row_index,
                utt_id,
                key,
                winner_source_index,
                winner_row_index,
                winner_utt_id,
                winner_key,
            ) in query:
                public_matches = public_by_fingerprint.get(str(fingerprint), [])
                if public_matches:
                    _write_jsonl_handle(
                        public_output,
                        {
                            "canonical_pcm_sha256": fingerprint,
                            "key": key,
                            "public_matches": public_matches,
                            "source_index": source_index,
                            "source_row_index": row_index,
                            "utt_id": utt_id,
                        },
                    )
                    public_rejected += 1
                    for match in public_matches:
                        public_match_counts[match["dataset"]] += 1
                    continue
                if (int(source_index), int(row_index)) != (
                    int(winner_source_index),
                    int(winner_row_index),
                ):
                    _write_jsonl_handle(
                        duplicate_output,
                        {
                            "canonical_pcm_sha256": fingerprint,
                            "key": key,
                            "source_index": source_index,
                            "source_row_index": row_index,
                            "utt_id": utt_id,
                            "winner": {
                                "key": winner_key,
                                "source_index": winner_source_index,
                                "source_row_index": winner_row_index,
                                "utt_id": winner_utt_id,
                            },
                        },
                    )
                    duplicate_rejected += 1
                    continue
                row = dict(
                    _materialized_row(
                        source_records,
                        source_cache,
                        source_index=int(source_index),
                        row_index=int(row_index),
                    )
                )
                row["stage211_social_pcm_sha256"] = str(fingerprint)
                writer.write(row)
                label = str(row["source_dataset"])
                selected_counts[label] += 1
                selected_duration_ms[label] += int(row["duration_ms"])

        manifest, part_records = writer.finalize(source_length_index_path=inventory_path)
        source_manifest = _load_json(
            Path(materialized["manifest_path"]),
            label="Stage211 social materialized bucket manifest",
        )
        eval_split = source_manifest.get("splits", {}).get("eval")
        if not isinstance(eval_split, dict) or int(eval_split.get("num_samples", -1)) != 256:
            raise ValueError("Stage211 social fixed-eval split changed.")
        manifest["splits"]["eval"] = eval_split
        staging_manifest_path = staging_root / "manifest.json"
        staging_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        final_root = output_root / "webdataset_buckets_audio_text"
        tree_records = _install_identical_tree(staging_root, final_root)
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    final_manifest_path = output_root / "webdataset_buckets_audio_text/manifest.json"
    final_duplicate_path = output_root / "webdataset_buckets_audio_text/duplicate_exclusions.jsonl"
    final_public_path = output_root / "webdataset_buckets_audio_text/public_overlap_exclusions.jsonl"
    selected_rows = sum(selected_counts.values())
    selected_duration_total = sum(selected_duration_ms.values())
    if selected_rows + duplicate_rejected + public_rejected != segment_count:
        raise ValueError("Stage211 social filtered row partition is incomplete.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": FILTERED_INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": True,
        "admission_state": "normalized_pcm_exact_filtered",
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "near_duplicate_detection_complete": False,
        "near_duplicate_scope_excluded": [
            "lossy_reencoding",
            "partial_embedding",
            "time_shift",
            "changed_vad_boundaries",
        ],
        "storage_kinds": ["tar"],
        "language": "zh",
        "uses_text_labels": False,
        "materialized_inventory_path": materialized["inventory_path"],
        "materialized_inventory_sha256": materialized["inventory_sha256"],
        "prefilter_rows": segment_count,
        "prefilter_hours": float(inventory["selected_hours"]),
        "selected_rows": selected_rows,
        "selected_hours": selected_duration_total / 3_600_000.0,
        "selected_counts_by_source": dict(sorted(selected_counts.items())),
        "selected_hours_by_source": {
            key: value / 3_600_000.0 for key, value in sorted(selected_duration_ms.items())
        },
        "dedupe": {
            "mode": "normalized_pcm_exact",
            "winner_priority": [
                "social_clean_vocals",
                "social_videos_extracted_wav",
                "source_path_source_index_segment_key",
            ],
            "fingerprinted_rows": segment_count,
            "exact_duplicate_groups": duplicate_groups,
            "rejected_rows": duplicate_rejected,
        },
        "public_overlap": {
            "mode": "normalized_pcm_exact",
            "datasets": sorted(public_manifests),
            "public_rows": sum(int(row["rows"]) for row in public_fingerprint_receipts),
            "rejected_rows": public_rejected,
            "rejected_rows_by_dataset": dict(sorted(public_match_counts.items())),
            "near_duplicate_complete": False,
        },
        "fixed_eval": inventory["fixed_eval"],
        "source_fingerprint_receipts": source_fingerprint_receipts,
        "public_fingerprint_receipts": public_fingerprint_receipts,
        "bucket_manifest_path": str(final_manifest_path),
        "bucket_manifest_sha256": _sha256(final_manifest_path),
        "part_records": part_records,
        "output_tree_records": tree_records,
        "outputs": {
            "duplicate_exclusions": {
                "path": str(final_duplicate_path),
                "rows": duplicate_rejected,
                "sha256": _sha256(final_duplicate_path),
            },
            "public_overlap_exclusions": {
                "path": str(final_public_path),
                "rows": public_rejected,
                "sha256": _sha256(final_public_path),
            },
        },
        "required_before_training": [
            "atomic base-supplemental plus filtered-social inventory migration",
            "independent combined profile receipt",
        ],
    }
    _immutable_json(inventory_path, payload)
    validate_filtered_inventory(inventory_path, verify_part_sha256=True)
    return payload


def _parse_public_manifests(values: list[str]) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_PUBLIC_MANIFESTS)
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--public-manifest must use DATASET=PATH")
        dataset, path = value.split("=", 1)
        dataset = dataset.strip()
        if not dataset or dataset in result:
            raise ValueError(f"Duplicate/empty public dataset: {dataset!r}")
        result[dataset] = Path(path).expanduser().resolve()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fingerprint, deduplicate, and public-filter Stage211 social VAD segments."
    )
    parser.add_argument("--materialized-inventory", type=Path, default=DEFAULT_MATERIALIZED_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--public-manifest", action="append", default=[])
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--build-public", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    materialized = validate_materialized_inventory(args.materialized_inventory)
    output_root = args.output_root.expanduser().resolve()
    public_manifests = _parse_public_manifests(args.public_manifest)
    expected_public_rows = (
        EXPECTED_PUBLIC_ROWS
        if set(public_manifests) == set(EXPECTED_PUBLIC_ROWS)
        else None
    )
    if args.finalize_only:
        result = finalize_filtered_inventory(
            materialized=materialized,
            public_manifests=public_manifests,
            output_root=output_root,
            expected_public_rows=expected_public_rows,
        )
        print(
            f"filtered_inventory={output_root / 'filtered_inventory.json'} "
            f"rows={result['selected_rows']} hours={result['selected_hours']:.6f}",
            flush=True,
        )
        return 0
    result = run_social_fingerprint_worker(
        materialized=materialized,
        output_root=output_root,
        worker_index=int(args.worker_index),
        num_workers=int(args.num_workers),
        max_sources=args.max_sources,
    )
    print(f"social_fingerprints={result}", flush=True)
    if args.build_public:
        build_public_fingerprints(
            public_manifests=public_manifests,
            output_root=output_root,
            expected_rows=expected_public_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
