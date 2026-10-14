#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rwkvasr.eval.stage211_supplemental import validate_stage211_supplemental_inventory

try:
    from scripts import audit_stage211_base_public_pcm_overlap as pcm_audit
    from scripts.filter_stage211_social_pcm_overlap import (
        _public_fingerprint_paths,
        _validate_public_fingerprint_receipt,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import audit_stage211_base_public_pcm_overlap as pcm_audit
    from filter_stage211_social_pcm_overlap import (
        _public_fingerprint_paths,
        _validate_public_fingerprint_receipt,
    )


OLD_ARCHIVE = Path("/media/usbhd/LLaSO-Align/LLaSO-Align-audio.part531.zip")
OLD_ARCHIVE_SHA256 = "2a00be09fea4a0d3355b2537706b87e97d2bc2c5014b120af53e4119309aaa68"
REPAIRED_ARCHIVE = (
    Path.home() / "rwkvasr_data/stage211_source_repair_v1/LLaSO-Align-audio.part531.zip"
)
REPAIRED_ARCHIVE_SHA256 = (
    "bbf90f6fd927c08739a7fab87af690c415363a8bffd72b75705ea1117035ca33"
)
ARCHIVE_SIZE_BYTES = 1_551_133_575
ARCHIVE_AUDIO_MEMBERS = 4_000
SOURCE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_supplemental_natural_v1/supplemental_inventory.json"
)
REPAIRED_BASE_ROOT = Path.home() / "rwkvasr_data/stage211_supplemental_natural_v2"
SOURCE_PCM_ROOT = Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v1"
REPAIRED_PCM_ROOT = (
    Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_repaired_v1"
)
SCHEMA_VERSION = 1
DATA_REPAIR_ARTIFACT = "stage211_supplemental_source_repair"
PCM_REUSE_ARTIFACT = "stage211_base_public_pcm_incremental_repair"
FINAL_REPAIR_ARTIFACT = "stage211_supplemental_source_repair_final"


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _immutable_bytes(path: Path, payload: bytes) -> str:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace different Stage211 repair evidence: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
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
        raise ValueError(f"Stage211 repair hardlink source is unavailable: {source}")
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise ValueError(f"Stage211 repair hardlink destination differs: {destination}")
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        os.link(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def _copy_file_immutable(source: Path, destination: Path) -> str:
    if destination.is_file():
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file, length=16 << 20)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def _part_path(manifest_path: Path, record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def _iter_jsonl_bytes(path: Path) -> Iterable[bytes]:
    with path.open("rb") as source:
        for line in source:
            if line.strip():
                yield line


def _validate_repaired_zip(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    expected_members: int,
    deep: bool,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size != expected_size_bytes:
        raise ValueError(f"Stage211 repaired archive size changed: {path}")
    digest = _sha256(path)
    if digest != expected_sha256:
        raise ValueError(f"Stage211 repaired archive SHA-256 changed: {path}")
    with zipfile.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != expected_members or len({info.filename for info in infos}) != len(infos):
            raise ValueError("Stage211 repaired archive member coverage changed.")
        if deep:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Stage211 repaired archive CRC failed: {bad_member}")
    return {
        "path": str(path),
        "size_bytes": expected_size_bytes,
        "sha256": digest,
        "members": len(infos),
        "full_crc_pass": deep,
    }


def _rewrite_part(
    *,
    source: Path,
    destination: Path,
    old_archive: str,
    repaired_archive: str,
) -> tuple[str, int, dict[str, tuple[int, int, int]]]:
    old_bytes = old_archive.encode("utf-8")
    repaired_bytes = repaired_archive.encode("utf-8")
    replacements = 0
    members: dict[str, tuple[int, int, int]] = {}
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_file, temporary.open("xb") as output_file:
            for line_number, line in enumerate(input_file, start=1):
                occurrences = line.count(old_bytes)
                if occurrences > 1:
                    raise ValueError(
                        f"Stage211 repair path occurs repeatedly: {source}:{line_number}"
                    )
                if occurrences:
                    row = json.loads(line)
                    if (
                        row.get("shard_name") != old_archive
                        or row.get("storage_kind") != "zip"
                        or not row.get("audio_member")
                    ):
                        raise ValueError(
                            f"Stage211 repair row contract changed: {source}:{line_number}"
                        )
                    member = str(row["audio_member"])
                    metadata = (
                        int(row.get("audio_size", -1)),
                        int(row.get("zip_crc32", -1)),
                        int(row.get("zip_compress_type", -1)),
                    )
                    if member in members:
                        raise ValueError(f"Stage211 repaired ZIP member is duplicated: {member}")
                    members[member] = metadata
                    replacements += 1
                    line = line.replace(old_bytes, repaired_bytes)
                output_file.write(line)
            output_file.flush()
            os.fsync(output_file.fileno())
        if destination.is_file():
            if destination.read_bytes() != temporary.read_bytes():
                raise ValueError(f"Stage211 repaired part differs: {destination}")
            temporary.unlink()
            status = "reused"
        else:
            temporary.replace(destination)
            status = "created"
    finally:
        temporary.unlink(missing_ok=True)
    return status, replacements, members


def _replace_manifest_parts(
    manifest: dict[str, Any],
    replacement_records: dict[str, dict[str, Any]],
) -> None:
    seen: Counter[str] = Counter()
    for split in (manifest.get("splits") or {}).values():
        for bucket in split.get("buckets") or []:
            parts = bucket.get("parts") or []
            for index, record in enumerate(parts):
                path = str(record.get("path") or "")
                if path in replacement_records:
                    parts[index] = dict(replacement_records[path])
                    seen[path] += 1
    if seen != Counter({path: 1 for path in replacement_records}):
        raise ValueError("Stage211 repaired manifest part coverage changed.")


def migrate_supplemental_inventory(
    *,
    source_inventory_path: Path,
    output_root: Path,
    old_archive: Path,
    repaired_archive: Path,
    old_archive_sha256: str,
    repaired_archive_sha256: str,
    archive_size_bytes: int,
    expected_archive_members: int,
    deep_zip_validation: bool,
) -> dict[str, Any]:
    source_inventory_path = source_inventory_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    old_archive = old_archive.expanduser().resolve()
    repaired_archive = repaired_archive.expanduser().resolve()
    source = validate_stage211_supplemental_inventory(
        source_inventory_path,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    source_inventory = source["inventory"]
    source_manifest_path = Path(source["bucket_manifest_path"])
    source_manifest = _load_json(source_manifest_path, label="Stage211 source bucket manifest")
    repaired_zip = _validate_repaired_zip(
        repaired_archive,
        expected_sha256=repaired_archive_sha256,
        expected_size_bytes=archive_size_bytes,
        expected_members=expected_archive_members,
        deep=deep_zip_validation,
    )

    archive_records = [dict(record) for record in source_inventory["archive_records"]]
    matching = [record for record in archive_records if Path(str(record["path"])).resolve() == old_archive]
    if len(matching) != 1 or str(matching[0].get("sha256")) != old_archive_sha256:
        raise ValueError("Stage211 corrupt source archive binding changed.")
    if int(matching[0].get("size_bytes", -1)) != archive_size_bytes:
        raise ValueError("Stage211 corrupt source archive size changed.")
    repaired_stat = repaired_archive.stat()
    matching[0].update(
        {
            "path": str(repaired_archive),
            "size_bytes": repaired_stat.st_size,
            "mtime_ns": repaired_stat.st_mtime_ns,
            "sha256": repaired_archive_sha256,
        }
    )

    destination_manifest_path = output_root / "webdataset_buckets_audio_text/manifest.json"
    destination_inventory_path = output_root / "supplemental_inventory.json"
    old_archive_text = str(old_archive)
    repaired_archive_text = str(repaired_archive)
    replacement_records: dict[str, dict[str, Any]] = {}
    changed_parts = 0
    hardlinked_parts = 0
    replaced_rows = 0
    zip_members: dict[str, tuple[int, int, int]] = {}
    for source_record in source_inventory["part_records"]:
        record = dict(source_record)
        source_part = _part_path(source_manifest_path, source_record)
        destination_part = _part_path(destination_manifest_path, source_record)
        if _sha256(source_part) != str(source_record["sha256"]):
            raise ValueError(f"Stage211 source train part changed: {source_part}")
        with source_part.open("rb") as handle:
            affected = old_archive_text.encode("utf-8") in handle.read()
        if affected:
            _, replacements, members = _rewrite_part(
                source=source_part,
                destination=destination_part,
                old_archive=old_archive_text,
                repaired_archive=repaired_archive_text,
            )
            overlap = set(zip_members).intersection(members)
            if overlap:
                raise ValueError(f"Stage211 repaired ZIP members repeat across parts: {overlap}")
            zip_members.update(members)
            replaced_rows += replacements
            changed_parts += 1
            if record.get("first_shard") == old_archive_text:
                record["first_shard"] = repaired_archive_text
            if record.get("last_shard") == old_archive_text:
                record["last_shard"] = repaired_archive_text
            record["size_bytes"] = destination_part.stat().st_size
            record["sha256"] = _sha256(destination_part)
        else:
            _immutable_hardlink(source_part, destination_part)
            if not os.path.samefile(source_part, destination_part):
                raise ValueError(f"Stage211 unaffected part is not hard-linked: {destination_part}")
            hardlinked_parts += 1
        replacement_records[str(record["path"])] = record

    if replaced_rows != expected_archive_members or len(zip_members) != expected_archive_members:
        raise ValueError(
            "Stage211 repaired row coverage changed: "
            f"rows={replaced_rows} members={len(zip_members)}"
        )
    with zipfile.ZipFile(repaired_archive) as archive:
        infos = {info.filename: info for info in archive.infolist() if not info.is_dir()}
    if set(infos) != set(zip_members):
        raise ValueError("Stage211 repaired row/member set changed.")
    for member, (size, crc, compress_type) in zip_members.items():
        info = infos[member]
        if (info.file_size, info.CRC, info.compress_type) != (size, crc, compress_type):
            raise ValueError(f"Stage211 repaired ZIP metadata changed: {member}")

    destination_manifest = json.loads(json.dumps(source_manifest))
    destination_manifest["source_length_index_path"] = str(destination_inventory_path)
    _replace_manifest_parts(destination_manifest, replacement_records)
    _immutable_json(destination_manifest_path, destination_manifest)

    destination_inventory = json.loads(json.dumps(source_inventory))
    destination_inventory["archive_records"] = archive_records
    destination_inventory["part_records"] = [
        replacement_records[str(record["path"])] for record in source_inventory["part_records"]
    ]
    destination_inventory["bucket_manifest_path"] = str(destination_manifest_path)
    destination_inventory["bucket_manifest_sha256"] = _sha256(destination_manifest_path)
    destination_inventory["source_repair"] = {
        "mode": "exact_archive_path_replacement_v1",
        "old_archive_path": old_archive_text,
        "old_archive_sha256": old_archive_sha256,
        "repaired_archive_path": repaired_archive_text,
        "repaired_archive_sha256": repaired_archive_sha256,
        "repaired_archive_size_bytes": archive_size_bytes,
        "repaired_archive_members": expected_archive_members,
        "replaced_rows": replaced_rows,
        "changed_parts": changed_parts,
        "hardlinked_parts": hardlinked_parts,
    }
    _immutable_json(destination_inventory_path, destination_inventory)
    repaired = validate_stage211_supplemental_inventory(
        destination_inventory_path,
        require_training_ready=True,
        verify_part_sha256=True,
    )
    if repaired["rows"] != source["rows"] or repaired["hours"] != source["hours"]:
        raise ValueError("Stage211 repaired inventory totals changed.")

    receipt_path = output_root / "source_repair_receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": DATA_REPAIR_ARTIFACT,
        "complete": True,
        "preservation_mode": "byte_exact_rows_except_shard_name_v1",
        "source_inventory_path": source["inventory_path"],
        "source_inventory_sha256": source["inventory_sha256"],
        "source_bucket_manifest_path": source["bucket_manifest_path"],
        "source_bucket_manifest_sha256": source["bucket_manifest_sha256"],
        "repaired_inventory_path": repaired["inventory_path"],
        "repaired_inventory_sha256": repaired["inventory_sha256"],
        "repaired_bucket_manifest_path": repaired["bucket_manifest_path"],
        "repaired_bucket_manifest_sha256": repaired["bucket_manifest_sha256"],
        "rows": repaired["rows"],
        "hours": repaired["hours"],
        "part_count": len(replacement_records),
        "changed_parts": changed_parts,
        "hardlinked_parts": hardlinked_parts,
        "replaced_rows": replaced_rows,
        "old_archive": {
            "path": old_archive_text,
            "size_bytes": archive_size_bytes,
            "sha256": old_archive_sha256,
        },
        "repaired_archive": repaired_zip,
        "zero_key_count_duration_split_bucket_drift": True,
    }
    _immutable_json(receipt_path, receipt)
    return validate_source_repair_receipt(receipt_path)


def validate_source_repair_receipt(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    receipt = _load_json(path, label="Stage211 supplemental source repair receipt")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": DATA_REPAIR_ARTIFACT,
        "complete": True,
        "preservation_mode": "byte_exact_rows_except_shard_name_v1",
        "zero_key_count_duration_split_bucket_drift": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 supplemental source repair contract changed.")
    source_path = Path(str(receipt["source_inventory_path"])).resolve()
    repaired_path = Path(str(receipt["repaired_inventory_path"])).resolve()
    if _sha256(source_path) != receipt.get("source_inventory_sha256"):
        raise ValueError("Stage211 source inventory changed after repair.")
    if _sha256(repaired_path) != receipt.get("repaired_inventory_sha256"):
        raise ValueError("Stage211 repaired inventory changed.")
    source = validate_stage211_supplemental_inventory(
        source_path, require_training_ready=True, verify_part_sha256=True
    )
    repaired = validate_stage211_supplemental_inventory(
        repaired_path, require_training_ready=True, verify_part_sha256=True
    )
    if (
        source["rows"] != repaired["rows"]
        or source["hours"] != repaired["hours"]
        or int(receipt.get("rows", -1)) != repaired["rows"]
        or float(receipt.get("hours", -1.0)) != repaired["hours"]
    ):
        raise ValueError("Stage211 source repair total preservation changed.")
    repair = repaired["inventory"].get("source_repair") or {}
    if (
        repair.get("old_archive_path") != receipt.get("old_archive", {}).get("path")
        or repair.get("repaired_archive_path") != receipt.get("repaired_archive", {}).get("path")
        or int(repair.get("replaced_rows", -1)) != int(receipt.get("replaced_rows", -2))
    ):
        raise ValueError("Stage211 repaired inventory provenance changed.")

    old_archive = str(receipt["old_archive"]["path"])
    repaired_archive = str(receipt["repaired_archive"]["path"])
    source_inventory = source["inventory"]
    repaired_inventory = repaired["inventory"]
    source_parts = {str(record["path"]): record for record in source_inventory["part_records"]}
    repaired_parts = {
        str(record["path"]): record for record in repaired_inventory["part_records"]
    }
    if source_parts.keys() != repaired_parts.keys():
        raise ValueError("Stage211 repaired part set changed.")
    changed_parts = 0
    hardlinked_parts = 0
    replaced_rows = 0
    source_manifest_path = Path(source["bucket_manifest_path"])
    repaired_manifest_path = Path(repaired["bucket_manifest_path"])
    for relative_path, source_record in source_parts.items():
        repaired_record = repaired_parts[relative_path]
        source_part = _part_path(source_manifest_path, source_record)
        repaired_part = _part_path(repaired_manifest_path, repaired_record)
        affected = (
            source_record.get("first_shard") == old_archive
            or source_record.get("last_shard") == old_archive
        )
        if affected:
            changed_parts += 1
            normalized_record = dict(repaired_record)
            for key in ("first_shard", "last_shard"):
                if normalized_record.get(key) == repaired_archive:
                    normalized_record[key] = old_archive
            normalized_record["size_bytes"] = source_record["size_bytes"]
            normalized_record["sha256"] = source_record["sha256"]
            if normalized_record != source_record or os.path.samefile(source_part, repaired_part):
                raise ValueError(f"Stage211 repaired part metadata drifted: {relative_path}")
            source_lines = _iter_jsonl_bytes(source_part)
            repaired_lines = _iter_jsonl_bytes(repaired_part)
            for source_line, repaired_line in zip(source_lines, repaired_lines, strict=True):
                occurrences = repaired_line.count(repaired_archive.encode("utf-8"))
                if repaired_line.replace(
                    repaired_archive.encode("utf-8"), old_archive.encode("utf-8")
                ) != source_line:
                    raise ValueError(f"Stage211 repaired row drifted: {relative_path}")
                replaced_rows += occurrences
        else:
            hardlinked_parts += 1
            if repaired_record != source_record or not os.path.samefile(
                source_part, repaired_part
            ):
                raise ValueError(f"Stage211 unaffected part drifted: {relative_path}")
    if (
        changed_parts != int(receipt.get("changed_parts", -1))
        or hardlinked_parts != int(receipt.get("hardlinked_parts", -1))
        or replaced_rows != int(receipt.get("replaced_rows", -1))
    ):
        raise ValueError("Stage211 repaired row/part preservation totals changed.")

    source_archives = source_inventory["archive_records"]
    repaired_archives = repaired_inventory["archive_records"]
    if len(source_archives) != len(repaired_archives):
        raise ValueError("Stage211 repaired archive set changed.")
    repaired_archive_changes = 0
    for source_record, repaired_record in zip(
        source_archives, repaired_archives, strict=True
    ):
        normalized = dict(repaired_record)
        if normalized.get("path") == repaired_archive:
            repaired_archive_changes += 1
            normalized.update(
                {
                    "path": old_archive,
                    "size_bytes": receipt["old_archive"]["size_bytes"],
                    "mtime_ns": source_record["mtime_ns"],
                    "sha256": receipt["old_archive"]["sha256"],
                }
            )
        if normalized != source_record:
            raise ValueError("Stage211 repaired archive metadata drifted.")
    if repaired_archive_changes != 1:
        raise ValueError("Stage211 repair did not replace exactly one archive record.")

    source_manifest = _load_json(source_manifest_path, label="Stage211 source bucket manifest")
    repaired_manifest = _load_json(
        repaired_manifest_path, label="Stage211 repaired bucket manifest"
    )
    normalized_manifest = json.loads(json.dumps(repaired_manifest))
    normalized_manifest["source_length_index_path"] = source_manifest[
        "source_length_index_path"
    ]
    _replace_manifest_parts(normalized_manifest, source_parts)
    if normalized_manifest != source_manifest:
        raise ValueError("Stage211 repaired bucket manifest drifted.")

    normalized_inventory = json.loads(json.dumps(repaired_inventory))
    normalized_inventory.pop("source_repair", None)
    normalized_inventory["archive_records"] = source_archives
    normalized_inventory["part_records"] = source_inventory["part_records"]
    normalized_inventory["bucket_manifest_path"] = source_inventory[
        "bucket_manifest_path"
    ]
    normalized_inventory["bucket_manifest_sha256"] = source_inventory[
        "bucket_manifest_sha256"
    ]
    if normalized_inventory != source_inventory:
        raise ValueError("Stage211 repaired inventory semantic fields drifted.")
    return receipt


def _copy_and_patch_location_index(
    *,
    source_database: Path,
    destination_database: Path,
    source_base: dict[str, Any],
    repaired_base: dict[str, Any],
    old_archive: Path,
    repaired_archive: Path,
) -> str:
    status = _copy_file_immutable(source_database, destination_database)
    old_text = str(old_archive.resolve())
    repaired_text = str(repaired_archive.resolve())
    repaired_record = next(
        record
        for record in repaired_base["inventory"]["archive_records"]
        if Path(str(record["path"])).resolve() == repaired_archive.resolve()
    )
    connection = sqlite3.connect(destination_database)
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if status == "created":
            entries = connection.execute(
                "UPDATE entries SET shard_path = ? WHERE shard_path = ?",
                (repaired_text, old_text),
            ).rowcount
            archives = connection.execute(
                "UPDATE archives SET shard_path = ?, archive_size_bytes = ?, "
                "archive_mtime_ns = ?, archive_sha256 = ? WHERE shard_path = ?",
                (
                    repaired_text,
                    int(repaired_record["size_bytes"]),
                    int(repaired_record["mtime_ns"]),
                    str(repaired_record["sha256"]),
                    old_text,
                ),
            ).rowcount
            if entries != int(repaired_record["audio_members"]) or archives != 1:
                raise ValueError(
                    f"Stage211 repaired location-index coverage changed: {entries}/{archives}"
                )
            replacements = {
                "base_inventory_path": repaired_base["inventory_path"],
                "base_inventory_sha256": repaired_base["inventory_sha256"],
                "base_bucket_manifest_path": repaired_base["bucket_manifest_path"],
                "base_bucket_manifest_sha256": repaired_base["bucket_manifest_sha256"],
            }
            connection.executemany(
                "UPDATE metadata SET value = ? WHERE key = ?",
                [(value, key) for key, value in replacements.items()],
            )
            connection.commit()
        else:
            expected = {
                "base_inventory_path": repaired_base["inventory_path"],
                "base_inventory_sha256": repaired_base["inventory_sha256"],
                "base_bucket_manifest_path": repaired_base["bucket_manifest_path"],
                "base_bucket_manifest_sha256": repaired_base["bucket_manifest_sha256"],
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("Existing Stage211 repaired location index differs.")
    except BaseException:
        connection.close()
        if status == "created":
            destination_database.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    return status


def _rebind_public_fingerprints(*, source_root: Path, output_root: Path) -> int:
    count = 0
    for dataset, manifest_path in sorted(pcm_audit.DEFAULT_PUBLIC_MANIFESTS.items()):
        source_part, source_receipt_path = _public_fingerprint_paths(source_root, dataset)
        recorded_source_receipt = _load_json(
            source_receipt_path,
            label=f"Stage211 source public PCM fingerprint receipt ({dataset})",
        )
        source_rows = int(recorded_source_receipt.get("rows", -1))
        if source_rows <= 0:
            raise ValueError(f"Stage211 source public PCM row count is invalid: {dataset}")
        source_receipt = _validate_public_fingerprint_receipt(
            source_receipt_path,
            dataset=dataset,
            manifest_path=manifest_path.resolve(),
            expected_rows=source_rows,
        )
        destination_part, destination_receipt_path = _public_fingerprint_paths(
            output_root, dataset
        )
        _immutable_hardlink(source_part, destination_part)
        destination_receipt = {
            **source_receipt,
            "part_path": str(destination_part.resolve()),
        }
        _immutable_json(destination_receipt_path, destination_receipt)
        _validate_public_fingerprint_receipt(
            destination_receipt_path,
            dataset=dataset,
            manifest_path=manifest_path.resolve(),
            expected_rows=source_rows,
        )
        if not os.path.samefile(source_part, destination_part):
            raise ValueError("Stage211 public fingerprint hardlink proof changed.")
        count += 1
    return count


def prepare_incremental_pcm_audit(
    *,
    source_inventory_path: Path,
    repaired_inventory_path: Path,
    source_root: Path,
    output_root: Path,
    old_archive: Path,
    repaired_archive: Path,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    source_base = pcm_audit.validate_base_inventory(source_inventory_path)
    repaired_base = pcm_audit.validate_base_inventory(repaired_inventory_path)
    source_index = pcm_audit.validate_location_index(source_root, base=source_base)
    source_database = Path(source_index["database_path"])
    destination_database = output_root / "manifest_location_index.sqlite"
    output_root.mkdir(parents=True, exist_ok=True)
    index_status = _copy_and_patch_location_index(
        source_database=source_database,
        destination_database=destination_database,
        source_base=source_base,
        repaired_base=repaired_base,
        old_archive=old_archive,
        repaired_archive=repaired_archive,
    )
    summary = pcm_audit._index_database_summary(destination_database, base=repaired_base)
    destination_index_receipt = output_root / "manifest_location_index.receipt.json"
    pcm_audit._immutable_json(
        destination_index_receipt,
        pcm_audit._index_receipt_payload(
            database_path=destination_database,
            base=repaired_base,
            summary=summary,
        ),
    )
    destination_index = pcm_audit.validate_location_index(output_root, base=repaired_base)
    repaired_indices = [
        int(record["archive_index"])
        for record in destination_index["archives"]
        if Path(str(record["shard_path"])).resolve() == repaired_archive.resolve()
    ]
    if len(repaired_indices) != 1:
        raise ValueError("Stage211 repaired archive index changed.")
    repaired_index = repaired_indices[0]

    statuses: Counter[str] = Counter()
    inode_digest = hashlib.sha256()
    rebound = 0
    for archive in destination_index["archives"]:
        archive_index = int(archive["archive_index"])
        if archive_index == repaired_index:
            continue
        source_archive = source_index["archives"][archive_index]
        if archive != source_archive:
            raise ValueError(f"Stage211 unaffected archive metadata changed: {archive_index}")
        source_fingerprint, source_receipt_path = pcm_audit._archive_paths(
            source_root, archive_index
        )
        source_receipt = pcm_audit._validate_archive_receipt(
            source_receipt_path,
            location_index=source_index,
            archive=source_archive,
        )
        destination_fingerprint, destination_receipt_path = pcm_audit._archive_paths(
            output_root, archive_index
        )
        statuses[_immutable_hardlink(source_fingerprint, destination_fingerprint)] += 1
        destination_receipt = {
            **source_receipt,
            "location_index_receipt_path": destination_index["receipt_path"],
            "location_index_receipt_sha256": destination_index["receipt_sha256"],
            "fingerprint_path": str(destination_fingerprint.resolve()),
        }
        _immutable_json(destination_receipt_path, destination_receipt)
        pcm_audit._validate_archive_receipt(
            destination_receipt_path,
            location_index=destination_index,
            archive=archive,
        )
        if not os.path.samefile(source_fingerprint, destination_fingerprint):
            raise ValueError(f"Stage211 fingerprint hardlink proof changed: {archive_index}")
        stat = destination_fingerprint.stat()
        inode_digest.update(f"{archive_index}:{stat.st_dev}:{stat.st_ino}\n".encode())
        rebound += 1
    public_count = _rebind_public_fingerprints(
        source_root=source_root,
        output_root=output_root,
    )
    receipt_path = output_root / "incremental_repair_receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": PCM_REUSE_ARTIFACT,
        "complete": True,
        "reuse_mode": "hardlink_unchanged_fingerprints_recompute_repaired_archive_v1",
        "source_inventory_path": source_base["inventory_path"],
        "source_inventory_sha256": source_base["inventory_sha256"],
        "repaired_inventory_path": repaired_base["inventory_path"],
        "repaired_inventory_sha256": repaired_base["inventory_sha256"],
        "source_location_index_receipt_path": source_index["receipt_path"],
        "source_location_index_receipt_sha256": source_index["receipt_sha256"],
        "destination_location_index_receipt_path": destination_index["receipt_path"],
        "destination_location_index_receipt_sha256": destination_index["receipt_sha256"],
        "archive_count": len(destination_index["archives"]),
        "repaired_archive_index": repaired_index,
        "rebound_archive_fingerprints": rebound,
        "rebound_public_fingerprints": public_count,
        "archive_hardlink_status": dict(sorted(statuses.items())),
        "hardlink_inode_set_sha256": inode_digest.hexdigest(),
        "location_index_copy_status": index_status,
    }
    _immutable_json(receipt_path, receipt)
    return receipt


def run_repaired_pcm_audit(
    *,
    repaired_inventory_path: Path,
    output_root: Path,
    repaired_archive: Path,
    decode_workers: int,
) -> dict[str, Any]:
    base = pcm_audit.validate_base_inventory(repaired_inventory_path)
    location_index = pcm_audit.validate_location_index(output_root, base=base)
    repaired_indices = [
        int(record["archive_index"])
        for record in location_index["archives"]
        if Path(str(record["shard_path"])).resolve() == repaired_archive.resolve()
    ]
    if len(repaired_indices) != 1:
        raise ValueError("Stage211 repaired archive index changed.")
    status = pcm_audit.fingerprint_base_archive(
        location_index=location_index,
        output_root=output_root,
        archive_index=repaired_indices[0],
        decode_workers=decode_workers,
    )
    print(
        f"[stage211-source-repair] repaired_archive_index={repaired_indices[0]} status={status}",
        flush=True,
    )
    return pcm_audit.finalize_audit(
        base=base,
        public_manifests=dict(pcm_audit.DEFAULT_PUBLIC_MANIFESTS),
        output_root=output_root,
        expected_public_rows=None,
        location_index=location_index,
    )


def finalize_repair(
    *,
    source_repair_receipt: Path,
    incremental_repair_receipt: Path,
    audit_receipt: Path,
) -> dict[str, Any]:
    source_repair = validate_source_repair_receipt(source_repair_receipt)
    incremental = _load_json(
        incremental_repair_receipt, label="Stage211 incremental PCM repair receipt"
    )
    if (
        incremental.get("artifact") != PCM_REUSE_ARTIFACT
        or incremental.get("complete") is not True
        or int(incremental.get("rebound_archive_fingerprints", -1))
        != int(incremental.get("archive_count", -1)) - 1
    ):
        raise ValueError("Stage211 incremental PCM repair contract changed.")
    audit = pcm_audit.validate_audit_receipt(
        audit_receipt,
        require_training_ready=True,
        verify_overlap_replay=True,
    )
    if (
        audit.get("base_inventory_path") != source_repair["repaired_inventory_path"]
        or int(audit.get("scanned_rows", -1)) != int(source_repair["rows"])
        or int(audit.get("decode_failures", -1)) != 0
        or int(audit.get("public_overlap_rows", -1)) != 0
        or len(audit.get("archive_fingerprint_receipts") or [])
        != int(incremental["archive_count"])
    ):
        raise ValueError("Stage211 final repaired PCM audit coverage changed.")
    output_path = audit_receipt.parent / "source_repair_final_receipt.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": FINAL_REPAIR_ARTIFACT,
        "complete": True,
        "training_ready": True,
        "source_repair_receipt_path": str(source_repair_receipt.resolve()),
        "source_repair_receipt_sha256": _sha256(source_repair_receipt),
        "incremental_repair_receipt_path": str(incremental_repair_receipt.resolve()),
        "incremental_repair_receipt_sha256": _sha256(incremental_repair_receipt),
        "audit_receipt_path": str(audit_receipt.resolve()),
        "audit_receipt_sha256": _sha256(audit_receipt),
        "rows": int(audit["scanned_rows"]),
        "hours": float(audit["scanned_hours"]),
        "archive_count": len(audit["archive_fingerprint_receipts"]),
        "decode_failures": 0,
        "public_overlap_rows": 0,
        "repaired_archive_index": int(incremental["repaired_archive_index"]),
        "reused_archive_fingerprints": int(incremental["rebound_archive_fingerprints"]),
        "recomputed_archive_fingerprints": 1,
        "zero_key_count_duration_split_bucket_drift": True,
    }
    _immutable_json(output_path, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair one corrupted Stage211 supplemental ZIP without rescanning good archives."
    )
    parser.add_argument("--source-inventory", type=Path, default=SOURCE_INVENTORY)
    parser.add_argument("--repaired-base-root", type=Path, default=REPAIRED_BASE_ROOT)
    parser.add_argument("--source-pcm-root", type=Path, default=SOURCE_PCM_ROOT)
    parser.add_argument("--repaired-pcm-root", type=Path, default=REPAIRED_PCM_ROOT)
    parser.add_argument("--old-archive", type=Path, default=OLD_ARCHIVE)
    parser.add_argument("--repaired-archive", type=Path, default=REPAIRED_ARCHIVE)
    parser.add_argument("--old-archive-sha256", default=OLD_ARCHIVE_SHA256)
    parser.add_argument("--repaired-archive-sha256", default=REPAIRED_ARCHIVE_SHA256)
    parser.add_argument("--archive-size-bytes", type=int, default=ARCHIVE_SIZE_BYTES)
    parser.add_argument("--archive-members", type=int, default=ARCHIVE_AUDIO_MEMBERS)
    parser.add_argument("--decode-workers", type=int, default=16)
    parser.add_argument("--skip-deep-zip-validation", action="store_true")
    parser.add_argument(
        "command",
        choices=("data", "prepare-audit", "audit", "finalize", "all"),
        default="all",
        nargs="?",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repaired_inventory = args.repaired_base_root / "supplemental_inventory.json"
    source_repair_receipt = args.repaired_base_root / "source_repair_receipt.json"
    incremental_receipt = args.repaired_pcm_root / "incremental_repair_receipt.json"
    audit_receipt = args.repaired_pcm_root / "audit_receipt.json"
    if args.command in {"data", "all"}:
        migrate_supplemental_inventory(
            source_inventory_path=args.source_inventory,
            output_root=args.repaired_base_root,
            old_archive=args.old_archive,
            repaired_archive=args.repaired_archive,
            old_archive_sha256=args.old_archive_sha256,
            repaired_archive_sha256=args.repaired_archive_sha256,
            archive_size_bytes=args.archive_size_bytes,
            expected_archive_members=args.archive_members,
            deep_zip_validation=not args.skip_deep_zip_validation,
        )
    if args.command in {"prepare-audit", "all"}:
        prepare_incremental_pcm_audit(
            source_inventory_path=args.source_inventory,
            repaired_inventory_path=repaired_inventory,
            source_root=args.source_pcm_root,
            output_root=args.repaired_pcm_root,
            old_archive=args.old_archive,
            repaired_archive=args.repaired_archive,
        )
    if args.command in {"audit", "all"}:
        run_repaired_pcm_audit(
            repaired_inventory_path=repaired_inventory,
            output_root=args.repaired_pcm_root,
            repaired_archive=args.repaired_archive,
            decode_workers=args.decode_workers,
        )
    if args.command in {"finalize", "all"}:
        receipt = finalize_repair(
            source_repair_receipt=source_repair_receipt,
            incremental_repair_receipt=incremental_receipt,
            audit_receipt=audit_receipt,
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
