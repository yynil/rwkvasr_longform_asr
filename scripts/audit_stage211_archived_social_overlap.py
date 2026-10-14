#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_ARCHIVE = Path("/media/usbhd/new_video.tar")
DEFAULT_SOCIAL_ROOT = Path("/media/usbhd/videos")
DEFAULT_SOURCE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_audit_v1/source_inventory_full.json"
)
DEFAULT_USB_COVERAGE_RECEIPT = (
    Path.home()
    / "rwkvasr_data/stage211_usb_top_level_coverage_v1/coverage_receipt.json"
)
DEFAULT_OUTPUT = (
    Path.home()
    / "rwkvasr_data/stage211_archived_social_overlap_v1/overlap_receipt.json"
)
EXPECTED_ARCHIVE_BYTES = 7_230_351_360
EXPECTED_FIRST_MEMBER_SHA256 = (
    "613499aab18dc837060e6e32b041b73656cb34e5617075881a71ff72e7660b05"
)
AUDIO_SUFFIXES = {".mp3", ".wav"}


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _validate_source_inventory(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    inventory = _load_json(path, label="Stage211 social source inventory")
    expected = {
        "schema_version": 1,
        "artifact": "stage211_social_vad_source_inventory",
        "complete": True,
        "training_ready": False,
        "admission_state": "source_inventory_only",
        "layout_valid": True,
        "media_content_hash_complete": True,
    }
    if any(inventory.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 social source inventory contract mismatch.")
    probe = inventory.get("audio_probe")
    records = inventory.get("source_records")
    if (
        not isinstance(probe, dict)
        or probe.get("complete") is not True
        or probe.get("errors") != []
        or not math.isfinite(float(probe.get("hours", float("nan"))))
        or not isinstance(records, list)
        or len(records) != int(inventory.get("canonical_audio_files", -1))
    ):
        raise ValueError("Stage211 social source inventory is not fully hash/probe complete.")
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Stage211 social source record is invalid.")
        source_path = Path(str(record.get("path") or "")).expanduser().resolve()
        source_hash = str(record.get("sha256") or "")
        if (
            str(source_path) in by_path
            or not source_path.is_file()
            or int(record.get("size_bytes", -1)) != source_path.stat().st_size
            or len(source_hash) != 64
        ):
            raise ValueError(f"Stage211 social source binding is invalid: {source_path}")
        by_path[str(source_path)] = record
    inventory["_path"] = str(path)
    inventory["_sha256"] = _sha256(path)
    inventory["_by_path"] = by_path
    return inventory


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe archived-social member path: {name!r}")
    return path


def index_archive(path: Path) -> tuple[list[dict[str, Any]], int]:
    path = path.expanduser().resolve()
    records: list[dict[str, Any]] = []
    directories = 0
    with tarfile.open(path, mode="r:") as archive:
        for member_index, member in enumerate(archive):
            member_path = _safe_member_name(member.name)
            if member.isdir():
                directories += 1
                continue
            if not member.isfile():
                raise ValueError(
                    f"Unsupported archived-social member type: {member.name!r} type={member.type!r}"
                )
            suffix = member_path.suffix.lower()
            if suffix not in AUDIO_SUFFIXES:
                raise ValueError(f"Unsupported archived-social regular member: {member.name!r}")
            if member.size <= 0 or member.offset_data < 0:
                raise ValueError(f"Invalid archived-social member bounds: {member.name!r}")
            records.append(
                {
                    "member_index": member_index,
                    "member_name": member.name,
                    "suffix": suffix,
                    "size_bytes": int(member.size),
                    "offset_data": int(member.offset_data),
                }
            )
    if not records:
        raise ValueError(f"Archived-social tar has no audio members: {path}")
    records.sort(key=lambda record: int(record["offset_data"]))
    previous_end = 0
    for record in records:
        start = int(record["offset_data"])
        end = start + int(record["size_bytes"])
        if start < previous_end:
            raise ValueError("Archived-social tar member byte ranges overlap.")
        previous_end = end
    return records, directories


def hash_archive_and_members(
    path: Path,
    records: list[dict[str, Any]],
    *,
    chunk_size: int = 8 << 20,
) -> tuple[str, list[str]]:
    archive_digest = hashlib.sha256()
    member_digests = [hashlib.sha256() for _ in records]
    member_bytes = [0 for _ in records]
    member_index = 0
    position = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            archive_digest.update(chunk)
            chunk_start = position
            chunk_end = chunk_start + len(chunk)
            while member_index < len(records):
                record = records[member_index]
                member_start = int(record["offset_data"])
                member_end = member_start + int(record["size_bytes"])
                if member_end <= chunk_start:
                    if member_bytes[member_index] != int(record["size_bytes"]):
                        raise ValueError("Archived-social member hashing skipped bytes.")
                    member_index += 1
                    continue
                if member_start >= chunk_end:
                    break
                overlap_start = max(chunk_start, member_start)
                overlap_end = min(chunk_end, member_end)
                member_digests[member_index].update(
                    chunk[overlap_start - chunk_start : overlap_end - chunk_start]
                )
                member_bytes[member_index] += overlap_end - overlap_start
                if member_end <= chunk_end:
                    if member_bytes[member_index] != int(record["size_bytes"]):
                        raise ValueError("Archived-social member hashing is incomplete.")
                    member_index += 1
                else:
                    break
            position = chunk_end
    if member_index != len(records) or any(
        actual != int(record["size_bytes"])
        for actual, record in zip(member_bytes, records)
    ):
        raise ValueError("Archived-social tar ended before every member was hashed.")
    return archive_digest.hexdigest(), [digest.hexdigest() for digest in member_digests]


def build_receipt(
    *,
    archive_path: Path,
    social_root: Path,
    source_inventory_path: Path,
    usb_coverage_receipt_path: Path,
    strict_production: bool,
) -> dict[str, Any]:
    archive_path = archive_path.expanduser().resolve()
    social_root = social_root.expanduser().resolve()
    if not archive_path.is_file() or not social_root.is_dir():
        raise ValueError("Archived-social source paths are unavailable.")
    if strict_production and archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("Archived-social production tar size changed.")
    inventory = _validate_source_inventory(source_inventory_path)
    usb_coverage_receipt_path = usb_coverage_receipt_path.expanduser().resolve()
    usb_coverage = _load_json(
        usb_coverage_receipt_path,
        label="Stage211 USB top-level coverage receipt",
    )
    if (
        usb_coverage.get("artifact") != "usb_top_level_coverage"
        or usb_coverage.get("classification_complete") is not True
    ):
        raise ValueError("Stage211 USB top-level coverage receipt is invalid.")

    records, directory_members = index_archive(archive_path)
    archive_sha256, member_hashes = hash_archive_and_members(archive_path, records)
    exact_duplicates = 0
    unique_members: list[dict[str, Any]] = []
    canonical_records: list[dict[str, Any]] = []
    by_path = inventory["_by_path"]
    for record, member_hash in zip(records, member_hashes):
        member_path = _safe_member_name(str(record["member_name"]))
        existing_path = (social_root / Path(*member_path.parts)).resolve()
        try:
            existing_path.relative_to(social_root)
        except ValueError as error:
            raise ValueError(f"Archived-social member escapes social root: {member_path}") from error
        existing_record = by_path.get(str(existing_path))
        existing_exists = existing_path.is_file()
        existing_size = existing_path.stat().st_size if existing_exists else None
        if existing_record is not None:
            existing_sha256 = str(existing_record["sha256"])
            hash_source = "bound_social_source_inventory"
        elif existing_exists and existing_size == int(record["size_bytes"]):
            existing_sha256 = _sha256(existing_path)
            hash_source = "direct_existing_path_hash"
        else:
            existing_sha256 = None
            hash_source = None
        duplicate = (
            existing_sha256 == member_hash
            and existing_size == int(record["size_bytes"])
        )
        resolved = {
            **record,
            "sha256": member_hash,
            "existing_path": str(existing_path),
            "existing_path_exists": existing_exists,
            "existing_size_bytes": existing_size,
            "existing_sha256": existing_sha256,
            "existing_hash_source": hash_source,
            "exact_existing_social_duplicate": duplicate,
        }
        canonical_records.append(resolved)
        if duplicate:
            exact_duplicates += 1
        else:
            unique_members.append(resolved)

    if strict_production and canonical_records[0]["sha256"] != EXPECTED_FIRST_MEMBER_SHA256:
        raise ValueError("Archived-social first-member production hash changed.")
    all_duplicates = exact_duplicates == len(canonical_records)
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "archived_social_overlap",
        "complete": True,
        "training_ready": False,
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "archive_directory_members": directory_members,
        "archive_audio_members": len(canonical_records),
        "archive_audio_bytes": sum(int(record["size_bytes"]) for record in canonical_records),
        "source_inventory_path": inventory["_path"],
        "source_inventory_sha256": inventory["_sha256"],
        "usb_top_level_coverage_receipt_path": str(usb_coverage_receipt_path),
        "usb_top_level_coverage_receipt_sha256": _sha256(usb_coverage_receipt_path),
        "comparison_mode": "exact_member_bytes_sha256_against_existing_social_path",
        "exact_duplicate_members": exact_duplicates,
        "unique_members": len(unique_members),
        "all_members_exact_existing_social_duplicates": all_duplicates,
        "archive_excluded_from_training_as_duplicate": all_duplicates,
        "requires_unique_member_vad_pipeline": not all_duplicates,
        "global_acoustic_near_duplicate_complete": False,
        "member_records": canonical_records,
        "unique_member_records": unique_members,
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_immutable(path: Path, value: dict[str, Any]) -> str:
    path = path.expanduser().resolve()
    payload = _canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace non-identical overlap receipt: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prove whether new_video.tar adds unique Stage211 social audio."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--social-root", type=Path, default=DEFAULT_SOCIAL_ROOT)
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument(
        "--usb-coverage-receipt",
        type=Path,
        default=DEFAULT_USB_COVERAGE_RECEIPT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-nonproduction-layout", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt = build_receipt(
        archive_path=args.archive,
        social_root=args.social_root,
        source_inventory_path=args.source_inventory,
        usb_coverage_receipt_path=args.usb_coverage_receipt,
        strict_production=not args.allow_nonproduction_layout,
    )
    status = _write_immutable(args.output, receipt)
    print(
        f"archived_social_overlap={args.output.expanduser().resolve()} status={status} "
        f"members={receipt['archive_audio_members']} "
        f"duplicates={receipt['exact_duplicate_members']} "
        f"unique={receipt['unique_members']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
