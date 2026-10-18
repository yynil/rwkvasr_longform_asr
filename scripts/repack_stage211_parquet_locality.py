#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from rwkvasr.eval.stage211_supplemental import validate_stage211_supplemental_inventory


DEFAULT_SOURCE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_supplemental_combined_v3/supplemental_inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_supplemental_combined_v4_locality"
PARQUET_SOURCES = frozenset({"peoples_speech_clean", "peoples_speech_dirty"})
POLICY = "parquet_row_group_max_frames_v1"
RECEIPT_ARTIFACT = "stage211_parquet_locality_receipt"
MASK_256 = (1 << 256) - 1


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_part_path(part: dict[str, Any], *, manifest_path: Path) -> Path:
    path = Path(str(part.get("path") or ""))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(str(path))
    return path


def _canonical_row(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(row: dict[str, Any]) -> str:
    source = str(row.get("source_dataset") or "")
    shard = str(row.get("shard_name") or "")
    row_group = row.get("parquet_row_group")
    row_index = row.get("parquet_row_index")
    key = str(row.get("key") or "")
    if (
        source not in PARQUET_SOURCES
        or str(row.get("storage_kind") or "") != "parquet"
        or not shard
        or not isinstance(row_group, int)
        or isinstance(row_group, bool)
        or row_group < 0
        or not isinstance(row_index, int)
        or isinstance(row_index, bool)
        or row_index < 0
        or not key
    ):
        raise ValueError(f"Invalid Stage211 Parquet row identity: {row!r}")
    return f"{source}\0{shard}\0{row_group:08d}\0{row_index:08d}\0{key}"


@dataclass
class _MultisetDigest:
    rows: int = 0
    xor: int = 0
    total: int = 0

    def update(self, payload: bytes) -> None:
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        self.rows += 1
        self.xor ^= value
        self.total = (self.total + value) & MASK_256

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "sha256_xor": f"{self.xor:064x}",
            "sha256_sum_mod_2_256": f"{self.total:064x}",
        }


@dataclass
class _InputStream:
    source: str
    bucket_id: int
    parts: tuple[Path, ...]

    def rows(self) -> Iterator[tuple[str, dict[str, Any]]]:
        previous: str | None = None
        for part in self.parts:
            with part.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"Non-object row in {part}:{line_number}")
                    identity = _identity(row)
                    if previous is not None and identity <= previous:
                        raise ValueError(
                            "Stage211 Parquet source part is not strictly archive-local: "
                            f"{part}:{line_number} identity={identity!r} previous={previous!r}"
                        )
                    previous = identity
                    yield identity, row


@dataclass
class _PartState:
    path: Path
    relative_path: str
    handle: Any
    digest: Any
    rows: int = 0
    first_shard: str | None = None
    last_shard: str | None = None


class _LocalityPartWriter:
    def __init__(self, root: Path, *, entries_per_part: int):
        self.root = root
        self.entries_per_part = int(entries_per_part)
        self._states: dict[tuple[int, str], _PartState] = {}
        self._next_part: Counter[tuple[int, str]] = Counter()
        self._parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._counts: Counter[int] = Counter()

    def _open(self, bucket_id: int, source: str) -> _PartState:
        key = (bucket_id, source)
        index = self._next_part[key]
        self._next_part[key] += 1
        source_hex = source.encode("utf-8").hex()
        relative = (
            f"train/bucket_{bucket_id:04d}/source_{source_hex}/part_{index:06d}.jsonl"
        )
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return _PartState(
            path=path,
            relative_path=relative,
            handle=path.open("wb"),
            digest=hashlib.sha256(),
        )

    def _close(self, key: tuple[int, str]) -> None:
        state = self._states.pop(key, None)
        if state is None:
            return
        state.handle.close()
        bucket_id, source = key
        self._parts[bucket_id].append(
            {
                "path": state.relative_path,
                "num_samples": state.rows,
                "first_shard": state.first_shard,
                "last_shard": state.last_shard,
                "source_label": source,
                "size_bytes": state.path.stat().st_size,
                "sha256": state.digest.hexdigest(),
            }
        )

    def write(self, row: dict[str, Any], *, bucket_id: int) -> None:
        source = str(row["source_dataset"])
        key = (int(bucket_id), source)
        state = self._states.get(key)
        if state is None:
            state = self._open(*key)
            self._states[key] = state
        payload = _canonical_row(row) + b"\n"
        state.handle.write(payload)
        state.digest.update(payload)
        state.rows += 1
        shard = str(row["shard_name"])
        state.first_shard = state.first_shard or shard
        state.last_shard = shard
        self._counts[int(bucket_id)] += 1
        if state.rows >= self.entries_per_part:
            self._close(key)

    def finalize(self) -> tuple[dict[int, list[dict[str, Any]]], Counter[int]]:
        for key in list(self._states):
            self._close(key)
        return dict(self._parts), self._counts


def _input_parts(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> tuple[dict[str, list[_InputStream]], dict[int, list[dict[str, Any]]]]:
    split = (manifest.get("splits") or {}).get("train")
    if not isinstance(split, dict):
        raise ValueError("Stage211 source manifest lacks a train split.")
    parquet: dict[str, list[_InputStream]] = defaultdict(list)
    unchanged: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for bucket in split.get("buckets") or []:
        bucket_id = int(bucket["bucket_id"])
        by_source: dict[str, list[Path]] = defaultdict(list)
        for part in bucket.get("parts") or []:
            source = str(part.get("source_label") or "")
            if source in PARQUET_SOURCES:
                by_source[source].append(
                    _resolve_part_path(part, manifest_path=manifest_path)
                )
            else:
                unchanged[bucket_id].append(dict(part))
        for source, paths in by_source.items():
            parquet[source].append(
                _InputStream(
                    source=source,
                    bucket_id=bucket_id,
                    parts=tuple(sorted(paths)),
                )
            )
    if set(parquet) != PARQUET_SOURCES:
        raise ValueError(
            "Stage211 source manifest lacks complete People's Speech parts: "
            f"observed={sorted(parquet)}"
        )
    return dict(parquet), dict(unchanged)


def _merge_source(streams: list[_InputStream]) -> Iterator[tuple[str, dict[str, Any]]]:
    iterators = [stream.rows() for stream in sorted(streams, key=lambda item: item.bucket_id)]
    yield from heapq.merge(*iterators, key=lambda item: item[0])


def _repack_parquet_rows(
    streams: dict[str, list[_InputStream]],
    writer: _LocalityPartWriter,
    *,
    bucket_width: int,
) -> dict[str, Any]:
    identity_digest = _MultisetDigest()
    payload_digest = _MultisetDigest()
    source_rows: Counter[str] = Counter()
    source_groups: Counter[str] = Counter()
    target_bucket_rows: Counter[int] = Counter()
    maximum_frames = 0

    for source in sorted(streams):
        previous_identity: str | None = None
        current_group: tuple[str, int] | None = None
        group_rows: list[tuple[str, dict[str, Any]]] = []

        def flush_group() -> None:
            nonlocal maximum_frames
            if not group_rows:
                return
            group_max = max(int(row["num_frames"]) for _, row in group_rows)
            if group_max <= 0:
                raise ValueError(f"Non-positive frame count in {source} group {current_group}")
            target_bucket = group_max // bucket_width
            upper_frames = (target_bucket + 1) * bucket_width
            if any(int(row["num_frames"]) > upper_frames for _, row in group_rows):
                raise ValueError("Stage211 row-group over-bucket exceeds its frame upper bound.")
            for identity, row in group_rows:
                writer.write(row, bucket_id=target_bucket)
                identity_digest.update(identity.encode("utf-8"))
                payload_digest.update(_canonical_row(row))
                source_rows[source] += 1
                target_bucket_rows[target_bucket] += 1
            source_groups[source] += 1
            maximum_frames = max(maximum_frames, group_max)

        for identity, row in _merge_source(streams[source]):
            if previous_identity is not None and identity <= previous_identity:
                raise ValueError(
                    "Stage211 merged Parquet identity is duplicated or unordered: "
                    f"identity={identity!r} previous={previous_identity!r}"
                )
            previous_identity = identity
            group = (str(row["shard_name"]), int(row["parquet_row_group"]))
            if current_group is not None and group != current_group:
                flush_group()
                group_rows.clear()
            current_group = group
            group_rows.append((identity, row))
        flush_group()

    return {
        "parquet_rows": sum(source_rows.values()),
        "parquet_row_groups": sum(source_groups.values()),
        "rows_by_source": dict(sorted(source_rows.items())),
        "row_groups_by_source": dict(sorted(source_groups.items())),
        "rows_by_target_bucket": {
            str(key): value for key, value in sorted(target_bucket_rows.items())
        },
        "maximum_num_frames": maximum_frames,
        "identity_multiset": identity_digest.as_dict(),
        "payload_multiset": payload_digest.as_dict(),
    }


def build_locality_inventory(
    *,
    source_inventory_path: Path,
    output_root: Path,
    validate_output: bool = True,
) -> dict[str, Any]:
    started = time.time()
    source_inventory_path = source_inventory_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to replace Stage211 locality output: {output_root}")
    source_inventory = _load_json(source_inventory_path)
    source_manifest_path = Path(str(source_inventory["bucket_manifest_path"])).resolve()
    source_manifest = _load_json(source_manifest_path)
    bucket_width = int(source_manifest["bucket_width"])
    entries_per_part = int(source_manifest["entries_per_part"])
    if bucket_width != 80 or entries_per_part != 100_000:
        raise ValueError("Stage211 source manifest bucketing contract changed.")

    staging = output_root.with_name(f"{output_root.name}.partial.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    manifest_root = staging / "webdataset_buckets_audio_text"
    manifest_root.mkdir(parents=True)
    try:
        parquet_streams, unchanged_parts = _input_parts(
            source_manifest,
            manifest_path=source_manifest_path,
        )
        writer = _LocalityPartWriter(manifest_root, entries_per_part=entries_per_part)
        locality = _repack_parquet_rows(
            parquet_streams,
            writer,
            bucket_width=bucket_width,
        )
        rewritten_parts, rewritten_counts = writer.finalize()

        buckets = []
        part_records = []
        for bucket_id in sorted(set(unchanged_parts) | set(rewritten_parts)):
            parts = sorted(
                unchanged_parts.get(bucket_id, []) + rewritten_parts.get(bucket_id, []),
                key=lambda part: (str(part.get("source_label") or ""), str(part["path"])),
            )
            rows = sum(int(part["num_samples"]) for part in parts)
            buckets.append({"bucket_id": bucket_id, "num_samples": rows, "parts": parts})
            part_records.extend(parts)
        selected_rows = int(source_inventory["selected_rows"])
        train_rows = sum(int(bucket["num_samples"]) for bucket in buckets)
        parquet_rows = int(locality["parquet_rows"])
        unchanged_rows = train_rows - parquet_rows
        expected_parquet_rows = sum(
            int(source_inventory["selected_counts_by_source"][source])
            for source in PARQUET_SOURCES
        )
        if (
            train_rows != selected_rows
            or parquet_rows != expected_parquet_rows
            or sum(rewritten_counts.values()) != parquet_rows
            or unchanged_rows != selected_rows - expected_parquet_rows
        ):
            raise ValueError(
                "Stage211 locality row accounting changed: "
                f"train={train_rows}/{selected_rows} parquet={parquet_rows}/{expected_parquet_rows}"
            )

        output_inventory_path = output_root / "supplemental_inventory.json"
        output_manifest_path = output_root / "webdataset_buckets_audio_text/manifest.json"
        manifest = {
            "version": 1,
            "root": "/",
            "source_length_index_path": str(output_inventory_path),
            "bucket_width": bucket_width,
            "entries_per_part": entries_per_part,
            "batching_policy": POLICY,
            "splits": {
                "train": {"num_samples": train_rows, "buckets": buckets},
                "eval": source_manifest["splits"]["eval"],
            },
        }
        staging_manifest_path = manifest_root / "manifest.json"
        staging_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256(staging_manifest_path)

        receipt_path = output_root / "parquet_locality_receipt.json"
        receipt = {
            "schema_version": 1,
            "artifact": RECEIPT_ARTIFACT,
            "complete": True,
            "policy": POLICY,
            "source_inventory_path": str(source_inventory_path),
            "source_inventory_sha256": _sha256(source_inventory_path),
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "output_manifest_path": str(output_manifest_path),
            "output_manifest_sha256": manifest_sha256,
            "selected_rows": selected_rows,
            "unchanged_rows": unchanged_rows,
            **locality,
            "created_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        }
        staging_receipt_path = staging / receipt_path.name
        staging_receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        inventory = dict(source_inventory)
        inventory["bucket_manifest_path"] = str(output_manifest_path)
        inventory["bucket_manifest_sha256"] = manifest_sha256
        inventory["part_records"] = part_records
        inventory["runtime_layout"] = {
            "policy": POLICY,
            "source_inventory_path": str(source_inventory_path),
            "source_inventory_sha256": receipt["source_inventory_sha256"],
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": receipt["source_manifest_sha256"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256(staging_receipt_path),
        }
        inventory["elapsed_seconds"] = time.time() - started
        (staging / output_inventory_path.name).write_text(
            json.dumps(inventory, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if validate_output:
        validated = validate_stage211_supplemental_inventory(
            output_inventory_path,
            require_training_ready=True,
            verify_part_sha256=True,
        )
        return validated["inventory"]
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repack Stage211 People's Speech metadata by Parquet row-group maximum length "
            "without changing admitted rows or audio."
        )
    )
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-deep-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = build_locality_inventory(
        source_inventory_path=args.source_inventory,
        output_root=args.output_root,
        validate_output=not args.skip_deep_validation,
    )
    print(
        f"inventory={args.output_root.expanduser().resolve() / 'supplemental_inventory.json'} "
        f"rows={int(inventory['selected_rows'])} "
        f"hours={float(inventory['selected_hours']):.6f} "
        f"policy={POLICY}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
