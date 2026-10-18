#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.validate_stage211_retention_replay import (
        PARQUET_SOURCES,
        RUNTIME_LAYOUT_ARTIFACT,
        RUNTIME_LAYOUT_POLICY,
        validate_retention_replay,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from validate_stage211_retention_replay import (  # type: ignore[no-redef]
        PARQUET_SOURCES,
        RUNTIME_LAYOUT_ARTIFACT,
        RUNTIME_LAYOUT_POLICY,
        validate_retention_replay,
    )


DEFAULT_SELECTION_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "retention_replay_v3"
    / "receipt.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "retention_replay_v4_locality"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_row(row: Mapping[str, Any]) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_part(raw_path: str, *, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(str(path))
    return path


def _row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("key") or row.get("id") or "")


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
    source = str(row.get("source_dataset") or "")
    shard = str(row.get("shard_name") or "")
    row_group = row.get("parquet_row_group")
    row_index = row.get("parquet_row_index")
    key = _row_key(row)
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
        raise ValueError(f"Invalid Stage211 replay Parquet row: {row!r}")
    return source, shard, row_group, row_index, key


def _is_parquet_label(value: Any) -> bool:
    return str(value or "").rsplit(":", 1)[-1] in PARQUET_SOURCES


@dataclass
class _PartState:
    path: Path
    relative_path: str
    source_label: str
    bucket_id: int
    handle: Any
    digest: Any
    rows: int = 0
    first_shard: str | None = None
    last_shard: str | None = None


class _Writer:
    def __init__(
        self,
        *,
        staging_manifest_root: Path,
        final_manifest_root: Path,
        entries_per_part: int,
    ):
        self.staging_manifest_root = staging_manifest_root
        self.final_manifest_root = final_manifest_root
        self.entries_per_part = int(entries_per_part)
        self.states: dict[tuple[int, str], _PartState] = {}
        self.next_parts: Counter[tuple[int, str]] = Counter()
        self.manifest_parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.receipt_parts: list[dict[str, Any]] = []

    def _open(self, *, bucket_id: int, source_label: str) -> _PartState:
        key = (bucket_id, source_label)
        index = self.next_parts[key]
        self.next_parts[key] += 1
        source_hex = source_label.encode("utf-8").hex()
        relative = (
            f"train/bucket_{bucket_id:04d}/source_{source_hex}/part_{index:06d}.jsonl"
        )
        path = self.staging_manifest_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return _PartState(
            path=path,
            relative_path=relative,
            source_label=source_label,
            bucket_id=bucket_id,
            handle=path.open("wb"),
            digest=hashlib.sha256(),
        )

    def _close(self, key: tuple[int, str]) -> None:
        state = self.states.pop(key, None)
        if state is None:
            return
        state.handle.close()
        manifest_record = {
            "path": state.relative_path,
            "num_samples": state.rows,
            "first_shard": state.first_shard,
            "last_shard": state.last_shard,
            "source_label": state.source_label,
            "size_bytes": state.path.stat().st_size,
            "sha256": state.digest.hexdigest(),
        }
        self.manifest_parts[state.bucket_id].append(manifest_record)
        self.receipt_parts.append(
            {
                **manifest_record,
                "path": str(self.final_manifest_root / state.relative_path),
                "bucket_id": state.bucket_id,
            }
        )

    def write_group(
        self,
        rows: list[tuple[tuple[str, str, int, int, str], dict[str, Any]]],
        *,
        bucket_id: int,
        source_label: str,
    ) -> None:
        if not rows or len(rows) > self.entries_per_part:
            raise ValueError("Stage211 replay row group does not fit one metadata part.")
        key = (bucket_id, source_label)
        state = self.states.get(key)
        if state is not None and state.rows + len(rows) > self.entries_per_part:
            self._close(key)
            state = None
        if state is None:
            state = self._open(bucket_id=bucket_id, source_label=source_label)
            self.states[key] = state
        for _, row in rows:
            payload = _canonical_row(row) + b"\n"
            state.handle.write(payload)
            state.digest.update(payload)
            state.rows += 1
            shard = str(row["shard_name"])
            state.first_shard = state.first_shard or shard
            state.last_shard = shard

    def finalize(self) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
        for key in list(self.states):
            self._close(key)
        return dict(self.manifest_parts), sorted(
            self.receipt_parts,
            key=lambda row: (int(row["bucket_id"]), str(row["source_label"]), str(row["path"])),
        )


def build_runtime_layout(
    *,
    selection_receipt_path: Path,
    output_root: Path,
    validate_output: bool = True,
) -> dict[str, Any]:
    started = time.time()
    selection_receipt_path = selection_receipt_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    receipt_path = output_root / "receipt.json"
    if receipt_path.is_file():
        if not validate_output:
            return _load_json(receipt_path)
        return validate_retention_replay(receipt_path)
    if output_root.exists():
        raise FileExistsError(f"Incomplete Stage211 replay locality output exists: {output_root}")
    selection = validate_retention_replay(selection_receipt_path)
    if selection.get("runtime_layout") is not None:
        raise ValueError("Stage211 replay locality source must be an immutable selection receipt.")
    source_manifest_path = Path(str(selection["manifest_path"])).resolve()
    source_manifest = _load_json(source_manifest_path)
    bucket_width = int(source_manifest.get("bucket_width", -1))
    entries_per_part = int(source_manifest.get("entries_per_part", -1))
    if bucket_width != 80 or entries_per_part <= 0:
        raise ValueError("Stage211 replay source bucketing contract changed.")

    train = (source_manifest.get("splits") or {}).get("train")
    eval_split = (source_manifest.get("splits") or {}).get("eval")
    if not isinstance(train, dict) or not isinstance(eval_split, dict):
        raise ValueError("Stage211 replay source manifest lacks train/eval splits.")
    unchanged_parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    groups: dict[
        tuple[str, str, str, int],
        list[tuple[tuple[str, str, int, int, str], dict[str, Any]]],
    ] = defaultdict(list)
    seen_keys: set[str] = set()
    source_parquet_parts = 0
    source_rows = 0
    parquet_rows = 0
    source_adjacent_same_row_group = 0
    previous_source_group: tuple[str, str, int] | None = None
    rows_by_source: Counter[str] = Counter()
    row_groups_by_source: Counter[str] = Counter()
    for bucket in train.get("buckets") or []:
        bucket_id = int(bucket["bucket_id"])
        observed_bucket_rows = 0
        for part in bucket.get("parts") or []:
            part = dict(part)
            part_path = _resolve_part(str(part.get("path") or ""), manifest_path=source_manifest_path)
            declared_rows = int(part.get("num_samples", -1))
            if declared_rows <= 0:
                raise ValueError(f"Stage211 replay source part count is invalid: {part_path}")
            observed_bucket_rows += declared_rows
            source_rows += declared_rows
            source_label = str(part.get("source_label") or "")
            if not _is_parquet_label(source_label):
                unchanged_parts[bucket_id].append(part)
                continue
            source_parquet_parts += 1
            observed_rows = 0
            with part_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    observed_rows += 1
                    row = json.loads(line)
                    identity = _identity(row)
                    key = identity[-1]
                    expected_label = f"{row.get('_stage211_replay_cell')}:{identity[0]}"
                    if source_label != expected_label or key in seen_keys:
                        raise ValueError(f"Stage211 replay source Parquet provenance changed: {key}")
                    seen_keys.add(key)
                    group_key = (source_label, identity[0], identity[1], identity[2])
                    groups[group_key].append((identity, row))
                    logical_group = identity[:3]
                    if logical_group == previous_source_group:
                        source_adjacent_same_row_group += 1
                    previous_source_group = logical_group
                    parquet_rows += 1
                    rows_by_source[identity[0]] += 1
            if observed_rows != declared_rows:
                raise ValueError(f"Stage211 replay source part count changed: {part_path}")
        if observed_bucket_rows != int(bucket.get("num_samples", -1)):
            raise ValueError("Stage211 replay source bucket count changed.")
    if source_rows != int(selection.get("validated_unique_keys", -1)):
        raise ValueError("Stage211 replay source train count changed.")

    staging = output_root.with_name(f"{output_root.name}.partial.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging_manifest_root = staging / "webdataset_buckets_audio_text"
    final_manifest_root = output_root / "webdataset_buckets_audio_text"
    staging_manifest_root.mkdir(parents=True)
    try:
        writer = _Writer(
            staging_manifest_root=staging_manifest_root,
            final_manifest_root=final_manifest_root,
            entries_per_part=entries_per_part,
        )
        rows_by_runtime_bucket: Counter[int] = Counter()
        maximum_num_frames = 0
        for group_key in sorted(groups):
            source_label, source_name, _, _ = group_key
            rows = sorted(groups[group_key], key=lambda item: item[0])
            group_max_frames = max(int(row["num_frames"]) for _, row in rows)
            if group_max_frames <= 0:
                raise ValueError(f"Stage211 replay row group has invalid frames: {group_key}")
            runtime_bucket = group_max_frames // bucket_width
            writer.write_group(
                rows,
                bucket_id=runtime_bucket,
                source_label=source_label,
            )
            row_groups_by_source[source_name] += 1
            rows_by_runtime_bucket[runtime_bucket] += len(rows)
            maximum_num_frames = max(maximum_num_frames, group_max_frames)
        rewritten_parts, rewritten_receipt_parts = writer.finalize()

        buckets = []
        for bucket_id in sorted(set(unchanged_parts) | set(rewritten_parts)):
            parts = sorted(
                unchanged_parts.get(bucket_id, []) + rewritten_parts.get(bucket_id, []),
                key=lambda part: (str(part.get("source_label") or ""), str(part["path"])),
            )
            buckets.append(
                {
                    "bucket_id": bucket_id,
                    "num_samples": sum(int(part["num_samples"]) for part in parts),
                    "parts": parts,
                }
            )
        runtime_rows = sum(int(bucket["num_samples"]) for bucket in buckets)
        if runtime_rows != source_rows or sum(rows_by_runtime_bucket.values()) != parquet_rows:
            raise ValueError("Stage211 replay runtime row accounting changed.")
        runtime_manifest_path = final_manifest_root / "manifest.json"
        runtime_manifest = {
            "version": int(source_manifest.get("version", 1)),
            "root": source_manifest.get("root", "/"),
            "source_length_index_path": str(receipt_path),
            "bucket_width": bucket_width,
            "entries_per_part": entries_per_part,
            "batching_policy": RUNTIME_LAYOUT_POLICY,
            "source_selection_manifest_path": str(source_manifest_path),
            "source_selection_manifest_sha256": _sha256(source_manifest_path),
            "splits": {
                "train": {"num_samples": runtime_rows, "buckets": buckets},
                "eval": eval_split,
            },
        }
        staging_manifest_path = staging_manifest_root / "manifest.json"
        staging_manifest_path.write_text(
            json.dumps(runtime_manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        eval_rows = int(eval_split.get("num_samples", 0))
        receipt = {
            "schema_version": 1,
            "pipeline": "stage211",
            "artifact": RUNTIME_LAYOUT_ARTIFACT,
            "complete": True,
            "policy": RUNTIME_LAYOUT_POLICY,
            "selection_receipt": {
                "path": str(selection_receipt_path),
                "sha256": _sha256(selection_receipt_path),
            },
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "runtime_manifest_path": str(runtime_manifest_path),
            "runtime_manifest_sha256": _sha256(staging_manifest_path),
            "selected_rows": source_rows,
            "eval_rows": eval_rows,
            "parquet_rows": parquet_rows,
            "parquet_row_groups": len(groups),
            "unchanged_rows": source_rows - parquet_rows,
            "source_parquet_parts": source_parquet_parts,
            "runtime_parquet_parts": len(rewritten_receipt_parts),
            "rows_by_source": dict(sorted(rows_by_source.items())),
            "row_groups_by_source": dict(sorted(row_groups_by_source.items())),
            "rows_by_runtime_bucket": {
                str(bucket): rows for bucket, rows in sorted(rows_by_runtime_bucket.items())
            },
            "maximum_num_frames": maximum_num_frames,
            "source_adjacent_same_row_group": source_adjacent_same_row_group,
            "runtime_adjacent_same_row_group": parquet_rows - len(groups),
            "rewritten_parts": rewritten_receipt_parts,
            "created_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        }
        (staging / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.replace(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if validate_output:
        return validate_retention_replay(receipt_path)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repack the immutable Stage211 retention selection into a Parquet "
            "row-group-local runtime manifest without changing selected rows."
        )
    )
    parser.add_argument("--selection-receipt", type=Path, default=DEFAULT_SELECTION_RECEIPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-deep-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay = build_runtime_layout(
        selection_receipt_path=args.selection_receipt,
        output_root=args.output_root,
        validate_output=not args.skip_deep_validation,
    )
    layout = replay.get("runtime_layout", replay)
    print(
        "stage211_retention_replay_locality "
        f"receipt={args.output_root.expanduser().resolve() / 'receipt.json'} "
        f"rows={int(layout['selected_rows'])} parquet_rows={int(layout['parquet_rows'])} "
        f"row_groups={int(layout['parquet_row_groups'])} policy={RUNTIME_LAYOUT_POLICY}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
