#!/usr/bin/env python3
"""Filter voxbox-style ASR WebDataset shards into a train-only curriculum root."""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


AUDIO_SUFFIXES = {"wav", "mp3", "flac"}
DEFAULT_EXCLUDED_SPLITS = {"test", "dev", "valid", "validation", "eval"}


@dataclass
class PendingSample:
    audio_bytes: bytes | None = None
    audio_suffix: str | None = None
    metadata: dict | None = None


class ShardWriter:
    def __init__(self, output_root: Path, target_bytes: int, prefix: str) -> None:
        self.output_root = output_root
        self.target_bytes = target_bytes
        self.prefix = _sanitize_key(prefix)
        self.shard_idx = 0
        self.sample_count = 0
        self.current_bytes = 0
        self.tar: tarfile.TarFile | None = None

    def close(self) -> None:
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def _open_next(self) -> None:
        self.close()
        shard_path = self.output_root / f"{self.prefix}_{self.shard_idx:06d}.tar"
        self.tar = tarfile.open(shard_path, mode="w")
        self.current_bytes = 0
        self.shard_idx += 1

    def write_sample(self, key: str, audio_suffix: str, audio_bytes: bytes, metadata: dict) -> None:
        if self.tar is None or (self.current_bytes >= self.target_bytes and self.sample_count > 0):
            self._open_next()
        assert self.tar is not None

        json_bytes = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._add_bytes(f"{key}.{audio_suffix}", audio_bytes)
        self._add_bytes(f"{key}.json", json_bytes)
        self.current_bytes += len(audio_bytes) + len(json_bytes)
        self.sample_count += 1

    def _add_bytes(self, name: str, payload: bytes) -> None:
        assert self.tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mtime = 0
        self.tar.addfile(info, io.BytesIO(payload))


def _parse_source(value: str) -> tuple[str, Path]:
    if ":" not in value:
        path = Path(value)
        return _sanitize_key(path.name), path
    label, path = value.split(":", 1)
    return _sanitize_key(label), Path(path)


def _sanitize_key(value: str) -> str:
    value = value.strip().replace("/", "_")
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    value = value.strip("._-")
    return value or "sample"


def _member_key_and_suffix(name: str) -> tuple[str, str] | None:
    base = Path(name).name
    if "." not in base:
        return None
    key, suffix = base.rsplit(".", 1)
    suffix = suffix.lower()
    if suffix != "json" and suffix not in AUDIO_SUFFIXES:
        return None
    return key, suffix


def _read_member(handle: BinaryIO | None) -> bytes:
    if handle is None:
        return b""
    with handle:
        return handle.read()


def _split_allowed(metadata: dict, excluded_splits: set[str]) -> bool:
    split = str(metadata.get("split", "")).strip().lower()
    if split in excluded_splits:
        return False
    text = str(metadata.get("text", "")).strip()
    return bool(text)


def _duration_allowed(metadata: dict, min_duration: float, max_duration: float, min_speech_ratio: float) -> bool:
    duration = metadata.get("duration")
    if duration is None:
        return True
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        return True
    if duration_value < min_duration or duration_value > max_duration:
        return False
    speech_duration = metadata.get("speech_duration")
    if speech_duration is None or duration_value <= 0.0:
        return True
    try:
        speech_ratio = float(speech_duration) / duration_value
    except (TypeError, ValueError):
        return True
    return speech_ratio >= min_speech_ratio


def _try_emit(
    *,
    source_label: str,
    source_shard: str,
    sample_key: str,
    pending: PendingSample,
    writer: ShardWriter,
    excluded_splits: set[str],
    min_duration: float,
    max_duration: float,
    min_speech_ratio: float,
    source_counts: dict[str, int],
) -> bool:
    if pending.audio_bytes is None or pending.audio_suffix is None or pending.metadata is None:
        return False
    metadata = dict(pending.metadata)
    if not _split_allowed(metadata, excluded_splits):
        return True
    if not _duration_allowed(metadata, min_duration, max_duration, min_speech_ratio):
        return True

    original_id = str(metadata.get("index") or metadata.get("id") or sample_key)
    output_key = _sanitize_key(f"{source_label}_{original_id}")
    metadata["id"] = original_id
    metadata["source_dataset"] = source_label
    metadata["source_shard"] = source_shard
    metadata["source_key"] = sample_key
    metadata["source_split"] = metadata.get("split")
    writer.write_sample(output_key, pending.audio_suffix, pending.audio_bytes, metadata)
    source_counts[source_label] = source_counts.get(source_label, 0) + 1
    return True


def filter_source(
    *,
    label: str,
    root: Path,
    writer: ShardWriter,
    excluded_splits: set[str],
    min_duration: float,
    max_duration: float,
    min_speech_ratio: float,
    source_counts: dict[str, int],
) -> None:
    tar_paths = sorted(root.glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No .tar shards found under {root}")
    for shard_idx, tar_path in enumerate(tar_paths, start=1):
        pending: dict[str, PendingSample] = {}
        seen_json = 0
        kept_before = source_counts.get(label, 0)
        print(f"[rwkvasr] filtering {label} shard {shard_idx}/{len(tar_paths)}: {tar_path}", flush=True)
        with tarfile.open(tar_path, mode="r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parsed = _member_key_and_suffix(member.name)
                if parsed is None:
                    continue
                sample_key, suffix = parsed
                item = pending.setdefault(sample_key, PendingSample())
                if suffix == "json":
                    payload = _read_member(tar.extractfile(member))
                    item.metadata = json.loads(payload.decode("utf-8"))
                    seen_json += 1
                else:
                    item.audio_bytes = _read_member(tar.extractfile(member))
                    item.audio_suffix = suffix
                emitted_or_skipped = _try_emit(
                    source_label=label,
                    source_shard=tar_path.name,
                    sample_key=sample_key,
                    pending=item,
                    writer=writer,
                    excluded_splits=excluded_splits,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_speech_ratio=min_speech_ratio,
                    source_counts=source_counts,
                )
                if emitted_or_skipped:
                    pending.pop(sample_key, None)
        kept_after = source_counts.get(label, 0)
        print(
            f"[rwkvasr] finished {label} shard={tar_path.name} json={seen_json} "
            f"kept={kept_after - kept_before} total_kept={kept_after}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="LABEL:PATH input root; may be repeated.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exclude-split", action="append", default=sorted(DEFAULT_EXCLUDED_SPLITS))
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--min-speech-ratio", type=float, default=0.25)
    parser.add_argument("--target-shard-size-mb", type=float, default=1024.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    marker = output_root / ".filter_complete.json"
    if marker.exists() and not args.overwrite:
        print(f"[rwkvasr] filtered curriculum root already exists: {marker}")
        return
    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in output_root.glob("*.tar"):
            old.unlink()
        if marker.exists():
            marker.unlink()

    source_counts: dict[str, int] = {}
    excluded_splits = {str(value).strip().lower() for value in args.exclude_split}
    for source in args.source:
        label, root = _parse_source(source)
        writer = ShardWriter(output_root, target_bytes=int(args.target_shard_size_mb * 1024 * 1024), prefix=label)
        try:
            filter_source(
                label=label,
                root=root,
                writer=writer,
                excluded_splits=excluded_splits,
                min_duration=float(args.min_duration),
                max_duration=float(args.max_duration),
                min_speech_ratio=float(args.min_speech_ratio),
                source_counts=source_counts,
            )
        finally:
            writer.close()

    summary = {
        "sources": source_counts,
        "num_samples": sum(source_counts.values()),
        "num_shards": len(list(output_root.glob("*.tar"))),
        "excluded_splits": sorted(excluded_splits),
        "min_duration": float(args.min_duration),
        "max_duration": float(args.max_duration),
        "min_speech_ratio": float(args.min_speech_ratio),
    }
    marker.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[rwkvasr] filter complete: {json.dumps(summary, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
