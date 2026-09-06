#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import wave
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator


DEFAULT_PEOPLES_ROOT = Path("/media/usbhd/MLCommons/peoples_speech")
DEFAULT_LLASO_ROOT = Path("/media/usbhd/LLaSO-Align")
DEFAULT_OUTPUT = Path.home() / "rwkvasr_data/stage211_supplemental_natural_v1"
DEFAULT_FIXED_EVAL_MANIFEST = (
    Path.home()
    / "rwkvasr_data/stage211_full_curriculum"
    / "stage179c_hard_dedup_audio_only_online_ctc"
    / "webdataset_buckets_audio_text/manifest_stage211_fixed_eval.json"
)
DEFAULT_STAGE179_GLOBAL_DEDUP_MANIFEST = Path(
    "/media/usbhd/training_data/asr/curriculum/"
    "stage22_stage21_soup_clean_repair_mix/stages/"
    "stage179_usbhd_dedup_online_ctc_alignment/"
    "stage179_usbhd_dedup_alignment_manifest.json"
)
DEFAULT_BUCKET_WIDTH = 80
DEFAULT_ENTRIES_PER_PART = 100_000
MIN_FRAMES = 20
FIXED_EVAL_ROWS = 256
EXPECTED_STAGE179_SOURCES = {
    "aishell3",
    "commonvoice_cn",
    "commonvoice_en",
    "cv22_en",
    "cv22_zh",
    "emilia_en",
    "emilia_zh",
    "gigaspeech",
    "librispeech",
    "wenetspeech",
}
EXPECTED_PEOPLE_TRAIN_FILES = {
    "peoples_speech_clean": 804,
    "peoples_speech_dirty": 3_140,
}
EXPECTED_ZIP_ARCHIVES = {
    "ljspeech": {"accepted": 3, "invalid": 0, "absent": 0},
    "vctk": {"accepted": 11, "invalid": 0, "absent": 0},
    "mls_english": {"accepted": 214, "invalid": 4, "absent": 0},
}


@dataclass(frozen=True, slots=True)
class ZipSourceSpec:
    source: str
    parts: tuple[int, ...]
    path_prefix: str
    sample_rate: int


ZIP_SOURCE_SPECS = (
    ZipSourceSpec("ljspeech", tuple(range(1, 4)), "LJSpeech/wavs/", 22_050),
    ZipSourceSpec("vctk", tuple(range(41, 52)), "VCTK-Corpus/wav48/", 48_000),
    ZipSourceSpec(
        "mls_english",
        tuple(range(317, 535)),
        "mls_english_opus/train_data_16000/",
        16_000,
    ),
)


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_key(source: str, identity: str) -> str:
    payload = f"{source}\0{identity}".encode("utf-8", errors="surrogatepass")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _dedupe_key(corpus: str, identity: str) -> bytes:
    payload = f"{corpus}\0{identity}".encode("utf-8", errors="surrogatepass")
    return hashlib.blake2b(payload, digest_size=16).digest()


def _path_record(path: Path, *, hash_archives: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(path) if hash_archives else None,
    }


def _resolved_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _load_fixed_eval_split(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_split = raw.get("splits", {}).get("eval")
    if not isinstance(eval_split, dict) or int(eval_split.get("num_samples", -1)) != FIXED_EVAL_ROWS:
        raise ValueError(
            f"Stage211 fixed-eval manifest must contain {FIXED_EVAL_ROWS} rows: {manifest_path}"
        )
    raw_buckets = eval_split.get("buckets")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError(f"Stage211 fixed-eval manifest has no eval buckets: {manifest_path}")
    buckets: list[dict[str, Any]] = []
    part_records: list[dict[str, Any]] = []
    counted_rows = 0
    for raw_bucket in raw_buckets:
        if not isinstance(raw_bucket, dict) or not isinstance(raw_bucket.get("parts"), list):
            raise ValueError(f"Stage211 fixed-eval manifest has an invalid bucket: {manifest_path}")
        parts: list[dict[str, Any]] = []
        bucket_rows = 0
        for raw_part in raw_bucket["parts"]:
            if not isinstance(raw_part, dict):
                raise ValueError(f"Stage211 fixed-eval manifest has an invalid part: {manifest_path}")
            part_path = _resolved_manifest_path(manifest_path, str(raw_part.get("path") or ""))
            rows = int(raw_part.get("num_samples", -1))
            if not part_path.is_file() or rows <= 0:
                raise ValueError(f"Stage211 fixed-eval part is unavailable or empty: {part_path}")
            part = {**raw_part, "path": str(part_path)}
            parts.append(part)
            part_records.append(
                {
                    "path": str(part_path),
                    "num_samples": rows,
                    "size_bytes": part_path.stat().st_size,
                    "sha256": _sha256(part_path),
                }
            )
            bucket_rows += rows
        if bucket_rows != int(raw_bucket.get("num_samples", -1)):
            raise ValueError(f"Stage211 fixed-eval bucket row count mismatch: {manifest_path}")
        counted_rows += bucket_rows
        buckets.append({**raw_bucket, "parts": parts})
    if counted_rows != FIXED_EVAL_ROWS:
        raise ValueError(
            f"Stage211 fixed-eval part rows mismatch: {counted_rows}/{FIXED_EVAL_ROWS}"
        )
    return (
        {"num_samples": FIXED_EVAL_ROWS, "buckets": buckets},
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "rows": FIXED_EVAL_ROWS,
            "parts": part_records,
        },
    )


def _load_stage179_source_binding(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = raw.get("inputs")
    if raw.get("version") != 1 or not isinstance(inputs, dict) or not inputs:
        raise ValueError(f"Invalid Stage179 global-dedup manifest: {manifest_path}")
    sources: set[str] = set()
    for source_input in inputs.values():
        if not isinstance(source_input, dict):
            raise ValueError(f"Invalid Stage179 source input: {manifest_path}")
        counts = source_input.get("accepted_counts_by_source")
        if not isinstance(counts, dict):
            raise ValueError(f"Stage179 source input lacks source counts: {manifest_path}")
        sources.update(str(source) for source in counts)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "total_unique_rows": int(raw.get("total_unique_audio_rows", -1)),
        "total_unique_hours": float(raw.get("total_unique_hours", float("nan"))),
        "sources": sorted(sources),
        "expected_source_set_match": sources == EXPECTED_STAGE179_SOURCES,
    }


def _num_frames_from_ms(duration_ms: int) -> int:
    return max(1, int(round(duration_ms / 10.0)))


def _people_rows(
    root: Path,
    *,
    hash_archives: bool,
    archive_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - production environment requirement
        raise RuntimeError("People's Speech indexing requires pyarrow.") from exc

    for subset in ("clean", "dirty"):
        source = f"peoples_speech_{subset}"
        paths = sorted((root / subset).glob("train-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No People's Speech train Parquet files under {root / subset}")
        for file_index, path in enumerate(paths, 1):
            archive_record = _path_record(path, hash_archives=hash_archives)
            archive_record.update(
                {
                    "storage_kind": "parquet",
                    "source": source,
                    "split": "train",
                    "status": "accepted",
                }
            )
            archive_records.append(archive_record)
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.metadata.num_row_groups):
                table = parquet.read_row_group(
                    row_group,
                    columns=["id", "duration_ms", "audio.path"],
                )
                ids = table.column("id")
                durations = table.column("duration_ms")
                audio_paths = table.column("audio")
                for row_index in range(len(table)):
                    identity = str(ids[row_index].as_py() or "").strip()
                    duration_ms = int(durations[row_index].as_py() or 0)
                    audio_path_value = audio_paths[row_index].as_py()
                    audio_member = (
                        str(audio_path_value.get("path") or "").strip()
                        if isinstance(audio_path_value, dict)
                        else ""
                    )
                    if not identity or not audio_member or duration_ms <= 0:
                        yield {
                            "_rejected": "invalid_people_metadata",
                            "source_dataset": source,
                        }
                        continue
                    num_frames = _num_frames_from_ms(duration_ms)
                    if num_frames < MIN_FRAMES:
                        yield {
                            "_rejected": "too_short",
                            "source_dataset": source,
                        }
                        continue
                    suffix = Path(audio_member).suffix.lower().lstrip(".")
                    if not suffix:
                        yield {
                            "_rejected": "missing_audio_format",
                            "source_dataset": source,
                        }
                        continue
                    key = _stable_key("peoples_speech", identity)
                    yield {
                        "shard_name": str(path.resolve()),
                        "key": key,
                        "utt_id": f"ps-{key}",
                        "split": "train",
                        "num_frames": num_frames,
                        "audio_member": audio_member,
                        "audio_format": suffix,
                        "audio_size": None,
                        "json_member": "",
                        "storage_kind": "parquet",
                        "parquet_row_group": row_group,
                        "parquet_row_index": row_index,
                        "parquet_id": identity,
                        "source_dataset": source,
                        "language": "en",
                        "sample_rate": 16_000,
                        "duration_ms": duration_ms,
                        "uses_text_labels": False,
                        "_supplemental_dedupe": _dedupe_key("peoples_speech", identity),
                    }
            if file_index % 100 == 0 or file_index == len(paths):
                print(
                    f"[stage211-supplemental] people source={source} "
                    f"files={file_index}/{len(paths)}",
                    flush=True,
                )


def _validate_zip_audio_format(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_sample_rate: int,
) -> None:
    with archive.open(info, "r") as source:
        payload = source.read()
    with wave.open(BytesIO(payload), "rb") as wav:
        if wav.getframerate() != expected_sample_rate:
            raise ValueError(
                f"Unexpected sample rate for {archive.filename}:{info.filename}: "
                f"expected {expected_sample_rate} got {wav.getframerate()}"
            )
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(
                f"Expected PCM16 mono WAV for {archive.filename}:{info.filename}; "
                f"channels={wav.getnchannels()} sample_width={wav.getsampwidth()}"
            )


def _zip_duration_ms(info: zipfile.ZipInfo, *, sample_rate: int) -> int:
    pcm_bytes = int(info.file_size) - 44
    if pcm_bytes <= 0:
        return 0
    return int(round(pcm_bytes / (sample_rate * 2) * 1000.0))


def _zip_rows(
    root: Path,
    *,
    specs: Iterable[ZipSourceSpec],
    hash_archives: bool,
    archive_records: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for spec in specs:
        accepted_archives = 0
        for part in spec.parts:
            path = root / f"LLaSO-Align-audio.part{part}.zip"
            if not path.exists():
                archive_records.append(
                    {
                        "path": str(path.resolve()),
                        "storage_kind": "zip",
                        "source": spec.source,
                        "split": "train",
                        "part": part,
                        "status": "absent",
                    }
                )
                continue
            record = _path_record(path, hash_archives=hash_archives)
            record.update(
                {
                    "storage_kind": "zip",
                    "source": spec.source,
                    "split": "train",
                    "part": part,
                }
            )
            try:
                archive = zipfile.ZipFile(path, "r")
            except zipfile.BadZipFile as exc:
                record.update({"status": "invalid", "error": str(exc)})
                archive_records.append(record)
                continue
            with archive:
                infos = [
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and info.filename.startswith(spec.path_prefix)
                    and info.filename.lower().endswith(".wav")
                ]
                if not infos:
                    record.update({"status": "invalid", "error": "no matching WAV members"})
                    archive_records.append(record)
                    continue
                _validate_zip_audio_format(
                    archive,
                    infos[0],
                    expected_sample_rate=spec.sample_rate,
                )
                accepted_archives += 1
                record.update({"status": "accepted", "audio_members": len(infos)})
                archive_records.append(record)
                for info in infos:
                    duration_ms = _zip_duration_ms(info, sample_rate=spec.sample_rate)
                    num_frames = _num_frames_from_ms(duration_ms)
                    if info.file_size <= 44 or num_frames < MIN_FRAMES:
                        yield {"_rejected": "too_short", "source_dataset": spec.source}
                        continue
                    key = _stable_key(spec.source, info.filename)
                    yield {
                        "shard_name": str(path.resolve()),
                        "key": key,
                        "utt_id": f"{spec.source}-{key}",
                        "split": "train",
                        "num_frames": num_frames,
                        "audio_member": info.filename,
                        "audio_format": "wav",
                        "audio_size": int(info.file_size),
                        "json_member": "",
                        "storage_kind": "zip",
                        "zip_crc32": int(info.CRC),
                        "zip_compress_type": int(info.compress_type),
                        "source_dataset": spec.source,
                        "language": "en",
                        "sample_rate": spec.sample_rate,
                        "duration_ms": duration_ms,
                        "uses_text_labels": False,
                        "_supplemental_dedupe": _dedupe_key(spec.source, info.filename),
                    }
        print(
            f"[stage211-supplemental] zip source={spec.source} "
            f"accepted_archives={accepted_archives}/{len(spec.parts)}",
            flush=True,
        )


@dataclass(slots=True)
class _PartState:
    path: Path
    relative_path: str
    handle: BinaryIO
    digest: Any
    num_samples: int = 0
    first_shard: str | None = None
    last_shard: str | None = None


class BucketManifestWriter:
    def __init__(self, root: Path, *, bucket_width: int, entries_per_part: int):
        self.root = root
        self.bucket_width = int(bucket_width)
        self.entries_per_part = int(entries_per_part)
        self._states: dict[tuple[int, str], _PartState] = {}
        self._next_part: Counter[tuple[int, str]] = Counter()
        self._parts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._bucket_counts: Counter[int] = Counter()

    def _open(self, bucket_id: int, source: str) -> _PartState:
        key = (bucket_id, source)
        part_index = self._next_part[key]
        self._next_part[key] += 1
        source_hex = source.encode("utf-8").hex()
        relative = (
            f"train/bucket_{bucket_id:04d}/source_{source_hex}/part_{part_index:06d}.jsonl"
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
                "num_samples": state.num_samples,
                "first_shard": state.first_shard,
                "last_shard": state.last_shard,
                "source_label": source,
                "size_bytes": state.path.stat().st_size,
                "sha256": state.digest.hexdigest(),
            }
        )

    def write(self, row: dict[str, Any]) -> None:
        source = str(row["source_dataset"])
        bucket_id = int(row["num_frames"]) // self.bucket_width
        key = (bucket_id, source)
        state = self._states.get(key)
        if state is None:
            state = self._open(bucket_id, source)
            self._states[key] = state
        payload_row = {k: v for k, v in row.items() if not k.startswith("_supplemental_")}
        payload = (json.dumps(payload_row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        state.handle.write(payload)
        state.digest.update(payload)
        state.num_samples += 1
        shard_name = str(row["shard_name"])
        state.first_shard = state.first_shard or shard_name
        state.last_shard = shard_name
        self._bucket_counts[bucket_id] += 1
        if state.num_samples >= self.entries_per_part:
            self._close(key)

    def finalize(self, *, source_length_index_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        for key in list(self._states):
            self._close(key)
        buckets = [
            {
                "bucket_id": bucket_id,
                "num_samples": self._bucket_counts[bucket_id],
                "parts": self._parts[bucket_id],
            }
            for bucket_id in sorted(self._parts)
        ]
        manifest = {
            "version": 1,
            "root": "/",
            "source_length_index_path": str(source_length_index_path),
            "bucket_width": self.bucket_width,
            "entries_per_part": self.entries_per_part,
            "splits": {
                "train": {
                    "num_samples": sum(self._bucket_counts.values()),
                    "buckets": buckets,
                }
            },
        }
        part_records = [part for bucket in buckets for part in bucket["parts"]]
        return manifest, part_records


def build_manifest(
    *,
    peoples_root: Path,
    llaso_root: Path,
    output: Path,
    bucket_width: int,
    entries_per_part: int,
    hash_archives: bool,
    require_production_layout: bool = True,
    fixed_eval_manifest_path: Path = DEFAULT_FIXED_EVAL_MANIFEST,
    stage179_global_dedup_manifest_path: Path = DEFAULT_STAGE179_GLOBAL_DEDUP_MANIFEST,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing supplemental manifest: {output}")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise FileExistsError(
            "Refusing to delete an interrupted Stage211 supplemental build; "
            f"inspect and recover or remove it explicitly: {staging}"
        )
    staging.mkdir(parents=True)
    archive_records: list[dict[str, Any]] = []
    writer = BucketManifestWriter(
        staging / "webdataset_buckets_audio_text",
        bucket_width=bucket_width,
        entries_per_part=entries_per_part,
    )
    source_counts: Counter[str] = Counter()
    source_frames: Counter[str] = Counter()
    source_duration_ms: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    seen: set[bytes] = set()
    started = time.time()
    fixed_eval_split, fixed_eval_binding = _load_fixed_eval_split(fixed_eval_manifest_path)
    stage179_binding = _load_stage179_source_binding(stage179_global_dedup_manifest_path)

    iterators = (
        _people_rows(
            peoples_root,
            hash_archives=hash_archives,
            archive_records=archive_records,
        ),
        _zip_rows(
            llaso_root,
            specs=ZIP_SOURCE_SPECS,
            hash_archives=hash_archives,
            archive_records=archive_records,
        ),
    )
    accepted = 0
    for rows in iterators:
        for row in rows:
            rejected = row.get("_rejected")
            if rejected:
                rejection_counts[str(rejected)] += 1
                continue
            dedupe = row.pop("_supplemental_dedupe")
            if dedupe in seen:
                duplicate_counts[str(row["source_dataset"])] += 1
                continue
            seen.add(dedupe)
            writer.write(row)
            source = str(row["source_dataset"])
            source_counts[source] += 1
            source_frames[source] += int(row["num_frames"])
            source_duration_ms[source] += int(row["duration_ms"])
            accepted += 1
            if accepted % 1_000_000 == 0:
                print(
                    f"[stage211-supplemental] accepted={accepted} "
                    f"hours={sum(source_frames.values()) / 360000.0:.3f}",
                    flush=True,
                )

    inventory_path = staging / "supplemental_inventory.json"
    manifest, part_records = writer.finalize(
        source_length_index_path=output / "supplemental_inventory.json"
    )
    manifest["splits"]["eval"] = fixed_eval_split
    manifest_path = staging / "webdataset_buckets_audio_text/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    layout_errors: list[str] = []
    if require_production_layout:
        if not stage179_binding["expected_source_set_match"]:
            layout_errors.append("Stage179 source set differs from the production contract")
        archive_status: Counter[tuple[str, str]] = Counter(
            (str(record.get("source")), str(record.get("status")))
            for record in archive_records
        )
        for source, expected_files in EXPECTED_PEOPLE_TRAIN_FILES.items():
            actual = archive_status[(source, "accepted")]
            if actual != expected_files:
                layout_errors.append(
                    f"{source} accepted archives expected={expected_files} actual={actual}"
                )
        for source, expected_statuses in EXPECTED_ZIP_ARCHIVES.items():
            for status, expected in expected_statuses.items():
                actual = archive_status[(source, status)]
                if actual != expected:
                    layout_errors.append(
                        f"{source} {status} archives expected={expected} actual={actual}"
                    )
        for record in archive_records:
            if record.get("status") == "accepted" and hash_archives and not record.get("sha256"):
                layout_errors.append(f"accepted archive has no SHA-256: {record.get('path')}")
    supplemental_sources = set(source_counts)
    stage179_sources = set(stage179_binding["sources"])
    source_sets_disjoint = supplemental_sources.isdisjoint(stage179_sources)
    if require_production_layout and not source_sets_disjoint:
        layout_errors.append(
            "supplemental source labels overlap Stage179: "
            + ",".join(sorted(supplemental_sources & stage179_sources))
        )
    training_ready = bool(
        hash_archives
        and require_production_layout
        and not layout_errors
        and source_sets_disjoint
    )
    inventory = {
        "schema_version": 1,
        "artifact": "stage211_supplemental_natural_inventory",
        "complete": True,
        "training_ready": training_ready,
        "hash_archives": bool(hash_archives),
        "require_production_layout": bool(require_production_layout),
        "layout_errors": layout_errors,
        "language": "en",
        "uses_text_labels": False,
        "storage_kinds": ["parquet", "zip"],
        "fixed_eval": fixed_eval_binding,
        "cross_pool_dedupe": {
            "mode": "source_identity_plus_known_corpus_exclusion",
            "content_fingerprint_complete": False,
            "stage179": stage179_binding,
            "supplemental_sources": sorted(supplemental_sources),
            "source_sets_disjoint": source_sets_disjoint,
            "known_overlap_exclusions": ["llaso_gigaspeech", "llaso_librispeech"],
        },
        "excluded_sources": {
            "llaso_librispeech": "already present in Stage179 and public-overlap risk",
            "llaso_gigaspeech": "already present in Stage179",
            "synthetic_tts": "augmentation only, not core natural-ASR coverage",
            "raw_video_or_vocals": "not admitted before speech segmentation audit",
            "people_test_validation": "evaluation splits excluded from training",
            "mls_dev_test": "evaluation splits excluded from training",
        },
        "dedupe": {
            "algorithm": "blake2b16(corpus + NUL + source_identity)",
            "accepted_unique_rows": accepted,
            "duplicates_by_source": dict(sorted(duplicate_counts.items())),
        },
        "selected_rows": accepted,
        "selected_hours": sum(source_duration_ms.values()) / 3_600_000.0,
        "selected_frame_derived_hours": sum(source_frames.values()) / 360000.0,
        "selected_counts_by_source": dict(sorted(source_counts.items())),
        "selected_hours_by_source": {
            key: value / 3_600_000.0 for key, value in sorted(source_duration_ms.items())
        },
        "rejections": dict(sorted(rejection_counts.items())),
        "archive_records": archive_records,
        "bucket_manifest_path": str(
            output / "webdataset_buckets_audio_text/manifest.json"
        ),
        "bucket_manifest_sha256": _sha256(manifest_path),
        "part_records": part_records,
        "elapsed_seconds": time.time() - started,
    }
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    inventory["inventory_sha256"] = _sha256(inventory_path)
    os.replace(staging, output)
    print(
        f"[stage211-supplemental] complete rows={accepted} "
        f"hours={inventory['selected_hours']:.6f} output={output}",
        flush=True,
    )
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Stage211 supplemental natural-audio bucket manifest."
    )
    parser.add_argument("--peoples-root", type=Path, default=DEFAULT_PEOPLES_ROOT)
    parser.add_argument("--llaso-root", type=Path, default=DEFAULT_LLASO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bucket-width", type=int, default=DEFAULT_BUCKET_WIDTH)
    parser.add_argument("--entries-per-part", type=int, default=DEFAULT_ENTRIES_PER_PART)
    parser.add_argument(
        "--fixed-eval-manifest",
        type=Path,
        default=DEFAULT_FIXED_EVAL_MANIFEST,
    )
    parser.add_argument(
        "--stage179-global-dedup-manifest",
        type=Path,
        default=DEFAULT_STAGE179_GLOBAL_DEDUP_MANIFEST,
    )
    parser.add_argument(
        "--hash-archives",
        action="store_true",
        help="Hash every source archive; required for training_ready=true.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_manifest(
        peoples_root=args.peoples_root,
        llaso_root=args.llaso_root,
        output=args.output,
        bucket_width=args.bucket_width,
        entries_per_part=args.entries_per_part,
        hash_archives=args.hash_archives,
        require_production_layout=True,
        fixed_eval_manifest_path=args.fixed_eval_manifest,
        stage179_global_dedup_manifest_path=args.stage179_global_dedup_manifest,
    )


if __name__ == "__main__":
    main()
