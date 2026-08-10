from __future__ import annotations

import json
import random
import tarfile
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from torch.utils.data import DataLoader, Dataset, Sampler

from .manifest import FeatureCollator, TokenizerLike, WenetFbankFeatureExtractor
from .webdataset import (
    WebDatasetConfig,
    decode_webdataset_sample,
    log_webdataset_decode_skip,
    preload_decoder_ctc_draft_cache,
)
from .webdataset_common import AUDIO_SUFFIXES
from .webdataset_index import StableHashSplitConfig, assign_split, resolve_sample_id

MAX_IN_MEMORY_LENGTH_INDEX_BYTES = 1 << 30
SUPPORTED_LENGTH_INDEX_STORAGE_KINDS = frozenset({"tar", "zip", "parquet"})


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class WebDatasetLengthEntry:
    shard_name: str
    key: str
    utt_id: str
    split: str
    num_frames: int
    audio_member: str
    audio_format: str
    json_member: str
    storage_kind: str = "tar"
    num_text_tokens: int | None = None
    num_text_chars: int | None = None
    text_bytes: int | None = None
    audio_offset: int | None = None
    audio_size: int | None = None
    json_offset: int | None = None
    json_size: int | None = None
    zip_crc32: int | None = None
    zip_compress_type: int | None = None
    parquet_row_group: int | None = None
    parquet_row_index: int | None = None
    raw: dict[str, Any] | None = None

    @property
    def wav_member(self) -> str:
        return self.audio_member


def _member_audio_format(member_name: str) -> str:
    return Path(member_name).suffix.lower().lstrip(".")


def _shard_source_label(shard_name: str | None) -> str:
    if not shard_name:
        return "unknown"
    if shard_name.startswith("GSXL"):
        return "gigaspeech"
    if shard_name.startswith("WSL"):
        return "wenetspeech"
    return shard_name.split("-", 1)[0].lower() or "unknown"


def resolve_webdataset_length_index_path(shard_root: str | Path, index_path: str | Path | None = None) -> Path:
    if index_path is not None:
        return Path(index_path)
    shard_root = Path(shard_root)
    if shard_root.is_dir():
        return shard_root / "webdataset_lengths.jsonl"
    return shard_root.with_suffix(".lengths.jsonl")


def resolve_webdataset_length_summary_path(
    shard_root: str | Path,
    summary_path: str | Path | None = None,
) -> Path:
    if summary_path is not None:
        return Path(summary_path)
    index_path = resolve_webdataset_length_index_path(shard_root)
    return index_path.with_suffix(".summary.json")


def format_num_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TiB"


def can_load_webdataset_length_index_in_memory(
    index_path: str | Path,
    *,
    max_bytes: int = MAX_IN_MEMORY_LENGTH_INDEX_BYTES,
) -> bool:
    return Path(index_path).stat().st_size <= int(max_bytes)


def infer_num_frames_from_metadata(metadata: dict[str, Any]) -> int:
    if "num_frames" in metadata:
        value = int(metadata["num_frames"])
        if value > 0:
            return value

    duration_sec: float | None = None
    if "begin_time" in metadata and "end_time" in metadata:
        duration_sec = float(metadata["end_time"]) - float(metadata["begin_time"])
    elif "duration" in metadata:
        duration_sec = float(metadata["duration"])
    elif "num_samples" in metadata and "sample_rate" in metadata:
        duration_sec = float(metadata["num_samples"]) / float(metadata["sample_rate"])

    if duration_sec is None or duration_sec <= 0.0:
        raise ValueError("Unable to infer acoustic frame length from metadata.")
    return max(1, int(round(duration_sec * 100.0)))


def inspect_webdataset_lengths(
    shard_root: str | Path,
    *,
    shard_pattern: str = "*.tar",
    split_config: StableHashSplitConfig | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    shard_root = Path(shard_root)
    split_config = split_config or StableHashSplitConfig()
    shards = [shard_root] if shard_root.is_file() else sorted(shard_root.glob(shard_pattern))
    if not shards:
        raise FileNotFoundError(f"No shard files matching {shard_pattern!r} under {shard_root}")

    output_file = resolve_webdataset_length_index_path(shard_root, output_path)
    summary_file = resolve_webdataset_length_summary_path(shard_root, summary_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {split_config.train_name: 0, split_config.eval_name: 0}
    bucket_counts: dict[str, int] = {}
    total_samples = 0
    min_frames: int | None = None
    max_frames = 0
    start_time = time.monotonic()
    _log(f"Inspecting WebDataset lengths under {shard_root} with {len(shards)} shard(s).")

    with output_file.open("w", encoding="utf-8") as handle:
        for shard_idx, shard_path in enumerate(shards, 1):
            pending: dict[str, dict[str, Any]] = {}
            with tarfile.open(shard_path, "r") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    member_name = member.name
                    basename = Path(member_name).name
                    if "." not in basename:
                        continue
                    key, suffix = basename.rsplit(".", 1)
                    suffix = suffix.lower()
                    if suffix != "json" and suffix not in AUDIO_SUFFIXES:
                        continue

                    sample = pending.setdefault(key, {})
                    if suffix != "json":
                        sample["audio_member"] = member_name
                        sample["audio_format"] = suffix
                        sample["audio_offset"] = int(member.offset_data)
                        sample["audio_size"] = int(member.size)
                    else:
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            continue
                        sample["json_member"] = member_name
                        sample["json_offset"] = int(member.offset_data)
                        sample["json_size"] = int(member.size)
                        sample["metadata"] = json.loads(extracted.read().decode("utf-8"))

                    if "audio_member" not in sample or "metadata" not in sample or "json_member" not in sample:
                        continue

                    metadata = sample["metadata"]
                    sample_id = resolve_sample_id(key, metadata, utt_id_key=split_config.utt_id_key)
                    split_name = assign_split(
                        sample_id if split_config.split_by == "sample_id" else shard_path.name,
                        split_config,
                    )
                    num_frames = infer_num_frames_from_metadata(metadata)
                    entry = {
                        "shard_name": shard_path.name,
                        "key": key,
                        "utt_id": sample_id,
                        "split": split_name,
                        "num_frames": num_frames,
                        "audio_member": sample["audio_member"],
                        "audio_format": sample["audio_format"],
                        "json_member": sample["json_member"],
                        "audio_offset": sample["audio_offset"],
                        "audio_size": sample["audio_size"],
                        "json_offset": sample["json_offset"],
                        "json_size": sample["json_size"],
                    }
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

                    total_samples += 1
                    split_counts[split_name] += 1
                    bucket_key = str(num_frames // 80)
                    bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1
                    min_frames = num_frames if min_frames is None else min(min_frames, num_frames)
                    max_frames = max(max_frames, num_frames)
                    pending.pop(key, None)

            elapsed = time.monotonic() - start_time
            _log(
                f"Length index progress: shards={shard_idx}/{len(shards)}, "
                f"samples={total_samples}, elapsed={elapsed:.1f}s, current={shard_path.name}"
            )

    summary = {
        "version": 2,
        "root": str(shard_root),
        "length_index_path": str(output_file),
        "num_shards": len(shards),
        "num_samples": total_samples,
        "min_frames": int(min_frames or 0),
        "max_frames": int(max_frames),
        "audio_suffixes": sorted(AUDIO_SUFFIXES),
        "split": {
            "type": "stable_hash",
            "split_by": split_config.split_by,
            "train_name": split_config.train_name,
            "eval_name": split_config.eval_name,
            "eval_ratio": split_config.eval_ratio,
            "hash_seed": split_config.hash_seed,
            "utt_id_key": split_config.utt_id_key,
        },
        "splits": {
            split_config.train_name: {"num_samples": split_counts[split_config.train_name]},
            split_config.eval_name: {"num_samples": split_counts[split_config.eval_name]},
        },
        "frame_buckets": bucket_counts,
    }
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _log(f"Finished WebDataset length inspection: samples={total_samples}, output={output_file}")
    return summary


def load_webdataset_length_entries(
    index_path: str | Path,
    *,
    split: str = "all",
) -> list[WebDatasetLengthEntry]:
    entries: list[WebDatasetLengthEntry] = []
    with Path(index_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if split != "all" and raw["split"] != split:
                continue
            entries.append(parse_webdataset_length_entry(raw))
    return entries


def parse_webdataset_length_entry(raw: dict[str, Any]) -> WebDatasetLengthEntry:
    storage_kind = str(raw.get("storage_kind") or "tar").strip().lower()
    if storage_kind not in SUPPORTED_LENGTH_INDEX_STORAGE_KINDS:
        raise ValueError(
            f"Unsupported length-index storage_kind={storage_kind!r}; "
            f"expected one of {sorted(SUPPORTED_LENGTH_INDEX_STORAGE_KINDS)}"
        )
    audio_size = (
        int(raw["audio_size"])
        if raw.get("audio_size") is not None
        else None
    )
    if audio_size is not None and audio_size <= 0:
        sample_key = str(raw.get("key") or raw.get("utt_id") or "unknown")
        raise ValueError(
            "WebDataset length entry has a non-positive audio_size: "
            f"key={sample_key!r} audio_size={audio_size}"
        )
    parquet_row_group = (
        int(raw["parquet_row_group"])
        if raw.get("parquet_row_group") is not None
        else None
    )
    parquet_row_index = (
        int(raw["parquet_row_index"])
        if raw.get("parquet_row_index") is not None
        else None
    )
    if storage_kind == "parquet" and (
        parquet_row_group is None
        or parquet_row_group < 0
        or parquet_row_index is None
        or parquet_row_index < 0
    ):
        raise ValueError(
            "Parquet length-index entries require non-negative "
            "parquet_row_group and parquet_row_index values."
        )
    json_member = str(raw.get("json_member") or "")
    if storage_kind == "tar" and not json_member:
        raise ValueError("Tar length-index entries require json_member.")
    return WebDatasetLengthEntry(
        shard_name=str(raw["shard_name"]),
        key=str(raw["key"]),
        utt_id=str(raw["utt_id"]),
        split=str(raw["split"]),
        num_frames=int(raw["num_frames"]),
        num_text_tokens=(
            int(raw["num_text_tokens"])
            if raw.get("num_text_tokens") is not None
            else None
        ),
        num_text_chars=(
            int(raw["num_text_chars"])
            if raw.get("num_text_chars") is not None
            else None
        ),
        text_bytes=(
            int(raw["text_bytes"])
            if raw.get("text_bytes") is not None
            else None
        ),
        audio_member=str(raw.get("audio_member") or raw["wav_member"]),
        audio_format=str(
            raw.get("audio_format")
            or _member_audio_format(str(raw.get("audio_member") or raw["wav_member"]))
        ),
        json_member=json_member,
        storage_kind=storage_kind,
        audio_offset=(
            int(raw["audio_offset"])
            if raw.get("audio_offset") is not None
            else None
        ),
        audio_size=audio_size,
        json_offset=(
            int(raw["json_offset"])
            if raw.get("json_offset") is not None
            else None
        ),
        json_size=(
            int(raw["json_size"])
            if raw.get("json_size") is not None
            else None
        ),
        zip_crc32=(
            int(raw["zip_crc32"])
            if raw.get("zip_crc32") is not None
            else None
        ),
        zip_compress_type=(
            int(raw["zip_compress_type"])
            if raw.get("zip_compress_type") is not None
            else None
        ),
        parquet_row_group=parquet_row_group,
        parquet_row_index=parquet_row_index,
        raw=dict(raw),
    )


class _TarShardReader:
    def __init__(self, shard_path: Path):
        self.shard_path = shard_path
        self._binary: BinaryIO | None = None
        self._archive: tarfile.TarFile | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"shard_path": self.shard_path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.shard_path = Path(state["shard_path"])
        self._binary = None
        self._archive = None

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        if self._binary is not None:
            self._binary.close()
            self._binary = None

    @property
    def is_open(self) -> bool:
        return self._archive is not None or self._binary is not None

    def _binary_handle(self) -> BinaryIO:
        if self._binary is None:
            self._binary = self.shard_path.open("rb")
        return self._binary

    def _archive_handle(self) -> tarfile.TarFile:
        if self._archive is None:
            self._archive = tarfile.open(self.shard_path, "r")
        return self._archive

    def read_member(self, member_name: str, *, offset: int | None, size: int | None) -> bytes:
        if offset is not None and size is not None:
            handle = self._binary_handle()
            handle.seek(offset)
            payload = handle.read(size)
            if len(payload) != size:
                raise EOFError(
                    f"Short read for {self.shard_path.name}:{member_name}; expected {size} bytes got {len(payload)}"
                )
            return payload

        extracted = self._archive_handle().extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Missing tar member {self.shard_path.name}:{member_name}")
        return extracted.read()


class _ZipShardReader:
    def __init__(self, shard_path: Path):
        self.shard_path = shard_path
        self._archive: zipfile.ZipFile | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"shard_path": self.shard_path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.shard_path = Path(state["shard_path"])
        self._archive = None

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    @property
    def is_open(self) -> bool:
        return self._archive is not None

    def _archive_handle(self) -> zipfile.ZipFile:
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.shard_path, "r")
        return self._archive

    def read_member(
        self,
        member_name: str,
        *,
        size: int | None,
        crc32: int | None,
        compress_type: int | None,
    ) -> bytes:
        archive = self._archive_handle()
        info = archive.getinfo(member_name)
        if size is not None and int(info.file_size) != int(size):
            raise ValueError(
                f"ZIP member size changed for {self.shard_path.name}:{member_name}; "
                f"expected {size} got {info.file_size}"
            )
        if crc32 is not None and int(info.CRC) != int(crc32):
            raise ValueError(
                f"ZIP member CRC changed for {self.shard_path.name}:{member_name}; "
                f"expected {crc32} got {info.CRC}"
            )
        if compress_type is not None and int(info.compress_type) != int(compress_type):
            raise ValueError(
                f"ZIP compression changed for {self.shard_path.name}:{member_name}; "
                f"expected {compress_type} got {info.compress_type}"
            )
        payload = archive.read(info)
        if size is not None and len(payload) != int(size):
            raise EOFError(
                f"Short ZIP read for {self.shard_path.name}:{member_name}; "
                f"expected {size} bytes got {len(payload)}"
            )
        return payload


class _ParquetShardReader:
    AUDIO_COLUMNS = ("id", "audio", "duration_ms")

    def __init__(self, shard_path: Path):
        self.shard_path = shard_path
        self._parquet_file: Any | None = None
        self._cached_row_group: int | None = None
        self._cached_table: Any | None = None

    def __getstate__(self) -> dict[str, Any]:
        return {"shard_path": self.shard_path}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.shard_path = Path(state["shard_path"])
        self._parquet_file = None
        self._cached_row_group = None
        self._cached_table = None

    def close(self) -> None:
        self._cached_table = None
        self._cached_row_group = None
        self._parquet_file = None

    @property
    def is_open(self) -> bool:
        return self._parquet_file is not None

    def _handle(self) -> Any:
        if self._parquet_file is None:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:  # pragma: no cover - covered by preprocess environments
                raise RuntimeError(
                    "Reading Parquet-backed audio requires the preprocess extra (pyarrow)."
                ) from exc
            self._parquet_file = pq.ParquetFile(self.shard_path)
        return self._parquet_file

    def read_audio_row(self, *, row_group: int, row_index: int) -> tuple[bytes, dict[str, Any]]:
        if self._cached_row_group != row_group or self._cached_table is None:
            self._cached_table = self._handle().read_row_group(
                row_group,
                columns=list(self.AUDIO_COLUMNS),
            )
            self._cached_row_group = int(row_group)
        table = self._cached_table
        if row_index < 0 or row_index >= len(table):
            raise IndexError(
                f"Parquet row index {row_index} outside row-group size {len(table)} "
                f"for {self.shard_path.name} row_group={row_group}"
            )
        audio_value = table.column("audio")[row_index].as_py()
        if not isinstance(audio_value, dict):
            raise ValueError(
                f"Parquet audio value is not a struct for {self.shard_path.name} "
                f"row_group={row_group} row={row_index}"
            )
        audio_bytes = audio_value.get("bytes")
        if not isinstance(audio_bytes, (bytes, bytearray, memoryview)) or not audio_bytes:
            raise ValueError(
                f"Parquet audio payload is empty for {self.shard_path.name} "
                f"row_group={row_group} row={row_index}"
            )
        metadata = {
            "id": str(table.column("id")[row_index].as_py() or ""),
            "duration_ms": int(table.column("duration_ms")[row_index].as_py() or 0),
            "audio_path": str(audio_value.get("path") or ""),
        }
        return bytes(audio_bytes), metadata


def _make_shard_reader(storage_kind: str, shard_path: Path) -> Any:
    if storage_kind == "tar":
        return _TarShardReader(shard_path)
    if storage_kind == "zip":
        return _ZipShardReader(shard_path)
    if storage_kind == "parquet":
        return _ParquetShardReader(shard_path)
    raise ValueError(f"Unsupported storage_kind={storage_kind!r}")


def _generated_metadata(entry: WebDatasetLengthEntry) -> dict[str, Any]:
    raw = entry.raw or {}
    metadata = {
        "text": str(raw.get("text") or ""),
        "sid": entry.utt_id,
        "utt_id": entry.utt_id,
        "id": entry.utt_id,
        "language": str(raw.get("language") or raw.get("lang") or "unknown"),
        "format": entry.audio_format,
        "duration": float(entry.num_frames) / 100.0,
        "num_frames": int(entry.num_frames),
    }
    if raw.get("sample_rate") is not None:
        metadata["sample_rate"] = int(raw["sample_rate"])
    return metadata


def _read_indexed_entry_payload(
    reader: Any,
    entry: WebDatasetLengthEntry,
) -> tuple[bytes, bytes]:
    if entry.storage_kind == "tar":
        audio_bytes = reader.read_member(
            entry.audio_member,
            offset=entry.audio_offset,
            size=entry.audio_size,
        )
        metadata_bytes = reader.read_member(
            entry.json_member,
            offset=entry.json_offset,
            size=entry.json_size,
        )
        return audio_bytes, metadata_bytes

    metadata = _generated_metadata(entry)
    if entry.storage_kind == "zip":
        audio_bytes = reader.read_member(
            entry.audio_member,
            size=entry.audio_size,
            crc32=entry.zip_crc32,
            compress_type=entry.zip_compress_type,
        )
    elif entry.storage_kind == "parquet":
        assert entry.parquet_row_group is not None
        assert entry.parquet_row_index is not None
        audio_bytes, parquet_metadata = reader.read_audio_row(
            row_group=entry.parquet_row_group,
            row_index=entry.parquet_row_index,
        )
        expected_id = str((entry.raw or {}).get("parquet_id") or entry.utt_id)
        actual_id = str(parquet_metadata.get("id") or "")
        if expected_id and actual_id != expected_id:
            raise ValueError(
                f"Parquet row identity changed for {reader.shard_path.name}: "
                f"expected {expected_id!r} got {actual_id!r}"
            )
        duration_ms = int(parquet_metadata.get("duration_ms") or 0)
        if duration_ms <= 0:
            raise ValueError(
                f"Parquet duration is non-positive for {reader.shard_path.name}: "
                f"row_group={entry.parquet_row_group} row={entry.parquet_row_index}"
            )
        metadata["duration_ms"] = duration_ms
        metadata["duration"] = duration_ms / 1000.0
        audio_path = str(parquet_metadata.get("audio_path") or "")
        if audio_path:
            metadata["audio_path"] = audio_path
    else:  # pragma: no cover - parser rejects this first
        raise ValueError(f"Unsupported storage_kind={entry.storage_kind!r}")
    return audio_bytes, json.dumps(metadata, ensure_ascii=False).encode("utf-8")


class LengthIndexedWebDatasetDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        shard_root: str | Path,
        entries: list[WebDatasetLengthEntry],
        *,
        tokenizer: TokenizerLike | None = None,
        decoder_tokenizer: TokenizerLike | None = None,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
        config: WebDatasetConfig | None = None,
    ):
        self.shard_root = Path(shard_root)
        self.entries = entries
        self.tokenizer = tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.config = config or WebDatasetConfig()
        preload_decoder_ctc_draft_cache(self.config)
        self._reader_cache: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reader_cache"] = {}
        return state

    def __del__(self) -> None:
        for reader in self._reader_cache.values():
            reader.close()

    def __len__(self) -> int:
        return len(self.entries)

    def _reader(self, entry: WebDatasetLengthEntry) -> Any:
        cache_key = f"{entry.storage_kind}\0{entry.shard_name}"
        reader = self._reader_cache.get(cache_key)
        if reader is None:
            reader = _make_shard_reader(
                entry.storage_kind,
                self.shard_root / entry.shard_name,
            )
            self._reader_cache[cache_key] = reader
            while len(self._reader_cache) > max(1, int(self.config.max_open_shards_per_worker)):
                _, evicted = self._reader_cache.popitem(last=False)
                evicted.close()
        else:
            self._reader_cache.move_to_end(cache_key)
        return reader

    def _decode_entry(self, entry: WebDatasetLengthEntry) -> dict[str, Any]:
        reader = self._reader(entry)
        audio_bytes, metadata_bytes = _read_indexed_entry_payload(reader, entry)
        return decode_webdataset_sample(
            key=entry.key,
            audio_bytes=audio_bytes,
            metadata_bytes=metadata_bytes,
            tokenizer=self.tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            feature_extractor=self.feature_extractor,
            text_key=self.config.text_key,
            utt_id_key=self.config.utt_id_key,
            token_ids_key=self.config.token_ids_key,
            append_eos=self.config.append_eos,
            decoder_append_eos=self.config.decoder_append_eos,
            text_normalization=self.config.text_normalization,
            decoder_text_normalization=self.config.decoder_text_normalization,
            decoder_prompt_before_audio=self.config.decoder_prompt_before_audio,
            decoder_prompt_before_audio_use_language=self.config.decoder_prompt_before_audio_use_language,
            decoder_ctc_draft_cache_path=self.config.decoder_ctc_draft_cache_path,
            decoder_ctc_draft_prompt_template=self.config.decoder_ctc_draft_prompt_template,
            decoder_ctc_draft_text_key=self.config.decoder_ctc_draft_text_key,
            decoder_ctc_draft_missing_policy=self.config.decoder_ctc_draft_missing_policy,
            decoder_ctc_draft_dropout_prob=self.config.decoder_ctc_draft_dropout_prob,
            decoder_ctc_draft_language_mismatch_dropout_prob=(
                self.config.decoder_ctc_draft_language_mismatch_dropout_prob
            ),
            decoder_ctc_draft_dropout_seed=self.config.decoder_ctc_draft_dropout_seed,
            ctc_label_override_cache_path=self.config.ctc_label_override_cache_path,
            ctc_label_override_text_key=self.config.ctc_label_override_text_key,
            decoder_target_prefix=self.config.decoder_target_prefix,
            decoder_target_prefix_use_language=self.config.decoder_target_prefix_use_language,
            decoder_language_confirmation_en=self.config.decoder_language_confirmation_en,
            decoder_language_confirmation_zh=self.config.decoder_language_confirmation_zh,
            decoder_prompt_language_label_noise_prob=self.config.decoder_prompt_language_label_noise_prob,
            decoder_prompt_language_label_noise_seed=self.config.decoder_prompt_language_label_noise_seed,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.config.skip_decode_errors:
            return self._decode_entry(self.entries[index])
        last_exc: Exception | None = None
        for offset in range(len(self.entries)):
            entry = self.entries[(index + offset) % len(self.entries)]
            try:
                return self._decode_entry(entry)
            except Exception as exc:
                last_exc = exc
                log_webdataset_decode_skip(
                    key=entry.key,
                    shard_name=entry.shard_name,
                    audio_member=entry.audio_member,
                    json_member=entry.json_member,
                    exc=exc,
                )
        raise RuntimeError("All length-index WebDataset samples failed to decode.") from last_exc


class LengthBucketedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        *,
        source_labels: list[str] | None = None,
        source_interleave: bool = False,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = True,
        frame_budget: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.lengths = list(lengths)
        self.source_labels = list(source_labels) if source_labels is not None else None
        if self.source_labels is not None and len(self.source_labels) != len(self.lengths):
            raise ValueError("source_labels must have the same length as lengths")
        self.source_interleave = bool(source_interleave and self.source_labels is not None)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.frame_budget = int(frame_budget) if frame_budget is not None else None
        self.epoch = 0

    @property
    def global_batch_size(self) -> int:
        return self.batch_size * self.world_size

    def _max_global_batch_size(self) -> int:
        return self.batch_size * self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _source_interleave_sorted_indices(self, indices: list[int], rng: random.Random) -> list[int]:
        if not self.source_interleave or self.source_labels is None:
            return indices
        window_size = max(self._max_global_batch_size() * 32, 256)
        interleaved: list[int] = []
        for offset in range(0, len(indices), window_size):
            window = indices[offset : offset + window_size]
            grouped: dict[str, list[int]] = {}
            for index in window:
                grouped.setdefault(self.source_labels[index], []).append(index)
            labels = sorted(grouped)
            if self.shuffle and len(labels) > 1:
                shift = rng.randrange(len(labels))
                labels = labels[shift:] + labels[:shift]
            cursors = {label: 0 for label in labels}
            totals = {label: len(grouped[label]) for label in labels}
            order = {label: index for index, label in enumerate(labels)}
            remaining = len(window)
            while remaining > 0 and labels:
                label = min(
                    labels,
                    key=lambda candidate: (
                        (cursors[candidate] + 0.5) / totals[candidate],
                        order[candidate],
                    ),
                )
                source_indices = grouped[label]
                source_cursor = cursors[label]
                if source_cursor >= len(source_indices):
                    labels.remove(label)
                    continue
                interleaved.append(source_indices[source_cursor])
                cursors[label] = source_cursor + 1
                remaining -= 1
        return interleaved

    def _build_global_batches(self, *, epoch: int) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + epoch)
        if self.shuffle:
            rng.shuffle(indices)
        if self.drop_last and self.world_size > 1:
            remainder = len(indices) % self.world_size
            if remainder:
                indices = indices[:-remainder]
        indices.sort(key=lambda idx: self.lengths[idx])
        indices = self._source_interleave_sorted_indices(indices, rng)
        max_global_batch = self._max_global_batch_size()
        global_batches: list[list[int]] = []
        offset = 0
        while offset < len(indices):
            remaining = len(indices) - offset
            if remaining < self.world_size:
                break
            if not self.frame_budget or self.frame_budget <= 0:
                batch_size = min(max_global_batch, remaining)
                if self.world_size > 1:
                    batch_size -= batch_size % self.world_size
                if batch_size <= 0:
                    break
            else:
                batch_size = _select_dynamic_global_batch_size(
                    self.lengths,
                    indices,
                    start=offset,
                    max_local_batch_size=self.batch_size,
                    world_size=self.world_size,
                    frame_budget=self.frame_budget,
                    drop_last=self.drop_last,
                )
                if batch_size <= 0:
                    break
            global_batches.append(indices[offset : offset + batch_size])
            offset += batch_size
        if self.shuffle:
            rng.shuffle(global_batches)
        return global_batches

    def __iter__(self) -> Iterable[list[int]]:
        global_batches = self._build_global_batches(epoch=self.epoch)

        for global_batch in global_batches:
            local_batch_size = len(global_batch) // self.world_size
            local_start = self.rank * local_batch_size
            local_end = local_start + local_batch_size
            local_batch = global_batch[local_start:local_end]
            if self.drop_last and len(local_batch) != local_batch_size:
                continue
            if local_batch:
                yield local_batch

    def __len__(self) -> int:
        return len(self._build_global_batches(epoch=0))


class _StaticBatchSampler(Sampler[list[int]]):
    def __init__(self, batches: Iterable[list[int]]):
        self.batches = [list(batch) for batch in batches]

    def __iter__(self) -> Iterable[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


class LengthBucketedDataLoader:
    def __init__(
        self,
        dataset: LengthIndexedWebDatasetDataset,
        *,
        batch_sampler: LengthBucketedBatchSampler,
        num_workers: int,
        collate_fn: FeatureCollator,
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.num_workers = int(num_workers)
        self.collate_fn = collate_fn

    def _build_loader(self, batch_sampler: Sampler[list[int]]) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def __iter__(self) -> Iterable[Any]:
        yield from self._build_loader(self.batch_sampler)

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def iter_from_batch_offset(
        self,
        batch_offset: int,
        *,
        progress_interval: int = 0,
        progress_callback: Any | None = None,
    ) -> tuple[Iterable[Any], int]:
        batches = list(self.batch_sampler)
        skipped = min(max(0, int(batch_offset)), len(batches))
        if progress_callback is not None and progress_interval > 0:
            for value in range(progress_interval, skipped + 1, progress_interval):
                progress_callback(value)
        remaining_sampler = _StaticBatchSampler(batches[skipped:])
        return self._build_loader(remaining_sampler), skipped


def _select_dynamic_global_batch_size(
    lengths: list[int],
    sorted_indices: list[int],
    *,
    start: int,
    max_local_batch_size: int,
    world_size: int,
    frame_budget: int,
    drop_last: bool,
) -> int:
    remaining = len(sorted_indices) - start
    if remaining <= 0:
        return 0
    step = 1 if world_size == 1 else world_size
    max_global_batch = min(max_local_batch_size * world_size, remaining)
    if world_size > 1:
        max_global_batch -= max_global_batch % world_size
    if max_global_batch <= 0:
        return 0

    best = 0
    candidate = step
    while candidate <= max_global_batch:
        local_batch_size = candidate if world_size == 1 else candidate // world_size
        max_frames = max(lengths[index] for index in sorted_indices[start : start + candidate])
        if local_batch_size * max_frames <= frame_budget:
            best = candidate
            candidate += step
            continue
        break

    if best > 0:
        return best
    if world_size == 1:
        return 1
    if remaining < world_size:
        return 0
    return world_size


def estimate_length_bucketed_steps(
    lengths: list[int],
    *,
    batch_size: int,
    world_size: int,
    frame_budget: int | None,
    drop_last: bool,
) -> int:
    sampler = LengthBucketedBatchSampler(
        lengths,
        batch_size=batch_size,
        rank=0,
        world_size=world_size,
        seed=0,
        shuffle=False,
        drop_last=drop_last,
        frame_budget=frame_budget,
    )
    return len(sampler)


def build_length_bucketed_webdataset_dataloader(
    shard_root: str | Path,
    *,
    length_index_path: str | Path,
    tokenizer: TokenizerLike | None = None,
    decoder_tokenizer: TokenizerLike | None = None,
    feature_extractor: WenetFbankFeatureExtractor | None = None,
    config: WebDatasetConfig | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[LengthBucketedDataLoader, LengthBucketedBatchSampler]:
    config = config or WebDatasetConfig()
    entries = load_webdataset_length_entries(length_index_path, split=config.split)
    dataset = LengthIndexedWebDatasetDataset(
        shard_root,
        entries,
        tokenizer=tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        feature_extractor=feature_extractor,
        config=config,
    )
    sampler = LengthBucketedBatchSampler(
        [entry.num_frames for entry in entries],
        source_labels=(
            [_shard_source_label(entry.shard_name) for entry in entries]
            if config.bucket_source_interleave
            else None
        ),
        source_interleave=config.bucket_source_interleave,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=config.seed,
        shuffle=config.shuffle_shards,
        drop_last=config.length_bucket_drop_last,
        frame_budget=config.length_bucket_frame_budget,
    )
    loader = LengthBucketedDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=FeatureCollator(),
    )
    return loader, sampler
