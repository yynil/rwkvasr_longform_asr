from __future__ import annotations

import queue
import json
import math
import random
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

from .manifest import ASRBatch, FeatureCollator, TokenizerLike, WenetFbankFeatureExtractor
from .webdataset import WebDatasetConfig, decode_webdataset_sample
from .webdataset_lengths import WebDatasetLengthEntry, parse_webdataset_length_entry


@dataclass(frozen=True, slots=True)
class WebDatasetBucketPart:
    path: str
    num_samples: int
    first_shard: str | None = None
    last_shard: str | None = None


@dataclass(frozen=True, slots=True)
class WebDatasetBucket:
    split: str
    bucket_id: int
    num_samples: int
    parts: tuple[WebDatasetBucketPart, ...]


@dataclass(frozen=True, slots=True)
class WebDatasetBucketManifest:
    manifest_path: Path
    root: str
    source_length_index_path: str
    bucket_width: int
    entries_per_part: int
    splits: dict[str, tuple[WebDatasetBucket, ...]]


def resolve_webdataset_bucket_manifest_path(
    shard_root: str | Path,
    manifest_path: str | Path | None = None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)
    shard_root = Path(shard_root)
    if shard_root.is_dir():
        return shard_root / "webdataset_buckets" / "manifest.json"
    return shard_root.with_suffix(".bucket_manifest.json")


def load_webdataset_bucket_manifest(manifest_path: str | Path) -> WebDatasetBucketManifest:
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = raw.get("splits", {})
    splits: dict[str, tuple[WebDatasetBucket, ...]] = {}
    for split_name, split_data in split_payload.items():
        buckets: list[WebDatasetBucket] = []
        for bucket_data in split_data.get("buckets", []):
            parts = tuple(
                WebDatasetBucketPart(
                    path=str(part["path"]),
                    num_samples=int(part["num_samples"]),
                    first_shard=part.get("first_shard"),
                    last_shard=part.get("last_shard"),
                )
                for part in bucket_data.get("parts", [])
            )
            buckets.append(
                WebDatasetBucket(
                    split=str(split_name),
                    bucket_id=int(bucket_data["bucket_id"]),
                    num_samples=int(bucket_data["num_samples"]),
                    parts=parts,
                )
            )
        buckets.sort(key=lambda item: item.bucket_id)
        splits[str(split_name)] = tuple(buckets)
    return WebDatasetBucketManifest(
        manifest_path=manifest_path,
        root=str(raw["root"]),
        source_length_index_path=str(raw["source_length_index_path"]),
        bucket_width=int(raw["bucket_width"]),
        entries_per_part=int(raw["entries_per_part"]),
        splits=splits,
    )


def compute_bucket_local_batch_size(
    *,
    bucket_id: int,
    bucket_width: int,
    max_local_batch_size: int,
    frame_budget: int | None,
) -> int:
    if max_local_batch_size <= 0:
        raise ValueError("max_local_batch_size must be positive.")
    if frame_budget is None or frame_budget <= 0:
        return int(max_local_batch_size)
    bucket_upper_frames = max(1, (int(bucket_id) + 1) * int(bucket_width))
    return max(1, min(int(max_local_batch_size), int(frame_budget) // bucket_upper_frames))


def estimate_bucket_manifest_steps(
    manifest: WebDatasetBucketManifest,
    *,
    split: str,
    batch_size: int,
    world_size: int,
    frame_budget: int | None,
    drop_last: bool = True,
) -> int:
    total_steps = 0
    for bucket in manifest.splits.get(split, ()):
        local_batch_size = compute_bucket_local_batch_size(
            bucket_id=bucket.bucket_id,
            bucket_width=manifest.bucket_width,
            max_local_batch_size=batch_size,
            frame_budget=frame_budget,
        )
        global_batch_size = max(1, local_batch_size * max(1, world_size))
        if drop_last:
            total_steps += bucket.num_samples // global_batch_size
        else:
            total_steps += math.ceil(bucket.num_samples / global_batch_size)
    return max(1, total_steps)


class _TarShardReader:
    def __init__(self, shard_path: Path):
        self.shard_path = shard_path
        self._binary: BinaryIO | None = None
        self._archive: tarfile.TarFile | None = None

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        if self._binary is not None:
            self._binary.close()
            self._binary = None

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


class _BucketEntryStream:
    def __init__(self, manifest_path: Path, bucket: WebDatasetBucket):
        self._manifest_path = manifest_path
        self._bucket = bucket
        self._part_index = 0
        self._handle = None

    def reset(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._part_index = 0

    def _open_next_part(self):
        if self._part_index >= len(self._bucket.parts):
            return None
        part = self._bucket.parts[self._part_index]
        self._part_index += 1
        self._handle = (self._manifest_path.parent / part.path).open("r", encoding="utf-8")
        return self._handle

    def take(self, num_entries: int) -> list[WebDatasetLengthEntry]:
        entries: list[WebDatasetLengthEntry] = []
        while len(entries) < num_entries:
            if self._handle is None:
                handle = self._open_next_part()
                if handle is None:
                    break
            assert self._handle is not None
            line = self._handle.readline()
            if not line:
                self._handle.close()
                self._handle = None
                continue
            entries.append(parse_webdataset_length_entry(json.loads(line)))
        return entries


class _ThreadLocalTarReaderPool:
    def __init__(self, shard_root: Path):
        self._shard_root = shard_root
        self._local = threading.local()
        self._created: list[_TarShardReader] = []
        self._created_lock = threading.Lock()

    def get(self, shard_name: str) -> _TarShardReader:
        readers = getattr(self._local, "readers", None)
        if readers is None:
            readers = {}
            self._local.readers = readers
        reader = readers.get(shard_name)
        if reader is None:
            reader = _TarShardReader(self._shard_root / shard_name)
            readers[shard_name] = reader
            with self._created_lock:
                self._created.append(reader)
        return reader

    def close(self) -> None:
        seen: set[int] = set()
        for reader in self._created:
            reader_id = id(reader)
            if reader_id in seen:
                continue
            seen.add(reader_id)
            reader.close()


class BucketedWebDatasetBatchLoader:
    def __init__(
        self,
        shard_root: str | Path,
        *,
        bucket_manifest_path: str | Path,
        tokenizer: TokenizerLike | None = None,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
        config: WebDatasetConfig | None = None,
        batch_size: int = 4,
        num_workers: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.shard_root = Path(shard_root)
        self.manifest = load_webdataset_bucket_manifest(bucket_manifest_path)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.config = config or WebDatasetConfig()
        self.batch_size = int(batch_size)
        self.num_workers = max(1, int(num_workers))
        self.decoded_batch_prefetch = max(0, int(self.config.decoded_batch_prefetch))
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.collator = FeatureCollator()
        self.epoch = 0
        self.dataset = self

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _split_name(self) -> str:
        return str(self.config.split)

    def _bucket_steps(self, bucket: WebDatasetBucket) -> int:
        local_batch = compute_bucket_local_batch_size(
            bucket_id=bucket.bucket_id,
            bucket_width=self.manifest.bucket_width,
            max_local_batch_size=self.batch_size,
            frame_budget=self.config.length_bucket_frame_budget,
        )
        global_batch = max(1, local_batch * self.world_size)
        if self.config.length_bucket_drop_last:
            return bucket.num_samples // global_batch
        return math.ceil(bucket.num_samples / global_batch)

    def _build_schedule(self) -> list[int]:
        schedule: list[int] = []
        for bucket in self.manifest.splits.get(self._split_name(), ()):
            schedule.extend([bucket.bucket_id] * self._bucket_steps(bucket))
        if self.config.shuffle_shards:
            random.Random(self.config.seed + self.epoch).shuffle(schedule)
        return schedule

    def __len__(self) -> int:
        return len(self._build_schedule())

    def _iter_local_entry_batches(self) -> Iterator[list[WebDatasetLengthEntry]]:
        split = self._split_name()
        buckets = {bucket.bucket_id: bucket for bucket in self.manifest.splits.get(split, ())}
        streams = {
            bucket_id: _BucketEntryStream(self.manifest.manifest_path, bucket)
            for bucket_id, bucket in buckets.items()
        }
        try:
            for bucket_id in self._build_schedule():
                bucket = buckets[bucket_id]
                local_batch = compute_bucket_local_batch_size(
                    bucket_id=bucket.bucket_id,
                    bucket_width=self.manifest.bucket_width,
                    max_local_batch_size=self.batch_size,
                    frame_budget=self.config.length_bucket_frame_budget,
                )
                global_batch = local_batch * self.world_size
                entries = streams[bucket_id].take(global_batch)
                if len(entries) < global_batch:
                    if self.config.length_bucket_drop_last:
                        continue
                    if len(entries) <= self.rank * local_batch:
                        continue
                local_start = self.rank * local_batch
                local_end = min(local_start + local_batch, len(entries))
                local_entries = entries[local_start:local_end]
                if not local_entries:
                    continue
                yield local_entries
        finally:
            for stream in streams.values():
                stream.reset()

    def _prefetch_local_entry_batches(self) -> Iterator[list[WebDatasetLengthEntry]]:
        max_prefetch = max(2, min(8, self.num_workers))
        items: queue.Queue[list[WebDatasetLengthEntry] | BaseException | None] = queue.Queue(maxsize=max_prefetch)
        sentinel = None

        def _producer() -> None:
            try:
                for local_entries in self._iter_local_entry_batches():
                    items.put(local_entries)
            except BaseException as exc:  # pragma: no cover - hard to force reliably in tests
                items.put(exc)
            finally:
                items.put(sentinel)

        producer = threading.Thread(
            target=_producer,
            name=f"bucket-prefetch-rank{self.rank}",
            daemon=True,
        )
        producer.start()
        while True:
            item = items.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        producer.join()

    def _decode_batch(
        self,
        local_entries: list[WebDatasetLengthEntry],
        *,
        reader_pool: _ThreadLocalTarReaderPool,
        executor: ThreadPoolExecutor | None,
    ) -> ASRBatch:
        if executor is None:
            samples = [self._decode_entry(entry, reader_pool) for entry in local_entries]
        else:
            samples = list(executor.map(lambda entry: self._decode_entry(entry, reader_pool), local_entries))
        return self.collator(samples)

    def _prefetch_decoded_batches(
        self,
        entry_batches: Iterator[list[WebDatasetLengthEntry]],
        *,
        reader_pool: _ThreadLocalTarReaderPool,
        executor: ThreadPoolExecutor | None,
    ) -> Iterator[ASRBatch]:
        max_prefetch = max(1, self.decoded_batch_prefetch)
        items: queue.Queue[ASRBatch | BaseException | None] = queue.Queue(maxsize=max_prefetch)
        sentinel = None
        stop_event = threading.Event()

        def _put(item: ASRBatch | BaseException | None) -> bool:
            while not stop_event.is_set():
                try:
                    items.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def _producer() -> None:
            try:
                for local_entries in entry_batches:
                    if stop_event.is_set():
                        break
                    batch = self._decode_batch(
                        local_entries,
                        reader_pool=reader_pool,
                        executor=executor,
                    )
                    if not _put(batch):
                        break
            except BaseException as exc:  # pragma: no cover - hard to force reliably in tests
                _put(exc)
            finally:
                _put(sentinel)

        producer = threading.Thread(
            target=_producer,
            name=f"bucket-batch-prefetch-rank{self.rank}",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = items.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_event.set()
            producer.join()

    def __iter__(self) -> Iterable[ASRBatch]:
        reader_pool = _ThreadLocalTarReaderPool(self.shard_root)
        entry_batches = self._prefetch_local_entry_batches() if self.num_workers > 1 else self._iter_local_entry_batches()
        executor: ThreadPoolExecutor | None = None
        if self.num_workers > 1:
            executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix=f"bucket-decode-r{self.rank}",
            )
        try:
            if self.num_workers > 1 and self.decoded_batch_prefetch > 0:
                yield from self._prefetch_decoded_batches(
                    entry_batches,
                    reader_pool=reader_pool,
                    executor=executor,
                )
            else:
                for local_entries in entry_batches:
                    yield self._decode_batch(
                        local_entries,
                        reader_pool=reader_pool,
                        executor=executor,
                    )
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
            reader_pool.close()

    def _decode_entry(
        self,
        entry: WebDatasetLengthEntry,
        reader_pool: _ThreadLocalTarReaderPool,
    ) -> dict[str, Any]:
        reader = reader_pool.get(entry.shard_name)
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
        return decode_webdataset_sample(
            key=entry.key,
            audio_bytes=audio_bytes,
            metadata_bytes=metadata_bytes,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            text_key=self.config.text_key,
            utt_id_key=self.config.utt_id_key,
            token_ids_key=self.config.token_ids_key,
        )


def build_bucketed_webdataset_loader(
    shard_root: str | Path,
    *,
    bucket_manifest_path: str | Path,
    tokenizer: TokenizerLike | None = None,
    feature_extractor: WenetFbankFeatureExtractor | None = None,
    config: WebDatasetConfig | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> BucketedWebDatasetBatchLoader:
    return BucketedWebDatasetBatchLoader(
        shard_root,
        bucket_manifest_path=bucket_manifest_path,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )
