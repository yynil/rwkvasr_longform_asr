from __future__ import annotations

import json
import random
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from torch.utils.data import DataLoader, Dataset, Sampler

from .manifest import FeatureCollator, TokenizerLike, WenetFbankFeatureExtractor
from .webdataset import WebDatasetConfig, decode_webdataset_sample
from .webdataset_index import StableHashSplitConfig, assign_split, resolve_sample_id


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class WebDatasetLengthEntry:
    shard_name: str
    key: str
    utt_id: str
    split: str
    num_frames: int
    wav_member: str
    json_member: str


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
                    if suffix not in {"wav", "json"}:
                        continue

                    sample = pending.setdefault(key, {})
                    if suffix == "wav":
                        sample["wav_member"] = member_name
                    else:
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            continue
                        sample["json_member"] = member_name
                        sample["metadata"] = json.loads(extracted.read().decode("utf-8"))

                    if "wav_member" not in sample or "metadata" not in sample or "json_member" not in sample:
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
                        "wav_member": sample["wav_member"],
                        "json_member": sample["json_member"],
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
        "version": 1,
        "root": str(shard_root),
        "length_index_path": str(output_file),
        "num_shards": len(shards),
        "num_samples": total_samples,
        "min_frames": int(min_frames or 0),
        "max_frames": int(max_frames),
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
            entries.append(
                WebDatasetLengthEntry(
                    shard_name=str(raw["shard_name"]),
                    key=str(raw["key"]),
                    utt_id=str(raw["utt_id"]),
                    split=str(raw["split"]),
                    num_frames=int(raw["num_frames"]),
                    wav_member=str(raw["wav_member"]),
                    json_member=str(raw["json_member"]),
                )
            )
    return entries


class LengthIndexedWebDatasetDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        shard_root: str | Path,
        entries: list[WebDatasetLengthEntry],
        *,
        tokenizer: TokenizerLike | None = None,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
        config: WebDatasetConfig | None = None,
    ):
        self.shard_root = Path(shard_root)
        self.entries = entries
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.config = config or WebDatasetConfig()
        self._tar_cache: dict[str, tarfile.TarFile] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tar_cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.entries)

    def _archive(self, shard_name: str) -> tarfile.TarFile:
        archive = self._tar_cache.get(shard_name)
        if archive is None:
            archive = tarfile.open(self.shard_root / shard_name, "r")
            self._tar_cache[shard_name] = archive
        return archive

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        archive = self._archive(entry.shard_name)
        wav_file = archive.extractfile(entry.wav_member)
        json_file = archive.extractfile(entry.json_member)
        if wav_file is None or json_file is None:
            raise FileNotFoundError(f"Missing sample members for {entry.shard_name}:{entry.key}")
        audio_bytes = wav_file.read()
        metadata_bytes = json_file.read()
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


class LengthBucketedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        *,
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
        max_frames = lengths[sorted_indices[start + candidate - 1]]
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
    feature_extractor: WenetFbankFeatureExtractor | None = None,
    config: WebDatasetConfig | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, LengthBucketedBatchSampler]:
    config = config or WebDatasetConfig()
    entries = load_webdataset_length_entries(length_index_path, split=config.split)
    dataset = LengthIndexedWebDatasetDataset(
        shard_root,
        entries,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=config,
    )
    sampler = LengthBucketedBatchSampler(
        [entry.num_frames for entry in entries],
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        seed=config.seed,
        shuffle=config.shuffle_shards,
        drop_last=config.length_bucket_drop_last,
        frame_budget=config.length_bucket_frame_budget,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=FeatureCollator(),
    )
    return loader, sampler
