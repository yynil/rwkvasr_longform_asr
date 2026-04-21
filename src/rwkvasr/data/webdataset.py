from __future__ import annotations

import io
import json
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .manifest import FeatureCollator, TokenizerLike, WenetFbankFeatureExtractor, build_text_tokenizer
from .webdataset_common import AUDIO_SUFFIXES
from .webdataset_index import StableHashSplitConfig, resolve_sample_id, sample_in_split, shard_in_split


@dataclass(frozen=True)
class WebDatasetConfig:
    shard_pattern: str = "*.tar"
    shuffle_shards: bool = True
    seed: int = 42
    text_key: str = "text"
    utt_id_key: str = "sid"
    token_ids_key: str = "token_ids"
    split: str = "all"
    eval_ratio: float = 0.0
    hash_seed: int = 0
    split_by: str = "shard_name"
    partition_by_rank: bool = True
    length_index_path: str | None = None
    use_length_bucketing: bool = False
    length_bucket_drop_last: bool = True
    length_bucket_frame_budget: int | None = None
    decoded_batch_prefetch: int = 2
    max_open_shards_per_worker: int = 8


def decode_webdataset_sample(
    *,
    key: str,
    audio_bytes: bytes,
    metadata_bytes: bytes,
    tokenizer: TokenizerLike | None,
    feature_extractor: WenetFbankFeatureExtractor,
    text_key: str,
    utt_id_key: str,
    token_ids_key: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import soundfile as sf

    metadata = metadata or json.loads(metadata_bytes.decode("utf-8"))
    token_ids = metadata.get(token_ids_key)
    text = metadata.get(text_key)
    if token_ids is None:
        if text is None:
            raise ValueError("WebDataset sample needs token_ids or text with a tokenizer.")
        if tokenizer is None:
            tokenizer = build_text_tokenizer("whisper_multilingual")
        token_ids = tokenizer.encode(text)

    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, sample_rate = sf.read(audio_buffer, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_array).transpose(0, 1)
    features = feature_extractor(waveform, sample_rate).float()
    targets = torch.tensor([int(token) for token in token_ids], dtype=torch.long)
    utt_id = str(metadata.get(utt_id_key) or key)

    return {
        "utt_id": utt_id,
        "features": features,
        "feature_length": features.size(0),
        "targets": targets,
        "target_length": targets.numel(),
        "text": text,
        "language": metadata.get("language"),
        "metadata": metadata,
    }


class WebDatasetASRIterableDataset(IterableDataset[dict[str, Any]]):
    def __init__(
        self,
        shard_root: str | Path,
        *,
        tokenizer: TokenizerLike | None = None,
        feature_extractor: WenetFbankFeatureExtractor | None = None,
        config: WebDatasetConfig | None = None,
    ):
        super().__init__()
        self.shard_root = Path(shard_root)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or WenetFbankFeatureExtractor()
        self.config = config or WebDatasetConfig()
        self.epoch = 0
        self.shards = self._resolve_shards()
        self.split_config = StableHashSplitConfig(
            eval_ratio=self.config.eval_ratio,
            hash_seed=self.config.hash_seed,
            utt_id_key=self.config.utt_id_key,
            split_by=self.config.split_by,
        )

    def _resolve_shards(self) -> list[Path]:
        if self.shard_root.is_file():
            shards = [self.shard_root]
        else:
            shards = sorted(self.shard_root.glob(self.config.shard_pattern))
        if not shards:
            raise FileNotFoundError(
                f"No shard files matching {self.config.shard_pattern!r} under {self.shard_root}"
            )
        return shards

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _partition_shards(self) -> list[Path]:
        shards = list(self.shards)
        if self.config.shuffle_shards:
            random.Random(self.config.seed + self.epoch).shuffle(shards)
        if self.config.split != "all" and self.split_config.split_by == "shard_name":
            shards = [shard for shard in shards if shard_in_split(shard.name, self.config.split, self.split_config)]

        if self.config.partition_by_rank:
            rank = int(os.environ.get("RANK", "0"))
            world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
            shards = shards[rank::world_size]

            worker = get_worker_info()
            if worker is not None:
                shards = shards[worker.id :: worker.num_workers]
        return shards

    def _iter_shard(self, shard_path: Path) -> Iterable[dict[str, Any]]:
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if "." not in name:
                    continue
                key, suffix = name.rsplit(".", 1)
                suffix = suffix.lower()
                if suffix != "json" and suffix not in AUDIO_SUFFIXES:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                sample = pending.setdefault(key, {})
                if suffix == "json":
                    sample["json"] = extracted.read()
                else:
                    sample["audio"] = extracted.read()
                if "audio" in sample and "json" in sample:
                    metadata = json.loads(sample["json"].decode("utf-8"))
                    include_sample = True
                    if self.split_config.split_by == "sample_id":
                        sample_id = resolve_sample_id(key, metadata, utt_id_key=self.config.utt_id_key)
                        include_sample = sample_in_split(
                            sample_id,
                            self.config.split,
                            self.split_config,
                            shard_name=shard_path.name,
                        )
                    if include_sample:
                        yield self._decode_sample(key, sample["audio"], sample["json"], metadata=metadata)
                    pending.pop(key, None)

    def _decode_sample(
        self,
        key: str,
        audio_bytes: bytes,
        metadata_bytes: bytes,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return decode_webdataset_sample(
            key=key,
            audio_bytes=audio_bytes,
            metadata_bytes=metadata_bytes,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            text_key=self.config.text_key,
            utt_id_key=self.config.utt_id_key,
            token_ids_key=self.config.token_ids_key,
            metadata=metadata,
        )

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for shard_path in self._partition_shards():
            yield from self._iter_shard(shard_path)


def build_webdataset_dataloader(
    shard_root: str | Path,
    *,
    tokenizer: TokenizerLike | None = None,
    feature_extractor: WenetFbankFeatureExtractor | None = None,
    config: WebDatasetConfig | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    dataset = WebDatasetASRIterableDataset(
        shard_root,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        config=config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=FeatureCollator(),
    )
