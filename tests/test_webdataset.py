import io
import json
import sys
import tarfile
import types
import wave
from pathlib import Path

import torch

from rwkvasr.data import (
    LengthBucketedBatchSampler,
    StableHashSplitConfig,
    WebDatasetASRIterableDataset,
    WebDatasetConfig,
    build_bucketed_webdataset_loader,
    assign_split,
    estimate_length_bucketed_steps,
    estimate_bucket_manifest_steps,
    build_webdataset_dataloader,
    build_length_bucketed_webdataset_dataloader,
    inspect_webdataset,
    inspect_webdataset_lengths,
    load_webdataset_index,
    load_webdataset_bucket_manifest,
    load_webdataset_length_entries,
)
from rwkvasr.data.webdataset_bucketed import (
    _BucketEntryStream,
    _ThreadLocalTarReaderPool,
    WebDatasetBucket,
    WebDatasetBucketPart,
)


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [idx + 1 for idx, _ in enumerate(text)]


def _make_waveform(num_frames: int) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, steps=num_frames)
    return torch.sin(2.0 * torch.pi * 220.0 * time)


def _make_wav_bytes(num_frames: int) -> bytes:
    waveform = (_make_waveform(num_frames).clamp(-1.0, 1.0) * 32767.0).to(dtype=torch.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(waveform.numpy().tobytes())
    return buffer.getvalue()


def _write_json_bytes(text: str, sid: str) -> bytes:
    return json.dumps(
        {
            "text": text,
            "language": "zh",
            "sample_rate": 16000,
            "format": "wav",
            "begin_time": 0.0,
            "end_time": 1.0,
            "sid": sid,
        },
        ensure_ascii=False,
    ).encode("utf-8")


def _write_shard(tmp_path: Path, shard_name: str, samples: list[tuple[str, str, str]]) -> Path:
    shard_path = tmp_path / shard_name
    with tarfile.open(shard_path, "w") as archive:
        for key, text, sid in samples:
            wav_bytes = _make_wav_bytes(16000)
            wav_info = tarfile.TarInfo(name=f"{key}.wav")
            wav_info.size = len(wav_bytes)
            archive.addfile(wav_info, io.BytesIO(wav_bytes))

            json_bytes = _write_json_bytes(text, sid)
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = len(json_bytes)
            archive.addfile(info, io.BytesIO(json_bytes))
    return shard_path


def _write_mp3_named_shard(tmp_path: Path, shard_name: str, samples: list[tuple[str, str, str]]) -> Path:
    shard_path = tmp_path / shard_name
    with tarfile.open(shard_path, "w") as archive:
        for key, text, sid in samples:
            audio_bytes = _make_wav_bytes(16000)
            audio_info = tarfile.TarInfo(name=f"{key}.mp3")
            audio_info.size = len(audio_bytes)
            archive.addfile(audio_info, io.BytesIO(audio_bytes))

            json_bytes = _write_json_bytes(text, sid)
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = len(json_bytes)
            archive.addfile(info, io.BytesIO(json_bytes))
    return shard_path


def _build_root(tmp_path: Path) -> Path:
    _write_shard(
        tmp_path,
        "shard_00000000.tar",
        [
            ("0000000001", "你好世界", "sid-1"),
            ("0000000002", "双向语音", "sid-2"),
        ],
    )
    _write_shard(
        tmp_path,
        "shard_00000001.tar",
        [
            ("0000000003", "流式识别", "sid-3"),
        ],
    )
    return tmp_path


def _write_bucket_manifest(
    tmp_path: Path,
    *,
    root: Path,
    entries_by_part: list[tuple[str, list[dict[str, object]]]],
    bucket_width: int = 80,
) -> Path:
    bucket_root = tmp_path / "webdataset_buckets"
    bucket_root.mkdir(parents=True, exist_ok=True)
    parts = []
    bucket_counts: dict[int, int] = {}
    for relative_path, entries in entries_by_part:
        part_path = bucket_root / relative_path
        part_path.parent.mkdir(parents=True, exist_ok=True)
        with part_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                bucket_id = int(entry["num_frames"]) // bucket_width
                bucket_counts[bucket_id] = bucket_counts.get(bucket_id, 0) + 1
        parts.append((relative_path, entries))

    grouped: dict[int, list[tuple[str, list[dict[str, object]]]]] = {}
    for relative_path, entries in parts:
        bucket_id = int(entries[0]["num_frames"]) // bucket_width
        grouped.setdefault(bucket_id, []).append((relative_path, entries))

    manifest = {
        "version": 1,
        "root": str(root),
        "source_length_index_path": str(tmp_path / "webdataset_lengths.jsonl"),
        "bucket_width": bucket_width,
        "entries_per_part": 100000,
        "splits": {
            "train": {
                "num_samples": sum(bucket_counts.values()),
                "buckets": [
                    {
                        "bucket_id": bucket_id,
                        "num_samples": bucket_counts[bucket_id],
                        "parts": [
                            {
                                "path": relative_path,
                                "num_samples": len(entries),
                                "first_shard": str(entries[0]["shard_name"]),
                                "last_shard": str(entries[-1]["shard_name"]),
                            }
                            for relative_path, entries in grouped[bucket_id]
                        ],
                    }
                    for bucket_id in sorted(grouped)
                ],
            }
        },
    }
    manifest_path = bucket_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _install_fake_whisper(monkeypatch) -> None:
    fake_tokenizer_module = types.ModuleType("whisper.tokenizer")

    class FakeProcessor:
        eot = 50000

        def encode(self, text: str) -> list[int]:
            return [201, 202]

        def decode(self, token_ids: list[int]) -> str:
            return "decoded"

    def fake_get_tokenizer(*, multilingual: bool, language=None, task=None):
        assert multilingual is True
        return FakeProcessor()

    fake_tokenizer_module.get_tokenizer = fake_get_tokenizer  # type: ignore[attr-defined]
    fake_whisper_module = types.ModuleType("whisper")
    fake_whisper_module.tokenizer = fake_tokenizer_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper_module)
    monkeypatch.setitem(sys.modules, "whisper.tokenizer", fake_tokenizer_module)


def test_webdataset_iterable_decodes_audio_and_tokenizes_text(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    dataset = WebDatasetASRIterableDataset(
        root,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(shuffle_shards=False),
    )

    samples = list(dataset)

    assert [sample["utt_id"] for sample in samples] == ["sid-1", "sid-2", "sid-3"]
    assert all(sample["features"].dim() == 2 for sample in samples)
    assert all(sample["features"].size(1) == 80 for sample in samples)
    assert [sample["target_length"] for sample in samples] == [4, 4, 4]


def test_webdataset_dataloader_batches_like_manifest_pipeline(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    loader = build_webdataset_dataloader(
        root,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(shuffle_shards=False),
        batch_size=2,
    )

    batch = next(iter(loader))

    assert batch.features.shape[0] == 2
    assert batch.features.shape[2] == 80
    assert batch.feature_lengths.shape == (2,)
    assert batch.target_lengths.tolist() == [4, 4]
    assert batch.utt_ids == ["sid-1", "sid-2"]


def test_webdataset_rank_partition_uses_disjoint_shards(tmp_path: Path, monkeypatch) -> None:
    root = _build_root(tmp_path)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    dataset = WebDatasetASRIterableDataset(
        root,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(shuffle_shards=False),
    )
    samples = list(dataset)

    assert [sample["utt_id"] for sample in samples] == ["sid-3"]


def test_webdataset_defaults_to_whisper_tokenizer_for_text(tmp_path: Path, monkeypatch) -> None:
    _install_fake_whisper(monkeypatch)
    root = _build_root(tmp_path)
    dataset = WebDatasetASRIterableDataset(
        root,
        config=WebDatasetConfig(shuffle_shards=False),
    )

    sample = next(iter(dataset))

    assert sample["targets"].tolist() == [201, 202]


def test_webdataset_sample_level_split_filters_without_shard_split_bias(tmp_path: Path) -> None:
    split_config = StableHashSplitConfig(eval_ratio=0.5, hash_seed=7, split_by="sample_id")
    shard0: list[tuple[str, str, str]] = []
    shard1: list[tuple[str, str, str]] = []
    shard_counts = {
        "shard0": {"train": 0, "eval": 0},
        "shard1": {"train": 0, "eval": 0},
    }

    sample_idx = 0
    while min(shard_counts["shard0"].values()) == 0 or min(shard_counts["shard1"].values()) == 0:
        sid = f"sid-{sample_idx}"
        split_name = assign_split(sid, split_config)
        sample = (f"{sample_idx + 1:010d}", f"文本{sample_idx}", sid)
        target_shard = "shard0" if sample_idx % 2 == 0 else "shard1"
        if target_shard == "shard0":
            shard0.append(sample)
        else:
            shard1.append(sample)
        shard_counts[target_shard][split_name] += 1
        sample_idx += 1

    _write_shard(tmp_path, "shard_00000000.tar", shard0)
    _write_shard(tmp_path, "shard_00000001.tar", shard1)

    train_dataset = WebDatasetASRIterableDataset(
        tmp_path,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split="train",
            eval_ratio=split_config.eval_ratio,
            hash_seed=split_config.hash_seed,
            split_by="sample_id",
        ),
    )
    eval_dataset = WebDatasetASRIterableDataset(
        tmp_path,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split="eval",
            eval_ratio=split_config.eval_ratio,
            hash_seed=split_config.hash_seed,
            split_by="sample_id",
        ),
    )

    train_ids = [sample["utt_id"] for sample in train_dataset]
    eval_ids = [sample["utt_id"] for sample in eval_dataset]

    assert train_ids
    assert eval_ids
    assert set(train_ids).isdisjoint(eval_ids)
    assert set(train_ids) | set(eval_ids) == {sample[2] for sample in shard0 + shard1}


def test_inspect_webdataset_writes_index_with_shard_and_split_counts(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    index_path = tmp_path / "webdataset_index.json"

    result = inspect_webdataset(
        root,
        output_path=index_path,
        split_config=StableHashSplitConfig(eval_ratio=0.5, hash_seed=3, split_by="sample_id"),
    )

    loaded = load_webdataset_index(index_path)
    assert result["num_shards"] == 2
    assert result["num_samples"] == 3
    assert loaded["num_samples"] == 3
    assert sum(shard["num_samples"] for shard in loaded["shards"]) == 3
    assert loaded["splits"]["train"]["num_samples"] + loaded["splits"]["eval"]["num_samples"] == 3
    assert loaded["split"]["split_by"] == "sample_id"


def test_webdataset_shard_level_split_filters_whole_shards(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    split_config = StableHashSplitConfig(eval_ratio=0.5, hash_seed=3, split_by="shard_name")

    train_dataset = WebDatasetASRIterableDataset(
        root,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split="train",
            eval_ratio=split_config.eval_ratio,
            hash_seed=split_config.hash_seed,
            split_by="shard_name",
        ),
    )
    eval_dataset = WebDatasetASRIterableDataset(
        root,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split="eval",
            eval_ratio=split_config.eval_ratio,
            hash_seed=split_config.hash_seed,
            split_by="shard_name",
        ),
    )

    train_ids = [sample["utt_id"] for sample in train_dataset]
    eval_ids = [sample["utt_id"] for sample in eval_dataset]

    assert train_ids
    assert eval_ids
    assert set(train_ids).isdisjoint(eval_ids)
    assert set(train_ids) | set(eval_ids) == {"sid-1", "sid-2", "sid-3"}


def test_inspect_webdataset_records_shard_level_assignments(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    index_path = tmp_path / "webdataset_shard_index.json"

    result = inspect_webdataset(
        root,
        output_path=index_path,
        split_config=StableHashSplitConfig(eval_ratio=0.5, hash_seed=3, split_by="shard_name"),
    )

    loaded = load_webdataset_index(index_path)
    assert result["split"]["split_by"] == "shard_name"
    assert loaded["split"]["split_by"] == "shard_name"
    assert all(shard["assigned_split"] in {"train", "eval"} for shard in loaded["shards"])


def test_inspect_webdataset_lengths_writes_per_sample_entries(tmp_path: Path) -> None:
    root = _build_root(tmp_path)
    index_path = tmp_path / "webdataset_lengths.jsonl"
    summary_path = tmp_path / "webdataset_lengths.summary.json"

    summary = inspect_webdataset_lengths(
        root,
        output_path=index_path,
        summary_path=summary_path,
        split_config=StableHashSplitConfig(eval_ratio=0.5, hash_seed=3, split_by="sample_id"),
    )
    entries = load_webdataset_length_entries(index_path)

    assert summary["num_samples"] == 3
    assert summary_path.exists()
    assert len(entries) == 3
    assert all(entry.num_frames == 100 for entry in entries)
    assert all(entry.audio_offset is not None for entry in entries)
    assert all(entry.audio_size is not None for entry in entries)
    assert all(entry.json_offset is not None for entry in entries)
    assert all(entry.json_size is not None for entry in entries)


def test_webdataset_supports_mp3_audio_members_and_random_access_length_reads(tmp_path: Path) -> None:
    root = tmp_path / "mp3_root"
    root.mkdir()
    _write_mp3_named_shard(
        root,
        "shard_00000000.tar",
        [
            ("0000000001", "你好世界", "sid-1"),
            ("0000000002", "双向语音", "sid-2"),
        ],
    )
    index_path = tmp_path / "mp3_lengths.jsonl"
    inspect_webdataset_lengths(root, output_path=index_path)
    entries = load_webdataset_length_entries(index_path)

    assert len(entries) == 2
    assert all(entry.audio_format == "mp3" for entry in entries)

    loader, _ = build_length_bucketed_webdataset_dataloader(
        root,
        length_index_path=index_path,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(shuffle_shards=False, length_bucket_frame_budget=250),
        batch_size=2,
        num_workers=0,
    )
    batch = next(iter(loader))

    assert batch.features.shape[0] >= 1
    assert batch.features.shape[2] == 80


def test_length_bucketed_batch_sampler_splits_global_batches_across_ranks() -> None:
    lengths = [10, 11, 12, 13, 100, 101, 102, 103]
    rank0 = list(
        LengthBucketedBatchSampler(
            lengths,
            batch_size=2,
            rank=0,
            world_size=2,
            seed=7,
            shuffle=False,
        )
    )
    rank1 = list(
        LengthBucketedBatchSampler(
            lengths,
            batch_size=2,
            rank=1,
            world_size=2,
            seed=7,
            shuffle=False,
        )
    )

    assert len(rank0) == len(rank1) == 2
    for batch0, batch1 in zip(rank0, rank1, strict=True):
        combined = [lengths[idx] for idx in [*batch0, *batch1]]
        assert max(combined) - min(combined) <= 3


def test_length_bucketed_batch_sampler_dynamically_adjusts_batch_size_by_length() -> None:
    lengths = [100, 110, 120, 130, 500, 520, 540, 560]
    batches = list(
        LengthBucketedBatchSampler(
            lengths,
            batch_size=4,
            rank=0,
            world_size=1,
            seed=0,
            shuffle=False,
            frame_budget=500,
        )
    )

    batch_sizes = [len(batch) for batch in batches]
    batch_max_lengths = [max(lengths[idx] for idx in batch) for batch in batches]

    assert batch_sizes == [3, 1, 1, 1, 1, 1]
    assert batch_max_lengths == [120, 130, 500, 520, 540, 560]


def test_estimate_length_bucketed_steps_matches_dynamic_batches() -> None:
    lengths = [100, 110, 120, 130, 500, 520, 540, 560]
    steps = estimate_length_bucketed_steps(
        lengths,
        batch_size=4,
        world_size=1,
        frame_budget=500,
        drop_last=True,
    )

    assert steps == 6


def test_length_bucketed_epoch_covers_all_lengths_and_shuffles_batch_order() -> None:
    lengths = [10, 11, 12, 13, 100, 101, 102, 103]
    sampler = LengthBucketedBatchSampler(
        lengths,
        batch_size=2,
        rank=0,
        world_size=1,
        seed=7,
        shuffle=True,
    )
    batches = list(sampler)
    flattened = [idx for batch in batches for idx in batch]
    batch_max_lengths = [max(lengths[idx] for idx in batch) for batch in batches]

    assert sorted(flattened) == list(range(len(lengths)))
    assert batch_max_lengths != sorted(batch_max_lengths)


def test_length_bucketed_webdataset_dataloader_reads_similar_length_batches(tmp_path: Path) -> None:
    root = tmp_path / "bucketed"
    root.mkdir()
    shard_path = root / "shard_00000000.tar"
    with tarfile.open(shard_path, "w") as archive:
        for key, sid, seconds in [
            ("0000000001", "sid-1", 1.0),
            ("0000000002", "sid-2", 1.2),
            ("0000000003", "sid-3", 3.0),
            ("0000000004", "sid-4", 3.2),
        ]:
            wav_bytes = _make_wav_bytes(int(16000 * seconds))
            wav_info = tarfile.TarInfo(name=f"{key}.wav")
            wav_info.size = len(wav_bytes)
            archive.addfile(wav_info, io.BytesIO(wav_bytes))

            json_bytes = json.dumps(
                {
                    "text": sid,
                    "language": "zh",
                    "sample_rate": 16000,
                    "format": "wav",
                    "begin_time": 0.0,
                    "end_time": seconds,
                    "sid": sid,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{key}.json")
            json_info.size = len(json_bytes)
            archive.addfile(json_info, io.BytesIO(json_bytes))
    index_path = tmp_path / "bucketed_lengths.jsonl"
    inspect_webdataset_lengths(root, output_path=index_path)
    loader, _ = build_length_bucketed_webdataset_dataloader(
        root,
        length_index_path=index_path,
        tokenizer=DummyTokenizer(),
        config=WebDatasetConfig(shuffle_shards=False, length_bucket_frame_budget=250),
        batch_size=4,
        num_workers=0,
    )

    batches = list(iter(loader))
    batch = batches[0]

    assert batch.features.size(0) == 2
    assert max(batch.feature_lengths.tolist()) - min(batch.feature_lengths.tolist()) <= 20
    assert [candidate.features.size(0) for candidate in batches] == [2, 1, 1]


def test_bucket_manifest_step_estimation_and_loading(tmp_path: Path) -> None:
    root = tmp_path / "bucket_manifest_root"
    root.mkdir()
    root = _build_root(root)
    manifest_path = _write_bucket_manifest(
        tmp_path,
        root=root,
        entries_by_part=[
            (
                "train/bucket_0000/part_000000.jsonl",
                [
                    {
                        "shard_name": "shard_00000000.tar",
                        "key": "0000000001",
                        "utt_id": "sid-1",
                        "split": "train",
                        "num_frames": 40,
                        "audio_member": "0000000001.wav",
                        "audio_format": "wav",
                        "json_member": "0000000001.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    },
                    {
                        "shard_name": "shard_00000000.tar",
                        "key": "0000000002",
                        "utt_id": "sid-2",
                        "split": "train",
                        "num_frames": 50,
                        "audio_member": "0000000002.wav",
                        "audio_format": "wav",
                        "json_member": "0000000002.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    },
                ],
            ),
            (
                "train/bucket_0001/part_000000.jsonl",
                [
                    {
                        "shard_name": "shard_00000001.tar",
                        "key": "0000000003",
                        "utt_id": "sid-3",
                        "split": "train",
                        "num_frames": 120,
                        "audio_member": "0000000003.wav",
                        "audio_format": "wav",
                        "json_member": "0000000003.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    },
                ],
            ),
        ],
    )

    manifest = load_webdataset_bucket_manifest(manifest_path)

    assert estimate_bucket_manifest_steps(
        manifest,
        split="train",
        batch_size=4,
        world_size=1,
        frame_budget=160,
        drop_last=True,
    ) == 2


def test_bucketed_webdataset_loader_splits_same_bucket_across_ranks(tmp_path: Path) -> None:
    root = tmp_path / "bucket_loader_root"
    root.mkdir()
    root = _build_root(root)
    manifest_path = _write_bucket_manifest(
        tmp_path,
        root=root,
        entries_by_part=[
            (
                "train/bucket_0000/part_000000.jsonl",
                [
                    {
                        "shard_name": "shard_00000000.tar",
                        "key": "0000000001",
                        "utt_id": "sid-1",
                        "split": "train",
                        "num_frames": 40,
                        "audio_member": "0000000001.wav",
                        "audio_format": "wav",
                        "json_member": "0000000001.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    },
                    {
                        "shard_name": "shard_00000000.tar",
                        "key": "0000000002",
                        "utt_id": "sid-2",
                        "split": "train",
                        "num_frames": 50,
                        "audio_member": "0000000002.wav",
                        "audio_format": "wav",
                        "json_member": "0000000002.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    },
                ],
            ),
        ],
    )

    config = WebDatasetConfig(
        shuffle_shards=False,
        split="train",
        length_bucket_frame_budget=80,
    )
    loader_rank0 = build_bucketed_webdataset_loader(
        root,
        bucket_manifest_path=manifest_path,
        tokenizer=DummyTokenizer(),
        config=config,
        batch_size=4,
        num_workers=2,
        rank=0,
        world_size=2,
    )
    loader_rank1 = build_bucketed_webdataset_loader(
        root,
        bucket_manifest_path=manifest_path,
        tokenizer=DummyTokenizer(),
        config=config,
        batch_size=4,
        num_workers=2,
        rank=1,
        world_size=2,
    )

    batch0 = next(iter(loader_rank0))
    batch1 = next(iter(loader_rank1))

    assert batch0.utt_ids == ["sid-1"]
    assert batch1.utt_ids == ["sid-2"]


def test_bucket_entry_stream_closes_part_handle_between_takes(tmp_path: Path) -> None:
    bucket_root = tmp_path / "webdataset_buckets"
    bucket_root.mkdir()
    part_path = bucket_root / "bucket-000000.jsonl"
    with part_path.open("w", encoding="utf-8") as handle:
        for idx in range(3):
            key = f"{idx:010d}"
            handle.write(
                json.dumps(
                    {
                        "shard_name": "shard_00000000.tar",
                        "key": key,
                        "utt_id": key,
                        "split": "train",
                        "num_frames": 40,
                        "audio_member": f"{key}.wav",
                        "audio_format": "wav",
                        "json_member": f"{key}.json",
                        "audio_offset": None,
                        "audio_size": None,
                        "json_offset": None,
                        "json_size": None,
                    }
                )
                + "\n"
            )
    manifest_path = bucket_root / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    stream = _BucketEntryStream(
        manifest_path,
        WebDatasetBucket(
            split="train",
            bucket_id=0,
            num_samples=3,
            parts=(
                WebDatasetBucketPart(
                    path=part_path.name,
                    num_samples=3,
                    first_shard="shard_00000000.tar",
                    last_shard="shard_00000000.tar",
                ),
            ),
        ),
    )

    first = stream.take(1)
    assert [entry.key for entry in first] == ["0000000000"]
    assert stream._handle is None

    rest = stream.take(2)
    assert [entry.key for entry in rest] == ["0000000001", "0000000002"]
    assert stream._handle is None
    assert stream.take(1) == []


def test_thread_local_tar_reader_pool_lru_closes_old_shards(tmp_path: Path) -> None:
    readers = []
    pool = _ThreadLocalTarReaderPool(tmp_path, max_open_shards_per_worker=2)
    for idx in range(5):
        shard_name = f"shard_{idx:08d}.tar"
        (tmp_path / shard_name).write_bytes(b"x")
        reader = pool.get(shard_name)
        reader._binary_handle()
        readers.append(reader)

    assert [reader.is_open for reader in readers] == [False, False, False, True, True]
    assert list(pool._local.readers.keys()) == ["shard_00000003.tar", "shard_00000004.tar"]

    pool.close()
    assert not any(reader.is_open for reader in readers)
