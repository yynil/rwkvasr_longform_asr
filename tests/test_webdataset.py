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
    assign_split,
    estimate_length_bucketed_steps,
    build_webdataset_dataloader,
    build_length_bucketed_webdataset_dataloader,
    inspect_webdataset,
    inspect_webdataset_lengths,
    load_webdataset_index,
    load_webdataset_length_entries,
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
