from __future__ import annotations

import hashlib
import json
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_SPLIT_BY = {"sample_id", "shard_name"}


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


@dataclass(frozen=True)
class StableHashSplitConfig:
    eval_ratio: float = 0.01
    hash_seed: int = 0
    train_name: str = "train"
    eval_name: str = "eval"
    utt_id_key: str = "sid"
    split_by: str = "shard_name"

    def __post_init__(self) -> None:
        if not 0.0 <= self.eval_ratio < 1.0:
            raise ValueError("eval_ratio must be within [0, 1).")
        if not self.train_name or not self.eval_name:
            raise ValueError("train_name and eval_name must be non-empty.")
        if self.train_name == self.eval_name:
            raise ValueError("train_name and eval_name must be different.")
        if not self.utt_id_key:
            raise ValueError("utt_id_key must be non-empty.")
        if self.split_by not in VALID_SPLIT_BY:
            raise ValueError(f"split_by must be one of {sorted(VALID_SPLIT_BY)!r}.")


def resolve_sample_id(key: str, metadata: dict[str, Any], *, utt_id_key: str) -> str:
    sample_id = metadata.get(utt_id_key) or key
    return str(sample_id)


def assign_split(sample_id: str, config: StableHashSplitConfig) -> str:
    if config.eval_ratio <= 0.0:
        return config.train_name
    digest = hashlib.sha1(f"{config.hash_seed}:{sample_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)
    if bucket < config.eval_ratio:
        return config.eval_name
    return config.train_name


def shard_in_split(shard_name: str, split: str, config: StableHashSplitConfig) -> bool:
    if split == "all":
        return True
    if split not in {config.train_name, config.eval_name}:
        raise ValueError(f"Unsupported split {split!r}; expected 'all', {config.train_name!r}, or {config.eval_name!r}.")
    return assign_split(shard_name, config) == split


def sample_in_split(
    sample_id: str,
    split: str,
    config: StableHashSplitConfig,
    *,
    shard_name: str | None = None,
) -> bool:
    if split == "all":
        return True
    if split not in {config.train_name, config.eval_name}:
        raise ValueError(f"Unsupported split {split!r}; expected 'all', {config.train_name!r}, or {config.eval_name!r}.")
    if config.split_by == "sample_id":
        return assign_split(sample_id, config) == split
    if shard_name is None:
        raise ValueError("shard_name is required when split_by='shard_name'.")
    return assign_split(shard_name, config) == split


def resolve_webdataset_index_path(shard_root: str | Path, index_path: str | Path | None = None) -> Path:
    if index_path is not None:
        return Path(index_path)
    shard_root = Path(shard_root)
    if shard_root.is_dir():
        return shard_root / "webdataset_index.json"
    return shard_root.with_suffix(".index.json")


def load_webdataset_index(index_path: str | Path) -> dict[str, Any]:
    index_path = Path(index_path)
    with index_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {index_path}, got {type(data)}")
    return data


def validate_webdataset_index(
    index_data: dict[str, Any],
    *,
    split_config: StableHashSplitConfig,
) -> None:
    split = index_data.get("split", {})
    index_split_by = str(split.get("split_by", "sample_id"))
    if index_split_by != split_config.split_by:
        raise ValueError(
            f"WebDataset index split_by={index_split_by!r} does not match requested split_by={split_config.split_by!r}."
        )
    index_eval_ratio = float(split.get("eval_ratio", 0.0))
    if index_eval_ratio != split_config.eval_ratio:
        raise ValueError(
            "WebDataset index eval_ratio does not match the requested eval_ratio: "
            f"{index_eval_ratio} vs {split_config.eval_ratio}."
        )
    index_hash_seed = int(split.get("hash_seed", 0))
    if index_hash_seed != split_config.hash_seed:
        raise ValueError(
            "WebDataset index hash_seed does not match the requested hash_seed: "
            f"{index_hash_seed} vs {split_config.hash_seed}."
        )
    index_utt_id_key = str(split.get("utt_id_key", "sid"))
    if index_utt_id_key != split_config.utt_id_key:
        raise ValueError(
            "WebDataset index utt_id_key does not match the requested utt_id_key: "
            f"{index_utt_id_key!r} vs {split_config.utt_id_key!r}."
        )


def index_split_sample_count(index_data: dict[str, Any], split: str) -> int:
    if split == "all":
        return int(index_data["num_samples"])
    splits = index_data.get("splits", {})
    if split not in splits:
        raise KeyError(f"Split {split!r} not found in WebDataset index.")
    return int(splits[split]["num_samples"])


def inspect_webdataset(
    shard_root: str | Path,
    *,
    shard_pattern: str = "*.tar",
    split_config: StableHashSplitConfig | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    shard_root = Path(shard_root)
    split_config = split_config or StableHashSplitConfig()
    if shard_root.is_file():
        shards = [shard_root]
    else:
        shards = sorted(shard_root.glob(shard_pattern))
    if not shards:
        raise FileNotFoundError(f"No shard files matching {shard_pattern!r} under {shard_root}")

    split_counts: dict[str, int] = {
        split_config.train_name: 0,
        split_config.eval_name: 0,
    }
    shard_entries: list[dict[str, Any]] = []
    total_samples = 0
    start_time = time.monotonic()
    _log(f"Inspecting WebDataset under {shard_root} with {len(shards)} shard(s).")

    for shard_idx, shard_path in enumerate(shards, 1):
        pending: dict[str, dict[str, Any]] = {}
        shard_split_counts = {
            split_config.train_name: 0,
            split_config.eval_name: 0,
        }
        shard_samples = 0
        shard_assigned_split: str | None = None
        if split_config.split_by == "shard_name":
            shard_assigned_split = assign_split(shard_path.name, split_config)

        with tarfile.open(shard_path, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if "." not in name:
                    continue
                key, suffix = name.rsplit(".", 1)
                suffix = suffix.lower()
                if suffix not in {"wav", "json"}:
                    continue

                sample = pending.setdefault(key, {"has_wav": False, "metadata_bytes": None})
                if suffix == "wav":
                    sample["has_wav"] = True
                else:
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    sample["metadata_bytes"] = extracted.read()

                if sample["has_wav"] and sample["metadata_bytes"] is not None:
                    if split_config.split_by == "sample_id":
                        metadata = json.loads(sample["metadata_bytes"].decode("utf-8"))
                        sample_id = resolve_sample_id(key, metadata, utt_id_key=split_config.utt_id_key)
                        split_name = assign_split(sample_id, split_config)
                    else:
                        split_name = shard_assigned_split or split_config.train_name
                    shard_samples += 1
                    total_samples += 1
                    shard_split_counts[split_name] += 1
                    split_counts[split_name] += 1
                    pending.pop(key, None)

        shard_entries.append(
            {
                "name": shard_path.name,
                "num_samples": shard_samples,
                "assigned_split": shard_assigned_split,
                "splits": {
                    split_config.train_name: {"num_samples": shard_split_counts[split_config.train_name]},
                    split_config.eval_name: {"num_samples": shard_split_counts[split_config.eval_name]},
                },
            }
        )
        elapsed = time.monotonic() - start_time
        _log(
            f"Index progress: shards={shard_idx}/{len(shards)}, "
            f"samples={total_samples}, elapsed={elapsed:.1f}s, current={shard_path.name}"
        )

    index_data = {
        "version": 1,
        "root": str(shard_root),
        "shard_pattern": shard_pattern,
        "num_shards": len(shards),
        "num_samples": total_samples,
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
        "shards": shard_entries,
    }

    output_file = resolve_webdataset_index_path(shard_root, output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(index_data, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    _log(
        f"Finished WebDataset inspection: shards={len(shards)}, samples={total_samples}, output={output_file}"
    )
    return index_data
