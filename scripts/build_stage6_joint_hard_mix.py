#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_ROOT = "/media/usbhd/training_data/asr/curriculum/stage6_joint_hard_mix"
DEFAULT_CLEAN_ROOT = "/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset"
DEFAULT_CLEAN_LENGTHS = (
    "/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/"
    "stages/difficulty_sensevoice/stage5_replay_balanced/webdataset_lengths.jsonl"
)
DEFAULT_HARD_ROOT = "/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset"
DEFAULT_HARD_LENGTHS = (
    "/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset/"
    "webdataset_lengths.jsonl"
)


@dataclass
class Reservoir:
    size: int
    rng: random.Random
    seen: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        if self.size <= 0:
            return
        self.seen += 1
        if len(self.rows) < self.size:
            self.rows.append(row)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            self.rows[index] = row


def _log(message: str) -> None:
    print(f"[rwkvasr-stage6-mix] {message}", flush=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _source_from_shard(shard_name: str) -> str:
    if shard_name.startswith("GSXL-"):
        return "gigaspeech"
    if shard_name.startswith("WSL-"):
        return "wenetspeech"
    if shard_name.startswith("librispeech_"):
        return "librispeech"
    if shard_name.startswith("aishell3_"):
        return "aishell3"
    if shard_name.startswith("commonvoice_en_"):
        return "commonvoice_en"
    if shard_name.startswith("commonvoice_cn_"):
        return "commonvoice_cn"
    return shard_name.split("_", 1)[0].split("-", 1)[0] or "unknown"


def _hard_source(row: dict[str, Any]) -> str | None:
    source = str(row.get("source_dataset") or "")
    if source in {"gigaspeech", "wenetspeech"}:
        return source
    shard_name = str(row.get("shard_name") or "")
    inferred = _source_from_shard(shard_name)
    if inferred in {"gigaspeech", "wenetspeech"}:
        return inferred
    return None


def _read_clean_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(path):
        split = str(row.get("split") or "train")
        rows[split].append(row)
    return rows


def _sample_clean_rows(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    split: str,
    count: int,
    rng: random.Random,
    with_replacement: bool = False,
) -> list[dict[str, Any]]:
    rows = rows_by_split.get(split, [])
    if count <= 0 or not rows:
        return []
    if with_replacement:
        chosen = [rng.choice(rows) for _ in range(count)]
    elif count >= len(rows):
        chosen = list(rows)
    else:
        chosen = rng.sample(rows, count)
    result: list[dict[str, Any]] = []
    for index, row in enumerate(chosen):
        item = dict(row)
        item["_stage6_mix_component"] = "clean_replay"
        if with_replacement:
            item["_stage6_clean_replay_index"] = index
        result.append(item)
    return result


def _sample_hard_rows(
    hard_lengths: Path,
    quotas: dict[tuple[str, str], int],
    *,
    seed: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    reservoirs = {
        key: Reservoir(size=size, rng=random.Random(seed + index * 9973))
        for index, (key, size) in enumerate(sorted(quotas.items()))
    }
    processed = 0
    for row in _iter_jsonl(hard_lengths):
        processed += 1
        split = str(row.get("split") or "train")
        source = _hard_source(row)
        if source is None:
            continue
        reservoir = reservoirs.get((split, source))
        if reservoir is None:
            continue
        item = dict(row)
        item.setdefault("source_dataset", source)
        item.setdefault("language", "en" if source == "gigaspeech" else "zh")
        item["_stage6_mix_component"] = f"hard_{source}"
        reservoir.add(item)
        if processed % 1_000_000 == 0:
            _log(f"hard sampling progress rows={processed}")
    sampled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, reservoir in reservoirs.items():
        sampled[key] = reservoir.rows
        if len(reservoir.rows) < reservoir.size:
            _log(f"warning: quota short key={key} wanted={reservoir.size} got={len(reservoir.rows)}")
    return sampled


def _sample_hard_rows_for_stages(
    hard_lengths: Path,
    stage_quotas: dict[str, dict[tuple[str, str], int]],
    *,
    seed: int,
) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
    reservoirs: dict[tuple[str, str, str], Reservoir] = {}
    for stage_index, (stage_name, quotas) in enumerate(sorted(stage_quotas.items())):
        for quota_index, (key, size) in enumerate(sorted(quotas.items())):
            split, source = key
            reservoirs[(stage_name, split, source)] = Reservoir(
                size=size,
                rng=random.Random(seed + stage_index * 1000003 + quota_index * 9973),
            )

    processed = 0
    for row in _iter_jsonl(hard_lengths):
        processed += 1
        split = str(row.get("split") or "train")
        source = _hard_source(row)
        if source is None:
            continue
        for stage_name in stage_quotas:
            reservoir = reservoirs.get((stage_name, split, source))
            if reservoir is None:
                continue
            item = dict(row)
            item.setdefault("source_dataset", source)
            item.setdefault("language", "en" if source == "gigaspeech" else "zh")
            item["_stage6_mix_component"] = f"hard_{source}"
            reservoir.add(item)
        if processed % 1_000_000 == 0:
            _log(f"hard sampling progress rows={processed}")

    sampled: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = defaultdict(dict)
    for (stage_name, split, source), reservoir in reservoirs.items():
        sampled[stage_name][(split, source)] = reservoir.rows
        if len(reservoir.rows) < reservoir.size:
            _log(
                "warning: quota short "
                f"stage={stage_name} key={(split, source)} wanted={reservoir.size} got={len(reservoir.rows)}"
            )
    return sampled


def _stage_quotas(
    *,
    target_samples: int,
    eval_ratio: float,
    clean_ratio: float,
) -> dict[str, dict[str, int]]:
    eval_count = max(1, int(round(target_samples * eval_ratio)))
    train_count = max(1, int(target_samples) - eval_count)
    quotas: dict[str, dict[str, int]] = {}
    for split, total in (("train", train_count), ("eval", eval_count)):
        clean = int(round(total * clean_ratio))
        hard = max(0, total - clean)
        giga = hard // 2
        wenet = hard - giga
        quotas[split] = {
            "clean": clean,
            "gigaspeech": giga,
            "wenetspeech": wenet,
        }
    return quotas


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        split = str(row.get("split") or "unknown")
        component = str(row.get("_stage6_mix_component") or "unknown")
        source = str(row.get("source_dataset") or _source_from_shard(str(row.get("shard_name") or "")))
        language = str(row.get("language") or "unknown")
        counters["split"][split] += 1
        counters["component"][component] += 1
        counters["source"][source] += 1
        counters["language"][language] += 1
        counters["split_component"][f"{split}/{component}"] += 1
        counters["split_source"][f"{split}/{source}"] += 1
    return {
        "num_samples": len(rows),
        "counts": {name: dict(sorted(counter.items())) for name, counter in sorted(counters.items())},
    }


def _link_shards(source_root: Path, output_root: Path) -> int:
    linked = 0
    for source in source_root.glob("*.tar"):
        dest = output_root / source.name
        if dest.exists() or dest.is_symlink():
            existing = dest.resolve()
            if existing == source.resolve():
                continue
            raise FileExistsError(f"refusing to replace existing shard {dest} -> {existing}")
        os.symlink(source, dest)
        linked += 1
    return linked


def _combine_indexes(
    clean_root: Path,
    hard_root: Path,
    output_root: Path,
    *,
    eval_ratio: float,
    hash_seed: int,
    split_by: str,
    utt_id_key: str,
    target_samples: int,
) -> None:
    shards: list[dict[str, Any]] = []
    total = 0
    for root in (clean_root, hard_root):
        index_path = root / "webdataset_index.json"
        if not index_path.exists():
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        for shard in payload.get("shards", []):
            item = dict(shard)
            item["source_root"] = str(root)
            shards.append(item)
            total += int(item.get("num_samples") or 0)
    output = {
        "version": 1,
        "root": str(output_root),
        "num_shards": len(shards),
        "num_samples": total,
        "split": {
            "type": "precomputed_in_length_indexes",
            "split_by": split_by,
            "eval_ratio": float(eval_ratio),
            "hash_seed": int(hash_seed),
            "utt_id_key": utt_id_key,
            "train_name": "train",
            "eval_name": "eval",
        },
        "splits": {
            "train": {
                "num_samples": max(
                    1,
                    int(target_samples) - max(1, int(round(int(target_samples) * float(eval_ratio)))),
                )
            },
            "eval": {
                "num_samples": max(1, int(round(int(target_samples) * float(eval_ratio))))
            },
        },
        "shards": shards,
    }
    (output_root / "webdataset_index.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _build_bucket_manifest(repo_root: Path, root: Path, stage_dir: Path, bucket_width: int) -> None:
    bucket_dir = stage_dir / "webdataset_buckets_audio_text"
    manifest = bucket_dir / "manifest.json"
    command = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        str(repo_root / "tools/Cargo.toml"),
        "--bin",
        "build_bucket_index",
        "--",
        "--shard-root",
        str(root),
        "--length-index-path",
        str(stage_dir / "webdataset_lengths.jsonl"),
        "--output-dir",
        str(bucket_dir),
        "--manifest-path",
        str(manifest),
        "--bucket-width",
        str(bucket_width),
        "--text-cost-source",
        "auto",
        "--text-cost-weight",
        "4",
        "--json-size-text-offset",
        "256",
        "--json-size-bytes-per-token",
        "4.0",
        "--entries-per-part",
        "100000",
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _parse_stage(value: str) -> tuple[str, float]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("stage must use NAME:CLEAN_RATIO")
    name, ratio = value.split(":", 1)
    clean_ratio = float(ratio)
    if not 0.0 <= clean_ratio <= 1.0:
        raise argparse.ArgumentTypeError("clean ratio must be in [0, 1]")
    return name, clean_ratio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sampled clean+GigaSpeech/WenetSpeech hard-mix WebDataset stages for joint training."
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clean-root", default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--clean-length-index-path", default=DEFAULT_CLEAN_LENGTHS)
    parser.add_argument("--hard-root", default=DEFAULT_HARD_ROOT)
    parser.add_argument("--hard-length-index-path", default=DEFAULT_HARD_LENGTHS)
    parser.add_argument("--target-samples", type=int, default=131012)
    parser.add_argument("--eval-ratio", type=float, default=0.005)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--split-by", default="sample_id")
    parser.add_argument("--utt-id-key", default="id")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--bucket-width", type=int, default=80)
    parser.add_argument(
        "--stage",
        action="append",
        type=_parse_stage,
        default=[],
        help="Stage spec NAME:CLEAN_RATIO. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-buckets", action="store_true")
    parser.add_argument(
        "--clean-with-replacement",
        action="store_true",
        help="Sample clean replay rows with replacement when the requested clean quota exceeds the source index size.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    clean_root = Path(args.clean_root)
    hard_root = Path(args.hard_root)
    clean_lengths = Path(args.clean_length_index_path)
    hard_lengths = Path(args.hard_length_index_path)
    stages: list[tuple[str, float]] = args.stage or [
        ("stage6b_clean70_hard30", 0.70),
        ("stage6c_clean50_hard50", 0.50),
        ("stage6d_clean30_hard70", 0.30),
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    linked = _link_shards(clean_root, output_root) + _link_shards(hard_root, output_root)
    _log(f"linked new shards={linked} root={output_root}")
    _combine_indexes(
        clean_root,
        hard_root,
        output_root,
        eval_ratio=float(args.eval_ratio),
        hash_seed=int(args.hash_seed),
        split_by=str(args.split_by),
        utt_id_key=str(args.utt_id_key),
        target_samples=int(args.target_samples),
    )

    clean_rows = _read_clean_rows(clean_lengths)
    stage_plans: list[tuple[int, str, float, dict[str, dict[str, int]]]] = []
    hard_stage_quotas: dict[str, dict[tuple[str, str], int]] = {}
    for stage_index, (stage_name, clean_ratio) in enumerate(stages):
        stage_dir = output_root / "stages" / stage_name
        lengths_path = stage_dir / "webdataset_lengths.jsonl"
        summary_path = stage_dir / "webdataset_lengths.summary.json"
        bucket_manifest = stage_dir / "webdataset_buckets_audio_text" / "manifest.json"
        if (
            not args.overwrite
            and lengths_path.exists()
            and summary_path.exists()
            and (args.skip_buckets or bucket_manifest.exists())
        ):
            _log(f"stage already exists, skipping stage={stage_name}")
            continue

        quotas = _stage_quotas(
            target_samples=int(args.target_samples),
            eval_ratio=float(args.eval_ratio),
            clean_ratio=float(clean_ratio),
        )
        stage_plans.append((stage_index, stage_name, clean_ratio, quotas))
        hard_stage_quotas[stage_name] = {
            (split, source): count
            for split, split_quotas in quotas.items()
            for source, count in split_quotas.items()
            if source in {"gigaspeech", "wenetspeech"} and count > 0
        }

    hard_rows_by_stage = _sample_hard_rows_for_stages(
        hard_lengths,
        hard_stage_quotas,
        seed=int(args.seed),
    ) if stage_plans else {}

    for stage_index, stage_name, clean_ratio, quotas in stage_plans:
        stage_dir = output_root / "stages" / stage_name
        lengths_path = stage_dir / "webdataset_lengths.jsonl"
        summary_path = stage_dir / "webdataset_lengths.summary.json"
        bucket_manifest = stage_dir / "webdataset_buckets_audio_text" / "manifest.json"
        hard_rows = hard_rows_by_stage.get(stage_name, {})
        if not hard_rows:
            hard_rows = _sample_hard_rows(
                hard_lengths,
                hard_stage_quotas.get(stage_name, {}),
                seed=int(args.seed) + stage_index * 1000003,
            )
        rng = random.Random(int(args.seed) + stage_index * 1009)
        rows: list[dict[str, Any]] = []
        for split, split_quotas in quotas.items():
            rows.extend(
                _sample_clean_rows(
                    clean_rows,
                    split=split,
                    count=int(split_quotas["clean"]),
                    rng=rng,
                    with_replacement=bool(args.clean_with_replacement),
                )
            )
            for source in ("gigaspeech", "wenetspeech"):
                rows.extend(hard_rows.get((split, source), []))
        rng.shuffle(rows)
        _write_jsonl(lengths_path, rows)
        summary = {
            "version": 1,
            "stage_name": stage_name,
            "output_root": str(output_root),
            "clean_root": str(clean_root),
            "hard_root": str(hard_root),
            "clean_ratio": clean_ratio,
            "hard_ratio": 1.0 - clean_ratio,
            "target_samples": int(args.target_samples),
            "eval_ratio": float(args.eval_ratio),
            "seed": int(args.seed),
            "clean_with_replacement": bool(args.clean_with_replacement),
            "quotas": quotas,
            "length_index_path": str(lengths_path),
            "bucket_manifest_path": str(bucket_manifest),
            **_summarize_rows(rows),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _log(f"wrote stage={stage_name} samples={len(rows)} lengths={lengths_path}")
        if not args.skip_buckets:
            _build_bucket_manifest(repo_root, output_root, stage_dir, int(args.bucket_width))
            _log(f"built bucket manifest stage={stage_name} path={bucket_manifest}")


if __name__ == "__main__":
    main()
