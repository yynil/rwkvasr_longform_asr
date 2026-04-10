from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from rwkvasr.data import (
    ASRManifestDataset,
    StableHashSplitConfig,
    WebDatasetASRIterableDataset,
    WebDatasetConfig,
    accumulate_global_cmvn_stats_from_samples,
    index_split_sample_count,
    load_webdataset_index,
    resolve_webdataset_index_path,
    validate_webdataset_index,
    write_wenet_cmvn_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute WeNet-style global CMVN stats from a manifest or WebDataset.")
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--webdataset-root", default=None)
    parser.add_argument("--webdataset-split", default="all")
    parser.add_argument("--webdataset-eval-ratio", default=0.0, type=float)
    parser.add_argument("--webdataset-hash-seed", default=0, type=int)
    parser.add_argument("--webdataset-split-by", default="shard_name")
    parser.add_argument("--webdataset-utt-id-key", default="sid")
    parser.add_argument("--output-path", required=True)
    return parser


def _wrap_with_progress(
    samples: Iterable[dict[str, object]],
    *,
    progress: Progress,
    task_id: int,
) -> Iterable[dict[str, object]]:
    for sample in samples:
        yield sample
        progress.advance(task_id, 1)


def _resolve_webdataset_total_samples(
    root: str,
    *,
    split: str,
    eval_ratio: float,
    hash_seed: int,
    split_by: str,
    utt_id_key: str,
) -> int | None:
    index_path = resolve_webdataset_index_path(root)
    if not index_path.exists():
        return None
    index_data = load_webdataset_index(index_path)
    validate_webdataset_index(
        index_data,
        split_config=StableHashSplitConfig(
            eval_ratio=eval_ratio,
            hash_seed=hash_seed,
            utt_id_key=utt_id_key,
            split_by=split_by,
        ),
    )
    return index_split_sample_count(index_data, split)


def main() -> None:
    args = build_parser().parse_args()
    has_manifest = args.manifest_path is not None
    has_webdataset = args.webdataset_root is not None
    if has_manifest == has_webdataset:
        raise ValueError("Exactly one of --manifest-path or --webdataset-root must be provided.")

    if has_manifest:
        dataset = ASRManifestDataset(args.manifest_path)
        total_samples = len(dataset)
        label = f"CMVN manifest {Path(args.manifest_path).name}"
    else:
        dataset = WebDatasetASRIterableDataset(
            args.webdataset_root,
            config=WebDatasetConfig(
                shuffle_shards=False,
                split=args.webdataset_split,
                eval_ratio=args.webdataset_eval_ratio,
                hash_seed=args.webdataset_hash_seed,
                split_by=args.webdataset_split_by,
                utt_id_key=args.webdataset_utt_id_key,
                partition_by_rank=False,
            ),
        )
        total_samples = _resolve_webdataset_total_samples(
            args.webdataset_root,
            split=args.webdataset_split,
            eval_ratio=args.webdataset_eval_ratio,
            hash_seed=args.webdataset_hash_seed,
            split_by=args.webdataset_split_by,
            utt_id_key=args.webdataset_utt_id_key,
        )
        label = f"CMVN webdataset {Path(args.webdataset_root).name}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(label, total=total_samples)
        stats = accumulate_global_cmvn_stats_from_samples(
            _wrap_with_progress(dataset, progress=progress, task_id=task_id),
            total_samples=total_samples,
            progress_label=label,
            progress_every_samples=0,
        )

    write_wenet_cmvn_json(args.output_path, stats)
    print(
        f"compute_cmvn frame_num={stats.frame_num} "
        f"feat_dim={stats.mean_stat.numel()} output={args.output_path}"
    )


if __name__ == "__main__":
    main()
