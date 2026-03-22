from __future__ import annotations

import argparse

from rwkvasr.data import StableHashSplitConfig, inspect_webdataset_lengths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a per-sample WebDataset length index from tar metadata.")
    parser.add_argument("--webdataset-root", required=True)
    parser.add_argument("--shard-pattern", default="*.tar")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--utt-id-key", default="sid")
    parser.add_argument("--split-by", default="shard_name", choices=("sample_id", "shard_name"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = inspect_webdataset_lengths(
        args.webdataset_root,
        shard_pattern=args.shard_pattern,
        split_config=StableHashSplitConfig(
            eval_ratio=args.eval_ratio,
            hash_seed=args.hash_seed,
            utt_id_key=args.utt_id_key,
            split_by=args.split_by,
        ),
        output_path=args.output_path,
        summary_path=args.summary_path,
    )
    print(
        "inspect_webdataset_lengths "
        f"samples={summary['num_samples']} "
        f"train={summary['splits']['train']['num_samples']} "
        f"eval={summary['splits']['eval']['num_samples']}"
    )


if __name__ == "__main__":
    main()
