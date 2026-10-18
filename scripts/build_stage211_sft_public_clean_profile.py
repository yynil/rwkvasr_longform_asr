#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rwkvasr.eval.stage211_sft_public_clean import (
    build_stage211_sft_public_clean_profile,
    validate_stage211_sft_public_clean_rebuild_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path.home() / "rwkvasr_data/stage211_sft_full_labeled_v2"
DEFAULT_SOURCE_PROFILE = DEFAULT_SOURCE_ROOT / "stage211_labeled_profile_receipt.json"
DEFAULT_REJECTED_OVERLAP = (
    Path.home() / "rwkvasr_data/stage211_sft_public_encoded_overlap_v1/receipt.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "rwkvasr_data/stage211_sft_full_labeled_v3_public_clean"
)
DEFAULT_BUILDER = REPO_ROOT / "tools/target/release/build_bucket_index"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the Stage211D labeled metadata without exact public-audio matches."
    )
    parser.add_argument("--source-profile", type=Path, default=DEFAULT_SOURCE_PROFILE)
    parser.add_argument("--rejected-overlap-receipt", type=Path, default=DEFAULT_REJECTED_OVERLAP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    receipt_path = (
        args.output_root.expanduser().resolve()
        / "public_overlap_exclusion_rebuild_receipt.json"
    )
    if args.validate_only:
        receipt = validate_stage211_sft_public_clean_rebuild_receipt(receipt_path)
    else:
        receipt = build_stage211_sft_public_clean_profile(
            source_profile_path=args.source_profile,
            overlap_receipt_path=args.rejected_overlap_receipt,
            output_root=args.output_root,
            repo_root=REPO_ROOT,
            builder_path=args.builder,
        )
    print(
        "[stage211-sft-public-clean] "
        f"source_rows={receipt['coverage']['source_rows']} "
        f"excluded={receipt['coverage']['excluded_rows']} "
        f"retained={receipt['coverage']['retained_rows']} "
        f"profile={receipt['output_profile_path']} "
        f"receipt={receipt_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
