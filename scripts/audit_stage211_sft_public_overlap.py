#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rwkvasr.eval.stage211_sft_public_overlap import (
    build_stage211_sft_public_overlap_audit,
    validate_stage211_sft_public_overlap_receipt,
)

try:
    from scripts.create_stage211_labeled_profile_receipt import (
        validate_receipt as validate_labeled_profile_receipt,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script fallback
    from create_stage211_labeled_profile_receipt import (  # type: ignore[no-redef]
        validate_receipt as validate_labeled_profile_receipt,
    )


DEFAULT_LABELED_ROOT = Path.home() / "rwkvasr_data/stage211_sft_full_labeled_v2"
DEFAULT_LABELED_PROFILE = DEFAULT_LABELED_ROOT / "stage211_labeled_profile_receipt.json"
DEFAULT_NANO_PROVENANCE = (
    Path.home() / "rwkvasr_eval/stage211_public_full/nano_2512/provenance_receipt.json"
)
DEFAULT_PUBLIC_PCM_AUDIT = (
    Path.home() / "rwkvasr_data/stage211_base_public_pcm_overlap_v2/audit_receipt.json"
)
DEFAULT_OUTPUT_DIR = Path.home() / "rwkvasr_data/stage211_sft_public_encoded_overlap_v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit full Stage211D labeled audio against all five public datasets."
    )
    parser.add_argument("--labeled-profile", type=Path, default=DEFAULT_LABELED_PROFILE)
    parser.add_argument("--nano-provenance", type=Path, default=DEFAULT_NANO_PROVENANCE)
    parser.add_argument("--public-pcm-audit", type=Path, default=DEFAULT_PUBLIC_PCM_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    profile_path = args.labeled_profile.expanduser().resolve()
    profile = validate_labeled_profile_receipt(profile_path)
    output_dir = args.output_dir.expanduser().resolve()
    receipt_path = output_dir / "receipt.json"
    if args.validate_only or receipt_path.is_file():
        receipt = validate_stage211_sft_public_overlap_receipt(
            receipt_path,
            expected_labeled_profile=profile_path,
            expected_nano_provenance=args.nano_provenance.expanduser().resolve(),
            expected_public_pcm_audit=args.public_pcm_audit.expanduser().resolve(),
        )
    else:
        receipt = build_stage211_sft_public_overlap_audit(
            labeled_profile_path=profile_path,
            nano_provenance_path=args.nano_provenance,
            public_pcm_audit_path=args.public_pcm_audit,
            output_dir=output_dir,
        )
    expected = profile["expected"]
    coverage = receipt["coverage"]
    overlap = receipt["overlap"]
    print(
        "[stage211-sft-public-overlap] "
        f"rows={coverage['scanned_rows']}/{expected['total_samples']} "
        f"candidates={coverage['candidate_rows']} "
        f"train_overlap={overlap['training_rows']} "
        f"eval_overlap={overlap['internal_eval_rows']} "
        f"training_ready={str(receipt['training_ready']).lower()} "
        f"receipt={receipt_path}",
        flush=True,
    )
    return 0 if receipt["training_ready"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
