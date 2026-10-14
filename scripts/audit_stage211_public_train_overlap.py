from __future__ import annotations

import argparse
from pathlib import Path

from rwkvasr.eval.stage211_public_overlap import create_stage211_public_overlap_audit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE178_INDEX = Path(
    "/media/usbhd/training_data/asr/curriculum/"
    "stage22_stage21_soup_clean_repair_mix/stages/"
    "stage178_cv22_audio_only_online_ctc_alignment/"
    "stage178a_cv22_large1m_online_ctc/webdataset_lengths.jsonl"
)
DEFAULT_CV22_ROOT = Path("/media/usbhd/common_voice_22/common_voice_22_0")
DEFAULT_OUTPUT_DIR = (
    Path.home() / "rwkvasr_data" / "stage211_full_curriculum" / "public_train_overlap_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage211 Common Voice public/training audio overlap by MP3 bytes."
    )
    parser.add_argument("--stage178-index", type=Path, default=DEFAULT_STAGE178_INDEX)
    parser.add_argument(
        "--commonvoice-test-tsv",
        type=Path,
        default=DEFAULT_CV22_ROOT / "transcript" / "en" / "test.tsv",
    )
    parser.add_argument(
        "--converter-source",
        type=Path,
        default=DEFAULT_CV22_ROOT / "convert_to_webdataset.py",
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "eval_benchmarks"
        / "manifests"
        / "commonvoice_en_test.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = create_stage211_public_overlap_audit(
        stage178_index=args.stage178_index,
        commonvoice_test_tsv=args.commonvoice_test_tsv,
        public_manifest=args.public_manifest,
        converter_source=args.converter_source,
        output_dir=args.output_dir,
        expected_stage178_english_rows=63_625,
        expected_public_rows=16_401,
        expected_candidate_rows=1_474,
    )
    coverage = receipt["coverage"]
    print(
        "[stage211-public-overlap] "
        f"candidates={coverage['candidate_training_rows']} "
        f"exact_training_rows={coverage['exact_byte_identical_training_rows']} "
        f"excluded_public_rows={coverage['excluded_public_rows']} "
        f"clean_public_rows={coverage['clean_public_rows']} "
        f"receipt={args.output_dir.expanduser().resolve() / 'receipt.json'}"
    )


if __name__ == "__main__":
    main()
