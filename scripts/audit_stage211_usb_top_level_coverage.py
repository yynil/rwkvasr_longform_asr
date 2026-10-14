#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/media/usbhd")
DEFAULT_STAGE179_RECEIPT = (
    Path.home()
    / "rwkvasr_data/stage211_full_curriculum/stage211_loaded_manifest_chain_receipt.json"
)
DEFAULT_TESTNEW_README = DEFAULT_ROOT / "testnew/README.md"
DEFAULT_WENET_CONVERTER = DEFAULT_ROOT / "wenetdata/convert_to_webdataseet.py"
DEFAULT_OUTPUT = (
    Path.home()
    / "rwkvasr_data/stage211_usb_top_level_coverage_v1/coverage_receipt.json"
)

STAGE179_ROWS = 62_072_225
STAGE179_HOURS = 118_465.16068055555
STAGE179_WENET_ROWS = 14_506_415
STAGE179_WENET_HOURS = 9_926.039
EXPECTED_WENET_RAW_SHARDS = 688
EXPECTED_WENET_RAW_SHARD_BYTES = 575_646_671_970
EXPECTED_NEW_VIDEO_TAR_BYTES = 7_230_351_360
EXPECTED_TESTNEW_PARQUET_BYTES = 170_259_304_060
EXPECTED_SYNTHETIC_PAIRS = {
    "liwu1_speech": 570,
    "liwu_speech": 755,
    "xiaohongshu_speech": 984,
}
EXPECTED_MINIMAX_TARS = {
    "liwu1_speech.tar": 5_368_606_720,
    "liwu_speech.tar": 6_316_032_000,
    "xiaohongshu_speech.tar": 3_280_957_440,
}

ENTRY_REGISTRY = {
    ".Trash-1000": (
        "directory",
        "non_dataset",
        "filesystem trash is not an active dataset root",
    ),
    ".cache": ("directory", "non_dataset", "download and tool cache"),
    "LLaSO-Align": (
        "directory",
        "queued_for_admission",
        "selected LJSpeech, VCTK, and MLS train members require the supplemental receipt",
    ),
    "MLCommons": (
        "directory",
        "queued_for_admission",
        "People's Speech Parquet sources require the supplemental receipt",
    ),
    "clean": ("directory", "non_dataset", "RWKVTTS source and environment scripts"),
    "clean_vocals": (
        "directory",
        "queued_for_admission",
        "vocal replacements require social VAD and normalized-PCM filtering",
    ),
    "common_voice_22": (
        "directory",
        "admitted",
        "only the Stage179-bound Common Voice subtrees are admitted",
    ),
    "github": ("directory", "non_dataset", "source repositories"),
    "hfd.sh": ("file", "non_dataset", "download helper script"),
    "liwu1": ("directory", "text_only", "blog JSONL text source"),
    "liwu1_speech": (
        "directory",
        "synthetic_tts_augmentation",
        "project-attested MiniMax/blog synthetic speech",
    ),
    "liwu_speech": (
        "directory",
        "synthetic_tts_augmentation",
        "project-attested MiniMax/blog synthetic speech",
    ),
    "liwu_texts": ("directory", "text_only", "blog JSONL text source"),
    "minimax_tars": (
        "directory",
        "synthetic_tts_augmentation",
        "archive packaging of local MiniMax/blog synthetic speech",
    ),
    "models": ("directory", "non_dataset", "model checkpoints and assets"),
    "new_video.tar": (
        "file",
        "queued_for_admission",
        "archived social audio requires indexing, dedupe, VAD, and public filtering",
    ),
    "rwkvasr_runs": ("directory", "non_dataset", "training outputs"),
    "testnew": (
        "directory",
        "synthetic_tts_augmentation",
        "README declares a text-to-speech Parquet dataset",
    ),
    "tmp": ("directory", "non_dataset", "scratch and generated intermediates"),
    "training_data": (
        "directory",
        "admitted",
        "only receipt-bound Stage179 subtrees are admitted",
    ),
    "videos": (
        "directory",
        "queued_for_admission",
        "raw social audio requires VAD and normalized-PCM filtering",
    ),
    "videos_bilibili5": (
        "directory",
        "queued_for_admission",
        "raw social audio requires VAD and normalized-PCM filtering",
    ),
    "videos_bilibili6": (
        "directory",
        "queued_for_admission",
        "raw social audio requires VAD and normalized-PCM filtering",
    ),
    "videos_bilibili7": (
        "directory",
        "queued_for_admission",
        "raw social audio requires VAD and normalized-PCM filtering",
    ),
    "wenetdata": (
        "directory",
        "same_corpus_repack",
        "alternate WenetSpeech raw/WebDataset copy already represented by Stage179",
    ),
    "xiaohongshu_speech": (
        "directory",
        "synthetic_tts_augmentation",
        "project-attested MiniMax/Xiaohongshu synthetic speech",
    ),
    "xiaohongshu_texts": (
        "directory",
        "text_only",
        "Xiaohongshu JSONL text source",
    ),
}


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not load {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _entry_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    if path.is_symlink():
        return "symlink"
    return "other"


def _count_suffix(root: Path, suffix: str) -> int:
    return sum(
        1
        for entry in os.scandir(root)
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(suffix)
    )


def _tar_first_regular_member(path: Path) -> dict[str, Any]:
    with tarfile.open(path, mode="r:") as archive:
        first_entry: str | None = None
        for member in archive:
            if first_entry is None:
                first_entry = member.name
            if member.isfile():
                return {
                    "first_entry": first_entry,
                    "first_regular_member": member.name,
                    "first_regular_member_size": int(member.size),
                    "first_regular_member_offset_data": int(member.offset_data),
                }
    raise ValueError(f"Archive has no regular member: {path}")


def _validate_stage179(receipt_path: Path, *, strict_production: bool) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage179 loaded-manifest receipt")
    manifest_path = Path(str(receipt.get("global_dedup_manifest_path") or "")).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Stage179 global manifest is missing: {manifest_path}")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != receipt.get("global_dedup_manifest_sha256"):
        raise ValueError("Stage179 global manifest SHA-256 differs from its loaded receipt.")
    manifest = _load_json(manifest_path, label="Stage179 global manifest")
    giga_wenet = manifest.get("inputs", {}).get("giga_wenet", {})
    wenet_rows = int(giga_wenet.get("accepted_counts_by_source", {}).get("wenetspeech", -1))
    wenet_hours = float(giga_wenet.get("accepted_hours_by_source", {}).get("wenetspeech", -1.0))
    rows = int(manifest.get("total_unique_audio_rows", -1))
    hours = float(manifest.get("total_unique_hours", -1.0))
    if strict_production and (
        rows != STAGE179_ROWS
        or not math.isclose(hours, STAGE179_HOURS, rel_tol=0.0, abs_tol=1e-9)
        or wenet_rows != STAGE179_WENET_ROWS
        or not math.isclose(wenet_hours, STAGE179_WENET_HOURS, rel_tol=0.0, abs_tol=0.001)
    ):
        raise ValueError("Stage179 production totals changed.")
    return {
        "loaded_receipt_path": str(receipt_path),
        "loaded_receipt_sha256": _sha256(receipt_path),
        "global_manifest_path": str(manifest_path),
        "global_manifest_sha256": manifest_sha256,
        "rows": rows,
        "hours": hours,
        "wenetspeech_rows": wenet_rows,
        "wenetspeech_hours": wenet_hours,
        "dedupe_scope": "Stage179 receipt-bound storage identity",
    }


def _wenet_repack_evidence(
    root: Path,
    converter_path: Path,
    *,
    strict_production: bool,
) -> dict[str, Any]:
    wenet_root = root / "wenetdata"
    metadata_path = wenet_root / "WenetSpeech.json"
    shard_root = wenet_root / "webdataset_output"
    converter_path = converter_path.expanduser().resolve()
    converter = converter_path.read_text(encoding="utf-8")
    if "for segment in audio_data.get('segments', [])" not in converter:
        raise ValueError("WenetSpeech converter traversal contract changed.")
    subset_filter_present = "subsets" in converter
    with metadata_path.open("rb") as handle:
        metadata_header = handle.read(4096).decode("utf-8", errors="replace")
    if '"dataset": "WenetSpeech"' not in metadata_header:
        raise ValueError("WenetSpeech metadata identity is missing.")
    shards = sorted(
        (
            entry
            for entry in os.scandir(shard_root)
            if entry.is_file(follow_symlinks=False)
            and re.fullmatch(r"shard_\d+\.tar", entry.name)
        ),
        key=lambda entry: entry.name,
    )
    shard_bytes = sum(entry.stat(follow_symlinks=False).st_size for entry in shards)
    if strict_production and (
        len(shards) != EXPECTED_WENET_RAW_SHARDS
        or shard_bytes != EXPECTED_WENET_RAW_SHARD_BYTES
        or subset_filter_present
    ):
        raise ValueError("WenetSpeech alternate-repack production evidence changed.")
    return {
        "corpus_identity": "WenetSpeech",
        "metadata_path": str(metadata_path.resolve()),
        "metadata_size_bytes": metadata_path.stat().st_size,
        "converter_path": str(converter_path),
        "converter_sha256": _sha256(converter_path),
        "converter_iterates_all_segments": True,
        "converter_subset_filter_present": subset_filter_present,
        "raw_webdataset_shards": len(shards),
        "raw_webdataset_bytes": shard_bytes,
        "admission": "excluded_same_corpus_repack_and_unsafe_split_filter",
        "exact_content_dedup_complete": False,
    }


def _synthetic_evidence(
    root: Path,
    readme_path: Path,
    *,
    strict_production: bool,
) -> dict[str, Any]:
    readme_path = readme_path.expanduser().resolve()
    readme = readme_path.read_text(encoding="utf-8")
    explicit_tts = bool(re.search(r"task_categories:\s*\n- text-to-speech", readme))
    clips_match = re.search(r"\*\*Total audio files\*\* \| ([\d,]+)", readme)
    hours_match = re.search(r"\*\*Total duration\*\* \| ([\d,.]+) hours", readme)
    if not explicit_tts or clips_match is None or hours_match is None:
        raise ValueError("testnew README no longer proves its TTS identity and totals.")
    clips = int(clips_match.group(1).replace(",", ""))
    hours = float(hours_match.group(1).replace(",", ""))
    parquet_path = readme_path.parent / "tts_dataset_combined.parquet"
    pairs = {
        name: {
            "wav": _count_suffix(root / name, ".wav"),
            "json": _count_suffix(root / name, ".json"),
        }
        for name in EXPECTED_SYNTHETIC_PAIRS
    }
    minimax_tars = {
        entry.name: entry.stat(follow_symlinks=False).st_size
        for entry in os.scandir(root / "minimax_tars")
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".tar")
    }
    if strict_production and (
        clips != 556_667
        or not math.isclose(hours, 1_024.71, rel_tol=0.0, abs_tol=0.001)
        or parquet_path.stat().st_size != EXPECTED_TESTNEW_PARQUET_BYTES
        or any(
            values != {"wav": expected, "json": expected}
            for name, expected in EXPECTED_SYNTHETIC_PAIRS.items()
            for values in (pairs[name],)
        )
        or minimax_tars != EXPECTED_MINIMAX_TARS
    ):
        raise ValueError("Synthetic-TTS production evidence changed.")
    return {
        "testnew_readme_path": str(readme_path),
        "testnew_readme_sha256": _sha256(readme_path),
        "testnew_explicit_text_to_speech": explicit_tts,
        "testnew_clips": clips,
        "testnew_hours": hours,
        "testnew_parquet_path": str(parquet_path.resolve()),
        "testnew_parquet_size_bytes": parquet_path.stat().st_size,
        "local_paired_files": pairs,
        "minimax_archive_files": dict(sorted(minimax_tars.items())),
        "natural_asr_admission": "excluded_synthetic_tts_augmentation_only",
        "minimax_provenance_mode": "project_attestation",
    }


def _archived_social_evidence(
    root: Path,
    *,
    strict_production: bool,
) -> dict[str, Any]:
    archive_path = root / "new_video.tar"
    if strict_production and archive_path.stat().st_size != EXPECTED_NEW_VIDEO_TAR_BYTES:
        raise ValueError("new_video.tar production size changed.")
    first = _tar_first_regular_member(archive_path)
    first_top_level = str(first["first_regular_member"]).split("/", 1)[0]
    social_roots = [
        root / "videos",
        root / "videos_bilibili5",
        root / "videos_bilibili6",
        root / "videos_bilibili7",
        root / "clean_vocals",
    ]
    matching_top_level_paths = sorted(
        str((social_root / first_top_level).resolve())
        for social_root in social_roots
        if (social_root / first_top_level).exists()
    )
    return {
        "archive_path": str(archive_path.resolve()),
        "archive_size_bytes": archive_path.stat().st_size,
        **first,
        "first_member_top_level": first_top_level,
        "matching_existing_social_top_level_paths": matching_top_level_paths,
        "path_tree_overlap_proven": bool(matching_top_level_paths),
        "content_overlap_proven": False,
        "admission": "pending_member_hash_decode_dedupe_vad_and_public_filter",
    }


def build_receipt(
    *,
    root: Path,
    stage179_receipt: Path,
    testnew_readme: Path,
    wenet_converter: Path,
    strict_production: bool,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"USB root is unavailable: {root}")
    observed = {entry.name: entry for entry in os.scandir(root)}
    expected_names = set(ENTRY_REGISTRY)
    unknown = sorted(set(observed) - expected_names)
    missing = sorted(expected_names - set(observed))
    if unknown or missing:
        raise ValueError(f"USB top-level registry mismatch: unknown={unknown} missing={missing}")

    entries: list[dict[str, Any]] = []
    for name, (expected_type, classification, rationale) in ENTRY_REGISTRY.items():
        path = root / name
        actual_type = _entry_type(path)
        if actual_type != expected_type:
            raise ValueError(
                f"USB top-level type changed for {name}: expected={expected_type} actual={actual_type}"
            )
        record: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "type": actual_type,
            "classification": classification,
            "rationale": rationale,
        }
        if actual_type == "file":
            record["size_bytes"] = path.stat().st_size
        entries.append(record)

    evidence = {
        "stage179": _validate_stage179(
            stage179_receipt,
            strict_production=strict_production,
        ),
        "wenetspeech_repack": _wenet_repack_evidence(
            root,
            wenet_converter,
            strict_production=strict_production,
        ),
        "synthetic_tts": _synthetic_evidence(
            root,
            testnew_readme,
            strict_production=strict_production,
        ),
        "archived_social": _archived_social_evidence(
            root,
            strict_production=strict_production,
        ),
    }
    pending = sorted(
        record["name"]
        for record in entries
        if record["classification"] == "queued_for_admission"
    )
    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "usb_top_level_coverage",
        "root": str(root),
        "classification_complete": True,
        "training_coverage_complete": False,
        "natural_asr_policy": "all_receipt_admissible_natural_audio_after_dedupe_and_public_filter",
        "top_level_entries": entries,
        "top_level_entry_count": len(entries),
        "pending_natural_admission": pending,
        "synthetic_tts_policy": "augmentation_only_not_natural_asr_coverage",
        "same_corpus_repack_policy": "exclude_without_counting_as_new_coverage",
        "global_content_fingerprint_complete": False,
        "acoustic_near_duplicate_complete": False,
        "evidence": evidence,
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _write_immutable(path: Path, value: dict[str, Any]) -> str:
    path = path.expanduser().resolve()
    payload = _canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace non-identical coverage receipt: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable Stage211 USB top-level coverage receipt."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage179-receipt", type=Path, default=DEFAULT_STAGE179_RECEIPT)
    parser.add_argument("--testnew-readme", type=Path, default=DEFAULT_TESTNEW_README)
    parser.add_argument("--wenet-converter", type=Path, default=DEFAULT_WENET_CONVERTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-nonproduction-layout",
        action="store_true",
        help="Derive fixture totals instead of enforcing the production constants.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    receipt = build_receipt(
        root=args.root,
        stage179_receipt=args.stage179_receipt,
        testnew_readme=args.testnew_readme,
        wenet_converter=args.wenet_converter,
        strict_production=not args.allow_nonproduction_layout,
    )
    status = _write_immutable(args.output, receipt)
    print(
        f"usb_top_level_coverage={args.output.expanduser().resolve()} status={status} "
        f"entries={receipt['top_level_entry_count']} "
        f"pending={len(receipt['pending_natural_admission'])} "
        f"training_coverage_complete={receipt['training_coverage_complete']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
