#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOTS = {
    "videos": Path("/media/usbhd/videos"),
    "videos_bilibili5": Path("/media/usbhd/videos_bilibili5"),
    "videos_bilibili6": Path("/media/usbhd/videos_bilibili6"),
    "videos_bilibili7": Path("/media/usbhd/videos_bilibili7"),
    "clean_vocals": Path("/media/usbhd/clean_vocals"),
}
DEFAULT_VAD_MODEL_ROOT = Path(
    "/home/yueyulin/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
)
DEFAULT_OUTPUT = Path.home() / "rwkvasr_data/stage211_social_vad_audit_v1/source_inventory.json"

AUDIO_SUFFIXES = {".mp3", ".wav"}
VIDEO_SUFFIXES = {".mp4"}
SIDECAR_SUFFIXES = {".ass", ".csv", ".json", ".jsonl", ".lrc", ".srt", ".tsv", ".txt", ".vtt"}
EXPECTED_EXTENSION_COUNTS = {".mp3": 7_773, ".wav": 283, ".mp4": 527}
EXPECTED_ROOT_COUNTS = {
    "videos": {"files": 5_264, ".mp3": 4_475, ".wav": 263, ".mp4": 526},
    "videos_bilibili5": {"files": 967, ".mp3": 967, ".wav": 0, ".mp4": 0},
    "videos_bilibili6": {"files": 893, ".mp3": 892, ".wav": 0, ".mp4": 1},
    "videos_bilibili7": {"files": 1_439, ".mp3": 1_439, ".wav": 0, ".mp4": 0},
    "clean_vocals": {"files": 20, ".mp3": 0, ".wav": 20, ".mp4": 0},
}
EXPECTED_TOTAL_FILES = 8_583
EXPECTED_TOTAL_BYTES = 92_610_944_135
EXPECTED_VOCAL_REPLACEMENTS = 20
EXPECTED_CANONICAL_AUDIO = 8_036
EXPECTED_VAD_HASHES = {
    "model.pt": "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5",
    "config.yaml": "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93",
    "am.mvn": "6820fef9687708c4fc3fab2530179c8fcea6262daa25514380056cd8f6eb1754",
}


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _title(path: Path, *, clean_vocal: bool = False) -> str:
    title = path.parent.name if clean_vocal else path.stem
    for suffix in ("_audio", "_video"):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
    return title


def _iter_files(root: Path) -> Iterable[Path]:
    return (path for path in sorted(root.rglob("*")) if path.is_file())


def _source_label(root_name: str, path: Path) -> str:
    if root_name == "clean_vocals":
        return "social_clean_vocals"
    if root_name == "videos" and path.suffix.lower() == ".wav":
        return "social_videos_extracted_wav"
    return f"social_{root_name}_mp3"


def _probe_audio(path: Path) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(str(path))
    duration_seconds = float(info.frames) / float(info.samplerate)
    if info.frames <= 0 or info.samplerate <= 0 or not math.isfinite(duration_seconds):
        raise ValueError(f"Invalid audio metadata: {path}")
    return {
        "frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": duration_seconds,
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def build_inventory(
    *,
    roots: dict[str, Path],
    vad_model_root: Path,
    probe_audio: bool,
    hash_media: bool,
    require_production_layout: bool,
) -> dict[str, Any]:
    resolved_roots = {name: path.expanduser().resolve() for name, path in roots.items()}
    layout_errors: list[str] = []
    root_records: dict[str, Any] = {}
    path_roots: dict[Path, str] = {}
    all_files: list[Path] = []
    extension_counts: Counter[str] = Counter()
    total_bytes = 0

    for name, root in resolved_roots.items():
        if not root.is_dir():
            layout_errors.append(f"missing root: {name}={root}")
            root_records[name] = {"path": str(root), "files": 0, "bytes": 0, "extensions": {}}
            continue
        files = list(_iter_files(root))
        root_extensions: Counter[str] = Counter(path.suffix.lower() for path in files)
        root_bytes = sum(path.stat().st_size for path in files)
        root_records[name] = {
            "path": str(root),
            "files": len(files),
            "bytes": root_bytes,
            "extensions": dict(sorted(root_extensions.items())),
        }
        for path in files:
            path_roots[path] = name
        all_files.extend(files)
        extension_counts.update(root_extensions)
        total_bytes += root_bytes

    primary_audio = [
        path
        for path in all_files
        if path_roots[path] != "clean_vocals" and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    clean_vocals = [
        path
        for path in all_files
        if path_roots[path] == "clean_vocals"
        and path.name == "vocals.wav"
        and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    by_title: dict[str, list[Path]] = defaultdict(list)
    for path in primary_audio:
        by_title[_title(path)].append(path)

    replacement_records: list[dict[str, str]] = []
    replaced_paths: set[Path] = set()
    for vocal_path in clean_vocals:
        title = _title(vocal_path, clean_vocal=True)
        matches = sorted(by_title.get(title, []))
        if len(matches) != 1:
            layout_errors.append(
                f"clean vocal title must map to exactly one primary audio: {title!r} matches={len(matches)}"
            )
            continue
        replaced = matches[0]
        replaced_paths.add(replaced)
        replacement_records.append(
            {"title": title, "replacement": str(vocal_path), "replaced": str(replaced)}
        )

    canonical_paths = sorted((set(primary_audio) - replaced_paths) | set(clean_vocals))
    sidecar_paths = sorted(path for path in all_files if path.suffix.lower() in SIDECAR_SUFFIXES)
    video_paths = sorted(path for path in all_files if path.suffix.lower() in VIDEO_SUFFIXES)
    unsupported_paths = sorted(
        path
        for path in all_files
        if path.suffix.lower() not in AUDIO_SUFFIXES | VIDEO_SUFFIXES | SIDECAR_SUFFIXES
    )

    if require_production_layout:
        if len(all_files) != EXPECTED_TOTAL_FILES:
            layout_errors.append(
                f"total files expected={EXPECTED_TOTAL_FILES} actual={len(all_files)}"
            )
        if total_bytes != EXPECTED_TOTAL_BYTES:
            layout_errors.append(
                f"total bytes expected={EXPECTED_TOTAL_BYTES} actual={total_bytes}"
            )
        if dict(extension_counts) != EXPECTED_EXTENSION_COUNTS:
            layout_errors.append(
                f"extension counts expected={EXPECTED_EXTENSION_COUNTS} actual={dict(extension_counts)}"
            )
        for name, expected in EXPECTED_ROOT_COUNTS.items():
            actual = root_records.get(name, {})
            actual_layout = {
                "files": int(actual.get("files", 0)),
                ".mp3": int(actual.get("extensions", {}).get(".mp3", 0)),
                ".wav": int(actual.get("extensions", {}).get(".wav", 0)),
                ".mp4": int(actual.get("extensions", {}).get(".mp4", 0)),
            }
            if actual_layout != expected:
                layout_errors.append(
                    f"root layout {name} expected={expected} actual={actual_layout}"
                )
        if len(replacement_records) != EXPECTED_VOCAL_REPLACEMENTS:
            layout_errors.append(
                "vocal replacements "
                f"expected={EXPECTED_VOCAL_REPLACEMENTS} actual={len(replacement_records)}"
            )
        if len(canonical_paths) != EXPECTED_CANONICAL_AUDIO:
            layout_errors.append(
                f"canonical audio expected={EXPECTED_CANONICAL_AUDIO} actual={len(canonical_paths)}"
            )
        if sidecar_paths:
            layout_errors.append(f"unexpected timestamp/transcript sidecars: {len(sidecar_paths)}")
        if unsupported_paths:
            layout_errors.append(f"unsupported files present: {len(unsupported_paths)}")

    vad_model_root = vad_model_root.expanduser().resolve()
    vad_assets: dict[str, Any] = {}
    for filename, expected_hash in EXPECTED_VAD_HASHES.items():
        path = vad_model_root / filename
        if not path.is_file():
            layout_errors.append(f"missing VAD asset: {path}")
            continue
        actual_hash = _sha256(path)
        vad_assets[filename] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_hash,
        }
        if require_production_layout and actual_hash != expected_hash:
            layout_errors.append(
                f"VAD asset hash mismatch {filename}: expected={expected_hash} actual={actual_hash}"
            )

    probe_errors: list[dict[str, str]] = []
    probe_counts: Counter[str] = Counter()
    sample_rate_counts: Counter[int] = Counter()
    channel_counts: Counter[int] = Counter()
    total_duration_seconds = 0.0
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    media_hashes: dict[str, list[str]] = defaultdict(list)
    source_records: list[dict[str, Any]] = []
    replacements_by_path = {
        Path(record["replacement"]): record["replaced"] for record in replacement_records
    }
    for path in canonical_paths:
        stat = path.stat()
        root_name = path_roots[path]
        record: dict[str, Any] = {
            "path": str(path),
            "root": root_name,
            "source_label": _source_label(root_name, path),
            "title": _title(path, clean_vocal=root_name == "clean_vocals"),
            "suffix": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        replaced = replacements_by_path.get(path)
        if replaced is not None:
            record["replaces"] = replaced
        if hash_media:
            digest = _sha256(path)
            record["sha256"] = digest
            media_hashes[digest].append(str(path))
        if probe_audio:
            try:
                probe = _probe_audio(path)
            except Exception as exc:  # pragma: no cover - production data dependent
                probe_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            else:
                record["probe"] = probe
                duration = float(probe["duration_seconds"])
                total_duration_seconds += duration
                min_duration_seconds = (
                    duration
                    if min_duration_seconds is None
                    else min(min_duration_seconds, duration)
                )
                max_duration_seconds = (
                    duration
                    if max_duration_seconds is None
                    else max(max_duration_seconds, duration)
                )
                probe_counts[str(probe["format"])] += 1
                sample_rate_counts[int(probe["sample_rate"])] += 1
                channel_counts[int(probe["channels"])] += 1
        source_records.append(record)

    exact_content_duplicates = [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(media_hashes.items())
        if len(paths) > 1
    ]
    source_counts = Counter(record["source_label"] for record in source_records)
    source_bytes = Counter()
    for record in source_records:
        source_bytes[str(record["source_label"])] += int(record["size_bytes"])

    return {
        "schema_version": 1,
        "artifact": "stage211_social_vad_source_inventory",
        "complete": True,
        "training_ready": False,
        "admission_state": "source_inventory_only",
        "layout_valid": not layout_errors,
        "layout_errors": layout_errors,
        "roots": root_records,
        "total_files": len(all_files),
        "total_bytes": total_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "sidecar_files": [str(path) for path in sidecar_paths],
        "unsupported_files": [str(path) for path in unsupported_paths],
        "excluded_video_files": len(video_paths),
        "vocal_replacements": replacement_records,
        "canonical_audio_files": len(source_records),
        "canonical_audio_bytes": sum(int(record["size_bytes"]) for record in source_records),
        "source_counts": dict(sorted(source_counts.items())),
        "source_bytes": dict(sorted(source_bytes.items())),
        "media_content_hash_complete": bool(hash_media),
        "exact_content_duplicate_groups": exact_content_duplicates,
        "audio_probe": {
            "complete": bool(probe_audio),
            "errors": probe_errors,
            "probed_files": len(source_records) - len(probe_errors) if probe_audio else 0,
            "hours": total_duration_seconds / 3600.0 if probe_audio else None,
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "format_counts": dict(sorted(probe_counts.items())),
            "sample_rate_counts": {
                str(key): value for key, value in sorted(sample_rate_counts.items())
            },
            "channel_counts": {str(key): value for key, value in sorted(channel_counts.items())},
        },
        "vad": {
            "implementation": "FunASR FsmnVADStreaming",
            "model_root": str(vad_model_root),
            "assets": vad_assets,
            "device": "cpu",
            "max_single_segment_time_ms": 30_000,
            "minimum_admitted_segment_ms": 500,
            "target_sample_rate": 16_000,
            "target_channels": 1,
        },
        "required_before_training": [
            "complete duration and decode probe",
            "immutable source-content hashes and duplicate decision",
            "deterministic VAD boundary receipt",
            "materialized mono 16 kHz segments between 0.5 and 30 seconds",
            "source-to-segment duration accounting",
            "complete segment decode verification",
            "public-evaluation overlap audit",
            "bucket manifest and fixed hidden-eval binding",
            "identical admitted pool for rwkv_layer, block, and logits",
        ],
        "source_records": source_records,
    }


def _write_immutable(path: Path, payload: dict[str, Any]) -> str:
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path = path.expanduser().resolve()
    if path.exists():
        if path.read_bytes() != rendered:
            raise FileExistsError(f"Refusing to replace different inventory: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered)
    return "created"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit raw USB social media before Stage211 VAD segmentation admission."
    )
    for name, default in DEFAULT_ROOTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}-root", type=Path, default=default)
    parser.add_argument("--vad-model-root", type=Path, default=DEFAULT_VAD_MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-audio", action="store_true")
    parser.add_argument("--hash-media", action="store_true")
    parser.add_argument("--allow-layout-mismatch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    roots = {name: getattr(args, f"{name}_root") for name in DEFAULT_ROOTS}
    inventory = build_inventory(
        roots=roots,
        vad_model_root=args.vad_model_root,
        probe_audio=bool(args.probe_audio),
        hash_media=bool(args.hash_media),
        require_production_layout=not bool(args.allow_layout_mismatch),
    )
    if not inventory["layout_valid"] and not args.allow_layout_mismatch:
        raise ValueError("; ".join(inventory["layout_errors"]))
    status = _write_immutable(args.output, inventory)
    print(
        f"inventory={args.output.expanduser().resolve()} status={status} "
        f"canonical_audio={inventory['canonical_audio_files']} "
        f"training_ready={str(inventory['training_ready']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
