#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE_INVENTORY = (
    Path.home() / "rwkvasr_data/stage211_social_vad_audit_v1/source_inventory_full.json"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "rwkvasr_data/stage211_social_vad_boundaries_v1"
EXPECTED_SOURCE_ARTIFACT = "stage211_social_vad_source_inventory"
BOUNDARY_ARTIFACT = "stage211_social_vad_source_boundaries"
INVENTORY_ARTIFACT = "stage211_social_vad_boundary_inventory"
MAX_SEGMENT_MS = 30_000
MIN_SEGMENT_MS = 500
MERGE_GAP_MS = 200
VAD_BATCH_SIZE_SECONDS = 300


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def validate_source_inventory(path: Path, *, verify_media_hashes: bool) -> dict[str, Any]:
    path = path.expanduser().resolve()
    inventory = _load_json(path, label="Stage211 social VAD source inventory")
    expected = {
        "schema_version": 1,
        "artifact": EXPECTED_SOURCE_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "source_inventory_only",
        "layout_valid": True,
        "media_content_hash_complete": True,
    }
    if any(inventory.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 social source inventory contract mismatch.")
    audio_probe = inventory.get("audio_probe")
    if (
        not isinstance(audio_probe, dict)
        or audio_probe.get("complete") is not True
        or audio_probe.get("errors") != []
        or int(audio_probe.get("probed_files", -1))
        != int(inventory.get("canonical_audio_files", -2))
        or not math.isfinite(float(audio_probe.get("hours", float("nan"))))
        or float(audio_probe["hours"]) <= 0.0
    ):
        raise ValueError("Stage211 social source inventory lacks a complete audio probe.")
    source_records = inventory.get("source_records")
    if (
        not isinstance(source_records, list)
        or not source_records
        or len(source_records) != int(inventory.get("canonical_audio_files", -1))
    ):
        raise ValueError("Stage211 social source records are incomplete.")

    seen_paths: set[str] = set()
    for record in source_records:
        if not isinstance(record, dict):
            raise ValueError("Stage211 social source record is invalid.")
        source_path = Path(str(record.get("path") or "")).expanduser().resolve()
        source_sha256 = str(record.get("sha256") or "")
        probe = record.get("probe")
        if (
            str(source_path) in seen_paths
            or not source_path.is_file()
            or len(source_sha256) != 64
            or not isinstance(probe, dict)
            or int(probe.get("frames", -1)) <= 0
            or int(probe.get("sample_rate", -1)) <= 0
            or int(probe.get("channels", -1)) <= 0
            or float(probe.get("duration_seconds", -1.0)) <= 0.0
        ):
            raise ValueError(f"Stage211 social source binding is invalid: {source_path}")
        stat = source_path.stat()
        if (
            int(record.get("size_bytes", -1)) != stat.st_size
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
        ):
            raise ValueError(f"Stage211 social source metadata changed: {source_path}")
        if verify_media_hashes and _sha256(source_path) != source_sha256:
            raise ValueError(f"Stage211 social source SHA-256 changed: {source_path}")
        seen_paths.add(str(source_path))

    vad = inventory.get("vad")
    assets = vad.get("assets") if isinstance(vad, dict) else None
    if (
        not isinstance(assets, dict)
        or int(vad.get("max_single_segment_time_ms", -1)) != MAX_SEGMENT_MS
        or int(vad.get("minimum_admitted_segment_ms", -1)) != MIN_SEGMENT_MS
    ):
        raise ValueError("Stage211 social source inventory VAD binding is invalid.")
    for filename in ("model.pt", "config.yaml", "am.mvn"):
        record = assets.get(filename)
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 social VAD asset binding is absent: {filename}")
        asset_path = Path(str(record.get("path") or "")).expanduser().resolve()
        if (
            not asset_path.is_file()
            or int(record.get("size_bytes", -1)) != asset_path.stat().st_size
            or _sha256(asset_path) != str(record.get("sha256") or "")
        ):
            raise ValueError(f"Stage211 social VAD asset changed: {asset_path}")
    inventory["_inventory_path"] = str(path)
    inventory["_inventory_sha256"] = _sha256(path)
    return inventory


def _source_priority(record: dict[str, Any]) -> tuple[int, str]:
    label = str(record.get("source_label") or "")
    priority = {
        "social_clean_vocals": 0,
        "social_videos_extracted_wav": 1,
    }.get(label, 2)
    return priority, str(record.get("path") or "")


def exact_duplicate_decisions(inventory: dict[str, Any]) -> dict[str, str]:
    by_path = {str(record["path"]): record for record in inventory["source_records"]}
    duplicate_of: dict[str, str] = {}
    groups = inventory.get("exact_content_duplicate_groups")
    if not isinstance(groups, list):
        raise ValueError("Stage211 social inventory duplicate groups are invalid.")
    grouped_paths: set[str] = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("paths"), list):
            raise ValueError("Stage211 social exact duplicate group is invalid.")
        paths = [str(Path(path).expanduser().resolve()) for path in group["paths"]]
        if len(paths) < 2 or any(path not in by_path for path in paths):
            raise ValueError("Stage211 social exact duplicate group has unknown paths.")
        if any(path in grouped_paths for path in paths):
            raise ValueError("Stage211 social source occurs in multiple duplicate groups.")
        expected_hash = str(group.get("sha256") or "")
        if any(str(by_path[path].get("sha256") or "") != expected_hash for path in paths):
            raise ValueError("Stage211 social exact duplicate hash binding is inconsistent.")
        winner = min((by_path[path] for path in paths), key=_source_priority)
        winner_path = str(winner["path"])
        for path in paths:
            grouped_paths.add(path)
            if path != winner_path:
                duplicate_of[path] = winner_path
    return duplicate_of


def _merge_raw_segments(
    segments: Iterable[Iterable[int]], *, duration_ms: int
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    previous_raw_start = -1
    for raw in segments:
        values = list(raw)
        if len(values) != 2:
            raise ValueError(f"VAD segment must contain start/end: {values}")
        start, end = int(values[0]), int(values[1])
        if start < previous_raw_start or start < 0 or end <= start:
            raise ValueError(f"VAD segments are invalid or unordered: {start},{end}")
        previous_raw_start = start
        start = min(start, duration_ms)
        end = min(end, duration_ms)
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= MERGE_GAP_MS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def normalize_vad_segments(
    segments: Iterable[Iterable[int]],
    *,
    duration_ms: int,
    minimum_ms: int = MIN_SEGMENT_MS,
    maximum_ms: int = MAX_SEGMENT_MS,
) -> list[tuple[int, int]]:
    if duration_ms <= 0 or minimum_ms <= 0 or maximum_ms < minimum_ms:
        raise ValueError("Invalid VAD normalization limits.")
    admitted: list[tuple[int, int]] = []
    for start, end in _merge_raw_segments(segments, duration_ms=duration_ms):
        span = end - start
        if span < minimum_ms:
            continue
        pieces = max(1, math.ceil(span / maximum_ms))
        boundaries = [start + (span * index) // pieces for index in range(pieces)] + [end]
        for piece_start, piece_end in zip(boundaries, boundaries[1:]):
            piece_duration = piece_end - piece_start
            if piece_duration < minimum_ms:
                raise ValueError(
                    f"VAD normalization produced a sub-minimum segment: {piece_start},{piece_end}"
                )
            if piece_duration > maximum_ms:
                raise ValueError(
                    f"VAD normalization produced an overlong segment: {piece_start},{piece_end}"
                )
            admitted.append((piece_start, piece_end))
    return admitted


def _part_path(output_root: Path, source_index: int) -> Path:
    return output_root / "sources" / f"source_{source_index:06d}.json"


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    rendered = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if path.is_file():
        if path.read_bytes() != rendered:
            raise ValueError(f"Refusing to replace a different Stage211 VAD artifact: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(rendered)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def _vad_binding(inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "implementation": inventory["vad"]["implementation"],
        "assets": inventory["vad"]["assets"],
        "max_single_segment_time_ms": MAX_SEGMENT_MS,
        "minimum_admitted_segment_ms": MIN_SEGMENT_MS,
        "merge_gap_ms": MERGE_GAP_MS,
        "batch_size_seconds": VAD_BATCH_SIZE_SECONDS,
    }


def build_source_payload(
    *,
    inventory: dict[str, Any],
    source_index: int,
    raw_segments: Iterable[Iterable[int]] | None,
    duplicate_of: str | None,
) -> dict[str, Any]:
    record = inventory["source_records"][source_index]
    source_path = str(record["path"])
    base: dict[str, Any] = {
        "schema_version": 1,
        "artifact": BOUNDARY_ARTIFACT,
        "complete": True,
        "source_inventory_path": inventory["_inventory_path"],
        "source_inventory_sha256": inventory["_inventory_sha256"],
        "source_index": source_index,
        "source_path": source_path,
        "source_sha256": record["sha256"],
        "source_size_bytes": int(record["size_bytes"]),
        "source_mtime_ns": int(record["mtime_ns"]),
        "source_label": record["source_label"],
        "source_probe": record["probe"],
        "vad": _vad_binding(inventory),
    }
    if duplicate_of is not None:
        return {
            **base,
            "status": "exact_duplicate_skipped",
            "duplicate_of": duplicate_of,
            "raw_segment_count": 0,
            "segments": [],
            "admitted_segments": 0,
            "admitted_duration_ms": 0,
        }
    if raw_segments is None:
        raise ValueError(f"Missing VAD segments for admitted source: {source_path}")
    raw_list = [list(segment) for segment in raw_segments]
    duration_ms = int(round(float(record["probe"]["duration_seconds"]) * 1000.0))
    normalized = normalize_vad_segments(raw_list, duration_ms=duration_ms)
    segments = [
        {
            "segment_index": index,
            "start_ms": start,
            "end_ms": end,
            "duration_ms": end - start,
            "key": hashlib.blake2b(
                f"{record['sha256']}\0{start}\0{end}".encode("ascii"), digest_size=16
            ).hexdigest(),
        }
        for index, (start, end) in enumerate(normalized)
    ]
    if not segments:
        return {
            **base,
            "status": "no_speech_rejected",
            "raw_segment_count": len(raw_list),
            "segments": [],
            "admitted_segments": 0,
            "admitted_duration_ms": 0,
        }
    return {
        **base,
        "status": "admitted",
        "raw_segment_count": len(raw_list),
        "segments": segments,
        "admitted_segments": len(segments),
        "admitted_duration_ms": sum(segment["duration_ms"] for segment in segments),
    }


def validate_source_payload(
    payload: dict[str, Any],
    *,
    inventory: dict[str, Any],
    source_index: int,
    duplicate_of: str | None,
) -> None:
    source = inventory["source_records"][source_index]
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact") != BOUNDARY_ARTIFACT
        or payload.get("complete") is not True
        or int(payload.get("source_index", -1)) != source_index
        or payload.get("source_inventory_path") != inventory["_inventory_path"]
        or payload.get("source_inventory_sha256") != inventory["_inventory_sha256"]
        or payload.get("source_path") != source["path"]
        or payload.get("source_sha256") != source["sha256"]
        or int(payload.get("source_size_bytes", -1)) != int(source["size_bytes"])
        or int(payload.get("source_mtime_ns", -1)) != int(source["mtime_ns"])
        or payload.get("source_label") != source["source_label"]
        or payload.get("source_probe") != source["probe"]
        or payload.get("vad") != _vad_binding(inventory)
    ):
        raise ValueError(f"Stage211 social VAD source receipt changed: index={source_index}")
    status = payload.get("status")
    segments = payload.get("segments")
    if not isinstance(segments, list) or int(payload.get("raw_segment_count", -1)) < 0:
        raise ValueError(f"Stage211 social VAD segment payload is invalid: index={source_index}")
    if duplicate_of is not None:
        if (
            status != "exact_duplicate_skipped"
            or payload.get("duplicate_of") != duplicate_of
            or segments != []
            or int(payload.get("admitted_segments", -1)) != 0
            or int(payload.get("admitted_duration_ms", -1)) != 0
        ):
            raise ValueError(f"Stage211 social duplicate receipt is invalid: index={source_index}")
        return
    if status == "no_speech_rejected":
        if (
            segments != []
            or int(payload.get("admitted_segments", -1)) != 0
            or int(payload.get("admitted_duration_ms", -1)) != 0
        ):
            raise ValueError(f"Stage211 no-speech receipt is invalid: index={source_index}")
        return
    if status != "admitted" or not segments:
        raise ValueError(f"Stage211 admitted VAD receipt is invalid: index={source_index}")
    previous_end = -1
    total_duration = 0
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Stage211 VAD segment is invalid: index={source_index}")
        start = int(segment.get("start_ms", -1))
        end = int(segment.get("end_ms", -1))
        duration = int(segment.get("duration_ms", -1))
        expected_key = hashlib.blake2b(
            f"{source['sha256']}\0{start}\0{end}".encode("ascii"), digest_size=16
        ).hexdigest()
        if (
            int(segment.get("segment_index", -1)) != segment_index
            or start < previous_end
            or end <= start
            or duration != end - start
            or not MIN_SEGMENT_MS <= duration <= MAX_SEGMENT_MS
            or segment.get("key") != expected_key
        ):
            raise ValueError(f"Stage211 VAD segment binding is invalid: index={source_index}")
        previous_end = end
        total_duration += duration
    if (
        int(payload.get("admitted_segments", -1)) != len(segments)
        or int(payload.get("admitted_duration_ms", -1)) != total_duration
    ):
        raise ValueError(f"Stage211 VAD segment totals are invalid: index={source_index}")


def _load_model(inventory: dict[str, Any], *, torch_threads: int):
    import torch
    from funasr import AutoModel

    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive.")
    torch.set_num_threads(torch_threads)
    model_root = Path(inventory["vad"]["assets"]["model.pt"]["path"]).parent
    return AutoModel(
        model=str(model_root),
        device="cpu",
        disable_update=True,
        max_single_segment_time=MAX_SEGMENT_MS,
    )


def _generate_raw_segments(model: Any, source_path: str) -> list[list[int]]:
    results = model.generate(
        input=source_path,
        batch_size_s=VAD_BATCH_SIZE_SECONDS,
        disable_pbar=True,
    )
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"FunASR VAD returned an invalid result for {source_path}")
    value = results[0].get("value") if isinstance(results[0], dict) else None
    if not isinstance(value, list):
        raise ValueError(f"FunASR VAD returned no segment list for {source_path}")
    return value


def run_worker(
    *,
    inventory: dict[str, Any],
    output_root: Path,
    worker_index: int,
    num_workers: int,
    torch_threads: int,
    max_sources: int | None,
) -> dict[str, int]:
    if num_workers <= 0 or worker_index < 0 or worker_index >= num_workers:
        raise ValueError("Invalid worker index/count.")
    output_root = output_root.expanduser().resolve()
    duplicate_of = exact_duplicate_decisions(inventory)
    assigned = [
        index
        for index in range(len(inventory["source_records"]))
        if index % num_workers == worker_index
    ]
    if max_sources is not None:
        if max_sources <= 0:
            raise ValueError("max_sources must be positive.")
        assigned = assigned[:max_sources]
    model = None
    created = 0
    reused = 0
    for position, source_index in enumerate(assigned, 1):
        path = _part_path(output_root, source_index)
        source_record = inventory["source_records"][source_index]
        duplicate = duplicate_of.get(str(source_record["path"]))
        if path.is_file():
            payload = _load_json(path, label=f"Stage211 social VAD source {source_index}")
            validate_source_payload(
                payload,
                inventory=inventory,
                source_index=source_index,
                duplicate_of=duplicate,
            )
            reused += 1
            if position % 25 == 0 or position == len(assigned):
                print(
                    f"[stage211-social-vad] worker={worker_index}/{num_workers} "
                    f"sources={position}/{len(assigned)} created={created} reused={reused}",
                    flush=True,
                )
            continue
        if duplicate is None:
            if model is None:
                model = _load_model(inventory, torch_threads=torch_threads)
            raw_segments = _generate_raw_segments(model, str(source_record["path"]))
        else:
            raw_segments = None
        payload = build_source_payload(
            inventory=inventory,
            source_index=source_index,
            raw_segments=raw_segments,
            duplicate_of=duplicate,
        )
        status = _write_immutable_json(path, payload)
        created += int(status == "created")
        reused += int(status == "reused")
        if position % 25 == 0 or position == len(assigned):
            print(
                f"[stage211-social-vad] worker={worker_index}/{num_workers} "
                f"sources={position}/{len(assigned)} created={created} reused={reused}",
                flush=True,
            )
    return {"assigned": len(assigned), "created": created, "reused": reused}


def finalize_boundaries(*, inventory: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    part_records: list[dict[str, Any]] = []
    admitted_sources = 0
    duplicate_sources = 0
    no_speech_sources = 0
    admitted_segments = 0
    admitted_duration_ms = 0
    source_duration_ms = 0
    source_counts: dict[str, int] = {}
    segment_counts: dict[str, int] = {}
    segment_hours: dict[str, float] = {}
    duplicate_of = exact_duplicate_decisions(inventory)
    for source_index, source in enumerate(inventory["source_records"]):
        path = _part_path(output_root, source_index)
        payload = _load_json(path, label=f"Stage211 social VAD source {source_index}")
        validate_source_payload(
            payload,
            inventory=inventory,
            source_index=source_index,
            duplicate_of=duplicate_of.get(str(source["path"])),
        )
        label = str(source["source_label"])
        source_counts[label] = source_counts.get(label, 0) + 1
        source_duration_ms += int(round(float(source["probe"]["duration_seconds"]) * 1000.0))
        status = payload.get("status")
        if status == "admitted":
            admitted_sources += 1
            count = int(payload.get("admitted_segments", -1))
            duration = int(payload.get("admitted_duration_ms", -1))
            if count <= 0 or duration <= 0 or len(payload.get("segments") or []) != count:
                raise ValueError(f"Stage211 social VAD admitted source is empty: {path}")
            for segment in payload["segments"]:
                segment_duration = int(segment.get("duration_ms", -1))
                if not MIN_SEGMENT_MS <= segment_duration <= MAX_SEGMENT_MS:
                    raise ValueError(f"Stage211 social VAD segment duration is invalid: {path}")
            admitted_segments += count
            admitted_duration_ms += duration
            segment_counts[label] = segment_counts.get(label, 0) + count
            segment_hours[label] = segment_hours.get(label, 0.0) + duration / 3_600_000.0
        elif status == "exact_duplicate_skipped":
            duplicate_sources += 1
        elif status == "no_speech_rejected":
            no_speech_sources += 1
        else:
            raise ValueError(f"Stage211 social VAD source status is invalid: {path}")
        part_records.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source_index": source_index,
                "status": status,
                "admitted_segments": int(payload.get("admitted_segments", 0)),
                "admitted_duration_ms": int(payload.get("admitted_duration_ms", 0)),
            }
        )
    payload = {
        "schema_version": 1,
        "artifact": INVENTORY_ARTIFACT,
        "complete": True,
        "training_ready": False,
        "admission_state": "vad_boundaries_only",
        "source_inventory_path": inventory["_inventory_path"],
        "source_inventory_sha256": inventory["_inventory_sha256"],
        "vad": _vad_binding(inventory),
        "source_files": len(inventory["source_records"]),
        "admitted_source_files": admitted_sources,
        "exact_duplicate_source_files": duplicate_sources,
        "no_speech_source_files": no_speech_sources,
        "source_hours": source_duration_ms / 3_600_000.0,
        "admitted_segments": admitted_segments,
        "admitted_hours": admitted_duration_ms / 3_600_000.0,
        "source_counts": dict(sorted(source_counts.items())),
        "segment_counts": dict(sorted(segment_counts.items())),
        "segment_hours": dict(sorted(segment_hours.items())),
        "part_records": part_records,
        "required_before_training": [
            "materialize receipt-bound mono 16 kHz segment shards",
            "verify every materialized segment decodes and matches its duration",
            "complete segment-level duplicate and public-overlap audit",
            "build bucket manifest with the shared fixed hidden-eval split",
            "atomically combine with the base supplemental-natural inventory",
        ],
    }
    _write_immutable_json(output_root / "boundary_inventory.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic FunASR FSMN-VAD boundaries for Stage211 social audio."
    )
    parser.add_argument("--source-inventory", type=Path, default=DEFAULT_SOURCE_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--verify-media-hashes", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = validate_source_inventory(
        args.source_inventory,
        verify_media_hashes=bool(args.verify_media_hashes),
    )
    if args.finalize_only:
        result = finalize_boundaries(inventory=inventory, output_root=args.output_root)
        print(
            f"boundary_inventory={args.output_root.expanduser().resolve() / 'boundary_inventory.json'} "
            f"segments={result['admitted_segments']} hours={result['admitted_hours']:.6f}",
            flush=True,
        )
        return 0
    run_worker(
        inventory=inventory,
        output_root=args.output_root,
        worker_index=int(args.worker_index),
        num_workers=int(args.num_workers),
        torch_threads=int(args.torch_threads),
        max_sources=args.max_sources,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
