#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_ROOT = (
    "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/"
    "stages/stage179_usbhd_dedup_online_ctc_alignment"
)

DEFAULT_INPUTS: tuple[tuple[str, str, str], ...] = (
    (
        "cv22",
        "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/"
        "stages/stage178_cv22_audio_only_online_ctc_alignment/stage178a_cv22_large1m_online_ctc/"
        "webdataset_lengths.jsonl",
        "/media/usbhd/common_voice_22/webdataset",
    ),
    (
        "clean_public",
        "/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/webdataset_lengths.jsonl",
        "/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset",
    ),
    (
        "giga_wenet",
        "/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset/webdataset_lengths.jsonl",
        "/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset",
    ),
    (
        "emilia",
        "/media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH/webdataset_lengths.jsonl",
        "/media/usbhd/training_data/asr/emilia/Emilia/MIX_EN_ZH",
    ),
)

SOURCE_LANGUAGES = {
    "aishell3": "zh",
    "commonvoice_cn": "zh",
    "commonvoice_en": "en",
    "cv22_en": "en",
    "cv22_zh": "zh",
    "emilia_en": "en",
    "emilia_zh": "zh",
    "gigaspeech": "en",
    "librispeech": "en",
    "wenetspeech": "zh",
}

DIFFICULTY_ORDER = ("easy", "medium", "hard", "long")
STAGE_NAMES = {
    "easy": "stage179a_easy_dedup_audio_only_online_ctc",
    "medium": "stage179b_medium_dedup_audio_only_online_ctc",
    "hard": "stage179c_hard_dedup_audio_only_online_ctc",
    "long": "stage179d_long_dedup_audio_only_online_ctc",
}


@dataclass(frozen=True)
class InputSpec:
    name: str
    length_index: Path
    root: Path


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def _input_specs(values: list[str] | None) -> list[InputSpec]:
    raw_specs = values or [":".join(item) for item in DEFAULT_INPUTS]
    specs: list[InputSpec] = []
    for raw in raw_specs:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "--input must be name:length_index:root, "
                f"got {raw!r}"
            )
        name, length_index, root = parts
        specs.append(InputSpec(name=name, length_index=Path(length_index), root=Path(root)))
    return specs


def _utt_id(row: dict[str, Any]) -> str:
    for key in ("utt_id", "id", "key", "sid"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _source_from_row(row: dict[str, Any], *, input_name: str) -> str:
    locale = str(row.get("_stage178_locale") or row.get("locale") or "").strip()
    if input_name == "cv22":
        if locale == "en":
            return "cv22_en"
        if locale in {"zh-CN", "zh-HK", "zh-TW"}:
            return "cv22_zh"

    source = str(row.get("source_dataset") or row.get("source") or row.get("dataset") or "").strip().lower()
    if source in SOURCE_LANGUAGES:
        return source
    if source == "commonvoice":
        lang = str(row.get("language") or row.get("lang") or "").lower()
        if lang.startswith("zh"):
            return "commonvoice_cn"
        if lang.startswith("en"):
            return "commonvoice_en"

    shard = Path(str(row.get("shard_name") or row.get("shard") or "")).name
    shard_lower = shard.lower()
    key_text = f"{row.get('key') or ''} {_utt_id(row)}".lower()
    if shard.startswith("GSXL-") or "gigaspeech" in key_text:
        return "gigaspeech"
    if shard.startswith("WSL-") or "wenetspeech" in key_text:
        return "wenetspeech"
    if shard_lower.startswith("aishell3") or "aishell" in key_text:
        return "aishell3"
    if shard_lower.startswith("librispeech") or "librispeech" in key_text:
        return "librispeech"
    if shard_lower.startswith("commonvoice_en"):
        return "commonvoice_en"
    if shard_lower.startswith("commonvoice_cn"):
        return "commonvoice_cn"
    if input_name == "emilia":
        if shard.startswith("EN-") or shard.startswith("EN_") or str(row.get("language") or "").lower() == "en":
            return "emilia_en"
        if shard.startswith("ZH-") or shard.startswith("ZH_") or str(row.get("language") or "").lower().startswith("zh"):
            return "emilia_zh"
    return source or f"unknown_{input_name}"


def _frame_count(row: dict[str, Any]) -> int:
    value = row.get("num_frames")
    if value is None:
        return 0
    return int(value)


def _resolve_tar_path(row: dict[str, Any], root: Path) -> Path:
    raw = str(row.get("tar_path") or row.get("shard_path") or "").strip()
    if raw:
        path = Path(raw)
    else:
        shard_name = str(row.get("shard_name") or row.get("shard") or "").strip()
        if not shard_name:
            raise ValueError(f"row has no shard_name/tar_path: {row}")
        path = Path(shard_name)
    if path.is_absolute():
        return path
    return root / path


def _dedupe_key(row: dict[str, Any], *, root: Path) -> bytes:
    tar_path = _resolve_tar_path(row, root)
    audio_member = str(row.get("audio_member") or row.get("wav_member") or row.get("audio") or "").strip()
    if not audio_member:
        audio_member = _utt_id(row)
    payload = "\0".join(
        (
            str(tar_path),
            audio_member,
            str(row.get("audio_offset") or ""),
            str(row.get("audio_size") or ""),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8", errors="surrogatepass"), digest_size=16).digest()


def _difficulty(source: str, frames: int) -> str:
    cleanish = {"aishell3", "librispeech"}
    crowd = {"cv22_en", "cv22_zh", "commonvoice_en", "commonvoice_cn"}
    broad = {"gigaspeech", "wenetspeech"}
    synthetic_or_web = {"emilia_en", "emilia_zh"}

    if frames > 3000:
        return "long"
    if source in cleanish and frames <= 900:
        return "easy"
    if source in cleanish and frames <= 1800:
        return "medium"
    if source in crowd and frames <= 700:
        return "easy"
    if source in crowd and frames <= 1500:
        return "medium"
    if source in broad and frames <= 900:
        return "medium"
    if source in synthetic_or_web and frames <= 800:
        return "medium"
    return "hard"


def _absolute_training_row(
    row: dict[str, Any],
    *,
    input_spec: InputSpec,
    source: str,
    difficulty: str,
    stage_name: str,
    seed: int,
    sample_index: int,
) -> dict[str, Any]:
    result = dict(row)
    tar_path = _resolve_tar_path(row, input_spec.root)
    result["shard_name"] = str(tar_path)
    result["tar_path"] = str(tar_path)
    result["source_dataset"] = source
    result["language"] = str(row.get("language") or row.get("lang") or SOURCE_LANGUAGES.get(source, "unknown"))
    result["_stage179_stage"] = stage_name
    result["_stage179_curriculum"] = "usbhd_dedup_audio_only_online_ctc_alignment"
    result["_stage179_input"] = input_spec.name
    result["_stage179_source"] = source
    result["_stage179_difficulty"] = difficulty
    result["_stage179_seed"] = int(seed)
    result["_stage179_sample_index"] = int(sample_index)
    result["_stage179_ctc_online_only"] = True
    result["_stage179_uses_text_labels"] = False
    return result


def _distribution_from_hist(hist: Counter[int]) -> dict[str, float | int | None]:
    count = sum(hist.values())
    if count <= 0:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    ordered = sorted(hist.items())
    total = sum(frame * n for frame, n in ordered)

    def percentile(q: float) -> float:
        target = q * (count - 1)
        running = 0
        for frame, n in ordered:
            next_running = running + n
            if target < next_running:
                return float(frame)
            running = next_running
        return float(ordered[-1][0])

    return {
        "count": int(count),
        "min": float(ordered[0][0]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": float(ordered[-1][0]),
        "mean": float(total / count),
    }


def _stats_payload(
    *,
    stage_name: str,
    difficulty: str,
    rows: int,
    unique_shards: set[str],
    source_counts: Counter[str],
    source_frames: Counter[str],
    frame_hist: Counter[int],
    length_path: Path,
) -> dict[str, Any]:
    hours = {source: frames / 100.0 / 3600.0 for source, frames in source_frames.items()}
    return {
        "version": 1,
        "stage_name": stage_name,
        "curriculum": "usbhd_dedup_audio_only_online_ctc_alignment",
        "root": "/",
        "difficulty": difficulty,
        "uses_text_labels": False,
        "teacher": "FunASR-Nano-2512 online CTC logits",
        "selected_rows": int(rows),
        "selected_audio_unique_rows": int(rows),
        "selected_hours": sum(source_frames.values()) / 100.0 / 3600.0,
        "selected_counts": dict(sorted(source_counts.items())),
        "selected_hours_by_source": dict(sorted((k, round(v, 3)) for k, v in hours.items())),
        "num_shards": len(unique_shards),
        "num_frames_distribution": _distribution_from_hist(frame_hist),
        "length_index_path": str(length_path),
        "bucket_manifest_path": str(length_path.parent / "webdataset_buckets_audio_text" / "manifest.json"),
    }


def _build_bucket_manifest(repo_root: Path, stage_dir: Path, bucket_width: int) -> None:
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
        "/",
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
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage179: deduplicated /media/usbhd audio-only CTC-logits distillation curriculum."
    )
    parser.add_argument("--input", action="append", help="name:length_index:root. Defaults cover CV22, clean, Giga/Wenet, Emilia.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-rows-per-input", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    parser.add_argument("--bucket-width", type=int, default=200)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--build-bucket-manifests", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    specs = _input_specs(args.input)
    output_root = Path(args.output_root)
    repo_root = Path(args.repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    handles: dict[str, Any] = {}
    stage_dirs: dict[str, Path] = {}
    length_paths: dict[str, Path] = {}
    if not args.analyze_only:
        for difficulty in DIFFICULTY_ORDER:
            stage_name = STAGE_NAMES[difficulty]
            stage_dir = output_root / stage_name
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_dirs[difficulty] = stage_dir
            length_path = stage_dir / "webdataset_lengths.jsonl"
            length_paths[difficulty] = length_path
            handles[difficulty] = length_path.open("w", encoding="utf-8")

    seen: set[bytes] = set()
    rows_by_diff: Counter[str] = Counter()
    frames_by_diff: Counter[str] = Counter()
    source_counts_by_diff: dict[str, Counter[str]] = defaultdict(Counter)
    source_frames_by_diff: dict[str, Counter[str]] = defaultdict(Counter)
    frame_hist_by_diff: dict[str, Counter[int]] = defaultdict(Counter)
    shards_by_diff: dict[str, set[str]] = defaultdict(set)
    input_summary: dict[str, Any] = {}
    skipped = Counter()

    try:
        for spec in specs:
            _log(f"scan start input={spec.name} path={spec.length_index}")
            scanned = 0
            split_rows = 0
            accepted = 0
            dupes = 0
            source_counts = Counter()
            source_frames = Counter()
            for row in _iter_jsonl(spec.length_index):
                scanned += 1
                if int(args.max_rows_per_input) > 0 and scanned > int(args.max_rows_per_input):
                    break
                if str(row.get("split") or "train") != str(args.split):
                    skipped["nontrain"] += 1
                    continue
                split_rows += 1
                frames = _frame_count(row)
                if frames < int(args.min_frames):
                    skipped["too_short"] += 1
                    continue
                key = _dedupe_key(row, root=spec.root)
                if key in seen:
                    dupes += 1
                    skipped["duplicate_audio"] += 1
                    continue
                seen.add(key)
                source = _source_from_row(row, input_name=spec.name)
                difficulty = _difficulty(source, frames)
                stage_name = STAGE_NAMES[difficulty]
                source_counts[source] += 1
                source_frames[source] += frames
                rows_by_diff[difficulty] += 1
                frames_by_diff[difficulty] += frames
                source_counts_by_diff[difficulty][source] += 1
                source_frames_by_diff[difficulty][source] += frames
                frame_hist_by_diff[difficulty][frames] += 1
                shards_by_diff[difficulty].add(str(_resolve_tar_path(row, spec.root)))
                accepted += 1
                if not args.analyze_only:
                    sample_index = rows_by_diff[difficulty] - 1
                    out_row = _absolute_training_row(
                        row,
                        input_spec=spec,
                        source=source,
                        difficulty=difficulty,
                        stage_name=stage_name,
                        seed=int(args.seed),
                        sample_index=sample_index,
                    )
                    handles[difficulty].write(json.dumps(out_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if int(args.progress_interval) > 0 and scanned % int(args.progress_interval) == 0:
                    _log(
                        f"scan progress input={spec.name} scanned={scanned} split={split_rows} "
                        f"accepted={accepted} dedup_total={len(seen)}"
                    )
            input_summary[spec.name] = {
                "length_index_path": str(spec.length_index),
                "root": str(spec.root),
                "rows_scanned": int(scanned),
                "split_rows": int(split_rows),
                "accepted_unique_audio_rows": int(accepted),
                "duplicate_audio_rows": int(dupes),
                "accepted_hours": round(sum(source_frames.values()) / 100.0 / 3600.0, 3),
                "accepted_counts_by_source": dict(sorted(source_counts.items())),
                "accepted_hours_by_source": dict(
                    sorted((source, round(frames / 100.0 / 3600.0, 3)) for source, frames in source_frames.items())
                ),
            }
            _log(
                f"scan done input={spec.name} scanned={scanned} split={split_rows} "
                f"accepted={accepted} duplicates={dupes}"
            )
    finally:
        for handle in handles.values():
            handle.close()

    stages: dict[str, Any] = {}
    for difficulty in DIFFICULTY_ORDER:
        stage_name = STAGE_NAMES[difficulty]
        stage_dir = output_root / stage_name
        length_path = stage_dir / "webdataset_lengths.jsonl"
        if args.analyze_only:
            length_path = output_root / stage_name / "webdataset_lengths.jsonl"
        payload = _stats_payload(
            stage_name=stage_name,
            difficulty=difficulty,
            rows=rows_by_diff[difficulty],
            unique_shards=shards_by_diff[difficulty],
            source_counts=source_counts_by_diff[difficulty],
            source_frames=source_frames_by_diff[difficulty],
            frame_hist=frame_hist_by_diff[difficulty],
            length_path=length_path,
        )
        stages[stage_name] = payload
        if not args.analyze_only:
            (stage_dir / "webdataset_lengths.summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    total_frames = sum(frames_by_diff.values())
    manifest = {
        "version": 1,
        "root": "/",
        "curriculum": "usbhd_dedup_audio_only_online_ctc_alignment",
        "split": str(args.split),
        "seed": int(args.seed),
        "min_frames": int(args.min_frames),
        "dedupe_key": "blake2b16(root_or_tar_path + audio_member + audio_offset + audio_size)",
        "inputs": input_summary,
        "skipped": dict(sorted(skipped.items())),
        "total_unique_audio_rows": int(sum(rows_by_diff.values())),
        "total_unique_hours": total_frames / 100.0 / 3600.0,
        "total_unique_hours_rounded": round(total_frames / 100.0 / 3600.0, 3),
        "rows_by_difficulty": dict((name, int(rows_by_diff[name])) for name in DIFFICULTY_ORDER),
        "hours_by_difficulty": dict(
            (name, round(frames_by_diff[name] / 100.0 / 3600.0, 3)) for name in DIFFICULTY_ORDER
        ),
        "stages": stages,
    }
    manifest_path = output_root / "stage179_usbhd_dedup_alignment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(
        "dedup summary "
        f"rows={manifest['total_unique_audio_rows']} hours={manifest['total_unique_hours_rounded']} "
        f"manifest={manifest_path}"
    )

    if args.build_bucket_manifests and not args.analyze_only:
        for difficulty in DIFFICULTY_ORDER:
            stage_dir = output_root / STAGE_NAMES[difficulty]
            _log(f"bucket manifest start difficulty={difficulty} dir={stage_dir}")
            _build_bucket_manifest(repo_root, stage_dir, int(args.bucket_width))
            _log(f"bucket manifest done difficulty={difficulty}")
    elif args.build_bucket_manifests and args.analyze_only:
        _log("--build-bucket-manifests ignored with --analyze-only")


if __name__ == "__main__":
    main()
