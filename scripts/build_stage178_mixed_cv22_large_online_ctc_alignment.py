#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CLEAN_LENGTHS = "/media/usbhd/training_data/asr/curriculum/clean_ctc_voxbox_webdataset/webdataset_lengths.jsonl"
DEFAULT_HARD_LENGTHS = "/media/usbhd/training_data/asr/mix/gigaspeech_xl_wenetspeech_l_webdataset/webdataset_lengths.jsonl"
DEFAULT_CV22_LENGTHS = (
    "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/"
    "stages/stage178_cv22_audio_only_online_ctc_alignment/stage178a_cv22_large1m_online_ctc/"
    "webdataset_lengths.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/"
    "stages/stage178_cv22_audio_only_online_ctc_alignment"
)
DEFAULT_STAGE_NAME = "stage178b_cv22_all_plus_large1m_online_ctc"
DEFAULT_TARGET_TOTAL = 1_000_000

SOURCE_LANGUAGES = {
    "aishell3": "zh",
    "cv22_en": "en",
    "cv22_zh": "zh",
    "gigaspeech": "en",
    "librispeech": "en",
    "wenetspeech": "zh",
}

DEFAULT_FIXED_QUOTAS: tuple[tuple[str, int], ...] = (
    ("aishell3", 62_952),
    ("librispeech", 180_000),
    ("gigaspeech", 300_000),
)


@dataclass
class Reservoir:
    size: int
    rng: random.Random
    seen: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    selected_utt_ids: set[str] = field(default_factory=set)

    def add(self, row: dict[str, Any], utt_id: str) -> None:
        if self.size <= 0 or utt_id in self.selected_utt_ids:
            return
        self.seen += 1
        if len(self.rows) < self.size:
            self.rows.append(row)
            self.selected_utt_ids.add(utt_id)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            old_row = self.rows[index]
            self.selected_utt_ids.discard(_utt_id(old_row))
            self.rows[index] = row
            self.selected_utt_ids.add(utt_id)


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


def _utt_id(row: dict[str, Any]) -> str:
    value = row.get("utt_id") or row.get("id") or row.get("key")
    if value is None:
        raise ValueError(f"row has no utt id: {row}")
    return str(value)


def _source_from_row(row: dict[str, Any]) -> str:
    stage178_locale = str(row.get("_stage178_locale") or "").strip()
    if stage178_locale == "en":
        return "cv22_en"
    if stage178_locale in {"zh-CN", "zh-HK", "zh-TW"}:
        return "cv22_zh"
    source = str(row.get("source_dataset") or row.get("source") or "").strip().lower()
    if source == "commonvoice_en":
        return "old_commonvoice_en"
    if source == "commonvoice_cn":
        return "old_commonvoice_cn"
    if source in SOURCE_LANGUAGES:
        return source
    shard_name = str(row.get("shard_name") or row.get("shard") or "")
    if shard_name.startswith("GSXL-"):
        return "gigaspeech"
    if shard_name.startswith("WSL-"):
        return "wenetspeech"
    if shard_name.startswith("aishell3_"):
        return "aishell3"
    if shard_name.startswith("librispeech_"):
        return "librispeech"
    if shard_name.startswith("commonvoice_en_"):
        return "old_commonvoice_en"
    if shard_name.startswith("commonvoice_cn_"):
        return "old_commonvoice_cn"
    return source or "unknown"


def _frame_count(row: dict[str, Any]) -> int:
    value = row.get("num_frames")
    if value is None:
        return 0
    return int(value)


def _parse_quota(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"quota must be source:count, got {value!r}")
    source, raw_count = value.split(":", 1)
    source = source.strip().lower()
    if source not in SOURCE_LANGUAGES and source != "wenetspeech":
        raise argparse.ArgumentTypeError(
            f"unsupported source {source!r}; expected one of {sorted(SOURCE_LANGUAGES)}"
        )
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"quota count must be an integer, got {value!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError(f"quota count must be non-negative, got {value!r}")
    return source, count


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "mean": None}
    values = sorted(values)

    def pct(q: float) -> float:
        pos = q * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(values[lo])
        frac = pos - lo
        return float(values[lo] * (1.0 - frac) + values[hi] * frac)

    return {
        "count": len(values),
        "min": float(values[0]),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": float(values[-1]),
        "mean": float(sum(values) / len(values)),
    }


def _resolve_tar_path(row: dict[str, Any], root: Path | None) -> Path:
    raw_tar_path = str(row.get("tar_path") or row.get("shard_path") or "").strip()
    if raw_tar_path:
        return Path(raw_tar_path)
    shard_name = str(row.get("shard_name") or row.get("shard") or "").strip()
    if not shard_name:
        raise ValueError(f"row has no shard_name/tar_path: {row}")
    shard_path = Path(shard_name)
    if shard_path.is_absolute():
        return shard_path
    if root is None:
        raise ValueError(f"relative shard_name has no root: {shard_name}")
    return root / shard_path


def _absolute_training_row(
    row: dict[str, Any],
    *,
    source: str,
    stage_name: str,
    input_path: Path,
    input_root: Path | None,
    seed: int,
    min_frames: int,
    max_frames: int,
    quota: int,
    sample_index: int,
) -> dict[str, Any]:
    result = dict(row)
    tar_path = _resolve_tar_path(row, input_root)
    if not tar_path.is_absolute():
        tar_path = tar_path.absolute()
    result["shard_name"] = str(tar_path)
    result["tar_path"] = str(tar_path)
    result["source_dataset"] = (
        "commonvoice_en"
        if source == "cv22_en"
        else "commonvoice_cn"
        if source == "cv22_zh"
        else source
    )
    result["language"] = SOURCE_LANGUAGES[source]
    result["_stage178b_stage"] = stage_name
    result["_stage178b_curriculum"] = "cv22_all_plus_large_audio_only_online_ctc_alignment"
    result["_stage178b_source"] = source
    result["_stage178b_source_length_index"] = str(input_path)
    result["_stage178b_seed"] = int(seed)
    result["_stage178b_min_frames"] = int(min_frames)
    result["_stage178b_max_frames"] = int(max_frames)
    result["_stage178b_source_quota"] = int(quota)
    result["_stage178b_sample_index"] = int(sample_index)
    result["_stage178b_ctc_online_only"] = True
    result["_stage178b_uses_text_labels"] = False
    return result


def _load_cv22_rows(
    *,
    cv22_lengths: Path,
    stage_name: str,
    seed: int,
    min_frames: int,
    max_frames: int,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped = {"too_short": 0, "too_long": 0, "old_commonvoice": 0, "other": 0}
    for row in _iter_jsonl(cv22_lengths):
        source = _source_from_row(row)
        if source not in {"cv22_en", "cv22_zh"}:
            skipped["other"] += 1
            continue
        frames = _frame_count(row)
        if frames < min_frames:
            skipped["too_short"] += 1
            continue
        if max_frames > 0 and frames > max_frames:
            skipped["too_long"] += 1
            continue
        counts[source] += 1
        rows.append(
            _absolute_training_row(
                row,
                source=source,
                stage_name=stage_name,
                input_path=cv22_lengths,
                input_root=None,
                seed=seed,
                min_frames=min_frames,
                max_frames=max_frames,
                quota=0,
                sample_index=counts[source] - 1,
            )
        )
    return rows, counts, skipped


def _scan_reservoir_sources(
    *,
    length_paths: list[tuple[Path, Path]],
    quotas: dict[str, int],
    stage_name: str,
    split: str,
    seed: int,
    min_frames: int,
    max_frames: int,
    max_rows_per_index: int,
    progress_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reservoirs = {
        source: Reservoir(size=quota, rng=random.Random(seed + index * 104_729))
        for index, (source, quota) in enumerate(sorted(quotas.items()))
        if quota > 0
    }
    total_rows = 0
    split_rows = 0
    candidate_rows = 0
    source_counts: Counter[str] = Counter()
    source_candidates: Counter[str] = Counter()
    skipped_by_source: Counter[str] = Counter()
    skipped_nontrain = 0
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_old_commonvoice = 0
    skipped_unknown_source = 0

    for input_path, input_root in length_paths:
        _log(f"scan start path={input_path}")
        input_rows = 0
        for row in _iter_jsonl(input_path):
            total_rows += 1
            input_rows += 1
            if str(row.get("split") or "train") != split:
                skipped_nontrain += 1
                continue
            split_rows += 1
            source = _source_from_row(row)
            source_counts[source] += 1
            if source in {"old_commonvoice_en", "old_commonvoice_cn"}:
                skipped_old_commonvoice += 1
                skipped_by_source[source] += 1
                continue
            if source not in reservoirs:
                skipped_unknown_source += 1
                skipped_by_source[source] += 1
                continue
            frames = _frame_count(row)
            if frames < min_frames:
                skipped_too_short += 1
                skipped_by_source[source] += 1
                continue
            if max_frames > 0 and frames > max_frames:
                skipped_too_long += 1
                skipped_by_source[source] += 1
                continue
            candidate_rows += 1
            source_candidates[source] += 1
            reservoirs[source].add(row, _utt_id(row))
            if progress_interval > 0 and total_rows % progress_interval == 0:
                _log(
                    "scan progress "
                    f"rows={total_rows} split={split_rows} candidates={candidate_rows} "
                    f"selected={sum(len(item.rows) for item in reservoirs.values())}"
                )
            if max_rows_per_index > 0 and input_rows >= max_rows_per_index:
                break
        _log(f"scan done path={input_path} rows={input_rows}")

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for source, reservoir in sorted(reservoirs.items()):
        if len(reservoir.rows) < reservoir.size:
            _log(f"warning source={source} requested={reservoir.size} selected={len(reservoir.rows)}")
        reservoir.rows.sort(key=lambda row: (int(row.get("num_frames") or 0), _utt_id(row)))
        source_input = next(
            (item for item in length_paths if _source_matches_path(source, item[0])),
            length_paths[0],
        )
        for sample_index, row in enumerate(reservoir.rows):
            input_path, input_root = source_input
            selected.append(
                _absolute_training_row(
                    row,
                    source=source,
                    stage_name=stage_name,
                    input_path=input_path,
                    input_root=input_root,
                    seed=seed,
                    min_frames=min_frames,
                    max_frames=max_frames,
                    quota=int(quotas[source]),
                    sample_index=sample_index,
                )
            )
            selected_counts[source] += 1

    summary = {
        "total_rows_scanned": int(total_rows),
        "split_rows": int(split_rows),
        "candidate_rows": int(candidate_rows),
        "source_counts_before_frame_filter": dict(sorted(source_counts.items())),
        "source_candidate_counts": dict(sorted(source_candidates.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "skipped_nontrain": int(skipped_nontrain),
        "skipped_too_short": int(skipped_too_short),
        "skipped_too_long": int(skipped_too_long),
        "skipped_old_commonvoice": int(skipped_old_commonvoice),
        "skipped_unknown_source": int(skipped_unknown_source),
        "skipped_by_source": dict(sorted(skipped_by_source.items())),
    }
    return selected, summary


def _source_matches_path(source: str, path: Path) -> bool:
    text = str(path)
    if source in {"gigaspeech", "wenetspeech"}:
        return "gigaspeech_xl_wenetspeech" in text
    return "clean_ctc_voxbox" in text


def _write_stage(stage_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    length_path = stage_dir / "webdataset_lengths.jsonl"
    with length_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    payload = {
        **summary,
        "length_index_path": str(length_path),
        "bucket_manifest_path": str(stage_dir / "webdataset_buckets_audio_text" / "manifest.json"),
    }
    (stage_dir / "webdataset_lengths.summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
        description="Build Stage178b: CV22 all rows plus large non-CommonVoice audio-only online CTC pool."
    )
    parser.add_argument("--cv22-lengths", default=DEFAULT_CV22_LENGTHS)
    parser.add_argument("--clean-lengths", default=DEFAULT_CLEAN_LENGTHS)
    parser.add_argument("--hard-lengths", default=DEFAULT_HARD_LENGTHS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--quota", action="append", type=_parse_quota)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--bucket-width", type=int, default=200)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--progress-interval", type=int, default=1_000_000)
    parser.add_argument("--max-rows-per-index", type=int, default=0)
    parser.add_argument("--skip-bucket-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cv22_lengths = Path(args.cv22_lengths)
    clean_lengths = Path(args.clean_lengths)
    hard_lengths = Path(args.hard_lengths)
    output_root = Path(args.output_root)
    stage_name = str(args.stage_name)
    stage_dir = output_root / stage_name
    repo_root = Path(args.repo_root)

    cv22_rows, cv22_counts, cv22_skipped = _load_cv22_rows(
        cv22_lengths=cv22_lengths,
        stage_name=stage_name,
        seed=int(args.seed),
        min_frames=int(args.min_frames),
        max_frames=int(args.max_frames),
    )
    quotas = dict(DEFAULT_FIXED_QUOTAS)
    if args.quota:
        quotas.update(dict(args.quota))
    fixed_total = len(cv22_rows) + sum(int(value) for value in quotas.values())
    quotas["wenetspeech"] = max(0, int(args.target_total) - fixed_total)
    _log(
        f"build stage={stage_name} cv22_rows={len(cv22_rows)} "
        f"non_cv_quotas={dict(sorted(quotas.items()))} target={int(args.target_total)} output={stage_dir}"
    )

    non_cv_rows, non_cv_summary = _scan_reservoir_sources(
        length_paths=[
            (clean_lengths, clean_lengths.parent),
            (hard_lengths, hard_lengths.parent),
        ],
        quotas=quotas,
        stage_name=stage_name,
        split=str(args.split),
        seed=int(args.seed),
        min_frames=int(args.min_frames),
        max_frames=int(args.max_frames),
        max_rows_per_index=int(args.max_rows_per_index),
        progress_interval=int(args.progress_interval),
    )
    rows = cv22_rows + non_cv_rows
    rows.sort(
        key=lambda row: (
            int(row.get("num_frames") or 0),
            str(row.get("_stage178b_source") or ""),
            str(row.get("utt_id") or row.get("key") or ""),
        )
    )
    frames = [float(row.get("num_frames") or 0.0) for row in rows]
    selected_counts: Counter[str] = Counter(str(row.get("_stage178b_source") or "unknown") for row in rows)
    summary = {
        "version": 1,
        "stage_name": stage_name,
        "curriculum": "cv22_all_plus_large_audio_only_online_ctc_alignment",
        "root": "/",
        "uses_text_labels": False,
        "teacher": "FunASR-Nano-2512 online CTC logits",
        "cv22_length_index_path": str(cv22_lengths),
        "clean_length_index_path": str(clean_lengths),
        "hard_length_index_path": str(hard_lengths),
        "split": str(args.split),
        "seed": int(args.seed),
        "min_frames": int(args.min_frames),
        "max_frames": int(args.max_frames),
        "target_total": int(args.target_total),
        "cv22_rows": len(cv22_rows),
        "cv22_selected_counts": dict(sorted(cv22_counts.items())),
        "cv22_skipped": cv22_skipped,
        "non_cv_quotas": dict(sorted(quotas.items())),
        "selected_rows": len(rows),
        "selected_unique_utt_ids": len({str(row.get("utt_id") or row.get("key") or "") for row in rows}),
        "selected_counts": dict(sorted(selected_counts.items())),
        "num_shards": len({str(row.get("shard_name") or "") for row in rows}),
        "num_frames_distribution": _distribution(frames),
        "non_cv_summary": non_cv_summary,
    }
    _write_stage(stage_dir, rows, summary)
    _log(f"length index written rows={len(rows)} dir={stage_dir}")
    if not args.skip_bucket_manifest:
        _build_bucket_manifest(repo_root, stage_dir, int(args.bucket_width))
    manifest = {
        "version": 1,
        "root": "/",
        "output_root": str(output_root),
        "stages": {
            stage_name: json.loads(
                (stage_dir / "webdataset_lengths.summary.json").read_text(encoding="utf-8")
            )
        },
    }
    (output_root / "stage178_alignment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"Stage178b mixed alignment pool complete output_root={output_root}")


if __name__ == "__main__":
    main()
