#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import random
import subprocess
import tarfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WEBDATASET_ROOT = "/media/usbhd/common_voice_22/webdataset"
DEFAULT_OUTPUT_ROOT = (
    "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/"
    "stages/stage178_cv22_audio_only_online_ctc_alignment"
)
DEFAULT_STAGE_NAME = "stage178a_cv22_large1m_online_ctc"

DEFAULT_QUOTAS: tuple[tuple[str, int], ...] = (
    ("en", 800_000),
    ("zh-CN", 120_000),
    ("zh-HK", 40_000),
    ("zh-TW", 40_000),
)

LOCALE_TO_SOURCE = {
    "en": "commonvoice_en",
    "zh-CN": "commonvoice_cn",
    "zh-HK": "commonvoice_cn",
    "zh-TW": "commonvoice_cn",
}

LOCALE_TO_LANGUAGE = {
    "en": "en",
    "zh-CN": "zh",
    "zh-HK": "zh",
    "zh-TW": "zh",
}


@dataclass
class Reservoir:
    size: int
    rng: random.Random
    seen: int = 0
    rows: list[dict[str, Any]] = field(default_factory=list)
    selected_ids: set[str] = field(default_factory=set)

    def add(self, row: dict[str, Any], sample_id: str) -> None:
        if self.size <= 0 or sample_id in self.selected_ids:
            return
        self.seen += 1
        if len(self.rows) < self.size:
            self.rows.append(row)
            self.selected_ids.add(sample_id)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            old_id = str(self.rows[index]["utt_id"])
            self.selected_ids.discard(old_id)
            self.rows[index] = row
            self.selected_ids.add(sample_id)


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _parse_quota(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"quota must be locale:count, got {value!r}")
    locale, raw_count = value.split(":", 1)
    locale = locale.strip()
    if locale not in LOCALE_TO_SOURCE:
        raise argparse.ArgumentTypeError(f"unsupported locale {locale!r}; expected one of {sorted(LOCALE_TO_SOURCE)}")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"quota count must be an integer, got {value!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError(f"quota count must be non-negative, got {value!r}")
    return locale, count


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


def _iter_shards(
    root: Path,
    locales: Iterable[str],
    *,
    max_shards_per_locale: int = 0,
) -> list[tuple[str, Path]]:
    shards: list[tuple[str, Path]] = []
    for locale in locales:
        locale_dir = root / locale
        locale_shards = sorted(locale_dir.glob("*.tar"))
        if max_shards_per_locale > 0:
            locale_shards = locale_shards[:max_shards_per_locale]
        shards.extend((locale, path) for path in locale_shards)
    return shards


def _safe_sample_id(locale: str, shard_path: Path, key: str) -> str:
    clean_locale = locale.replace("-", "_")
    return f"cv22_{clean_locale}_{shard_path.stem}_{key}"


def _audio_num_frames(audio_bytes: bytes) -> int:
    import soundfile as sf

    info = sf.info(io.BytesIO(audio_bytes))
    if info.samplerate <= 0:
        raise ValueError("audio has no valid sample rate")
    duration = float(info.frames) / float(info.samplerate)
    if duration <= 0.0:
        raise ValueError("audio has non-positive duration")
    return max(1, int(round(duration * 100.0)))


def _json_text_stats(metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    text = metadata.get("text")
    if text is None:
        return None, None
    text_value = str(text)
    return len(text_value), len(text_value.encode("utf-8"))


def _make_row(
    *,
    root: Path,
    locale: str,
    shard_path: Path,
    key: str,
    metadata: dict[str, Any],
    num_frames: int,
    audio_member: str,
    audio_offset: int,
    audio_size: int,
    json_member: str,
    json_offset: int,
    json_size: int,
    stage_name: str,
    seed: int,
    min_frames: int,
    max_frames: int,
    quota: int,
) -> dict[str, Any]:
    utt_id = _safe_sample_id(locale, shard_path, key)
    num_text_chars, text_bytes = _json_text_stats(metadata)
    relative_shard = shard_path.relative_to(root).as_posix()
    source = LOCALE_TO_SOURCE[locale]
    language = LOCALE_TO_LANGUAGE[locale]
    row: dict[str, Any] = {
        "shard_name": relative_shard,
        "tar_path": str(shard_path),
        "key": utt_id,
        "utt_id": utt_id,
        "split": "train",
        "num_frames": int(num_frames),
        "audio_member": audio_member,
        "audio_format": Path(audio_member).suffix.lower().lstrip("."),
        "json_member": json_member,
        "audio_offset": int(audio_offset),
        "audio_size": int(audio_size),
        "json_offset": int(json_offset),
        "json_size": int(json_size),
        "source_dataset": source,
        "language": language,
        "_stage178_stage": stage_name,
        "_stage178_curriculum": "cv22_audio_only_online_ctc_alignment",
        "_stage178_locale": locale,
        "_stage178_source": source,
        "_stage178_language": language,
        "_stage178_seed": int(seed),
        "_stage178_min_frames": int(min_frames),
        "_stage178_max_frames": int(max_frames),
        "_stage178_locale_quota": int(quota),
        "_stage178_ctc_online_only": True,
        "_stage178_uses_text_labels": False,
    }
    sentence_id = metadata.get("sentence_id")
    client_id = metadata.get("client_id")
    if sentence_id is not None:
        row["cv22_sentence_id"] = str(sentence_id)
    if client_id is not None:
        row["cv22_client_id"] = str(client_id)
    if metadata.get("up_votes") is not None:
        row["cv22_up_votes"] = metadata.get("up_votes")
    if metadata.get("down_votes") is not None:
        row["cv22_down_votes"] = metadata.get("down_votes")
    if num_text_chars is not None:
        row["num_text_chars"] = int(num_text_chars)
    if text_bytes is not None:
        row["text_bytes"] = int(text_bytes)
    return row


def _scan_shard(
    *,
    root: Path,
    locale: str,
    shard_path: Path,
    reservoirs: dict[str, Reservoir],
    stage_name: str,
    seed: int,
    min_frames: int,
    max_frames: int,
    max_pairs_per_shard: int,
    skip_decode_errors: bool,
) -> dict[str, Any]:
    pending: dict[str, dict[str, Any]] = {}
    seen_json = 0
    seen_audio = 0
    paired = 0
    selected_before = sum(len(item.rows) for item in reservoirs.values())
    skipped_too_short = 0
    skipped_too_long = 0
    decode_errors = 0
    missing_pairs = 0
    frame_values: list[float] = []

    with tarfile.open(shard_path, "r:") as archive, shard_path.open("rb") as raw:
        for member in archive:
            if not member.isfile():
                continue
            basename = Path(member.name).name
            if "." not in basename:
                continue
            key, suffix = basename.rsplit(".", 1)
            suffix = suffix.lower()
            if suffix not in {"json", "mp3", "wav", "flac"}:
                continue

            sample = pending.setdefault(key, {})
            if suffix == "json":
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                metadata = json.loads(extracted.read().decode("utf-8"))
                sample["metadata"] = metadata
                sample["json_member"] = member.name
                sample["json_offset"] = int(member.offset_data)
                sample["json_size"] = int(member.size)
                seen_json += 1
            else:
                sample["audio_member"] = member.name
                sample["audio_offset"] = int(member.offset_data)
                sample["audio_size"] = int(member.size)
                seen_audio += 1

            if "metadata" not in sample or "audio_member" not in sample:
                continue

            try:
                raw.seek(int(sample["audio_offset"]))
                audio_bytes = raw.read(int(sample["audio_size"]))
                if len(audio_bytes) != int(sample["audio_size"]):
                    raise EOFError(f"short audio read for {member.name}")
                num_frames = _audio_num_frames(audio_bytes)
            except Exception:
                decode_errors += 1
                if not skip_decode_errors:
                    raise
                pending.pop(key, None)
                continue

            paired += 1
            if num_frames < min_frames:
                skipped_too_short += 1
                pending.pop(key, None)
                continue
            if max_frames > 0 and num_frames > max_frames:
                skipped_too_long += 1
                pending.pop(key, None)
                continue

            frame_values.append(float(num_frames))
            row = _make_row(
                root=root,
                locale=locale,
                shard_path=shard_path,
                key=key,
                metadata=sample["metadata"],
                num_frames=num_frames,
                audio_member=str(sample["audio_member"]),
                audio_offset=int(sample["audio_offset"]),
                audio_size=int(sample["audio_size"]),
                json_member=str(sample["json_member"]),
                json_offset=int(sample["json_offset"]),
                json_size=int(sample["json_size"]),
                stage_name=stage_name,
                seed=seed,
                min_frames=min_frames,
                max_frames=max_frames,
                quota=reservoirs[locale].size,
            )
            reservoirs[locale].add(row, str(row["utt_id"]))
            pending.pop(key, None)
            if max_pairs_per_shard > 0 and paired >= max_pairs_per_shard:
                break

    for sample in pending.values():
        if "metadata" in sample or "audio_member" in sample:
            missing_pairs += 1
    selected_after = sum(len(item.rows) for item in reservoirs.values())
    return {
        "locale": locale,
        "shard": shard_path.relative_to(root).as_posix(),
        "seen_json": int(seen_json),
        "seen_audio": int(seen_audio),
        "paired": int(paired),
        "candidate_rows": int(len(frame_values)),
        "selected_added_net": int(selected_after - selected_before),
        "skipped_too_short": int(skipped_too_short),
        "skipped_too_long": int(skipped_too_long),
        "decode_errors": int(decode_errors),
        "missing_pairs": int(missing_pairs),
        "num_frames_distribution": _distribution(frame_values),
    }


def _build_rows(
    *,
    root: Path,
    quotas: dict[str, int],
    stage_name: str,
    seed: int,
    min_frames: int,
    max_frames: int,
    max_shards_per_locale: int,
    max_pairs_per_shard: int,
    skip_decode_errors: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reservoirs = {
        locale: Reservoir(size=count, rng=random.Random(seed + index * 65_537))
        for index, (locale, count) in enumerate(sorted(quotas.items()))
        if count > 0
    }
    shards = _iter_shards(root, reservoirs.keys(), max_shards_per_locale=max_shards_per_locale)
    if not shards:
        raise FileNotFoundError(f"No CV22 tar shards found under {root}")

    start_time = time.monotonic()
    shard_summaries: list[dict[str, Any]] = []
    locale_candidates: Counter[str] = Counter()
    locale_paired: Counter[str] = Counter()
    locale_skipped_short: Counter[str] = Counter()
    locale_skipped_long: Counter[str] = Counter()
    locale_decode_errors: Counter[str] = Counter()

    _log(f"scan start root={root} shards={len(shards)} target_rows={sum(quotas.values())}")
    for shard_index, (locale, shard_path) in enumerate(shards, start=1):
        shard_summary = _scan_shard(
            root=root,
            locale=locale,
            shard_path=shard_path,
            reservoirs=reservoirs,
            stage_name=stage_name,
            seed=seed,
            min_frames=min_frames,
            max_frames=max_frames,
            max_pairs_per_shard=max_pairs_per_shard,
            skip_decode_errors=skip_decode_errors,
        )
        shard_summaries.append(shard_summary)
        locale_candidates[locale] += int(shard_summary["candidate_rows"])
        locale_paired[locale] += int(shard_summary["paired"])
        locale_skipped_short[locale] += int(shard_summary["skipped_too_short"])
        locale_skipped_long[locale] += int(shard_summary["skipped_too_long"])
        locale_decode_errors[locale] += int(shard_summary["decode_errors"])
        elapsed = time.monotonic() - start_time
        selected = sum(len(item.rows) for item in reservoirs.values())
        _log(
            "scan progress "
            f"shards={shard_index}/{len(shards)} locale={locale} "
            f"candidates={sum(locale_candidates.values())} selected={selected} "
            f"elapsed={elapsed:.1f}s current={shard_summary['shard']}"
        )

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for locale, reservoir in sorted(reservoirs.items()):
        if len(reservoir.rows) < reservoir.size:
            _log(f"warning locale={locale} requested={reservoir.size} selected={len(reservoir.rows)}")
        reservoir.rows.sort(key=lambda row: (int(row["num_frames"]), str(row["utt_id"])))
        for sample_index, row in enumerate(reservoir.rows):
            row["_stage178_sample_index"] = int(sample_index)
            selected.append(row)
            selected_counts[locale] += 1

    selected.sort(
        key=lambda row: (
            int(row["num_frames"]),
            str(row["_stage178_locale"]),
            str(row["utt_id"]),
        )
    )
    frames = [float(row["num_frames"]) for row in selected]
    summary = {
        "version": 1,
        "stage_name": stage_name,
        "curriculum": "cv22_audio_only_online_ctc_alignment",
        "root": str(root),
        "uses_text_labels": False,
        "teacher": "FunASR-Nano-2512 online CTC logits",
        "seed": int(seed),
        "min_frames": int(min_frames),
        "max_frames": int(max_frames),
        "max_shards_per_locale": int(max_shards_per_locale),
        "max_pairs_per_shard": int(max_pairs_per_shard),
        "quotas": dict(sorted(quotas.items())),
        "num_shards": len(shards),
        "paired_rows": int(sum(locale_paired.values())),
        "candidate_rows": int(sum(locale_candidates.values())),
        "selected_rows": len(selected),
        "selected_unique_utt_ids": len({str(row["utt_id"]) for row in selected}),
        "locale_paired_counts": dict(sorted(locale_paired.items())),
        "locale_candidate_counts": dict(sorted(locale_candidates.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "skipped_too_short": int(sum(locale_skipped_short.values())),
        "skipped_too_long": int(sum(locale_skipped_long.values())),
        "decode_errors": int(sum(locale_decode_errors.values())),
        "locale_skipped_too_short": dict(sorted(locale_skipped_short.items())),
        "locale_skipped_too_long": dict(sorted(locale_skipped_long.items())),
        "locale_decode_errors": dict(sorted(locale_decode_errors.items())),
        "num_frames_distribution": _distribution(frames),
        "shards": shard_summaries,
    }
    return selected, summary


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


def _build_bucket_manifest(repo_root: Path, webdataset_root: Path, stage_dir: Path, bucket_width: int) -> None:
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
        str(webdataset_root),
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
        description="Build a CV22 audio-only pool for online FunASR-Nano CTC logits alignment."
    )
    parser.add_argument("--webdataset-root", default=DEFAULT_WEBDATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage-name", default=DEFAULT_STAGE_NAME)
    parser.add_argument("--quota", action="append", type=_parse_quota)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--max-shards-per-locale", type=int, default=0)
    parser.add_argument("--max-pairs-per-shard", type=int, default=0)
    parser.add_argument("--bucket-width", type=int, default=200)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skip-decode-errors", action="store_true")
    parser.add_argument("--skip-bucket-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    webdataset_root = Path(args.webdataset_root)
    output_root = Path(args.output_root)
    stage_name = str(args.stage_name)
    stage_dir = output_root / stage_name
    repo_root = Path(args.repo_root)
    quotas = dict(DEFAULT_QUOTAS)
    if args.quota:
        quotas.update(dict(args.quota))
    rows, summary = _build_rows(
        root=webdataset_root,
        quotas=quotas,
        stage_name=stage_name,
        seed=int(args.seed),
        min_frames=int(args.min_frames),
        max_frames=int(args.max_frames),
        max_shards_per_locale=int(args.max_shards_per_locale),
        max_pairs_per_shard=int(args.max_pairs_per_shard),
        skip_decode_errors=bool(args.skip_decode_errors),
    )
    summary.update({"output_root": str(output_root), "target_total_quota": int(sum(quotas.values()))})
    _write_stage(stage_dir, rows, summary)
    _log(f"length index written rows={len(rows)} dir={stage_dir}")
    if not args.skip_bucket_manifest:
        _build_bucket_manifest(repo_root, webdataset_root, stage_dir, int(args.bucket_width))
    manifest = {
        "version": 1,
        "root": str(webdataset_root),
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
    _log(f"Stage178 CV22 alignment pool complete output_root={output_root}")


if __name__ == "__main__":
    main()
