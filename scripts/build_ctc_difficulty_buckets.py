#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import tarfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, TextIO

from rwkvasr.data import build_text_tokenizer
from rwkvasr.data.text_normalization import normalize_asr_text


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _safe_name(value: str | None) -> str:
    value = (value or "unknown").strip().lower() or "unknown"
    value = re.sub(r"[^a-z0-9._+-]+", "_", value)
    return value.strip("_") or "unknown"


def _source_from_entry(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    source = metadata.get("source_dataset")
    if source:
        return _safe_name(str(source))
    shard_name = str(entry.get("shard_name") or "")
    if shard_name:
        return _safe_name(shard_name.split("_", 1)[0].split("-", 1)[0])
    return "unknown"


def _language_from_metadata(metadata: dict[str, Any]) -> str:
    return _safe_name(str(metadata.get("language") or "unknown"))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    pos = max(0.0, min(1.0, q)) * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p10": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _duration_score(duration_sec: float) -> int:
    if duration_sec <= 4.0:
        return 0
    if duration_sec <= 8.0:
        return 1
    if duration_sec <= 14.0:
        return 2
    return 3


def _token_score(num_tokens: int) -> int:
    if num_tokens <= 16:
        return 0
    if num_tokens <= 40:
        return 1
    if num_tokens <= 80:
        return 2
    return 3


def _frame_token_score(frames_per_token: float) -> int:
    if frames_per_token >= 18.0:
        return 0
    if frames_per_token >= 10.0:
        return 1
    if frames_per_token >= 6.0:
        return 2
    return 3


def _speech_ratio_score(speech_ratio: float | None) -> int:
    if speech_ratio is None:
        return 1
    if speech_ratio >= 0.80:
        return 0
    if speech_ratio >= 0.60:
        return 1
    if speech_ratio >= 0.40:
        return 2
    return 3


def _source_score(source: str) -> int:
    if source in {"librispeech", "aishell3", "aishell-3"}:
        return 0
    if source.startswith("commonvoice"):
        return 1
    return 1


def _difficulty_tier(score: int) -> str:
    if score <= 2:
        return "very_easy"
    if score <= 4:
        return "easy"
    if score <= 7:
        return "medium"
    if score <= 10:
        return "hard"
    return "extreme"


def _score_sample(
    *,
    source: str,
    num_frames: int,
    num_tokens: int,
    duration_sec: float,
    speech_ratio: float | None,
) -> tuple[str, int, dict[str, int], float]:
    frames_per_token = float(num_frames) / max(1, int(num_tokens))
    components = {
        "duration": _duration_score(duration_sec),
        "tokens": _token_score(num_tokens),
        "frames_per_token": _frame_token_score(frames_per_token),
        "speech_ratio": _speech_ratio_score(speech_ratio),
        "source": _source_score(source),
    }
    score = int(sum(components.values()))
    return _difficulty_tier(score), score, components, frames_per_token


class TarJsonReader:
    def __init__(self, shard_root: Path):
        self.shard_root = shard_root
        self._shard_name: str | None = None
        self._binary: BinaryIO | None = None
        self._archive: tarfile.TarFile | None = None

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        if self._binary is not None:
            self._binary.close()
            self._binary = None
        self._shard_name = None

    def _switch(self, shard_name: str) -> None:
        if shard_name == self._shard_name:
            return
        self.close()
        shard_path = self.shard_root / shard_name
        self._binary = shard_path.open("rb")
        self._shard_name = shard_name

    def read_json(self, entry: dict[str, Any]) -> dict[str, Any]:
        shard_name = str(entry["shard_name"])
        self._switch(shard_name)
        assert self._binary is not None
        offset = entry.get("json_offset")
        size = entry.get("json_size")
        if offset is not None and size is not None:
            self._binary.seek(int(offset))
            payload = self._binary.read(int(size))
            if len(payload) != int(size):
                raise EOFError(f"Short JSON read for {shard_name}:{entry.get('json_member')}")
            return json.loads(payload.decode("utf-8"))

        if self._archive is None:
            assert self._shard_name is not None
            self._archive = tarfile.open(self.shard_root / self._shard_name, "r")
        extracted = self._archive.extractfile(str(entry["json_member"]))
        if extracted is None:
            raise FileNotFoundError(f"Missing JSON member {shard_name}:{entry.get('json_member')}")
        return json.loads(extracted.read().decode("utf-8"))


@dataclass
class PartWriter:
    output_dir: Path
    entries_per_part: int
    count: int = 0
    part_index: int = 0
    current_count: int = 0
    handle: TextIO | None = None
    parts: list[dict[str, Any]] | None = None
    first_shard: str | None = None
    last_shard: str | None = None

    def __post_init__(self) -> None:
        self.parts = []

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.close()
        assert self.parts is not None
        self.parts.append(
            {
                "path": str(self._current_path.relative_to(self.output_dir.parent.parent.parent)),
                "num_samples": int(self.current_count),
                "first_shard": self.first_shard,
                "last_shard": self.last_shard,
            }
        )
        self.handle = None
        self.current_count = 0
        self.first_shard = None
        self.last_shard = None

    def _open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current_path = self.output_dir / f"part_{self.part_index:06d}.jsonl"
        self.part_index += 1
        self.handle = self._current_path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if self.handle is None or self.current_count >= self.entries_per_part:
            self.close()
            self._open()
        assert self.handle is not None
        shard_name = str(row.get("shard_name") or "")
        if self.first_shard is None:
            self.first_shard = shard_name
        self.last_shard = shard_name
        self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1
        self.current_count += 1


def _iter_length_entries(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CTC curriculum difficulty buckets from a WebDataset length index.")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--length-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-type", default="sensevoice_tiktoken")
    parser.add_argument("--tokenizer-model-path", default="assets/fun-asr-nano-2512/multilingual.tiktoken")
    parser.add_argument("--text-normalization", default="ctc")
    parser.add_argument("--entries-per-part", type=int, default=100000)
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shard_root = Path(args.shard_root)
    length_index_path = Path(args.length_index_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_text_tokenizer(
        args.tokenizer_type,
        model_path=args.tokenizer_model_path,
    )
    reader = TarJsonReader(shard_root)
    annotated_path = output_dir / "difficulty_annotated.jsonl"
    summary_path = output_dir / "difficulty_summary.json"

    counts: Counter[tuple[str, str, str, str]] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    component_counts: dict[str, Counter[int]] = defaultdict(Counter)
    distributions: dict[str, list[float]] = defaultdict(list)
    by_tier_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    writers: dict[tuple[str, str], PartWriter] = {}
    source_tier_counts: Counter[tuple[str, str]] = Counter()
    language_tier_counts: Counter[tuple[str, str]] = Counter()

    started = time.monotonic()
    processed = 0
    skipped = 0
    with annotated_path.open("w", encoding="utf-8") as annotated:
        try:
            for entry in _iter_length_entries(length_index_path):
                if args.max_samples and processed >= int(args.max_samples):
                    break
                try:
                    metadata = reader.read_json(entry)
                    text = metadata.get("text")
                    if text is None:
                        skipped += 1
                        continue
                    language = _language_from_metadata(metadata)
                    source = _source_from_entry(entry, metadata)
                    normalized_text = normalize_asr_text(
                        str(text),
                        language=None if language == "unknown" else language,
                        mode=str(args.text_normalization),
                    )
                    token_ids = tokenizer.encode(normalized_text)
                    num_tokens = len(token_ids)
                    num_frames = int(entry["num_frames"])
                    duration_sec = float(metadata.get("duration") or (num_frames / 100.0))
                    speech_duration = metadata.get("speech_duration")
                    speech_ratio = None
                    if speech_duration is not None and duration_sec > 0:
                        speech_ratio = max(0.0, min(1.0, float(speech_duration) / duration_sec))
                    tier, score, components, frames_per_token = _score_sample(
                        source=source,
                        num_frames=num_frames,
                        num_tokens=num_tokens,
                        duration_sec=duration_sec,
                        speech_ratio=speech_ratio,
                    )
                    row = {
                        **entry,
                        "source_dataset": source,
                        "language": language,
                        "duration_sec": round(duration_sec, 4),
                        "speech_duration_sec": (
                            round(float(speech_duration), 4) if speech_duration is not None else None
                        ),
                        "speech_ratio": round(float(speech_ratio), 6) if speech_ratio is not None else None,
                        "normalized_text_chars": len(normalized_text),
                        "ctc_num_tokens": num_tokens,
                        "frames_per_token": round(frames_per_token, 6),
                        "difficulty_tier": tier,
                        "difficulty_score": score,
                        "difficulty_components": components,
                    }
                    annotated.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

                    split = str(row["split"])
                    counts[(split, source, language, tier)] += 1
                    split_counts[split] += 1
                    source_counts[source] += 1
                    language_counts[language] += 1
                    tier_counts[tier] += 1
                    source_tier_counts[(source, tier)] += 1
                    language_tier_counts[(language, tier)] += 1
                    for component_name, component_value in components.items():
                        component_counts[component_name][int(component_value)] += 1
                    for metric_name, metric_value in (
                        ("duration_sec", duration_sec),
                        ("num_frames", float(num_frames)),
                        ("ctc_num_tokens", float(num_tokens)),
                        ("frames_per_token", frames_per_token),
                        ("difficulty_score", float(score)),
                    ):
                        distributions[metric_name].append(float(metric_value))
                        by_tier_values[tier][metric_name].append(float(metric_value))
                    if speech_ratio is not None:
                        distributions["speech_ratio"].append(float(speech_ratio))
                        by_tier_values[tier]["speech_ratio"].append(float(speech_ratio))

                    writer_key = (split, tier)
                    writer = writers.get(writer_key)
                    if writer is None:
                        writer = PartWriter(
                            output_dir=output_dir / "buckets" / split / tier,
                            entries_per_part=int(args.entries_per_part),
                        )
                        writers[writer_key] = writer
                    writer.write(row)
                    processed += 1
                    if args.progress_every > 0 and processed % int(args.progress_every) == 0:
                        elapsed = time.monotonic() - started
                        _log(
                            f"difficulty progress samples={processed} skipped={skipped} "
                            f"elapsed={elapsed:.1f}s current={entry.get('shard_name')}"
                        )
                except Exception as exc:
                    skipped += 1
                    _log(
                        f"skipped difficulty sample shard={entry.get('shard_name')} "
                        f"key={entry.get('key')}: {type(exc).__name__}: {exc}"
                    )
        finally:
            reader.close()
            for writer in writers.values():
                writer.close()

    bucket_manifest: dict[str, Any] = {
        "version": 1,
        "root": str(shard_root),
        "length_index_path": str(length_index_path),
        "annotated_path": str(annotated_path),
        "tokenizer_type": str(args.tokenizer_type),
        "tokenizer_model_path": str(Path(args.tokenizer_model_path).resolve()),
        "text_normalization": str(args.text_normalization),
        "entries_per_part": int(args.entries_per_part),
        "splits": {},
    }
    for (split, tier), writer in sorted(writers.items()):
        assert writer.parts is not None
        split_payload = bucket_manifest["splits"].setdefault(split, {"tiers": {}})
        split_payload["tiers"][tier] = {
            "num_samples": int(writer.count),
            "parts": writer.parts,
        }

    summary = {
        "version": 1,
        "root": str(shard_root),
        "length_index_path": str(length_index_path),
        "output_dir": str(output_dir),
        "annotated_path": str(annotated_path),
        "bucket_manifest_path": str(output_dir / "difficulty_manifest.json"),
        "num_samples": int(processed),
        "num_skipped": int(skipped),
        "counts": {
            "by_split": dict(sorted(split_counts.items())),
            "by_source": dict(sorted(source_counts.items())),
            "by_language": dict(sorted(language_counts.items())),
            "by_tier": dict(sorted(tier_counts.items())),
            "by_source_tier": {
                f"{source}/{tier}": count for (source, tier), count in sorted(source_tier_counts.items())
            },
            "by_language_tier": {
                f"{language}/{tier}": count for (language, tier), count in sorted(language_tier_counts.items())
            },
            "by_split_source_language_tier": {
                f"{split}/{source}/{language}/{tier}": count
                for (split, source, language, tier), count in sorted(counts.items())
            },
            "difficulty_components": {
                name: {str(k): v for k, v in sorted(counter.items())}
                for name, counter in sorted(component_counts.items())
            },
        },
        "distributions": {
            metric_name: _distribution(values)
            for metric_name, values in sorted(distributions.items())
        },
        "tier_distributions": {
            tier: {
                metric_name: _distribution(values)
                for metric_name, values in sorted(metrics.items())
            }
            for tier, metrics in sorted(by_tier_values.items())
        },
    }

    (output_dir / "difficulty_manifest.json").write_text(
        json.dumps(bucket_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(
        f"difficulty bucketing complete samples={processed} skipped={skipped} "
        f"summary={summary_path} annotated={annotated_path}"
    )


if __name__ == "__main__":
    main()
