from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf

SUPPORTED_AUDIO_SUFFIXES = {"wav", "mp3", "flac"}
TEXT_KEYS = ("text", "sentence", "normalized_text", "transcription", "transcript")
ID_KEYS = ("id", "sid", "utt_id", "utterance_id", "segment_id", "key")


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _safe_key(value: object, *, fallback: str) -> str:
    raw = str(value or "").strip() or fallback
    raw = re.sub(r"[^A-Za-z0-9_.=-]+", "_", raw).strip("._")
    if not raw:
        raw = fallback
    if len(raw) <= 180:
        return raw
    import hashlib

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{raw[:150]}_{digest}"


def _audio_suffix_from_path(path: object | None) -> str | None:
    if not path:
        return None
    suffix = Path(str(path)).suffix.lower().lstrip(".")
    if suffix in SUPPORTED_AUDIO_SUFFIXES:
        return suffix
    return None


def _audio_suffix_from_bytes(audio: bytes) -> str | None:
    if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
        return "wav"
    if audio.startswith(b"fLaC"):
        return "flac"
    if audio.startswith(b"ID3") or audio[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "mp3"
    return None


def _infer_audio_suffix(audio: bytes, source_path: object | None = None) -> str:
    suffix = _audio_suffix_from_path(source_path) or _audio_suffix_from_bytes(audio)
    if suffix is None:
        raise ValueError("unsupported or unknown audio encoding; expected wav/mp3/flac")
    return suffix


def _infer_duration(audio_bytes: bytes, sample_rate: int | None = None) -> tuple[float, int | None]:
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception:
        return 0.0, sample_rate
    rate = int(info.samplerate) if info.samplerate else sample_rate
    duration = float(info.frames) / float(rate) if rate else float(info.duration or 0.0)
    return duration, rate


def _row_get_first(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalise_language(value: object | None, default: str) -> str:
    raw = str(value or default).strip().lower()
    if raw in {"chinese", "mandarin", "zh-cn", "zh_cn", "cmn"}:
        return "zh"
    if raw in {"english", "en-us", "en_us"}:
        return "en"
    return raw or default


def _normalise_text_for_runtime(text: str, language: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode != "runtime":
        raise ValueError(f"Unsupported text normalization mode: {mode}")

    replacements = {
        "COMMA": ",",
        "PERIOD": ".",
        "FULLSTOP": ".",
        "DOT": ".",
        "QUESTION": "?",
        "QUESTIONMARK": "?",
        "EXCLAMATION": "!",
        "EXCLAMATIONMARK": "!",
        "EXCLAMATIONPOINT": "!",
        "COLON": ":",
        "SEMICOLON": ";",
        "DASH": "-",
        "HYPHEN": "-",
        "APOSTROPHE": "'",
    }

    def replace_tag(match: re.Match[str]) -> str:
        tag = re.sub(r"[\s_-]+", "", match.group(1)).upper()
        return replacements.get(tag, " ")

    normalized = re.sub(r"<([^<>]+)>", replace_tag, text)
    if language == "en" or language.startswith(("en-", "en_")):
        pieces: list[str] = []
        cursor = 0
        for match in re.finditer(r"<[^<>]+>", normalized):
            pieces.append(normalized[cursor : match.start()].lower())
            pieces.append(match.group(0))
            cursor = match.end()
        pieces.append(normalized[cursor:].lower())
        normalized = "".join(pieces)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"\s+([,.?!:;])", r"\1", normalized)
    normalized = re.sub(r"([,.?!:;])(?=\S)", r"\1 ", normalized)
    return normalized.strip()


def _tarinfo(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    return info


@dataclass
class ConvertStats:
    input_shards: int = 0
    converted_shards: int = 0
    samples: int = 0
    skipped_samples: int = 0
    skipped_input_shards: int = 0
    missing_audio: int = 0
    missing_text: int = 0
    unsupported_audio: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_shards": self.input_shards,
            "converted_shards": self.converted_shards,
            "samples": self.samples,
            "skipped_samples": self.skipped_samples,
            "skipped_input_shards": self.skipped_input_shards,
            "missing_audio": self.missing_audio,
            "missing_text": self.missing_text,
            "unsupported_audio": self.unsupported_audio,
            "elapsed_sec": round(time.monotonic() - self.started_at, 3),
        }


class WebDatasetTarWriter:
    def __init__(
        self,
        output_root: str | Path,
        *,
        shard_prefix: str,
        samples_per_shard: int,
        overwrite: bool,
    ) -> None:
        self.output_root = Path(output_root)
        self.shard_prefix = shard_prefix
        self.samples_per_shard = int(samples_per_shard)
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.samples_per_shard <= 0:
            raise ValueError("--samples-per-shard must be positive.")
        existing = sorted(self.output_root.glob(f"{self.shard_prefix}-*.tar"))
        if existing and not overwrite:
            raise FileExistsError(
                f"{self.output_root} already contains {self.shard_prefix}-*.tar; "
                "pass --overwrite to replace them."
            )
        if overwrite:
            for path in existing:
                path.unlink()
        self._archive: tarfile.TarFile | None = None
        self._shard_index = 0
        self._samples_in_shard = 0
        self._written_shards = 0

    @property
    def written_shards(self) -> int:
        return self._written_shards

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    def _open_next_shard(self) -> None:
        self.close()
        shard_path = self.output_root / f"{self.shard_prefix}-{self._shard_index:06d}.tar"
        self._archive = tarfile.open(shard_path, "w")
        self._shard_index += 1
        self._samples_in_shard = 0
        self._written_shards += 1

    def add_sample(
        self,
        *,
        key: str,
        audio_suffix: str,
        audio_bytes: bytes,
        metadata: dict[str, Any],
    ) -> None:
        if self._archive is None or self._samples_in_shard >= self.samples_per_shard:
            self._open_next_shard()
        assert self._archive is not None
        json_bytes = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._archive.addfile(_tarinfo(f"{key}.{audio_suffix}", audio_bytes), io.BytesIO(audio_bytes))
        self._archive.addfile(_tarinfo(f"{key}.json", json_bytes), io.BytesIO(json_bytes))
        self._samples_in_shard += 1


def _load_audio_from_parquet_row(
    row: dict[str, Any],
    *,
    input_root: Path,
    parquet_path: Path,
) -> tuple[bytes | None, str | None, int | None]:
    audio = row.get("audio")
    source_path: str | None = None
    sample_rate: int | None = None
    audio_bytes: bytes | None = None

    if isinstance(audio, dict):
        raw_bytes = audio.get("bytes")
        if raw_bytes is not None:
            audio_bytes = bytes(raw_bytes)
        source_path = audio.get("path")
        if audio.get("sampling_rate") is not None:
            sample_rate = int(audio["sampling_rate"])
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        audio_bytes = bytes(audio)

    if audio_bytes is None:
        for key in ("audio_bytes", "wav", "bytes"):
            value = row.get(key)
            if isinstance(value, (bytes, bytearray, memoryview)):
                audio_bytes = bytes(value)
                break

    if source_path is None:
        source_path = str(row.get("audio_path") or row.get("path") or "") or None

    if sample_rate is None and row.get("sampling_rate") is not None:
        sample_rate = int(row["sampling_rate"])

    if audio_bytes is None and source_path:
        candidates = [
            Path(source_path),
            parquet_path.parent / source_path,
            input_root / source_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                audio_bytes = candidate.read_bytes()
                break

    return audio_bytes, source_path, sample_rate


def convert_gigaspeech_parquet(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Python GigaSpeech parquet fallback requires pyarrow. "
            "Run `uv sync --extra preprocess`, or use the default Rust converter."
        ) from exc

    input_root = Path(args.input_root)
    parquet_files = sorted(input_root.glob(args.input_glob))
    if args.max_input_shards:
        parquet_files = parquet_files[: args.max_input_shards]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matching {args.input_glob!r} under {input_root}")

    writer = WebDatasetTarWriter(
        args.output_root,
        shard_prefix=args.shard_prefix,
        samples_per_shard=args.samples_per_shard,
        overwrite=args.overwrite,
    )
    stats = ConvertStats(input_shards=len(parquet_files))
    try:
        for shard_idx, parquet_path in enumerate(parquet_files, 1):
            try:
                parquet = pq.ParquetFile(parquet_path)
            except Exception as exc:
                if args.skip_missing:
                    stats.skipped_input_shards += 1
                    _log(f"Skipping unreadable parquet shard {parquet_path}: {exc}")
                    continue
                raise

            stats.converted_shards += 1
            source_split = parquet_path.name.split("-", 1)[0]
            local_row_index = 0
            for batch in parquet.iter_batches(batch_size=args.parquet_batch_size):
                for row in batch.to_pylist():
                    local_row_index += 1
                    if args.max_samples and stats.samples >= args.max_samples:
                        break
                    text = _row_get_first(row, TEXT_KEYS)
                    if text is None:
                        stats.missing_text += 1
                        stats.skipped_samples += 1
                        continue
                    audio_bytes, audio_path, sample_rate = _load_audio_from_parquet_row(
                        row,
                        input_root=input_root,
                        parquet_path=parquet_path,
                    )
                    if not audio_bytes:
                        stats.missing_audio += 1
                        stats.skipped_samples += 1
                        continue
                    try:
                        audio_suffix = _infer_audio_suffix(audio_bytes, audio_path)
                    except ValueError:
                        stats.unsupported_audio += 1
                        stats.skipped_samples += 1
                        continue

                    duration = row.get("duration")
                    if duration is None and row.get("begin_time") is not None and row.get("end_time") is not None:
                        duration = float(row["end_time"]) - float(row["begin_time"])
                    if duration is None or float(duration) <= 0.0:
                        duration, sample_rate = _infer_duration(audio_bytes, sample_rate)
                    if duration is None or float(duration) <= 0.0:
                        stats.skipped_samples += 1
                        continue

                    raw_id = _row_get_first(row, ID_KEYS)
                    fallback_id = f"{parquet_path.stem}_{local_row_index:08d}"
                    sample_id = _safe_key(raw_id, fallback=fallback_id)
                    key = _safe_key(f"gigaspeech_{sample_id}", fallback=f"gigaspeech_{fallback_id}")
                    language = _normalise_language(row.get("language"), args.language)
                    normalized_text = _normalise_text_for_runtime(str(text), language, args.text_normalization)
                    metadata = {
                        "id": key,
                        "sid": key,
                        "text": normalized_text,
                        "language": language,
                        "duration": float(duration),
                        "sample_rate": int(sample_rate) if sample_rate else None,
                        "source": "gigaspeech",
                        "source_split": source_split,
                        "source_shard": parquet_path.name,
                        "source_id": str(raw_id or fallback_id),
                        "text_normalization": args.text_normalization,
                    }
                    if normalized_text != str(text):
                        metadata["source_text"] = str(text)
                    if audio_path:
                        metadata["source_audio_path"] = str(audio_path)
                    writer.add_sample(
                        key=key,
                        audio_suffix=audio_suffix,
                        audio_bytes=audio_bytes,
                        metadata={k: v for k, v in metadata.items() if v is not None},
                    )
                    stats.samples += 1
                    if args.progress_every and stats.samples % args.progress_every == 0:
                        elapsed = time.monotonic() - stats.started_at
                        _log(
                            f"GigaSpeech convert progress: input={shard_idx}/{len(parquet_files)} "
                            f"samples={stats.samples} skipped={stats.skipped_samples} elapsed={elapsed:.1f}s"
                        )
                if args.max_samples and stats.samples >= args.max_samples:
                    break

            elapsed = time.monotonic() - stats.started_at
            _log(
                f"GigaSpeech shard done: {shard_idx}/{len(parquet_files)} "
                f"current={parquet_path.name} samples={stats.samples} elapsed={elapsed:.1f}s"
            )
            if args.max_samples and stats.samples >= args.max_samples:
                break
    finally:
        writer.close()

    result = stats.as_dict()
    result["output_root"] = str(args.output_root)
    result["output_shards"] = writer.written_shards
    result["text_normalization"] = args.text_normalization
    _write_summary(args.output_root, "gigaspeech_parquet_conversion_summary.json", result)
    return result


def _wenet_cut_metadata(
    raw: dict[str, Any], *, default_language: str, text_normalization: str
) -> dict[str, Any] | None:
    cut_id = raw.get("id")
    if not cut_id:
        return None
    supervisions = raw.get("supervisions") or []
    supervision = supervisions[0] if supervisions else {}
    text = supervision.get("text") or raw.get("text")
    if not text:
        return None
    recording = raw.get("recording") or {}
    duration = raw.get("duration") or supervision.get("duration") or recording.get("duration")
    sample_rate = recording.get("sampling_rate")
    source_audio_path = None
    sources = recording.get("sources") or []
    if sources:
        source_audio_path = sources[0].get("source")
    if duration is None or float(duration) <= 0.0:
        return None
    sample_id = _safe_key(cut_id, fallback="wenetspeech_cut")
    key = _safe_key(f"wenetspeech_{sample_id}", fallback=f"wenetspeech_{sample_id}")
    language = _normalise_language(supervision.get("language"), default_language)
    normalized_text = _normalise_text_for_runtime(str(text), language, text_normalization)
    metadata = {
        "id": key,
        "sid": key,
        "text": normalized_text,
        "language": language,
        "duration": float(duration),
        "sample_rate": int(sample_rate) if sample_rate else None,
        "source": "wenetspeech",
        "source_id": str(cut_id),
        "source_recording_id": str(supervision.get("recording_id") or recording.get("id") or ""),
        "source_audio_path": source_audio_path,
        "num_frames": int(raw.get("features", {}).get("num_frames") or round(float(duration) * 100.0)),
        "text_normalization": text_normalization,
    }
    if normalized_text != str(text):
        metadata["source_text"] = str(text)
    return metadata


def _load_wenet_metadata(
    jsonl_path: Path, *, language: str, text_normalization: str
) -> dict[str, dict[str, Any]]:
    metadata_by_cut: dict[str, dict[str, Any]] = {}
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            metadata = _wenet_cut_metadata(
                raw, default_language=language, text_normalization=text_normalization
            )
            if metadata is None:
                continue
            source_id = str(metadata["source_id"])
            metadata["source_shard"] = jsonl_path.name
            metadata_by_cut[source_id] = metadata
    return metadata_by_cut


def _paired_wenet_tar(jsonl_path: Path) -> Path:
    return jsonl_path.with_suffix("").with_suffix(".tar.gz")


def convert_wenetspeech_lhotse(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root)
    jsonl_files = sorted(input_root.glob(args.input_glob))
    if args.max_input_shards:
        jsonl_files = jsonl_files[: args.max_input_shards]
    if not jsonl_files:
        raise FileNotFoundError(f"No Lhotse cut jsonl files matching {args.input_glob!r} under {input_root}")

    writer = WebDatasetTarWriter(
        args.output_root,
        shard_prefix=args.shard_prefix,
        samples_per_shard=args.samples_per_shard,
        overwrite=args.overwrite,
    )
    stats = ConvertStats(input_shards=len(jsonl_files))
    try:
        for shard_idx, jsonl_path in enumerate(jsonl_files, 1):
            tar_path = _paired_wenet_tar(jsonl_path)
            if not tar_path.exists():
                if args.skip_missing:
                    stats.skipped_input_shards += 1
                    _log(f"Skipping unpaired WenetSpeech shard {jsonl_path.name}; missing {tar_path.name}")
                    continue
                raise FileNotFoundError(f"Missing paired tar file for {jsonl_path}: {tar_path}")

            metadata_by_cut = _load_wenet_metadata(
                jsonl_path, language=args.language, text_normalization=args.text_normalization
            )
            stats.converted_shards += 1
            matched = 0
            try:
                archive = tarfile.open(tar_path, "r:gz")
            except Exception as exc:
                if args.skip_missing:
                    stats.skipped_input_shards += 1
                    _log(f"Skipping unreadable tar shard {tar_path}: {exc}")
                    continue
                raise
            with archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    suffix = Path(member.name).suffix.lower().lstrip(".")
                    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
                        continue
                    source_id = Path(member.name).stem
                    metadata = metadata_by_cut.get(source_id)
                    if metadata is None:
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        stats.missing_audio += 1
                        stats.skipped_samples += 1
                        continue
                    audio_bytes = extracted.read()
                    key = str(metadata["id"])
                    metadata = dict(metadata)
                    metadata["source_tar"] = tar_path.name
                    metadata["source_audio_member"] = member.name
                    writer.add_sample(
                        key=key,
                        audio_suffix=suffix,
                        audio_bytes=audio_bytes,
                        metadata=metadata,
                    )
                    matched += 1
                    stats.samples += 1
                    if args.max_samples and stats.samples >= args.max_samples:
                        break
                    if args.progress_every and stats.samples % args.progress_every == 0:
                        elapsed = time.monotonic() - stats.started_at
                        _log(
                            f"WenetSpeech convert progress: input={shard_idx}/{len(jsonl_files)} "
                            f"samples={stats.samples} skipped={stats.skipped_samples} elapsed={elapsed:.1f}s"
                        )
            if not (args.max_samples and stats.samples >= args.max_samples):
                missing_in_tar = max(0, len(metadata_by_cut) - matched)
                stats.missing_audio += missing_in_tar
                stats.skipped_samples += missing_in_tar
            elapsed = time.monotonic() - stats.started_at
            _log(
                f"WenetSpeech shard done: {shard_idx}/{len(jsonl_files)} "
                f"current={jsonl_path.name} matched={matched}/{len(metadata_by_cut)} "
                f"samples={stats.samples} elapsed={elapsed:.1f}s"
            )
            if args.max_samples and stats.samples >= args.max_samples:
                break
    finally:
        writer.close()

    result = stats.as_dict()
    result["output_root"] = str(args.output_root)
    result["output_shards"] = writer.written_shards
    result["text_normalization"] = args.text_normalization
    _write_summary(args.output_root, "wenetspeech_lhotse_conversion_summary.json", result)
    return result


def _write_summary(output_root: str | Path, filename: str, data: dict[str, Any]) -> None:
    path = Path(output_root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--shard-prefix", required=True)
    parser.add_argument("--samples-per-shard", type=int, default=5000)
    parser.add_argument("--max-input-shards", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--text-normalization", choices=("none", "runtime"), default="none")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ASR corpora into the project's canonical WebDataset tar format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    giga = subparsers.add_parser("gigaspeech-parquet", help="Convert HuggingFace GigaSpeech parquet shards.")
    _add_common_args(giga)
    giga.add_argument("--input-glob", default="*.parquet")
    giga.add_argument("--parquet-batch-size", type=int, default=128)

    wenet = subparsers.add_parser("wenetspeech-lhotse", help="Convert WenetSpeech Lhotse cuts jsonl/tar shards.")
    _add_common_args(wenet)
    wenet.add_argument("--input-glob", default="cuts_*.jsonl.gz")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "gigaspeech-parquet":
        result = convert_gigaspeech_parquet(args)
    elif args.command == "wenetspeech-lhotse":
        result = convert_wenetspeech_lhotse(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
