#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


DATASET_INFO: dict[str, dict[str, str]] = {
    "aishell1_test": {"language": "zh", "condition": "clean"},
    "librispeech_test_clean": {"language": "en", "condition": "clean"},
    "librispeech_test_other": {"language": "en", "condition": "clean"},
    "commonvoice_en_test": {"language": "en", "condition": "non_clean"},
    "wenetspeech_test_net": {"language": "zh", "condition": "non_clean"},
}


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _extract_tar_once(tar_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{tar_path.name}.extracted"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(["tar", "-xf", str(tar_path), "-C", str(output_dir)])
    marker.write_text("ok\n", encoding="utf-8")


def _extract_targz_once(tar_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{tar_path.name}.extracted"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(["tar", "-xzf", str(tar_path), "-C", str(output_dir)])
    marker.write_text("ok\n", encoding="utf-8")


def _librispeech_records(root: Path, subset: str) -> Iterable[dict[str, str]]:
    subset_root = root / "LibriSpeech" / subset
    if not subset_root.exists():
        raise FileNotFoundError(f"LibriSpeech subset not found: {subset_root}")
    for transcript_path in sorted(subset_root.rglob("*.trans.txt")):
        for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            utt_id, text = raw_line.split(" ", 1)
            audio_path = transcript_path.parent / f"{utt_id}.flac"
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing LibriSpeech audio for {utt_id}: {audio_path}")
            yield {
                "utt_id": utt_id,
                "audio_filepath": str(audio_path),
                "text": text,
                "dataset": f"librispeech_{subset.replace('-', '_')}",
            }


def _aishell_hf_records(parquet_dir: Path, audio_output_dir: Path) -> Iterable[dict[str, str]]:
    import pyarrow.parquet as pq

    parquet_paths = sorted(parquet_dir.glob("test-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No AISHELL-1 test parquet files under {parquet_dir}")
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                utt_id = str(row["name"])
                text = " ".join(str(row["text"]).split())
                audio = row.get("audio") or {}
                audio_bytes = audio.get("bytes")
                if not isinstance(audio_bytes, (bytes, bytearray)):
                    raise ValueError(f"AISHELL-1 row {utt_id} has no audio bytes.")
                audio_path = audio_output_dir / f"{utt_id}.wav"
                if not audio_path.exists() or audio_path.stat().st_size == 0:
                    audio_path.write_bytes(bytes(audio_bytes))
                yield {
                    "utt_id": utt_id,
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "dataset": "aishell1_test",
                }


def _common_voice_records(root: Path, locale: str, split: str, audio_output_dir: Path) -> Iterable[dict[str, str]]:
    transcript_path = root / "transcript" / locale / f"{split}.tsv"
    tar_dir = root / "audio" / locale / split
    if not transcript_path.exists():
        raise FileNotFoundError(f"Common Voice TSV not found: {transcript_path}")
    for tar_path in sorted(tar_dir.glob("*.tar")):
        _extract_tar_once(tar_path, audio_output_dir)
    audio_by_name = {path.name: path for path in audio_output_dir.rglob("*.mp3")}
    with transcript_path.open("r", encoding="utf-8", newline="") as handle:
        # Common Voice TSV stores transcript quote marks literally and does not
        # escape them as CSV fields. CSV quote parsing can merge physical rows.
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            path_name = str(row["path"])
            audio_path = audio_by_name.get(path_name)
            if audio_path is None:
                raise FileNotFoundError(f"Common Voice audio {path_name} not found after extracting {tar_dir}")
            utt_id = Path(path_name).stem
            yield {
                "utt_id": utt_id,
                "audio_filepath": str(audio_path),
                "text": str(row["sentence"]),
                "dataset": f"commonvoice_{locale}_{split}",
            }


def _wenetspeech_records(data_dir: Path, split: str, audio_output_dir: Path) -> Iterable[dict[str, str]]:
    jsonl_paths = sorted(data_dir.glob(f"cuts_{split}.*.jsonl.gz"))
    tar_paths = sorted(data_dir.glob(f"cuts_{split}.*.tar.gz"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No WenetSpeech {split} jsonl.gz files under {data_dir}")
    if not tar_paths:
        raise FileNotFoundError(f"No WenetSpeech {split} audio tar.gz files under {data_dir}")
    for tar_path in tar_paths:
        _extract_targz_once(tar_path, audio_output_dir)
    for jsonl_path in jsonl_paths:
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                record = json.loads(raw)
                utt_id = str(record["id"])
                supervisions = record.get("supervisions") or []
                text = str(supervisions[0].get("text", "")) if supervisions else ""
                sources = ((record.get("recording") or {}).get("sources")) or []
                source = str(sources[0].get("source", "")) if sources else ""
                if source.startswith("data/"):
                    source = source[len("data/") :]
                audio_path = audio_output_dir / source
                if not audio_path.exists():
                    raise FileNotFoundError(f"WenetSpeech audio for {utt_id} not found: {audio_path}")
                yield {
                    "utt_id": utt_id,
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "dataset": f"wenetspeech_{split.lower()}",
                }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build public ASR eval manifests for RWKVASR benchmark runs.")
    parser.add_argument("--output-dir", default="artifacts/eval_benchmarks/manifests")
    parser.add_argument("--audio-cache-dir", default="artifacts/eval_benchmarks/audio")
    parser.add_argument(
        "--librispeech-root",
        default="/media/usbhd/training_data/asr/eval_benchmarks/librispeech",
    )
    parser.add_argument(
        "--aishell-hf-dir",
        default="/media/usbhd/training_data/asr/eval_benchmarks/aishell_hf",
    )
    parser.add_argument(
        "--commonvoice-root",
        default="/media/usbhd/common_voice_22/common_voice_22_0",
    )
    parser.add_argument(
        "--wenetspeech-data-dir",
        default="/media/usbhd/training_data/asr/wenet-e2e/wenetspeech/data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    audio_cache_dir = Path(args.audio_cache_dir).resolve()
    manifest_counts: dict[str, int] = {}

    jobs: list[tuple[str, Iterable[dict[str, str]]]] = [
        (
            "librispeech_test_clean",
            _librispeech_records(Path(args.librispeech_root), "test-clean"),
        ),
        (
            "librispeech_test_other",
            _librispeech_records(Path(args.librispeech_root), "test-other"),
        ),
        (
            "aishell1_test",
            _aishell_hf_records(Path(args.aishell_hf_dir), audio_cache_dir / "aishell1_test"),
        ),
        (
            "commonvoice_en_test",
            _common_voice_records(
                Path(args.commonvoice_root),
                "en",
                "test",
                audio_cache_dir / "commonvoice_en_test",
            ),
        ),
        (
            "wenetspeech_test_net",
            _wenetspeech_records(
                Path(args.wenetspeech_data_dir),
                "TEST_NET",
                audio_cache_dir / "wenetspeech_test_net",
            ),
        ),
    ]

    for name, records in jobs:
        count = _write_jsonl(output_dir / f"{name}.jsonl", records)
        manifest_counts[name] = count
        print(f"{name}: {count}")

    summary = {
        "manifests": {
            name: {
                **DATASET_INFO[name],
                "path": str(output_dir / f"{name}.jsonl"),
                "num_samples": count,
            }
            for name, count in manifest_counts.items()
        }
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
