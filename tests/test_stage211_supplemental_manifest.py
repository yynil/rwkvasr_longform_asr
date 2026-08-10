from __future__ import annotations

import importlib
import json
import sys
import wave
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from rwkvasr.data import load_webdataset_bucket_manifest
from rwkvasr.data.webdataset_lengths import load_webdataset_length_entries


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
builder = importlib.import_module("scripts.build_stage211_supplemental_natural_manifest")


def _wav_bytes(sample_rate: int, *, seconds: float = 1.0) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * int(sample_rate * seconds))
    return buffer.getvalue()


def _write_people_parquet(path: Path, rows: list[tuple[str, int]]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "id": pa.array([identity for identity, _ in rows]),
            "audio": pa.array(
                [
                    {"bytes": _wav_bytes(16_000), "path": f"{identity}.wav"}
                    for identity, _ in rows
                ],
                type=audio_type,
            ),
            "duration_ms": pa.array([duration for _, duration in rows], type=pa.int32()),
            "text": pa.array(["unused"] * len(rows)),
        }
    )
    pq.write_table(table, path, row_group_size=1)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_build_supplemental_manifest_deduplicates_and_excludes_eval_paths(
    tmp_path: Path,
) -> None:
    peoples = tmp_path / "peoples"
    (peoples / "clean").mkdir(parents=True)
    (peoples / "dirty").mkdir(parents=True)
    _write_people_parquet(
        peoples / "clean/train-00000-of-00001.parquet",
        [("shared", 1000)],
    )
    _write_people_parquet(
        peoples / "dirty/train-00000-of-00001.parquet",
        [("shared", 1000), ("dirty-only", 1000)],
    )
    _write_people_parquet(
        peoples / "clean/test-00000-of-00001.parquet",
        [("must-not-train", 1000)],
    )

    llaso = tmp_path / "llaso"
    llaso.mkdir()
    _write_zip(
        llaso / "LLaSO-Align-audio.part1.zip",
        {"LJSpeech/wavs/ljs.wav": _wav_bytes(22_050)},
    )
    _write_zip(
        llaso / "LLaSO-Align-audio.part41.zip",
        {"VCTK-Corpus/wav48/vctk.wav": _wav_bytes(48_000)},
    )
    _write_zip(
        llaso / "LLaSO-Align-audio.part317.zip",
        {
            "mls_english_opus/train_data_16000/train.wav": _wav_bytes(16_000),
            "mls_english_opus/dev_data_16000/dev.wav": _wav_bytes(16_000),
        },
    )
    output = tmp_path / "supplemental"

    inventory = builder.build_manifest(
        peoples_root=peoples,
        llaso_root=llaso,
        output=output,
        bucket_width=80,
        entries_per_part=2,
        hash_archives=False,
        require_production_layout=False,
    )

    assert inventory["complete"] is True
    assert inventory["training_ready"] is False
    assert inventory["selected_rows"] == 5
    assert inventory["dedupe"]["duplicates_by_source"] == {"peoples_speech_dirty": 1}
    assert inventory["selected_counts_by_source"] == {
        "ljspeech": 1,
        "mls_english": 1,
        "peoples_speech_clean": 1,
        "peoples_speech_dirty": 1,
        "vctk": 1,
    }
    manifest = load_webdataset_bucket_manifest(
        output / "webdataset_buckets_audio_text/manifest.json"
    )
    assert sum(bucket.num_samples for bucket in manifest.splits["train"]) == 5
    rows = []
    for bucket in manifest.splits["train"]:
        for part in bucket.parts:
            rows.extend(
                load_webdataset_length_entries(
                    output / "webdataset_buckets_audio_text" / part.path,
                )
            )
    assert {row.storage_kind for row in rows} == {"parquet", "zip"}
    assert "must-not-train" not in json.dumps([row.raw for row in rows])
    assert all(row.split == "train" for row in rows)


def test_zip_inventory_refuses_truncated_archive(tmp_path: Path) -> None:
    path = tmp_path / "LLaSO-Align-audio.part1.zip"
    _write_zip(path, {"LJSpeech/wavs/ljs.wav": _wav_bytes(22_050)})
    path.write_bytes(path.read_bytes()[:-22])
    records: list[dict[str, object]] = []

    rows = list(
        builder._zip_rows(
            tmp_path,
            specs=(
                builder.ZipSourceSpec("ljspeech", (1,), "LJSpeech/wavs/", 22_050),
            ),
            hash_archives=False,
            archive_records=records,
        )
    )

    assert rows == []
    assert records[0]["status"] == "invalid"
