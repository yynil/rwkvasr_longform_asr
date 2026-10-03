from __future__ import annotations

import io
import importlib
import json
import sys
import tarfile
import wave
from pathlib import Path

import pytest

from rwkvasr.data.webdataset_lengths import parse_webdataset_length_entry


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
repair_zero_audio_row = importlib.import_module(
    "scripts.repair_stage211_zero_audio"
).repair_zero_audio_row


def test_length_entry_rejects_explicit_zero_audio_size() -> None:
    with pytest.raises(ValueError, match="non-positive audio_size"):
        parse_webdataset_length_entry(
            {
                "shard_name": "source.tar",
                "key": "zero",
                "utt_id": "zero",
                "split": "train",
                "num_frames": 10,
                "audio_member": "zero.wav",
                "json_member": "zero.json",
                "audio_size": 0,
            }
        )


def _write_wav(path: Path, *, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * frames)


def _add_tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_repair_stage211_zero_audio_row_is_audited_and_idempotent(tmp_path: Path) -> None:
    sample_key = "sample-zero"
    audio_member = f"{sample_key}.wav"
    json_member = f"{sample_key}.json"
    metadata = json.dumps({"id": sample_key, "text": "HELLO WORLD"}).encode()
    source_tar = tmp_path / "source.tar"
    with tarfile.open(source_tar, "w") as archive:
        _add_tar_member(archive, audio_member, b"")
        _add_tar_member(archive, json_member, metadata)
    with tarfile.open(source_tar, "r") as archive:
        audio_info = archive.getmember(audio_member)
        json_info = archive.getmember(json_member)

    row = {
        "shard_name": str(source_tar),
        "tar_path": str(source_tar),
        "key": sample_key,
        "utt_id": sample_key,
        "split": "train",
        "num_frames": 10,
        "audio_member": audio_member,
        "audio_format": "wav",
        "json_member": json_member,
        "audio_offset": audio_info.offset_data,
        "audio_size": 0,
        "json_offset": json_info.offset_data,
        "json_size": json_info.size,
        "_stage179_sample_index": 7,
    }
    part = tmp_path / "part.jsonl"
    part.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    waveform = tmp_path / "crop.wav"
    episode = tmp_path / "episode.wav"
    _write_wav(waveform)
    _write_wav(episode, frames=3200)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    import hashlib

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictions = tmp_path / "nano.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "utt_id": sample_key,
                "pred_text": "hello world",
                "ref_text": "HELLO WORLD <PERIOD>",
                "debug": {"logit_length": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "nano.report.json"
    report.write_text(
        json.dumps(
            {
                "sample_count": 1,
                "metrics": {"avg_wer": 0.0},
                "model_checkpoint_path": str(checkpoint),
                "model_checkpoint_sha256": checkpoint_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    repair_tar = tmp_path / "repair.tar"
    receipt_path = tmp_path / "receipt.json"

    receipt = repair_zero_audio_row(
        bucket_manifest_path=manifest,
        bucket_part_path=part,
        sample_key=sample_key,
        waveform_path=waveform,
        episode_path=episode,
        repair_shard_path=repair_tar,
        receipt_path=receipt_path,
        nano_report_path=report,
        nano_predictions_path=predictions,
        language="en",
        episode_page_url="https://example.test/episode",
        episode_audio_url="https://example.test/episode.mp3",
        crop_start_sec=1.0,
        crop_duration_sec=0.1,
    )

    repaired = json.loads(part.read_text(encoding="utf-8"))
    assert repaired["key"] == sample_key
    assert repaired["num_frames"] == row["num_frames"]
    assert repaired["audio_size"] == waveform.stat().st_size
    assert repaired["_stage211_audio_repair_receipt"] == str(receipt_path.resolve())
    assert receipt["preserved_contract"]["row_count_delta"] == 0
    assert receipt["nano_verification"]["error_rate"] == 0.0
    assert repair_zero_audio_row(
        bucket_manifest_path=manifest,
        bucket_part_path=part,
        sample_key=sample_key,
        waveform_path=waveform,
        episode_path=episode,
        repair_shard_path=repair_tar,
        receipt_path=receipt_path,
        nano_report_path=report,
        nano_predictions_path=predictions,
        language="en",
        episode_page_url="https://example.test/episode",
        episode_audio_url="https://example.test/episode.mp3",
        crop_start_sec=1.0,
        crop_duration_sec=0.1,
    ) == receipt
