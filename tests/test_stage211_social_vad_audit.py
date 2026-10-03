from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
audit = importlib.import_module("scripts.audit_stage211_social_vad_sources")


def _roots(tmp_path: Path) -> dict[str, Path]:
    roots = {name: tmp_path / name for name in audit.DEFAULT_ROOTS}
    for root in roots.values():
        root.mkdir(parents=True)
    return roots


def _vad_root(tmp_path: Path) -> Path:
    root = tmp_path / "vad"
    root.mkdir()
    for filename in audit.EXPECTED_VAD_HASHES:
        (root / filename).write_bytes(filename.encode("ascii"))
    return root


def test_social_vad_inventory_replaces_matching_original_and_excludes_video(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    original = roots["videos"] / "speaker" / "same title.mp3"
    original.parent.mkdir()
    original.write_bytes(b"original")
    extracted = roots["videos"] / "speaker" / "other title_audio.wav"
    extracted.write_bytes(b"wav")
    (roots["videos"] / "speaker" / "other title_video.mp4").write_bytes(b"video")
    vocal = roots["clean_vocals"] / "htdemucs" / "same title" / "vocals.wav"
    vocal.parent.mkdir(parents=True)
    vocal.write_bytes(b"vocal")

    inventory = audit.build_inventory(
        roots=roots,
        vad_model_root=_vad_root(tmp_path),
        probe_audio=False,
        hash_media=False,
        require_production_layout=False,
    )

    assert inventory["layout_valid"] is True
    assert inventory["training_ready"] is False
    assert inventory["canonical_audio_files"] == 2
    assert inventory["excluded_video_files"] == 1
    assert inventory["sidecar_files"] == []
    assert inventory["vocal_replacements"] == [
        {
            "title": "same title",
            "replacement": str(vocal.resolve()),
            "replaced": str(original.resolve()),
        }
    ]
    selected = {record["path"]: record for record in inventory["source_records"]}
    assert str(original.resolve()) not in selected
    assert selected[str(vocal.resolve())]["replaces"] == str(original.resolve())
    assert selected[str(extracted.resolve())]["title"] == "other title"


def test_social_vad_inventory_aggregates_probe_and_exact_content_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    first = roots["videos_bilibili5"] / "first.mp3"
    second = roots["videos_bilibili6"] / "second.mp3"
    first.write_bytes(b"identical")
    second.write_bytes(b"identical")

    monkeypatch.setattr(
        audit,
        "_probe_audio",
        lambda path: {
            "frames": 32_000,
            "sample_rate": 16_000,
            "channels": 1,
            "duration_seconds": 2.0,
            "format": "MP3",
            "subtype": "MPEG_LAYER_III",
        },
    )
    inventory = audit.build_inventory(
        roots=roots,
        vad_model_root=_vad_root(tmp_path),
        probe_audio=True,
        hash_media=True,
        require_production_layout=False,
    )

    assert inventory["audio_probe"]["complete"] is True
    assert inventory["audio_probe"]["probed_files"] == 2
    assert inventory["audio_probe"]["hours"] == pytest.approx(4.0 / 3600.0)
    assert inventory["audio_probe"]["sample_rate_counts"] == {"16000": 2}
    assert inventory["media_content_hash_complete"] is True
    assert len(inventory["exact_content_duplicate_groups"]) == 1
    assert inventory["exact_content_duplicate_groups"][0]["paths"] == sorted(
        [str(first.resolve()), str(second.resolve())]
    )


def test_social_vad_inventory_write_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    assert audit._write_immutable(output, {"value": 1}) == "created"
    assert audit._write_immutable(output, {"value": 1}) == "reused"
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        audit._write_immutable(output, {"value": 2})
