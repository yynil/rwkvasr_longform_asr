from __future__ import annotations

import importlib
import json
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
audit = importlib.import_module("scripts.audit_stage211_usb_top_level_coverage")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "usb"
    root.mkdir()
    for name, (entry_type, _, _) in audit.ENTRY_REGISTRY.items():
        path = root / name
        if entry_type == "directory":
            path.mkdir()
        else:
            path.write_bytes(b"fixture")

    stage179_manifest = tmp_path / "stage179.json"
    stage179_manifest.write_text(
        json.dumps(
            {
                "total_unique_audio_rows": audit.STAGE179_ROWS,
                "total_unique_hours": audit.STAGE179_HOURS,
                "inputs": {
                    "giga_wenet": {
                        "accepted_counts_by_source": {
                            "wenetspeech": audit.STAGE179_WENET_ROWS
                        },
                        "accepted_hours_by_source": {
                            "wenetspeech": audit.STAGE179_WENET_HOURS
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stage179_receipt = tmp_path / "loaded.json"
    stage179_receipt.write_text(
        json.dumps(
            {
                "global_dedup_manifest_path": str(stage179_manifest),
                "global_dedup_manifest_sha256": audit._sha256(stage179_manifest),
            }
        ),
        encoding="utf-8",
    )

    wenet = root / "wenetdata"
    (wenet / "WenetSpeech.json").write_text(
        '{"dataset": "WenetSpeech", "audios": []}', encoding="utf-8"
    )
    (wenet / "webdataset_output").mkdir()
    converter = wenet / "convert_to_webdataseet.py"
    converter.write_text(
        "for segment in audio_data.get('segments', []):\n    pass\n",
        encoding="utf-8",
    )

    testnew = root / "testnew"
    readme = testnew / "README.md"
    readme.write_text(
        "task_categories:\n- text-to-speech\n"
        "| **Total audio files** | 556,667 |\n"
        "| **Total duration** | 1,024.71 hours |\n",
        encoding="utf-8",
    )
    (testnew / "tts_dataset_combined.parquet").write_bytes(b"parquet")
    for name in audit.EXPECTED_SYNTHETIC_PAIRS:
        synthetic = root / name
        (synthetic / "sample.wav").write_bytes(b"wav")
        (synthetic / "sample.json").write_text("{}", encoding="ascii")
    for name in audit.EXPECTED_MINIMAX_TARS:
        (root / "minimax_tars" / name).write_bytes(b"tar")

    new_video = root / "new_video.tar"
    new_video.unlink()
    source = tmp_path / "source.mp3"
    source.write_bytes(b"audio")
    with tarfile.open(new_video, "w") as archive:
        archive.add(source, arcname="new_author/example.mp3")
    return root, stage179_receipt, readme, converter


def test_usb_coverage_classifies_every_entry_and_stays_incomplete(tmp_path: Path) -> None:
    root, stage179_receipt, readme, converter = _fixture(tmp_path)
    receipt = audit.build_receipt(
        root=root,
        stage179_receipt=stage179_receipt,
        testnew_readme=readme,
        wenet_converter=converter,
        strict_production=False,
    )

    assert receipt["classification_complete"] is True
    assert receipt["training_coverage_complete"] is False
    assert receipt["top_level_entry_count"] == len(audit.ENTRY_REGISTRY)
    classifications = {
        record["name"]: record["classification"]
        for record in receipt["top_level_entries"]
    }
    assert classifications["wenetdata"] == "same_corpus_repack"
    assert classifications["testnew"] == "synthetic_tts_augmentation"
    assert classifications["new_video.tar"] == "queued_for_admission"
    assert receipt["evidence"]["wenetspeech_repack"][
        "converter_subset_filter_present"
    ] is False
    assert receipt["evidence"]["archived_social"]["content_overlap_proven"] is False


def test_usb_coverage_fails_on_unknown_top_level_entry(tmp_path: Path) -> None:
    root, stage179_receipt, readme, converter = _fixture(tmp_path)
    (root / "unclassified_audio").mkdir()
    with pytest.raises(ValueError, match="unknown=.*unclassified_audio"):
        audit.build_receipt(
            root=root,
            stage179_receipt=stage179_receipt,
            testnew_readme=readme,
            wenet_converter=converter,
            strict_production=False,
        )


def test_usb_coverage_receipt_write_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert audit._write_immutable(output, {"value": 1}) == "created"
    assert audit._write_immutable(output, {"value": 1}) == "reused"
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        audit._write_immutable(output, {"value": 2})
