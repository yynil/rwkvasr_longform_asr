from __future__ import annotations

import importlib
import json
import sys
import tarfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
audit = importlib.import_module("scripts.audit_stage211_archived_social_overlap")


def _source_inventory(tmp_path: Path, paths: list[Path]) -> Path:
    output = tmp_path / "source_inventory.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "stage211_social_vad_source_inventory",
                "complete": True,
                "training_ready": False,
                "admission_state": "source_inventory_only",
                "layout_valid": True,
                "media_content_hash_complete": True,
                "canonical_audio_files": len(paths),
                "audio_probe": {"complete": True, "errors": [], "hours": 1.0},
                "source_records": [
                    {
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": audit._sha256(path),
                    }
                    for path in paths
                ],
            }
        ),
        encoding="utf-8",
    )
    return output


def _coverage(tmp_path: Path) -> Path:
    output = tmp_path / "coverage.json"
    output.write_text(
        json.dumps(
            {
                "artifact": "usb_top_level_coverage",
                "classification_complete": True,
            }
        ),
        encoding="utf-8",
    )
    return output


def _archive(path: Path, members: dict[str, bytes]) -> None:
    source_root = path.parent / "sources"
    source_root.mkdir(exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for index, (name, payload) in enumerate(members.items()):
            source = source_root / f"source-{index}.mp3"
            source.write_bytes(payload)
            archive.add(source, arcname=name)


def test_archived_social_all_exact_duplicates_are_excluded(tmp_path: Path) -> None:
    social = tmp_path / "videos"
    (social / "author").mkdir(parents=True)
    first = social / "author/one.mp3"
    second = social / "author/two.mp3"
    first.write_bytes(b"first audio")
    second.write_bytes(b"second audio")
    archive = tmp_path / "new_video.tar"
    _archive(
        archive,
        {"author/one.mp3": first.read_bytes(), "author/two.mp3": second.read_bytes()},
    )

    receipt = audit.build_receipt(
        archive_path=archive,
        social_root=social,
        source_inventory_path=_source_inventory(tmp_path, [first, second]),
        usb_coverage_receipt_path=_coverage(tmp_path),
        strict_production=False,
    )

    assert receipt["archive_audio_members"] == 2
    assert receipt["exact_duplicate_members"] == 2
    assert receipt["unique_members"] == 0
    assert receipt["all_members_exact_existing_social_duplicates"] is True
    assert receipt["archive_excluded_from_training_as_duplicate"] is True
    assert all(
        record["existing_hash_source"] == "bound_social_source_inventory"
        for record in receipt["member_records"]
    )


def test_archived_social_unique_member_requires_vad(tmp_path: Path) -> None:
    social = tmp_path / "videos"
    social.mkdir()
    archive = tmp_path / "new_video.tar"
    _archive(archive, {"new_author/unique.mp3": b"unique audio"})

    receipt = audit.build_receipt(
        archive_path=archive,
        social_root=social,
        source_inventory_path=_source_inventory(tmp_path, []),
        usb_coverage_receipt_path=_coverage(tmp_path),
        strict_production=False,
    )

    assert receipt["exact_duplicate_members"] == 0
    assert receipt["unique_members"] == 1
    assert receipt["archive_excluded_from_training_as_duplicate"] is False
    assert receipt["requires_unique_member_vad_pipeline"] is True


def test_archived_social_rejects_unsafe_member_path(tmp_path: Path) -> None:
    social = tmp_path / "videos"
    social.mkdir()
    archive = tmp_path / "new_video.tar"
    _archive(archive, {"../escape.mp3": b"audio"})
    with pytest.raises(ValueError, match="Unsafe archived-social member path"):
        audit.index_archive(archive)


def test_archived_social_receipt_write_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert audit._write_immutable(output, {"value": 1}) == "created"
    assert audit._write_immutable(output, {"value": 1}) == "reused"
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        audit._write_immutable(output, {"value": 2})
