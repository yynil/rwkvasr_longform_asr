from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pytest

from rwkvasr.eval.stage211_public_overlap import (
    create_stage211_public_overlap_audit,
    validate_stage211_public_overlap_receipt,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_tar(path: Path, members: dict[str, bytes]) -> dict[str, tuple[int, int]]:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    offsets: dict[str, tuple[int, int]] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive:
            offsets[member.name] = (member.offset_data, member.size)
    return offsets


def _fixture(tmp_path: Path) -> dict[str, Path]:
    public_audio = tmp_path / "public"
    public_audio.mkdir()
    exact_audio = public_audio / "test_exact.mp3"
    mismatch_audio = public_audio / "test_mismatch.mp3"
    exact_audio.write_bytes(b"exact-mp3-bytes")
    mismatch_audio.write_bytes(b"public-mismatch-bytes")

    public_manifest = tmp_path / "public.jsonl"
    _write_jsonl(
        public_manifest,
        [
            {
                "utt_id": "test_exact",
                "audio_filepath": str(exact_audio),
                "text": "exact",
                "dataset": "commonvoice_en_test",
            },
            {
                "utt_id": "test_mismatch",
                "audio_filepath": str(mismatch_audio),
                "text": "mismatch",
                "dataset": "commonvoice_en_test",
            },
        ],
    )

    test_tsv = tmp_path / "test.tsv"
    with test_tsv.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=["client_id", "path", "sentence_id", "sentence"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "client_id": "client-a",
                "path": "test_exact.mp3",
                "sentence_id": "sentence-a",
                "sentence": "exact",
            }
        )
        writer.writerow(
            {
                "client_id": "client-b",
                "path": "test_mismatch.mp3",
                "sentence_id": "sentence-b",
                "sentence": "mismatch",
            }
        )

    training_tar = tmp_path / "training.tar"
    offsets = _write_tar(
        training_tar,
        {
            "exact.mp3": exact_audio.read_bytes(),
            "mismatch.mp3": b"different-training-bytes",
            "unrelated.mp3": b"unrelated",
        },
    )
    rows: list[dict[str, object]] = []
    for utt_id, member, client_id, sentence_id, source in (
        ("train-exact", "exact.mp3", "client-a", "sentence-a", "commonvoice_en"),
        ("train-mismatch", "mismatch.mp3", "client-b", "sentence-b", "commonvoice_en"),
        ("train-unrelated", "unrelated.mp3", "client-z", "sentence-z", "commonvoice_en"),
        ("train-zh", "unrelated.mp3", "client-a", "sentence-a", "commonvoice_cn"),
    ):
        offset, size = offsets[member]
        rows.append(
            {
                "_stage178_source": source,
                "audio_member": member,
                "audio_offset": offset,
                "audio_size": size,
                "cv22_client_id": client_id,
                "cv22_sentence_id": sentence_id,
                "tar_path": str(training_tar),
                "utt_id": utt_id,
            }
        )
    stage178_index = tmp_path / "stage178.jsonl"
    _write_jsonl(stage178_index, rows)
    converter = tmp_path / "converter.py"
    converter.write_text("# merged split fixture\n", encoding="utf-8")
    loaded_receipt = tmp_path / "loaded.json"
    loaded_receipt.write_text("{}\n", encoding="utf-8")
    return {
        "converter": converter,
        "loaded_receipt": loaded_receipt,
        "public_manifest": public_manifest,
        "stage178_index": stage178_index,
        "test_tsv": test_tsv,
    }


def test_overlap_audit_excludes_only_byte_identical_audio(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_dir = tmp_path / "audit"
    receipt = create_stage211_public_overlap_audit(
        stage178_index=fixture["stage178_index"],
        commonvoice_test_tsv=fixture["test_tsv"],
        public_manifest=fixture["public_manifest"],
        converter_source=fixture["converter"],
        output_dir=output_dir,
        loaded_manifest_receipt=fixture["loaded_receipt"],
        validate_loaded_binding=False,
        expected_stage178_english_rows=3,
        expected_public_rows=2,
        expected_candidate_rows=2,
    )

    assert receipt["coverage"]["candidate_training_rows"] == 2
    assert receipt["coverage"]["exact_byte_identical_training_rows"] == 1
    assert receipt["coverage"]["excluded_public_rows"] == 1
    assert receipt["coverage"]["clean_public_rows"] == 1
    exclusions = [
        json.loads(line)
        for line in (output_dir / "commonvoice_en_test_exclusions.jsonl").read_text().splitlines()
    ]
    assert [row["public_utt_id"] for row in exclusions] == ["test_exact"]
    clean = [
        json.loads(line)
        for line in (output_dir / "commonvoice_en_test.clean.jsonl").read_text().splitlines()
    ]
    assert [row["utt_id"] for row in clean] == ["test_mismatch"]
    validate_stage211_public_overlap_receipt(
        output_dir / "receipt.json",
        expected_public_manifest=fixture["public_manifest"],
        expected_loaded_manifest_receipt=fixture["loaded_receipt"],
    )


def test_overlap_audit_rejects_public_test_identity_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    public_rows = [json.loads(line) for line in fixture["public_manifest"].read_text().splitlines()]
    public_rows.pop()
    _write_jsonl(fixture["public_manifest"], public_rows)

    with pytest.raises(ValueError, match="identity mismatch"):
        create_stage211_public_overlap_audit(
            stage178_index=fixture["stage178_index"],
            commonvoice_test_tsv=fixture["test_tsv"],
            public_manifest=fixture["public_manifest"],
            converter_source=fixture["converter"],
            output_dir=tmp_path / "audit",
            loaded_manifest_receipt=None,
        )


def test_overlap_receipt_rejects_changed_clean_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_dir = tmp_path / "audit"
    create_stage211_public_overlap_audit(
        stage178_index=fixture["stage178_index"],
        commonvoice_test_tsv=fixture["test_tsv"],
        public_manifest=fixture["public_manifest"],
        converter_source=fixture["converter"],
        output_dir=output_dir,
        loaded_manifest_receipt=None,
    )
    clean = output_dir / "commonvoice_en_test.clean.jsonl"
    clean.write_text(clean.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output changed"):
        validate_stage211_public_overlap_receipt(output_dir / "receipt.json")
