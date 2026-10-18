from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from rwkvasr.eval.stage211_gate import sha256_file
from rwkvasr.eval.stage211_sft_public_overlap import (
    EXPECTED_DATASETS,
    _load_profile_binding,
    _resolve_labeled_shard,
    build_stage211_sft_public_overlap_audit,
    validate_stage211_sft_public_overlap_receipt,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_tar(path: Path, payloads: dict[str, bytes]) -> dict[str, tuple[int, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    offsets: dict[str, tuple[int, int]] = {}
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if member.isfile():
                offsets[member.name] = (member.offset_data, member.size)
    return offsets


def _fixture(
    tmp_path: Path,
    *,
    overlap_train: bool,
    matching_sizes: bool = True,
) -> tuple[Path, Path, Path, Path]:
    public_payloads = {
        dataset: (f"public-{index:02d}-audio").encode()
        for index, dataset in enumerate(EXPECTED_DATASETS)
    }
    nano_results: list[dict[str, Any]] = []
    fingerprint_records: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        public_id = f"{dataset}-utt"
        manifest = tmp_path / "public" / "manifests" / f"{dataset}.jsonl"
        _write_jsonl(manifest, [{"utt_id": public_id, "text": "reference"}])
        nano_results.append(
            {
                "dataset": dataset,
                "manifest_path": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "sample_count": 1,
            }
        )
        part = tmp_path / "public" / "fingerprints" / f"{dataset}.jsonl"
        payload = public_payloads[dataset]
        _write_jsonl(
            part,
            [
                {
                    "audio_sha256": hashlib.sha256(payload).hexdigest(),
                    "audio_size_bytes": len(payload),
                    "dataset": dataset,
                    "utt_id": public_id,
                }
            ],
        )
        fingerprint_receipt = part.with_suffix(".receipt.json")
        _write_json(fingerprint_receipt, {"dataset": dataset, "rows": 1})
        fingerprint_records.append(
            {
                "dataset": dataset,
                "part_path": str(part),
                "part_sha256": sha256_file(part),
                "path": str(fingerprint_receipt),
                "rows": 1,
                "sha256": sha256_file(fingerprint_receipt),
            }
        )

    nano_provenance = tmp_path / "public" / "nano_provenance.json"
    _write_json(
        nano_provenance,
        {
            "artifact": "nano_public_baseline_provenance",
            "complete": True,
            "pipeline": "stage211",
            "results": nano_results,
            "total_samples": len(EXPECTED_DATASETS),
        },
    )
    public_pcm_audit = tmp_path / "public" / "pcm_audit.json"
    _write_json(
        public_pcm_audit,
        {
            "artifact": "stage211_base_public_pcm_overlap_audit",
            "complete": True,
            "pipeline": "stage211",
            "public_fingerprint_receipts": fingerprint_records,
            "public_overlap_rows": 0,
        },
    )

    first_public = public_payloads[EXPECTED_DATASETS[0]]
    second_public = public_payloads[EXPECTED_DATASETS[1]]
    train_payload = (
        first_public
        if overlap_train
        else b"x" * len(first_public)
        if matching_sizes
        else b"xyz"
    )
    eval_payload = b"y" * len(second_public) if matching_sizes else b"abcde"
    labeled_root = tmp_path / "labeled"
    shard = labeled_root / "samples.tar"
    offsets = _write_tar(shard, {"train.wav": train_payload, "eval.flac": eval_payload})
    length_index = labeled_root / "webdataset_lengths.jsonl"
    _write_jsonl(
        length_index,
        [
            {
                "audio_member": "train.wav",
                "audio_offset": offsets["train.wav"][0],
                "audio_size": offsets["train.wav"][1],
                "key": "train-key",
                "shard_name": shard.name,
                "source_dataset": "commonvoice_en",
                "split": "train",
                "utt_id": "train-utt",
            },
            {
                "audio_member": "eval.flac",
                "audio_offset": offsets["eval.flac"][0],
                "audio_size": offsets["eval.flac"][1],
                "key": "eval-key",
                "shard_name": shard.name,
                "source_dataset": "librispeech",
                "split": "eval",
                "utt_id": "eval-utt",
            },
        ],
    )
    bucket_manifest = labeled_root / "webdataset_buckets_audio_text" / "manifest.json"
    _write_json(bucket_manifest, {"splits": {"train": {}, "eval": {}}})
    profile = labeled_root / "stage211_labeled_profile_receipt.json"
    _write_json(
        profile,
        {
            "artifact": "stage211_labeled_profile_receipt",
            "bucket_manifest_path": str(bucket_manifest),
            "bucket_manifest_sha256": sha256_file(bucket_manifest),
            "complete": True,
            "expected": {
                "eval_samples": 1,
                "source_counts": {"commonvoice_en": 1, "librispeech": 1},
                "total_samples": 2,
                "train_samples": 1,
            },
            "labeled_webdataset_root": str(labeled_root),
            "length_index_path": str(length_index),
            "length_index_sha256": sha256_file(length_index),
            "phase": "sft",
            "pipeline": "stage211",
            "schema_version": 2,
        },
    )
    return profile, nano_provenance, public_pcm_audit, tmp_path / "audit"


def test_sft_public_overlap_audit_accepts_complete_clear_scan(tmp_path: Path) -> None:
    profile, nano, pcm, output = _fixture(tmp_path, overlap_train=False)
    receipt = build_stage211_sft_public_overlap_audit(
        labeled_profile_path=profile,
        nano_provenance_path=nano,
        public_pcm_audit_path=pcm,
        output_dir=output,
    )

    assert receipt["training_ready"] is True
    assert receipt["coverage"]["scanned_rows"] == 2
    assert receipt["coverage"]["candidate_rows"] == 2
    assert receipt["overlap"]["training_rows"] == 0
    assert receipt["matches_output"]["rows"] == 0
    assert validate_stage211_sft_public_overlap_receipt(output / "receipt.json") == receipt


def test_sft_public_overlap_audit_accepts_zero_size_candidates(tmp_path: Path) -> None:
    profile, nano, pcm, output = _fixture(
        tmp_path,
        overlap_train=False,
        matching_sizes=False,
    )
    receipt = build_stage211_sft_public_overlap_audit(
        labeled_profile_path=profile,
        nano_provenance_path=nano,
        public_pcm_audit_path=pcm,
        output_dir=output,
    )

    assert receipt["training_ready"] is True
    assert receipt["coverage"]["candidate_rows"] == 0
    assert receipt["coverage"]["opened_shards"] == 0
    assert validate_stage211_sft_public_overlap_receipt(output / "receipt.json") == receipt


def test_sft_public_overlap_audit_rejects_training_match(tmp_path: Path) -> None:
    profile, nano, pcm, output = _fixture(tmp_path, overlap_train=True)
    receipt = build_stage211_sft_public_overlap_audit(
        labeled_profile_path=profile,
        nano_provenance_path=nano,
        public_pcm_audit_path=pcm,
        output_dir=output,
    )

    assert receipt["training_ready"] is False
    assert receipt["overlap"]["training_rows"] == 1
    assert receipt["overlap"]["pair_count"] == 1
    with pytest.raises(ValueError, match="overlaps the public benchmark"):
        validate_stage211_sft_public_overlap_receipt(output / "receipt.json")
    assert (
        validate_stage211_sft_public_overlap_receipt(
            output / "receipt.json",
            require_training_ready=False,
        )["admission_state"]
        == "public_overlap_detected"
    )


def test_sft_public_overlap_audit_rejects_changed_match_output(tmp_path: Path) -> None:
    profile, nano, pcm, output = _fixture(tmp_path, overlap_train=False)
    build_stage211_sft_public_overlap_audit(
        labeled_profile_path=profile,
        nano_provenance_path=nano,
        public_pcm_audit_path=pcm,
        output_dir=output,
    )
    (output / "exact_encoded_audio_matches.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matches output (?:size )?changed"):
        validate_stage211_sft_public_overlap_receipt(output / "receipt.json")


def test_sft_public_overlap_audit_rejects_forged_public_match(tmp_path: Path) -> None:
    profile, nano, pcm, output = _fixture(tmp_path, overlap_train=True)
    build_stage211_sft_public_overlap_audit(
        labeled_profile_path=profile,
        nano_provenance_path=nano,
        public_pcm_audit_path=pcm,
        output_dir=output,
    )
    matches_path = output / "exact_encoded_audio_matches.jsonl"
    match = json.loads(matches_path.read_text(encoding="utf-8"))
    match["public_utt_id"] = "forged-public-id"
    _write_jsonl(matches_path, [match])
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["matches_output"]["sha256"] = sha256_file(matches_path)
    receipt["matches_output"]["size_bytes"] = matches_path.stat().st_size
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="absent from the public fingerprints"):
        validate_stage211_sft_public_overlap_receipt(
            receipt_path,
            require_training_ready=False,
        )


def test_sft_public_overlap_binds_public_clean_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, _, _, _ = _fixture(tmp_path, overlap_train=False)
    root = profile.parent
    rebuild_path = root / "public_overlap_exclusion_rebuild_receipt.json"
    _write_json(rebuild_path, {"artifact": "test-placeholder"})

    def validate_rebuild(path: Path) -> dict[str, Any]:
        assert path == rebuild_path
        return {
            "output_root": str(root),
            "output_profile_path": str(profile),
            "output_profile_sha256": sha256_file(profile),
            "source_profile_path": str(profile),
        }

    monkeypatch.setattr(
        "rwkvasr.eval.stage211_sft_public_clean."
        "validate_stage211_sft_public_clean_rebuild_receipt",
        validate_rebuild,
    )

    _, binding = _load_profile_binding(profile)

    assert binding["public_clean_rebuild_receipt_path"] == str(rebuild_path)
    assert binding["public_clean_rebuild_receipt_sha256"] == sha256_file(rebuild_path)
    assert binding["public_clean_source_webdataset_root"] == str(root)


def test_resolve_labeled_shard_accepts_only_bound_external_symlink(tmp_path: Path) -> None:
    root = tmp_path / "clean"
    source_root = tmp_path / "source"
    unrelated_root = tmp_path / "unrelated"
    for directory in (root, source_root, unrelated_root):
        directory.mkdir()
    shard_name = "samples.tar"
    source_shard = source_root / shard_name
    source_shard.write_bytes(b"source")
    unrelated_shard = unrelated_root / shard_name
    unrelated_shard.write_bytes(b"unrelated")
    clean_shard = root / shard_name
    clean_shard.symlink_to(source_shard)

    assert _resolve_labeled_shard(
        root=root,
        shard_relative=Path(shard_name),
        trusted_source_root=source_root,
        line_number=1,
    ) == source_shard
    with pytest.raises(ValueError, match="escapes its labeled root"):
        _resolve_labeled_shard(
            root=root,
            shard_relative=Path(shard_name),
            trusted_source_root=None,
            line_number=1,
        )

    clean_shard.unlink()
    clean_shard.symlink_to(unrelated_shard)
    with pytest.raises(ValueError, match="unbound external shard target"):
        _resolve_labeled_shard(
            root=root,
            shard_relative=Path(shard_name),
            trusted_source_root=source_root,
            line_number=1,
        )
