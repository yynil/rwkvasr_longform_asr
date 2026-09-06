from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from rwkvasr.eval.stage211_sft_public_clean import (
    EXCLUSION_REASON,
    filter_length_index,
    updated_summary,
    updated_webdataset_index,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(*, key: str, offset: int, split: str = "train") -> dict[str, Any]:
    return {
        "audio_member": f"{key}.mp3",
        "audio_offset": offset,
        "audio_size": 100 + offset,
        "ctc_num_tokens": 5,
        "key": key,
        "language": "en",
        "num_frames": 200,
        "shard_name": "commonvoice_en_000000.tar",
        "source_dataset": "commonvoice_en",
        "split": split,
        "stage211_sft_interleave_lane": "en:commonvoice_en:lane_00",
        "utt_id": f"utt-{key}",
    }


def _exclusion(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "audio_member",
            "audio_offset",
            "audio_size",
            "key",
            "shard_name",
            "source_dataset",
            "split",
            "utt_id",
        )
    } | {
        "public_matches": [
            {
                "audio_sha256": "a" * 64,
                "dataset": "commonvoice_en_test",
                "public_utt_id": "public-1",
            }
        ]
    }


def test_filter_length_index_excludes_every_bound_identity(tmp_path: Path) -> None:
    rows = [_row(key="keep", offset=0), _row(key="drop", offset=200)]
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    _write_jsonl(source, rows)
    exclusion = _exclusion(rows[1])

    result = filter_length_index(
        source_path=source,
        output_path=output,
        exclusions={
            (rows[1]["shard_name"], rows[1]["audio_offset"], rows[1]["key"]): exclusion
        },
    )

    assert result["source_rows"] == 2
    assert result["retained_rows"] == 1
    assert result["excluded_rows"] == 1
    assert result["retained_split_counts"] == {"train": 1}
    assert json.loads(output.read_text(encoding="utf-8"))["key"] == "keep"
    assert result["excluded_records"][0]["public_matches"] == exclusion["public_matches"]


def test_filter_length_index_rejects_unresolved_exclusion(tmp_path: Path) -> None:
    row = _row(key="keep", offset=0)
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [row])
    missing = _row(key="missing", offset=100)

    with pytest.raises(ValueError, match="not found exactly once"):
        filter_length_index(
            source_path=source,
            output_path=tmp_path / "output.jsonl",
            exclusions={
                (missing["shard_name"], missing["audio_offset"], missing["key"]): _exclusion(
                    missing
                )
            },
        )


def test_updated_summary_accounts_public_exclusion() -> None:
    summary = {
        "output_dir": "/old",
        "length_index_path": "/old/webdataset_lengths.jsonl",
        "index_path": "/old/webdataset_index.json",
        "num_kept_samples": 10,
        "num_dropped_samples": 2,
        "counts": {
            "kept_by_split": {"train": 9, "eval": 1},
            "kept_by_source": {"commonvoice_en": 10},
            "kept_by_language": {"en": 10},
            "kept_by_label_source": {"metadata": 10},
            "kept_by_split_source": {"train/commonvoice_en": 9, "eval/commonvoice_en": 1},
            "kept_by_split_language": {"train/en": 9, "eval/en": 1},
            "kept_by_interleave_lane": {"en:commonvoice_en:lane_00": 10},
            "dropped_by_reason": {"non_pronunciation_token": 2},
            "dropped_by_source": {"commonvoice_en": 2},
            "dropped_by_language": {"en": 2},
        },
    }
    record = {
        **_row(key="drop", offset=200),
        "public_matches": [],
    }

    result = updated_summary(
        summary,
        output_root=Path("/clean"),
        excluded_records=[record],
    )

    assert result["num_kept_samples"] == 9
    assert result["num_dropped_samples"] == 3
    assert result["counts"]["kept_by_split"]["train"] == 8
    assert result["counts"]["kept_by_source"]["commonvoice_en"] == 9
    assert result["counts"]["dropped_by_reason"] == {
        "non_pronunciation_token": 2,
        EXCLUSION_REASON: 1,
    }
    assert result["counts"]["dropped_by_source"]["commonvoice_en"] == 3


def test_updated_webdataset_index_preserves_split_and_shard_accounting() -> None:
    index = {
        "root": "/old",
        "num_samples": 10,
        "splits": {"train": {"num_samples": 9}, "eval": {"num_samples": 1}},
        "shards": [
            {
                "name": "commonvoice_en_000000.tar",
                "num_samples": 10,
                "splits": {
                    "train": {"num_samples": 9},
                    "eval": {"num_samples": 1},
                },
            }
        ],
    }
    record = {**_row(key="drop", offset=200), "public_matches": []}

    result = updated_webdataset_index(
        index,
        output_root=Path("/clean"),
        excluded_records=[record],
    )

    assert result["num_samples"] == 9
    assert result["splits"]["train"]["num_samples"] == 8
    assert result["shards"][0]["num_samples"] == 9
    assert result["shards"][0]["splits"]["train"]["num_samples"] == 8


def test_stage211_sft_production_defaults_use_public_clean_artifacts() -> None:
    audit = importlib.import_module("scripts.audit_stage211_sft_public_overlap")
    correction_builder = importlib.import_module(
        "scripts.build_stage211_sft_correction_profile"
    )
    profile_builder = importlib.import_module(
        "scripts.create_stage211_labeled_profile_receipt"
    )
    sft_runner = importlib.import_module("scripts.run_stage211_labeled_sft")
    correction_runner = importlib.import_module("scripts.run_stage211_sft_correction")
    correction_loop = importlib.import_module("scripts.run_stage211_sft_correction_loop")

    data_root = Path.home() / "rwkvasr_data"
    clean_root = data_root / "stage211_sft_full_labeled_v3_public_clean"
    clean_overlap = data_root / "stage211_sft_public_encoded_overlap_v2_clean"
    clean_correction = (
        data_root / "stage211_sft_source_balanced_correction_v3_public_clean"
    )
    assert audit.DEFAULT_LABELED_ROOT == clean_root
    assert audit.DEFAULT_LABELED_PROFILE == clean_root / "stage211_labeled_profile_receipt.json"
    assert audit.DEFAULT_OUTPUT_DIR == clean_overlap
    assert correction_builder.DEFAULT_FULL_ROOT == clean_root
    assert correction_builder.DEFAULT_FULL_PROFILE == (
        clean_root / "stage211_labeled_profile_receipt.json"
    )
    assert correction_builder.DEFAULT_OUTPUT_ROOT == clean_correction
    assert profile_builder.DEFAULT_LABELED_ROOT == clean_root
    assert sft_runner.DEFAULT_LABELED_ROOT == clean_root
    assert sft_runner.DEFAULT_LABELED_PROFILE_RECEIPT == (
        clean_root / "stage211_labeled_profile_receipt.json"
    )
    assert sft_runner.DEFAULT_SFT_PUBLIC_OVERLAP_RECEIPT == clean_overlap / "receipt.json"
    expected_correction_profile = clean_correction / "stage211_sft_correction_profile.json"
    assert correction_runner.DEFAULT_PROFILE == expected_correction_profile
    assert correction_loop.DEFAULT_PROFILE_ROOT == clean_correction

    bootstrap = (REPO_ROOT / "scripts/start_stage211_abcd_after_calibration.sh").read_text(
        encoding="utf-8"
    )
    monitor = (REPO_ROOT / "scripts/monitor_stage211_abcd.sh").read_text(encoding="utf-8")
    assert "stage211_sft_full_labeled_v3_public_clean" in bootstrap
    assert "stage211_sft_public_encoded_overlap_v2_clean" in bootstrap
    assert "stage211_sft_source_balanced_correction_v3_public_clean" in bootstrap
    assert "stage211_sft_full_labeled_v3_public_clean" in monitor
