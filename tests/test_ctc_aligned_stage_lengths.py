from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.eval.stage211_gate import (
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT,
    STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256,
    stage211_sft_ctc_suppressed_token_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
builder = importlib.import_module("scripts.build_ctc_aligned_stage_lengths")
labeled_profile = importlib.import_module("scripts.create_stage211_labeled_profile_receipt")
strict_runner = importlib.import_module("scripts.run_stage211_strict_chained_alignment")


class _Tokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = token_ids

    def encode(self, _: str) -> list[int]:
        return list(self._token_ids)


def _write_length_index(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "shard_name": "sample.tar",
                "key": "sample-1",
                "utt_id": "sample-1",
                "split": "train",
                "num_frames": 120,
                "json_member": "sample-1.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _arguments(source: Path, index: Path, output: Path, *extra: str) -> list[str]:
    return [
        "build_ctc_aligned_stage_lengths.py",
        "--shard-root",
        str(source),
        "--length-index-path",
        str(index),
        "--output-dir",
        str(output),
        "--tokenizer-type",
        "sensevoice_tiktoken",
        "--tokenizer-model-path",
        str(source / "tokenizer.tiktoken"),
        "--text-normalization",
        "ctc",
        "--frontend-downsample",
        "sensevoice_lfr6",
        *extra,
    ]


def test_stage211_non_pronunciation_target_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    index = tmp_path / "lengths.jsonl"
    output = tmp_path / "output"
    _write_length_index(index)
    suppressed_id = stage211_sft_ctc_suppressed_token_ids()[0]
    monkeypatch.setattr(
        builder, "build_text_tokenizer", lambda *_args, **_kwargs: _Tokenizer([suppressed_id])
    )
    monkeypatch.setattr(
        builder.TarJsonReader,
        "read_json",
        lambda _self, _entry: {
            "id": "sample-1",
            "text": "hello",
            "language": "en",
            "source_dataset": "fixture",
            "source_split": "train",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(
            source,
            index,
            output,
            "--forbid-stage211-non-pronunciation-tokens",
            "--fail-on-error",
        ),
    )

    builder.main()

    assert (output / "webdataset_lengths.jsonl").read_text(encoding="utf-8") == ""
    summary = json.loads((output / "webdataset_lengths.summary.json").read_text(encoding="utf-8"))
    assert summary["num_input_samples"] == 1
    assert summary["num_kept_samples"] == 0
    assert summary["counts"]["dropped_by_reason"] == {"non_pronunciation_token": 1}
    assert summary["forbidden_token_ids_count"] == (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT)
    assert summary["forbidden_token_ids_sha256"] == (STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256)


def test_strict_preparation_failure_does_not_replace_completed_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    index = tmp_path / "lengths.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    completed_index = output / "webdataset_lengths.jsonl"
    completed_index.write_text("stable-complete-index\n", encoding="utf-8")
    _write_length_index(index)
    monkeypatch.setattr(builder, "build_text_tokenizer", lambda *_args, **_kwargs: _Tokenizer([1]))

    def _fail_read(_self: object, _entry: dict[str, object]) -> dict[str, object]:
        raise ValueError("broken metadata")

    monkeypatch.setattr(builder.TarJsonReader, "read_json", _fail_read)
    monkeypatch.setattr(
        sys,
        "argv",
        _arguments(source, index, output, "--fail-on-error"),
    )

    with pytest.raises(RuntimeError, match="Failed strict CTC-label preparation"):
        builder.main()

    assert completed_index.read_text(encoding="utf-8") == "stable-complete-index\n"


def test_source_split_aliases_are_normalized() -> None:
    assert builder._source_split_values(
        {
            "source_split": " Test ",
            "subset": ["DEV", "train"],
            "partition": None,
        }
    ) == {"test", "dev", "train"}


def test_stage211_full_labeled_profile_is_immutable_and_deeply_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "labeled"
    bucket_dir = root / "webdataset_buckets_audio_text"
    bucket_dir.mkdir(parents=True)
    length_index = root / "webdataset_lengths.jsonl"
    bucket_manifest = bucket_dir / "manifest.json"
    summary = root / "webdataset_lengths.summary.json"
    log = root / "prepare_ctc_aligned.log"
    source_filter = root / ".filter_complete.json"
    for path in (length_index, bucket_manifest, summary, log, source_filter):
        path.write_text(f"{path.name}\n", encoding="utf-8")
    expected = {
        "train_samples": 100,
        "eval_samples": 2,
        "total_samples": 102,
        "total_hours": 1.25,
        "ctc_tokens": 500,
        "ctc_unk_tokens": 0,
        "ctc_forbidden_tokens": 0,
        "unique_utterance_ids": 102,
        "pronunciation_target_samples": 102,
        "ctc_feasible_samples": 102,
        "estimated_train_steps": 5,
        "tail_padding_samples_per_epoch": 1,
        "tail_padding_sample_exposures": 1,
        "executed_sample_exposures": 101,
        "source_counts": {"english": 60, "chinese": 42},
        "language_counts": {"en": 60, "zh": 42},
        "interleave_lane_counts": {"en:english:lane_00": 60, "zh:chinese:lane_00": 42},
    }
    audit = {
        **expected,
        "label_preparation": {
            "profile_schema_version": 2,
            "source_filter": {
                "path": str(source_filter),
                "sha256": labeled_profile.sha256_file(source_filter),
            },
            "summary_path": str(summary),
            "summary_sha256": labeled_profile.sha256_file(summary),
            "log_path": str(log),
            "log_sha256": labeled_profile.sha256_file(log),
            "interleave_source_field": "stage211_sft_interleave_lane",
            "interleave_source_lanes": {"chinese": 1, "english": 1},
        },
    }
    monkeypatch.setattr(labeled_profile, "_audit_labeled_data", lambda **_kwargs: audit)

    receipt = labeled_profile.build_receipt(labeled_root=root)
    receipt_path = root / "stage211_labeled_profile_receipt.json"
    labeled_profile.write_immutable_receipt(receipt_path, receipt)

    assert labeled_profile.validate_receipt(receipt_path, labeled_root=root) == receipt
    changed = dict(receipt)
    changed["expected"] = {**expected, "ctc_tokens": 499}
    with pytest.raises(ValueError, match="Refusing to replace"):
        labeled_profile.write_immutable_receipt(receipt_path, changed)


def test_stage211_full_label_preparation_binds_filter_and_pronunciation_support(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    source_index = source / "webdataset_lengths.jsonl"
    output_index = output / "webdataset_lengths.jsonl"
    source_index.write_text("source-index\n", encoding="utf-8")
    output_index.write_text("output-index\n", encoding="utf-8")
    source_counts = dict(strict_runner.STAGE211_FULL_LABELED_INPUT_SOURCE_COUNTS)
    language_counts = dict(strict_runner.STAGE211_FULL_LABELED_INPUT_LANGUAGE_COUNTS)
    lane_counts: dict[str, int] = {}
    for (
        source_name,
        lane_count,
    ) in strict_runner.STAGE211_FULL_LABELED_INTERLEAVE_SOURCE_LANES.items():
        language = strict_runner.STAGE211_FULL_LABELED_INTERLEAVE_SOURCE_LANGUAGES[source_name]
        source_samples = source_counts[source_name]
        samples_per_lane, remainder = divmod(source_samples, lane_count)
        for lane_id in range(lane_count):
            lane_counts[f"{language}:{source_name}:lane_{lane_id:02d}"] = samples_per_lane + int(
                lane_id < remainder
            )
    (source / ".filter_complete.json").write_text(
        json.dumps(
            {
                "sources": source_counts,
                "num_samples": strict_runner.STAGE211_FULL_LABELED_INPUT_SAMPLES,
                "num_shards": 107,
                "excluded_splits": list(strict_runner.STAGE211_FULL_LABELED_REJECTED_SOURCE_SPLITS),
                "min_duration": 0.5,
                "max_duration": 20.0,
                "min_speech_ratio": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    total = strict_runner.STAGE211_FULL_LABELED_INPUT_SAMPLES
    summary = {
        "version": 1,
        "source_root": str(source),
        "source_length_index_path": str(source_index),
        "output_dir": str(output),
        "length_index_path": str(output_index),
        "tokenizer_type": "sensevoice_tiktoken",
        "tokenizer_model_path": str(
            (REPO_ROOT / "assets/fun-asr-nano-2512/multilingual.tiktoken").resolve()
        ),
        "text_normalization": "ctc",
        "frontend_downsample": "sensevoice_lfr6",
        "drop_unk_token": True,
        "unk_token_id": None,
        "forbid_stage211_non_pronunciation_tokens": True,
        "forbidden_token_ids_count": STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_COUNT,
        "forbidden_token_ids_sha256": STAGE211_SFT_CTC_SUPPRESSED_TOKEN_IDS_SHA256,
        "reject_source_splits": list(strict_runner.STAGE211_FULL_LABELED_REJECTED_SOURCE_SPLITS),
        "fail_on_error": True,
        "interleave_source_lanes": dict(
            strict_runner.STAGE211_FULL_LABELED_INTERLEAVE_SOURCE_LANES
        ),
        "num_input_samples": total,
        "num_kept_samples": total,
        "num_dropped_samples": 0,
        "counts": {
            "input_by_split": {"eval": 7_165, "train": 1_423_636},
            "input_by_source": source_counts,
            "input_by_language": language_counts,
            "source_split_labels": {"train": total},
            "kept_by_split": {"eval": 7_165, "train": 1_423_636},
            "kept_by_source": source_counts,
            "kept_by_language": language_counts,
            "kept_by_label_source": {"metadata": total},
            "kept_by_split_source": {},
            "kept_by_split_language": {},
            "dropped_by_reason": {},
            "dropped_by_source": {},
            "dropped_by_language": {},
            "dropped_unk_tokens_by_source": {},
            "dropped_unk_tokens_by_language": {},
            "dropped_forbidden_tokens_by_source": {},
            "dropped_forbidden_tokens_by_language": {},
            "rejected_source_split_labels": {},
            "kept_by_interleave_lane": lane_counts,
        },
    }
    (output / "webdataset_lengths.summary.json").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    (output / "prepare_ctc_aligned.log").write_text(
        "tokenizer_type=sensevoice_tiktoken\n"
        "text_normalization=ctc\n"
        "frontend_downsample=sensevoice_lfr6\n"
        "drop_unk_token=1\n"
        "forbid_stage211_non_pronunciation_tokens=1\n"
        "reject_source_splits=dev,eval,test,valid,validation\n"
        "fail_on_error=1\n"
        "interleave_source_lanes="
        "aishell3:2,commonvoice_cn:1,commonvoice_en:35,librispeech:7\n"
        "bucket_source_field=stage211_sft_interleave_lane\n"
        "CTC-aligned clean preprocessing complete\n",
        encoding="utf-8",
    )

    proof = strict_runner._label_preparation_proof(
        webdataset_root=output,
        length_index_path=output_index,
    )

    assert proof["profile_schema_version"] == 2
    assert proof["accepted_samples"] == total
    assert proof["source_counts"] == source_counts
    assert proof["interleave_lane_counts"] == lane_counts
    assert proof["source_filter"]["source_length_index_sha256"] == (
        labeled_profile.sha256_file(source_index)
    )


def test_stage211_full_input_languages_bind_observed_metadata_counts() -> None:
    assert strict_runner.STAGE211_FULL_LABELED_INPUT_LANGUAGE_COUNTS == {
        "en": 1_334_783,
        "zh": 96_018,
    }
    source_derived_english = sum(
        strict_runner.STAGE211_FULL_LABELED_INPUT_SOURCE_COUNTS[source]
        for source in ("commonvoice_en", "librispeech")
    )
    assert source_derived_english == (
        strict_runner.STAGE211_FULL_LABELED_INPUT_LANGUAGE_COUNTS["en"] + 2
    )


def test_interleave_source_lanes_are_parsed_and_assigned_deterministically() -> None:
    assert builder._parse_interleave_source_lanes(
        ["commonvoice_en:35,librispeech:7", "aishell3:2", "commonvoice_cn:1"]
    ) == {
        "aishell3": 2,
        "commonvoice_cn": 1,
        "commonvoice_en": 35,
        "librispeech": 7,
    }
    first = builder._interleave_lane(
        source="commonvoice_en",
        language="en",
        sample_id="fixture-1",
        lanes=35,
    )
    assert first == builder._interleave_lane(
        source="commonvoice_en",
        language="en",
        sample_id="fixture-1",
        lanes=35,
    )
    assert first.startswith("en:commonvoice_en:lane_")
    assert 0 <= int(first.rsplit("_", 1)[1]) < 35


def test_full_labeled_audit_binds_bucket_manifest_to_interleave_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "labeled"
    root.mkdir()
    length_index = root / "webdataset_lengths.jsonl"
    lane = "zh:commonvoice_cn:lane_00"
    rows = [
        {
            "key": f"fixture-{split}",
            "utt_id": f"fixture-{split}",
            "split": split,
            "num_frames": 120,
            "json_member": f"fixture-{split}.json",
            "normalized_text_chars": 2,
            "ctc_num_tokens": 2,
            "ctc_unk_tokens": 0,
            "ctc_forbidden_tokens": 0,
            "ctc_required_frames": 2,
            "ctc_logit_frames": 20,
            "source_dataset": "commonvoice_cn",
            "language": "zh",
            "stage211_sft_interleave_lane": lane,
        }
        for split in ("train", "eval")
    ]
    length_index.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest_payload = {
        "version": 1,
        "root": str(root),
        "source_length_index_path": str(length_index),
        "bucket_width": 80,
        "entries_per_part": 100_000,
        "source_field": "stage211_sft_interleave_lane",
        "splits": {
            split: {
                "num_samples": 1,
                "buckets": [
                    {
                        "bucket_id": 1,
                        "num_samples": 1,
                        "parts": [
                            {
                                "path": f"{split}.jsonl",
                                "num_samples": 1,
                                "source_label": lane,
                            }
                        ],
                    }
                ],
            }
            for split in ("train", "eval")
        },
    }
    manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
    preparation = {
        "profile_schema_version": 2,
        "source_counts": {"commonvoice_cn": 2},
        "language_counts": {"zh": 2},
        "interleave_lane_counts": {lane: 2},
    }
    monkeypatch.setattr(
        strict_runner,
        "_label_preparation_proof",
        lambda **_kwargs: preparation,
    )

    audit = strict_runner._audit_labeled_data(
        webdataset_root=root,
        length_index_path=length_index,
        bucket_manifest_path=manifest,
    )
    assert audit["interleave_lane_counts"] == {lane: 2}

    manifest_payload["source_field"] = "source_dataset"
    manifest.write_text(json.dumps(manifest_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong interleave field"):
        strict_runner._audit_labeled_data(
            webdataset_root=root,
            length_index_path=length_index,
            bucket_manifest_path=manifest,
        )
