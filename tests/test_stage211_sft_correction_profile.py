from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
builder = importlib.import_module("scripts.build_stage211_sft_correction_profile")


def _row(
    *,
    source: str,
    language: str,
    index: int,
    split: str,
    num_frames: int,
) -> dict[str, object]:
    key = f"{source}-{split}-{index:03d}"
    return {
        "shard_name": f"{source}.tar",
        "key": f"audio-{key}",
        "utt_id": key,
        "split": split,
        "num_frames": num_frames,
        "num_text_chars": 8,
        "num_text_tokens": 4,
        "text_bytes": 16,
        "audio_member": f"{key}.wav",
        "audio_format": "wav",
        "json_member": f"{key}.json",
        "audio_offset": index * 1024,
        "audio_size": 512,
        "json_offset": index * 1024 + 512,
        "json_size": 400,
        "source_dataset": source,
        "language": language,
        "normalized_text_chars": 4,
        "ctc_num_tokens": 4,
        "ctc_adjacent_repeats": 0,
        "ctc_unk_tokens": 0,
        "ctc_forbidden_tokens": 0,
        "ctc_required_frames": 4,
        "ctc_logit_frames": max(5, num_frames // 6),
        "ctc_logit_required_ratio": max(5, num_frames // 6) / 4,
        "ctc_label_source": "metadata",
        "stage211_sft_interleave_lane": f"{language}:{source}:lane_00",
    }


def _write_full_profile_fixture(root: Path) -> tuple[Path, dict[str, object], list[dict]]:
    root.mkdir(parents=True)
    rows: list[dict] = []
    specs = {
        "aishell3": ("zh", 6),
        "commonvoice_cn": ("zh", 2),
        "commonvoice_en": ("en", 8),
        "librispeech": ("en", 8),
    }
    for source, (language, train_rows) in specs.items():
        for index in range(train_rows):
            rows.append(
                _row(
                    source=source,
                    language=language,
                    index=index,
                    split="train",
                    num_frames=(40 if index % 2 == 0 else 120) + index,
                )
            )
        rows.append(
            _row(
                source=source,
                language=language,
                index=99,
                split="eval",
                num_frames=160,
            )
        )
    length_index = root / "webdataset_lengths.jsonl"
    length_index.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    eval_rows = [row for row in rows if row["split"] == "eval"]
    eval_part = root / "full_eval_part.jsonl"
    eval_part.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
            for row in eval_rows
        ),
        encoding="utf-8",
    )
    manifest = root / "webdataset_buckets_audio_text" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "root": str(root.resolve()),
                "source_length_index_path": str(length_index.resolve()),
                "bucket_width": 80,
                "bucket_metric": "audio_frames_plus_text_cost",
                "text_cost_source": "auto",
                "text_cost_weight": 4.0,
                "json_size_text_offset": 256,
                "json_size_bytes_per_token": 4.0,
                "entries_per_part": 100,
                "source_field": "stage211_sft_interleave_lane",
                "splits": {
                    "eval": {
                        "num_samples": len(eval_rows),
                        "buckets": [
                            {
                                "bucket_id": 2,
                                "num_samples": len(eval_rows),
                                "parts": [
                                    {
                                        "path": str(eval_part.resolve()),
                                        "num_samples": len(eval_rows),
                                        "source_label": "full-eval",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    profile_path = root / "stage211_labeled_profile_receipt.json"
    profile_path.write_text("{}\n", encoding="utf-8")
    split_counts = Counter(str(row["split"]) for row in rows)
    source_counts = Counter(str(row["source_dataset"]) for row in rows)
    language_counts = Counter(str(row["language"]) for row in rows)
    profile: dict[str, object] = {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "stage211_labeled_profile_receipt",
        "complete": True,
        "labeled_webdataset_root": str(root.resolve()),
        "length_index_path": str(length_index.resolve()),
        "bucket_manifest_path": str(manifest.resolve()),
        "source_filter": {
            "excluded_splits": ["dev", "eval", "test", "valid", "validation"]
        },
        "expected": {
            "train_samples": split_counts["train"],
            "eval_samples": split_counts["eval"],
            "total_samples": len(rows),
            "unique_utterance_ids": len(rows),
            "source_counts": dict(sorted(source_counts.items())),
            "language_counts": dict(sorted(language_counts.items())),
        },
        "labeled_data_audit": {
            "label_preparation": {
                "tokenizer_type": "sensevoice_tiktoken",
                "tokenizer_model_path": "/tokenizer.tiktoken",
                "tokenizer_model_sha256": "1" * 64,
                "text_normalization": "ctc",
                "frontend_downsample": "sensevoice_lfr6",
                "drop_unk_token": True,
                "ctc_suppressed_token_ids_count": 2114,
                "ctc_suppressed_token_ids_sha256": "2" * 64,
            }
        },
    }
    return profile_path, profile, rows


def test_stage211_sft_correction_selection_is_balanced_and_excludes_eval(
    tmp_path: Path,
) -> None:
    _, _, rows = _write_full_profile_fixture(tmp_path / "full")
    selection = builder.select_correction_rows(
        tmp_path / "full" / "webdataset_lengths.jsonl",
        seed=2114,
        bucket_width=80,
    )

    selected_keys = {candidate.key for candidate in selection.candidates}
    chinese_train_keys = {
        str(row["utt_id"])
        for row in rows
        if row["split"] == "train" and row["language"] == "zh"
    }
    eval_keys = {str(row["utt_id"]) for row in rows if row["split"] == "eval"}
    languages = Counter(candidate.language for candidate in selection.candidates)

    assert languages == {"en": 8, "zh": 8}
    assert chinese_train_keys <= selected_keys
    assert selected_keys.isdisjoint(eval_keys)
    assert selection.source_quotas == {"commonvoice_en": 4, "librispeech": 4}
    assert set(selection.bucket_quotas.values()) == {2}
    assert selection.scan["eval_unique_keys"] == 4
    assert selection.scan["english_second_pass_rows"] == 16


def test_stage211_sft_correction_profile_is_deeply_bound_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path, profile, rows = _write_full_profile_fixture(tmp_path / "full")
    monkeypatch.setattr(
        builder,
        "validate_full_labeled_profile_receipt",
        lambda path: profile,
    )
    output_root = tmp_path / "correction"
    receipt = builder.build_correction_profile(
        full_profile_path=profile_path,
        output_root=output_root,
        seed=2114,
        bucket_width=80,
        entries_per_part=3,
    )

    assert receipt["train_samples"] == 16
    assert receipt["unique_train_keys"] == 16
    assert receipt["language_counts"] == {"en": 8, "zh": 8}
    assert receipt["source_counts"]["commonvoice_en"] == 4
    assert receipt["source_counts"]["librispeech"] == 4
    assert receipt["eval_monitor_samples"] == 4
    assert receipt["eval_monitor_excluded_from_train_accounting"] is True
    assert receipt["estimated_train_steps"] > 0
    assert len(receipt["bucket_output_parts"]) > 1
    assert receipt["full_eval_parts"][0]["sha256"] == builder.sha256_file(
        tmp_path / "full" / "full_eval_part.jsonl"
    )
    selected_rows = [
        json.loads(line)
        for line in (output_root / "webdataset_lengths.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["split"] == "train" for row in selected_rows)
    assert {row[builder.CORRECTION_FIELD] for row in selected_rows} == {"en", "zh"}
    eval_keys = {str(row["utt_id"]) for row in rows if row["split"] == "eval"}
    assert {str(row["utt_id"]) for row in selected_rows}.isdisjoint(eval_keys)

    repeated = builder.build_correction_profile(
        full_profile_path=profile_path,
        output_root=output_root,
        seed=2114,
        bucket_width=80,
        entries_per_part=3,
    )
    assert repeated == receipt
    with pytest.raises(ValueError, match="seed differs"):
        builder.build_correction_profile(
            full_profile_path=profile_path,
            output_root=output_root,
            seed=2115,
            bucket_width=80,
            entries_per_part=3,
        )

    first_part = Path(str(receipt["bucket_output_parts"][0]["path"]))
    first_part.write_text(first_part.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        builder.validate_correction_profile(
            output_root / "stage211_sft_correction_profile.json",
            expected_full_profile_path=profile_path,
        )


def test_stage211_sft_correction_rejects_duplicate_source_ids(tmp_path: Path) -> None:
    _, _, _ = _write_full_profile_fixture(tmp_path / "full")
    index = tmp_path / "full" / "webdataset_lengths.jsonl"
    first = index.read_text(encoding="utf-8").splitlines()[0]
    index.write_text(index.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicated"):
        builder.select_correction_rows(index)
