from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
summary = importlib.import_module("scripts.summarize_nano_ctc_imitation")


def test_load_teacher_accepts_native_nano_eval_prediction_fields(tmp_path: Path) -> None:
    path = tmp_path / "nano.jsonl"
    path.write_text(
        json.dumps(
            {
                "utt_id": "utt-a",
                "pred_text": "hello world",
                "pred_token_ids": [1, 2, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = summary._load_teacher(
        path,
        teacher_text_key="funasr_ctc_text",
        teacher_token_key="funasr_ctc_token_ids",
    )

    assert rows["utt-a"]["text"] == "hello world"
    assert rows["utt-a"]["tokens"] == [1, 2, 3]


def test_accumulator_reports_token_edit_decomposition() -> None:
    accumulator = summary.Accumulator()
    common = {
        "pred_text": "",
        "teacher_text": "",
        "language": "en",
        "normalization": "ctc",
        "student_blank_prob": None,
        "teacher_blank_prob": None,
        "student_blank_top1": None,
        "teacher_blank_top1": None,
    }
    accumulator.add(
        **common,
        pred_tokens=[1, 4, 3, 5],
        teacher_tokens=[1, 2, 3],
    )
    accumulator.add(
        **common,
        pred_tokens=[8],
        teacher_tokens=[7, 8],
    )

    stats = accumulator.as_dict()

    assert stats["token_insertions"] == 1
    assert stats["token_deletions"] == 1
    assert stats["token_substitutions"] == 1
    assert stats["token_errors"] == 3
    assert stats["nano_token_insertion_rate"] == 0.2
    assert stats["nano_token_deletion_rate"] == 0.2
    assert stats["nano_token_substitution_rate"] == 0.2
    assert stats["nano_token_er"] == 0.6
