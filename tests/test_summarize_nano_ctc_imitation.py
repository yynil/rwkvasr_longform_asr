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
