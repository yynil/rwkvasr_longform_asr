from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from rwkvasr.eval.funasr_nano_ctc import (
    FunASRNanoCTCManifestEvalConfig,
    evaluate_funasr_nano_ctc_manifest,
)


class _FakeTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return {1: "hello", 2: "world", 3: "wrong"}.get(token_ids[0], "") + (
            " world" if token_ids == [1, 2] else ""
        )


class _FakeTeacher:
    def __init__(self) -> None:
        self.model = SimpleNamespace(ctc_tokenizer=_FakeTokenizer())
        self.sources: list[str] = []

    def topk_records(self, utt_ids, audio_rows):
        utt_id = str(utt_ids[0])
        self.sources.append(str(audio_rows[0]["source_dataset"]))
        token_ids = [1, 2] if utt_id == "u1" else [3]
        return {
            utt_id: {
                "argmax_token_ids": torch.tensor(token_ids),
                "topk_token_ids": torch.tensor([[60515, 1], [1, 60515]]),
                "blank_log_probs": torch.log(torch.tensor([0.8, 0.2])),
                "project_blank_id": 60515,
                "num_frames": 2,
            }
        }


def test_evaluate_funasr_nano_ctc_manifest_writes_normalized_report(tmp_path: Path) -> None:
    model_dir = tmp_path / "nano"
    model_dir.mkdir()
    model_checkpoint = model_dir / "model.pt"
    model_checkpoint.write_bytes(b"nano-checkpoint")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"utt_id": "u1", "audio_filepath": "one.wav", "text": "HELLO WORLD"},
        {"utt_id": "u2", "audio_filepath": "two.wav", "text": "wrong answer"},
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    predictions = tmp_path / "nano.jsonl"
    report_path = tmp_path / "nano.report.json"
    teacher = _FakeTeacher()

    report = evaluate_funasr_nano_ctc_manifest(
        FunASRNanoCTCManifestEvalConfig(
            manifest_path=str(manifest),
            model_path=str(model_dir),
            predictions_path=str(predictions),
            report_path=str(report_path),
            language="en",
            progress_interval=0,
        ),
        teacher=teacher,
    )

    assert teacher.sources == ["librispeech", "librispeech"]
    assert predictions.read_text(encoding="utf-8").count("\n") == 2
    assert report["sample_count"] == 2
    assert report["model_checkpoint_path"] == str(model_checkpoint.resolve())
    assert len(report["model_checkpoint_sha256"]) == 64
    assert report["metrics"]["avg_wer"] == 0.25
    assert report["diagnostics"]["pred_ref_unit_ratio"] == 0.75
    assert report["diagnostics"]["mean_blank_top1_ratio"] == 0.5
    assert json.loads(report_path.read_text(encoding="utf-8"))["decode"] == "greedy_ctc"
