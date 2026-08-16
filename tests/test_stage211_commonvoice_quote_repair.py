from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from rwkvasr.data.manifest import build_text_tokenizer
from rwkvasr.data.text_normalization import normalize_asr_text
from rwkvasr.eval.stage211_gate import STAGE211_PUBLIC_BENCHMARKS, sha256_file


quote_repair = importlib.import_module("scripts.prepare_stage211_commonvoice_quote_repair")
unicode_correction = importlib.import_module(
    "scripts.install_stage211_unicode_metric_correction"
)


def test_quote_repair_merges_restored_predictions_in_manifest_order() -> None:
    manifest_rows = [
        {"utt_id": "legacy", "text": 'The "legacy" row.'},
        {"utt_id": "restored", "text": "A restored row."},
    ]
    legacy = [{"utt_id": "legacy", "ref_text": "stale", "pred_text": "legacy"}]
    restored = [{"utt_id": "restored", "ref_text": "stale", "pred_text": "restored"}]

    repaired = quote_repair._repair_nano_rows(manifest_rows, legacy, restored)

    assert [row["utt_id"] for row in repaired] == ["legacy", "restored"]
    assert [row["ref_text"] for row in repaired] == [
        'The "legacy" row.',
        "A restored row.",
    ]


def test_quote_repair_rebuilds_calibration_ctc_references() -> None:
    tokenizer_model = quote_repair.DEFAULT_TOKENIZER_MODEL
    manifest_rows = [
        {"utt_id": "legacy", "text": 'Hello, "WORLD"!'},
        {"utt_id": "restored", "text": "A restored row."},
    ]
    legacy = [
        {
            "utt_id": "legacy",
            "ref_text": "stale",
            "ref_token_ids": [1],
            "pred_text": "legacy",
            "debug": {"ref_token_count": 1},
        }
    ]
    restored = [
        {
            "utt_id": "restored",
            "ref_text": "stale",
            "ref_token_ids": [1],
            "pred_text": "restored",
            "debug": {"ref_token_count": 1},
        }
    ]

    repaired = quote_repair._repair_calibration_rows(
        manifest_rows,
        legacy,
        restored,
        tokenizer_model=tokenizer_model,
    )

    tokenizer = build_text_tokenizer("sensevoice_tiktoken", model_path=str(tokenizer_model))
    for manifest, row in zip(manifest_rows, repaired, strict=True):
        expected = normalize_asr_text(str(manifest["text"]), language="en", mode="ctc")
        assert row["ref_text"] == expected
        assert tokenizer.decode(row["ref_token_ids"]) == expected
        assert row["debug"]["ref_token_count"] == len(row["ref_token_ids"])


def test_unicode_correction_prior_install_requires_v2_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlap_receipt = tmp_path / "overlap.json"
    overlap_receipt.write_text("{}\n", encoding="utf-8")
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = manifest_dir / "commonvoice_en_test.jsonl"
    manifest.write_text('{"utt_id":"sample"}\n', encoding="utf-8")
    derivation = tmp_path / "derivation.json"
    derivation.write_text(
        json.dumps(
            {
                "total_samples": sum(
                    int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values()
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )
    installed_files = [{"path": str(manifest), "sha256": sha256_file(manifest)}]
    for index in range(11):
        artifact = tmp_path / f"artifact-{index}.json"
        artifact.write_text(f"{index}\n", encoding="utf-8")
        installed_files.append({"path": str(artifact), "sha256": sha256_file(artifact)})
    receipt = tmp_path / "canonical-install.json"
    payload = {
        "pipeline": "stage211",
        "artifact": "stage211_clean_public_canonical_install",
        "complete": True,
        "overlap_receipt_path": str(overlap_receipt),
        "overlap_receipt_sha256": sha256_file(overlap_receipt),
        "derivation_receipt_path": str(derivation),
        "derivation_receipt_sha256": sha256_file(derivation),
        "commonvoice_clean_metrics": {
            "sample_count": int(STAGE211_PUBLIC_BENCHMARKS["commonvoice_en_test"]["samples"])
        },
        "installed_files": installed_files,
    }
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        unicode_correction,
        "DEFAULT_STAGE211_PUBLIC_OVERLAP_RECEIPT",
        overlap_receipt,
    )
    monkeypatch.setattr(
        unicode_correction,
        "STAGE211_PUBLIC_OVERLAP_RECEIPT_SHA256",
        sha256_file(overlap_receipt),
    )
    monkeypatch.setattr(unicode_correction, "DEFAULT_MANIFEST_DIR", manifest_dir)
    monkeypatch.setattr(
        unicode_correction,
        "STAGE211_CLEAN_COMMONVOICE_MANIFEST_SHA256",
        sha256_file(manifest),
    )

    assert unicode_correction._validate_prior_install(receipt)["complete"] is True

    payload["overlap_receipt_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical-install receipt is invalid"):
        unicode_correction._validate_prior_install(receipt)
