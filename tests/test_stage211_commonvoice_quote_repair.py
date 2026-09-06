from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from rwkvasr.data.manifest import build_text_tokenizer
from rwkvasr.data.text_normalization import normalize_asr_text
from rwkvasr.eval.stage211_gate import STAGE211_PUBLIC_BENCHMARKS, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
quote_repair = importlib.import_module("scripts.prepare_stage211_commonvoice_quote_repair")
clean_install = importlib.import_module("scripts.install_stage211_clean_public_eval")
unicode_correction = importlib.import_module("scripts.install_stage211_unicode_metric_correction")


def test_quote_repair_handoff_validates_corrected_public_before_supplemental_publish() -> None:
    source = (REPO_ROOT / "scripts" / "build_stage211_quote_repair_handoff.sh").read_text(
        encoding="utf-8"
    )

    assert source.count("scripts/install_stage211_unicode_metric_correction.py") == 2
    clean_index = source.index("scripts/install_stage211_clean_public_eval.py")
    unicode_install_index = source.index("scripts/install_stage211_unicode_metric_correction.py")
    validation_index = source.index("  --validate-only \\\n")
    combined_index = source.index("scripts/build_stage211_combined_supplemental_inventory.py")
    assert clean_index < unicode_install_index < validation_index < combined_index
    clean_command = source[clean_index:unicode_install_index]
    for binding in (
        '--clean-root "${PUBLIC_CLEAN_ROOT}"',
        '--manifest-dir "${PUBLIC_MANIFEST_DIR}"',
        '--nano-root "${NANO_EVAL_ROOT}"',
        '--nano-checkpoint "${NANO_CHECKPOINT}"',
        '--calibration-root "${CALIBRATION_EVAL_ROOT}"',
        '--overlap-receipt "${PUBLIC_OVERLAP_RECEIPT}"',
    ):
        assert binding in clean_command
    for variable in (
        "PUBLIC_MANIFEST_DIR",
        "NANO_EVAL_ROOT",
        "CALIBRATION_EVAL_ROOT",
        "INITIALIZATION_RECEIPT",
        "NANO_CHECKPOINT",
        "NANO_BASELINE_RECEIPT",
    ):
        assert f"${{{variable}}}" in source
    for argument in (
        "--correction-receipt",
        "--prior-install-receipt",
        "--nano-root",
        "--calibration-root",
        "--manifest-dir",
        "--initialization-receipt",
        "--nano-checkpoint",
        "--nano-baseline-receipt",
    ):
        assert argument in source[validation_index:combined_index]


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
    for index in range(15):
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


def test_corrected_public_readiness_replays_complete_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "metric"
    correction_receipt = output_root / "custom-correction.json"
    correction_receipt.parent.mkdir(parents=True)
    correction_receipt.write_text("{}\n", encoding="utf-8")
    prior_install_receipt = tmp_path / "canonical-install.json"
    prior_install_receipt.write_text("{}\n", encoding="utf-8")
    nano_root = tmp_path / "nano"
    calibration_root = tmp_path / "calibration"
    calibration_public = calibration_root / "public"
    calibration_public.mkdir(parents=True)
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    initialization_receipt = tmp_path / "initialization.json"
    initialization_receipt.write_text("{}\n", encoding="utf-8")
    nano_checkpoint = tmp_path / "nano.pt"
    nano_checkpoint.write_bytes(b"nano")
    nano_baseline_receipt = nano_root / "provenance_receipt.json"
    nano_baseline_receipt.parent.mkdir(parents=True)
    nano_baseline_receipt.write_text("{}\n", encoding="utf-8")
    public_overlap_receipt = tmp_path / "public-overlap.json"
    public_overlap_receipt.write_text("{}\n", encoding="utf-8")

    benchmark = {
        "results": [
            {"dataset": dataset, "sample_count": int(expected["samples"])}
            for dataset, expected in STAGE211_PUBLIC_BENCHMARKS.items()
        ]
    }
    rebuilt_reuse = {
        "checkpoint_path": str(tmp_path / "calibration.pt"),
        "public_overlap": {"receipt_path": str(public_overlap_receipt)},
        "public_benchmark": benchmark,
    }
    calibration_reuse_receipt = calibration_public / "reuse_receipt.json"
    calibration_reuse_receipt.write_text(
        json.dumps(rebuilt_reuse, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nano_sha256 = sha256_file(nano_checkpoint)
    expected_installed_paths = unicode_correction._affected_paths(
        nano_root=nano_root,
        calibration_root=calibration_root,
        initialization_receipt=initialization_receipt,
    )
    correction = {
        "prior_clean_install_receipt_path": str(prior_install_receipt),
        "prior_clean_install_receipt_sha256": sha256_file(prior_install_receipt),
        "prior_clean_install_artifact": "stage211_clean_public_canonical_install",
        "nano_checkpoint_path": str(nano_checkpoint),
        "nano_checkpoint_sha256": nano_sha256,
        "calibration_reuse_receipt_sha256": sha256_file(calibration_reuse_receipt),
        "initialization_receipt_sha256": sha256_file(initialization_receipt),
        "installed_files": [
            {"label": label, "path": str(path)} for label, path in expected_installed_paths.items()
        ],
    }
    monkeypatch.setattr(
        unicode_correction,
        "_validate_prior_install",
        lambda path: {
            "artifact": "stage211_clean_public_canonical_install",
            "complete": True,
        },
    )
    monkeypatch.setattr(
        unicode_correction,
        "validate_completed_correction",
        lambda *args, **kwargs: correction,
    )
    monkeypatch.setattr(
        unicode_correction,
        "build_reuse_receipt",
        lambda **kwargs: rebuilt_reuse,
    )
    monkeypatch.setattr(
        unicode_correction,
        "validate_stage211_public_benchmark",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        unicode_correction,
        "validate_stage211_nano_public_baseline_receipt",
        lambda *args, **kwargs: {
            "nano_checkpoint_path": str(nano_checkpoint),
            "nano_checkpoint_sha256": nano_sha256,
        },
    )
    monkeypatch.setattr(
        unicode_correction,
        "validate_stage211_initialization_receipt",
        lambda *args, **kwargs: {
            "calibration_reuse_receipt_path": str(calibration_reuse_receipt),
            "calibration_reuse_receipt_sha256": sha256_file(calibration_reuse_receipt),
        },
    )
    immutable_inputs = (
        correction_receipt,
        prior_install_receipt,
        calibration_reuse_receipt,
        initialization_receipt,
        nano_checkpoint,
        nano_baseline_receipt,
        public_overlap_receipt,
    )
    before = {path: path.read_bytes() for path in immutable_inputs}

    readiness = unicode_correction.validate_corrected_public_readiness(
        output_root=output_root,
        correction_receipt=correction_receipt,
        prior_install_receipt=prior_install_receipt,
        nano_root=nano_root,
        calibration_root=calibration_root,
        manifest_dir=manifest_dir,
        initialization_receipt=initialization_receipt,
        nano_checkpoint=nano_checkpoint,
        nano_baseline_receipt=nano_baseline_receipt,
    )

    assert readiness["correction_receipt_path"] == str(correction_receipt)
    assert readiness["public_sample_count"] == sum(
        int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values()
    )
    assert {path: path.read_bytes() for path in immutable_inputs} == before


def _canonical_install_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    clean_root = tmp_path / "clean"
    manifest_dir = tmp_path / "canonical" / "manifests"
    nano_root = tmp_path / "canonical" / "nano"
    calibration_root = tmp_path / "canonical" / "calibration"
    archive_root = tmp_path / "archive"
    nano_checkpoint = tmp_path / "model" / "model.pt"
    nano_checkpoint.parent.mkdir(parents=True)
    nano_checkpoint.write_bytes(b"nano-checkpoint")
    overlap_receipt = tmp_path / "overlap.json"
    overlap_receipt.write_text("{}\n", encoding="utf-8")
    derivation_path = clean_root / "derivation_receipt.json"
    derivation_path.parent.mkdir(parents=True)
    derivation = {
        "total_samples": sum(int(row["samples"]) for row in STAGE211_PUBLIC_BENCHMARKS.values())
    }
    derivation_path.write_text(json.dumps(derivation) + "\n", encoding="utf-8")

    mapping = clean_install._canonical_archive_map(
        manifest_dir=manifest_dir,
        nano_root=nano_root,
        calibration_root=calibration_root,
        archive_root=archive_root,
    )
    archived_files = []
    for index, (canonical, archived) in enumerate(mapping.items()):
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(f"installed-{index}\n", encoding="utf-8")
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_text(f"archived-{index}\n", encoding="utf-8")
        archive_sha256 = sha256_file(archived)
        archived_files.append(
            {
                "archive_path": str(archived.resolve()),
                "archive_sha256": archive_sha256,
                "canonical_path": str(canonical.resolve()),
                "canonical_sha256": archive_sha256,
            }
        )
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        report = nano_root / "reports" / f"{dataset}.json"
        report.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "model_path": str(nano_checkpoint.parent),
                    "preserved": {"value": dataset},
                }
            )
            + "\n",
            encoding="utf-8",
        )
    identity = clean_install._bind_nano_report_checkpoint_identity(
        nano_root=nano_root,
        nano_checkpoint=nano_checkpoint,
    )
    installed_files = [
        {"path": str(canonical.resolve()), "sha256": sha256_file(canonical)}
        for canonical in mapping
    ]
    archive = {
        "artifact": "stage211_contaminated_public_archive",
        "complete": True,
        "files": archived_files,
        "pipeline": "stage211",
        "schema_version": 1,
    }
    archive_receipt = archive_root / "archive_receipt.json"
    archive_receipt.write_text(json.dumps(archive) + "\n", encoding="utf-8")
    receipt = {
        "archive_receipt_path": str(archive_receipt.resolve()),
        "archive_receipt_sha256": sha256_file(archive_receipt),
        "artifact": "stage211_clean_public_canonical_install",
        "commonvoice_clean_metrics": {
            "sample_count": int(STAGE211_PUBLIC_BENCHMARKS["commonvoice_en_test"]["samples"])
        },
        "complete": True,
        "derivation_receipt_path": str(derivation_path.resolve()),
        "derivation_receipt_sha256": sha256_file(derivation_path),
        "installed_files": installed_files,
        "nano_report_checkpoint_identity": identity,
        "overlap_receipt_path": str(overlap_receipt.resolve()),
        "overlap_receipt_sha256": sha256_file(overlap_receipt),
        "pipeline": "stage211",
        "schema_version": 1,
        "source_archive": archive,
    }
    receipt_path = clean_root / "canonical_install_receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        clean_install,
        "validate_stage211_public_overlap_receipt",
        lambda path: {},
    )
    monkeypatch.setattr(clean_install, "_validate_derivation", lambda *args: derivation)
    monkeypatch.setattr(
        clean_install,
        "validate_stage211_nano_public_baseline_receipt",
        lambda *args, **kwargs: {"provenance_mode": "embedded_checkpoint_sha256"},
    )
    return {
        "archive_root": archive_root,
        "calibration_root": calibration_root,
        "clean_root": clean_root,
        "manifest_dir": manifest_dir,
        "nano_root": nano_root,
        "nano_checkpoint": nano_checkpoint,
        "overlap_receipt": overlap_receipt,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def _validate_canonical_install_fixture(fixture: dict[str, object]) -> dict[str, object] | None:
    return clean_install._validate_existing_install(
        fixture["receipt_path"],
        clean_root=fixture["clean_root"],
        manifest_dir=fixture["manifest_dir"],
        nano_root=fixture["nano_root"],
        calibration_root=fixture["calibration_root"],
        archive_root=fixture["archive_root"],
        overlap_receipt=fixture["overlap_receipt"],
        nano_checkpoint=fixture["nano_checkpoint"],
    )


def test_nano_report_identity_binding_preserves_reports_and_rolls_back_all_five(
    tmp_path: Path,
) -> None:
    nano_root = tmp_path / "nano"
    checkpoint = tmp_path / "model" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"nano-checkpoint")
    originals: dict[Path, bytes] = {}
    archive_mapping: dict[Path, Path] = {}
    for index, dataset in enumerate(STAGE211_PUBLIC_BENCHMARKS):
        report_path = nano_root / "reports" / f"{dataset}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset,
            "model_path": str(checkpoint.parent),
            "nested": {"index": index, "text": "preserve-me"},
        }
        if dataset == "commonvoice_en_test":
            payload["model_checkpoint_path"] = str(checkpoint)
            payload["model_checkpoint_sha256"] = sha256_file(checkpoint)
        report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        originals[report_path] = report_path.read_bytes()
        archive_mapping[report_path] = tmp_path / "archive" / f"{dataset}.json"

    clean_install._archive_originals(archive_mapping, tmp_path / "archive")
    proof = clean_install._bind_nano_report_checkpoint_identity(
        nano_root=nano_root,
        nano_checkpoint=checkpoint,
    )

    assert proof["mode"] == "embedded_checkpoint_sha256"
    assert len(proof["reports"]) == len(STAGE211_PUBLIC_BENCHMARKS)
    clean_install._validate_nano_report_checkpoint_identity(
        proof,
        nano_root=nano_root,
        nano_checkpoint=checkpoint,
    )
    for index, dataset in enumerate(STAGE211_PUBLIC_BENCHMARKS):
        report = json.loads((nano_root / "reports" / f"{dataset}.json").read_text())
        assert report["nested"] == {"index": index, "text": "preserve-me"}
        assert report["model_checkpoint_path"] == str(checkpoint.resolve())
        assert report["model_checkpoint_sha256"] == sha256_file(checkpoint)

    clean_install._restore(archive_mapping)
    assert {path: path.read_bytes() for path in originals} == originals


def test_clean_public_install_exact_reuse_is_deeply_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _canonical_install_fixture(tmp_path, monkeypatch)

    assert _validate_canonical_install_fixture(fixture) == fixture["receipt"]

    receipt = dict(fixture["receipt"])
    installed = [dict(row) for row in receipt["installed_files"]]
    installed[-1] = dict(installed[0])
    receipt["installed_files"] = installed
    fixture["receipt_path"].write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="paths are duplicated"):
        _validate_canonical_install_fixture(fixture)


@pytest.mark.parametrize(
    "alternate",
    ("overlap_receipt", "clean_root", "manifest_dir", "archive_root"),
)
def test_clean_public_install_rejects_byte_identical_alternate_request_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alternate: str,
) -> None:
    fixture = _canonical_install_fixture(tmp_path, monkeypatch)
    requested = dict(fixture)
    original = fixture[alternate]
    assert isinstance(original, Path)
    replacement = tmp_path / "alternate" / alternate
    if original.is_file():
        replacement.parent.mkdir(parents=True, exist_ok=True)
        replacement.write_bytes(original.read_bytes())
    else:
        replacement.mkdir(parents=True)
        if alternate == "clean_root":
            derivation = original / "derivation_receipt.json"
            (replacement / "derivation_receipt.json").write_bytes(derivation.read_bytes())
    requested[alternate] = replacement

    with pytest.raises(ValueError, match="mismatch"):
        _validate_canonical_install_fixture(requested)


def test_clean_public_install_rejects_changed_embedded_archive_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _canonical_install_fixture(tmp_path, monkeypatch)
    receipt = dict(fixture["receipt"])
    embedded = dict(receipt["source_archive"])
    embedded["complete"] = False
    receipt["source_archive"] = embedded
    fixture["receipt_path"].write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="embedded archive receipt mismatch"):
        _validate_canonical_install_fixture(fixture)
