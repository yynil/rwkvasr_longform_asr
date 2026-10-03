from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from rwkvasr.data.manifest import build_text_tokenizer
from rwkvasr.data.text_normalization import normalize_asr_text
from rwkvasr.eval import compute_text_error_stats
from rwkvasr.eval.stage211_gate import STAGE211_PUBLIC_BENCHMARKS, sha256_file
from rwkvasr.eval.stage211_public_metrics import (
    STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
)
from rwkvasr.eval.stage211_public_overlap import validate_stage211_public_overlap_receipt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAP_RECEIPT = (
    Path.home()
    / "rwkvasr_data"
    / "stage211_full_curriculum"
    / "public_train_overlap_v2"
    / "receipt.json"
)
DEFAULT_LEGACY_ARCHIVE = Path.home() / "rwkvasr_eval" / "stage211_public_contaminated_archive_v1"
DEFAULT_ACTIVE_MANIFEST_DIR = REPO_ROOT / "artifacts" / "eval_benchmarks" / "manifests"
DEFAULT_ACTIVE_NANO_ROOT = Path.home() / "rwkvasr_eval" / "stage211_public_full" / "nano_2512"
DEFAULT_ACTIVE_CALIBRATION_PUBLIC = (
    Path.home() / "rwkvasr_eval" / "stage211_calibration_selected_full" / "public"
)
DEFAULT_RESTORED_INFERENCE_ROOT = (
    Path.home() / "rwkvasr_eval" / "stage211_public_quote_repair_v2" / "inference"
)
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "rwkvasr_eval" / "stage211_public_quote_repair_v2" / "parent_exact"
)
DEFAULT_TOKENIZER_MODEL = REPO_ROOT / "assets" / "fun-asr-nano-2512" / "multilingual.tiktoken"
EXPECTED_FULL_ROWS = 16_401
EXPECTED_CLEAN_ROWS = 14_927
EXPECTED_LEGACY_ROWS = 16_396
EXPECTED_RESTORED_IDS = frozenset(
    {
        "common_voice_en_117312",
        "common_voice_en_154952",
        "common_voice_en_23428991",
        "common_voice_en_26240905",
        "common_voice_en_33006819",
    }
)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} is not an object.")
            utt_id = str(row.get("utt_id") or "").strip()
            if not utt_id or utt_id in seen:
                raise ValueError(f"{label} has invalid/duplicate utt_id={utt_id!r}.")
            seen.add(utt_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _by_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row["utt_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{label} has duplicate utterance IDs.")
    return indexed


def _copy_unchanged_public_files(
    *,
    output_root: Path,
    active_manifest_dir: Path,
    active_nano_root: Path,
    active_calibration_public: Path,
) -> None:
    for dataset in STAGE211_PUBLIC_BENCHMARKS:
        if dataset == "commonvoice_en_test":
            continue
        copies = (
            (
                active_manifest_dir / f"{dataset}.jsonl",
                output_root / "manifests" / f"{dataset}.jsonl",
            ),
            (
                active_nano_root / "predictions" / f"{dataset}.ctc.jsonl",
                output_root / "nano_2512" / "predictions" / f"{dataset}.ctc.jsonl",
            ),
            (
                active_nano_root / "reports" / f"{dataset}.json",
                output_root / "nano_2512" / "reports" / f"{dataset}.json",
            ),
            (
                active_calibration_public / "predictions" / f"{dataset}.ctc.jsonl",
                output_root / "calibration" / "predictions" / f"{dataset}.ctc.jsonl",
            ),
        )
        for source, destination in copies:
            if not source.is_file() or source.stat().st_size <= 0:
                raise ValueError(f"Stage211 unchanged public artifact is missing: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def _repair_nano_rows(
    manifest_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    restored_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = _by_id(legacy_rows + restored_rows, label="Nano repaired predictions")
    manifest_ids = {str(row["utt_id"]) for row in manifest_rows}
    if set(predictions) != manifest_ids:
        raise ValueError("Nano repaired prediction coverage differs from the corrected manifest.")
    repaired: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        utt_id = str(manifest_row["utt_id"])
        row = dict(predictions[utt_id])
        row["ref_text"] = str(manifest_row["text"])
        repaired.append(row)
    return repaired


def _repair_calibration_rows(
    manifest_rows: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    restored_rows: list[dict[str, Any]],
    *,
    tokenizer_model: Path,
) -> list[dict[str, Any]]:
    predictions = _by_id(legacy_rows + restored_rows, label="calibration repaired predictions")
    manifest_ids = {str(row["utt_id"]) for row in manifest_rows}
    if set(predictions) != manifest_ids:
        raise ValueError(
            "Calibration repaired prediction coverage differs from the corrected manifest."
        )
    tokenizer = build_text_tokenizer("sensevoice_tiktoken", model_path=str(tokenizer_model))
    repaired: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        utt_id = str(manifest_row["utt_id"])
        normalized = normalize_asr_text(
            str(manifest_row["text"]),
            language="en",
            mode="ctc",
        )
        ref_token_ids = [int(token_id) for token_id in tokenizer.encode(normalized)]
        ref_text = str(tokenizer.decode(ref_token_ids))
        if ref_text != normalized:
            raise ValueError(f"Calibration reference does not round trip for {utt_id}.")
        row = dict(predictions[utt_id])
        row["ref_text"] = ref_text
        row["ref_token_ids"] = ref_token_ids
        debug = row.get("debug")
        if isinstance(debug, dict):
            row["debug"] = {**debug, "ref_token_count": len(ref_token_ids)}
        repaired.append(row)
    return repaired


def _output_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_prepared_bundle(receipt_path: Path) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json(receipt_path, label="Stage211 Common Voice quote-repair receipt")
    expected = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stage211_commonvoice_tsv_quote_repair_parent",
        "complete": True,
        "full_rows": EXPECTED_FULL_ROWS,
        "clean_rows": EXPECTED_CLEAN_ROWS,
        "legacy_rows": EXPECTED_LEGACY_ROWS,
        "restored_ids": sorted(EXPECTED_RESTORED_IDS),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage211 Common Voice quote-repair receipt contract changed.")
    for group in ("sources", "outputs"):
        bindings = receipt.get(group)
        if not isinstance(bindings, dict) or not bindings:
            raise ValueError(f"Stage211 quote-repair receipt lacks {group} bindings.")
        for label, binding in bindings.items():
            if not isinstance(binding, dict):
                raise ValueError(f"Stage211 quote-repair {group} binding is invalid: {label}")
            path = Path(str(binding.get("path") or "")).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != int(binding.get("size_bytes", -1))
                or sha256_file(path) != binding.get("sha256")
            ):
                raise ValueError(f"Stage211 quote-repair binding changed: {path}")
    return receipt


def prepare_bundle(
    *,
    overlap_receipt: Path,
    legacy_archive: Path,
    active_manifest_dir: Path,
    active_nano_root: Path,
    active_calibration_public: Path,
    restored_inference_root: Path,
    output_root: Path,
    tokenizer_model: Path,
) -> dict[str, Any]:
    paths = [
        overlap_receipt,
        legacy_archive,
        active_manifest_dir,
        active_nano_root,
        active_calibration_public,
        restored_inference_root,
        output_root,
        tokenizer_model,
    ]
    (
        overlap_receipt,
        legacy_archive,
        active_manifest_dir,
        active_nano_root,
        active_calibration_public,
        restored_inference_root,
        output_root,
        tokenizer_model,
    ) = [path.expanduser().resolve() for path in paths]
    receipt_path = output_root / "quote_repair_parent_receipt.json"
    if receipt_path.is_file():
        return validate_prepared_bundle(receipt_path)

    overlap = validate_stage211_public_overlap_receipt(overlap_receipt)
    coverage = overlap["coverage"]
    if (
        int(coverage["public_rows"]) != EXPECTED_FULL_ROWS
        or int(coverage["clean_public_rows"]) != EXPECTED_CLEAN_ROWS
    ):
        raise ValueError("Stage211 corrected Common Voice overlap coverage is unexpected.")
    full_manifest = Path(overlap["source_bindings"]["public_manifest_path"]).resolve()
    full_rows = _load_jsonl(full_manifest, label="corrected full Common Voice manifest")
    if len(full_rows) != EXPECTED_FULL_ROWS:
        raise ValueError("Stage211 corrected Common Voice full-manifest count changed.")

    legacy_manifest = legacy_archive / "manifest.jsonl"
    legacy_manifest_rows = _load_jsonl(legacy_manifest, label="legacy Common Voice manifest")
    restored_ids = {str(row["utt_id"]) for row in full_rows} - {
        str(row["utt_id"]) for row in legacy_manifest_rows
    }
    if len(legacy_manifest_rows) != EXPECTED_LEGACY_ROWS or restored_ids != EXPECTED_RESTORED_IDS:
        raise ValueError("Stage211 Common Voice restored-ID proof changed.")

    legacy_nano = legacy_archive / "nano" / "commonvoice_en_test.ctc.jsonl"
    legacy_calibration = legacy_archive / "calibration" / "commonvoice_en_test.ctc.jsonl"
    restored_nano = restored_inference_root / "nano" / "commonvoice_en_test.ctc.jsonl"
    restored_nano_report = restored_inference_root / "nano" / "commonvoice_en_test.json"
    restored_calibration = restored_inference_root / "calibration" / "commonvoice_en_test.ctc.jsonl"
    restored_nano_rows = _load_jsonl(restored_nano, label="restored Nano predictions")
    restored_calibration_rows = _load_jsonl(
        restored_calibration,
        label="restored calibration predictions",
    )
    if {str(row["utt_id"]) for row in restored_nano_rows} != EXPECTED_RESTORED_IDS or {
        str(row["utt_id"]) for row in restored_calibration_rows
    } != EXPECTED_RESTORED_IDS:
        raise ValueError("Stage211 restored inference does not cover the exact five IDs.")
    nano_report = _load_json(restored_nano_report, label="restored Nano report")
    if int(nano_report.get("sample_count", -1)) != len(EXPECTED_RESTORED_IDS):
        raise ValueError("Stage211 restored Nano report coverage changed.")

    _copy_unchanged_public_files(
        output_root=output_root,
        active_manifest_dir=active_manifest_dir,
        active_nano_root=active_nano_root,
        active_calibration_public=active_calibration_public,
    )
    parent_manifest = output_root / "manifests" / "commonvoice_en_test.jsonl"
    parent_nano = output_root / "nano_2512" / "predictions" / "commonvoice_en_test.ctc.jsonl"
    parent_calibration = (
        output_root / "calibration" / "predictions" / "commonvoice_en_test.ctc.jsonl"
    )
    parent_report = output_root / "nano_2512" / "reports" / "commonvoice_en_test.json"
    _atomic_copy(full_manifest, parent_manifest)
    repaired_nano = _repair_nano_rows(
        full_rows,
        _load_jsonl(legacy_nano, label="legacy Nano predictions"),
        restored_nano_rows,
    )
    repaired_calibration = _repair_calibration_rows(
        full_rows,
        _load_jsonl(legacy_calibration, label="legacy calibration predictions"),
        restored_calibration_rows,
        tokenizer_model=tokenizer_model,
    )
    _write_jsonl(parent_nano, repaired_nano)
    _write_jsonl(parent_calibration, repaired_calibration)

    report = _load_json(
        legacy_archive / "nano" / "commonvoice_en_test.report.json",
        label="legacy Nano Common Voice report",
    )
    metrics = compute_text_error_stats(
        parent_nano,
        language="en",
        normalization="ctc",
        strip_language_confirmation=STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
    )
    report.update(
        {
            "manifest_path": str(parent_manifest.resolve()),
            "model_checkpoint_path": str(nano_report["model_checkpoint_path"]),
            "model_checkpoint_sha256": str(nano_report["model_checkpoint_sha256"]),
            "metrics": {
                "avg_cer": float(metrics["avg_cer"]),
                "avg_wer": float(metrics["avg_wer"]),
                "sample_count": EXPECTED_FULL_ROWS,
            },
            "predictions_path": str(parent_nano.resolve()),
            "requested_limit": None,
            "sample_count": EXPECTED_FULL_ROWS,
            "strip_language_confirmation": STAGE211_PUBLIC_STRIP_LANGUAGE_CONFIRMATION,
            "quote_repair": {
                "overlap_receipt_path": str(overlap_receipt),
                "overlap_receipt_sha256": sha256_file(overlap_receipt),
                "restored_ids": sorted(EXPECTED_RESTORED_IDS),
                "restored_nano_report_path": str(restored_nano_report),
                "restored_nano_report_sha256": sha256_file(restored_nano_report),
            },
        }
    )
    _write_json(parent_report, report)

    source_paths = {
        "overlap_receipt": overlap_receipt,
        "legacy_manifest": legacy_manifest,
        "legacy_nano_predictions": legacy_nano,
        "legacy_calibration_predictions": legacy_calibration,
        "restored_nano_predictions": restored_nano,
        "restored_nano_report": restored_nano_report,
        "restored_calibration_predictions": restored_calibration,
        "tokenizer_model": tokenizer_model,
    }
    output_paths = {
        "full_manifest": parent_manifest,
        "full_nano_predictions": parent_nano,
        "full_nano_report": parent_report,
        "full_calibration_predictions": parent_calibration,
    }
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "stage211_commonvoice_tsv_quote_repair_parent",
        "complete": True,
        "full_rows": EXPECTED_FULL_ROWS,
        "clean_rows": EXPECTED_CLEAN_ROWS,
        "legacy_rows": EXPECTED_LEGACY_ROWS,
        "restored_ids": sorted(EXPECTED_RESTORED_IDS),
        "sources": {label: _output_binding(path) for label, path in source_paths.items()},
        "outputs": {label: _output_binding(path) for label, path in output_paths.items()},
    }
    _write_json(receipt_path, receipt)
    return validate_prepared_bundle(receipt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a hash-bound Stage211 Common Voice literal-TSV repair parent."
    )
    parser.add_argument("--overlap-receipt", type=Path, default=DEFAULT_OVERLAP_RECEIPT)
    parser.add_argument("--legacy-archive", type=Path, default=DEFAULT_LEGACY_ARCHIVE)
    parser.add_argument("--active-manifest-dir", type=Path, default=DEFAULT_ACTIVE_MANIFEST_DIR)
    parser.add_argument("--active-nano-root", type=Path, default=DEFAULT_ACTIVE_NANO_ROOT)
    parser.add_argument(
        "--active-calibration-public",
        type=Path,
        default=DEFAULT_ACTIVE_CALIBRATION_PUBLIC,
    )
    parser.add_argument(
        "--restored-inference-root",
        type=Path,
        default=DEFAULT_RESTORED_INFERENCE_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-model", type=Path, default=DEFAULT_TOKENIZER_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = prepare_bundle(
        overlap_receipt=args.overlap_receipt,
        legacy_archive=args.legacy_archive,
        active_manifest_dir=args.active_manifest_dir,
        active_nano_root=args.active_nano_root,
        active_calibration_public=args.active_calibration_public,
        restored_inference_root=args.restored_inference_root,
        output_root=args.output_root,
        tokenizer_model=args.tokenizer_model,
    )
    print(
        "[stage211-commonvoice-quote-repair] "
        f"full_rows={receipt['full_rows']} clean_rows={receipt['clean_rows']} "
        f"restored={len(receipt['restored_ids'])} "
        f"receipt={args.output_root.expanduser().resolve() / 'quote_repair_parent_receipt.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
