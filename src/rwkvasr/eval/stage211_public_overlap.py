from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rwkvasr.eval.stage211_gate import (
    DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    sha256_file,
    validate_stage211_loaded_manifest_receipt,
)


STAGE211_PUBLIC_OVERLAP_SCHEMA_VERSION = 1


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


def _canonical_json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rendered = "".join(f"{_canonical_json_line(row)}\n" for row in rows)
    _atomic_write_text(path, rendered)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, rendered)


def _sha256_region(path: Path, *, offset: int, size: int) -> str:
    if offset < 0 or size <= 0:
        raise ValueError(f"Invalid audio region offset={offset} size={size}: {path}")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as source:
        source.seek(offset)
        while remaining:
            chunk = source.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Audio region ends early at offset={offset} size={size}: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _count_jsonl_rows(path: Path) -> int:
    with path.open("rb") as source:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: source.read(1024 * 1024), b""))


def _resolved_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is unavailable: {resolved}")
    return resolved


def _load_test_rows(path: Path) -> tuple[list[dict[str, str]], dict[tuple[str, str], list[str]]]:
    rows: list[dict[str, str]] = []
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"client_id", "path", "sentence_id", "sentence"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Common Voice test TSV lacks required columns: {path}")
        for line_number, row in enumerate(reader, start=2):
            client_id = str(row.get("client_id") or "").strip()
            sentence_id = str(row.get("sentence_id") or "").strip()
            utt_id = Path(str(row.get("path") or "").strip()).stem
            if not client_id or not sentence_id or not utt_id:
                raise ValueError(f"Common Voice test TSV row {line_number} lacks identity fields.")
            record = {key: str(value or "") for key, value in row.items() if key is not None}
            record["utt_id"] = utt_id
            rows.append(record)
            by_pair[(client_id, sentence_id)].append(utt_id)
    return rows, dict(by_pair)


def _load_public_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _load_jsonl(path, label="Common Voice public manifest")
    by_utt_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        utt_id = str(row.get("utt_id") or "").strip()
        if not utt_id or utt_id in by_utt_id:
            raise ValueError(
                f"Common Voice public manifest has an invalid/duplicate utt_id: {utt_id}"
            )
        audio_path = _resolved_file(
            str(row.get("audio_filepath") or ""),
            label=f"Common Voice public audio {utt_id}",
        )
        normalized = dict(row)
        normalized["audio_filepath"] = str(audio_path)
        by_utt_id[utt_id] = normalized
    return rows, by_utt_id


def _validate_public_test_identity(
    test_rows: list[dict[str, str]],
    public_by_utt_id: dict[str, dict[str, Any]],
) -> None:
    test_ids = [row["utt_id"] for row in test_rows]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Common Voice test TSV contains duplicate audio path stems.")
    if set(test_ids) != set(public_by_utt_id):
        missing = len(set(test_ids) - set(public_by_utt_id))
        extra = len(set(public_by_utt_id) - set(test_ids))
        raise ValueError(
            "Common Voice test TSV/public manifest identity mismatch: "
            f"missing={missing} extra={extra}."
        )


def _validate_loaded_source_binding(
    *,
    loaded_manifest_receipt_path: Path,
    stage178_index_path: Path,
) -> dict[str, Any]:
    loaded_receipt = validate_stage211_loaded_manifest_receipt(loaded_manifest_receipt_path)
    global_manifest_path = _resolved_file(
        str(loaded_receipt.get("global_dedup_manifest_path") or ""),
        label="Stage211 global-dedup manifest",
    )
    global_manifest = _load_json_object(
        global_manifest_path,
        label="Stage211 global-dedup manifest",
    )
    inputs = global_manifest.get("inputs")
    cv22 = inputs.get("cv22") if isinstance(inputs, dict) else None
    if not isinstance(cv22, dict):
        raise ValueError("Stage211 global-dedup manifest lacks the CV22 source binding.")
    recorded_index = Path(str(cv22.get("length_index_path") or "")).expanduser().resolve()
    if recorded_index != stage178_index_path:
        raise ValueError("Stage211 CV22 source index does not match the overlap audit input.")
    expected = {
        "rows_scanned": 112_603,
        "split_rows": 112_603,
        "accepted_unique_audio_rows": 112_603,
        "duplicate_audio_rows": 0,
    }
    if any(int(cv22.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("Stage211 CV22 global-dedup coverage is incomplete or changed.")
    source_counts = cv22.get("accepted_counts_by_source")
    if not isinstance(source_counts, dict) or int(source_counts.get("cv22_en", -1)) != 63_625:
        raise ValueError("Stage211 CV22 English accepted-row count changed.")
    return {
        "loaded_manifest_receipt_path": str(loaded_manifest_receipt_path),
        "loaded_manifest_receipt_sha256": sha256_file(loaded_manifest_receipt_path),
        "global_dedup_manifest_path": str(global_manifest_path),
        "global_dedup_manifest_sha256": sha256_file(global_manifest_path),
        "cv22_rows_scanned": int(cv22["rows_scanned"]),
        "cv22_accepted_unique_audio_rows": int(cv22["accepted_unique_audio_rows"]),
        "cv22_english_accepted_rows": int(source_counts["cv22_en"]),
    }


def create_stage211_public_overlap_audit(
    *,
    stage178_index: str | Path,
    commonvoice_test_tsv: str | Path,
    public_manifest: str | Path,
    converter_source: str | Path,
    output_dir: str | Path,
    loaded_manifest_receipt: str | Path | None = DEFAULT_STAGE211_LOADED_MANIFEST_RECEIPT,
    validate_loaded_binding: bool = True,
    expected_stage178_english_rows: int | None = None,
    expected_public_rows: int | None = None,
    expected_candidate_rows: int | None = None,
) -> dict[str, Any]:
    stage178_index_path = _resolved_file(stage178_index, label="Stage178 CV22 index")
    test_tsv_path = _resolved_file(commonvoice_test_tsv, label="Common Voice test TSV")
    public_manifest_path = _resolved_file(public_manifest, label="Common Voice public manifest")
    converter_source_path = _resolved_file(converter_source, label="CV22 converter source")
    output_dir_path = Path(output_dir).expanduser().resolve()

    source_binding: dict[str, Any] | None = None
    loaded_receipt_path: Path | None = None
    if loaded_manifest_receipt is not None:
        loaded_receipt_path = _resolved_file(
            loaded_manifest_receipt,
            label="Stage211 loaded-manifest receipt",
        )
        if validate_loaded_binding:
            source_binding = _validate_loaded_source_binding(
                loaded_manifest_receipt_path=loaded_receipt_path,
                stage178_index_path=stage178_index_path,
            )
        else:
            source_binding = {
                "loaded_manifest_receipt_path": str(loaded_receipt_path),
                "loaded_manifest_receipt_sha256": sha256_file(loaded_receipt_path),
            }

    test_rows, test_ids_by_pair = _load_test_rows(test_tsv_path)
    public_rows, public_by_utt_id = _load_public_manifest(public_manifest_path)
    _validate_public_test_identity(test_rows, public_by_utt_id)

    public_audio_hashes: dict[str, str] = {}
    candidate_rows: list[dict[str, Any]] = []
    stage178_rows = 0
    stage178_english_rows = 0
    with stage178_index_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stage178_rows += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Stage178 row {line_number} is not a JSON object.")
            if str(row.get("_stage178_source") or "") != "commonvoice_en":
                continue
            stage178_english_rows += 1
            pair = (
                str(row.get("cv22_client_id") or "").strip(),
                str(row.get("cv22_sentence_id") or "").strip(),
            )
            matching_test_ids = test_ids_by_pair.get(pair)
            if not matching_test_ids:
                continue
            tar_path = _resolved_file(
                str(row.get("tar_path") or ""),
                label=f"Stage178 candidate tar row {line_number}",
            )
            offset = int(row.get("audio_offset", -1))
            size = int(row.get("audio_size", -1))
            candidate_rows.append(
                {
                    "audio_member": str(row.get("audio_member") or ""),
                    "audio_offset": offset,
                    "audio_size": size,
                    "client_id": pair[0],
                    "sentence_id": pair[1],
                    "stage178_line_number": line_number,
                    "stage178_utt_id": str(row.get("utt_id") or row.get("key") or ""),
                    "tar_path": str(tar_path),
                    "test_utt_ids": sorted(matching_test_ids),
                }
            )

    if expected_stage178_english_rows is not None and (
        stage178_english_rows != expected_stage178_english_rows
    ):
        raise ValueError(
            "Stage178 Common Voice English count mismatch: "
            f"expected={expected_stage178_english_rows} actual={stage178_english_rows}."
        )
    if expected_public_rows is not None and len(public_rows) != expected_public_rows:
        raise ValueError(
            f"Common Voice public count mismatch: expected={expected_public_rows} "
            f"actual={len(public_rows)}."
        )
    if expected_candidate_rows is not None and len(candidate_rows) != expected_candidate_rows:
        raise ValueError(
            f"Common Voice overlap candidate count mismatch: expected={expected_candidate_rows} "
            f"actual={len(candidate_rows)}."
        )

    exact_matches_by_public_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    compared_candidates: list[dict[str, Any]] = []
    tar_paths_touched: set[str] = set()
    training_bytes_hashed = 0
    for candidate in sorted(
        candidate_rows,
        key=lambda row: (str(row["tar_path"]), int(row["audio_offset"])),
    ):
        tar_path = Path(str(candidate["tar_path"]))
        training_sha256 = _sha256_region(
            tar_path,
            offset=int(candidate["audio_offset"]),
            size=int(candidate["audio_size"]),
        )
        tar_paths_touched.add(str(tar_path))
        training_bytes_hashed += int(candidate["audio_size"])
        comparisons: list[dict[str, Any]] = []
        for public_utt_id in candidate["test_utt_ids"]:
            public_row = public_by_utt_id[public_utt_id]
            public_audio_path = Path(str(public_row["audio_filepath"]))
            public_sha256 = public_audio_hashes.setdefault(
                public_utt_id,
                sha256_file(public_audio_path),
            )
            byte_identical = training_sha256 == public_sha256
            comparison = {
                "byte_identical": byte_identical,
                "public_audio_path": str(public_audio_path),
                "public_audio_sha256": public_sha256,
                "public_utt_id": public_utt_id,
            }
            comparisons.append(comparison)
            if byte_identical:
                exact_matches_by_public_id[public_utt_id].append(
                    {
                        "audio_member": candidate["audio_member"],
                        "audio_offset": candidate["audio_offset"],
                        "audio_size": candidate["audio_size"],
                        "stage178_line_number": candidate["stage178_line_number"],
                        "stage178_utt_id": candidate["stage178_utt_id"],
                        "tar_path": candidate["tar_path"],
                        "training_audio_sha256": training_sha256,
                    }
                )
        compared = dict(candidate)
        compared["comparisons"] = comparisons
        compared["training_audio_sha256"] = training_sha256
        compared["byte_identical_public_ids"] = sorted(
            comparison["public_utt_id"]
            for comparison in comparisons
            if comparison["byte_identical"]
        )
        compared_candidates.append(compared)

    exclusion_rows: list[dict[str, Any]] = []
    for public_utt_id in sorted(exact_matches_by_public_id):
        public_row = public_by_utt_id[public_utt_id]
        exclusion_rows.append(
            {
                "classification": "byte_identical_stage211_training_recording",
                "public_audio_path": str(public_row["audio_filepath"]),
                "public_audio_sha256": public_audio_hashes[public_utt_id],
                "public_utt_id": public_utt_id,
                "training_matches": sorted(
                    exact_matches_by_public_id[public_utt_id],
                    key=lambda row: (
                        str(row["tar_path"]),
                        int(row["audio_offset"]),
                    ),
                ),
            }
        )

    excluded_ids = set(exact_matches_by_public_id)
    clean_manifest_rows = [
        row for row in public_rows if str(row.get("utt_id") or "") not in excluded_ids
    ]
    candidate_output = output_dir_path / "commonvoice_en_test_overlap_candidates.jsonl"
    exclusion_output = output_dir_path / "commonvoice_en_test_exclusions.jsonl"
    clean_manifest_output = output_dir_path / "commonvoice_en_test.clean.jsonl"
    receipt_output = output_dir_path / "receipt.json"
    _write_jsonl(candidate_output, compared_candidates)
    _write_jsonl(exclusion_output, exclusion_rows)
    _write_jsonl(clean_manifest_output, clean_manifest_rows)

    exact_training_rows = sum(1 for row in compared_candidates if row["byte_identical_public_ids"])
    receipt: dict[str, Any] = {
        "artifact": "stage211_public_train_overlap_audit",
        "audio_hash_algorithm": "sha256_exact_encoded_mp3_bytes",
        "complete": True,
        "coverage": {
            "candidate_pair_count": len(
                {(row["client_id"], row["sentence_id"]) for row in candidate_rows}
            ),
            "candidate_training_rows": len(candidate_rows),
            "clean_public_rows": len(clean_manifest_rows),
            "excluded_public_rows": len(exclusion_rows),
            "exact_byte_identical_training_rows": exact_training_rows,
            "public_rows": len(public_rows),
            "stage178_english_rows": stage178_english_rows,
            "stage178_rows": stage178_rows,
            "tar_files_touched": len(tar_paths_touched),
            "test_pair_count": len(test_ids_by_pair),
            "test_tsv_rows": len(test_rows),
            "training_bytes_hashed": training_bytes_hashed,
        },
        "dataset": "commonvoice_en_test",
        "outputs": {
            "candidate_rows": {
                "path": str(candidate_output),
                "rows": len(compared_candidates),
                "sha256": sha256_file(candidate_output),
            },
            "clean_manifest": {
                "path": str(clean_manifest_output),
                "rows": len(clean_manifest_rows),
                "sha256": sha256_file(clean_manifest_output),
            },
            "exclusions": {
                "path": str(exclusion_output),
                "rows": len(exclusion_rows),
                "sha256": sha256_file(exclusion_output),
            },
        },
        "pipeline": "stage211",
        "schema_version": STAGE211_PUBLIC_OVERLAP_SCHEMA_VERSION,
        "source_bindings": {
            "commonvoice_test_tsv_path": str(test_tsv_path),
            "commonvoice_test_tsv_sha256": sha256_file(test_tsv_path),
            "converter_source_path": str(converter_source_path),
            "converter_source_sha256": sha256_file(converter_source_path),
            "loaded_manifest": source_binding,
            "public_manifest_path": str(public_manifest_path),
            "public_manifest_sha256": sha256_file(public_manifest_path),
            "stage178_index_path": str(stage178_index_path),
            "stage178_index_sha256": sha256_file(stage178_index_path),
        },
    }
    _write_json(receipt_output, receipt)
    return receipt


def validate_stage211_public_overlap_receipt(
    receipt_path: str | Path,
    *,
    expected_public_manifest: str | Path | None = None,
    expected_loaded_manifest_receipt: str | Path | None = None,
) -> dict[str, Any]:
    path = _resolved_file(receipt_path, label="Stage211 public/train overlap receipt")
    receipt = _load_json_object(path, label="Stage211 public/train overlap receipt")
    expected_fields = {
        "artifact": "stage211_public_train_overlap_audit",
        "audio_hash_algorithm": "sha256_exact_encoded_mp3_bytes",
        "complete": True,
        "dataset": "commonvoice_en_test",
        "pipeline": "stage211",
        "schema_version": STAGE211_PUBLIC_OVERLAP_SCHEMA_VERSION,
    }
    if any(receipt.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Stage211 public/train overlap receipt contract mismatch.")

    source_bindings = receipt.get("source_bindings")
    if not isinstance(source_bindings, dict):
        raise ValueError("Stage211 public/train overlap receipt lacks source bindings.")
    source_pairs = (
        ("commonvoice_test_tsv_path", "commonvoice_test_tsv_sha256"),
        ("converter_source_path", "converter_source_sha256"),
        ("public_manifest_path", "public_manifest_sha256"),
        ("stage178_index_path", "stage178_index_sha256"),
    )
    for path_key, sha_key in source_pairs:
        source_path = _resolved_file(str(source_bindings.get(path_key) or ""), label=path_key)
        if str(source_bindings.get(sha_key) or "") != sha256_file(source_path):
            raise ValueError(f"Stage211 public/train overlap source changed: {source_path}")
    public_manifest_path = Path(str(source_bindings["public_manifest_path"])).resolve()
    if (
        expected_public_manifest is not None
        and public_manifest_path != Path(expected_public_manifest).expanduser().resolve()
    ):
        raise ValueError("Stage211 public/train overlap public manifest path mismatch.")

    loaded = source_bindings.get("loaded_manifest")
    if expected_loaded_manifest_receipt is not None:
        if not isinstance(loaded, dict):
            raise ValueError("Stage211 public/train overlap receipt lacks loaded-manifest proof.")
        expected_loaded = Path(expected_loaded_manifest_receipt).expanduser().resolve()
        recorded_loaded = Path(str(loaded.get("loaded_manifest_receipt_path") or "")).resolve()
        if recorded_loaded != expected_loaded or str(
            loaded.get("loaded_manifest_receipt_sha256") or ""
        ) != sha256_file(expected_loaded):
            raise ValueError("Stage211 public/train overlap loaded-manifest binding mismatch.")

    outputs = receipt.get("outputs")
    coverage = receipt.get("coverage")
    if not isinstance(outputs, dict) or not isinstance(coverage, dict):
        raise ValueError("Stage211 public/train overlap receipt lacks outputs/coverage.")
    output_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("candidate_rows", "clean_manifest", "exclusions"):
        record = outputs.get(name)
        if not isinstance(record, dict):
            raise ValueError(f"Stage211 public/train overlap receipt lacks output {name}.")
        output_path = _resolved_file(str(record.get("path") or ""), label=name)
        if str(record.get("sha256") or "") != sha256_file(output_path):
            raise ValueError(f"Stage211 public/train overlap output changed: {output_path}")
        if int(record.get("rows", -1)) != _count_jsonl_rows(output_path):
            raise ValueError(
                f"Stage211 public/train overlap output row count changed: {output_path}"
            )
        output_rows[name] = _load_jsonl(output_path, label=name)

    public_rows = _load_jsonl(public_manifest_path, label="bound public manifest")
    public_ids = {str(row.get("utt_id") or "") for row in public_rows}
    clean_ids = {str(row.get("utt_id") or "") for row in output_rows["clean_manifest"]}
    excluded_ids = {str(row.get("public_utt_id") or "") for row in output_rows["exclusions"]}
    candidate_exact_ids = {
        str(public_utt_id)
        for row in output_rows["candidate_rows"]
        for public_utt_id in row.get("byte_identical_public_ids", [])
    }
    if not public_ids or "" in public_ids or "" in clean_ids or "" in excluded_ids:
        raise ValueError("Stage211 public/train overlap output contains invalid utterance IDs.")
    if clean_ids & excluded_ids or clean_ids | excluded_ids != public_ids:
        raise ValueError("Stage211 clean/excluded Common Voice IDs do not partition the source.")
    if excluded_ids != candidate_exact_ids:
        raise ValueError("Stage211 exclusion IDs do not match exact-byte candidate evidence.")
    expected_counts = {
        "candidate_training_rows": len(output_rows["candidate_rows"]),
        "clean_public_rows": len(clean_ids),
        "excluded_public_rows": len(excluded_ids),
        "public_rows": len(public_ids),
    }
    if any(int(coverage.get(key, -1)) != value for key, value in expected_counts.items()):
        raise ValueError("Stage211 public/train overlap coverage totals mismatch.")
    return receipt
