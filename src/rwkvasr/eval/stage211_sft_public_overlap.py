from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from rwkvasr.eval.stage211_gate import sha256_file


SCHEMA_VERSION = 1
ARTIFACT = "stage211_sft_public_encoded_overlap_audit"
COMPARISON_MODE = "exact_encoded_audio_bytes"
HASH_ALGORITHM = "sha256_exact_encoded_audio_bytes"
EXPECTED_DATASETS = (
    "aishell1_test",
    "librispeech_test_clean",
    "librispeech_test_other",
    "commonvoice_en_test",
    "wenetspeech_test_net",
)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _iter_jsonl(path: Path, *, label: str) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"{label} is unavailable: {path}")
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not a JSON object.")
            yield line_number, value


def _immutable_write(path: Path, payload: bytes) -> None:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to replace a different immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _checked_binding(path_value: Any, sha_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is unavailable: {path}")
    if sha256_file(path) != str(sha_value or ""):
        raise ValueError(f"{label} changed: {path}")
    return path


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def _load_profile_binding(profile_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = profile_path.expanduser().resolve()
    profile = _load_json_object(profile_path, label="Stage211D labeled profile")
    expected_contract = {
        "artifact": "stage211_labeled_profile_receipt",
        "complete": True,
        "phase": "sft",
        "pipeline": "stage211",
        "schema_version": 2,
    }
    if any(profile.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Stage211D labeled profile contract mismatch.")
    root = Path(str(profile.get("labeled_webdataset_root") or "")).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Stage211D labeled WebDataset root is unavailable: {root}")
    length_index = _checked_binding(
        profile.get("length_index_path"),
        profile.get("length_index_sha256"),
        label="Stage211D labeled length index",
    )
    bucket_manifest = _checked_binding(
        profile.get("bucket_manifest_path"),
        profile.get("bucket_manifest_sha256"),
        label="Stage211D labeled bucket manifest",
    )
    expected = profile.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("Stage211D labeled profile lacks expected coverage.")
    required_counts = ("total_samples", "train_samples", "eval_samples", "source_counts")
    if any(key not in expected for key in required_counts):
        raise ValueError("Stage211D labeled profile coverage is incomplete.")
    binding = {
        "bucket_manifest_path": str(bucket_manifest),
        "bucket_manifest_sha256": sha256_file(bucket_manifest),
        "labeled_profile_path": str(profile_path),
        "labeled_profile_sha256": sha256_file(profile_path),
        "labeled_webdataset_root": str(root),
        "length_index_path": str(length_index),
        "length_index_sha256": sha256_file(length_index),
    }
    rebuild_path = root / "public_overlap_exclusion_rebuild_receipt.json"
    if rebuild_path.is_file():
        from rwkvasr.eval.stage211_sft_public_clean import (
            validate_stage211_sft_public_clean_rebuild_receipt,
        )

        rebuild = validate_stage211_sft_public_clean_rebuild_receipt(rebuild_path)
        if (
            Path(str(rebuild.get("output_root") or "")).resolve() != root
            or Path(str(rebuild.get("output_profile_path") or "")).resolve()
            != profile_path
            or rebuild.get("output_profile_sha256") != sha256_file(profile_path)
        ):
            raise ValueError("Stage211D public-clean rebuild uses a different labeled profile.")
        source_profile_path = Path(str(rebuild.get("source_profile_path") or "")).resolve()
        source_profile = _load_json_object(
            source_profile_path,
            label="Stage211D public-clean source profile",
        )
        source_root = Path(
            str(source_profile.get("labeled_webdataset_root") or "")
        ).resolve()
        if not source_root.is_dir():
            raise ValueError("Stage211D public-clean source root is unavailable.")
        binding.update(
            {
                "public_clean_rebuild_receipt_path": str(rebuild_path),
                "public_clean_rebuild_receipt_sha256": sha256_file(rebuild_path),
                "public_clean_source_webdataset_root": str(source_root),
            }
        )
    return profile, binding


def _load_public_fingerprints(
    *,
    nano_provenance_path: Path,
    public_pcm_audit_path: Path,
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    nano_provenance_path = nano_provenance_path.expanduser().resolve()
    nano = _load_json_object(nano_provenance_path, label="Stage211 Nano public provenance")
    if (
        nano.get("artifact") != "nano_public_baseline_provenance"
        or nano.get("complete") is not True
        or nano.get("pipeline") != "stage211"
    ):
        raise ValueError("Stage211 Nano public provenance contract mismatch.")
    nano_results = nano.get("results")
    if not isinstance(nano_results, list):
        raise ValueError("Stage211 Nano public provenance lacks result rows.")
    nano_by_dataset: dict[str, dict[str, Any]] = {}
    nano_ids: dict[str, set[str]] = {}
    for result in nano_results:
        if not isinstance(result, dict):
            raise ValueError("Stage211 Nano public result must be an object.")
        dataset = str(result.get("dataset") or "")
        if dataset in nano_by_dataset:
            raise ValueError(f"Duplicate Stage211 Nano public dataset: {dataset}")
        manifest = _checked_binding(
            result.get("manifest_path"),
            result.get("manifest_sha256"),
            label=f"Stage211 public manifest {dataset}",
        )
        ids: set[str] = set()
        manifest_rows = 0
        for _, row in _iter_jsonl(manifest, label=f"Stage211 public manifest {dataset}"):
            manifest_rows += 1
            utt_id = str(row.get("utt_id") or "")
            if not utt_id or utt_id in ids:
                raise ValueError(f"Stage211 public manifest identity changed: {dataset}")
            ids.add(utt_id)
        if manifest_rows != int(result.get("sample_count", -1)):
            raise ValueError(f"Stage211 public manifest coverage changed: {dataset}")
        nano_by_dataset[dataset] = result
        nano_ids[dataset] = ids
    if tuple(nano_by_dataset) != EXPECTED_DATASETS:
        raise ValueError("Stage211 Nano public dataset order or coverage changed.")
    if sum(len(ids) for ids in nano_ids.values()) != int(nano.get("total_samples", -1)):
        raise ValueError("Stage211 Nano public total coverage changed.")

    public_pcm_audit_path = public_pcm_audit_path.expanduser().resolve()
    pcm_audit = _load_json_object(public_pcm_audit_path, label="Stage211 public PCM audit")
    if (
        pcm_audit.get("artifact") != "stage211_base_public_pcm_overlap_audit"
        or pcm_audit.get("complete") is not True
        or pcm_audit.get("pipeline") != "stage211"
        or pcm_audit.get("public_overlap_rows") != 0
    ):
        raise ValueError("Stage211 public PCM audit is not a clean complete source.")
    receipts = pcm_audit.get("public_fingerprint_receipts")
    if not isinstance(receipts, list):
        raise ValueError("Stage211 public PCM audit lacks fingerprint receipts.")
    receipt_by_dataset = {
        str(record.get("dataset") or ""): record
        for record in receipts
        if isinstance(record, dict)
    }
    if tuple(sorted(receipt_by_dataset)) != tuple(sorted(EXPECTED_DATASETS)):
        raise ValueError("Stage211 public PCM fingerprint dataset coverage changed.")

    by_size: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    public_rows_by_dataset: dict[str, int] = {}
    fingerprint_bindings: list[dict[str, Any]] = []
    for dataset in EXPECTED_DATASETS:
        record = receipt_by_dataset[dataset]
        receipt_path = _checked_binding(
            record.get("path"),
            record.get("sha256"),
            label=f"Stage211 public fingerprint receipt {dataset}",
        )
        part_path = _checked_binding(
            record.get("part_path"),
            record.get("part_sha256"),
            label=f"Stage211 public fingerprint part {dataset}",
        )
        ids: set[str] = set()
        rows = 0
        for line_number, row in _iter_jsonl(
            part_path,
            label=f"Stage211 public fingerprint part {dataset}",
        ):
            if str(row.get("dataset") or "") != dataset:
                raise ValueError(
                    f"Stage211 public fingerprint dataset mismatch at {dataset}:{line_number}."
                )
            utt_id = str(row.get("utt_id") or "")
            audio_sha256 = str(row.get("audio_sha256") or "")
            audio_size = int(row.get("audio_size_bytes", -1))
            if not utt_id or not _is_sha256(audio_sha256) or audio_size <= 0:
                raise ValueError(
                    f"Stage211 public fingerprint row is malformed at {dataset}:{line_number}."
                )
            if utt_id in ids:
                raise ValueError(f"Duplicate Stage211 public fingerprint ID: {dataset}/{utt_id}")
            ids.add(utt_id)
            by_size[audio_size][audio_sha256].append(
                {"dataset": dataset, "public_utt_id": utt_id}
            )
            rows += 1
        if rows != int(record.get("rows", -1)) or ids != nano_ids[dataset]:
            raise ValueError(f"Stage211 public fingerprint identity changed: {dataset}")
        public_rows_by_dataset[dataset] = rows
        fingerprint_bindings.append(
            {
                "dataset": dataset,
                "fingerprint_receipt_path": str(receipt_path),
                "fingerprint_receipt_sha256": sha256_file(receipt_path),
                "fingerprint_part_path": str(part_path),
                "fingerprint_part_sha256": sha256_file(part_path),
                "rows": rows,
            }
        )
    return dict(by_size), {
        "nano_public_provenance_path": str(nano_provenance_path),
        "nano_public_provenance_sha256": sha256_file(nano_provenance_path),
        "public_pcm_audit_path": str(public_pcm_audit_path),
        "public_pcm_audit_sha256": sha256_file(public_pcm_audit_path),
        "public_fingerprint_receipts": fingerprint_bindings,
        "public_rows": sum(public_rows_by_dataset.values()),
        "public_rows_by_dataset": public_rows_by_dataset,
    }


def _source_name(row: Mapping[str, Any]) -> str:
    source = str(row.get("source_dataset") or "").strip()
    if source:
        return source
    shard_name = Path(str(row.get("shard_name") or "")).name
    return shard_name.split("_", 1)[0]


def _read_region(handle: BinaryIO, *, offset: int, size: int, path: Path) -> bytes:
    handle.seek(offset)
    payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"Short audio payload read from {path}: {len(payload)} != {size}")
    return payload


def _resolve_labeled_shard(
    *,
    root: Path,
    shard_relative: Path,
    trusted_source_root: Path | None,
    line_number: int,
) -> Path:
    shard_entry = root / shard_relative
    shard_path = shard_entry.resolve()
    if shard_path.is_relative_to(root):
        return shard_path
    if trusted_source_root is None:
        raise ValueError(f"Stage211D row {line_number} escapes its labeled root.")
    trusted_shard = (trusted_source_root / shard_relative).resolve()
    if shard_path != trusted_shard:
        raise ValueError(
            f"Stage211D row {line_number} uses an unbound external shard target."
        )
    return shard_path


def build_stage211_sft_public_overlap_audit(
    *,
    labeled_profile_path: Path,
    nano_provenance_path: Path,
    public_pcm_audit_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    profile, labeled_binding = _load_profile_binding(labeled_profile_path)
    public_by_size, public_binding = _load_public_fingerprints(
        nano_provenance_path=nano_provenance_path,
        public_pcm_audit_path=public_pcm_audit_path,
    )
    root = Path(labeled_binding["labeled_webdataset_root"])
    length_index = Path(labeled_binding["length_index_path"])
    profile_expected = dict(profile["expected"])
    trusted_source_root_value = labeled_binding.get("public_clean_source_webdataset_root")
    trusted_source_root = (
        Path(str(trusted_source_root_value)).resolve()
        if trusted_source_root_value is not None
        else None
    )

    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    candidate_split_counts: Counter[str] = Counter()
    candidate_source_counts: Counter[str] = Counter()
    overlap_split_counts: Counter[str] = Counter()
    overlap_source_counts: Counter[str] = Counter()
    overlap_pair_split_counts: Counter[str] = Counter()
    overlap_public_counts: Counter[str] = Counter()
    matches: list[dict[str, Any]] = []
    candidate_rows = 0
    candidate_audio_bytes = 0
    scanned_rows = 0
    handles: dict[Path, BinaryIO] = {}
    shard_sizes: dict[Path, int] = {}
    try:
        for line_number, row in _iter_jsonl(length_index, label="Stage211D length index"):
            scanned_rows += 1
            split = str(row.get("split") or "")
            if split not in {"train", "eval"}:
                raise ValueError(f"Stage211D row {line_number} has unsupported split {split!r}.")
            source_name = _source_name(row)
            if not source_name:
                raise ValueError(f"Stage211D row {line_number} lacks a source dataset.")
            split_counts[split] += 1
            source_counts[source_name] += 1

            shard_name = str(row.get("shard_name") or "")
            shard_relative = Path(shard_name)
            if (
                not shard_name
                or shard_relative.is_absolute()
                or ".." in shard_relative.parts
            ):
                raise ValueError(f"Stage211D row {line_number} escapes its labeled root.")
            shard_path = _resolve_labeled_shard(
                root=root,
                shard_relative=shard_relative,
                trusted_source_root=trusted_source_root,
                line_number=line_number,
            )
            audio_offset = int(row.get("audio_offset", -1))
            audio_size = int(row.get("audio_size", -1))
            if audio_offset < 0 or audio_size <= 0 or not shard_path.is_file():
                raise ValueError(f"Stage211D row {line_number} has invalid audio storage metadata.")
            shard_size = shard_sizes.setdefault(shard_path, shard_path.stat().st_size)
            if audio_offset + audio_size > shard_size:
                raise ValueError(f"Stage211D row {line_number} escapes its shard payload bounds.")
            public_hashes = public_by_size.get(audio_size)
            if public_hashes is None:
                continue
            candidate_rows += 1
            candidate_audio_bytes += audio_size
            candidate_split_counts[split] += 1
            candidate_source_counts[source_name] += 1
            handle = handles.get(shard_path)
            if handle is None:
                handle = shard_path.open("rb")
                handles[shard_path] = handle
            payload = _read_region(
                handle,
                offset=audio_offset,
                size=audio_size,
                path=shard_path,
            )
            audio_sha256 = hashlib.sha256(payload).hexdigest()
            public_matches = public_hashes.get(audio_sha256)
            if not public_matches:
                continue
            overlap_split_counts[split] += 1
            overlap_source_counts[source_name] += 1
            for public_match in public_matches:
                dataset = str(public_match["dataset"])
                overlap_pair_split_counts[split] += 1
                overlap_public_counts[dataset] += 1
                matches.append(
                    {
                        "audio_member": str(row.get("audio_member") or ""),
                        "audio_offset": audio_offset,
                        "audio_sha256": audio_sha256,
                        "audio_size": audio_size,
                        "dataset": dataset,
                        "key": str(row.get("key") or ""),
                        "length_index_line_number": line_number,
                        "public_utt_id": str(public_match["public_utt_id"]),
                        "shard_name": shard_name,
                        "source_dataset": source_name,
                        "split": split,
                        "utt_id": str(row.get("utt_id") or ""),
                    }
                )
    finally:
        for handle in handles.values():
            handle.close()

    expected_total = int(profile_expected["total_samples"])
    expected_split_counts = {
        "eval": int(profile_expected["eval_samples"]),
        "train": int(profile_expected["train_samples"]),
    }
    expected_source_counts = {
        str(key): int(value)
        for key, value in dict(profile_expected["source_counts"]).items()
    }
    if scanned_rows != expected_total or dict(sorted(split_counts.items())) != expected_split_counts:
        raise ValueError("Stage211D public-overlap scan split coverage changed.")
    if dict(sorted(source_counts.items())) != expected_source_counts:
        raise ValueError("Stage211D public-overlap scan source coverage changed.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matches_path = output_dir / "exact_encoded_audio_matches.jsonl"
    _immutable_write(matches_path, _jsonl_bytes(matches))
    training_overlap_rows = int(overlap_split_counts.get("train", 0))
    internal_eval_overlap_rows = int(overlap_split_counts.get("eval", 0))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "stage211",
        "artifact": ARTIFACT,
        "complete": True,
        "training_ready": training_overlap_rows == 0,
        "admission_state": (
            "exact_encoded_public_clear"
            if training_overlap_rows == 0
            else "public_overlap_detected"
        ),
        "comparison_mode": COMPARISON_MODE,
        "hash_algorithm": HASH_ALGORITHM,
        "size_prefilter_lossless_for_exact_bytes": True,
        "normalized_pcm_complete": False,
        "near_duplicate_complete": False,
        "labeled_profile": labeled_binding,
        "public_benchmark": public_binding,
        "coverage": {
            "scanned_rows": scanned_rows,
            "split_counts": dict(sorted(split_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "candidate_rows": candidate_rows,
            "candidate_audio_bytes": candidate_audio_bytes,
            "candidate_split_counts": dict(sorted(candidate_split_counts.items())),
            "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
            "opened_shards": len(handles),
        },
        "overlap": {
            "training_rows": training_overlap_rows,
            "internal_eval_rows": internal_eval_overlap_rows,
            "pair_count": len(matches),
            "rows_by_split": dict(sorted(overlap_split_counts.items())),
            "rows_by_source": dict(sorted(overlap_source_counts.items())),
            "pairs_by_split": dict(sorted(overlap_pair_split_counts.items())),
            "pairs_by_public_dataset": dict(sorted(overlap_public_counts.items())),
        },
        "matches_output": {
            "path": str(matches_path),
            "rows": len(matches),
            "size_bytes": matches_path.stat().st_size,
            "sha256": sha256_file(matches_path),
        },
    }
    receipt_path = output_dir / "receipt.json"
    _immutable_write(receipt_path, _json_bytes(receipt))
    return validate_stage211_sft_public_overlap_receipt(
        receipt_path,
        expected_labeled_profile=labeled_profile_path,
        expected_nano_provenance=nano_provenance_path,
        expected_public_pcm_audit=public_pcm_audit_path,
        require_training_ready=False,
    )


def validate_stage211_sft_public_overlap_receipt(
    receipt_path: Path,
    *,
    expected_labeled_profile: Path | None = None,
    expected_nano_provenance: Path | None = None,
    expected_public_pcm_audit: Path | None = None,
    require_training_ready: bool = True,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _load_json_object(receipt_path, label="Stage211D public-overlap receipt")
    expected_contract = {
        "artifact": ARTIFACT,
        "comparison_mode": COMPARISON_MODE,
        "complete": True,
        "hash_algorithm": HASH_ALGORITHM,
        "near_duplicate_complete": False,
        "normalized_pcm_complete": False,
        "pipeline": "stage211",
        "schema_version": SCHEMA_VERSION,
        "size_prefilter_lossless_for_exact_bytes": True,
    }
    if any(receipt.get(key) != value for key, value in expected_contract.items()):
        raise ValueError("Stage211D public-overlap receipt contract mismatch.")

    labeled = receipt.get("labeled_profile")
    public = receipt.get("public_benchmark")
    coverage = receipt.get("coverage")
    overlap = receipt.get("overlap")
    output = receipt.get("matches_output")
    if not all(isinstance(value, dict) for value in (labeled, public, coverage, overlap, output)):
        raise ValueError("Stage211D public-overlap receipt is structurally incomplete.")
    assert isinstance(labeled, dict)
    assert isinstance(public, dict)
    assert isinstance(coverage, dict)
    assert isinstance(overlap, dict)
    assert isinstance(output, dict)

    profile_path = _checked_binding(
        labeled.get("labeled_profile_path"),
        labeled.get("labeled_profile_sha256"),
        label="Stage211D overlap labeled profile",
    )
    profile, rebuilt_labeled = _load_profile_binding(profile_path)
    if labeled != rebuilt_labeled:
        raise ValueError("Stage211D overlap labeled-profile binding changed.")
    if expected_labeled_profile is not None and profile_path != expected_labeled_profile.resolve():
        raise ValueError("Stage211D overlap audit uses a different labeled profile.")

    nano_path = _checked_binding(
        public.get("nano_public_provenance_path"),
        public.get("nano_public_provenance_sha256"),
        label="Stage211D overlap Nano public provenance",
    )
    pcm_path = _checked_binding(
        public.get("public_pcm_audit_path"),
        public.get("public_pcm_audit_sha256"),
        label="Stage211D overlap public PCM audit",
    )
    if expected_nano_provenance is not None and nano_path != expected_nano_provenance.resolve():
        raise ValueError("Stage211D overlap audit uses different Nano public provenance.")
    if expected_public_pcm_audit is not None and pcm_path != expected_public_pcm_audit.resolve():
        raise ValueError("Stage211D overlap audit uses a different public PCM audit.")
    public_by_size, rebuilt_public = _load_public_fingerprints(
        nano_provenance_path=nano_path,
        public_pcm_audit_path=pcm_path,
    )
    if public != rebuilt_public:
        raise ValueError("Stage211D overlap public-benchmark binding changed.")

    expected = dict(profile["expected"])
    expected_split_counts = {
        "eval": int(expected["eval_samples"]),
        "train": int(expected["train_samples"]),
    }
    expected_source_counts = {
        str(key): int(value) for key, value in dict(expected["source_counts"]).items()
    }
    candidate_rows = int(coverage.get("candidate_rows", -1))
    candidate_audio_bytes = int(coverage.get("candidate_audio_bytes", -1))
    opened_shards = int(coverage.get("opened_shards", -1))
    candidate_split_counts = coverage.get("candidate_split_counts")
    candidate_source_counts = coverage.get("candidate_source_counts")
    candidate_counts_valid = (
        isinstance(candidate_split_counts, dict)
        and isinstance(candidate_source_counts, dict)
        and set(candidate_split_counts).issubset(expected_split_counts)
        and set(candidate_source_counts).issubset(expected_source_counts)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (*candidate_split_counts.values(), *candidate_source_counts.values())
        )
        and sum(candidate_split_counts.values()) == candidate_rows
        and sum(candidate_source_counts.values()) == candidate_rows
    )
    if (
        int(coverage.get("scanned_rows", -1)) != int(expected["total_samples"])
        or coverage.get("split_counts") != expected_split_counts
        or coverage.get("source_counts") != expected_source_counts
        or not 0 <= candidate_rows <= int(expected["total_samples"])
        or candidate_audio_bytes < candidate_rows
        or not candidate_counts_valid
        or not 0 <= opened_shards <= candidate_rows
        or (candidate_rows == 0) != (opened_shards == 0)
    ):
        raise ValueError("Stage211D public-overlap coverage changed.")

    matches_path = _checked_binding(
        output.get("path"),
        output.get("sha256"),
        label="Stage211D exact encoded-audio matches",
    ) if int(output.get("size_bytes", -1)) > 0 else Path(str(output.get("path") or "")).resolve()
    if not matches_path.is_file():
        raise ValueError("Stage211D exact encoded-audio matches output is unavailable.")
    if matches_path.stat().st_size != int(output.get("size_bytes", -1)):
        raise ValueError("Stage211D exact encoded-audio matches output size changed.")
    if sha256_file(matches_path) != str(output.get("sha256") or ""):
        raise ValueError("Stage211D exact encoded-audio matches output changed.")
    replay_pair_split_counts: Counter[str] = Counter()
    replay_dataset_counts: Counter[str] = Counter()
    replay_unique_rows: dict[tuple[str, int, str], tuple[str, str]] = {}
    replay_pairs: set[tuple[tuple[str, int, str], str, str]] = set()
    replay_rows = 0
    for _, row in _iter_jsonl(matches_path, label="Stage211D exact encoded-audio matches"):
        replay_rows += 1
        split = str(row.get("split") or "")
        source_name = str(row.get("source_dataset") or "")
        dataset = str(row.get("dataset") or "")
        public_utt_id = str(row.get("public_utt_id") or "")
        audio_sha256 = str(row.get("audio_sha256") or "")
        audio_size = int(row.get("audio_size", -1))
        if (
            split not in expected_split_counts
            or source_name not in expected_source_counts
            or dataset not in EXPECTED_DATASETS
            or not public_utt_id
            or not _is_sha256(audio_sha256)
            or audio_size <= 0
        ):
            raise ValueError("Stage211D overlap match row is malformed.")
        expected_public_matches = public_by_size.get(audio_size, {}).get(audio_sha256, [])
        if {"dataset": dataset, "public_utt_id": public_utt_id} not in expected_public_matches:
            raise ValueError("Stage211D overlap match is absent from the public fingerprints.")
        replay_pair_split_counts[split] += 1
        replay_dataset_counts[dataset] += 1
        identity = (
            str(row.get("shard_name") or ""),
            int(row.get("audio_offset", -1)),
            str(row.get("key") or ""),
        )
        if not identity[0] or identity[1] < 0 or not identity[2]:
            raise ValueError("Stage211D overlap match row identity is malformed.")
        pair = (identity, dataset, public_utt_id)
        if pair in replay_pairs:
            raise ValueError("Stage211D overlap match pair is duplicated.")
        replay_pairs.add(pair)
        prior = replay_unique_rows.setdefault(identity, (split, source_name))
        if prior != (split, source_name):
            raise ValueError("Stage211D overlap match row identity is inconsistent.")
    replay_split_counts = Counter(split for split, _ in replay_unique_rows.values())
    replay_source_counts = Counter(source for _, source in replay_unique_rows.values())
    if (
        replay_rows != int(output.get("rows", -1))
        or replay_rows != int(overlap.get("pair_count", -1))
        or dict(sorted(replay_split_counts.items())) != overlap.get("rows_by_split")
        or dict(sorted(replay_source_counts.items())) != overlap.get("rows_by_source")
        or dict(sorted(replay_pair_split_counts.items())) != overlap.get("pairs_by_split")
        or dict(sorted(replay_dataset_counts.items()))
        != overlap.get("pairs_by_public_dataset")
        or int(overlap.get("training_rows", -1)) != replay_split_counts.get("train", 0)
        or int(overlap.get("internal_eval_rows", -1)) != replay_split_counts.get("eval", 0)
    ):
        raise ValueError("Stage211D exact encoded-audio overlap replay changed.")
    training_ready = int(overlap["training_rows"]) == 0
    if receipt.get("training_ready") is not training_ready:
        raise ValueError("Stage211D public-overlap training-ready decision changed.")
    expected_state = "exact_encoded_public_clear" if training_ready else "public_overlap_detected"
    if receipt.get("admission_state") != expected_state:
        raise ValueError("Stage211D public-overlap admission state changed.")
    if require_training_ready and not training_ready:
        raise ValueError("Stage211D labeled training audio overlaps the public benchmark.")
    return receipt
