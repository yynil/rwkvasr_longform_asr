from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rwkvasr.config import load_yaml
from rwkvasr.data.manifest import build_text_tokenizer
from rwkvasr.eval.text_metrics import (
    _normalize_text_for_error_tokens,
    edit_counts,
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)


STAGE211_PUBLIC_MAX_RELATIVE_RATIO = 1.20
STAGE211_PUBLIC_MAX_ABSOLUTE_GAP_POINTS = 3.0
STAGE211_PUBLIC_MAX_PROGRESS_REGRESSION = 0.03
STAGE211_STUDENT_BLANK_ID = 60_515
STAGE211_STUDENT_TOKENIZER_VOCAB_SIZE = 60_515
STAGE211_STUDENT_CTC_VOCAB_SIZE = 60_516

_PUBLIC_LABELS = {
    "aishell1_test": "AISHELL-1 test",
    "librispeech_test_clean": "LibriSpeech test-clean",
    "librispeech_test_other": "LibriSpeech test-other",
    "commonvoice_en_test": "Common Voice 22 en test",
    "wenetspeech_test_net": "WenetSpeech TEST_NET",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bound_file(path_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Stage211 {label} is missing or empty: {path}")
    return path


def build_stage211_student_ctc_execution_provenance(
    *,
    checkpoint_path: Path,
    model_config_path: Path,
    tokenizer_config_path: Path,
    mode: str,
    beam_size: int,
    token_prune_topk: int | None,
    decoder_rescore_topk: int,
    blank_logit_bias: float,
    hotwords_path: str | Path | None,
    text_normalization: str,
    save_debug_lengths: bool,
) -> dict[str, Any]:
    checkpoint_path = _bound_file(checkpoint_path, label="student CTC execution checkpoint")
    model_config_path = _bound_file(
        model_config_path,
        label="student CTC execution model config",
    )
    tokenizer_config_path = _bound_file(
        tokenizer_config_path,
        label="student CTC execution tokenizer config",
    )
    if model_config_path != (checkpoint_path.parent / "model_config.yaml").resolve():
        raise ValueError("Stage211 student CTC execution must use the checkpoint-local model config.")
    if tokenizer_config_path != (checkpoint_path.parent / "tokenizer_config.yaml").resolve():
        raise ValueError(
            "Stage211 student CTC execution must use the checkpoint-local tokenizer config."
        )

    model_config = load_yaml(model_config_path)
    tokenizer_config = load_yaml(tokenizer_config_path)
    if not isinstance(model_config, dict) or not isinstance(tokenizer_config, dict):
        raise ValueError("Stage211 student CTC execution configs must be mappings.")
    blank_id = int(model_config.get("blank_id", -1))
    tokenizer_vocab_size = int(tokenizer_config.get("vocab_size", -1))
    model_vocab_size = int(model_config.get("vocab_size", -1))
    ctc_vocab_size = max(model_vocab_size, blank_id + 1)
    tokenizer_model_path = _bound_file(
        tokenizer_config.get("tokenizer_model_path"),
        label="student CTC tokenizer model",
    )
    ctc_decoder_type = str(model_config.get("ctc_decoder_type") or "").lower()
    suppressed_token_ids = tuple(
        sorted({int(token_id) for token_id in model_config.get("ctc_suppressed_token_ids", ())})
    )
    contract_valid = (
        blank_id == STAGE211_STUDENT_BLANK_ID
        and model_vocab_size == STAGE211_STUDENT_TOKENIZER_VOCAB_SIZE
        and tokenizer_vocab_size == STAGE211_STUDENT_TOKENIZER_VOCAB_SIZE
        and ctc_vocab_size == STAGE211_STUDENT_CTC_VOCAB_SIZE
        and tokenizer_config.get("tokenizer_type") == "sensevoice_tiktoken"
        and ctc_decoder_type in {"funasr_nano_transformer", "nano_transformer"}
        and int(model_config.get("ctc_decoder_num_layers", -1)) == 5
        and model_config.get("decoder_enabled") is False
        and str(mode) == "bi"
        and int(beam_size) == 1
        and int(token_prune_topk or 0) == 16
        and int(decoder_rescore_topk) == 0
        and math.isclose(float(blank_logit_bias), 0.0, rel_tol=0.0, abs_tol=0.0)
        and hotwords_path is None
        and str(text_normalization) == "ctc"
        and bool(save_debug_lengths)
    )
    if not contract_valid:
        raise ValueError(
            "Stage211 student CTC execution does not satisfy the strict checkpoint/tokenizer/"
            "greedy-decode contract."
        )
    if any(
        token_id < 0
        or token_id >= STAGE211_STUDENT_CTC_VOCAB_SIZE
        or token_id == STAGE211_STUDENT_BLANK_ID
        for token_id in suppressed_token_ids
    ):
        raise ValueError("Stage211 student CTC suppression IDs are invalid.")

    return {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "student_ctc_prediction_execution",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_config_path": str(model_config_path),
        "model_config_sha256": _sha256_file(model_config_path),
        "tokenizer_config_path": str(tokenizer_config_path),
        "tokenizer_config_sha256": _sha256_file(tokenizer_config_path),
        "tokenizer_type": "sensevoice_tiktoken",
        "tokenizer_model_path": str(tokenizer_model_path),
        "tokenizer_model_sha256": _sha256_file(tokenizer_model_path),
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "blank_id": blank_id,
        "ctc_vocab_size": ctc_vocab_size,
        "ctc_decoder_type": ctc_decoder_type,
        "ctc_decoder_num_layers": 5,
        "ctc_suppressed_token_ids_count": len(suppressed_token_ids),
        "ctc_suppressed_token_ids_sha256": _canonical_json_sha256(suppressed_token_ids),
        "mode": "bi",
        "decode_strategy": "ctc_greedy",
        "beam_size": 1,
        "token_prune_topk": 16,
        "decoder_rescore_topk": 0,
        "blank_logit_bias": 0.0,
        "hotwords_enabled": False,
        "text_normalization": "ctc",
        "save_debug_lengths": True,
        "llm_decoder_enabled": False,
        "ctc_only": True,
    }


def validate_stage211_student_ctc_execution_provenance(
    payload: Any,
    *,
    expected_checkpoint: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Stage211 student prediction lacks CTC execution provenance.")
    rebuilt = build_stage211_student_ctc_execution_provenance(
        checkpoint_path=expected_checkpoint,
        model_config_path=Path(str(payload.get("model_config_path") or "")),
        tokenizer_config_path=Path(str(payload.get("tokenizer_config_path") or "")),
        mode=str(payload.get("mode") or ""),
        beam_size=int(payload.get("beam_size", -1)),
        token_prune_topk=int(payload.get("token_prune_topk", -1)),
        decoder_rescore_topk=int(payload.get("decoder_rescore_topk", -1)),
        blank_logit_bias=float(payload.get("blank_logit_bias", float("nan"))),
        hotwords_path=("enabled" if payload.get("hotwords_enabled") else None),
        text_normalization=str(payload.get("text_normalization") or ""),
        save_debug_lengths=bool(payload.get("save_debug_lengths")),
    )
    if payload != rebuilt:
        raise ValueError(
            "Stage211 student CTC execution provenance does not match current checkpoint/config files."
        )
    return rebuilt


def _jsonl_references(
    path: Path,
    *,
    language: str,
    reference_keys: tuple[str, ...],
) -> dict[str, str]:
    references: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Stage211 public row is invalid at {path}:{line_number}")
            utt_id = str(row.get("utt_id") or row.get("id") or row.get("key") or "")
            if not utt_id:
                raise ValueError(f"Stage211 public row lacks an ID at {path}:{line_number}")
            if utt_id in references:
                raise ValueError(
                    f"Stage211 public row duplicates {utt_id!r} at {path}:{line_number}"
                )
            reference = next(
                (str(row[key]) for key in reference_keys if row.get(key) is not None),
                None,
            )
            if reference is None:
                raise ValueError(
                    f"Stage211 public row lacks a reference for {utt_id!r} at {path}:{line_number}"
                )
            references[utt_id] = normalize_asr_text_for_metrics(
                reference,
                language=language,
                normalization="ctc",
            )
    if not references:
        raise ValueError(f"Stage211 public JSONL is empty: {path}")
    return references


def _relative_ratio(*, nano_rate: float, student_rate: float) -> float:
    if nano_rate > 0.0:
        return student_rate / nano_rate
    return 1.0 if student_rate <= 0.0 else math.inf


def _jsonl_predictions(
    path: Path,
    *,
    language: str,
) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Stage211 public row is invalid at {path}:{line_number}")
            utt_id = str(row.get("utt_id") or "")
            if not utt_id:
                raise ValueError(f"Stage211 public row lacks an ID at {path}:{line_number}")
            if utt_id in records:
                raise ValueError(
                    f"Stage211 public row duplicates {utt_id!r} at {path}:{line_number}"
                )
            if row.get("ref_text") is None or row.get("pred_text") is None:
                raise ValueError(
                    f"Stage211 public prediction row is incomplete at {path}:{line_number}"
                )
            reference = normalize_asr_text_for_metrics(
                str(row["ref_text"]),
                language=language,
                normalization="ctc",
            )
            prediction = normalize_asr_text_for_metrics(
                str(row["pred_text"]),
                language=language,
                normalization="ctc",
            )
            records[utt_id] = (reference, prediction)
    if not records:
        raise ValueError(f"Stage211 public prediction JSONL is empty: {path}")
    return records


def _jsonl_student_ctc_predictions(
    path: Path,
    *,
    language: str,
    expected_checkpoint: Path,
    expected_execution_provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    records: dict[str, tuple[str, str]] = {}
    execution_provenance = expected_execution_provenance
    tokenizer: Any | None = None
    suppressed_token_ids: set[int] = set()
    decoded_token_ids: dict[tuple[int, ...], str] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Stage211 public row is invalid at {path}:{line_number}")
            utt_id = str(row.get("utt_id") or "")
            if not utt_id:
                raise ValueError(f"Stage211 public row lacks an ID at {path}:{line_number}")
            if utt_id in records:
                raise ValueError(
                    f"Stage211 public row duplicates {utt_id!r} at {path}:{line_number}"
                )
            if row.get("ref_text") is None or row.get("pred_text") is None:
                raise ValueError(
                    f"Stage211 public prediction row is incomplete at {path}:{line_number}"
                )
            if execution_provenance is None:
                execution_provenance = validate_stage211_student_ctc_execution_provenance(
                    row.get("inference_provenance"),
                    expected_checkpoint=expected_checkpoint,
                )
            elif row.get("inference_provenance") != execution_provenance:
                raise ValueError(
                    f"Stage211 student CTC execution provenance changed at {path}:{line_number}"
                )
            if tokenizer is None:
                tokenizer_config = load_yaml(Path(execution_provenance["tokenizer_config_path"]))
                model_config = load_yaml(Path(execution_provenance["model_config_path"]))
                tokenizer = build_text_tokenizer(
                    str(tokenizer_config["tokenizer_type"]),
                    model_path=str(tokenizer_config["tokenizer_model_path"]),
                    language=tokenizer_config.get("tokenizer_language"),
                    task=tokenizer_config.get("tokenizer_task"),
                )
                suppressed_token_ids = {
                    int(token_id)
                    for token_id in model_config.get("ctc_suppressed_token_ids", ())
                }

            if (
                row.get("mode") != "bi"
                or row.get("decode_strategy") != "ctc_greedy"
                or row.get("ctc_score") is not None
                or row.get("decoder_score") is not None
                or row.get("combined_score") is not None
            ):
                raise ValueError(
                    f"Stage211 student row is not CTC-only greedy output at {path}:{line_number}"
                )
            pred_token_ids = row.get("pred_token_ids")
            ref_token_ids = row.get("ref_token_ids")
            if (
                not isinstance(pred_token_ids, list)
                or not isinstance(ref_token_ids, list)
                or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in pred_token_ids)
                or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in ref_token_ids)
            ):
                raise ValueError(
                    f"Stage211 student row has invalid token IDs at {path}:{line_number}"
                )
            if any(
                token_id < 0
                or token_id >= STAGE211_STUDENT_TOKENIZER_VOCAB_SIZE
                or token_id in suppressed_token_ids
                for token_id in (*pred_token_ids, *ref_token_ids)
            ):
                raise ValueError(
                    f"Stage211 student row contains blank, suppressed, or out-of-range token IDs "
                    f"at {path}:{line_number}"
                )
            if tokenizer is None:
                raise AssertionError("Stage211 student tokenizer was not initialized.")
            pred_token_key = tuple(pred_token_ids)
            pred_decoded = decoded_token_ids.get(pred_token_key)
            if pred_decoded is None:
                pred_decoded = str(tokenizer.decode(pred_token_ids))
                decoded_token_ids[pred_token_key] = pred_decoded
            if pred_decoded != str(row["pred_text"]):
                raise ValueError(
                    f"Stage211 student prediction text/token mismatch at {path}:{line_number}"
                )
            ref_token_key = tuple(ref_token_ids)
            ref_decoded = decoded_token_ids.get(ref_token_key)
            if ref_decoded is None:
                ref_decoded = str(tokenizer.decode(ref_token_ids))
                decoded_token_ids[ref_token_key] = ref_decoded
            if ref_decoded != str(row["ref_text"]):
                raise ValueError(
                    f"Stage211 student reference text/token mismatch at {path}:{line_number}"
                )
            debug = row.get("debug")
            if (
                not isinstance(debug, dict)
                or int(debug.get("pred_token_count", -1)) != len(pred_token_ids)
                or int(debug.get("ref_token_count", -1)) != len(ref_token_ids)
                or int(debug.get("feature_length", -1)) <= 0
                or int(debug.get("logit_length", -1)) <= 0
                or not math.isfinite(float(debug.get("blank_top1_ratio", float("nan"))))
                or not math.isfinite(float(debug.get("avg_blank_prob", float("nan"))))
            ):
                raise ValueError(
                    f"Stage211 student row has invalid CTC debug evidence at {path}:{line_number}"
                )
            alignments = row.get("alignments")
            if (
                not isinstance(alignments, list)
                or len(alignments) != len(pred_token_ids)
                or not all(isinstance(alignment, dict) for alignment in alignments)
                or [int(alignment.get("token_id", -1)) for alignment in alignments]
                != pred_token_ids
            ):
                raise ValueError(
                    f"Stage211 student row has invalid CTC alignments at {path}:{line_number}"
                )

            reference = normalize_asr_text_for_metrics(
                str(row["ref_text"]),
                language=language,
                normalization="ctc",
            )
            prediction = normalize_asr_text_for_metrics(
                str(row["pred_text"]),
                language=language,
                normalization="ctc",
            )
            records[utt_id] = (reference, prediction)
    if not records or execution_provenance is None:
        raise ValueError(f"Stage211 public prediction JSONL is empty: {path}")
    return records, execution_provenance


def build_stage211_student_public_prediction_receipt(
    *,
    checkpoint_path: Path,
    manifest_paths: dict[str, Path],
    prediction_paths: dict[str, Path],
    benchmarks: dict[str, dict[str, str | int]],
) -> dict[str, Any]:
    checkpoint_path = _bound_file(checkpoint_path, label="student public checkpoint")
    if set(manifest_paths) != set(benchmarks):
        raise ValueError("Stage211 student public receipt manifest map is incomplete.")
    if set(prediction_paths) != set(benchmarks):
        raise ValueError("Stage211 student public receipt prediction map is incomplete.")

    results: list[dict[str, Any]] = []
    shared_execution_provenance: dict[str, Any] | None = None
    for dataset, expected in benchmarks.items():
        language = str(expected["language"])
        metric = str(expected["metric"])
        expected_samples = int(expected["samples"])
        manifest_path = _bound_file(
            manifest_paths[dataset],
            label=f"{dataset} student public manifest",
        )
        prediction_path = _bound_file(
            prediction_paths[dataset],
            label=f"{dataset} student public predictions",
        )
        manifest_references = _jsonl_references(
            manifest_path,
            language=language,
            reference_keys=("text", "transcript", "ref_text", "reference"),
        )
        predictions, execution_provenance = _jsonl_student_ctc_predictions(
            prediction_path,
            language=language,
            expected_checkpoint=checkpoint_path,
            expected_execution_provenance=shared_execution_provenance,
        )
        if shared_execution_provenance is None:
            shared_execution_provenance = execution_provenance
        elif execution_provenance != shared_execution_provenance:
            raise ValueError(
                f"Stage211 {dataset} student CTC execution provenance differs across datasets."
            )
        if (
            len(manifest_references) != expected_samples
            or len(predictions) != expected_samples
            or set(predictions) != set(manifest_references)
        ):
            raise ValueError(f"Stage211 {dataset} student public receipt coverage mismatch.")
        if any(
            predictions[utt_id][0] != reference
            for utt_id, reference in manifest_references.items()
        ):
            raise ValueError(
                f"Stage211 {dataset} student public receipt normalized references mismatch."
            )
        results.append(
            {
                "dataset": dataset,
                "language": language,
                "metric": metric,
                "sample_count": expected_samples,
                "identical_utt_coverage": True,
                "normalized_reference_mismatch_count": 0,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "student_prediction_path": str(prediction_path),
                "student_prediction_sha256": _sha256_file(prediction_path),
                "student_ctc_execution_provenance_sha256": _canonical_json_sha256(
                    execution_provenance
                ),
                "all_rows_ctc_only_greedy": True,
                "all_token_ids_within_student_tokenizer": True,
            }
        )
    if shared_execution_provenance is None:
        raise ValueError("Stage211 student public receipt lacks CTC execution provenance.")
    return {
        "schema_version": 2,
        "pipeline": "stage211",
        "artifact": "student_public_prediction_receipt",
        "complete": True,
        "decode": "greedy_ctc",
        "mode": "bi",
        "normalization": "ctc",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "student_ctc_execution_provenance": shared_execution_provenance,
        "student_ctc_execution_provenance_sha256": _canonical_json_sha256(
            shared_execution_provenance
        ),
        "all_rows_ctc_only_greedy": True,
        "all_rows_checkpoint_bound": True,
        "all_token_ids_within_student_tokenizer": True,
        "dataset_count": len(results),
        "total_samples": sum(int(result["sample_count"]) for result in results),
        "results": results,
    }


def validate_stage211_student_public_prediction_receipt(
    receipt_path: Path,
    *,
    expected_checkpoint: Path,
    expected_manifest_paths: dict[str, Path],
    expected_prediction_paths: dict[str, Path],
    benchmarks: dict[str, dict[str, str | int]],
) -> dict[str, Any]:
    receipt_path = _bound_file(receipt_path, label="student public prediction receipt")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage211 student public prediction receipt must be a JSON object.")
    expected_checkpoint = expected_checkpoint.expanduser().resolve()
    if (
        payload.get("schema_version") != 2
        or payload.get("pipeline") != "stage211"
        or payload.get("artifact") != "student_public_prediction_receipt"
        or payload.get("complete") is not True
        or payload.get("decode") != "greedy_ctc"
        or payload.get("mode") != "bi"
        or payload.get("normalization") != "ctc"
        or payload.get("all_rows_ctc_only_greedy") is not True
        or payload.get("all_rows_checkpoint_bound") is not True
        or payload.get("all_token_ids_within_student_tokenizer") is not True
    ):
        raise ValueError("Stage211 student public prediction receipt contract mismatch.")
    if Path(str(payload.get("checkpoint_path") or "")).expanduser().resolve() != (
        expected_checkpoint
    ) or payload.get("checkpoint_sha256") != _sha256_file(expected_checkpoint):
        raise ValueError("Stage211 student public prediction receipt checkpoint mismatch.")
    rebuilt = build_stage211_student_public_prediction_receipt(
        checkpoint_path=expected_checkpoint,
        manifest_paths={
            dataset: path.expanduser().resolve()
            for dataset, path in expected_manifest_paths.items()
        },
        prediction_paths={
            dataset: path.expanduser().resolve()
            for dataset, path in expected_prediction_paths.items()
        },
        benchmarks=benchmarks,
    )
    if payload != rebuilt:
        raise ValueError(
            "Stage211 student public prediction receipt does not match current files."
        )
    return payload


def _score_prediction_records(
    records: dict[str, tuple[str, str]],
    *,
    metric: str,
) -> dict[str, Any]:
    if metric not in {"wer", "cer"}:
        raise ValueError(f"Unsupported Stage211 public metric: {metric!r}")
    total_wer_errors = 0
    total_cer_errors = 0
    total_ref_words = 0
    total_ref_chars = 0
    primary_insertions = 0
    primary_deletions = 0
    primary_substitutions = 0
    primary_reference_units = 0
    primary_prediction_units = 0
    per_sample_primary: dict[str, float] = {}
    normalized_predictions: dict[str, str] = {}
    for utt_id, (reference, prediction) in records.items():
        ref_words = tokenize_for_wer(reference)
        pred_words = tokenize_for_wer(prediction)
        ref_chars = tokenize_for_cer(reference)
        pred_chars = tokenize_for_cer(prediction)
        if metric == "wer":
            insertions, deletions, substitutions = edit_counts(ref_words, pred_words)
            wer_errors = insertions + deletions + substitutions
            cer_errors = edit_distance(pred_chars, ref_chars)
            ref_primary = ref_words
            pred_primary = pred_words
        else:
            insertions, deletions, substitutions = edit_counts(ref_chars, pred_chars)
            cer_errors = insertions + deletions + substitutions
            wer_errors = edit_distance(pred_words, ref_words)
            ref_primary = ref_chars
            pred_primary = pred_chars
        total_wer_errors += wer_errors
        total_cer_errors += cer_errors
        total_ref_words += len(ref_words)
        total_ref_chars += len(ref_chars)
        primary_insertions += insertions
        primary_deletions += deletions
        primary_substitutions += substitutions
        primary_reference_units += len(ref_primary)
        primary_prediction_units += len(pred_primary)
        per_sample_primary[utt_id] = float(insertions + deletions + substitutions) / float(
            max(1, len(ref_primary))
        )
        normalized_predictions[utt_id] = _normalize_text_for_error_tokens(prediction)
    primary_denominator = max(1, primary_reference_units)
    return {
        "sample_count": len(records),
        "avg_wer": float(total_wer_errors) / float(max(1, total_ref_words)),
        "avg_cer": float(total_cer_errors) / float(max(1, total_ref_chars)),
        "prediction_reference_unit_ratio": primary_prediction_units / primary_denominator,
        "insertion_rate": primary_insertions / primary_denominator,
        "deletion_rate": primary_deletions / primary_denominator,
        "substitution_rate": primary_substitutions / primary_denominator,
        "per_sample_primary": per_sample_primary,
        "normalized_predictions": normalized_predictions,
    }


def _recompute_dataset(
    *,
    dataset: str,
    language: str,
    metric: str,
    nano_path: Path,
    student_path: Path,
    nano_records: dict[str, tuple[str, str]],
    student_records: dict[str, tuple[str, str]],
    max_relative_ratio: float,
    max_absolute_gap_points: float,
) -> dict[str, Any]:
    nano_stats = _score_prediction_records(nano_records, metric=metric)
    student_stats = _score_prediction_records(student_records, metric=metric)
    metric_key = f"avg_{metric}"
    nano_rate = float(nano_stats[metric_key])
    student_rate = float(student_stats[metric_key])
    absolute_gap_points = (student_rate - nano_rate) * 100.0
    relative_ratio = _relative_ratio(
        nano_rate=nano_rate,
        student_rate=student_rate,
    )
    improved = 0
    worsened = 0
    unchanged = 0
    changed_predictions = 0
    for utt_id in nano_records:
        nano_error = float(nano_stats["per_sample_primary"][utt_id])
        student_error = float(student_stats["per_sample_primary"][utt_id])
        if student_error < nano_error - 1.0e-9:
            improved += 1
        elif student_error > nano_error + 1.0e-9:
            worsened += 1
        else:
            unchanged += 1
        if (
            nano_stats["normalized_predictions"][utt_id]
            != student_stats["normalized_predictions"][utt_id]
        ):
            changed_predictions += 1
    absolute_gate_pass = absolute_gap_points <= max_absolute_gap_points
    relative_gate_pass = relative_ratio <= max_relative_ratio
    return {
        "dataset": dataset,
        "label": _PUBLIC_LABELS[dataset],
        "language": language,
        "metric": metric,
        "sample_count": int(nano_stats["sample_count"]),
        "identical_utt_coverage": True,
        "normalized_reference_mismatch_count": 0,
        "nano_wer": float(nano_stats["avg_wer"]),
        "student_wer": float(student_stats["avg_wer"]),
        "nano_cer": float(nano_stats["avg_cer"]),
        "student_cer": float(student_stats["avg_cer"]),
        "nano_error_rate": nano_rate,
        "student_error_rate": student_rate,
        "absolute_gap_points": absolute_gap_points,
        "relative_ratio": relative_ratio,
        "absolute_gate_pass": absolute_gate_pass,
        "relative_gate_pass": relative_gate_pass,
        "gate_pass": absolute_gate_pass and relative_gate_pass,
        "nano_prediction_reference_unit_ratio": float(
            nano_stats["prediction_reference_unit_ratio"]
        ),
        "student_prediction_reference_unit_ratio": float(
            student_stats["prediction_reference_unit_ratio"]
        ),
        "nano_insertion_rate": float(nano_stats["insertion_rate"]),
        "student_insertion_rate": float(student_stats["insertion_rate"]),
        "nano_deletion_rate": float(nano_stats["deletion_rate"]),
        "student_deletion_rate": float(student_stats["deletion_rate"]),
        "nano_substitution_rate": float(nano_stats["substitution_rate"]),
        "student_substitution_rate": float(student_stats["substitution_rate"]),
        "changed_prediction_count": changed_predictions,
        "student_improved_count": improved,
        "student_worsened_count": worsened,
        "unchanged_count": unchanged,
        "nano_prediction_path": str(nano_path),
        "student_prediction_path": str(student_path),
    }


def _validate_recomputed_result(
    reported: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    dataset: str,
) -> None:
    for key, expected in recomputed.items():
        actual = reported.get(key)
        if key in {"nano_prediction_path", "student_prediction_path"}:
            if Path(str(actual or "")).expanduser().resolve() != Path(str(expected)).resolve():
                raise ValueError(f"Stage211 {dataset} replayed {key} path mismatch.")
        elif isinstance(expected, bool):
            if actual is not expected:
                raise ValueError(f"Stage211 {dataset} replayed {key} mismatch.")
        elif isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(actual),
                    expected,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"Stage211 {dataset} replayed {key} mismatch.")
        elif actual != expected:
            raise ValueError(f"Stage211 {dataset} replayed {key} mismatch.")


def replay_stage211_public_comparison(
    report: dict[str, Any],
    *,
    manifest_paths: dict[str, Path],
    benchmarks: dict[str, dict[str, str | int]],
    expected_checkpoint: Path | None = None,
    require_student_prediction_receipt: bool = False,
) -> dict[str, Any]:
    if report.get("decode") != "greedy_ctc" or report.get("normalization") != "ctc":
        raise ValueError("Stage211 public comparison must use greedy CTC and CTC normalization.")
    gate = report.get("gate")
    if (
        not isinstance(gate, dict)
        or gate.get("requires_every_dataset") is not True
        or not math.isclose(
            float(gate.get("max_relative_ratio", float("nan"))),
            STAGE211_PUBLIC_MAX_RELATIVE_RATIO,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(gate.get("max_absolute_gap_points", float("nan"))),
            STAGE211_PUBLIC_MAX_ABSOLUTE_GAP_POINTS,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("Stage211 public comparison metric-gate contract mismatch.")
    if set(manifest_paths) != set(benchmarks):
        raise ValueError("Stage211 public manifest map is incomplete or unexpected.")
    if expected_checkpoint is not None:
        expected_checkpoint = expected_checkpoint.expanduser().resolve()
        if Path(
            str(report.get("student_checkpoint_path") or "")
        ).expanduser().resolve() != expected_checkpoint or report.get(
            "student_checkpoint_sha256"
        ) != _sha256_file(expected_checkpoint):
            raise ValueError("Stage211 public comparison checkpoint binding mismatch.")
    raw_results = report.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(benchmarks):
        raise ValueError("Stage211 public comparison result coverage mismatch.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(benchmarks):
        raise ValueError("Stage211 public comparison dataset set is incomplete or unexpected.")

    receipt_path_value = report.get("student_prediction_receipt_path")
    receipt_sha256 = report.get("student_prediction_receipt_sha256")
    has_receipt_binding = receipt_path_value is not None or receipt_sha256 is not None
    if require_student_prediction_receipt and not has_receipt_binding:
        raise ValueError("Stage211 public comparison lacks student prediction provenance.")
    if has_receipt_binding:
        if receipt_path_value is None or receipt_sha256 is None:
            raise ValueError("Stage211 student prediction provenance binding is incomplete.")
        receipt_path = _bound_file(
            receipt_path_value,
            label="student public prediction receipt",
        )
        if receipt_sha256 != _sha256_file(receipt_path):
            raise ValueError("Stage211 student public prediction receipt SHA-256 mismatch.")
        receipt_checkpoint = expected_checkpoint
        if receipt_checkpoint is None:
            receipt_checkpoint = Path(
                str(report.get("student_checkpoint_path") or "")
            ).expanduser().resolve()
        validate_stage211_student_public_prediction_receipt(
            receipt_path,
            expected_checkpoint=receipt_checkpoint,
            expected_manifest_paths=manifest_paths,
            expected_prediction_paths={
                dataset: Path(str(by_dataset[dataset].get("student_prediction_path") or ""))
                for dataset in benchmarks
            },
            benchmarks=benchmarks,
        )

    replayed_results: list[dict[str, Any]] = []
    for dataset, expected in benchmarks.items():
        reported = by_dataset[dataset]
        language = str(expected["language"])
        metric = str(expected["metric"])
        manifest_path = _bound_file(manifest_paths[dataset], label=f"{dataset} manifest")
        nano_path = _bound_file(
            reported.get("nano_prediction_path"),
            label=f"{dataset} Nano predictions",
        )
        student_path = _bound_file(
            reported.get("student_prediction_path"),
            label=f"{dataset} student predictions",
        )
        manifest_references = _jsonl_references(
            manifest_path,
            language=language,
            reference_keys=("text", "transcript", "ref_text", "reference"),
        )
        nano_records = _jsonl_predictions(nano_path, language=language)
        student_records = _jsonl_predictions(student_path, language=language)
        nano_references = {utt_id: reference for utt_id, (reference, _) in nano_records.items()}
        student_references = {
            utt_id: reference for utt_id, (reference, _) in student_records.items()
        }
        expected_samples = int(expected["samples"])
        if (
            len(manifest_references) != expected_samples
            or set(nano_references) != set(manifest_references)
            or set(student_references) != set(manifest_references)
        ):
            raise ValueError(f"Stage211 {dataset} public utterance coverage mismatch.")
        if any(
            nano_references[utt_id] != reference or student_references[utt_id] != reference
            for utt_id, reference in manifest_references.items()
        ):
            raise ValueError(f"Stage211 {dataset} normalized public references mismatch.")
        recomputed = _recompute_dataset(
            dataset=dataset,
            language=language,
            metric=metric,
            nano_path=nano_path,
            student_path=student_path,
            nano_records=nano_records,
            student_records=student_records,
            max_relative_ratio=STAGE211_PUBLIC_MAX_RELATIVE_RATIO,
            max_absolute_gap_points=STAGE211_PUBLIC_MAX_ABSOLUTE_GAP_POINTS,
        )
        _validate_recomputed_result(reported, recomputed, dataset=dataset)
        replayed_results.append(
            {
                **reported,
                **recomputed,
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "nano_prediction_path": str(nano_path),
                "nano_prediction_sha256": _sha256_file(nano_path),
                "student_prediction_path": str(student_path),
                "student_prediction_sha256": _sha256_file(student_path),
                "metric_source_recomputed": True,
            }
        )
    all_datasets_pass = all(bool(row["gate_pass"]) for row in replayed_results)
    if report.get("all_datasets_pass") is not all_datasets_pass:
        raise ValueError("Stage211 public comparison every-dataset decision mismatch.")
    return {
        **report,
        "all_datasets_complete": True,
        "all_datasets_pass": all_datasets_pass,
        "results": replayed_results,
    }


def build_stage211_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    benchmarks: dict[str, dict[str, str | int]],
    max_dataset_regression: float = STAGE211_PUBLIC_MAX_PROGRESS_REGRESSION,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(benchmarks) or set(candidate_results) != set(benchmarks):
        raise ValueError("Stage211 baseline/candidate public datasets differ.")
    benchmark_languages = {str(expected["language"]) for expected in benchmarks.values()}
    if benchmark_languages != {"en", "zh"}:
        raise ValueError("Stage211 public progress requires English and Chinese benchmarks.")
    rows: list[dict[str, Any]] = []
    for dataset, expected in benchmarks.items():
        baseline_result = baseline_results[dataset]
        candidate_result = candidate_results[dataset]
        for hash_key in ("manifest_sha256", "nano_prediction_sha256"):
            if baseline_result.get(hash_key) != candidate_result.get(hash_key):
                raise ValueError(f"Stage211 {dataset} baseline/candidate {hash_key} differs.")
        baseline_error = float(baseline_result["student_error_rate"])
        candidate_error = float(candidate_result["student_error_rate"])
        baseline_deletion = float(baseline_result["student_deletion_rate"])
        candidate_deletion = float(candidate_result["student_deletion_rate"])
        rows.append(
            {
                "dataset": dataset,
                "language": str(expected["language"]),
                "metric": str(expected["metric"]),
                "baseline_error_rate": baseline_error,
                "candidate_error_rate": candidate_error,
                "absolute_change": candidate_error - baseline_error,
                "baseline_deletion_rate": baseline_deletion,
                "candidate_deletion_rate": candidate_deletion,
                "within_regression_limit": candidate_error
                <= baseline_error + max_dataset_regression,
                "improved": candidate_error < baseline_error,
            }
        )
    macro_baseline_error = sum(float(row["baseline_error_rate"]) for row in rows) / len(rows)
    macro_candidate_error = sum(float(row["candidate_error_rate"]) for row in rows) / len(rows)
    macro_baseline_deletion = sum(float(row["baseline_deletion_rate"]) for row in rows) / len(rows)
    macro_candidate_deletion = sum(float(row["candidate_deletion_rate"]) for row in rows) / len(
        rows
    )
    language_summaries: dict[str, dict[str, Any]] = {}
    for language in ("en", "zh"):
        language_rows = [row for row in rows if row["language"] == language]
        metrics = {str(row["metric"]) for row in language_rows}
        if not language_rows or len(metrics) != 1:
            raise ValueError(f"Stage211 {language} public progress metric coverage mismatch.")
        baseline_error = sum(float(row["baseline_error_rate"]) for row in language_rows) / len(
            language_rows
        )
        candidate_error = sum(float(row["candidate_error_rate"]) for row in language_rows) / len(
            language_rows
        )
        baseline_deletion = sum(
            float(row["baseline_deletion_rate"]) for row in language_rows
        ) / len(language_rows)
        candidate_deletion = sum(
            float(row["candidate_deletion_rate"]) for row in language_rows
        ) / len(language_rows)
        error_improved = candidate_error < baseline_error
        deletion_improved = candidate_deletion < baseline_deletion
        improved_datasets = sum(bool(row["improved"]) for row in language_rows)
        language_summaries[language] = {
            "language": language,
            "metric": next(iter(metrics)),
            "dataset_count": len(language_rows),
            "datasets": [str(row["dataset"]) for row in language_rows],
            "macro_baseline_error_rate": baseline_error,
            "macro_candidate_error_rate": candidate_error,
            "macro_baseline_deletion_rate": baseline_deletion,
            "macro_candidate_deletion_rate": candidate_deletion,
            "improved_datasets": improved_datasets,
            "error_improved": error_improved,
            "deletion_improved": deletion_improved,
            "gate_passed": error_improved and deletion_improved and improved_datasets > 0,
        }
    gate_passed = (
        all(bool(row["within_regression_limit"]) for row in rows)
        and macro_candidate_error < macro_baseline_error
        and macro_candidate_deletion < macro_baseline_deletion
        and all(bool(summary["gate_passed"]) for summary in language_summaries.values())
    )
    return {
        "gate_passed": gate_passed,
        "max_dataset_regression": max_dataset_regression,
        "macro_baseline_error_rate": macro_baseline_error,
        "macro_candidate_error_rate": macro_candidate_error,
        "macro_baseline_deletion_rate": macro_baseline_deletion,
        "macro_candidate_deletion_rate": macro_candidate_deletion,
        "improved_datasets": sum(bool(row["improved"]) for row in rows),
        "language_summaries": language_summaries,
        "results": rows,
        "baseline_public_benchmark": baseline,
    }


def build_stage211_sft_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    benchmarks: dict[str, dict[str, str | int]],
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(benchmarks) or set(candidate_results) != set(benchmarks):
        raise ValueError("Stage211D baseline/candidate public datasets differ.")
    rows: list[dict[str, Any]] = []
    for dataset, expected in benchmarks.items():
        baseline_result = baseline_results[dataset]
        candidate_result = candidate_results[dataset]
        for hash_key in ("manifest_sha256", "nano_prediction_sha256"):
            if baseline_result.get(hash_key) != candidate_result.get(hash_key):
                raise ValueError(f"Stage211D {dataset} baseline/candidate {hash_key} differs.")
        baseline_error = float(baseline_result["student_error_rate"])
        candidate_error = float(candidate_result["student_error_rate"])
        if not math.isfinite(baseline_error) or not math.isfinite(candidate_error):
            raise ValueError(f"Stage211D {dataset} error rate must be finite.")
        rows.append(
            {
                "dataset": dataset,
                "metric": expected["metric"],
                "baseline_error_rate": baseline_error,
                "candidate_error_rate": candidate_error,
                "absolute_change": candidate_error - baseline_error,
                "no_regression": candidate_error <= baseline_error + tolerance,
                "improved": candidate_error < baseline_error - tolerance,
            }
        )
    macro_baseline = sum(float(row["baseline_error_rate"]) for row in rows) / len(rows)
    macro_candidate = sum(float(row["candidate_error_rate"]) for row in rows) / len(rows)
    no_regressions = all(bool(row["no_regression"]) for row in rows)
    improved_datasets = sum(bool(row["improved"]) for row in rows)
    macro_improved = macro_candidate < macro_baseline - tolerance
    gate_passed = no_regressions and macro_improved and improved_datasets > 0
    return {
        "gate_passed": gate_passed,
        "tolerance": tolerance,
        "no_dataset_regression": no_regressions,
        "macro_improved": macro_improved,
        "improved_datasets": improved_datasets,
        "macro_baseline_error_rate": macro_baseline,
        "macro_candidate_error_rate": macro_candidate,
        "results": rows,
    }


def build_stage211_sft_correction_public_progress(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    benchmarks: dict[str, dict[str, str | int]],
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    baseline_results = {str(result["dataset"]): result for result in baseline["results"]}
    candidate_results = {str(result["dataset"]): result for result in candidate["results"]}
    if set(baseline_results) != set(benchmarks) or set(candidate_results) != set(benchmarks):
        raise ValueError("Stage211D correction baseline/candidate public datasets differ.")
    rows: list[dict[str, Any]] = []
    for dataset, expected in benchmarks.items():
        baseline_result = baseline_results[dataset]
        candidate_result = candidate_results[dataset]
        for hash_key in ("manifest_sha256", "nano_prediction_sha256"):
            if baseline_result.get(hash_key) != candidate_result.get(hash_key):
                raise ValueError(
                    f"Stage211D correction {dataset} baseline/candidate {hash_key} differs."
                )
        metric = str(expected["metric"])
        language = "en" if metric == "wer" else "zh"
        baseline_error = float(baseline_result["student_error_rate"])
        candidate_error = float(candidate_result["student_error_rate"])
        if not math.isfinite(baseline_error) or not math.isfinite(candidate_error):
            raise ValueError(f"Stage211D correction {dataset} error rate must be finite.")
        rows.append(
            {
                "dataset": dataset,
                "language": language,
                "metric": metric,
                "baseline_error_rate": baseline_error,
                "candidate_error_rate": candidate_error,
                "absolute_change": candidate_error - baseline_error,
                "no_regression": candidate_error <= baseline_error + tolerance,
                "improved": candidate_error < baseline_error - tolerance,
            }
        )
    language_summaries: dict[str, dict[str, Any]] = {}
    for language, metric in (("en", "wer"), ("zh", "cer")):
        language_rows = [row for row in rows if row["language"] == language]
        if not language_rows:
            raise ValueError(f"Stage211D correction has no {language} public datasets.")
        baseline_macro = sum(float(row["baseline_error_rate"]) for row in language_rows) / len(
            language_rows
        )
        candidate_macro = sum(
            float(row["candidate_error_rate"]) for row in language_rows
        ) / len(language_rows)
        improved_datasets = sum(bool(row["improved"]) for row in language_rows)
        macro_improved = candidate_macro < baseline_macro - tolerance
        language_summaries[language] = {
            "metric": metric,
            "datasets": [str(row["dataset"]) for row in language_rows],
            "dataset_count": len(language_rows),
            "macro_baseline_error_rate": baseline_macro,
            "macro_candidate_error_rate": candidate_macro,
            "macro_improved": macro_improved,
            "improved_datasets": improved_datasets,
            "gate_passed": macro_improved and improved_datasets > 0,
        }
    macro_baseline = sum(float(row["baseline_error_rate"]) for row in rows) / len(rows)
    macro_candidate = sum(float(row["candidate_error_rate"]) for row in rows) / len(rows)
    no_regressions = all(bool(row["no_regression"]) for row in rows)
    gate_passed = no_regressions and all(
        bool(summary["gate_passed"]) for summary in language_summaries.values()
    )
    return {
        "gate_passed": gate_passed,
        "tolerance": tolerance,
        "no_dataset_regression": no_regressions,
        "bilingual_macro_improved": all(
            bool(summary["macro_improved"]) for summary in language_summaries.values()
        ),
        "macro_baseline_error_rate": macro_baseline,
        "macro_candidate_error_rate": macro_candidate,
        "language_summaries": language_summaries,
        "results": rows,
    }


def _load_bound_comparison(
    report: dict[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    label: str,
) -> dict[str, Any]:
    path = _bound_file(report.get(path_key), label=label)
    if report.get(sha256_key) != _sha256_file(path):
        raise ValueError(f"Stage211 {label} SHA-256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage211 {label} must be a JSON object.")
    return payload


def _validate_replayed_value(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"Stage211 {label} does not match replayed public evidence.")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Stage211 {label} field coverage mismatch.")
        for key, value in expected.items():
            _validate_replayed_value(actual[key], value, label=f"{label}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Stage211 {label} sequence mismatch.")
        for index, value in enumerate(expected):
            _validate_replayed_value(actual[index], value, label=f"{label}[{index}]")
        return
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise ValueError(f"Stage211 {label} does not match replayed public evidence.")
        return
    if isinstance(expected, float):
        try:
            matches = math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"Stage211 {label} does not match replayed public evidence.")
        return
    if actual != expected:
        raise ValueError(f"Stage211 {label} does not match replayed public evidence.")


def replay_stage211_sft_public_evidence(
    report: dict[str, Any],
    *,
    benchmarks: dict[str, dict[str, str | int]],
    expected_baseline_checkpoint: Path,
    expected_candidate_checkpoint: Path,
) -> dict[str, Any]:
    embedded_candidate = report.get("public_benchmark")
    if not isinstance(embedded_candidate, dict):
        raise ValueError("Stage211D report lacks its public benchmark.")
    raw_results = embedded_candidate.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Stage211D public benchmark results are invalid.")
    by_dataset = {
        str(result.get("dataset")): result for result in raw_results if isinstance(result, dict)
    }
    if set(by_dataset) != set(benchmarks):
        raise ValueError("Stage211D public benchmark dataset coverage mismatch.")
    manifest_paths = {
        dataset: Path(str(result.get("manifest_path") or "")).expanduser().resolve()
        for dataset, result in by_dataset.items()
    }
    baseline_source = _load_bound_comparison(
        report,
        path_key="baseline_public_comparison_report_path",
        sha256_key="baseline_public_comparison_report_sha256",
        label="SFT baseline public comparison",
    )
    candidate_source = _load_bound_comparison(
        report,
        path_key="public_comparison_report_path",
        sha256_key="public_comparison_report_sha256",
        label="SFT candidate public comparison",
    )
    replayed_baseline = replay_stage211_public_comparison(
        baseline_source,
        manifest_paths=manifest_paths,
        benchmarks=benchmarks,
        expected_checkpoint=expected_baseline_checkpoint,
    )
    replayed_candidate = replay_stage211_public_comparison(
        candidate_source,
        manifest_paths=manifest_paths,
        benchmarks=benchmarks,
        expected_checkpoint=expected_candidate_checkpoint,
        require_student_prediction_receipt=True,
    )
    replayed_progress = build_stage211_sft_public_progress(
        baseline=replayed_baseline,
        candidate=replayed_candidate,
        benchmarks=benchmarks,
    )
    _validate_replayed_value(
        report.get("baseline_public_benchmark"),
        replayed_baseline,
        label="SFT baseline public benchmark",
    )
    _validate_replayed_value(
        embedded_candidate,
        replayed_candidate,
        label="SFT candidate public benchmark",
    )
    _validate_replayed_value(
        report.get("public_progress"),
        replayed_progress,
        label="SFT public progress",
    )
    return {
        "baseline_public_benchmark": replayed_baseline,
        "public_benchmark": replayed_candidate,
        "public_progress": replayed_progress,
    }
