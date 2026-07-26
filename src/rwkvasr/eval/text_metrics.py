from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from rwkvasr.data import normalize_asr_text

_WER_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+|[\u4e00-\u9fff]"
)

_APOSTROPHE_CHARS = {"'", "\u2019", "\u02bc", "\uff07"}

_METRIC_EQUIVALENCE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bo\s+k\b"), "ok"),
    (re.compile(r"\bokay\b"), "ok"),
)

_LANGUAGE_CONFIRMATION_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*this\s+is\s+english\s+text\s*[\.\!\?。:：,，;；]*\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*这是中文文字\s*[\.\!\?。:：,，;；]*\s*"),
)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def strip_asr_language_confirmation_prefix(text: str | None) -> str:
    """Remove project-generated AR language metadata before ASR content scoring."""

    if text is None:
        return ""
    stripped = str(text).strip()
    for _ in range(len(_LANGUAGE_CONFIRMATION_PREFIX_PATTERNS)):
        for pattern in _LANGUAGE_CONFIRMATION_PREFIX_PATTERNS:
            updated = pattern.sub("", stripped, count=1).lstrip()
            if updated != stripped:
                stripped = updated
                break
        else:
            break
    return stripped


def normalize_asr_text_for_metrics(
    text: str | None,
    *,
    language: str | None = None,
    normalization: str = "ctc",
    strip_language_confirmation: bool = True,
) -> str:
    if text is None:
        return ""
    if strip_language_confirmation:
        text = strip_asr_language_confirmation_prefix(text)
    if normalization == "none":
        return text.strip()
    return normalize_asr_text(text, language=language, mode=normalization).strip()


def _is_ignored_metric_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return category.startswith("P") or category.startswith("S")


def _normalize_text_for_error_tokens(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    pieces: list[str] = []
    for ch in text:
        if ch in _APOSTROPHE_CHARS:
            continue
        if _is_ignored_metric_char(ch):
            pieces.append(" ")
        else:
            pieces.append(ch)
    normalized = re.sub(r"\s+", " ", "".join(pieces)).strip()
    for pattern, replacement in _METRIC_EQUIVALENCE_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def tokenize_for_wer(text: str) -> list[str]:
    normalized = _normalize_text_for_error_tokens(text)
    if not normalized:
        return []
    return _WER_TOKEN_PATTERN.findall(normalized)


def tokenize_for_cer(text: str) -> list[str]:
    normalized = _normalize_text_for_error_tokens(text)
    return [ch for ch in normalized if not ch.isspace()]


def edit_distance(source: list[str], target: list[str]) -> int:
    if not source:
        return len(target)
    if not target:
        return len(source)
    prev = list(range(len(target) + 1))
    for source_idx, source_token in enumerate(source, start=1):
        current = [source_idx]
        for target_idx, target_token in enumerate(target, start=1):
            cost = 0 if source_token == target_token else 1
            current.append(
                min(
                    prev[target_idx] + 1,
                    current[target_idx - 1] + 1,
                    prev[target_idx - 1] + cost,
                )
            )
        prev = current
    return prev[-1]


def edit_counts(reference: list[Any], hypothesis: list[Any]) -> tuple[int, int, int]:
    """Return insertion, deletion, and substitution counts for one optimal edit path."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    for ref_index in range(1, rows):
        costs[ref_index][0] = ref_index
    for hyp_index in range(1, columns):
        costs[0][hyp_index] = hyp_index
    for ref_index, ref_token in enumerate(reference, start=1):
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = costs[ref_index - 1][hyp_index - 1] + int(ref_token != hyp_token)
            insertion = costs[ref_index][hyp_index - 1] + 1
            deletion = costs[ref_index - 1][hyp_index] + 1
            costs[ref_index][hyp_index] = min(substitution, insertion, deletion)

    insertions = 0
    deletions = 0
    substitutions = 0
    ref_index = len(reference)
    hyp_index = len(hypothesis)
    while ref_index > 0 or hyp_index > 0:
        if (
            ref_index > 0
            and hyp_index > 0
            and reference[ref_index - 1] == hypothesis[hyp_index - 1]
            and costs[ref_index][hyp_index] == costs[ref_index - 1][hyp_index - 1]
        ):
            ref_index -= 1
            hyp_index -= 1
            continue
        if (
            ref_index > 0
            and hyp_index > 0
            and costs[ref_index][hyp_index] == costs[ref_index - 1][hyp_index - 1] + 1
        ):
            substitutions += 1
            ref_index -= 1
            hyp_index -= 1
            continue
        if ref_index > 0 and costs[ref_index][hyp_index] == costs[ref_index - 1][hyp_index] + 1:
            deletions += 1
            ref_index -= 1
            continue
        if hyp_index <= 0:
            raise RuntimeError("Invalid edit-distance backtrace.")
        insertions += 1
        hyp_index -= 1
    return insertions, deletions, substitutions


def _load_normalized_records(
    path: Path,
    *,
    language: str | None,
    normalization: str,
    strip_language_confirmation: bool = True,
) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    records: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            utt_id = str(record.get("utt_id", ""))
            pred = _as_text(record.get("pred_text"))
            ref = _as_text(record.get("ref_text"))
            if pred is None and ref is None:
                continue
            pred_norm = normalize_asr_text_for_metrics(
                pred,
                language=language,
                normalization=normalization,
                strip_language_confirmation=strip_language_confirmation,
            )
            ref_norm = normalize_asr_text_for_metrics(
                ref,
                language=language,
                normalization=normalization,
                strip_language_confirmation=strip_language_confirmation,
            )
            records[utt_id] = (pred_norm, ref_norm)
    return records


def compute_text_error_stats(
    path: Path,
    *,
    language: str | None = None,
    normalization: str = "ctc",
    strip_language_confirmation: bool = True,
) -> dict[str, Any]:
    records = _load_normalized_records(
        path,
        language=language,
        normalization=normalization,
        strip_language_confirmation=strip_language_confirmation,
    )
    if not records:
        return {}
    total_wer_errors = 0
    total_cer_errors = 0
    total_ref_words = 0
    total_ref_chars = 0
    per_sample_wer: dict[str, float] = {}
    per_sample_cer: dict[str, float] = {}
    for utt_id, (pred_text, ref_text) in records.items():
        pred_wer = tokenize_for_wer(pred_text)
        ref_wer = tokenize_for_wer(ref_text)
        pred_cer = tokenize_for_cer(pred_text)
        ref_cer = tokenize_for_cer(ref_text)
        wer_edits = edit_distance(pred_wer, ref_wer)
        cer_edits = edit_distance(pred_cer, ref_cer)
        ref_word_count = len(ref_wer)
        ref_char_count = len(ref_cer)
        total_wer_errors += wer_edits
        total_cer_errors += cer_edits
        total_ref_words += ref_word_count
        total_ref_chars += ref_char_count
        per_sample_wer[utt_id] = float(wer_edits) / float(max(1, ref_word_count))
        per_sample_cer[utt_id] = float(cer_edits) / float(max(1, ref_char_count))
    return {
        "sample_count": len(records),
        "avg_wer": float(total_wer_errors) / float(max(1, total_ref_words)),
        "avg_cer": float(total_cer_errors) / float(max(1, total_ref_chars)),
        "per_sample_wer": per_sample_wer,
        "per_sample_cer": per_sample_cer,
    }


def compute_text_error_decomposition(
    path: Path,
    *,
    language: str | None = None,
    normalization: str = "ctc",
    metric: str,
    strip_language_confirmation: bool = True,
) -> dict[str, Any]:
    if metric not in {"wer", "cer"}:
        raise ValueError("metric must be either 'wer' or 'cer'.")
    records = _load_normalized_records(
        path,
        language=language,
        normalization=normalization,
        strip_language_confirmation=strip_language_confirmation,
    )
    if not records:
        return {}

    insertions = 0
    deletions = 0
    substitutions = 0
    reference_units = 0
    prediction_units = 0
    tokenizer = tokenize_for_wer if metric == "wer" else tokenize_for_cer
    for prediction, reference in records.values():
        prediction_tokens = tokenizer(prediction)
        reference_tokens = tokenizer(reference)
        sample_insertions, sample_deletions, sample_substitutions = edit_counts(
            reference_tokens,
            prediction_tokens,
        )
        insertions += sample_insertions
        deletions += sample_deletions
        substitutions += sample_substitutions
        reference_units += len(reference_tokens)
        prediction_units += len(prediction_tokens)

    denominator = max(1, reference_units)
    return {
        "sample_count": len(records),
        "metric": metric,
        "reference_units": reference_units,
        "prediction_units": prediction_units,
        "prediction_reference_unit_ratio": prediction_units / denominator,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "insertion_rate": insertions / denominator,
        "deletion_rate": deletions / denominator,
        "substitution_rate": substitutions / denominator,
        "error_rate": (insertions + deletions + substitutions) / denominator,
    }


def compare_prediction_text_sets(
    baseline_jsonl: Path,
    candidate_jsonl: Path,
    *,
    baseline_label: str,
    candidate_label: str,
    language: str | None = None,
    normalization: str = "ctc",
    metric: str = "wer",
    strip_language_confirmation: bool = True,
) -> dict[str, Any]:
    if metric not in {"wer", "cer"}:
        raise ValueError("metric must be either 'wer' or 'cer'.")
    baseline_stats = compute_text_error_stats(
        baseline_jsonl,
        language=language,
        normalization=normalization,
        strip_language_confirmation=strip_language_confirmation,
    )
    candidate_stats = compute_text_error_stats(
        candidate_jsonl,
        language=language,
        normalization=normalization,
        strip_language_confirmation=strip_language_confirmation,
    )
    if not baseline_stats or not candidate_stats:
        return {}

    baseline_key = f"per_sample_{metric}"
    candidate_key = f"per_sample_{metric}"
    baseline_records = baseline_stats[baseline_key]
    candidate_records = candidate_stats[candidate_key]
    if not isinstance(baseline_records, dict) or not isinstance(candidate_records, dict):
        return {}

    baseline_preds = _load_normalized_records(
        baseline_jsonl,
        language=language,
        normalization=normalization,
        strip_language_confirmation=strip_language_confirmation,
    )
    candidate_preds = _load_normalized_records(
        candidate_jsonl,
        language=language,
        normalization=normalization,
        strip_language_confirmation=strip_language_confirmation,
    )
    shared_utts = sorted(set(baseline_records) & set(candidate_records))

    if not shared_utts:
        return {
            "baseline_avg_wer": baseline_stats.get("avg_wer"),
            "candidate_avg_wer": candidate_stats.get("avg_wer"),
            "baseline_avg_cer": baseline_stats.get("avg_cer"),
            "candidate_avg_cer": candidate_stats.get("avg_cer"),
            "shared_sample_count": 0,
            "changed_prediction_count": 0,
            "improved_count": 0,
            "worsened_count": 0,
            "unchanged_count": 0,
            "verdict": "no shared utt_id between baseline and candidate",
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "metric": metric,
        }

    improved = 0
    worsened = 0
    unchanged = 0
    changed_prediction = 0
    for utt_id in shared_utts:
        base_metric = float(baseline_records[utt_id])
        cand_metric = float(candidate_records[utt_id])
        if cand_metric < base_metric - 1e-9:
            improved += 1
        elif cand_metric > base_metric + 1e-9:
            worsened += 1
        else:
            unchanged += 1
        base_pred = baseline_preds.get(utt_id, ("", ""))[0]
        cand_pred = candidate_preds.get(utt_id, ("", ""))[0]
        if _normalize_text_for_error_tokens(base_pred) != _normalize_text_for_error_tokens(cand_pred):
            changed_prediction += 1

    baseline_avg_wer = baseline_stats.get("avg_wer")
    candidate_avg_wer = candidate_stats.get("avg_wer")
    baseline_avg_cer = baseline_stats.get("avg_cer")
    candidate_avg_cer = candidate_stats.get("avg_cer")
    verdict = "comparison unavailable"
    compare_base = baseline_avg_wer if metric == "wer" else baseline_avg_cer
    compare_cand = candidate_avg_wer if metric == "wer" else candidate_avg_cer
    if isinstance(compare_base, float) and isinstance(compare_cand, float):
        if compare_cand < compare_base - 0.01:
            verdict = (
                f"{candidate_label} improved preview ASR-content normalized {metric.upper()} vs {baseline_label}"
            )
        elif compare_cand > compare_base + 0.01:
            verdict = (
                f"{baseline_label} remained better than {candidate_label} on preview ASR-content normalized {metric.upper()}"
            )
        else:
            verdict = (
                f"{baseline_label} and {candidate_label} were roughly neutral on preview ASR-content normalized {metric.upper()}"
            )
    return {
        "baseline_avg_wer": baseline_avg_wer,
        "candidate_avg_wer": candidate_avg_wer,
        "baseline_avg_cer": baseline_avg_cer,
        "candidate_avg_cer": candidate_avg_cer,
        "shared_sample_count": len(shared_utts),
        "changed_prediction_count": changed_prediction,
        "improved_count": improved,
        "worsened_count": worsened,
        "unchanged_count": unchanged,
        "verdict": verdict,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "metric": metric,
    }
