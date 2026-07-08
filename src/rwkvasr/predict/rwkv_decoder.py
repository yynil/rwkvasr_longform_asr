from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rwkvasr.data import build_text_tokenizer
from rwkvasr.data.manifest import load_ctc_draft_cache
from rwkvasr.eval.text_metrics import (
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)
from rwkvasr.modules import build_inference_direction_mask

from .ctc import (
    CTCLabeledPrediction,
    PredictionConfig,
    _build_labeled_prediction_loader,
    _load_prediction_model,
    _maybe_report_prediction_progress,
    _resolve_prediction_total,
)


@dataclass(frozen=True)
class RWKVDecoderDecodeDebug:
    feature_length: int
    encoded_length: int
    pred_token_count: int
    ref_token_count: int | None
    eos_emitted: bool
    avg_logprob: float
    ctc_draft_fallback: bool = False
    ctc_draft_fallback_reason: str | None = None
    ctc_draft_cer_distance: float | None = None
    ctc_draft_length_ratio: float | None = None
    raw_pred_text: str | None = None


_REPEATED_SPAN_PATTERN = re.compile(r"(.{2,12})\1\1")


def _resolve_eos_token_id(tokenizer: object) -> int:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return int(eos_token_id)
    return 0


def _heuristic_max_new_tokens(
    encoded_length: int,
    *,
    explicit_max_new_tokens: int | None,
    max_new_tokens_factor: float,
) -> int:
    if explicit_max_new_tokens is not None:
        return max(1, int(explicit_max_new_tokens))
    estimated = int(math.ceil(float(encoded_length) * float(max_new_tokens_factor)))
    return max(16, min(1024, estimated))


def _contains_repeated_span(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if bool(_REPEATED_SPAN_PATTERN.search(compact)):
        return True
    if re.search(r"(.)\1{5,}", compact):
        return True
    tokens = tokenize_for_wer(text)
    if tokens:
        repeated_single = 1
        for previous, current in zip(tokens, tokens[1:], strict=False):
            repeated_single = repeated_single + 1 if previous == current else 1
            if repeated_single >= 6:
                return True
        max_ngram = min(6, len(tokens) // 3)
        for width in range(2, max_ngram + 1):
            counts = Counter(tuple(tokens[start : start + width]) for start in range(0, len(tokens) - width + 1))
            if any(count >= 3 for count in counts.values()):
                return True
    max_width = min(32, max(0, len(compact) // 3))
    for width in range(2, max_width + 1):
        for start in range(0, len(compact) - (width * 3) + 1):
            span = compact[start : start + width]
            if compact[start + width : start + 2 * width] != span:
                continue
            if compact[start + 2 * width : start + 3 * width] == span:
                return True
    return False


def _ctc_draft_fallback_decision(
    *,
    pred_text: str | None,
    draft_text: str | None,
    max_cer: float,
    min_length_ratio: float,
    max_length_ratio: float,
    reject_repetition: bool,
    metric_normalization: str,
) -> tuple[bool, str | None, float | None, float | None]:
    if draft_text is None:
        return False, None, None, None
    draft_norm = normalize_asr_text_for_metrics(draft_text, normalization=metric_normalization)
    if not draft_norm:
        return False, None, None, None
    pred_norm = normalize_asr_text_for_metrics(pred_text, normalization=metric_normalization)
    pred_chars = tokenize_for_cer(pred_norm)
    draft_chars = tokenize_for_cer(draft_norm)
    if not draft_chars:
        return False, None, None, None
    cer_distance = float(edit_distance(pred_chars, draft_chars)) / float(max(1, len(draft_chars)))
    length_ratio = float(len(pred_chars)) / float(max(1, len(draft_chars)))
    if not pred_chars:
        return True, "empty_ar", cer_distance, length_ratio
    if reject_repetition and _contains_repeated_span(pred_norm):
        return True, "repeated_span", cer_distance, length_ratio
    if length_ratio < min_length_ratio:
        return True, "too_short_vs_ctc_draft", cer_distance, length_ratio
    if length_ratio > max_length_ratio:
        return True, "too_long_vs_ctc_draft", cer_distance, length_ratio
    if cer_distance > max_cer:
        return True, "too_far_from_ctc_draft", cer_distance, length_ratio
    return False, None, cer_distance, length_ratio


def predict_rwkv_decoder_labeled(
    config: PredictionConfig,
    *,
    limit: int | None = None,
    max_new_tokens: int | None = None,
    max_new_tokens_factor: float = 2.0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    ctc_draft_fallback_max_cer: float | None = None,
    ctc_draft_fallback_min_length_ratio: float = 0.75,
    ctc_draft_fallback_max_length_ratio: float = 1.25,
    ctc_draft_fallback_reject_repetition: bool = True,
    ctc_draft_fallback_metric_normalization: str = "ctc",
) -> tuple[list[CTCLabeledPrediction], list[RWKVDecoderDecodeDebug]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided.")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1 when provided.")
    if max_new_tokens_factor <= 0:
        raise ValueError("max_new_tokens_factor must be > 0.")
    if do_sample:
        if not (temperature > 0):
            raise ValueError("temperature must be > 0 when do_sample=True.")
        if top_k < 0:
            raise ValueError("top_k must be >= 0.")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in the interval (0.0, 1.0].")
    if ctc_draft_fallback_max_cer is not None:
        if ctc_draft_fallback_max_cer < 0:
            raise ValueError("ctc_draft_fallback_max_cer must be >= 0 when provided.")
        if ctc_draft_fallback_min_length_ratio <= 0:
            raise ValueError("ctc_draft_fallback_min_length_ratio must be > 0.")
        if ctc_draft_fallback_max_length_ratio <= 0:
            raise ValueError("ctc_draft_fallback_max_length_ratio must be > 0.")
        if ctc_draft_fallback_min_length_ratio > ctc_draft_fallback_max_length_ratio:
            raise ValueError(
                "ctc_draft_fallback_min_length_ratio must be <= ctc_draft_fallback_max_length_ratio."
            )

    device = torch.device(config.device)
    model, feature_dtype = _load_prediction_model(config, device=device)
    if model.decoder is None:
        raise RuntimeError("RWKV decoder prediction requires decoder_enabled=True in the checkpoint config.")

    tokenizer = build_text_tokenizer(
        config.tokenizer_type,
        model_path=config.tokenizer_model_path,
        language=config.tokenizer_language,
        task=config.tokenizer_task,
    )
    decode_fn = getattr(tokenizer, "decode", None)
    eos_token_id = _resolve_eos_token_id(tokenizer)
    loader = _build_labeled_prediction_loader(config, tokenizer=tokenizer)
    ctc_draft_cache: dict[str, str] = {}
    if ctc_draft_fallback_max_cer is not None and config.decoder_ctc_draft_cache_path:
        ctc_draft_cache = load_ctc_draft_cache(
            str(config.decoder_ctc_draft_cache_path),
            text_key=str(config.decoder_ctc_draft_text_key or "pred_text"),
        )

    predictions: list[CTCLabeledPrediction] = []
    debug_rows: list[RWKVDecoderDecodeDebug] = []
    progress_total = _resolve_prediction_total(loader, limit=limit)
    progress_started_at = time.monotonic()
    progress_last_reported = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, feature_dtype=feature_dtype)
            mask = build_inference_direction_mask(
                model.config.num_layers,
                mode=config.mode,
                device=batch.features.device,
            )
            encoded, encoded_lengths, _ = model.encoder(
                batch.features,
                batch.feature_lengths,
                direction_mask=mask,
            )
            if encoded_lengths is None:
                encoded_lengths = torch.full(
                    (encoded.size(0),),
                    int(encoded.size(1)),
                    dtype=torch.long,
                    device=encoded.device,
                )

            target_offset = 0
            decoder_prompt_offset = 0
            for batch_idx, utt_id in enumerate(batch.utt_ids):
                encoded_length = int(encoded_lengths[batch_idx].item())
                sample_encoded = encoded[batch_idx : batch_idx + 1, :encoded_length, :]
                sample_lengths = encoded_lengths[batch_idx : batch_idx + 1]
                sample_max_new_tokens = _heuristic_max_new_tokens(
                    encoded_length,
                    explicit_max_new_tokens=max_new_tokens,
                    max_new_tokens_factor=max_new_tokens_factor,
                )
                generated_batch, score_batch, eos_flags = model.decoder_greedy_decode(
                    sample_encoded,
                    sample_lengths,
                    eos_token_id=eos_token_id,
                    max_new_tokens=sample_max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    decoder_prompt_before_audio=(
                        batch.decoder_prompt_before_audio[
                            decoder_prompt_offset : decoder_prompt_offset
                            + int(batch.decoder_prompt_before_audio_lengths[batch_idx].item())
                        ]
                        if batch.decoder_prompt_before_audio is not None
                        and batch.decoder_prompt_before_audio_lengths is not None
                        else None
                    ),
                    decoder_prompt_before_audio_lengths=(
                        batch.decoder_prompt_before_audio_lengths[batch_idx : batch_idx + 1]
                        if batch.decoder_prompt_before_audio_lengths is not None
                        else None
                    ),
                )
                if batch.decoder_prompt_before_audio_lengths is not None:
                    decoder_prompt_offset += int(batch.decoder_prompt_before_audio_lengths[batch_idx].item())
                pred_token_ids = [int(token_id) for token_id in generated_batch[0]]

                target_length = int(batch.target_lengths[batch_idx].item())
                ref_token_ids = [int(token_id) for token_id in batch.targets[target_offset : target_offset + target_length].tolist()]
                target_offset += target_length
                if ref_token_ids and ref_token_ids[-1] == eos_token_id:
                    ref_token_ids = ref_token_ids[:-1]

                pred_text = str(decode_fn(pred_token_ids)) if callable(decode_fn) else None
                raw_pred_text = pred_text
                fallback = False
                fallback_reason = None
                ctc_draft_cer_distance = None
                ctc_draft_length_ratio = None
                if ctc_draft_fallback_max_cer is not None and ctc_draft_cache:
                    draft_text = ctc_draft_cache.get(str(utt_id))
                    (
                        fallback,
                        fallback_reason,
                        ctc_draft_cer_distance,
                        ctc_draft_length_ratio,
                    ) = _ctc_draft_fallback_decision(
                        pred_text=pred_text,
                        draft_text=draft_text,
                        max_cer=float(ctc_draft_fallback_max_cer),
                        min_length_ratio=float(ctc_draft_fallback_min_length_ratio),
                        max_length_ratio=float(ctc_draft_fallback_max_length_ratio),
                        reject_repetition=bool(ctc_draft_fallback_reject_repetition),
                        metric_normalization=str(ctc_draft_fallback_metric_normalization),
                    )
                    if fallback:
                        pred_text = draft_text
                ref_text = batch.texts[batch_idx]
                if ref_text is None and callable(decode_fn):
                    ref_text = str(decode_fn(ref_token_ids))

                avg_logprob = float(score_batch[0].item())
                debug = RWKVDecoderDecodeDebug(
                    feature_length=int(batch.feature_lengths[batch_idx].item()),
                    encoded_length=encoded_length,
                    pred_token_count=len(pred_token_ids),
                    ref_token_count=len(ref_token_ids),
                    eos_emitted=bool(eos_flags[0]),
                    avg_logprob=avg_logprob,
                    ctc_draft_fallback=fallback,
                    ctc_draft_fallback_reason=fallback_reason,
                    ctc_draft_cer_distance=ctc_draft_cer_distance,
                    ctc_draft_length_ratio=ctc_draft_length_ratio,
                    raw_pred_text=raw_pred_text if fallback else None,
                )
                debug_rows.append(debug)
                predictions.append(
                    CTCLabeledPrediction(
                        utt_id=str(utt_id),
                        pred_token_ids=pred_token_ids,
                        ref_token_ids=ref_token_ids,
                        pred_text=pred_text,
                        ref_text=ref_text,
                        score=avg_logprob,
                        mode=config.mode,
                        alignments=[],
                        debug=None,
                        decode_strategy="rwkv_decoder_ar",
                        ctc_score=None,
                        decoder_score=avg_logprob,
                        combined_score=None,
                    )
                )
                if limit is not None and len(predictions) >= limit:
                    _maybe_report_prediction_progress(
                        kind="ar_sampling" if do_sample else "ar",
                        count=len(predictions),
                        total=progress_total,
                        started_at=progress_started_at,
                        last_reported=progress_last_reported,
                        interval=config.progress_interval,
                        final=True,
                    )
                    return predictions, debug_rows

                progress_last_reported = _maybe_report_prediction_progress(
                    kind="ar_sampling" if do_sample else "ar",
                    count=len(predictions),
                    total=progress_total,
                    started_at=progress_started_at,
                    last_reported=progress_last_reported,
                    interval=config.progress_interval,
                )

    _maybe_report_prediction_progress(
        kind="ar_sampling" if do_sample else "ar",
        count=len(predictions),
        total=progress_total,
        started_at=progress_started_at,
        last_reported=progress_last_reported,
        interval=config.progress_interval,
        final=True,
    )
    return predictions, debug_rows


def write_rwkv_decoder_labeled_predictions_jsonl(
    path: str | Path,
    predictions: list[CTCLabeledPrediction],
    debug_rows: list[RWKVDecoderDecodeDebug],
) -> Path:
    if len(predictions) != len(debug_rows):
        raise ValueError("predictions and debug_rows must have identical lengths.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction, debug in zip(predictions, debug_rows, strict=True):
            handle.write(
                json.dumps(
                    {
                        "utt_id": prediction.utt_id,
                        "pred_token_ids": prediction.pred_token_ids,
                        "ref_token_ids": prediction.ref_token_ids,
                        "pred_text": prediction.pred_text,
                        "ref_text": prediction.ref_text,
                        "score": prediction.score,
                        "decode_strategy": prediction.decode_strategy,
                        "ctc_score": prediction.ctc_score,
                        "decoder_score": prediction.decoder_score,
                        "combined_score": prediction.combined_score,
                        "mode": prediction.mode,
                        "alignments": [],
                        "debug": asdict(debug),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output_path
