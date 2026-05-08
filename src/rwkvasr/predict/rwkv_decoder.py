from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rwkvasr.data import build_text_tokenizer
from rwkvasr.modules import build_inference_direction_mask

from .ctc import (
    CTCLabeledPrediction,
    PredictionConfig,
    _build_labeled_prediction_loader,
    _load_prediction_model,
)


@dataclass(frozen=True)
class RWKVDecoderDecodeDebug:
    feature_length: int
    encoded_length: int
    pred_token_count: int
    ref_token_count: int | None
    eos_emitted: bool
    avg_logprob: float


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


def predict_rwkv_decoder_labeled(
    config: PredictionConfig,
    *,
    limit: int | None = None,
    max_new_tokens: int | None = None,
    max_new_tokens_factor: float = 2.0,
) -> tuple[list[CTCLabeledPrediction], list[RWKVDecoderDecodeDebug]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided.")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1 when provided.")
    if max_new_tokens_factor <= 0:
        raise ValueError("max_new_tokens_factor must be > 0.")

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
    loader = _build_labeled_prediction_loader(config)

    predictions: list[CTCLabeledPrediction] = []
    debug_rows: list[RWKVDecoderDecodeDebug] = []
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
                )
                pred_token_ids = [int(token_id) for token_id in generated_batch[0]]

                target_length = int(batch.target_lengths[batch_idx].item())
                ref_token_ids = [int(token_id) for token_id in batch.targets[target_offset : target_offset + target_length].tolist()]
                target_offset += target_length
                if ref_token_ids and ref_token_ids[-1] == eos_token_id:
                    ref_token_ids = ref_token_ids[:-1]

                pred_text = str(decode_fn(pred_token_ids)) if callable(decode_fn) else None
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
                    return predictions, debug_rows
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
