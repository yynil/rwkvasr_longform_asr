#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from rwkvasr.eval.text_metrics import normalize_asr_text_for_metrics


SOURCE_LANGUAGES = {
    "librispeech": "en",
    "commonvoice_en": "en",
    "gigaspeech": "en",
    "aishell3": "zh",
    "commonvoice_cn": "zh",
    "wenetspeech": "zh",
}


SOURCE_FIELDS = (
    "_stage162_source",
    "_stage161_source",
    "_stage160_source",
    "_stage157_source",
    "_stage149_source",
    "_stage108_source",
    "_stage104_source",
    "_stage98_source",
    "_stage61_source",
    "_stage59_source",
    "_stage47_source",
    "_stage43_source",
    "_stage39_source",
    "_stage37_selector_source",
    "source_dataset",
    "source",
    "dataset",
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield line_number, json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def _source_from_row(row: dict[str, Any]) -> str:
    for field in SOURCE_FIELDS:
        value = str(row.get(field) or "").strip().lower()
        if value:
            return value
    shard_name = str(row.get("shard_name") or row.get("shard") or "").lower()
    if shard_name.startswith("gigaspeech") or shard_name.startswith("gsxl-"):
        return "gigaspeech"
    if shard_name.startswith("wenetspeech") or shard_name.startswith("wsl-"):
        return "wenetspeech"
    if shard_name.startswith("librispeech"):
        return "librispeech"
    if shard_name.startswith("commonvoice_en"):
        return "commonvoice_en"
    if shard_name.startswith("commonvoice_cn"):
        return "commonvoice_cn"
    if shard_name.startswith("aishell3"):
        return "aishell3"
    return "unknown"


def _row_id(row: dict[str, Any]) -> str:
    for field in ("utt_id", "id", "key"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    audio_path = str(row.get("audio_path") or "").strip()
    if audio_path:
        return Path(audio_path).stem
    raise ValueError("row has no utt_id/id/key/audio_path")


def _load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for _, row in _iter_jsonl(path):
        utt_id = row.get("utt_id")
        if utt_id is not None:
            completed.add(str(utt_id))
    return completed


def _ctc_argmax_text(auto_model, audio_path: str, key: str, *, language: str | None) -> dict[str, Any]:
    model = auto_model.model
    kwargs = dict(auto_model.kwargs)
    tokenizer = kwargs.pop("tokenizer")
    frontend = kwargs.pop("frontend")
    kwargs["disable_pbar"] = True
    prompt = model.get_prompt([], language, True)
    chatml = model.generate_chatml(prompt, audio_path)
    with torch.no_grad():
        _, _, _, _, meta_data = model.inference_prepare(
            [chatml],
            key=[key],
            tokenizer=tokenizer,
            frontend=frontend,
            **kwargs,
        )
        encoder_out = meta_data["encoder_out"]
        encoder_out_lens = meta_data["encoder_out_lens"]
        decoder_out, decoder_out_lens = model.ctc_decoder(encoder_out, encoder_out_lens)
        ctc_logp = model.ctc.log_softmax(decoder_out)
        frame_count = int(decoder_out_lens[0].item())
        x = ctc_logp[0, :frame_count, :]
        top1_logp, yseq = x.max(dim=-1)
        unique = torch.unique_consecutive(yseq, dim=-1)
        token_ids = unique[unique != model.blank_id].detach().cpu().tolist()
        text = model.ctc_tokenizer.decode(token_ids).replace("<|nospeech|>", "").strip()
        blank_top1 = (yseq == model.blank_id).to(torch.float32).mean().item() if frame_count else 0.0
        avg_top1_logp = top1_logp.to(torch.float32).mean().item() if frame_count else 0.0
        avg_blank_prob = x[:, model.blank_id].exp().to(torch.float32).mean().item() if frame_count else 0.0
    return {
        "funasr_ctc_text": text,
        "funasr_ctc_token_ids": token_ids,
        "funasr_ctc_num_tokens": len(token_ids),
        "funasr_ctc_num_frames": frame_count,
        "funasr_ctc_blank_id": int(model.blank_id),
        "funasr_ctc_vocab_size": int(x.shape[-1]),
        "funasr_ctc_blank_top1_ratio": float(blank_top1),
        "funasr_ctc_avg_top1_logp": float(avg_top1_logp),
        "funasr_ctc_avg_blank_prob": float(avg_blank_prob),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export FunASR-Nano CTC argmax text from a length-index JSONL."
    )
    parser.add_argument("--length-index-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    length_index_path = Path(args.length_index_path)
    output_jsonl = Path(args.output_jsonl)
    summary_path = Path(args.summary_path) if args.summary_path else output_jsonl.with_suffix(".summary.json")
    if output_jsonl.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{output_jsonl} exists; pass --resume or --overwrite")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    completed = _load_completed(output_jsonl) if args.resume else set()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    from funasr import AutoModel

    auto_model = AutoModel(
        model=str(args.model_path),
        trust_remote_code=True,
        device=str(args.device),
        disable_update=True,
    )
    if auto_model.model.ctc_decoder is None:
        raise RuntimeError("FunASR-Nano ctc_decoder is missing; use the current ModelScope model.pt")

    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    output_mode = "a" if args.resume else "w"
    selected = 0
    written = 0
    with output_jsonl.open(output_mode, encoding="utf-8") as handle:
        for _, row in _iter_jsonl(length_index_path):
            counts["input_rows"] += 1
            split = str(row.get("split") or "train")
            if args.split != "all" and split != str(args.split):
                continue
            utt_id = _row_id(row)
            if utt_id in completed:
                counts["skipped_completed"] += 1
                continue
            if args.limit is not None and selected >= int(args.limit):
                break
            selected += 1
            source = _source_from_row(row)
            source_counts[source] += 1
            language = SOURCE_LANGUAGES.get(source)
            audio_path = str(row.get("audio_path") or "").strip()
            out = {
                "utt_id": utt_id,
                "key": row.get("key"),
                "source": source,
                "language": language,
                "split": split,
                "audio_path": audio_path,
            }
            try:
                if not audio_path:
                    raise ValueError("row has no audio_path")
                if not Path(audio_path).exists():
                    raise FileNotFoundError(audio_path)
                out.update(_ctc_argmax_text(auto_model, audio_path, utt_id, language=language))
                out["funasr_ctc_text_tn"] = normalize_asr_text_for_metrics(
                    out["funasr_ctc_text"],
                    language=language,
                    normalization="ctc",
                )
                out["funasr_ctc_error"] = None
                counts["ok"] += 1
            except Exception as exc:  # pragma: no cover - runtime diagnostics
                out["funasr_ctc_text"] = None
                out["funasr_ctc_text_tn"] = None
                out["funasr_ctc_error"] = repr(exc)
                error_counts[type(exc).__name__] += 1
                counts["error"] += 1
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if written % 20 == 0:
                handle.flush()
            if args.progress_interval > 0 and written % int(args.progress_interval) == 0:
                print(
                    f"[funasr-nano-ctc] written={written} ok={counts['ok']} error={counts['error']}",
                    flush=True,
                )
    summary = {
        "version": 1,
        "length_index_path": str(length_index_path),
        "output_jsonl": str(output_jsonl),
        "model_path": str(args.model_path),
        "device": str(args.device),
        "split": str(args.split),
        "limit": args.limit,
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "funasr_ctc_blank_id": int(auto_model.model.blank_id),
        "funasr_ctc_vocab_size": int(auto_model.model.ctc.ctc_lo.weight.shape[0]),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_jsonl} rows={written} ok={counts['ok']} error={counts['error']}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
