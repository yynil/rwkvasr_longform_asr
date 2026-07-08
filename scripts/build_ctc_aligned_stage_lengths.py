#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from rwkvasr.data import build_text_tokenizer
from rwkvasr.data.text_normalization import normalize_asr_text
from rwkvasr.eval.text_metrics import edit_distance, tokenize_for_cer, tokenize_for_wer


def _log(message: str) -> None:
    print(f"[rwkvasr] {message}", flush=True)


def _safe_name(value: str | None) -> str:
    value = (value or "unknown").strip().lower() or "unknown"
    value = re.sub(r"[^a-z0-9._+-]+", "_", value)
    return value.strip("_") or "unknown"


def _source_from_entry(entry: dict[str, Any], metadata: dict[str, Any]) -> str:
    source = metadata.get("source_dataset")
    if source:
        return _safe_name(str(source))
    shard_name = str(entry.get("shard_name") or "")
    if shard_name:
        return _safe_name(shard_name.split("_", 1)[0].split("-", 1)[0])
    return "unknown"


def _language_from_metadata(metadata: dict[str, Any]) -> str:
    return _safe_name(str(metadata.get("language") or "unknown"))


def _sample_id_from_entry(entry: dict[str, Any], metadata: dict[str, Any], utt_id_key: str) -> str:
    candidate_keys = [utt_id_key, "utt_id", "id", "audio_id", "sid", "key"]
    seen: set[str] = set()
    for key_name in candidate_keys:
        if key_name in seen:
            continue
        seen.add(key_name)
        value = metadata.get(key_name)
        if value is not None:
            return str(value)
    for key_name in ("sample_id", "key", "id", "audio_member", "json_member"):
        value = entry.get(key_name)
        if value is not None:
            return str(value)
    return ""


def _load_label_cache(paths: list[str], *, text_key: str) -> dict[str, str]:
    cache: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in teacher cache {path}:{line_number}") from exc
                text = row.get(text_key)
                if text is None and text_key != "pred_text":
                    text = row.get("pred_text")
                if text is None:
                    text = row.get("teacher_text_tn") or row.get("ctc_text") or row.get("text")
                if text is None:
                    continue
                for key_name in ("utt_id", "id", "audio_id", "sid", "key", "sample_id"):
                    value = row.get(key_name)
                    if value is not None:
                        cache[str(value)] = str(text)
    return cache


def _resolve_cached_label_text(
    cache: dict[str, str],
    *,
    sample_id: str,
    entry: dict[str, Any],
    metadata: dict[str, Any],
) -> str | None:
    candidate_keys: list[str] = [sample_id]
    for key_name in ("utt_id", "id", "audio_id", "sid", "key"):
        value = metadata.get(key_name)
        if value is not None:
            candidate_keys.append(str(value))
    for key_name in ("sample_id", "key", "id", "audio_member", "json_member"):
        value = entry.get(key_name)
        if value is not None:
            candidate_keys.append(str(value))
    seen: set[str] = set()
    for candidate in candidate_keys:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in cache:
            return cache[candidate]
    return None


def _label_error_rates(candidate_text: str, reference_text: str) -> tuple[float, float]:
    candidate_words = tokenize_for_wer(candidate_text)
    reference_words = tokenize_for_wer(reference_text)
    if reference_words:
        wer = edit_distance(candidate_words, reference_words) / len(reference_words)
    else:
        wer = 0.0 if not candidate_words else 1.0
    candidate_chars = tokenize_for_cer(candidate_text)
    reference_chars = tokenize_for_cer(reference_text)
    if reference_chars:
        cer = edit_distance(candidate_chars, reference_chars) / len(reference_chars)
    else:
        cer = 0.0 if not candidate_chars else 1.0
    return float(wer), float(cer)


def _aut_conv2d8_out_length(num_frames: int) -> int:
    length = int(num_frames)
    for _ in range(3):
        length = (length + 1) // 2
    return max(0, length)


def _sensevoice_lfr6_out_length(num_frames: int) -> int:
    length = int(num_frames)
    if length <= 0:
        return 0
    return (length + 5) // 6


def _frontend_logit_length(num_frames: int, frontend_downsample: str) -> int:
    if frontend_downsample == "aut_conv2d8":
        return _aut_conv2d8_out_length(num_frames)
    if frontend_downsample == "sensevoice_lfr6":
        return _sensevoice_lfr6_out_length(num_frames)
    if frontend_downsample == "none":
        return int(num_frames)
    raise ValueError(f"Unsupported frontend_downsample: {frontend_downsample}")


def _ctc_required_frames(token_ids: list[int]) -> int:
    adjacent_repeats = sum(1 for left, right in zip(token_ids, token_ids[1:]) if int(left) == int(right))
    return int(len(token_ids) + adjacent_repeats)


def _tokenizer_unknown_id(tokenizer: Any) -> int | None:
    processor = getattr(tokenizer, "processor", None)
    unk_id = getattr(processor, "unk_id", None)
    if callable(unk_id):
        value = int(unk_id())
        return value if value >= 0 else None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is None:
        return None
    return int(unk_token_id)


class TarJsonReader:
    def __init__(self, shard_root: Path):
        self.shard_root = shard_root
        self._shard_name: str | None = None
        self._binary: BinaryIO | None = None
        self._archive: tarfile.TarFile | None = None

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        if self._binary is not None:
            self._binary.close()
            self._binary = None
        self._shard_name = None

    def _switch(self, shard_name: str) -> None:
        if shard_name == self._shard_name:
            return
        self.close()
        self._binary = (self.shard_root / shard_name).open("rb")
        self._shard_name = shard_name

    def read_json(self, entry: dict[str, Any]) -> dict[str, Any]:
        shard_name = str(entry["shard_name"])
        self._switch(shard_name)
        assert self._binary is not None
        offset = entry.get("json_offset")
        size = entry.get("json_size")
        if offset is not None and size is not None:
            self._binary.seek(int(offset))
            payload = self._binary.read(int(size))
            if len(payload) != int(size):
                raise EOFError(f"Short JSON read for {shard_name}:{entry.get('json_member')}")
            return json.loads(payload.decode("utf-8"))

        if self._archive is None:
            assert self._shard_name is not None
            self._archive = tarfile.open(self.shard_root / self._shard_name, "r")
        extracted = self._archive.extractfile(str(entry["json_member"]))
        if extracted is None:
            raise FileNotFoundError(f"Missing JSON member {shard_name}:{entry.get('json_member')}")
        return json.loads(extracted.read().decode("utf-8"))


def _iter_length_entries(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a WebDataset length index filtered to samples that are alignable by CTC."
    )
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--length-index-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-type", default="sentencepiece")
    parser.add_argument("--tokenizer-model-path", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--text-normalization", default="ctc")
    parser.add_argument(
        "--frontend-downsample",
        choices=("aut_conv2d8", "sensevoice_lfr6", "none"),
        default="aut_conv2d8",
    )
    parser.add_argument("--min-logit-required-ratio", type=float, default=1.0)
    parser.add_argument("--drop-unk-token", action="store_true")
    parser.add_argument("--unk-token-id", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.005)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--split-by", default="sample_id")
    parser.add_argument("--utt-id-key", default="id")
    parser.add_argument("--label-cache-path", default=None)
    parser.add_argument("--label-cache-text-key", default="ctc_text")
    parser.add_argument("--label-cache-include", choices=("kept", "teacher"), default="kept")
    parser.add_argument("--teacher-cache-path", action="append", default=[])
    parser.add_argument("--teacher-text-key", default="teacher_text_tn")
    parser.add_argument("--prefer-teacher-labels", action="store_true")
    parser.add_argument("--require-teacher-label", action="store_true")
    parser.add_argument("--max-teacher-wer-vs-original", type=float, default=-1.0)
    parser.add_argument("--max-teacher-cer-vs-original", type=float, default=-1.0)
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shard_root = Path(args.shard_root)
    length_index_path = Path(args.length_index_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_text_tokenizer(args.tokenizer_type, model_path=args.tokenizer_model_path)
    unknown_id = int(args.unk_token_id) if args.unk_token_id is not None else _tokenizer_unknown_id(tokenizer)
    teacher_cache = _load_label_cache(
        [str(path) for path in (args.teacher_cache_path or [])],
        text_key=str(args.teacher_text_key),
    )
    reader = TarJsonReader(shard_root)
    output_lengths_path = output_dir / "webdataset_lengths.jsonl"
    summary_path = output_dir / "webdataset_lengths.summary.json"
    index_path = output_dir / "webdataset_index.json"
    label_cache_path = Path(args.label_cache_path) if args.label_cache_path else None
    label_cache_output = None
    if label_cache_path is not None:
        label_cache_path.parent.mkdir(parents=True, exist_ok=True)
        label_cache_output = label_cache_path.open("w", encoding="utf-8")

    counts: dict[str, Counter[Any]] = defaultdict(Counter)
    shard_counts: dict[str, Counter[str]] = defaultdict(Counter)
    kept = 0
    processed = 0
    started = time.monotonic()

    try:
        with output_lengths_path.open("w", encoding="utf-8") as output:
            for entry in _iter_length_entries(length_index_path):
                if args.max_samples and processed >= int(args.max_samples):
                    break
                processed += 1
                split = str(entry.get("split") or "unknown")
                counts["input_split"][split] += 1
                try:
                    metadata = reader.read_json(entry)
                    language = _language_from_metadata(metadata)
                    source = _source_from_entry(entry, metadata)
                    sample_id = _sample_id_from_entry(entry, metadata, str(args.utt_id_key))
                    text = metadata.get(str(args.text_key))
                    teacher_text = _resolve_cached_label_text(
                        teacher_cache,
                        sample_id=sample_id,
                        entry=entry,
                        metadata=metadata,
                    )
                    if bool(args.require_teacher_label) and teacher_text is None:
                        counts["drop_reason"]["missing_teacher_label"] += 1
                        counts["drop_source"][source] += 1
                        counts["drop_language"][language] += 1
                        continue
                    use_teacher = teacher_text is not None and (
                        bool(args.prefer_teacher_labels) or bool(args.require_teacher_label)
                    )
                    if text is None and not use_teacher:
                        counts["drop_reason"]["missing_text"] += 1
                        continue
                    original_normalized_text = (
                        normalize_asr_text(
                            str(text),
                            language=None if language == "unknown" else language,
                            mode=str(args.text_normalization),
                        )
                        if text is not None
                        else ""
                    )
                    label_text = teacher_text if use_teacher else text
                    label_source = "teacher" if use_teacher else "metadata"
                    normalized_text = normalize_asr_text(
                        str(label_text or ""),
                        language=None if language == "unknown" else language,
                        mode=str(args.text_normalization),
                    )
                    teacher_original_wer = None
                    teacher_original_cer = None
                    if use_teacher:
                        if not normalized_text:
                            counts["drop_reason"]["empty_teacher_target"] += 1
                            counts["drop_source"][source] += 1
                            counts["drop_language"][language] += 1
                            continue
                        if original_normalized_text:
                            teacher_original_wer, teacher_original_cer = _label_error_rates(
                                normalized_text,
                                original_normalized_text,
                            )
                            if (
                                float(args.max_teacher_wer_vs_original) >= 0.0
                                and teacher_original_wer > float(args.max_teacher_wer_vs_original)
                            ):
                                counts["drop_reason"]["teacher_wer_vs_original"] += 1
                                counts["drop_source"][source] += 1
                                counts["drop_language"][language] += 1
                                continue
                            if (
                                float(args.max_teacher_cer_vs_original) >= 0.0
                                and teacher_original_cer > float(args.max_teacher_cer_vs_original)
                            ):
                                counts["drop_reason"]["teacher_cer_vs_original"] += 1
                                counts["drop_source"][source] += 1
                                counts["drop_language"][language] += 1
                                continue
                    token_ids = tokenizer.encode(normalized_text)
                    if not token_ids:
                        counts["drop_reason"]["empty_target"] += 1
                        continue
                    unk_count = 0
                    if unknown_id is not None:
                        unk_count = sum(1 for token_id in token_ids if int(token_id) == int(unknown_id))
                    if bool(args.drop_unk_token) and unk_count > 0:
                        counts["drop_reason"]["unk_token"] += 1
                        counts["drop_source"][source] += 1
                        counts["drop_language"][language] += 1
                        counts["unk_source"][source] += unk_count
                        counts["unk_language"][language] += unk_count
                        continue
                    required_frames = _ctc_required_frames(token_ids)
                    num_frames = int(entry["num_frames"])
                    logit_frames = _frontend_logit_length(num_frames, str(args.frontend_downsample))
                    ratio = float(logit_frames) / max(1, required_frames)
                    if required_frames > logit_frames:
                        counts["drop_reason"]["ctc_unalignable"] += 1
                        counts["drop_source"][source] += 1
                        counts["drop_language"][language] += 1
                        continue
                    if ratio < float(args.min_logit_required_ratio):
                        counts["drop_reason"]["below_ratio"] += 1
                        counts["drop_source"][source] += 1
                        counts["drop_language"][language] += 1
                        continue

                    row = {
                        **entry,
                        "source_dataset": source,
                        "language": language,
                        "num_text_tokens": len(token_ids),
                        "normalized_text_chars": len(normalized_text),
                        "ctc_num_tokens": len(token_ids),
                        "ctc_adjacent_repeats": required_frames - len(token_ids),
                        "ctc_unk_tokens": unk_count,
                        "ctc_required_frames": required_frames,
                        "ctc_logit_frames": logit_frames,
                        "ctc_logit_required_ratio": round(ratio, 6),
                        "ctc_label_source": label_source,
                    }
                    if teacher_original_wer is not None:
                        row["teacher_original_wer"] = round(float(teacher_original_wer), 6)
                    if teacher_original_cer is not None:
                        row["teacher_original_cer"] = round(float(teacher_original_cer), 6)
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if label_cache_output is not None and (
                        str(args.label_cache_include) == "kept" or label_source == "teacher"
                    ):
                        label_cache_row = {
                            "utt_id": sample_id,
                            "source": source,
                            "language": language,
                            "ctc_label_source": label_source,
                            "raw_text": None if text is None else str(text),
                            str(args.label_cache_text_key): normalized_text,
                        }
                        if str(args.label_cache_text_key) != "ctc_text":
                            label_cache_row["ctc_text"] = normalized_text
                        if teacher_text is not None:
                            label_cache_row["teacher_text"] = str(teacher_text)
                        if teacher_original_wer is not None:
                            label_cache_row["teacher_original_wer"] = round(
                                float(teacher_original_wer),
                                6,
                            )
                        if teacher_original_cer is not None:
                            label_cache_row["teacher_original_cer"] = round(
                                float(teacher_original_cer),
                                6,
                            )
                        label_cache_output.write(
                            json.dumps(label_cache_row, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                    kept += 1
                    shard_name = str(row["shard_name"])
                    shard_counts[shard_name][split] += 1
                    counts["split"][split] += 1
                    counts["source"][source] += 1
                    counts["language"][language] += 1
                    counts["label_source"][label_source] += 1
                    counts["split_source"][(split, source)] += 1
                    counts["split_language"][(split, language)] += 1
                except Exception as exc:
                    counts["drop_reason"][f"error:{type(exc).__name__}"] += 1
                    _log(
                        f"skipped CTC-align sample shard={entry.get('shard_name')} "
                        f"key={entry.get('key')}: {type(exc).__name__}: {exc}"
                    )
                if args.progress_every > 0 and processed % int(args.progress_every) == 0:
                    elapsed = time.monotonic() - started
                    _log(
                        f"ctc-align progress processed={processed} kept={kept} "
                        f"elapsed={elapsed:.1f}s current={entry.get('shard_name')}"
                    )
    finally:
        reader.close()
        if label_cache_output is not None:
            label_cache_output.close()

    split_counts = dict(sorted(counts["split"].items()))
    index_payload = {
        "version": 1,
        "root": str(output_dir),
        "shard_pattern": "*.tar",
        "num_shards": len(shard_counts),
        "num_samples": int(kept),
        "split": {
            "type": "stable_hash",
            "split_by": str(args.split_by),
            "train_name": "train",
            "eval_name": "eval",
            "eval_ratio": float(args.eval_ratio),
            "hash_seed": int(args.hash_seed),
            "utt_id_key": str(args.utt_id_key),
        },
        "splits": {
            "train": {"num_samples": int(split_counts.get("train", 0))},
            "eval": {"num_samples": int(split_counts.get("eval", 0))},
        },
        "shards": [
            {
                "name": shard_name,
                "num_samples": int(counter["train"] + counter["eval"]),
                "assigned_split": None,
                "splits": {
                    "train": {"num_samples": int(counter["train"])},
                    "eval": {"num_samples": int(counter["eval"])},
                },
            }
            for shard_name, counter in sorted(shard_counts.items())
        ],
    }
    summary_payload = {
        "version": 1,
        "source_root": str(shard_root),
        "source_length_index_path": str(length_index_path),
        "output_dir": str(output_dir),
        "length_index_path": str(output_lengths_path),
        "index_path": str(index_path),
        "tokenizer_type": str(args.tokenizer_type),
        "tokenizer_model_path": str(Path(args.tokenizer_model_path).resolve()),
        "text_key": str(args.text_key),
        "text_normalization": str(args.text_normalization),
        "frontend_downsample": str(args.frontend_downsample),
        "min_logit_required_ratio": float(args.min_logit_required_ratio),
        "drop_unk_token": bool(args.drop_unk_token),
        "unk_token_id": unknown_id,
        "label_cache_path": None if label_cache_path is None else str(label_cache_path),
        "label_cache_text_key": str(args.label_cache_text_key),
        "label_cache_include": str(args.label_cache_include),
        "teacher_cache_paths": [str(path) for path in (args.teacher_cache_path or [])],
        "teacher_text_key": str(args.teacher_text_key),
        "prefer_teacher_labels": bool(args.prefer_teacher_labels),
        "require_teacher_label": bool(args.require_teacher_label),
        "teacher_cache_keys": len(teacher_cache),
        "max_teacher_wer_vs_original": float(args.max_teacher_wer_vs_original),
        "max_teacher_cer_vs_original": float(args.max_teacher_cer_vs_original),
        "num_input_samples": int(processed),
        "num_kept_samples": int(kept),
        "num_dropped_samples": int(processed - kept),
        "counts": {
            "input_by_split": dict(sorted(counts["input_split"].items())),
            "kept_by_split": split_counts,
            "kept_by_source": dict(sorted(counts["source"].items())),
            "kept_by_language": dict(sorted(counts["language"].items())),
            "kept_by_label_source": dict(sorted(counts["label_source"].items())),
            "kept_by_split_source": {
                f"{split}/{source}": value for (split, source), value in sorted(counts["split_source"].items())
            },
            "kept_by_split_language": {
                f"{split}/{language}": value
                for (split, language), value in sorted(counts["split_language"].items())
            },
            "dropped_by_reason": dict(sorted(counts["drop_reason"].items())),
            "dropped_by_source": dict(sorted(counts["drop_source"].items())),
            "dropped_by_language": dict(sorted(counts["drop_language"].items())),
            "dropped_unk_tokens_by_source": dict(sorted(counts["unk_source"].items())),
            "dropped_unk_tokens_by_language": dict(sorted(counts["unk_language"].items())),
        },
    }
    _write_json(index_path, index_payload)
    _write_json(summary_path, summary_payload)
    _log(
        f"ctc-align lengths complete processed={processed} kept={kept} "
        f"lengths={output_lengths_path} summary={summary_path}"
        + ("" if label_cache_path is None else f" label_cache={label_cache_path}")
    )


if __name__ == "__main__":
    main()
