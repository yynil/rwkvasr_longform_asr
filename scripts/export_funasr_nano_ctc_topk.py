#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch


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


def _safe_tensor_name(utt_id: str) -> str:
    digest = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()
    return f"{digest}.pt"


def _safe_audio_name(utt_id: str, audio_member: str | None) -> str:
    digest = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()
    suffix = Path(str(audio_member or "")).suffix
    return f"{digest}{suffix or '.audio'}"


def _relative_to_jsonl_parent(path: Path, jsonl_parent: Path) -> str:
    try:
        return str(path.resolve().relative_to(jsonl_parent.resolve()))
    except ValueError:
        return str(path.resolve())


def _int_rows(tensor: torch.Tensor) -> list[list[int]]:
    return [[int(value) for value in row] for row in tensor.cpu().tolist()]


def _float_rows(tensor: torch.Tensor, precision: int | None) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in tensor.cpu().tolist():
        if precision is None:
            rows.append([float(value) for value in row])
        else:
            rows.append([round(float(value), precision) for value in row])
    return rows


def _float_list(tensor: torch.Tensor, precision: int | None) -> list[float]:
    values = tensor.cpu().tolist()
    if precision is None:
        return [float(value) for value in values]
    return [round(float(value), precision) for value in values]


def _load_shard_paths(webdataset_index_path: Path | None) -> dict[str, Path]:
    if webdataset_index_path is None:
        return {}
    data = json.loads(webdataset_index_path.read_text(encoding="utf-8"))
    shards = data.get("shards") if isinstance(data, dict) else None
    if not isinstance(shards, list):
        raise ValueError(f"webdataset index has no shards list: {webdataset_index_path}")
    paths: dict[str, Path] = {}
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        name = str(shard.get("name") or "").strip()
        if not name:
            continue
        raw_path = str(shard.get("path") or shard.get("tar_path") or "").strip()
        if raw_path:
            path = Path(raw_path)
        else:
            source_root = str(shard.get("source_root") or data.get("root") or "").strip()
            if not source_root:
                continue
            path = Path(source_root) / name
        paths[name] = path
    return paths


def _resolve_audio_path(
    row: dict[str, Any],
    utt_id: str,
    *,
    shard_paths: dict[str, Path],
    audio_cache_dir: Path | None,
) -> str:
    audio_path = str(row.get("audio_path") or "").strip()
    if audio_path:
        path = Path(audio_path)
        if path.exists():
            return str(path)
        if audio_cache_dir is None:
            raise FileNotFoundError(audio_path)

    if audio_cache_dir is None:
        raise ValueError("row has no audio_path")

    tar_path_value = str(row.get("tar_path") or row.get("shard_path") or "").strip()
    if not tar_path_value:
        shard_name = str(row.get("shard_name") or row.get("shard") or "").strip()
        if shard_name:
            tar_path_value = str(shard_paths.get(shard_name) or "")
    if not tar_path_value:
        raise ValueError("row has no audio_path or resolvable tar_path")

    tar_path = Path(tar_path_value)
    if not tar_path.exists():
        raise FileNotFoundError(str(tar_path))

    audio_member = str(row.get("audio_member") or "").strip()
    if not audio_member:
        raise ValueError("row has no audio_member for tar-backed audio")

    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_cache_dir / _safe_audio_name(utt_id, audio_member)
    audio_size_raw = row.get("audio_size")
    expected_size = int(audio_size_raw) if audio_size_raw is not None else None
    if out_path.exists() and (expected_size is None or out_path.stat().st_size == expected_size):
        return str(out_path)

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    audio_offset_raw = row.get("audio_offset")
    if audio_offset_raw is not None and expected_size is not None:
        with tar_path.open("rb") as source, tmp_path.open("wb") as target:
            source.seek(int(audio_offset_raw))
            target.write(source.read(expected_size))
    else:
        with tarfile.open(tar_path, "r:*") as tar:
            member = tar.extractfile(audio_member)
            if member is None:
                raise FileNotFoundError(f"{audio_member} in {tar_path}")
            with member, tmp_path.open("wb") as target:
                target.write(member.read())
    tmp_path.replace(out_path)
    return str(out_path)


def _ctc_topk(
    auto_model,
    audio_path: str,
    key: str,
    *,
    language: str | None,
    top_k: int,
    project_blank_id: int,
    project_ignored_token_ids: list[int],
) -> dict[str, Any]:
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
        x = ctc_logp[0, :frame_count, :].float()
        vocab_size = int(x.shape[-1])
        k = min(max(1, int(top_k)), vocab_size)
        topk_log_probs, topk_ids = torch.topk(x, k=k, dim=-1)
        teacher_blank_id = int(model.blank_id)
        blank_log_probs = x[:, teacher_blank_id]
        blank_in_topk = topk_ids.eq(teacher_blank_id).any(dim=-1)
        if frame_count > 0:
            topk_ids = topk_ids.clone()
            topk_log_probs = topk_log_probs.clone()
            topk_ids[~blank_in_topk, -1] = teacher_blank_id
            topk_log_probs[~blank_in_topk, -1] = blank_log_probs[~blank_in_topk]
        mapped_ids = topk_ids.clone()
        mapped_ids[mapped_ids == teacher_blank_id] = int(project_blank_id)

        top1_logp, yseq = x.max(dim=-1)
        unique = torch.unique_consecutive(yseq, dim=-1)
        token_ids = unique[unique != teacher_blank_id].detach().cpu().tolist()
        text = model.ctc_tokenizer.decode(token_ids).replace("<|nospeech|>", "").strip()
        blank_top1 = (yseq == teacher_blank_id).to(torch.float32).mean().item() if frame_count else 0.0
        avg_top1_logp = top1_logp.to(torch.float32).mean().item() if frame_count else 0.0
        avg_blank_prob = blank_log_probs.exp().to(torch.float32).mean().item() if frame_count else 0.0

    return {
        "format": "funasr_nano_ctc_topk_v1",
        "teacher": "FunASR-Nano-2512",
        "topk": k,
        "num_frames": frame_count,
        "teacher_blank_id": teacher_blank_id,
        "teacher_vocab_size": vocab_size,
        "project_blank_id": int(project_blank_id),
        "project_ignored_token_ids": [int(value) for value in project_ignored_token_ids],
        "topk_token_ids_tensor": mapped_ids.cpu().to(dtype=torch.int32),
        "topk_log_probs_tensor": topk_log_probs.cpu().to(dtype=torch.float32),
        "blank_log_probs_tensor": blank_log_probs.cpu().to(dtype=torch.float32),
        "funasr_ctc_text": text,
        "funasr_ctc_token_ids": [int(value) for value in token_ids],
        "funasr_ctc_blank_top1_ratio": float(blank_top1),
        "funasr_ctc_avg_top1_logp": float(avg_top1_logp),
        "funasr_ctc_avg_blank_prob": float(avg_blank_prob),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export mapped FunASR-Nano CTC frame top-k log-probs for project CTC distillation."
    )
    parser.add_argument("--length-index-path", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--webdataset-index-path", default=None)
    parser.add_argument("--audio-cache-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--project-blank-id", type=int, default=60515)
    parser.add_argument("--project-ignored-token-id", action="append", type=int, default=None)
    parser.add_argument("--float-precision", type=int, default=6)
    parser.add_argument("--store-tensors-dir", default=None)
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
    if int(args.num_shards) <= 0:
        raise ValueError("--num-shards must be positive")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")

    project_ignored_token_ids = args.project_ignored_token_id or [60514]
    completed = _load_completed(output_jsonl) if args.resume else set()
    seen_utt_ids = set(completed)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_dir = Path(args.store_tensors_dir) if args.store_tensors_dir else None
    if tensor_dir is not None:
        tensor_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = _load_shard_paths(Path(args.webdataset_index_path)) if args.webdataset_index_path else {}
    audio_cache_dir = Path(args.audio_cache_dir) if args.audio_cache_dir else None

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
    frame_counts: Counter[str] = Counter()
    output_mode = "a" if args.resume else "w"
    eligible_index = 0
    selected = 0
    written = 0
    precision = None if int(args.float_precision) < 0 else int(args.float_precision)
    with output_jsonl.open(output_mode, encoding="utf-8") as handle:
        for _, row in _iter_jsonl(length_index_path):
            counts["input_rows"] += 1
            split = str(row.get("split") or "train")
            if args.split != "all" and split != str(args.split):
                continue
            current_eligible_index = eligible_index
            eligible_index += 1
            if current_eligible_index % int(args.num_shards) != int(args.shard_index):
                counts["skipped_other_shard"] += 1
                continue
            utt_id = _row_id(row)
            if utt_id in seen_utt_ids:
                if utt_id in completed:
                    counts["skipped_completed"] += 1
                else:
                    counts["skipped_duplicate_utt_id"] += 1
                continue
            seen_utt_ids.add(utt_id)
            if args.limit is not None and selected >= int(args.limit):
                break
            selected += 1
            source = _source_from_row(row)
            source_counts[source] += 1
            language = SOURCE_LANGUAGES.get(source)
            out = {
                "utt_id": utt_id,
                "key": row.get("key"),
                "source": source,
                "language": language,
                "split": split,
                "audio_path": str(row.get("audio_path") or "").strip(),
            }
            try:
                audio_path = _resolve_audio_path(
                    row,
                    utt_id,
                    shard_paths=shard_paths,
                    audio_cache_dir=audio_cache_dir,
                )
                out["audio_path"] = audio_path
                topk = _ctc_topk(
                    auto_model,
                    audio_path,
                    utt_id,
                    language=language,
                    top_k=int(args.top_k),
                    project_blank_id=int(args.project_blank_id),
                    project_ignored_token_ids=[int(value) for value in project_ignored_token_ids],
                )
                topk_ids = topk.pop("topk_token_ids_tensor")
                topk_log_probs = topk.pop("topk_log_probs_tensor")
                blank_log_probs = topk.pop("blank_log_probs_tensor")
                out.update(topk)
                if tensor_dir is None:
                    out["topk_token_ids"] = _int_rows(topk_ids)
                    out["topk_log_probs"] = _float_rows(topk_log_probs, precision)
                    out["blank_log_probs"] = _float_list(blank_log_probs, precision)
                else:
                    tensor_path = tensor_dir / _safe_tensor_name(utt_id)
                    torch.save(
                        {
                            "utt_id": utt_id,
                            "topk_token_ids": topk_ids,
                            "topk_log_probs": topk_log_probs,
                            "blank_log_probs": blank_log_probs,
                        },
                        tensor_path,
                    )
                    out["topk_tensor_path"] = _relative_to_jsonl_parent(tensor_path, output_jsonl.parent)
                out["error"] = None
                counts["ok"] += 1
                frame_counts["total_frames"] += int(out["num_frames"])
                frame_counts["max_frames"] = max(frame_counts["max_frames"], int(out["num_frames"]))
            except Exception as exc:  # pragma: no cover - runtime diagnostics
                out["error"] = repr(exc)
                error_counts[type(exc).__name__] += 1
                counts["error"] += 1
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if written % 20 == 0:
                handle.flush()
            if args.progress_interval > 0 and written % int(args.progress_interval) == 0:
                print(
                    f"[funasr-nano-ctc-topk] written={written} ok={counts['ok']} error={counts['error']}",
                    flush=True,
                )

    ok = int(counts["ok"])
    summary = {
        "version": 1,
        "length_index_path": str(length_index_path),
        "output_jsonl": str(output_jsonl),
        "tensor_dir": str(tensor_dir) if tensor_dir is not None else None,
        "model_path": str(args.model_path),
        "webdataset_index_path": str(args.webdataset_index_path) if args.webdataset_index_path else None,
        "audio_cache_dir": str(audio_cache_dir) if audio_cache_dir is not None else None,
        "device": str(args.device),
        "split": str(args.split),
        "limit": args.limit,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "top_k": int(args.top_k),
        "project_blank_id": int(args.project_blank_id),
        "project_ignored_token_ids": [int(value) for value in project_ignored_token_ids],
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "funasr_ctc_blank_id": int(auto_model.model.blank_id),
        "funasr_ctc_vocab_size": int(auto_model.model.ctc.ctc_lo.weight.shape[0]),
        "total_frames": int(frame_counts["total_frames"]),
        "max_frames": int(frame_counts["max_frames"]),
        "avg_frames": (float(frame_counts["total_frames"]) / ok) if ok > 0 else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_jsonl} rows={written} ok={counts['ok']} error={counts['error']}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
