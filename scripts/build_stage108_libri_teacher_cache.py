#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from rwkvasr.eval.text_metrics import (
    edit_distance,
    normalize_asr_text_for_metrics,
    tokenize_for_cer,
    tokenize_for_wer,
)


DEFAULT_ROOT = "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix"
DEFAULT_STAGE22_LENGTHS = (
    DEFAULT_ROOT + "/stages/stage22_public90_hard10_from_stage21_soup/webdataset_lengths.jsonl"
)
DEFAULT_OUTPUT_DIR = "/media/usbhd/rwkvasr_runs/stage108_expanded_libri_teacher_20260621"
DEFAULT_SELECTOR_LENGTHS = (
    "/media/usbhd/rwkvasr_runs/stage37_ctc_decode_audit_20260620/"
    "stage30_selector256.lengths.jsonl"
)
DEFAULT_PRIOR_TEACHER_PATHS = (
    "/media/usbhd/rwkvasr_runs/stage39_large_teacher_repair_20260620/stage39_teacher_accepted.cache.jsonl",
    "/media/usbhd/training_data/asr/curriculum/stage22_stage21_soup_clean_repair_mix/"
    "stages/stage104_stage95_libri_teacher_guard_repair/stage104_libri_teacher_sampled.cache.jsonl",
    "/media/usbhd/rwkvasr_runs/stage32_teacher_label_repair_probe_20260620/stage32_teacher510.cache.jsonl",
)
DEFAULT_STAGE_PREFIX = "stage108"
DEFAULT_SOURCE = "librispeech"
SOURCE_LANGUAGES = {
    "librispeech": "en",
    "commonvoice_en": "en",
    "gigaspeech": "en",
    "aishell3": "zh",
    "commonvoice_cn": "zh",
    "wenetspeech": "zh",
}

SOURCE_FIELDS = (
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
NAME_FIELDS = ("shard_name", "key", "audio_member", "json_member", "utt_id", "id")


def _validate_source(source: str) -> str:
    canonical = _canonical_source(source) or source
    if canonical not in SOURCE_LANGUAGES:
        raise ValueError(f"unsupported source={source!r}; expected one of {sorted(SOURCE_LANGUAGES)}")
    return canonical


def _stage_prefix(args: argparse.Namespace) -> str:
    prefix = str(getattr(args, "stage_prefix", DEFAULT_STAGE_PREFIX)).strip().lstrip("_")
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError(f"invalid stage prefix: {prefix!r}")
    return prefix


def _artifact_prefix(args: argparse.Namespace) -> str:
    prefix = _stage_prefix(args)
    source = _validate_source(str(getattr(args, "source", DEFAULT_SOURCE)))
    if prefix == "stage108" and source == "librispeech":
        return "stage108_libri"
    return f"{prefix}_{source}"


def _is_legacy_stage108_libri(args: argparse.Namespace) -> bool:
    return _stage_prefix(args) == "stage108" and _validate_source(str(args.source)) == "librispeech"


def _audit_key(args: argparse.Namespace, name: str) -> str:
    return f"_{_stage_prefix(args)}_{name}"


def _audit_value(row: dict[str, Any], args: argparse.Namespace, name: str, *fallbacks: str) -> Any:
    keys = [_audit_key(args, name), f"_stage108_{name}", *fallbacks]
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _utt_id(row: dict[str, Any]) -> str:
    return str(row.get("utt_id") or row.get("id") or row.get("key") or "")


def _canonical_source(raw: Any) -> str | None:
    value = str(raw or "").lower()
    if not value:
        return None
    if "librispeech" in value or "libri" in value:
        return "librispeech"
    if "aishell" in value:
        return "aishell3"
    if "commonvoice_en" in value or "commonvoice-en" in value or "cv_en" in value:
        return "commonvoice_en"
    if "commonvoice_cn" in value or "commonvoice-cn" in value or "cv_cn" in value:
        return "commonvoice_cn"
    if "gigaspeech" in value or "gsxl" in value or "gs-" in value:
        return "gigaspeech"
    if "wenetspeech" in value or "wenet" in value or "wsl-" in value:
        return "wenetspeech"
    return None


def _infer_source(row: dict[str, Any]) -> str:
    for field in SOURCE_FIELDS:
        source = _canonical_source(row.get(field))
        if source:
            return source
    for field in NAME_FIELDS:
        source = _canonical_source(row.get(field))
        if source:
            return source
    return "unknown"


def _source_language(source: str) -> str:
    return SOURCE_LANGUAGES[_validate_source(source)]


def _load_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            utt_id = _utt_id(row)
            if utt_id:
                ids.add(utt_id)
    return ids


def _sample_candidates(args: argparse.Namespace) -> None:
    rng = random.Random(int(args.seed))
    source_filter = _validate_source(str(args.source))
    artifact_prefix = _artifact_prefix(args)
    length_path = Path(args.length_index)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{artifact_prefix}_candidate.lengths.jsonl"
    summary_path = output_dir / f"{artifact_prefix}_candidate.summary.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")

    exclude_paths = [Path(path) for path in args.exclude_ids_path]
    prior_defaults = (
        []
        if bool(args.disable_default_prior_teacher_cache) or source_filter != DEFAULT_SOURCE
        else list(DEFAULT_PRIOR_TEACHER_PATHS)
    )
    prior_paths = [Path(path) for path in [*prior_defaults, *(args.prior_teacher_cache or [])]]
    exclude_ids = _load_ids(exclude_paths + prior_paths)
    target = int(args.candidate_count)
    max_frames = int(args.max_frames)
    seen = 0
    skipped_source = 0
    skipped_excluded = 0
    skipped_long = 0
    reservoir: list[dict[str, Any]] = []

    for row in _iter_jsonl(length_path):
        source = _infer_source(row)
        if source != source_filter:
            skipped_source += 1
            continue
        utt_id = _utt_id(row)
        if utt_id in exclude_ids:
            skipped_excluded += 1
            continue
        frames = int(row.get("num_frames", 0) or 0)
        if max_frames > 0 and frames > max_frames:
            skipped_long += 1
            continue
        seen += 1
        enriched = dict(row)
        enriched[_audit_key(args, "source")] = source_filter
        enriched[_audit_key(args, "component")] = "candidate"
        enriched[_audit_key(args, "selector")] = f"expanded_{source_filter}_teacher"
        enriched[_audit_key(args, "seed")] = int(args.seed)
        if len(reservoir) < target:
            reservoir.append(enriched)
            continue
        replace_index = rng.randrange(seen)
        if replace_index < target:
            reservoir[replace_index] = enriched

    rng.shuffle(reservoir)
    _write_jsonl(output_path, reservoir)
    part_count = max(1, int(args.parts))
    for part in range(part_count):
        part_rows = reservoir[part::part_count]
        _write_jsonl(
            output_dir / f"{artifact_prefix}_candidate.part{part:02d}.lengths.jsonl",
            part_rows,
        )
    summary = {
        "version": 1,
        "stage_prefix": _stage_prefix(args),
        "source": source_filter,
        "language": _source_language(source_filter),
        "seed": int(args.seed),
        "length_index": str(length_path),
        "output_path": str(output_path),
        "candidate_count": len(reservoir),
        "parts": part_count,
        "eligible_seen": seen,
        "excluded_ids": len(exclude_ids),
        "skipped_source": skipped_source,
        "skipped_excluded": skipped_excluded,
        "skipped_long": skipped_long,
        "max_frames": max_frames,
        "unique_utt_ids": len({_utt_id(row) for row in reservoir}),
        "prior_teacher_cache": [str(path) for path in prior_paths],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} rows={len(reservoir)}")
    print(f"wrote {summary_path}")


def _load_length_by_utt(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        keys = [
            row.get("utt_id"),
            row.get("id"),
            row.get("key"),
            row.get("audio_member"),
            row.get("json_member"),
        ]
        audio_member = str(row.get("audio_member") or "")
        if audio_member:
            keys.append(Path(audio_member).stem)
        json_member = str(row.get("json_member") or "")
        if json_member:
            keys.append(Path(json_member).stem)
        for key in keys:
            key_text = str(key or "").strip()
            if key_text:
                rows[key_text] = row
    return rows


def _sample_error(pred_text: str, ref_text: str, *, language: str) -> tuple[float, float]:
    pred_norm = normalize_asr_text_for_metrics(pred_text, language=language, normalization="ctc")
    ref_norm = normalize_asr_text_for_metrics(ref_text, language=language, normalization="ctc")
    pred_words = tokenize_for_wer(pred_norm)
    ref_words = tokenize_for_wer(ref_norm)
    pred_chars = tokenize_for_cer(pred_norm)
    ref_chars = tokenize_for_cer(ref_norm)
    wer = edit_distance(pred_words, ref_words) / float(max(1, len(ref_words)))
    cer = edit_distance(pred_chars, ref_chars) / float(max(1, len(ref_chars)))
    return wer, cer


def _score_predictions(args: argparse.Namespace) -> None:
    source_filter = _validate_source(str(args.source))
    language = _source_language(source_filter)
    artifact_prefix = _artifact_prefix(args)
    primary_metric = str(getattr(args, "primary_metric", "wer")).lower()
    if primary_metric not in {"wer", "cer", "max"}:
        raise ValueError(f"unsupported primary_metric={primary_metric!r}")
    candidate_path = Path(args.candidate_lengths)
    output_dir = Path(args.output_dir)
    scored_name = (
        "stage108_stage104_candidate.scored.jsonl"
        if _is_legacy_stage108_libri(args)
        else f"{artifact_prefix}_stage104_candidate.scored.jsonl"
    )
    scored_path = output_dir / scored_name
    selected_path = output_dir / f"{artifact_prefix}_teacher_targets.lengths.jsonl"
    summary_path = output_dir / f"{artifact_prefix}_student_selection.summary.json"
    if selected_path.exists() and not args.overwrite:
        raise FileExistsError(f"{selected_path} exists; pass --overwrite")

    length_by_utt = _load_length_by_utt(candidate_path)
    scored_rows: list[dict[str, Any]] = []
    missing_lengths = 0
    for pred_path_raw in args.prediction_jsonl:
        pred_path = Path(pred_path_raw)
        for pred in _iter_jsonl(pred_path):
            utt_id = _utt_id(pred)
            length_row = length_by_utt.get(utt_id)
            if length_row is None:
                missing_lengths += 1
                continue
            pred_text = str(pred.get("pred_text") or "")
            ref_text = str(pred.get("ref_text") or "")
            student_wer, student_cer = _sample_error(pred_text, ref_text, language=language)
            if primary_metric == "cer":
                primary_error = student_cer
            elif primary_metric == "max":
                primary_error = max(student_wer, student_cer)
            else:
                primary_error = student_wer
            scored = dict(length_row)
            scored[_audit_key(args, "component")] = "student_scored"
            scored[_audit_key(args, "source")] = source_filter
            scored[_audit_key(args, "student_wer")] = student_wer
            scored[_audit_key(args, "student_cer")] = student_cer
            scored[_audit_key(args, "primary_error")] = primary_error
            scored[_audit_key(args, "primary_metric")] = primary_metric
            scored[_audit_key(args, "pred_text")] = pred_text
            scored[_audit_key(args, "ref_text")] = ref_text
            scored_rows.append(scored)

    scored_rows.sort(
        key=lambda row: (
            float(row.get(_audit_key(args, "primary_error"), 0.0) or 0.0),
            float(row.get(_audit_key(args, "student_cer"), 0.0) or 0.0),
            int(row.get("num_frames", 0) or 0),
        ),
        reverse=True,
    )
    min_error = float(args.min_primary_error)
    selected_rows = [
        row
        for row in scored_rows
        if float(row.get(_audit_key(args, "primary_error"), 0.0) or 0.0) >= min_error
    ][: int(args.target_count)]
    for row in selected_rows:
        row[_audit_key(args, "component")] = "teacher_target"

    _write_jsonl(scored_path, scored_rows)
    _write_jsonl(selected_path, selected_rows)
    errors = [float(row.get(_audit_key(args, "primary_error"), 0.0) or 0.0) for row in scored_rows]
    selected_errors = [
        float(row.get(_audit_key(args, "primary_error"), 0.0) or 0.0) for row in selected_rows
    ]
    summary = {
        "version": 1,
        "stage_prefix": _stage_prefix(args),
        "source": source_filter,
        "language": language,
        "candidate_lengths": str(candidate_path),
        "prediction_jsonl": [str(Path(p)) for p in args.prediction_jsonl],
        "scored_path": str(scored_path),
        "selected_path": str(selected_path),
        "scored_rows": len(scored_rows),
        "selected_rows": len(selected_rows),
        "missing_lengths": missing_lengths,
        "min_primary_error": min_error,
        "primary_metric": primary_metric,
        "primary_error_avg": sum(errors) / len(errors) if errors else 0.0,
        "selected_primary_error_avg": sum(selected_errors) / len(selected_errors) if selected_errors else 0.0,
        "selected_primary_error_min": min(selected_errors) if selected_errors else 0.0,
        "selected_primary_error_max": max(selected_errors) if selected_errors else 0.0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {scored_path} rows={len(scored_rows)}")
    print(f"wrote {selected_path} rows={len(selected_rows)}")


def _read_audio_bytes(root: Path, row: dict[str, Any]) -> tuple[bytes, Path]:
    shard_name = str(row.get("shard_name") or "")
    audio_member = str(row.get("audio_member") or "")
    if not shard_name or not audio_member:
        raise ValueError(f"row missing shard/audio member for utt_id={_utt_id(row)}")
    tar_path = root / shard_name
    offset = row.get("audio_offset")
    size = row.get("audio_size")
    if offset is not None and size is not None:
        with tar_path.open("rb") as handle:
            handle.seek(int(offset))
            return handle.read(int(size)), tar_path
    with tarfile.open(tar_path, "r") as archive:
        extracted = archive.extractfile(audio_member)
        if extracted is None:
            raise FileNotFoundError(f"{audio_member} not found in {tar_path}")
        return extracted.read(), tar_path


def _read_json_metadata(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    shard_name = str(row.get("shard_name") or "")
    json_member = str(row.get("json_member") or "")
    if not shard_name or not json_member:
        raise ValueError(f"row missing shard/json member for utt_id={_utt_id(row)}")
    tar_path = root / shard_name
    offset = row.get("json_offset")
    size = row.get("json_size")
    if offset is not None and size is not None:
        with tar_path.open("rb") as handle:
            handle.seek(int(offset))
            raw = handle.read(int(size))
    else:
        with tarfile.open(tar_path, "r") as archive:
            extracted = archive.extractfile(json_member)
            if extracted is None:
                raise FileNotFoundError(f"{json_member} not found in {tar_path}")
            raw = extracted.read()
    metadata = json.loads(raw.decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata JSON is not an object for utt_id={_utt_id(row)}")
    return metadata


def _metadata_text(metadata: dict[str, Any]) -> str:
    for key in ("text", "sentence", "transcript", "normalized_text", "raw_text"):
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _recover_text(args: argparse.Namespace) -> None:
    artifact_prefix = _artifact_prefix(args)
    root = Path(args.root)
    candidate_path = Path(args.candidate_lengths)
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{artifact_prefix}_candidate.text.lengths.jsonl"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    rows: list[dict[str, Any]] = []
    missing_text = 0
    for row in _iter_jsonl(candidate_path):
        metadata = _read_json_metadata(root, row)
        text = _metadata_text(metadata)
        if not text:
            missing_text += 1
        enriched = dict(row)
        if text:
            enriched["text"] = text
        if "language" not in enriched and metadata.get("language") is not None:
            enriched["language"] = metadata.get("language")
        enriched[_audit_key(args, "text_recovered")] = bool(text)
        rows.append(enriched)
    _write_jsonl(output_path, rows)
    part_count = max(1, int(args.parts))
    for part in range(part_count):
        _write_jsonl(
            output_dir / f"{artifact_prefix}_candidate.text.part{part:02d}.lengths.jsonl",
            rows[part::part_count],
        )
    summary_path = output_dir / f"{artifact_prefix}_candidate.text.summary.json"
    summary = {
        "version": 1,
        "stage_prefix": _stage_prefix(args),
        "source": _validate_source(str(args.source)),
        "root": str(root),
        "candidate_lengths": str(candidate_path),
        "output_path": str(output_path),
        "rows": len(rows),
        "missing_text": missing_text,
        "parts": part_count,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path} rows={len(rows)} missing_text={missing_text}")
    print(f"wrote {summary_path}")


def _extract_audio(args: argparse.Namespace) -> None:
    artifact_prefix = _artifact_prefix(args)
    root = Path(args.root)
    target_path = Path(args.target_lengths)
    output_dir = Path(args.output_dir)
    legacy = _is_legacy_stage108_libri(args)
    audio_dir = output_dir / ("audio_teacher_targets" if legacy else f"{artifact_prefix}_audio_teacher_targets")
    extracted_path = output_dir / (
        "stage108_teacher_targets.extracted.jsonl"
        if legacy
        else f"{artifact_prefix}_teacher_targets.extracted.jsonl"
    )
    if extracted_path.exists() and not args.overwrite:
        raise FileExistsError(f"{extracted_path} exists; pass --overwrite")
    rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(target_path):
        utt_id = _utt_id(row)
        audio_bytes, tar_path = _read_audio_bytes(root, row)
        suffix = Path(str(row.get("audio_member") or "")).suffix or ".flac"
        audio_path = audio_dir / f"{utt_id}{suffix}"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(audio_bytes)
        out = dict(row)
        out["audio_path"] = str(audio_path)
        out["tar_path"] = str(tar_path)
        rows.append(out)
    _write_jsonl(extracted_path, rows)
    if int(args.parts) > 1:
        for part in range(int(args.parts)):
            _write_jsonl(
                output_dir
                / (
                    f"stage108_teacher_targets.part{part:02d}.extracted.jsonl"
                    if legacy
                    else f"{artifact_prefix}_teacher_targets.part{part:02d}.extracted.jsonl"
                ),
                rows[part:: int(args.parts)],
            )
    summary_path = output_dir / (
        "stage108_teacher_targets.extraction.summary.json"
        if legacy
        else f"{artifact_prefix}_teacher_targets.extraction.summary.json"
    )
    summary = {
        "version": 1,
        "stage_prefix": _stage_prefix(args),
        "source": _validate_source(str(args.source)),
        "root": str(root),
        "target_lengths": str(target_path),
        "extracted_path": str(extracted_path),
        "audio_dir": str(audio_dir),
        "rows": len(rows),
        "parts": int(args.parts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {extracted_path} rows={len(rows)}")


def _run_teacher(args: argparse.Namespace) -> None:
    from funasr import AutoModel

    input_path = Path(args.extracted_jsonl)
    output_path = Path(args.output_jsonl)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModel(
        model=str(args.model_path),
        trust_remote_code=True,
        device=str(args.device),
        disable_update=True,
    )
    rows = list(_iter_jsonl(input_path))
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            out = dict(row)
            try:
                result = model.generate(
                    input=str(row["audio_path"]),
                    batch_size_s=float(args.batch_size_s),
                    disable_pbar=True,
                )
                item = result[0] if isinstance(result, list) and result else {}
                out["teacher_raw_text"] = item.get("text")
                out["teacher_text_tn"] = item.get("text_tn") or item.get("text")
                out["funasr_key"] = item.get("key")
                out["teacher_error"] = None
            except Exception as exc:  # pragma: no cover - defensive runtime logging
                out["teacher_raw_text"] = None
                out["teacher_text_tn"] = None
                out["funasr_key"] = None
                out["teacher_error"] = repr(exc)
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            if args.progress_interval and index % int(args.progress_interval) == 0:
                print(f"[stage108-teacher] {index}/{len(rows)}", flush=True)
    print(f"wrote {output_path} rows={len(rows)}")


def _filter_teacher(args: argparse.Namespace) -> None:
    source_filter = _validate_source(str(args.source))
    language = _source_language(source_filter)
    artifact_prefix = _artifact_prefix(args)
    improvement_metric = str(getattr(args, "improvement_metric", "wer")).lower()
    if improvement_metric not in {"wer", "cer", "max"}:
        raise ValueError(f"unsupported improvement_metric={improvement_metric!r}")
    output_dir = Path(args.output_dir)
    cache_path = output_dir / f"{artifact_prefix}_teacher_accepted.cache.jsonl"
    lengths_path = output_dir / f"{artifact_prefix}_teacher_accepted.lengths.jsonl"
    summary_path = output_dir / f"{artifact_prefix}_teacher_filter.summary.json"
    if cache_path.exists() and not args.overwrite:
        raise FileExistsError(f"{cache_path} exists; pass --overwrite")

    accepted_cache: list[dict[str, Any]] = []
    accepted_lengths: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    teacher_rows = 0
    for teacher_path_raw in args.teacher_jsonl:
        for row in _iter_jsonl(Path(teacher_path_raw)):
            teacher_rows += 1
            utt_id = _utt_id(row)
            teacher_text = str(row.get("teacher_text_tn") or "").strip()
            ref_text = str(_audit_value(row, args, "ref_text", "ref_text") or "")
            student_text = str(_audit_value(row, args, "pred_text", "pred_text") or "")
            if row.get("teacher_error") is not None:
                rejected["teacher_runtime_error"] += 1
                continue
            if not teacher_text:
                rejected["empty_teacher"] += 1
                continue
            student_wer = float(_audit_value(row, args, "student_wer") or 0.0)
            student_cer = float(_audit_value(row, args, "student_cer") or 0.0)
            teacher_wer, teacher_cer = _sample_error(teacher_text, ref_text, language=language)
            wer_improvement = student_wer - teacher_wer
            cer_improvement = student_cer - teacher_cer
            if improvement_metric == "cer":
                primary_improvement = cer_improvement
            elif improvement_metric == "max":
                primary_improvement = max(wer_improvement, cer_improvement)
            else:
                primary_improvement = wer_improvement
            if teacher_wer > float(args.max_teacher_wer):
                rejected["teacher_wer"] += 1
                continue
            if teacher_cer > float(args.max_teacher_cer):
                rejected["teacher_cer"] += 1
                continue
            if primary_improvement < float(args.min_improvement):
                rejected["insufficient_improvement"] += 1
                continue
            cache_row = {
                "utt_id": utt_id,
                "teacher_text_tn": teacher_text,
                "teacher_raw_text": row.get("teacher_raw_text") or teacher_text,
                "source": source_filter,
                "student_pred_text": student_text,
                "ref_text": ref_text,
                "student_wer": student_wer,
                "student_cer": student_cer,
                "teacher_wer": teacher_wer,
                "teacher_cer": teacher_cer,
                "wer_improvement": wer_improvement,
                "cer_improvement": cer_improvement,
                "improvement_metric": improvement_metric,
                "primary_improvement": primary_improvement,
            }
            length_row = dict(row)
            length_row[_audit_key(args, "component")] = f"{source_filter}_teacher_accepted"
            length_row[_audit_key(args, "source")] = source_filter
            length_row[_audit_key(args, "teacher_text_tn")] = teacher_text
            length_row[_audit_key(args, "teacher_wer")] = teacher_wer
            length_row[_audit_key(args, "teacher_cer")] = teacher_cer
            length_row[_audit_key(args, "wer_improvement")] = wer_improvement
            length_row[_audit_key(args, "cer_improvement")] = cer_improvement
            length_row[_audit_key(args, "improvement_metric")] = improvement_metric
            length_row[_audit_key(args, "primary_improvement")] = primary_improvement
            accepted_cache.append(cache_row)
            accepted_lengths.append(length_row)

    _write_jsonl(cache_path, accepted_cache)
    _write_jsonl(lengths_path, accepted_lengths)
    if accepted_cache:
        avg_student_wer = sum(float(r["student_wer"]) for r in accepted_cache) / len(accepted_cache)
        avg_teacher_wer = sum(float(r["teacher_wer"]) for r in accepted_cache) / len(accepted_cache)
        avg_student_cer = sum(float(r["student_cer"]) for r in accepted_cache) / len(accepted_cache)
        avg_teacher_cer = sum(float(r["teacher_cer"]) for r in accepted_cache) / len(accepted_cache)
    else:
        avg_student_wer = avg_teacher_wer = avg_student_cer = avg_teacher_cer = 0.0
    summary = {
        "version": 1,
        "stage_prefix": _stage_prefix(args),
        "source": source_filter,
        "language": language,
        "teacher_jsonl": [str(Path(path)) for path in args.teacher_jsonl],
        "teacher_rows": teacher_rows,
        "accepted_rows": len(accepted_cache),
        "rejected_reasons": dict(sorted(rejected.items())),
        "cache_path": str(cache_path),
        "lengths_path": str(lengths_path),
        "avg_student_wer": avg_student_wer,
        "avg_teacher_wer": avg_teacher_wer,
        "avg_student_cer": avg_student_cer,
        "avg_teacher_cer": avg_teacher_cer,
        "min_improvement": float(args.min_improvement),
        "improvement_metric": improvement_metric,
        "max_teacher_wer": float(args.max_teacher_wer),
        "max_teacher_cer": float(args.max_teacher_cer),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {cache_path} rows={len(accepted_cache)}")
    print(f"wrote {lengths_path} rows={len(accepted_lengths)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage108 expanded Libri teacher-cache pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample-candidates")
    sample.add_argument("--length-index", default=DEFAULT_STAGE22_LENGTHS)
    sample.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    sample.add_argument("--stage-prefix", default=DEFAULT_STAGE_PREFIX)
    sample.add_argument("--source", default=DEFAULT_SOURCE)
    sample.add_argument("--candidate-count", type=int, default=4096)
    sample.add_argument("--max-frames", type=int, default=1800)
    sample.add_argument("--parts", type=int, default=4)
    sample.add_argument("--seed", type=int, default=20260621)
    sample.add_argument("--exclude-ids-path", action="append", default=[DEFAULT_SELECTOR_LENGTHS])
    sample.add_argument("--prior-teacher-cache", action="append", default=None)
    sample.add_argument("--disable-default-prior-teacher-cache", action="store_true")
    sample.add_argument("--overwrite", action="store_true")
    sample.set_defaults(func=_sample_candidates)

    score = subparsers.add_parser("score-predictions")
    score.add_argument("--candidate-lengths", default=str(Path(DEFAULT_OUTPUT_DIR) / "stage108_libri_candidate.lengths.jsonl"))
    score.add_argument("--prediction-jsonl", action="append", required=True)
    score.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    score.add_argument("--stage-prefix", default=DEFAULT_STAGE_PREFIX)
    score.add_argument("--source", default=DEFAULT_SOURCE)
    score.add_argument("--target-count", type=int, default=1024)
    score.add_argument("--min-primary-error", type=float, default=0.20)
    score.add_argument("--primary-metric", choices=("wer", "cer", "max"), default="wer")
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(func=_score_predictions)

    recover = subparsers.add_parser("recover-text")
    recover.add_argument("--root", default=DEFAULT_ROOT)
    recover.add_argument(
        "--candidate-lengths",
        default=str(Path(DEFAULT_OUTPUT_DIR) / "stage108_libri_candidate.lengths.jsonl"),
    )
    recover.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    recover.add_argument("--stage-prefix", default=DEFAULT_STAGE_PREFIX)
    recover.add_argument("--source", default=DEFAULT_SOURCE)
    recover.add_argument("--parts", type=int, default=4)
    recover.add_argument("--overwrite", action="store_true")
    recover.set_defaults(func=_recover_text)

    extract = subparsers.add_parser("extract-audio")
    extract.add_argument("--root", default=DEFAULT_ROOT)
    extract.add_argument("--target-lengths", default=str(Path(DEFAULT_OUTPUT_DIR) / "stage108_libri_teacher_targets.lengths.jsonl"))
    extract.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    extract.add_argument("--stage-prefix", default=DEFAULT_STAGE_PREFIX)
    extract.add_argument("--source", default=DEFAULT_SOURCE)
    extract.add_argument("--parts", type=int, default=4)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(func=_extract_audio)

    teacher = subparsers.add_parser("run-teacher")
    teacher.add_argument("--extracted-jsonl", required=True)
    teacher.add_argument("--output-jsonl", required=True)
    teacher.add_argument("--model-path", default="assets/fun-asr-nano-2512")
    teacher.add_argument("--device", default="cuda:0")
    teacher.add_argument("--batch-size-s", type=float, default=60.0)
    teacher.add_argument("--progress-interval", type=int, default=50)
    teacher.add_argument("--overwrite", action="store_true")
    teacher.set_defaults(func=_run_teacher)

    filt = subparsers.add_parser("filter-teacher")
    filt.add_argument("--teacher-jsonl", action="append", required=True)
    filt.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    filt.add_argument("--stage-prefix", default=DEFAULT_STAGE_PREFIX)
    filt.add_argument("--source", default=DEFAULT_SOURCE)
    filt.add_argument("--min-improvement", type=float, default=0.10)
    filt.add_argument("--improvement-metric", choices=("wer", "cer", "max"), default="wer")
    filt.add_argument("--max-teacher-wer", type=float, default=0.20)
    filt.add_argument("--max-teacher-cer", type=float, default=0.15)
    filt.add_argument("--overwrite", action="store_true")
    filt.set_defaults(func=_filter_teacher)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
