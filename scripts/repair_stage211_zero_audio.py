from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import soundfile as sf

from rwkvasr.eval.text_metrics import normalize_asr_text_for_metrics


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _find_bucket_row(part_path: Path, sample_key: str) -> tuple[int, bytes, dict[str, Any]]:
    match: tuple[int, bytes, dict[str, Any]] | None = None
    with part_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            row_key = str(row.get("key") or row.get("utt_id") or row.get("id") or "")
            if row_key != sample_key:
                continue
            if match is not None:
                raise ValueError(
                    f"Duplicate sample key {sample_key!r} in bucket part {part_path}"
                )
            match = (line_number, raw_line.rstrip(b"\r\n"), row)
    if match is None:
        raise ValueError(f"Sample key {sample_key!r} not found in bucket part {part_path}")
    return match


def _read_indexed_bytes(
    path: Path,
    *,
    member_name: str,
    offset: int | None,
    size: int | None,
) -> bytes:
    if offset is not None and size is not None:
        with path.open("rb") as source:
            source.seek(offset)
            payload = source.read(size)
        if len(payload) != size:
            raise EOFError(
                f"Short read for {path}:{member_name}; expected={size} actual={len(payload)}"
            )
        return payload
    with tarfile.open(path, "r") as archive:
        extracted = archive.extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Missing tar member {path}:{member_name}")
        return extracted.read()


def _tar_info(name: str, payload_size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = payload_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _build_repair_tar(
    *,
    output_path: Path,
    audio_member: str,
    audio_payload: bytes,
    json_member: str,
    json_payload: bytes,
) -> dict[str, dict[str, int]]:
    if not audio_payload:
        raise ValueError("Repaired waveform payload must be non-empty.")
    if not json_payload:
        raise ValueError("Source metadata payload must be non-empty.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing repair shard: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(temporary, "w", format=tarfile.GNU_FORMAT) as archive:
            archive.addfile(
                _tar_info(audio_member, len(audio_payload)),
                io.BytesIO(audio_payload),
            )
            archive.addfile(
                _tar_info(json_member, len(json_payload)),
                io.BytesIO(json_payload),
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    members: dict[str, dict[str, int]] = {}
    with tarfile.open(output_path, "r") as archive:
        for name in (audio_member, json_member):
            info = archive.getmember(name)
            members[name] = {"offset": int(info.offset_data), "size": int(info.size)}
    if members[audio_member]["size"] <= 0:
        raise ValueError("Repair shard contains a non-positive audio member.")
    return members


def _validate_nano_verification(
    *,
    sample_key: str,
    report_path: Path,
    predictions_path: Path,
    language: str,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report.get("sample_count") or 0) != 1:
        raise ValueError("Nano repair verification must contain exactly one sample.")
    metric_name = "avg_wer" if language == "en" else "avg_cer"
    metric = float((report.get("metrics") or {}).get(metric_name, float("inf")))
    if metric != 0.0:
        raise ValueError(
            f"Nano repair verification requires normalized {metric_name}=0.0, got {metric}"
        )
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 1 or str(rows[0].get("utt_id") or "") != sample_key:
        raise ValueError("Nano repair prediction does not bind the requested sample key.")
    prediction = rows[0]
    normalized_prediction = normalize_asr_text_for_metrics(
        str(prediction.get("pred_text") or ""),
        language=language,
        normalization="ctc",
    )
    normalized_reference = normalize_asr_text_for_metrics(
        str(prediction.get("ref_text") or ""),
        language=language,
        normalization="ctc",
    )
    if normalized_prediction != normalized_reference:
        raise ValueError("Nano repair prediction and reference differ after CTC normalization.")
    checkpoint_path = Path(str(report.get("model_checkpoint_path") or "")).resolve()
    checkpoint_sha256 = str(report.get("model_checkpoint_sha256") or "")
    if (
        not checkpoint_path.is_file()
        or len(checkpoint_sha256) != 64
        or _sha256_file(checkpoint_path) != checkpoint_sha256
    ):
        raise ValueError("Nano repair verification checkpoint is missing or changed.")
    return {
        "report_path": str(report_path.resolve()),
        "report_sha256": _sha256_file(report_path),
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": _sha256_file(predictions_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "metric": metric_name,
        "error_rate": metric,
        "prediction": str(prediction.get("pred_text") or ""),
        "reference": str(prediction.get("ref_text") or ""),
        "normalized_text": normalized_reference,
        "logit_frames": int((prediction.get("debug") or {}).get("logit_length") or 0),
    }


def _verify_source_parquet_zero_audio(
    *,
    path: Path,
    row_group: int | None,
    row_index: int | None,
    expected_source_id: str,
) -> dict[str, Any]:
    if row_group is None or row_index is None:
        raise ValueError(
            "Source Parquet verification requires row-group and row-index values."
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to verify the zero-byte source row.") from exc
    parquet = pq.ParquetFile(path)
    if row_group < 0 or row_group >= parquet.num_row_groups:
        raise ValueError(f"Source Parquet row group is out of range: {row_group}")
    table = parquet.read_row_group(
        row_group,
        columns=("segment_id", "audio", "begin_time", "end_time", "audio_id"),
    )
    if row_index < 0 or row_index >= table.num_rows:
        raise ValueError(f"Source Parquet row index is out of range: {row_index}")
    segment_id = str(table["segment_id"][row_index].as_py() or "")
    if segment_id != expected_source_id:
        raise ValueError(
            "Source Parquet segment id mismatch: "
            f"expected={expected_source_id!r} actual={segment_id!r}"
        )
    audio = table["audio"][row_index].as_py() or {}
    audio_bytes = audio.get("bytes") or b""
    if len(audio_bytes) != 0:
        raise ValueError(
            "Stage211 source repair is only valid for a zero-byte Parquet payload: "
            f"segment_id={segment_id!r} audio_bytes={len(audio_bytes)}"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "row_group": row_group,
        "row_index_in_group": row_index,
        "segment_id": segment_id,
        "audio_path": str(audio.get("path") or ""),
        "audio_bytes_observed": 0,
        "begin_time": float(table["begin_time"][row_index].as_py()),
        "end_time": float(table["end_time"][row_index].as_py()),
        "audio_id": str(table["audio_id"][row_index].as_py() or ""),
    }


def _patch_bucket_part(
    *,
    part_path: Path,
    line_number: int,
    expected_line: bytes,
    repaired_row: dict[str, Any],
) -> tuple[str, str, bytes]:
    before_sha256 = _sha256_file(part_path)
    rendered = json.dumps(repaired_row, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    temporary = part_path.with_name(f".{part_path.name}.tmp-{os.getpid()}")
    replaced = False
    try:
        with part_path.open("rb") as source, temporary.open("wb") as output:
            for current_line_number, raw_line in enumerate(source, start=1):
                if current_line_number == line_number:
                    if raw_line.rstrip(b"\r\n") != expected_line:
                        raise ValueError(
                            f"Bucket part changed before repair at line {line_number}: {part_path}"
                        )
                    output.write(rendered + b"\n")
                    replaced = True
                else:
                    output.write(raw_line)
        if not replaced:
            raise ValueError(f"Bucket part ended before repair line {line_number}: {part_path}")
        temporary.replace(part_path)
    finally:
        temporary.unlink(missing_ok=True)
    return before_sha256, _sha256_file(part_path), rendered


def repair_zero_audio_row(
    *,
    bucket_manifest_path: Path,
    bucket_part_path: Path,
    sample_key: str,
    waveform_path: Path,
    episode_path: Path,
    repair_shard_path: Path,
    receipt_path: Path,
    nano_report_path: Path,
    nano_predictions_path: Path,
    language: str,
    episode_page_url: str,
    episode_audio_url: str,
    crop_start_sec: float,
    crop_duration_sec: float,
    source_parquet_path: Path | None = None,
    source_parquet_row_group: int | None = None,
    source_parquet_row_index: int | None = None,
) -> dict[str, Any]:
    paths = (
        bucket_manifest_path,
        bucket_part_path,
        waveform_path,
        episode_path,
        nano_report_path,
        nano_predictions_path,
    )
    for path in paths:
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(str(path))
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("sample_key") != sample_key:
            raise ValueError(f"Existing repair receipt binds another sample: {receipt_path}")
        if receipt.get("repaired_bucket_part_sha256") != _sha256_file(bucket_part_path):
            raise ValueError("Repaired bucket part changed after receipt creation.")
        if receipt.get("repair_shard_sha256") != _sha256_file(repair_shard_path):
            raise ValueError("Repair shard changed after receipt creation.")
        if source_parquet_path is not None:
            source_parquet_path = source_parquet_path.resolve()
            verified_source = _verify_source_parquet_zero_audio(
                path=source_parquet_path,
                row_group=source_parquet_row_group,
                row_index=source_parquet_row_index,
                expected_source_id=str(
                    (receipt.get("source_metadata") or {}).get("source_id") or ""
                ),
            )
            if receipt.get("source_parquet") != verified_source:
                raise ValueError("Source Parquet verification differs from the repair receipt.")
        return receipt

    line_number, original_line, original_row = _find_bucket_row(bucket_part_path, sample_key)
    original_audio_size = original_row.get("audio_size")
    if original_audio_size is None or int(original_audio_size) > 0:
        raise ValueError(
            "Stage211 zero-audio repair requires an explicit non-positive audio_size; "
            f"key={sample_key!r} audio_size={original_audio_size!r}"
        )
    original_shard_path = Path(
        str(original_row.get("tar_path") or original_row.get("shard_name") or "")
    ).resolve()
    if not original_shard_path.is_file():
        raise FileNotFoundError(str(original_shard_path))
    audio_member = str(original_row.get("audio_member") or original_row.get("wav_member") or "")
    json_member = str(original_row.get("json_member") or "")
    if not audio_member or not json_member:
        raise ValueError("Source row is missing audio_member/json_member.")
    metadata_payload = _read_indexed_bytes(
        original_shard_path,
        member_name=json_member,
        offset=(int(original_row["json_offset"]) if original_row.get("json_offset") is not None else None),
        size=(int(original_row["json_size"]) if original_row.get("json_size") is not None else None),
    )
    source_metadata = json.loads(metadata_payload)

    waveform_info = sf.info(waveform_path)
    waveform_duration = float(waveform_info.frames / waveform_info.samplerate)
    if waveform_info.samplerate != 16_000 or waveform_info.channels != 1:
        raise ValueError("Repaired waveform must be 16 kHz mono.")
    if abs(waveform_duration - float(crop_duration_sec)) > 1.0 / 16_000.0:
        raise ValueError(
            "Repaired waveform duration does not match the declared crop: "
            f"waveform={waveform_duration} crop={crop_duration_sec}"
        )
    nano_verification = _validate_nano_verification(
        sample_key=sample_key,
        report_path=nano_report_path,
        predictions_path=nano_predictions_path,
        language=language,
    )
    source_parquet: dict[str, Any] | None = None
    if source_parquet_path is not None:
        source_parquet_path = source_parquet_path.resolve()
        if not source_parquet_path.is_file() or source_parquet_path.stat().st_size <= 0:
            raise FileNotFoundError(str(source_parquet_path))
        source_parquet = _verify_source_parquet_zero_audio(
            path=source_parquet_path,
            row_group=source_parquet_row_group,
            row_index=source_parquet_row_index,
            expected_source_id=str(source_metadata.get("source_id") or ""),
        )
    audio_payload = waveform_path.read_bytes()
    member_records = _build_repair_tar(
        output_path=repair_shard_path,
        audio_member=audio_member,
        audio_payload=audio_payload,
        json_member=json_member,
        json_payload=metadata_payload,
    )

    repaired_row = dict(original_row)
    repaired_row["shard_name"] = str(repair_shard_path.resolve())
    repaired_row["tar_path"] = str(repair_shard_path.resolve())
    repaired_row["audio_offset"] = member_records[audio_member]["offset"]
    repaired_row["audio_size"] = member_records[audio_member]["size"]
    repaired_row["json_offset"] = member_records[json_member]["offset"]
    repaired_row["json_size"] = member_records[json_member]["size"]
    repaired_row["_stage211_audio_repair_receipt"] = str(receipt_path.resolve())
    part_before_sha256, part_after_sha256, repaired_line = _patch_bucket_part(
        part_path=bucket_part_path,
        line_number=line_number,
        expected_line=original_line,
        repaired_row=repaired_row,
    )

    episode_info = sf.info(episode_path)
    receipt = {
        "schema_version": 1,
        "pipeline": "stage211",
        "artifact": "zero_audio_source_repair",
        "sample_key": sample_key,
        "bucket_manifest_path": str(bucket_manifest_path.resolve()),
        "bucket_manifest_sha256": _sha256_file(bucket_manifest_path),
        "bucket_part_path": str(bucket_part_path.resolve()),
        "bucket_part_line_number": line_number,
        "original_bucket_part_sha256": part_before_sha256,
        "repaired_bucket_part_sha256": part_after_sha256,
        "original_row_sha256": _sha256_bytes(original_line),
        "repaired_row_sha256": _sha256_bytes(repaired_line),
        "original_row": original_row,
        "repaired_row": repaired_row,
        "original_shard_path": str(original_shard_path),
        "original_audio_size": int(original_audio_size),
        "source_metadata": source_metadata,
        "source_parquet": source_parquet,
        "episode": {
            "page_url": episode_page_url,
            "audio_url": episode_audio_url,
            "path": str(episode_path.resolve()),
            "sha256": _sha256_file(episode_path),
            "size_bytes": episode_path.stat().st_size,
            "sample_rate": int(episode_info.samplerate),
            "channels": int(episode_info.channels),
            "duration_sec": float(episode_info.frames / episode_info.samplerate),
        },
        "crop": {
            "start_sec": float(crop_start_sec),
            "duration_sec": float(crop_duration_sec),
            "end_sec": float(crop_start_sec + crop_duration_sec),
            "waveform_path": str(waveform_path.resolve()),
            "waveform_sha256": _sha256_file(waveform_path),
            "waveform_size_bytes": waveform_path.stat().st_size,
            "sample_rate": int(waveform_info.samplerate),
            "channels": int(waveform_info.channels),
            "frames": int(waveform_info.frames),
        },
        "nano_verification": nano_verification,
        "repair_shard_path": str(repair_shard_path.resolve()),
        "repair_shard_sha256": _sha256_file(repair_shard_path),
        "repair_shard_size_bytes": repair_shard_path.stat().st_size,
        "repair_members": member_records,
        "preserved_contract": {
            "key": sample_key,
            "sample_index": repaired_row.get("_stage179_sample_index"),
            "num_frames": repaired_row.get("num_frames"),
            "split": repaired_row.get("split"),
            "row_count_delta": 0,
            "order_changed": False,
        },
        "complete": True,
    }
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair one audited Stage211 zero-byte audio row.")
    parser.add_argument("--bucket-manifest-path", type=Path, required=True)
    parser.add_argument("--bucket-part-path", type=Path, required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--waveform-path", type=Path, required=True)
    parser.add_argument("--episode-path", type=Path, required=True)
    parser.add_argument("--repair-shard-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--nano-report-path", type=Path, required=True)
    parser.add_argument("--nano-predictions-path", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--episode-page-url", required=True)
    parser.add_argument("--episode-audio-url", required=True)
    parser.add_argument("--crop-start-sec", type=float, required=True)
    parser.add_argument("--crop-duration-sec", type=float, required=True)
    parser.add_argument("--source-parquet-path", type=Path)
    parser.add_argument("--source-parquet-row-group", type=int)
    parser.add_argument("--source-parquet-row-index", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    receipt = repair_zero_audio_row(
        bucket_manifest_path=args.bucket_manifest_path,
        bucket_part_path=args.bucket_part_path,
        sample_key=args.sample_key,
        waveform_path=args.waveform_path,
        episode_path=args.episode_path,
        repair_shard_path=args.repair_shard_path,
        receipt_path=args.receipt_path,
        nano_report_path=args.nano_report_path,
        nano_predictions_path=args.nano_predictions_path,
        language=args.language,
        episode_page_url=args.episode_page_url,
        episode_audio_url=args.episode_audio_url,
        crop_start_sec=args.crop_start_sec,
        crop_duration_sec=args.crop_duration_sec,
        source_parquet_path=args.source_parquet_path,
        source_parquet_row_group=args.source_parquet_row_group,
        source_parquet_row_index=args.source_parquet_row_index,
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
