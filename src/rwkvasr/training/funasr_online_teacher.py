from __future__ import annotations

import hashlib
import json
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


SOURCE_LANGUAGES = {
    "librispeech": "en",
    "commonvoice_en": "en",
    "cv22_en": "en",
    "emilia_en": "en",
    "gigaspeech": "en",
    "aishell3": "zh",
    "commonvoice_cn": "zh",
    "cv22_zh": "zh",
    "emilia_zh": "zh",
    "wenetspeech": "zh",
}

SOURCE_FIELDS = (
    "_stage162_source",
    "_stage161_source",
    "_stage160_source",
    "_stage157_source",
    "_stage179_source",
    "_stage178b_source",
    "_stage178_source",
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


@dataclass(frozen=True)
class FunASROnlineCTCTeacherConfig:
    model_path: str
    audio_index_path: str | None = None
    webdataset_index_path: str | None = None
    webdataset_root: str | None = None
    audio_cache_dir: str | None = None
    keep_audio_cache: bool = True
    device: str = "cuda"
    split: str = "train"
    top_k: int = 16
    project_blank_id: int = 60515
    project_ignored_token_ids: tuple[int, ...] = (60514,)
    project_vocab_size: int | None = None
    return_full_log_probs: bool = False
    return_encoder_out: bool = False
    return_encoder_input: bool = False
    return_layer_hiddens: bool = False


@dataclass(frozen=True)
class _ResolvedAudioPath:
    path: str
    temporary: bool = False


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"Expected tensor module output, got {type(value)!r}.")


def _funasr_encoder_layers(audio_encoder: Any) -> list[Any]:
    return [
        *list(audio_encoder.encoders0),
        *list(audio_encoder.encoders),
        *list(audio_encoder.tp_encoders),
    ]


@contextmanager
def _capture_funasr_encoder_hiddens(
    audio_encoder: Any,
    layer_ids: tuple[int, ...],
    *,
    capture_input: bool,
) -> Iterator[dict[str, Any]]:
    layers = _funasr_encoder_layers(audio_encoder)
    invalid = [layer_id for layer_id in layer_ids if not 0 <= int(layer_id) < len(layers)]
    if invalid:
        raise ValueError(f"Nano encoder layer ids are out of range: {invalid}; layers={len(layers)}")

    captured: dict[str, Any] = {"layers": {}}
    handles: list[Any] = []
    if capture_input:
        def capture_encoder_input(_module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                raise RuntimeError("Nano audio encoder pre-hook received no inputs.")
            # SenseVoiceEncoderSmall scales its input in-place, so the audit
            # snapshot must not share storage with the live encoder tensor.
            captured["encoder_input"] = _first_tensor(args[0]).detach().clone()
            if len(args) > 1 and isinstance(args[1], torch.Tensor):
                captured["encoder_input_lengths"] = args[1].detach()

        handles.append(audio_encoder.register_forward_pre_hook(capture_encoder_input))

    for layer_id in layer_ids:
        layer_capture: dict[str, torch.Tensor] = {}
        captured["layers"][int(layer_id)] = layer_capture
        layer = layers[int(layer_id)]

        def capture_layer_input(
            _module: Any,
            args: tuple[Any, ...],
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            if not args:
                raise RuntimeError("Nano encoder layer pre-hook received no inputs.")
            target["input"] = _first_tensor(args[0]).detach()

        def capture_mixer(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            target["mixer"] = _first_tensor(output).detach()

        def capture_ffn(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            target["ffn"] = _first_tensor(output).detach()

        def capture_block(
            _module: Any,
            _args: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, torch.Tensor] = layer_capture,
        ) -> None:
            target["block"] = _first_tensor(output).detach()

        handles.append(layer.register_forward_pre_hook(capture_layer_input))
        handles.append(layer.self_attn.register_forward_hook(capture_mixer))
        handles.append(layer.feed_forward.register_forward_hook(capture_ffn))
        handles.append(layer.register_forward_hook(capture_block))

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


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


def _row_id(row: dict[str, Any]) -> str:
    for field in ("utt_id", "id", "key", "sid"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    audio_path = str(row.get("audio_path") or row.get("audio_filepath") or "").strip()
    if audio_path:
        return Path(audio_path).stem
    raise ValueError("row has no utt_id/id/key/sid/audio_path")


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


def _safe_audio_name(utt_id: str, audio_member: str | None) -> str:
    digest = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()
    suffix = Path(str(audio_member or "")).suffix
    return f"{digest}{suffix or '.audio'}"


def _load_shard_paths(webdataset_index_path: Path | None, webdataset_root: Path | None) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if webdataset_index_path is not None:
        data = json.loads(webdataset_index_path.read_text(encoding="utf-8"))
        shards = data.get("shards") if isinstance(data, dict) else None
        if not isinstance(shards, list):
            raise ValueError(f"webdataset index has no shards list: {webdataset_index_path}")
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
    if webdataset_root is not None:
        for shard_path in webdataset_root.glob("*.tar"):
            paths.setdefault(shard_path.name, shard_path)
    return paths


def _load_audio_rows(path: Path, *, split: str) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"CTC online teacher audio index missing or empty: {path}")
    rows: dict[str, dict[str, Any]] = {}
    requested_split = str(split or "all")
    for _, row in _iter_jsonl(path):
        if requested_split != "all":
            row_split = str(row.get("split") or "train")
            if row_split != requested_split:
                continue
        try:
            utt_id = _row_id(row)
        except ValueError:
            continue
        if utt_id not in rows:
            rows[utt_id] = row
    if not rows:
        raise ValueError(f"No usable rows found in CTC online teacher audio index: {path}")
    return rows


def _resolve_audio_path(
    row: dict[str, Any],
    utt_id: str,
    *,
    shard_paths: dict[str, Path],
    audio_cache_dir: Path | None,
    keep_audio_cache: bool = True,
) -> _ResolvedAudioPath:
    audio_path = str(row.get("audio_path") or row.get("audio_filepath") or "").strip()
    if audio_path:
        path = Path(audio_path)
        if path.exists():
            return _ResolvedAudioPath(str(path), temporary=False)
        if audio_cache_dir is None:
            raise FileNotFoundError(audio_path)

    if audio_cache_dir is None:
        raise ValueError("row has no usable audio_path; set ctc_teacher_online_audio_cache_dir")

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

    audio_member = str(row.get("audio_member") or row.get("wav_member") or "").strip()
    if not audio_member:
        raise ValueError("row has no audio_member/wav_member for tar-backed audio")

    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_cache_dir / _safe_audio_name(utt_id, audio_member)
    audio_size_raw = row.get("audio_size")
    expected_size = int(audio_size_raw) if audio_size_raw is not None else None
    if (
        keep_audio_cache
        and out_path.exists()
        and (expected_size is None or out_path.stat().st_size == expected_size)
    ):
        return _ResolvedAudioPath(str(out_path), temporary=False)

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
    return _ResolvedAudioPath(str(out_path), temporary=not bool(keep_audio_cache))


class FunASRNanoCTCTopKOnlineTeacher:
    def __init__(self, config: FunASROnlineCTCTeacherConfig):
        self.config = config
        self.audio_rows = (
            _load_audio_rows(Path(config.audio_index_path), split=config.split)
            if config.audio_index_path
            else {}
        )
        self.shard_paths = _load_shard_paths(
            Path(config.webdataset_index_path) if config.webdataset_index_path else None,
            Path(config.webdataset_root) if config.webdataset_root else None,
        )
        self.audio_cache_dir = Path(config.audio_cache_dir) if config.audio_cache_dir else None

        from funasr import AutoModel

        self.auto_model = AutoModel(
            model=str(config.model_path),
            trust_remote_code=True,
            device=str(config.device),
            disable_update=True,
        )
        self.model = self.auto_model.model
        if self.model.ctc_decoder is None:
            raise RuntimeError("FunASR-Nano ctc_decoder is missing; use the current ModelScope model.pt")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def num_audio_rows(self) -> int:
        return len(self.audio_rows)

    def _build_record(
        self,
        *,
        utt_id: str,
        source: str,
        language: str | None,
        ctc_logp: torch.Tensor,
        decoder_out_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        hidden_capture: dict[str, Any],
        capture_layer_ids: tuple[int, ...],
        sample_idx: int,
    ) -> dict[str, Any]:
        frame_count = int(decoder_out_lens[sample_idx].item())
        encoder_frame_count = int(encoder_out_lens[sample_idx].item())
        x = ctc_logp[sample_idx, :frame_count, :].float()
        vocab_size = int(x.shape[-1])
        yseq = torch.unique_consecutive(x.argmax(dim=-1), dim=-1)
        k = min(max(1, int(self.config.top_k)), vocab_size)
        topk_log_probs, topk_ids = torch.topk(x, k=k, dim=-1)
        teacher_blank_id = int(self.model.blank_id)
        argmax_token_ids = yseq[yseq != teacher_blank_id].to(dtype=torch.int32)
        blank_log_probs = x[:, teacher_blank_id]
        blank_in_topk = topk_ids.eq(teacher_blank_id).any(dim=-1)
        if frame_count > 0:
            topk_ids = topk_ids.clone()
            topk_log_probs = topk_log_probs.clone()
            topk_ids[~blank_in_topk, -1] = teacher_blank_id
            topk_log_probs[~blank_in_topk, -1] = blank_log_probs[~blank_in_topk]
        mapped_ids = topk_ids.clone()
        mapped_ids[mapped_ids == teacher_blank_id] = int(self.config.project_blank_id)

        result = {
            "format": "funasr_nano_ctc_topk_online_v1",
            "teacher": "FunASR-Nano-2512",
            "utt_id": utt_id,
            "source": source,
            "language": language,
            "topk": int(k),
            "num_frames": int(frame_count),
            "teacher_blank_id": teacher_blank_id,
            "teacher_vocab_size": vocab_size,
            "project_blank_id": int(self.config.project_blank_id),
            "project_ignored_token_ids": [int(value) for value in self.config.project_ignored_token_ids],
            "topk_token_ids": mapped_ids.detach().cpu().to(dtype=torch.int32),
            "topk_log_probs": topk_log_probs.detach().cpu().to(dtype=torch.float32),
            "blank_log_probs": blank_log_probs.detach().cpu().to(dtype=torch.float32),
            "argmax_token_ids": argmax_token_ids.detach().cpu(),
        }
        if self.config.return_encoder_out:
            result["encoder_out"] = (
                encoder_out[sample_idx, :encoder_frame_count, :].detach().cpu().to(dtype=torch.float16)
            )
            result["encoder_out_lens"] = int(encoder_frame_count)
        if self.config.return_encoder_input:
            encoder_input = hidden_capture.get("encoder_input")
            if not isinstance(encoder_input, torch.Tensor):
                raise RuntimeError(f"Nano encoder input capture is missing for utt_id={utt_id!r}.")
            encoder_input_lengths = hidden_capture.get("encoder_input_lengths")
            input_frame_count = (
                int(encoder_input_lengths.flatten()[sample_idx].item())
                if isinstance(encoder_input_lengths, torch.Tensor)
                else int(encoder_input.size(1))
            )
            result["encoder_input"] = (
                encoder_input[sample_idx, :input_frame_count, :].detach().cpu().to(dtype=torch.float16)
            )
            result["encoder_input_lens"] = int(input_frame_count)
        if capture_layer_ids:
            layer_hiddens: dict[str, dict[str, torch.Tensor]] = {}
            captured_layers = hidden_capture.get("layers", {})
            for layer_id in capture_layer_ids:
                captured_components = captured_layers.get(layer_id, {})
                missing_components = [
                    name for name in ("input", "mixer", "ffn", "block")
                    if not isinstance(captured_components.get(name), torch.Tensor)
                ]
                if missing_components:
                    raise RuntimeError(
                        f"Nano layer hidden capture is incomplete for utt_id={utt_id!r} "
                        f"layer={layer_id}: missing={missing_components}"
                    )
                layer_hiddens[str(layer_id)] = {
                    name: captured_components[name][sample_idx, :encoder_frame_count, :]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float16)
                    for name in ("input", "mixer", "ffn", "block")
                }
            result["encoder_layer_hiddens"] = layer_hiddens
            result["encoder_layer_ids"] = list(capture_layer_ids)
        if self.config.return_full_log_probs:
            project_vocab_size = max(
                int(self.config.project_vocab_size or 0),
                vocab_size,
                int(self.config.project_blank_id) + 1,
                *(int(value) + 1 for value in self.config.project_ignored_token_ids),
            )
            full_log_probs = x.new_full((frame_count, project_vocab_size), float("-inf"))
            full_log_probs[:, :vocab_size] = x
            if teacher_blank_id != int(self.config.project_blank_id):
                full_log_probs[:, int(self.config.project_blank_id)] = x[:, teacher_blank_id]
                full_log_probs[:, teacher_blank_id] = float("-inf")
            for ignored_id in self.config.project_ignored_token_ids:
                ignored_id = int(ignored_id)
                if ignored_id != int(self.config.project_blank_id) and 0 <= ignored_id < project_vocab_size:
                    full_log_probs[:, ignored_id] = float("-inf")
            result["full_log_probs"] = full_log_probs.detach().cpu().to(dtype=torch.float16)
        return result

    def _forward_one(
        self,
        *,
        utt_id: str,
        row: dict[str, Any],
        layer_ids: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        resolved_audio = _resolve_audio_path(
            row,
            utt_id,
            shard_paths=self.shard_paths,
            audio_cache_dir=self.audio_cache_dir,
            keep_audio_cache=bool(self.config.keep_audio_cache),
        )
        audio_path = resolved_audio.path
        try:
            source = _source_from_row(row)
            language = SOURCE_LANGUAGES.get(source)
            kwargs = dict(self.auto_model.kwargs)
            tokenizer = kwargs.pop("tokenizer")
            frontend = kwargs.pop("frontend")
            kwargs["disable_pbar"] = True
            prompt = self.model.get_prompt([], language, True)
            chatml = self.model.generate_chatml(prompt, audio_path)
            capture_layer_ids = tuple(sorted(set(int(value) for value in layer_ids)))
            if capture_layer_ids and not self.config.return_layer_hiddens:
                raise ValueError("layer_ids were requested but return_layer_hiddens is disabled.")
            with _capture_funasr_encoder_hiddens(
                self.model.audio_encoder,
                capture_layer_ids,
                capture_input=bool(self.config.return_encoder_input),
            ) as hidden_capture:
                with torch.inference_mode():
                    _, _, _, _, meta_data = self.model.inference_prepare(
                        [chatml],
                        key=[utt_id],
                        tokenizer=tokenizer,
                        frontend=frontend,
                        **kwargs,
                    )
                    encoder_out = meta_data["encoder_out"]
                    encoder_out_lens = meta_data["encoder_out_lens"]
                    decoder_out, decoder_out_lens = self.model.ctc_decoder(encoder_out, encoder_out_lens)
                    ctc_logp = self.model.ctc.log_softmax(decoder_out)
            return self._build_record(
                utt_id=utt_id,
                source=source,
                language=language,
                ctc_logp=ctc_logp,
                decoder_out_lens=decoder_out_lens,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                hidden_capture=hidden_capture,
                capture_layer_ids=capture_layer_ids,
                sample_idx=0,
            )
        finally:
            if resolved_audio.temporary:
                Path(audio_path).unlink(missing_ok=True)

    def feature_records(
        self,
        utt_ids: list[str] | tuple[str, ...],
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        *,
        audio_rows: list[dict[str, Any] | None] | tuple[dict[str, Any] | None, ...] | None = None,
        layer_ids: tuple[int, ...] | list[int] | None = None,
        include_ctc_outputs: bool = True,
    ) -> dict[str, dict[str, Any]]:
        batch_size = min(len(utt_ids), int(features.size(0)), int(feature_lengths.numel()))
        if batch_size <= 0:
            return {}
        if features.ndim != 3:
            raise ValueError(f"Nano feature teacher expects [B, T, D], got {tuple(features.shape)}.")
        capture_layer_ids = tuple(sorted(set(int(value) for value in (layer_ids or ()))))
        if capture_layer_ids and not self.config.return_layer_hiddens:
            raise ValueError("layer_ids were requested but return_layer_hiddens is disabled.")

        teacher_device = next(self.model.audio_encoder.parameters()).device
        teacher_features = features[:batch_size].detach().to(device=teacher_device, dtype=torch.float32).clone()
        teacher_lengths = feature_lengths[:batch_size].detach().to(device=teacher_device, dtype=torch.long)
        with _capture_funasr_encoder_hiddens(
            self.model.audio_encoder,
            capture_layer_ids,
            capture_input=bool(self.config.return_encoder_input),
        ) as hidden_capture:
            with torch.inference_mode():
                encoder_out, encoder_out_lens = self.model.audio_encoder(teacher_features, teacher_lengths)
                if include_ctc_outputs:
                    decoder_out, decoder_out_lens = self.model.ctc_decoder(encoder_out, encoder_out_lens)
                    ctc_logp = self.model.ctc.log_softmax(decoder_out)
                else:
                    decoder_out_lens = None
                    ctc_logp = None

        records: dict[str, dict[str, Any]] = {}
        for sample_idx in range(batch_size):
            utt_id = str(utt_ids[sample_idx])
            row = None
            if audio_rows is not None and sample_idx < len(audio_rows):
                candidate = audio_rows[sample_idx]
                if isinstance(candidate, dict) and candidate:
                    row = candidate
            if row is None:
                row = self.audio_rows.get(utt_id)
            source = _source_from_row(row) if row is not None else "unknown"
            if isinstance(ctc_logp, torch.Tensor) and isinstance(decoder_out_lens, torch.Tensor):
                records[utt_id] = self._build_record(
                    utt_id=utt_id,
                    source=source,
                    language=SOURCE_LANGUAGES.get(source),
                    ctc_logp=ctc_logp,
                    decoder_out_lens=decoder_out_lens,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    hidden_capture=hidden_capture,
                    capture_layer_ids=capture_layer_ids,
                    sample_idx=sample_idx,
                )
                continue

            encoder_frame_count = int(encoder_out_lens[sample_idx].item())
            captured_layers = hidden_capture.get("layers", {})
            layer_hiddens: dict[str, dict[str, torch.Tensor]] = {}
            for layer_id in capture_layer_ids:
                captured_components = captured_layers.get(layer_id, {})
                missing_components = [
                    name for name in ("input", "mixer", "ffn", "block")
                    if not isinstance(captured_components.get(name), torch.Tensor)
                ]
                if missing_components:
                    raise RuntimeError(
                        f"Nano layer hidden capture is incomplete for utt_id={utt_id!r} "
                        f"layer={layer_id}: missing={missing_components}"
                    )
                layer_hiddens[str(layer_id)] = {
                    name: captured_components[name][sample_idx, :encoder_frame_count, :]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float16)
                    for name in ("input", "mixer", "ffn", "block")
                }
            hidden_record: dict[str, Any] = {
                "format": "funasr_nano_encoder_hidden_online_v1",
                "teacher": "FunASR-Nano-2512",
                "utt_id": utt_id,
                "source": source,
                "language": SOURCE_LANGUAGES.get(source),
                "encoder_out_lens": encoder_frame_count,
                "encoder_layer_hiddens": layer_hiddens,
                "encoder_layer_ids": list(capture_layer_ids),
            }
            if self.config.return_encoder_out:
                hidden_record["encoder_out"] = (
                    encoder_out[sample_idx, :encoder_frame_count, :]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float16)
                )
            if self.config.return_encoder_input:
                encoder_input = hidden_capture.get("encoder_input")
                if not isinstance(encoder_input, torch.Tensor):
                    raise RuntimeError(f"Nano encoder input capture is missing for utt_id={utt_id!r}.")
                hidden_record["encoder_input"] = (
                    encoder_input[sample_idx, :encoder_frame_count, :]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float16)
                )
                hidden_record["encoder_input_lens"] = encoder_frame_count
            records[utt_id] = hidden_record
        return records

    def topk_records(
        self,
        utt_ids: list[str] | tuple[str, ...],
        audio_rows: list[dict[str, Any] | None] | tuple[dict[str, Any] | None, ...] | None = None,
        *,
        layer_ids: tuple[int, ...] | list[int] | None = None,
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        requested_layer_ids = tuple(int(value) for value in (layer_ids or ()))
        for index, utt_id_value in enumerate(utt_ids):
            utt_id = str(utt_id_value)
            row = None
            if audio_rows is not None and index < len(audio_rows):
                candidate = audio_rows[index]
                if isinstance(candidate, dict) and candidate:
                    row = candidate
            if row is None:
                row = self.audio_rows.get(utt_id)
            if row is None:
                continue
            records[utt_id] = self._forward_one(
                utt_id=utt_id,
                row=row,
                layer_ids=requested_layer_ids,
            )
        return records
