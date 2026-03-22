from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from rwkvasr.data import ASRManifestDataset, FeatureCollator, WebDatasetConfig, build_webdataset_dataloader
from rwkvasr.modules import RWKVCTCModel, RWKVCTCModelConfig
from rwkvasr.training.checkpoint import load_checkpoint
from rwkvasr.training.ctc_task import RWKVDualModeCTCTrainer


def ctc_greedy_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    blank_id: int,
) -> list[list[int]]:
    token_ids = logits.argmax(dim=-1)
    if lengths is None:
        lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)

    results: list[list[int]] = []
    for batch_idx in range(logits.size(0)):
        length = int(lengths[batch_idx].item())
        seq = token_ids[batch_idx, :length].tolist()
        decoded: list[int] = []
        prev = None
        for token in seq:
            if token != blank_id and token != prev:
                decoded.append(int(token))
            prev = token
        results.append(decoded)
    return results


def edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    dp = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        dp[i][0] = i
    for j in range(len(hypothesis) + 1):
        dp[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[-1][-1]


def token_error_rate(references: list[list[int]], hypotheses: list[list[int]]) -> float:
    total_edits = 0
    total_tokens = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        total_edits += edit_distance(ref, hyp)
        total_tokens += len(ref)
    if total_tokens == 0:
        return 0.0
    return total_edits / total_tokens


@dataclass(frozen=True)
class EvalConfig:
    checkpoint_path: str
    batch_size: int
    model_config: RWKVCTCModelConfig
    manifest_path: str | None = None
    webdataset_root: str | None = None
    webdataset_split: str = "all"
    webdataset_eval_ratio: float = 0.0
    webdataset_hash_seed: int = 0
    webdataset_split_by: str = "shard_name"
    device: str = "cpu"
    modes: tuple[str, ...] = ("bi", "l2r", "alt")


def _build_eval_loader(config: EvalConfig) -> DataLoader:
    has_manifest = config.manifest_path is not None
    has_webdataset = config.webdataset_root is not None
    if has_manifest == has_webdataset:
        raise ValueError("Exactly one of manifest_path or webdataset_root must be provided for evaluation.")
    if has_manifest:
        dataset = ASRManifestDataset(str(config.manifest_path))
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=FeatureCollator(),
        )
    return build_webdataset_dataloader(
        str(config.webdataset_root),
        config=WebDatasetConfig(
            shuffle_shards=False,
            split=config.webdataset_split,
            eval_ratio=config.webdataset_eval_ratio,
            hash_seed=config.webdataset_hash_seed,
            split_by=config.webdataset_split_by,
        ),
        batch_size=config.batch_size,
        num_workers=0,
    )


def evaluate_ctc_modes(config: EvalConfig) -> dict[str, dict[str, float]]:
    device = torch.device(config.device)
    feature_dtype = torch.bfloat16 if device.type == "cuda" else None
    model = RWKVCTCModel(config.model_config)
    if feature_dtype is not None:
        model = model.to(device=device, dtype=feature_dtype)
    else:
        model = model.to(device)
    load_checkpoint(config.checkpoint_path, model=model, map_location=device.type)
    model.eval()

    task = RWKVDualModeCTCTrainer(model)
    loader = _build_eval_loader(config)

    metrics: dict[str, dict[str, float]] = {}
    for mode in config.modes:
        references: list[list[int]] = []
        hypotheses: list[list[int]] = []
        exact = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, feature_dtype=feature_dtype)
                logits, logit_lengths = task.inference_logits(
                    batch.features,
                    batch.feature_lengths,
                    mode=mode,
                )
                preds = ctc_greedy_decode(logits, logit_lengths, blank_id=model.config.blank_id)

                offset = 0
                refs: list[list[int]] = []
                for target_len in batch.target_lengths.tolist():
                    refs.append(batch.targets[offset : offset + target_len].tolist())
                    offset += target_len

                references.extend(refs)
                hypotheses.extend(preds)
                exact += sum(int(ref == hyp) for ref, hyp in zip(refs, preds, strict=True))
                total += len(refs)

        metrics[mode] = {
            "token_error_rate": token_error_rate(references, hypotheses),
            "exact_match": exact / total if total > 0 else 0.0,
            "num_utts": float(total),
        }
    return metrics
