from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def init_wandb_run(
    *,
    enabled: bool,
    project: str | None,
    run_name: str | None,
    output_dir: str | Path,
    config: dict[str, Any],
    base_url: str | None = None,
    init_timeout_sec: float = 30.0,
    logger: Callable[[str], None] | None = None,
) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "wandb logging is enabled but the `wandb` package is not installed. Run `uv sync` first."
        ) from exc
    output_dir = Path(output_dir)
    settings_kwargs: dict[str, Any] = {
        "init_timeout": float(init_timeout_sec),
        "reinit": "create_new",
    }
    if base_url:
        settings_kwargs["base_url"] = str(base_url)
    try:
        run = wandb.init(
            project=project or "rwkvasr_longform_asr",
            name=run_name,
            dir=str(output_dir),
            config=config,
            settings=wandb.Settings(**settings_kwargs),
        )
    except Exception as exc:  # pragma: no cover - depends on local environment and connectivity
        if logger is not None:
            logger(
                "wandb init failed; continuing without experiment tracking. "
                f"base_url={base_url or '<wandb settings>'} "
                f"timeout={float(init_timeout_sec):.1f}s "
                f"error={type(exc).__name__}: {exc}"
            )
        return None
    return run


def log_wandb(run: Any | None, metrics: dict[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(metrics)
        return
    run.log(metrics, step=int(step))


def finish_wandb(run: Any | None) -> None:
    if run is None:
        return
    run.finish()
