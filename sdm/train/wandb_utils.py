"""Lazy W&B wrapper.

Init is a no-op unless the master process has `wandb` installed and
`WANDB_API_KEY` (or a previous `wandb login`) is configured. All log/finish
calls are safe to invoke from any process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_run = None


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "sdm"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] | None = None


def init(cfg: WandbConfig | None, hyperparams: dict[str, Any], *, is_master: bool) -> None:
    global _run
    if not is_master or cfg is None or not cfg.enabled:
        return
    if os.environ.get("WANDB_DISABLED") == "true":
        return
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        return
    _run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=cfg.run_name,
        group=cfg.group,
        tags=cfg.tags,
        config=hyperparams,
        resume="allow",
    )


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    if _run is None:
        return
    _run.log(metrics, step=step)


def finish() -> None:
    global _run
    if _run is None:
        return
    _run.finish()
    _run = None
