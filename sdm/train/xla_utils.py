"""Thin wrappers around torch_xla so the rest of the codebase can stay
device-agnostic. Every helper falls back to the equivalent CPU/GPU op when
torch_xla is not installed, so unit tests run in any environment.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from sdm.train import io as ckpt_io


def is_xla() -> bool:
    try:
        import torch_xla  # noqa: F401, PLC0415
    except ImportError:
        return False
    return os.environ.get("SDM_FORCE_CPU") != "1"


def get_device() -> torch.device:
    if is_xla():
        import torch_xla.core.xla_model as xm  # noqa: PLC0415

        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mark_step() -> None:
    if not is_xla():
        return
    import torch_xla.core.xla_model as xm  # noqa: PLC0415

    xm.mark_step()


def world_size() -> int:
    if not is_xla():
        return 1
    import torch_xla.runtime as xr  # noqa: PLC0415

    return xr.world_size()


def global_ordinal() -> int:
    if not is_xla():
        return 0
    import torch_xla.runtime as xr  # noqa: PLC0415

    return xr.global_ordinal()


def is_master() -> bool:
    return global_ordinal() == 0


def shard_module_fsdp(module: nn.Module) -> nn.Module:
    """Wrap a module with FSDP-via-XLA. No-op on non-XLA backends."""
    if not is_xla():
        return module
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP  # noqa: PLC0415

    return FSDP(module)


def reduce_gradients(optimizer: torch.optim.Optimizer) -> None:
    if not is_xla():
        return
    import torch_xla.core.xla_model as xm  # noqa: PLC0415

    xm.reduce_gradients(optimizer)


def save_checkpoint(state: dict[str, Any], path: str) -> None:
    """Master-only save. On XLA we collect a CPU state-dict before writing.

    `path` may be a local filesystem path or a `gs://` URI.
    """
    if is_xla():
        import torch_xla.core.xla_model as xm  # noqa: PLC0415

        cpu_state = {
            k: ({sk: sv.cpu() if torch.is_tensor(sv) else sv for sk, sv in v.items()}
                if isinstance(v, dict)
                else (v.cpu() if torch.is_tensor(v) else v))
            for k, v in state.items()
        }
        if xm.is_master_ordinal():
            ckpt_io.save_state(cpu_state, path)
        xm.rendezvous("sdm-save")
    elif is_master():
        ckpt_io.save_state(state, path)


def loader_per_device(loader: Iterable, device: torch.device) -> Iterable:
    """Wrap a torch DataLoader so each step's batch lands on the right device."""
    if is_xla():
        import torch_xla.distributed.parallel_loader as pl  # noqa: PLC0415

        return pl.MpDeviceLoader(loader, device)
    # CPU/GPU: a thin generator that moves tensors.
    def _gen() -> Iterable:
        for batch in loader:
            yield {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

    return _gen()
