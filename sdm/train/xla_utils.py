"""Thin wrappers around torch_xla so the rest of the codebase can stay
device-agnostic. Helpers fall back to CPU/GPU only when XLA was not requested;
TPU runs fail loudly if torch_xla is unavailable.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from sdm.train import io as ckpt_io

_TPU_ENV_VARS = (
    "TPU_NAME",
    "TPU_WORKER_ID",
    "TPU_ACCELERATOR_TYPE",
    "CLOUD_TPU_TASK_ID",
    "TPU_PROCESS_BOUNDS",
    "TPU_VISIBLE_DEVICES",
    "XRT_TPU_CONFIG",
)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def xla_required() -> bool:
    """Return whether this process should refuse non-XLA fallback."""
    if _truthy_env("SDM_FORCE_CPU"):
        return False
    if _truthy_env("SDM_REQUIRE_XLA"):
        return True
    if os.environ.get("PJRT_DEVICE", "").strip().upper() == "TPU":
        return True
    if any(os.environ.get(name) for name in _TPU_ENV_VARS):
        return True
    return os.path.exists("/dev/accel0")


def _import_torch_xla():
    import torch_xla  # noqa: PLC0415

    return torch_xla


def _xla_unavailable_error() -> RuntimeError:
    return RuntimeError(
        "XLA/TPU training was requested, but torch_xla could not be imported. "
        "Refusing to fall back to CPU/CUDA. Fix the torch/torch_xla/libtpu install, "
        "or set SDM_FORCE_CPU=1 only for an intentional local CPU run."
    )


def is_xla() -> bool:
    if _truthy_env("SDM_FORCE_CPU"):
        return False
    try:
        _import_torch_xla()
    except ImportError as exc:
        if xla_required():
            raise _xla_unavailable_error() from exc
        return False
    return True


def get_device(*, require_xla: bool | None = None) -> torch.device:
    if require_xla is None:
        require_xla = xla_required()
    if require_xla:
        try:
            import torch_xla.core.xla_model as xm  # noqa: PLC0415
        except ImportError as exc:
            raise _xla_unavailable_error() from exc

        return xm.xla_device()
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


def clear_metrics() -> None:
    if not is_xla():
        return
    import torch_xla.debug.metrics as met  # noqa: PLC0415

    met.clear_all()


def compile_metrics() -> dict[str, int]:
    """Return compact XLA compile counters for recompilation diagnostics."""
    if not is_xla():
        return {}
    import torch_xla.debug.metrics as met  # noqa: PLC0415

    values: dict[str, int] = {}

    if hasattr(met, "counter_value"):
        for name in ("UncachedCompile", "CachedCompile", "CreateCompileHandles"):
            try:
                value = met.counter_value(name)
            except Exception:
                continue
            if value is not None:
                values[name] = int(value)

    if hasattr(met, "metric_data"):
        try:
            compile_time = met.metric_data("CompileTime")
        except Exception:
            compile_time = None
        if compile_time is not None:
            try:
                values["CompileTimeSamples"] = int(compile_time[0])
            except (IndexError, TypeError, ValueError):
                pass

    if "CompileTimeSamples" not in values:
        report = met.metrics_report()
        metric_match = re.search(
            r"Metric:\s+CompileTime\s+.*?TotalSamples:\s+(\d+)",
            report,
            flags=re.DOTALL,
        )
        if metric_match:
            values["CompileTimeSamples"] = int(metric_match.group(1))
        for name in ("UncachedCompile", "CachedCompile", "CreateCompileHandles"):
            if name in values:
                continue
            counter_match = re.search(
                rf"Counter:\s+{re.escape(name)}\s+.*?Value:\s+(\d+)",
                report,
                flags=re.DOTALL,
            )
            if counter_match:
                values[name] = int(counter_match.group(1))

    return values


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
