"""Teacher modules for in-loop and offline distillation targets.

Each teacher exposes a ``__call__(audio, chunk_mask=None) -> Tensor`` method
returning ``(B, N_chunks, D)`` where ``audio`` has shape
``(B, N_chunks, samples_per_chunk)``. The chunk grid is fixed by the data
loader, so teachers only need to mean-pool their internal frame outputs onto
that grid.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


class Teacher(Protocol):
    target_dim: int

    def __call__(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


def build_teacher(cfg: Any, *, device: torch.device | str = "cpu") -> Teacher:
    """Dispatch on ``cfg.kind`` to construct the right teacher implementation."""

    kind = getattr(cfg, "kind", None) or cfg["kind"]
    if kind == "hf_ssl":
        from sdm.data.teachers.hf_ssl import HfSslTeacher

        return HfSslTeacher(cfg, device=device)
    raise NotImplementedError(
        f"teacher kind {kind!r} is not implemented yet; add a module under sdm/data/teachers/."
    )


__all__ = ["Teacher", "build_teacher"]
