"""WORLD F0 teacher (pyworld DIO + StoneMask).

Computes the per-utterance F0 contour with WORLD's DIO + StoneMask, then
mean-pools voiced frames within each 1-second chunk to a single scalar.
Unvoiced frames (F0 == 0) are excluded from the mean so silent / noisy
chunks don't drag the target toward zero. Output: ``(B, N_chunks, 1)``.

The pyworld import is lazy so non-prosody training / CI hosts do not
require the pyworld wheel. Provide a ``f0_extractor`` callable for
deterministic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

# (f0_array, frame_period_ms) callable returning F0 contour for a 1-D float32
# waveform at the configured sample rate.
F0Extractor = Callable[[np.ndarray, int], tuple[np.ndarray, float]]


@dataclass
class PyworldF0Config:
    target_dim: int = 1
    pooled: str = "chunked"
    sample_rate: int = 16000
    chunk_seconds: float = 1.0
    frame_period_ms: float = 5.0
    f0_floor: float = 71.0
    f0_ceil: float = 800.0
    target_layernorm: bool = False


def _coerce_config(cfg: Any) -> PyworldF0Config:
    if isinstance(cfg, PyworldF0Config):
        return cfg
    fields = (
        "target_dim",
        "pooled",
        "sample_rate",
        "chunk_seconds",
        "frame_period_ms",
        "f0_floor",
        "f0_ceil",
        "target_layernorm",
    )
    if hasattr(cfg, "kind"):
        return PyworldF0Config(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return PyworldF0Config(**{f: cfg[f] for f in fields if f in cfg})


def _default_pyworld_extractor(
    f0_floor: float, f0_ceil: float, frame_period: float
) -> F0Extractor:
    def _extract(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        import pyworld  # type: ignore

        wf = waveform.astype(np.float64, copy=False)
        f0, t = pyworld.dio(
            wf,
            sample_rate,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            frame_period=frame_period,
        )
        f0 = pyworld.stonemask(wf, f0, t, sample_rate)
        return f0.astype(np.float32, copy=False), float(frame_period)

    return _extract


class PyworldF0Teacher(nn.Module):
    """Per-chunk mean voiced F0 (Hz) target."""

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        f0_extractor: F0Extractor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        if self.target_dim != 1:
            raise ValueError(
                f"pyworld_f0 emits a 1-D scalar; got target_dim={self.target_dim}"
            )
        self._extractor = f0_extractor or _default_pyworld_extractor(
            self.cfg.f0_floor, self.cfg.f0_ceil, self.cfg.frame_period_ms
        )
        self._device = torch.device(device) if not isinstance(device, torch.device) else device

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if audio.dim() != 3:
            raise ValueError(f"expected (B, N_chunks, T), got {tuple(audio.shape)}")
        b, n, t = audio.shape
        if chunk_mask is None:
            chunk_mask = torch.ones((b, n), dtype=torch.bool, device=audio.device)

        out = torch.zeros((b, n, 1), dtype=torch.float32)
        sample_rate = int(self.cfg.sample_rate)
        chunk_secs = float(self.cfg.chunk_seconds)
        for i in range(b):
            valid_n = int(chunk_mask[i].sum().item())
            if valid_n <= 0:
                continue
            waveform = audio[i, :valid_n].reshape(-1).to(torch.float32).cpu().numpy()
            f0, frame_period_ms = self._extractor(waveform, sample_rate)
            frame_period_s = float(frame_period_ms) / 1000.0
            frames_per_chunk = max(1, int(round(chunk_secs / frame_period_s)))
            for c in range(valid_n):
                start = c * frames_per_chunk
                end = min(start + frames_per_chunk, f0.shape[0])
                if end <= start:
                    continue
                segment = f0[start:end]
                voiced = segment[segment > 0]
                if voiced.size == 0:
                    continue
                out[i, c, 0] = float(voiced.mean())

        out = out.to(audio.device)
        if self.cfg.target_layernorm:
            # Single scalar — LN would zero it out. Honor explicit opt-in.
            out = torch.nn.functional.layer_norm(out, (1,))
        out = out * chunk_mask.unsqueeze(-1).to(out.dtype)
        return out

    def __call__(  # type: ignore[override]
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **ctx: Any,
    ) -> torch.Tensor:
        return self.forward(audio, chunk_mask=chunk_mask, **ctx)


__all__ = ["F0Extractor", "PyworldF0Config", "PyworldF0Teacher"]
