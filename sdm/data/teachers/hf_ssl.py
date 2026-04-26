"""HuggingFace SSL teacher (e.g. XLS-R / wav2vec2 family).

Loads ``AutoModel`` for the configured ``model_id`` and extracts a fixed
hidden layer. The teacher is run on the full chunk-flattened batch with
``output_hidden_states=True``; the per-chunk frame outputs are mean-pooled to
yield ``(B, N_chunks, hidden_size)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sdm.dotenv import hf_token_kwargs


@dataclass
class HfSslConfig:
    model_id: str
    layer: int
    target_dim: int
    pooled: str = "chunked"


def _coerce_config(cfg: Any) -> HfSslConfig:
    if isinstance(cfg, HfSslConfig):
        return cfg
    fields = ("model_id", "layer", "target_dim", "pooled")
    if hasattr(cfg, "model_id"):
        return HfSslConfig(**{f: getattr(cfg, f) for f in fields if hasattr(cfg, f)})
    return HfSslConfig(**{f: cfg[f] for f in fields if f in cfg})


class HfSslTeacher(nn.Module):
    """In-loop SSL teacher returning ``(B, N_chunks, hidden_size)`` features."""

    def __init__(self, cfg: Any, *, device: torch.device | str = "cpu", model: nn.Module | None = None) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        if model is None:
            from transformers import AutoModel  # type: ignore

            model = AutoModel.from_pretrained(self.cfg.model_id, **hf_token_kwargs())
        self.model = model
        self.layer = int(self.cfg.layer)
        self.target_dim = int(self.cfg.target_dim)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.to(device)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if audio.dim() != 3:
            raise ValueError(f"expected (B, N_chunks, T), got {tuple(audio.shape)}")
        b, n, t = audio.shape
        flat = audio.reshape(b * n, t)
        out = self.model(input_values=flat, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[self.layer]
        pooled = hidden.mean(dim=1)  # (B*N, D)
        d = pooled.shape[-1]
        if d != self.target_dim:
            raise ValueError(
                f"teacher hidden dim {d} != configured target_dim {self.target_dim}"
            )
        features = pooled.reshape(b, n, d)
        if chunk_mask is not None:
            features = features * chunk_mask.unsqueeze(-1).to(features.dtype)
        return features

    def __call__(  # type: ignore[override]
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(audio, chunk_mask=chunk_mask)


__all__ = ["HfSslConfig", "HfSslTeacher"]
