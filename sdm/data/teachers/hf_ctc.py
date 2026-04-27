"""HuggingFace CTC teacher (e.g. wav2vec2 ASR encoder).

Loads ``AutoModelForCTC`` for the configured ``model_id`` and extracts a
fixed encoder hidden layer. Mirrors :class:`HfSslTeacher` but with a CTC
head instead of the bare SSL backbone — TTSDS2 uses the *encoder* hidden
states of an ASR-finetuned wav2vec2 (intelligibility factor), so we pull
the same intermediate layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sdm.dotenv import hf_token_kwargs


@dataclass
class HfCtcConfig:
    model_id: str
    target_dim: int
    layer: int = -1
    pooled: str = "chunked"
    target_layernorm: bool = True


def _coerce_config(cfg: Any) -> HfCtcConfig:
    if isinstance(cfg, HfCtcConfig):
        return cfg
    fields = ("model_id", "layer", "target_dim", "pooled", "target_layernorm")
    if hasattr(cfg, "model_id"):
        return HfCtcConfig(**{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None})
    return HfCtcConfig(**{f: cfg[f] for f in fields if f in cfg})


class HfCtcTeacher(nn.Module):
    """In-loop CTC encoder teacher returning ``(B, N_chunks, hidden_size)``."""

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        if model is None:
            from transformers import AutoModelForCTC  # type: ignore

            model = AutoModelForCTC.from_pretrained(self.cfg.model_id, **hf_token_kwargs())
        self.model = model
        self.layer = int(self.cfg.layer)
        self.target_dim = int(self.cfg.target_dim)
        self.target_layernorm = bool(self.cfg.target_layernorm)
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
        **_: Any,
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
        if self.target_layernorm:
            features = torch.nn.functional.layer_norm(features, (d,))
        if chunk_mask is not None:
            features = features * chunk_mask.unsqueeze(-1).to(features.dtype)
        return features

    def __call__(  # type: ignore[override]
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **ctx: Any,
    ) -> torch.Tensor:
        return self.forward(audio, chunk_mask=chunk_mask, **ctx)


__all__ = ["HfCtcConfig", "HfCtcTeacher"]
