"""WeSpeaker ResNet34 speaker-embedding teacher.

The pyannote release ``pyannote/wespeaker-voxceleb-resnet34-LM`` ships a
``pytorch_model.bin`` plus a config that wraps a WeSpeaker ResNet34
embedding model. The native ``pyannote.audio.Inference`` runs on CPU
and is too slow for in-loop distillation; this teacher loads the bare
``pyannote.audio.Model`` (or any compatible ``nn.Module``) directly so
it runs on the training accelerator.

Each chunk is embedded independently. Output: ``(B, N_chunks, target_dim)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sdm.dotenv import hf_token_kwargs


@dataclass
class WespeakerResnet34Config:
    model_id: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    target_dim: int = 256
    pooled: str = "chunked"
    sample_rate: int = 16000
    target_layernorm: bool = True


def _coerce_config(cfg: Any) -> WespeakerResnet34Config:
    if isinstance(cfg, WespeakerResnet34Config):
        return cfg
    fields = ("model_id", "target_dim", "pooled", "sample_rate", "target_layernorm")
    if hasattr(cfg, "kind"):
        return WespeakerResnet34Config(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return WespeakerResnet34Config(**{f: cfg[f] for f in fields if f in cfg})


def _load_pyannote_wespeaker(model_id: str) -> nn.Module:
    """Load the bare embedding nn.Module from a pyannote checkpoint.

    pyannote.audio renamed ``use_auth_token`` to ``token`` somewhere
    around 3.1; try the new spelling first and fall back for older
    installs.
    """
    from pyannote.audio import Model  # type: ignore

    token = hf_token_kwargs().get("token")
    try:
        return Model.from_pretrained(model_id, token=token).eval()
    except TypeError:
        return Model.from_pretrained(model_id, use_auth_token=token).eval()


class WespeakerResnet34Teacher(nn.Module):
    """Per-chunk WeSpeaker ResNet34 embedding teacher."""

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        self.target_layernorm = bool(self.cfg.target_layernorm)
        if model is None:
            model = _load_pyannote_wespeaker(self.cfg.model_id)
        self.model = model
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
        if chunk_mask is None:
            chunk_mask = torch.ones((b, n), dtype=torch.bool, device=audio.device)

        # pyannote.audio.Model expects (B, 1, T) waveform input.
        flat = audio.reshape(b * n, 1, t).to(torch.float32)
        emb = self.model(flat)
        if emb.dim() == 3:  # (B, 1, D) — squeeze the singleton speaker axis
            emb = emb.squeeze(1)
        if emb.dim() != 2 or emb.shape[-1] != self.target_dim:
            raise ValueError(
                f"wespeaker model returned shape {tuple(emb.shape)}; "
                f"expected (B*N, {self.target_dim})"
            )
        out = emb.reshape(b, n, self.target_dim).to(audio.device, dtype=torch.float32)
        if self.target_layernorm:
            out = torch.nn.functional.layer_norm(out, (self.target_dim,))
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


__all__ = ["WespeakerResnet34Config", "WespeakerResnet34Teacher"]
