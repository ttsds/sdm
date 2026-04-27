"""d-Vector speaker-embedding teacher.

The TTSDS d-vector ships as a torchscript checkpoint paired with a Wav2Mel
front-end (resampled to 16 kHz and converted to log-mel before the
torchscript module runs). Each chunk is processed independently — d-Vector
is a windowed embedding model, so feeding it per-chunk audio is the
canonical TTSDS use. Output: ``(B, N_chunks, target_dim)``.

The torchscript binding is loaded lazily so test environments without
``ttsds`` / ``yin``-style speaker tooling can still import this module.
Pass ``embedder``/``wav2mel`` callables for deterministic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from sdm.dotenv import hf_token_kwargs

# Expects (B, T_samples) waveform at 16 kHz, returns (B, n_mels, T_frames).
Wav2MelFn = Callable[[torch.Tensor], torch.Tensor]
# Expects mel features (B, n_mels, T_frames), returns (B, target_dim).
EmbedderFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DvectorTorchscriptConfig:
    model_id: str = "ttsds/dvector"
    target_dim: int = 256
    pooled: str = "chunked"
    sample_rate: int = 16000
    target_layernorm: bool = True


def _coerce_config(cfg: Any) -> DvectorTorchscriptConfig:
    if isinstance(cfg, DvectorTorchscriptConfig):
        return cfg
    fields = ("model_id", "target_dim", "pooled", "sample_rate", "target_layernorm")
    if hasattr(cfg, "kind"):
        return DvectorTorchscriptConfig(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return DvectorTorchscriptConfig(**{f: cfg[f] for f in fields if f in cfg})


def _load_default_backend(model_id: str) -> tuple[Wav2MelFn, EmbedderFn]:
    """Load the canonical TTSDS d-vector + Wav2Mel torchscript bundle.

    Both files ship together in the HF repo (``ttsds/dvector`` by default).
    We use ``hf_hub_download`` so the same access pattern works for
    public and gated repos.
    """
    from huggingface_hub import hf_hub_download  # type: ignore

    wav2mel_path = hf_hub_download(model_id, "wav2mel.pt", **hf_token_kwargs())
    dvector_path = hf_hub_download(model_id, "dvector.pt", **hf_token_kwargs())
    wav2mel = torch.jit.load(wav2mel_path, map_location="cpu").eval()
    embedder = torch.jit.load(dvector_path, map_location="cpu").eval()

    def _wav2mel(audio: torch.Tensor) -> torch.Tensor:
        return wav2mel(audio)

    def _embed(mels: torch.Tensor) -> torch.Tensor:
        return embedder.embed_utterance(mels) if hasattr(embedder, "embed_utterance") else embedder(mels)

    return _wav2mel, _embed


class DvectorTorchscriptTeacher(nn.Module):
    """Per-chunk d-vector embedding teacher."""

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        wav2mel: Wav2MelFn | None = None,
        embedder: EmbedderFn | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        self.target_layernorm = bool(self.cfg.target_layernorm)

        if wav2mel is None or embedder is None:
            loaded_w2m, loaded_emb = _load_default_backend(self.cfg.model_id)
            wav2mel = wav2mel or loaded_w2m
            embedder = embedder or loaded_emb
        self._wav2mel = wav2mel
        self._embedder = embedder
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

        flat = audio.reshape(b * n, t).to(self._device, dtype=torch.float32)
        mels = self._wav2mel(flat)
        emb = self._embedder(mels)
        if emb.dim() != 2 or emb.shape[-1] != self.target_dim:
            raise ValueError(
                f"d-vector embedder returned shape {tuple(emb.shape)}; "
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


__all__ = [
    "DvectorTorchscriptConfig",
    "DvectorTorchscriptTeacher",
    "EmbedderFn",
    "Wav2MelFn",
]
