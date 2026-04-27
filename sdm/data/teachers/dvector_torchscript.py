"""d-Vector speaker-embedding teacher.

The d-vector ships as a torchscript checkpoint paired with a ``Wav2Mel``
front-end. We use the official VoxCeleb1-trained release from
https://github.com/yistLin/dvector/releases (v1.1.1):

* ``wav2mel.pt``                — log-mel front-end (60 dB normalize + 40-bin mel)
* ``dvector-step250000.pt``     — 3-layer LSTM + attentive pool, 256-dim emb

Both are TorchScript modules with a per-utterance API:
    ``mel = wav2mel(wav_1d, sample_rate)``  -> ``(T, 40)``
    ``emb = dvector.embed_utterance(mel)``  -> ``(256,)``

We download them once into the HF cache (just for a stable on-disk
location) and wrap them so :class:`DvectorTorchscriptTeacher.forward`
keeps its batched ``(B*N, T) -> (B*N, 256)`` interface.

Each chunk is processed independently — d-Vector is a windowed embedding
model, so feeding it per-chunk audio is the canonical TTSDS use. Output:
``(B, N_chunks, target_dim)``.

The torchscript binding is loaded lazily so test environments without
the checkpoints can still import this module. Pass
``embedder``/``wav2mel`` callables for deterministic tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

# Expects (B*N, T_samples) waveform at 16 kHz, returns (B*N, n_mels, T_frames)
# *or* anything the embedder accepts; we don't constrain shape here because
# real-world wav2mel/embedder pairs are tightly coupled.
Wav2MelFn = Callable[[torch.Tensor], torch.Tensor]
# Maps the wav2mel output to embeddings of shape (B*N, target_dim).
EmbedderFn = Callable[[torch.Tensor], torch.Tensor]


# Pinned to the v1.1.1 release so checkpoints don't drift under us.
_DVECTOR_RELEASE = "https://github.com/yistLin/dvector/releases/download/v1.1.1"
_WAV2MEL_URL = f"{_DVECTOR_RELEASE}/wav2mel.pt"
_DVECTOR_URL = f"{_DVECTOR_RELEASE}/dvector-step250000.pt"


@dataclass
class DvectorTorchscriptConfig:
    # ``model_id`` is kept for backwards compatibility with the YAML schema
    # but is unused for the default backend (assets are pinned to the GH
    # release). Override it to point at a custom torchscript bundle and
    # supply your own loader.
    model_id: str = "yistLin/dvector"
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


def _cache_dir() -> Path:
    root = Path(os.environ.get("SDM_TEACHER_CACHE", Path.home() / ".cache" / "sdm" / "dvector"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    import urllib.request  # noqa: PLC0415

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — fixed asset URL
    tmp.rename(dest)
    return dest


def _load_default_backend(sample_rate: int) -> tuple[Wav2MelFn, EmbedderFn]:
    """Download yistLin/dvector v1.1.1 release assets and wrap them.

    The torchscript modules expose a *per-utterance* API:
        ``wav2mel(wav_1d, sample_rate) -> (T, 40)``
        ``dvector.embed_utterance(mel)  -> (256,)``
    so we loop over the leading batch axis and stack. This is fine because
    each chunk is ~1 s and we're CPU-bound here anyway.
    """
    cache = _cache_dir()
    wav2mel_path = _download(_WAV2MEL_URL, cache / "wav2mel.pt")
    dvector_path = _download(_DVECTOR_URL, cache / "dvector-step250000.pt")
    wav2mel_mod = torch.jit.load(str(wav2mel_path), map_location="cpu").eval()
    dvector_mod = torch.jit.load(str(dvector_path), map_location="cpu").eval()

    def _wav2mel(audio: torch.Tensor) -> torch.Tensor:
        # audio: (B*N, T) float32. Returns (B*N, T_frames, 40) — kept as a
        # list-of-mels would also work, but the embedder wrapper indexes
        # this dim explicitly so the torch.stack happens in one place.
        mels = [wav2mel_mod(wav, sample_rate) for wav in audio]
        return torch.nn.utils.rnn.pad_sequence(mels, batch_first=True)

    def _embed(mels: torch.Tensor) -> torch.Tensor:
        # mels: (B*N, T_frames, 40). embed_utterance wants (T_frames, 40).
        embs = [dvector_mod.embed_utterance(m) for m in mels]
        return torch.stack(embs, dim=0)

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
            loaded_w2m, loaded_emb = _load_default_backend(self.cfg.sample_rate)
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
