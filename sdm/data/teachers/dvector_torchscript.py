"""d-Vector speaker-embedding teacher.

The d-vector ships as a torchscript checkpoint trained on log-mel input.
We use the official VoxCeleb1-trained release from
https://github.com/yistLin/dvector/releases (v1.1.1):

* ``dvector-step250000.pt``     — 3-layer LSTM + attentive pool, 256-dim emb

The release also bundles ``wav2mel.pt`` (a torchscript Wav2Mel front-end),
but it depends on ``torchaudio::sox_effects_apply_effects_tensor`` which
is not registered in modern torchaudio builds (>= 2.0). Instead, we
reimplement the front-end in pure Python — peak-normalize to -3 dB then
``torchaudio.transforms.MelSpectrogram`` + log. This matches the TTSDS
benchmark implementation
(https://github.com/ttsds/ttsds/blob/main/src/ttsds/benchmarks/speaker/dvector.py)
which uses the same parameters
(40 mels, 25 ms / 10 ms window/hop, f_min=50 Hz).

Each chunk is processed independently — d-Vector is a windowed embedding
model, so feeding it per-chunk audio is the canonical TTSDS use. Output:
``(B, N_chunks, target_dim)``.

The torchscript binding is loaded lazily so test environments without
the checkpoint can still import this module. Pass
``embedder``/``wav2mel`` callables for deterministic tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

# Expects (B*N, T_samples) waveform at 16 kHz, returns the per-utterance
# mel feature shape the embedder expects (here: (B*N, T_frames, 40)).
Wav2MelFn = Callable[[torch.Tensor], torch.Tensor]
# Maps the wav2mel output to embeddings of shape (B*N, target_dim).
EmbedderFn = Callable[[torch.Tensor], torch.Tensor]


# Pinned to the v1.1.1 release so the dvector checkpoint doesn't drift.
_DVECTOR_RELEASE = "https://github.com/yistLin/dvector/releases/download/v1.1.1"
_DVECTOR_URL = f"{_DVECTOR_RELEASE}/dvector-step250000.pt"

# Mel front-end constants from the upstream Wav2Mel definition (TTSDS copy).
_NORM_DB = -3.0
_FFT_WINDOW_MS = 25.0
_FFT_HOP_MS = 10.0
_F_MIN = 50.0
_N_MELS = 40


@dataclass
class DvectorTorchscriptConfig:
    # Kept for YAML schema compatibility; the default backend ignores it
    # because the dvector checkpoint URL is pinned in code.
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


def _make_wav2mel(sample_rate: int) -> Wav2MelFn:
    """Build the pure-Python Wav2Mel front-end.

    Replicates ``torchaudio.sox_effects.apply_effects_tensor([['norm','-3']])``
    + ``MelSpectrogram(...)`` + ``log(clamp(..., 1e-9))``. Input is assumed
    mono and at ``sample_rate`` already (no resample step needed: the caller
    has chunked at that rate).
    """
    from torchaudio.transforms import MelSpectrogram  # noqa: PLC0415

    melspec = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=int(sample_rate * _FFT_WINDOW_MS / 1000),
        hop_length=int(sample_rate * _FFT_HOP_MS / 1000),
        f_min=_F_MIN,
        n_mels=_N_MELS,
    ).eval()
    # Linear gain that maps peak |x|=1 to ``_NORM_DB``. Matches sox `norm -3`
    # which scales so the new peak hits 10**(norm_db/20) of full-scale.
    norm_target = 10.0 ** (_NORM_DB / 20.0)

    @torch.no_grad()
    def _wav2mel(audio: torch.Tensor) -> torch.Tensor:
        # audio: (B*N, T) float32.
        peak = audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
        wav = audio * (norm_target / peak)
        # MelSpectrogram returns (..., n_mels, T_frames); we want (..., T, n_mels).
        mel = melspec(wav)
        mel = mel.transpose(-1, -2)
        return torch.log(torch.clamp(mel, min=1e-9))

    return _wav2mel


def _load_default_backend(sample_rate: int) -> tuple[Wav2MelFn, EmbedderFn]:
    """Build the pure-Python Wav2Mel and load the yistLin dvector torchscript.

    The dvector torchscript exposes a per-utterance API:
        ``dvector.embed_utterance(mel) -> (256,)``  with mel shaped (T, 40).
    We loop over the leading batch axis because each chunk is ~1 s and we're
    CPU-bound here anyway.
    """
    cache = _cache_dir()
    dvector_path = _download(_DVECTOR_URL, cache / "dvector-step250000.pt")
    dvector_mod = torch.jit.load(str(dvector_path), map_location="cpu").eval()

    wav2mel = _make_wav2mel(sample_rate)

    def _embed(mels: torch.Tensor) -> torch.Tensor:
        # mels: (B*N, T_frames, 40). embed_utterance wants (T_frames, 40).
        embs = [dvector_mod.embed_utterance(m) for m in mels]
        return torch.stack(embs, dim=0)

    return wav2mel, _embed


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
        # The dvector torchscript and our pure-Python Wav2Mel both live on
        # CPU (torchaudio.MelSpectrogram registers its window as a CPU
        # buffer; torch.stft would otherwise raise "input and window must
        # be on the same device" when audio arrives on xla:0). Ignore the
        # caller-provided device for compute and move outputs back at the
        # end of forward().
        self._device = torch.device("cpu")

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
