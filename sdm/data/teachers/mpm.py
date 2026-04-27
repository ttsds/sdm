"""Masked Prosody Model (MPM) layer-7 teacher.

MPM is an English-trained prosody encoder published at
https://github.com/MiniXC/masked_prosody_model. It expects 22.05 kHz
audio, computes pitch / energy / VAD features per frame, bucketizes them
to integer indices, and runs a Conformer encoder; layer 7 was the layer
used in the paper. The released model exposes ``process_audio(path,
layer=7)`` which returns the per-frame layer-7 representations for the
entire audio file. We:

1. Per utterance, stitch the valid (unpadded) chunks into a contiguous
   waveform (still at the streaming-loader's 16 kHz).
2. Resample 16 kHz -> 22.05 kHz inside the teacher (the data pipeline
   only emits 16 kHz mono).
3. Write a temporary WAV and invoke ``model.process_audio`` since that
   bundles the pitch/energy/VAD measures + bucketization + forward.
4. Mean-pool the returned per-frame representations into the 1-second
   chunk grid.

Output: ``(B, N_chunks, filter_size)``.

A custom ``representation_fn`` callable can be injected for tests so we
do not need ``masked_prosody_model`` and its dependencies in CI.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

# Receives the per-utterance 22.05 kHz waveform (1-D float32 numpy) and
# returns the layer-N representations as a tensor of shape (T_frames, D).
RepresentationFn = Callable[[np.ndarray, int], torch.Tensor]


@dataclass
class MpmConfig:
    model_id: str = "cdminix/masked_prosody_model"
    layer: int = 7
    target_dim: int = 256
    pooled: str = "chunked"
    sample_rate: int = 16000           # input rate from the data loader
    teacher_sample_rate: int = 22050   # MPM's expected rate
    chunk_seconds: float = 1.0
    target_layernorm: bool = True


def _coerce_config(cfg: Any) -> MpmConfig:
    if isinstance(cfg, MpmConfig):
        return cfg
    fields = (
        "model_id",
        "layer",
        "target_dim",
        "pooled",
        "sample_rate",
        "teacher_sample_rate",
        "chunk_seconds",
        "target_layernorm",
    )
    if hasattr(cfg, "kind"):
        return MpmConfig(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return MpmConfig(**{f: cfg[f] for f in fields if f in cfg})


def _load_default_representation_fn(model_id: str, layer: int) -> RepresentationFn:
    """Wrap ``MaskedProsodyModel.process_audio`` so we can call it on numpy waveforms."""
    from masked_prosody_model import MaskedProsodyModel  # type: ignore
    import soundfile as sf  # type: ignore

    model = MaskedProsodyModel.from_pretrained(model_id).eval()

    @torch.no_grad()
    def _repr_fn(waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
        # process_audio reads from disk + resamples internally to 22050.
        # Write a temp WAV at the input sample_rate and let MPM handle it.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform.astype(np.float32, copy=False), sample_rate)
            try:
                rep = model.process_audio(tmp.name, layer=layer)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        # ``process_audio`` concatenates per-window representations along
        # the time axis: shape (T_frames, filter_size).
        return rep.detach().to(torch.float32)

    return _repr_fn


class MpmTeacher(nn.Module):
    """In-loop MPM layer-N teacher.

    Bear in mind: MPM is English-only, so on the multilingual Emilia
    stream the targets for non-English utterances will be noisy. The
    paper still reports useful prosody factor signal at layer 7.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        representation_fn: RepresentationFn | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        self.target_layernorm = bool(self.cfg.target_layernorm)
        if representation_fn is None:
            representation_fn = _load_default_representation_fn(
                self.cfg.model_id, int(self.cfg.layer)
            )
        self._representation_fn = representation_fn
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
        b, n, _ = audio.shape
        if chunk_mask is None:
            chunk_mask = torch.ones((b, n), dtype=torch.bool, device=audio.device)

        out = torch.zeros((b, n, self.target_dim), dtype=torch.float32)
        chunk_secs = float(self.cfg.chunk_seconds)
        sample_rate = int(self.cfg.sample_rate)
        for i in range(b):
            valid_n = int(chunk_mask[i].sum().item())
            if valid_n <= 0:
                continue
            waveform = audio[i, :valid_n].reshape(-1).to(torch.float32).cpu().numpy()
            rep = self._representation_fn(waveform, sample_rate)
            if rep.dim() != 2 or rep.shape[-1] != self.target_dim:
                raise ValueError(
                    f"MPM representation_fn returned shape {tuple(rep.shape)}; "
                    f"expected (T_frames, {self.target_dim})"
                )
            # Map the per-frame representations onto the chunk grid via
            # mean-pool. ``valid_n`` chunks span ``valid_n * chunk_secs``
            # seconds; MPM's frame rate is implicit in ``rep.shape[0]``.
            t_frames = rep.shape[0]
            frames_per_chunk = max(1, t_frames // valid_n)
            for c in range(valid_n):
                start = c * frames_per_chunk
                end = start + frames_per_chunk if c < valid_n - 1 else t_frames
                if end <= start:
                    continue
                out[i, c] = rep[start:end].mean(dim=0)

        out = out.to(audio.device)
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


__all__ = ["MpmConfig", "MpmTeacher", "RepresentationFn"]
