"""Whisper encoder teacher (multilingual).

Whisper always processes 30-second log-mel inputs internally; its encoder
produces 1500 frames (= 50 frames per second). We feed the unpadded
utterance through ``WhisperFeatureExtractor`` once, run the encoder, and
mean-pool encoder frames into 1-second chunks so the output grid matches
the rest of the distillation pipeline.

Output shape: ``(B, N_chunks, hidden_size)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from sdm.dotenv import hf_token_kwargs

# Whisper's encoder emits this many frames per second of input audio
# (30s -> 1500 frames). Used to map the chunk grid onto encoder frames.
_WHISPER_ENCODER_FPS = 50


@dataclass
class WhisperEncoderConfig:
    model_id: str
    target_dim: int
    pooled: str = "chunked"
    target_layernorm: bool = True
    chunk_seconds: float = 1.0
    sample_rate: int = 16000


def _coerce_config(cfg: Any) -> WhisperEncoderConfig:
    if isinstance(cfg, WhisperEncoderConfig):
        return cfg
    fields = ("model_id", "target_dim", "pooled", "target_layernorm", "chunk_seconds", "sample_rate")
    if hasattr(cfg, "model_id"):
        return WhisperEncoderConfig(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return WhisperEncoderConfig(**{f: cfg[f] for f in fields if f in cfg})


class WhisperEncoderTeacher(nn.Module):
    """In-loop multilingual Whisper encoder teacher."""

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        encoder: nn.Module | None = None,
        feature_extractor: Any | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        self.target_layernorm = bool(self.cfg.target_layernorm)
        self.chunk_seconds = float(self.cfg.chunk_seconds)
        self.sample_rate = int(self.cfg.sample_rate)
        self._frames_per_chunk = max(1, int(round(self.chunk_seconds * _WHISPER_ENCODER_FPS)))

        if encoder is None or feature_extractor is None:
            from transformers import WhisperFeatureExtractor, WhisperModel  # type: ignore

            if encoder is None:
                full = WhisperModel.from_pretrained(self.cfg.model_id, **hf_token_kwargs())
                encoder = full.encoder  # type: ignore[assignment]
            if feature_extractor is None:
                feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    self.cfg.model_id, **hf_token_kwargs()
                )
        self.encoder = encoder
        self.feature_extractor = feature_extractor
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self.encoder.to(self._device)

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

        # Fully batched path: flatten chunks back into per-utterance
        # waveforms, run the feature extractor in one batched call, run
        # the encoder on the whole batch once, then mean-pool encoder
        # frames into the 1-second chunk grid. Invalid (padded) chunks
        # are zeroed via ``chunk_mask`` at the end.
        #
        # Note: Whisper truncates to 30s of audio (1500 encoder frames).
        # If ``n * chunk_seconds > 30`` the trailing chunks are zero-padded.
        flat = audio.reshape(b, n * t).to(torch.float32).cpu().numpy()
        proc = self.feature_extractor(
            [row for row in flat],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_features = proc["input_features"].to(self._device)
        enc_out = self.encoder(input_features, return_dict=True)
        hidden = enc_out.last_hidden_state  # (B, T_enc, D)
        d = hidden.shape[-1]
        if d != self.target_dim:
            raise ValueError(
                f"whisper encoder hidden dim {d} != configured target_dim {self.target_dim}"
            )

        needed = n * self._frames_per_chunk
        if hidden.shape[1] < needed:
            pad = torch.zeros(
                (b, needed - hidden.shape[1], d),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            hidden = torch.cat([hidden, pad], dim=1)
        hidden = hidden[:, :needed]  # (B, n*fpc, D)
        out = hidden.reshape(b, n, self._frames_per_chunk, d).mean(dim=2)
        out = out.to(audio.device, dtype=torch.float32)

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


__all__ = ["WhisperEncoderConfig", "WhisperEncoderTeacher"]
