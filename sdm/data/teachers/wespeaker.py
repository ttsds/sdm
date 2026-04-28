"""WeSpeaker speaker-embedding teacher.

Backed by `MiniXC/wespeaker-unofficial-pypi`_ (a fork of WeNet's
`wespeaker`_ packaged on PyPI). The package's ``wespeaker.load_model``
returns a ``Speaker`` object whose ``.model`` attribute is a bare
``nn.Module`` embedding network ingesting Kaldi-style log-mel fbank
features. We expose that bare module as a per-chunk teacher: each chunk
is converted to fbank on CPU (``kaldi.fbank`` has no XLA implementation),
batched, and pushed through the embedding network on the configured
device. Output: ``(B, N_chunks, target_dim)``.

Default ``model_id`` is ``vblinkf`` -- the multilingual VoxBlink2 +
VoxCeleb2 fine-tuned SimAMResNet34, which beats the English-only
``pyannote/wespeaker-voxceleb-resnet34-LM`` checkpoint we used previously
on non-English data.

.. _MiniXC/wespeaker-unofficial-pypi: https://github.com/MiniXC/wespeaker-unofficial-pypi
.. _wespeaker: https://github.com/wenet-e2e/wespeaker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class WespeakerResnet34Config:
    # wespeaker.cli.hub.Hub.Assets keys: "chinese", "english", "campplus",
    # "eres2net", "vblinkp", "vblinkf". "vblinkf" is the multilingual
    # SimAMResNet34 fine-tuned on VoxCeleb2.
    model_id: str = "vblinkf"
    target_dim: int = 256
    pooled: str = "chunked"
    sample_rate: int = 16000
    target_layernorm: bool = True
    # WeSpeaker's CLI computes Kaldi fbank on int16-scale waveforms (the
    # default torchaudio.load(normalize=False) output). Our pipeline hands
    # in float32 [-1, 1] so we rescale before fbank to match the training
    # distribution. Models marked wavform_norm=True (campplus, eres2net)
    # were trained on float32 directly; set this to 1.0 for those.
    waveform_scale: float = 32768.0
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    cmn: bool = True


def _coerce_config(cfg: Any) -> WespeakerResnet34Config:
    if isinstance(cfg, WespeakerResnet34Config):
        return cfg
    fields = (
        "model_id",
        "target_dim",
        "pooled",
        "sample_rate",
        "target_layernorm",
        "waveform_scale",
        "num_mel_bins",
        "frame_length_ms",
        "frame_shift_ms",
        "cmn",
    )
    if hasattr(cfg, "kind"):
        return WespeakerResnet34Config(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return WespeakerResnet34Config(**{f: cfg[f] for f in fields if f in cfg})


def _load_wespeaker_model(model_id: str) -> nn.Module:
    """Load the bare embedding nn.Module from wespeaker-unofficial.

    We bypass ``wespeaker.load_model`` because it instantiates a
    ``Speaker`` wrapper whose ``__init__`` eagerly loads silero-VAD,
    which transitively requires ``onnxruntime``. We don't use VAD here
    (chunk masks come from the dataloader), so we skip the wrapper and
    load the checkpoint directly from the modelscope-downloaded
    ``~/.wespeaker/<model_id>/{avg_model.pt, config.yaml}``.
    """
    import os

    import yaml  # type: ignore
    from wespeaker.cli.hub import Hub  # type: ignore
    from wespeaker.models.speaker_model import get_speaker_model  # type: ignore
    from wespeaker.utils.checkpoint import load_checkpoint  # type: ignore

    model_dir = Hub.get_model(model_id)
    with open(os.path.join(model_dir, "config.yaml"), "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = get_speaker_model(configs["model"])(**configs["model_args"])
    load_checkpoint(model, os.path.join(model_dir, "avg_model.pt"))
    return model.eval()


class WespeakerResnet34Teacher(nn.Module):
    """Per-chunk WeSpeaker speaker-embedding teacher."""

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
            model = _load_wespeaker_model(self.cfg.model_id)
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.to(device)

    def _fbank(self, waveform_1d_cpu: torch.Tensor) -> torch.Tensor:
        import torchaudio.compliance.kaldi as kaldi  # type: ignore

        feat = kaldi.fbank(
            waveform_1d_cpu.unsqueeze(0),  # (1, T)
            num_mel_bins=self.cfg.num_mel_bins,
            frame_length=self.cfg.frame_length_ms,
            frame_shift=self.cfg.frame_shift_ms,
            sample_frequency=self.cfg.sample_rate,
        )
        if self.cfg.cmn:
            feat = feat - feat.mean(dim=0, keepdim=True)
        return feat

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

        # kaldi.fbank has no XLA implementation; pull audio to CPU once,
        # rescale, compute fbank per chunk, restack. The model itself
        # runs on whatever device its parameters live on (set by .to()).
        flat = (audio.reshape(b * n, t).to(torch.float32) * float(self.cfg.waveform_scale)).cpu()
        fbanks = [self._fbank(flat[i]) for i in range(flat.shape[0])]
        feats = torch.stack(fbanks, dim=0)
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = audio.device
        feats = feats.to(model_device, dtype=torch.float32)

        emb = self.model(feats)
        # Some wespeaker architectures return (last_layer_emb, embedding)
        # tuples; keep the last (the speaker embedding).
        if isinstance(emb, tuple):
            emb = emb[-1]
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
