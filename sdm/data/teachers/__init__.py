"""Teacher modules for in-loop and offline distillation targets.

Each teacher exposes a ``__call__(audio, chunk_mask=None, **ctx) -> Tensor``
method returning ``(B, N_chunks, D)`` where ``audio`` has shape
``(B, N_chunks, samples_per_chunk)``. The chunk grid is fixed by the data
loader, so teachers only need to mean-pool their internal frame outputs onto
that grid. ``**ctx`` carries optional batch metadata (``texts``,
``languages``, ``n_chunks``); audio-only teachers ignore it.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch


class Teacher(Protocol):
    target_dim: int

    def __call__(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **ctx: Any,
    ) -> torch.Tensor:
        ...


def build_teacher(cfg: Any, *, device: torch.device | str = "cpu") -> Teacher:
    """Dispatch on ``cfg.kind`` to construct the right teacher implementation."""

    kind = getattr(cfg, "kind", None) or cfg["kind"]
    if kind == "hf_ssl":
        from sdm.data.teachers.hf_ssl import HfSslTeacher

        return HfSslTeacher(cfg, device=device)
    if kind == "hf_ctc":
        from sdm.data.teachers.hf_ctc import HfCtcTeacher

        return HfCtcTeacher(cfg, device=device)
    if kind == "whisper_encoder":
        from sdm.data.teachers.whisper_encoder import WhisperEncoderTeacher

        return WhisperEncoderTeacher(cfg, device=device)
    if kind == "pyworld_f0":
        from sdm.data.teachers.pyworld_f0 import PyworldF0Teacher

        return PyworldF0Teacher(cfg, device=device)
    if kind == "g2p_speaking_rate":
        from sdm.data.teachers.g2p_speaking_rate import G2pSpeakingRateTeacher

        return G2pSpeakingRateTeacher(cfg, device=device)
    if kind == "dvector_torchscript":
        from sdm.data.teachers.dvector_torchscript import DvectorTorchscriptTeacher

        return DvectorTorchscriptTeacher(cfg, device=device)
    if kind == "wespeaker_resnet34":
        from sdm.data.teachers.wespeaker import WespeakerResnet34Teacher

        return WespeakerResnet34Teacher(cfg, device=device)
    if kind == "emotion2vec":
        from sdm.data.teachers.emotion2vec import Emotion2vecTeacher

        return Emotion2vecTeacher(cfg, device=device)
    if kind == "mpm":
        from sdm.data.teachers.mpm import MpmTeacher

        return MpmTeacher(cfg, device=device)
    raise NotImplementedError(
        f"teacher kind {kind!r} is not implemented yet; add a module under sdm/data/teachers/."
    )


__all__ = ["Teacher", "build_teacher"]
