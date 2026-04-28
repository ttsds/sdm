from __future__ import annotations

import torch
from torch import nn

from sdm.data.teachers import build_teacher
from sdm.data.teachers.wespeaker import (
    WespeakerResnet34Config,
    WespeakerResnet34Teacher,
)


class _FakeWespeaker(nn.Module):
    """Fake wespeaker embedding net: ingests Kaldi fbank ``(B, F, M)`` and
    returns ``(B, target_dim)`` by mean-pooling and broadcasting a scalar.
    The real model does something smarter; the teacher only cares about
    output shape semantics.
    """

    def __init__(self, target_dim: int = 8):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        scalar = x.reshape(b, -1).mean(dim=-1, keepdim=True)
        return scalar.expand(b, self.target_dim)


# kaldi.fbank needs at least one full window (25 ms @ 16 kHz = 400 samples).
# Use 1 s of audio per chunk so fbank computation succeeds in tests.
_CHUNK_SAMPLES = 16000


def test_wespeaker_returns_per_chunk_embeddings():
    cfg = WespeakerResnet34Config(target_dim=8, target_layernorm=False)
    teacher = WespeakerResnet34Teacher(cfg, model=_FakeWespeaker(8))
    audio = torch.zeros(2, 3, _CHUNK_SAMPLES)
    chunk_mask = torch.tensor([[True, True, True], [True, False, False]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert out.shape == (2, 3, 8)
    assert torch.all(out[1, 1:] == 0)


def test_wespeaker_dim_mismatch_raises():
    cfg = WespeakerResnet34Config(target_dim=99, target_layernorm=False)
    teacher = WespeakerResnet34Teacher(cfg, model=_FakeWespeaker(8))
    try:
        teacher(torch.zeros(1, 1, _CHUNK_SAMPLES))
    except ValueError as exc:
        assert "wespeaker" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on dim mismatch")


def test_build_teacher_dispatches_wespeaker(monkeypatch):
    class _Cfg:
        kind = "wespeaker_resnet34"
        model_id = "fake"
        target_dim = 8
        pooled = "chunked"
        target_layernorm = False

    import sdm.data.teachers.wespeaker as mod

    orig = mod.WespeakerResnet34Teacher.__init__

    def _patched(self, cfg, *, device="cpu", model=None):
        orig(self, cfg, device=device, model=_FakeWespeaker(8))

    monkeypatch.setattr(mod.WespeakerResnet34Teacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, WespeakerResnet34Teacher)
