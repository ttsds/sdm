from __future__ import annotations

import torch

from sdm.data.teachers import build_teacher
from sdm.data.teachers.dvector_torchscript import (
    DvectorTorchscriptConfig,
    DvectorTorchscriptTeacher,
)


def _identity_wav2mel(audio: torch.Tensor) -> torch.Tensor:
    # Pretend each waveform produces a (1, T) "mel" — only the embedder uses it.
    return audio.unsqueeze(1)


def _zero_embedder(target_dim: int):
    def _embed(mels: torch.Tensor) -> torch.Tensor:
        b = mels.shape[0]
        return torch.arange(b, dtype=torch.float32).unsqueeze(-1).expand(b, target_dim).contiguous()

    return _embed


def test_dvector_returns_per_chunk_embeddings():
    cfg = DvectorTorchscriptConfig(target_dim=8, target_layernorm=False)
    teacher = DvectorTorchscriptTeacher(
        cfg, wav2mel=_identity_wav2mel, embedder=_zero_embedder(8)
    )
    audio = torch.zeros(2, 3, 16)
    chunk_mask = torch.tensor([[True, True, True], [True, True, False]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert out.shape == (2, 3, 8)
    assert torch.all(out[1, 2] == 0)


def test_dvector_target_dim_mismatch_raises():
    cfg = DvectorTorchscriptConfig(target_dim=99, target_layernorm=False)
    teacher = DvectorTorchscriptTeacher(
        cfg, wav2mel=_identity_wav2mel, embedder=_zero_embedder(8)
    )
    try:
        teacher(torch.zeros(1, 1, 16))
    except ValueError as exc:
        assert "256" in str(exc) or "8" in str(exc) or "99" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on dim mismatch")


def test_build_teacher_dispatches_dvector(monkeypatch):
    class _Cfg:
        kind = "dvector_torchscript"
        model_id = "fake"
        target_dim = 8
        pooled = "chunked"
        sample_rate = 16000
        target_layernorm = False

    import sdm.data.teachers.dvector_torchscript as mod

    orig = mod.DvectorTorchscriptTeacher.__init__

    def _patched(self, cfg, *, device="cpu", wav2mel=None, embedder=None):
        orig(self, cfg, device=device, wav2mel=_identity_wav2mel, embedder=_zero_embedder(8))

    monkeypatch.setattr(mod.DvectorTorchscriptTeacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, DvectorTorchscriptTeacher)
