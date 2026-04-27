from __future__ import annotations

import numpy as np
import torch

from sdm.data.teachers import build_teacher
from sdm.data.teachers.mpm import MpmConfig, MpmTeacher


def _ramp_repr_fn(target_dim: int, frames_per_sec: int = 100):
    """Deterministic stand-in for ``MaskedProsodyModel.process_audio``."""

    def _fn(waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
        seconds = len(waveform) / sample_rate
        n_frames = max(1, int(round(seconds * frames_per_sec)))
        ramp = torch.arange(n_frames, dtype=torch.float32).unsqueeze(-1)
        return ramp.expand(n_frames, target_dim).contiguous()

    return _fn


def test_mpm_pools_frames_into_chunk_grid():
    cfg = MpmConfig(target_dim=8, chunk_seconds=1.0, sample_rate=16000, target_layernorm=False)
    teacher = MpmTeacher(cfg, representation_fn=_ramp_repr_fn(8))

    audio = torch.zeros(1, 3, 16000)
    chunk_mask = torch.tensor([[True, True, True]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert out.shape == (1, 3, 8)
    # 3 chunks * 100 frames/chunk; means of [0..99], [100..199], [200..299].
    expected = torch.tensor([(0 + 99) / 2, (100 + 199) / 2, (200 + 299) / 2])
    for i in range(3):
        assert torch.allclose(out[0, i], torch.full((8,), expected[i].item()))


def test_mpm_zeros_padded_chunks():
    cfg = MpmConfig(target_dim=8, chunk_seconds=1.0, sample_rate=16000, target_layernorm=False)
    teacher = MpmTeacher(cfg, representation_fn=_ramp_repr_fn(8))

    audio = torch.zeros(2, 4, 16000)
    chunk_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    out = teacher(audio, chunk_mask=chunk_mask)
    assert torch.all(out[0, 2:] == 0)
    assert torch.all(out[1, 1:] == 0)


def test_build_teacher_dispatches_mpm(monkeypatch):
    class _Cfg:
        kind = "mpm"
        model_id = "fake"
        layer = 7
        target_dim = 8
        pooled = "chunked"
        sample_rate = 16000
        teacher_sample_rate = 22050
        chunk_seconds = 1.0
        target_layernorm = False

    import sdm.data.teachers.mpm as mod

    orig = mod.MpmTeacher.__init__

    def _patched(self, cfg, *, device="cpu", representation_fn=None):
        orig(self, cfg, device=device, representation_fn=_ramp_repr_fn(8))

    monkeypatch.setattr(mod.MpmTeacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, MpmTeacher)
