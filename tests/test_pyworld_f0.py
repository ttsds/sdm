from __future__ import annotations

import numpy as np
import torch

from sdm.data.teachers import build_teacher
from sdm.data.teachers.pyworld_f0 import PyworldF0Config, PyworldF0Teacher


def _const_extractor(value: float, frames_per_sec: int = 200):
    """Returns a constant F0 contour at the requested rate (5 ms frames)."""

    def _extract(waveform: np.ndarray, sample_rate: int):
        n_frames = max(1, int(round(len(waveform) / sample_rate * frames_per_sec)))
        return np.full(n_frames, value, dtype=np.float32), 1000.0 / frames_per_sec

    return _extract


def test_pyworld_f0_returns_voiced_mean_per_chunk():
    cfg = PyworldF0Config(chunk_seconds=1.0, sample_rate=16000, frame_period_ms=5.0)
    teacher = PyworldF0Teacher(cfg, f0_extractor=_const_extractor(220.0))

    audio = torch.zeros(1, 3, 16000)
    chunk_mask = torch.tensor([[True, True, True]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert out.shape == (1, 3, 1)
    assert torch.allclose(out[0, :, 0], torch.full((3,), 220.0))


def test_pyworld_f0_zeros_padded_chunks():
    cfg = PyworldF0Config(chunk_seconds=1.0, sample_rate=16000)
    teacher = PyworldF0Teacher(cfg, f0_extractor=_const_extractor(150.0))

    audio = torch.zeros(2, 4, 16000)
    chunk_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert torch.all(out[0, 2:] == 0)
    assert torch.all(out[1, 1:] == 0)
    assert torch.allclose(out[0, :2, 0], torch.full((2,), 150.0))


def test_pyworld_f0_excludes_unvoiced_frames():
    def _half_voiced(waveform: np.ndarray, sample_rate: int):
        # 5 ms frames: 200 frames per second -> 200 frames for a 1s chunk.
        # Alternating voiced 100 Hz / unvoiced 0 across the chunk.
        n_frames = int(round(len(waveform) / sample_rate * 200))
        f0 = np.zeros(n_frames, dtype=np.float32)
        f0[::2] = 100.0
        return f0, 5.0

    cfg = PyworldF0Config(chunk_seconds=1.0, sample_rate=16000)
    teacher = PyworldF0Teacher(cfg, f0_extractor=_half_voiced)
    out = teacher(torch.zeros(1, 1, 16000), chunk_mask=torch.ones(1, 1, dtype=torch.bool))
    # voiced frames only -> mean is 100 Hz, not 50 Hz.
    assert torch.allclose(out[0, 0], torch.tensor([100.0]))


def test_build_teacher_dispatches_pyworld(monkeypatch):
    class _Cfg:
        kind = "pyworld_f0"
        target_dim = 1
        pooled = "chunked"
        chunk_seconds = 1.0
        sample_rate = 16000
        target_layernorm = False

    import sdm.data.teachers.pyworld_f0 as mod

    orig = mod.PyworldF0Teacher.__init__

    def _patched(self, cfg, *, device="cpu", f0_extractor=None):
        orig(self, cfg, device=device, f0_extractor=_const_extractor(123.0))

    monkeypatch.setattr(mod.PyworldF0Teacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, PyworldF0Teacher)
