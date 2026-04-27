from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sdm.data.teachers import build_teacher
from sdm.data.teachers.whisper_encoder import (
    WhisperEncoderConfig,
    WhisperEncoderTeacher,
)


class _FakeFeatureExtractor:
    """Returns a fixed-length log-mel-shaped tensor regardless of input."""

    def __call__(self, waveform, sampling_rate, return_tensors="pt"):
        # Whisper-small ships 80 mels x 3000 frames at 30s. We don't actually
        # need those values for the test since the encoder is also fake.
        x = torch.from_numpy(np.zeros((1, 80, 3000), dtype=np.float32))
        return {"input_features": x}


class _FakeEncoderOut:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_features, return_dict=True):
        # Whisper encoder downsamples the 3000-frame input by 2 to 1500.
        # Emit a deterministic ramp so we can check the chunk pooling.
        b = input_features.shape[0]
        ramp = torch.arange(1500, dtype=torch.float32).unsqueeze(-1).expand(1500, self.hidden_size)
        out = ramp.unsqueeze(0).expand(b, 1500, self.hidden_size).contiguous()
        return _FakeEncoderOut(out)


def test_whisper_encoder_pools_50_frames_per_chunk():
    cfg = WhisperEncoderConfig(
        model_id="fake", target_dim=8, chunk_seconds=1.0, target_layernorm=False
    )
    teacher = WhisperEncoderTeacher(
        cfg,
        encoder=_FakeEncoder(hidden_size=8),
        feature_extractor=_FakeFeatureExtractor(),
    )
    audio = torch.zeros(1, 3, 16000)
    chunk_mask = torch.tensor([[True, True, True]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert out.shape == (1, 3, 8)
    # Chunk i mean-pools encoder frames [i*50 : (i+1)*50]; the ramp values
    # average to (start + end - 1) / 2.
    expected = torch.tensor([(0 + 49) / 2, (50 + 99) / 2, (100 + 149) / 2])
    for i in range(3):
        assert torch.allclose(out[0, i], torch.full((8,), expected[i].item()))


def test_whisper_encoder_zeros_padded_chunks():
    cfg = WhisperEncoderConfig(model_id="fake", target_dim=8, target_layernorm=False)
    teacher = WhisperEncoderTeacher(
        cfg,
        encoder=_FakeEncoder(hidden_size=8),
        feature_extractor=_FakeFeatureExtractor(),
    )
    audio = torch.zeros(1, 4, 16000)
    chunk_mask = torch.tensor([[True, True, False, False]])
    out = teacher(audio, chunk_mask=chunk_mask)
    assert out.shape == (1, 4, 8)
    assert torch.all(out[0, 2] == 0)
    assert torch.all(out[0, 3] == 0)


def test_build_teacher_dispatches_whisper(monkeypatch):
    class _Cfg:
        kind = "whisper_encoder"
        model_id = "fake"
        target_dim = 8
        pooled = "chunked"
        target_layernorm = False
        chunk_seconds = 1.0
        sample_rate = 16000

    import sdm.data.teachers.whisper_encoder as mod

    orig_init = mod.WhisperEncoderTeacher.__init__

    def _patched(self, cfg, *, device="cpu", encoder=None, feature_extractor=None):
        orig_init(
            self,
            cfg,
            device=device,
            encoder=_FakeEncoder(hidden_size=8),
            feature_extractor=_FakeFeatureExtractor(),
        )

    monkeypatch.setattr(mod.WhisperEncoderTeacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, WhisperEncoderTeacher)
