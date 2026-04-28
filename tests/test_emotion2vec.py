from __future__ import annotations

import numpy as np
import torch

from sdm.data.teachers import build_teacher
from sdm.data.teachers.emotion2vec import (
    Emotion2vecConfig,
    Emotion2vecTeacher,
)


class _FakeFunasrModel:
    """Stand-in for funasr.AutoModel: ``generate(list_of_np)`` returns a
    list of dicts with ``feats`` of shape ``(D,)``.
    """

    def __init__(self, target_dim: int = 8):
        self.target_dim = target_dim

    def generate(self, wav_list, **kwargs):  # noqa: ANN001
        out = []
        for wav in wav_list:
            assert isinstance(wav, np.ndarray)
            scalar = float(wav.mean())
            feats = np.full((self.target_dim,), scalar, dtype=np.float32)
            out.append({"feats": feats, "labels": [], "scores": []})
        return out


def test_emotion2vec_returns_per_chunk_embeddings():
    cfg = Emotion2vecConfig(target_dim=8, target_layernorm=False)
    teacher = Emotion2vecTeacher(cfg, model=_FakeFunasrModel(8))
    audio = torch.zeros(2, 3, 16000)
    chunk_mask = torch.tensor([[True, True, True], [True, False, False]])
    out = teacher(audio, chunk_mask=chunk_mask)

    assert out.shape == (2, 3, 8)
    assert torch.all(out[1, 1:] == 0)


def test_emotion2vec_dim_mismatch_raises():
    cfg = Emotion2vecConfig(target_dim=99, target_layernorm=False)
    teacher = Emotion2vecTeacher(cfg, model=_FakeFunasrModel(8))
    try:
        teacher(torch.zeros(1, 1, 16000))
    except ValueError as exc:
        assert "emotion2vec" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on dim mismatch")


def test_build_teacher_dispatches_emotion2vec(monkeypatch):
    class _Cfg:
        kind = "emotion2vec"
        model_id = "fake"
        hub = "hf"
        target_dim = 8
        pooled = "chunked"
        target_layernorm = False

    import sdm.data.teachers.emotion2vec as mod

    orig = mod.Emotion2vecTeacher.__init__

    def _patched(self, cfg, *, device="cpu", model=None):
        orig(self, cfg, device=device, model=_FakeFunasrModel(8))

    monkeypatch.setattr(mod.Emotion2vecTeacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, Emotion2vecTeacher)
