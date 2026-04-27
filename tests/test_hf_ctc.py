from __future__ import annotations


import torch
from torch import nn

from sdm.data.teachers import build_teacher
from sdm.data.teachers.hf_ctc import HfCtcConfig, HfCtcTeacher


class _FakeOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeCtcModel(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_values, output_hidden_states=True, return_dict=True):
        x = input_values.unsqueeze(-1)
        h0 = self.proj(x)
        h1 = h0 + 1.0
        h2 = h1 + 1.0
        return _FakeOutputs((h0, h1, h2))


def test_hf_ctc_pools_to_chunks():
    fake = _FakeCtcModel(hidden_size=8)
    teacher = HfCtcTeacher(HfCtcConfig(model_id="fake", layer=-1, target_dim=8), model=fake)
    audio = torch.randn(2, 3, 16)
    out = teacher(audio)
    assert out.shape == (2, 3, 8)


def test_hf_ctc_applies_chunk_mask():
    fake = _FakeCtcModel(hidden_size=8)
    teacher = HfCtcTeacher(HfCtcConfig(model_id="fake", layer=1, target_dim=8), model=fake)
    audio = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    out = teacher(audio, chunk_mask=mask)
    assert torch.all(out[0, 2] == 0)
    assert torch.all(out[0, 3] == 0)


def test_build_teacher_dispatches_hf_ctc(monkeypatch):
    fake = _FakeCtcModel(hidden_size=8)

    class _Cfg:
        kind = "hf_ctc"
        model_id = "fake"
        layer = -1
        target_dim = 8
        pooled = "chunked"
        target_layernorm = True

    import sdm.data.teachers.hf_ctc as mod

    orig_init = mod.HfCtcTeacher.__init__

    def _patched(self, cfg, *, device="cpu", model=None):
        orig_init(self, cfg, device=device, model=fake)

    monkeypatch.setattr(mod.HfCtcTeacher, "__init__", _patched)
    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, HfCtcTeacher)
    assert teacher(torch.randn(1, 2, 4)).shape == (1, 2, 8)
