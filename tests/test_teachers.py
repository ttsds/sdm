from __future__ import annotations

import torch
from torch import nn

from sdm.data.teachers import build_teacher
from sdm.data.teachers.hf_ssl import HfSslConfig, HfSslTeacher


class _FakeOutputs:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeSslModel(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_values, output_hidden_states=True, return_dict=True):
        x = input_values.unsqueeze(-1)
        h0 = self.proj(x)
        h1 = h0 + 1.0
        h2 = h1 + 1.0
        return _FakeOutputs((h0, h1, h2))


def test_hf_ssl_pools_to_chunks():
    fake = _FakeSslModel(hidden_size=8)
    teacher = HfSslTeacher(HfSslConfig(model_id="fake", layer=2, target_dim=8), model=fake)
    audio = torch.randn(2, 3, 16)
    out = teacher(audio)
    assert out.shape == (2, 3, 8)


def test_hf_ssl_applies_chunk_mask():
    fake = _FakeSslModel(hidden_size=8)
    teacher = HfSslTeacher(HfSslConfig(model_id="fake", layer=1, target_dim=8), model=fake)
    audio = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    out = teacher(audio, chunk_mask=mask)
    assert torch.all(out[0, 2] == 0)
    assert torch.all(out[0, 3] == 0)
    assert not torch.all(out[0, 0] == 0)


def test_hf_ssl_target_dim_mismatch_raises():
    fake = _FakeSslModel(hidden_size=8)
    teacher = HfSslTeacher(HfSslConfig(model_id="fake", layer=0, target_dim=99), model=fake)
    try:
        teacher(torch.randn(1, 1, 4))
    except ValueError as exc:
        assert "target_dim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected target_dim mismatch")


def test_build_teacher_dispatches_hf_ssl():
    fake = _FakeSslModel(hidden_size=8)

    class _CfgObj:
        kind = "hf_ssl"
        model_id = "fake"
        layer = 1
        target_dim = 8
        pooled = "chunked"

    # Patch the AutoModel call so dispatch path doesn't hit HF.
    import sdm.data.teachers.hf_ssl as hf_ssl_mod

    orig_init = hf_ssl_mod.HfSslTeacher.__init__

    def _patched_init(self, cfg, *, device="cpu", model=None):
        orig_init(self, cfg, device=device, model=fake)

    hf_ssl_mod.HfSslTeacher.__init__ = _patched_init  # type: ignore[assignment]
    try:
        teacher = build_teacher(_CfgObj())
        assert isinstance(teacher, HfSslTeacher)
        assert teacher(torch.randn(1, 2, 4)).shape == (1, 2, 8)
    finally:
        hf_ssl_mod.HfSslTeacher.__init__ = orig_init  # type: ignore[assignment]


def test_build_teacher_unknown_kind_raises():
    class _CfgObj:
        kind = "definitely-not-real"

    try:
        build_teacher(_CfgObj())
    except NotImplementedError as exc:
        assert "definitely-not-real" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError")
