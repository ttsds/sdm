from __future__ import annotations

import torch
from torch import nn

from sdm.modeling.distill_model import BackboneConfig, build_backbone, load_backbone


class _FakeOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int = 6, num_hidden_layers: int = 2):
        super().__init__()
        self.config = type(
            "FakeConfig",
            (),
            {"hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers},
        )()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_values, attention_mask=None, output_hidden_states=True, return_dict=True):
        x = input_values.unsqueeze(-1)
        hidden0 = self.proj(x)
        hidden1 = hidden0 + 1.0
        hidden2 = hidden1 + 1.0
        return _FakeOutput((hidden0, hidden1, hidden2))


def test_build_backbone_pools_chunks(monkeypatch, tmp_path):
    fake = _FakeBackbone(hidden_size=6)
    monkeypatch.setattr(
        "sdm.modeling.distill_model.AutoModel.from_pretrained",
        lambda model_id, **kwargs: fake,
    )

    model = build_backbone(BackboneConfig(model_id="fake/mhubert", hidden_size=6, layer_idx=-1))
    audio = torch.randn(2, 3, 5)
    out = model(audio)

    assert out.shape == (2, 3, 6)
    ckpt = tmp_path / "backbone.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    restored = load_backbone(ckpt, model_id="fake/mhubert")
    restored_out = restored(audio)
    assert restored_out.shape == (2, 3, 6)


def test_build_backbone_checks_hidden_size(monkeypatch):
    monkeypatch.setattr(
        "sdm.modeling.distill_model.AutoModel.from_pretrained",
        lambda model_id, **kwargs: _FakeBackbone(hidden_size=5),
    )

    try:
        build_backbone(BackboneConfig(model_id="fake/mhubert", hidden_size=6))
    except ValueError as exc:
        assert "hidden_size mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected hidden-size mismatch")


def test_build_backbone_uses_hf_token(monkeypatch):
    captured = {}

    def _fake_from_pretrained(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return _FakeBackbone(hidden_size=6)

    monkeypatch.setenv("hf_token", "secret-token")
    monkeypatch.setattr("sdm.modeling.distill_model.AutoModel.from_pretrained", _fake_from_pretrained)

    build_backbone(BackboneConfig(model_id="private/backbone", hidden_size=6))

    assert captured["model_id"] == "private/backbone"
    assert captured["kwargs"]["token"] == "secret-token"