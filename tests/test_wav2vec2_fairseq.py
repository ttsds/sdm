"""Smoke tests for the in-tree fairseq wav2vec2 port.

These tests don't hit HuggingFace; they exercise the architecture directly
on a small config to verify shapes, gradient flow, and parity between the
pre-LN and post-LN paths.
"""

from __future__ import annotations

import torch

from sdm.modeling.wav2vec2_fairseq import (
    ConvFeatureExtractionModel,
    Fp32GroupNorm,
    Fp32LayerNorm,
    Wav2Vec2Config,
    Wav2Vec2Model,
)


def _tiny_cfg(**overrides) -> Wav2Vec2Config:
    base = dict(
        encoder_layers=2,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=4,
        conv_feature_layers=[(32, 10, 5), (32, 3, 2), (64, 3, 2)],
        conv_pos=16,
        conv_pos_groups=4,
        conv_bias=True,
    )
    base.update(overrides)
    return Wav2Vec2Config(**base)


def test_feature_extractor_layer_norm_shapes():
    cfg = _tiny_cfg(extractor_mode="layer_norm")
    extractor = ConvFeatureExtractionModel(
        cfg.conv_feature_layers,
        mode="layer_norm",
        conv_bias=cfg.conv_bias,
    )
    x = torch.randn(2, 16000)
    out = extractor(x)
    assert out.shape[0] == 2
    assert out.shape[1] == cfg.conv_feature_layers[-1][0]
    assert torch.isfinite(out).all()


def test_fp32_norms_are_finite_in_bf16_input():
    gn = Fp32GroupNorm(4, 4, affine=True)
    ln = Fp32LayerNorm(8, elementwise_affine=True)
    x_gn = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    x_ln = torch.randn(2, 8, dtype=torch.bfloat16)
    assert torch.isfinite(gn(x_gn)).all()
    assert torch.isfinite(ln(x_ln)).all()


def test_full_forward_layer_norm_first_finite():
    cfg = _tiny_cfg(extractor_mode="layer_norm", layer_norm_first=True)
    model = Wav2Vec2Model(cfg)
    x = torch.randn(2, 16000)
    out = model(x, output_hidden_states=True)
    assert out.last_hidden_state.dim() == 3
    assert out.last_hidden_state.shape[0] == 2
    assert out.last_hidden_state.shape[2] == cfg.encoder_embed_dim
    assert len(out.hidden_states) == cfg.encoder_layers + 1
    assert torch.isfinite(out.last_hidden_state).all()


def test_full_forward_post_ln_finite():
    cfg = _tiny_cfg(extractor_mode="layer_norm", layer_norm_first=False)
    model = Wav2Vec2Model(cfg)
    x = torch.randn(2, 16000)
    out = model(x, output_hidden_states=False)
    assert torch.isfinite(out.last_hidden_state).all()


def test_backward_runs_on_full_forward():
    cfg = _tiny_cfg(extractor_mode="layer_norm", layer_norm_first=True)
    model = Wav2Vec2Model(cfg)
    x = torch.randn(2, 16000)
    out = model(x).last_hidden_state
    loss = out.float().pow(2).mean()
    loss.backward()
    # The positional conv had its weight_norm baked out; check we still got
    # a real gradient on the conv weight.
    grad = model.encoder.pos_conv[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
