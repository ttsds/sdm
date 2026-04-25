"""Smoke tests for distillation heads + loss on synthetic data."""

import torch

from sdm.modeling.deberta_neucodec import SdmConfig, build_model
from sdm.modeling.distillation_heads import (
    DistillationModel,
    HeadSpec,
    distillation_loss,
)


def _tiny_model():
    cfg = SdmConfig(
        fsq_vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        position_buckets=16,
    )
    return cfg, build_model(cfg)


def test_sequence_head_shapes():
    cfg, backbone = _tiny_model()
    spec = HeadSpec(name="hubert", target_dim=16, pooled=False)
    model = DistillationModel(backbone, [spec])
    ids = torch.randint(4, cfg.vocab_size, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    out = model(ids, attn)
    assert "hubert" in out
    assert out["hubert"].shape == (2, 16, 16)


def test_pooled_head_shapes():
    cfg, backbone = _tiny_model()
    spec = HeadSpec(name="whisper", target_dim=24, pooled=True)
    model = DistillationModel(backbone, [spec])
    ids = torch.randint(4, cfg.vocab_size, (3, 16))
    attn = torch.ones(3, 16, dtype=torch.long)
    out = model(ids, attn)
    assert out["whisper"].shape == (3, 24)


def test_distillation_loss_decreases_with_overfit():
    cfg, backbone = _tiny_model()
    spec_a = HeadSpec(name="hubert", target_dim=8, pooled=False)
    spec_b = HeadSpec(name="whisper", target_dim=8, pooled=True)
    model = DistillationModel(backbone, [spec_a, spec_b])

    torch.manual_seed(0)
    ids = torch.randint(4, cfg.vocab_size, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    targets = {
        "hubert": torch.randn(2, 16, 8),
        "whisper": torch.randn(2, 8),
    }

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        optim.zero_grad()
        preds = model(ids, attn)
        loss, _ = distillation_loss(preds, targets, attention_mask=attn)
        loss.backward()
        optim.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.9, f"loss did not decrease enough: {losses[0]} -> {losses[-1]}"
