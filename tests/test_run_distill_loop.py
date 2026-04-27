from __future__ import annotations

import numpy as np
import torch
from torch import nn

from sdm.data.streaming_emilia import EmiliaConfig as StreamCfg
from sdm.data.streaming_emilia import StreamingEmiliaDataset, collate
from sdm.modeling.distill_model import BackboneConfig, build_backbone
from sdm.train import run_distill
from sdm.train.run_distill import (
    DistillConfig,
    DistillTrainConfig,
    EmiliaConfig,
    TeacherConfig,
    _masked_mse,
    load_config,
    train,
)


def test_load_config_dry_run_path():
    cfg = load_config("configs/finetune_xlsr_fairseq.yaml")
    assert cfg.experiment == "sdm-xlsr-fairseq"
    assert cfg.teacher.kind == "hf_ssl"
    assert cfg.backbone.kind == "fairseq_w2v2"


def test_masked_mse_zeroes_out_padding():
    pred = torch.zeros(1, 3, 4)
    target = torch.ones(1, 3, 4)
    full_mask = torch.ones(1, 3, dtype=torch.bool)
    full_loss = _masked_mse(pred, target, full_mask).item()
    assert abs(full_loss - 1.0) < 1e-6

    partial_mask = torch.tensor([[True, False, False]])
    pred2 = torch.zeros(1, 3, 4)
    target2 = torch.zeros(1, 3, 4)
    target2[0, 0] = 2.0  # only the masked-in chunk has error
    target2[0, 1] = 99.0  # padding, must be ignored
    loss = _masked_mse(pred2, target2, partial_mask).item()
    assert abs(loss - 4.0) < 1e-6


def test_masked_mse_uses_float32_for_bfloat16_inputs():
    pred = torch.tensor([[[1.0, 3.0]]], dtype=torch.bfloat16)
    target = torch.tensor([[[0.0, 1.0]]], dtype=torch.bfloat16)
    mask = torch.tensor([[True]])

    loss = _masked_mse(pred, target, mask)

    assert loss.dtype == torch.float32
    assert torch.isclose(loss, torch.tensor(2.5))


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int = 6):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden_size, "num_hidden_layers": 2})()
        self.proj = nn.Linear(1, hidden_size)

    def forward(self, input_values, attention_mask=None, output_hidden_states=True, return_dict=True):
        x = input_values.unsqueeze(-1)
        h0 = self.proj(x)
        return type("O", (), {"hidden_states": (h0, h0 + 1.0)})()


class _FakeTeacher(nn.Module):
    target_dim = 4

    def __init__(self, target_dim: int = 4):
        super().__init__()
        self.target_dim = target_dim

    def __call__(self, audio, *, chunk_mask=None, **_):  # type: ignore[override]
        b, n, _ = audio.shape
        out = torch.zeros(b, n, self.target_dim, dtype=audio.dtype, device=audio.device)
        if chunk_mask is not None:
            out = out * chunk_mask.unsqueeze(-1).to(out.dtype)
        return out


class _FakeSslTeacher(nn.Module):
    def __init__(self, cfg, *, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.target_dim = cfg.target_dim
        self.device = device

    def __call__(self, audio, *, chunk_mask=None):  # type: ignore[override]
        b, n, _ = audio.shape
        return torch.zeros(b, n, self.target_dim, dtype=audio.dtype, device=audio.device)


def _fake_records(n: int, seconds: float = 1.5, sr: int = 16000):
    rng = np.random.default_rng(0)
    for i in range(n):
        yield {
            "audio": {
                "array": rng.standard_normal(int(seconds * sr)).astype(np.float32),
                "sampling_rate": sr,
            },
            "id": f"u{i}",
            "language": "en",
        }


def _patch_fake_feature_extractor(monkeypatch):
    """Stub AutoFeatureExtractor so make_collate doesn't try to hit HF Hub."""

    class _FakeExtractor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return cls()

        def __call__(self, audio_list, sampling_rate, return_tensors, padding):
            arr = np.stack([np.asarray(a, dtype=np.float32) for a in audio_list], axis=0)
            return {"input_values": torch.from_numpy(arr)}

    import transformers

    monkeypatch.setattr(transformers, "AutoFeatureExtractor", _FakeExtractor, raising=False)


def test_build_teacher_hf_ssl_uses_concrete_config(monkeypatch):
    captured = {}

    def _fake_init(self, cfg, *, device="cpu"):
        nn.Module.__init__(self)
        captured["cfg"] = cfg
        captured["device"] = device
        self.target_dim = cfg.target_dim

    monkeypatch.setattr("sdm.data.teachers.hf_ssl.HfSslTeacher.__init__", _fake_init)
    cfg = DistillConfig(
        experiment="test",
        backbone=BackboneConfig(model_id="fake/mhubert", hidden_size=6, layer_idx=-1),
        teacher=TeacherConfig(kind="hf_ssl", target_dim=1024, pooled="chunked", model_id="fake", layer=8),
        data=EmiliaConfig(repo_id="fake"),
        train=DistillTrainConfig(),
    )

    teacher = run_distill._build_teacher(cfg, torch.device("cpu"))

    assert teacher.target_dim == 1024
    assert captured["cfg"].model_id == "fake"
    assert captured["cfg"].layer == 8
    assert captured["device"] == torch.device("cpu")


def test_train_runs_for_few_steps(monkeypatch, tmp_path):
    # Fake mHuBERT
    monkeypatch.setattr(
        "sdm.modeling.distill_model.AutoModel.from_pretrained",
        lambda model_id: _FakeBackbone(hidden_size=6),
    )
    # Fake teacher build path
    monkeypatch.setattr(run_distill, "_build_teacher", lambda cfg, device: _FakeTeacher(target_dim=4))

    # Fake teacher feature extractor (collate path)
    _patch_fake_feature_extractor(monkeypatch)

    # Fake streaming source
    records = list(_fake_records(8, seconds=1.5, sr=16000))
    monkeypatch.setattr(
        "sdm.data.streaming_emilia._open_emilia_stream",
        lambda cfg: iter(records),
    )

    cfg = DistillConfig(
        experiment="test",
        backbone=BackboneConfig(model_id="fake/mhubert", hidden_size=6, layer_idx=-1),
        teacher=TeacherConfig(kind="hf_ssl", target_dim=4, pooled="chunked", model_id="fake", layer=1),
        data=EmiliaConfig(repo_id="fake", sample_rate=16000, chunk_seconds=1.0, max_chunks=2, num_workers=0),
        train=DistillTrainConfig(
            batch_size=2,
            grad_accum=1,
            lr=1e-3,
            warmup_steps=1,
            total_steps=3,
            log_every=1,
            ckpt_every=0,
            ckpt_dir=str(tmp_path),
            resume_from_latest=False,
            fsdp=False,
        ),
    )

    train(cfg)
    # Final checkpoint written
    assert (tmp_path / "final.pt").exists()
    assert (tmp_path / "latest.pt").exists()


def test_train_skips_checkpoint_with_nonfinite_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "sdm.modeling.distill_model.AutoModel.from_pretrained",
        lambda model_id: _FakeBackbone(hidden_size=6),
    )
    monkeypatch.setattr(run_distill, "_build_teacher", lambda cfg, device: _FakeTeacher(target_dim=4))
    _patch_fake_feature_extractor(monkeypatch)
    records = list(_fake_records(6, seconds=1.5, sr=16000))
    monkeypatch.setattr("sdm.data.streaming_emilia._open_emilia_stream", lambda cfg: iter(records))

    clean = build_backbone(BackboneConfig(model_id="fake/mhubert", hidden_size=6), target_dim=4)
    state = clean.state_dict()
    first_key = next(iter(state))
    state[first_key] = state[first_key].clone()
    state[first_key].view(-1)[0] = float("nan")
    torch.save({"model": state, "step": 50}, tmp_path / "latest.pt")

    cfg = DistillConfig(
        experiment="test",
        backbone=BackboneConfig(model_id="fake/mhubert", hidden_size=6, layer_idx=-1),
        teacher=TeacherConfig(kind="hf_ssl", target_dim=4, pooled="chunked", model_id="fake", layer=1),
        data=EmiliaConfig(repo_id="fake", sample_rate=16000, chunk_seconds=1.0, max_chunks=2, num_workers=0),
        train=DistillTrainConfig(
            batch_size=2,
            total_steps=1,
            log_every=1,
            ckpt_every=0,
            ckpt_dir=str(tmp_path),
            resume_from_latest=True,
            fsdp=False,
        ),
    )

    train(cfg)

    out = capsys.readouterr().out
    assert "skipping checkpoint" in out
    assert "contains NaN or Inf" in out
    final = torch.load(tmp_path / "final.pt", map_location="cpu")
    assert final["step"] == 1


