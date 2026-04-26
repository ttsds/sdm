"""xla_utils must degrade cleanly on non-XLA hosts (CPU/GPU)."""

import torch
import pytest

from sdm.train import xla_utils


def test_helpers_on_cpu():
    assert xla_utils.world_size() == 1
    assert xla_utils.global_ordinal() == 0
    assert xla_utils.is_master() is True
    assert xla_utils.get_device().type in {"cpu", "cuda"}


def test_required_xla_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.delenv("SDM_FORCE_CPU", raising=False)
    monkeypatch.setattr(
        xla_utils,
        "_import_torch_xla",
        lambda: (_ for _ in ()).throw(ImportError("missing torch_xla")),
    )

    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        xla_utils.is_xla()


def test_force_cpu_keeps_explicit_local_fallback(monkeypatch):
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    monkeypatch.setenv("SDM_FORCE_CPU", "1")

    assert xla_utils.xla_required() is False
    assert xla_utils.get_device().type in {"cpu", "cuda"}


def test_shard_module_fsdp_is_noop_off_xla():
    m = torch.nn.Linear(4, 4)
    out = xla_utils.shard_module_fsdp(m)
    assert out is m


def test_loader_per_device_moves_tensors():
    device = xla_utils.get_device()
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "ids": ["a", "b"]}
    loader = [batch]
    wrapped = xla_utils.loader_per_device(loader, device)
    out = next(iter(wrapped))
    assert out["input_ids"].device == device
    assert out["ids"] == ["a", "b"]
