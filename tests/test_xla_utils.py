"""xla_utils must degrade cleanly on non-XLA hosts (CPU/GPU)."""

import torch

from sdm.train import xla_utils


def test_helpers_on_cpu():
    assert xla_utils.world_size() == 1
    assert xla_utils.global_ordinal() == 0
    assert xla_utils.is_master() is True
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
