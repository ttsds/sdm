"""Read teacher target shards produced by `scripts/extract_teacher_targets.py`
and pair them with NeuCodec input ids ready for distillation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from sdm.data.neucodec_dataset import codes_to_input_ids


@dataclass
class TeacherCacheConfig:
    shard_dir: Path
    teachers: list[str]  # names matching the shard's *_concat / *_lengths keys
    max_length: int = 2048
    pooled: dict[str, bool] | None = None  # True if a teacher's target is per-utterance


def _slice(concat: np.ndarray, lengths: np.ndarray, idx: int) -> np.ndarray:
    start = int(lengths[:idx].sum())
    end = start + int(lengths[idx])
    return concat[start:end]


class TeacherShardDataset(IterableDataset):
    """Streams (input_ids, attention_mask, targets) from .npz shards on disk."""

    def __init__(self, cfg: TeacherCacheConfig):
        self.cfg = cfg
        self._shards = sorted(Path(cfg.shard_dir).glob("shard-*.npz"))
        if not self._shards:
            raise FileNotFoundError(f"No shards under {cfg.shard_dir}")

    def __iter__(self) -> Iterator[dict]:
        pooled = self.cfg.pooled or {}
        for shard in self._shards:
            with np.load(shard, allow_pickle=True) as data:
                ids = data["ids"]
                codes_concat = data["codes_concat"]
                codes_lengths = data["codes_lengths"]
                tcaches = {
                    name: (data[f"{name}_concat"], data[f"{name}_lengths"])
                    for name in self.cfg.teachers
                }
                for i in range(len(ids)):
                    codes = _slice(codes_concat, codes_lengths, i)
                    input_ids, attn = codes_to_input_ids(codes.tolist(), self.cfg.max_length)
                    targets: dict[str, torch.Tensor] = {}
                    for name, (concat, lengths) in tcaches.items():
                        flat = _slice(concat, lengths, i)
                        if pooled.get(name, False):
                            targets[name] = torch.from_numpy(flat.astype(np.float32))
                        else:
                            # Truncate / right-pad to model max length minus CLS+SEP.
                            target_dim = flat.shape[-1] if flat.ndim > 1 else 1
                            arr = flat.reshape(-1, target_dim).astype(np.float32)
                            T = min(arr.shape[0], self.cfg.max_length - 2)
                            buf = np.zeros((self.cfg.max_length, target_dim), dtype=np.float32)
                            buf[1 : 1 + T] = arr[:T]  # align with CLS-prefixed input_ids
                            targets[name] = torch.from_numpy(buf)
                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attn,
                        "targets": targets,
                        "id": str(ids[i]),
                    }


def collate_teacher_batch(batch: list[dict]) -> dict:
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "ids": [b["id"] for b in batch],
    }
    target_keys = list(batch[0]["targets"])
    out["targets"] = {
        k: torch.stack([b["targets"][k] for b in batch], dim=0) for k in target_keys
    }
    return out
