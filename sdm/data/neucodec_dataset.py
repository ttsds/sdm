"""Streaming dataset loader for `neuphonic/emilia-yodas-english-neucodec`.

The dataset emits records with fields:
    id, dnsmos, duration, phone_count, speaker, text, codes (list[int])

`codes` are NeuCodec FSQ tokens at ~50 Hz. We reserve four special token ids
at the bottom of the model vocab and offset every FSQ code by `NUM_SPECIAL`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

PAD_ID = 0
CLS_ID = 1
SEP_ID = 2
MASK_ID = 3
NUM_SPECIAL = 4


def codes_to_input_ids(
    codes: list[int] | np.ndarray,
    max_length: int,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a sequence of NeuCodec codes into (input_ids, attention_mask).

    Codes are shifted by NUM_SPECIAL so model token id 0..3 stay reserved.
    Truncates to fit max_length including CLS/SEP if requested.
    """
    arr = np.asarray(codes, dtype=np.int64) + NUM_SPECIAL

    overhead = 2 if add_special_tokens else 0
    if arr.shape[0] + overhead > max_length:
        arr = arr[: max_length - overhead]

    parts: list[np.ndarray] = []
    if add_special_tokens:
        parts.append(np.array([CLS_ID], dtype=np.int64))
    parts.append(arr)
    if add_special_tokens:
        parts.append(np.array([SEP_ID], dtype=np.int64))
    seq = np.concatenate(parts)

    input_ids = np.full((max_length,), PAD_ID, dtype=np.int64)
    input_ids[: seq.shape[0]] = seq
    attn = np.zeros((max_length,), dtype=np.int64)
    attn[: seq.shape[0]] = 1
    return torch.from_numpy(input_ids), torch.from_numpy(attn)


@dataclass
class NeucodecConfig:
    repo_id: str = "neuphonic/emilia-yodas-english-neucodec"
    split: str = "train"
    max_length: int = 2048
    min_codes: int = 16  # discard utterances shorter than this many tokens
    streaming: bool = True
    shuffle_buffer: int = 10_000
    seed: int = 0


def stream_records(cfg: NeucodecConfig) -> Iterator[dict]:
    """Yield raw dataset records. Imported lazily so torch-only smoke tests
    don't need `datasets` installed.
    """
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(cfg.repo_id, split=cfg.split, streaming=cfg.streaming)
    if cfg.streaming and cfg.shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)

    for record in ds:
        codes = record.get("codes")
        if codes is None or len(codes) < cfg.min_codes:
            continue
        yield record


def iter_examples(cfg: NeucodecConfig) -> Iterator[dict]:
    """Yield model-ready examples: tensors + light metadata."""
    for rec in stream_records(cfg):
        input_ids, attn = codes_to_input_ids(rec["codes"], cfg.max_length)
        yield {
            "input_ids": input_ids,
            "attention_mask": attn,
            "id": rec["id"],
            "speaker": rec.get("speaker", ""),
            "duration": float(rec.get("duration", 0.0)),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "ids": [b["id"] for b in batch],
    }
