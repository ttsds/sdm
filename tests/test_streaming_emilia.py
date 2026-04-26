from __future__ import annotations

import numpy as np
import torch

from sdm.data.streaming_emilia import (
    EmiliaConfig,
    StreamingEmiliaDataset,
    chunk_audio,
    collate,
    iter_chunks,
    samples_per_chunk,
)


def _fake_record(seconds: float, sr: int = 16000, *, idx: int = 0, language: str = "en") -> dict:
    n = int(seconds * sr)
    rng = np.random.default_rng(idx)
    return {
        "audio": {"array": rng.standard_normal(n).astype(np.float32), "sampling_rate": sr},
        "id": f"utt-{idx}",
        "language": language,
    }


def test_chunk_audio_truncates_and_pads():
    spc = 16
    # 2.5 chunks worth of samples -> expect 3 valid chunks, last padded.
    waveform = np.arange(40, dtype=np.float32)
    audio, mask, n = chunk_audio(waveform, samples_per_chunk=spc, max_chunks=8)

    assert audio.shape == (8, spc)
    assert mask.shape == (8,)
    assert n == 3
    assert mask.tolist() == [True, True, True, False, False, False, False, False]
    # First chunk preserved verbatim
    assert torch.equal(audio[0], torch.arange(0, 16, dtype=torch.float32))
    # Last valid chunk: 8 real samples + 8 zeros
    assert torch.equal(audio[2, :8], torch.arange(32, 40, dtype=torch.float32))
    assert torch.all(audio[2, 8:] == 0)


def test_chunk_audio_caps_at_max_chunks():
    spc = 4
    waveform = np.zeros(100, dtype=np.float32)
    _, mask, n = chunk_audio(waveform, samples_per_chunk=spc, max_chunks=5)
    assert n == 5
    assert mask.sum().item() == 5


def test_iter_chunks_emits_fixed_shape():
    cfg = EmiliaConfig(sample_rate=16000, chunk_seconds=1.0, max_chunks=4)
    records = [_fake_record(seconds=2.5, idx=i) for i in range(3)]
    items = list(iter_chunks(cfg, records_iter=records))

    spc = samples_per_chunk(cfg)
    assert len(items) == 3
    for item in items:
        assert item["audio"].shape == (4, spc)
        assert item["chunk_mask"].shape == (4,)
        assert item["n_chunks"] == 3
        assert item["language"] == "en"
        assert item["id"].startswith("utt-")


def test_iter_chunks_take_limits_records():
    cfg = EmiliaConfig(sample_rate=8000, chunk_seconds=0.5, max_chunks=2)
    records = (_fake_record(seconds=1.0, sr=8000, idx=i) for i in range(10))
    items = list(iter_chunks(cfg, records_iter=records, take=4))
    assert len(items) == 4


def test_collate_stacks_batch():
    cfg = EmiliaConfig(sample_rate=16000, chunk_seconds=1.0, max_chunks=2)
    records = [_fake_record(seconds=1.5, idx=i) for i in range(3)]
    batch = list(iter_chunks(cfg, records_iter=records))
    out = collate(batch)

    spc = samples_per_chunk(cfg)
    assert out["audio"].shape == (3, 2, spc)
    assert out["chunk_mask"].shape == (3, 2)
    assert out["n_chunks"].tolist() == [2, 2, 2]
    assert out["ids"] == ["utt-0", "utt-1", "utt-2"]


def test_streaming_dataset_iterates():
    cfg = EmiliaConfig(sample_rate=8000, chunk_seconds=1.0, max_chunks=3)
    records = [_fake_record(seconds=2.0, sr=8000, idx=i) for i in range(2)]
    ds = StreamingEmiliaDataset(cfg, records_iter=records)
    items = list(ds)
    assert len(items) == 2
    assert items[0]["audio"].dtype == torch.float32
