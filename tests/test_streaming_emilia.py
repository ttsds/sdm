from __future__ import annotations

import numpy as np
import torch

from sdm.data.streaming_emilia import (
    EmiliaConfig,
    StreamingEmiliaDataset,
    _cast_audio_to_plain_dict,
    _extract_audio,
    _open_emilia_stream,
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


def test_open_emilia_stream_passes_hf_token(monkeypatch):
    captured = {}

    class _FakeStream(list):
        features = {}

        def shuffle(self, *, seed, buffer_size):
            captured["shuffle"] = (seed, buffer_size)
            return self

    def _fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeStream()

    import datasets

    monkeypatch.setenv("hf_token", "secret-token")
    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    cfg = EmiliaConfig(repo_id="private/repo", split="train", streaming=True, shuffle_buffer=4, seed=7)
    _open_emilia_stream(cfg)

    assert captured["args"] == ("private/repo",)
    assert captured["kwargs"]["token"] == "secret-token"
    assert captured["shuffle"] == (7, 4)


def test_cast_audio_to_plain_dict_avoids_audio_feature():
    from datasets import Audio, Features, Value

    class _FakeStream(list):
        features = Features({"audio": Audio(), "id": Value("string")})

        def cast(self, features):
            self.features = features
            return self

    ds = _cast_audio_to_plain_dict(_FakeStream())

    assert not isinstance(ds.features["audio"], Audio)
    assert set(ds.features["audio"]) == {"array", "sampling_rate", "path", "bytes"}


def test_extract_audio_reads_embedded_wav_bytes():
    import soundfile as sf
    from io import BytesIO

    buffer = BytesIO()
    sf.write(buffer, np.linspace(-0.5, 0.5, 8, dtype=np.float32), 8000, format="WAV")

    array, sample_rate = _extract_audio({"audio": {"bytes": buffer.getvalue(), "path": None}})

    assert sample_rate == 8000
    assert array.shape == (8,)
    assert array.dtype == np.float32
