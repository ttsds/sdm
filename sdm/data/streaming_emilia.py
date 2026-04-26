"""Streaming, language-shuffled loader for `amphion/Emilia-Dataset`.

Emits fixed-shape chunked audio tensors so the per-experiment distillation runs
can pool both backbone and teacher representations onto identical 1-second
chunk grids. The HF dataset is opened in `streaming=True` mode and shuffled
with a buffer so the per-language shards interleave naturally.

The ``records_iter`` argument lets tests inject a synthetic iterable; in
production it defaults to opening the HF dataset.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

from sdm.dotenv import hf_token, hf_token_kwargs


@dataclass
class EmiliaConfig:
    repo_id: str = "amphion/Emilia-Dataset"
    split: str = "train"
    streaming: bool = True
    shuffle_buffer: int = 10000
    seed: int = 0
    fraction: float = 1.0
    sample_rate: int = 16000
    chunk_seconds: float = 1.0
    max_chunks: int = 32
    num_workers: int = 0


def samples_per_chunk(cfg: EmiliaConfig) -> int:
    return int(round(cfg.sample_rate * cfg.chunk_seconds))


def _resample_if_needed(array: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return array
    # Lazy import to keep CPU-only test paths cheap.
    import librosa  # type: ignore

    return librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def _to_mono(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        # HF Audio occasionally returns (channels, samples) or (samples, channels).
        if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
            return array.mean(axis=0)
        return array.mean(axis=-1)
    raise ValueError(f"unexpected audio rank {array.ndim}")


def chunk_audio(
    waveform: np.ndarray,
    *,
    samples_per_chunk: int,
    max_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Split a 1-D waveform into fixed-size chunks.

    Returns ``(audio, mask, n_chunks)`` where ``audio`` has shape
    ``(max_chunks, samples_per_chunk)``, padded with zeros, and ``mask`` is a
    bool tensor marking valid chunks.
    """

    if waveform.ndim != 1:
        raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")

    total = waveform.shape[0]
    n_full = total // samples_per_chunk
    remainder = total - n_full * samples_per_chunk
    n_chunks = min(max_chunks, n_full + (1 if remainder > 0 else 0))
    if n_chunks <= 0:
        n_chunks = 1  # always emit at least one chunk; shorter clips get padded

    audio = np.zeros((max_chunks, samples_per_chunk), dtype=np.float32)
    mask = np.zeros((max_chunks,), dtype=bool)
    for i in range(n_chunks):
        start = i * samples_per_chunk
        end = min(start + samples_per_chunk, total)
        if start >= total:
            break
        chunk = waveform[start:end]
        audio[i, : chunk.shape[0]] = chunk
        mask[i] = True

    valid = int(mask.sum())
    return torch.from_numpy(audio), torch.from_numpy(mask), valid


def _open_emilia_stream(cfg: EmiliaConfig) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(cfg.repo_id, split=cfg.split, streaming=cfg.streaming, **hf_token_kwargs())
    ds = _cast_audio_to_plain_dict(ds)
    if cfg.streaming and cfg.shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=cfg.shuffle_buffer)
    return ds


def _cast_audio_to_plain_dict(ds: Any) -> Any:
    """Keep HF Audio columns as raw dicts so TorchCodec is not needed on TPUs."""

    features = getattr(ds, "features", None)
    if not features or "audio" not in features:
        return ds

    from datasets import Audio, Features, Sequence, Value  # type: ignore

    if not isinstance(features["audio"], Audio):
        return ds

    plain_features = dict(features)
    plain_features["audio"] = {
        "array": Sequence(Value("float32")),
        "sampling_rate": Value("int64"),
        "path": Value("string"),
        "bytes": Value("binary"),
    }
    return ds.cast(Features(plain_features))


def _read_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # type: ignore

    array, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(array, dtype=np.float32), int(sample_rate)


def _read_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf  # type: ignore

    array, sample_rate = sf.read(BytesIO(payload), dtype="float32", always_2d=False)
    return np.asarray(array, dtype=np.float32), int(sample_rate)


def _read_audio_path(path: str) -> tuple[np.ndarray, int]:
    if "://" not in path and "::" not in path:
        return _read_audio_file(path)

    from datasets.download.download_config import DownloadConfig  # type: ignore
    from datasets.utils.file_utils import xopen  # type: ignore

    with xopen(path, "rb", download_config=DownloadConfig(token=hf_token())) as handle:
        return _read_audio_bytes(handle.read())


def _extract_audio(record: dict[str, Any]) -> tuple[np.ndarray, int]:
    audio = record.get("audio")
    if audio is None:
        raise KeyError(f"record missing 'audio' field; keys={list(record)}")
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            array = np.asarray(audio["array"])
            sr = int(audio["sampling_rate"])
            return array.astype(np.float32, copy=False), sr
        if audio.get("bytes") is not None:
            return _read_audio_bytes(audio["bytes"])
        if audio.get("path") is not None:
            return _read_audio_path(str(audio["path"]))
        raise ValueError(f"audio dict has no array, bytes, or path: keys={list(audio)}")
    else:  # tuple/list fallback
        array, sr = audio
        array = np.asarray(array)
        sr = int(sr)
    return array.astype(np.float32, copy=False), sr


def _extract_id(record: dict[str, Any]) -> str:
    for key in ("id", "utt_id", "audio_id", "filename", "path"):
        if key in record and record[key] is not None:
            return str(record[key])
    audio = record.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return ""


def _extract_language(record: dict[str, Any]) -> str | None:
    for key in ("language", "lang", "lang_code"):
        if key in record and record[key] is not None:
            return str(record[key])
    return None


def iter_chunks(
    cfg: EmiliaConfig,
    *,
    records_iter: Iterable[dict[str, Any]] | Callable[[EmiliaConfig], Iterable[dict[str, Any]]] | None = None,
    take: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield chunked-audio examples for distillation training."""

    if records_iter is None:
        source: Iterable[dict[str, Any]] = _open_emilia_stream(cfg)
    elif callable(records_iter):
        source = records_iter(cfg)
    else:
        source = records_iter

    spc = samples_per_chunk(cfg)
    emitted = 0
    for record in source:
        array, sr = _extract_audio(record)
        array = _to_mono(array)
        array = _resample_if_needed(array, sr, cfg.sample_rate)
        audio, mask, n = chunk_audio(array, samples_per_chunk=spc, max_chunks=cfg.max_chunks)
        yield {
            "audio": audio,
            "chunk_mask": mask,
            "n_chunks": n,
            "id": _extract_id(record),
            "language": _extract_language(record),
        }
        emitted += 1
        if take is not None and emitted >= take:
            break


class StreamingEmiliaDataset(IterableDataset):
    """Thin IterableDataset wrapper around :func:`iter_chunks`."""

    def __init__(
        self,
        cfg: EmiliaConfig,
        *,
        records_iter: Iterable[dict[str, Any]] | Callable[[EmiliaConfig], Iterable[dict[str, Any]]] | None = None,
        take: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._records_iter = records_iter
        self._take = take

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter_chunks(self.cfg, records_iter=self._records_iter, take=self._take)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    audio = torch.stack([b["audio"] for b in batch], dim=0)
    chunk_mask = torch.stack([b["chunk_mask"] for b in batch], dim=0)
    n_chunks = torch.tensor([b["n_chunks"] for b in batch], dtype=torch.long)
    return {
        "audio": audio,
        "chunk_mask": chunk_mask,
        "n_chunks": n_chunks,
        "ids": [b["id"] for b in batch],
        "languages": [b["language"] for b in batch],
    }


__all__ = [
    "EmiliaConfig",
    "StreamingEmiliaDataset",
    "_cast_audio_to_plain_dict",
    "chunk_audio",
    "collate",
    "iter_chunks",
    "samples_per_chunk",
]
