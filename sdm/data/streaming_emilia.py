"""Streaming, language-shuffled loader for `amphion/Emilia-Dataset`.

Emits fixed-shape chunked audio tensors so the per-experiment distillation runs
can pool both backbone and teacher representations onto identical 1-second
chunk grids. The HF dataset is opened in `streaming=True` mode and shuffled
with a buffer so the per-language shards interleave naturally.

The ``records_iter`` argument lets tests inject a synthetic iterable; in
production it defaults to opening the HF dataset.
"""

from __future__ import annotations

import os
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


def _sanitize_waveform(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    max_abs = float(np.max(np.abs(array)))
    if max_abs > 2.0:
        array = array / max_abs
    elif max_abs > 1.0:
        array = np.clip(array, -1.0, 1.0)
    return array.astype(np.float32, copy=False)


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

    _disable_audio_encode_torchcodec()
    ds = load_dataset(cfg.repo_id, split=cfg.split, streaming=cfg.streaming, **hf_token_kwargs())
    ds = _cast_audio_to_plain_dict(ds)
    rank, world_size = _xla_worker_info()
    if world_size > 1 and hasattr(ds, "shard"):
        ds = ds.shard(num_shards=world_size, index=rank)
    if cfg.streaming and cfg.shuffle_buffer > 0:
        ds = ds.shuffle(seed=cfg.seed + rank, buffer_size=_effective_shuffle_buffer(cfg, world_size))
    return ds


def _xla_worker_info() -> tuple[int, int]:
    try:
        from sdm.train import xla_utils

        if xla_utils.is_xla():
            return xla_utils.global_ordinal(), xla_utils.world_size()
    except Exception:
        pass
    return 0, 1


def _effective_shuffle_buffer(cfg: EmiliaConfig, world_size: int) -> int:
    override = os.environ.get("SDM_EMILIA_SHUFFLE_BUFFER")
    if override is not None:
        return max(1, int(override))
    if world_size <= 1:
        return cfg.shuffle_buffer
    cap = int(os.environ.get("SDM_TPU_SHUFFLE_BUFFER_CAP", "1024"))
    return max(1, min(cfg.shuffle_buffer, cap))


_AUDIO_ENCODE_PATCHED = False


def _disable_audio_encode_torchcodec() -> None:
    """Monkey-patch ``datasets.features.audio.Audio.encode_example``.

    HF >=4 imports ``torchcodec`` whenever a streaming example contains an
    audio dict with ``array`` + ``sampling_rate`` so it can re-encode it to
    bytes. We never need that round-trip on TPUs (the loader decodes the raw
    array directly), so replace ``encode_example`` with a passthrough that
    keeps the original dict and only delegates to the upstream implementation
    for the cheap path-only / bytes-only cases.
    """

    global _AUDIO_ENCODE_PATCHED
    if _AUDIO_ENCODE_PATCHED:
        return

    try:
        from datasets.features.audio import Audio  # type: ignore
    except Exception:  # pragma: no cover - datasets unavailable
        return

    original = Audio.encode_example

    def _safe_encode_example(self, value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict) and "array" in value:
            # Already-decoded HF audio dict. Keep it as-is so downstream
            # consumers can read ``array`` / ``sampling_rate`` without going
            # through TorchCodec.
            return value
        return original(self, value)

    Audio.encode_example = _safe_encode_example  # type: ignore[assignment]
    _AUDIO_ENCODE_PATCHED = True


def _cast_audio_to_plain_dict(ds: Any) -> Any:
    """Best-effort: drop the HF Audio feature so iteration skips re-encoding.

    Streaming ``IterableDataset.cast`` does not always re-route the encode
    path used by ``_apply_feature_types_on_example`` (the original feature
    info is preserved on the underlying ex_iterable). We still apply the cast
    where possible, but we also forcibly clear ``_info.features`` so HF will
    treat each example as an unstructured dict and never call
    ``Audio.encode_example`` at all. The :func:`_disable_audio_encode_torchcodec`
    monkey-patch acts as a final safety net.
    """

    features = getattr(ds, "features", None)
    if features and "audio" in features:
        try:
            from datasets import Audio, Features, Sequence, Value  # type: ignore

            if isinstance(features["audio"], Audio):
                plain_features = dict(features)
                plain_features["audio"] = {
                    "array": Sequence(Value("float32")),
                    "sampling_rate": Value("int64"),
                    "path": Value("string"),
                    "bytes": Value("binary"),
                }
                ds = ds.cast(Features(plain_features))
        except Exception:
            pass

    info = getattr(ds, "_info", None)
    if info is not None and getattr(info, "features", None) is not None:
        try:
            info.features = None
        except Exception:
            pass

    return ds


def _read_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # type: ignore

    array, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(array, dtype=np.float32), int(sample_rate)


def _read_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf  # type: ignore

    try:
        array, sample_rate = sf.read(BytesIO(payload), dtype="float32", always_2d=False)
        return np.asarray(array, dtype=np.float32), int(sample_rate)
    except Exception:
        # libsndfile may not support the codec (e.g. old builds + mp3). Fall back
        # to librosa, which uses audioread/ffmpeg under the hood.
        import tempfile

        import librosa  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as fh:
            fh.write(payload)
            fh.flush()
            array, sample_rate = librosa.load(fh.name, sr=None, mono=False)
        return np.asarray(array, dtype=np.float32), int(sample_rate)


def _read_audio_path(path: str) -> tuple[np.ndarray, int]:
    if "://" not in path and "::" not in path:
        return _read_audio_file(path)

    from datasets.download.download_config import DownloadConfig  # type: ignore
    from datasets.utils.file_utils import xopen  # type: ignore

    with xopen(path, "rb", download_config=DownloadConfig(token=hf_token())) as handle:
        return _read_audio_bytes(handle.read())


_WEBDATASET_AUDIO_KEYS = ("flac", "wav", "mp3", "ogg", "opus", "m4a")


def _extract_audio(record: dict[str, Any]) -> tuple[np.ndarray, int]:
    audio = record.get("audio")
    if audio is None:
        # WebDataset-style records (e.g. amphion/Emilia-Dataset) use the file
        # extension as the key and store raw bytes as the value.
        for key in _WEBDATASET_AUDIO_KEYS:
            payload = record.get(key)
            if payload is None:
                continue
            if isinstance(payload, (bytes, bytearray, memoryview)):
                return _read_audio_bytes(bytes(payload))
            if isinstance(payload, dict):
                audio = payload
                break
            if isinstance(payload, str):
                return _read_audio_path(payload)
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


def _record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return ``record['json']`` as a parsed dict if present.

    WebDataset Emilia shards bundle metadata in a sibling ``json`` value that
    can either already be a dict (HF parses it) or a raw bytes/str payload.
    """

    raw = record.get("json")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            import json as _json

            return _json.loads(bytes(raw).decode("utf-8"))
        except Exception:
            return {}
    if isinstance(raw, str):
        try:
            import json as _json

            return _json.loads(raw)
        except Exception:
            return {}
    return {}


def _extract_id(record: dict[str, Any]) -> str:
    for key in ("id", "utt_id", "audio_id", "filename", "path", "__key__"):
        if key in record and record[key] is not None:
            return str(record[key])
    meta = _record_metadata(record)
    for key in ("id", "utt_id", "audio_id", "filename", "path"):
        if meta.get(key) is not None:
            return str(meta[key])
    audio = record.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return ""


def _extract_language(record: dict[str, Any]) -> str | None:
    for key in ("language", "lang", "lang_code"):
        if key in record and record[key] is not None:
            return str(record[key])
    meta = _record_metadata(record)
    for key in ("language", "lang", "lang_code"):
        if meta.get(key) is not None:
            return str(meta[key])
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
        array = _sanitize_waveform(array)
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


def make_collate(
    *,
    teacher_processor_id: str | None = None,
    sample_rate: int = 16000,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate fn that optionally runs an HF AutoFeatureExtractor.

    When ``teacher_processor_id`` is set, the resulting collate emits an extra
    ``teacher_audio`` tensor of the same shape as ``audio`` (B, N, T) but
    pre-processed by the teacher's HF feature extractor. This is the canonical
    HF input path: e.g. for ``facebook/wav2vec2-xls-r-300m`` the extractor
    applies per-clip zero-mean / unit-variance normalization, which is required
    by ``feat_extract_norm="layer"`` models. For ``mHuBERT-147`` the extractor
    is a no-op since ``do_normalize=False``.

    The student backbone keeps consuming raw ``audio`` directly, matching its
    own processor's behaviour (mHuBERT: ``do_normalize=False``).
    """

    if teacher_processor_id is None:
        return collate

    from transformers import AutoFeatureExtractor  # type: ignore

    extractor = AutoFeatureExtractor.from_pretrained(
        teacher_processor_id, **hf_token_kwargs()
    )

    def _collate_with_processor(batch: list[dict[str, Any]]) -> dict[str, Any]:
        out = collate(batch)
        audio = out["audio"]  # (B, N, T) float32 cpu
        b, n, t = audio.shape
        flat_np = audio.reshape(b * n, t).numpy()
        # All chunks are exactly ``t`` samples (zero-padded for trailing chunks)
        # so we can disable padding and skip attention_mask: the processor
        # only normalizes per-clip. Padded all-zero chunks normalize to zeros
        # (eps in the variance term avoids div-by-zero), and the per-chunk
        # ``chunk_mask`` keeps them out of the loss downstream.
        processed = extractor(
            list(flat_np),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False,
        )
        teacher_audio = processed["input_values"].reshape(b, n, t).to(torch.float32)
        out["teacher_audio"] = teacher_audio
        return out

    return _collate_with_processor


__all__ = [
    "EmiliaConfig",
    "StreamingEmiliaDataset",
    "_cast_audio_to_plain_dict",
    "chunk_audio",
    "collate",
    "iter_chunks",
    "make_collate",
    "samples_per_chunk",
]
