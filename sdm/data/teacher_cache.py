"""Per-utterance teacher feature extraction for distillation targets.

Each teacher is a callable `(wav: np.ndarray, sr: int) -> np.ndarray` that returns
either a 1-D scalar/vector or a 2-D `(time, dim)` array. They wrap (and where
useful, replicate the internals of) the corresponding TTSDS benchmarks at
`/tmp/ttsds/src/ttsds/benchmarks/`.

This module's heavy deps (transformers, librosa, soundfile, pyworld, pyannote.audio,
neucodec) are imported lazily so that the rest of sdm doesn't pay for them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Teacher(Protocol):
    name: str
    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray: ...


@dataclass
class _SSLTeacher:
    """HuBERT / WavLM / wav2vec2 — same shape, parameterised by model+layer."""

    name: str
    model_id: str
    layer: int
    processor_id: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        import librosa  # noqa: F401, PLC0415
        import torch  # noqa: PLC0415
        from transformers import AutoFeatureExtractor, AutoModel  # noqa: PLC0415

        self._torch = torch
        self._librosa = __import__("librosa")
        self._processor = AutoFeatureExtractor.from_pretrained(
            self.processor_id or self.model_id
        )
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            wav = self._librosa.resample(wav, orig_sr=sr, target_sr=16000)
        inputs = self._processor(wav, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        with self._torch.no_grad():
            out = self._model(input_values, output_hidden_states=True).hidden_states
        return out[self.layer].squeeze(0).cpu().numpy()


def make_hubert(device: str = "cpu") -> Teacher:
    return _SSLTeacher(
        name="hubert",
        model_id="facebook/hubert-base-ls960",
        processor_id="facebook/hubert-large-ls960-ft",
        layer=7,
        device=device,
    )


def make_wavlm(device: str = "cpu") -> Teacher:
    return _SSLTeacher(
        name="wavlm",
        model_id="microsoft/wavlm-base-plus",
        layer=11,
        device=device,
    )


def make_wav2vec2(device: str = "cpu") -> Teacher:
    return _SSLTeacher(
        name="wav2vec2",
        model_id="facebook/wav2vec2-base",
        layer=8,
        device=device,
    )


@dataclass
class _WhisperEncoderPooled:
    """Replicates the per-utterance computation in TTSDS's
    WhisperActivationsBenchmark._get_distribution: mean-pool encoder hidden
    states across layers and across time, returning a fixed-size vector.
    """

    name: str = "whisper"
    model_id: str = "openai/whisper-small.en"
    device: str = "cpu"

    def __post_init__(self) -> None:
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415

        self._torch = torch
        self._librosa = librosa
        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = (
            WhisperForConditionalGeneration.from_pretrained(self.model_id).to(self.device).eval()
        )

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            wav = self._librosa.resample(wav, orig_sr=sr, target_sr=16000)
        feats = self._processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device)
        with self._torch.no_grad():
            enc = self._model.model.encoder(feats, output_hidden_states=True).hidden_states
            # mean over time per layer, then mean over layers -> (hidden,)
            per_layer = self._torch.stack([h.mean(dim=1) for h in enc])  # (L, B, H)
            pooled = per_layer.mean(dim=0).squeeze(0)
        return pooled.cpu().numpy()


def make_whisper(device: str = "cpu") -> Teacher:
    return _WhisperEncoderPooled(device=device)


@dataclass
class _PyworldF0:
    """WORLD F0 contour at 5 ms hop. 1-D scalar sequence per utterance."""

    name: str = "f0"
    target_sr: int = 16000

    def __post_init__(self) -> None:
        import pyworld  # noqa: PLC0415

        self._pw = pyworld

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.target_sr:
            import librosa  # noqa: PLC0415

            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
        f0, _ = self._pw.dio(wav.astype(np.float64), self.target_sr)
        return f0.astype(np.float32)


def make_f0() -> Teacher:
    return _PyworldF0()


# Teachers that need significant porting from TTSDS — each raises until ported.
def _todo(name: str, ttsds_path: str) -> Callable[..., Teacher]:
    def _factory(*_a, **_kw):
        raise NotImplementedError(
            f"Teacher '{name}' not yet ported. See {ttsds_path} for the upstream "
            "implementation. Wrap its per-utterance path into a (wav, sr) callable."
        )

    return _factory


make_wespeaker = _todo("wespeaker", "/tmp/ttsds/src/ttsds/benchmarks/speaker/wespeaker.py")
make_dvector = _todo("dvector", "/tmp/ttsds/src/ttsds/benchmarks/speaker/dvector.py")
make_mpm = _todo("mpm", "/tmp/ttsds/src/ttsds/benchmarks/prosody/mpm.py")
make_allosaurus = _todo("allosaurus", "/tmp/ttsds/src/ttsds/benchmarks/prosody/allosaurus.py")
make_xlsr = _todo("xlsr", "/tmp/ttsds/src/ttsds/benchmarks/generic/wav2vec2_xlsr.py")
make_mhubert = _todo("mhubert", "/tmp/ttsds/src/ttsds/benchmarks/generic/mhubert.py")
make_hubert_token_rate = _todo(
    "hubert_token_rate", "/tmp/ttsds/src/ttsds/benchmarks/prosody/hubert_token.py"
)


FACTOR_TO_TEACHERS: dict[str, list[str]] = {
    "generic": ["hubert", "wavlm", "wav2vec2"],
    "speaker": ["wespeaker", "dvector"],
    "prosody": ["f0", "mpm", "allosaurus", "hubert_token_rate"],
    "intelligibility": ["whisper"],  # add wav2vec2-960h, xlsr after porting
}


_REGISTRY: dict[str, Callable[..., Teacher]] = {
    "hubert": make_hubert,
    "wavlm": make_wavlm,
    "wav2vec2": make_wav2vec2,
    "whisper": make_whisper,
    "f0": make_f0,
    "wespeaker": make_wespeaker,
    "dvector": make_dvector,
    "mpm": make_mpm,
    "allosaurus": make_allosaurus,
    "xlsr": make_xlsr,
    "mhubert": make_mhubert,
    "hubert_token_rate": make_hubert_token_rate,
}


def get_teacher(name: str, device: str = "cpu") -> Teacher:
    factory = _REGISTRY[name]
    try:
        return factory(device=device)
    except TypeError:
        return factory()


def get_factor_teachers(factor: str, device: str = "cpu") -> dict[str, Teacher]:
    teachers: dict[str, Teacher] = {}
    skipped: list[str] = []
    for name in FACTOR_TO_TEACHERS[factor]:
        try:
            teachers[name] = get_teacher(name, device=device)
        except NotImplementedError as e:
            skipped.append(f"{name}: {e}")
    if skipped:
        import warnings  # noqa: PLC0415

        warnings.warn(
            "Skipping teachers awaiting port:\n  " + "\n  ".join(skipped),
            stacklevel=2,
        )
    return teachers


def decode_neucodec(codes: list[int] | np.ndarray, model=None) -> tuple[np.ndarray, int]:
    """Decode NeuCodec FSQ codes back to a 24 kHz mono waveform.

    Loads `neuphonic/neucodec` lazily; pass a pre-loaded `model` to amortise
    initialisation across many calls (recommended in extraction loops).
    """
    import torch  # noqa: PLC0415

    if model is None:
        from neucodec import NeuCodec  # noqa: PLC0415

        model = NeuCodec.from_pretrained("neuphonic/neucodec")
        model.eval()
    fsq = torch.tensor(codes, dtype=torch.long)[None, None, :]
    with torch.no_grad():
        wav = model.decode_code(fsq)
    arr = wav.squeeze().cpu().numpy().astype(np.float32)
    return arr, 24000


__all__ = [
    "FACTOR_TO_TEACHERS",
    "Teacher",
    "decode_neucodec",
    "get_factor_teachers",
    "get_teacher",
    "make_f0",
    "make_hubert",
    "make_wav2vec2",
    "make_wavlm",
    "make_whisper",
]
