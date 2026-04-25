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


# Teachers ported from /tmp/ttsds/src/ttsds/benchmarks (MIT-licensed). Each
# wraps the per-utterance computation of the corresponding `_get_distribution`
# loop body.


@dataclass
class _Wav2Vec2XLSR(_SSLTeacher):
    """Wav2Vec2-XLSR-53 hidden states, default layer 8 (TTSDS)."""


def make_xlsr(device: str = "cpu") -> Teacher:
    return _Wav2Vec2XLSR(
        name="xlsr",
        model_id="facebook/wav2vec2-large-xlsr-53",
        processor_id="facebook/wav2vec2-base-960h",
        layer=8,
        device=device,
    )


def make_mhubert(device: str = "cpu") -> Teacher:
    return _SSLTeacher(
        name="mhubert",
        model_id="utter-project/mHuBERT-147",
        processor_id="facebook/hubert-large-ls960-ft",
        layer=7,
        device=device,
    )


@dataclass
class _Wav2Vec2Pooled:
    """wav2vec2-base-960h last-hidden-state, mean-pooled across time -> (768,)."""

    name: str = "wav2vec2_960h"
    model_id: str = "facebook/wav2vec2-base-960h"
    device: str = "cpu"

    def __post_init__(self) -> None:
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from transformers import Wav2Vec2Model, Wav2Vec2Processor  # noqa: PLC0415

        self._torch = torch
        self._librosa = librosa
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self._model = Wav2Vec2Model.from_pretrained(self.model_id).to(self.device).eval()

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            wav = self._librosa.resample(wav, orig_sr=sr, target_sr=16000)
        inputs = self._processor(wav, return_tensors="pt", sampling_rate=16000)
        iv = inputs["input_values"].to(self.device)
        with self._torch.no_grad():
            out = self._model(iv).last_hidden_state.mean(dim=1).squeeze(0)
        return out.cpu().numpy()


def make_wav2vec2_960h(device: str = "cpu") -> Teacher:
    return _Wav2Vec2Pooled(device=device)


@dataclass
class _WeSpeaker:
    """Sliding-window pyannote/wespeaker embeddings -> (n_windows, 256)."""

    name: str = "wespeaker"
    window_duration: float = 1.0
    window_step: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        import librosa  # noqa: PLC0415
        from pyannote.audio import Inference, Model  # noqa: PLC0415

        self._librosa = librosa
        self._model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
        self._inference = Inference(
            self._model,
            window="sliding",
            duration=self.window_duration,
            step=self.window_step,
        )

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import soundfile as sf  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        if sr != 16000:
            wav = self._librosa.resample(wav, orig_sr=sr, target_sr=16000)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav, 16000)
            sliding = self._inference(f.name)
        return np.stack([np.asarray(x[1]) for x in sliding]).astype(np.float32)


def make_wespeaker(device: str = "cpu") -> Teacher:
    return _WeSpeaker(device=device)


@dataclass
class _DVector:
    """Sliding-window DVector embeddings -> (n_windows, 256). Uses the
    `dvector.pt` shipped inside the `ttsds` package (MIT)."""

    name: str = "dvector"
    window_duration: float = 1.0
    window_step: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        import importlib.resources  # noqa: PLC0415

        import torch  # noqa: PLC0415

        # Reuse upstream's Wav2Mel for byte-exact preprocessing.
        from ttsds.benchmarks.speaker.dvector import Wav2Mel  # noqa: PLC0415

        self._torch = torch
        self._wav2mel = Wav2Mel()
        with importlib.resources.path("ttsds", "dvector") as dp:
            self._dvector = torch.jit.load(str(dp / "dvector.pt")).to(self.device).eval()

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        win = int(self.window_duration * sr)
        hop = int(self.window_step * sr)
        embs: list = []
        for i in range(0, max(1, len(wav)), hop):
            chunk = wav[i : i + win]
            if len(chunk) <= win // 2:
                continue
            wav_t = self._torch.tensor(chunk).float().unsqueeze(0)
            mel = self._wav2mel(wav_t, sr)
            with self._torch.no_grad():
                emb = self._dvector.embed_utterance(mel)
            embs.append(emb.detach().cpu().numpy())
        if not embs:
            # Fallback: single embedding from whole utterance.
            wav_t = self._torch.tensor(wav).float().unsqueeze(0)
            mel = self._wav2mel(wav_t, sr)
            with self._torch.no_grad():
                emb = self._dvector.embed_utterance(mel)
            embs = [emb.detach().cpu().numpy()]
        return np.stack(embs).astype(np.float32)


def make_dvector(device: str = "cpu") -> Teacher:
    return _DVector(device=device)


@dataclass
class _MPM:
    """Masked Prosody Model layer-7 hidden states. Same preprocessing as TTSDS."""

    name: str = "mpm"
    model_id: str = "cdminix/masked_prosody_model"
    layer: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from ttsds.util.measures import (  # noqa: PLC0415
            EnergyMeasure,
            PitchMeasure,
            VoiceActivityMeasure,
        )
        from ttsds.util.mpm import MaskedProsodyModel  # noqa: PLC0415

        self._torch = torch
        self._librosa = librosa
        self._model = MaskedProsodyModel.from_pretrained(self.model_id).to(self.device).eval()
        self._pitch = PitchMeasure()
        self._energy = EnergyMeasure()
        self._vad = VoiceActivityMeasure()
        self._bins = torch.linspace(0, 1, 128)
        self._vad_bins = torch.linspace(0, 1, 2)

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        torch = self._torch
        if sr != 22050:
            wav = self._librosa.resample(wav, orig_sr=sr, target_sr=22050)
        pitch = torch.tensor(self._pitch(wav, np.array([1000]))["measure"])
        energy = torch.tensor(self._energy(wav, np.array([1000]))["measure"])
        vad = torch.tensor(self._vad(wav, np.array([1000]))["measure"])
        for x in (pitch, energy, vad):
            x[torch.isnan(x)] = -1000
        pitch = torch.clip(pitch, 50, 300) / (300 - 50)
        energy = torch.clip(energy, 0, 0.2) / 0.2
        vad = torch.clip(vad, 0, 1)
        pitch = torch.bucketize(pitch, self._bins)
        energy = torch.bucketize(energy, self._bins)
        vad = torch.bucketize(vad, self._vad_bins)
        n = min(len(pitch), len(energy), len(vad))
        x = torch.stack([pitch[:n], energy[:n], vad[:n]]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            reprs = self._model(x, return_layer=self.layer)["representations"]
        return reprs.squeeze(0).cpu().numpy().astype(np.float32)


def make_mpm(device: str = "cpu") -> Teacher:
    return _MPM(device=device)


@dataclass
class _AllosaurusSR:
    """Speaking rate from Allosaurus phone count / duration. Returns (1,)."""

    name: str = "allosaurus"

    def __post_init__(self) -> None:
        from allosaurus.app import read_recognizer  # noqa: PLC0415

        self._model = read_recognizer()

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import soundfile as sf  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav, sr)
            result = self._model.recognize(f.name, timestamp=True)
        if not result.strip():
            rate = 0.0
        else:
            rate = len(result.strip().split("\n")) / (len(wav) / sr)
        return np.array([rate], dtype=np.float32)


def make_allosaurus(device: str = "cpu") -> Teacher:  # noqa: ARG001
    return _AllosaurusSR()


@dataclass
class _HubertTokenRate:
    """Speaking rate from HuBERT-layer-7 KMeans run-length compression.

    Requires a precomputed cluster-centres array. Drop one at
    `<cache>/hubert_token_kmeans.npy` (shape `(K, 768)`) before extraction;
    use TTSDS' own `HubertTokenSRBenchmark.create_clusters` to bootstrap one
    from a held-out subset of the data.
    """

    name: str = "hubert_token_rate"
    cluster_centers_path: str = "hubert_token_kmeans.npy"
    hubert_model: str = "facebook/hubert-base-ls960"
    layer: int = 7
    device: str = "cpu"

    def __post_init__(self) -> None:
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from sklearn.cluster import KMeans  # noqa: PLC0415
        from transformers import HubertModel, Wav2Vec2Processor  # noqa: PLC0415

        self._torch = torch
        self._librosa = librosa
        self._processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self._model = HubertModel.from_pretrained(self.hubert_model).to(self.device).eval()
        centers = np.load(self.cluster_centers_path)
        self._kmeans = KMeans(n_clusters=centers.shape[0], n_init=1)
        self._kmeans.fit(np.zeros((centers.shape[0] + 1, centers.shape[1])))
        self._kmeans.cluster_centers_ = centers

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            wav = self._librosa.resample(wav, orig_sr=sr, target_sr=16000)
        iv = self._processor(wav, return_tensors="pt", sampling_rate=16000).input_values
        iv = iv.to(self.device)
        with self._torch.no_grad():
            feats = self._model(iv, output_hidden_states=True).hidden_states[self.layer]
        feats = feats.squeeze(0).cpu().numpy()
        clusters = self._kmeans.predict(feats)
        # Run-length compression: count distinct token runs.
        runs = 1 + int(np.sum(clusters[1:] != clusters[:-1]))
        rate = runs / (len(wav) / 16000)
        return np.array([rate], dtype=np.float32)


def make_hubert_token_rate(
    device: str = "cpu",
    cluster_centers_path: str = "hubert_token_kmeans.npy",
) -> Teacher:
    return _HubertTokenRate(cluster_centers_path=cluster_centers_path, device=device)


FACTOR_TO_TEACHERS: dict[str, list[str]] = {
    "generic": ["hubert", "wavlm", "wav2vec2", "xlsr", "mhubert"],
    "speaker": ["wespeaker", "dvector"],
    "prosody": ["f0", "mpm", "allosaurus", "hubert_token_rate"],
    "intelligibility": ["whisper", "wav2vec2_960h"],
}


_REGISTRY: dict[str, Callable[..., Teacher]] = {
    "hubert": make_hubert,
    "wavlm": make_wavlm,
    "wav2vec2": make_wav2vec2,
    "wav2vec2_960h": make_wav2vec2_960h,
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


def get_teacher(name: str, device: str = "cpu", **kwargs) -> Teacher:
    factory = _REGISTRY[name]
    try:
        return factory(device=device, **kwargs)
    except TypeError:
        return factory(**kwargs) if kwargs else factory()


def get_factor_teachers(factor: str, device: str = "cpu") -> dict[str, Teacher]:
    teachers: dict[str, Teacher] = {}
    skipped: list[str] = []
    for name in FACTOR_TO_TEACHERS[factor]:
        try:
            teachers[name] = get_teacher(name, device=device)
        except (NotImplementedError, FileNotFoundError, ImportError) as e:
            skipped.append(f"{name}: {type(e).__name__}: {e}")
    if skipped:
        import warnings  # noqa: PLC0415

        warnings.warn(
            "Skipping teachers (missing weights or unported deps):\n  "
            + "\n  ".join(skipped),
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
    "make_allosaurus",
    "make_dvector",
    "make_f0",
    "make_hubert",
    "make_hubert_token_rate",
    "make_mhubert",
    "make_mpm",
    "make_wav2vec2",
    "make_wav2vec2_960h",
    "make_wavlm",
    "make_wespeaker",
    "make_whisper",
    "make_xlsr",
]
