"""Per-chunk speaking-rate teacher using Allosaurus phone recognition.

Unlike :mod:`sdm.data.teachers.g2p_speaking_rate`, which derives a single
utterance-level rate from the transcript and broadcasts it across all
chunks, this teacher runs Allosaurus on the *audio* and buckets the
resulting timestamped IPA phones into per-chunk vowel-group counts. Each
chunk gets its own ``vowel_groups / chunk_seconds`` scalar, so the student
must learn local prosody rather than a constant per-utterance number.

Allosaurus emits one phone per line as ``"<start_sec> <duration_sec> <ipa>"``.
We assign each phone to chunk ``int(start_sec // chunk_seconds)`` and
collapse adjacent vowels within a chunk into a single syllable nucleus
(matching the vowel-group definition used by the G2P teacher).

Performance: Allosaurus on CPU runs at RTF ~0.02 (~100 ms per 5 s
utterance, single thread). With the dataloader's worker processes this
is comparable in cost to the espeak-based G2P path.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from sdm.data.teachers.g2p_speaking_rate import _IPA_VOWELS


# ``ChunkVowelCounter`` returns one count per chunk index. It receives the
# already-trimmed mono audio for a single utterance plus the chunk grid,
# and is the dependency-injection seam that keeps tests Allosaurus-free.
ChunkVowelCounter = Callable[[torch.Tensor, int, float, int], list[int]]


# Map Emilia language codes to Allosaurus language ids. ``ipa`` is the
# universal model; language-specific ids constrain the inventory and tend
# to give cleaner counts but only for languages Allosaurus ships.
_LANG_TO_ALLOSAURUS = {
    "en": "eng",
    "en-us": "eng",
    "en-gb": "eng",
    "zh": "cmn",
    "zh-cn": "cmn",
    "cmn": "cmn",
    "ja": "jpn",
    "ko": "kor",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "nl": "nld",
}


@dataclass
class AllosaurusSpeakingRateConfig:
    target_dim: int = 1
    pooled: str = "chunked"
    chunk_seconds: float = 1.0
    sample_rate: int = 16000
    default_language: str = "en"
    target_layernorm: bool = False  # scalar target — LN would zero it
    # Linear target normalisation: out -> (out - target_mean) / target_scale.
    target_mean: float = 0.0
    target_scale: float = 1.0
    # Allosaurus model name (passed to ``read_recognizer``); ``"latest"`` is
    # the multi-lingual default. Override for ablations only.
    allosaurus_model: str = "latest"
    # ``emit`` controls the phone-emission threshold; values <1.0 emit more
    # phones (helpful for fast speech). Allosaurus default is 1.0.
    emit: float = 1.0


def _coerce_config(cfg: Any) -> AllosaurusSpeakingRateConfig:
    if isinstance(cfg, AllosaurusSpeakingRateConfig):
        return cfg
    fields = (
        "target_dim",
        "pooled",
        "chunk_seconds",
        "sample_rate",
        "default_language",
        "target_layernorm",
        "target_mean",
        "target_scale",
        "allosaurus_model",
        "emit",
    )
    if hasattr(cfg, "kind"):
        return AllosaurusSpeakingRateConfig(
            **{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None}
        )
    return AllosaurusSpeakingRateConfig(**{f: cfg[f] for f in fields if f in cfg})


def _bucket_phones_into_chunks(
    phones: Sequence[tuple[float, str]],
    chunk_seconds: float,
    n_chunks: int,
) -> list[int]:
    """Assign timestamped IPA phones to chunks and count vowel groups.

    ``phones`` is an iterable of ``(start_sec, ipa)`` pairs sorted by start
    time. Within each chunk we collapse adjacent vowels into a single
    nucleus -- diphthongs like ``aɪ`` count as one. Chunk boundaries reset
    the in-vowel state, so a phone that straddles a boundary still counts
    only on its assigned chunk.
    """
    counts = [0] * n_chunks
    in_vowel_per_chunk = [False] * n_chunks
    for start, ipa in phones:
        idx = int(start // chunk_seconds)
        if idx < 0 or idx >= n_chunks:
            continue
        is_vowel = any(ch in _IPA_VOWELS for ch in ipa)
        if is_vowel:
            if not in_vowel_per_chunk[idx]:
                counts[idx] += 1
                in_vowel_per_chunk[idx] = True
        else:
            in_vowel_per_chunk[idx] = False
    return counts


class _AllosaurusCounter:
    """Wraps :func:`allosaurus.app.read_recognizer` with per-chunk bucketing.

    The model is loaded lazily on first call so dataloader workers don't
    pay the import cost until they actually need it. Each utterance is
    written to a temporary WAV file (Allosaurus only accepts file paths
    in 1.0.x); we use ``tmpfs`` (``/dev/shm``) when available to avoid
    disk IO inside the training loop.
    """

    def __init__(self, model_name: str = "latest", emit: float = 1.0) -> None:
        self._model_name = model_name
        self._emit = float(emit)
        self._recognizer: Any = None
        self._import_error: Exception | None = None
        self._import_traceback: str | None = None
        self._tmpdir: str | None = None
        self._fallback_warned = False

    def _ensure(self) -> None:
        if self._recognizer is not None or self._import_error is not None:
            return
        try:
            from allosaurus.app import read_recognizer  # type: ignore

            self._recognizer = read_recognizer(self._model_name)
        except Exception as exc:  # pragma: no cover - environment-dependent
            import traceback

            self._import_error = exc
            self._import_traceback = traceback.format_exc()

    def _temp_path(self) -> str:
        if self._tmpdir is None:
            preferred = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
            self._tmpdir = tempfile.mkdtemp(prefix="sdm-allosaurus-", dir=preferred)
        return tempfile.mktemp(suffix=".wav", dir=self._tmpdir)

    def __call__(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        chunk_seconds: float,
        n_chunks: int,
        *,
        language: str | None = None,
    ) -> list[int]:
        self._ensure()
        if self._recognizer is None:
            if not self._fallback_warned:
                import warnings

                warnings.warn(
                    "allosaurus is unavailable "
                    f"({self._import_error!r}); per-chunk speaking-rate "
                    "targets will be all-zero. Install the `teachers` extra "
                    "to enable accurate audio-based counts.\n"
                    f"Traceback:\n{self._import_traceback or '(none)'}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback_warned = True
            return [0] * n_chunks
        if audio.numel() == 0:
            return [0] * n_chunks
        # Allosaurus reads via soundfile; write a 16-bit PCM WAV.
        try:
            import soundfile as sf  # type: ignore
        except Exception as exc:  # pragma: no cover - covered by `teachers` extra
            raise RuntimeError(
                "soundfile is required for the allosaurus speaking-rate teacher"
            ) from exc
        path = self._temp_path()
        try:
            wav = audio.detach().to(torch.float32).cpu().numpy()
            sf.write(path, wav, sample_rate, subtype="PCM_16")
            lang_id = _LANG_TO_ALLOSAURUS.get((language or "eng").lower(), "ipa")
            try:
                raw = self._recognizer.recognize(
                    path, lang_id=lang_id, timestamp=True, emit=self._emit
                )
            except Exception:
                return [0] * n_chunks
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        phones: list[tuple[float, str]] = []
        for line in (raw or "").splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
            except ValueError:
                continue
            ipa = parts[2]
            phones.append((start, ipa))
        return _bucket_phones_into_chunks(phones, chunk_seconds, n_chunks)


class AllosaurusSpeakingRateTeacher(nn.Module):
    """Per-chunk speaking-rate (vowel groups / second) from audio.

    Output shape ``(B, N_chunks, 1)``. Each chunk gets its own scalar
    rate computed by running Allosaurus over the utterance's valid audio
    and bucketing IPA phones by their start timestamp. Padded chunks are
    masked to zero. ``target_mean`` / ``target_scale`` apply the same
    linear renormalisation as the G2P teacher so the L1 loss starts at
    a sensible scale.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        counter: ChunkVowelCounter | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        if self.target_dim != 1:
            raise ValueError(
                f"allosaurus_speaking_rate teacher emits a 1-D scalar; got target_dim={self.target_dim}"
            )
        if counter is None:
            counter = _AllosaurusCounter(
                model_name=self.cfg.allosaurus_model, emit=self.cfg.emit
            )
            # Eagerly load the recognizer in the main process so the model
            # files / network downloads are resolved once, before any
            # dataloader workers fork. Otherwise each worker would try to
            # initialise allosaurus concurrently and may race on the
            # bundled pretrained dir (FileNotFoundError under load).
            if isinstance(counter, _AllosaurusCounter):
                counter._ensure()
        self._counter: ChunkVowelCounter = counter
        self._device = torch.device(device) if not isinstance(device, torch.device) else device

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        languages: list[str | None] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if audio.dim() != 3:
            raise ValueError(f"expected (B, N_chunks, T), got {tuple(audio.shape)}")
        b, n, t = audio.shape
        if chunk_mask is None:
            chunk_mask = torch.ones((b, n), dtype=torch.bool, device=audio.device)

        chunk_secs = float(self.cfg.chunk_seconds)
        sr = int(self.cfg.sample_rate)
        per_chunk_counts = torch.zeros((b, n), dtype=torch.float32)

        valid_per_utt = chunk_mask.to(torch.long).sum(dim=1).tolist()
        cpu_audio = audio.detach().to(torch.float32).cpu()
        for i in range(b):
            n_valid = int(valid_per_utt[i])
            if n_valid <= 0:
                continue
            # Concatenate valid chunks into a single waveform. We rely on
            # the dataloader convention that valid chunks are contiguous
            # at the front and padded chunks come after.
            wav = cpu_audio[i, :n_valid].reshape(-1)
            lang = (
                languages[i]
                if languages is not None and i < len(languages) and languages[i]
                else self.cfg.default_language
            )
            counts = self._counter(wav, sr, chunk_secs, n_valid, language=lang)
            if len(counts) != n_valid:
                # Defensive: pad/truncate to the expected chunk count.
                counts = list(counts[:n_valid]) + [0] * max(0, n_valid - len(counts))
            for j, c in enumerate(counts):
                per_chunk_counts[i, j] = float(c)

        rates = (per_chunk_counts / max(chunk_secs, 1e-6)).to(audio.device, dtype=torch.float32)
        scale = float(self.cfg.target_scale)
        if scale != 1.0 or float(self.cfg.target_mean) != 0.0:
            if scale == 0.0:
                raise ValueError("target_scale must be non-zero")
            rates = (rates - float(self.cfg.target_mean)) / scale
        out = rates.unsqueeze(-1)
        out = out * chunk_mask.unsqueeze(-1).to(out.dtype)
        return out

    def __call__(  # type: ignore[override]
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        **ctx: Any,
    ) -> torch.Tensor:
        return self.forward(audio, chunk_mask=chunk_mask, **ctx)


__all__ = [
    "AllosaurusSpeakingRateConfig",
    "AllosaurusSpeakingRateTeacher",
    "ChunkVowelCounter",
    "_bucket_phones_into_chunks",
]
