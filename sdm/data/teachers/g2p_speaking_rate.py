"""Speaking-rate teacher derived from the dataset transcripts.

The TTSDS2 paper uses Allosaurus to estimate phones-per-second on each
utterance. Allosaurus has no TPU implementation and would force an offline
pre-extraction step. Since Emilia ships per-utterance transcripts, we
reproduce the same target via grapheme-to-phoneme on the transcript:

    speaking_rate = #phonemes(text) / utterance_duration_seconds

The scalar is broadcast to every valid chunk so the per-chunk loss masks
behave the same as the other prosody teachers. ``chunk_mask`` zeros out the
padded chunks. Default backend is :mod:`phonemizer` (espeak-ng); a custom
``counter`` callable can be injected for tests / alternative G2Ps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


# Map Emilia language codes to espeak-ng voices. Anything outside this set
# falls back to ``en-us`` which still produces a roughly-monotonic phoneme
# count (the alternative is dropping the example, which would silently bias
# training toward the listed languages).
_LANG_TO_ESPEAK = {
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "zh": "cmn",
    "zh-cn": "cmn",
    "cmn": "cmn",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr-fr",
    "de": "de",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "nl": "nl",
}

PhonemeCounter = Callable[[str, str | None], int]


@dataclass
class G2pSpeakingRateConfig:
    target_dim: int = 1
    pooled: str = "chunked"
    chunk_seconds: float = 1.0
    sample_rate: int = 16000
    default_language: str = "en"
    target_layernorm: bool = False  # scalar target — LN would zero it


def _coerce_config(cfg: Any) -> G2pSpeakingRateConfig:
    if isinstance(cfg, G2pSpeakingRateConfig):
        return cfg
    fields = (
        "target_dim",
        "pooled",
        "chunk_seconds",
        "sample_rate",
        "default_language",
        "target_layernorm",
    )
    if hasattr(cfg, "kind"):
        return G2pSpeakingRateConfig(**{f: getattr(cfg, f) for f in fields if hasattr(cfg, f) and getattr(cfg, f) is not None})
    return G2pSpeakingRateConfig(**{f: cfg[f] for f in fields if f in cfg})


class _PhonemizerCounter:
    """Lazy wrapper around :mod:`phonemizer` with one backend per language.

    Counts phonemes by splitting on whitespace after running the espeak
    backend with a per-phone separator. Falls back to a simple character-
    based heuristic if phonemizer / espeak-ng is unavailable; this keeps
    the dataloader from crashing in environments where the backend is not
    installed (tests / smoke). The fallback is documented in the warning
    that fires the first time it is used.
    """

    def __init__(self) -> None:
        self._backends: dict[str, Any] = {}
        self._fallback_warned = False
        self._import_error: Exception | None = None
        try:
            from phonemizer.backend import EspeakBackend  # type: ignore
            from phonemizer.separator import Separator  # type: ignore

            self._EspeakBackend = EspeakBackend
            self._Separator = Separator(phone=" ", word=" | ")
        except Exception as exc:  # pragma: no cover - exercised in fallback path
            self._EspeakBackend = None
            self._Separator = None
            self._import_error = exc

    def _backend_for(self, language: str | None):
        if self._EspeakBackend is None:
            return None
        voice = _LANG_TO_ESPEAK.get((language or "en").lower(), "en-us")
        backend = self._backends.get(voice)
        if backend is None:
            try:
                # phonemizer logs "words count mismatch" warnings whenever
                # espeak-ng's output token count differs from the input
                # (contractions, numbers, punctuation). We only count phones,
                # so word alignment is irrelevant — silence the spam.
                import logging

                quiet = logging.getLogger("phonemizer")
                quiet.setLevel(logging.ERROR)
                backend = self._EspeakBackend(
                    voice,
                    preserve_punctuation=False,
                    with_stress=False,
                    language_switch="remove-flags",
                    logger=quiet,
                )
            except Exception as exc:  # pragma: no cover - environment-dependent
                self._import_error = exc
                return None
            self._backends[voice] = backend
        return backend

    def __call__(self, text: str, language: str | None) -> int:
        if not text or not text.strip():
            return 0
        backend = self._backend_for(language)
        if backend is None:
            if not self._fallback_warned:
                import warnings

                warnings.warn(
                    "phonemizer/espeak-ng is unavailable "
                    f"({self._import_error!r}); falling back to a whitespace+"
                    "character heuristic for speaking-rate targets. Install "
                    "the `teachers` extra and the espeak-ng system package "
                    "for accurate phoneme counts.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback_warned = True
            # Heuristic: ~1 phoneme per non-space character. Crude but bounded.
            return sum(1 for ch in text if not ch.isspace())
        try:
            phonemized = backend.phonemize(
                [text],
                separator=self._Separator,
                strip=True,
                njobs=1,
            )
        except Exception:
            return 0
        if not phonemized:
            return 0
        joined = phonemized[0]
        # ``Separator(phone=' ', word=' | ')`` produces e.g. ``"h ə l ˈoʊ | w ɝː l d"``.
        # Strip word boundary markers and split on whitespace.
        return sum(1 for tok in joined.replace("|", " ").split() if tok)


class G2pSpeakingRateTeacher(nn.Module):
    """Per-utterance speaking rate (phonemes / second) broadcast to chunks.

    Output shape: ``(B, N_chunks, 1)`` — the scalar speaking rate is
    repeated across every valid chunk and zeroed on padded chunks. The
    target is computed on CPU from the batch's transcripts and language
    codes (no audio model is loaded).
    """

    def __init__(
        self,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        counter: PhonemeCounter | None = None,
    ) -> None:
        super().__init__()
        self.cfg = _coerce_config(cfg)
        self.target_dim = int(self.cfg.target_dim)
        if self.target_dim != 1:
            raise ValueError(
                f"g2p_speaking_rate teacher emits a 1-D scalar; got target_dim={self.target_dim}"
            )
        self._counter: PhonemeCounter = counter or _PhonemizerCounter()
        self._device = torch.device(device) if not isinstance(device, torch.device) else device

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        *,
        chunk_mask: torch.Tensor | None = None,
        texts: list[str | None] | None = None,
        languages: list[str | None] | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if audio.dim() != 3:
            raise ValueError(f"expected (B, N_chunks, T), got {tuple(audio.shape)}")
        b, n, _ = audio.shape
        if chunk_mask is None:
            chunk_mask = torch.ones((b, n), dtype=torch.bool, device=audio.device)
        # Per-utterance duration uses the chunk grid (one valid chunk =
        # ``chunk_seconds``). Padded chunks are excluded so short clips
        # produce a meaningful rate rather than getting diluted by zero-pad.
        chunk_secs = float(self.cfg.chunk_seconds)
        valid_per_utt = chunk_mask.to(torch.float32).sum(dim=1).clamp(min=1.0)
        durations = (valid_per_utt * chunk_secs).cpu().tolist()

        rates = torch.zeros((b,), dtype=torch.float32)
        if texts is not None:
            for i, text in enumerate(texts):
                if not text:
                    continue
                lang = (
                    languages[i]
                    if languages is not None and i < len(languages) and languages[i]
                    else self.cfg.default_language
                )
                count = int(self._counter(text, lang))
                rates[i] = count / max(durations[i], 1e-6)

        out = rates.view(b, 1, 1).expand(b, n, 1).contiguous().to(audio.device, dtype=torch.float32)
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


__all__ = ["G2pSpeakingRateConfig", "G2pSpeakingRateTeacher", "PhonemeCounter"]
