from __future__ import annotations

import torch

from sdm.data.teachers import build_teacher
from sdm.data.teachers.allosaurus_speaking_rate import (
    AllosaurusSpeakingRateConfig,
    AllosaurusSpeakingRateTeacher,
    _bucket_phones_into_chunks,
)


def test_bucket_phones_collapses_diphthongs_within_chunk():
    # Two adjacent vowels in chunk 0 -> 1 vowel group; consonant; vowel
    # in chunk 1 -> 1 group; chunk 2 has no phones -> 0.
    phones = [(0.10, "a"), (0.20, "ɪ"), (0.40, "k"), (1.30, "u")]
    counts = _bucket_phones_into_chunks(phones, chunk_seconds=1.0, n_chunks=3)
    assert counts == [1, 1, 0]


def test_bucket_phones_consonant_resets_in_vowel_state():
    # vowel, consonant, vowel within the same chunk -> 2 groups.
    phones = [(0.10, "a"), (0.20, "k"), (0.30, "i")]
    counts = _bucket_phones_into_chunks(phones, chunk_seconds=1.0, n_chunks=1)
    assert counts == [2]


def test_bucket_phones_chunk_boundary_resets_state():
    # Two adjacent vowels straddling a chunk boundary should count once
    # in each chunk (boundary resets the in-vowel state).
    phones = [(0.95, "a"), (1.05, "ɪ")]
    counts = _bucket_phones_into_chunks(phones, chunk_seconds=1.0, n_chunks=2)
    assert counts == [1, 1]


def test_allosaurus_teacher_uses_injected_counter_per_chunk():
    cfg = AllosaurusSpeakingRateConfig(chunk_seconds=1.0, sample_rate=16000)

    seen: list[tuple[int, int, str | None]] = []

    def _counter(audio, sr, chunk_secs, n_valid, *, language=None):
        seen.append((int(audio.numel()), int(n_valid), language))
        # 1 vowel group in chunk 0, 3 in chunk 1, 0 elsewhere.
        return [1, 3, 0][:n_valid]

    teacher = AllosaurusSpeakingRateTeacher(cfg, counter=_counter)

    audio = torch.zeros(2, 4, 16000)
    chunk_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    out = teacher(audio, chunk_mask=chunk_mask, languages=["en", None])

    assert out.shape == (2, 4, 1)
    # First utterance: 3 valid chunks, counts [1, 3, 0] / 1.0 sec.
    assert torch.allclose(out[0, :3, 0], torch.tensor([1.0, 3.0, 0.0]))
    assert out[0, 3, 0] == 0.0
    # Second utterance: 2 valid chunks, counts [1, 3].
    assert torch.allclose(out[1, :2, 0], torch.tensor([1.0, 3.0]))
    assert torch.all(out[1, 2:] == 0)
    # Counter was called once per utterance with concatenated valid audio.
    assert len(seen) == 2
    assert seen[0] == (3 * 16000, 3, "en")
    # Default language fills in when None is passed.
    assert seen[1][:2] == (2 * 16000, 2)
    assert seen[1][2] == cfg.default_language


def test_allosaurus_teacher_applies_target_normalisation():
    cfg = AllosaurusSpeakingRateConfig(target_mean=4.0, target_scale=1.5)

    def _counter(audio, sr, chunk_secs, n_valid, *, language=None):
        return [4] * n_valid  # raw rate = 4 syl/sec

    teacher = AllosaurusSpeakingRateTeacher(cfg, counter=_counter)
    out = teacher(
        torch.zeros(1, 2, 16000),
        chunk_mask=torch.ones(1, 2, dtype=torch.bool),
        languages=["en"],
    )
    # (4 - 4) / 1.5 = 0
    assert torch.allclose(out, torch.zeros_like(out))


def test_allosaurus_teacher_skips_zero_valid_chunks():
    def _counter(*args, **kwargs):  # pragma: no cover - should never be hit
        raise AssertionError("counter must not run when no chunks are valid")

    teacher = AllosaurusSpeakingRateTeacher(
        AllosaurusSpeakingRateConfig(), counter=_counter
    )
    out = teacher(
        torch.zeros(1, 2, 16000),
        chunk_mask=torch.zeros(1, 2, dtype=torch.bool),
        languages=["en"],
    )
    assert torch.all(out == 0)


def test_build_teacher_dispatches_allosaurus_speaking_rate():
    class _Cfg:
        kind = "allosaurus_speaking_rate"
        target_dim = 1
        pooled = "chunked"
        chunk_seconds = 1.0
        sample_rate = 16000
        default_language = "en"
        target_layernorm = False

    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, AllosaurusSpeakingRateTeacher)
