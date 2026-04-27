from __future__ import annotations

import torch

from sdm.data.teachers import build_teacher
from sdm.data.teachers.g2p_speaking_rate import (
    G2pSpeakingRateConfig,
    G2pSpeakingRateTeacher,
)


def _fixed_counter(text: str, language: str | None) -> int:
    """Deterministic stand-in for phonemizer in tests."""
    return len(text.split())


def test_speaking_rate_broadcasts_utterance_scalar_to_chunks():
    cfg = G2pSpeakingRateConfig(chunk_seconds=1.0)
    teacher = G2pSpeakingRateTeacher(cfg, counter=_fixed_counter)

    audio = torch.zeros(2, 4, 16000)
    chunk_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    texts = ["a b c d e f", "x y"]
    languages = ["en", "en"]

    out = teacher(audio, chunk_mask=chunk_mask, texts=texts, languages=languages)

    assert out.shape == (2, 4, 1)
    # 6 phonemes / 3 valid chunks * 1.0s = 2.0 phonemes/sec
    assert torch.allclose(out[0, :3, 0], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.all(out[0, 3] == 0)  # padded chunk masked
    # 2 phonemes / 2 valid chunks * 1.0s = 1.0 phonemes/sec
    assert torch.allclose(out[1, :2, 0], torch.tensor([1.0, 1.0]))
    assert torch.all(out[1, 2:] == 0)


def test_speaking_rate_handles_missing_text():
    teacher = G2pSpeakingRateTeacher(G2pSpeakingRateConfig(), counter=_fixed_counter)
    audio = torch.zeros(1, 2, 16000)
    out = teacher(
        audio,
        chunk_mask=torch.ones(1, 2, dtype=torch.bool),
        texts=[None],
        languages=[None],
    )
    assert out.shape == (1, 2, 1)
    assert torch.all(out == 0)


def test_speaking_rate_uses_default_language_when_missing():
    seen_langs: list[str | None] = []

    def _counter(text: str, language: str | None) -> int:
        seen_langs.append(language)
        return 1

    teacher = G2pSpeakingRateTeacher(
        G2pSpeakingRateConfig(default_language="fr"),
        counter=_counter,
    )
    teacher(
        torch.zeros(1, 1, 8),
        chunk_mask=torch.ones(1, 1, dtype=torch.bool),
        texts=["hello"],
        languages=[None],
    )
    assert seen_langs == ["fr"]


def test_build_teacher_dispatches_speaking_rate():
    class _Cfg:
        kind = "g2p_speaking_rate"
        target_dim = 1
        pooled = "chunked"
        chunk_seconds = 1.0
        default_language = "en"
        target_layernorm = False

    teacher = build_teacher(_Cfg())
    assert isinstance(teacher, G2pSpeakingRateTeacher)
