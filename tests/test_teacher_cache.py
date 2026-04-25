"""Light tests for teacher_cache. Heavy deps (transformers, neucodec, pyworld)
are not present in the default `dev` env, so we only check API contracts and
the registry layout."""

import pytest

from sdm.data import teacher_cache


def test_factor_to_teachers_covers_four_factors():
    assert set(teacher_cache.FACTOR_TO_TEACHERS) == {
        "generic",
        "speaker",
        "prosody",
        "intelligibility",
    }


def test_unported_teachers_raise_with_pointer():
    with pytest.raises(NotImplementedError) as exc:
        teacher_cache.get_teacher("wespeaker")
    msg = str(exc.value)
    assert "ttsds" in msg.lower()
    assert "wespeaker" in msg.lower()


def test_get_factor_teachers_skips_unported(recwarn):
    # All speaker teachers are still TODO -> we expect a warning and an empty dict.
    teachers = teacher_cache.get_factor_teachers("speaker")
    assert teachers == {}
    assert any("Skipping teachers" in str(w.message) for w in recwarn.list)
