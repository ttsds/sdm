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


def test_ported_teachers_raise_importerror_without_deps():
    # Heavy deps (pyannote, allosaurus, ttsds, etc.) are not in the dev env;
    # constructing the teacher should fail with ImportError, not NotImplementedError.
    with pytest.raises((ImportError, FileNotFoundError, OSError)):
        teacher_cache.get_teacher("wespeaker")


def test_get_factor_teachers_skips_when_deps_missing(recwarn):
    # Speaker teachers require pyannote / ttsds; in the dev env they should be
    # skipped with a warning, leaving an empty dict.
    teachers = teacher_cache.get_factor_teachers("speaker")
    assert teachers == {}
    assert any("Skipping teachers" in str(w.message) for w in recwarn.list)
