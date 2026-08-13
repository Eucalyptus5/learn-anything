import string

from tests.fakes import FakeSynthesizer
from tutor.openers import OPENER_PHRASES, synthesize_openers

BANNED_WORDS = {"i", "me", "my"}


def test_key_set_is_exactly_four_openers() -> None:
    assert set(OPENER_PHRASES) == {"thinking", "file_hit", "many_files", "empty"}


def test_synthesize_openers_calls_synth_once_per_key() -> None:
    synth = FakeSynthesizer()

    result = synthesize_openers(synth)

    assert synth.calls == list(OPENER_PHRASES.values())
    assert set(result) == set(OPENER_PHRASES)


def test_phrases_are_short_ascii_and_pronoun_free() -> None:
    for phrase in OPENER_PHRASES.values():
        assert phrase.isascii()
        words = phrase.split()
        assert 1 <= len(words) <= 4
        for word in words:
            stripped = word.strip(string.punctuation).lower()
            assert stripped not in BANNED_WORDS
