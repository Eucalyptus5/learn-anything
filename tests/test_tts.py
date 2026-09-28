from pathlib import Path

import numpy as np
import pytest

from tutor import tts
from tutor.constants import TTS_SAMPLE_RATE
from tutor.tts import KokoroSynthesizer, to_int16

WEIGHTS = Path("models/kokoro/kokoro-v1.0.fp16.onnx")
VOICES = Path("models/kokoro/voices-v1.0.bin")


class FakeKokoro:
    def __init__(self, model_path: str, voices_path: str) -> None:
        self.model_path = model_path
        self.voices_path = voices_path
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.styles: list[str] = []
        self._audio = np.array([-1.0, 0.0, 1.0, 0.5], dtype=np.float32)
        self._rate = TTS_SAMPLE_RATE

    def get_voice_style(self, name: str) -> np.ndarray:
        self.styles.append(name)
        return np.repeat(np.arange(510, dtype=np.float32).reshape(510, 1, 1), 256, axis=2)

    def create(self, text: str, **kwargs: object):
        self.calls.append((text, kwargs))
        return self._audio, self._rate


class BadRateKokoro(FakeKokoro):
    def __init__(self, model_path: str, voices_path: str) -> None:
        super().__init__(model_path, voices_path)
        self._rate = TTS_SAMPLE_RATE + 1


def build(
    monkeypatch: pytest.MonkeyPatch, voice: str = "af_heart", fake_cls: type = FakeKokoro
) -> tuple[KokoroSynthesizer, FakeKokoro]:
    fakes: list[FakeKokoro] = []

    def new_kokoro(model_path: str, voices_path: str) -> FakeKokoro:
        fakes.append(fake_cls(model_path, voices_path))
        return fakes[-1]

    monkeypatch.setattr(tts, "Kokoro", new_kokoro)
    synth = KokoroSynthesizer(WEIGHTS, VOICES, voice)
    return synth, fakes[0]


def test_to_int16_clamps_and_scales() -> None:
    audio = np.array([-1.0, 0.0, 1.0, 0.5, 1.5, -2.0], dtype=np.float32)

    out = to_int16(audio)

    assert out.dtype == np.int16
    assert out.ndim == 1
    np.testing.assert_array_equal(
        out, np.array([-32767, 0, 32767, 16383, 32767, -32767], dtype=np.int16)
    )


def test_kokoro_is_built_with_string_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = build(monkeypatch)

    assert fake.model_path == str(WEIGHTS)
    assert fake.voices_path == str(VOICES)


def test_synthesize_passes_row_190_of_the_default_voice_and_appends_the_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synth, fake = build(monkeypatch)

    out = synth.synthesize("Checking the acquire path.")
    synth.synthesize("Then the release path.")

    assert fake.styles == ["af_heart"]
    assert [text for text, _ in fake.calls] == [
        "Checking the acquire path.",
        "Then the release path.",
    ]
    row_190 = np.full((1, 1, 256), 190, dtype=np.float32)
    for _, kwargs in fake.calls:
        assert list(kwargs) == ["voice"]
        assert kwargs["voice"].shape == (1, 1, 256)
        np.testing.assert_array_equal(kwargs["voice"], row_190)
    expected = np.concatenate(
        [np.array([-32767, 0, 32767, 16383], dtype=np.int16), np.zeros(6000, dtype=np.int16)]
    )
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.int16
    assert out.ndim == 1


def test_a_custom_voice_names_the_style_table(monkeypatch: pytest.MonkeyPatch) -> None:
    synth, fake = build(monkeypatch, voice="am_adam")

    out = synth.synthesize("hello")

    assert fake.styles == ["am_adam"]
    [(text, kwargs)] = fake.calls
    assert text == "hello"
    assert kwargs["voice"].shape == (1, 1, 256)
    np.testing.assert_array_equal(kwargs["voice"], np.full((1, 1, 256), 190, dtype=np.float32))
    assert len(out) == 4 + 6000
    assert not out[4:].any()


def test_wrong_sample_rate_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    synth, _ = build(monkeypatch, fake_cls=BadRateKokoro)

    with pytest.raises(ValueError):
        synth.synthesize("hello")
