from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tutor import stt
from tutor.stt import FINAL_CPU_THREADS, PARTIAL_CPU_THREADS, Transcriber, load_whisper

MODEL_DIR = Path("models/whisper")


class FakeWhisperModel:
    def __init__(self, model_size_or_path: str, **kwargs: object) -> None:
        self.model_size_or_path = model_size_or_path
        self.kwargs = kwargs


class FakeModel:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: object):
        self.calls.append((audio, kwargs))
        segments = (SimpleNamespace(text=text) for text in self._texts)
        return segments, None


def test_segments_are_joined_and_stripped() -> None:
    model = FakeModel(["  hello there.", "how are you  "])

    text = Transcriber(model).transcribe(np.zeros(160, dtype=np.int16))

    assert text == "hello there. how are you"
    assert model.calls[0][1] == {"language": "en", "beam_size": 5}


def test_no_segments_gives_empty_string() -> None:
    model = FakeModel([])

    assert Transcriber(model).transcribe(np.zeros(160, dtype=np.int16)) == ""


def test_int16_audio_is_converted_before_the_call() -> None:
    model = FakeModel(["ok"])
    audio = np.array([-32768, 0, 32767], dtype=np.int16)

    Transcriber(model).transcribe(audio)

    handed = model.calls[0][0]
    assert handed.dtype == np.float32
    np.testing.assert_array_equal(handed, np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))


def test_non_int16_audio_raises_before_the_call() -> None:
    model = FakeModel(["ok"])

    for bad in (np.zeros(160, dtype=np.float32), np.zeros(160, dtype=np.int32)):
        with pytest.raises(ValueError):
            Transcriber(model).transcribe(bad)

    assert model.calls == []


def test_load_whisper_pins_the_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    assert PARTIAL_CPU_THREADS == 1
    assert FINAL_CPU_THREADS == 4
    monkeypatch.setattr(stt, "WhisperModel", FakeWhisperModel)

    final = load_whisper(MODEL_DIR)
    partial = load_whisper(MODEL_DIR, PARTIAL_CPU_THREADS)

    assert final.model_size_or_path == "base.en"
    assert final.kwargs == {
        "device": "cpu",
        "compute_type": "int8",
        "download_root": "models/whisper",
        "cpu_threads": FINAL_CPU_THREADS,
    }
    assert partial.kwargs["cpu_threads"] == PARTIAL_CPU_THREADS
