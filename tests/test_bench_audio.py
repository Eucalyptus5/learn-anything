import importlib.util
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE
from tutor.endpointer import SILENCE_WINDOW_MS

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_audio.py"
_spec = importlib.util.spec_from_file_location("bench_audio", SCRIPT)
bench_audio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_audio)


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: np.ndarray, **options: Any) -> tuple[Any, None]:
        self.calls += 1
        return iter(()), None


def test_unconverted_audio_never_reaches_the_model() -> None:
    model = FakeWhisperModel()

    with pytest.raises(ValueError):
        bench_audio.whisper_transcribe(model, np.zeros(SAMPLE_RATE, dtype=np.int16))

    assert model.calls == 0


def test_to_float32_rejects_audio_that_is_already_converted() -> None:
    with pytest.raises(ValueError):
        bench_audio.to_float32(np.zeros(16, dtype=np.float32))


def test_to_float32_maps_int16_extremes_into_the_unit_range() -> None:
    scaled = bench_audio.to_float32(np.array([-32768, 0, 32767], dtype=np.int16))

    assert scaled.dtype == np.float32
    assert scaled[0] == -1.0
    assert scaled[1] == 0.0
    assert scaled[2] == 32767 / 32768


def test_report_us_reports_microseconds(capsys: pytest.CaptureFixture[str]) -> None:
    bench_audio.report_us("per-frame inference", [0.000123, 0.000456, 0.000789])
    out = capsys.readouterr().out

    assert "median=  456us" in out
    assert "ms" not in out


def test_ensure_fixture_returns_int16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    path = tmp_path / "utterance.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    monkeypatch.setattr(bench_audio, "FIXTURE", path)

    audio = bench_audio.ensure_fixture()

    assert audio.dtype == np.int16
    np.testing.assert_array_equal(audio, samples)


def silence_frames_to_end_of_turn() -> int:
    window = SILENCE_WINDOW_MS * SAMPLE_RATE // 1000
    return window // FRAME_SAMPLES + 1


def test_replay_times_a_turn_from_the_last_silence_start() -> None:
    tail = silence_frames_to_end_of_turn()
    probabilities = [0.9] * 3 + [0.0] * 2 + [0.9] * 3 + [0.0] * tail
    frames = [(p, float(i)) for i, p in enumerate(probabilities)]

    turns = bench_audio.replay_turns(frames, [100.0])

    assert len(turns) == 1
    assert turns[0].silence_at == 8.0
    assert turns[0].silence_starts == 2
    assert turns[0].end_at == 100.0
    assert turns[0].wait == 92.0


def test_replay_pairs_turns_with_end_stamps_in_order() -> None:
    utterance = [0.9] * 3 + [0.0] * silence_frames_to_end_of_turn()
    frames = [(p, float(i)) for i, p in enumerate(utterance * 2)]

    turns = bench_audio.replay_turns(frames, [50.0, 90.0])

    assert [t.silence_at for t in turns] == [3.0, 22.0]
    assert [t.end_at for t in turns] == [50.0, 90.0]
    assert [t.silence_starts for t in turns] == [1, 1]
