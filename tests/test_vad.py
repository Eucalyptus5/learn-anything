from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from tutor.constants import FRAME_SAMPLES
from tutor.vad import CONTEXT_SAMPLES, STATE_SHAPE, SileroVad

MODEL_PATH = Path("silero_vad.onnx")


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []
        self.states: list[np.ndarray] = []

    def run(self, outputs: None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: value.copy() for name, value in feeds.items()})
        n = len(self.calls)
        state = np.full(STATE_SHAPE, n, dtype=np.float32)
        self.states.append(state)
        return [np.array([[n / 8.0]], dtype=np.float32), state]


def build(monkeypatch: pytest.MonkeyPatch) -> tuple[SileroVad, FakeSession]:
    sessions: list[FakeSession] = []

    def new_session(model_path: str, **options: object) -> FakeSession:
        sessions.append(FakeSession())
        return sessions[-1]

    monkeypatch.setattr(ort, "InferenceSession", new_session)
    return SileroVad(MODEL_PATH), sessions[0]


def test_frame_size_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = build(monkeypatch)

    for bad in (
        np.zeros(FRAME_SAMPLES - 1, dtype=np.int16),
        np.zeros(FRAME_SAMPLES + 1, dtype=np.int16),
        np.zeros(FRAME_SAMPLES, dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            vad(bad)

    assert session.calls == []


def test_context_tail_is_previous_64_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = build(monkeypatch)
    first = (np.arange(FRAME_SAMPLES) * 3 - 700).astype(np.int16)
    second = (np.arange(FRAME_SAMPLES) * -5 + 400).astype(np.int16)

    vad(first)
    vad(second)

    assert not session.calls[0]["input"][:, :CONTEXT_SAMPLES].any()

    x = session.calls[1]["input"]
    assert x.shape == (1, CONTEXT_SAMPLES + FRAME_SAMPLES)
    np.testing.assert_array_equal(
        x[0, :CONTEXT_SAMPLES], first[-CONTEXT_SAMPLES:].astype(np.float32) / 32768.0
    )
    np.testing.assert_array_equal(x[0, CONTEXT_SAMPLES:], second.astype(np.float32) / 32768.0)


def test_state_is_fed_back_between_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = build(monkeypatch)

    vad(np.full(FRAME_SAMPLES, 1000, dtype=np.int16))
    vad(np.full(FRAME_SAMPLES, -1000, dtype=np.int16))

    opening = session.calls[0]["state"]
    assert opening.shape == STATE_SHAPE
    assert opening.dtype == np.float32
    assert not opening.any()
    np.testing.assert_array_equal(session.calls[1]["state"], session.states[0])


def test_reset_clears_state_and_context(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = build(monkeypatch)

    vad(np.full(FRAME_SAMPLES, 9000, dtype=np.int16))
    vad.reset()
    vad(np.full(FRAME_SAMPLES, -9000, dtype=np.int16))

    after = session.calls[1]
    assert not after["state"].any()
    assert not after["input"][:, :CONTEXT_SAMPLES].any()


def test_int16_frame_scales_to_unit_float(monkeypatch: pytest.MonkeyPatch) -> None:
    vad, session = build(monkeypatch)
    frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    frame[0] = -32768
    frame[1] = 0
    frame[2] = 32767

    probability = vad(frame)

    x = session.calls[0]["input"]
    assert x.dtype == np.float32
    assert x[0, CONTEXT_SAMPLES] == -1.0
    assert x[0, CONTEXT_SAMPLES + 1] == 0.0
    assert x[0, CONTEXT_SAMPLES + 2] == 32767 / 32768
    assert isinstance(probability, float)
    assert probability == 1 / 8.0
