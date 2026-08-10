import threading
from collections.abc import AsyncIterator

import numpy as np

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE
from tutor.endpointer import SILENCE_WINDOW_MS
from tutor.input_path import PRE_ROLL_FRAMES, START_FRAMES, EndOfTurn, InputPath, SpeechStarted

WINDOW_SAMPLES = SILENCE_WINDOW_MS * SAMPLE_RATE // 1000
SILENT_FRAMES_TO_CLOSE = -(-WINDOW_SAMPLES // FRAME_SAMPLES)
SPEECH_FRAMES = 40
SILENCE_FRAMES = 20
TRANSCRIPT = "walk me through the endpointer"


class FakeConnection:
    def __init__(self, audio: list[np.ndarray]) -> None:
        self._audio = audio
        self.closed = False

    async def frames(self) -> AsyncIterator[np.ndarray]:
        try:
            for frame in self._audio:
                yield frame
        finally:
            self.closed = True


class FakeVad:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = list(probabilities)
        self.threads: list[int] = []

    def __call__(self, frame: np.ndarray) -> float:
        self.threads.append(threading.get_ident())
        return self._probabilities.pop(0)


class FakeTranscriber:
    def __init__(self) -> None:
        self.handed: list[np.ndarray] = []
        self.threads: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.handed.append(audio)
        self.threads.append(threading.get_ident())
        return TRANSCRIPT


def scripted_frames(count: int) -> list[np.ndarray]:
    return [np.full(FRAME_SAMPLES, i + 1, dtype=np.int16) for i in range(count)]


def build(lead: int = 0) -> tuple[InputPath, FakeVad, FakeTranscriber]:
    probabilities = [0.0] * lead + [0.9] * SPEECH_FRAMES + [0.0] * SILENCE_FRAMES
    vad = FakeVad(probabilities)
    transcriber = FakeTranscriber()
    connection = FakeConnection(scripted_frames(len(probabilities)))
    return InputPath(connection, vad, transcriber), vad, transcriber


async def test_speech_started_then_end_of_turn_in_order() -> None:
    path, _, _ = build()

    events = [event async for event in path.events()]

    assert events == [SpeechStarted(), EndOfTurn(text=TRANSCRIPT)]


async def test_end_of_turn_carries_the_final_transcript() -> None:
    path, _, transcriber = build()

    events = [event async for event in path.events()]

    assert events[-1].text == TRANSCRIPT
    assert len(transcriber.handed) == 1
    audio = transcriber.handed[0]
    assert audio.dtype == np.int16
    assert audio.size == (SPEECH_FRAMES + SILENT_FRAMES_TO_CLOSE) * FRAME_SAMPLES


async def test_silent_stream_emits_nothing() -> None:
    silent = scripted_frames(60)
    transcriber = FakeTranscriber()
    path = InputPath(FakeConnection(silent), FakeVad([0.0] * len(silent)), transcriber)

    events = [event async for event in path.events()]

    assert events == []
    assert transcriber.handed == []


async def test_the_buffer_keeps_the_frames_before_speech_start() -> None:
    lead = START_FRAMES + PRE_ROLL_FRAMES
    path, _, transcriber = build(lead=lead)

    events = [event async for event in path.events()]

    assert len(events) == 2
    first = int(transcriber.handed[0][0]) - 1
    assert first == lead - PRE_ROLL_FRAMES


async def test_vad_does_not_run_on_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    path, vad, _ = build()

    events = [event async for event in path.events()]

    assert len(events) == 2
    assert len(vad.threads) == SPEECH_FRAMES + SILENCE_FRAMES
    assert all(ident != loop_thread for ident in vad.threads)


async def test_transcription_does_not_run_on_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    path, _, transcriber = build()

    events = [event async for event in path.events()]

    assert len(events) == 2
    assert len(transcriber.threads) == 1
    assert transcriber.threads[0] != loop_thread


async def test_aclose_closes_the_frame_generator_after_the_consumer_stops() -> None:
    probabilities = [0.9] * SPEECH_FRAMES + [0.0] * SILENCE_FRAMES
    connection = FakeConnection(scripted_frames(len(probabilities)))
    path = InputPath(connection, FakeVad(probabilities), FakeTranscriber())

    async for event in path.events():
        assert event == SpeechStarted()
        break

    await path.aclose()

    assert connection.closed
    await path.aclose()
