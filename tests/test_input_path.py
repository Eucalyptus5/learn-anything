import asyncio
import threading
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from tutor.constants import FRAME_SAMPLES
from tutor.input_path import (
    PRE_ROLL_FRAMES,
    START_FRAMES,
    EndOfTurn,
    InputEvent,
    InputPath,
    PartialTranscript,
    SpeechStarted,
)

SPEECH_FRAMES = 40
SILENCE_FRAMES = 20
CANONICAL = [0.9] * SPEECH_FRAMES + [0.0] * SILENCE_FRAMES
FIRST_PARTIAL_FRAMES = 32
SECOND_PARTIAL_FRAMES = 64


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
        self.resets = 0

    def __call__(self, frame: np.ndarray) -> float:
        self.threads.append(threading.get_ident())
        return self._probabilities.pop(0)

    def reset(self) -> None:
        self.resets += 1


class FakeTranscriber:
    def __init__(self, name: str) -> None:
        self._name = name
        self.handed: list[np.ndarray] = []
        self.threads: list[int] = []

    def transcribe(self, audio: np.ndarray) -> str:
        self.handed.append(audio)
        self.threads.append(threading.get_ident())
        return f"{self._name} {audio.size}"


class GatedTranscriber(FakeTranscriber):
    def __init__(self, name: str, release: threading.Event) -> None:
        super().__init__(name)
        self._release = release

    def transcribe(self, audio: np.ndarray) -> str:
        text = super().transcribe(audio)
        self._release.wait()
        return text


class HeldTranscriber(FakeTranscriber):
    def __init__(
        self,
        name: str,
        release: threading.Event,
        started: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(name)
        self._release = release
        self._started = started
        self._loop = loop

    def transcribe(self, audio: np.ndarray) -> str:
        text = super().transcribe(audio)
        self._loop.call_soon_threadsafe(self._started.set)
        self._release.wait()
        return text


class ReleasingTranscriber(FakeTranscriber):
    def __init__(self, name: str, release: threading.Event) -> None:
        super().__init__(name)
        self._release = release

    def transcribe(self, audio: np.ndarray) -> str:
        self._release.set()
        return super().transcribe(audio)


@pytest.fixture
def release() -> Iterator[threading.Event]:
    event = threading.Event()
    yield event
    event.set()


def scripted_frames(count: int) -> list[np.ndarray]:
    return [np.full(FRAME_SAMPLES, i + 1, dtype=np.int16) for i in range(count)]


def build(
    probabilities: list[float],
    partial: FakeTranscriber | None = None,
    final: FakeTranscriber | None = None,
) -> tuple[InputPath, FakeVad, FakeTranscriber, FakeTranscriber]:
    vad = FakeVad(probabilities)
    partial = partial or FakeTranscriber("partial")
    final = final or FakeTranscriber("final")
    connection = FakeConnection(scripted_frames(len(probabilities)))
    return InputPath(connection, vad, partial, final), vad, partial, final


def without_partials(events: list[InputEvent]) -> list[InputEvent]:
    return [event for event in events if not isinstance(event, PartialTranscript)]


async def test_speech_started_then_end_of_turn_in_order() -> None:
    path, _, _, _ = build(CANONICAL)

    events = [event async for event in path.events()]

    assert without_partials(events) == [
        SpeechStarted(),
        EndOfTurn(text=f"final {(SPEECH_FRAMES + 1) * FRAME_SAMPLES}"),
    ]


async def test_end_of_turn_carries_the_final_transcript() -> None:
    path, _, _, final = build(CANONICAL)

    events = [event async for event in path.events()]

    assert events[-1] == EndOfTurn(text=f"final {(SPEECH_FRAMES + 1) * FRAME_SAMPLES}")
    assert len(final.handed) == 1
    audio = final.handed[0]
    assert audio.dtype == np.int16
    assert audio.size == (SPEECH_FRAMES + 1) * FRAME_SAMPLES


async def test_final_job_starts_at_silence_start() -> None:
    path, _, _, final = build(CANONICAL)

    events = [event async for event in path.events()]

    assert len(final.handed) == 1
    audio = final.handed[0]
    assert audio.size == (SPEECH_FRAMES + 1) * FRAME_SAMPLES
    assert int(audio[-1]) == SPEECH_FRAMES + 1
    assert events[-1] == EndOfTurn(text=f"final {audio.size}")


async def test_silent_stream_emits_nothing() -> None:
    path, _, partial, final = build([0.0] * 60)

    events = [event async for event in path.events()]

    assert events == []
    assert partial.handed == []
    assert final.handed == []


async def test_the_buffer_keeps_the_frames_before_speech_start() -> None:
    lead = START_FRAMES + PRE_ROLL_FRAMES
    path, _, _, final = build([0.0] * lead + CANONICAL)

    events = [event async for event in path.events()]

    assert len(without_partials(events)) == 2
    first = int(final.handed[0][0]) - 1
    assert first == lead - PRE_ROLL_FRAMES


async def test_partial_transcript_is_empty_before_speech() -> None:
    path, _, _, _ = build(CANONICAL)
    assert path.partial_transcript == ""

    async for event in path.events():
        assert event == SpeechStarted()
        assert path.partial_transcript == ""
        break

    await path.aclose()


async def test_partial_transcript_updates_while_speaking() -> None:
    path, _, partial, final = build([0.9] * 90)

    events = []
    async for event in path.events():
        events.append(event)
        if isinstance(event, PartialTranscript):
            assert path.partial_transcript == event.text

    assert [audio.size for audio in partial.handed] == [
        FIRST_PARTIAL_FRAMES * FRAME_SAMPLES,
        SECOND_PARTIAL_FRAMES * FRAME_SAMPLES,
    ]
    assert events == [
        SpeechStarted(),
        PartialTranscript(text=f"partial {FIRST_PARTIAL_FRAMES * FRAME_SAMPLES}"),
        PartialTranscript(text=f"partial {SECOND_PARTIAL_FRAMES * FRAME_SAMPLES}"),
    ]
    assert final.handed == []


async def test_no_partial_below_the_buffer_floor() -> None:
    speech = FIRST_PARTIAL_FRAMES - 2
    path, _, partial, _ = build([0.9] * speech + [0.0] * SILENCE_FRAMES)

    events = [event async for event in path.events()]

    assert partial.handed == []
    assert events == [SpeechStarted(), EndOfTurn(text=f"final {(speech + 1) * FRAME_SAMPLES}")]


async def test_late_partial_is_discarded(release: threading.Event) -> None:
    path, _, partial, _ = build(
        CANONICAL,
        partial=GatedTranscriber("partial", release),
        final=ReleasingTranscriber("final", release),
    )

    events = [event async for event in path.events()]

    assert [audio.size for audio in partial.handed] == [FIRST_PARTIAL_FRAMES * FRAME_SAMPLES]
    assert events == [
        SpeechStarted(),
        EndOfTurn(text=f"final {(SPEECH_FRAMES + 1) * FRAME_SAMPLES}"),
    ]


async def test_no_second_partial_while_one_is_in_flight(release: threading.Event) -> None:
    path, _, partial, _ = build([0.9] * 90, partial=GatedTranscriber("partial", release))

    events = [event async for event in path.events()]

    assert [audio.size for audio in partial.handed] == [FIRST_PARTIAL_FRAMES * FRAME_SAMPLES]
    assert events == [SpeechStarted()]


async def test_resume_discards_the_pending_final_and_rearms() -> None:
    gap = 5
    resumed = [0.9] * SPEECH_FRAMES + [0.0] * gap + [0.9] * SPEECH_FRAMES + [0.0] * SILENCE_FRAMES
    path, _, _, final = build(resumed)

    events = [event async for event in path.events()]

    last = final.handed[-1]
    assert last.size == (2 * SPEECH_FRAMES + gap + 1) * FRAME_SAMPLES
    assert without_partials(events) == [SpeechStarted(), EndOfTurn(text=f"final {last.size}")]


async def test_vad_does_not_run_on_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    path, vad, _, _ = build(CANONICAL)

    events = [event async for event in path.events()]

    assert len(without_partials(events)) == 2
    assert len(vad.threads) == len(CANONICAL)
    assert all(ident != loop_thread for ident in vad.threads)


async def test_transcription_does_not_run_on_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    path, _, partial, final = build(CANONICAL)

    events = [event async for event in path.events()]

    assert len(without_partials(events)) == 2
    assert len(final.threads) == 1
    assert final.threads[0] != loop_thread
    assert partial.threads[0] != loop_thread
    assert partial.threads[0] != final.threads[0]


async def test_aclose_closes_the_frame_generator_after_the_consumer_stops() -> None:
    connection = FakeConnection(scripted_frames(len(CANONICAL)))
    path = InputPath(
        connection, FakeVad(CANONICAL), FakeTranscriber("partial"), FakeTranscriber("final")
    )

    async for event in path.events():
        assert event == SpeechStarted()
        break

    await path.aclose()

    assert connection.closed
    await path.aclose()


async def test_cancelling_the_consumer_reraises_cancelled_error() -> None:
    path, vad, partial, final = build([0.9] * 90)
    speaking = asyncio.Event()

    async def consume() -> list[InputEvent]:
        events = []
        async for event in path.events():
            events.append(event)
            if isinstance(event, SpeechStarted):
                speaking.set()
        return events

    consumer = asyncio.create_task(consume())
    await speaking.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert vad.resets == 1
    assert path.partial_transcript == ""
    assert partial.handed == []
    assert final.handed == []

    rest = [event async for event in path.events()]

    assert without_partials(rest) == [SpeechStarted()]


async def test_aclose_is_idempotent(release: threading.Event) -> None:
    started = asyncio.Event()
    partial = HeldTranscriber("partial", release, started, asyncio.get_running_loop())
    connection = FakeConnection(scripted_frames(90))
    path = InputPath(connection, FakeVad([0.9] * 90), partial, FakeTranscriber("final"))

    async def consume() -> list[InputEvent]:
        return [event async for event in path.events()]

    consumer = asyncio.create_task(consume())
    await started.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    await path.aclose()
    await path.aclose()

    assert connection.closed
    assert not release.is_set()
    assert len(partial.handed) == 1


async def test_transport_iterator_exhaustion_ends_the_event_stream() -> None:
    connection = FakeConnection(scripted_frames(SPEECH_FRAMES))
    final = FakeTranscriber("final")
    path = InputPath(connection, FakeVad([0.9] * SPEECH_FRAMES), FakeTranscriber("partial"), final)

    events = [event async for event in path.events()]

    assert without_partials(events) == [SpeechStarted()]
    assert final.handed == []
    assert connection.closed
    await path.aclose()
