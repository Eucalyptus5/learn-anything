import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from tests.fakes import FakeSynthesizer, FakeTransport
from tutor.constants import TTS_SAMPLE_RATE
from tutor.speech import Chunk, Speaker

CHUNKS = ["one", "a longer clause", "two words"]
TIMING_CHUNKS = ["quorum", "the acquire path", "zebra crossing"]
SPAN_PATTERN = re.compile(r"^tts\.(synthesize|first_audio)( [a-z_]+=\d+)+$")


class LoggingSynthesizer(FakeSynthesizer):
    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__()
        self._log = log

    def synthesize(self, text: str) -> np.ndarray:
        self._log.append(("synthesize", text))
        return super().synthesize(text)


class LoggingTransport(FakeTransport):
    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__()
        self._log = log

    async def play(self, pcm: np.ndarray) -> None:
        self._log.append(("play", len(pcm)))
        await super().play(pcm)


class HeldSynthesizer(FakeSynthesizer):
    def __init__(
        self,
        release: threading.Event,
        started: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._release = release
        self._started = started
        self._loop = loop
        self.log: list[str] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.log.append("call")
        self._loop.call_soon_threadsafe(self._started.set)
        self._release.wait()
        audio = super().synthesize(text)
        self.log.append("done")
        return audio


class SignallingSynthesizer(HeldSynthesizer):
    def __init__(
        self,
        release: threading.Event,
        started: asyncio.Event,
        finished: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(release, started, loop)
        self._finished = finished

    def synthesize(self, text: str) -> np.ndarray:
        audio = super().synthesize(text)
        self._loop.call_soon_threadsafe(self._finished.set)
        return audio


class OverlappingSynthesizer(FakeSynthesizer):
    def __init__(
        self,
        release: threading.Event,
        started: asyncio.Event,
        finished: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._release = release
        self._started = started
        self._finished = finished
        self._loop = loop
        self._entered = threading.Event()
        self._first_done = threading.Event()
        self.first_done_at_replacement: bool | None = None

    def synthesize(self, text: str) -> np.ndarray:
        if self._entered.is_set():
            self.first_done_at_replacement = self._first_done.is_set()
            self._release.set()
            return super().synthesize(text)
        self._entered.set()
        audio = super().synthesize(text)
        self._loop.call_soon_threadsafe(self._started.set)
        self._release.wait()
        self._first_done.set()
        self._loop.call_soon_threadsafe(self._finished.set)
        return audio


class FlushLoggingTransport(FakeTransport):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    def flush_playout(self) -> None:
        self._log.append("flush")
        super().flush_playout()


class BackloggedTransport(FakeTransport):
    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__()
        self._log = log

    async def play(self, pcm: np.ndarray) -> None:
        self._log.append(("play", len(pcm)))
        await super().play(pcm)
        self.backlog_s += 0.5


class SizedSynthesizer(FakeSynthesizer):
    def synthesize(self, text: str) -> np.ndarray:
        super().synthesize(text)
        return np.zeros(len(text.split()) * TTS_SAMPLE_RATE // 4, dtype=np.int16)


async def no_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
    return None


@pytest.fixture
def release() -> Iterator[threading.Event]:
    event = threading.Event()
    yield event
    event.set()


async def source(chunks: list[str]) -> AsyncIterator[Chunk]:
    for n, text in enumerate(chunks, start=1):
        yield Chunk(n, text)


async def test_one_array_per_chunk_in_source_order() -> None:
    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    await speaker.speak(source(CHUNKS), no_play)

    assert synth.calls == CHUNKS
    assert [len(pcm) for pcm in transport.played] == [len(chunk) for chunk in CHUNKS]
    assert all(pcm.dtype == np.int16 for pcm in transport.played)


async def test_a_chunk_is_synthesized_only_after_the_previous_one_is_enqueued() -> None:
    log: list[tuple[str, object]] = []
    speaker = Speaker(LoggingSynthesizer(log), LoggingTransport(log))

    await speaker.speak(source(CHUNKS), no_play)

    assert log == [
        ("synthesize", CHUNKS[0]),
        ("play", len(CHUNKS[0])),
        ("synthesize", CHUNKS[1]),
        ("play", len(CHUNKS[1])),
        ("synthesize", CHUNKS[2]),
        ("play", len(CHUNKS[2])),
    ]


async def test_on_play_follows_the_enqueue_with_the_backlog_measured_before_it() -> None:
    log: list[tuple[str, object]] = []
    speaker = Speaker(FakeSynthesizer(), BackloggedTransport(log))

    async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
        log.append(("on_play", (chunk, lead_ms)))

    await speaker.speak(source(CHUNKS), on_play)

    assert log == [
        ("play", len(CHUNKS[0])),
        ("on_play", (Chunk(1, CHUNKS[0]), 0)),
        ("play", len(CHUNKS[1])),
        ("on_play", (Chunk(2, CHUNKS[1]), 500)),
        ("play", len(CHUNKS[2])),
        ("on_play", (Chunk(3, CHUNKS[2]), 1000)),
    ]


async def test_on_play_gets_the_chunk_its_lead_and_its_audio_length() -> None:
    log: list[tuple[str, object]] = []
    speaker = Speaker(SizedSynthesizer(), BackloggedTransport(log))
    plays: list[tuple[Chunk, int, int]] = []

    async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
        plays.append((chunk, lead_ms, audio_ms))

    await speaker.speak(source(CHUNKS), on_play)

    assert plays == [
        (Chunk(1, "one"), 0, 250),
        (Chunk(2, "a longer clause"), 500, 750),
        (Chunk(3, "two words"), 1000, 500),
    ]


async def test_the_speaker_synthesizes_the_text_and_never_renumbers_a_chunk() -> None:
    synth = FakeSynthesizer()
    ids: list[int] = []

    async def numbered() -> AsyncIterator[Chunk]:
        yield Chunk(7, "seven")
        yield Chunk(9, "nine")

    async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
        ids.append(chunk.id)

    await Speaker(synth, FakeTransport()).speak(numbered(), on_play)

    assert synth.calls == ["seven", "nine"] and ids == [7, 9]


async def test_a_cancelled_utterance_calls_on_play_for_nothing_after_the_cancel(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    synth = SignallingSynthesizer(release, started, finished, loop)
    speaker = Speaker(synth, FakeTransport())
    plays: list[tuple[Chunk, int]] = []

    async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
        plays.append((chunk, lead_ms))

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), on_play))
    await started.wait()
    await speaker.cancel()
    release.set()
    await finished.wait()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    assert synth.calls == [CHUNKS[0]]
    assert plays == []


async def test_on_play_stops_with_the_utterance_when_the_cancel_comes_between_chunks() -> None:
    waiting = asyncio.Event()
    gate = asyncio.Event()

    async def gated() -> AsyncIterator[Chunk]:
        yield Chunk(1, CHUNKS[0])
        waiting.set()
        await gate.wait()
        yield Chunk(2, CHUNKS[1])

    plays: list[tuple[Chunk, int]] = []

    async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
        plays.append((chunk, lead_ms))

    speaker = Speaker(FakeSynthesizer(), FakeTransport())
    utterance = asyncio.create_task(speaker.speak(gated(), on_play))
    await waiting.wait()
    await speaker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    gate.set()
    assert plays == [(Chunk(1, CHUNKS[0]), 0)]


async def test_speak_returns_when_the_source_is_exhausted() -> None:
    transport = FakeTransport()
    speaker = Speaker(FakeSynthesizer(), transport)

    await speaker.speak(source(CHUNKS), no_play)

    assert len(transport.played) == len(CHUNKS)
    assert speaker._utterance is None


async def test_an_empty_source_enqueues_nothing() -> None:
    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    await speaker.speak(source([]), no_play)

    assert synth.calls == []
    assert transport.played == []
    assert speaker._utterance is None


async def test_synthesis_does_not_run_on_the_event_loop(release: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()
    in_flight = list(synth.log)
    pending = list(transport.played)
    release.set()
    await utterance

    assert in_flight == ["call"]
    assert pending == []
    assert len(synth.threads) == len(CHUNKS)
    assert all(ident != loop_thread for ident in synth.threads)
    assert len(transport.played) == len(CHUNKS)


async def test_a_second_speak_while_one_is_in_flight_is_refused(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()

    with pytest.raises(RuntimeError):
        await speaker.speak(source(["rejected"]), no_play)

    assert "rejected" not in synth.calls

    release.set()
    await utterance

    assert synth.calls == CHUNKS
    assert len(transport.played) == len(CHUNKS)

    await speaker.speak(source(["after"]), no_play)

    assert synth.calls == [*CHUNKS, "after"]
    assert [len(pcm) for pcm in transport.played] == [
        *(len(chunk) for chunk in CHUNKS),
        len("after"),
    ]


async def test_cancel_drops_playout_once_before_speak_unwinds(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    log: list[str] = []
    transport = FlushLoggingTransport(log)
    speaker = Speaker(synth, transport)

    async def unwinding() -> None:
        try:
            await speaker.speak(source(CHUNKS), no_play)
        except asyncio.CancelledError:
            log.append("unwound")
            raise

    utterance = asyncio.create_task(unwinding())
    await started.wait()
    await speaker.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    assert transport.flushes == 1
    assert log == ["flush", "unwound"]


async def test_the_caller_awaiting_speak_sees_cancelled_error(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    speaker = Speaker(synth, FakeTransport())

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()
    await speaker.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    assert utterance.cancelled()
    assert speaker._utterance is None


async def test_cancel_while_suspended_on_the_source_stops_the_utterance() -> None:
    waiting = asyncio.Event()
    gate = asyncio.Event()

    async def gated() -> AsyncIterator[Chunk]:
        yield Chunk(1, CHUNKS[0])
        waiting.set()
        await gate.wait()
        yield Chunk(2, CHUNKS[1])

    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(gated(), no_play))
    await waiting.wait()
    await speaker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    gate.set()

    assert synth.calls == [CHUNKS[0]]
    assert [len(pcm) for pcm in transport.played] == [len(CHUNKS[0])]
    assert transport.flushes == 1


async def test_cancel_returns_normally_to_a_task_that_was_not_cancelled(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    speaker = Speaker(synth, FakeTransport())
    log: list[str] = []
    cancelling: list[int] = []

    async def canceller() -> None:
        await speaker.cancel()
        log.append("continued")
        cancelling.append(asyncio.current_task().cancelling())

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()
    caller = asyncio.create_task(canceller())
    await caller
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    assert not caller.cancelled()
    assert caller.result() is None
    assert log == ["continued"]
    assert cancelling == [0]


async def test_audio_already_in_the_worker_thread_is_never_enqueued(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    synth = SignallingSynthesizer(release, started, finished, loop)
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()
    await speaker.cancel()
    release.set()
    await finished.wait()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    assert synth.calls == [CHUNKS[0]]
    assert transport.played == []


async def test_speak_accepts_a_new_utterance_after_cancel(release: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    synth = SignallingSynthesizer(release, started, finished, loop)
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()
    await speaker.cancel()
    release.set()
    await finished.wait()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    await speaker.speak(source(["after"]), no_play)

    assert synth.calls[-1] == "after"
    assert [len(pcm) for pcm in transport.played] == [len("after")]
    assert speaker._utterance is None


async def test_cancel_on_an_idle_speaker_is_a_no_op() -> None:
    transport = FakeTransport()
    speaker = Speaker(FakeSynthesizer(), transport)

    await speaker.cancel()

    assert transport.flushes == 0

    await speaker.speak(source(CHUNKS), no_play)
    await speaker.cancel()

    assert transport.flushes == 0
    assert speaker._utterance is None


async def test_the_replacement_utterance_runs_while_the_abandoned_call_is_in_flight(
    release: threading.Event,
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    synth = OverlappingSynthesizer(release, started, finished, loop)
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
    await started.wait()
    await speaker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    await speaker.speak(source(["after"]), no_play)
    await finished.wait()

    assert synth.first_done_at_replacement is False
    assert synth.calls == [CHUNKS[0], "after"]
    assert [len(pcm) for pcm in transport.played] == [len("after")]
    assert transport.flushes == 1


async def test_a_cut_utterance_unwinding_late_leaves_the_replacement_in_place() -> None:
    parked = asyncio.Event()
    gate = asyncio.Event()
    replacement_parked = asyncio.Event()
    replacement_gate = asyncio.Event()

    async def parking() -> AsyncIterator[Chunk]:
        yield Chunk(1, CHUNKS[0])
        parked.set()
        await gate.wait()

    async def replacement() -> AsyncIterator[Chunk]:
        yield Chunk(2, CHUNKS[1])
        replacement_parked.set()
        await replacement_gate.wait()

    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    first = asyncio.create_task(speaker.speak(parking(), no_play))
    await parked.wait()
    first.cancel()
    second = asyncio.create_task(speaker.speak(replacement(), no_play))

    with pytest.raises(asyncio.CancelledError):
        await first
    await replacement_parked.wait()

    assert first.cancelled()
    assert not second.done()
    assert speaker._utterance is not None
    assert not speaker._utterance.done()

    replacement_gate.set()
    await second

    assert speaker._utterance is None


def tutor_speech_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "tutor.speech"]


async def test_one_utterance_logs_a_first_audio_span_and_one_synthesize_span_per_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        await speaker.speak(source(TIMING_CHUNKS), no_play)

    records = tutor_speech_records(caplog)
    messages = [record.getMessage() for record in records]

    first_audio = [m for m in messages if m.startswith("tts.first_audio ")]
    synthesize = [m for m in messages if m.startswith("tts.synthesize ")]

    assert len(first_audio) == 1
    assert f"words={len(TIMING_CHUNKS[0].split())}" in first_audio[0]

    assert len(synthesize) == len(TIMING_CHUNKS)
    for message, chunk in zip(synthesize, TIMING_CHUNKS, strict=True):
        assert f"words={len(chunk.split())}" in message

    for record in records:
        assert record.levelno == logging.DEBUG
        assert SPAN_PATTERN.match(record.getMessage())


async def test_no_span_carries_the_spoken_text(caplog: pytest.LogCaptureFixture) -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        await speaker.speak(source(TIMING_CHUNKS), no_play)

    messages = [record.getMessage() for record in tutor_speech_records(caplog)]

    for chunk in TIMING_CHUNKS:
        assert all(chunk not in message for message in messages)


async def test_an_empty_source_logs_no_spans(caplog: pytest.LogCaptureFixture) -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        await speaker.speak(source([]), no_play)

    assert tutor_speech_records(caplog) == []


async def test_cancel_before_first_audio_logs_no_first_audio_span(
    release: threading.Event, caplog: pytest.LogCaptureFixture
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    speaker = Speaker(synth, FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        utterance = asyncio.create_task(speaker.speak(source(CHUNKS), no_play))
        await started.wait()
        await speaker.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await utterance

    messages = [record.getMessage() for record in tutor_speech_records(caplog)]

    assert not any(m.startswith("tts.first_audio") for m in messages)
    assert not any(m.startswith("tts.synthesize") for m in messages)
