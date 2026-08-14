import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from tests.fakes import FakeSynthesizer, FakeTransport
from tutor.openers import OPENER_PHRASES
from tutor.speech import Speaker

CHUNKS = ["one", "a longer clause", "two words"]
TIMING_CHUNKS = ["quorum", "the acquire path", "zebra crossing"]
SPAN_PATTERN = re.compile(r"^tts\.(synthesize|first_audio|opener)( [a-z_]+=\d+)+$")


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


@pytest.fixture
def release() -> Iterator[threading.Event]:
    event = threading.Event()
    yield event
    event.set()


async def source(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def test_one_array_per_chunk_in_source_order() -> None:
    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    await speaker.speak(source(CHUNKS))

    assert synth.calls == CHUNKS
    assert [len(pcm) for pcm in transport.played] == [len(chunk) for chunk in CHUNKS]
    assert all(pcm.dtype == np.int16 for pcm in transport.played)


async def test_a_chunk_is_synthesized_only_after_the_previous_one_is_enqueued() -> None:
    log: list[tuple[str, object]] = []
    speaker = Speaker(LoggingSynthesizer(log), LoggingTransport(log))

    await speaker.speak(source(CHUNKS))

    assert log == [
        ("synthesize", CHUNKS[0]),
        ("play", len(CHUNKS[0])),
        ("synthesize", CHUNKS[1]),
        ("play", len(CHUNKS[1])),
        ("synthesize", CHUNKS[2]),
        ("play", len(CHUNKS[2])),
    ]


async def test_speak_returns_when_the_source_is_exhausted() -> None:
    transport = FakeTransport()
    speaker = Speaker(FakeSynthesizer(), transport)

    await speaker.speak(source(CHUNKS))

    assert len(transport.played) == len(CHUNKS)
    assert speaker._utterance is None


async def test_an_empty_source_enqueues_nothing() -> None:
    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    await speaker.speak(source([]))

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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
    await started.wait()

    with pytest.raises(RuntimeError):
        await speaker.speak(source(["rejected"]))

    assert "rejected" not in synth.calls

    release.set()
    await utterance

    assert synth.calls == CHUNKS
    assert len(transport.played) == len(CHUNKS)

    await speaker.speak(source(["after"]))

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
            await speaker.speak(source(CHUNKS))
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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
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

    async def gated() -> AsyncIterator[str]:
        yield CHUNKS[0]
        waiting.set()
        await gate.wait()
        yield CHUNKS[1]

    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    utterance = asyncio.create_task(speaker.speak(gated()))
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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
    await started.wait()
    await speaker.cancel()
    release.set()
    await finished.wait()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    await speaker.speak(source(["after"]))

    assert synth.calls[-1] == "after"
    assert [len(pcm) for pcm in transport.played] == [len("after")]
    assert speaker._utterance is None


async def test_cancel_on_an_idle_speaker_is_a_no_op() -> None:
    transport = FakeTransport()
    speaker = Speaker(FakeSynthesizer(), transport)

    await speaker.cancel()

    assert transport.flushes == 0

    await speaker.speak(source(CHUNKS))
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

    utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
    await started.wait()
    await speaker.cancel()

    with pytest.raises(asyncio.CancelledError):
        await utterance

    await speaker.speak(source(["after"]))
    await finished.wait()

    assert synth.first_done_at_replacement is False
    assert synth.calls == [CHUNKS[0], "after"]
    assert [len(pcm) for pcm in transport.played] == [len("after")]
    assert transport.flushes == 1


async def test_warm_fills_the_cache_and_calls_the_synth_once_per_opener() -> None:
    synth = FakeSynthesizer()
    speaker = Speaker(synth, FakeTransport())

    await speaker.warm()

    assert synth.calls == list(OPENER_PHRASES.values())
    assert set(speaker._openers) == set(OPENER_PHRASES)


async def test_warm_does_not_block_the_event_loop(release: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    speaker = Speaker(synth, FakeTransport())

    warming = asyncio.create_task(speaker.warm())
    await started.wait()

    assert synth.log == ["call"]
    assert speaker._openers == {}

    release.set()
    await warming

    assert len(synth.threads) == len(OPENER_PHRASES)
    assert all(ident != loop_thread for ident in synth.threads)
    assert set(speaker._openers) == set(OPENER_PHRASES)


async def test_speak_opener_enqueues_the_cached_array_without_resynthesizing() -> None:
    synth = FakeSynthesizer()
    transport = FakeTransport()
    speaker = Speaker(synth, transport)

    await speaker.warm()
    calls_after_warm = list(synth.calls)

    await speaker.speak_opener("thinking")

    assert synth.calls == calls_after_warm
    assert len(transport.played) == 1
    np.testing.assert_array_equal(transport.played[0], speaker._openers["thinking"])


async def test_speak_opener_with_an_unknown_key_raises_key_error() -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())
    await speaker.warm()

    with pytest.raises(KeyError):
        await speaker.speak_opener("nope")


def tutor_speech_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "tutor.speech"]


async def test_one_utterance_logs_a_first_audio_span_and_one_synthesize_span_per_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        await speaker.speak(source(TIMING_CHUNKS))

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
        await speaker.speak(source(TIMING_CHUNKS))

    messages = [record.getMessage() for record in tutor_speech_records(caplog)]

    for chunk in TIMING_CHUNKS:
        assert all(chunk not in message for message in messages)


async def test_an_empty_source_logs_no_spans(caplog: pytest.LogCaptureFixture) -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        await speaker.speak(source([]))

    assert tutor_speech_records(caplog) == []


async def test_speak_opener_logs_one_opener_span_without_the_phrase_or_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    speaker = Speaker(FakeSynthesizer(), FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        await speaker.warm()
        warm_records = tutor_speech_records(caplog)
        await speaker.speak_opener("thinking")
        opener_records = tutor_speech_records(caplog)[len(warm_records) :]

    assert warm_records == []
    assert len(opener_records) == 1

    message = opener_records[0].getMessage()
    assert message.startswith("tts.opener ")
    assert SPAN_PATTERN.match(message)
    assert "thinking" not in message
    assert OPENER_PHRASES["thinking"] not in message


async def test_cancel_before_first_audio_logs_no_first_audio_span(
    release: threading.Event, caplog: pytest.LogCaptureFixture
) -> None:
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    synth = HeldSynthesizer(release, started, loop)
    speaker = Speaker(synth, FakeTransport())

    with caplog.at_level(logging.DEBUG, logger="tutor.speech"):
        utterance = asyncio.create_task(speaker.speak(source(CHUNKS)))
        await started.wait()
        await speaker.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await utterance

    messages = [record.getMessage() for record in tutor_speech_records(caplog)]

    assert not any(m.startswith("tts.first_audio") for m in messages)
    assert not any(m.startswith("tts.synthesize") for m in messages)
