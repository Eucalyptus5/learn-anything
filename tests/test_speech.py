import asyncio
import threading
from collections.abc import AsyncIterator, Iterator

import numpy as np
import pytest

from tests.fakes import FakeSynthesizer, FakeTransport
from tutor.speech import Speaker

CHUNKS = ["one", "a longer clause", "two words"]


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
