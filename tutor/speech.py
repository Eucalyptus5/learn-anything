import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from tutor.constants import TTS_SAMPLE_RATE
from tutor.transport import Connection
from tutor.tts import KokoroSynthesizer

logger = logging.getLogger(__name__)

OnPlay = Callable[[str, int], Awaitable[None]]


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class Speaker:
    def __init__(self, synth: KokoroSynthesizer, transport: Connection) -> None:
        self._synth = synth
        self._transport = transport
        self._utterance: asyncio.Task[None] | None = None

    async def _drain(self, chunks: AsyncIterator[str], on_play: OnPlay) -> None:
        start = time.perf_counter()
        first = True
        async for chunk in chunks:
            chunk_start = time.perf_counter()
            audio = await asyncio.to_thread(self._synth.synthesize, chunk)
            words = len(chunk.split())
            logger.debug(
                "tts.synthesize words=%d audio_ms=%d ms=%d",
                words,
                len(audio) * 1000 // TTS_SAMPLE_RATE,
                _elapsed_ms(chunk_start),
            )
            backlog = self._transport.playout_backlog_s()
            await self._transport.play(audio)
            if first:
                logger.debug("tts.first_audio words=%d ms=%d", words, _elapsed_ms(start))
                first = False
            await on_play(chunk, int(backlog * 1000))

    async def speak(self, chunks: AsyncIterator[str], on_play: OnPlay) -> None:
        if self._utterance is not None and not self._utterance.done():
            raise RuntimeError("an utterance is already in flight")
        utterance = asyncio.create_task(self._drain(chunks, on_play))
        self._utterance = utterance
        try:
            await utterance
        finally:
            if self._utterance is utterance:
                self._utterance = None

    async def cancel(self) -> None:
        utterance = self._utterance
        if utterance is None or utterance.done():
            return
        self._transport.flush_playout()
        utterance.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await utterance
