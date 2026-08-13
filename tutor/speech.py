import asyncio
from collections.abc import AsyncIterator

import numpy as np

from tutor.openers import synthesize_openers
from tutor.transport import Connection
from tutor.tts import KokoroSynthesizer


class Speaker:
    def __init__(self, synth: KokoroSynthesizer, transport: Connection) -> None:
        self._synth = synth
        self._transport = transport
        self._utterance: asyncio.Task[None] | None = None
        self._openers: dict[str, np.ndarray] = {}

    async def warm(self) -> None:
        self._openers = await asyncio.to_thread(synthesize_openers, self._synth)

    async def speak_opener(self, key: str) -> None:
        await self._transport.play(self._openers[key])

    async def _drain(self, chunks: AsyncIterator[str]) -> None:
        async for chunk in chunks:
            audio = await asyncio.to_thread(self._synth.synthesize, chunk)
            await self._transport.play(audio)

    async def speak(self, chunks: AsyncIterator[str]) -> None:
        if self._utterance is not None and not self._utterance.done():
            raise RuntimeError("an utterance is already in flight")
        self._utterance = asyncio.create_task(self._drain(chunks))
        try:
            await self._utterance
        finally:
            self._utterance = None
