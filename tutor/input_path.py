import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from tutor.endpointer import Endpointer, EndpointEvent
from tutor.stt import Transcriber
from tutor.transport import Connection
from tutor.vad import SileroVad

logger = logging.getLogger(__name__)

START_FRAMES = 3
PRE_ROLL_FRAMES = 2


class SpeechStarted(BaseModel):
    pass


class PartialTranscript(BaseModel):
    text: str


class EndOfTurn(BaseModel):
    text: str


InputEvent = SpeechStarted | PartialTranscript | EndOfTurn


class InputPath:
    def __init__(self, connection: Connection, vad: SileroVad, transcriber: Transcriber) -> None:
        self._frames = connection.frames()
        self._vad = vad
        self._transcriber = transcriber
        self._endpointer = Endpointer(start_frames=START_FRAMES)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=START_FRAMES + PRE_ROLL_FRAMES)
        self._utterance: list[np.ndarray] = []

    async def events(self) -> AsyncIterator[InputEvent]:
        loop = asyncio.get_running_loop()
        async for frame in self._frames:
            if self._utterance:
                self._utterance.append(frame)
            else:
                self._pre_roll.append(frame)

            probability = await asyncio.to_thread(self._vad, frame)
            event = self._endpointer.push(probability)

            if event is EndpointEvent.SPEECH_START:
                self._utterance.extend(self._pre_roll)
                self._pre_roll.clear()
                yield SpeechStarted()
            elif event is EndpointEvent.END_OF_TURN:
                audio = np.concatenate(self._utterance)
                text = await loop.run_in_executor(
                    self._executor, self._transcriber.transcribe, audio
                )
                self._utterance.clear()
                logger.info("end_of_turn samples=%d", audio.size)
                yield EndOfTurn(text=text)

    async def aclose(self) -> None:
        await self._frames.aclose()
        self._executor.shutdown(wait=False)
