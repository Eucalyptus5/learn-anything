import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE
from tutor.endpointer import Endpointer, EndpointEvent
from tutor.stt import Transcriber
from tutor.transport import Connection
from tutor.vad import SileroVad

logger = logging.getLogger(__name__)

START_FRAMES = 3
START_THRESHOLD = 0.5
PRE_ROLL_FRAMES = 2
# below this faster-whisper's temperature fallback re-decodes the fragment and takes seconds
PARTIAL_FLOOR_MS = 1024
PARTIAL_STRIDE_MS = 1000
PARTIAL_FLOOR_SAMPLES = PARTIAL_FLOOR_MS * SAMPLE_RATE // 1000
PARTIAL_STRIDE_SAMPLES = PARTIAL_STRIDE_MS * SAMPLE_RATE // 1000


class SpeechStarted(BaseModel):
    pass


class PartialTranscript(BaseModel):
    text: str


class EndOfTurn(BaseModel):
    text: str


InputEvent = SpeechStarted | PartialTranscript | EndOfTurn


class InputPath:
    def __init__(
        self,
        connection: Connection,
        vad: SileroVad,
        partial_transcriber: Transcriber,
        final_transcriber: Transcriber,
    ) -> None:
        self._frames = connection.frames()
        self._vad = vad
        self._partial_transcriber = partial_transcriber
        self._final_transcriber = final_transcriber
        self._endpointer = Endpointer(start_frames=START_FRAMES, start_threshold=START_THRESHOLD)
        self._partial_executor = ThreadPoolExecutor(max_workers=1)
        self._final_executor = ThreadPoolExecutor(max_workers=1)
        # the confirming frames are still in the deque at SPEECH_START and eat into its room
        self._pre_roll: deque[np.ndarray] = deque(maxlen=START_FRAMES + PRE_ROLL_FRAMES)
        self._utterance: list[np.ndarray] = []
        self._partial: asyncio.Future[str] | None = None
        self._final: asyncio.Future[str] | None = None
        self._partial_samples = 0
        self._partial_text = ""

    @property
    def partial_transcript(self) -> str:
        return self._partial_text

    async def events(self) -> AsyncIterator[InputEvent]:
        loop = asyncio.get_running_loop()
        async for frame in self._frames:
            if self._utterance:
                self._utterance.append(frame)
            else:
                self._pre_roll.append(frame)

            probability = await asyncio.to_thread(self._vad, frame)

            if self._partial is not None and self._partial.done():
                self._partial_text = self._partial.result()
                self._partial = None
                yield PartialTranscript(text=self._partial_text)

            event = self._endpointer.push(probability)

            if event is EndpointEvent.SPEECH_START:
                self._utterance.extend(self._pre_roll)
                self._pre_roll.clear()
                yield SpeechStarted()
            elif event is EndpointEvent.SILENCE_START:
                if self._partial is not None:
                    self._partial.cancel()
                    self._partial = None
                audio = np.concatenate(self._utterance)
                self._final = loop.run_in_executor(
                    self._final_executor, self._final_transcriber.transcribe, audio
                )
            elif event is EndpointEvent.END_OF_TURN:
                text = await self._final
                self._final = None
                samples = len(self._utterance) * FRAME_SAMPLES
                self._utterance.clear()
                self._partial_samples = 0
                self._partial_text = ""
                logger.info("end_of_turn samples=%d", samples)
                yield EndOfTurn(text=text)
            elif self._final is not None and probability > START_THRESHOLD:
                self._final.cancel()
                self._final = None

            if self._utterance and self._final is None and self._partial is None:
                samples = len(self._utterance) * FRAME_SAMPLES
                if (
                    samples >= PARTIAL_FLOOR_SAMPLES
                    and samples - self._partial_samples >= PARTIAL_STRIDE_SAMPLES
                ):
                    audio = np.concatenate(self._utterance)
                    self._partial = loop.run_in_executor(
                        self._partial_executor, self._partial_transcriber.transcribe, audio
                    )
                    self._partial_samples = samples

    async def aclose(self) -> None:
        self._partial_executor.shutdown(wait=False)
        self._final_executor.shutdown(wait=False)
        await self._frames.aclose()
