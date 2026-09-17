import asyncio
import fractions
import time
from collections.abc import Awaitable, Callable

import av
import numpy as np
from aiortc.mediastreams import MediaStreamTrack

from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.resample import OutboundResampler

PLAYOUT_QUEUE_CHUNKS = 32


class PlayoutTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self,
        pace: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        super().__init__()
        self._pace = pace
        self._clock = clock
        self._resampler = OutboundResampler()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=PLAYOUT_QUEUE_CHUNKS)
        self._queued = 0
        self._start = 0.0
        self._timestamp = 0
        self._started = False

    async def enqueue(self, pcm: np.ndarray) -> None:
        await self._queue.put(pcm)
        self._queued += len(pcm)

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queued = 0
        self._resampler.flush()

    def backlog_s(self) -> float:
        return self._queued / TTS_SAMPLE_RATE + self._resampler.available() / WEBRTC_SAMPLE_RATE

    async def recv(self) -> av.AudioFrame:
        if self._started:
            self._timestamp += WEBRTC_FRAME_SAMPLES
            await self._pace(self._start + self._timestamp / WEBRTC_SAMPLE_RATE - self._clock())
        else:
            self._start = self._clock()
            self._started = True

        while True:
            try:
                pcm = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queued -= len(pcm)
            self._resampler.push(pcm)

        pcm = self._resampler.pull(WEBRTC_FRAME_SAMPLES)
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        return frame
