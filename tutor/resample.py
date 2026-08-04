import fractions
from collections import deque

import av
import numpy as np

from tutor.constants import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
)

INBOUND_STARTUP_DELAY_SAMPLES = 16  # fixed algorithmic delay of 48k -> 16k, measured
OUTBOUND_STARTUP_DELAY_SAMPLES = 32  # fixed algorithmic delay of 24k -> 48k, measured


class InboundResampler:
    def __init__(self) -> None:
        self._resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
            frame_size=FRAME_SAMPLES,
        )

    def push(self, frame: av.AudioFrame) -> list[np.ndarray]:
        return [out.to_ndarray().reshape(-1) for out in self._resampler.resample(frame)]


def _outbound_resampler() -> av.AudioResampler:
    return av.AudioResampler(
        format="s16",
        layout="mono",
        rate=WEBRTC_SAMPLE_RATE,
        frame_size=WEBRTC_FRAME_SAMPLES,
    )


class OutboundResampler:
    def __init__(self) -> None:
        self._resampler = _outbound_resampler()
        self._buffer: deque[np.ndarray] = deque()
        self._pts = 0

    def push(self, pcm: np.ndarray) -> None:
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = TTS_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, TTS_SAMPLE_RATE)
        self._pts += len(pcm)
        for out in self._resampler.resample(frame):
            self._buffer.append(out.to_ndarray().reshape(-1))

    def available(self) -> int:
        return sum(len(chunk) for chunk in self._buffer)

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.int16)
        filled = 0
        while filled < n and self._buffer:
            head = self._buffer[0]
            take = min(n - filled, len(head))
            out[filled : filled + take] = head[:take]
            filled += take
            if take == len(head):
                self._buffer.popleft()
            else:
                self._buffer[0] = head[take:]
        return out

    def flush(self) -> None:
        self._buffer.clear()
        self._resampler = _outbound_resampler()
