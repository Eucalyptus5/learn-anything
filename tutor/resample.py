import av
import numpy as np

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE

INBOUND_STARTUP_DELAY_SAMPLES = 16  # fixed algorithmic delay of 48k -> 16k, measured


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
