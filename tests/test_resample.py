import fractions

import av
import numpy as np
import pytest

from tutor.constants import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
)
from tutor.resample import INBOUND_STARTUP_DELAY_SAMPLES, InboundResampler

OUTPUT_SAMPLES_PER_WEBRTC_FRAME = WEBRTC_FRAME_SAMPLES * SAMPLE_RATE // WEBRTC_SAMPLE_RATE


def webrtc_frame(samples: np.ndarray, pts: int) -> av.AudioFrame:
    interleaved = np.repeat(samples, 2).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
    frame.sample_rate = WEBRTC_SAMPLE_RATE
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
    return frame


def silence_frames(count: int, n_samples: int) -> list[av.AudioFrame]:
    silence = np.zeros(n_samples, dtype=np.int16)
    return [webrtc_frame(silence, i * n_samples) for i in range(count)]


@pytest.mark.parametrize("n_samples", [160, 480, 960, 1920])
def test_every_output_array_is_one_vad_frame(n_samples: int) -> None:
    resampler = InboundResampler()
    outputs = [array for frame in silence_frames(20, n_samples) for array in resampler.push(frame)]

    assert outputs
    for array in outputs:
        assert array.dtype == np.int16
        assert array.ndim == 1
        assert len(array) == FRAME_SAMPLES


@pytest.mark.parametrize("n_frames", [30, 300, 1500])
def test_shortfall_is_bounded_and_does_not_accumulate(n_frames: int) -> None:
    resampler = InboundResampler()
    produced = sum(
        len(array)
        for frame in silence_frames(n_frames, WEBRTC_FRAME_SAMPLES)
        for array in resampler.push(frame)
    )

    shortfall = n_frames * OUTPUT_SAMPLES_PER_WEBRTC_FRAME - produced
    assert 0 <= shortfall < FRAME_SAMPLES + INBOUND_STARTUP_DELAY_SAMPLES


def test_long_run_lags_no_further_behind_than_a_short_one() -> None:
    def shortfall(n_frames: int) -> int:
        resampler = InboundResampler()
        produced = sum(
            len(array)
            for frame in silence_frames(n_frames, WEBRTC_FRAME_SAMPLES)
            for array in resampler.push(frame)
        )
        return n_frames * OUTPUT_SAMPLES_PER_WEBRTC_FRAME - produced

    assert shortfall(1500) <= shortfall(30)


def test_a_tone_survives_the_conversion() -> None:
    tone_hz = 440.0
    n_frames = 50
    t = np.arange(n_frames * WEBRTC_FRAME_SAMPLES) / WEBRTC_SAMPLE_RATE
    tone = (np.sin(2 * np.pi * tone_hz * t) * 30000).astype(np.int16)

    resampler = InboundResampler()
    outputs = [
        array
        for i in range(n_frames)
        for array in resampler.push(
            webrtc_frame(
                tone[i * WEBRTC_FRAME_SAMPLES : (i + 1) * WEBRTC_FRAME_SAMPLES],
                i * WEBRTC_FRAME_SAMPLES,
            )
        )
    ]
    signal = np.concatenate(outputs).astype(np.float64)

    magnitudes = np.abs(np.fft.rfft(signal))
    magnitudes[0] = 0.0
    peak_hz = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)[int(np.argmax(magnitudes))]

    assert abs(peak_hz - tone_hz) <= SAMPLE_RATE / len(signal)
