import fractions

import av
import numpy as np
import pytest

from tutor.constants import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
)
from tutor.resample import (
    INBOUND_STARTUP_DELAY_SAMPLES,
    OUTBOUND_STARTUP_DELAY_SAMPLES,
    InboundResampler,
    OutboundResampler,
)

OUTPUT_SAMPLES_PER_WEBRTC_FRAME = WEBRTC_FRAME_SAMPLES * SAMPLE_RATE // WEBRTC_SAMPLE_RATE
TTS_SAMPLES_PER_WEBRTC_FRAME = WEBRTC_FRAME_SAMPLES * TTS_SAMPLE_RATE // WEBRTC_SAMPLE_RATE
WEBRTC_SAMPLES_PER_TTS_SAMPLE = WEBRTC_SAMPLE_RATE // TTS_SAMPLE_RATE


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


def tts_tone(n_chunks: int, tone_hz: float) -> np.ndarray:
    t = np.arange(n_chunks * TTS_SAMPLES_PER_WEBRTC_FRAME) / TTS_SAMPLE_RATE
    return (np.sin(2 * np.pi * tone_hz * t) * 30000).astype(np.int16)


def push_chunks(resampler: OutboundResampler, pcm: np.ndarray) -> None:
    for i in range(0, len(pcm), TTS_SAMPLES_PER_WEBRTC_FRAME):
        resampler.push(pcm[i : i + TTS_SAMPLES_PER_WEBRTC_FRAME])


def test_pull_returns_exactly_what_it_was_asked_for() -> None:
    resampler = OutboundResampler()
    push_chunks(resampler, tts_tone(10, 440.0))
    pulled = resampler.pull(WEBRTC_FRAME_SAMPLES)

    assert pulled.dtype == np.int16
    assert pulled.ndim == 1
    assert len(pulled) == WEBRTC_FRAME_SAMPLES


def test_pull_on_an_empty_buffer_is_silence_not_a_short_read() -> None:
    resampler = OutboundResampler()

    assert resampler.available() == 0
    pulled = resampler.pull(WEBRTC_FRAME_SAMPLES)
    assert len(pulled) == WEBRTC_FRAME_SAMPLES
    assert not pulled.any()


@pytest.mark.parametrize("n_chunks", [20, 200])
def test_outbound_shortfall_is_bounded(n_chunks: int) -> None:
    resampler = OutboundResampler()
    pcm = tts_tone(n_chunks, 440.0)
    push_chunks(resampler, pcm)

    shortfall = WEBRTC_SAMPLES_PER_TTS_SAMPLE * len(pcm) - resampler.available()
    assert 0 <= shortfall < WEBRTC_FRAME_SAMPLES + OUTBOUND_STARTUP_DELAY_SAMPLES


def test_a_long_outbound_run_lags_no_further_behind_than_a_short_one() -> None:
    def shortfall(n_chunks: int) -> int:
        resampler = OutboundResampler()
        pcm = tts_tone(n_chunks, 440.0)
        push_chunks(resampler, pcm)
        return WEBRTC_SAMPLES_PER_TTS_SAMPLE * len(pcm) - resampler.available()

    assert shortfall(200) <= shortfall(20)


def test_flush_leaves_the_next_pull_silent() -> None:
    resampler = OutboundResampler()
    tone = tts_tone(10, 440.0)

    push_chunks(resampler, tone)
    assert resampler.available() > 0

    resampler.flush()
    assert resampler.available() == 0
    assert not resampler.pull(WEBRTC_FRAME_SAMPLES).any()

    push_chunks(resampler, tone)
    assert resampler.available() > 0


def test_a_tone_survives_the_outbound_conversion() -> None:
    tone_hz = 440.0
    resampler = OutboundResampler()
    push_chunks(resampler, tts_tone(TTS_SAMPLE_RATE // TTS_SAMPLES_PER_WEBRTC_FRAME, tone_hz))
    signal = resampler.pull(resampler.available()).astype(np.float64)

    magnitudes = np.abs(np.fft.rfft(signal))
    magnitudes[0] = 0.0
    peak_hz = np.fft.rfftfreq(len(signal), 1 / WEBRTC_SAMPLE_RATE)[int(np.argmax(magnitudes))]

    assert abs(peak_hz - tone_hz) <= WEBRTC_SAMPLE_RATE / len(signal)
