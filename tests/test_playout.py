import fractions

import numpy as np
import pytest

from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.playout import PlayoutTrack
from tutor.resample import OUTBOUND_STARTUP_DELAY_SAMPLES

TTS_CHUNK_SAMPLES = TTS_SAMPLE_RATE * WEBRTC_FRAME_SAMPLES // WEBRTC_SAMPLE_RATE
FRAME_SECONDS = WEBRTC_FRAME_SAMPLES / WEBRTC_SAMPLE_RATE
CHUNKS_PER_SIDE = 10
LOUD = 8000


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RecordingPace:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.waits: list[float] = []

    async def __call__(self, wait: float) -> None:
        self.waits.append(wait)
        self._clock.now += max(wait, 0.0)


def paced_track() -> tuple[PlayoutTrack, FakeClock, RecordingPace]:
    clock = FakeClock()
    pace = RecordingPace(clock)
    return PlayoutTrack(pace=pace, clock=clock), clock, pace


def chunk(value: int) -> np.ndarray:
    return np.full(TTS_CHUNK_SAMPLES, value, dtype=np.int16)


async def enqueue_chunks(track: PlayoutTrack, value: int, count: int) -> None:
    for _ in range(count):
        await track.enqueue(chunk(value))


async def drain(track: PlayoutTrack, n_frames: int) -> np.ndarray:
    frames = [await track.recv() for _ in range(n_frames)]
    return np.concatenate([frame.to_ndarray().reshape(-1) for frame in frames])


async def test_frames_have_the_webrtc_shape_and_a_rising_pts() -> None:
    track, _, _ = paced_track()

    for i in range(5):
        frame = await track.recv()
        assert frame.samples == WEBRTC_FRAME_SAMPLES
        assert frame.sample_rate == WEBRTC_SAMPLE_RATE
        assert frame.time_base == fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        assert frame.pts == i * WEBRTC_FRAME_SAMPLES


async def test_an_empty_queue_yields_silence() -> None:
    track, _, _ = paced_track()

    frame = await track.recv()
    assert not frame.to_ndarray().any()


async def test_enqueued_audio_comes_back_in_order() -> None:
    track, _, _ = paced_track()
    await enqueue_chunks(track, LOUD, CHUNKS_PER_SIDE)
    await enqueue_chunks(track, -LOUD, CHUNKS_PER_SIDE)

    signal = await drain(track, 2 * CHUNKS_PER_SIDE - 1)
    first = signal[:WEBRTC_FRAME_SAMPLES]
    last = signal[-WEBRTC_FRAME_SAMPLES:]

    assert first.mean() > LOUD * 0.75
    assert last.mean() < -LOUD * 0.75


async def test_flush_drops_what_was_queued_before_it() -> None:
    track, _, _ = paced_track()
    await enqueue_chunks(track, LOUD, CHUNKS_PER_SIDE)
    track.flush()
    await enqueue_chunks(track, -LOUD, CHUNKS_PER_SIDE)

    signal = await drain(track, CHUNKS_PER_SIDE - 1)

    assert signal.min() < -6000
    assert signal.max() < 2000


async def test_a_wait_shrinks_by_the_work_the_previous_frame_did() -> None:
    track, clock, pace = paced_track()

    await track.recv()
    assert pace.waits == []

    clock.now += FRAME_SECONDS / 4
    await track.recv()

    assert pace.waits == [pytest.approx(FRAME_SECONDS * 3 / 4)]


async def test_an_overrunning_frame_yields_a_non_positive_wait() -> None:
    track, clock, pace = paced_track()

    await track.recv()
    clock.now += FRAME_SECONDS * 1.5
    await track.recv()

    assert len(pace.waits) == 1
    assert pace.waits[0] <= 0


async def test_a_long_run_consumes_one_frame_of_time_per_frame_after_the_first() -> None:
    track, clock, pace = paced_track()
    n_frames = 300

    for _ in range(n_frames):
        await track.recv()

    assert clock.now == pytest.approx((n_frames - 1) * FRAME_SECONDS)
    assert len(pace.waits) == n_frames - 1
    for wait in pace.waits:
        assert wait == pytest.approx(FRAME_SECONDS)


async def test_the_backlog_is_zero_on_a_fresh_track() -> None:
    track, _, _ = paced_track()

    assert track.backlog_s() == 0.0


async def test_the_backlog_grows_by_each_enqueued_chunk() -> None:
    track, _, _ = paced_track()

    for n in range(1, 4):
        await track.enqueue(chunk(LOUD))
        assert track.backlog_s() == pytest.approx(n * TTS_CHUNK_SAMPLES / TTS_SAMPLE_RATE)


async def test_the_backlog_shrinks_by_one_frame_per_pull() -> None:
    track, _, _ = paced_track()
    await enqueue_chunks(track, LOUD, CHUNKS_PER_SIDE)
    queued = CHUNKS_PER_SIDE * TTS_CHUNK_SAMPLES / TTS_SAMPLE_RATE
    resampled = CHUNKS_PER_SIDE * WEBRTC_FRAME_SAMPLES
    whole = (
        (resampled - OUTBOUND_STARTUP_DELAY_SAMPLES) // WEBRTC_FRAME_SAMPLES * WEBRTC_FRAME_SAMPLES
    )

    await track.recv()
    after_first = track.backlog_s()

    assert after_first == pytest.approx((whole - WEBRTC_FRAME_SAMPLES) / WEBRTC_SAMPLE_RATE)
    assert after_first < queued
    for pulled in range(2, CHUNKS_PER_SIDE):
        await track.recv()
        assert track.backlog_s() == pytest.approx(after_first - (pulled - 1) * FRAME_SECONDS)


async def test_flush_zeroes_the_backlog() -> None:
    track, _, _ = paced_track()
    await enqueue_chunks(track, LOUD, CHUNKS_PER_SIDE)
    await track.recv()
    await track.enqueue(chunk(LOUD))
    assert track.backlog_s() > 0.0

    track.flush()

    assert track.backlog_s() == 0.0
    await track.recv()
    assert track.backlog_s() == 0.0
