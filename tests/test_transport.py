import asyncio
import fractions
import json
from collections import deque
from collections.abc import Callable

import av
import numpy as np
import pytest
from aiortc import RTCConfiguration, RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError, MediaStreamTrack

from tutor.constants import (
    FRAME_SAMPLES,
    TTS_SAMPLE_RATE,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
)
from tutor.transport import Connection, negotiate

TTS_CHUNK_SAMPLES = TTS_SAMPLE_RATE * WEBRTC_FRAME_SAMPLES // WEBRTC_SAMPLE_RATE
TONE_HZ = 440
TONE_AMPLITUDE = 12000
LOUD = 1000
LOOPBACK_TIMEOUT_S = 20.0
PAYLOAD = {"kind": "diagram", "html": "<p>x</p>"}


def tone(count: int, start: int, rate: int) -> np.ndarray:
    phase = 2 * np.pi * TONE_HZ * (np.arange(count) + start) / rate
    return (TONE_AMPLITUDE * np.sin(phase)).astype(np.int16)


def webrtc_frame(samples: np.ndarray, pts: int) -> av.AudioFrame:
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = WEBRTC_SAMPLE_RATE
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
    return frame


def tone_frames(count: int) -> list[av.AudioFrame]:
    starts = [i * WEBRTC_FRAME_SAMPLES for i in range(count)]
    return [
        webrtc_frame(tone(WEBRTC_FRAME_SAMPLES, start, WEBRTC_SAMPLE_RATE), start)
        for start in starts
    ]


class FiniteTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, frames: list[av.AudioFrame]) -> None:
        super().__init__()
        self._frames = deque(frames)

    async def recv(self) -> av.AudioFrame:
        if not self._frames:
            raise MediaStreamError
        return self._frames.popleft()


class StallingTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, frames: list[av.AudioFrame], log: list[str]) -> None:
        super().__init__()
        self._frames = deque(frames)
        self._log = log

    async def recv(self) -> av.AudioFrame:
        if self._frames:
            return self._frames.popleft()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self._log.append("reader")
            raise


class VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    async def recv(self) -> av.VideoFrame:
        self._log.append("video")
        raise AssertionError("a video track must never be read")


class ToneTrack(AudioStreamTrack):
    async def recv(self) -> av.AudioFrame:
        frame = await super().recv()
        frame.planes[0].update(tone(frame.samples, frame.pts, frame.sample_rate).tobytes())
        return frame


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.deliver: Callable[[str], None]

    def on(self, event: str) -> Callable[[Callable[[str], None]], Callable[[str], None]]:
        def register(handler: Callable[[str], None]) -> Callable[[str], None]:
            self.deliver = handler
            return handler

        return register

    def send(self, message: str) -> None:
        self.sent.append(message)


def local_peer() -> RTCPeerConnection:
    return RTCPeerConnection(RTCConfiguration(iceServers=[]))


async def loud_array(queue: asyncio.Queue[np.ndarray]) -> np.ndarray:
    while True:
        array = await queue.get()
        if np.abs(array).max() > LOUD:
            return array


async def test_frames_yields_vad_frames_until_the_track_ends() -> None:
    pc = local_peer()
    connection = Connection(pc)
    pc.emit("track", FiniteTrack(tone_frames(20)))

    frames = [array async for array in connection.frames()]

    assert frames
    for array in frames:
        assert array.dtype == np.int16
        assert array.ndim == 1
        assert len(array) == FRAME_SAMPLES
    assert np.abs(frames[-1]).max() > LOUD
    await connection.close()


async def test_a_video_track_is_ignored_and_leaves_the_audio_reader_alone() -> None:
    pc = local_peer()
    connection = Connection(pc)
    log: list[str] = []
    pc.emit("track", FiniteTrack(tone_frames(20)))
    pc.emit("track", VideoTrack(log))

    frames = [array async for array in connection.frames()]

    await connection.close()
    assert log == []
    assert len(frames) > 0


async def test_a_malformed_message_is_dropped_and_the_next_one_still_arrives() -> None:
    pc = local_peer()
    connection = Connection(pc)
    received: list[dict[str, object]] = []
    connection.on_json(received.append)
    channel = FakeChannel()
    pc.emit("datachannel", channel)

    channel.deliver("{not json")
    channel.deliver("[1, 2]")
    channel.deliver(json.dumps(PAYLOAD))

    assert received == [PAYLOAD]
    await connection.close()


async def test_send_json_waits_for_the_channel_to_arrive() -> None:
    pc = local_peer()
    connection = Connection(pc)
    channel = FakeChannel()
    sending = asyncio.create_task(connection.send_json(PAYLOAD))

    pc.emit("datachannel", channel)
    await sending

    assert [json.loads(message) for message in channel.sent] == [PAYLOAD]
    await connection.close()


async def test_cancelling_the_frames_consumer_unwinds_the_reader_first() -> None:
    pc = local_peer()
    connection = Connection(pc)
    log: list[str] = []
    pc.emit("track", StallingTrack(tone_frames(4), log))
    delivered = asyncio.Event()

    async def consume() -> None:
        try:
            async for _ in connection.frames():
                delivered.set()
        except asyncio.CancelledError:
            log.append("consumer")
            raise

    consumer = asyncio.create_task(consume())
    await delivered.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert log == ["reader", "consumer"]
    assert connection._reader.done()
    await connection.close()


async def test_loopback_carries_json_and_audio_both_ways() -> None:
    offerer = local_peer()
    channel = offerer.createDataChannel("tutor")
    offerer.addTrack(ToneTrack())

    connected = asyncio.Event()
    opened = asyncio.Event()
    from_tutor: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    heard: asyncio.Queue[np.ndarray] = asyncio.Queue()
    drainers: list[asyncio.Task[None]] = []

    @offerer.on("connectionstatechange")
    def _on_state() -> None:
        if offerer.connectionState == "connected":
            connected.set()

    @channel.on("open")
    def _on_open() -> None:
        opened.set()

    @channel.on("message")
    def _on_message(message: str) -> None:
        from_tutor.put_nowait(json.loads(message))

    async def drain(track: MediaStreamTrack) -> None:
        while True:
            frame = await track.recv()
            heard.put_nowait(frame.to_ndarray().reshape(-1))

    @offerer.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        drainers.append(asyncio.create_task(drain(track)))

    await offerer.setLocalDescription(await offerer.createOffer())
    connection, answer = await negotiate(offerer.localDescription)

    assert answer.type == "answer"
    assert "a=candidate" in answer.sdp

    await offerer.setRemoteDescription(answer)
    await asyncio.wait_for(connected.wait(), LOOPBACK_TIMEOUT_S)
    await asyncio.wait_for(opened.wait(), LOOPBACK_TIMEOUT_S)

    to_tutor: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    connection.on_json(to_tutor.put_nowait)
    await connection.send_json(PAYLOAD)
    channel.send(json.dumps(PAYLOAD))

    assert await asyncio.wait_for(from_tutor.get(), LOOPBACK_TIMEOUT_S) == PAYLOAD
    assert await asyncio.wait_for(to_tutor.get(), LOOPBACK_TIMEOUT_S) == PAYLOAD

    inbound: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def speak() -> None:
        start = 0
        while True:
            await connection.play(tone(TTS_CHUNK_SAMPLES, start, TTS_SAMPLE_RATE))
            start += TTS_CHUNK_SAMPLES

    async def listen() -> None:
        async for array in connection.frames():
            inbound.put_nowait(array)

    speaker = asyncio.create_task(speak())
    listener = asyncio.create_task(listen())

    from_browser = await asyncio.wait_for(loud_array(inbound), LOOPBACK_TIMEOUT_S)
    from_tutor_audio = await asyncio.wait_for(loud_array(heard), LOOPBACK_TIMEOUT_S)

    assert from_browser.dtype == np.int16
    assert len(from_browser) == FRAME_SAMPLES
    assert from_tutor_audio.dtype == np.int16

    speaker.cancel()
    listener.cancel()
    for task in drainers:
        task.cancel()
    await asyncio.gather(speaker, listener, *drainers, return_exceptions=True)
    await connection.close()
    await offerer.close()


async def test_a_peer_that_goes_away_tears_the_connection_down() -> None:
    offerer = local_peer()
    offerer.createDataChannel("tutor")
    offerer.addTrack(ToneTrack())

    connected = asyncio.Event()

    @offerer.on("connectionstatechange")
    def _on_state() -> None:
        if offerer.connectionState == "connected":
            connected.set()

    await offerer.setLocalDescription(await offerer.createOffer())
    connection, answer = await negotiate(offerer.localDescription)
    gone = asyncio.Event()
    connection.on_close(gone.set)

    await offerer.setRemoteDescription(answer)
    await asyncio.wait_for(connected.wait(), LOOPBACK_TIMEOUT_S)
    await offerer.close()
    await asyncio.wait_for(gone.wait(), LOOPBACK_TIMEOUT_S)

    assert connection.closed
