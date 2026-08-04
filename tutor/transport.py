import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable

import numpy as np
from aiortc import RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from tutor.playout import PlayoutTrack
from tutor.resample import InboundResampler

logger = logging.getLogger(__name__)


class Connection:
    def __init__(self, pc: RTCPeerConnection) -> None:
        self._pc = pc
        self._playout = PlayoutTrack()
        self._inbound: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._channel: RTCDataChannel | None = None
        self._channel_ready = asyncio.Event()
        self._handlers: list[Callable[[dict[str, object]], None]] = []

        pc.addTrack(self._playout)

        @pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            self._reader = asyncio.create_task(self._read(track))

        @pc.on("datachannel")
        def _on_datachannel(channel: RTCDataChannel) -> None:
            self._channel = channel
            self._channel_ready.set()

            @channel.on("message")
            def _on_message(message: str) -> None:
                self._dispatch(message)

    async def _read(self, track: MediaStreamTrack) -> None:
        resampler = InboundResampler()
        try:
            while True:
                frame = await track.recv()
                for pcm in resampler.push(frame):
                    self._inbound.put_nowait(pcm)
        except MediaStreamError:
            logger.info("inbound_track_ended")
        finally:
            self._inbound.put_nowait(None)

    def _dispatch(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("datachannel_drop reason=not_json")
            return
        if not isinstance(payload, dict):
            logger.warning("datachannel_drop reason=not_an_object")
            return
        for handler in self._handlers:
            handler(payload)

    async def _stop_reader(self) -> None:
        if self._reader is None:
            return
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader

    async def frames(self) -> AsyncIterator[np.ndarray]:
        try:
            while True:
                pcm = await self._inbound.get()
                if pcm is None:
                    return
                yield pcm
        except asyncio.CancelledError:
            await self._stop_reader()
            raise

    async def play(self, pcm: np.ndarray) -> None:
        await self._playout.enqueue(pcm)

    def flush_playout(self) -> None:
        self._playout.flush()

    async def send_json(self, payload: dict[str, object]) -> None:
        await self._channel_ready.wait()
        self._channel.send(json.dumps(payload))

    def on_json(self, handler: Callable[[dict[str, object]], None]) -> None:
        self._handlers.append(handler)

    async def close(self) -> None:
        await self._stop_reader()
        await self._pc.close()


async def negotiate(offer: RTCSessionDescription) -> tuple[Connection, RTCSessionDescription]:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    connection = Connection(pc)
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(await pc.createAnswer())
    return connection, pc.localDescription
