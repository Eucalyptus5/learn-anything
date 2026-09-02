"""Latency of the local WebRTC loopback: impulse round trip, and time to a connected peer.

The latency figure spans Opus encode, the jitter buffer, Opus decode and the inbound reframer,
and excludes browser capture, browser playout buffering, audio hardware and the outbound playout
queue. The reframer emits whole 512-sample frames, so the samples it still holds back when a
frame arrives stay uncorrected: on this machine that residual took one of two values, 64 or 128
samples, 4.0 or 8.0 ms, mean 6.1 ms over n=31, which the figure carries as a positive bias.
Both blocks discard one warm-up and report n, median, p95, min and max in whole milliseconds.
No STUN server is contacted, so nothing leaves the machine.

Browser tab RSS is measured by hand instead: start `uv run tutor`, open the
page in Chrome, connect, hold 60 s, read the tab's memory footprint from Chrome's Task Manager,
n=5 with a fresh tab each time.
"""

import argparse
import asyncio
import fractions
import statistics
import sys
import time
from pathlib import Path

import av
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack, MediaStreamTrack

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tutor.constants import SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.resample import InboundResampler
from tutor.transport import Connection, negotiate

DETECT_THRESHOLD = 8000
BURST_SAMPLES = WEBRTC_SAMPLE_RATE // 1000
BURST_AMPLITUDE = 32767
FRAMES_BETWEEN_BURSTS = 12
REFRACTORY_SAMPLES = SAMPLE_RATE // 10
DETECT_TIMEOUT = 5.0
CONNECT_TIMEOUT = 10.0


def report(label: str, values: list[float]) -> None:
    ms = sorted(round(v * 1000) for v in values)
    p95 = ms[max(0, int(len(ms) * 0.95) - 1)]
    print(
        f"  {label:32s} n={len(ms):4d} median={int(statistics.median(ms)):5d}ms "
        f"p95={p95:5d}ms min={ms[0]:5d}ms max={ms[-1]:5d}ms"
    )


class ImpulseTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, stamps: list[float]) -> None:
        super().__init__()
        self._stamps = stamps
        self._start = 0.0
        self._timestamp = 0
        self._index = 0
        self._started = False

    async def recv(self) -> av.AudioFrame:
        if self._started:
            self._timestamp += WEBRTC_FRAME_SAMPLES
            await asyncio.sleep(
                self._start + self._timestamp / WEBRTC_SAMPLE_RATE - time.perf_counter()
            )
        else:
            self._start = time.perf_counter()
            self._started = True

        pcm = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)
        if self._index % FRAMES_BETWEEN_BURSTS == 0:
            pcm[:BURST_SAMPLES] = BURST_AMPLITUDE
            self._stamps.append(time.perf_counter())
        self._index += 1

        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        return frame


async def detect(track: MediaStreamTrack, arrivals: asyncio.Queue[float]) -> None:
    resampler = InboundResampler()
    emitted = 0
    last = -REFRACTORY_SAMPLES
    while True:
        frame = await track.recv()
        arrived = time.perf_counter()
        outputs = resampler.push(frame)
        if not outputs:
            continue
        pcm = np.concatenate(outputs)
        base = emitted
        emitted += len(pcm)
        hits = np.flatnonzero(np.abs(pcm.astype(np.int32)) >= DETECT_THRESHOLD)
        if not hits.size:
            continue
        index = int(hits[0])
        if base + index - last < REFRACTORY_SAMPLES:
            continue
        last = base + index
        arrivals.put_nowait(arrived - (len(pcm) - index) / SAMPLE_RATE)


async def bench_latency(samples: int) -> None:
    print("\n--- loopback audio latency ---")
    print(
        f"  burst {BURST_SAMPLES} samples at {WEBRTC_SAMPLE_RATE} Hz every "
        f"{FRAMES_BETWEEN_BURSTS} frames, threshold {DETECT_THRESHOLD}"
    )

    stamps: list[float] = []
    arrivals: asyncio.Queue[float] = asyncio.Queue()
    reader: asyncio.Task[None] | None = None
    sender = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    receiver = RTCPeerConnection(RTCConfiguration(iceServers=[]))

    @receiver.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        nonlocal reader
        reader = asyncio.create_task(detect(track, arrivals))

    try:
        sender.addTrack(ImpulseTrack(stamps))
        await sender.setLocalDescription(await sender.createOffer())
        await receiver.setRemoteDescription(sender.localDescription)
        await receiver.setLocalDescription(await receiver.createAnswer())
        await sender.setRemoteDescription(receiver.localDescription)

        latencies: list[float] = []
        for index in range(samples + 1):
            arrival = await asyncio.wait_for(arrivals.get(), DETECT_TIMEOUT)
            if index >= 1:
                latencies.append(arrival - stamps[index])
            print(f"  {index + 1}/{samples + 1}", end="\r", file=sys.stderr)
        print(" " * 20, end="\r", file=sys.stderr)
    finally:
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        await sender.close()
        await receiver.close()

    print(f"  bursts emitted {len(stamps)}")
    report("impulse send to reframed", latencies)


async def connect_once() -> float:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    connection: Connection | None = None
    connected = asyncio.Event()

    @pc.on("connectionstatechange")
    def _on_state() -> None:
        if pc.connectionState == "connected":
            connected.set()

    try:
        pc.createDataChannel("tutor")
        pc.addTrack(AudioStreamTrack())
        start = time.perf_counter()
        await pc.setLocalDescription(await pc.createOffer())
        connection, answer = await negotiate(pc.localDescription)
        await pc.setRemoteDescription(answer)
        await asyncio.wait_for(connected.wait(), CONNECT_TIMEOUT)
        return time.perf_counter() - start
    finally:
        await pc.close()
        if connection is not None:
            await connection.close()


async def bench_connect(samples: int) -> None:
    print("\n--- loopback connect time ---")

    values: list[float] = []
    for index in range(samples + 1):
        values.append(await connect_once())
        print(f"  {index + 1}/{samples + 1}", end="\r", file=sys.stderr)
    print(" " * 20, end="\r", file=sys.stderr)

    print(f"  warm-up sample (discarded)       {int(values[0] * 1000)}ms")
    report("offer to connectionState connected", values[1:])


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="latency,connect")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    blocks = [x.strip() for x in args.blocks.split(",")]
    if "latency" in blocks:
        await bench_latency(args.samples)
    if "connect" in blocks:
        await bench_connect(args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
