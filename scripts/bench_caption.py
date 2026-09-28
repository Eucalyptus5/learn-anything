"""Caption lead against the playout: a fake synthesizer of known audio lengths drives the real
Speaker into a real PlayoutTrack pulled at wall-clock pace, and for every clause the harness
records when the clause left the source, when its audio was enqueued, the lead_ms the speaker
reported, and the time of the frame that first carried its samples. Two figures come out: the
pull-to-first-sound lead, which is how far a caption pushed at handoff runs ahead of the sound
when the source keeps one clause ahead of the playout, as a model streaming at speech pace does,
and the error between lead_ms and the measured enqueue-to-first-sound wait, which is how far a
caption held for lead_ms lands from the sound. No model, no browser, no network.
"""

import argparse
import asyncio
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.playout import PlayoutTrack
from tutor.speech import Chunk, Speaker

SAMPLES = 30
CLAUSE_MS = (1200, 800, 1600, 2400, 900)
SYNTH_RTF = 0.25
FRAME_S = WEBRTC_FRAME_SAMPLES / WEBRTC_SAMPLE_RATE


def frame_index(samples_before: int) -> int:
    return samples_before * WEBRTC_SAMPLE_RATE // TTS_SAMPLE_RATE // WEBRTC_FRAME_SAMPLES


def ms(t0: float, t: float) -> int:
    return round((t - t0) * 1000)


class ClauseSample(BaseModel):
    pulled_ms: int
    enqueued_ms: int
    lead_ms: int
    first_sound_ms: int

    @property
    def handoff_lead_ms(self) -> int:
        return self.first_sound_ms - self.pulled_ms

    @property
    def lead_error_ms(self) -> int:
        return abs(self.lead_ms - (self.first_sound_ms - self.enqueued_ms))


class LengthsSynth:
    def __init__(self, lengths_ms: tuple[int, ...], rtf: float) -> None:
        self._lengths = lengths_ms
        self._rtf = rtf
        self.calls = 0

    def synthesize(self, text: str) -> np.ndarray:
        length_ms = self._lengths[self.calls % len(self._lengths)]
        self.calls += 1
        time.sleep(length_ms / 1000 * self._rtf)
        return np.zeros(length_ms * TTS_SAMPLE_RATE // 1000, dtype=np.int16)


class TrackTransport:
    def __init__(self, track: PlayoutTrack) -> None:
        self._track = track
        self.enqueued: list[tuple[float, int]] = []
        self.first = asyncio.Event()

    async def play(self, pcm: np.ndarray) -> None:
        await self._track.enqueue(pcm)
        self.enqueued.append((time.perf_counter(), len(pcm)))
        self.first.set()

    def flush_playout(self) -> None:
        self._track.flush()

    def playout_backlog_s(self) -> float:
        return self._track.backlog_s()


async def pull(track: PlayoutTrack, started: asyncio.Event, frames: list[float]) -> None:
    await started.wait()
    while True:
        await track.recv()
        frames.append(time.perf_counter())


async def clauses(
    n: int,
    pulled: list[float],
    backlog_s: Callable[[], float],
    pace: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[Chunk]:
    for k in range(n):
        while backlog_s() >= CLAUSE_MS[k % len(CLAUSE_MS)] / 1000:
            await pace(FRAME_S)
        pulled.append(time.perf_counter())
        yield Chunk(k + 1, f"clause {k}")


def attribute(
    t0: float,
    pulled: list[float],
    enqueued: list[tuple[float, int]],
    leads: list[int],
    frames: list[float],
) -> list[ClauseSample]:
    samples: list[ClauseSample] = []
    before = 0
    for k, ((at, length), lead) in enumerate(zip(enqueued, leads, strict=True)):
        samples.append(
            ClauseSample(
                pulled_ms=ms(t0, pulled[k]),
                enqueued_ms=ms(t0, at),
                lead_ms=lead,
                first_sound_ms=ms(t0, frames[frame_index(before)]),
            )
        )
        before += length
    return samples


async def run(n: int) -> list[ClauseSample]:
    track = PlayoutTrack()
    transport = TrackTransport(track)
    speaker = Speaker(LengthsSynth(CLAUSE_MS, SYNTH_RTF), transport)
    pulled: list[float] = []
    leads: list[int] = []
    frames: list[float] = []

    async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
        leads.append(lead_ms)

    puller = asyncio.create_task(pull(track, transport.first, frames))
    t0 = time.perf_counter()
    try:
        await speaker.speak(clauses(n, pulled, track.backlog_s), on_play)
        before_last = sum(length for _, length in transport.enqueued[:-1])
        while len(frames) <= frame_index(before_last):
            await asyncio.sleep(FRAME_S)
    finally:
        puller.cancel()
        await asyncio.gather(puller, return_exceptions=True)
    return attribute(t0, pulled, transport.enqueued, leads, frames)


def report(samples: list[ClauseSample]) -> None:
    print(
        "pull to first sound: from the clause leaving the source, where a caption pushed at "
        "handoff fires, to the first 48 kHz frame carrying that clause's samples leaving the "
        "playout track; synthesis and the playout backlog both sit inside it. The source hands "
        "over a clause only once the backlog is under that clause's length, so this is the lead "
        "under a model streaming at speech pace; an unpaced source grows it with every clause."
    )
    print(
        "lead error: the absolute difference between the lead_ms the speaker reported when the "
        "clause was enqueued and the measured wait from that enqueue to the same frame; a caption "
        "held for lead_ms lands this far from the sound, before the browser's jitter buffer."
    )
    print(
        "attribution is by cumulative sample count, the resampler's output being time-aligned "
        "with its input, so it is exact only while the queue never runs dry; with synth_rtf under "
        "one and no clause more than four times longer than the one before it, synthesis always "
        "finishes before the previous clause has played out. Frame granularity 20 ms."
    )
    print(f"clauses={len(samples)}  clause_ms={CLAUSE_MS}  synth_rtf={SYNTH_RTF}")
    summarize("pull to first sound", [s.handoff_lead_ms for s in samples])
    summarize("lead error", [s.lead_error_ms for s in samples])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    return parser


async def main() -> int:
    args = build_parser().parse_args()
    report(await run(args.samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
