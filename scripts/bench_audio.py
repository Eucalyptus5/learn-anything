"""Latency and memory for the local audio path: Silero VAD, Faster-Whisper, Kokoro, and the
assembled input path driven in real time.

Every block discards one warm-up and reports n, median, and p95 in whole milliseconds, or
whole microseconds for the per-frame figures, with process RSS and system swap sampled before
and after. No microphone is opened; the speech fixture is synthesized once with the macOS `say`
command.
"""

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tutor.constants import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, TTS_SAMPLE_RATE

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from tutor.stt import Transcriber
    from tutor.vad import SileroVad

MODELS = REPO / "models"
FIXTURE = MODELS / "bench" / "utterance.wav"

INPUT_PATH_TAIL_FRAMES = 20  # 640 ms of silence, enough to close a turn on a 500 ms window

FRAGMENT_MS = [256, 384, 512, 640, 768, 1024, 2000, 4000, 8000]
FRAGMENT_OFFSETS = 5
FRAGMENT_RUNS = 5
# end-of-turn budget: 690 ms median, 1230 ms p95, from speech stopping to the final transcript
PARTIAL_FLOOR_MEDIAN_MS = 690
PARTIAL_FLOOR_MAX_MS = 1230

FIXTURE_TEXT = (
    "The connection pool acquires a semaphore before it hands out a socket, and the worker "
    "releases it in a finally block so a panic during a flush cannot leak a permit."
)
TTS_TEXTS = [
    ("lead-in", "Right, let me look."),
    ("sentence", "The session actor owns the audio buffers."),
    (
        "full turn",
        (
            "The session actor owns the audio buffers, so when the detector fires mid sentence "
            "it cancels the synthesis task first and only then drains the output queue."
        ),
    ),
]


def rss_mb(pid: int | None = None) -> float:
    pid = pid or os.getpid()
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return int(out.stdout.strip() or 0) / 1024


def swap_used_mb() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    )
    parts = out.stdout.replace("=", " ").split()
    return float(parts[parts.index("used") + 1].rstrip("M"))


def heavy_processes(limit: int = 5) -> list[str]:
    out = subprocess.run(["ps", "-Ao", "rss=,comm="], capture_output=True, text=True, check=False)
    rows = []
    for line in out.stdout.splitlines():
        rss, _, comm = line.strip().partition(" ")
        if rss.isdigit():
            rows.append((int(rss), comm.strip().split("/")[-1]))
    rows.sort(reverse=True)
    return [f"{r / 1024:.0f} MB {c}" for r, c in rows[:limit]]


@dataclass
class Block:
    name: str
    rss_before: float
    swap_before: float


def begin(name: str) -> Block:
    b = Block(name, rss_mb(), swap_used_mb())
    print(f"\n--- {name} ---")
    print(f"  rss before {b.rss_before:.0f} MB   swap used before {b.swap_before:.0f} MB")
    return b


def end(b: Block) -> None:
    print(
        f"  rss after  {rss_mb():.0f} MB (delta {rss_mb() - b.rss_before:+.0f} MB)"
        f"   swap used after {swap_used_mb():.0f} MB"
    )


def report(label: str, values: list[float]) -> None:
    ms = sorted(round(v * 1000) for v in values)
    p95 = ms[max(0, int(len(ms) * 0.95) - 1)]
    print(
        f"  {label:32s} n={len(ms):4d} median={int(statistics.median(ms)):5d}ms "
        f"p95={p95:5d}ms min={ms[0]:5d}ms max={ms[-1]:5d}ms"
    )


def report_us(label: str, values: list[float]) -> None:
    us = sorted(round(v * 1_000_000) for v in values)
    p95 = us[max(0, int(len(us) * 0.95) - 1)]
    print(
        f"  {label:32s} n={len(us):4d} median={int(statistics.median(us)):5d}us "
        f"p95={p95:5d}us min={us[0]:5d}us max={us[-1]:5d}us"
    )


def ensure_fixture() -> np.ndarray:
    if not FIXTURE.exists():
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        aiff = FIXTURE.with_suffix(".aiff")
        subprocess.run(["say", "-o", str(aiff), FIXTURE_TEXT], check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(aiff),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                str(FIXTURE),
            ],
            check=True,
        )
        aiff.unlink()
    with wave.open(str(FIXTURE)) as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16)


def to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.int16:
        raise ValueError(f"expected int16 audio, got {audio.dtype}")
    return audio.astype(np.float32) / 32768.0


def bench_silero(audio: np.ndarray) -> None:
    from tutor.vad import SileroVad

    b = begin("silero vad (onnxruntime, no torch)")
    load = time.perf_counter()
    vad = SileroVad(MODELS / "silero" / "silero_vad.onnx")
    print(f"  model load {int((time.perf_counter() - load) * 1000)}ms")

    frames = [
        audio[i : i + FRAME_SAMPLES]
        for i in range(0, len(audio) - FRAME_SAMPLES + 1, FRAME_SAMPLES)
    ]

    timings: list[float] = []
    for index, frame in enumerate(frames):
        t = time.perf_counter()
        vad(frame)
        elapsed = time.perf_counter() - t
        if index >= 1:
            timings.append(elapsed)

    print(f"  frame size {FRAME_SAMPLES} samples ({FRAME_MS} ms of audio at {SAMPLE_RATE} Hz)")
    report_us("per-frame inference", timings)
    end(b)


def whisper_transcribe(model: "WhisperModel", audio: np.ndarray) -> str:
    if audio.dtype != np.float32:
        raise ValueError(f"expected float32 audio, got {audio.dtype}")
    if np.max(np.abs(audio)) > 1.0:
        raise ValueError("expected audio in [-1, 1]")
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    return " ".join(s.text for s in segments).strip()


def select_partial_floor_ms(curve: dict[int, list[float]]) -> int | None:
    for length_ms in sorted(curve):
        timings = curve[length_ms]
        if (
            statistics.median(timings) * 1000 < PARTIAL_FLOOR_MEDIAN_MS
            and max(timings) * 1000 < PARTIAL_FLOOR_MAX_MS
        ):
            return length_ms
    return None


def bench_whisper(audio: np.ndarray, samples: int) -> None:
    from faster_whisper import WhisperModel

    audio = to_float32(audio)
    b = begin("faster-whisper base.en (ctranslate2, int8, cpu)")
    load = time.perf_counter()
    model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS / "whisper"),
    )
    print(f"  model load {int((time.perf_counter() - load) * 1000)}ms")
    duration = len(audio) / SAMPLE_RATE
    print(f"  utterance {duration:.2f}s of speech")

    timings: list[float] = []
    text = ""
    for index in range(samples + 1):
        t = time.perf_counter()
        text = whisper_transcribe(model, audio)
        elapsed = time.perf_counter() - t
        if index >= 1:
            timings.append(elapsed)
        print(f"  {index + 1}/{samples + 1}", end="\r", file=sys.stderr)
    print(" " * 20, end="\r", file=sys.stderr)

    report("full utterance transcription", timings)
    rtf = statistics.median(timings) / duration
    print(f"  real-time factor {rtf:.3f} ({1 / rtf:.1f}x faster than real time)")
    print(f"  transcript: {text[:110]}")
    end(b)


def bench_whisper_fragments(audio: np.ndarray) -> None:
    from faster_whisper import WhisperModel

    audio = to_float32(audio)
    b = begin("faster-whisper base.en fragment cost curve")
    load = time.perf_counter()
    model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        download_root=str(MODELS / "whisper"),
    )
    print(f"  model load {int((time.perf_counter() - load) * 1000)}ms")
    print(f"  fixture {len(audio) / SAMPLE_RATE:.4f}s ({len(audio)} samples at {SAMPLE_RATE} Hz)")
    print(f"  {FRAGMENT_OFFSETS} offsets x {FRAGMENT_RUNS} timed runs per length")
    whisper_transcribe(model, audio)
    assert max(FRAGMENT_MS) * SAMPLE_RATE // 1000 <= len(audio)

    total = len(FRAGMENT_MS) * FRAGMENT_OFFSETS * FRAGMENT_RUNS
    done = 0
    curve: dict[int, list[float]] = {}
    for length_ms in FRAGMENT_MS:
        width = length_ms * SAMPLE_RATE // 1000
        slack = len(audio) - width
        timings: list[float] = []
        for step in range(FRAGMENT_OFFSETS):
            start = round(step * slack / (FRAGMENT_OFFSETS - 1))
            fragment = audio[start : start + width]
            for _ in range(FRAGMENT_RUNS):
                t = time.perf_counter()
                whisper_transcribe(model, fragment)
                timings.append(time.perf_counter() - t)
                done += 1
                print(f"  {done}/{total}", end="\r", file=sys.stderr)
        print(" " * 20, end="\r", file=sys.stderr)
        curve[length_ms] = timings
        print(f"  [{length_ms} ms] offsets spread over {slack} samples of slack")
        report(f"{length_ms} ms ({width} samples)", timings)

    print(f"  thresholds median<{PARTIAL_FLOOR_MEDIAN_MS}ms max<{PARTIAL_FLOOR_MAX_MS}ms")
    floor = select_partial_floor_ms(curve)
    if floor is None:
        print("  no fragment length is under both thresholds")
    else:
        print(f"  PARTIAL_FLOOR_MS {floor}")
    end(b)


async def bench_kokoro(samples: int, weights: str) -> None:
    from kokoro_onnx import Kokoro

    b = begin(f"kokoro-82m (onnxruntime, {weights}, cpu)")
    load = time.perf_counter()
    kokoro = Kokoro(
        str(MODELS / "kokoro" / f"kokoro-v1.0.{weights}.onnx"),
        str(MODELS / "kokoro" / "voices-v1.0.bin"),
    )
    print(f"  model load {int((time.perf_counter() - load) * 1000)}ms")

    for label, text in TTS_TEXTS:
        first: list[float] = []
        full: list[float] = []
        spoken = 0.0
        chunks = 0
        for index in range(samples + 1):
            t = time.perf_counter()
            first_at = None
            total_samples = 0
            chunks = 0
            rate = TTS_SAMPLE_RATE
            async for chunk, rate in kokoro.create_stream(text, voice="af_heart", lang="en-us"):
                if first_at is None:
                    first_at = time.perf_counter() - t
                total_samples += len(chunk)
                chunks += 1
            elapsed = time.perf_counter() - t
            spoken = total_samples / rate
            if index >= 1:
                first.append(first_at)
                full.append(elapsed)
            print(f"  {label} {index + 1}/{samples + 1}", end="\r", file=sys.stderr)
        print(" " * 40, end="\r", file=sys.stderr)

        words = len(text.split())
        print(f"  [{label}] {words} words -> {spoken:.2f}s of audio in {chunks} chunk(s)")
        report(f"[{label}] time to first audio", first)
        report(f"[{label}] full synthesis", full)
        rtf = statistics.median(full) / spoken
        print(f"  [{label}] real-time factor {rtf:.3f} ({1 / rtf:.1f}x faster than real time)")
    end(b)


class PacedSource:
    def __init__(self, frames: list[np.ndarray]) -> None:
        from tutor.transport import INBOUND_CAPACITY

        self._frames = frames
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=INBOUND_CAPACITY)
        self.hops: list[float] = []
        self.dropped_frames = 0

    # the real reader runs on the peer's clock, so a stalled consumer loses the oldest frames
    def _offer(self, item: np.ndarray | None) -> None:
        while self._queue.full():
            self._queue.get_nowait()
            self.dropped_frames += 1
        self._queue.put_nowait(item)

    async def produce(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        for frame in self._frames:
            self._offer(frame)
            deadline += FRAME_MS / 1000
            await asyncio.sleep(max(0.0, deadline - loop.time()))
        self._offer(None)

    async def frames(self) -> AsyncGenerator[np.ndarray, None]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            handed = time.perf_counter()
            yield frame
            self.hops.append(time.perf_counter() - handed)


class TimedVad:
    def __init__(self, vad: "SileroVad") -> None:
        self._vad = vad
        self.probabilities: list[tuple[float, float]] = []

    def __call__(self, frame: np.ndarray) -> float:
        probability = self._vad(frame)
        self.probabilities.append((probability, time.perf_counter()))
        return probability

    def reset(self) -> None:
        self._vad.reset()


class TimedTranscriber:
    def __init__(self, transcriber: "Transcriber") -> None:
        self._transcriber = transcriber
        self.spans: list[tuple[float, float]] = []

    def transcribe(self, audio: np.ndarray) -> str:
        started = time.perf_counter()
        text = self._transcriber.transcribe(audio)
        self.spans.append((started, time.perf_counter()))
        return text


@dataclass
class TurnTiming:
    silence_at: float
    end_at: float
    silence_starts: int

    @property
    def wait(self) -> float:
        return self.end_at - self.silence_at


def replay_turns(
    probabilities: list[tuple[float, float]], end_stamps: list[float]
) -> list[TurnTiming]:
    from tutor.endpointer import Endpointer, EndpointEvent
    from tutor.input_path import START_FRAMES, START_THRESHOLD

    endpointer = Endpointer(start_frames=START_FRAMES, start_threshold=START_THRESHOLD)
    replayed: list[tuple[float, int]] = []
    silence_at = 0.0
    starts = 0
    for probability, stamp in probabilities:
        event = endpointer.push(probability)
        if event is EndpointEvent.SILENCE_START:
            silence_at = stamp
            starts += 1
        elif event is EndpointEvent.END_OF_TURN:
            replayed.append((silence_at, starts))
            starts = 0
    return [TurnTiming(at, end, n) for (at, n), end in zip(replayed, end_stamps)]


async def bench_input_path(audio: np.ndarray, samples: int) -> None:
    from tutor.endpointer import SILENCE_WINDOW_MS
    from tutor.input_path import EndOfTurn, InputPath, PartialTranscript
    from tutor.stt import FINAL_CPU_THREADS, PARTIAL_CPU_THREADS, Transcriber, load_whisper
    from tutor.vad import SileroVad

    b = begin("input path (silero + two whisper workers, real time)")
    load = time.perf_counter()
    vad = SileroVad(MODELS / "silero" / "silero_vad.onnx")
    print(f"  vad load {int((time.perf_counter() - load) * 1000)}ms")
    load = time.perf_counter()
    partial_model = load_whisper(MODELS / "whisper", cpu_threads=PARTIAL_CPU_THREADS)
    print(
        f"  partial whisper load {int((time.perf_counter() - load) * 1000)}ms "
        f"({PARTIAL_CPU_THREADS} cpu thread)"
    )
    load = time.perf_counter()
    final_model = load_whisper(MODELS / "whisper", cpu_threads=FINAL_CPU_THREADS)
    print(
        f"  final whisper load {int((time.perf_counter() - load) * 1000)}ms "
        f"({FINAL_CPU_THREADS} cpu threads)"
    )
    print(f"  rss with the vad and both whisper models loaded {rss_mb():.0f} MB")

    whole = len(audio) - len(audio) % FRAME_SAMPLES
    tail = np.zeros(INPUT_PATH_TAIL_FRAMES * FRAME_SAMPLES, dtype=np.int16)
    utterance = np.concatenate([audio[:whole], tail])
    stream = np.tile(utterance, samples + 1)
    frames = [stream[i : i + FRAME_SAMPLES] for i in range(0, len(stream), FRAME_SAMPLES)]
    per_utterance = len(utterance) // FRAME_SAMPLES
    print(
        f"  {samples + 1} utterances of {per_utterance} frames "
        f"({len(utterance) / SAMPLE_RATE:.2f}s each, {len(frames) * FRAME_MS / 1000:.1f}s of stream)"
    )

    timed_vad = TimedVad(vad)
    partial = TimedTranscriber(Transcriber(partial_model))
    final = TimedTranscriber(Transcriber(final_model))
    source = PacedSource(frames)
    path = InputPath(source, timed_vad, partial, final)
    producer = asyncio.create_task(source.produce())

    end_stamps: list[float] = []
    end_frames: list[int] = []
    partials: list[int] = []
    seen = 0
    async for event in path.events():
        if isinstance(event, PartialTranscript):
            seen += 1
        elif isinstance(event, EndOfTurn):
            end_stamps.append(time.perf_counter())
            end_frames.append(len(source.hops))
            partials.append(seen)
            seen = 0
            print(f"  {len(end_stamps)}/{samples + 1}", end="\r", file=sys.stderr)
    print(" " * 20, end="\r", file=sys.stderr)
    await producer
    await path.aclose()

    turns = replay_turns(timed_vad.probabilities, end_stamps)[1:]
    at_end_of_turn = set(end_frames)
    hops = [
        hop
        for index, hop in enumerate(source.hops)
        if index >= per_utterance and index not in at_end_of_turn
    ]
    starts = [t.silence_starts for t in turns]
    counts = partials[1:]
    decodes = [end - start for start, end in final.spans if start > end_stamps[0]]
    overruns = sum(1 for t in turns if t.wait * 1000 > SILENCE_WINDOW_MS)
    in_flight = sum(1 for t in turns if any(s <= t.silence_at <= e for s, e in partial.spans))

    report_us("hand-off to next request (loop)", hops)
    report("silence start to final transcript", [t.wait for t in turns])
    print(f"  {overruns} of {len(turns)} turns wait past the {SILENCE_WINDOW_MS} ms silence window")
    report("final transcriber run", decodes)
    print(f"  {len(decodes)} final decodes ran across {len(turns)} timed turns")
    print(
        f"  silence starts per utterance median {statistics.median(starts):.1f} max {max(starts)}"
    )
    print(f"  partials per utterance median {statistics.median(counts):.1f} max {max(counts)}")
    print(f"  {in_flight} of {len(turns)} timed turns had a partial in flight at the silence start")
    print(f"  dropped frames {source.dropped_frames}")
    end(b)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="silero,whisper,kokoro")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--kokoro-weights", default="fp16", choices=["fp16", "int8"])
    args = parser.parse_args()

    print(f"machine swap used {swap_used_mb():.0f} MB at start")
    print("heaviest processes at start:")
    for row in heavy_processes():
        print(f"  {row}")

    audio = ensure_fixture()
    blocks = [x.strip() for x in args.blocks.split(",")]
    if "silero" in blocks:
        bench_silero(audio)
    if "whisper" in blocks:
        bench_whisper(audio, args.samples)
    if "whisper-fragments" in blocks:
        bench_whisper_fragments(audio)
    if "input-path" in blocks:
        await bench_input_path(audio, args.samples)
    if "kokoro" in blocks:
        await bench_kokoro(args.samples, args.kokoro_weights)

    print(f"\ncombined process rss {rss_mb():.0f} MB")
    print(f"machine swap used {swap_used_mb():.0f} MB at end")
    print("heaviest processes at end:")
    for row in heavy_processes():
        print(f"  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
