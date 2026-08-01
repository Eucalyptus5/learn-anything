"""Latency and memory for the local audio path: Silero VAD, Faster-Whisper, Kokoro.

Every block discards one warm-up and reports n, median, and p95 in whole milliseconds,
with process RSS and system swap sampled before and after. No microphone is opened; the
speech fixture is synthesized once with the macOS `say` command.
"""

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
FIXTURE = MODELS / "bench" / "utterance.wav"

SAMPLE_RATE = 16000
VAD_FRAME = 512
VAD_CONTEXT = 64

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
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def bench_silero(audio: np.ndarray) -> None:
    import onnxruntime as ort

    b = begin("silero vad (onnxruntime, no torch)")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    load = time.perf_counter()
    session = ort.InferenceSession(
        str(MODELS / "silero" / "silero_vad.onnx"),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    print(f"  model load {int((time.perf_counter() - load) * 1000)}ms")

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    frames = [audio[i : i + VAD_FRAME] for i in range(0, len(audio) - VAD_FRAME + 1, VAD_FRAME)]

    timings: list[float] = []
    for index, frame in enumerate(frames):
        x = np.concatenate([context, frame[None, :]], axis=1).astype(np.float32)
        t = time.perf_counter()
        _, state = session.run(None, {"input": x, "state": state, "sr": sr})
        elapsed = time.perf_counter() - t
        context = x[:, -VAD_CONTEXT:]
        if index >= 1:
            timings.append(elapsed)

    frame_ms = VAD_FRAME / SAMPLE_RATE * 1000
    print(f"  frame size {VAD_FRAME} samples ({frame_ms:.0f} ms of audio at {SAMPLE_RATE} Hz)")
    report("per-frame inference", timings)
    end(b)


def bench_whisper(audio: np.ndarray, samples: int) -> None:
    from faster_whisper import WhisperModel

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
        segments, _ = model.transcribe(audio, language="en", beam_size=5)
        text = " ".join(s.text for s in segments)
        elapsed = time.perf_counter() - t
        if index >= 1:
            timings.append(elapsed)
        print(f"  {index + 1}/{samples + 1}", end="\r", file=sys.stderr)
    print(" " * 20, end="\r", file=sys.stderr)

    report("full utterance transcription", timings)
    rtf = statistics.median(timings) / duration
    print(f"  real-time factor {rtf:.3f} ({1 / rtf:.1f}x faster than real time)")
    print(f"  transcript: {text.strip()[:110]}")
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
            rate = 24000
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
