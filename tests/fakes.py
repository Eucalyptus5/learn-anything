import asyncio
import fractions
import threading
from pathlib import Path

import av
import numpy as np
import pytest
from aiortc import RTCConfiguration, RTCPeerConnection

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE


class Spawned:
    def __init__(self) -> None:
        self.argv: list[str] = []
        self.kwargs: dict[str, object] = {}
        self.proc: asyncio.subprocess.Process | None = None
        self.calls = 0
        self.started = asyncio.Event()


def record_spawns(monkeypatch: pytest.MonkeyPatch, replacement: list[str] | None) -> Spawned:
    holder = Spawned()
    real = asyncio.create_subprocess_exec

    async def passthrough(*argv: str, **kwargs: object) -> asyncio.subprocess.Process:
        holder.argv = list(argv)
        holder.kwargs = dict(kwargs)
        holder.calls += 1
        proc = await real(*(replacement or argv), **kwargs)
        holder.proc = proc
        holder.started.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", passthrough)
    return holder


SPLIT_SEPARATORS = ["\x0b", "\x1c", "\x85", "\u2028"]


def forged_names_tree(root: Path) -> Path:
    (root / ".rgignore").write_text("/private_notes/\n")
    (root / "private_notes").mkdir()
    (root / "private_notes" / "secret.txt").write_text("CANARY placeholder row\n")
    (root / "visible.py").write_text("CANARY ordinary row\n")
    for separator in SPLIT_SEPARATORS:
        carrier = root / f"x{separator}." / "private_notes"
        carrier.mkdir(parents=True)
        (carrier / "secret.txt").write_text("CANARY decoy row\n")
    return root


def match_record(path: str, number: int, text: str, submatches: list[dict]) -> dict:
    return {
        "type": "match",
        "data": {
            "path": {"text": path},
            "lines": {"text": text},
            "line_number": number,
            "absolute_offset": 0,
            "submatches": submatches,
        },
    }


class FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.threads: list[int] = []

    def synthesize(self, text: str) -> np.ndarray:
        self.calls.append(text)
        self.threads.append(threading.get_ident())
        return np.zeros(len(text), dtype=np.int16)


class FakeTransport:
    def __init__(self) -> None:
        self.played: list[np.ndarray] = []
        self.flushes = 0

    async def play(self, pcm: np.ndarray) -> None:
        self.played.append(pcm)

    def flush_playout(self) -> None:
        self.flushes += 1


def numbered_frame(value: int, pts: int) -> av.AudioFrame:
    samples = np.full(FRAME_SAMPLES, value, dtype=np.int16)
    frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = SAMPLE_RATE
    frame.pts = pts
    frame.time_base = fractions.Fraction(1, SAMPLE_RATE)
    return frame


def numbered_frames(count: int) -> list[av.AudioFrame]:
    return [numbered_frame(i + 1, i * FRAME_SAMPLES) for i in range(count)]


def local_peer() -> RTCPeerConnection:
    return RTCPeerConnection(RTCConfiguration(iceServers=[]))
