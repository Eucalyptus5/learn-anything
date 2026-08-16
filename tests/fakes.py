import asyncio
import threading

import numpy as np
import pytest


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
