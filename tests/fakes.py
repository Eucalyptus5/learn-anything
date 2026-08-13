import threading

import numpy as np


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
