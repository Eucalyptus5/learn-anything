from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

from tutor.constants import TTS_SAMPLE_RATE


def to_int16(audio: np.ndarray) -> np.ndarray:
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16)


class KokoroSynthesizer:
    def __init__(self, weights: Path, voices: Path, voice: str = "af_heart") -> None:
        self._kokoro = Kokoro(str(weights), str(voices))
        self._voice = voice

    def synthesize(self, text: str) -> np.ndarray:
        audio, sample_rate = self._kokoro.create(text, voice=self._voice)
        if sample_rate != TTS_SAMPLE_RATE:
            raise ValueError(f"expected sample rate {TTS_SAMPLE_RATE}, got {sample_rate}")
        return to_int16(audio)
