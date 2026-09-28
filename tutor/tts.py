from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

from tutor.constants import TTS_SAMPLE_RATE

# median token count of the 20 listening replies; one fixed style by owner ruling 2026-09-27
FIXED_STYLE_TOKENS = 191
SENTENCE_GAP_S = 0.25


def to_int16(audio: np.ndarray) -> np.ndarray:
    clamped = np.clip(audio, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16)


class KokoroSynthesizer:
    def __init__(self, weights: Path, voices: Path, voice: str = "af_heart") -> None:
        self._kokoro = Kokoro(str(weights), str(voices))
        styles = self._kokoro.get_voice_style(voice)
        self._style = styles[FIXED_STYLE_TOKENS - 1 : FIXED_STYLE_TOKENS]
        self._gap = np.zeros(int(SENTENCE_GAP_S * TTS_SAMPLE_RATE), dtype=np.int16)

    def synthesize(self, text: str) -> np.ndarray:
        audio, sample_rate = self._kokoro.create(text, voice=self._style)
        if sample_rate != TTS_SAMPLE_RATE:
            raise ValueError(f"expected sample rate {TTS_SAMPLE_RATE}, got {sample_rate}")
        return np.concatenate([to_int16(audio), self._gap])
