from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

PARTIAL_CPU_THREADS = 1
FINAL_CPU_THREADS = 4


def load_whisper(model_dir: Path, cpu_threads: int = FINAL_CPU_THREADS) -> WhisperModel:
    return WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        download_root=str(model_dir),
        cpu_threads=cpu_threads,
    )


class Transcriber:
    def __init__(self, model: WhisperModel) -> None:
        self._model = model

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.dtype != np.int16:
            raise ValueError(f"expected int16 audio, got {audio.dtype}")

        segments, _ = self._model.transcribe(
            audio.astype(np.float32) / 32768.0, language="en", beam_size=5
        )
        return " ".join(s.text for s in segments).strip()
