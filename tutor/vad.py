from pathlib import Path

import numpy as np
import onnxruntime as ort

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE

CONTEXT_SAMPLES = 64  # lookback silero expects in front of every frame at 16 kHz
STATE_SHAPE = (2, 1, 128)


class SileroVad:
    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def __call__(self, frame: np.ndarray) -> float:
        if frame.dtype != np.int16:
            raise ValueError(f"expected int16 frame, got {frame.dtype}")
        if frame.shape != (FRAME_SAMPLES,):
            raise ValueError(f"expected {FRAME_SAMPLES} samples, got {frame.shape}")

        x = np.concatenate([self._context, frame[None, :].astype(np.float32) / 32768.0], axis=1)
        feeds = {"input": x, "state": self._state, "sr": self._sr}
        out, self._state = self._session.run(None, feeds)
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(out[0, 0])

    def reset(self) -> None:
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
