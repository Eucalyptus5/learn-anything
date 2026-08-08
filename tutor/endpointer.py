from enum import Enum, StrEnum, auto

from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE

SILENCE_WINDOW_MS = 500


class EndpointEvent(StrEnum):
    SPEECH_START = auto()
    SILENCE_START = auto()
    END_OF_TURN = auto()


class _State(Enum):
    IDLE = auto()
    SPEECH = auto()
    SILENCE = auto()


class Endpointer:
    def __init__(
        self,
        silence_window_ms: int = SILENCE_WINDOW_MS,
        start_threshold: float = 0.5,
        continue_threshold: float = 0.35,
        start_frames: int = 3,
    ) -> None:
        self._window_samples = silence_window_ms * SAMPLE_RATE // 1000
        self._start_threshold = start_threshold
        self._continue_threshold = continue_threshold
        self._start_frames = start_frames
        self.reset()

    def push(self, probability: float) -> EndpointEvent | None:
        if self._state is _State.IDLE:
            if probability <= self._start_threshold:
                self._run = 0
                return None
            self._run += 1
            if self._run < self._start_frames:
                return None
            self._enter_speech()
            return EndpointEvent.SPEECH_START

        if self._state is _State.SPEECH:
            if probability >= self._continue_threshold:
                return None
            self._state = _State.SILENCE
            self._silence = FRAME_SAMPLES
            return EndpointEvent.SILENCE_START

        if probability > self._start_threshold:
            self._enter_speech()
            return None
        self._silence += FRAME_SAMPLES
        if self._silence < self._window_samples:
            return None
        self.reset()
        return EndpointEvent.END_OF_TURN

    def reset(self) -> None:
        self._state = _State.IDLE
        self._run = 0
        self._silence = 0

    def _enter_speech(self) -> None:
        self._state = _State.SPEECH
        self._run = 0
        self._silence = 0
