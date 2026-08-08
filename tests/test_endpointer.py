from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE
from tutor.endpointer import SILENCE_WINDOW_MS, Endpointer, EndpointEvent

WINDOW_SAMPLES = SILENCE_WINDOW_MS * SAMPLE_RATE // 1000
SILENT_FRAMES_TO_CLOSE = -(-WINDOW_SAMPLES // FRAME_SAMPLES)


def drive(endpointer: Endpointer, probabilities: list[float]) -> list[EndpointEvent | None]:
    return [endpointer.push(p) for p in probabilities]


def confirmed() -> Endpointer:
    endpointer = Endpointer()
    assert drive(endpointer, [0.9, 0.9, 0.9]) == [None, None, EndpointEvent.SPEECH_START]
    return endpointer


def test_no_event_until_speech_is_confirmed() -> None:
    endpointer = Endpointer()

    assert drive(endpointer, [0.9, 0.9, 0.1]) == [None, None, None]
    assert drive(endpointer, [0.9, 0.5]) == [None, None]
    assert drive(endpointer, [0.9, 0.9]) == [None, None]
    assert endpointer.push(0.9) is EndpointEvent.SPEECH_START


def test_speech_start_emitted_once() -> None:
    endpointer = confirmed()

    assert drive(endpointer, [0.99, 0.9, 0.35, 0.4, 0.9, 0.6]) == [None] * 6


def test_silence_start_precedes_end_of_turn() -> None:
    endpointer = confirmed()

    events = drive(endpointer, [0.0] * SILENT_FRAMES_TO_CLOSE)

    assert events[0] is EndpointEvent.SILENCE_START
    assert events[1:-1] == [None] * (SILENT_FRAMES_TO_CLOSE - 2)
    assert events[-1] is EndpointEvent.END_OF_TURN
    assert drive(endpointer, [0.9, 0.9]) == [None, None]
    assert endpointer.push(0.9) is EndpointEvent.SPEECH_START


def test_short_silence_does_not_end_the_turn() -> None:
    endpointer = confirmed()
    quiet = [0.0 if i % 2 else 0.4 for i in range(SILENT_FRAMES_TO_CLOSE - 2)]

    events = drive(endpointer, [0.0, *quiet])

    assert events[0] is EndpointEvent.SILENCE_START
    assert events[1:] == [None] * (SILENT_FRAMES_TO_CLOSE - 2)
    assert EndpointEvent.END_OF_TURN not in events
    assert endpointer.push(0.4) is EndpointEvent.END_OF_TURN


def test_resume_before_the_window_cancels_pending_silence() -> None:
    endpointer = confirmed()

    assert drive(endpointer, [0.0, 0.0, 0.0]) == [EndpointEvent.SILENCE_START, None, None]
    assert endpointer.push(0.9) is None
    assert endpointer.push(0.0) is EndpointEvent.SILENCE_START
    assert drive(endpointer, [0.0] * (SILENT_FRAMES_TO_CLOSE - 2)) == [None] * (
        SILENT_FRAMES_TO_CLOSE - 2
    )
    assert endpointer.push(0.0) is EndpointEvent.END_OF_TURN


def test_reset_returns_to_idle() -> None:
    endpointer = confirmed()
    assert endpointer.push(0.0) is EndpointEvent.SILENCE_START

    endpointer.reset()

    assert drive(endpointer, [0.0] * SILENT_FRAMES_TO_CLOSE) == [None] * SILENT_FRAMES_TO_CLOSE
    assert drive(endpointer, [0.9, 0.9]) == [None, None]

    endpointer.reset()

    assert drive(endpointer, [0.9, 0.9]) == [None, None]
    assert endpointer.push(0.9) is EndpointEvent.SPEECH_START
