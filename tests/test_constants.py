from tutor.constants import (
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
)


def test_frame_ms_is_the_exact_duration_of_a_vad_frame() -> None:
    assert FRAME_SAMPLES * 1000 % SAMPLE_RATE == 0
    assert FRAME_MS == FRAME_SAMPLES * 1000 // SAMPLE_RATE


def test_webrtc_rate_converts_by_whole_factors() -> None:
    assert WEBRTC_SAMPLE_RATE % SAMPLE_RATE == 0
    assert WEBRTC_SAMPLE_RATE % TTS_SAMPLE_RATE == 0


def test_webrtc_frame_is_a_whole_number_of_milliseconds() -> None:
    assert WEBRTC_FRAME_SAMPLES * 1000 % WEBRTC_SAMPLE_RATE == 0
