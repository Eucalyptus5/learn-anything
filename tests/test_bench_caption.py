import importlib.util
from pathlib import Path

import numpy as np

from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_caption.py"
_spec = importlib.util.spec_from_file_location("bench_caption", SCRIPT)
bench_caption = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_caption)


def test_the_frame_index_follows_the_cumulative_samples() -> None:
    frame_index = bench_caption.frame_index
    boundary = WEBRTC_FRAME_SAMPLES * TTS_SAMPLE_RATE // WEBRTC_SAMPLE_RATE

    assert frame_index(0) == 0
    assert frame_index(boundary - 1) == 0
    assert frame_index(boundary) == 1
    assert frame_index(TTS_SAMPLE_RATE) == WEBRTC_SAMPLE_RATE // WEBRTC_FRAME_SAMPLES


def test_attribution_reads_the_frame_that_first_carries_each_clause() -> None:
    frames = [1.0 + k * WEBRTC_FRAME_SAMPLES / WEBRTC_SAMPLE_RATE for k in range(200)]
    half = TTS_SAMPLE_RATE // 2

    samples = bench_caption.attribute(
        0.5, [0.9, 0.95], [(1.0, half), (1.0, half)], [0, 500], frames
    )

    assert [s.pulled_ms for s in samples] == [400, 450]
    assert [s.enqueued_ms for s in samples] == [500, 500]
    assert [s.first_sound_ms for s in samples] == [500, 1000]
    assert [s.handoff_lead_ms for s in samples] == [100, 550]
    assert [s.lead_error_ms for s in samples] == [0, 0]


async def test_the_source_holds_a_clause_until_the_backlog_is_under_its_length() -> None:
    lengths_s = [length / 1000 for length in bench_caption.CLAUSE_MS[:2]]
    readings = iter([0.0, lengths_s[0], lengths_s[1] + 0.02, lengths_s[1], lengths_s[1] - 0.02])
    log: list[tuple[str, object]] = []

    def backlog_s() -> float:
        reading = next(readings)
        log.append(("backlog", reading))
        return reading

    async def pace(delay: float) -> None:
        log.append(("wait", delay))

    pulled: list[float] = []
    async for clause in bench_caption.clauses(2, pulled, backlog_s, pace):
        log.append(("clause", clause))

    assert log == [
        ("backlog", 0.0),
        ("clause", "clause 0"),
        ("backlog", lengths_s[0]),
        ("wait", bench_caption.FRAME_S),
        ("backlog", lengths_s[1] + 0.02),
        ("wait", bench_caption.FRAME_S),
        ("backlog", lengths_s[1]),
        ("wait", bench_caption.FRAME_S),
        ("backlog", lengths_s[1] - 0.02),
        ("clause", "clause 1"),
    ]
    assert len(pulled) == 2


def test_the_fake_synthesizer_cycles_the_clause_lengths_without_waiting() -> None:
    synth = bench_caption.LengthsSynth((1200, 800), 0.0)

    lengths = [len(synth.synthesize("x")) for _ in range(3)]

    assert lengths == [
        1200 * TTS_SAMPLE_RATE // 1000,
        800 * TTS_SAMPLE_RATE // 1000,
        1200 * TTS_SAMPLE_RATE // 1000,
    ]
    assert synth.synthesize("x").dtype == np.int16


def test_the_report_prints_both_figures(capsys) -> None:
    ClauseSample = bench_caption.ClauseSample
    samples = [
        ClauseSample(pulled_ms=0, enqueued_ms=300, lead_ms=0, first_sound_ms=320),
        ClauseSample(pulled_ms=310, enqueued_ms=600, lead_ms=920, first_sound_ms=1520),
    ]

    bench_caption.report(samples)

    lines = capsys.readouterr().out.splitlines()
    (handoff,) = [
        line for line in lines if line.startswith("pull to first sound") and " n=" in line
    ]
    (error,) = [line for line in lines if line.startswith("lead error") and " n=" in line]
    assert "n=  2 median=   765ms" in handoff
    assert "n=  2 median=    10ms" in error
    assert any(line.startswith("clauses=2 ") for line in lines)


def test_parser_defaults() -> None:
    args = bench_caption.build_parser().parse_args([])

    assert args.samples == 30
