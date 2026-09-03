import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_bargein.py"
_spec = importlib.util.spec_from_file_location("bench_bargein", SCRIPT)
bench_bargein = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_bargein)

Enqueued = bench_bargein.Enqueued

FRAMES = [(10.0, True), (10.25, True), (10.5, True), (10.75, False), (11.0, True)]


def test_last_old_frame_is_the_last_real_frame_before_the_flush() -> None:
    assert bench_bargein.last_old_frame_ms(FRAMES, 10.375, 10.625) == 125


def test_last_old_frame_before_the_barge_in_floors_at_zero() -> None:
    assert bench_bargein.last_old_frame_ms(FRAMES, 10.625, 10.875) == 0


def test_last_old_frame_with_no_real_frames_is_zero() -> None:
    silent = [(t, False) for t, _ in FRAMES]

    assert bench_bargein.last_old_frame_ms(silent, 10.375, 10.625) == 0
    assert bench_bargein.last_old_frame_ms([], 10.375, 10.625) == 0


def test_overlap_when_the_cancelled_call_outlives_the_next_start() -> None:
    intervals = [(1.0, 1.5, "old"), (1.25, 1.75, "new"), (1.875, 2.0, "later")]

    assert bench_bargein.overlap_ms(intervals, 1.125) == (250, 500)


def test_overlap_when_the_cancelled_call_ends_first_is_zero() -> None:
    intervals = [(1.0, 1.125, "old"), (1.25, 1.75, "new")]

    assert bench_bargein.overlap_ms(intervals, 1.0625) == (0, 500)


def test_overlap_picks_the_earliest_call_after_the_barge_in() -> None:
    intervals = [(1.0, 1.5, "old"), (1.375, 1.625, "second"), (1.25, 1.75, "new")]

    assert bench_bargein.overlap_ms(intervals, 1.125) == (250, 500)


def test_overlap_with_no_call_in_flight_is_none() -> None:
    assert bench_bargein.overlap_ms([(1.0, 1.0625, "old"), (1.25, 1.75, "new")], 1.125) is None
    assert bench_bargein.overlap_ms([(1.0, 1.5, "old")], 1.125) is None
    assert bench_bargein.overlap_ms([], 1.125) is None


def test_accept_is_the_first_opener_after_the_barge_in() -> None:
    ledger = [
        Enqueued(0.5, None, 100),
        Enqueued(1.0, "a clause", 200),
        Enqueued(1.375, None, 100),
        Enqueued(1.875, None, 100),
    ]

    assert bench_bargein.accept_ms(ledger, 1.125) == 250


def test_accept_without_an_opener_after_the_barge_in_is_none() -> None:
    ledger = [Enqueued(0.5, None, 100), Enqueued(1.25, "a clause", 200)]

    assert bench_bargein.accept_ms(ledger, 1.125) is None


class FakeSynth:
    def __init__(self, probe: "list[bool]", fail: bool = False) -> None:
        self._probe = probe
        self._fail = fail
        self.wrapper: bench_bargein.BusySynth | None = None

    def synthesize(self, text: str) -> np.ndarray:
        self._probe.append(self.wrapper.busy)
        if self._fail:
            raise RuntimeError("phonemizer")
        return np.zeros(4, dtype=np.int16)


def test_busy_synth_is_busy_only_inside_the_call_and_records_the_interval() -> None:
    probe: list[bool] = []
    fake = FakeSynth(probe)
    synth = bench_bargein.BusySynth(fake)
    fake.wrapper = synth

    assert not synth.busy
    audio = synth.synthesize("hello there")

    assert probe == [True]
    assert not synth.busy
    assert len(audio) == 4
    assert synth.last == "hello there"
    assert synth.errors == []
    [(start, end, text)] = synth.intervals
    assert start <= end
    assert text == "hello there"


def test_busy_synth_records_and_reraises_an_exception() -> None:
    probe: list[bool] = []
    fake = FakeSynth(probe, fail=True)
    synth = bench_bargein.BusySynth(fake)
    fake.wrapper = synth

    with pytest.raises(RuntimeError):
        synth.synthesize("hello")

    assert not synth.busy
    assert synth.errors == ["RuntimeError"]
    assert [text for _, _, text in synth.intervals] == ["hello"]


def test_parser_defaults() -> None:
    args = bench_bargein.build_parser().parse_args([])

    assert args.root.parts[-3:] == ("tests", "data", "fixture_repo")
    assert args.model is None
