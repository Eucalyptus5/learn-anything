import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.input_path import EndOfTurn, SpeechStarted
from tutor.session import OutcomeSplitter

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_turn.py"
_spec = importlib.util.spec_from_file_location("bench_turn", SCRIPT)
bench_turn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_turn)

TIMES = [float(n) for n in range(60)]

DELTAS = [
    "The `acquire` method pops ",
    "from the available list. Then it adds",
    " the connection to in_use, and returns it.",
    "\n<outcome>",
    '{"signal": "covered", "settling_positions": []}',
]


async def _loop_pump(deltas: list[str]) -> list[str]:
    splitter = OutcomeSplitter()
    pieces = [text for text in (splitter.feed(delta) for delta in deltas) if text]
    tail, _ = splitter.finish()
    if tail:
        pieces.append(tail)

    async def replay() -> AsyncIterator[str]:
        for piece in pieces:
            yield piece

    return [clause async for clause in clause_chunks(spoken_text(replay(), Scrubber()))]


def test_substance_frame_is_the_first_frame_holding_the_model_buffer() -> None:
    assert bench_turn.substance_frame_time([24000, 12000], 1, TIMES) == 50.0


def test_substance_frame_for_the_first_buffer_is_frame_zero() -> None:
    assert bench_turn.substance_frame_time([24000, 12000], 0, TIMES) == 0.0


def test_substance_frame_when_the_boundary_falls_inside_a_frame() -> None:
    assert bench_turn.substance_frame_time([500, 100], 1, TIMES) == 1.0
    assert bench_turn.substance_frame_time([480, 100], 1, TIMES) == 1.0
    assert bench_turn.substance_frame_time([479, 100], 1, TIMES) == 0.0


def test_substance_frame_not_yet_emitted_is_none() -> None:
    assert bench_turn.substance_frame_time([24000, 12000], 1, TIMES[:50]) is None


async def test_loop_clauses_classify_as_model_and_lead_in_does_not() -> None:
    clauses = await _loop_pump(DELTAS)
    model = bench_turn.model_text([DELTAS])

    assert clauses
    assert all(bench_turn.is_model(clause, model) for clause in clauses)
    assert "`" not in model
    assert "<outcome>" not in model
    assert not bench_turn.is_model("The match is in src/pool.py line 12.", model)
    assert not bench_turn.is_model(None, model)


async def test_follow_up_stream_joins_the_first_with_a_newline() -> None:
    first = ["Two things to check"]
    second = ["and here is the second one."]

    assert (
        bench_turn.model_text([first, second]) == "Two things to check\nand here is the second one."
    )


async def test_scripted_source_yields_in_order_and_ends_on_close() -> None:
    source = bench_turn.ScriptedSource()
    source.inject(EndOfTurn(text="one"))
    source.inject(SpeechStarted())
    source.inject(EndOfTurn(text="two"))
    source.close()

    events = [event async for event in source.events()]

    assert events == [EndOfTurn(text="one"), SpeechStarted(), EndOfTurn(text="two")]


def test_floor_lifts_every_zero_sample_and_leaves_the_original_alone() -> None:
    pcm = np.array([0, 5, 0, -3, 0], dtype=np.int16)

    floored = bench_turn.floored(pcm)

    assert floored.dtype == np.int16
    assert floored.tolist() == [1, 5, 1, -3, 1]
    assert pcm.tolist() == [0, 5, 0, -3, 0]
    assert floored is not pcm


def test_parser_defaults() -> None:
    args = bench_turn.build_parser().parse_args([])

    assert args.root.parts[-3:] == ("tests", "data", "fixture_repo")
    assert args.model is None
