import importlib.util
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.config import Settings
from tutor.cost import UsageLedger
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


POWERMETRICS_BLOCKS = """Machine model: Mac14,2
OS version: 25C56
Boot arguments: 
Boot time: Thu Sep 10 23:42:17 2026



*** Sampled system activity (Fri Sep 11 13:41:46 2026 -0700) (1002.90ms elapsed) ***


**** Processor usage ****

E-Cluster HW active frequency: 1248 MHz
E-Cluster HW active residency:  77.43% (600 MHz:   0% 912 MHz:  54% 2424 MHz: 8.7%)
E-Cluster idle residency:  22.57%
CPU 0 frequency: 1359 MHz

P-Cluster HW active frequency: 1145 MHz
P-Cluster HW active residency:  56.23% (660 MHz:  36% 924 MHz: .54% 3504 MHz:   0%)
P-Cluster idle residency:  43.77%
CPU 4 frequency: 2039 MHz

CPU Power: 823 mW
GPU Power: 26 mW
ANE Power: 323 mW
Combined Power (CPU + GPU + ANE): 1172 mW


**** Thermal pressure ****

Current pressure level: Nominal


*** Sampled system activity (Fri Sep 11 13:41:47 2026 -0700) (1010.09ms elapsed) ***


**** Processor usage ****

E-Cluster HW active frequency: 1049 MHz
E-Cluster HW active residency:  77.05% (600 MHz:   0% 912 MHz:  65% 2424 MHz: 1.8%)
E-Cluster idle residency:  22.95%
CPU 0 frequency: 1045 MHz

P-Cluster HW active frequency: 715 MHz
P-Cluster HW active residency:  51.36% (660 MHz:  47% 924 MHz: .40% 3504 MHz:   0%)
P-Cluster idle residency:  48.64%
CPU 4 frequency: 1220 MHz

CPU Power: 205 mW
GPU Power: 19 mW
ANE Power: 221 mW
Combined Power (CPU + GPU + ANE): 445 mW


**** Thermal pressure ****

Current pressure level: Nominal
"""


def _blocks(text: str) -> list[str]:
    return list(bench_turn.powermetrics_blocks(text.splitlines(keepends=True)))


def test_powermetrics_blocks_frames_on_the_pressure_line() -> None:
    blocks = _blocks(POWERMETRICS_BLOCKS)

    assert len(blocks) == 2
    assert all(block.startswith(bench_turn.POWERMETRICS_HEADER) for block in blocks)
    assert all(block.rstrip().endswith("Current pressure level: Nominal") for block in blocks)
    assert "Machine model" not in blocks[0]


def test_powermetrics_blocks_holds_a_block_until_its_pressure_line() -> None:
    lines = POWERMETRICS_BLOCKS.splitlines(keepends=True)
    cut = lines[: len(lines) - 1]

    assert len(list(bench_turn.powermetrics_blocks(cut))) == 1


def test_parse_powermetrics_reads_each_block() -> None:
    first, second = _blocks(POWERMETRICS_BLOCKS)

    a = bench_turn.parse_powermetrics(first)
    b = bench_turn.parse_powermetrics(second)

    assert a == bench_turn.PowerSample(
        cpu_power_mw=823,
        combined_power_mw=1172,
        p_cluster_mhz=1145,
        e_cluster_mhz=1248,
        pressure="Nominal",
    )
    assert b == bench_turn.PowerSample(
        cpu_power_mw=205,
        combined_power_mw=445,
        p_cluster_mhz=715,
        e_cluster_mhz=1049,
        pressure="Nominal",
    )


def test_parse_powermetrics_without_pressure_is_none() -> None:
    first, _ = _blocks(POWERMETRICS_BLOCKS)
    cut = first[: first.index("**** Thermal pressure ****")]

    assert bench_turn.parse_powermetrics(cut) is None


def test_parse_swap_truncates_to_whole_megabytes() -> None:
    line = "total = 5120.00M  used = 3972.81M  free = 1147.19M  (encrypted)\n"

    assert bench_turn.parse_swap(line) == 3972


def test_edge_buckets_split_the_first_and_last_window() -> None:
    offsets = [0.0, 100.0, 299.0, 300.0, 1500.0, 1501.0, 1799.0]

    first, last = bench_turn.edge_buckets(offsets, minutes=30, edge_min=5)

    assert first == [0, 1, 2]
    assert last == [4, 5, 6]


def test_edge_buckets_overlap_on_a_short_run() -> None:
    offsets = [0.0, 60.0, 120.0]

    first, last = bench_turn.edge_buckets(offsets, minutes=2, edge_min=5)

    assert first == [0, 1, 2]
    assert last == [0, 1, 2]


def test_parser_soak_defaults_to_none() -> None:
    args = bench_turn.build_parser().parse_args([])

    assert args.soak is None
    assert bench_turn.build_parser().parse_args(["--soak", "30"]).soak == 30


def _record(at: float, rss_mb: int, swap_used_mb: int) -> bench_turn.SoakRecord:
    power = bench_turn.PowerSample(
        cpu_power_mw=500,
        combined_power_mw=800,
        p_cluster_mhz=1000,
        e_cluster_mhz=1000,
        pressure="Nominal",
    )
    return bench_turn.SoakRecord(at, power, rss_mb, swap_used_mb)


def test_report_soak_buckets_rss_and_swap_at_the_edges(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_turn.build_parser().parse_args(["--soak", "30"])
    records = [
        _record(0.0, 1500, 100),
        _record(120.0, 1600, 110),
        _record(1560.0, 2400, 300),
        _record(1680.0, 2500, 310),
    ]

    bench_turn.report_soak(cfg, args, [], records, UsageLedger())

    lines = capsys.readouterr().out.splitlines()
    rss, rss_first, rss_last = [line for line in lines if line.startswith("process rss")]
    swap, swap_first, swap_last = [line for line in lines if line.startswith("swap used")]
    assert "n=  4 min=  1500MB median=  2000MB max=  2500MB" in rss
    assert rss_first.startswith("process rss, first 5 min")
    assert "n=  2 min=  1500MB median=  1550MB max=  1600MB" in rss_first
    assert rss_last.startswith("process rss, last 5 min")
    assert "n=  2 min=  2400MB median=  2450MB max=  2500MB" in rss_last
    assert "n=  4 min=   100MB median=   205MB max=   310MB" in swap
    assert swap_first.startswith("swap used, first 5 min")
    assert "n=  2 min=   100MB median=   105MB max=   110MB" in swap_first
    assert swap_last.startswith("swap used, last 5 min")
    assert "n=  2 min=   300MB median=   305MB max=   310MB" in swap_last
