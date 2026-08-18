import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_search.py"
_spec = importlib.util.spec_from_file_location("bench_search", SCRIPT)
bench_search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_search)

FIXTURE_ROOT = Path(__file__).parent / "data" / "fixture_repo"


async def test_time_runs_discards_exactly_three_warmups() -> None:
    calls = 0

    async def fake() -> None:
        nonlocal calls
        calls += 1

    result = await bench_search.time_runs(fake, 5)

    assert len(result) == 5
    assert calls == 8


def test_summarize_applies_the_p95_rule() -> None:
    median, p95 = bench_search.summarize([i / 1000 for i in range(1, 21)])

    assert (median, p95) == (10, 19)
    assert isinstance(median, int)
    assert isinstance(p95, int)


def test_samples_defaults_to_thirty() -> None:
    parser = bench_search.build_parser()
    args = parser.parse_args(["--root", "tests/data/fixture_repo"])

    assert args.samples == 30


def test_root_is_required() -> None:
    parser = bench_search.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_file_count_counts_the_fixture_repo() -> None:
    assert bench_search.file_count(FIXTURE_ROOT) == 8
