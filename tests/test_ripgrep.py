import asyncio
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fakes import Spawned, match_record, record_spawns
from tutor.tools.models import SearchBudget
from tutor.tools.ripgrep import RipgrepFailed, RipgrepUnavailable, probe_ripgrep, run_ripgrep

FAKE_RG_DIR = str(Path(__file__).parent / "data" / "fake_rg")
FIXTURE_ROOT = Path(__file__).parent / "data" / "fixture_repo"


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> Spawned:
    return record_spawns(monkeypatch, None)


@pytest.fixture
def blocking_child(monkeypatch: pytest.MonkeyPatch) -> Spawned:
    return record_spawns(
        monkeypatch, [sys.executable, "-c", "import threading; threading.Event().wait()"]
    )


@pytest.fixture
def unreadable_helper() -> Iterator[Path]:
    path = FIXTURE_ROOT / "src" / "pool_helpers.py"
    mode = path.stat().st_mode
    path.chmod(0o000)
    try:
        yield path
    finally:
        path.chmod(mode)


def matches(records: list[dict]) -> list[dict]:
    return [record for record in records if record["type"] == "match"]


def metered(records: list[dict]) -> int:
    return sum(len(json.dumps(record, separators=(",", ":")).encode()) for record in records)


def test_probe_ripgrep_returns_version_from_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", FAKE_RG_DIR)

    assert probe_ripgrep() == "15.2.0"


def test_probe_ripgrep_rejects_version_below_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", FAKE_RG_DIR)
    monkeypatch.setenv("FAKE_RG_VERSION", "12.0.0")

    with pytest.raises(RipgrepUnavailable, match="12.0.0"):
        probe_ripgrep()


def test_probe_ripgrep_raises_on_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RipgrepUnavailable, match="rg"):
        probe_ripgrep()


async def test_run_ripgrep_rejects_empty_globs(spawned: Spawned) -> None:
    with pytest.raises(ValueError):
        await run_ripgrep("acquire", [], FIXTURE_ROOT, SearchBudget())

    assert spawned.calls == 0


async def test_run_ripgrep_argv(spawned: Spawned) -> None:
    await run_ripgrep(
        "acquire", ["src/*.py", "docs/*.md"], FIXTURE_ROOT, SearchBudget(context_lines=3)
    )

    argv = spawned.argv
    assert argv[0] == "rg"
    assert "--json" in argv
    assert "--no-require-git" in argv
    assert argv[argv.index("--sort") + 1] == "path"
    assert argv[argv.index("--context") + 1] == "3"
    globs = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--glob"]
    assert globs == ["src/*.py", "docs/*.md"]
    assert argv[argv.index("-e") + 1] == "acquire"
    assert not any(arg.startswith("--max-columns") for arg in argv)
    assert "--no-ignore" not in argv
    assert argv[-1] == "."
    assert argv.count(".") == 1
    assert Path(spawned.kwargs["cwd"]) == FIXTURE_ROOT


async def test_no_match_returns_empty() -> None:
    assert await run_ripgrep("zzznomatch", ["**/*.py"], FIXTURE_ROOT, SearchBudget()) == (
        [],
        0,
        False,
        False,
    )


async def test_bad_pattern_raises_with_stderr() -> None:
    with pytest.raises(RipgrepFailed, match="regex parse error"):
        await run_ripgrep("[", ["**/*.py"], FIXTURE_ROOT, SearchBudget())


async def test_unreadable_file_beside_match_still_returns_match(
    spawned: Spawned, unreadable_helper: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="tutor.tools.ripgrep")

    records, _, truncated, _ = await run_ripgrep(
        "acquire", ["src/pool*.py"], FIXTURE_ROOT, SearchBudget()
    )

    assert any(record["data"]["path"]["text"] == "./src/pool.py" for record in matches(records))
    assert truncated is False
    assert any("pool_helpers.py" in record.getMessage() for record in caplog.records)


async def test_wide_lines_clip_to_max_columns() -> None:
    records, byte_count, truncated, oversized = await run_ripgrep(
        "WIDE_ROW", ["src/wide.py"], FIXTURE_ROOT, SearchBudget(max_bytes=64000)
    )

    assert len(matches(records)) == 60
    assert len(records) == 60
    assert all(len(record["data"]["lines"]["text"]) == 400 for record in records)
    assert all(
        submatch["end"] <= 400 for record in records for submatch in record["data"]["submatches"]
    )
    assert truncated is False
    assert oversized is False
    assert byte_count == metered(records)
    assert byte_count <= 64000


async def test_byte_budget_truncates_wide(spawned: Spawned) -> None:
    records, byte_count, truncated, _ = await run_ripgrep(
        "WIDE_ROW", ["src/wide.py"], FIXTURE_ROOT, SearchBudget(max_bytes=12000)
    )

    assert 18 <= len(records) <= 22
    assert truncated is True
    assert byte_count <= 12000
    assert spawned.proc.returncode is not None


async def test_byte_budget_kills_a_child_still_writing(spawned: Spawned) -> None:
    records, _, truncated, _ = await run_ripgrep(
        '= "', ["**/*.py"], FIXTURE_ROOT, SearchBudget(max_bytes=12000)
    )

    assert truncated is True
    assert {record["data"]["path"]["text"] for record in records} == {"./src/wide.py"}
    assert spawned.proc.returncode < 0


async def test_oversized_record_is_dropped_and_search_continues(spawned: Spawned) -> None:
    records, _, truncated, oversized = await run_ripgrep(
        "B", ["vendor/*.py"], FIXTURE_ROOT, SearchBudget(max_record_bytes=1000)
    )

    assert oversized is True
    assert truncated is False
    assert [record["data"]["path"]["text"] for record in matches(records)] == ["./vendor/small.py"]
    assert spawned.proc.returncode == 0


async def test_oversized_applies_to_a_terminated_record() -> None:
    records, _, truncated, oversized = await run_ripgrep(
        "WIDE_ROW", ["src/wide.py"], FIXTURE_ROOT, SearchBudget(max_record_bytes=500)
    )

    assert records == []
    assert oversized is True
    assert truncated is False


async def test_big_line_clips_under_default_budget() -> None:
    records, _, truncated, oversized = await run_ripgrep(
        "B", ["vendor/*.py"], FIXTURE_ROOT, SearchBudget()
    )

    found = matches(records)
    assert [record["data"]["path"]["text"] for record in found] == [
        "./vendor/big.py",
        "./vendor/small.py",
    ]
    assert len(found[0]["data"]["lines"]["text"]) == 400
    assert oversized is False
    assert truncated is False


async def test_submatch_offsets_survive_non_ascii_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    accent = "\u00e9"
    payload = "".join(
        json.dumps(record) + "\n"
        for record in (
            match_record(
                "./src/accents.py",
                1,
                accent * 300 + "NEEDLE\n",
                [{"match": {"text": "NEEDLE"}, "start": 600, "end": 606}],
            ),
            match_record(
                "./src/accents.py",
                2,
                accent * 398 + "NEEDLE" + accent * 50 + "\n",
                [
                    {"match": {"text": "NEEDLE"}, "start": 796, "end": 802},
                    {"match": {"text": accent}, "start": 900, "end": 902},
                ],
            ),
        )
    )
    record_spawns(
        monkeypatch, [sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload]
    )

    records, _, truncated, oversized = await run_ripgrep(
        "NEEDLE", ["src/*.py"], FIXTURE_ROOT, SearchBudget()
    )

    first, second = records
    assert len(first["data"]["lines"]["text"]) == 307
    assert first["data"]["submatches"] == [{"match": {"text": "NEEDLE"}, "start": 600, "end": 606}]
    assert len(second["data"]["lines"]["text"]) == 400
    assert second["data"]["submatches"] == [{"match": {"text": "NE"}, "start": 796, "end": 798}]
    for record in records:
        line = record["data"]["lines"]["text"].encode()
        for submatch in record["data"]["submatches"]:
            assert line[submatch["start"] : submatch["end"]].decode() == submatch["match"]["text"]
    assert truncated is False
    assert oversized is False


async def test_cancel_kills_child_and_reraises(blocking_child: Spawned) -> None:
    task = asyncio.create_task(run_ripgrep("x", ["**/*.py"], FIXTURE_ROOT, SearchBudget()))
    await blocking_child.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert blocking_child.proc.returncode < 0


async def test_timeout_kills_child_and_returns_truncated(blocking_child: Spawned) -> None:
    assert await run_ripgrep("x", ["**/*.py"], FIXTURE_ROOT, SearchBudget(timeout_ms=1)) == (
        [],
        0,
        True,
        False,
    )
    assert blocking_child.proc.returncode < 0
