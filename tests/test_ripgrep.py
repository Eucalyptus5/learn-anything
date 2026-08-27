import asyncio
import json
import logging
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fakes import (
    SPLIT_SEPARATORS,
    Spawned,
    forged_names_tree,
    match_record,
    record_spawns,
)
from tutor.tools.models import SearchBudget
from tutor.tools.ripgrep import (
    VISIBILITY_WALK_TIMEOUT_MS,
    RipgrepFailed,
    RipgrepUnavailable,
    probe_ripgrep,
    reap,
    run_ripgrep,
    visible_files,
)

FAKE_RG_DIR = str(Path(__file__).parent / "data" / "fake_rg")
FIXTURE_ROOT = Path(__file__).parent / "data" / "fixture_repo"
GLOB_FIXTURE = Path(__file__).parent / "data" / "glob_fixture"
HANG_GUARD_S = 20.0
OVER_READER_LIMIT_BYTES = 200000
FIXTURE_PY = {
    "./generated/out.py",
    "./src/httpclient.py",
    "./src/pool.py",
    "./src/pool_helpers.py",
    "./src/wide.py",
    "./vendor/big.py",
    "./vendor/small.py",
}


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
    assert "--sort" not in argv
    assert "--threads" not in argv
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
    records, byte_count, truncated, _ = await run_ripgrep(
        '= "', ["**/*.py"], FIXTURE_ROOT, SearchBudget(max_bytes=12000)
    )

    assert truncated is True
    assert records
    assert {record["data"]["path"]["text"] for record in records} <= FIXTURE_PY
    assert byte_count <= 12000
    assert spawned.proc.returncode < 0


async def test_records_arrive_in_contiguous_per_file_blocks() -> None:
    records, _, truncated, oversized = await run_ripgrep(
        '= "', ["**/*.py"], FIXTURE_ROOT, SearchBudget(max_bytes=400000)
    )

    blocks: list[tuple[str, list[int]]] = []
    for record in records:
        path = record["data"]["path"]["text"]
        if not blocks or blocks[-1][0] != path:
            blocks.append((path, []))
        blocks[-1][1].append(record["data"]["line_number"])

    assert truncated is False
    assert oversized is False
    assert len({path for path, _ in blocks}) > 1
    assert len(blocks) == len({path for path, _ in blocks})
    for _, lines in blocks:
        assert lines == sorted(set(lines))


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
    assert sorted(record["data"]["path"]["text"] for record in found) == [
        "./vendor/big.py",
        "./vendor/small.py",
    ]
    big = next(record for record in found if record["data"]["path"]["text"] == "./vendor/big.py")
    assert len(big["data"]["lines"]["text"]) == 400
    assert oversized is False
    assert truncated is False


async def test_submatch_offsets_survive_non_ascii_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    accent = "\u00e9"
    payload = "".join(
        json.dumps(record) + "\n"
        for record in (
            match_record(
                "./src/httpclient.py",
                1,
                accent * 300 + "NEEDLE\n",
                [{"match": {"text": "NEEDLE"}, "start": 600, "end": 606}],
            ),
            match_record(
                "./src/httpclient.py",
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


async def test_records_outside_the_visible_tree_are_dropped_before_the_meter() -> None:
    records, byte_count, truncated, oversized = await run_ripgrep(
        "SENTINEL", ["**/*"], GLOB_FIXTURE, SearchBudget(max_bytes=2000)
    )

    assert {record["data"]["path"]["text"] for record in records} == {
        "./sub/deep.py",
        "./visible.py",
    }
    assert truncated is False
    assert oversized is False
    assert byte_count == metered(records)


def install_walk_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    shim = tmp_path / "rg"
    shim.write_text(f"#!/bin/sh\n{body}\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))


def begin_record(path: str) -> dict:
    return {"type": "begin", "data": {"path": {"text": path}}}


def test_a_separator_in_a_filename_cannot_forge_a_visible_entry(tmp_path: Path) -> None:
    root = forged_names_tree(tmp_path)

    visible = visible_files(root)

    assert "./private_notes/secret.txt" not in visible
    assert "./visible.py" in visible
    assert len(visible) == len(SPLIT_SEPARATORS) + 1


async def test_a_forged_entry_cannot_admit_an_ignored_file(tmp_path: Path) -> None:
    root = forged_names_tree(tmp_path)

    records, _, _, _ = await run_ripgrep("CANARY", ["**/*"], root, SearchBudget())

    paths = {record["data"]["path"]["text"] for record in matches(records)}
    assert "./private_notes/secret.txt" not in paths
    assert "./visible.py" in paths


async def test_oversized_flags_a_visible_record_and_ignores_an_invisible_one() -> None:
    hidden, _, _, hidden_oversized = await run_ripgrep(
        "row 00", ["**/*"], GLOB_FIXTURE, SearchBudget(max_record_bytes=250)
    )

    assert hidden == []
    assert hidden_oversized is False

    shown, _, _, shown_oversized = await run_ripgrep(
        "SENTINEL", ["**/*"], GLOB_FIXTURE, SearchBudget(max_record_bytes=150)
    )

    assert {record["data"]["path"]["text"] for record in shown} == {
        "./sub/deep.py",
        "./visible.py",
    }
    assert shown_oversized is True


@pytest.mark.parametrize(("path", "flagged"), [("./src/pool.py", True), ("./ignored.py", False)])
async def test_a_record_too_large_to_buffer_is_attributed_to_its_block(
    monkeypatch: pytest.MonkeyPatch, path: str, flagged: bool
) -> None:
    huge = match_record(
        path, 1, "x" * 70000 + "\n", [{"match": {"text": "x"}, "start": 0, "end": 1}]
    )
    payload = json.dumps(begin_record(path)) + "\n" + json.dumps(huge) + "\n"
    record_spawns(
        monkeypatch, [sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload]
    )

    records, _, truncated, oversized = await run_ripgrep(
        "x", ["src/*.py"], FIXTURE_ROOT, SearchBudget(max_record_bytes=200)
    )

    assert records == []
    assert truncated is False
    assert oversized is flagged


def test_a_walk_that_produced_nothing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_walk_shim(tmp_path, monkeypatch, "echo 'rg: /nope: Permission denied' >&2\nexit 2")

    with pytest.raises(RipgrepFailed, match="Permission denied"):
        visible_files(FIXTURE_ROOT)


def test_a_partial_walk_narrows_and_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="tutor.tools.ripgrep")
    install_walk_shim(
        tmp_path,
        monkeypatch,
        "printf './src/pool.py\\0'\necho 'rg: /nope: Permission denied' >&2\nexit 2",
    )

    assert visible_files(FIXTURE_ROOT) == frozenset({"./src/pool.py"})
    assert any("ripgrep_walk_partial" in record.getMessage() for record in caplog.records)


async def test_the_walk_is_bounded(spawned: Spawned, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    real = subprocess.run

    def capture(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(kwargs.get("timeout"))
        return real(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture)

    await run_ripgrep("acquire", ["src/*.py"], FIXTURE_ROOT, SearchBudget())

    assert seen == [VISIBILITY_WALK_TIMEOUT_MS / 1000]


async def test_a_timed_out_walk_fails_closed_and_visibly(
    spawned: Spawned, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timing_out(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(argv, 2.0)

    monkeypatch.setattr(subprocess, "run", timing_out)

    with pytest.raises(RipgrepFailed, match="timed out"):
        await run_ripgrep("acquire", ["src/*.py"], FIXTURE_ROOT, SearchBudget())

    assert spawned.calls == 0


@pytest.mark.parametrize("pipe", ["stdout", "stderr"])
async def test_reap_returns_when_a_pipe_reader_is_paused(pipe: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import threading; threading.Event().wait()",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    # Past the reader limit the transport is paused, so it never delivers EOF on its own.
    getattr(proc, pipe).feed_data(b"x" * OVER_READER_LIMIT_BYTES)

    await asyncio.wait_for(reap(proc), HANG_GUARD_S)

    assert proc.returncode < 0


async def test_a_truncated_final_walk_entry_cannot_admit_an_ignored_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rgignore").write_text("/notes.txt\n")
    (tmp_path / "notes.txt").write_text("CANARY ignored row\n")
    (tmp_path / "notes.txt.md").write_text("CANARY visible row\n")

    def cut_short(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b"./notes.txt.md\x00./notes.txt", b"")

    monkeypatch.setattr(subprocess, "run", cut_short)

    assert visible_files(tmp_path) == frozenset({"./notes.txt.md"})

    records, _, _, _ = await run_ripgrep("CANARY", ["**/*"], tmp_path, SearchBudget())

    assert {record["data"]["path"]["text"] for record in records} == {"./notes.txt.md"}


def test_a_signal_killed_walk_that_named_nothing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_walk_shim(tmp_path, monkeypatch, "kill -9 $$")

    with pytest.raises(RipgrepFailed, match="exited -9"):
        visible_files(FIXTURE_ROOT)


def test_a_signal_killed_walk_that_named_some_files_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="tutor.tools.ripgrep")
    install_walk_shim(tmp_path, monkeypatch, "printf './src/pool.py\\0'\nkill -9 $$")

    assert visible_files(FIXTURE_ROOT) == frozenset({"./src/pool.py"})
    assert any("ripgrep_walk_partial" in record.getMessage() for record in caplog.records)


def test_a_tree_holding_no_visible_files_is_not_a_walk_failure(tmp_path: Path) -> None:
    assert visible_files(tmp_path) == frozenset()
