import asyncio
import sys
from pathlib import Path

import pytest

from tests.fakes import Spawned, record_spawns
from tutor.prompt import derive_globs
from tutor.tools.models import PATH_EXTENSIONS, SearchBudget
from tutor.tools.search import search

FIXTURE_ROOT = Path(__file__).parent / "data" / "fixture_repo"


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> Spawned:
    return record_spawns(monkeypatch, None)


@pytest.fixture
def blocking_child(monkeypatch: pytest.MonkeyPatch) -> Spawned:
    return record_spawns(
        monkeypatch, [sys.executable, "-c", "import threading; threading.Event().wait()"]
    )


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


async def test_bare_turn_on_a_go_tree_yields_go_globs(tmp_path: Path) -> None:
    _write(tmp_path, "cmd/server/main.go", "package main\n")
    _write(tmp_path, "internal/router/router.go", "package router\n")
    _write(tmp_path, "internal/router/route_test.go", "package router\n")
    _write(tmp_path, "internal/store/store.go", "package store\n")
    _write(tmp_path, "docs/design.md", "notes\n")

    globs = await derive_globs("how does the request get handled", tmp_path)

    assert globs[0] == "**/*.go"
    assert "**/*" not in globs
    assert all(glob.removeprefix("**/*") in PATH_EXTENSIONS for glob in globs)
    assert len(globs) <= 6


async def test_named_path_anchors_to_its_directory(spawned: Spawned) -> None:
    globs = await derive_globs("walk me through src/pool.py", FIXTURE_ROOT)

    assert globs == ["src/**/*.py"]
    assert spawned.calls == 0


async def test_prose_decimal_is_not_a_path() -> None:
    globs = await derive_globs(
        "we moved to version 3.12 last week, where is acquire called", FIXTURE_ROOT
    )

    assert all("3." not in glob and ".12" not in glob for glob in globs)
    assert "**/*.py" in globs


async def test_derived_globs_are_a_legal_search_argument() -> None:
    globs = await derive_globs("where is the connection handed back", FIXTURE_ROOT)

    result = await search("acquire", globs, FIXTURE_ROOT, SearchBudget())

    assert result.matches
    assert result.globs == globs


async def test_language_word_maps_to_its_extensions(spawned: Spawned) -> None:
    globs = await derive_globs("show me the python side of the pool", FIXTURE_ROOT)

    assert globs[0] == "**/*.py"
    assert spawned.calls == 0


async def test_completed_walk_is_cached_per_root(spawned: Spawned, tmp_path: Path) -> None:
    _write(tmp_path, "src/app.py", "x = 1\n")

    first = await derive_globs("where does this start", tmp_path)
    second = await derive_globs("where does this start", tmp_path)

    assert first == second
    assert first == ["**/*.py"]
    assert spawned.calls == 1


async def test_cancelled_walk_kills_the_child(blocking_child: Spawned, tmp_path: Path) -> None:
    task = asyncio.create_task(derive_globs("anything", tmp_path))
    await blocking_child.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert blocking_child.proc.returncode < 0


async def test_timed_out_walk_kills_the_child_and_caches_nothing(
    blocking_child: Spawned, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("tutor.prompt.WALK_TIMEOUT_MS", 1)

    assert await derive_globs("anything", tmp_path) == []
    assert blocking_child.proc.returncode < 0

    await derive_globs("anything", tmp_path)

    assert blocking_child.calls == 2
