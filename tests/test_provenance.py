import logging

import pytest

from tutor.tools.models import ContextLine, Position, SearchMatch, SearchResult
from tutor.tools.provenance import TurnRegistry


def _result(*matches: SearchMatch) -> SearchResult:
    return SearchResult(
        tool="search_code",
        query="acquire",
        globs=["**/*.py"],
        matches=list(matches),
        truncated=False,
        oversized=False,
        byte_count=1,
    )


def _sample_result() -> SearchResult:
    return _result(
        SearchMatch(
            path="src/pool.py",
            line=11,
            text="    def acquire(self):",
            before=[ContextLine(line=9, text=""), ContextLine(line=10, text="")],
            after=[ContextLine(line=12, text="        pass")],
        ),
        SearchMatch(
            path="src/httpclient.py",
            line=12,
            text="        conn = self.pool.acquire()",
            before=[],
            after=[ContextLine(line=13, text="        return conn")],
        ),
    )


def test_record_makes_every_returned_position_known_in_its_turn() -> None:
    registry = TurnRegistry()
    result = _sample_result()

    registry.open_turn("t1")
    registry.record("t1", result)

    for position in result.positions():
        assert registry.known("t1", position)
    assert registry.known("t1", Position(path="src/pool.py", line=None))
    assert registry.known("t1", Position(path="src/httpclient.py", line=None))


def test_positions_from_turn_a_are_unknown_in_turn_b() -> None:
    registry = TurnRegistry()
    result = _sample_result()

    registry.open_turn("a")
    registry.record("a", result)
    registry.open_turn("b")

    for position in result.positions():
        assert not registry.known("b", position)
        assert not registry.known("a", position)


def test_record_for_unopened_turn_is_a_logged_no_op(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="tutor.tools.provenance")
    registry = TurnRegistry()
    result = _sample_result()

    registry.open_turn("t1")
    registry.record("never", result)

    for position in result.positions():
        assert not registry.known("t1", position)
    assert any(
        record.name == "tutor.tools.provenance" and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_record_before_any_turn_opened_is_a_logged_no_op(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="tutor.tools.provenance")
    registry = TurnRegistry()
    result = _sample_result()

    registry.record("t1", result)

    for position in result.positions():
        assert not registry.known("t1", position)
    assert any(
        record.name == "tutor.tools.provenance" and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_record_after_abandon_is_a_logged_no_op(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="tutor.tools.provenance")
    registry = TurnRegistry()
    result = _sample_result()

    registry.open_turn("t1")
    registry.abandon("t1")
    registry.record("t1", result)

    for position in result.positions():
        assert not registry.known("t1", position)
    assert any(
        record.name == "tutor.tools.provenance" and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_abandon_of_a_turn_that_is_not_open_leaves_the_open_turn_intact() -> None:
    registry = TurnRegistry()
    result = _sample_result()

    registry.open_turn("t1")
    registry.record("t1", result)
    registry.abandon("t2")

    for position in result.positions():
        assert registry.known("t1", position)


def test_basename_collision_makes_the_shared_basename_unknown() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")

    first = _result(
        SearchMatch(path="src/util.py", line=99, text="x", before=[], after=[]),
        SearchMatch(path="src/pool.py", line=1, text="y", before=[], after=[]),
    )
    registry.record("t1", first)

    assert registry.known("t1", Position(path="util.py", line=99))
    assert registry.known("t1", Position(path="util.py", line=None))
    assert registry.known("t1", Position(path="src/pool.py", line=1))
    assert registry.known("t1", Position(path="pool.py", line=1))
    assert registry.known("t1", Position(path="pool.py", line=None))

    second = _result(SearchMatch(path="lib/util.py", line=5, text="z", before=[], after=[]))
    registry.record("t1", second)

    assert not registry.known("t1", Position(path="util.py", line=99))
    assert not registry.known("t1", Position(path="util.py", line=None))
    assert registry.known("t1", Position(path="src/util.py", line=99))
    assert registry.known("t1", Position(path="src/util.py", line=None))
    assert registry.known("t1", Position(path="lib/util.py", line=5))
    assert registry.known("t1", Position(path="lib/util.py", line=None))
    assert registry.known("t1", Position(path="src/pool.py", line=1))
    assert registry.known("t1", Position(path="pool.py", line=1))
    assert registry.known("t1", Position(path="pool.py", line=None))


def test_known_ignores_symbol_on_the_queried_position() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    assert registry.known("t1", Position(path="src/pool.py", line=11, symbol="acquire"))


def test_no_directory_path_keeps_full_keys_through_a_same_basename_collision() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")

    registry.record(
        "t1", _result(SearchMatch(path="README.md", line=1, text="x", before=[], after=[]))
    )
    assert registry.known("t1", Position(path="README.md", line=1))
    assert registry.known("t1", Position(path="README.md", line=None))

    registry.record(
        "t1", _result(SearchMatch(path="docs/README.md", line=3, text="y", before=[], after=[]))
    )

    assert registry.known("t1", Position(path="README.md", line=1))
    assert registry.known("t1", Position(path="README.md", line=None))
    assert registry.known("t1", Position(path="docs/README.md", line=3))
    assert registry.known("t1", Position(path="docs/README.md", line=None))
