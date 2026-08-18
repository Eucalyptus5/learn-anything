import logging

import pytest

from tutor.chunker import split_clauses
from tutor.tools.models import (
    ContextLine,
    GroundingVerdict,
    Position,
    SearchMatch,
    SearchResult,
)
from tutor.tools.provenance import TurnRegistry, extract_positions


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


def test_verify_accepts_a_recorded_path_with_a_recorded_line() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    verdict = registry.verify("t1", "the acquire helper sits in src/pool.py line 11")

    assert verdict.ok
    assert verdict.ungrounded == []


def test_verify_rejects_a_path_no_tool_returned() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    verdict = registry.verify("t1", "the acquire helper sits in src/queue.py")

    assert not verdict.ok
    assert verdict.ungrounded == [Position(path="src/queue.py")]


def test_verify_rejects_a_recorded_path_with_an_invented_line() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    verdict = registry.verify("t1", "the acquire helper sits in src/pool.py line 44")

    assert not verdict.ok
    assert verdict.ungrounded == [Position(path="src/pool.py", line=44)]


def test_abandoned_turn_leaves_nothing_verifiable_in_the_next_turn() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.abandon("t1")
    registry.record("t1", _sample_result())
    registry.open_turn("t2")

    verdict = registry.verify("t2", "the acquire helper sits in src/pool.py line 11")

    assert not verdict.ok
    assert verdict.ungrounded == [Position(path="src/pool.py", line=11)]


_LEAD_IN = "Here is the thing you should look at now,"


def _pieces(text: str) -> list[str]:
    clauses, remainder = split_clauses(text, 8, 12)
    return clauses + ([remainder.strip()] if remainder.strip() else [])


def _chunked(
    registry: TurnRegistry, turn_id: str, text: str
) -> tuple[list[str], list[GroundingVerdict]]:
    pieces = _pieces(text)
    return pieces, [registry.verify_chunk(turn_id, piece) for piece in pieces]


def _ungrounded(verdicts: list[GroundingVerdict]) -> list[Position]:
    return [position for verdict in verdicts for position in verdict.ungrounded]


@pytest.mark.parametrize(
    "tail",
    [
        "the fix lands in src/pool.py",
        "the fix lands in `src/pool.py`",
        "the fix lands in pool.py",
        "the fix lands in `pool.py`",
        "the fix lands in pool.py:11",
        "the fix lands in `pool.py:11`",
        "the fix lands in pool.py line 11",
        "the fix lands in `pool.py` line 11",
        "the fix lands in pool.py line twelve",
        "the fix lands in `pool.py` line twelve",
        "the fix lands in src/pool.py lines 9 to 10",
        "the fix lands in `src/pool.py` lines 9 to 10",
        "the fix lands on line 11 of src/pool.py",
        "the fix lands on line 11 of `src/pool.py`",
        "the fix lands on the 11th line of src/pool.py",
        "the fix lands on the 11th line of `src/pool.py`",
        "the fix lands on lines 9 to 10 of src/pool.py",
        "the fix lands on lines 9 to 10 of `src/pool.py`",
        "the fix lands in src/pool.py line 9",
        "the fix lands in `src/pool.py` line 9",
        "the fix lands on the 11th and 12th lines of src/pool.py",
        "the fix lands on the 11th and 12th lines of `src/pool.py`",
    ],
)
def test_recorded_token_forms_verify_clean(tail: str) -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    pieces, verdicts = _chunked(registry, "t1", f"{_LEAD_IN} {tail}")

    assert len(pieces) == 2
    assert _ungrounded(verdicts) == []
    assert all(verdict.ok for verdict in verdicts)


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("the fix lands in src/queue.py", [Position(path="src/queue.py")]),
        ("the fix lands in `src/queue.py`", [Position(path="src/queue.py")]),
        ("the fix lands in queue.py", [Position(path="queue.py")]),
        ("the fix lands in `queue.py`", [Position(path="queue.py")]),
        ("the fix lands in pool.py:44", [Position(path="pool.py", line=44)]),
        ("the fix lands in `pool.py:44`", [Position(path="pool.py", line=44)]),
        ("the fix lands in pool.py line 44", [Position(path="pool.py", line=44)]),
        ("the fix lands in `pool.py` line 44", [Position(path="pool.py", line=44)]),
        ("the fix lands in pool.py line one forty two", [Position(path="pool.py", line=142)]),
        ("the fix lands in `pool.py` line one forty two", [Position(path="pool.py", line=142)]),
        (
            "the fix lands in src/pool.py lines 40 to 41",
            [Position(path="src/pool.py", line=40), Position(path="src/pool.py", line=41)],
        ),
        (
            "the fix lands in `src/pool.py` lines 40 to 41",
            [Position(path="src/pool.py", line=40), Position(path="src/pool.py", line=41)],
        ),
        ("the fix lands on line 44 of src/pool.py", [Position(path="src/pool.py", line=44)]),
        ("the fix lands on line 44 of `src/pool.py`", [Position(path="src/pool.py", line=44)]),
        ("the fix lands on the 44th line of src/pool.py", [Position(path="src/pool.py", line=44)]),
        (
            "the fix lands on the 44th line of `src/pool.py`",
            [Position(path="src/pool.py", line=44)],
        ),
        (
            "the fix lands on lines 40 to 41 of src/pool.py",
            [Position(path="src/pool.py", line=40), Position(path="src/pool.py", line=41)],
        ),
        (
            "the fix lands on lines 40 to 41 of `src/pool.py`",
            [Position(path="src/pool.py", line=40), Position(path="src/pool.py", line=41)],
        ),
    ],
)
def test_invented_token_forms_are_reported(tail: str, expected: list[Position]) -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    pieces, verdicts = _chunked(registry, "t1", f"{_LEAD_IN} {tail}")

    assert len(pieces) == 2
    assert _ungrounded(verdicts) == expected
    assert not verdicts[1].ok


@pytest.mark.parametrize(
    "tail",
    [
        "it sits in os.path.join today",
        "the call to threading.Lock blocks",
        "the read/write split matters here",
        "the tutor/tools package holds it",
        "the notes at https://example.com/pool.py explain",
    ],
)
def test_dotted_names_and_urls_yield_no_position(tail: str) -> None:
    pieces = _pieces(f"{_LEAD_IN} {tail}")

    assert len(pieces) == 2
    assert [position for piece in pieces for position in extract_positions(piece)] == []


def test_a_count_that_is_not_a_line_number_yields_only_the_path() -> None:
    pieces = _pieces(f"{_LEAD_IN} pool.py has three callers")

    assert len(pieces) == 2
    assert [position for piece in pieces for position in extract_positions(piece)] == [
        Position(path="pool.py")
    ]


def test_verify_chunk_carries_the_path_into_the_next_clause() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    text = (
        "The connection pool implementation lives over there in src/pool.py,"
        " and you want lines 9 to 10"
    )
    pieces, verdicts = _chunked(registry, "t1", text)

    assert len(pieces) == 2
    assert _ungrounded(verdicts) == []
    assert all(verdict.ok for verdict in verdicts)


def test_verify_chunk_reports_invented_endpoints_against_the_carried_path() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    text = (
        "The connection pool implementation lives over there in src/pool.py,"
        " and you want lines 40 to 41"
    )
    pieces, verdicts = _chunked(registry, "t1", text)

    assert len(pieces) == 2
    assert verdicts[0].ok
    assert not verdicts[1].ok
    assert _ungrounded(verdicts) == [
        Position(path="src/pool.py", line=40),
        Position(path="src/pool.py", line=41),
    ]


def test_a_line_mention_before_any_path_has_no_path_to_bind_to() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())

    pieces, verdicts = _chunked(
        registry, "t1", "On lines 40 to 41 you will find the missing branch"
    )

    assert len(pieces) == 1
    assert not verdicts[0].ok
    assert _ungrounded(verdicts) == [Position(path="", line=40), Position(path="", line=41)]


def test_open_turn_clears_the_carried_path() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())
    _chunked(registry, "t1", "Open src/pool.py and read the acquire body")

    registry.open_turn("t2")
    registry.record("t2", _sample_result())
    pieces, verdicts = _chunked(registry, "t2", "On lines 9 to 10 you will find the acquire body")

    assert len(pieces) == 1
    assert _ungrounded(verdicts) == [Position(path="", line=9), Position(path="", line=10)]


def test_abandon_clears_the_carried_path() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())
    _chunked(registry, "t1", "Open src/pool.py and read the acquire body")

    registry.abandon("t1")
    pieces, verdicts = _chunked(registry, "t1", "On lines 9 to 10 you will find the acquire body")

    assert len(pieces) == 1
    assert _ungrounded(verdicts) == [Position(path="", line=9), Position(path="", line=10)]


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("the standup is at 12:30 today", []),
        ("the blob lives in vendor/big.dat now", [Position(path="vendor/big.dat")]),
        (
            "the fix lands in pool.py:11-15 there",
            [Position(path="pool.py", line=11), Position(path="pool.py", line=15)],
        ),
        ("the forty-second line of pool.py matters", [Position(path="pool.py", line=42)]),
        (
            "you should look at line 12:14 for the bug",
            [Position(path="", line=12), Position(path="", line=14)],
        ),
        ("the fix lands in pool.py line two hundred fifty", [Position(path="pool.py", line=250)]),
        (
            "the fix lands in pool.py line one thousand two hundred thirty four",
            [Position(path="pool.py", line=1234)],
        ),
        ("the second argument to acquire in src/pool.py", [Position(path="src/pool.py")]),
        ("the first case handles it", []),
        ("forty-second", []),
    ],
)
def test_extraction_of_ranges_scales_ordinals_and_clock_times(
    tail: str, expected: list[Position]
) -> None:
    pieces = _pieces(f"{_LEAD_IN} {tail}")

    assert len(pieces) == 2
    assert [position for piece in pieces for position in extract_positions(piece)] == expected


def test_a_scaled_number_word_does_not_truncate_into_a_recorded_line() -> None:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record(
        "t1",
        _result(SearchMatch(path="src/pool.py", line=1, text="import os", before=[], after=[])),
    )

    text = f"{_LEAD_IN} the lock lives in pool.py line one hundred"
    pieces, verdicts = _chunked(registry, "t1", text)

    assert len(pieces) == 2
    assert not verdicts[1].ok
    assert _ungrounded(verdicts) == [Position(path="pool.py", line=100)]
