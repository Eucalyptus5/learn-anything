import pytest

from tutor.chunker import split_clauses
from tutor.tools.citations import extract_positions
from tutor.tools.models import Position

_LEAD_IN = "Here is the thing you should look at now,"


def _pieces(text: str) -> list[str]:
    clauses, remainder = split_clauses(text, 8, 12)
    return clauses + ([remainder.strip()] if remainder.strip() else [])


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


def test_and_after_a_joined_number_starts_a_second_number() -> None:
    assert extract_positions("src/pool.py lines nine hundred and twelve and thirteen") == [
        Position(path="src/pool.py", line=912),
        Position(path="src/pool.py", line=13),
    ]
    assert extract_positions("src/pool.py lines nine hundred twelve and thirteen") == [
        Position(path="src/pool.py", line=912),
        Position(path="src/pool.py", line=13),
    ]


def test_the_suffix_branch_keeps_urls_and_clock_times_ahead_of_path_shapes() -> None:
    assert extract_positions("the notes at https://example.com:8080 explain it") == []
    assert extract_positions("the standup is at 10:30 today") == []
    assert extract_positions("look at line 10:30 for the bug") == [
        Position(path="", line=10),
        Position(path="", line=30),
    ]
    assert extract_positions("24") == [Position(path="", line=24)]
    assert extract_positions("handler.php") == []
