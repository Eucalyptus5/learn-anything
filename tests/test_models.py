import pytest
from pydantic import ValidationError

from tutor.tools.models import (
    MAX_COLUMNS,
    PATH_EXTENSIONS,
    ContextLine,
    Position,
    SearchBudget,
    SearchMatch,
    SearchResult,
)


def _sample_result() -> SearchResult:
    return SearchResult(
        tool="search_code",
        query="acquire",
        globs=["src/**/*.py"],
        matches=[
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
        ],
        truncated=False,
        oversized=False,
        byte_count=512,
    )


def test_search_result_round_trips_through_json() -> None:
    result = _sample_result()

    restored = SearchResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_positions_returns_one_position_per_returned_line() -> None:
    result = _sample_result()

    positions = result.positions()

    assert len(positions) == 6
    assert [p.key() for p in positions] == [
        ("src/pool.py", 9),
        ("src/pool.py", 10),
        ("src/pool.py", 11),
        ("src/pool.py", 12),
        ("src/httpclient.py", 12),
        ("src/httpclient.py", 13),
    ]
    assert all(p.symbol is None for p in positions)


def test_position_is_hashable_and_equal_by_path_and_line() -> None:
    a = Position(path="a.py", line=3)
    b = Position(path="a.py", line=3)

    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_position_key_ignores_symbol() -> None:
    with_symbol = Position(path="a.py", line=3, symbol="f")
    without_symbol = Position(path="a.py", line=3)

    assert with_symbol.key() == without_symbol.key()


def test_position_is_frozen() -> None:
    position = Position(path="a.py", line=3)

    with pytest.raises(ValidationError):
        position.line = 4


def test_search_budget_defaults() -> None:
    budget = SearchBudget()

    assert budget.max_matches == 40
    assert budget.max_bytes == 24000
    assert budget.max_record_bytes == 1000000
    assert budget.context_lines == 2
    assert budget.timeout_ms == 2000


def test_search_budget_rejects_non_positive_fields() -> None:
    with pytest.raises(ValidationError):
        SearchBudget(max_bytes=0)
    with pytest.raises(ValidationError):
        SearchBudget(max_bytes=-1)
    with pytest.raises(ValidationError):
        SearchBudget(max_matches=0)


def test_search_budget_context_lines_allows_zero_rejects_negative() -> None:
    assert SearchBudget(context_lines=0).context_lines == 0

    with pytest.raises(ValidationError):
        SearchBudget(context_lines=-1)


def test_path_extensions_are_lowercase_dotted_ascii() -> None:
    for ext in PATH_EXTENSIONS:
        assert ext.startswith(".")
        assert ext == ext.lower()
        assert ext.isascii()

    assert ".py" in PATH_EXTENSIONS
    assert ".md" in PATH_EXTENSIONS
    assert ".exe" not in PATH_EXTENSIONS


def test_max_columns_is_a_positive_int() -> None:
    assert isinstance(MAX_COLUMNS, int)
    assert MAX_COLUMNS > 0
