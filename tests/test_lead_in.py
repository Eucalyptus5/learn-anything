from collections.abc import Callable

import pytest

from tutor.lead_in import lead_in_sentence, opener_key
from tutor.openers import OPENER_PHRASES
from tutor.tools.models import ContextLine, Position, SearchMatch, SearchResult
from tutor.tools.provenance import TurnRegistry, extract_positions

Shape = Callable[[], list[SearchResult]]


def _result(*matches: SearchMatch) -> SearchResult:
    return SearchResult(
        tool="search",
        query="acquire",
        globs=["**/*.py"],
        matches=list(matches),
        truncated=False,
        oversized=False,
        byte_count=1,
    )


def _single_file() -> list[SearchResult]:
    return [_result(SearchMatch(path="src/pool.py", line=11, text="        conn = acquire()"))]


def _many_files() -> list[SearchResult]:
    return [
        _result(
            SearchMatch(path="src/pool.py", line=11, text="        conn = acquire()"),
            SearchMatch(path="src/httpclient.py", line=12, text="        self.pool.acquire()"),
            SearchMatch(path="src/queue.py", line=13, text="        acquire()"),
        )
    ]


def _symbol_definition() -> list[SearchResult]:
    return [
        _result(
            SearchMatch(
                path="tutor/session.py",
                line=42,
                text="    def advance(self, outcome: TurnOutcome) -> Phase:",
            )
        )
    ]


def _file_range() -> list[SearchResult]:
    return [
        _result(
            SearchMatch(
                path="src/pool.py",
                line=11,
                text="        conn = self._free.pop()",
                before=[
                    ContextLine(line=9, text="    def acquire(self):"),
                    ContextLine(line=10, text="        self._lock.acquire()"),
                ],
                after=[ContextLine(line=12, text="        return conn")],
            )
        )
    ]


def _sibling_match_gap() -> list[SearchResult]:
    return [
        _result(
            SearchMatch(
                path="src/pool.py",
                line=10,
                text="        first = pool.acquire()",
                before=[
                    ContextLine(line=8, text="    def warm(self):"),
                    ContextLine(line=9, text="        pool = self._pool"),
                ],
                after=[ContextLine(line=12, text="        first.close()")],
            ),
            SearchMatch(
                path="src/pool.py",
                line=11,
                text="        second = pool.acquire()",
                after=[ContextLine(line=13, text="        second.close()")],
            ),
        )
    ]


def _undecodable_line_gap() -> list[SearchResult]:
    return [
        _result(
            SearchMatch(
                path="src/pool.py",
                line=10,
                text="        conn = self._free.acquire()",
                before=[ContextLine(line=8, text="    def checkout(self):")],
                after=[ContextLine(line=11, text="        return conn")],
            )
        )
    ]


def _empty_matches() -> list[SearchResult]:
    return [_result()]


SHAPES = [
    _single_file,
    _many_files,
    _symbol_definition,
    _file_range,
    _sibling_match_gap,
    _undecodable_line_gap,
]


def _registry(results: list[SearchResult]) -> TurnRegistry:
    registry = TurnRegistry()
    registry.open_turn("t1")
    for result in results:
        registry.record("t1", result)
    return registry


@pytest.mark.parametrize("shape", SHAPES)
def test_sentence_verifies_clean_against_the_turn_registry(shape: Shape) -> None:
    results = shape()
    sentence = lead_in_sentence(results)

    assert extract_positions(sentence)
    assert _registry(results).verify("t1", sentence).ok is True


@pytest.mark.parametrize("shape", SHAPES)
def test_sentence_names_only_positions_the_results_hold(shape: Shape) -> None:
    results = shape()
    paths = {match.path for result in results for match in result.matches}
    held = {position.key() for result in results for position in result.positions()}
    spoken = extract_positions(lead_in_sentence(results))

    assert {position.path for position in spoken} == paths
    for position in spoken:
        assert position.path in paths
        if position.line is not None:
            assert position.key() in held


@pytest.mark.parametrize("shape", SHAPES)
def test_sentence_is_byte_stable(shape: Shape) -> None:
    results = shape()
    sentence = lead_in_sentence(results)

    assert sentence.endswith(".")
    assert lead_in_sentence(results) == sentence
    assert lead_in_sentence(shape()) == sentence
    assert sentence.isascii()


@pytest.mark.parametrize("shape", SHAPES)
def test_sentence_spells_every_path_whole(shape: Shape) -> None:
    results = shape()
    sentence = lead_in_sentence(results)

    for result in results:
        for match in result.matches:
            assert match.path in sentence
    assert " slash " not in sentence
    assert " dot " not in sentence


@pytest.mark.parametrize("shape", SHAPES)
def test_opener_key_is_a_key_the_opener_table_holds(shape: Shape) -> None:
    assert opener_key(shape()) in OPENER_PHRASES


@pytest.mark.parametrize("shape", SHAPES)
def test_every_line_inside_a_spoken_range_is_a_line_the_results_hold(shape: Shape) -> None:
    results = shape()
    held: dict[str, set[int]] = {}
    for result in results:
        for position in result.positions():
            held.setdefault(position.path, set()).add(position.line)

    spoken: dict[str, set[int]] = {}
    for position in extract_positions(lead_in_sentence(results)):
        if position.line is not None:
            spoken.setdefault(position.path, set()).add(position.line)

    for path, lines in spoken.items():
        for line in range(min(lines), max(lines) + 1):
            assert line in held[path]


def test_single_file_hit_speaks_its_line_number() -> None:
    results = _single_file()

    assert extract_positions(lead_in_sentence(results)) == [Position(path="src/pool.py", line=11)]
    assert opener_key(results) == "file_hit"


def test_many_files_sentence_names_every_path_once() -> None:
    results = _many_files()
    sentence = lead_in_sentence(results)

    for match in results[0].matches:
        assert sentence.count(match.path) == 1
    assert opener_key(results) == "many_files"


def test_symbol_spoken_comes_from_the_match_text() -> None:
    assert "advance" in lead_in_sentence(_symbol_definition())

    renamed = [_result(SearchMatch(path="tutor/session.py", line=42, text="class TurnRegistry:"))]
    sentence = lead_in_sentence(renamed)

    assert "TurnRegistry" in sentence
    assert "advance" not in sentence


def test_range_spoken_lies_inside_the_lines_the_result_holds() -> None:
    results = _file_range()
    held = {position.line for position in results[0].positions()}
    spoken = {
        position.line
        for position in extract_positions(lead_in_sentence(results))
        if position.line is not None
    }

    assert spoken
    assert spoken <= held
    assert min(spoken) >= min(held)
    assert max(spoken) <= max(held)


@pytest.mark.parametrize("shape", [_sibling_match_gap, _undecodable_line_gap])
def test_a_gapped_span_speaks_only_the_match_line(shape: Shape) -> None:
    sentence = lead_in_sentence(shape())

    assert extract_positions(sentence) == [Position(path="src/pool.py", line=10)]


@pytest.mark.parametrize("results", [[], _empty_matches()])
def test_empty_results_say_nothing_was_found(results: list[SearchResult]) -> None:
    sentence = lead_in_sentence(results)

    assert "nothing" in sentence.lower()
    assert not any(character.isdigit() for character in sentence)
    assert "/" not in sentence
    assert extract_positions(sentence) == []
    assert opener_key(results) in OPENER_PHRASES


def test_opener_key_on_no_results_is_a_key_the_opener_table_holds() -> None:
    assert opener_key([]) in OPENER_PHRASES
    assert opener_key([]) == "empty"
