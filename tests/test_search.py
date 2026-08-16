import base64
import json
import sys
from pathlib import Path

import pytest

from tests.fakes import match_record, record_spawns
from tutor.tools.models import ContextLine, SearchBudget, SearchResult
from tutor.tools.search import search

FIXTURE_ROOT = Path(__file__).parent / "data" / "fixture_repo"

GOLDEN = [
    ("src/httpclient.py", 12),
    ("src/pool.py", 1),
    ("src/pool.py", 11),
    ("src/pool.py", 14),
    ("src/pool.py", 15),
    ("src/pool.py", 16),
    ("src/pool.py", 26),
    ("src/pool.py", 29),
    ("src/pool.py", 31),
    ("src/pool_helpers.py", 5),
    ("src/pool_helpers.py", 12),
    ("src/pool_helpers.py", 13),
]


def pairs(result: SearchResult) -> list[tuple[str, int]]:
    return [(match.path, match.line) for match in result.matches]


async def test_golden_pairs() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    assert pairs(result) == GOLDEN


async def test_repeated_search_is_byte_stable() -> None:
    first = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())
    second = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    assert first.model_dump_json() == second.model_dump_json()


async def test_paths_are_relative_and_sorted() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    for match in result.matches:
        assert not match.path.startswith("/")
        assert not Path(match.path).is_absolute()
    assert pairs(result) == sorted(pairs(result))


async def test_globs_scope_the_search() -> None:
    scoped = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())
    wide = await search("acquire", ["**/*.py"], FIXTURE_ROOT, SearchBudget())

    assert "generated/out.py" not in {match.path for match in scoped.matches}
    assert "generated/out.py" in {match.path for match in wide.matches}


async def test_leading_dot_slash_is_stripped() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    assert result.matches
    for match in result.matches:
        assert "./" not in match.path


async def test_union_fields_decode_and_invalid_utf8_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submatches = [{"match": {"text": "acquire"}, "start": 0, "end": 7}]
    valid_bytes = match_record("./src/b.py", 2, "", submatches)
    valid_bytes["data"]["lines"] = {"bytes": base64.b64encode(b"caf\xc3\xa9 acquire\n").decode()}
    invalid_line = match_record("./src/c.py", 3, "", submatches)
    invalid_line["data"]["lines"] = {"bytes": base64.b64encode(b"\xff\xfe acquire\n").decode()}
    invalid_path = match_record("", 4, "delta acquire\n", submatches)
    invalid_path["data"]["path"] = {"bytes": base64.b64encode(b"./src/\xff.py").decode()}
    payload = "".join(
        json.dumps(record) + "\n"
        for record in (
            match_record("./src/a.py", 1, "alpha acquire\n", submatches),
            valid_bytes,
            invalid_line,
            invalid_path,
        )
    )
    record_spawns(
        monkeypatch, [sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload]
    )

    result = await search("acquire", ["src/*.py"], FIXTURE_ROOT, SearchBudget())

    assert [(match.path, match.line, match.text) for match in result.matches] == [
        ("src/a.py", 1, "alpha acquire"),
        ("src/b.py", 2, "caf\u00e9 acquire"),
    ]


async def test_oversized_and_truncated_reach_the_result() -> None:
    oversized = await search(
        "B", ["vendor/*.py"], FIXTURE_ROOT, SearchBudget(max_record_bytes=1000)
    )

    assert oversized.oversized is True
    assert oversized.truncated is False
    assert [match.path for match in oversized.matches] == ["vendor/small.py"]

    truncated = await search(
        "WIDE_ROW", ["src/wide.py"], FIXTURE_ROOT, SearchBudget(max_bytes=12000)
    )

    assert truncated.truncated is True


async def test_no_match_gives_an_empty_result() -> None:
    result = await search("zzznomatch", ["**/*.py"], FIXTURE_ROOT, SearchBudget())

    assert result.matches == []
    assert result.truncated is False
    assert result.oversized is False
    assert result.byte_count == 0
    assert result.tool == "search"
    assert result.query == "zzznomatch"
    assert result.globs == ["**/*.py"]


async def test_match_text_has_no_trailing_newline() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    assert result.matches
    for match in result.matches:
        assert not match.text.endswith("\n")


GOLDEN_CONTEXT = {
    ("src/httpclient.py", 12): ([10, 11], [13, 14]),
    ("src/pool.py", 1): ([], [2, 3]),
    ("src/pool.py", 11): ([9, 10], [12, 13]),
    ("src/pool.py", 14): ([], []),
    ("src/pool.py", 15): ([], [17]),
    ("src/pool.py", 16): ([], [18]),
    ("src/pool.py", 26): ([24, 25], [27, 28]),
    ("src/pool.py", 29): ([], [30]),
    ("src/pool.py", 31): ([], []),
    ("src/pool_helpers.py", 5): ([3, 4], [6, 7]),
    ("src/pool_helpers.py", 12): ([10, 11], []),
    ("src/pool_helpers.py", 13): ([], []),
}


async def test_first_line_match_has_no_before_context() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    first = next(m for m in result.matches if (m.path, m.line) == ("src/pool.py", 1))
    assert first.before == []
    assert [line.line for line in first.after] == [2, 3]


async def test_last_line_match_has_no_after_context() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    last_of_pool = next(m for m in result.matches if (m.path, m.line) == ("src/pool.py", 31))
    last_of_helpers = next(
        m for m in result.matches if (m.path, m.line) == ("src/pool_helpers.py", 13)
    )
    assert last_of_pool.after == []
    assert last_of_helpers.after == []


async def test_overlapping_windows_do_not_duplicate_context() -> None:
    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget())

    earlier = next(m for m in result.matches if (m.path, m.line) == ("src/pool.py", 11))
    later = next(m for m in result.matches if (m.path, m.line) == ("src/pool.py", 14))
    assert [line.line for line in earlier.after] == [12, 13]
    assert later.before == []

    assert {
        (match.path, match.line): (
            [line.line for line in match.before],
            [line.line for line in match.after],
        )
        for match in result.matches
    } == GOLDEN_CONTEXT

    seen: list[tuple[str, int]] = []
    for match in result.matches:
        for line in match.before + match.after:
            assert not line.text.endswith("\n")
            seen.append((match.path, line.line))
    assert len(seen) == len(set(seen))
    assert set(seen).isdisjoint(set(pairs(result)))


async def test_zero_context_lines_reaches_the_child_and_yields_empty_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = record_spawns(monkeypatch, None)

    result = await search("acquire", ["src/**/*.py"], FIXTURE_ROOT, SearchBudget(context_lines=0))

    assert spawned.argv[spawned.argv.index("--context") + 1] == "0"
    assert pairs(result) == GOLDEN
    for match in result.matches:
        assert match.before == []
        assert match.after == []


async def test_context_line_numbers_come_from_the_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        "type": "context",
        "data": {
            "path": {"text": "./src/a.py"},
            "lines": {"text": "eight\n"},
            "line_number": 8,
            "absolute_offset": 0,
            "submatches": [],
        },
    }
    after = {
        "type": "context",
        "data": {
            "path": {"text": "./src/a.py"},
            "lines": {"text": "twelve\n"},
            "line_number": 12,
            "absolute_offset": 0,
            "submatches": [],
        },
    }
    other_file = {
        "type": "context",
        "data": {
            "path": {"text": "./src/b.py"},
            "lines": {"text": "b eleven\n"},
            "line_number": 11,
            "absolute_offset": 0,
            "submatches": [],
        },
    }
    hit = match_record(
        "./src/a.py", 10, "ten acquire\n", [{"match": {"text": "acquire"}, "start": 4, "end": 11}]
    )
    payload = "".join(json.dumps(record) + "\n" for record in (before, hit, after, other_file))
    record_spawns(
        monkeypatch, [sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", payload]
    )

    result = await search("acquire", ["src/*.py"], FIXTURE_ROOT, SearchBudget())

    assert len(result.matches) == 1
    match = result.matches[0]
    assert (match.path, match.line) == ("src/a.py", 10)
    assert match.before == [ContextLine(line=8, text="eight")]
    assert match.after == [ContextLine(line=12, text="twelve")]
