import importlib.util
from pathlib import Path

import pytest

from tutor.tools.models import ContextLine, SearchMatch, SearchResult
from tutor.tools.provenance import TurnRegistry, extract_positions

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_gate.py"
_spec = importlib.util.spec_from_file_location("bench_gate", SCRIPT)
bench_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_gate)

TABLE = [
    "The acquire method sits in src/pool.py line 11.",
    "The retry loop is in src/pool.py line 90.",
    "There are 3 callers of it.",
    "It runs on Python 3.12 without changes.",
    "It handles roughly 100 frames per second.",
    "The same shape appears in handler.php:412 upstream.",
    "Look in tutor/session for the loop.",
    "The pool hands out one connection at a time.",
]

EXPECTED = {
    "current": [
        "admitted",
        "withheld_bound",
        "withheld_bound",
        "admitted",
        "withheld_bound",
        "withheld_bound",
        "admitted",
        "admitted",
    ],
    "digits": [
        "admitted",
        "withheld_bound",
        "withheld_bound",
        "admitted",
        "withheld_bound",
        "withheld_bound",
        "admitted",
        "admitted",
    ],
    "pathshape": [
        "admitted",
        "withheld_bound",
        "withheld_bound",
        "withheld_pathshape",
        "withheld_bound",
        "withheld_bound",
        "withheld_pathshape",
        "admitted",
    ],
}


def _sample_result() -> SearchResult:
    return SearchResult(
        tool="search_code",
        query="acquire",
        globs=["**/*.py"],
        matches=[
            SearchMatch(
                path="src/pool.py",
                line=11,
                text="    def acquire(self):",
                before=[ContextLine(line=9, text=""), ContextLine(line=10, text="")],
                after=[ContextLine(line=12, text="        pass"), ContextLine(line=13, text="")],
            )
        ],
        truncated=False,
        oversized=False,
        byte_count=1,
    )


@pytest.fixture
def registry() -> TurnRegistry:
    registry = TurnRegistry()
    registry.open_turn("t1")
    registry.record("t1", _sample_result())
    return registry


@pytest.mark.parametrize("policy", ["current", "digits", "pathshape"])
def test_classify_over_the_eight_utterance_table(registry: TurnRegistry, policy: str) -> None:
    assert bench_gate.classify(TABLE, registry, "t1", "model", policy) == EXPECTED[policy]


def test_classify_reproduces_a_seam_dependent_withhold(registry: TurnRegistry) -> None:
    chunks = ["The queue reader sits on line", "ninety of that same file and it never blocks."]

    assert bench_gate.classify(chunks, registry, "t1", "model", "current") == [
        "admitted",
        "withheld_bound",
    ]
    assert extract_positions(chunks[1]) == []


def test_classify_keeps_the_seam_per_source(registry: TurnRegistry) -> None:
    bench_gate.classify(["The queue reader sits on line"], registry, "t1", "lead_in", "current")

    assert bench_gate.classify(
        ["ninety of that same file and it never blocks."], registry, "t1", "model", "current"
    ) == ["admitted"]


def test_unbound_tokens_over_the_table() -> None:
    digits, shapes = bench_gate.unbound_tokens(TABLE)

    assert set(digits) == set()
    assert set(shapes) == {"3.12", "tutor/session"}


def test_policies_are_the_three_named() -> None:
    assert bench_gate.POLICIES == ("current", "digits", "pathshape")


def test_turns_defaults_to_no_capture() -> None:
    args = bench_gate.build_parser().parse_args([])

    assert args.turns == 0
    assert args.corpus.parts[-2:] == ("captures", "gate_corpus.json")
