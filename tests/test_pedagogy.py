import json

import pytest

from tutor.pedagogy import PedagogyState, Phase, TurnOutcome, parse_outcome
from tutor.prompt import MAX_GLOBS
from tutor.tools.models import Position

SETTLING = [
    Position(path="src/pool.py", line=11),
    Position(path="src/pool.py", line=12),
]

MALFORMED = [
    "",
    "the engineer explained it back correctly",
    "{signal: misconception}",
    "[1, 2, 3]",
    '"misconception"',
    '{"signal": "sure_thing"}',
    '{"signal": 7}',
    '{"signal": "misconception", "settling_positions": [{"line": 11}]}',
    '{"settling_positions": "src/pool.py"}',
]


def settling(*paths: str) -> str:
    positions = [{"path": path, "line": n} for n, path in enumerate(paths, start=11)]
    return json.dumps({"signal": "misconception", "settling_positions": positions})


BAD_SCOPES = [
    settling(""),
    settling(" "),
    settling("!src/pool.py"),
    settling(*[f"src/pool_{n}.py" for n in range(MAX_GLOBS + 1)]),
]


def teaching() -> PedagogyState:
    state = PedagogyState()
    assert state.phase is Phase.TEACH
    return state


def exploring() -> PedagogyState:
    state = teaching()
    assert state.advance(TurnOutcome(signal="covered")) is Phase.EXPLORE
    return state


def questioning() -> PedagogyState:
    state = exploring()
    assert state.advance(TurnOutcome(signal="covered")) is Phase.REVERSE_FEYNMAN
    return state


def test_teach_moves_to_explore_once_the_subsystem_is_covered() -> None:
    state = teaching()

    assert state.advance(TurnOutcome(signal="covered")) is Phase.EXPLORE
    assert state.phase is Phase.EXPLORE


def test_explore_moves_to_reverse_feynman_once_the_files_are_covered() -> None:
    state = exploring()

    assert state.advance(TurnOutcome(signal="covered")) is Phase.REVERSE_FEYNMAN
    assert state.phase is Phase.REVERSE_FEYNMAN


def test_reverse_feynman_returns_to_teach_on_a_correct_explanation() -> None:
    state = questioning()

    assert state.advance(TurnOutcome(signal="correct")) is Phase.TEACH
    assert state.phase is Phase.TEACH


def test_reverse_feynman_returns_to_explore_on_a_misconception() -> None:
    state = questioning()

    assert state.advance(TurnOutcome(signal="misconception")) is Phase.EXPLORE
    assert state.phase is Phase.EXPLORE


def test_explore_holds_on_a_follow_up_question() -> None:
    state = exploring()

    assert state.advance(TurnOutcome(signal="follow_up")) is Phase.EXPLORE
    assert state.advance(TurnOutcome(signal="follow_up")) is Phase.EXPLORE


def test_misconception_carries_the_settling_positions_into_the_next_explore_turn() -> None:
    state = questioning()

    state.advance(TurnOutcome(signal="misconception", settling_positions=SETTLING))

    assert state.settling_positions == SETTLING
    assert state.advance(TurnOutcome(signal="follow_up")) is Phase.EXPLORE
    assert state.settling_positions == SETTLING


def test_settling_positions_clear_when_the_phase_moves_on() -> None:
    state = questioning()
    state.advance(TurnOutcome(signal="misconception", settling_positions=SETTLING))

    assert state.advance(TurnOutcome(signal="covered")) is Phase.REVERSE_FEYNMAN
    assert state.settling_positions == []


def test_a_correct_explanation_leaves_no_settling_scope_behind() -> None:
    state = questioning()

    assert state.advance(TurnOutcome(signal="correct", settling_positions=SETTLING)) is Phase.TEACH
    assert state.settling_positions == []


def test_a_parsed_model_outcome_drives_the_transition() -> None:
    state = questioning()
    text = (
        '{"signal": "misconception", "settling_positions": [{"path": "src/pool.py", "line": 11}]}'
    )

    assert state.advance(parse_outcome(text)) is Phase.EXPLORE
    assert state.settling_positions == [Position(path="src/pool.py", line=11)]


def test_extra_model_fields_do_not_defeat_the_parse() -> None:
    state = questioning()

    assert state.advance(parse_outcome('{"signal": "correct", "confidence": 0.9}')) is Phase.TEACH


@pytest.mark.parametrize("phase", list(Phase))
@pytest.mark.parametrize("text", [*MALFORMED, *BAD_SCOPES])
def test_malformed_model_text_leaves_the_phase_unchanged(text: str, phase: Phase) -> None:
    state = PedagogyState(phase=phase)

    assert state.advance(parse_outcome(text)) is phase
    assert state.phase is phase


@pytest.mark.parametrize("text", BAD_SCOPES)
def test_an_unusable_settling_path_neutralizes_the_whole_outcome(text: str) -> None:
    outcome = parse_outcome(text)

    assert outcome.signal is None
    assert outcome.settling_positions == []
    state = questioning()
    assert state.advance(outcome) is Phase.REVERSE_FEYNMAN
    assert state.settling_positions == []


def test_the_settling_scope_admits_max_globs_distinct_paths() -> None:
    paths = [f"src/pool_{n}.py" for n in range(MAX_GLOBS)]
    state = questioning()

    assert state.advance(parse_outcome(settling(*paths))) is Phase.EXPLORE
    assert [position.path for position in state.settling_positions] == paths


def test_the_settling_cap_counts_distinct_paths_not_positions() -> None:
    paths = ["src/pool.py"] * MAX_GLOBS + ["src/lease.py"]
    state = questioning()

    assert state.advance(parse_outcome(settling(*paths))) is Phase.EXPLORE
    assert [position.path for position in state.settling_positions] == paths
    assert len(state.settling_positions) == MAX_GLOBS + 1


def test_malformed_model_text_leaves_the_settling_scope_in_place() -> None:
    state = questioning()
    state.advance(TurnOutcome(signal="misconception", settling_positions=SETTLING))

    assert state.advance(parse_outcome("nice try")) is Phase.EXPLORE
    assert state.settling_positions == SETTLING


def test_prompt_directive_follows_the_phase() -> None:
    state = teaching()

    teach = state.prompt_directive()
    state.advance(TurnOutcome(signal="covered"))
    explore = state.prompt_directive()
    state.advance(TurnOutcome(signal="covered"))
    feynman = state.prompt_directive()

    assert len({teach, explore, feynman}) == 3
    assert all(directive.strip() and directive.isascii() for directive in (teach, explore, feynman))
    state.advance(TurnOutcome(signal="correct"))
    assert state.prompt_directive() == teach
