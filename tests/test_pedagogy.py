import json

import pytest
from pydantic import ValidationError

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
    assert state.advance(TurnOutcome(signal="covered")) is Phase.CONCRETE
    return state


def questioning() -> PedagogyState:
    state = exploring()
    assert state.advance(TurnOutcome(signal="covered")) is Phase.INTERROGATE
    return state


def test_teach_moves_to_concrete_once_the_mechanism_is_covered() -> None:
    state = teaching()

    assert state.advance(TurnOutcome(signal="covered")) is Phase.CONCRETE
    assert state.phase is Phase.CONCRETE


def test_concrete_moves_to_interrogate_once_it_is_covered() -> None:
    state = exploring()

    assert state.advance(TurnOutcome(signal="covered")) is Phase.INTERROGATE
    assert state.phase is Phase.INTERROGATE


def test_interrogate_returns_to_teach_on_a_correct_explanation() -> None:
    state = questioning()

    assert state.advance(TurnOutcome(signal="correct")) is Phase.TEACH
    assert state.phase is Phase.TEACH


def test_interrogate_returns_to_concrete_on_a_misconception() -> None:
    state = questioning()

    assert state.advance(TurnOutcome(signal="misconception")) is Phase.CONCRETE
    assert state.phase is Phase.CONCRETE


def test_concrete_holds_on_a_follow_up_question() -> None:
    state = exploring()

    assert state.advance(TurnOutcome(signal="follow_up")) is Phase.CONCRETE
    assert state.advance(TurnOutcome(signal="follow_up")) is Phase.CONCRETE


def test_misconception_carries_the_settling_positions_into_the_next_concrete_turn() -> None:
    state = questioning()

    state.advance(TurnOutcome(signal="misconception", settling_positions=SETTLING))

    assert state.settling_positions == SETTLING
    assert state.advance(TurnOutcome(signal="follow_up")) is Phase.CONCRETE
    assert state.settling_positions == SETTLING


def test_settling_positions_clear_when_the_phase_moves_on() -> None:
    state = questioning()
    state.advance(TurnOutcome(signal="misconception", settling_positions=SETTLING))

    assert state.advance(TurnOutcome(signal="covered")) is Phase.INTERROGATE
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

    assert state.advance(parse_outcome(text)) is Phase.CONCRETE
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
    assert state.advance(outcome) is Phase.INTERROGATE
    assert state.settling_positions == []


def test_the_settling_scope_admits_max_globs_distinct_paths() -> None:
    paths = [f"src/pool_{n}.py" for n in range(MAX_GLOBS)]
    state = questioning()

    assert state.advance(parse_outcome(settling(*paths))) is Phase.CONCRETE
    assert [position.path for position in state.settling_positions] == paths


def test_the_settling_cap_counts_distinct_paths_not_positions() -> None:
    paths = ["src/pool.py"] * MAX_GLOBS + ["src/lease.py"]
    state = questioning()

    assert state.advance(parse_outcome(settling(*paths))) is Phase.CONCRETE
    assert [position.path for position in state.settling_positions] == paths
    assert len(state.settling_positions) == MAX_GLOBS + 1


def test_malformed_model_text_leaves_the_settling_scope_in_place() -> None:
    state = questioning()
    state.advance(TurnOutcome(signal="misconception", settling_positions=SETTLING))

    assert state.advance(parse_outcome("nice try")) is Phase.CONCRETE
    assert state.settling_positions == SETTLING


def test_prompt_directive_follows_the_phase() -> None:
    state = teaching()

    teach = state.prompt_directive()
    state.advance(TurnOutcome(signal="covered"))
    concrete = state.prompt_directive()
    state.advance(TurnOutcome(signal="covered"))
    interrogate = state.prompt_directive()

    assert len({teach, concrete, interrogate}) == 3
    assert all(
        directive.strip() and directive.isascii() for directive in (teach, concrete, interrogate)
    )
    state.advance(TurnOutcome(signal="correct"))
    assert state.prompt_directive() == teach


def test_interrogate_returns_to_teach_when_the_learner_asks_to_be_told() -> None:
    state = PedagogyState(Phase.INTERROGATE)
    assert state.advance(TurnOutcome(signal="told")) is Phase.TEACH
    assert state.settling_positions == []
    assert state.settling == ""


def test_a_misconception_carries_the_settling_note_into_the_next_concrete_turn() -> None:
    state = PedagogyState(Phase.INTERROGATE)
    outcome = TurnOutcome(
        signal="misconception",
        settling="the ratio at 1.3 with epsilon 0.2 gives a flat objective",
    )
    assert state.advance(outcome) is Phase.CONCRETE
    assert state.settling == outcome.settling
    assert "flat objective" in state.prompt_directive()


def test_the_settling_note_clears_when_the_phase_moves_on() -> None:
    state = PedagogyState(Phase.INTERROGATE)
    state.advance(TurnOutcome(signal="misconception", settling="a note"))
    state.advance(TurnOutcome(signal="covered"))
    assert state.settling == ""
    assert "a note" not in state.prompt_directive()


def test_the_settling_note_is_capped() -> None:
    with pytest.raises(ValidationError):
        TurnOutcome(signal="misconception", settling="x" * 501)


def test_settling_positions_appear_in_the_concrete_directive() -> None:
    state = PedagogyState(Phase.INTERROGATE)
    state.advance(
        TurnOutcome(
            signal="misconception", settling_positions=[Position(path="src/pool.py", line=42)]
        )
    )
    assert "Search src/pool.py line 42 first" in state.prompt_directive()


def test_told_outside_interrogate_holds_the_phase() -> None:
    state = PedagogyState(Phase.TEACH)
    assert state.advance(TurnOutcome(signal="told")) is Phase.TEACH
