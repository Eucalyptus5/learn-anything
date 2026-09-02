import logging
from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from tutor.prompt import MAX_GLOBS
from tutor.tools.models import Position

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    TEACH = auto()
    EXPLORE = auto()
    REVERSE_FEYNMAN = auto()


class TurnOutcome(BaseModel):
    signal: Literal["covered", "follow_up", "correct", "misconception"] | None = None
    settling_positions: list[Position] = Field(default_factory=list)

    @field_validator("settling_positions", mode="after")
    @classmethod
    def _bound_search_scope(cls, positions: list[Position]) -> list[Position]:
        for position in positions:
            if not position.path.strip():
                raise ValueError("a settling path is blank")
            if position.path.startswith("!"):
                raise ValueError("a settling path is negated")
        if len({position.path for position in positions}) > MAX_GLOBS:
            raise ValueError(f"more than {MAX_GLOBS} distinct settling paths")
        return positions


TRANSITIONS: dict[tuple[Phase, str | None], Phase] = {
    (Phase.TEACH, "covered"): Phase.EXPLORE,
    (Phase.EXPLORE, "covered"): Phase.REVERSE_FEYNMAN,
    (Phase.REVERSE_FEYNMAN, "correct"): Phase.TEACH,
    (Phase.REVERSE_FEYNMAN, "misconception"): Phase.EXPLORE,
}

DIRECTIVES: dict[Phase, str] = {
    Phase.TEACH: (
        "Introduce one subsystem: the design patterns behind it and the control flow through it, "
        "with a visual update in the same breath. Stay on the mechanism, not on syntax."
    ),
    Phase.EXPLORE: (
        "Walk the engineer into the files themselves: entry points, invariants, failure paths. "
        "Name exact positions from this turn's tool results and never read syntax aloud."
    ),
    Phase.REVERSE_FEYNMAN: (
        "Stop lecturing and test. Put a realistic edge case, race or failure mode to the engineer "
        "and have them explain the mechanism back. On a misconception, interrupt and correct."
    ),
}


def parse_outcome(text: str) -> TurnOutcome:
    try:
        return TurnOutcome.model_validate_json(text)
    except ValidationError:
        logger.info("turn_outcome_unparsed chars=%d", len(text))
        return TurnOutcome()


class PedagogyState:
    def __init__(self, phase: Phase = Phase.TEACH) -> None:
        self.phase = phase
        self.settling_positions: list[Position] = []

    def advance(self, outcome: TurnOutcome) -> Phase:
        moved = TRANSITIONS.get((self.phase, outcome.signal), self.phase)
        if moved is not self.phase:
            self.settling_positions = (
                list(outcome.settling_positions) if outcome.signal == "misconception" else []
            )
        self.phase = moved
        return self.phase

    def prompt_directive(self) -> str:
        return DIRECTIVES[self.phase]
