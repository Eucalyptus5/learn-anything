import logging
from enum import StrEnum, auto
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from tutor.prompt import MAX_GLOBS
from tutor.tools.models import Position

logger = logging.getLogger(__name__)


class Phase(StrEnum):
    TEACH = auto()
    CONCRETE = auto()
    INTERROGATE = auto()


class TurnOutcome(BaseModel):
    signal: Literal["covered", "follow_up", "correct", "misconception", "told"] | None = None
    settling_positions: list[Position] = Field(default_factory=list)
    settling: str = Field(default="", max_length=500)

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
    (Phase.TEACH, "covered"): Phase.CONCRETE,
    (Phase.CONCRETE, "covered"): Phase.INTERROGATE,
    (Phase.INTERROGATE, "correct"): Phase.TEACH,
    (Phase.INTERROGATE, "told"): Phase.TEACH,
    (Phase.INTERROGATE, "misconception"): Phase.CONCRETE,
}

DIRECTIVES: dict[Phase, str] = {
    Phase.TEACH: (
        "Teach: introduce one mechanism, why it exists and how it works, one idea this turn, and "
        "end with a question the learner can answer from what you just said. Never ask about a "
        "term you have not introduced. Stay on the mechanism, not on notation."
    ),
    Phase.CONCRETE: (
        "Concrete: make the mechanism tangible. A worked example with numbers, one step of the "
        "derivation, a plot, a trace of one iteration; with a folder attached, the exact lines "
        "from this turn's search results. Never read syntax aloud; say what it does."
    ),
    Phase.INTERROGATE: (
        "Interrogate: stop lecturing and test. Pose an edge case, a failure mode or a limit and "
        "have the learner explain the mechanism back. At most two probes per gap, then explain. "
        "If they ask to be told, tell them. On a misconception, cut in, correct it in one "
        "sentence, and signal misconception with what settles it."
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
        self.settling = ""

    def advance(self, outcome: TurnOutcome) -> Phase:
        moved = TRANSITIONS.get((self.phase, outcome.signal), self.phase)
        if moved is not self.phase:
            settling = outcome.signal == "misconception"
            self.settling_positions = list(outcome.settling_positions) if settling else []
            self.settling = outcome.settling if settling else ""
        self.phase = moved
        return self.phase

    def prompt_directive(self) -> str:
        directive = DIRECTIVES[self.phase]
        if self.settling:
            directive += f" Settle the misconception here: {self.settling}"
        if self.settling_positions:
            scope = ", ".join(
                f"{p.path} line {p.line}" if p.line else p.path for p in self.settling_positions
            )
            directive += f" Search {scope} first and settle it from what you find."
        return directive
