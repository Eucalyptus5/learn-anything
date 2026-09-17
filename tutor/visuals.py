from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from tutor.tools.models import Position
from tutor.tools.provenance import TurnRegistry
from tutor.transport import Connection


class DiagramPush(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["diagram.push"] = "diagram.push"
    id: str = Field(max_length=64)
    kind: Literal["flowchart", "sequence"]
    source: str = Field(max_length=8000)
    title: str = Field(max_length=80)


class DiagramClear(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["diagram.clear"] = "diagram.clear"


class SourceHighlight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["source.highlight"] = "source.highlight"
    path: str = Field(max_length=4096)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("end_line precedes start_line")
        return self


class AppPush(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["app.push"] = "app.push"
    id: str = Field(max_length=64)
    html: str = Field(max_length=64000)
    title: str = Field(max_length=80)


VisualPayload = Annotated[
    DiagramPush | DiagramClear | SourceHighlight | AppPush, Field(discriminator="type")
]


class TurnState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["state"] = "state"
    state: Literal["listening", "thinking", "speaking"]
    phase: Literal["teach", "concrete", "interrogate"]
    interrupted: bool = False


class Caption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["caption"] = "caption"
    turn_id: str = Field(max_length=32)
    text: str = Field(max_length=2000)
    lead_ms: int = Field(ge=0)


class LearnerText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["transcript"] = "transcript"
    turn_id: str = Field(max_length=32)
    text: str = Field(max_length=4000)


class VisualPending(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["visual.pending"] = "visual.pending"
    turn_id: str = Field(max_length=32)
    title: str = Field(max_length=80)


ChannelPayload = Annotated[
    DiagramPush
    | DiagramClear
    | SourceHighlight
    | AppPush
    | TurnState
    | Caption
    | LearnerText
    | VisualPending,
    Field(discriminator="type"),
]


class ThemeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["theme"] = "theme"
    theme: Literal["light", "dark"]


class SayMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["say"] = "say"
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


ClientMessage = Annotated[ThemeMessage | SayMessage, Field(discriminator="type")]
CLIENT_MESSAGE = TypeAdapter(ClientMessage)


class UngroundedVisual(Exception):
    pass


class VisualChannel:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._seq = 0
        self._registry: TurnRegistry | None = None
        self._turn_id = ""

    def set_grounding(self, registry: TurnRegistry, turn_id: str) -> None:
        self._registry = registry
        self._turn_id = turn_id

    async def push(self, payload: ChannelPayload) -> None:
        if isinstance(payload, SourceHighlight):
            path = payload.path.removeprefix("./")
            for line in range(payload.start_line, payload.end_line + 1):
                if self._registry is None:
                    raise UngroundedVisual(f"ungrounded highlight {path!r}:{line}")
                if not self._registry.known(self._turn_id, Position(path=path, line=line)):
                    raise UngroundedVisual(f"ungrounded highlight {path!r}:{line}")

        body = payload.model_dump(mode="json")
        self._seq += 1
        body["seq"] = self._seq
        await self._connection.send_json(body)
