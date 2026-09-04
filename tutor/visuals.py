from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tutor.tools.models import Position
from tutor.tools.provenance import TurnRegistry
from tutor.transport import Connection


class DiagramPush(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["diagram.push"] = "diagram.push"
    id: str = Field(max_length=64)
    kind: Literal["flowchart", "sequence"]
    source: str = Field(max_length=8000)


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


VisualPayload = Annotated[
    DiagramPush | DiagramClear | SourceHighlight | AppPush, Field(discriminator="type")
]


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

    async def push(self, payload: VisualPayload) -> None:
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
