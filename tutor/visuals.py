from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
