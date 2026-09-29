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


SCENE_ID = r"^[a-z][a-z0-9-]{0,31}$"
StepLine = Annotated[str, Field(min_length=1, max_length=120)]


SceneId = Annotated[str, Field(pattern=SCENE_ID)]
STATUS = Literal["planned", "building", "built", "failed", "done"]
AckReason = Literal[
    "stale_epoch",
    "stale_barrier",
    "stale_revision",
    "barrier",
    "range",
    "invalid",
    "runtime",
    "layout",
    "owned",
]


class SceneStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: SceneId
    title: str = Field(max_length=80)
    status: STATUS


class LessonStatePush(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.state"] = "lesson.state"
    scenes: list[SceneStatus] = Field(max_length=12)
    current: SceneId | None


class LessonAttach(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.attach"] = "lesson.attach"
    epoch: int = Field(ge=1)


class StepCue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["step"] = "step"
    n: int = Field(ge=1, le=5)


class SceneCue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["scene"] = "scene"
    n: int = Field(ge=1, le=12)
    scene_id: SceneId


CueTag = Annotated[StepCue | SceneCue, Field(discriminator="kind")]


class LessonCue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.cue"] = "lesson.cue"
    epoch: int = Field(ge=1)
    barrier: int = Field(ge=0)
    cue_id: int = Field(ge=1)
    chunk_id: int = Field(ge=0)
    scene_id: SceneId | None
    revision: int = Field(ge=0)
    lead_ms: int = Field(ge=0)
    audio_ms: int = Field(ge=0)
    tag: CueTag


class LessonSync(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.sync"] = "lesson.sync"
    epoch: int = Field(ge=1)
    barrier: int = Field(ge=1)


class LessonAck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.ack"] = "lesson.ack"
    epoch: int = Field(ge=1)
    barrier: int = Field(ge=0)
    cue_id: int = Field(ge=1)
    outcome: Literal["fired", "dropped", "failed"]
    reason: AckReason | None
    scene_id: SceneId | None
    step: int = Field(ge=0)
    revision: int = Field(ge=0)

    @model_validator(mode="after")
    def _reason_matches_outcome(self) -> Self:
        if (self.outcome == "fired") != (self.reason is None):
            raise ValueError("a fired ack has no reason; a dropped or failed one has one")
        return self


class LessonSynced(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.synced"] = "lesson.synced"
    epoch: int = Field(ge=1)
    barrier: int = Field(ge=1)
    scene_id: SceneId | None
    step: int = Field(ge=0)
    revision: int = Field(ge=0)
    last_cue: int = Field(ge=0)


class LessonCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["lesson.checkpoint"] = "lesson.checkpoint"
    epoch: int = Field(ge=1)
    scene_id: SceneId | None
    version: int = Field(ge=0)
    step: int = Field(ge=0)
    revision: int = Field(ge=0)


class ScenePush(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["scene.push"] = "scene.push"
    scene_id: str = Field(pattern=SCENE_ID)
    title: str = Field(max_length=80)
    html: str = Field(min_length=1, max_length=200000)
    steps: list[StepLine] = Field(min_length=1, max_length=8)


class SceneShow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["scene.show"] = "scene.show"
    scene_id: str = Field(pattern=SCENE_ID)
    at: int = Field(ge=1)


class SceneStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["scene.step"] = "scene.step"
    scene_id: str = Field(pattern=SCENE_ID)
    n: int = Field(ge=1)
    lead_ms: int = Field(ge=0)


ChannelPayload = Annotated[
    DiagramPush
    | DiagramClear
    | SourceHighlight
    | AppPush
    | TurnState
    | Caption
    | LearnerText
    | ScenePush
    | SceneShow
    | SceneStep
    | LessonAttach
    | LessonCue
    | LessonSync
    | LessonStatePush,
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


class SceneReady(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["scene.ready"] = "scene.ready"
    scene_id: str = Field(pattern=SCENE_ID)
    ok: bool
    steps: int = Field(ge=0, le=8)
    error: str = Field(max_length=500)


ClientMessage = Annotated[
    ThemeMessage | SayMessage | SceneReady | LessonAck | LessonSynced | LessonCheckpoint,
    Field(discriminator="type"),
]
CLIENT_MESSAGE = TypeAdapter(ClientMessage)


class UngroundedVisual(Exception):
    pass


class VisualChannel:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._seq = 0
        self._unsent = 0
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
        self._unsent += 1
        body["seq"] = self._seq
        try:
            await self._connection.send_json(body)
        finally:
            self._unsent -= 1

    def push_nowait(self, payload: LessonSync) -> bool:
        # The page drops any seq at or below its last, so a sync never passes one still unsent.
        if self._unsent:
            return False
        body = payload.model_dump(mode="json")
        body["seq"] = self._seq + 1
        if not self._connection.send_json_nowait(body):
            return False
        self._seq += 1
        return True
