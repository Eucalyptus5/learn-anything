import logging
from typing import NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tutor.visuals import (
    SCENE_ID,
    LessonAck,
    LessonCheckpoint,
    LessonSynced,
    SceneCue,
    SceneStatus,
    StepCue,
    StepLine,
)

logger = logging.getLogger(__name__)

OPENING_TEXT = "(the lesson begins)"


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show: str = Field(
        max_length=300,
        description=(
            "Required on every step, including a step that asks. One visible change to the "
            "picture, with the numbers and the case."
        ),
    )
    ask: str = Field(
        default="",
        max_length=300,
        description=(
            "Optional. A question for the learner to predict or explain before this step "
            "appears, answered by this step's picture; empty when they just watch."
        ),
    )


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

    id: str = Field(
        pattern=SCENE_ID, description="Short, unique, lowercase: letters, digits and hyphens."
    )
    # the models leave a property named title out of their calls; stored plans keep the name
    title: str = Field(
        alias="heading",
        max_length=80,
        description="Required. The scene's title, under eight words.",
    )
    show: str = Field(
        max_length=600, description="Required. What the picture must show as a whole."
    )
    steps: list[Step] = Field(
        min_length=3, max_length=5, description="Three to five steps, never more."
    )


class LessonPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = Field(
        max_length=800,
        description=(
            "The learner in a few sentences: only what they said and what follows directly from it."
        ),
    )
    scenes: list[Scene] = Field(
        min_length=1,
        max_length=12,
        description="The lesson in teaching order, one to twelve scenes.",
    )

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [scene.id for scene in self.scenes]
        if len(set(ids)) != len(ids):
            raise ValueError("scene ids repeat")
        return self


class Cursor(BaseModel):
    scene: int = Field(default=0, ge=0)
    step: int = Field(default=0, ge=0)


class BuiltScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(pattern=SCENE_ID)
    version: int = Field(ge=1)
    say: list[StepLine] = Field(min_length=3, max_length=5)
    html: str = Field(min_length=1, max_length=200000)


def rerun_conflict(old: LessonPlan, new: LessonPlan, protected: int) -> str | None:
    if len(new.scenes) < protected:
        return "committed_removed"
    for position in range(protected):
        if new.scenes[position] != old.scenes[position]:
            return "committed_changed"
    return None


def splice_rerun(old: LessonPlan, new: LessonPlan, protected: int) -> LessonPlan | str:
    prefix = old.scenes[:protected]
    drawn = {scene.id for scene in prefix}
    tail = [scene for scene in new.scenes if scene.id not in drawn]
    try:
        return LessonPlan.model_validate(
            {**new.model_dump(exclude={"scenes"}), "scenes": [*prefix, *tail]}
        )
    except ValidationError:
        return "splice_invalid"


class SentCue(NamedTuple):
    cue_id: int
    tag: StepCue | SceneCue
    scene_id: str | None
    revision: int
    to: Cursor


def _spelled(tag: StepCue | SceneCue) -> str:
    return f"<{tag.kind} {tag.n}>"


class LessonState:
    def __init__(self) -> None:
        self.plan: LessonPlan | None = None
        self.acked = Cursor()
        self.revision = 0
        self.sent: list[SentCue] = []
        self.checkpoint: LessonCheckpoint | None = None
        self.dropped: list[str] = []
        self.committed: set[str] = set()
        self.built: dict[str, BuiltScene] = {}
        self.failed: set[str] = set()
        self.resync = False
        self.opened = False
        self.first_answer_done = False
        self.planned_through: str | None = None
        self._cue_id = 0
        self._stale: set[int] = set()

    def adopt(self, plan: LessonPlan) -> None:
        self.plan = plan

    def accept(self, plan: LessonPlan, shown: int, exposed: int) -> str | None:
        if self.plan is None:
            self.adopt(plan)
            return None
        live = max(self.protected_count(), exposed)
        note = rerun_conflict(self.plan, plan, shown)
        if note is None and rerun_conflict(self.plan, plan, live) is not None:
            note = "stale_prefix"
        if note is not None:
            logger.info("lesson.prefix_touched reason=%s", note)
        spliced = splice_rerun(self.plan, plan, max(shown, live))
        if isinstance(spliced, str):
            return spliced
        self.adopt(spliced)
        return None

    def scene_at(self, n: int) -> Scene | None:
        if self.plan is None or not 1 <= n <= len(self.plan.scenes):
            return None
        return self.plan.scenes[n - 1]

    def position(self) -> Cursor:
        return self.sent[-1].to if self.sent else self.acked

    def current(self) -> Scene | None:
        return self.scene_at(max(self.position().scene, 1))

    def next_scene(self) -> Scene | None:
        return self.scene_at(max(self.position().scene, 1) + 1)

    def step_tag(self, n: int) -> str | None:
        at = self.position()
        scene = self.scene_at(at.scene)
        if self.plan is None:
            reason = "no_plan"
        elif scene is None:
            reason = "not_open"
        elif n <= at.step:
            reason = "not_rising"
        elif n > len(scene.steps):
            reason = "past_end"
        else:
            self._accept(StepCue(n=n), scene.id, Cursor(scene=at.scene, step=n))
            return None
        logger.info(
            "step.dropped scene_id=%s n=%d reason=%s",
            None if scene is None else scene.id,
            n,
            reason,
        )
        self.dropped.append(f"<step {n}>: {reason}")
        return reason

    def scene_tag(self, n: int) -> str | None:
        at = self.position()
        if self.plan is None:
            reason = "no_plan"
        elif n != at.scene + 1:
            reason = "not_next"
        elif n > len(self.plan.scenes):
            reason = "past_end"
        else:
            here = self.scene_at(at.scene)
            cue = SceneCue(n=n, scene_id=self.plan.scenes[n - 1].id)
            self._accept(cue, None if here is None else here.id, Cursor(scene=n, step=1))
            return None
        logger.info("scene.dropped n=%d reason=%s", n, reason)
        self.dropped.append(f"<scene {n}>: {reason}")
        return reason

    def _accept(self, tag: StepCue | SceneCue, scene_id: str | None, to: Cursor) -> None:
        self._cue_id += 1
        self.sent.append(SentCue(self._cue_id, tag, scene_id, self.revision + len(self.sent), to))

    def _at(self, scene_id: str | None, step: int) -> Cursor | None:
        if scene_id is None:
            return Cursor()
        for n, scene in enumerate(self.plan.scenes if self.plan else [], start=1):
            if scene.id == scene_id:
                return Cursor(scene=n, step=step)
        logger.warning("lesson.unknown_scene scene_id=%s", scene_id)
        return None

    def acknowledge(self, ack: LessonAck) -> None:
        if ack.cue_id in self._stale:
            self._stale.discard(ack.cue_id)
            return
        if not self.sent or self.sent[0].cue_id != ack.cue_id:
            logger.info("lesson.ack_unknown cue_id=%d outcome=%s", ack.cue_id, ack.outcome)
            return
        if ack.outcome == "fired":
            at = self._at(ack.scene_id, ack.step)
            if at is None:
                return
            self.sent.pop(0)
            self.acked, self.revision = at, ack.revision
            return
        head, *later = self.sent
        if ack.outcome == "failed":
            self.resync = True
        self.sent = []
        self._stale.update(cue.cue_id for cue in later)
        if ack.reason != "barrier":
            self.dropped.append(f"{_spelled(head.tag)}: {ack.reason}")
            self.dropped.extend(f"{_spelled(cue.tag)}: stale_revision" for cue in later)

    def synced(self, message: LessonSynced) -> None:
        at = self._at(message.scene_id, message.step)
        if at is None:
            return
        self.sent = []
        self._stale.clear()
        self.resync = False
        self.acked, self.revision = at, message.revision

    def retain(self, message: LessonCheckpoint) -> None:
        at = self._at(message.scene_id, message.step)
        if at is None or message.revision < self.revision:
            logger.info("lesson.checkpoint_ignored revision=%d", message.revision)
            return
        if (at, message.revision) != (self.acked, self.revision):
            self.resync = True
        self.checkpoint = message
        self.acked, self.revision = at, message.revision

    def forget(self) -> None:
        self.sent = []
        self._stale.clear()

    def protected_count(self) -> int:
        if self.plan is None:
            return 0
        committed = [n for n, s in enumerate(self.plan.scenes, start=1) if s.id in self.committed]
        return max([self.position().scene, *committed])

    def next_to_build(self) -> Scene | None:
        if self.plan is None:
            return None
        first = max(self.acked.scene, 1)
        for position in range(first, min(first + 1, len(self.plan.scenes)) + 1):
            scene = self.plan.scenes[position - 1]
            if scene.id not in self.committed:
                return scene
        return None

    def being_taught(self, scene_id: str) -> bool:
        if self.plan is None:
            return False
        opened = self.plan.scenes[: self.position().scene]
        return any(scene.id == scene_id for scene in opened)

    def statuses(self) -> tuple[list[SceneStatus], str | None]:
        if self.plan is None:
            return [], None
        rows: list[SceneStatus] = []
        for position, scene in enumerate(self.plan.scenes, start=1):
            if position < self.acked.scene:
                status = "done"
            elif scene.id in self.failed:
                status = "failed"
            elif scene.id in self.built:
                status = "built"
            elif scene.id in self.committed:
                status = "building"
            else:
                status = "planned"
            rows.append(SceneStatus(id=scene.id, title=scene.title, status=status))
        shown = self.scene_at(self.acked.scene)
        return rows, None if shown is None else shown.id
