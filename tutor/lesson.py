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
REPEAT = "repeat"


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


class ScriptChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1, le=5)
    question: bool
    text: str = Field(min_length=1)


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
        self.scripts: dict[str, list[ScriptChunk]] = {}
        self.asked: set[tuple[str, int]] = set()
        self._cue_id = 0
        self._stale: set[int] = set()
        self._turn_start = Cursor()
        self._learner_spoke = False

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

    def pending(self) -> int | None:
        at = self.position()
        scene = self.scene_at(at.scene)
        if scene is None or at.step >= len(scene.steps):
            scene, at = self.scene_at(at.scene + 1), Cursor(scene=at.scene + 1)
        if scene is None or not scene.steps[at.step].ask:
            return None
        return at.step + 1 if (scene.id, at.step + 1) in self.asked else None

    def done(self) -> bool:
        if self.plan is None:
            return False
        scenes = self.plan.scenes
        return self.position() == Cursor(scene=len(scenes), step=len(scenes[-1].steps))

    def begin_turn(self, learner_spoke: bool) -> None:
        self._turn_start = self.position()
        self._learner_spoke = learner_spoke

    def step_tag(self, n: int) -> str | None:
        at = self.position()
        scene = self.scene_at(at.scene)
        # Learner text can answer only the step right after where its turn began; any later
        # ask in the same reply was put after the learner last spoke.
        answered = at.step + 1 if self._learner_spoke and at == self._turn_start else at.step
        if self.plan is None:
            reason = "no_plan"
        elif scene is None:
            reason = "not_open"
        elif n == at.step:
            return REPEAT
        elif n < at.step:
            reason = "not_rising"
        elif n > len(scene.steps):
            reason = "past_end"
        elif any(step.ask for step in scene.steps[answered:n]):
            reason = "not_answered"
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


TAG_RULES = (
    "Tag every step you narrate, from <step 1>: write its <step n> at the start of the sentence "
    "where it appears, in every scene, one just opened included. For a step that asks, put its "
    "question with no tag and hold its <step n> until the learner has answered."
)
NO_PLAN = (
    "There is no lesson plan yet. Teach from the subject and the starting-from line in words, "
    "one idea at a time, and write no tags."
)
OPEN_SCENE_ONE = (
    "write <scene 1> at the start of the sentence where it opens; <scene 1> and its <step 1> "
    "open the same picture. Say what its first step shows, then go on through the steps "
    "that do not ask, writing <step n> at the start of the sentence where step n should appear, "
    "until the next step that asks; put that question and stop."
)


def _steps(steps: list[Step]) -> list[str]:
    return [
        f"{n}. {step.show} (ask first, with no tag: {step.ask} Its tag waits for the answer.)"
        if step.ask
        else f"{n}. {step.show}"
        for n, step in enumerate(steps, start=1)
    ]


def _directive(state: LessonState, scene: Scene, learner_spoke: bool) -> str:
    at = state.position()
    if at.scene == 0:
        if state.opened:
            lead = (
                "Scene one is not open yet. If the learner said something, answer it in a "
                "sentence; then open scene one: "
            )
        elif learner_spoke:
            lead = (
                "The learner spoke before the lesson was ready. Answer what they said in a "
                "sentence, then open scene one: "
            )
        else:
            lead = "This is the opening of the lesson. There is no learner text. Open scene one: "
        return f"{lead}{OPEN_SCENE_ONE} {TAG_RULES}"
    if at.step >= len(scene.steps):
        if state.next_scene() is None:
            return (
                "This was the last scene and it is done: close the lesson in a few sentences, "
                "and keep talking about whatever the learner raises."
            )
        return (
            f"This scene is done. Open the next scene in the same breath: write "
            f"<scene {at.scene + 1}> at the start of the sentence where it opens, say what its "
            "first step shows, then go on through its steps by the same rules; "
            f"<scene {at.scene + 1}> and its <step 1> open the same picture. {TAG_RULES}"
        )
    n = at.step + 1
    following = scene.steps[at.step]
    if following.ask:
        if state.next_scene() is None:
            onward = ", or until the scene ends."
        else:
            onward = (
                ". If the scene ends first, open the next scene in the same breath: write "
                f"<scene {at.scene + 1}> at the start of the sentence where it opens, say what "
                "its first step shows, then go on through its steps by the same rules, writing "
                f"the tag of every step you narrate; <scene {at.scene + 1}> and its <step 1> "
                "open the same picture."
            )
        return (
            f"The next step, {n}, asks first: {following.ask} Its tag is held back until the "
            "learner has answered. If that question is not yet in the conversation above, put "
            "it with no tag and stop. If the learner has just answered it, say in one sentence "
            f"whether they have it, then write <step {n}> at the start of the sentence that "
            "explains what appears, and continue through the steps that do not ask until the "
            "next one that does, whose question you put with no tag before you stop"
            f"{onward} If the learner has asked to be told, explain the rest of the scene "
            "without questions, writing each step's tag. On the learner's own question, answer "
            f"it in words first and return to the step. {TAG_RULES}"
        )
    return (
        f"The next step, {n}, does not ask: write <step {n}> at the start of the sentence where "
        "it should appear and explain it, and continue through the steps that do not ask until "
        "the next one that does; put that question and stop. On the learner's own question, "
        f"answer it in words first and return to the step. {TAG_RULES}"
    )


def lesson_block(state: LessonState, learner_spoke: bool, dropped: list[str]) -> str:
    plan = state.plan
    scene = state.current()
    if plan is None or scene is None:
        return NO_PLAN
    at = max(state.position().scene, 1)
    lines = [f"Profile: {plan.profile}", "Lesson, in order:"]
    for n, each in enumerate(plan.scenes, start=1):
        lines.append(f"{n}. {each.title}" + (" (now)" if n == at else ""))
    lines.append("")
    lines.append(
        f"Current scene, {at} of {len(plan.scenes)}: {scene.title}. "
        f"As a whole it shows: {scene.show}"
    )
    lines.append("Its steps:")
    lines.extend(_steps(scene.steps))
    built = state.built.get(scene.id)
    if built is not None:
        lines.append("The picture is drawn; what each step shows, as drawn:")
        lines.extend(f"{n}. {say}" for n, say in enumerate(built.say, start=1))
    elif scene.id in state.failed:
        lines.append(
            "The picture for this scene is not coming; the board stays blank under its title. "
            "Teach it in words and still write the step tags."
        )
    else:
        lines.append(
            "The board is blank for this scene: its picture is not drawn yet. Teach it in words "
            "and still write the step tags."
        )
    following = state.next_scene()
    if following is not None:
        lines.append(f"Next scene, {at + 1}: {following.title}. It shows: {following.show}")
        lines.extend(_steps(following.steps))
    lines.append("")
    shown = state.scene_at(state.acked.scene)
    if shown is None:
        lines.append("The page has not opened a scene yet.")
    else:
        lines.append(
            f"The page shows scene {state.acked.scene} at step {state.acked.step} "
            f"of {len(shown.steps)}."
        )
    if dropped:
        lines.append(f"Tags dropped from your last reply: {'; '.join(dropped)}.")
    lines.append(_directive(state, scene, learner_spoke))
    return "\n".join(lines)
