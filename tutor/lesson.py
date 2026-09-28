from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tutor.visuals import SCENE_ID, StepLine

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
