import pytest

from tutor.lesson import OPENING_TEXT, BuiltScene, Cursor, LessonPlan, Scene, Step, rerun_conflict

STEP = {"show": "The ratio axis from 0.5 to 2.0", "ask": ""}
ASKING = {"show": "The clip band at 0.8 and 1.2", "ask": "Where does the band sit?"}


def scene(n: int = 1, steps: int = 3) -> dict[str, object]:
    return {
        "id": f"scene-{n}",
        "title": f"Scene {n}",
        "show": "The surrogate against the ratio, epsilon 0.2",
        "steps": [STEP] * (steps - 1) + [ASKING],
    }


def plan(scenes: int = 2) -> dict[str, object]:
    return {
        "profile": "Knows policy gradients.",
        "scenes": [scene(n) for n in range(1, scenes + 1)],
    }


def test_a_plan_validates_with_the_bounds() -> None:
    parsed = LessonPlan.model_validate(plan())
    assert [s.id for s in parsed.scenes] == ["scene-1", "scene-2"]
    assert parsed.scenes[0].steps[2].ask == "Where does the band sit?"
    assert parsed.scenes[0].steps[0].ask == ""
    assert LessonPlan.model_validate(plan(12)).scenes[-1].id == "scene-12"


@pytest.mark.parametrize(
    "body",
    [
        {"profile": "p", "scenes": []},
        {"profile": "p", "scenes": [scene(n) for n in range(1, 14)]},
        {"profile": "p" * 801, "scenes": [scene()]},
        {"profile": "p", "scenes": [scene()], "settled": True},
        {"profile": "p", "scenes": [scene(), scene()]},
        {"scenes": [scene()]},
    ],
)
def test_a_plan_outside_its_bounds_is_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        LessonPlan.model_validate(body)


@pytest.mark.parametrize(
    "body",
    [
        {**scene(), "id": "Scene-1"},
        {**scene(), "id": "s" * 33},
        {**scene(), "title": "t" * 81},
        {**scene(), "show": "s" * 601},
        {**scene(), "steps": [STEP, STEP]},
        {**scene(), "steps": [STEP] * 6},
        {**scene(), "mode": "probe"},
    ],
)
def test_a_scene_outside_its_bounds_is_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Scene.model_validate(body)


@pytest.mark.parametrize(
    "body",
    [
        {"show": "s" * 301, "ask": ""},
        {"show": "s", "ask": "a" * 301},
        {"show": "s", "ask": "", "check": "c"},
        {"ask": ""},
    ],
)
def test_a_step_outside_its_bounds_is_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Step.model_validate(body)


def test_a_scene_loads_from_heading_or_title_and_stores_title() -> None:
    stored = scene()
    reply = {("heading" if key == "title" else key): value for key, value in stored.items()}
    assert Scene.model_validate(reply) == Scene.model_validate(stored)
    assert Scene.model_validate(reply).title == "Scene 1"
    assert list(Scene.model_validate(reply).model_dump()) == ["id", "title", "show", "steps"]
    assert "heading" in Scene.model_validate(stored).model_dump(by_alias=True)
    assert LessonPlan.model_validate(plan()).scenes[0].title == "Scene 1"
    with pytest.raises(ValueError):
        Scene.model_validate({**reply, "heading": "t" * 81})


def test_the_cursor_starts_at_nothing_and_never_goes_negative() -> None:
    assert Cursor().model_dump() == {"scene": 0, "step": 0}
    with pytest.raises(ValueError):
        Cursor(scene=-1, step=0)
    with pytest.raises(ValueError):
        Cursor(scene=0, step=-1)


def test_a_built_scene_holds_its_version_and_the_say_lines_the_builder_wrote() -> None:
    built = BuiltScene(scene_id="scene-1", version=1, say=["a", "b", "c"], html="<p>x</p>")
    assert built.say == ["a", "b", "c"] and built.version == 1
    good = {"scene_id": "scene-1", "version": 1, "say": ["a", "b", "c"], "html": "x"}
    for body in (
        {**good, "scene_id": "Bad"},
        {**good, "version": 0},
        {**good, "say": ["a", "b"]},
        {**good, "say": ["a"] * 6},
        {**good, "say": ["a", "", "c"]},
        {**good, "say": ["a", "b", "c" * 121]},
        {**good, "html": ""},
        {**good, "html": "x" * 200001},
        {**good, "steps": ["a", "b", "c"]},
        {key: value for key, value in good.items() if key != "version"},
    ):
        with pytest.raises(ValueError):
            BuiltScene.model_validate(body)


def test_the_opening_text_is_fixed() -> None:
    assert OPENING_TEXT == "(the lesson begins)" and OPENING_TEXT.isascii()


def test_a_rerun_may_not_touch_the_protected_prefix() -> None:
    old = LessonPlan.model_validate(plan(3))
    same = LessonPlan.model_validate(plan(3))
    assert rerun_conflict(old, same, 2) is None
    appended = LessonPlan.model_validate(plan(4))
    assert rerun_conflict(old, appended, 2) is None
    reshaped = LessonPlan.model_validate({**plan(2), "scenes": [scene(1), scene(3)]})
    assert rerun_conflict(old, reshaped, 1) is None
    retitled = plan(3)
    retitled["scenes"][1]["title"] = "Another"
    assert rerun_conflict(old, LessonPlan.model_validate(retitled), 2) == "committed_changed"
    restepped = plan(3)
    restepped["scenes"][0]["steps"][0] = {"show": "Other", "ask": ""}
    assert rerun_conflict(old, LessonPlan.model_validate(restepped), 1) == "committed_changed"
    dropped = LessonPlan.model_validate({**plan(1), "scenes": [scene(1)]})
    assert rerun_conflict(old, dropped, 2) == "committed_removed"
    inserted = LessonPlan.model_validate({**plan(3), "scenes": [scene(9), scene(1), scene(2)]})
    assert rerun_conflict(old, inserted, 1) == "committed_changed"
    assert rerun_conflict(old, inserted, 0) is None
