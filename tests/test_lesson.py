import logging

import pytest

from tutor.lesson import (
    NO_PLAN,
    OPENING_TEXT,
    BuiltScene,
    Cursor,
    LessonPlan,
    LessonState,
    Scene,
    SentCue,
    Step,
    lesson_block,
    rerun_conflict,
    splice_rerun,
)
from tutor.visuals import LessonAck, LessonCheckpoint, LessonSynced, SceneCue, StepCue

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


def test_a_splice_puts_the_drawn_scenes_back_from_the_current_plan() -> None:
    old = LessonPlan.model_validate(plan(3))
    reshown = plan(3)
    reshown["scenes"][0]["steps"][1] = {"show": "The ratio axis from 0.5 to 2", "ask": ""}
    reshown["scenes"][1]["show"] = "The surrogate against the ratio"
    reshown["scenes"][2]["title"] = "Retold"
    restored = LessonPlan.model_validate(
        {**plan(3), "scenes": [scene(1), scene(2), reshown["scenes"][2]]}
    )
    assert splice_rerun(old, LessonPlan.model_validate(reshown), 2) == restored
    dropped = LessonPlan.model_validate({**plan(2), "scenes": [scene(1), scene(3)]})
    assert splice_rerun(old, dropped, 2) == old
    inserted = {**plan(3), "scenes": [scene(9), {**scene(1), "title": "Other"}, scene(2)]}
    moved = LessonPlan.model_validate({**plan(3), "scenes": [scene(1), scene(9), scene(2)]})
    assert splice_rerun(old, LessonPlan.model_validate(inserted), 1) == moved


def test_a_splice_keeps_the_reply_tail_and_profile() -> None:
    old = LessonPlan.model_validate(plan(3))
    appended = plan(4)
    appended["scenes"][0]["title"] = "Other"
    longer = LessonPlan.model_validate(plan(4))
    assert splice_rerun(old, LessonPlan.model_validate(appended), 2) == longer
    profile = "Knows the clip, not why it is there."
    reprofiled = {**plan(3), "profile": profile}
    reprofiled["scenes"][0]["show"] = "The surrogate"
    kept = LessonPlan.model_validate({**plan(3), "profile": profile})
    assert splice_rerun(old, LessonPlan.model_validate(reprofiled), 1) == kept


def test_a_splice_past_twelve_scenes_is_invalid() -> None:
    old = LessonPlan.model_validate(plan(2))
    twelve = LessonPlan.model_validate({**plan(), "scenes": [scene(n) for n in range(3, 15)]})
    assert splice_rerun(old, twelve, 2) == "splice_invalid"
    assert splice_rerun(old, twelve, 1) == "splice_invalid"
    ten = LessonPlan.model_validate({**plan(), "scenes": [scene(n) for n in range(3, 13)]})
    assert splice_rerun(old, ten, 2) == LessonPlan.model_validate(plan(12))


def three_scenes() -> LessonState:
    state = LessonState()
    state.adopt(LessonPlan.model_validate(plan(3)))
    return state


def ack(
    cue_id: int,
    outcome: str = "fired",
    reason: str | None = None,
    scene_id: str | None = "scene-1",
    step: int = 1,
    revision: int = 1,
) -> LessonAck:
    return LessonAck(
        epoch=1,
        barrier=0,
        cue_id=cue_id,
        outcome=outcome,
        reason=reason,
        scene_id=scene_id,
        step=step,
        revision=revision,
    )


def opened() -> LessonState:
    state = three_scenes()
    assert state.scene_tag(1) is None
    state.acknowledge(ack(1))
    return state


def test_a_new_state_has_no_position_until_the_page_opens_a_scene() -> None:
    empty = LessonState()
    assert empty.plan is None and empty.acked == Cursor() and empty.revision == 0
    assert empty.current() is None and empty.statuses() == ([], None)
    assert empty.step_tag(2) == "no_plan" and empty.scene_tag(1) == "no_plan"
    assert empty.dropped == ["<step 2>: no_plan", "<scene 1>: no_plan"]
    state = three_scenes()
    assert state.acked == Cursor() and state.sent == [] and state.checkpoint is None
    assert state.current().id == "scene-1" and state.next_scene().id == "scene-2"
    assert state.step_tag(2) == "not_open"
    assert state.scene_tag(2) == "not_next"
    assert state.dropped == ["<step 2>: not_open", "<scene 2>: not_next"]


def test_scene_one_opens_by_its_tag_and_the_cursor_moves_only_on_the_ack() -> None:
    state = three_scenes()
    assert state.scene_tag(1) is None
    assert state.sent == [
        SentCue(1, SceneCue(n=1, scene_id="scene-1"), None, 0, Cursor(scene=1, step=1))
    ]
    assert state.acked == Cursor() and state.position() == Cursor(scene=1, step=1)
    assert state.step_tag(2) is None
    assert state.sent[-1] == SentCue(2, StepCue(n=2), "scene-1", 1, Cursor(scene=1, step=2))
    assert state.acked == Cursor()
    state.acknowledge(ack(1, revision=1))
    assert state.acked == Cursor(scene=1, step=1) and state.revision == 1
    assert [cue.cue_id for cue in state.sent] == [2]
    state.acknowledge(ack(2, step=2, revision=2))
    assert state.acked == Cursor(scene=1, step=2) and state.revision == 2 and state.sent == []


def test_a_step_tag_must_rise_past_what_was_acked_or_sent_and_stay_in_range(caplog) -> None:
    state = opened()
    with caplog.at_level(logging.INFO, logger="tutor.lesson"):
        assert state.step_tag(1) == "not_rising"
        assert state.step_tag(4) == "past_end"
        assert state.step_tag(3) is None
        assert state.step_tag(2) == "not_rising"
        assert state.step_tag(3) == "not_rising"
    assert state.acked == Cursor(scene=1, step=1)
    assert state.dropped == [
        "<step 1>: not_rising",
        "<step 4>: past_end",
        "<step 2>: not_rising",
        "<step 3>: not_rising",
    ]
    assert "step.dropped scene_id=scene-1 n=4 reason=past_end" in caplog.messages


def test_a_scene_tag_names_the_next_scene_only(caplog) -> None:
    state = opened()
    with caplog.at_level(logging.INFO, logger="tutor.lesson"):
        assert state.scene_tag(1) == "not_next"
        assert state.scene_tag(3) == "not_next"
        assert state.scene_tag(2) is None
        assert state.sent[-1].tag == SceneCue(n=2, scene_id="scene-2")
        assert state.sent[-1].scene_id == "scene-1"
        assert state.step_tag(1) == "not_rising"
        assert state.scene_tag(3) is None
        assert state.scene_tag(4) == "past_end"
    assert "scene.dropped n=4 reason=past_end" in caplog.messages


def test_each_cue_expects_the_acked_revision_plus_the_cues_ahead_of_it() -> None:
    state = opened()
    assert state.revision == 1
    assert state.step_tag(2) is None and state.step_tag(3) is None and state.scene_tag(2) is None
    assert [cue.revision for cue in state.sent] == [1, 2, 3]
    state.acknowledge(ack(2, step=2, revision=2))
    assert state.step_tag(2) is None
    assert state.sent[-1].scene_id == "scene-2" and state.sent[-1].revision == 4


def test_a_dropped_head_takes_every_later_cue_with_it(caplog) -> None:
    state = opened()
    assert state.step_tag(2) is None and state.step_tag(3) is None and state.scene_tag(2) is None
    with caplog.at_level(logging.INFO, logger="tutor.lesson"):
        state.acknowledge(ack(2, outcome="dropped", reason="range"))
        assert state.sent == [] and state.position() == Cursor(scene=1, step=1)
        state.acknowledge(ack(3, outcome="dropped", reason="stale_revision"))
        state.acknowledge(ack(4, outcome="dropped", reason="stale_revision"))
    assert not any(m.startswith("lesson.ack_unknown") for m in caplog.messages)
    assert state.dropped == [
        "<step 2>: range",
        "<step 3>: stale_revision",
        "<scene 2>: stale_revision",
    ]
    assert state.acked == Cursor(scene=1, step=1) and state.revision == 1 and not state.resync
    assert state.step_tag(2) is None and state.sent[-1].revision == 1


def test_a_failed_cue_waits_for_the_checkpoint_the_page_sends_after_it() -> None:
    state = opened()
    assert not state.resync
    assert state.step_tag(2) is None
    state.acknowledge(ack(2, outcome="failed", reason="runtime"))
    assert state.sent == [] and state.acked == Cursor(scene=1, step=1) and state.resync
    assert state.dropped == ["<step 2>: runtime"]
    restored = LessonCheckpoint(epoch=1, scene_id="scene-1", version=0, step=1, revision=2)
    state.retain(restored)
    assert state.checkpoint == restored and state.resync
    assert state.acked == Cursor(scene=1, step=1) and state.revision == 2
    older = LessonCheckpoint(epoch=1, scene_id="scene-2", version=0, step=1, revision=1)
    state.retain(older)
    assert state.checkpoint == restored and state.acked == Cursor(scene=1, step=1)
    state.forget()
    assert state.resync
    state.synced(
        LessonSynced(epoch=1, barrier=1, scene_id="scene-1", step=1, revision=2, last_cue=1)
    )
    assert not state.resync and state.acked == Cursor(scene=1, step=1)


def test_a_checkpoint_that_moves_acked_or_the_revision_is_a_restore_and_owes_a_sync() -> None:
    state = opened()
    state.retain(LessonCheckpoint(epoch=1, scene_id="scene-1", version=1, step=1, revision=1))
    assert not state.resync and state.acked == Cursor(scene=1, step=1)
    state.retain(LessonCheckpoint(epoch=1, scene_id="scene-1", version=1, step=1, revision=2))
    assert state.resync and state.revision == 2
    state.synced(
        LessonSynced(epoch=1, barrier=1, scene_id="scene-1", step=1, revision=2, last_cue=1)
    )
    assert not state.resync
    state.retain(LessonCheckpoint(epoch=1, scene_id="scene-2", version=0, step=1, revision=1))
    assert not state.resync and state.acked == Cursor(scene=1, step=1)
    state.retain(LessonCheckpoint(epoch=1, scene_id="scene-1", version=1, step=2, revision=2))
    assert state.resync and state.acked == Cursor(scene=1, step=2)


def test_the_barrier_clears_the_overlay_and_the_page_names_the_position() -> None:
    state = opened()
    assert state.step_tag(2) is None and state.step_tag(3) is None
    state.acknowledge(ack(2, step=2, revision=2))
    state.acknowledge(ack(3, outcome="dropped", reason="barrier", step=2, revision=2))
    state.synced(
        LessonSynced(epoch=1, barrier=1, scene_id="scene-1", step=2, revision=2, last_cue=2)
    )
    assert state.sent == [] and state.acked == Cursor(scene=1, step=2) and state.revision == 2
    assert state.dropped == []
    assert state.step_tag(3) is None and state.sent[-1].revision == 2


def test_synced_also_clears_cues_whose_chunk_never_played() -> None:
    state = opened()
    assert state.step_tag(2) is None and state.step_tag(3) is None
    state.synced(
        LessonSynced(epoch=1, barrier=1, scene_id="scene-1", step=1, revision=1, last_cue=1)
    )
    assert state.sent == [] and state.position() == Cursor(scene=1, step=1)


def test_an_ack_that_matches_no_head_moves_nothing(caplog) -> None:
    state = opened()
    assert state.step_tag(2) is None
    with caplog.at_level(logging.INFO, logger="tutor.lesson"):
        state.acknowledge(ack(9, step=3, revision=5))
        state.acknowledge(ack(2, scene_id="scene-9", step=2, revision=2))
        state.synced(
            LessonSynced(epoch=1, barrier=1, scene_id="scene-9", step=1, revision=4, last_cue=2)
        )
    assert state.acked == Cursor(scene=1, step=1) and state.revision == 1
    assert [cue.cue_id for cue in state.sent] == [2]
    assert "lesson.ack_unknown cue_id=9 outcome=fired" in caplog.messages
    assert caplog.messages.count("lesson.unknown_scene scene_id=scene-9") == 2


def test_forgetting_the_overlay_keeps_the_acked_position() -> None:
    state = opened()
    assert state.step_tag(2) is None
    state.forget()
    assert state.sent == [] and state.acked == Cursor(scene=1, step=1)
    assert state.step_tag(2) is None and state.sent[-1].revision == 1


def test_the_queue_builds_one_scene_ahead_of_the_acked_scene() -> None:
    state = three_scenes()
    assert state.next_to_build().id == "scene-1"
    state.committed.add("scene-1")
    assert state.next_to_build().id == "scene-2"
    state.committed.add("scene-2")
    assert state.next_to_build() is None
    assert state.scene_tag(1) is None and state.scene_tag(2) is None
    assert state.next_to_build() is None
    state.acknowledge(ack(1))
    assert state.next_to_build() is None
    state.acknowledge(ack(2, scene_id="scene-2", step=1, revision=2))
    assert state.next_to_build().id == "scene-3"
    assert LessonState().next_to_build() is None


def test_a_scene_opened_in_speech_is_being_taught_before_its_ack() -> None:
    fresh = three_scenes()
    assert not fresh.being_taught("scene-1") and not fresh.being_taught("scene-2")
    assert fresh.scene_tag(1) is None
    assert fresh.being_taught("scene-1") and fresh.acked == Cursor()
    state = opened()
    assert state.scene_tag(2) is None
    assert state.being_taught("scene-2") and state.being_taught("scene-1")
    assert not state.being_taught("scene-3") and not state.being_taught("scene-9")
    assert state.acked == Cursor(scene=1, step=1)


def test_statuses_follow_the_acknowledged_position() -> None:
    state = three_scenes()
    rows, current = state.statuses()
    assert [(r.id, r.status) for r in rows] == [
        ("scene-1", "planned"),
        ("scene-2", "planned"),
        ("scene-3", "planned"),
    ]
    assert current is None
    state.committed.add("scene-1")
    assert state.statuses()[0][0].status == "building"
    state.built["scene-1"] = BuiltScene(
        scene_id="scene-1", version=1, say=["a", "b", "c"], html="<p>x</p>"
    )
    assert state.statuses()[0][0].status == "built"
    state.committed.add("scene-2")
    state.failed.add("scene-2")
    assert state.statuses()[0][1].status == "failed"
    assert state.scene_tag(1) is None and state.scene_tag(2) is None
    assert state.statuses()[1] is None
    state.acknowledge(ack(1))
    assert state.statuses()[1] == "scene-1"
    state.acknowledge(ack(2, scene_id="scene-2", step=1, revision=2))
    rows, current = state.statuses()
    assert [r.status for r in rows] == ["done", "failed", "planned"] and current == "scene-2"


def test_a_rerun_keeps_the_larger_of_the_prefix_it_was_shown_and_the_live_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fresh = LessonState()
    assert fresh.accept(LessonPlan.model_validate(plan(3)), 0, 0) is None
    assert fresh.plan is not None
    state = three_scenes()
    state.committed.add("scene-1")
    reshaped = LessonPlan.model_validate({**plan(3), "scenes": [scene(1), scene(7), scene(8)]})
    assert state.accept(reshaped, 1, 0) is None and state.plan == reshaped
    assert state.scene_tag(1) is None and state.scene_tag(2) is None
    assert state.protected_count() == 2
    touched = LessonPlan.model_validate({**plan(3), "scenes": [scene(1), scene(9), scene(8)]})
    third = LessonPlan.model_validate({**plan(3), "scenes": [scene(1), scene(7), scene(5)]})
    shorter = LessonPlan.model_validate({**plan(1), "scenes": [scene(1)]})
    crowded = LessonPlan.model_validate({**plan(12), "scenes": [scene(n) for n in range(20, 32)]})
    with caplog.at_level(logging.INFO, logger="tutor.lesson"):
        assert state.accept(touched, 2, 0) is None
        assert [s.id for s in state.plan.scenes] == ["scene-1", "scene-7", "scene-9", "scene-8"]
        assert state.accept(third, 2, 3) is None
        assert [s.id for s in state.plan.scenes] == ["scene-1", "scene-7", "scene-9", "scene-5"]
        assert state.accept(shorter, 2, 0) is None
        assert [s.id for s in state.plan.scenes] == ["scene-1", "scene-7"]
        assert state.accept(crowded, 2, 0) == "splice_invalid"
        assert [s.id for s in state.plan.scenes] == ["scene-1", "scene-7"]
    assert [m for m in caplog.messages if m.startswith("lesson.prefix_touched")] == [
        "lesson.prefix_touched reason=committed_changed",
        "lesson.prefix_touched reason=stale_prefix",
        "lesson.prefix_touched reason=committed_removed",
        "lesson.prefix_touched reason=committed_changed",
    ]
    assert state.acked == Cursor()


def test_the_block_without_a_plan_asks_for_words_and_no_tags() -> None:
    text = lesson_block(LessonState(), learner_spoke=True, dropped=[])
    assert text == NO_PLAN
    assert "no lesson plan yet" in text and "write no tags" in text and text.isascii()


def test_the_block_lists_every_title_marks_the_current_scene_and_numbers_the_steps() -> None:
    state = opened()
    state.opened = True
    text = lesson_block(state, learner_spoke=True, dropped=[])
    assert text.startswith("Profile: Knows policy gradients.")
    assert "1. Scene 1 (now)\n2. Scene 2\n3. Scene 3" in text
    assert "Current scene, 1 of 3: Scene 1." in text
    assert (
        "1. The ratio axis from 0.5 to 2.0\n2. The ratio axis from 0.5 to 2.0\n"
        "3. The clip band at 0.8 and 1.2 (ask first, with no tag: Where does the band sit? "
        "Its tag waits for the answer.)"
    ) in text
    assert "Next scene, 2: Scene 2." in text
    assert "The page shows scene 1 at step 1 of 3." in text
    assert (
        "The next step, 2, does not ask: write <step 2> at the start of the sentence where" in text
    )
    assert "phase" not in text.lower() and text.isascii()


def test_the_block_directs_the_opening_and_a_learner_who_spoke_first() -> None:
    state = three_scenes()
    opening = lesson_block(state, learner_spoke=False, dropped=[])
    assert "There is no learner text" in opening and "Open scene one" in opening
    assert "write <scene 1> at the start of the sentence where it opens" in opening
    assert "<scene 1> opens step 1, so <step 1> is never written" in opening
    assert "The page has not opened a scene yet." in opening
    assert "1. Scene 1 (now)" in opening
    spoke = lesson_block(state, learner_spoke=True, dropped=[])
    assert "spoke before the lesson was ready" in spoke and "open scene one" in spoke
    assert "Answer what they said in a sentence" in spoke
    state.opened = True
    missed = lesson_block(state, learner_spoke=True, dropped=[])
    assert "Scene one is not open yet" in missed and "write <scene 1>" in missed


def test_the_block_asks_before_an_asking_step_and_closes_a_scene_and_the_lesson() -> None:
    state = opened()
    state.opened = True
    assert state.step_tag(2) is None
    state.acknowledge(ack(2, step=2, revision=2))
    text = lesson_block(state, learner_spoke=True, dropped=[])
    assert "The next step, 3, asks first: Where does the band sit?" in text
    assert "put it with no tag and stop" in text and "write <step 3>" in text
    assert "Its tag is held back until the learner has answered." in text
    assert "write <step 3> at the start of the sentence that explains what appears" in text
    assert "open the next scene in the same breath: write <scene 2>" in text
    assert "step tags in a new scene start at 2" in text
    assert "asked to be told" in text and "without questions" in text
    assert state.step_tag(3) is None
    state.acknowledge(ack(3, step=3, revision=3))
    done = lesson_block(state, learner_spoke=True, dropped=[])
    assert "This scene is done" in done and "write <scene 2>" in done and "start at 2" in done
    assert state.scene_tag(2) is None
    state.acknowledge(ack(4, scene_id="scene-2", step=1, revision=4))
    assert state.scene_tag(3) is None
    state.acknowledge(ack(5, scene_id="scene-3", step=1, revision=5))
    assert state.step_tag(3) is None
    state.acknowledge(ack(6, scene_id="scene-3", step=3, revision=6))
    last = lesson_block(state, learner_spoke=True, dropped=[])
    assert "last scene" in last and "close the lesson" in last and "<scene" not in last
    assert "3. Scene 3 (now)" in last and "Next scene" not in last


def test_the_ask_directive_in_the_last_scene_opens_no_scene() -> None:
    state = opened()
    state.opened = True
    assert state.scene_tag(2) is None and state.scene_tag(3) is None
    assert state.step_tag(2) is None
    text = lesson_block(state, learner_spoke=True, dropped=[])
    assert "3. Scene 3 (now)" in text
    assert "The next step, 3, asks first: Where does the band sit?" in text
    assert "put it with no tag and stop" in text
    assert "whose question you put with no tag before you stop, or until the scene ends." in text
    assert "<scene" not in text and "If the scene ends first" not in text


def test_the_block_reads_the_position_with_tags_already_sent() -> None:
    state = opened()
    state.opened = True
    assert state.step_tag(2) is None
    text = lesson_block(state, learner_spoke=True, dropped=[])
    assert "The page shows scene 1 at step 1 of 3." in text
    assert "The next step, 3, asks first" in text


def test_the_block_shows_the_drawn_say_lines_or_says_the_board_is_blank() -> None:
    state = opened()
    state.opened = True
    unbuilt = lesson_block(state, learner_spoke=True, dropped=[])
    assert "The board is blank for this scene" in unbuilt and "still write the step tags" in unbuilt
    state.built["scene-1"] = BuiltScene(
        scene_id="scene-1", version=1, say=["The axes", "The curve", "The band"], html="<p>x</p>"
    )
    text = lesson_block(state, learner_spoke=True, dropped=[])
    assert "what each step shows, as drawn:\n1. The axes\n2. The curve\n3. The band" in text
    assert "blank" not in text
    del state.built["scene-1"]
    state.failed.add("scene-1")
    failed = lesson_block(state, learner_spoke=True, dropped=[])
    assert "not coming" in failed and "blank" in failed


def test_the_block_names_the_tags_dropped_from_the_last_reply() -> None:
    state = opened()
    state.opened = True
    text = lesson_block(
        state, learner_spoke=True, dropped=["<step 4>: past_end", "<set>: unsupported"]
    )
    assert "Tags dropped from your last reply: <step 4>: past_end; <set>: unsupported." in text
    assert "Tags dropped" not in lesson_block(state, learner_spoke=True, dropped=[])
