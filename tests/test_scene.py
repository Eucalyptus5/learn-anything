import asyncio
import json
import logging
import re
from pathlib import Path

from tests.test_session import FakeReasoning, visual_call
from tutor.brief import VisualBrief
from tutor.reasoning import TurnChunk
from tutor.scene import (
    CONTRACT,
    EMPTY_REPLY,
    GUIDE,
    NO_TOOL_CALL,
    SCENE_TOOL,
    SCENE_TOOLS,
    SceneDraft,
    run_scene_build,
    scene_prompt,
)

BRIEF = VisualBrief(kind="app", title="Clipped objective", show="surrogate vs ratio, epsilon 0.2")
DRAFT = {"html": "<!doctype html><p>x</p>", "steps": ["The axes", "The curve", "The band"]}
DRAFT_ARGUMENTS = json.dumps(DRAFT)
VARIABLES = (
    "--ink",
    "--ink-2",
    "--ink-3",
    "--accent",
    "--surface",
    "--hair",
    "--sans",
    "--serif",
    "--mono",
)


def call(arguments: str, name: str = SCENE_TOOL) -> TurnChunk:
    return visual_call(name, arguments, "call-s")


def test_the_guide_is_ascii_bounded_and_names_every_variable_the_helper_defines() -> None:
    path = Path("tutor/prompts/scene_guide.md")
    text = path.read_text()
    assert text == GUIDE
    assert text.isascii()
    assert len(text.encode()) < 12000
    for variable in VARIABLES:
        assert f"var({variable})" in text, variable
    for literal in ("lesson.scene({", "gsap.timeline({ paused: true })", 'addLabel("step-1")'):
        assert text.count(literal) >= 2, literal
    assert not re.search(r"#[0-9a-fA-F]{6}\b", text)
    assert "eight thousand" not in text and "quickly" not in text


def test_the_prompt_is_the_contract_the_guide_the_subject_and_the_brief() -> None:
    prompt = scene_prompt("PPO", BRIEF, "dark")

    assert prompt.system.startswith(CONTRACT.split("{", 1)[0])
    assert "The page is dark" in prompt.system
    assert prompt.system.endswith(GUIDE)
    assert prompt.history == [] and prompt.tool_context == [] and prompt.tool_exchange == []
    assert prompt.user_text == (
        "Subject: PPO\nTitle: Clipped objective\nShow: surrogate vs ratio, epsilon 0.2"
    )
    assert prompt.system.isascii() and "http" not in prompt.system
    for tag in (
        '<script src="/lesson.js"></script>',
        '<script src="/vendor/gsap.min.js"></script>',
        "/vendor/plotly.min.js",
        "/vendor/katex/katex.min.js",
        "/vendor/katex/katex.min.css",
        "/vendor/p5.min.js",
        "/vendor/d3.min.js",
        "/vendor/mermaid.min.js",
    ):
        assert tag in CONTRACT, tag
    assert CONTRACT.index("/lesson.js") < CONTRACT.index("/vendor/gsap.min.js")
    assert "lesson.scene({root, timeline, steps})" in prompt.system
    assert "under eighty characters" in CONTRACT
    assert "three to five" in CONTRACT
    for word in ("eight thousand", "quickly", "prefer a library call"):
        assert word not in CONTRACT, word


def test_a_retry_carries_the_previous_error_and_nothing_else_changes() -> None:
    first = scene_prompt("PPO", BRIEF, "light")
    retry = scene_prompt("PPO", BRIEF, "light", error="timeline lacks labels step-3")

    assert retry.system == first.system
    assert retry.user_text == first.user_text + (
        "\n\nThe previous attempt failed its check: timeline lacks labels step-3\n"
        "Write the whole scene again."
    )


def test_the_tool_schema_is_the_draft_and_nothing_else() -> None:
    (tool,) = SCENE_TOOLS
    json.dumps(SCENE_TOOLS)
    assert tool["type"] == "function"
    function = tool["function"]
    assert function["name"] == SCENE_TOOL
    parameters = function["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"html", "steps"}
    assert parameters["required"] == ["html", "steps"]
    assert parameters["properties"]["html"]["maxLength"] == 200000
    assert parameters["properties"]["steps"]["minItems"] == 3
    assert parameters["properties"]["steps"]["maxItems"] == 5
    assert parameters["properties"]["steps"]["items"]["maxLength"] == 120
    assert "steps" in function["description"] and "lesson.scene" in function["description"]


def test_the_draft_holds_three_to_five_lines_and_a_document() -> None:
    draft = SceneDraft.model_validate(DRAFT)
    assert draft.steps == DRAFT["steps"]
    for body in (
        {"html": "", "steps": ["a", "b", "c"]},
        {"html": "x", "steps": ["a", "b"]},
        {"html": "x", "steps": ["a", "b", "c", "d", "e", "f"]},
        {"html": "x", "steps": ["a", "b", ""]},
        {"html": "x" * 200001, "steps": ["a", "b", "c"]},
        {"html": "x", "steps": ["a", "b", "c"], "title": "t"},
    ):
        try:
            SceneDraft.model_validate(body)
        except ValueError:
            continue
        raise AssertionError(body)


async def test_a_tool_call_becomes_a_draft_and_is_logged_without_its_text(caplog) -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(log, [], asyncio.Event(), visual=[call(DRAFT_ARGUMENTS)])
    with caplog.at_level(logging.INFO, logger="tutor.scene"):
        result = await run_scene_build(
            reasoning, scene_prompt("PPO", BRIEF, "light"), 32000, "high", model="draw-1"
        )

    assert result == SceneDraft.model_validate(DRAFT)
    assert reasoning.tools == [SCENE_TOOLS]
    assert reasoning.tool_choices == ["required"]
    assert reasoning.max_tokens == [32000]
    assert reasoning.efforts == ["high"]
    assert reasoning.models == ["draw-1"]
    (line,) = [m for m in caplog.messages if m.startswith("scene.call")]
    assert re.fullmatch(r"scene\.call ms=\d+ finish=\S+ chars=23 steps=3", line)
    assert "<p>" not in " ".join(caplog.messages)


async def test_prose_without_a_tool_call_is_the_no_tool_string(caplog) -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(
        log, [], asyncio.Event(), visual=[TurnChunk(kind="spoken", text="here is a scene")]
    )
    with caplog.at_level(logging.INFO, logger="tutor.scene"):
        result = await run_scene_build(
            reasoning, scene_prompt("PPO", BRIEF, "light"), 32000, "high"
        )

    assert result == NO_TOOL_CALL
    assert "scene.prose chars=15" in caplog.messages


async def test_a_stream_with_neither_prose_nor_a_call_is_empty_and_one_line(caplog) -> None:
    log: list[tuple[str, object]] = []
    capped = FakeReasoning(log, [], asyncio.Event(), visual=[], visual_finish="length")
    with caplog.at_level(logging.INFO, logger="tutor.scene"):
        result = await run_scene_build(capped, scene_prompt("PPO", BRIEF, "light"), 128000, "high")

    assert result == EMPTY_REPLY == "scene: error: empty reply"
    assert [m for m in caplog.messages if m.startswith("scene.")] == ["scene.empty finish=length"]

    caplog.clear()
    dropped = FakeReasoning(log, [], asyncio.Event(), visual=[])
    with caplog.at_level(logging.INFO, logger="tutor.scene"):
        result = await run_scene_build(dropped, scene_prompt("PPO", BRIEF, "light"), 128000, "high")

    assert result == EMPTY_REPLY
    assert [m for m in caplog.messages if m.startswith("scene.")] == ["scene.empty finish=None"]


async def test_an_unexpected_tool_and_a_truncated_stream_are_error_strings(caplog) -> None:
    log: list[tuple[str, object]] = []
    other = FakeReasoning(log, [], asyncio.Event(), visual=[call("{}", "push_app")])
    with caplog.at_level(logging.WARNING, logger="tutor.scene"):
        result = await run_scene_build(other, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert result == "scene: error: unexpected tool push_app"
    assert "scene.unexpected_tool chars=2" in caplog.messages
    assert "push_app" not in " ".join(caplog.messages)
    cut = FakeReasoning(
        log, [], asyncio.Event(), visual=[call(DRAFT_ARGUMENTS[:20])], visual_finish="length"
    )
    with caplog.at_level(logging.INFO, logger="tutor.scene"):
        result = await run_scene_build(cut, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert result == "scene: error: truncated at 32000 tokens"
    assert "scene.truncated chars=20" in caplog.messages


async def test_bad_arguments_are_error_strings_that_carry_no_text(caplog) -> None:
    log: list[tuple[str, object]] = []
    broken = FakeReasoning(log, [], asyncio.Event(), visual=[call("{")])
    with caplog.at_level(logging.WARNING, logger="tutor.scene"):
        result = await run_scene_build(broken, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert result == "scene: error: arguments are not valid JSON"
    assert "scene.tool_arguments chars=1" in caplog.messages

    short = FakeReasoning(
        log, [], asyncio.Event(), visual=[call(json.dumps({"html": "<p>x</p>", "steps": ["a"]}))]
    )
    with caplog.at_level(logging.WARNING, logger="tutor.scene"):
        result = await run_scene_build(short, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert isinstance(result, str) and result.startswith("scene: error: steps: ")
    assert "<p>" not in result
    assert "scene.rejected errors=1" in caplog.messages

    extra = FakeReasoning(
        log,
        [],
        asyncio.Event(),
        visual=[call(json.dumps({**DRAFT, "<p>secret</p>": 1}))],
    )
    with caplog.at_level(logging.WARNING, logger="tutor.scene"):
        result = await run_scene_build(extra, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert result == "scene: error: extra: Extra inputs are not permitted"
    assert caplog.messages.count("scene.rejected errors=1") == 2
    assert "<p>" not in " ".join(caplog.messages)

    digits = FakeReasoning(
        log,
        [],
        asyncio.Event(),
        visual=[call('{"html": "<p>x</p>", "steps": ["a", "b", "c"], "n": ' + "1" * 5000 + "}")],
    )
    with caplog.at_level(logging.WARNING, logger="tutor.scene"):
        result = await run_scene_build(digits, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert result == "scene: error: arguments are not valid JSON"
    assert "scene.tool_arguments chars=5053" in caplog.messages

    nested = FakeReasoning(log, [], asyncio.Event(), visual=[call("[" * 20000)])
    with caplog.at_level(logging.WARNING, logger="tutor.scene"):
        result = await run_scene_build(nested, scene_prompt("PPO", BRIEF, "light"), 32000, "high")
    assert result == "scene: error: arguments are not valid JSON"
    assert "scene.tool_arguments chars=20000" in caplog.messages
