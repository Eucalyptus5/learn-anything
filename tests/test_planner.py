import asyncio
import json
import logging
import re

import pytest

from tests.test_lesson import plan
from tests.test_session import FakeReasoning, visual_call
from tutor.lesson import LessonPlan
from tutor.planner import (
    EMPTY_REPLY,
    NO_TOOL_CALL,
    PLAN_TOOL,
    PLAN_TOOLS,
    PLANNER_PROMPT,
    POSITION_REFUSED,
    RERUN,
    plan_prompt,
    run_planner,
)
from tutor.prompt import Message
from tutor.reasoning import TurnChunk

PLAN = plan()
PLAN_ARGUMENTS = LessonPlan.model_validate(PLAN).model_dump_json(by_alias=True)
STARTING = "I know policy gradients and the advantage; I have not read the PPO paper"
TRANSCRIPT = [
    Message(role="user", content="teach me ppo"),
    Message(role="assistant", content="Where would the ratio sit after one bad step?"),
    Message(role="user", content="above one I think"),
]


def call(arguments: str, name: str = PLAN_TOOL) -> TurnChunk:
    return visual_call(name, arguments, "call-p")


def test_the_tool_schema_is_the_plan_and_nothing_else() -> None:
    (tool,) = PLAN_TOOLS
    json.dumps(PLAN_TOOLS)
    assert tool["type"] == "function"
    function = tool["function"]
    assert function["name"] == PLAN_TOOL
    parameters = function["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"profile", "scenes"}
    assert parameters["required"] == ["profile", "scenes"]
    assert parameters["properties"]["scenes"]["minItems"] == 1
    assert parameters["properties"]["scenes"]["maxItems"] == 12
    scene = parameters["$defs"]["Scene"]
    assert scene["additionalProperties"] is False
    assert set(scene["properties"]) == {"id", "heading", "show", "steps"}
    assert scene["required"] == ["id", "heading", "show", "steps"]
    assert scene["properties"]["steps"]["minItems"] == 3
    assert scene["properties"]["steps"]["maxItems"] == 5
    step = parameters["$defs"]["Step"]
    assert set(step["properties"]) == {"show", "ask"}
    assert step["required"] == ["show"]
    assert "title" not in parameters and "title" not in scene and "title" not in step
    assert '"title":' not in json.dumps(PLAN_TOOLS)
    assert "scenes" in function["description"] and "steps" in function["description"]


def test_every_field_of_the_tool_schema_carries_its_description() -> None:
    (tool,) = PLAN_TOOLS
    parameters = tool["function"]["parameters"]
    for group in (parameters, parameters["$defs"]["Scene"], parameters["$defs"]["Step"]):
        for name, field in group["properties"].items():
            assert field["description"], name
    heading = parameters["$defs"]["Scene"]["properties"]["heading"]
    assert heading["description"] == "Required. The scene's title, under eight words."


def test_the_connect_prompt_carries_the_form_and_no_history() -> None:
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    assert prompt.system == PLANNER_PROMPT
    assert prompt.history == [] and prompt.tool_context == [] and prompt.tool_exchange == []
    assert prompt.user_text == f"Subject: PPO\nStarting from: {STARTING}"
    assert prompt.system.isascii() and "phase" not in prompt.system.lower()
    assert "A label is at most five words." in prompt.system
    for word in ("profile", "three to five", "ask", "misconception", "numbers", "file path"):
        assert word in prompt.system, word
    folder = plan_prompt("a client", "", True, None, [], [])
    assert folder.user_text == (
        "Subject: a client\n"
        "A folder of source code is attached; the tutor can search it while teaching."
    )


def test_the_rerun_prompt_carries_the_transcript_the_plan_and_the_protected_ids() -> None:
    current = LessonPlan.model_validate(PLAN)
    prompt = plan_prompt("PPO", STARTING, False, current, ["scene-1"], TRANSCRIPT)
    assert prompt.system == PLANNER_PROMPT + RERUN
    assert prompt.history == TRANSCRIPT
    assert prompt.user_text == (
        f"Subject: PPO\nStarting from: {STARTING}\nProtected: scene-1\n"
        f"Current plan: {current.model_dump_json(by_alias=True)}"
    )
    assert '"heading":"Scene 1"' in prompt.user_text and '"title":' not in prompt.user_text
    none = plan_prompt("PPO", STARTING, False, current, [], TRANSCRIPT)
    assert "Protected: none\n" in none.user_text
    assert "exactly as they are" in RERUN and "never insert" in RERUN


async def test_a_tool_call_becomes_a_plan_and_is_logged_without_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(log, [], asyncio.Event(), visual=[call(PLAN_ARGUMENTS)])
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        result = await run_planner(reasoning, prompt, 8000, "high", model="plan-1")

    assert result == LessonPlan.model_validate(PLAN)
    assert reasoning.tools == [PLAN_TOOLS]
    assert reasoning.tool_choices == ["required"]
    assert reasoning.max_tokens == [8000]
    assert reasoning.efforts == ["high"]
    assert reasoning.models == ["plan-1"]
    (line,) = [m for m in caplog.messages if m.startswith("planner.call")]
    assert re.fullmatch(r"planner\.call ms=\d+ finish=\S+ scenes=2 steps=6", line)
    assert "Knows policy" not in " ".join(caplog.messages)


async def test_prose_without_a_tool_call_is_the_no_tool_string(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(
        log, [], asyncio.Event(), visual=[TurnChunk(kind="spoken", text="here is a plan")]
    )
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        result = await run_planner(reasoning, prompt, 8000, "high")

    assert result == NO_TOOL_CALL
    assert "planner.prose chars=14" in caplog.messages


async def test_a_stream_with_neither_prose_nor_a_call_is_empty_and_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(log, [], asyncio.Event(), visual=[], visual_finish="length")
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        result = await run_planner(reasoning, prompt, 8000, "high")

    assert result == EMPTY_REPLY
    assert [m for m in caplog.messages if m.startswith("planner.")] == [
        "planner.empty finish=length"
    ]


async def test_an_unexpected_tool_and_a_truncated_stream_are_error_strings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    other = FakeReasoning(log, [], asyncio.Event(), visual=[call(PLAN_ARGUMENTS, "write_scene")])
    cut = FakeReasoning(
        log, [], asyncio.Event(), visual=[call(PLAN_ARGUMENTS[:40])], visual_finish="length"
    )
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        unexpected = await run_planner(other, prompt, 8000, "high")
        truncated = await run_planner(cut, prompt, 8000, "high")

    assert unexpected == "planner: error: unexpected tool write_scene"
    assert truncated == "planner: error: truncated at 8000 tokens"
    assert f"planner.unexpected_tool chars={len(PLAN_ARGUMENTS)}" in caplog.messages
    assert "planner.truncated chars=40" in caplog.messages


async def test_bad_arguments_are_error_strings_that_carry_no_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    broken = FakeReasoning(log, [], asyncio.Event(), visual=[call("{not json")])
    invalid = FakeReasoning(
        log, [], asyncio.Event(), visual=[call(json.dumps({"profile": "p", "scenes": []}))]
    )
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        not_json = await run_planner(broken, prompt, 8000, "high")
        rejected = await run_planner(invalid, prompt, 8000, "high")

    assert not_json == "planner: error: arguments are not valid JSON"
    assert rejected.startswith("planner: error: scenes:")
    assert "planner.tool_arguments chars=9" in caplog.messages
    assert "planner.invalid errors=1" in caplog.messages
    assert "not json" not in " ".join(caplog.messages)


async def test_a_plan_that_names_a_source_path_is_refused_without_its_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    named = plan()
    named["scenes"][0]["steps"][0] = {
        "show": "The second argument to acquire in src/pool.py",
        "ask": "",
    }
    numbered = plan()
    numbered["scenes"][0]["steps"][0] = {"show": "The number line from 0 to 1", "ask": ""}
    log: list[tuple[str, object]] = []
    refusing = FakeReasoning(log, [], asyncio.Event(), visual=[call(json.dumps(named))])
    keeping = FakeReasoning(log, [], asyncio.Event(), visual=[call(json.dumps(numbered))])
    prompt = plan_prompt("a client", "", True, None, [], [])
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        refused = await run_planner(refusing, prompt, 8000, "high")
        kept = await run_planner(keeping, prompt, 8000, "high")

    assert refused == POSITION_REFUSED
    assert kept == LessonPlan.model_validate(numbered)
    assert "planner.positions count=1" in caplog.messages
    assert not any("pool" in m or "number line" in m for m in caplog.messages)


async def test_a_decimal_ratio_in_a_step_is_not_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ratio = plan()
    ratio["scenes"][0]["steps"][0] = {
        "show": "The ratio r = 0.6/0.4 = 1.5 on one sample",
        "ask": "",
    }
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(log, [], asyncio.Event(), visual=[call(json.dumps(ratio))])
    prompt = plan_prompt("PPO", STARTING, False, None, [], [])
    with caplog.at_level(logging.INFO, logger="tutor.planner"):
        result = await run_planner(reasoning, prompt, 8000, "high")

    assert result == LessonPlan.model_validate(ratio)
    assert not any(m.startswith("planner.positions") for m in caplog.messages)
