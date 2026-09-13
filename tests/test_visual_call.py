import asyncio
import json
import logging

from tests.test_session import FakeReasoning, LoggingTransport, visual_call
from tutor.brief import VisualBrief
from tutor.prompt import Message, TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.visual_call import VISUAL_SYSTEM_PROMPT, run_visual_call, visual_prompt
from tutor.visual_tools import VISUAL_CALL_TOOLS
from tutor.visuals import VisualChannel

BRIEF = VisualBrief(kind="app", title="Clipped objective", show="surrogate vs ratio, epsilon 0.2")
PREVIOUS = VisualBrief(kind="diagram", title="PPO update loop", show="collect, estimate, update")
SNAPSHOT = TurnPrompt(
    system="voice system",
    history=[
        Message(role="user", content="teach me ppo"),
        Message(role="assistant", content="PPO clips."),
    ],
    user_text="why clip",
)
APP_ARGUMENTS = json.dumps(
    {"id": "clip", "title": "Clipped objective", "html": "<!doctype html><p>x</p>"}
)


def test_the_visual_prompt_carries_the_snapshot_the_brief_and_the_previous_one() -> None:
    prompt = visual_prompt(SNAPSHOT, BRIEF, PREVIOUS, "dark")
    assert prompt.history == SNAPSHOT.history
    assert prompt.tool_context == SNAPSHOT.tool_context
    assert prompt.system.startswith(VISUAL_SYSTEM_PROMPT.split("{", 1)[0])
    assert "near-black" in prompt.system and "light text" in prompt.system
    assert "Previous visual: diagram, PPO update loop" in prompt.system
    assert prompt.user_text == f"Brief: {BRIEF.model_dump_json()}\nThe learner just said: why clip"
    assert "under eight thousand characters" in prompt.system
    assert prompt.system.isascii()


def test_no_previous_visual_is_said_so() -> None:
    assert "Previous visual: none" in visual_prompt(SNAPSHOT, BRIEF, None, "light").system


def test_the_prompt_names_every_vendored_tag_and_nothing_else() -> None:
    system = visual_prompt(SNAPSHOT, BRIEF, None, "light").system
    for tag in (
        "/vendor/plotly.min.js",
        "/vendor/katex/katex.min.js",
        "/vendor/katex/katex.min.css",
        "/vendor/p5.min.js",
        "/vendor/d3.min.js",
    ):
        assert tag in system
    assert "http" not in system


async def test_a_tool_call_is_dispatched_and_the_result_string_returned() -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(
        log, [], asyncio.Event(), visual=[visual_call("push_app", APP_ARGUMENTS, "call-v")]
    )
    channel = VisualChannel(LoggingTransport(log))
    result = await run_visual_call(
        reasoning, visual_prompt(SNAPSHOT, BRIEF, None, "light"), channel, 3000, lambda: True
    )
    assert result == "push_app: sent"
    assert reasoning.tools == [VISUAL_CALL_TOOLS]
    assert reasoning.tool_choices == ["required"]
    assert reasoning.max_tokens == [3000]
    (payload,) = [p for name, p in log if name == "send_json"]
    assert payload["type"] == "app.push" and payload["title"] == "Clipped objective"


async def test_a_superseded_visual_is_not_dispatched() -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(
        log, [], asyncio.Event(), visual=[visual_call("push_app", APP_ARGUMENTS, "call-v")]
    )
    result = await run_visual_call(
        reasoning,
        visual_prompt(SNAPSHOT, BRIEF, None, "light"),
        VisualChannel(LoggingTransport(log)),
        3000,
        lambda: False,
    )
    assert result == "visual: superseded"
    assert not [p for name, p in log if name == "send_json"]


async def test_prose_without_a_tool_call_is_an_error_string(caplog) -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(
        log, [], asyncio.Event(), visual=[TurnChunk(kind="spoken", text="here is a plot")]
    )
    with caplog.at_level(logging.INFO, logger="tutor.visual_call"):
        result = await run_visual_call(
            reasoning,
            visual_prompt(SNAPSHOT, BRIEF, None, "light"),
            VisualChannel(LoggingTransport(log)),
            3000,
            lambda: True,
        )
    assert result == "visual: error: no tool call"
    assert any(m.startswith("visual.prose chars=") for m in caplog.messages)


async def test_a_call_for_a_voice_tool_is_an_error_string() -> None:
    log: list[tuple[str, object]] = []
    reasoning = FakeReasoning(
        log, [], asyncio.Event(), visual=[visual_call("clear_diagram", "{}", "call-v")]
    )
    result = await run_visual_call(
        reasoning,
        visual_prompt(SNAPSHOT, BRIEF, None, "light"),
        VisualChannel(LoggingTransport(log)),
        3000,
        lambda: True,
    )
    assert result == "visual: error: unexpected tool clear_diagram"
    assert not [p for name, p in log if name == "send_json"]


async def test_a_call_cut_by_max_tokens_is_reported_as_truncated(caplog) -> None:
    log: list[tuple[str, object]] = []
    cut = visual_call("push_app", APP_ARGUMENTS[:40], "call-v")
    reasoning = FakeReasoning(log, [], asyncio.Event(), visual=[cut], visual_finish="length")
    with caplog.at_level(logging.INFO, logger="tutor.visual_call"):
        result = await run_visual_call(
            reasoning,
            visual_prompt(SNAPSHOT, BRIEF, None, "light"),
            VisualChannel(LoggingTransport(log)),
            3000,
            lambda: True,
        )
    assert result == "visual: error: truncated at 3000 tokens"
    assert not [p for name, p in log if name == "send_json"]
    assert any(m.startswith("visual.truncated tool=push_app chars=") for m in caplog.messages)
