import asyncio
import hashlib
import json
import logging
import re
from collections import deque
from collections.abc import AsyncIterator

import httpx2
import pytest
from openai import APIError

from tests.test_session import LESSON
from tutor.lesson import ScriptChunk
from tutor.planner import named_paths
from tutor.prompt import TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.script import (
    SCRIPT_PROMPT,
    SCRIPT_TOOL,
    SCRIPT_TOOLS,
    chunk_order,
    script_fault,
    script_prompt,
    write_script,
)

STARTING = "I know policy gradients"
GOOD = [
    {
        "step": 1,
        "question": False,
        "text": "Here are the old and the new policy over three actions.",
    },
    {"step": 2, "question": True, "text": "Pick one action. What is the ratio where they agree?"},
    {"step": 2, "question": False, "text": "Where the two agree, the ratio is exactly one."},
    {"step": 3, "question": False, "text": "Across all three actions the ratio moves around one."},
]
UNASKED = [GOOD[0], GOOD[2], GOOD[3]]
OUT_OF_ORDER = (
    "the chunks run step 1, step 2, step 3; they must run step 1, question 2, step 2, step 3"
)
SETTINGS = {"model": "glm-5.3", "effort": "low", "max_tokens": 4000}


class FakeStream:
    def __init__(
        self,
        chunks: list[TurnChunk],
        finish_reason: str | None = "tool_calls",
        raises: BaseException | None = None,
        hold: asyncio.Event | None = None,
    ) -> None:
        self._chunks = chunks
        self._raises = raises
        self._hold = hold
        self.finish_reason = finish_reason
        self.started = asyncio.Event()
        self.cancels = 0

    async def _drain(self) -> AsyncIterator[TurnChunk]:
        self.started.set()
        for chunk in self._chunks:
            yield chunk
        if self._hold is not None:
            await self._hold.wait()
        if self._raises is not None:
            raise self._raises

    def __aiter__(self) -> AsyncIterator[TurnChunk]:
        return self._drain()

    async def cancel(self) -> None:
        self.cancels += 1


class FakeWriter:
    def __init__(self, streams: list[FakeStream]) -> None:
        self._streams = deque(streams)
        self.prompts: list[TurnPrompt] = []
        self.calls: list[dict[str, object]] = []

    def start_turn(self, prompt: TurnPrompt, **kwargs: object) -> FakeStream:
        self.prompts.append(prompt)
        self.calls.append(kwargs)
        return self._streams.popleft()


def call(arguments: object, name: str = SCRIPT_TOOL) -> TurnChunk:
    text = arguments if isinstance(arguments, str) else json.dumps({"chunks": arguments})
    return TurnChunk(kind="tool_call", text=text, tool_call_id="call-s", tool_name=name)


def chunks(rows: list[dict[str, object]]) -> list[ScriptChunk]:
    return [ScriptChunk.model_validate(row) for row in rows]


def edited(index: int, text: str) -> list[ScriptChunk]:
    rows = [dict(row) for row in GOOD]
    rows[index]["text"] = text
    return chunks(rows)


async def write(writer: FakeWriter, timeout_s: float = 30.0) -> list[ScriptChunk] | str:
    return await write_script(writer, "PPO", STARTING, LESSON, 1, **SETTINGS, timeout_s=timeout_s)


def rejections(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [m for m in caplog.messages if m.startswith("script.rejected")]


def test_the_prompt_and_tool_are_the_prototypes() -> None:
    assert (
        hashlib.sha256(SCRIPT_PROMPT.encode()).hexdigest()
        == "764987bb8614e07286519bf73045207e68161ccc945d2f92c4af93fde75f6b54"
    )
    tools = json.dumps(SCRIPT_TOOLS, sort_keys=True, separators=(",", ":")).encode()
    assert (
        hashlib.sha256(tools).hexdigest()
        == "f48809440e49f396ab1ff0ab39de4939a3dc20de7e8600bfe8c664b62ff2d025"
    )


def test_the_writer_input_is_the_prototypes() -> None:
    prompt = script_prompt("PPO", STARTING, LESSON, 1, None)
    assert prompt.system == SCRIPT_PROMPT
    assert prompt.user_text == (
        "Subject: PPO\n"
        "Starting from: I know policy gradients\n"
        "Previous scene: none; this scene opens the lesson\n"
        "This scene: The ratio\n"
        "As a whole it shows: The probability ratio between the new and the old policy\n"
        "Its steps:\n"
        "1. The old and the new policy over three actions\n"
        "2. Their ratio at one action\n"
        "   Step 2 asks first: What is the ratio where they agree?\n"
        "   Step 2's picture, above, is the answer: no chunk before the question may give it.\n"
        "3. The ratio across all three actions\n"
        "Write the chunks in this order: step 1, question 2, step 2, step 3."
    )
    second = script_prompt("PPO", STARTING, LESSON, 2, None).user_text.splitlines()
    assert second[2] == "Previous scene: The ratio"
    assert second[3] == "This scene: The clip"
    assert second[-1] == "Write the chunks in this order: step 1, step 2, step 3."
    bare = script_prompt("PPO", "", LESSON, 1, None).user_text.splitlines()
    assert bare[:2] == ["Subject: PPO", "Previous scene: none; this scene opens the lesson"]
    assert not any(line.startswith("Starting from") for line in bare)


def test_chunks_out_of_order_are_refused_with_the_prototypes_reason() -> None:
    scene = LESSON.scenes[0]
    assert chunk_order(scene) == [(1, False), (2, True), (2, False), (3, False)]
    assert script_fault(scene, chunks(GOOD)) is None
    assert script_fault(scene, chunks(UNASKED)) == OUT_OF_ORDER
    assert script_fault(scene, []) == (
        "the chunks run empty; they must run step 1, question 2, step 2, step 3"
    )
    swapped = chunks([GOOD[0], GOOD[2], GOOD[1], GOOD[3]])
    assert script_fault(scene, swapped) == (
        "the chunks run step 1, step 2, question 2, step 3; "
        "they must run step 1, question 2, step 2, step 3"
    )


def test_each_text_rule_refuses() -> None:
    scene = LESSON.scenes[0]
    cases = [
        (edited(0, "   "), "the chunk for step 1 is empty"),
        (
            edited(3, "The ratio moves \u2014 caf\u00e9 style."),
            "the chunk for step 3 has characters outside ASCII: '\\u2014', '\\xe9'",
        ),
        (edited(2, "The ratio <step 2> is one."), "the chunk for step 2 has an angle bracket"),
        (
            edited(0, "Here are the old and the new policy"),
            (
                "the chunk for step 1 does not end with a full stop, a question mark or an "
                "exclamation mark"
            ),
        ),
        (
            edited(1, "Pick one action and say the ratio where they agree."),
            "the question for step 2 does not end with a question mark",
        ),
    ]
    for script, reason in cases:
        assert script_fault(scene, script) == reason
    assert script_fault(scene, edited(3, "  Across all three it moves.  ")) is None


async def test_the_retry_carries_the_rejection_line(caplog: pytest.LogCaptureFixture) -> None:
    valid = FakeStream(
        [
            TurnChunk(kind="spoken", text="Here is the script."),
            call(GOOD),
            call(UNASKED),
        ]
    )
    writer = FakeWriter([FakeStream([call(UNASKED)]), valid])
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        result = await write(writer)
    assert result == chunks(GOOD)
    first, second = writer.prompts
    assert first == script_prompt("PPO", STARTING, LESSON, 1, None)
    assert second.system == SCRIPT_PROMPT
    assert second.user_text == first.user_text + (
        f"\n\nYour previous call was rejected: {OUT_OF_ORDER}. "
        "Call write_script again with the whole scene."
    )
    expected = {
        "tools": SCRIPT_TOOLS,
        "effort": "low",
        "max_tokens": 4000,
        "tool_choice": "required",
        "model": "glm-5.3",
    }
    assert writer.calls == [expected, expected]
    assert rejections(caplog) == ["script.rejected scene_id=ratio attempt=1 reason=order"]
    (written,) = [m for m in caplog.messages if m.startswith("script.written")]
    assert re.fullmatch(r"script\.written scene_id=ratio attempt=2 ms=\d+ chunks=4", written)
    logged = " ".join(caplog.messages)
    assert not any(row["text"] in logged for row in GOOD)
    assert "Subject" not in logged and SCRIPT_TOOL not in logged

    caplog.clear()
    unended = [dict(row) for row in GOOD]
    unended[0]["text"] = "Here are the old and the new policy"
    writer = FakeWriter([FakeStream([call(unended)]), FakeStream([call(GOOD)])])
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        assert await write(writer) == chunks(GOOD)
    assert rejections(caplog) == ["script.rejected scene_id=ratio attempt=1 reason=text"]
    assert writer.prompts[1].user_text.endswith(
        "Your previous call was rejected: the chunk for step 1 does not end with a full stop, "
        "a question mark or an exclamation mark. Call write_script again with the whole scene."
    )


async def test_a_script_naming_a_path_is_refused(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = asyncio.to_thread
    threaded: list[tuple[object, tuple[object, ...]]] = []

    async def recording(func: object, /, *args: object, **kwargs: object) -> object:
        threaded.append((func, args))
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording)
    rows = [dict(row) for row in GOOD]
    rows[3]["text"] = "The ratio is computed in tutor/session.py across all three actions."
    writer = FakeWriter([FakeStream([call(rows)]), FakeStream([call(rows)])])
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        result = await write(writer)
    assert result == "script: error: the script names a source position"
    assert writer.prompts[1].user_text.endswith(
        "Your previous call was rejected: the script names a source position. "
        "Call write_script again with the whole scene."
    )
    assert threaded == [(named_paths, ([row["text"] for row in rows],))] * 2
    assert rejections(caplog) == [
        "script.rejected scene_id=ratio attempt=1 reason=position",
        "script.rejected scene_id=ratio attempt=2 reason=position",
    ]
    assert "tutor/session.py" not in " ".join(caplog.messages)


async def test_a_client_error_or_timeout_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = APIError("upstream", httpx2.Request("POST", "https://reasoning.invalid"), body=None)
    failed = FakeStream([], raises=error)
    outlasts = FakeStream([TurnChunk(kind="spoken", text="Hm.")], hold=asyncio.Event())
    writer = FakeWriter([failed, outlasts])
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        result = await write(writer, timeout_s=0.01)
    assert result == "script: error: timeout"
    assert writer.prompts[1].user_text.endswith(
        "Your previous call was rejected: call failed: APIError. "
        "Call write_script again with the whole scene."
    )
    assert rejections(caplog) == [
        "script.rejected scene_id=ratio attempt=1 reason=call",
        "script.rejected scene_id=ratio attempt=2 reason=timeout",
    ]
    assert (failed.cancels, outlasts.cancels) == (1, 1)

    caplog.clear()
    inner = [FakeStream([], raises=TimeoutError()), FakeStream([], raises=httpx2.ReadError("x"))]
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        result = await write(FakeWriter(inner))
    assert result == "script: error: call failed: ReadError"
    assert [stream.cancels for stream in inner] == [1, 1]
    assert rejections(caplog) == [
        "script.rejected scene_id=ratio attempt=1 reason=call",
        "script.rejected scene_id=ratio attempt=2 reason=call",
    ]

    broken = FakeStream([], raises=RuntimeError("client closed"))
    with pytest.raises(RuntimeError, match="client closed"):
        await write(FakeWriter([broken]))

    held = FakeStream([], hold=asyncio.Event())
    task = asyncio.create_task(write(FakeWriter([held])))
    await held.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_truncation_and_bad_arguments_refuse(caplog: pytest.LogCaptureFixture) -> None:
    cut = [FakeStream([call(GOOD)], finish_reason="length") for _ in range(2)]
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        assert await write(FakeWriter(cut)) == "script: error: truncated at 4000 tokens"
    assert rejections(caplog)[-1] == "script.rejected scene_id=ratio attempt=2 reason=truncated"

    caplog.clear()
    missing = call([{"step": 1, "question": False}])
    zero = call([{"step": 0, "question": False, "text": ""}])
    writer = FakeWriter([FakeStream([missing]), FakeStream([zero])])
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        result = await write(writer)
    assert writer.prompts[1].user_text.endswith(
        "Your previous call was rejected: the arguments do not fit write_script: chunks.0.text: "
        "Field required. Call write_script again with the whole scene."
    )
    assert result == (
        "script: error: the arguments do not fit write_script: chunks.0.step: Input should be "
        "greater than or equal to 1; chunks.0.text: String should have at least 1 character"
    )
    assert rejections(caplog) == [
        "script.rejected scene_id=ratio attempt=1 reason=arguments",
        "script.rejected scene_id=ratio attempt=2 reason=arguments",
    ]
    writer = FakeWriter([FakeStream([call("not json")]), FakeStream([call(GOOD)])])
    assert await write(writer) == chunks(GOOD)
    assert "rejected: the arguments do not fit write_script: body: Invalid JSON" in (
        writer.prompts[1].user_text
    )

    caplog.clear()
    stray = [
        FakeStream([TurnChunk(kind="spoken", text="No call.")]),
        FakeStream([call(GOOD, name="write_plan")]),
    ]
    with caplog.at_level(logging.INFO, logger="tutor.script"):
        assert await write(FakeWriter(stray)) == "script: error: unexpected tool write_plan"
    assert rejections(caplog) == [
        "script.rejected scene_id=ratio attempt=1 reason=no_tool_call",
        "script.rejected scene_id=ratio attempt=2 reason=unexpected_tool",
    ]
