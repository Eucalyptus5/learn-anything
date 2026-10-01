import asyncio
import logging
import time

import httpx2
from openai import APIError
from pydantic import BaseModel, ConfigDict, ValidationError

from tutor.lesson import LessonPlan, Scene, ScriptChunk
from tutor.planner import named_paths
from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient, TurnChunk

logger = logging.getLogger(__name__)

SCRIPT_TOOL = "write_script"
SCRIPT_ERROR = "script: error: "
_CALL_ERRORS = (APIError, TimeoutError, httpx2.ReadError, httpx2.RemoteProtocolError)

SCRIPT_PROMPT = """
You write the spoken script for one scene of a lesson that a tutor teaches aloud to one
learner. A picture beside the learner is built up in steps, and each piece of your script plays
as its step appears. You produce exactly one call to write_script and no prose.

Write the chunks in playing order. Every step gets one chunk: two to four spoken sentences that
say what that step's picture shows as it appears, and what it means. Narrate exactly that step:
nothing a later step draws, and nothing the step does not show. A step that asks gets a second
chunk, its question, placed just before the step's own chunk and marked question true. The
question chunk may lead in with one short sentence and ends on the question, put in words close
to the ask line. Every other chunk is marked question false.

The step that asks is the step whose picture answers its question, and you are told that
picture so you can hold it back. No chunk before the question, the question itself included,
may state, hint at or work out the answer, give the numbers that answer it, or describe what
that step draws. The step's own chunk, after the question, shows and explains the answer. It
plays after the learner has answered and the tutor has already said whether they were right, so
it never says whether they were right, and it must also make sense if the question was skipped.

The first chunk opens the scene. It may play straight after the end of the previous scene, so
it can begin with a short turn from that scene, but it never repeats it. It never refers to an
earlier session, as in "last time": the previous scene played moments ago in this same session.

The chunks are synthesized to audio. Write plain spoken English in ASCII only: no symbols, no
equations in symbols, no markdown, no lists, no angle brackets. Say Greek letters and operators
as words, such as theta, epsilon, times and over. End every chunk with a full stop, a question
mark or an exclamation mark. Never name a scene number or a step number aloud.
""".strip()

SCRIPT_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": SCRIPT_TOOL,
            "description": "Hand over the scene's spoken script: its chunks in playing order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunks": {
                        "type": "array",
                        "description": "The chunks in playing order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {
                                    "type": "integer",
                                    "description": "The step this chunk belongs to, from 1.",
                                },
                                "question": {
                                    "type": "boolean",
                                    "description": (
                                        "True for the question put just before an asking "
                                        "step's own chunk; false for a step's own narration."
                                    ),
                                },
                                "text": {
                                    "type": "string",
                                    "description": "The spoken words: plain ASCII sentences.",
                                },
                            },
                            "required": ["step", "question", "text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["chunks"],
                "additionalProperties": False,
            },
        },
    }
]


class _Script(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunks: list[ScriptChunk]


def _spell(order: list[tuple[int, bool]]) -> str:
    return ", ".join(f"question {n}" if question else f"step {n}" for n, question in order)


def chunk_order(scene: Scene) -> list[tuple[int, bool]]:
    order: list[tuple[int, bool]] = []
    for n, step in enumerate(scene.steps, start=1):
        if step.ask:
            order.append((n, True))
        order.append((n, False))
    return order


def script_fault(scene: Scene, chunks: list[ScriptChunk]) -> str | None:
    found = [(chunk.step, chunk.question) for chunk in chunks]
    wanted = chunk_order(scene)
    if found != wanted:
        return f"the chunks run {_spell(found) or 'empty'}; they must run {_spell(wanted)}"
    for chunk in chunks:
        name = f"the {'question' if chunk.question else 'chunk'} for step {chunk.step}"
        text = chunk.text.strip()
        if not text:
            return f"{name} is empty"
        if not text.isascii():
            odd = ", ".join(sorted({ascii(char) for char in text if not char.isascii()}))
            return f"{name} has characters outside ASCII: {odd}"
        if "<" in text or ">" in text:
            return f"{name} has an angle bracket"
        if text[-1] not in ".?!":
            return f"{name} does not end with a full stop, a question mark or an exclamation mark"
        if chunk.question and not text.endswith("?"):
            return f"{name} does not end with a question mark"
    return None


def script_prompt(
    subject: str, starting_from: str, plan: LessonPlan, n: int, rejected: str | None
) -> TurnPrompt:
    scene = plan.scenes[n - 1]
    previous = plan.scenes[n - 2].title if n > 1 else "none; this scene opens the lesson"
    lines = [f"Subject: {subject}"]
    if starting_from:
        lines.append(f"Starting from: {starting_from}")
    lines += [
        f"Previous scene: {previous}",
        f"This scene: {scene.title}",
        f"As a whole it shows: {scene.show}",
        "Its steps:",
    ]
    for k, step in enumerate(scene.steps, start=1):
        lines.append(f"{k}. {step.show}")
        if step.ask:
            lines.append(f"   Step {k} asks first: {step.ask}")
            lines.append(
                f"   Step {k}'s picture, above, is the answer: no chunk before the question may "
                "give it."
            )
    lines.append(f"Write the chunks in this order: {_spell(chunk_order(scene))}.")
    text = "\n".join(lines)
    if rejected is not None:
        text += (
            f"\n\nYour previous call was rejected: {rejected}. Call write_script again with the "
            "whole scene."
        )
    return TurnPrompt(system=SCRIPT_PROMPT, user_text=text)


async def _attempt(
    reasoning: ReasoningClient,
    prompt: TurnPrompt,
    scene: Scene,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    timeout_s: float,
) -> list[ScriptChunk] | tuple[str, str]:
    call: TurnChunk | None = None
    stream = reasoning.start_turn(
        prompt,
        tools=SCRIPT_TOOLS,
        effort=effort,
        max_tokens=max_tokens,
        tool_choice="required",
        model=model,
    )
    try:
        async with asyncio.timeout(timeout_s) as bound:
            async for chunk in stream:
                if chunk.kind == "tool_call" and call is None:
                    call = chunk
    except _CALL_ERRORS as error:
        await stream.cancel()
        if bound.expired():
            return "timeout", "timeout"
        return "call", f"call failed: {type(error).__name__}"
    if call is None or not call.tool_name:
        return "no_tool_call", "no tool call"
    if call.tool_name != SCRIPT_TOOL:
        return "unexpected_tool", f"unexpected tool {call.tool_name}"
    if stream.finish_reason == "length":
        return "truncated", f"truncated at {max_tokens} tokens"
    try:
        chunks = _Script.model_validate_json(call.text).chunks
    except ValidationError as exc:
        fit = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
            for error in exc.errors()
        )
        return "arguments", f"the arguments do not fit write_script: {fit}"
    fault = script_fault(scene, chunks)
    if fault is not None:
        ordered = [(chunk.step, chunk.question) for chunk in chunks] == chunk_order(scene)
        return ("text" if ordered else "order"), fault
    if await asyncio.to_thread(named_paths, [chunk.text for chunk in chunks]):
        return "position", "the script names a source position"
    return chunks


async def write_script(
    reasoning: ReasoningClient,
    subject: str,
    starting_from: str,
    plan: LessonPlan,
    n: int,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    timeout_s: float,
) -> list[ScriptChunk] | str:
    scene = plan.scenes[n - 1]
    rejected: str | None = None
    for attempt in (1, 2):
        start = time.perf_counter()
        outcome = await _attempt(
            reasoning,
            script_prompt(subject, starting_from, plan, n, rejected),
            scene,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
        if isinstance(outcome, list):
            logger.info(
                "script.written scene_id=%s attempt=%d ms=%d chunks=%d",
                scene.id,
                attempt,
                int((time.perf_counter() - start) * 1000),
                len(outcome),
            )
            return outcome
        code, rejected = outcome
        logger.info("script.rejected scene_id=%s attempt=%d reason=%s", scene.id, attempt, code)
    return f"{SCRIPT_ERROR}{rejected}"
