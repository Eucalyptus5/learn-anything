import asyncio
import json
import logging
import time
from collections.abc import Iterable, Sequence

from pydantic import ValidationError

from tutor.lesson import LessonPlan
from tutor.prompt import Message, TurnPrompt
from tutor.reasoning import ReasoningClient
from tutor.tools.citations import extract_positions

logger = logging.getLogger(__name__)

PLAN_TOOL = "write_plan"
NO_TOOL_CALL = "planner: error: no tool call"
EMPTY_REPLY = "planner: error: empty reply"
POSITION_REFUSED = "planner: error: the plan names a source position"

PLANNER_PROMPT = """
You plan a spoken lesson on one subject for one learner. A tutor teaches it aloud, scene by
scene, and one picture per scene is drawn from your words alone. You produce exactly one call
to write_plan and no prose.

Write a profile of the learner in a few sentences. Say only what the learner said and what
follows directly from it; invent no knowledge, gaps or preferences.

Write the scenes in teaching order. Each scene is one picture teaching one idea; a second idea
gets its own scene. A scene has a title under eight words, a show line saying what the picture
must show as a whole, and three to five steps. Each step is one visible change to the picture,
written so that a builder who has only that scene can draw it, with the numbers and the case.
Every step adds, moves or changes something on the canvas; no step only restates or keeps what
is already there.

Each scene stands alone. Never refer to another scene or another picture, earlier or later;
where an earlier number matters, repeat the number. Never put a scene number on the canvas.

Text on the canvas is short labels, numbers and equations. A label is at most five words.
Explaining is the voice's job, so no caption, note or bubble holds a sentence. Draw with plain
pieces first: axes, curves, points, bars, arrows, boxes, short labels and equations. Use an
object such as a dial or a stamp only when the idea needs it.

The tutor asks a step's question, waits for the answer, then shows that step. Put a question
in ask only where predicting or explaining helps the learner, never one per scene by habit.
The step that carries a question is the step whose picture answers it.

Connect the first new idea to what the learner said they know. Motivate each new quantity
before using it. Introduce every term and every piece of a formula in the scene that first
uses it, never later. Where the learner has an obvious objection or a simpler alternative,
give it a scene, and give one scene to each likely misconception at the place where it would
arise.

When a claim holds only in one case, such as one sign or one regime, say which case. Call a
quantity that is maximized an objective, not a loss.

Never write a file path, a line number or a name from source code: the tutor finds those by
searching while it teaches, and the pictures show only the idea. Write a library's or a
runtime's name without a file extension (Node, not Node.js). Scene ids are short, unique,
lowercase: letters, digits and hyphens.
""".strip()

RERUN = """

This is a rerun. The conversation above is what the learner and the tutor have said since the
plan was last written. The scenes whose ids are listed as protected are already drawn,
building, being taught, or exposed to a voice reply:
return them exactly as they are, in the same positions, and never insert a scene before them.
Revise what comes after them for this learner: rewrite or drop any unprotected scene, append
scenes past the end if the learner needs them, and say in the profile what changed."""


def _parameters() -> dict[str, object]:
    schema = LessonPlan.model_json_schema()
    schema.pop("title")
    for prop in schema["properties"].values():
        prop.pop("title")
    for definition in schema["$defs"].values():
        definition.pop("title")
        for prop in definition["properties"].values():
            prop.pop("title")
    return schema


def named_paths(texts: Iterable[str]) -> int:
    return sum(1 for text in texts for position in extract_positions(text) if position.path)


def _named_paths(plan: LessonPlan) -> int:
    texts = [plan.profile]
    for scene in plan.scenes:
        texts += [scene.title, scene.show]
        texts += [text for step in scene.steps for text in (step.show, step.ask)]
    return named_paths(texts)


PLAN_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": PLAN_TOOL,
            "description": (
                "Hand over the lesson plan: profile is the learner in a few sentences, scenes "
                "is the lesson in teaching order, each with three to five steps."
            ),
            "parameters": _parameters(),
        },
    }
]


def plan_prompt(
    subject: str,
    starting_from: str,
    folder: bool,
    plan: LessonPlan | None,
    protected: Sequence[str],
    transcript: list[Message],
) -> TurnPrompt:
    lines = [f"Subject: {subject}"]
    if starting_from:
        lines.append(f"Starting from: {starting_from}")
    if folder:
        lines.append("A folder of source code is attached; the tutor can search it while teaching.")
    system = PLANNER_PROMPT
    if plan is not None:
        system += RERUN
        lines.append(f"Protected: {', '.join(protected) or 'none'}")
        lines.append(f"Current plan: {plan.model_dump_json(by_alias=True)}")
    return TurnPrompt(system=system, history=list(transcript), user_text="\n".join(lines))


async def run_planner(
    reasoning: ReasoningClient,
    prompt: TurnPrompt,
    max_tokens: int,
    effort: str,
    model: str | None = None,
) -> LessonPlan | str:
    start = time.perf_counter()
    prose = 0
    call = None
    stream = reasoning.start_turn(
        prompt,
        tools=PLAN_TOOLS,
        effort=effort,
        max_tokens=max_tokens,
        tool_choice="required",
        model=model,
    )
    async for chunk in stream:
        if chunk.kind == "spoken":
            prose += len(chunk.text)
        elif chunk.kind == "tool_call" and call is None:
            call = chunk
    if prose:
        logger.info("planner.prose chars=%d", prose)
    if call is None or not call.tool_name:
        if prose:
            return NO_TOOL_CALL
        logger.info("planner.empty finish=%s", stream.finish_reason)
        return EMPTY_REPLY
    if call.tool_name != PLAN_TOOL:
        logger.warning("planner.unexpected_tool chars=%d", len(call.text))
        return f"planner: error: unexpected tool {call.tool_name}"
    if stream.finish_reason == "length":
        logger.info("planner.truncated chars=%d", len(call.text))
        return f"planner: error: truncated at {max_tokens} tokens"
    try:
        body = json.loads(call.text)
    except (ValueError, RecursionError):
        logger.warning("planner.tool_arguments chars=%d", len(call.text))
        return "planner: error: arguments are not valid JSON"
    try:
        plan = LessonPlan.model_validate(body)
    except ValidationError as exc:
        reasons: list[str] = []
        for error in exc.errors():
            where = (
                "extra"
                if error["type"] == "extra_forbidden"
                else ".".join(str(part) for part in error["loc"])
            )
            reasons.append(f"{where or 'body'}: {error['msg']}")
        logger.warning("planner.invalid errors=%d", len(reasons))
        return f"planner: error: {'; '.join(reasons)}"
    named = await asyncio.to_thread(_named_paths, plan)
    if named:
        logger.warning("planner.positions count=%d", named)
        return POSITION_REFUSED
    logger.info(
        "planner.call ms=%d finish=%s scenes=%d steps=%d",
        int((time.perf_counter() - start) * 1000),
        stream.finish_reason,
        len(plan.scenes),
        sum(len(scene.steps) for scene in plan.scenes),
    )
    return plan
