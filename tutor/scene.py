import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient
from tutor.visuals import StepLine

logger = logging.getLogger(__name__)

GUIDE = (Path(__file__).parent / "prompts" / "scene_guide.md").read_text()
SCENE_TOOL = "write_scene"
NO_TOOL_CALL = "scene: error: no tool call"
EMPTY_REPLY = "scene: error: empty reply"

CONTRACT = """
You build one scene for a spoken lesson: a complete HTML document that draws one picture and
moves it in steps as the tutor explains it. You receive the subject, the picture's title and
what it must show. You produce exactly one call to write_scene and no prose.

The document is markup, style and script inline, with nothing fetched from anywhere but these
tags, used exactly as written and in this order where used:
<script src="/lesson.js"></script>
<script src="/vendor/gsap.min.js"></script>
<script src="/vendor/plotly.min.js"></script>
<link rel="stylesheet" href="/vendor/katex/katex.min.css"><script src="/vendor/katex/katex.min.js"></script>
<script src="/vendor/p5.min.js"></script>
<script src="/vendor/d3.min.js"></script>
<script src="/vendor/mermaid.min.js"></script>
The first two are required. The first sets the page's colours and fonts as CSS variables and
lays the picture out beside a column of step lines; the second is the timeline library.

The page is {theme}. Every colour is one of var(--ink), var(--ink-2), var(--ink-3),
var(--accent), var(--surface) and var(--hair); every font is var(--sans), var(--serif) or
var(--mono). Put the whole picture inside one root element and do not style body.

Build one paused gsap timeline. Step 1 is the first frame; every later step adds one part. After
the tweens of each step n add the label "step-n", so the labels run step-1 to step-n. Then,
once, call lesson.scene({{root, timeline, steps}}) with the root element or its selector, the
timeline, and one say line per step, under eighty characters, naming what that step shows.
Write three to five steps. Pass the same say lines, in the same order, as the steps argument
of write_scene. The say lines are the only prose; the picture carries the rest.

The guide below is how the scene is judged. Read it before drawing.
""".strip()

RETRY = "\n\nThe previous attempt failed its check: {error}\nWrite the whole scene again."


class SceneDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    html: str = Field(min_length=1, max_length=200000)
    steps: list[StepLine] = Field(min_length=3, max_length=5)


def _parameters() -> dict[str, object]:
    schema = SceneDraft.model_json_schema()
    schema.pop("title")
    for prop in schema["properties"].values():
        prop.pop("title")
    return schema


SCENE_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": SCENE_TOOL,
            "description": (
                "Hand over the finished scene: html is the complete document, steps is the list "
                "of say lines passed to lesson.scene, identical and in order."
            ),
            "parameters": _parameters(),
        },
    }
]


def scene_prompt(subject: str, title: str, show: str, theme: str, error: str = "") -> TurnPrompt:
    user_text = f"Subject: {subject}\nTitle: {title}\nShow: {show}"
    if error:
        user_text += RETRY.format(error=error)
    return TurnPrompt(system=f"{CONTRACT.format(theme=theme)}\n\n{GUIDE}", user_text=user_text)


async def run_scene_build(
    reasoning: ReasoningClient,
    prompt: TurnPrompt,
    max_tokens: int,
    effort: str,
    model: str | None = None,
) -> SceneDraft | str:
    start = time.perf_counter()
    prose = 0
    call = None
    stream = reasoning.start_turn(
        prompt,
        tools=SCENE_TOOLS,
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
        logger.info("scene.prose chars=%d", prose)
    if call is None or not call.tool_name:
        if prose:
            return NO_TOOL_CALL
        logger.info("scene.empty finish=%s", stream.finish_reason)
        return EMPTY_REPLY
    if call.tool_name != SCENE_TOOL:
        logger.warning("scene.unexpected_tool chars=%d", len(call.text))
        return f"scene: error: unexpected tool {call.tool_name}"
    if stream.finish_reason == "length":
        logger.info("scene.truncated chars=%d", len(call.text))
        return f"scene: error: truncated at {max_tokens} tokens"
    try:
        body = json.loads(call.text)
    except (ValueError, RecursionError):
        logger.warning("scene.tool_arguments chars=%d", len(call.text))
        return "scene: error: arguments are not valid JSON"
    try:
        draft = SceneDraft.model_validate(body)
    except ValidationError as exc:
        reasons: list[str] = []
        for error in exc.errors():
            where = (
                "extra"
                if error["type"] == "extra_forbidden"
                else ".".join(str(part) for part in error["loc"])
            )
            reasons.append(f"{where or 'body'}: {error['msg']}")
        logger.warning("scene.rejected errors=%d", len(reasons))
        return f"scene: error: {'; '.join(reasons)}"
    logger.info(
        "scene.call ms=%d finish=%s chars=%d steps=%d",
        int((time.perf_counter() - start) * 1000),
        stream.finish_reason,
        len(draft.html),
        len(draft.steps),
    )
    return draft
