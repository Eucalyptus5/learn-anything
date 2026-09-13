import logging
import time
from collections.abc import Callable

from tutor.brief import VisualBrief
from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient
from tutor.visual_tools import VISUAL_CALL_TOOLS, dispatch_visual_tool
from tutor.visuals import VisualChannel

logger = logging.getLogger(__name__)

CALL_TOOL_NAMES = frozenset(t["function"]["name"] for t in VISUAL_CALL_TOOLS)
THEMES = {
    "light": ("light", "an off-white", "dark"),
    "dark": ("dark", "a near-black", "light"),
}

VISUAL_SYSTEM_PROMPT = """
You draw one picture for a spoken lesson. You receive the lesson so far, the tutor's brief for
this turn, and what the canvas showed last turn. You produce exactly one tool call and no prose.

For kind diagram call push_diagram with a mermaid flowchart or sequence diagram, structural,
under thirty nodes. For kind app call push_app with one complete HTML document: markup, style
and script inline, nothing fetched from the network, no fonts loaded from anywhere but the
tags below, under eight thousand characters, written quickly, so prefer a library call to
hand-written markup. Libraries available by exact tag, and nothing else:
<script src="/vendor/plotly.min.js"></script>
<link rel="stylesheet" href="/vendor/katex/katex.min.css"><script src="/vendor/katex/katex.min.js"></script>
<script src="/vendor/p5.min.js"></script>
<script src="/vendor/d3.min.js"></script>
The page is {theme}: draw on {background} background with {foreground} text and one accent,
and set the body background explicitly. The picture shows what the brief's show line says,
with its numbers and its case; title is the brief's title. Label axes and name the
quantities. No explanatory paragraph in the picture; the voice carries the explanation.
""".strip()


def visual_prompt(
    snapshot: TurnPrompt, brief: VisualBrief, previous: VisualBrief | None, theme: str
) -> TurnPrompt:
    name, background, foreground = THEMES[theme]
    system = VISUAL_SYSTEM_PROMPT.format(theme=name, background=background, foreground=foreground)
    last = f"{previous.kind}, {previous.title}" if previous is not None else "none"
    return TurnPrompt(
        system=f"{system}\n\nPrevious visual: {last}",
        history=snapshot.history,
        tool_context=snapshot.tool_context,
        user_text=f"Brief: {brief.model_dump_json()}\nThe learner just said: {snapshot.user_text}",
    )


async def run_visual_call(
    reasoning: ReasoningClient,
    prompt: TurnPrompt,
    channel: VisualChannel,
    max_tokens: int,
    may_land: Callable[[], bool],
) -> str:
    start = time.perf_counter()
    prose = 0
    call = None
    stream = reasoning.start_turn(
        prompt, tools=VISUAL_CALL_TOOLS, max_tokens=max_tokens, tool_choice="required"
    )
    async for chunk in stream:
        if chunk.kind == "spoken":
            prose += len(chunk.text)
        elif chunk.kind == "tool_call" and call is None:
            call = chunk
    if prose:
        logger.info("visual.prose chars=%d", prose)
    if call is None or not call.tool_name:
        return "visual: error: no tool call"
    if call.tool_name not in CALL_TOOL_NAMES:
        return f"visual: error: unexpected tool {call.tool_name}"
    if stream.finish_reason == "length":
        logger.info("visual.truncated tool=%s chars=%d", call.tool_name, len(call.text))
        return f"visual: error: truncated at {max_tokens} tokens"
    if not may_land():
        return "visual: superseded"
    result = await dispatch_visual_tool(call.tool_name, call.text, channel)
    logger.info(
        "visual.call tool=%s ms=%d finish=%s result=%s",
        call.tool_name,
        int((time.perf_counter() - start) * 1000),
        stream.finish_reason,
        "sent" if result.endswith(": sent") else "error",
    )
    return result
