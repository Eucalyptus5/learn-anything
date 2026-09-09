import json
import logging

from pydantic import BaseModel, ValidationError

from tutor.visuals import (
    AppPush,
    DiagramClear,
    DiagramPush,
    SourceHighlight,
    UngroundedVisual,
    VisualChannel,
)

logger = logging.getLogger(__name__)

_MODELS: dict[str, type[BaseModel]] = {
    "push_diagram": DiagramPush,
    "clear_diagram": DiagramClear,
    "highlight_source": SourceHighlight,
    "push_app": AppPush,
}

_DESCRIPTIONS = {
    "push_diagram": (
        "Render a mermaid flowchart or sequence diagram on the learner's canvas, replacing "
        "the diagram currently shown."
    ),
    "clear_diagram": "Remove every diagram from the learner's canvas.",
    "highlight_source": (
        "Highlight a range of lines in one file of the target repository on the learner's "
        "canvas; the path and lines must come from a search_code result in this turn."
    ),
    "push_app": (
        "Render a self-contained HTML page inside a sandboxed frame on the learner's canvas, "
        "replacing the page currently shown."
    ),
}


def _parameters(model: type[BaseModel]) -> dict[str, object]:
    schema = model.model_json_schema()
    schema.pop("title")
    del schema["properties"]["type"]
    for prop in schema["properties"].values():
        prop.pop("title")
    return schema


VISUAL_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": _DESCRIPTIONS[name],
            "parameters": _parameters(model),
        },
    }
    for name, model in _MODELS.items()
]


async def dispatch_visual_tool(name: str, arguments: str, channel: VisualChannel) -> str:
    model = _MODELS.get(name)
    if model is None:
        return f"{name}: error: unknown visual tool"
    try:
        body = json.loads(arguments)
    except json.JSONDecodeError:
        logger.warning("visual.tool_arguments tool=%s chars=%d", name, len(arguments))
        return f"{name}: error: arguments are not valid JSON"
    try:
        payload = model.model_validate(body)
    except ValidationError as exc:
        reasons = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg']}"
            for error in exc.errors()
        )
        return f"{name}: error: {reasons}"
    try:
        await channel.push(payload)
    except UngroundedVisual as exc:
        return f"{name}: error: {exc}"
    return f"{name}: sent"
