import json
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


class TurnChunk(BaseModel):
    kind: Literal["reasoning", "spoken", "tool_call"]
    text: str
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class _PartialCall:
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""


class ToolCallAccumulator:
    def __init__(self) -> None:
        self._partial: dict[int, _PartialCall] = {}
        self._emitted: set[int] = set()

    def add(self, fragment: object) -> None:
        index = getattr(fragment, "index", None)
        if not isinstance(index, int):
            return
        partial = self._partial.setdefault(index, _PartialCall())
        call_id = getattr(fragment, "id", None)
        if call_id:
            partial.call_id = call_id
        function = getattr(fragment, "function", None)
        name = getattr(function, "name", None)
        if name:
            partial.name = name
        arguments = getattr(function, "arguments", None)
        if arguments:
            partial.arguments += arguments

    def ready(self) -> list[TurnChunk]:
        chunks: list[TurnChunk] = []
        for index in sorted(self._partial):
            if index in self._emitted:
                continue
            partial = self._partial[index]
            try:
                json.loads(partial.arguments)
            except json.JSONDecodeError:
                continue
            self._emitted.add(index)
            chunks.append(
                TurnChunk(
                    kind="tool_call",
                    text=partial.arguments,
                    tool_call_id=partial.call_id,
                    tool_name=partial.name,
                )
            )
        return chunks


def parse_chunk(chunk: object, tool_calls: ToolCallAccumulator) -> TurnChunk | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None

    for fragment in getattr(delta, "tool_calls", None) or ():
        tool_calls.add(fragment)

    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        return TurnChunk(kind="reasoning", text=reasoning)

    content = getattr(delta, "content", None)
    if isinstance(content, str) and content:
        return TurnChunk(kind="spoken", text=content)

    return None
