import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx2
from openai import AsyncOpenAI
from pydantic import BaseModel

from tutor.config import Settings
from tutor.prompt import TurnPrompt

logger = logging.getLogger(__name__)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


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


class TurnStream:
    def __init__(self, client: AsyncOpenAI, request: dict[str, object], start: float) -> None:
        self._client = client
        self._request = request
        self._start = start
        self.spoke = False
        self.first_chunk_ms: int | None = None
        self.first_spoken_ms: int | None = None

    async def _drain(self) -> AsyncIterator[TurnChunk]:
        stream = await self._client.chat.completions.create(**self._request)
        tool_calls = ToolCallAccumulator()
        async for raw in stream:
            if self.first_chunk_ms is None:
                self.first_chunk_ms = _elapsed_ms(self._start)
                logger.debug("llm_first_chunk ms=%d", self.first_chunk_ms)
            chunk = parse_chunk(raw, tool_calls)
            if chunk is None:
                continue
            if chunk.kind == "spoken" and not self.spoke:
                self.spoke = True
                self.first_spoken_ms = _elapsed_ms(self._start)
                logger.debug("llm_first_spoken ms=%d", self.first_spoken_ms)
            yield chunk

    def __aiter__(self) -> AsyncIterator[TurnChunk]:
        return self._drain()


class ReasoningClient:
    def __init__(self, cfg: Settings, http_client: httpx2.AsyncClient | None = None) -> None:
        self._cfg = cfg
        injected = {"http_client": http_client} if http_client is not None else {}
        self._client = AsyncOpenAI(
            api_key=cfg.reasoning_api_key.get_secret_value(),
            base_url=cfg.reasoning_api_base,
            **injected,
        )

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> TurnStream:
        request: dict[str, object] = {
            "model": self._cfg.reasoning_model,
            "messages": prompt.messages(),
            "reasoning_effort": effort if effort is not None else self._cfg.reasoning_effort,
            "max_tokens": max_tokens if max_tokens is not None else self._cfg.reasoning_max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools is not None:
            request["tools"] = list(tools)
        return TurnStream(self._client, request, time.perf_counter())

    async def aclose(self) -> None:
        await self._client.close()
