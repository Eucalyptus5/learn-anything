import json
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx2
from openai import AsyncOpenAI, AsyncStream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from tutor.config import Settings
from tutor.cost import TurnUsage
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

    def _emit(self, index: int) -> TurnChunk:
        partial = self._partial[index]
        self._emitted.add(index)
        return TurnChunk(
            kind="tool_call",
            text=partial.arguments,
            tool_call_id=partial.call_id,
            tool_name=partial.name,
        )

    def ready(self) -> list[TurnChunk]:
        chunks: list[TurnChunk] = []
        for index in sorted(self._partial):
            if index in self._emitted:
                continue
            try:
                json.loads(self._partial[index].arguments)
            except json.JSONDecodeError:
                continue
            chunks.append(self._emit(index))
        return chunks

    def flush(self) -> list[TurnChunk]:
        pending = [index for index in sorted(self._partial) if index not in self._emitted]
        return [self._emit(index) for index in pending]


def _turn_usage(raw_usage: CompletionUsage, reasoning_chars: int) -> TurnUsage:
    details = getattr(raw_usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0
    return TurnUsage(
        prompt_tokens=raw_usage.prompt_tokens,
        completion_tokens=raw_usage.completion_tokens,
        cached_tokens=cached_tokens,
        reasoning_chars=reasoning_chars,
    )


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
        self._stream: AsyncStream[ChatCompletionChunk] | None = None
        self._cancelled = False
        self.spoke = False
        self.finish_reason: str | None = None
        self.first_chunk_ms: int | None = None
        self.first_spoken_ms: int | None = None
        self.usage: TurnUsage | None = None

    async def _drain(self) -> AsyncIterator[TurnChunk]:
        if self._cancelled:
            return
        stream = await self._client.chat.completions.create(**self._request)
        self._stream = stream
        tool_calls = ToolCallAccumulator()
        reasoning_chars = 0
        raw_usage: CompletionUsage | None = None
        try:
            if self._cancelled:
                return
            async for raw in stream:
                if self._cancelled:
                    return
                if self.first_chunk_ms is None:
                    self.first_chunk_ms = _elapsed_ms(self._start)
                    logger.debug("llm_first_chunk ms=%d", self.first_chunk_ms)
                choices = getattr(raw, "choices", None)
                reason = getattr(choices[0], "finish_reason", None) if choices else None
                if reason:
                    self.finish_reason = reason
                usage = getattr(raw, "usage", None)
                if usage is not None:
                    raw_usage = usage
                chunk = parse_chunk(raw, tool_calls)
                for call in tool_calls.ready():
                    yield call
                if chunk is None:
                    continue
                if chunk.kind == "reasoning":
                    reasoning_chars += len(chunk.text)
                if chunk.kind == "spoken" and not self.spoke:
                    self.spoke = True
                    self.first_spoken_ms = _elapsed_ms(self._start)
                    logger.debug("llm_first_spoken ms=%d", self.first_spoken_ms)
                yield chunk
            if raw_usage is not None:
                self.usage = _turn_usage(raw_usage, reasoning_chars)
            if self._cancelled:
                return
            for call in tool_calls.flush():
                yield call
            if not self.spoke:
                logger.info(
                    "silent_turn finish_reason=%s prompt_tokens=%s completion_tokens=%s",
                    self.finish_reason,
                    self.usage.prompt_tokens if self.usage else None,
                    self.usage.completion_tokens if self.usage else None,
                )
        except (httpx2.ReadError, httpx2.RemoteProtocolError):
            if not self._cancelled:
                raise
        finally:
            await stream.close()

    def __aiter__(self) -> AsyncIterator[TurnChunk]:
        return self._drain()

    async def cancel(self) -> None:
        self._cancelled = True
        if self._stream is not None:
            await self._stream.close()


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
        tool_choice: str | None = None,
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
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return TurnStream(self._client, request, time.perf_counter())

    async def aclose(self) -> None:
        await self._client.close()
