import asyncio
import http.client
import io
import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace

import httpx2
import pytest
from openai.types.chat import ChatCompletionChunk

from tutor.config import Settings
from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient, ToolCallAccumulator, TurnChunk, TurnStream, parse_chunk

HANG_GUARD_S = 20.0


def _chunk(**delta_fields: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(**delta_fields))]
    )


def _fragment(index: int, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(index=index, **fields)


def test_reasoning_delta_becomes_reasoning_chunk() -> None:
    chunk = parse_chunk(_chunk(reasoning_content="the pool is bounded"), ToolCallAccumulator())
    assert chunk is not None
    assert chunk.kind == "reasoning"
    assert chunk.text == "the pool is bounded"

    typed = ChatCompletionChunk.model_validate(
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {"index": 0, "delta": {"reasoning_content": "the pool is bounded"}},
            ],
        }
    )
    from_sdk = parse_chunk(typed, ToolCallAccumulator())
    assert from_sdk is not None
    assert from_sdk.kind == "reasoning"
    assert from_sdk.text == "the pool is bounded"


def test_content_delta_becomes_spoken_chunk() -> None:
    chunk = parse_chunk(_chunk(content="the connection pool"), ToolCallAccumulator())
    assert chunk is not None
    assert chunk.kind == "spoken"
    assert chunk.text == "the connection pool"
    assert chunk.tool_call_id is None
    assert chunk.tool_name is None


def test_reasoning_alias_field_accepted() -> None:
    chunk = parse_chunk(_chunk(reasoning="weighing two call sites"), ToolCallAccumulator())
    assert chunk is not None
    assert chunk.kind == "reasoning"
    assert chunk.text == "weighing two call sites"


def test_usage_only_chunk_yields_no_text() -> None:
    usage = SimpleNamespace(prompt_tokens=210, completion_tokens=64, total_tokens=274)
    chunk = SimpleNamespace(choices=[], usage=usage)
    assert parse_chunk(chunk, ToolCallAccumulator()) is None


def test_empty_choices_list_is_skipped() -> None:
    assert parse_chunk(SimpleNamespace(choices=[]), ToolCallAccumulator()) is None
    assert parse_chunk(SimpleNamespace(), ToolCallAccumulator()) is None
    assert parse_chunk(SimpleNamespace(choices=None), ToolCallAccumulator()) is None
    assert parse_chunk(SimpleNamespace(choices=[SimpleNamespace()]), ToolCallAccumulator()) is None


def test_null_content_is_skipped() -> None:
    assert parse_chunk(_chunk(content=None), ToolCallAccumulator()) is None
    assert parse_chunk(_chunk(content=""), ToolCallAccumulator()) is None
    assert parse_chunk(_chunk(), ToolCallAccumulator()) is None


def test_tool_call_fragments_accumulate_by_index() -> None:
    accumulator = ToolCallAccumulator()

    opening = _chunk(
        tool_calls=[
            _fragment(
                0,
                id="call_a",
                function=SimpleNamespace(name="search_code", arguments='{"pat'),
            )
        ]
    )
    assert parse_chunk(opening, accumulator) is None
    assert accumulator.ready() == []

    middle = _chunk(tool_calls=[_fragment(0, function=SimpleNamespace(arguments='tern": "acq'))])
    assert parse_chunk(middle, accumulator) is None
    assert accumulator.ready() == []

    closing = _chunk(tool_calls=[_fragment(0, function=SimpleNamespace(arguments='uire"}'))])
    assert parse_chunk(closing, accumulator) is None

    emitted = accumulator.ready()
    assert len(emitted) == 1
    assert emitted[0].kind == "tool_call"
    assert emitted[0].tool_call_id == "call_a"
    assert emitted[0].tool_name == "search_code"
    assert emitted[0].text == '{"pattern": "acquire"}'

    assert accumulator.ready() == []


def test_two_indices_yield_two_chunks() -> None:
    accumulator = ToolCallAccumulator()

    fragments = [
        _fragment(0, id="call_a", function=SimpleNamespace(name="search_code", arguments='{"pat')),
        _fragment(1, id="call_b", function=SimpleNamespace(name="read_file", arguments='{"pa')),
        _fragment(0, function=SimpleNamespace(arguments='tern": "acq')),
        _fragment(1, function=SimpleNamespace(arguments='th": "tutor/tts')),
        _fragment(0, function=SimpleNamespace(arguments='uire"}')),
    ]
    for fragment in fragments:
        assert parse_chunk(_chunk(tool_calls=[fragment]), accumulator) is None

    tail = _fragment(1, function=SimpleNamespace(arguments='.py"}'))
    assert parse_chunk(_chunk(tool_calls=[tail]), accumulator) is None

    emitted = accumulator.ready()
    assert len(emitted) == 2
    assert [chunk.kind for chunk in emitted] == ["tool_call", "tool_call"]
    assert [chunk.tool_call_id for chunk in emitted] == ["call_a", "call_b"]
    assert [chunk.tool_name for chunk in emitted] == ["search_code", "read_file"]
    assert emitted[0].text == '{"pattern": "acquire"}'
    assert emitted[1].text == '{"path": "tutor/tts.py"}'
    assert accumulator.ready() == []


FAKE_BASE = "https://reasoning.invalid/v1"
FAKE_KEY = "sk-test-not-a-real-key"

PROMPT = TurnPrompt(
    system="you are the tutor", history=[], tool_context=[], user_text="walk me through it"
)


def _settings(**overrides: object) -> Settings:
    fields: dict[str, object] = {"reasoning_api_base": FAKE_BASE, "reasoning_api_key": FAKE_KEY}
    return Settings(_env_file=None, **{**fields, **overrides})


def _frame(payload: dict[str, object]) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


def _delta_frame(delta: dict[str, object]) -> str:
    return _frame(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
    )


def _sse_body(deltas: list[dict[str, object]]) -> str:
    frames = [_delta_frame(delta) for delta in deltas]
    frames.append(
        _frame(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        )
    )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def _handler(
    deltas: list[dict[str, object]], bodies: list[dict[str, object]]
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse_body(deltas),
        )

    return handle


def _mock_client(
    deltas: list[dict[str, object]], bodies: list[dict[str, object]]
) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(deltas, bodies)))


TWO_THEN_THREE: list[dict[str, object]] = [
    {"reasoning_content": "the pool is bounded"},
    {"reasoning_content": " by max_size"},
    {"content": "the connection pool"},
    {"content": " holds at most"},
    {"content": " sixteen sockets"},
]


async def test_reasoning_chunks_precede_spoken_chunks() -> None:
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_mock_client(TWO_THEN_THREE, bodies))

    chunks = [chunk async for chunk in client.start_turn(PROMPT)]
    await client.aclose()

    assert [chunk.kind for chunk in chunks] == [
        "reasoning",
        "reasoning",
        "spoken",
        "spoken",
        "spoken",
    ]
    assert [chunk.text for chunk in chunks] == [
        "the pool is bounded",
        " by max_size",
        "the connection pool",
        " holds at most",
        " sixteen sockets",
    ]


async def test_spoke_flag_set_on_first_content() -> None:
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_mock_client(TWO_THEN_THREE, bodies))

    stream = client.start_turn(PROMPT)
    assert stream.spoke is False

    seen = 0
    async for chunk in stream:
        seen += 1
        if chunk.kind == "reasoning":
            assert stream.spoke is False
        else:
            assert stream.spoke is True

    await client.aclose()

    assert seen == 5
    assert stream.spoke is True


async def test_first_spoken_ms_recorded_once() -> None:
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_mock_client(TWO_THEN_THREE, bodies))

    stream = client.start_turn(PROMPT)
    assert stream.first_spoken_ms is None

    recorded: int | None = None
    async for chunk in stream:
        if chunk.kind == "reasoning":
            assert stream.first_spoken_ms is None
            continue
        if recorded is None:
            recorded = stream.first_spoken_ms
            assert isinstance(recorded, int)
            assert recorded >= 0
        assert stream.first_spoken_ms == recorded

    await client.aclose()

    assert stream.first_spoken_ms == recorded


async def test_first_chunk_ms_recorded() -> None:
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_mock_client(TWO_THEN_THREE, bodies))

    stream = client.start_turn(PROMPT)
    assert stream.first_chunk_ms is None

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        assert isinstance(stream.first_chunk_ms, int)
        assert stream.first_chunk_ms >= 0
    assert len(chunks) == 5

    silent = ReasoningClient(_settings(), http_client=_mock_client([], bodies))
    quiet = silent.start_turn(PROMPT)
    assert [chunk async for chunk in quiet] == []

    await client.aclose()
    await silent.aclose()

    assert isinstance(quiet.first_chunk_ms, int)
    assert quiet.first_chunk_ms >= 0
    assert quiet.spoke is False
    assert quiet.first_spoken_ms is None


async def test_client_reuses_one_http_client() -> None:
    bodies: list[dict[str, object]] = []
    http = _mock_client(TWO_THEN_THREE, bodies)
    client = ReasoningClient(_settings(), http_client=http)

    first = [chunk async for chunk in client.start_turn(PROMPT)]
    second = [chunk async for chunk in client.start_turn(PROMPT)]

    assert len(first) == 5
    assert len(second) == 5
    assert len(bodies) == 2

    await client.aclose()
    assert http.is_closed is True


async def test_start_turn_arguments_override_settings() -> None:
    bodies: list[dict[str, object]] = []
    cfg = _settings(reasoning_effort="medium", reasoning_max_tokens=123)
    client = ReasoningClient(cfg, http_client=_mock_client(TWO_THEN_THREE, bodies))

    assert [chunk async for chunk in client.start_turn(PROMPT)]

    assert bodies[0]["model"] == "glm-5.3-flash"
    assert bodies[0]["reasoning_effort"] == "medium"
    assert bodies[0]["max_tokens"] == 123
    assert bodies[0]["stream"] is True
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert bodies[0]["messages"] == PROMPT.messages()
    assert "tools" not in bodies[0]

    tools = [
        {"type": "function", "function": {"name": "search_code", "parameters": {"type": "object"}}}
    ]
    assert [
        chunk
        async for chunk in client.start_turn(PROMPT, tools=tools, effort="high", max_tokens=77)
    ]

    await client.aclose()

    assert bodies[1]["reasoning_effort"] == "high"
    assert bodies[1]["max_tokens"] == 77
    assert bodies[1]["tools"] == tools


class _ParkedBody(httpx2.AsyncByteStream):
    def __init__(self, head: list[str], release: asyncio.Event, tail: list[str]) -> None:
        self._head = head
        self._release = release
        self._tail = tail
        self.parked = asyncio.Event()
        self.closes = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._head:
            yield frame.encode()
        self.parked.set()
        await self._release.wait()
        for frame in self._tail:
            yield frame.encode()

    async def aclose(self) -> None:
        self.closes += 1


def _parked_client(body: _ParkedBody, bodies: list[dict[str, object]]) -> httpx2.AsyncClient:
    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handle))


async def _collect(stream: TurnStream, got_first: asyncio.Event) -> list[TurnChunk]:
    seen: list[TurnChunk] = []
    async for chunk in stream:
        seen.append(chunk)
        got_first.set()
    return seen


FIRST = _delta_frame({"content": "the pool"})
SECOND = _delta_frame({"content": " is bounded"})


async def test_cancel_closes_the_response() -> None:
    release = asyncio.Event()
    body = _ParkedBody([FIRST], release, [SECOND])
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_parked_client(body, bodies))
    stream = client.start_turn(PROMPT)
    got_first = asyncio.Event()
    consumer = asyncio.create_task(_collect(stream, got_first))

    await got_first.wait()
    await stream.cancel()
    assert body.closes == 1

    release.set()
    chunks = await consumer
    await client.aclose()

    assert [chunk.text for chunk in chunks] == ["the pool"]
    assert stream.spoke is True


async def test_cancel_is_idempotent() -> None:
    release = asyncio.Event()
    body = _ParkedBody([FIRST], release, [])
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_parked_client(body, bodies))
    stream = client.start_turn(PROMPT)
    got_first = asyncio.Event()
    consumer = asyncio.create_task(_collect(stream, got_first))

    await got_first.wait()
    await stream.cancel()
    await stream.cancel()
    assert body.closes == 1

    release.set()
    assert [chunk.text for chunk in await consumer] == ["the pool"]
    await stream.cancel()
    assert body.closes == 1

    untouched = client.start_turn(PROMPT)
    await untouched.cancel()
    await untouched.cancel()
    assert len(bodies) == 1

    plain = ReasoningClient(_settings(), http_client=_mock_client(TWO_THEN_THREE, bodies))
    finished = plain.start_turn(PROMPT)
    assert len([chunk async for chunk in finished]) == 5
    await finished.cancel()

    await client.aclose()
    await plain.aclose()


async def test_cancelled_error_propagates_to_caller() -> None:
    release = asyncio.Event()
    body = _ParkedBody([FIRST], release, [SECOND])
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_parked_client(body, bodies))
    stream = client.start_turn(PROMPT)
    got_first = asyncio.Event()
    consumer = asyncio.create_task(_collect(stream, got_first))

    await got_first.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await client.aclose()

    assert consumer.cancelled()
    assert body.closes == 1


async def test_cancel_before_first_chunk_is_clean() -> None:
    release = asyncio.Event()
    body = _ParkedBody([], release, [FIRST])
    bodies: list[dict[str, object]] = []
    client = ReasoningClient(_settings(), http_client=_parked_client(body, bodies))
    stream = client.start_turn(PROMPT)
    consumer = asyncio.create_task(_collect(stream, asyncio.Event()))

    await body.parked.wait()
    await stream.cancel()
    assert body.closes == 1

    release.set()
    assert await consumer == []
    assert stream.spoke is False
    assert len(bodies) == 1
    await client.aclose()

    unsent: list[dict[str, object]] = []
    quiet = ReasoningClient(_settings(), http_client=_parked_client(body, unsent))
    fresh = quiet.start_turn(PROMPT)
    await fresh.cancel()
    assert [chunk async for chunk in fresh] == []
    assert unsent == []
    assert fresh.spoke is False
    await quiet.aclose()


async def test_cancel_stops_the_consumer() -> None:
    park = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        _, _, raw_headers = head.partition(b"\r\n")
        headers = http.client.parse_headers(io.BytesIO(raw_headers))
        await reader.readexactly(int(headers["content-length"]))
        frame = FIRST.encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"content-type: text/event-stream\r\n"
            b"transfer-encoding: chunked\r\n"
            b"\r\n" + f"{len(frame):x}\r\n".encode() + frame + b"\r\n"
        )
        await writer.drain()
        await park.wait()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ReasoningClient(_settings(reasoning_api_base=f"http://127.0.0.1:{port}/v1"))
    try:
        stream = client.start_turn(PROMPT)
        got_first = asyncio.Event()
        consumer = asyncio.create_task(_collect(stream, got_first))

        await got_first.wait()
        await stream.cancel()
        async with asyncio.timeout(HANG_GUARD_S):
            chunks = await consumer

        assert [chunk.text for chunk in chunks] == ["the pool"]
        assert stream.spoke is True
    finally:
        park.set()
        server.close()
        await server.wait_closed()
        await client.aclose()
