import json
from collections.abc import Callable

import httpx2
import pytest

from tutor.config import Settings
from tutor.cost import TurnUsage, UsageLedger, turn_cost_usd
from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient

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


def _delta_frame(delta: dict[str, object], finish_reason: str | None = None) -> str:
    return _frame(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _usage_frame(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int | None = None,
) -> str:
    usage: dict[str, object] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return _frame(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": usage,
        }
    )


DONE = "data: [DONE]\n\n"


def _handler(
    body: str, bodies: list[dict[str, object]]
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    return handle


def _client(body: str, bodies: list[dict[str, object]]) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(body, bodies)))


def test_turn_cost_matches_measured_baseline() -> None:
    usage = TurnUsage(prompt_tokens=2300, completion_tokens=150)
    assert turn_cost_usd(usage) == pytest.approx(0.00021, rel=1e-6)

    with_extras = TurnUsage(
        prompt_tokens=2300, completion_tokens=150, cached_tokens=1800, reasoning_chars=4000
    )
    assert turn_cost_usd(with_extras) == turn_cost_usd(usage)


def test_list_pricing_doubles_the_turn() -> None:
    usage = TurnUsage(prompt_tokens=2300, completion_tokens=150)
    assert turn_cost_usd(usage, list_price=True) == pytest.approx(2 * turn_cost_usd(usage))
    assert turn_cost_usd(usage, list_price=False) == turn_cost_usd(usage)


def test_ledger_accumulates_across_turns() -> None:
    ledger = UsageLedger()
    assert ledger.turns == 0
    assert ledger.total == TurnUsage(prompt_tokens=0, completion_tokens=0)
    assert ledger.cost_usd() == 0.0

    turns = [
        TurnUsage(
            prompt_tokens=2300, completion_tokens=150, cached_tokens=1800, reasoning_chars=400
        ),
        TurnUsage(prompt_tokens=500, completion_tokens=80, cached_tokens=100, reasoning_chars=50),
        TurnUsage(
            prompt_tokens=1200, completion_tokens=300, cached_tokens=600, reasoning_chars=900
        ),
    ]
    for usage in turns:
        ledger.add(usage)

    assert ledger.turns == 3
    assert ledger.total.prompt_tokens == sum(t.prompt_tokens for t in turns)
    assert ledger.total.completion_tokens == sum(t.completion_tokens for t in turns)
    assert ledger.total.cached_tokens == sum(t.cached_tokens for t in turns)
    assert ledger.total.reasoning_chars == sum(t.reasoning_chars for t in turns)

    expected = sum(turn_cost_usd(t) for t in turns)
    assert ledger.cost_usd() == pytest.approx(expected)
    assert ledger.cost_usd(list_price=True) == pytest.approx(2 * expected)


async def test_missing_usage_chunk_leaves_usage_none() -> None:
    bodies: list[dict[str, object]] = []
    body = (
        _delta_frame({"reasoning_content": "the pool is bounded"})
        + _delta_frame({"content": "the connection pool"})
        + DONE
    )
    client = ReasoningClient(_settings(), http_client=_client(body, bodies))
    stream = client.start_turn(PROMPT)
    chunks = [chunk async for chunk in stream]
    await client.aclose()

    assert stream.usage is None
    assert stream.spoke is True
    assert len(chunks) == 2

    bodies_two: list[dict[str, object]] = []
    body_no_done = _delta_frame({"reasoning_content": "unfinished"})
    client_two = ReasoningClient(_settings(), http_client=_client(body_no_done, bodies_two))
    stream_two = client_two.start_turn(PROMPT)
    assert [chunk async for chunk in stream_two]
    await client_two.aclose()

    assert stream_two.usage is None


async def test_usage_populated_after_include_usage_drain() -> None:
    bodies: list[dict[str, object]] = []
    body = (
        _delta_frame({"reasoning_content": "the pool is "})
        + _delta_frame({"reasoning_content": "bounded"})
        + _delta_frame({"content": "the connection pool"})
        + _delta_frame({"content": " holds sockets"})
        + _delta_frame({}, "stop")
        + _usage_frame(2300, 150, cached_tokens=1800)
        + DONE
    )
    client = ReasoningClient(_settings(), http_client=_client(body, bodies))
    stream = client.start_turn(PROMPT)
    assert stream.usage is None

    chunks = [chunk async for chunk in stream]
    await client.aclose()

    assert chunks
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert stream.usage == TurnUsage(
        prompt_tokens=2300,
        completion_tokens=150,
        cached_tokens=1800,
        reasoning_chars=len("the pool is ") + len("bounded"),
    )
    assert turn_cost_usd(stream.usage) > 0.0

    bodies_two: list[dict[str, object]] = []
    body_no_details = _delta_frame({"content": "hello"}) + _usage_frame(10, 5) + DONE
    client_two = ReasoningClient(_settings(), http_client=_client(body_no_details, bodies_two))
    stream_two = client_two.start_turn(PROMPT)
    assert [chunk async for chunk in stream_two]
    await client_two.aclose()

    assert stream_two.usage is not None
    assert stream_two.usage.cached_tokens == 0
