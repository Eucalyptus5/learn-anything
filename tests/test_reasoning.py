from types import SimpleNamespace

from openai.types.chat import ChatCompletionChunk

from tutor.reasoning import ToolCallAccumulator, parse_chunk


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
