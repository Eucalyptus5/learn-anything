import inspect
import json

from tutor.brief import BRIEF_END, BRIEF_MARKER
from tutor.prompt import (
    SEARCH_CODE_TOOL,
    SYSTEM_PROMPT,
    TOOL_CONTEXT_BYTES,
    Message,
    ToolCall,
    ToolCallFunction,
    TurnPrompt,
)
from tutor.tools.models import SearchMatch, SearchResult
from tutor.tools.search import search


def _match(path: str, line: int, text: str) -> SearchMatch:
    return SearchMatch(path=path, line=line, text=text, before=[], after=[])


def _result(
    query: str,
    match_count: int,
    text_len: int,
    byte_count: int = 0,
    truncated: bool = False,
) -> SearchResult:
    return SearchResult(
        tool="search_code",
        query=query,
        globs=["src/**/*.py"],
        matches=[_match("src/pool.py", i + 1, "x" * text_len) for i in range(match_count)],
        truncated=truncated,
        oversized=False,
        byte_count=byte_count,
    )


def test_system_message_is_byte_identical_across_turns() -> None:
    history = [Message(role="user", content="what does acquire do")]
    tool_context = [_result("acquire", 2, 20)]

    first = TurnPrompt(
        system="you are the tutor", history=history, tool_context=tool_context, user_text="one"
    )
    second = TurnPrompt(
        system="you are the tutor", history=history, tool_context=tool_context, user_text="two"
    )

    first_zero = json.dumps(first.messages()[0], sort_keys=True).encode()
    second_zero = json.dumps(second.messages()[0], sort_keys=True).encode()

    assert first_zero == second_zero
    assert first.messages()[-1] != second.messages()[-1]


def test_tool_context_precedes_user_turn() -> None:
    history = [
        Message(role="user", content="how does the pool work"),
        Message(role="assistant", content="it hands out connections"),
    ]
    ctx0 = _result("acquire", 1, 10)
    ctx1 = _result("release", 1, 10)

    prompt = TurnPrompt(
        system="you are the tutor",
        history=history,
        tool_context=[ctx0, ctx1],
        user_text="walk me through it",
    )

    messages = prompt.messages()

    assert messages == [
        {"role": "system", "content": "you are the tutor"},
        {"role": "user", "content": "how does the pool work"},
        {"role": "assistant", "content": "it hands out connections"},
        {"role": "user", "content": ctx0.model_dump_json()},
        {"role": "user", "content": ctx1.model_dump_json()},
        {"role": "user", "content": "walk me through it"},
    ]


def test_history_order_preserved() -> None:
    history = [
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
        Message(role="user", content="second question"),
        Message(role="assistant", content="second answer"),
    ]

    prompt = TurnPrompt(
        system="you are the tutor", history=history, tool_context=[], user_text="third question"
    )

    messages = prompt.messages()
    middle = messages[1:5]

    assert middle == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]


def test_no_volatile_content_in_prefix() -> None:
    system = "you are the tutor"
    prompt = TurnPrompt(system=system, history=[], tool_context=[], user_text="anything")

    assert prompt.messages()[0] == {"role": "system", "content": system}
    assert prompt.messages()[0] == prompt.messages()[0]

    other = TurnPrompt(
        system=system,
        history=[Message(role="user", content="different history")],
        tool_context=[_result("acquire", 1, 10)],
        user_text="something else entirely",
    )

    assert other.messages()[0] == {"role": "system", "content": system}


def test_tool_context_capped_on_serialized_bytes() -> None:
    fitting = [_result(f"query{i}", match_count=3, text_len=200, byte_count=0) for i in range(5)]
    overflowing = _result("overflow", match_count=40, text_len=500, byte_count=0)
    dropped = _result("dropped", match_count=3, text_len=200, byte_count=0)

    prompt = TurnPrompt(
        system="you are the tutor",
        history=[],
        tool_context=[*fitting, overflowing, dropped],
        user_text="anything",
    )

    total = sum(len(r.model_dump_json()) for r in prompt.tool_context)
    assert total <= TOOL_CONTEXT_BYTES

    for original, kept in zip(fitting, prompt.tool_context[: len(fitting)]):
        assert kept == original

    assert len(prompt.tool_context) == len(fitting) + 1
    survivor = prompt.tool_context[-1]
    assert survivor.query == "overflow"
    assert len(survivor.matches) < len(overflowing.matches)
    assert survivor.matches == overflowing.matches[: len(survivor.matches)]
    assert survivor.truncated is True

    messages = prompt.messages()
    ctx_messages = messages[1 : 1 + len(prompt.tool_context)]
    assert [m["content"] for m in ctx_messages] == [
        r.model_dump_json() for r in prompt.tool_context
    ]


def test_tool_context_ignores_byte_count_when_result_is_small() -> None:
    tiny = _result("tiny", match_count=1, text_len=5, byte_count=10**7)

    prompt = TurnPrompt(
        system="you are the tutor", history=[], tool_context=[tiny], user_text="anything"
    )

    assert prompt.tool_context == [tiny]


def test_result_zero_keeps_one_oversized_match() -> None:
    huge = _result("huge", match_count=1, text_len=TOOL_CONTEXT_BYTES * 2, byte_count=0)

    prompt = TurnPrompt(
        system="you are the tutor", history=[], tool_context=[huge], user_text="anything"
    )

    assert len(prompt.tool_context) == 1
    assert prompt.tool_context[0] is huge
    assert prompt.tool_context[0].truncated is False


def test_result_zero_with_several_oversized_matches_keeps_exactly_one() -> None:
    huge = _result("huge", match_count=5, text_len=TOOL_CONTEXT_BYTES // 2, byte_count=0)

    prompt = TurnPrompt(
        system="you are the tutor", history=[], tool_context=[huge], user_text="anything"
    )

    assert len(prompt.tool_context) == 1
    assert len(prompt.tool_context[0].matches) == 1
    assert prompt.tool_context[0].truncated is True


def test_one_byte_overflow_drops_a_match_rather_than_relabeling_all() -> None:
    matches = [_match("src/pool.py", i + 1, "x" * 10) for i in range(3)]
    result = SearchResult(
        tool="search_code",
        query="acquire",
        globs=["src/**/*.py"],
        matches=matches,
        truncated=False,
        oversized=False,
        byte_count=0,
    )
    target = TOOL_CONTEXT_BYTES + 1
    pad = target - len(result.model_dump_json())
    matches[-1] = _match("src/pool.py", 3, "x" * (10 + pad))
    result = result.model_copy(update={"matches": matches})
    assert len(result.model_dump_json()) == target

    prompt = TurnPrompt(
        system="you are the tutor", history=[], tool_context=[result], user_text="anything"
    )

    kept = prompt.tool_context[0]
    assert len(kept.matches) < len(result.matches)
    assert kept.truncated is True


def test_second_result_dropped_when_its_fitting_prefix_would_be_empty() -> None:
    zero = _result("zero", match_count=1, text_len=11800, byte_count=0)
    one = _result("one", match_count=1, text_len=1000, byte_count=0)
    remaining = TOOL_CONTEXT_BYTES - len(zero.model_dump_json())
    assert 0 < remaining < len(_match("src/pool.py", 1, "x" * 1000).model_dump_json())

    prompt = TurnPrompt(
        system="you are the tutor", history=[], tool_context=[zero, one], user_text="anything"
    )

    assert prompt.tool_context == [zero]
    assert prompt.tool_context[0] is zero
    assert prompt.tool_context[0].truncated == zero.truncated

    messages = prompt.messages()
    tool_messages = [
        m for m in messages if m["role"] == "user" and m["content"] == zero.model_dump_json()
    ]
    assert len(tool_messages) == 1


def _tool_call(call_id: str, arguments: str) -> ToolCall:
    return ToolCall(id=call_id, function=ToolCallFunction(name="search_code", arguments=arguments))


def test_tool_exchange_follows_the_user_turn() -> None:
    ctx = _result("acquire", 1, 10)
    call = _tool_call("call_a", '{"query": "acquire", "globs": ["src/**/*.py"]}')

    prompt = TurnPrompt(
        system="you are the tutor",
        history=[Message(role="user", content="how does the pool work")],
        tool_context=[ctx],
        user_text="walk me through it",
        tool_exchange=[
            Message(role="assistant", content="", tool_calls=[call]),
            Message(role="tool", content=ctx.model_dump_json(), tool_call_id="call_a"),
        ],
    )

    messages = prompt.messages()

    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "user",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[3] == {"role": "user", "content": "walk me through it"}
    assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"]


def test_emitted_tool_call_is_api_shaped() -> None:
    arguments = '{"query": "acquire", "globs": ["src/**/*.py"]}'

    prompt = TurnPrompt(
        system="you are the tutor",
        user_text="walk me through it",
        tool_exchange=[
            Message(role="assistant", content="", tool_calls=[_tool_call("call_a", arguments)])
        ],
    )

    entry = json.loads(json.dumps(prompt.messages()[-1]))["tool_calls"][0]

    assert entry == {
        "id": "call_a",
        "type": "function",
        "function": {"name": "search_code", "arguments": arguments},
    }
    assert json.loads(entry["function"]["arguments"]) == {
        "query": "acquire",
        "globs": ["src/**/*.py"],
    }


def test_optional_keys_appear_only_where_set() -> None:
    prompt = TurnPrompt(
        system="you are the tutor",
        history=[Message(role="assistant", content="it hands out connections")],
        user_text="walk me through it",
        tool_exchange=[
            Message(role="assistant", content="", tool_calls=[_tool_call("call_a", "{}")]),
            Message(role="tool", content="{}", tool_call_id="call_a"),
        ],
    )

    messages = prompt.messages()

    for plain in messages[:3]:
        assert set(plain) == {"role", "content"}
    assert set(messages[-2]) == {"role", "content", "tool_calls"}
    assert set(messages[-1]) == {"role", "content", "tool_call_id"}


def test_two_calls_keep_their_pairing_and_order() -> None:
    calls = [
        _tool_call("call_a", '{"query": "acquire"}'),
        _tool_call("call_b", '{"query": "release"}'),
    ]

    prompt = TurnPrompt(
        system="you are the tutor",
        user_text="walk me through it",
        tool_exchange=[
            Message(role="assistant", content="", tool_calls=calls),
            Message(role="tool", content="acquire result", tool_call_id="call_a"),
            Message(role="tool", content="release result", tool_call_id="call_b"),
        ],
    )

    messages = prompt.messages()

    assert [entry["id"] for entry in messages[-3]["tool_calls"]] == ["call_a", "call_b"]
    assert [(m["tool_call_id"], m["content"]) for m in messages[-2:]] == [
        ("call_a", "acquire result"),
        ("call_b", "release result"),
    ]


def test_search_code_schema_is_json_data() -> None:
    assert json.loads(json.dumps(SEARCH_CODE_TOOL)) == SEARCH_CODE_TOOL
    assert SEARCH_CODE_TOOL["function"]["name"] == "search_code"


def test_search_code_schema_describes_only_model_chosen_arguments() -> None:
    properties = SEARCH_CODE_TOOL["function"]["parameters"]["properties"]

    assert set(properties) == {"query", "globs"}
    assert set(properties) <= set(inspect.signature(search).parameters)


def test_prompt_without_a_tool_exchange_is_unchanged() -> None:
    ctx = _result("acquire", 1, 10)

    prompt = TurnPrompt(
        system="you are the tutor",
        history=[Message(role="user", content="how does the pool work")],
        tool_context=[ctx],
        user_text="walk me through it",
    )

    assert prompt.tool_exchange == []
    assert prompt.messages() == [
        {"role": "system", "content": "you are the tutor"},
        {"role": "user", "content": "how does the pool work"},
        {"role": "user", "content": ctx.model_dump_json()},
        {"role": "user", "content": "walk me through it"},
    ]


def test_the_system_prompt_names_the_brief_markers_and_stays_ascii() -> None:
    assert BRIEF_MARKER in SYSTEM_PROMPT
    assert BRIEF_END in SYSTEM_PROMPT
    assert SYSTEM_PROMPT.isascii()
    assert "push_diagram" not in SYSTEM_PROMPT


def test_the_system_prompt_names_the_three_phases() -> None:
    assert "Teach:" in SYSTEM_PROMPT
    assert "Concrete:" in SYSTEM_PROMPT
    assert "Interrogate:" in SYSTEM_PROMPT
    assert "Explore" not in SYSTEM_PROMPT
    assert "Reverse Feynman" not in SYSTEM_PROMPT
