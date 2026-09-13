import asyncio
import json
import logging

import pytest

from tests.fakes import FakeConnection, grounded_registry
from tutor.visual_tools import (
    VISUAL_CALL_TOOLS,
    VISUAL_TOOLS,
    VOICE_VISUAL_TOOLS,
    dispatch_visual_tool,
)
from tutor.visuals import AppPush, DiagramClear, DiagramPush, SourceHighlight, VisualChannel

_TOOL_MODELS = [
    ("push_diagram", DiagramPush),
    ("clear_diagram", DiagramClear),
    ("highlight_source", SourceHighlight),
    ("push_app", AppPush),
]

_HOSTILE_HTML = "<script>alert(1)</script><img src=x onerror=alert(2)>"


def _grounded_channel(connection: FakeConnection) -> VisualChannel:
    channel = VisualChannel(connection)
    channel.set_grounding(grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9]), "t1")
    return channel


def test_tool_schema_matches_the_payload_models() -> None:
    assert [t["function"]["name"] for t in VISUAL_TOOLS] == [name for name, _ in _TOOL_MODELS]
    json.dumps(VISUAL_TOOLS)

    for tool, (name, model) in zip(VISUAL_TOOLS, _TOOL_MODELS, strict=True):
        assert set(tool) == {"type", "function"}
        assert tool["type"] == "function"
        function = tool["function"]
        assert function["name"] == name
        assert isinstance(function["description"], str) and function["description"]
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert set(parameters["properties"]) == set(model.model_fields) - {"type"}
        expected_required = [
            field for field, info in model.model_fields.items() if info.is_required()
        ]
        assert parameters.get("required", []) == expected_required

    by_name = {t["function"]["name"]: t["function"]["parameters"] for t in VISUAL_TOOLS}
    assert by_name["push_diagram"]["required"] == ["id", "kind", "source", "title"]
    assert by_name["highlight_source"]["required"] == ["path", "start_line", "end_line"]
    assert by_name["push_app"]["required"] == ["id", "html", "title"]
    assert by_name["push_diagram"]["properties"]["id"]["maxLength"] == 64
    assert by_name["push_diagram"]["properties"]["source"]["maxLength"] == 8000
    assert by_name["push_diagram"]["properties"]["kind"]["enum"] == ["flowchart", "sequence"]
    assert by_name["highlight_source"]["properties"]["path"]["maxLength"] == 4096
    assert by_name["highlight_source"]["properties"]["start_line"]["minimum"] == 1
    assert by_name["highlight_source"]["properties"]["end_line"]["minimum"] == 1
    assert by_name["push_app"]["properties"]["id"]["maxLength"] == 64
    assert by_name["push_app"]["properties"]["html"]["maxLength"] == 64000
    assert "search_code" in VISUAL_TOOLS[2]["function"]["description"]


def test_the_tool_schema_carries_title_for_both_pushes() -> None:
    by_name = {t["function"]["name"]: t["function"] for t in VISUAL_TOOLS}

    for name in ("push_diagram", "push_app"):
        parameters = by_name[name]["parameters"]
        assert "title" in parameters["required"]
        assert parameters["properties"]["title"] == {"type": "string", "maxLength": 80}
        assert "under eighty characters" in by_name[name]["description"]
    assert "title" not in by_name["clear_diagram"]["parameters"]["properties"]
    assert "title" not in by_name["highlight_source"]["parameters"]["properties"]


def test_the_call_tools_and_voice_tools_partition_the_surface() -> None:
    call_names = [t["function"]["name"] for t in VISUAL_CALL_TOOLS]
    voice_names = [t["function"]["name"] for t in VOICE_VISUAL_TOOLS]

    assert call_names == ["push_diagram", "push_app"]
    assert voice_names == ["clear_diagram", "highlight_source"]
    assert sorted(call_names + voice_names) == sorted(t["function"]["name"] for t in VISUAL_TOOLS)
    assert all(t in VISUAL_TOOLS for t in [*VISUAL_CALL_TOOLS, *VOICE_VISUAL_TOOLS])


@pytest.mark.parametrize("arguments", ["{", "", "not json", "{'id': 1}"])
async def test_malformed_json_arguments_return_an_error_string(arguments: str) -> None:
    connection = FakeConnection()

    result = await dispatch_visual_tool("push_diagram", arguments, VisualChannel(connection))

    assert isinstance(result, str)
    assert result.startswith("push_diagram: error:")
    assert "\n" not in result
    assert connection.sent == []


async def test_malformed_json_arguments_are_logged_without_their_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection = FakeConnection()
    arguments = json.dumps(
        {"id": "d1", "kind": "flowchart", "source": "graph TD; A-->B", "title": "reader"}
    )[:-6]

    with caplog.at_level(logging.WARNING, logger="tutor.visual_tools"):
        result = await dispatch_visual_tool("push_diagram", arguments, VisualChannel(connection))

    assert result.startswith("push_diagram: error:")
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [f"visual.tool_arguments tool=push_diagram chars={len(arguments)}"]
    assert "graph TD" not in messages[0]
    assert connection.sent == []


@pytest.mark.parametrize("arguments", ["[]", "null", '"x"', "3"])
async def test_non_object_json_returns_an_error_string(arguments: str) -> None:
    connection = FakeConnection()

    result = await dispatch_visual_tool("push_diagram", arguments, VisualChannel(connection))

    assert isinstance(result, str)
    assert result.startswith("push_diagram: error:")
    assert "\n" not in result
    assert connection.sent == []


@pytest.mark.parametrize(
    ("name", "body"),
    [
        (
            "push_diagram",
            {"id": "d1", "kind": "flowchart", "source": "g", "title": "r", "extra": 1},
        ),
        ("push_diagram", {"id": "d1", "kind": "gantt", "source": "g", "title": "r"}),
        ("push_diagram", {"id": "d1", "kind": "flowchart", "title": "r"}),
        ("push_diagram", {"id": "d1", "kind": "flowchart", "source": "x" * 8001, "title": "r"}),
        ("push_diagram", {"id": "d1", "kind": "flowchart", "source": "g"}),
        ("push_diagram", {"id": "d1", "kind": "flowchart", "source": "g", "title": "t" * 81}),
        ("push_app", {"id": "a1", "html": "<p>hi</p>"}),
        ("highlight_source", {"path": "src/pool.py", "start_line": 9, "end_line": 3}),
        ("highlight_source", {"path": "src/pool.py", "start_line": 0, "end_line": 3}),
        ("highlight_source", {"path": "src/pool.py", "start_line": "three", "end_line": 3}),
        (
            "push_diagram",
            {"type": "app.push", "id": "d1", "kind": "flowchart", "source": "g", "title": "r"},
        ),
    ],
    ids=[
        "extra-field",
        "bad-kind",
        "missing-field",
        "oversized-source",
        "missing-diagram-title",
        "oversized-title",
        "missing-app-title",
        "reversed-range",
        "zero-line",
        "non-numeric-line",
        "wrong-discriminator",
    ],
)
async def test_invalid_fields_return_an_error_string(name: str, body: dict[str, object]) -> None:
    connection = FakeConnection()
    channel = _grounded_channel(connection)

    result = await dispatch_visual_tool(name, json.dumps(body), channel)

    assert isinstance(result, str)
    assert result.startswith(f"{name}: error:")
    assert "\n" not in result
    assert "xxxxxxxx" not in result
    assert connection.sent == []


@pytest.mark.parametrize("name", ["search_code", "push_diagrams", ""])
async def test_unknown_tool_name_returns_an_error_string(name: str) -> None:
    connection = FakeConnection()
    arguments = json.dumps({"id": "d1", "kind": "flowchart", "source": "g", "title": "r"})

    result = await dispatch_visual_tool(name, arguments, VisualChannel(connection))

    assert result == f"{name}: error: unknown visual tool"
    assert connection.sent == []


async def test_ungrounded_highlight_returns_an_error_string_and_sends_nothing() -> None:
    connection = FakeConnection()
    arguments = json.dumps({"path": "src/other.py", "start_line": 3, "end_line": 3})

    result = await dispatch_visual_tool(
        "highlight_source", arguments, _grounded_channel(connection)
    )

    assert result.startswith("highlight_source: error: ungrounded highlight")
    assert connection.sent == []

    arguments = json.dumps({"path": "src/pool.py", "start_line": 3, "end_line": 9})
    result = await dispatch_visual_tool("highlight_source", arguments, VisualChannel(connection))

    assert result.startswith("highlight_source: error: ungrounded highlight")
    assert connection.sent == []


async def test_grounded_highlight_is_sent_through_the_tool() -> None:
    connection = FakeConnection()
    arguments = json.dumps({"path": "src/pool.py", "start_line": 3, "end_line": 9})

    result = await dispatch_visual_tool(
        "highlight_source", arguments, _grounded_channel(connection)
    )

    assert result == "highlight_source: sent"
    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "src/pool.py",
            "start_line": 3,
            "end_line": 9,
            "seq": 1,
        }
    ]


async def test_each_tool_pushes_its_model() -> None:
    connection = FakeConnection()
    channel = _grounded_channel(connection)
    calls = [
        (
            "push_diagram",
            {"id": "d1", "kind": "flowchart", "source": "graph TD; A-->B", "title": "reader"},
        ),
        ("clear_diagram", {}),
        ("highlight_source", {"path": "src/pool.py", "start_line": 3, "end_line": 9}),
        ("push_app", {"id": "a1", "html": "<p>hi</p>", "title": "reader"}),
    ]

    results = [await dispatch_visual_tool(name, json.dumps(body), channel) for name, body in calls]

    assert results == [f"{name}: sent" for name, _ in calls]
    assert [body["type"] for body in connection.sent] == [
        "diagram.push",
        "diagram.clear",
        "source.highlight",
        "app.push",
    ]
    assert [body["seq"] for body in connection.sent] == [1, 2, 3, 4]


async def test_html_with_a_script_tag_still_only_reaches_push() -> None:
    connection = FakeConnection()
    arguments = json.dumps({"id": "a1", "html": _HOSTILE_HTML, "title": "reader"})

    result = await dispatch_visual_tool("push_app", arguments, VisualChannel(connection))

    assert result == "push_app: sent"
    assert "<script" not in result
    assert connection.sent == [
        {"type": "app.push", "id": "a1", "html": _HOSTILE_HTML, "title": "reader", "seq": 1}
    ]


async def test_cancelled_push_propagates_through_dispatch() -> None:
    connection = FakeConnection()
    connection.raises = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await dispatch_visual_tool("clear_diagram", "{}", VisualChannel(connection))
    assert connection.sent == []
