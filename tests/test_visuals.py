import asyncio
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.fakes import FakeConnection, grounded_registry, search_result
from tutor.visuals import (
    AppPush,
    Caption,
    ChannelPayload,
    DiagramClear,
    DiagramPush,
    LearnerText,
    SourceHighlight,
    TurnState,
    UngroundedVisual,
    VisualChannel,
    VisualPayload,
)

_adapter = TypeAdapter(VisualPayload)
_channel_adapter = TypeAdapter(ChannelPayload)


_VALID_SHAPES = [
    {
        "type": "diagram.push",
        "id": "d1",
        "kind": "flowchart",
        "source": "graph TD; A-->B",
        "title": "reader",
    },
    {"type": "diagram.clear"},
    {"type": "source.highlight", "path": "src/pool.py", "start_line": 3, "end_line": 9},
    {"type": "app.push", "id": "a1", "html": "<p>hi</p>", "title": "reader"},
    {"type": "state", "state": "thinking", "phase": "teach"},
    {"type": "caption", "turn_id": "turn-1", "text": "PPO clips."},
    {"type": "transcript", "turn_id": "turn-1", "text": "teach me ppo"},
]


def test_diagram_push_round_trips_through_json() -> None:
    push = DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader")

    dumped = push.model_dump(mode="json")

    assert dumped == {
        "type": "diagram.push",
        "id": "d1",
        "kind": "flowchart",
        "source": "graph TD; A-->B",
        "title": "reader",
    }
    restored = _adapter.validate_python(dumped)
    assert isinstance(restored, DiagramPush)
    assert restored == push


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _adapter.validate_python(
            {"type": "diagram.explode", "id": "d1", "kind": "flowchart", "source": "graph TD"}
        )

    assert excinfo.value.errors()[0]["type"] == "union_tag_invalid"


@pytest.mark.parametrize("shape", _VALID_SHAPES, ids=[s["type"] for s in _VALID_SHAPES])
def test_extra_fields_are_rejected(shape: dict[str, object]) -> None:
    assert _channel_adapter.validate_python(shape).type == shape["type"]

    with pytest.raises(ValidationError):
        _channel_adapter.validate_python({**shape, "extra": 1})


def test_a_push_without_a_title_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DiagramPush(id="x", kind="flowchart", source="graph TD; a")
    with pytest.raises(ValidationError):
        AppPush(id="x", html="<p>hi</p>")


def test_a_title_over_eighty_characters_is_rejected() -> None:
    assert len(DiagramPush(id="x", kind="flowchart", source="g", title="t" * 80).title) == 80
    assert len(AppPush(id="x", html="<p>hi</p>", title="t" * 80).title) == 80

    with pytest.raises(ValidationError):
        DiagramPush(id="x", kind="flowchart", source="g", title="t" * 81)
    with pytest.raises(ValidationError):
        AppPush(id="x", html="<p>hi</p>", title="t" * 81)


@pytest.mark.parametrize(
    ("payload", "tag"),
    [
        (TurnState(state="thinking", phase="teach"), "state"),
        (Caption(turn_id="turn-1", text="PPO clips."), "caption"),
        (LearnerText(turn_id="turn-1", text="teach me ppo"), "transcript"),
    ],
    ids=["state", "caption", "transcript"],
)
def test_state_caption_and_transcript_round_trip_through_json(
    payload: TurnState | Caption | LearnerText, tag: str
) -> None:
    dumped = payload.model_dump(mode="json")

    json.dumps(dumped)
    assert dumped["type"] == tag
    restored = _channel_adapter.validate_python(dumped)
    assert type(restored) is type(payload)
    assert restored == payload


def test_an_unknown_state_or_phase_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TurnState(state="idle", phase="teach")
    with pytest.raises(ValidationError):
        TurnState(state="thinking", phase="review")
    with pytest.raises(ValidationError):
        _channel_adapter.validate_python(
            {"type": "state", "state": "thinking", "phase": "teach", "x": 1}
        )


def test_caption_and_transcript_are_capped() -> None:
    assert len(Caption(turn_id="turn-1", text="x" * 2000).text) == 2000
    assert len(LearnerText(turn_id="turn-1", text="x" * 4000).text) == 4000
    assert len(Caption(turn_id="t" * 32, text="x").turn_id) == 32

    with pytest.raises(ValidationError):
        Caption(turn_id="turn-1", text="x" * 2001)
    with pytest.raises(ValidationError):
        LearnerText(turn_id="turn-1", text="x" * 4001)
    with pytest.raises(ValidationError):
        Caption(turn_id="t" * 33, text="x")
    with pytest.raises(ValidationError):
        LearnerText(turn_id="t" * 33, text="x")


async def test_state_pushes_carry_seq_in_order() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)

    await channel.push(TurnState(state="thinking", phase="teach"))
    await channel.push(Caption(turn_id="turn-1", text="PPO clips."))
    await channel.push(
        DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader")
    )

    assert connection.sent == [
        {"type": "state", "state": "thinking", "phase": "teach", "seq": 1},
        {"type": "caption", "turn_id": "turn-1", "text": "PPO clips.", "seq": 2},
        {
            "type": "diagram.push",
            "id": "d1",
            "kind": "flowchart",
            "source": "graph TD; A-->B",
            "title": "reader",
            "seq": 3,
        },
    ]


def test_diagram_kind_outside_the_literal_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DiagramPush(id="d1", kind="gantt", source="gantt", title="reader")


@pytest.mark.parametrize(("start_line", "end_line"), [(10, 3), (0, 3), (1, 0)])
def test_reversed_line_range_is_rejected(start_line: int, end_line: int) -> None:
    assert SourceHighlight(path="a.py", start_line=3, end_line=3).end_line == 3

    with pytest.raises(ValidationError):
        SourceHighlight(path="a.py", start_line=start_line, end_line=end_line)


def test_oversized_source_is_rejected() -> None:
    assert (
        len(DiagramPush(id="d1", kind="sequence", source="x" * 8000, title="reader").source) == 8000
    )
    assert len(AppPush(id="a1", html="x" * 64000, title="reader").html) == 64000
    assert len(DiagramPush(id="i" * 64, kind="sequence", source="x", title="reader").id) == 64
    assert len(SourceHighlight(path="p" * 4096, start_line=1, end_line=1).path) == 4096

    with pytest.raises(ValidationError):
        DiagramPush(id="d1", kind="sequence", source="x" * 8001, title="reader")
    with pytest.raises(ValidationError):
        AppPush(id="a1", html="x" * 64001, title="reader")
    with pytest.raises(ValidationError):
        DiagramPush(id="i" * 65, kind="sequence", source="x", title="reader")
    with pytest.raises(ValidationError):
        SourceHighlight(path="p" * 4097, start_line=1, end_line=1)


async def test_push_sends_the_serialized_payload() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)

    await channel.push(
        DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader")
    )

    assert connection.sent == [
        {
            "type": "diagram.push",
            "id": "d1",
            "kind": "flowchart",
            "source": "graph TD; A-->B",
            "title": "reader",
            "seq": 1,
        }
    ]


async def test_seq_increases_by_one_per_push() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)

    for _ in range(3):
        await channel.push(DiagramClear())

    assert [body["seq"] for body in connection.sent] == [1, 2, 3]


async def test_push_order_is_preserved_on_the_wire() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    channel.set_grounding(grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9]), "t1")
    payloads: list[VisualPayload] = [
        DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader"),
        SourceHighlight(path="src/pool.py", start_line=3, end_line=9),
        DiagramClear(),
        AppPush(id="a1", html="<p>hi</p>", title="reader"),
    ]

    for payload in payloads:
        await channel.push(payload)

    assert connection.sent == [
        {**payload.model_dump(mode="json"), "seq": seq}
        for seq, payload in enumerate(payloads, start=1)
    ]


async def test_payload_json_is_plain_types() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    channel.set_grounding(grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9]), "t1")

    await channel.push(
        DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader")
    )
    await channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=9))

    for body in connection.sent:
        json.dumps(body)
        assert isinstance(body["seq"], int)
        assert isinstance(body["type"], str)


async def test_cancelling_a_push_reraises() -> None:
    connection = FakeConnection()
    connection.raises = asyncio.CancelledError()
    channel = VisualChannel(connection)

    with pytest.raises(asyncio.CancelledError):
        await channel.push(DiagramClear())


async def test_ungrounded_highlight_raises() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9])
    channel.set_grounding(registry, "t1")

    with pytest.raises(UngroundedVisual):
        await channel.push(SourceHighlight(path="src/other.py", start_line=3, end_line=3))
    assert connection.sent == []

    fresh_channel = VisualChannel(connection)
    with pytest.raises(UngroundedVisual):
        await fresh_channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=9))
    assert connection.sent == []


async def test_grounded_highlight_is_sent() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9])
    channel.set_grounding(registry, "t1")

    await channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=9))

    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "src/pool.py",
            "start_line": 3,
            "end_line": 9,
            "seq": 1,
        }
    ]

    await channel.push(SourceHighlight(path="./src/pool.py", start_line=3, end_line=9))

    assert connection.sent[1] == {
        "type": "source.highlight",
        "path": "./src/pool.py",
        "start_line": 3,
        "end_line": 9,
        "seq": 2,
    }


async def test_grounding_is_cleared_between_turns() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9])
    channel.set_grounding(registry, "t1")

    registry.open_turn("t2")
    channel.set_grounding(registry, "t2")

    with pytest.raises(UngroundedVisual):
        await channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=9))
    assert connection.sent == []

    other_channel = VisualChannel(connection)
    other_channel.set_grounding(registry, "t1")
    registry.open_turn("t2")
    registry.record("t2", search_result("src/pool.py", [3, 4, 5, 6, 7, 8, 9]))

    with pytest.raises(UngroundedVisual):
        await other_channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=9))
    assert connection.sent == []


async def test_every_highlighted_line_must_be_known_to_the_turn() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3, 4, 6])
    channel.set_grounding(registry, "t1")

    await channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=4))
    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "src/pool.py",
            "start_line": 3,
            "end_line": 4,
            "seq": 1,
        }
    ]

    with pytest.raises(UngroundedVisual) as excinfo:
        await channel.push(SourceHighlight(path="src/pool.py", start_line=3, end_line=6))
    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "src/pool.py",
            "start_line": 3,
            "end_line": 4,
            "seq": 1,
        }
    ]
    assert str(excinfo.value) == "ungrounded highlight 'src/pool.py':5"


async def test_basename_alias_grounds_a_highlight() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3])
    channel.set_grounding(registry, "t1")

    await channel.push(SourceHighlight(path="pool.py", start_line=3, end_line=3))

    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "pool.py",
            "start_line": 3,
            "end_line": 3,
            "seq": 1,
        }
    ]


async def test_diagram_push_ignores_the_grounding_set() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)

    await channel.push(
        DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader")
    )
    await channel.push(DiagramClear())
    await channel.push(AppPush(id="a1", html="<p>hi</p>", title="reader"))

    assert [body["seq"] for body in connection.sent] == [1, 2, 3]


@pytest.mark.parametrize(
    "path",
    [
        "",
        "..",
        "../src/pool.py",
        "/src/pool.py",
        "src/pool.py\n",
        "src/pool.py\x00",
        "././src/pool.py",
        "SRC/pool.py",
    ],
    ids=[
        "empty",
        "dotdot",
        "relative-escape",
        "absolute",
        "trailing-newline",
        "trailing-nul",
        "doubled-dot-slash",
        "uppercase-dir",
    ],
)
async def test_hostile_paths_never_reach_the_wire(path: str) -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3, 4, 5, 6, 7, 8, 9])
    channel.set_grounding(registry, "t1")

    with pytest.raises(UngroundedVisual):
        await channel.push(SourceHighlight(path=path, start_line=3, end_line=3))
    assert connection.sent == []


async def test_laxly_coerced_lines_are_gated_as_ints() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    registry = grounded_registry("t1", "src/pool.py", [3])
    channel.set_grounding(registry, "t1")

    await channel.push(SourceHighlight(path="src/pool.py", start_line="3", end_line=3.0))

    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "src/pool.py",
            "start_line": 3,
            "end_line": 3,
            "seq": 1,
        }
    ]

    with pytest.raises(UngroundedVisual):
        await channel.push(SourceHighlight(path="src/pool.py", start_line=True, end_line="3"))
    assert connection.sent == [
        {
            "type": "source.highlight",
            "path": "src/pool.py",
            "start_line": 3,
            "end_line": 3,
            "seq": 1,
        }
    ]
