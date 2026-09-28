import asyncio
import json
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from tests.fakes import FakeConnection, grounded_registry, local_peer, search_result
from tests.test_transport import FakeChannel
from tutor.transport import Connection
from tutor.visuals import (
    CLIENT_MESSAGE,
    AckReason,
    AppPush,
    Caption,
    ChannelPayload,
    ClientMessage,
    DiagramClear,
    DiagramPush,
    LearnerText,
    LessonAck,
    LessonAttach,
    LessonCheckpoint,
    LessonCue,
    LessonStatePush,
    LessonSync,
    LessonSynced,
    SayMessage,
    SceneCue,
    ScenePush,
    SceneReady,
    SceneShow,
    SceneStatus,
    SceneStep,
    SourceHighlight,
    StepCue,
    ThemeMessage,
    TurnState,
    UngroundedVisual,
    VisualChannel,
    VisualPayload,
    VisualPending,
)

_adapter = TypeAdapter(VisualPayload)
_channel_adapter = TypeAdapter(ChannelPayload)
_client_adapter = TypeAdapter(ClientMessage)


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
    {"type": "caption", "turn_id": "turn-1", "text": "PPO clips.", "lead_ms": 0},
    {"type": "transcript", "turn_id": "turn-1", "text": "teach me ppo"},
    {"type": "visual.pending", "turn_id": "turn-1", "title": "Clipped objective"},
    {
        "type": "scene.push",
        "scene_id": "turn-4",
        "title": "Clipped objective",
        "html": "<!doctype html><p>x</p>",
        "steps": ["The ratio axis", "The clip band", "The flat region"],
    },
    {"type": "scene.show", "scene_id": "turn-4", "at": 1},
    {"type": "scene.step", "scene_id": "turn-4", "n": 2, "lead_ms": 1200},
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
        (Caption(turn_id="turn-1", text="PPO clips.", lead_ms=1200), "caption"),
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
    assert len(Caption(turn_id="turn-1", text="x" * 2000, lead_ms=0).text) == 2000
    assert len(LearnerText(turn_id="turn-1", text="x" * 4000).text) == 4000
    assert len(Caption(turn_id="t" * 32, text="x", lead_ms=0).turn_id) == 32

    with pytest.raises(ValidationError):
        Caption(turn_id="turn-1", text="x" * 2001, lead_ms=0)
    with pytest.raises(ValidationError):
        LearnerText(turn_id="turn-1", text="x" * 4001)
    with pytest.raises(ValidationError):
        Caption(turn_id="t" * 33, text="x", lead_ms=0)
    with pytest.raises(ValidationError):
        LearnerText(turn_id="t" * 33, text="x")


async def test_state_pushes_carry_seq_in_order() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)

    await channel.push(TurnState(state="thinking", phase="teach"))
    await channel.push(Caption(turn_id="turn-1", text="PPO clips.", lead_ms=0))
    await channel.push(
        DiagramPush(id="d1", kind="flowchart", source="graph TD; A-->B", title="reader")
    )

    assert connection.sent == [
        {"type": "state", "state": "thinking", "phase": "teach", "interrupted": False, "seq": 1},
        {"type": "caption", "turn_id": "turn-1", "text": "PPO clips.", "lead_ms": 0, "seq": 2},
        {
            "type": "diagram.push",
            "id": "d1",
            "kind": "flowchart",
            "source": "graph TD; A-->B",
            "title": "reader",
            "seq": 3,
        },
    ]


def test_a_caption_requires_a_non_negative_lead() -> None:
    assert Caption(turn_id="turn-1", text="x", lead_ms=0).lead_ms == 0
    assert Caption(turn_id="turn-1", text="x", lead_ms=4500).lead_ms == 4500

    with pytest.raises(ValidationError):
        Caption(turn_id="turn-1", text="x")
    with pytest.raises(ValidationError):
        Caption(turn_id="turn-1", text="x", lead_ms=-1)
    with pytest.raises(ValidationError):
        _channel_adapter.validate_python(
            {"type": "caption", "turn_id": "turn-1", "text": "x", "lead_ms": 1.5}
        )


def test_a_state_is_not_interrupted_unless_said_so() -> None:
    assert TurnState(state="listening", phase="teach").interrupted is False

    dumped = TurnState(state="listening", phase="teach", interrupted=True).model_dump(mode="json")

    assert dumped == {"type": "state", "state": "listening", "phase": "teach", "interrupted": True}
    assert _channel_adapter.validate_python(dumped).interrupted is True
    with pytest.raises(ValidationError):
        _channel_adapter.validate_python(
            {"type": "state", "state": "listening", "phase": "teach", "interrupted": "maybe"}
        )


def test_a_pending_visual_round_trips_and_allows_an_empty_title() -> None:
    for title in ("Clipped objective", ""):
        payload = VisualPending(turn_id="turn-1", title=title)
        dumped = payload.model_dump(mode="json")
        assert dumped == {"type": "visual.pending", "turn_id": "turn-1", "title": title}
        assert _channel_adapter.validate_python(dumped) == payload
    assert len(VisualPending(turn_id="turn-1", title="t" * 80).title) == 80

    with pytest.raises(ValidationError):
        VisualPending(turn_id="turn-1", title="t" * 81)
    with pytest.raises(ValidationError):
        VisualPending(turn_id="t" * 33, title="t")
    with pytest.raises(ValidationError):
        _channel_adapter.validate_python(
            {"type": "visual.pending", "turn_id": "turn-1", "title": "t", "x": 1}
        )


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


def test_client_messages_validate_by_type() -> None:
    say = _client_adapter.validate_python({"type": "say", "text": "why clip"})
    theme = _client_adapter.validate_python({"type": "theme", "theme": "dark"})

    assert isinstance(say, SayMessage)
    assert say.text == "why clip"
    assert isinstance(theme, ThemeMessage)
    assert theme.theme == "dark"


def test_a_say_is_stripped_and_a_blank_one_is_rejected() -> None:
    say = _client_adapter.validate_python({"type": "say", "text": "  why clip  "})

    assert say.text == "why clip"
    with pytest.raises(ValidationError):
        _client_adapter.validate_python({"type": "say", "text": "   "})


def test_a_say_over_the_cap_is_rejected() -> None:
    assert len(_client_adapter.validate_python({"type": "say", "text": "x" * 4000}).text) == 4000
    with pytest.raises(ValidationError):
        _client_adapter.validate_python({"type": "say", "text": "x" * 4001})


def test_an_unknown_client_message_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _client_adapter.validate_python({"type": "shout", "text": "x"})
    with pytest.raises(ValidationError):
        _client_adapter.validate_python({"text": "x"})


def test_extra_keys_on_a_client_message_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _client_adapter.validate_python({"type": "say", "text": "x", "turn_id": "turn-1"})
    with pytest.raises(ValidationError):
        _client_adapter.validate_python({"type": "theme", "theme": "dark", "text": "x"})


def test_scene_payloads_round_trip_through_json() -> None:
    push = ScenePush(
        scene_id="turn-4", title="Clipped objective", html="<p>x</p>", steps=["a", "b", "c"]
    )
    show = SceneShow(scene_id="turn-4", at=1)
    step = SceneStep(scene_id="turn-4", n=2, lead_ms=1200)
    for payload in (push, show, step):
        dumped = payload.model_dump(mode="json")
        json.dumps(dumped)
        assert _channel_adapter.validate_python(dumped) == payload
    assert push.model_dump(mode="json")["type"] == "scene.push"
    assert show.model_dump(mode="json") == {"type": "scene.show", "scene_id": "turn-4", "at": 1}
    assert step.model_dump(mode="json") == {
        "type": "scene.step",
        "scene_id": "turn-4",
        "n": 2,
        "lead_ms": 1200,
    }


@pytest.mark.parametrize(
    "scene_id", ["", "Turn-4", "4turn", "a" * 33, "turn 4", "turn_4", "t<b>"], ids=repr
)
def test_a_scene_id_is_a_short_lowercase_token(scene_id: str) -> None:
    with pytest.raises(ValidationError):
        SceneShow(scene_id=scene_id, at=1)
    assert SceneShow(scene_id="a" * 32, at=1).scene_id == "a" * 32
    assert SceneShow(scene_id="turn-4", at=1).scene_id == "turn-4"


def test_a_scene_push_caps_its_document_and_its_steps() -> None:
    assert len(ScenePush(scene_id="s", title="t", html="x" * 200000, steps=["a"]).html) == 200000
    assert len(ScenePush(scene_id="s", title="t", html="x", steps=["a"] * 8).steps) == 8
    assert len(ScenePush(scene_id="s", title="t", html="x", steps=["a" * 120]).steps[0]) == 120
    for html, steps in (
        ("x" * 200001, ["a"]),
        ("", ["a"]),
        ("x", []),
        ("x", ["a"] * 9),
        ("x", ["a" * 121]),
        ("x", [""]),
        ("x", ["a", 1]),
    ):
        with pytest.raises(ValidationError):
            ScenePush(scene_id="s", title="t", html=html, steps=steps)
    with pytest.raises(ValidationError):
        ScenePush(scene_id="s", title="t" * 81, html="x", steps=["a"])
    with pytest.raises(ValidationError):
        _channel_adapter.validate_python(
            {
                "type": "scene.push",
                "scene_id": "s",
                "title": "t",
                "html": "x",
                "steps": ["a"],
                "x": 1,
            }
        )


def test_show_and_step_count_from_one_and_the_lead_is_non_negative() -> None:
    assert SceneShow(scene_id="s", at=1).at == 1
    assert SceneStep(scene_id="s", n=1, lead_ms=0).lead_ms == 0
    for at in (0, -1):
        with pytest.raises(ValidationError):
            SceneShow(scene_id="s", at=at)
    for n, lead in ((0, 0), (1, -1)):
        with pytest.raises(ValidationError):
            SceneStep(scene_id="s", n=n, lead_ms=lead)
    with pytest.raises(ValidationError):
        _channel_adapter.validate_python({"type": "scene.step", "scene_id": "s", "n": 1})


def test_a_scene_ready_message_validates_as_a_client_message() -> None:
    ready = _client_adapter.validate_python(
        {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": 3, "error": ""}
    )

    assert isinstance(ready, SceneReady)
    assert ready.ok is True and ready.steps == 3 and ready.error == ""
    failed = _client_adapter.validate_python(
        {"type": "scene.ready", "scene_id": "turn-4", "ok": False, "steps": 0, "error": "e" * 500}
    )
    assert failed.ok is False and len(failed.error) == 500
    for body in (
        {"type": "scene.ready", "scene_id": "turn-4", "ok": "maybe", "steps": 3, "error": ""},
        {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": 9, "error": ""},
        {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": -1, "error": ""},
        {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": 3, "error": "e" * 501},
        {"type": "scene.ready", "scene_id": "Turn", "ok": True, "steps": 3, "error": ""},
        {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": 3},
        {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": 3, "error": "", "x": 1},
    ):
        with pytest.raises(ValidationError):
            _client_adapter.validate_python(body)


LESSON_STATE = {
    "type": "lesson.state",
    "scenes": [{"id": "scene-1", "title": "Clipped objective", "status": "building"}],
    "current": "scene-1",
}
LESSON_CUE = {
    "type": "lesson.cue",
    "epoch": 1,
    "barrier": 0,
    "cue_id": 3,
    "chunk_id": 7,
    "scene_id": "scene-1",
    "revision": 2,
    "lead_ms": 480,
    "audio_ms": 1650,
    "tag": {"kind": "step", "n": 2},
}
LESSON_ACK = {
    "type": "lesson.ack",
    "epoch": 1,
    "barrier": 0,
    "cue_id": 3,
    "outcome": "fired",
    "reason": None,
    "scene_id": "scene-1",
    "step": 2,
    "revision": 3,
}
LESSON_SYNCED = {
    "type": "lesson.synced",
    "epoch": 1,
    "barrier": 1,
    "scene_id": "scene-1",
    "step": 2,
    "revision": 3,
    "last_cue": 3,
}
LESSON_CHECKPOINT = {
    "type": "lesson.checkpoint",
    "epoch": 1,
    "scene_id": "scene-1",
    "version": 1,
    "step": 2,
    "revision": 3,
}


def without(body: dict[str, object], key: str) -> dict[str, object]:
    return {k: v for k, v in body.items() if k != key}


def test_a_scene_status_is_one_of_five_words() -> None:
    row = SceneStatus(id="scene-1", title="Clipped objective", status="building")
    assert row.model_dump() == {"id": "scene-1", "title": "Clipped objective", "status": "building"}
    for body in (
        {"id": "scene-1", "title": "t", "status": "drawing"},
        {"id": "Scene-1", "title": "t", "status": "planned"},
        {"id": "scene-1", "title": "t" * 81, "status": "planned"},
        {"id": "scene-1", "title": "t", "status": "planned", "at": 1},
    ):
        with pytest.raises(ValueError):
            SceneStatus.model_validate(body)


@pytest.mark.parametrize(
    "model, body",
    [
        (LessonStatePush, LESSON_STATE),
        (LessonStatePush, {**LESSON_STATE, "scenes": [], "current": None}),
        (LessonAttach, {"type": "lesson.attach", "epoch": 1}),
        (LessonCue, LESSON_CUE),
        (LessonCue, {**LESSON_CUE, "chunk_id": 0}),
        (
            LessonCue,
            {
                **LESSON_CUE,
                "scene_id": None,
                "tag": {"kind": "scene", "n": 1, "scene_id": "scene-1"},
            },
        ),
        (LessonSync, {"type": "lesson.sync", "epoch": 1, "barrier": 1}),
        (LessonAck, LESSON_ACK),
        (LessonAck, {**LESSON_ACK, "outcome": "dropped", "reason": "barrier"}),
        (
            LessonAck,
            {**LESSON_ACK, "outcome": "failed", "reason": "runtime", "scene_id": None, "step": 0},
        ),
        (LessonSynced, LESSON_SYNCED),
        (LessonCheckpoint, LESSON_CHECKPOINT),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "scene_id": None, "version": 0, "step": 0}),
    ],
)
def test_the_lesson_models_round_trip_through_json(model: type, body: dict[str, object]) -> None:
    assert json.loads(model.model_validate(body).model_dump_json()) == body


def test_every_ack_reason_is_accepted_on_a_dropped_ack() -> None:
    for reason in get_args(AckReason):
        assert LessonAck.model_validate({**LESSON_ACK, "outcome": "dropped", "reason": reason})


@pytest.mark.parametrize(
    "model, body",
    [
        (LessonStatePush, {**LESSON_STATE, "scenes": LESSON_STATE["scenes"] * 13}),
        (LessonStatePush, {**LESSON_STATE, "current": "Scene-1"}),
        (LessonStatePush, {**LESSON_STATE, "phase": "teach"}),
        (LessonStatePush, without(LESSON_STATE, "current")),
        (LessonAttach, {"type": "lesson.attach", "epoch": 0}),
        (LessonAttach, {"type": "lesson.attach", "epoch": 1, "checkpoint": None}),
        (LessonAttach, {"type": "lesson.attach"}),
        (LessonCue, {**LESSON_CUE, "type": "lesson.position"}),
        (LessonCue, {**LESSON_CUE, "epoch": 0}),
        (LessonCue, {**LESSON_CUE, "barrier": -1}),
        (LessonCue, {**LESSON_CUE, "cue_id": 0}),
        (LessonCue, {**LESSON_CUE, "chunk_id": -1}),
        (LessonCue, {**LESSON_CUE, "scene_id": "Scene-1"}),
        (LessonCue, {**LESSON_CUE, "revision": -1}),
        (LessonCue, {**LESSON_CUE, "lead_ms": -1}),
        (LessonCue, {**LESSON_CUE, "audio_ms": -1}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "step", "n": 0}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "step", "n": 6}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "step", "n": 2, "at": 1}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "scene", "n": 0, "scene_id": "scene-1"}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "scene", "n": 13, "scene_id": "scene-1"}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "scene", "n": 1, "scene_id": "Bad"}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "scene", "n": 1}}),
        (LessonCue, {**LESSON_CUE, "tag": {"kind": "set", "n": 1}}),
        (LessonCue, {**LESSON_CUE, "html": "<p>x</p>"}),
        (LessonCue, without(LESSON_CUE, "chunk_id")),
        (LessonCue, without(LESSON_CUE, "scene_id")),
        (LessonSync, {"type": "lesson.sync", "epoch": 0, "barrier": 1}),
        (LessonSync, {"type": "lesson.sync", "epoch": 1, "barrier": 0}),
        (LessonSync, {"type": "lesson.sync", "epoch": 1, "barrier": 1, "cue_id": 1}),
        (LessonAck, {**LESSON_ACK, "type": "lesson.applied"}),
        (LessonAck, {**LESSON_ACK, "epoch": 0}),
        (LessonAck, {**LESSON_ACK, "barrier": -1}),
        (LessonAck, {**LESSON_ACK, "cue_id": 0}),
        (LessonAck, {**LESSON_ACK, "outcome": "applied"}),
        (LessonAck, {**LESSON_ACK, "reason": "barrier"}),
        (LessonAck, {**LESSON_ACK, "outcome": "dropped"}),
        (LessonAck, {**LESSON_ACK, "outcome": "failed", "reason": "late"}),
        (LessonAck, {**LESSON_ACK, "scene_id": "Scene-1"}),
        (LessonAck, {**LESSON_ACK, "step": -1}),
        (LessonAck, {**LESSON_ACK, "revision": -1}),
        (LessonAck, {**LESSON_ACK, "text": "x"}),
        (LessonAck, without(LESSON_ACK, "reason")),
        (LessonSynced, {**LESSON_SYNCED, "epoch": 0}),
        (LessonSynced, {**LESSON_SYNCED, "barrier": 0}),
        (LessonSynced, {**LESSON_SYNCED, "scene_id": "Scene-1"}),
        (LessonSynced, {**LESSON_SYNCED, "step": -1}),
        (LessonSynced, {**LESSON_SYNCED, "revision": -1}),
        (LessonSynced, {**LESSON_SYNCED, "last_cue": -1}),
        (LessonSynced, {**LESSON_SYNCED, "cue_id": 3}),
        (LessonSynced, without(LESSON_SYNCED, "last_cue")),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "epoch": 0}),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "scene_id": "Scene-1"}),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "version": -1}),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "step": -1}),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "revision": -1}),
        (LessonCheckpoint, {**LESSON_CHECKPOINT, "dials": {}}),
        (LessonCheckpoint, without(LESSON_CHECKPOINT, "version")),
    ],
)
def test_a_lesson_model_rejects_each_bad_field(model: type, body: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        model.model_validate(body)


OPENING_CUE = LessonCue(
    epoch=1,
    barrier=0,
    cue_id=1,
    chunk_id=1,
    scene_id=None,
    revision=0,
    lead_ms=0,
    audio_ms=900,
    tag=SceneCue(n=1, scene_id="ratio"),
)
STEP_CUE = LessonCue(
    epoch=1,
    barrier=0,
    cue_id=2,
    chunk_id=3,
    scene_id="ratio",
    revision=1,
    lead_ms=640,
    audio_ms=1800,
    tag=StepCue(n=2),
)
PAGE_ACK = {
    "type": "lesson.ack",
    "epoch": 1,
    "barrier": 0,
    "cue_id": 2,
    "outcome": "fired",
    "reason": None,
    "scene_id": "ratio",
    "step": 2,
    "revision": 2,
}
PAGE_SYNCED = {
    "type": "lesson.synced",
    "epoch": 1,
    "barrier": 1,
    "scene_id": "ratio",
    "step": 2,
    "revision": 2,
    "last_cue": 2,
}
PAGE_CHECKPOINT = {
    "type": "lesson.checkpoint",
    "epoch": 1,
    "scene_id": "ratio",
    "version": 0,
    "step": 2,
    "revision": 2,
}


def test_the_lesson_payloads_travel_on_the_channel() -> None:
    state = LessonStatePush(
        scenes=[SceneStatus(id="ratio", title="The ratio", status="building")], current=None
    )
    for payload in (
        LessonAttach(epoch=1),
        OPENING_CUE,
        STEP_CUE,
        LessonSync(epoch=1, barrier=1),
        state,
    ):
        assert _channel_adapter.validate_python(payload.model_dump(mode="json")) == payload
    assert OPENING_CUE.model_dump(mode="json")["tag"] == {
        "kind": "scene",
        "n": 1,
        "scene_id": "ratio",
    }
    assert STEP_CUE.model_dump(mode="json")["tag"] == {"kind": "step", "n": 2}
    assert OPENING_CUE.model_dump(mode="json")["scene_id"] is None


def test_the_page_messages_are_client_messages_and_never_a_say() -> None:
    assert isinstance(CLIENT_MESSAGE.validate_python(PAGE_ACK), LessonAck)
    assert isinstance(CLIENT_MESSAGE.validate_python(PAGE_SYNCED), LessonSynced)
    assert isinstance(CLIENT_MESSAGE.validate_python(PAGE_CHECKPOINT), LessonCheckpoint)
    for body in (PAGE_ACK, PAGE_SYNCED, PAGE_CHECKPOINT):
        with pytest.raises(ValidationError):
            CLIENT_MESSAGE.validate_python({**body, "text": "go on"})


async def test_a_cue_reaches_the_connection_with_its_tag_nested_and_a_seq() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    await channel.push(LessonAttach(epoch=1))
    await channel.push(STEP_CUE)
    assert connection.sent == [
        {"type": "lesson.attach", "epoch": 1, "seq": 1},
        {**STEP_CUE.model_dump(mode="json"), "seq": 2},
    ]


def test_push_nowait_sends_a_sync_with_the_next_seq_only_when_it_goes_out() -> None:
    connection = FakeConnection()
    channel = VisualChannel(connection)
    assert channel.push_nowait(LessonSync(epoch=1, barrier=1)) is True
    connection.open = False
    assert channel.push_nowait(LessonSync(epoch=1, barrier=2)) is False
    connection.open = True
    assert channel.push_nowait(LessonSync(epoch=1, barrier=3)) is True
    assert [body["seq"] for body in connection.sent] == [1, 2]
    assert [body["barrier"] for body in connection.sent] == [1, 3]


def wire_order(wire: FakeChannel) -> list[tuple[str, int]]:
    return [(body["type"], body["seq"]) for body in map(json.loads, wire.sent)]


async def reach_the_channel_wait() -> None:
    waited = asyncio.Event()
    asyncio.get_running_loop().call_soon(waited.set)
    await waited.wait()


async def test_a_sync_never_overtakes_a_push_still_waiting_on_the_data_channel() -> None:
    pc = local_peer()
    connection = Connection(pc)
    channel = VisualChannel(connection)
    wire = FakeChannel()
    attaching = asyncio.create_task(channel.push(LessonAttach(epoch=1)))
    await reach_the_channel_wait()

    pc.emit("datachannel", wire)
    delivered = channel.push_nowait(LessonSync(epoch=1, barrier=1))
    await attaching

    assert wire_order(wire) == [("lesson.attach", 1)]
    assert delivered is False
    await channel.push(LessonSync(epoch=1, barrier=1))
    assert wire_order(wire) == [("lesson.attach", 1), ("lesson.sync", 2)]
    await connection.close()


async def test_a_push_cancelled_on_the_channel_wait_releases_its_hold_on_the_sync() -> None:
    pc = local_peer()
    connection = Connection(pc)
    channel = VisualChannel(connection)
    wire = FakeChannel()
    waiting = asyncio.create_task(channel.push(LessonAttach(epoch=1)))
    await reach_the_channel_wait()

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    pc.emit("datachannel", wire)

    assert channel.push_nowait(LessonSync(epoch=1, barrier=1)) is True
    assert wire_order(wire) == [("lesson.sync", 2)]
    await connection.close()
