import asyncio
import contextlib
import json
import logging
import re
import typing
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from tests.fakes import local_peer, numbered_frames
from tests.test_transport import FakeChannel
from tutor.chunker import split_clauses
from tutor.constants import FRAME_SAMPLES, SAMPLE_RATE
from tutor.endpointer import SILENCE_WINDOW_MS
from tutor.input_path import (
    START_FRAMES,
    EndOfTurn,
    InputEvent,
    InputPath,
    PartialTranscript,
    SpeechStarted,
)
from tutor.lead_in import lead_in_sentence, lead_in_stages
from tutor.lesson import (
    NO_PLAN,
    OPENING_TEXT,
    Cursor,
    LessonPlan,
    LessonState,
    Scene,
    ScriptChunk,
    Step,
    lesson_block,
)
from tutor.planner import PLAN_TOOL
from tutor.prompt import SEARCH_CODE_TOOL, Message, TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.reply import LIVE_PROMPT, NOT_READY
from tutor.scene import SCENE_TOOL, planned_scene_prompt
from tutor.script import SCRIPT_TOOL
from tutor.session import (
    BAD_ARGUMENTS,
    LESSON_SYNC_TIMEOUT_S,
    SPOKEN_DEPTH,
    TurnLoop,
    TurnLoopConfig,
)
from tutor.speech import Chunk, OnPlay
from tutor.tools.models import (
    GroundingVerdict,
    Position,
    SearchBudget,
    SearchMatch,
    SearchResult,
)
from tutor.tools.provenance import TurnRegistry
from tutor.transport import INBOUND_CAPACITY, Connection
from tutor.visual_tools import VOICE_VISUAL_TOOLS
from tutor.visuals import LessonCheckpoint

HANG_GUARD_S = 20.0
TICK_S = 0.25
SYSTEM = "you tutor an engineer through a codebase"
SUBJECT = "the inbound frame queue"
USER_TEXT = "walk me through tutor/transport.py"
SECOND_TEXT = "and where does tutor/playout.py fit"
FIRST_PARTIAL = "walk me"
PARTIAL_TEXT = "walk me through"
GROUNDED_PATH = "tutor/transport.py"
UNGROUNDED_PATH = "tutor/playout.py"
UNPARSEABLE_PATH = "tutor/my transport.py"
SPOKEN_DELTAS = ["It lives in ", "the reader. ", "It drains ", "the track."]
SPOKEN_CLAUSES = ["It lives in the reader.", "It drains the track."]
FOLLOW_DELTAS = [" Both call sites ", "drain it."]
PATH_DELTAS = [
    f"It lives in {GROUNDED_PATH}. ",
    f"And the other copy of the same reader sits in {UNGROUNDED_PATH}. ",
    "Both call sites drain it.",
]
PATH_CLAUSES = [
    f"It lives in {GROUNDED_PATH}.",
    f"And the other copy of the same reader sits in {UNGROUNDED_PATH}.",
    "Both call sites drain it.",
]
LINE_DELTAS = [f"It lives in {GROUNDED_PATH} line 24. ", "And the reader drains the track."]
LINE_CLAUSES = [
    f"It lives in {GROUNDED_PATH} line 24.",
    "And the reader drains the track.",
]
COUNT_DELTAS = [f"It lives in {GROUNDED_PATH} line 24. ", "There are 3 callers of it."]
SPLIT_LINE = 4021
SPLIT_DELTAS = [
    "It lives there. The queue reader that drains ",
    (
        "the inbound audio track hands every frame it pulls to the resampler and then to the "
        "voice detector and after both of those steps have run does it reach the code that sits "
        f"on line {SPLIT_LINE} of that same file and it never blocks."
    ),
]
SPLIT_CLAUSES = [
    "It lives there.",
    (
        "The queue reader that drains the inbound audio track hands every frame it pulls to the "
        "resampler and then to the voice detector and after both of those steps have run does it "
        "reach the code that sits on line"
    ),
    f"{SPLIT_LINE} of that same file and it never blocks.",
]
CHAINED_CLAUSES = [
    "It lives in the reader.",
    "It drains the track.",
    "Both call sites drain it.",
]
REASONING_TEXT = "weighing two call sites"
MODEL_QUERY = "class Connection"
MODEL_GLOBS = ["tutor/transport.py"]
SEARCH_ARGUMENTS = json.dumps({"query": MODEL_QUERY, "globs": MODEL_GLOBS})
SEARCH_CALL_ID = "call-search-1"
SEARCH_CALL = TurnChunk(
    kind="tool_call", text=SEARCH_ARGUMENTS, tool_call_id=SEARCH_CALL_ID, tool_name="search_code"
)
TRUNCATED_ARGUMENTS = json.dumps({"query": "def recv", "globs": MODEL_GLOBS})
BAD_ARGUMENT_BODIES = [
    pytest.param(json.dumps({"query": MODEL_QUERY, "globs": []}), id="empty-globs"),
    pytest.param(json.dumps({"query": MODEL_QUERY}), id="missing-globs"),
    pytest.param(json.dumps({"query": MODEL_QUERY, "globs": MODEL_GLOBS[0]}), id="globs-string"),
    pytest.param(SEARCH_ARGUMENTS.removesuffix("]}"), id="truncated-json"),
]
VISUAL_TOOL = "clear_diagram"
VISUAL_ARGUMENTS = json.dumps({"mermaid": "graph TD; reader-->queue"})
VISUAL_CALL_ID = "call-visual-1"
DIAGRAM_ARGUMENTS = json.dumps(
    {"id": "reader", "kind": "flowchart", "source": "graph TD; reader-->queue", "title": "reader"}
)
CLEAR_ARGUMENTS = "{}"
CLEAR_PAYLOAD = {"type": "diagram.clear"}
GROUNDED_HIGHLIGHT = json.dumps({"path": GROUNDED_PATH, "start_line": 24, "end_line": 24})
UNGROUNDED_HIGHLIGHT = json.dumps({"path": UNGROUNDED_PATH, "start_line": 30, "end_line": 30})
RATE_LIMIT_DETAIL = "quota exhausted for project acct-9"
TURN_TASK = "turn-1"
DRAIN_TASK = "turn-1-drain"
STAGER_TASK = "turn-1-stager"
SPECULATION_TASK = "turn-1-speculation"
SECOND_PATH = "tutor/resample.py"
CARRY_DELTAS = [
    "Line 30 is the reader. ",
    f"It lives in the queue reader inside {GROUNDED_PATH}. ",
    "Line 24 is where it drains.",
]
CARRY_CLAUSES = [
    f"It lives in the queue reader inside {GROUNDED_PATH}.",
    "Line 24 is where it drains.",
]
FIRST_CLAUSE_DELTA = "It lives in the reader. "
FILLER_DELTA = "and "
# One delta reaches the chunker, SPOKEN_DEPTH sit in the queue, and the next put blocks.
QUEUE_FULL_DELTAS = SPOKEN_DEPTH + 2
CONCEPT_TEXT = "teach me ppo"
CONCEPT_PARTIAL = "teach me"
STARTING_FROM = "I know policy gradients"
CONCEPT_DELTAS = ["We run 10 epochs ", "on 2048 samples. ", "Look in setup.py ", "for the flags."]
CONCEPT_CLAUSES = ["We run 10 epochs on 2048 samples.", "Look in setup.py for the flags."]
SEARCH_SPAN = re.compile(r"^turn\.search turn_id=turn-1 ms=250$")
SPOKEN_SPAN = re.compile(r"^turn\.spoken turn_id=\S+ ms=750$")

SILENCE_WINDOW_FRAMES = -(-SILENCE_WINDOW_MS * SAMPLE_RATE // (1000 * FRAME_SAMPLES))
SPEECH_FRAMES = 4 * START_FRAMES
LEAD_FRAMES = SPEECH_FRAMES + SILENCE_WINDOW_FRAMES  # END_OF_TURN lands on the last of these
BURST_SILENCE_FRAMES = 60
BURST_SPEECH_FRAMES = 4 * START_FRAMES
BURST_FRAMES = BURST_SILENCE_FRAMES + BURST_SPEECH_FRAMES
PACED_PROBABILITIES = (
    [0.9] * SPEECH_FRAMES
    + [0.0] * (SILENCE_WINDOW_FRAMES + BURST_SILENCE_FRAMES)
    + [0.9] * BURST_SPEECH_FRAMES
)


def found(path: str = GROUNDED_PATH) -> SearchResult:
    return SearchResult(
        tool="search",
        query=MODEL_QUERY,
        globs=MODEL_GLOBS,
        matches=[SearchMatch(path=path, line=24, text="        self._inbound = queue")],
        truncated=False,
        oversized=False,
        byte_count=64,
    )


def found_many() -> SearchResult:
    return SearchResult(
        tool="search",
        query=MODEL_QUERY,
        globs=MODEL_GLOBS,
        matches=[
            SearchMatch(path=GROUNDED_PATH, line=24, text="        self._inbound = queue"),
            SearchMatch(path=UNGROUNDED_PATH, line=30, text="        frame = await self.recv()"),
            SearchMatch(path=SECOND_PATH, line=41, text="        return self._state.resample(pcm)"),
        ],
        truncated=False,
        oversized=False,
        byte_count=192,
    )


def config(root: Path) -> TurnLoopConfig:
    return TurnLoopConfig(system=SYSTEM, subject=SUBJECT, root=root, planned=False)


def concept_cfg() -> TurnLoopConfig:
    return TurnLoopConfig(
        system=SYSTEM, subject="PPO", starting_from=STARTING_FROM, root=None, planned=False
    )


class FakeClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        elapsed = self.calls * TICK_S
        self.calls += 1
        return elapsed


class HeldPace:
    def __init__(self, log: list[tuple[str, object]] | None = None) -> None:
        self._log = log
        self.waits: list[float] = []
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, wait: float) -> None:
        self.waits.append(wait)
        if self._log is not None:
            self._log.append(("pace", wait))
        self.held.set()
        await self.release.wait()
        self.release.clear()

    def release_once(self) -> None:
        self.held.clear()
        self.release.set()


class ScriptedSource:
    def __init__(self, script: list[InputEvent | asyncio.Event]) -> None:
        self._script = script
        self.blocked = asyncio.Event()

    async def events(self) -> AsyncIterator[InputEvent]:
        for item in self._script:
            if isinstance(item, asyncio.Event):
                self.blocked.set()
                await item.wait()
            else:
                yield item


class SerialSource:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    async def events(self) -> AsyncIterator[InputEvent]:
        for n, text in enumerate(self._texts, start=1):
            yield EndOfTurn(text=text)
            turns = [task for task in asyncio.all_tasks() if task.get_name() == f"turn-{n}"]
            await asyncio.wait(turns)


class RecordingSource:
    def __init__(self, path: InputPath) -> None:
        self._path = path
        self.seen: list[InputEvent] = []

    async def events(self) -> AsyncIterator[InputEvent]:
        async for event in self._path.events():
            self.seen.append(event)
            yield event


class SearchFailed(Exception):
    pass


class FakeSearch:
    def __init__(
        self,
        log: list[tuple[str, object]],
        result: SearchResult,
        gate: asyncio.Event | None = None,
        fails: str | None = None,
        later: SearchResult | None = None,
        holds: str | None = None,
    ) -> None:
        self._log = log
        self._result = result
        self._gate = gate
        self._fails = fails
        self._later = later
        self._holds = holds
        self.calls: list[tuple[str, list[str], Path, SearchBudget]] = []
        self.entered = asyncio.Event()
        self.held = asyncio.Event()

    async def __call__(
        self, query: str, globs: Sequence[str], root: Path, budget: SearchBudget
    ) -> SearchResult:
        self.calls.append((query, list(globs), root, budget))
        self.entered.set()
        if self._gate is not None and (self._holds is None or self._holds == query):
            self.held.set()
            try:
                await self._gate.wait()
            except asyncio.CancelledError:
                self._log.append(("search_cancelled", query))
                raise
        if query == self._fails:
            raise SearchFailed
        self._log.append(("search_done", query))
        # rg answers on a later tick; a result landing inside the caller's own step would let
        # the pump race a stager that does not exist yet.
        settled = asyncio.get_running_loop().create_future()
        settled.get_loop().call_soon(settled.set_result, None)
        await settled
        if self._later is not None and len(self.calls) > 1:
            return self._later
        return self._result


class FakeRegistry:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self._log = log

    def open_turn(self, turn_id: str) -> None:
        self._log.append(("open_turn", turn_id))

    def record(self, turn_id: str, result: SearchResult) -> None:
        self._log.append(("record", turn_id))

    def abandon(self, turn_id: str) -> None:
        self._log.append(("abandon", turn_id))

    def verify_chunk(self, turn_id: str, text: str, source: str = "model") -> GroundingVerdict:
        return GroundingVerdict(ok=True)


class RecordingRegistry(FakeRegistry):
    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__(log)
        self.verified: list[tuple[str, str, str]] = []

    def verify_chunk(self, turn_id: str, text: str, source: str = "model") -> GroundingVerdict:
        self.verified.append((turn_id, text, source))
        return GroundingVerdict(ok=True)


class TrippingRegistry(FakeRegistry):
    def __init__(self, log: list[tuple[str, object]], trip: asyncio.Event, clause: str) -> None:
        super().__init__(log)
        self._trip = trip
        self._clause = clause

    def verify_chunk(self, turn_id: str, text: str, source: str = "model") -> GroundingVerdict:
        if source == "model" and text == self._clause:
            self._trip.set()
        return GroundingVerdict(ok=True)


class FakeSpeaker:
    def __init__(
        self, log: list[tuple[str, object]], gate: asyncio.Event | None = None, hold_at: int = 2
    ) -> None:
        self._log = log
        self._gate = gate
        self._hold_at = hold_at
        self.utterances: list[list[str]] = []
        self.chunks: list[Chunk] = []
        self.openers: list[str] = []
        self.received = asyncio.Event()
        self.held = asyncio.Event()
        self.finished = asyncio.Event()

    async def speak_opener(self, key: str) -> None:
        self.openers.append(key)
        self._log.append(("speak_opener", key))

    async def speak(self, chunks: AsyncIterator[Chunk], on_play: OnPlay) -> None:
        spoken: list[str] = []
        self.utterances.append(spoken)
        try:
            async for chunk in chunks:
                spoken.append(chunk.text)
                self._log.append(("speak", chunk.text))
                self.chunks.append(chunk)
                await on_play(chunk, 0, 0)
                self.received.set()
                if self._gate is not None and len(spoken) >= self._hold_at:
                    self.held.set()
                    await self._gate.wait()
        except asyncio.CancelledError:
            self._log.append(("speak_cancelled", len(spoken)))
            raise
        self.finished.set()


VISUAL_PAYLOAD_TYPES = frozenset(
    {"diagram.push", "diagram.clear", "source.highlight", "app.push", "scene.push", "scene.show"}
)


def sent(log: list[tuple[str, object]], kind: str) -> list[dict[str, object]]:
    return [p for name, p in log if name == "send_json" and p["type"] == kind]


def without_seq(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "seq"}


class LoggingTransport:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self._log = log
        self.handlers: list[Callable[[dict[str, object]], None]] = []

    def flush_playout(self) -> None:
        self._log.append(("flush_playout", None))

    async def send_json(self, payload: dict[str, object]) -> None:
        self._log.append(("send_json", payload))

    def send_json_nowait(self, payload: dict[str, object]) -> bool:
        self._log.append(("send_json", payload))
        return True

    def on_json(self, handler: Callable[[dict[str, object]], None]) -> None:
        self.handlers.append(handler)


class CheckingTransport(LoggingTransport):
    def __init__(self, log: list[tuple[str, object]], ok: bool = True, error: str = "") -> None:
        super().__init__(log)
        self.ok = ok
        self.error = error
        self.reports: list[dict[str, object]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self._log.append(("send_json", payload))
        if payload["type"] != "scene.push":
            return
        steps = len(payload["steps"]) if self.ok else 0
        report = {
            "type": "scene.ready",
            "scene_id": payload["scene_id"],
            "ok": self.ok,
            "steps": steps,
            "error": self.error,
        }
        self.reports.append(report)
        asyncio.get_running_loop().call_soon(lambda: [h(report) for h in self.handlers])


def visual_call(name: str, arguments: str, call_id: str) -> TurnChunk:
    return TurnChunk(kind="tool_call", text=arguments, tool_call_id=call_id, tool_name=name)


class HeldTransport(LoggingTransport):
    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__(log)
        self.held = asyncio.Event()
        self.release = asyncio.Event()

    async def send_json(self, payload: dict[str, object]) -> None:
        self._log.append(("send_json", payload))
        if payload["type"] not in VISUAL_PAYLOAD_TYPES:
            return
        self.held.set()
        await self.release.wait()


class ChannelClosed(Exception):
    pass


class ClosedTransport(LoggingTransport):
    async def send_json(self, payload: dict[str, object]) -> None:
        raise ChannelClosed()


class ClosedListeningTransport(LoggingTransport):
    async def send_json(self, payload: dict[str, object]) -> None:
        if payload["type"] == "state" and payload["state"] == "listening":
            raise ChannelClosed()
        self._log.append(("send_json", payload))


class ClosedCanvasTransport(LoggingTransport):
    async def send_json(self, payload: dict[str, object]) -> None:
        if payload["type"] in VISUAL_PAYLOAD_TYPES:
            raise ChannelClosed()
        self._log.append(("send_json", payload))


class RateLimited(Exception):
    pass


class FakeStream:
    def __init__(
        self,
        chunks: list[TurnChunk],
        log: list[tuple[str, object]],
        received: asyncio.Event,
        gate: asyncio.Event | None = None,
        holds_at: int = 0,
        close_gate: asyncio.Event | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._log = log
        self._received = received
        self._gate = gate
        self._holds_at = holds_at
        self._close_gate = close_gate
        self._raises = raises
        self.iterations = 0
        self.cancels = 0
        self.yielded = 0
        self.finish_reason: str | None = None
        self.closed = False
        self.entered = asyncio.Event()
        self.held = asyncio.Event()
        self.filled = asyncio.Event()
        self.closing = asyncio.Event()

    async def _drain(self) -> AsyncIterator[TurnChunk]:
        self._log.append(("stream_open", self._received.is_set()))
        self.entered.set()
        try:
            for chunk in self._chunks:
                if self._gate is not None and self.yielded == self._holds_at:
                    self.held.set()
                    await self._gate.wait()
                self.yielded += 1
                if self.yielded == QUEUE_FULL_DELTAS:
                    self.filled.set()
                yield chunk
            if self._raises is not None:
                raise self._raises
        finally:
            self._log.append(("stream_closed", self.yielded))
            self.closed = True
            if self._close_gate is not None:
                self.closing.set()
                await self._close_gate.wait()
                self._log.append(("stream_released", self.yielded))

    def __aiter__(self) -> AsyncIterator[TurnChunk]:
        self.iterations += 1
        return self._drain()

    async def cancel(self) -> None:
        self.cancels += 1


class FakeReasoning:
    def __init__(
        self,
        log: list[tuple[str, object]],
        chunks: list[TurnChunk],
        received: asyncio.Event,
        follow_up: list[TurnChunk] | None = None,
        gate: asyncio.Event | None = None,
        fails: str | None = None,
        holds_at: int = 0,
        close_gate: asyncio.Event | None = None,
        turns: list[list[TurnChunk]] | None = None,
        follow_up_gate: asyncio.Event | None = None,
        visual: list[TurnChunk] | None = None,
        visual_finish: str | None = None,
        visual_gates: Sequence[asyncio.Event] | None = None,
        plans: Sequence[list[TurnChunk]] | None = None,
        plan_gates: Sequence[asyncio.Event] | None = None,
        plan_raises: BaseException | None = None,
        plan_finish: str | None = None,
        stream_raises: BaseException | None = None,
        scripts: Sequence[list[TurnChunk]] | None = None,
        script_gates: Sequence[asyncio.Event] | None = None,
        script_raises: BaseException | None = None,
    ) -> None:
        self._log = log
        self._chunks = chunks
        self._stream_raises = stream_raises
        self._follow_up = follow_up if follow_up is not None else []
        self._received = received
        self._gate = gate
        self._fails = fails
        self._holds_at = holds_at
        self._close_gate = close_gate
        self._turns = deque(turns) if turns is not None else deque()
        self._follow_up_gate = follow_up_gate
        self._visual = visual
        self._visual_finish = visual_finish
        self._visual_gates = deque(visual_gates) if visual_gates is not None else deque()
        self._plans = deque(plans) if plans is not None else deque()
        self._plan_gates = deque(plan_gates) if plan_gates is not None else deque()
        self._plan_raises = plan_raises
        self._plan_finish = plan_finish
        self._scripts = deque(scripts) if scripts is not None else deque()
        self._script_gates = deque(script_gates) if script_gates is not None else deque()
        self._script_raises = script_raises
        self.prompts: list[TurnPrompt] = []
        self.tools: list[list[dict] | None] = []
        self.max_tokens: list[int | None] = []
        self.tool_choices: list[str | None] = []
        self.models: list[str | None] = []
        self.efforts: list[str | None] = []
        self.streams: list[FakeStream] = []
        self.started = asyncio.Event()
        self.visual_started = asyncio.Event()
        self.build_prompts: list[TurnPrompt] = []
        self.build_tools: list[list[dict]] = []
        self.build_tool_choices: list[str | None] = []
        self.build_max_tokens: list[int | None] = []
        self.build_models: list[str | None] = []
        self.build_efforts: list[str | None] = []
        self.build_streams: list[FakeStream] = []
        self.plan_prompts: list[TurnPrompt] = []
        self.plan_tools: list[list[dict]] = []
        self.plan_tool_choices: list[str | None] = []
        self.plan_models: list[str | None] = []
        self.plan_efforts: list[str | None] = []
        self.plan_max_tokens: list[int | None] = []
        self.plan_streams: list[FakeStream] = []
        self.planning = asyncio.Event()
        self.script_prompts: list[TurnPrompt] = []
        self.script_models: list[str | None] = []
        self.script_streams: list[FakeStream] = []
        self.scripts_open: list[int] = []
        self.script_started = asyncio.Event()

    async def scripted(self, count: int) -> None:
        while len(self.script_prompts) < count:
            self.script_started.clear()
            await self.script_started.wait()

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
    ) -> FakeStream:
        if tools and tools[0]["function"]["name"] == PLAN_TOOL:
            self.plan_prompts.append(prompt)
            self.plan_tools.append(list(tools))
            self.plan_tool_choices.append(tool_choice)
            self.plan_models.append(model)
            self.plan_efforts.append(effort)
            self.plan_max_tokens.append(max_tokens)
            self._log.append(("plan", None))
            self.planning.set()
            if self._plan_raises is not None:
                raise self._plan_raises
            gate = self._plan_gates.popleft() if self._plan_gates else None
            chunks = self._plans.popleft() if self._plans else []
            stream = FakeStream(chunks, self._log, self._received, gate)
            stream.finish_reason = self._plan_finish or ("tool_calls" if chunks else "stop")
            self.plan_streams.append(stream)
            return stream
        if tools and tools[0]["function"]["name"] == SCRIPT_TOOL:
            self.scripts_open.append(sum(not stream.closed for stream in self.script_streams))
            self.script_prompts.append(prompt)
            self.script_models.append(model)
            self._log.append(("script", prompt.user_text))
            if self._scripts:
                chunks = self._scripts.popleft()
                gate = self._script_gates.popleft() if self._script_gates else None
            else:
                chunks, gate = [TurnChunk(kind="spoken", text="")], asyncio.Event()
            raises, self._script_raises = self._script_raises, None
            stream = FakeStream(chunks, self._log, self._received, gate, raises=raises)
            stream.finish_reason = "tool_calls" if chunks else "stop"
            self.script_streams.append(stream)
            self.script_started.set()
            return stream
        if tools and tools[0]["function"]["name"] == SCENE_TOOL:
            self.build_prompts.append(prompt)
            self.build_tools.append(list(tools))
            self.build_tool_choices.append(tool_choice)
            self.build_max_tokens.append(max_tokens)
            self.build_models.append(model)
            self.build_efforts.append(effort)
            self._log.append(("build", prompt.user_text))
            if self._visual is None:
                stream = FakeStream(
                    [TurnChunk(kind="spoken", text="")], self._log, self._received, asyncio.Event()
                )
            else:
                gate = self._visual_gates.popleft() if self._visual_gates else None
                stream = FakeStream(self._visual, self._log, self._received, gate)
                stream.finish_reason = self._visual_finish
            self.build_streams.append(stream)
            self.visual_started.set()
            return stream
        self.prompts.append(prompt)
        self.tools.append(list(tools) if tools is not None else None)
        self.max_tokens.append(max_tokens)
        self.tool_choices.append(tool_choice)
        self.models.append(model)
        self.efforts.append(effort)
        self._log.append(("start_turn", prompt.user_text))
        if prompt.user_text == self._fails:
            raise RateLimited(RATE_LIMIT_DETAIL)
        answering = bool(prompt.tool_exchange)
        if answering:
            chunks = self._follow_up
        elif self._turns:
            chunks = self._turns.popleft()
        else:
            chunks = self._chunks
        stream = FakeStream(
            chunks,
            self._log,
            self._received,
            self._follow_up_gate if answering else self._gate,
            self._holds_at,
            None if self.streams else self._close_gate,
            self._stream_raises,
        )
        self.streams.append(stream)
        self.started.set()
        return stream


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray) -> str:
        return USER_TEXT


class PacingVad:
    def __init__(
        self,
        probabilities: list[float],
        consumed: asyncio.Event,
        drained: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._probabilities = list(probabilities)
        self._consumed = consumed
        self._drained = drained
        self._loop = loop

    def __call__(self, frame: np.ndarray) -> float:
        probability = self._probabilities.pop(0)
        self._loop.call_soon_threadsafe(self._consumed.set)
        if not self._probabilities:
            self._loop.call_soon_threadsafe(self._drained.set)
        return probability

    def reset(self) -> None:
        return None


class PacedTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(
        self, frames: list[av.AudioFrame], pace: asyncio.Event, emitted: asyncio.Event
    ) -> None:
        super().__init__()
        self._frames = deque(frames)
        self._pace = pace
        self._emitted = emitted

    async def recv(self) -> av.AudioFrame:
        await self._pace.wait()
        self._pace.clear()
        if not self._frames:
            raise MediaStreamError
        frame = self._frames.popleft()
        self._emitted.set()
        return frame


def session_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "tutor.session"]


def spoken_chunks(deltas: list[str]) -> list[TurnChunk]:
    return [TurnChunk(kind="spoken", text=delta) for delta in deltas]


async def test_the_reasoning_call_starts_with_no_search_ahead_of_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [TurnChunk(kind="reasoning", text=REASONING_TEXT), *spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == []
    assert speaker.openers == []
    steps = [name for name, _ in log]
    assert "search_done" not in steps
    assert steps.index("open_turn") < steps.index("start_turn")
    assert ("start_turn", USER_TEXT) in log
    assert ("stream_open", False) in log
    assert [tool["function"]["name"] for tool in reasoning.tools[0]] == [
        "search_code",
        "clear_diagram",
        "highlight_source",
    ]

    prompt = reasoning.prompts[0]
    assert prompt.tool_context == []
    assert prompt.user_text == USER_TEXT
    assert SYSTEM in prompt.system
    assert SUBJECT in prompt.system
    assert speaker.utterances == [SPOKEN_CLAUSES]

    messages = session_messages(caplog)
    assert any(SPOKEN_SPAN.match(message) for message in messages)
    assert all(USER_TEXT not in message for message in messages)
    await loop.aclose()


async def test_frames_keep_arriving_while_a_turn_is_in_flight(tmp_path: Path) -> None:
    assert BURST_FRAMES > INBOUND_CAPACITY
    pc = local_peer()
    connection = Connection(pc)
    consumed = asyncio.Event()
    drained = asyncio.Event()
    pace = asyncio.Event()
    emitted = asyncio.Event()
    vad = PacingVad(PACED_PROBABILITIES, consumed, drained, asyncio.get_running_loop())
    path = InputPath(connection, vad, FakeTranscriber(), FakeTranscriber())
    source = RecordingSource(path)
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, found(), gate=gate)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received)
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        connection,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())
    pc.emit("datachannel", FakeChannel())
    pc.emit("track", PacedTrack(numbered_frames(len(PACED_PROBABILITIES)), pace, emitted))

    for _ in range(LEAD_FRAMES):
        consumed.clear()
        pace.set()
        await asyncio.wait_for(consumed.wait(), HANG_GUARD_S)
    await asyncio.wait_for(search.entered.wait(), HANG_GUARD_S)

    for _ in range(BURST_FRAMES):
        emitted.clear()
        pace.set()
        await asyncio.wait_for(emitted.wait(), HANG_GUARD_S)

    assert connection.dropped_frames == 0
    await asyncio.wait_for(drained.wait(), HANG_GUARD_S)

    assert connection.dropped_frames == 0
    assert len(search.calls) == 1
    assert [name for name, _ in log if name == "speak"] == []
    endpoints = [event for event in source.seen if not isinstance(event, PartialTranscript)]
    assert endpoints == [SpeechStarted(), EndOfTurn(text=USER_TEXT), SpeechStarted()]

    gate.set()
    pace.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert connection.dropped_frames == 0
    await loop.aclose()
    await path.aclose()
    await connection.close()


async def test_aclose_cancels_a_turn_still_in_flight(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, found(), gate=gate)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received)
    release = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), release])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(search.entered.wait(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    steps = [name for name, _ in log]
    assert "search_done" not in steps
    assert steps.index("search_cancelled") < steps.index("abandon")
    turn_ids = [turn_id for name, turn_id in log if name == "open_turn"]
    assert [turn_id for name, turn_id in log if name == "abandon"] == turn_ids
    assert "speak" not in steps
    assert len(reasoning.prompts) == 1

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert "speak" not in [name for name, _ in log]
    assert len(reasoning.prompts) == 1


async def test_a_turn_with_no_spoken_chunk_accepts_the_next_turn(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(
        log, [TurnChunk(kind="reasoning", text=REASONING_TEXT)], speaker.received
    )
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), speaker.finished, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[], []]
    assert [prompt.user_text for prompt in reasoning.prompts] == [USER_TEXT, SECOND_TEXT]
    turn_ids = [turn_id for name, turn_id in log if name == "open_turn"]
    assert len(set(turn_ids)) == 2
    assert "abandon" not in [name for name, _ in log]
    assert search.calls == []
    await loop.aclose()


async def test_a_failing_search_in_the_model_round_is_reported_and_the_loop_keeps_running(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found(), fails=MODEL_QUERY)
    turns = [[SEARCH_CALL], spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(
        log, [], speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS), turns=turns
    )
    source = SerialSource([USER_TEXT, SECOND_TEXT])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.reasoning_failed")] == [
        "turn.reasoning_failed turn_id=turn-1 error=SearchFailed"
    ]
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert search.calls == [(MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget())]
    assert "record" not in [name for name, _ in log]
    assert speaker.utterances == [[], SPOKEN_CLAUSES]
    assert [prompt.user_text for prompt in reasoning.prompts] == [USER_TEXT, SECOND_TEXT]
    captured = [record.getMessage() for record in caplog.records]
    assert all(USER_TEXT not in message and SECOND_TEXT not in message for message in captured)
    await loop.aclose()


async def test_a_tool_call_chunk_never_reaches_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [
        TurnChunk(kind="reasoning", text=REASONING_TEXT),
        TurnChunk(
            kind="tool_call",
            text=VISUAL_ARGUMENTS,
            tool_call_id=VISUAL_CALL_ID,
            tool_name=VISUAL_TOOL,
        ),
    ]
    follow_up = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [SPOKEN_CLAUSES]
    spoken = [text for name, text in log if name == "speak"]
    assert all(VISUAL_ARGUMENTS not in str(text) for text in spoken)
    assert all(REASONING_TEXT not in str(text) for text in spoken)

    recorded = [payload for name, payload in log if name in ("open_turn", "record", "abandon")]
    assert recorded == ["turn-1"]
    assert search.calls == []

    assert len(reasoning.prompts) == 2
    exchange = reasoning.prompts[1].tool_exchange
    assert [message.role for message in exchange] == ["assistant", "tool"]
    assert exchange[0].tool_calls[0].id == VISUAL_CALL_ID
    assert exchange[1].tool_call_id == VISUAL_CALL_ID
    assert VISUAL_TOOL in exchange[1].content
    assert exchange[1].content.startswith(f"{VISUAL_TOOL}: error:")

    assert all(VISUAL_ARGUMENTS not in message for message in session_messages(caplog))
    await loop.aclose()


async def test_a_search_code_call_dispatches_one_search_and_one_follow_up(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = RecordingRegistry(log)
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == [(MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget())]
    assert len(reasoning.prompts) == 2
    assert reasoning.tools == [[SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS], None]
    assert reasoning.prompts[0].tool_context == []

    exchange = reasoning.prompts[1].tool_exchange
    assert exchange[0].tool_calls[0].id == SEARCH_CALL_ID
    assert exchange[0].tool_calls[0].function.name == "search_code"
    assert exchange[0].tool_calls[0].function.arguments == SEARCH_ARGUMENTS
    assert exchange[1].tool_call_id == SEARCH_CALL_ID
    assert exchange[1].content == result.model_dump_json()

    steps = [name for name, _ in log]
    assert steps.index("start_turn") < steps.index("search_done") < steps.index("record")
    assert [turn_id for name, turn_id in log if name == "record"] == ["turn-1"]
    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [[lead_in, *SPOKEN_CLAUSES]]
    assert ("turn-1", lead_in, "lead_in") in registry.verified
    assert log.index(("record", "turn-1")) < log.index(("speak", lead_in))

    messages = session_messages(caplog)
    assert any(SEARCH_SPAN.match(message) for message in messages)
    assert all(MODEL_QUERY not in message for message in messages)
    await loop.aclose()


async def test_a_tool_round_does_not_glue_the_sentences_around_it(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [
        TurnChunk(kind="spoken", text="It lives there."),
        TurnChunk(
            kind="tool_call",
            text=SEARCH_ARGUMENTS,
            tool_call_id=SEARCH_CALL_ID,
            tool_name="search_code",
        ),
    ]
    follow_up = [TurnChunk(kind="spoken", text="Diagram incoming for the pool.")]
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log, deltas, speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    speaker.received.clear()
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    assert speaker.utterances == [["It lives there.", lead_in_sentence([result])]]
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert speaker.utterances == [
        ["It lives there.", lead_in_sentence([result]), "Diagram incoming for the pool."]
    ]
    await loop.aclose()


async def test_a_payload_in_the_stream_never_reaches_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [
        TurnChunk(kind="spoken", text="The pool is a free list. "),
        TurnChunk(kind="spoken", text='```json\n{"type":"diagram"}\n```'),
        TurnChunk(kind="spoken", text=" The acquire path takes a slot from it."),
    ]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [
        [
            "The pool is a free list.",
            "The acquire path takes a slot from it.",
        ]
    ]
    dropped = [m for m in session_messages(caplog) if m.startswith("turn.markup_dropped")]
    assert dropped == ["turn.markup_dropped turn_id=turn-1 fence=1"]
    await loop.aclose()


async def test_a_rate_limit_ends_the_turn_and_keeps_the_loop_running(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, fails=USER_TEXT)
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), speaker.finished, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[], SPOKEN_CLAUSES]
    assert search.calls == []

    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.reasoning_failed")] == [
        "turn.reasoning_failed turn_id=turn-1 error=RateLimited"
    ]
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert any(message.startswith("turn.spoken turn_id=turn-1 ") for message in messages)

    captured = [record.getMessage() for record in caplog.records]
    assert all(RATE_LIMIT_DETAIL not in message for message in captured)
    assert all(USER_TEXT not in message and SECOND_TEXT not in message for message in captured)
    await loop.aclose()


async def test_each_reasoning_stream_is_iterated_once(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [SEARCH_CALL]
    deltas += [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    follow_up = [TurnChunk(kind="spoken", text=delta) for delta in FOLLOW_DELTAS]
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log, deltas, speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    speaker.received.clear()
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    assert speaker.utterances == [CHAINED_CLAUSES[:2]]
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert [stream.iterations for stream in reasoning.streams] == [1, 1]
    assert len([name for name, _ in log if name == "start_turn"]) == 2
    assert speaker.utterances == [CHAINED_CLAUSES]
    await loop.aclose()


def turn_task(name: str = TURN_TASK) -> asyncio.Task[None]:
    turns = [task for task in asyncio.all_tasks() if task.get_name() == name]
    assert len(turns) == 1
    return turns[0]


def drain_task() -> asyncio.Task[None]:
    drains = [task for task in asyncio.all_tasks() if task.get_name() == DRAIN_TASK]
    assert len(drains) == 1
    return drains[0]


async def pull_past(source: ScriptedSource, gate: asyncio.Event) -> None:
    source.blocked.clear()
    gate.set()
    await asyncio.wait_for(source.blocked.wait(), HANG_GUARD_S)


async def test_a_barge_in_before_the_first_chunk_cancels_the_drain(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
    release = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), release])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].entered.wait(), HANG_GUARD_S)
    drain = drain_task()

    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert drain.cancelled()
    assert reasoning.streams[0].cancels == 0
    assert speaker.utterances == [[]]
    assert ("abandon", "turn-1") in log

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)


async def test_a_barge_in_on_a_full_spoken_queue_cancels_the_drain(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    search = FakeSearch(log, found())
    deltas = [TurnChunk(kind="spoken", text=FIRST_CLAUSE_DELTA)]
    deltas += [TurnChunk(kind="spoken", text=FILLER_DELTA) for _ in range(2 * QUEUE_FULL_DELTAS)]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    release = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), release])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].filled.wait(), HANG_GUARD_S)
    drain = drain_task()

    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert drain.cancelled()
    assert reasoning.streams[0].yielded == QUEUE_FULL_DELTAS
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]]]
    assert ("abandon", "turn-1") in log

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)


async def test_speech_start_mid_synthesis_cancels_flushes_and_takes_the_next_turn(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate, holds_at=3)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    drain = drain_task()
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]]]
    turn = turn_task()

    log.append(("barge", None))
    await pull_past(source, barge)
    gate.set()
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert drain.cancelled()
    assert reasoning.streams[0].cancels == 0
    steps = [name for name, _ in log]
    (sync,) = sent(log, "lesson.sync")
    assert log.index(("send_json", sync)) < steps.index("flush_playout")
    assert (
        steps.index("barge")
        < steps.index("flush_playout")
        < steps.index("stream_closed")
        < steps.index("speak_cancelled")
        < log.index(("abandon", "turn-1"))
    )
    assert ("stream_closed", 3) in log
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]]]
    assert ("open_turn", "turn-2") not in log

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert log.index(("abandon", "turn-1")) < log.index(("open_turn", "turn-2"))
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]], SPOKEN_CLAUSES]
    assert [entry for entry in log if entry[0] == "speak_cancelled"] == [("speak_cancelled", 1)]
    await loop.aclose()


async def test_speech_start_with_no_utterance_in_flight_still_flushes_playout(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = ScriptedSource([SpeechStarted(), EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    steps = [name for name, _ in log]
    assert steps.index("flush_playout") < steps.index("open_turn") < steps.index("speak")
    assert "abandon" not in steps
    assert "speak_cancelled" not in steps
    assert search.calls == []
    assert speaker.utterances == [SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_clause_past_the_chunker_at_speech_start_is_never_spoken(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    trip = asyncio.Event()
    registry = TrippingRegistry(log, trip, SPOKEN_CLAUSES[0])
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), trip, SpeechStarted()])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
        pace=HeldPace(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert SPOKEN_CLAUSES[0] not in [text for name, text in log if name == "speak"]
    assert speaker.utterances == [[lead_in_sentence([result])]]
    steps = [name for name, _ in log]
    assert steps.index("flush_playout") < log.index(("abandon", "turn-1"))
    await loop.aclose()


async def test_a_second_speech_start_during_cancellation_is_idempotent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    close_gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(
        log, deltas, speaker.received, gate=gate, holds_at=3, close_gate=close_gate
    )
    barge = asyncio.Event()
    second = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [
            EndOfTurn(text=USER_TEXT),
            barge,
            SpeechStarted(),
            second,
            SpeechStarted(),
            resume,
            EndOfTurn(text=SECOND_TEXT),
        ]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        turn = turn_task()

        await pull_past(source, barge)
        steps = [name for name, _ in log]
        assert steps.count("flush_playout") == 1
        await asyncio.wait_for(reasoning.streams[0].closing.wait(), HANG_GUARD_S)
        assert ("stream_closed", 3) in log

        await pull_past(source, second)
        steps = [name for name, _ in log]
        assert steps.count("flush_playout") == 2
        assert steps.count("speak_cancelled") == 1
        assert steps.count("abandon") == 0

        close_gate.set()
        await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
        assert turn.cancelled()
        assert ("stream_released", 3) in log
        assert [name for name, _ in log].count("abandon") == 1

        hold.set()
        gate.set()
        resume.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    steps = [name for name, _ in log]
    assert steps.count("flush_playout") == 2
    assert steps.count("speak_cancelled") == 1
    assert steps.count("abandon") == 1
    assert log.index(("abandon", "turn-1")) < log.index(("open_turn", "turn-2"))
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]], SPOKEN_CLAUSES]
    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.failed")] == []
    await loop.aclose()


async def test_aclose_during_cancellation_lets_the_turn_finish_its_cleanup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    close_gate = asyncio.Event()
    deltas = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(
        log, deltas, speaker.received, gate=gate, holds_at=3, close_gate=close_gate
    )
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), barge, SpeechStarted(), resume])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        turn = turn_task()

        await pull_past(source, barge)
        await asyncio.wait_for(reasoning.streams[0].closing.wait(), HANG_GUARD_S)
        assert ("stream_closed", 3) in log
        assert [name for name, _ in log].count("abandon") == 0

        closing = asyncio.create_task(loop.aclose())
        close_gate.set()
        await asyncio.wait_for(closing, HANG_GUARD_S)

        assert turn.cancelled()
        assert [name for name, _ in log].count("abandon") == 1
        assert ("stream_released", 3) in log
        messages = session_messages(caplog)
        assert [message for message in messages if message.startswith("turn.failed")] == []

        resume.set()
        await asyncio.wait_for(running, HANG_GUARD_S)


async def test_a_cancelled_turns_tool_results_do_not_ground_the_next_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    first = found()
    second = found(UNGROUNDED_PATH)
    registry = TurnRegistry()
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, first, later=second)
    follow_up = spoken_chunks(PATH_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        assert registry.known("turn-1", Position(path=GROUNDED_PATH, line=24))
        turn = turn_task()

        await pull_past(source, barge)
        hold.set()
        await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
        assert turn.cancelled()
        assert not registry.known("turn-1", Position(path=GROUNDED_PATH, line=24))

        resume.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert not registry.known("turn-2", Position(path=GROUNDED_PATH, line=24))
    assert registry.known("turn-2", Position(path=UNGROUNDED_PATH, line=24))
    assert speaker.utterances == [
        [lead_in_sentence([first]), PATH_CLAUSES[0]],
        [lead_in_sentence([second]), PATH_CLAUSES[1], PATH_CLAUSES[2]],
    ]
    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-2 source=model ungrounded=1" in messages
    await loop.aclose()


@pytest.mark.parametrize("arguments", BAD_ARGUMENT_BODIES)
async def test_unusable_search_code_arguments_answer_the_call_without_searching(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, arguments: str
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [
        TurnChunk(
            kind="tool_call",
            text=arguments,
            tool_call_id=SEARCH_CALL_ID,
            tool_name="search_code",
        )
    ]
    follow_up = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=follow_up)
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), speaker.finished, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == []
    assert "record" not in [name for name, _ in log]

    assert len(reasoning.prompts) == 4
    exchange = reasoning.prompts[1].tool_exchange
    assert exchange[0].tool_calls[0].id == SEARCH_CALL_ID
    assert exchange[1].tool_call_id == SEARCH_CALL_ID
    assert exchange[1].content == BAD_ARGUMENTS

    assert speaker.utterances == [SPOKEN_CLAUSES, SPOKEN_CLAUSES]
    spoken = [text for name, text in log if name == "speak"]
    assert all(arguments not in str(text) for text in spoken)

    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert [message for message in messages if message.startswith("turn.reasoning_failed")] == []
    assert all(arguments not in message for message in messages)
    await loop.aclose()


async def test_a_tool_call_missing_its_id_is_dropped_from_the_exchange(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [
        TurnChunk(kind="tool_call", text=TRUNCATED_ARGUMENTS, tool_name="search_code"),
        TurnChunk(
            kind="tool_call",
            text=SEARCH_ARGUMENTS,
            tool_call_id=SEARCH_CALL_ID,
            tool_name="search_code",
        ),
    ]
    follow_up = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == [(MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget())]

    assert len(reasoning.prompts) == 2
    exchange = reasoning.prompts[1].tool_exchange
    assert [call.id for call in exchange[0].tool_calls] == [SEARCH_CALL_ID]
    assert [message.tool_call_id for message in exchange[1:]] == [SEARCH_CALL_ID]
    assert exchange[1].content == result.model_dump_json()

    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
    spoken = [text for name, text in log if name == "speak"]
    assert all(TRUNCATED_ARGUMENTS not in str(text) for text in spoken)

    messages = session_messages(caplog)
    assert "turn.tool_call_incomplete turn_id=turn-1" in messages
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert [message for message in messages if message.startswith("turn.reasoning_failed")] == []
    await loop.aclose()


async def test_a_lone_tool_call_missing_its_id_skips_the_follow_up(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [TurnChunk(kind="tool_call", text=TRUNCATED_ARGUMENTS, tool_name="search_code")]
    deltas += [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), speaker.finished, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert [prompt.user_text for prompt in reasoning.prompts] == [USER_TEXT, SECOND_TEXT]
    assert reasoning.tools == [
        [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS],
        [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS],
    ]
    assert search.calls == []

    assert speaker.utterances == [SPOKEN_CLAUSES, SPOKEN_CLAUSES]
    spoken = [text for name, text in log if name == "speak"]
    assert all(TRUNCATED_ARGUMENTS not in str(text) for text in spoken)

    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.tool_call_incomplete")] == [
        "turn.tool_call_incomplete turn_id=turn-1",
        "turn.tool_call_incomplete turn_id=turn-2",
    ]
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert [message for message in messages if message.startswith("turn.reasoning_failed")] == []
    await loop.aclose()


async def test_a_visual_tool_call_reaches_the_channel_before_the_sentence_that_follows_it(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [visual_call("clear_diagram", CLEAR_ARGUMENTS, VISUAL_CALL_ID)]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS))
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    pushes = sent(log, "diagram.clear")
    assert [without_seq(payload) for payload in pushes] == [CLEAR_PAYLOAD]
    assert log.index(("send_json", pushes[0])) < log.index(("speak", SPOKEN_CLAUSES[0]))
    assert reasoning.prompts[1].tool_exchange[1].content == "clear_diagram: sent"
    assert [tool["function"]["name"] for tool in reasoning.tools[0]] == [
        "search_code",
        "clear_diagram",
        "highlight_source",
    ]
    assert reasoning.tools[1] is None
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert search.calls == []
    await loop.aclose()


async def test_a_payload_on_a_cancelled_turn_is_never_sent(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    gate = asyncio.Event()
    deltas = [
        visual_call("clear_diagram", CLEAR_ARGUMENTS, VISUAL_CALL_ID),
        TurnChunk(kind="spoken", text=SPOKEN_DELTAS[0]),
    ]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate, holds_at=1)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), barge, SpeechStarted(), resume])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    turn = turn_task()
    drain = drain_task()

    await pull_past(source, barge)
    gate.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert drain.cancelled()
    assert sent(log, "diagram.clear") == []
    assert len(reasoning.prompts) == 1
    assert ("abandon", "turn-1") in log

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await loop.aclose()


async def test_a_barge_in_during_a_push_sends_nothing_further(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [
        visual_call("clear_diagram", CLEAR_ARGUMENTS, VISUAL_CALL_ID),
        visual_call("clear_diagram", CLEAR_ARGUMENTS, "call-visual-2"),
    ]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS))
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), barge, SpeechStarted(), resume])
    transport = HeldTransport(log)
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        transport,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(transport.held.wait(), HANG_GUARD_S)
    turn = turn_task()
    drain = drain_task()

    await pull_past(source, barge)
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert drain.cancelled()
    assert len(sent(log, "diagram.clear")) == 1
    assert len(reasoning.prompts) == 1
    assert ("abandon", "turn-1") in log
    steps = [name for name, _ in log]
    assert steps.index("flush_playout") < log.index(("abandon", "turn-1"))

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await loop.aclose()


async def test_grounding_reaches_the_channel_before_a_highlight_and_an_ungrounded_one_is_an_error_string(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [
        SEARCH_CALL,
        visual_call("highlight_source", GROUNDED_HIGHLIGHT, VISUAL_CALL_ID),
        visual_call("highlight_source", UNGROUNDED_HIGHLIGHT, "call-visual-2"),
    ]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS))
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        TurnRegistry(),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert [without_seq(payload) for payload in sent(log, "source.highlight")] == [
        {"type": "source.highlight", "path": GROUNDED_PATH, "start_line": 24, "end_line": 24}
    ]
    exchange = reasoning.prompts[1].tool_exchange
    assert [message.content for message in exchange[1:]] == [
        result.model_dump_json(),
        "highlight_source: sent",
        f"highlight_source: error: ungrounded highlight {UNGROUNDED_PATH!r}:30",
    ]
    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_transport_error_in_a_push_ends_the_drain_and_the_next_turn_is_taken(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [visual_call("clear_diagram", CLEAR_ARGUMENTS, VISUAL_CALL_ID)]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS))
    source = SerialSource([USER_TEXT, SECOND_TEXT])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        ClosedCanvasTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "turn.reasoning_failed turn_id=turn-1 error=ChannelClosed" in messages
    assert "turn.reasoning_failed turn_id=turn-2 error=ChannelClosed" in messages
    assert [prompt.user_text for prompt in reasoning.prompts] == [USER_TEXT, SECOND_TEXT]
    assert speaker.utterances == [[], []]
    await loop.aclose()


async def test_a_push_diagram_in_the_voice_stream_is_unrouted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [visual_call("push_diagram", DIAGRAM_ARGUMENTS, VISUAL_CALL_ID)]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS))
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    exchange = reasoning.prompts[1].tool_exchange
    assert exchange[1].content == "push_diagram is not available in this turn"
    assert not any(sent(log, kind) for kind in VISUAL_PAYLOAD_TYPES)
    assert "turn.tool_unrouted turn_id=turn-1 tool=push_diagram" in session_messages(caplog)
    assert speaker.utterances == [SPOKEN_CLAUSES]
    await loop.aclose()


async def test_the_tool_round_carries_its_own_token_cap_and_the_follow_up_the_default(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [SEARCH_CALL]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS))
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    cfg = config(tmp_path)
    loop = TurnLoop(
        cfg,
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert reasoning.max_tokens == [cfg.tool_round_max_tokens, None]
    await loop.aclose()


async def test_an_ungrounded_chunk_is_withheld_from_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    follow_up = spoken_chunks(PATH_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[lead_in_sentence([result]), PATH_CLAUSES[0], PATH_CLAUSES[2]]]

    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    assert all(UNGROUNDED_PATH not in message for message in messages)
    await loop.aclose()


async def test_a_bare_count_is_withheld_from_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    follow_up = spoken_chunks(COUNT_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[lead_in_sentence([result]), LINE_CLAUSES[0]]]

    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    await loop.aclose()


async def test_a_recorded_position_reaches_the_speaker_unchanged(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    follow_up = spoken_chunks(LINE_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[lead_in_sentence([result])] + LINE_CLAUSES]
    await loop.aclose()


async def test_last_turns_position_does_not_authorize_this_turn(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    first = found()
    second = found(UNGROUNDED_PATH)
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, first, later=second)
    follow_up = spoken_chunks(PATH_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), speaker.finished, EndOfTurn(text=SECOND_TEXT)]
    )
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [
        [lead_in_sentence([first]), PATH_CLAUSES[0], PATH_CLAUSES[2]],
        [lead_in_sentence([second]), PATH_CLAUSES[1], PATH_CLAUSES[2]],
    ]
    await loop.aclose()


async def test_a_lead_in_the_gate_cannot_reparse_is_withheld(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found(UNPARSEABLE_PATH)
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(log, [SEARCH_CALL], speaker.received, follow_up=follow_up)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert lead_in_sentence([result]) not in [text for name, text in log if name == "speak"]
    assert speaker.utterances == [SPOKEN_CLAUSES]

    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=lead_in ungrounded=1" in messages
    assert all(UNPARSEABLE_PATH not in message for message in messages)
    await loop.aclose()


async def test_a_tool_result_answering_an_abandoned_turn_reaches_nobody(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, found(UNGROUNDED_PATH), gate=gate)
    deltas = [SEARCH_CALL]
    follow_up = [TurnChunk(kind="spoken", text=delta) for delta in PATH_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, follow_up=follow_up)
    release = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), release])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(search.held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    gate.set()
    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert len(search.calls) == 1
    assert ("search_cancelled", MODEL_QUERY) in log
    assert ("search_done", MODEL_QUERY) not in log
    assert not registry.known("turn-1", Position(path=UNGROUNDED_PATH, line=24))
    assert not registry.known("turn-1", Position(path=UNGROUNDED_PATH))
    assert speaker.utterances == [[]]
    assert PATH_CLAUSES[1] not in [text for name, text in log if name == "speak"]


async def test_a_line_number_split_across_the_cut_never_reaches_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPLIT_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    clauses, remainder = split_clauses("".join(SPLIT_DELTAS), 40)
    spoken = [text for name, text in log if name == "speak"]
    assert clauses + [remainder] == SPLIT_CLAUSES
    assert speaker.utterances == [SPLIT_CLAUSES[:2]]
    assert all(str(SPLIT_LINE) not in chunk for chunk in spoken)

    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    await loop.aclose()


async def test_a_stage_waits_out_the_gap_and_stops_once_the_model_speaks(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result, later=found())
    gate = asyncio.Event()
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(
        log, [SEARCH_CALL], speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), speaker.finished, EndOfTurn(text=SECOND_TEXT)]
    )
    pace = HeldPace()
    cfg = config(tmp_path)
    loop = TurnLoop(
        cfg,
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace=pace,
    )
    running = asyncio.create_task(loop.run())
    lead_in = lead_in_sentence([result])
    stages = list(lead_in_stages(result))

    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    stagers = [task for task in asyncio.all_tasks() if task.get_name() == STAGER_TASK]
    assert len(stagers) == 1
    assert speaker.utterances == [[lead_in]]
    assert search.calls == [(MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget())]
    assert pace.waits == [cfg.stage_gap_ms / 1000]

    speaker.received.clear()
    pace.release_once()
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    assert speaker.utterances == [[lead_in, stages[0]]]
    assert pace.waits == [cfg.stage_gap_ms / 1000] * 2
    assert not stagers[0].done()

    gate.set()
    await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
    assert stagers[0].done()
    assert stagers[0].cancelled()

    pace.release_once()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert pace.waits == [cfg.stage_gap_ms / 1000] * 2
    assert speaker.utterances == [
        [lead_in, stages[0], *SPOKEN_CLAUSES],
        [lead_in_sentence([found()]), *SPOKEN_CLAUSES],
    ]
    await loop.aclose()


async def test_every_stage_is_verified_as_a_lead_in_chunk_on_the_turn(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    registry = RecordingRegistry(log)
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    gate = asyncio.Event()
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(
        log, [SEARCH_CALL], speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    pace = HeldPace()
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
        pace=pace,
    )
    running = asyncio.create_task(loop.run())
    stages = list(lead_in_stages(result))

    for _ in stages[:2]:
        await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
        speaker.received.clear()
        pace.release_once()
        await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [[lead_in, *stages[:2], *SPOKEN_CLAUSES]]
    assert ("turn-1", lead_in, "lead_in") in registry.verified
    for sentence in stages[:2]:
        assert ("turn-1", sentence, "lead_in") in registry.verified
    sources = {source for _, text, source in registry.verified if text in stages}
    assert sources == {"lead_in"}
    assert [source for _, text, source in registry.verified if text in SPOKEN_CLAUSES] == [
        "model",
        "model",
    ]
    await loop.aclose()


async def test_a_stage_never_lends_its_path_to_the_model_stream(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    gate = asyncio.Event()
    follow_up = spoken_chunks(CARRY_DELTAS)
    reasoning = FakeReasoning(
        log, [SEARCH_CALL], speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    pace = HeldPace()
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        registry,
        FakeClock(),
        pace=pace,
    )
    running = asyncio.create_task(loop.run())
    stages = list(lead_in_stages(result))

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        for _ in stages[:2]:
            await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
            speaker.received.clear()
            pace.release_once()
            await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
        gate.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert UNGROUNDED_PATH in stages[1]
    assert speaker.utterances == [[lead_in_sentence([result]), *stages[:2], *CARRY_CLAUSES]]
    assert CARRY_DELTAS[0].strip() not in [text for name, text in log if name == "speak"]
    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    assert [message for message in messages if message.startswith("turn.stage")] == [
        "turn.stage turn_id=turn-1 n=1",
        "turn.stage turn_id=turn-1 n=2",
        "turn.stage turn_id=turn-1 n=3",
    ]
    await loop.aclose()


async def test_the_first_spoken_chunk_mid_stage_rides_the_turns_one_iterator(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    gate = asyncio.Event()
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(
        log, [SEARCH_CALL], speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    pace = HeldPace()
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace=pace,
    )
    running = asyncio.create_task(loop.run())
    stages = list(lead_in_stages(result))

    for _ in stages[:2]:
        await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
        speaker.received.clear()
        pace.release_once()
        await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert len(speaker.utterances) == 1
    assert speaker.utterances == [[lead_in_sentence([result]), *stages[:2], *SPOKEN_CLAUSES]]
    assert len([entry for entry in log if entry[0] == "speak"]) == 3 + len(SPOKEN_CLAUSES)
    await loop.aclose()


async def test_a_stalled_speaker_holds_the_next_stage_back(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, result)
    gate = asyncio.Event()
    follow_up = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(
        log, [SEARCH_CALL], speaker.received, follow_up=follow_up, follow_up_gate=gate
    )
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    pace = HeldPace(log)
    cfg = config(tmp_path)
    loop = TurnLoop(
        cfg,
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace=pace,
    )
    running = asyncio.create_task(loop.run())
    lead_in = lead_in_sentence([result])
    stages = list(lead_in_stages(result))

    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    speaker.received.clear()
    pace.release_once()
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    assert speaker.utterances == [[lead_in, stages[0]]]

    log.append(("gate_released", None))
    hold.set()
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)

    assert pace.waits == [cfg.stage_gap_ms / 1000] * 2
    paces = [i for i, entry in enumerate(log) if entry[0] == "pace"]
    assert log.index(("speak", stages[0])) < log.index(("gate_released", None)) < paces[1]

    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert pace.waits == [cfg.stage_gap_ms / 1000] * 2
    assert speaker.utterances == [[lead_in, stages[0], *SPOKEN_CLAUSES]]
    await loop.aclose()


def speculative(root: Path) -> TurnLoopConfig:
    cfg = TurnLoopConfig(
        system=SYSTEM, subject=SUBJECT, root=root, speculative_reasoning=True, planned=False
    )
    assert cfg.speculative_reasoning
    return cfg


def speculation_task() -> asyncio.Task[None]:
    tasks = [task for task in asyncio.all_tasks() if task.get_name() == SPECULATION_TASK]
    assert len(tasks) == 1
    return tasks[0]


def spoken_spans(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [message for message in session_messages(caplog) if message.startswith("turn.spoken")]


async def test_partials_are_ignored_with_speculation_off(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=USER_TEXT),
            speaker.finished,
        ]
    )
    cfg = config(tmp_path)
    assert cfg.speculative_reasoning is False
    loop = TurnLoop(
        cfg,
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(source.blocked.wait(), HANG_GUARD_S)
    log.remove(("send_json", {"type": "lesson.attach", "epoch": 1, "seq": 1}))
    assert search.calls == []
    assert reasoning.prompts == []
    assert log == []

    await pull_past(source, endpoint)
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert search.calls == []
    assert [entry for entry in log if entry[0] == "start_turn"] == [("start_turn", USER_TEXT)]
    assert speaker.utterances == [SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_partial_issues_the_request_and_a_matching_final_reuses_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received, gate=gate)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=""),
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=USER_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        assert search.calls == []
        assert [entry for entry in log if entry[0] == "start_turn"] == [
            ("start_turn", PARTIAL_TEXT)
        ]
        assert ("stream_open", False) in log
        assert reasoning.streams[0].iterations == 1
        assert reasoning.prompts[0].tool_context == []
        assert speaker.openers == []
        assert speaker.utterances == []

        await pull_past(source, endpoint)
        assert [entry for entry in log if entry[0] == "start_turn"] == [
            ("start_turn", PARTIAL_TEXT)
        ]

        gate.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert search.calls == []
    assert [entry for entry in log if entry[0] == "start_turn"] == [("start_turn", PARTIAL_TEXT)]
    assert reasoning.streams[-1].iterations == 1
    assert speaker.openers == []
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert [entry for entry in log if entry[0] == "open_turn"] == [("open_turn", TURN_TASK)]
    assert "record" not in [name for name, _ in log]
    assert "abandon" not in [name for name, _ in log]
    assert len(spoken_spans(caplog)) == 1
    messages = session_messages(caplog)
    assert all(PARTIAL_TEXT not in message for message in messages)
    await loop.aclose()


async def test_a_speculative_round_carries_no_visual_tools(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received, gate=gate)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=""),
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=USER_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert reasoning.tools == [[SEARCH_CODE_TOOL]]
    assert search.calls == []
    assert not any(sent(log, kind) for kind in VISUAL_PAYLOAD_TYPES)
    assert speaker.utterances == [SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_later_partial_cancels_and_reissues_the_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
    longer = asyncio.Event()
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=FIRST_PARTIAL),
            longer,
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=USER_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        first = speculation_task()
        reasoning.started.clear()

        await pull_past(source, longer)
        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[1].held.wait(), HANG_GUARD_S)

        assert first.cancelled()
        assert reasoning.streams[0].cancels == 0
        assert [entry for entry in log if entry[0] == "start_turn"] == [
            ("start_turn", FIRST_PARTIAL),
            ("start_turn", PARTIAL_TEXT),
        ]
        opens = [i for i, entry in enumerate(log) if entry == ("open_turn", TURN_TASK)]
        assert len(opens) == 2
        assert log.index(("stream_closed", 0)) < log.index(("abandon", TURN_TASK)) < opens[1]
        assert [name for name, _ in log].count("abandon") == 1
        assert speaker.utterances == []

        await pull_past(source, endpoint)
        gate.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert [name for name, _ in log].count("start_turn") == 2
    assert reasoning.streams[-1].iterations == 1
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert search.calls == []
    assert [name for name, _ in log].count("abandon") == 1
    assert len(spoken_spans(caplog)) == 1
    await loop.aclose()


async def test_a_mismatching_final_discards_the_speculation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=SECOND_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        speculation = speculation_task()
        reasoning.started.clear()

        await pull_past(source, endpoint)
        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        gate.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert speculation.cancelled()
    assert reasoning.streams[0].cancels == 0
    assert [entry for entry in log if entry[0] == "start_turn"] == [
        ("start_turn", PARTIAL_TEXT),
        ("start_turn", SECOND_TEXT),
    ]
    opens = [i for i, entry in enumerate(log) if entry == ("open_turn", TURN_TASK)]
    assert len(opens) == 2
    assert (
        log.index(("stream_closed", 0))
        < log.index(("abandon", TURN_TASK))
        < opens[1]
        < log.index(("start_turn", SECOND_TEXT))
    )
    assert [name for name, _ in log].count("abandon") == 1
    assert search.calls == []
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert len(spoken_spans(caplog)) == 1
    await loop.aclose()


async def test_speech_start_cancels_the_speculation(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=PARTIAL_TEXT),
            barge,
            SpeechStarted(),
            resume,
            EndOfTurn(text=USER_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    speculation = speculation_task()
    reasoning.started.clear()

    await pull_past(source, barge)
    await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
    assert speculation.cancelled()
    assert reasoning.streams[0].cancels == 0
    steps = [name for name, _ in log]
    assert steps.index("flush_playout") < steps.index("stream_closed") < steps.index("abandon")
    assert speaker.utterances == []

    await pull_past(source, resume)
    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert [entry for entry in log if entry[0] == "start_turn"] == [
        ("start_turn", PARTIAL_TEXT),
        ("start_turn", USER_TEXT),
    ]
    assert log.index(("abandon", TURN_TASK)) < log.index(("start_turn", USER_TEXT))
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert search.calls == []
    await loop.aclose()


async def test_aclose_cancels_a_speculation_in_flight(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
    release = asyncio.Event()
    source = ScriptedSource([PartialTranscript(text=PARTIAL_TEXT), release])
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    speculation = speculation_task()

    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speculation.cancelled()
    assert ("stream_closed", 0) in log
    assert reasoning.streams[0].cancels == 0
    assert ("abandon", TURN_TASK) in log
    assert speaker.openers == []
    assert speaker.utterances == []

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert speaker.utterances == []
    assert search.calls == []
    assert [name for name, _ in log].count("start_turn") == 1


async def test_a_failed_speculation_is_reported_and_the_final_starts_its_own_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), speaker.received, fails=PARTIAL_TEXT
    )
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=USER_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(source.blocked.wait(), HANG_GUARD_S)
        assert [entry for entry in log if entry[0] == "start_turn"] == [
            ("start_turn", PARTIAL_TEXT)
        ]
        assert speaker.utterances == []

        await pull_past(source, endpoint)
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert search.calls == []
    assert [entry for entry in log if entry[0] == "start_turn"] == [
        ("start_turn", PARTIAL_TEXT),
        ("start_turn", USER_TEXT),
    ]
    assert speaker.utterances == [SPOKEN_CLAUSES]
    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.speculation_failed")] == [
        "turn.speculation_failed turn_id=turn-1 error=RateLimited"
    ]
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert len(spoken_spans(caplog)) == 1
    captured = [record.getMessage() for record in caplog.records]
    assert all(PARTIAL_TEXT not in message for message in captured)
    assert all(RATE_LIMIT_DETAIL not in message for message in captured)
    await loop.aclose()


async def test_a_claimed_speculation_that_fails_ends_the_turn_as_a_reasoning_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, found(), gate=gate, fails=MODEL_QUERY)
    turns = [[SEARCH_CALL], spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(
        log, [], speaker.received, follow_up=spoken_chunks(SPOKEN_DELTAS), turns=turns
    )
    endpoint = asyncio.Event()
    second = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=USER_TEXT),
            second,
            EndOfTurn(text=SECOND_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        speculative(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(search.held.wait(), HANG_GUARD_S)
        speculation = speculation_task()

        await pull_past(source, endpoint)
        turn = turn_task()
        assert speaker.openers == []
        assert speaker.utterances == [[]]

        gate.set()
        await asyncio.wait_for(asyncio.wait([speculation, turn]), HANG_GUARD_S)
        assert not turn.cancelled()
        assert turn.exception() is None
        assert isinstance(speculation.exception(), SearchFailed)
        assert speaker.utterances == [[]]
        assert len(reasoning.prompts) == 1
        assert [entry for entry in log if entry[0] in ("open_turn", "record", "abandon")] == [
            ("open_turn", TURN_TASK)
        ]

        await pull_past(source, second)
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert search.calls == [(MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget())]
    assert speaker.utterances == [[], SPOKEN_CLAUSES]
    assert [prompt.user_text for prompt in reasoning.prompts] == [PARTIAL_TEXT, SECOND_TEXT]
    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.reasoning_failed")] == [
        "turn.reasoning_failed turn_id=turn-1 error=SearchFailed"
    ]
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert [message for message in messages if message.startswith("turn.speculation_failed")] == []
    assert len(spoken_spans(caplog)) == 2
    captured = [record.getMessage() for record in caplog.records]
    assert all(PARTIAL_TEXT not in message for message in captured)
    await loop.aclose()


def states(log: list[tuple[str, object]]) -> list[str]:
    return [str(payload["state"]) for payload in sent(log, "state")]


async def test_a_concept_turn_makes_one_call_with_no_tools_and_no_search() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == []
    assert speaker.openers == []
    assert reasoning.tools == [None]
    prompt = reasoning.prompts[0]
    assert prompt.tool_context == []
    assert prompt.user_text == CONCEPT_TEXT
    assert "Subject: PPO" in prompt.system
    assert f"Starting from: {STARTING_FROM}" in prompt.system
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert ("open_turn", "turn-1") in log
    await loop.aclose()


async def test_a_concept_turn_sends_thinking_then_speaking_then_listening() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert [without_seq(p) for p in sent(log, "state")] == [
        {"type": "state", "state": state, "interrupted": False}
        for state in ("thinking", "speaking", "listening")
    ]
    speaking = next(
        i for i, (n, p) in enumerate(log) if n == "send_json" and p.get("state") == "speaking"
    )
    assert speaking < log.index(("speak", SPOKEN_CLAUSES[0]))
    await loop.aclose()


async def test_chunk_ids_rise_by_one_across_turns() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT, "why clip"])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.chunks == [
        Chunk(1, SPOKEN_CLAUSES[0]),
        Chunk(2, SPOKEN_CLAUSES[1]),
        Chunk(3, SPOKEN_CLAUSES[0]),
        Chunk(4, SPOKEN_CLAUSES[1]),
    ]
    await loop.aclose()


async def test_every_spoken_clause_is_captioned_once_it_reaches_the_playout() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    captions = sent(log, "caption")
    assert [p["text"] for p in captions] == SPOKEN_CLAUSES
    for caption in captions:
        assert caption["turn_id"] == "turn-1"
        assert caption["lead_ms"] == 0
        assert log.index(("speak", caption["text"])) < log.index(("send_json", caption))
    await loop.aclose()


async def test_the_learner_text_reaches_the_channel_before_the_call() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    transcript = {"type": "transcript", "turn_id": "turn-1", "text": CONCEPT_TEXT, "seq": 2}
    assert sent(log, "transcript") == [transcript]
    assert log.index(("send_json", transcript)) < log.index(("start_turn", CONCEPT_TEXT))
    await loop.aclose()


async def test_the_second_turn_carries_the_first_in_its_history() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT, "why clip"])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert reasoning.prompts[0].history == []
    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]
    await loop.aclose()


async def test_history_is_bounded_by_the_config() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT, "two", "three"])
    loop = TurnLoop(
        concept_cfg().model_copy(update={"history_turns": 1}),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert reasoning.prompts[2].history == [
        Message(role="user", content="two"),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]
    await loop.aclose()


async def test_a_cancelled_concept_turn_keeps_what_was_spoken_and_ends_listening() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate, holds_at=3)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text="two")]
    )
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    drain = drain_task()
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]]]
    turn = turn_task()

    log.append(("barge", None))
    await pull_past(source, barge)
    gate.set()
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert drain.cancelled()
    steps = [name for name, _ in log]
    assert (
        steps.index("barge")
        < steps.index("flush_playout")
        < steps.index("stream_closed")
        < steps.index("speak_cancelled")
        < log.index(("abandon", "turn-1"))
    )
    assert states(log) == ["thinking", "speaking", "listening"]
    assert ("open_turn", "turn-2") not in log

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert reasoning.prompts[1].history[1].content == SPOKEN_CLAUSES[0]
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]], SPOKEN_CLAUSES]
    assert states(log) == ["thinking", "speaking", "listening"] * 2
    await loop.aclose()


async def test_only_the_barged_listening_state_is_marked_interrupted() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate, holds_at=3)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text="two")]
    )
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    turn = turn_task()

    await pull_past(source, barge)
    gate.set()
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert [(p["state"], p["interrupted"]) for p in sent(log, "state")] == [
        ("thinking", False),
        ("speaking", False),
        ("listening", True),
    ]

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert [p["interrupted"] for p in sent(log, "state")] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    await loop.aclose()


async def test_a_failed_listening_push_on_a_cancelled_turn_keeps_the_cancellation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate, holds_at=3)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text="two")]
    )
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        ClosedListeningTransport(log),
        reasoning,
        FakeRegistry(log),
    )
    with caplog.at_level(logging.WARNING, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
        drain = drain_task()
        turn = turn_task()

        await pull_past(source, barge)
        gate.set()
        hold.set()
        await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

        assert turn.cancelled()
        assert drain.cancelled()
        messages = session_messages(caplog)
        assert "turn.state_push_failed turn_id=turn-1 error=ChannelClosed" in messages
        assert not [message for message in messages if message.startswith("turn.failed")]
        assert ("abandon", "turn-1") in log

        resume.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
    await loop.aclose()


async def test_a_turn_without_a_folder_speaks_bare_numbers_and_withholds_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(CONCEPT_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        TurnRegistry(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[CONCEPT_CLAUSES[0]]]
    withheld = [m for m in session_messages(caplog) if m.startswith("turn.chunk_withheld")]
    assert withheld == ["turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1"]
    await loop.aclose()


async def test_a_turn_with_a_folder_still_withholds_a_bare_number(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(CONCEPT_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT])
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        TurnRegistry(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [[]]
    withheld = [m for m in session_messages(caplog) if m.startswith("turn.chunk_withheld")]
    assert withheld == [
        "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=2",
        "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1",
    ]
    await loop.aclose()


async def test_a_cancelled_turns_listening_push_yields_to_the_next_turn() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    close_gate = asyncio.Event()
    deltas = spoken_chunks(SPOKEN_DELTAS)
    reasoning = FakeReasoning(
        log, deltas, speaker.received, gate=gate, holds_at=3, close_gate=close_gate
    )
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text="two")]
    )
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    turn = turn_task()

    await pull_past(source, barge)
    await asyncio.wait_for(reasoning.streams[0].closing.wait(), HANG_GUARD_S)
    assert not turn.done()
    assert states(log) == ["thinking", "speaking"]

    speaker.received.clear()
    resume.set()
    await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
    assert len(reasoning.streams) == 2
    assert ("open_turn", "turn-2") in log
    assert not turn.done()

    close_gate.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    assert turn.cancelled()
    gate.set()
    hold.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert states(log) == ["thinking", "speaking", "thinking", "speaking", "listening"]
    await loop.aclose()


async def test_a_transport_error_on_a_state_push_ends_the_turn_and_the_next_is_taken(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([CONCEPT_TEXT, "two"])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        ClosedTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    with caplog.at_level(logging.ERROR, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "turn.failed turn_id=turn-1 error=ChannelClosed" in messages
    assert "turn.failed turn_id=turn-2 error=ChannelClosed" in messages
    await loop.aclose()


async def test_a_partial_with_no_folder_starts_a_speculation_with_no_tools() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received, gate=gate)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=CONCEPT_PARTIAL),
            endpoint,
            EndOfTurn(text=CONCEPT_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        concept_cfg().model_copy(update={"speculative_reasoning": True}),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    assert CONCEPT_TEXT.startswith(CONCEPT_PARTIAL)
    assert [
        task.get_name() for task in asyncio.all_tasks() if task.get_name() == SPECULATION_TASK
    ] == [SPECULATION_TASK]

    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert search.calls == []
    assert len(reasoning.prompts) == 1
    assert reasoning.tools == [None]
    assert reasoning.prompts[0].tool_context == []
    assert reasoning.prompts[0].user_text == CONCEPT_PARTIAL
    assert [entry for entry in log if entry[0] == "open_turn"] == [("open_turn", TURN_TASK)]
    assert speaker.utterances == [SPOKEN_CLAUSES]
    await loop.aclose()


def concept_loop(
    log: list[tuple[str, object]],
    source: InputPath,
    speaker: FakeSpeaker,
    reasoning: FakeReasoning,
    cfg: TurnLoopConfig | None = None,
    transport: LoggingTransport | None = None,
) -> TurnLoop:
    return TurnLoop(
        cfg if cfg is not None else concept_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        transport if transport is not None else CheckingTransport(log),
        reasoning,
        FakeRegistry(log),
    )


TYPED_TEXT = "why clip"
BAD_CLIENT_MESSAGES: list[dict[str, object]] = [
    {"type": "say", "text": ""},
    {"type": "say"},
    {"type": "shout", "text": "x"},
    {"type": "say", "text": "x" * 4001},
]


class ExclusiveSpeaker(FakeSpeaker):
    def __init__(self, log: list[tuple[str, object]], gate: asyncio.Event) -> None:
        super().__init__(log, gate=gate, hold_at=1)
        self.in_flight = False

    async def speak(self, chunks: AsyncIterator[Chunk], on_play: OnPlay) -> None:
        if self.in_flight:
            raise RuntimeError("an utterance is already in flight")
        self.in_flight = True
        try:
            await super().speak(chunks, on_play)
        finally:
            self.in_flight = False


async def test_a_typed_say_dispatches_a_turn_like_an_end_of_turn() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    hold = asyncio.Event()
    source = ScriptedSource([hold])
    loop = concept_loop(log, source, speaker, reasoning, transport=transport)
    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(source.blocked.wait(), HANG_GUARD_S)

    transport.handlers[0]({"type": "say", "text": CONCEPT_TEXT})
    await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
    hold.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert ("start_turn", CONCEPT_TEXT) in log
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert [without_seq(p) for p in sent(log, "transcript")] == [
        {"type": "transcript", "turn_id": "turn-1", "text": CONCEPT_TEXT}
    ]
    assert states(log) == ["thinking", "speaking", "listening"]
    await loop.aclose()


async def test_a_typed_say_while_speaking_interrupts_first() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), speaker.received, gate=gate, holds_at=3
    )
    transport = LoggingTransport(log)
    end = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=CONCEPT_TEXT), end])
    loop = concept_loop(log, source, speaker, reasoning, transport=transport)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    turn = turn_task()
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]]]

    transport.handlers[0]({"type": "say", "text": SECOND_TEXT})
    second = turn_task("turn-2")
    gate.set()
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn, second]), HANG_GUARD_S)

    assert turn.cancelled()
    assert log.index(("flush_playout", None)) < log.index(("abandon", "turn-1"))
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]], SPOKEN_CLAUSES]
    assert [prompt.user_text for prompt in reasoning.prompts] == [CONCEPT_TEXT, SECOND_TEXT]
    end.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await loop.aclose()


async def test_a_say_between_speech_start_and_end_of_turn_never_overlaps_the_spoken_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = ExclusiveSpeaker(log, hold)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    barge = asyncio.Event()
    typed = asyncio.Event()
    end = asyncio.Event()
    source = ScriptedSource(
        [
            EndOfTurn(text=CONCEPT_TEXT),
            barge,
            SpeechStarted(),
            typed,
            EndOfTurn(text=SECOND_TEXT),
            end,
        ]
    )
    loop = concept_loop(log, source, speaker, reasoning, transport=transport)
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        first = turn_task()
        speaker.held.clear()
        await pull_past(source, barge)
        await asyncio.wait_for(asyncio.wait([first]), HANG_GUARD_S)
        assert first.cancelled()

        transport.handlers[0]({"type": "say", "text": TYPED_TEXT})
        second = turn_task("turn-2")
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        speaker.held.clear()
        await pull_past(source, typed)
        third = turn_task("turn-3")
        await asyncio.wait_for(asyncio.wait([second]), HANG_GUARD_S)
        hold.set()
        await asyncio.wait_for(asyncio.wait([third]), HANG_GUARD_S)
        end.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert second.cancelled()
    assert ("abandon", "turn-2") in log
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]], [SPOKEN_CLAUSES[0]], SPOKEN_CLAUSES]
    assert not [line for line in session_messages(caplog) if line.startswith("turn.failed")]
    assert [prompt.user_text for prompt in reasoning.prompts] == [
        CONCEPT_TEXT,
        TYPED_TEXT,
        SECOND_TEXT,
    ]
    await loop.aclose()


async def test_an_invalid_client_message_is_logged_and_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    loop = concept_loop(log, ScriptedSource([]), speaker, reasoning, transport=transport)

    with caplog.at_level(logging.WARNING, logger="tutor.session"):
        for payload in BAD_CLIENT_MESSAGES:
            transport.handlers[0](payload)
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert session_messages(caplog) == [
        "client.message_rejected type=say",
        "client.message_rejected type=say",
        "client.message_rejected type=shout",
        "client.message_rejected type=say",
    ]
    assert reasoning.prompts == []
    assert sent(log, "transcript") == []
    assert ("flush_playout", None) not in log
    await loop.aclose()


async def test_a_scene_ready_with_no_scene_waiting_is_logged_and_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    loop = concept_loop(log, ScriptedSource([]), speaker, reasoning, transport=transport)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        transport.handlers[0](
            {"type": "scene.ready", "scene_id": "turn-4", "ok": True, "steps": 3, "error": ""}
        )
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert "scene.ready_unexpected scene_id=turn-4" in session_messages(caplog)
    assert reasoning.prompts == []
    assert sent(log, "transcript") == []
    assert ("flush_playout", None) not in log
    await loop.aclose()


async def test_a_typed_turn_and_a_spoken_turn_share_the_id_sequence() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    typed = asyncio.Event()
    source = ScriptedSource([typed, EndOfTurn(text=SECOND_TEXT)])
    loop = concept_loop(log, source, speaker, reasoning, transport=transport)
    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(source.blocked.wait(), HANG_GUARD_S)

    transport.handlers[0]({"type": "say", "text": CONCEPT_TEXT})
    first = turn_task()
    await asyncio.wait_for(asyncio.wait([first]), HANG_GUARD_S)
    typed.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert [(p["turn_id"], p["text"]) for p in sent(log, "transcript")] == [
        ("turn-1", CONCEPT_TEXT),
        ("turn-2", SECOND_TEXT),
    ]
    assert speaker.utterances == [SPOKEN_CLAUSES, SPOKEN_CLAUSES]
    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]
    await loop.aclose()


async def test_a_ready_report_for_no_waiting_scene_is_one_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=transport)
    (handler,) = transport.handlers

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        handler({"type": "scene.ready", "scene_id": "turn-9", "ok": True, "steps": 3, "error": ""})
        handler({"type": "scene.ready", "scene_id": "turn-9", "ok": True, "steps": 3})
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "scene.ready_unexpected scene_id=turn-9" in messages
    assert "client.message_rejected type=scene.ready" in messages
    assert speaker.utterances == [SPOKEN_CLAUSES]


PAGE_MESSAGES = [
    pytest.param(
        {
            "type": "lesson.ack",
            "epoch": 2,
            "barrier": 0,
            "cue_id": 9,
            "outcome": "fired",
            "reason": None,
            "scene_id": "ratio",
            "step": 2,
            "revision": 1,
        },
        "lesson.ack_ignored epoch=2 cue_id=9",
        id="ack-other-epoch",
    ),
    pytest.param(
        {
            "type": "lesson.synced",
            "epoch": 1,
            "barrier": 4,
            "scene_id": None,
            "step": 0,
            "revision": 0,
            "last_cue": 0,
        },
        "lesson.synced_ignored epoch=1 barrier=4",
        id="synced-no-barrier",
    ),
    pytest.param(
        {
            "type": "lesson.checkpoint",
            "epoch": 2,
            "scene_id": None,
            "version": 0,
            "step": 0,
            "revision": 0,
        },
        "lesson.checkpoint_ignored epoch=2",
        id="checkpoint-other-epoch",
    ),
]


@pytest.mark.parametrize(("message", "line"), PAGE_MESSAGES)
async def test_a_page_message_is_handled_apart_from_a_say(
    message: dict[str, object], line: str, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=transport)
    (handler,) = transport.handlers

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        handler(message)
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert line in session_messages(caplog)
    assert ("flush_playout", None) not in log
    assert [prompt.user_text for prompt in reasoning.prompts] == [CONCEPT_TEXT]
    assert speaker.utterances == [SPOKEN_CLAUSES]


async def test_an_ack_of_this_epoch_reaches_the_lesson_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=transport)
    (handler,) = transport.handlers
    ack = {
        "type": "lesson.ack",
        "epoch": 1,
        "barrier": 0,
        "cue_id": 9,
        "outcome": "fired",
        "reason": None,
        "scene_id": "ratio",
        "step": 2,
        "revision": 1,
    }

    with caplog.at_level(logging.INFO):
        handler(ack)
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    lesson = [record.getMessage() for record in caplog.records if record.name == "tutor.lesson"]
    assert "lesson.ack_unknown cue_id=9 outcome=fired" in lesson
    assert not any(line.startswith("lesson.ack_ignored") for line in session_messages(caplog))
    assert ("flush_playout", None) not in log
    assert [prompt.user_text for prompt in reasoning.prompts] == [CONCEPT_TEXT]
    assert speaker.utterances == [SPOKEN_CLAUSES]


async def test_a_checkpoint_from_the_attached_page_is_retained() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    transport = LoggingTransport(log)
    loop = concept_loop(log, ScriptedSource([]), speaker, reasoning, transport=transport)
    checkpoint = {
        "type": "lesson.checkpoint",
        "epoch": 1,
        "scene_id": None,
        "version": 0,
        "step": 0,
        "revision": 0,
    }

    transport.handlers[0](checkpoint)

    assert loop._lesson.checkpoint == LessonCheckpoint.model_validate(checkpoint)
    await loop.aclose()


LESSON = LessonPlan(
    profile="Knows policy gradients; new to clipping.",
    scenes=[
        Scene(
            id="ratio",
            title="The ratio",
            show="The probability ratio between the new and the old policy",
            steps=[
                Step(show="The old and the new policy over three actions"),
                Step(show="Their ratio at one action", ask="What is the ratio where they agree?"),
                Step(show="The ratio across all three actions"),
            ],
        ),
        Scene(
            id="clip",
            title="The clip",
            show="The clipped surrogate against the ratio, epsilon 0.2",
            steps=[
                Step(show="The ratio axis from 0.5 to 2.0"),
                Step(show="The clip band at 0.8 and 1.2"),
                Step(show="The flat regions outside the band"),
            ],
        ),
        Scene(
            id="epochs",
            title="Several epochs",
            show="Four epochs of updates on one batch",
            steps=[
                Step(show="One batch of samples"),
                Step(show="The ratio after each epoch"),
                Step(show="The ratio held inside the band"),
            ],
        ),
    ],
)
WATCHED = LessonPlan.model_validate(
    {
        **LESSON.model_dump(),
        "scenes": [
            {**scene.model_dump(), "steps": [{"show": step.show} for step in scene.steps]}
            for scene in LESSON.scenes
        ],
    }
)
SCRIPTS = {
    "ratio": [
        ScriptChunk(
            step=1,
            question=False,
            text="Here are the old and the new policy over three actions. Each bar is the "
            "chance one of them gives an action.",
        ),
        ScriptChunk(
            step=2,
            question=True,
            text="Divide the new chance by the old one at a single action. What is the ratio "
            "where they agree?",
        ),
        ScriptChunk(
            step=2,
            question=False,
            text="Where the two policies agree the ratio is exactly one. Above one the new "
            "policy favours that action more.",
        ),
        ScriptChunk(
            step=3,
            question=False,
            text="Now the ratio sits over all three actions. Two rise above one and one "
            "falls below it.",
        ),
    ],
    "clip": [
        ScriptChunk(
            step=1,
            question=False,
            text="This axis is the ratio, from one half to two. The objective is drawn against it.",
        ),
        ScriptChunk(
            step=2,
            question=False,
            text="The band runs from 0.8 to 1.2. Inside it the objective follows the ratio.",
        ),
        ScriptChunk(
            step=3,
            question=False,
            text="Outside the band the objective goes flat. A flat region gives no gradient "
            "to push further.",
        ),
    ],
    "epochs": [
        ScriptChunk(
            step=1,
            question=False,
            text="This is one batch of samples. Every epoch reuses the same batch.",
        ),
        ScriptChunk(
            step=2,
            question=False,
            text="Each epoch moves the ratio a little. Left alone it would drift far from one.",
        ),
        ScriptChunk(
            step=3,
            question=False,
            text="The clip holds the ratio inside the band. The policy cannot run away from "
            "the batch.",
        ),
    ],
}
WATCHED_SCRIPTS = {
    scene_id: [chunk for chunk in chunks if not chunk.question]
    for scene_id, chunks in SCRIPTS.items()
}
RESTORE = {
    "type": "lesson.checkpoint",
    "epoch": 1,
    "scene_id": "ratio",
    "version": 0,
    "step": 1,
    "revision": 3,
}


class FakePage(LoggingTransport):
    def __init__(
        self,
        log: list[tuple[str, object]],
        fires: bool = False,
        syncs: bool = True,
        ready: dict[str, str] | None = None,
    ) -> None:
        super().__init__(log)
        self.fires = fires
        self.syncs = syncs
        self.ready = ready
        self.held: dict[int, dict[str, object]] = {}
        self.scene_id: str | None = None
        self.step = 0
        self.revision = 0
        self.last_cue = 0
        self.syncs_seen: list[dict[str, object]] = []
        self.arrived: dict[str, asyncio.Event] = {}

    def send_json_nowait(self, payload: dict[str, object]) -> bool:
        self._log.append(("send_json", payload))
        self._receive(payload)
        return True

    async def send_json(self, payload: dict[str, object]) -> None:
        self._log.append(("send_json", payload))
        self._receive(payload)

    def arrival(self, kind: str) -> asyncio.Event:
        return self.arrived.setdefault(kind, asyncio.Event())

    async def state_where(self, match: Callable[[dict[str, object]], bool]) -> dict[str, object]:
        while True:
            for payload in sent(self._log, "lesson.state"):
                if match(payload):
                    return payload
            self.arrival("lesson.state").clear()
            await self.arrival("lesson.state").wait()

    async def seen(self, kind: str, count: int) -> None:
        while len(sent(self._log, kind)) < count:
            self.arrival(kind).clear()
            await self.arrival(kind).wait()

    def _receive(self, payload: dict[str, object]) -> None:
        kind = str(payload["type"])
        if kind == "lesson.cue":
            self.held[int(payload["cue_id"])] = payload
            if self.fires:
                asyncio.get_running_loop().call_soon(self.fire, int(payload["cue_id"]))
        elif kind == "lesson.sync":
            self.syncs_seen.append(payload)
            if self.syncs:
                asyncio.get_running_loop().call_soon(self.answer_sync)
        elif kind == "scene.push" and self.ready is not None:
            scene_id = str(payload["scene_id"])
            if scene_id in self.ready:
                error = self.ready[scene_id]
                steps = len(typing.cast(list[str], payload["steps"]))
                report = {
                    "type": "scene.ready",
                    "scene_id": scene_id,
                    "ok": error == "",
                    "steps": steps if error == "" else 0,
                    "error": error,
                }
                asyncio.get_running_loop().call_soon(self.reply, report)
        self.arrival(kind).set()

    def fire(self, cue_id: int) -> None:
        cue = self.held.pop(cue_id)
        tag = typing.cast(dict[str, object], cue["tag"])
        if tag["kind"] == "scene":
            self.scene_id, self.step = str(tag["scene_id"]), 1
        else:
            self.step = int(typing.cast(int, tag["n"]))
        self.revision += 1
        self.last_cue = cue_id
        self._ack(cue, "fired", None)

    def fail(self, cue_id: int, reason: str) -> None:
        self._ack(self.held.pop(cue_id), "failed", reason)

    def reply(self, message: dict[str, object]) -> None:
        for handler in self.handlers:
            handler(message)

    def _ack(self, cue: dict[str, object], outcome: str, reason: str | None) -> None:
        self.reply(
            {
                "type": "lesson.ack",
                "epoch": cue["epoch"],
                "barrier": cue["barrier"],
                "cue_id": cue["cue_id"],
                "outcome": outcome,
                "reason": reason,
                "scene_id": self.scene_id,
                "step": self.step,
                "revision": self.revision,
            }
        )

    def answer_sync(self) -> None:
        sync = self.syncs_seen[-1]
        for cue_id in sorted(self.held):
            self._ack(self.held.pop(cue_id), "dropped", "barrier")
        self.reply(
            {
                "type": "lesson.synced",
                "epoch": sync["epoch"],
                "barrier": sync["barrier"],
                "scene_id": self.scene_id,
                "step": self.step,
                "revision": self.revision,
                "last_cue": self.last_cue,
            }
        )


def page_ack(
    cue_id: int,
    outcome: str,
    reason: str | None,
    scene_id: str | None,
    step: int,
    revision: int,
    barrier: int = 0,
) -> dict[str, object]:
    return {
        "type": "lesson.ack",
        "epoch": 1,
        "barrier": barrier,
        "cue_id": cue_id,
        "outcome": outcome,
        "reason": reason,
        "scene_id": scene_id,
        "step": step,
        "revision": revision,
    }


TAGGED_DELTAS = [
    "<sce",
    "ne 1>The ratio compares two policies. ",
    "<set lr 0.9><Step 3>They agree ",
    "at one. <step 2>",
]


def spoken_texts(speaker: FakeSpeaker) -> list[str]:
    return [text for utterance in speaker.utterances for text in utterance]


class TimedSpeaker(FakeSpeaker):
    async def speak(self, chunks: AsyncIterator[Chunk], on_play: OnPlay) -> None:
        async def timed(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
            await on_play(chunk, 300, 900)

        await super().speak(chunks, timed)


class HeldCuePage(FakePage):
    def __init__(self, log: list[tuple[str, object]]) -> None:
        super().__init__(log, syncs=False)
        self.holding = asyncio.Event()
        self.release = asyncio.Event()

    async def send_json(self, payload: dict[str, object]) -> None:
        await super().send_json(payload)
        if payload["type"] == "lesson.cue":
            self.holding.set()
            await self.release.wait()


UNGROUNDED = "Look in setup.py for the flags."


def heard_texts(reasoning: FakeReasoning) -> list[str]:
    return [prompt.user_text.split("The learner now says: ")[-1] for prompt in reasoning.prompts]


def heard_at(log: list[tuple[str, object]], text: str) -> int | None:
    heard = f"The learner now says: {text}"
    calls = (n for n, (kind, body) in enumerate(log) if kind == "start_turn")
    return next((n for n in calls if str(log[n][1]).endswith(heard)), None)


async def test_the_page_is_attached_before_anything_else_is_sent() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    loop = concept_loop(
        log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=LoggingTransport(log)
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    first = next(payload for name, payload in log if name == "send_json")
    assert first == {"type": "lesson.attach", "epoch": 1, "seq": 1}
    assert sent(log, "lesson.attach") == [first]
    await loop.aclose()


async def test_a_barge_in_with_a_cue_unacknowledged_syncs_first_and_the_next_turn_waits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    gate = asyncio.Event()
    turns = [spoken_chunks(SPOKEN_DELTAS), spoken_chunks(["Back to the ratio then."])]
    reasoning = FakeReasoning(log, [], speaker.received, gate=gate, holds_at=3, turns=turns)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text=SECOND_TEXT)]
    )
    pace = HeldPace(log)
    loop = TurnLoop(
        concept_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace,
    )
    loop._lesson.adopt(LESSON)
    loop._lesson.scripts = dict(SCRIPTS)
    assert loop._lesson.scene_tag(1) is None
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    turn = turn_task()

    log.append(("barge", None))
    await pull_past(source, barge)
    gate.set()
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    (sync,) = sent(log, "lesson.sync")
    assert without_seq(sync) == {"type": "lesson.sync", "epoch": 1, "barrier": 1}
    steps = [name for name, _ in log]
    assert (
        steps.index("barge")
        < log.index(("send_json", sync))
        < steps.index("flush_playout")
        < steps.index("stream_closed")
        < steps.index("speak_cancelled")
    )
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    assert pace.waits == [LESSON_SYNC_TIMEOUT_S]

    page.arrival("transcript").clear()
    resume.set()
    await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
    assert heard_at(log, SECOND_TEXT) is None

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.answer_sync()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert heard_at(log, SECOND_TEXT) > log.index(("send_json", sync))
    assert speaker.utterances[1] == ["Back to the ratio then.", *PREPARED_OPENING]
    assert [entry.cue_id for entry in loop._lesson.sent] == [2]
    assert loop._lesson.acked == Cursor(scene=0, step=0)
    messages = session_messages(caplog)
    assert "lesson.synced epoch=1 barrier=1 scene_id=None step=0 revision=0 last_cue=0" in messages
    assert not any(message.startswith("lesson.sync_timeout") for message in messages)
    await loop.aclose()


async def test_a_page_that_never_syncs_releases_the_next_turn_at_the_bound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    turns = [spoken_chunks(SPOKEN_DELTAS), spoken_chunks(["Back to the ratio then."])]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text=SECOND_TEXT)]
    )
    pace = HeldPace(log)
    loop = TurnLoop(
        concept_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace,
    )
    loop._lesson.adopt(LESSON)
    loop._lesson.scripts = dict(SCRIPTS)
    assert loop._lesson.scene_tag(1) is None
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    turn = turn_task()
    await pull_past(source, barge)
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)

    page.arrival("transcript").clear()
    resume.set()
    await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
    assert heard_at(log, SECOND_TEXT) is None

    with caplog.at_level(logging.WARNING, logger="tutor.session"):
        pace.release_once()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert "lesson.sync_timeout epoch=1 barrier=1" in session_messages(caplog)
    assert [entry.cue_id for entry in loop._lesson.sent] == [2]
    assert loop._lesson.acked == Cursor(scene=0, step=0)
    assert heard_at(log, SECOND_TEXT) is not None
    await loop.aclose()


async def test_a_barge_in_with_nothing_in_flight_syncs_the_page_and_waits_for_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    turns = [spoken_chunks(SPOKEN_DELTAS), spoken_chunks(["Back to the ratio then."])]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    barge = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), EndOfTurn(text=SECOND_TEXT)]
    )
    pace = HeldPace(log)
    loop = TurnLoop(
        concept_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace,
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        turn = turn_task()
        barge.set()
        await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
        hold.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert turn.cancelled()
    (sync,) = sent(log, "lesson.sync")
    steps = [name for name, _ in log]
    assert log.index(("send_json", sync)) < steps.index("flush_playout")
    assert pace.waits == []
    assert log.index(("start_turn", SECOND_TEXT)) > log.index(("send_json", sync))
    assert page.syncs_seen == [sync]
    assert "lesson.sync epoch=1 barrier=1 waiting=False reason=barge_in" in session_messages(caplog)
    await loop.aclose()


async def test_a_second_barge_in_before_the_page_syncs_waits_for_the_newer_barrier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    turns = [spoken_chunks(SPOKEN_DELTAS), spoken_chunks(["Back to the ratio then."])]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    barge = asyncio.Event()
    again = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [
            EndOfTurn(text=CONCEPT_TEXT),
            barge,
            SpeechStarted(),
            again,
            SpeechStarted(),
            resume,
            EndOfTurn(text=SECOND_TEXT),
        ]
    )
    loop = TurnLoop(
        concept_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        HeldPace(log),
    )
    loop._lesson.adopt(LESSON)
    loop._lesson.scripts = dict(SCRIPTS)
    assert loop._lesson.scene_tag(1) is None
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    turn = turn_task()
    await pull_past(source, barge)
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    await pull_past(source, again)
    assert [without_seq(sync)["barrier"] for sync in sent(log, "lesson.sync")] == [1, 2]

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply(
            {
                "type": "lesson.synced",
                "epoch": 1,
                "barrier": 1,
                "scene_id": None,
                "step": 0,
                "revision": 0,
                "last_cue": 0,
            }
        )
        assert [entry.cue_id for entry in loop._lesson.sent] == [1]
        page.arrival("transcript").clear()
        resume.set()
        await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
        assert heard_at(log, SECOND_TEXT) is None
        page.answer_sync()
        await asyncio.wait_for(running, HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "lesson.synced_ignored epoch=1 barrier=1" in messages
    assert "lesson.synced epoch=1 barrier=2 scene_id=None step=0 revision=0 last_cue=0" in messages
    assert heard_at(log, SECOND_TEXT) is not None
    assert [entry.cue_id for entry in loop._lesson.sent] == [2]
    await loop.aclose()


async def test_a_failed_ack_is_answered_with_a_sync_before_the_next_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, syncs=False)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    after = asyncio.Event()
    done = asyncio.Event()
    source = ScriptedSource([after, EndOfTurn(text=CONCEPT_TEXT), done])
    loop = concept_loop(log, source, speaker, reasoning, transport=page)
    loop._lesson.adopt(LESSON)
    loop._lesson.scripts = dict(SCRIPTS)
    assert loop._lesson.scene_tag(1) is None
    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(page.arrival("lesson.attach").wait(), HANG_GUARD_S)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply(page_ack(1, "failed", "runtime", None, 0, 0))
        (sync,) = sent(log, "lesson.sync")
        assert log[-1] == ("send_json", sync)
        assert without_seq(sync) == {"type": "lesson.sync", "epoch": 1, "barrier": 1}
        assert loop._lesson.resync and loop._lesson.sent == []

        page.arrival("transcript").clear()
        await pull_past(source, after)
        await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
        assert heard_at(log, CONCEPT_TEXT) is None
        page.answer_sync()
        await asyncio.wait_for(asyncio.wait([turn_task()]), HANG_GUARD_S)

    assert heard_at(log, CONCEPT_TEXT) is not None
    assert not loop._lesson.resync
    messages = session_messages(caplog)
    assert "lesson.ack cue_id=1 outcome=failed reason=runtime" in messages
    assert "lesson.sync epoch=1 barrier=1 waiting=True reason=failed" in messages
    done.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await loop.aclose()


async def test_a_restore_checkpoint_is_answered_with_one_sync_until_the_page_answers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, syncs=False)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    loop = concept_loop(log, ScriptedSource([]), speaker, reasoning, transport=page)
    loop._lesson.adopt(LESSON)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply(RESTORE)
        page.reply({**RESTORE, "revision": 4})

    (sync,) = sent(log, "lesson.sync")
    assert without_seq(sync) == {"type": "lesson.sync", "epoch": 1, "barrier": 1}
    assert loop._lesson.resync
    assert "lesson.sync epoch=1 barrier=1 waiting=True reason=restore" in session_messages(caplog)
    page.scene_id, page.step, page.revision = "ratio", 1, 4
    page.answer_sync()
    assert not loop._lesson.resync
    assert loop._lesson.acked == Cursor(scene=1, step=1) and loop._lesson.revision == 4
    await loop.aclose()


async def test_a_barge_in_while_a_sync_is_still_owed_waits_for_the_page(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, syncs=False)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    pace = HeldPace(log)
    loop = TurnLoop(
        concept_cfg(),
        ScriptedSource([]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace,
    )
    loop._lesson.adopt(LESSON)
    page.reply(RESTORE)
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    (bound,) = [task for task in asyncio.all_tasks() if task.get_name() == "lesson-sync-bound"]
    pace.release_once()
    await asyncio.wait_for(asyncio.wait([bound]), HANG_GUARD_S)
    assert loop._lesson.resync

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply({"type": "say", "text": CONCEPT_TEXT})
        await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
        assert heard_at(log, CONCEPT_TEXT) is None
        page.scene_id, page.step, page.revision = "ratio", 1, 3
        page.answer_sync()
        await asyncio.wait_for(asyncio.wait([turn_task()]), HANG_GUARD_S)

    assert "lesson.sync epoch=1 barrier=2 waiting=True reason=barge_in" in session_messages(caplog)
    assert heard_at(log, CONCEPT_TEXT) is not None
    assert not loop._lesson.resync
    await loop.aclose()


async def test_a_speculation_waits_for_the_page_to_answer_an_owed_sync() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, syncs=False)
    gate = asyncio.Event()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received, gate=gate)
    restored = asyncio.Event()
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            restored,
            PartialTranscript(text=CONCEPT_PARTIAL),
            endpoint,
            EndOfTurn(text=CONCEPT_TEXT),
            speaker.finished,
        ]
    )
    loop = TurnLoop(
        concept_cfg().model_copy(update={"speculative_reasoning": True}),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        HeldPace(log),
    )
    loop._lesson.adopt(LESSON)
    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(page.arrival("lesson.attach").wait(), HANG_GUARD_S)
    page.reply({**RESTORE, "scene_id": "epochs", "step": 3})
    assert [without_seq(sync) for sync in sent(log, "lesson.sync")] == [
        {"type": "lesson.sync", "epoch": 1, "barrier": 1}
    ]

    await pull_past(source, restored)
    assert ("open_turn", TURN_TASK) not in log
    assert reasoning.prompts == []

    log.append(("synced", None))
    page.scene_id, page.step, page.revision = "epochs", 3, 3
    page.answer_sync()
    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    assert log.index(("open_turn", TURN_TASK)) > log.index(("synced", None))

    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert [entry for entry in log if entry[0] == "start_turn"] == [("start_turn", CONCEPT_PARTIAL)]
    assert [entry for entry in log if entry[0] == "open_turn"] == [("open_turn", TURN_TASK)]
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert not loop._lesson.resync
    await loop.aclose()


async def test_tags_never_reach_the_speaker_and_every_tag_drops_on_the_voice_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, spoken_chunks(TAGGED_DELTAS), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)
    caplog.set_level(logging.INFO, logger="tutor.lesson")

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    spoken = spoken_texts(speaker)
    assert spoken and not any("<" in text or ">" in text for text in spoken)
    assert "They agree at one." in " ".join(spoken)
    messages = session_messages(caplog)
    assert sent(log, "lesson.cue") == []
    assert loop._lesson.sent == []
    assert [message for message in messages if message.startswith("tag.dropped ")] == [
        "tag.dropped turn_id=turn-1 kind=scene reason=voice chars=7",
        "tag.dropped turn_id=turn-1 kind=set reason=voice chars=10",
        "tag.dropped turn_id=turn-1 kind=step reason=voice chars=6",
        "tag.dropped turn_id=turn-1 kind=step reason=voice chars=6",
    ]
    lesson = [record.getMessage() for record in caplog.records if record.name == "tutor.lesson"]
    assert not [message for message in lesson if message.startswith("scene.dropped ")]
    assert not [message for message in lesson if message.startswith("step.dropped ")]


@pytest.mark.parametrize("done", [False, True], ids=["no_plan", "lesson_done"])
async def test_a_well_formed_voice_tag_is_dropped_on_the_voice_path(
    caplog: pytest.LogCaptureFixture, done: bool
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    deltas = ["<scene 1>The ratio compares two policies. ", "<step 2>They agree at one."]
    reasoning = FakeReasoning(log, spoken_chunks(deltas), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)
    if done:
        loop._lesson.adopt(LESSON)
        loop._lesson.acked = Cursor(scene=3, step=3)
    caplog.set_level(logging.INFO, logger="tutor.lesson")

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [["The ratio compares two policies.", "They agree at one."]]
    assert sent(log, "lesson.cue") == [] and loop._lesson.sent == []
    assert [m for m in session_messages(caplog) if m.startswith("tag.dropped ")] == [
        "tag.dropped turn_id=turn-1 kind=scene reason=voice chars=7",
        "tag.dropped turn_id=turn-1 kind=step reason=voice chars=6",
    ]
    lesson = [record.getMessage() for record in caplog.records if record.name == "tutor.lesson"]
    assert not [m for m in lesson if m.startswith(("scene.dropped ", "step.dropped "))]


def watched_loop(log: list[tuple[str, object]], speaker: FakeSpeaker, page: FakePage) -> TurnLoop:
    reasoning = FakeReasoning(log, [], speaker.received, turns=[spoken_chunks(["go_on"])])
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)
    loop._lesson.adopt(WATCHED)
    loop._lesson.scripts = dict(WATCHED_SCRIPTS)
    return loop


async def test_each_cue_goes_out_right_after_its_chunks_caption() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    loop = watched_loop(log, speaker, page)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    cues = sent(log, "lesson.cue")
    assert [cue["tag"] for cue in cues] == [scene_cue(1, "ratio"), step_cue(2), step_cue(3)]
    assert [
        (c["cue_id"], c["scene_id"], c["revision"], c["barrier"], c["epoch"]) for c in cues
    ] == [
        (1, None, 0, 0, 1),
        (2, "ratio", 1, 0, 1),
        (3, "ratio", 2, 0, 1),
    ]
    captions = sent(log, "caption")
    for cue, step in zip(cues, (1, 2, 3)):
        (carrier,) = [c for c in captions if c["text"] == said("ratio", step)[0]]
        assert log.index(("send_json", cue)) == log.index(("send_json", carrier)) + 1
    assert cues[0]["chunk_id"] < cues[1]["chunk_id"] < cues[2]["chunk_id"]
    assert [entry.cue_id for entry in loop._lesson.sent] == [1, 2, 3]


async def test_a_fired_ack_moves_the_cursor_and_logs_the_position(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    loop = watched_loop(log, speaker, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert loop._lesson.acked == Cursor(scene=1, step=3)
    assert loop._lesson.sent == []
    messages = session_messages(caplog)
    assert "cursor.scene scene_id=ratio n=1 cue_id=1" in messages
    assert "cursor.step scene_id=ratio n=2 cue_id=2" in messages
    assert "cursor.step scene_id=ratio n=3 cue_id=3" in messages


async def test_a_fired_ack_the_state_refuses_moves_nothing_and_logs_no_cursor_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    loop = concept_loop(log, ScriptedSource([]), speaker, reasoning, transport=page)
    loop._lesson.adopt(WATCHED)
    assert loop._lesson.scene_tag(1) is None
    assert loop._lesson.step_tag(2) is None

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply(page_ack(2, "fired", None, "ratio", 2, 2))
        page.reply(page_ack(1, "fired", None, "gone", 1, 1))

    assert not any(message.startswith("cursor.") for message in session_messages(caplog))
    assert [entry.cue_id for entry in loop._lesson.sent] == [1, 2]
    assert loop._lesson.acked == Cursor()
    await loop.aclose()


async def test_a_trailing_cue_takes_the_last_chunks_timing_and_no_chunk_discards_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = TimedSpeaker(log)
    page = FakePage(log)
    turns = [spoken_chunks(["go_on"]), spoken_chunks(["go_on"])]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    loop = TurnLoop(
        concept_cfg(),
        SerialSource([CONCEPT_TEXT, "go on"]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        TurnRegistry(),
        FakeClock(),
    )
    loop._lesson.adopt(WATCHED)
    unheard = [ScriptChunk(step=n, question=False, text=UNGROUNDED) for n in (1, 2, 3)]
    loop._lesson.scripts = {**with_chunk("ratio", 3, False, UNGROUNDED), "clip": unheard}

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    opening, _, trailing = sent(log, "lesson.cue")
    assert trailing["tag"] == step_cue(3)
    (last,) = [chunk for chunk in speaker.chunks if chunk.text == said("ratio", 2)[-1]]
    assert trailing["chunk_id"] == last.id
    assert (trailing["lead_ms"], trailing["audio_ms"]) == (950, 0)
    assert (opening["lead_ms"], opening["audio_ms"]) == (300, 900)
    assert "tag.discarded turn_id=turn-2 count=3 reason=no_chunk" in session_messages(caplog)
    assert [entry.cue_id for entry in loop._lesson.sent] == [1, 2, 3]


async def test_a_tag_in_the_follow_up_round_is_stripped_and_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [SEARCH_CALL],
        speaker.received,
        follow_up=spoken_chunks(["<scene 1>Both call sites ", "drain it."]),
    )
    loop = TurnLoop(
        config(tmp_path),
        SerialSource([USER_TEXT]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert sent(log, "lesson.cue") == []
    assert "tag.dropped turn_id=turn-1 kind=scene reason=voice chars=7" in session_messages(caplog)
    assert not any("<" in text for text in spoken_texts(speaker))


async def test_a_cue_checked_after_a_failed_ack_in_the_same_reply_is_discarded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    loop = watched_loop(log, speaker, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        (opening,) = sent(log, "lesson.cue")
        page.fail(1, "runtime")
        (sync,) = sent(log, "lesson.sync")
        assert log[-1] == ("send_json", sync)
        hold.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert sent(log, "lesson.cue") == [opening]
    assert loop._lesson.sent == [] and loop._lesson.resync
    assert "tag.discarded turn_id=turn-1 count=1 reason=barrier" in session_messages(caplog)
    await loop.aclose()


async def test_a_barge_in_after_a_cue_went_out_drops_it_at_the_page_and_the_next_turn_waits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    turns = [spoken_chunks(["go_on"]), spoken_chunks(["Back to the ratio then."])]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text=SECOND_TEXT)]
    )
    pace = HeldPace(log)
    loop = TurnLoop(
        concept_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
        pace,
    )
    loop._lesson.adopt(LESSON)
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    (opening,) = sent(log, "lesson.cue")
    turn = turn_task()
    await pull_past(source, barge)
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert sent(log, "lesson.cue") == [opening]
    await asyncio.wait_for(pace.held.wait(), HANG_GUARD_S)
    page.arrival("transcript").clear()
    resume.set()
    await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
    assert heard_at(log, SECOND_TEXT) is None

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.answer_sync()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert speaker.utterances[1] == ["Back to the ratio then.", *PREPARED_OPENING]
    assert [entry.cue_id for entry in loop._lesson.sent] == [2]
    assert loop._lesson.acked == Cursor(scene=0, step=0)
    assert "lesson.ack cue_id=1 outcome=dropped reason=barrier" in session_messages(caplog)
    await loop.aclose()


async def test_a_cue_before_a_withheld_clause_rides_the_next_admitted_clause(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, turns=[spoken_chunks(["go_on"])])
    loop = TurnLoop(
        concept_cfg(),
        SerialSource([CONCEPT_TEXT]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        TurnRegistry(),
    )
    loop._lesson.adopt(LESSON)
    loop._lesson.scripts = with_chunk("ratio", 1, False, f"{UNGROUNDED} They agree at one.")

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [["They agree at one.", *said("ratio", 2, True)]]
    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == scene_cue(1, "ratio")
    (carrier,) = [chunk for chunk in speaker.chunks if chunk.text == "They agree at one."]
    assert cue["chunk_id"] == carrier.id
    (caption,) = [caption for caption in sent(log, "caption") if caption["text"] == carrier.text]
    assert log.index(("send_json", cue)) == log.index(("send_json", caption)) + 1


async def test_a_failed_ack_while_a_cue_push_waits_discards_the_next_marker_of_that_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = HeldCuePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, turns=[spoken_chunks(["go_on"])])
    loop = TurnLoop(
        concept_cfg(),
        SerialSource([CONCEPT_TEXT]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        TurnRegistry(),
        FakeClock(),
        HeldPace(),
    )
    loop._lesson.adopt(WATCHED)
    loop._lesson.scripts = with_chunk("ratio", 1, False, UNGROUNDED)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(page.holding.wait(), HANG_GUARD_S)
        page.fail(1, "runtime")
        page.release.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == scene_cue(1, "ratio")
    (carrier,) = [chunk for chunk in speaker.chunks if chunk.id == cue["chunk_id"]]
    assert carrier.text == said("ratio", 2)[0]
    assert [without_seq(sync) for sync in sent(log, "lesson.sync")] == [
        {"type": "lesson.sync", "epoch": 1, "barrier": 1}
    ]
    discards = [m for m in session_messages(caplog) if m.startswith("tag.discarded ")]
    assert discards == ["tag.discarded turn_id=turn-1 count=1 reason=barrier"] * 2
    assert loop._lesson.sent == []
    await loop.aclose()


async def test_the_voice_prompt_is_the_lesson_block_under_the_system_text() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    (prompt,) = reasoning.prompts
    assert prompt.system.startswith(f"{SYSTEM}\n\nSubject: PPO\n\n")
    assert prompt.system.endswith(lesson_block(LessonState(), True))
    assert "<outcome>" not in prompt.system and "<visual>" not in prompt.system


def planned_call(plan: LessonPlan) -> list[TurnChunk]:
    return [
        TurnChunk(
            kind="tool_call",
            text=plan.model_dump_json(),
            tool_call_id="call-plan",
            tool_name=PLAN_TOOL,
        )
    ]


def lesson_cfg() -> TurnLoopConfig:
    return concept_cfg().model_copy(update={"planned": True})


def with_step_show(plan: LessonPlan, scene: int, step: int, show: str) -> LessonPlan:
    body = plan.model_dump()
    body["scenes"][scene]["steps"][step]["show"] = show
    return LessonPlan.model_validate(body)


def planner_task() -> asyncio.Task[None]:
    (task,) = [task for task in asyncio.all_tasks() if task.get_name() == "lesson-planner"]
    return task


def opened() -> asyncio.Event:
    gate = asyncio.Event()
    gate.set()
    return gate


async def test_a_valid_plan_at_connect_opens_the_lesson_once_with_no_learner_speech() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    loop = TurnLoop(
        lesson_cfg(),
        ScriptedSource([stop]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
    )
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.prompts == []
    assert speaker.utterances == [PREPARED_OPENING]
    assert sent(log, "transcript") == []
    assert log.index(("plan", None)) < log.index(("send_json", sent(log, "caption")[0]))
    first_state = sent(log, "lesson.state")[0]
    assert without_seq(first_state) == {
        "type": "lesson.state",
        "scenes": [
            {"id": "ratio", "title": "The ratio", "status": "planned"},
            {"id": "clip", "title": "The clip", "status": "planned"},
            {"id": "epochs", "title": "Several epochs", "status": "planned"},
        ],
        "current": None,
    }
    assert len(reasoning.plan_prompts) == 1
    assert reasoning.plan_efforts == [lesson_cfg().planner_effort]


async def test_a_learner_who_speaks_first_is_turn_one_and_waits_for_the_plan() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plans=[planned_call(LESSON)],
        plan_gates=[held],
    )
    loop = concept_loop(
        log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
    assert heard_at(log, CONCEPT_TEXT) is None
    held.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    (live,) = reasoning.prompts
    assert live.system == LIVE_PROMPT
    assert heard_texts(reasoning) == [CONCEPT_TEXT]
    assert log.index(("plan", None)) < heard_at(log, CONCEPT_TEXT)
    assert speaker.utterances == [[*SPOKEN_CLAUSES, *PREPARED_OPENING]]


async def test_two_failed_connect_calls_open_once_in_words(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.plan_prompts) == 2
    assert "planner.failed stage=connect reason=empty" in session_messages(caplog)
    (voice,) = reasoning.prompts
    assert voice.user_text == OPENING_TEXT
    assert voice.system.endswith(NO_PLAN)
    assert sent(log, "lesson.state") == []


async def test_a_connect_plan_that_names_a_source_path_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    pathed = with_step_show(LESSON, 0, 1, "The Connection class in tutor/transport.py line 112")
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plans=[planned_call(pathed), planned_call(pathed)],
    )
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.plan_prompts) == 2
    assert "planner.failed stage=connect reason=position" in session_messages(caplog)
    assert loop._lesson.plan is None and sent(log, "lesson.state") == []
    (voice,) = reasoning.prompts
    assert voice.system.endswith(NO_PLAN)


async def test_a_reasoning_error_in_the_connect_call_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plan_raises=RateLimited(RATE_LIMIT_DETAIL),
    )
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert len(reasoning.plan_prompts) == 2
    assert messages.count("planner.call_failed stage=connect error=RateLimited") == 2
    assert "planner.failed stage=connect reason=call" in messages
    assert not any(RATE_LIMIT_DETAIL in message for message in messages)
    assert not any(message.startswith("planner.task_failed") for message in messages)
    (voice,) = reasoning.prompts
    assert voice.user_text == OPENING_TEXT


async def test_a_fired_scene_cue_publishes_the_list_with_its_scene_current() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    state = await asyncio.wait_for(
        page.state_where(lambda payload: payload["current"] == "ratio"), HANG_GUARD_S
    )
    (fired,) = sent(log, "lesson.cue")
    assert log.index(("send_json", fired)) < log.index(("send_json", state))
    assert [row["status"] for row in state["scenes"]][1:] == ["planned", "planned"]
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)
    assert loop._lesson.acked == Cursor(scene=1, step=1)


async def test_a_barge_in_never_cancels_the_planner() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on"])],
        plans=[planned_call(LESSON)],
        plan_gates=[held],
    )
    barge = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([barge, SpeechStarted(), EndOfTurn(text=CONCEPT_TEXT), stop])
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[0].held.wait(), HANG_GUARD_S)
    planner = planner_task()
    await pull_past(source, barge)
    assert not planner.cancelling()
    assert sent(log, "lesson.sync") != []

    held.set()
    await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert not planner.cancelled()
    assert loop._lesson.plan == LESSON
    assert heard_texts(reasoning) == [CONCEPT_TEXT]


async def test_a_plan_that_lands_while_the_learner_speaks_opens_no_turn() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plans=[planned_call(LESSON)],
        plan_gates=[held],
    )
    heard = asyncio.Event()
    source = ScriptedSource(
        [SpeechStarted(), heard, EndOfTurn(text=CONCEPT_TEXT), speaker.finished]
    )
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(source.blocked.wait(), HANG_GUARD_S)
    planner = planner_task()
    held.set()
    await asyncio.wait_for(asyncio.wait([planner]), HANG_GUARD_S)
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("turn-")]
    await pull_past(source, heard)
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert loop._lesson.plan == LESSON
    assert heard_texts(reasoning) == [CONCEPT_TEXT]
    assert speaker.utterances == [[*SPOKEN_CLAUSES, *PREPARED_OPENING]]


async def test_closing_during_the_connect_call_dispatches_no_opening() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plans=[planned_call(LESSON)],
        plan_gates=[held],
    )
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[0].held.wait(), HANG_GUARD_S)
    planner = planner_task()
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert planner.cancelled()
    assert reasoning.prompts == []
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith(("lesson-", "turn-"))
    ]


SECOND_CONCEPT_TEXT = "and the ratio"


def with_title(plan: LessonPlan, scene: int, title: str) -> LessonPlan:
    body = plan.model_dump()
    body["scenes"][scene]["title"] = title
    return LessonPlan.model_validate(body)


def with_fourth_scene(plan: LessonPlan) -> LessonPlan:
    body = plan.model_dump()
    body["scenes"].append(
        {
            "id": "trust",
            "title": "The trust region",
            "show": "The clip as a region the update stays inside",
            "steps": [
                {"show": "The old policy as a point", "ask": ""},
                {"show": "The region around it", "ask": ""},
                {"show": "An update that stays inside", "ask": ""},
            ],
        }
    )
    return LessonPlan.model_validate(body)


RETITLED = with_title(LESSON, 2, "Epochs on one batch")


async def test_the_first_answer_sends_the_planner_in_once() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plans=[planned_call(LESSON), planned_call(RETITLED)],
    )
    first = asyncio.Event()
    second = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource(
        [first, EndOfTurn(text=CONCEPT_TEXT), second, EndOfTurn(text=SECOND_CONCEPT_TEXT), stop]
    )
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await opening_done(page)
    await turn_after(source, first, "turn-2")
    await asyncio.wait_for(
        page.state_where(lambda payload: payload["scenes"][2]["title"] == "Epochs on one batch"),
        HANG_GUARD_S,
    )
    await turn_after(source, second, "turn-3")
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.plan_prompts) == 2
    assert reasoning.plan_prompts[1].history == [
        Message(role="user", content=OPENING_TEXT),
        Message(role="assistant", content=" ".join(PREPARED_OPENING)),
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]
    assert heard_texts(reasoning) == [CONCEPT_TEXT, SECOND_CONCEPT_TEXT]
    assert loop._lesson.plan == RETITLED


async def test_an_interrupted_opening_leaves_the_next_turn_to_open_the_lesson() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        plans=[planned_call(LESSON), planned_call(RETITLED)],
    )
    barge = asyncio.Event()
    answered = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource(
        [
            barge,
            SpeechStarted(),
            EndOfTurn(text=CONCEPT_TEXT),
            answered,
            EndOfTurn(text=SECOND_CONCEPT_TEXT),
            stop,
        ]
    )
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    opening = turn_task("turn-1")
    await pull_past(source, barge)
    hold.set()
    await asyncio.wait_for(asyncio.wait([opening, turn_task("turn-2")]), HANG_GUARD_S)
    assert opening.cancelled()
    assert len(reasoning.plan_prompts) == 1
    await pull_past(source, answered)
    await asyncio.wait_for(asyncio.wait([turn_task("turn-3")]), HANG_GUARD_S)
    await asyncio.wait_for(
        page.state_where(lambda payload: payload["scenes"][2]["title"] == "Epochs on one batch"),
        HANG_GUARD_S,
    )
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert heard_texts(reasoning)[0] == CONCEPT_TEXT
    assert reasoning.prompts[0].system == LIVE_PROMPT
    assert speaker.utterances[1] == [*SPOKEN_CLAUSES, *PREPARED_OPENING]
    assert len(reasoning.plan_prompts) == 2
    assert reasoning.plan_prompts[1].history[-2:] == [
        Message(role="user", content=SECOND_CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]


async def test_a_fired_scene_ack_past_scene_one_sends_the_planner_in(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on"])],
        plans=[planned_call(WATCHED), planned_call(with_title(WATCHED, 2, "Epochs on one batch"))],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="go on"), stop])
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(WATCHED_SCRIPTS)
    loop._lesson.first_answer_done = True

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await turn_after(source, first, "turn-2")
        await asyncio.wait_for(
            page.state_where(
                lambda payload: payload["scenes"][2]["title"] == "Epochs on one batch"
            ),
            HANG_GUARD_S,
        )
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.plan_prompts) == 2
    assert "Protected: ratio, clip" in reasoning.plan_prompts[1].user_text
    messages = session_messages(caplog)
    assert "planner.accepted stage=boundary scenes=3" in messages
    assert not any("stage=first_answer" in message for message in messages)
    assert "clip" in [payload["current"] for payload in sent(log, "lesson.state")]


async def test_a_rerun_asked_for_while_one_runs_waits_in_one_slot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on"]), spoken_chunks(["go_on"])],
        plans=[
            planned_call(WATCHED),
            planned_call(WATCHED),
            planned_call(with_fourth_scene(WATCHED)),
        ],
        plan_gates=[opened(), held],
    )
    answer = asyncio.Event()
    again = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([answer, EndOfTurn(text="go on"), again, EndOfTurn(text="go on"), stop])
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(WATCHED_SCRIPTS)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        reasoning.planning.clear()
        await pull_past(source, answer)
        await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.plan_streams[1].held.wait(), HANG_GUARD_S)
        boundary = planner_task()
        await asyncio.wait_for(asyncio.wait([turn_task("turn-2")]), HANG_GUARD_S)
        await turn_after(source, again, "turn-3")
        await asyncio.wait_for(
            page.state_where(lambda payload: payload["current"] == "epochs"), HANG_GUARD_S
        )
        assert len(reasoning.plan_prompts) == 2
        held.set()
        await asyncio.wait_for(asyncio.wait([boundary]), HANG_GUARD_S)
        await asyncio.wait_for(
            page.state_where(lambda payload: len(payload["scenes"]) == 4), HANG_GUARD_S
        )
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert messages.count("planner.queued stage=boundary") == 1
    assert messages.count("planner.queued stage=first_answer") == 1
    assert len(reasoning.plan_prompts) == 3
    assert [m.split(" ms=")[0] for m in messages if m.startswith("planner.result")] == [
        "planner.result stage=connect",
        "planner.result stage=boundary",
        "planner.result stage=boundary",
    ]


async def test_a_rerun_that_changes_a_protected_scene_gets_the_drawn_scene_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    appended = with_fourth_scene(LESSON)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[
            planned_call(LESSON),
            planned_call(with_title(LESSON, 0, "The probability ratio")),
            planned_call(appended),
        ],
    )
    answer = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([answer, EndOfTurn(text="one"), stop])
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = dict(SCRIPTS)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await pull_past(source, answer)
        await asyncio.wait_for(
            page.state_where(lambda payload: len(payload["scenes"]) == 4), HANG_GUARD_S
        )
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "planner.accepted stage=boundary scenes=3" in messages
    assert "planner.accepted stage=first_answer scenes=4" in messages
    assert loop._lesson.plan == appended


async def test_a_rerun_cannot_rewrite_the_scene_the_scripter_protected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    rerun = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[planned_call(LESSON), planned_call(RETITLED), planned_call(RETITLED)],
        plan_gates=[opened(), rerun],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)
    loop._lesson.scripts = {scene_id: SCRIPTS[scene_id] for scene_id in ("ratio", "clip")}
    caplog.set_level(logging.INFO, logger="tutor.lesson")

    with (
        caplog.at_level(logging.INFO, logger="tutor.session"),
        line_seen("planner.accepted stage=boundary") as boundary,
    ):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        reasoning.planning.clear()
        await turn_after(source, first, "turn-2")
        await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.plan_streams[1].held.wait(), HANG_GUARD_S)
        assert "epochs" not in loop._lesson.scripting
        rerun_task = planner_task()
        for cue_id in range(1, 7):
            page.fire(cue_id)
        await asyncio.wait_for(reasoning.scripted(1), HANG_GUARD_S)
        assert loop._lesson.scripting == {"epochs"}
        rerun.set()
        await asyncio.wait_for(asyncio.wait([rerun_task]), HANG_GUARD_S)
        await asyncio.wait_for(boundary.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    protected = [
        [line for line in prompt.user_text.splitlines() if line.startswith("Protected: ")]
        for prompt in reasoning.plan_prompts[1:]
    ]
    assert protected == [["Protected: ratio, clip"], ["Protected: ratio, clip, epochs"]]
    assert "lesson.prefix_touched reason=stale_prefix" in caplog.messages
    messages = session_messages(caplog)
    assert "planner.accepted stage=first_answer scenes=3" in messages
    assert loop._lesson.plan == LESSON


async def test_after_two_failed_connect_calls_the_first_answer_recovers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), speaker.received, plans=[[], [], planned_call(LESSON)]
    )
    answer = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([answer, EndOfTurn(text=CONCEPT_TEXT), stop])
    loop = concept_loop(log, source, speaker, reasoning, cfg=lesson_cfg(), transport=page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(asyncio.wait([turn_task("turn-1")]), HANG_GUARD_S)
        assert sent(log, "lesson.state") == []
        await pull_past(source, answer)
        await asyncio.wait_for(page.state_where(lambda payload: True), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "planner.failed stage=connect reason=empty" in messages
    assert "lesson.planned scenes=3" in messages
    assert loop._lesson.plan == LESSON
    assert len(reasoning.plan_prompts) == 3
    assert "Protected:" not in reasoning.plan_prompts[2].user_text


class LineSeen(logging.Handler):
    def __init__(self, line: str) -> None:
        super().__init__()
        self._line = line
        self.seen = asyncio.Event()

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith(self._line):
            self.seen.set()


@contextlib.contextmanager
def line_seen(line: str, name: str = "tutor.session") -> Iterator[asyncio.Event]:
    handler = LineSeen(line)
    watched = logging.getLogger(name)
    level = watched.level
    watched.setLevel(logging.INFO)
    watched.addHandler(handler)
    try:
        yield handler.seen
    finally:
        watched.removeHandler(handler)
        watched.setLevel(level)


def said_call(lines: list[str]) -> TurnChunk:
    body = {"html": "<!doctype html><p>scene</p>", "steps": lines}
    return TurnChunk(
        kind="tool_call", text=json.dumps(body), tool_call_id="call-scene", tool_name=SCENE_TOOL
    )


def draft_call(steps: int) -> TurnChunk:
    return said_call([f"say {n}" for n in range(1, steps + 1)])


def pushed(log: list[tuple[str, object]]) -> list[str]:
    return [str(payload["scene_id"]) for payload in sent(log, "scene.push")]


def builds(reasoning: FakeReasoning, title: str) -> list[TurnPrompt]:
    return [
        prompt for prompt in reasoning.build_prompts if f"\nTitle: {title}\n" in prompt.user_text
    ]


def builder_task() -> asyncio.Task[None]:
    (task,) = [task for task in asyncio.all_tasks() if task.get_name() == "lesson-builder"]
    return task


def status_is(scene_id: str, status: str) -> Callable[[dict[str, object]], bool]:
    def match(payload: dict[str, object]) -> bool:
        rows = typing.cast(list[dict[str, str]], payload["scenes"])
        return any(row["id"] == scene_id and row["status"] == status for row in rows)

    return match


READY = {"ratio": "", "clip": "", "epochs": ""}


def queue_reasoning(
    log: list[tuple[str, object]],
    speaker: FakeSpeaker,
    live: list[str],
    plans: list[LessonPlan] | None = None,
    visual: list[TurnChunk] | None = None,
    plan_gates: list[asyncio.Event] | None = None,
    visual_gates: list[asyncio.Event] | None = None,
) -> FakeReasoning:
    return FakeReasoning(
        log,
        spoken_chunks(SPOKEN_DELTAS),
        speaker.received,
        turns=[spoken_chunks(live)],
        plans=[planned_call(plan) for plan in (plans or [LESSON])],
        plan_gates=plan_gates,
        visual=visual if visual is not None else [draft_call(3)],
        visual_finish="tool_calls",
        visual_gates=visual_gates,
    )


async def test_one_scene_builds_ahead_of_the_page() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    reasoning = queue_reasoning(log, speaker, ["answered_right"], plans=[LESSON] * 3)
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(page.seen("scene.push", 1), HANG_GUARD_S)
    await opening_done(page)
    assert pushed(log) == ["ratio"]
    log.append(("fired", "ratio"))
    page.fire(1)
    await asyncio.wait_for(page.seen("scene.push", 2), HANG_GUARD_S)
    await turn_after(source, first, "turn-2")
    page.fire(2)
    page.fire(3)
    log.append(("fired", "clip"))
    page.fire(4)
    await asyncio.wait_for(page.seen("scene.push", 3), HANG_GUARD_S)

    second, third = sent(log, "scene.push")[1:]
    assert pushed(log) == ["ratio", "clip", "epochs"]
    assert log.index(("fired", "ratio")) < log.index(("send_json", second))
    assert log.index(("fired", "clip")) < log.index(("send_json", third))
    assert sent(log, "scene.show") == []
    assert loop._lesson.committed == {"ratio", "clip", "epochs"}
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)


async def test_scene_two_is_checked_only_after_scene_one_opens() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    reasoning = queue_reasoning(log, speaker, [])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)

    with line_seen("scene.held scene_id=clip") as held:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(page.seen("lesson.cue", 1), HANG_GUARD_S)
    assert pushed(log) == ["ratio"] and "ratio" in loop._lesson.built
    log.append(("fired", "ratio"))
    page.fire(1)
    await asyncio.wait_for(page.seen("scene.push", 2), HANG_GUARD_S)

    assert log.index(("fired", "ratio")) < log.index(("send_json", sent(log, "scene.push")[1]))
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)


async def test_no_build_starts_while_a_planner_call_is_in_flight() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready=READY)
    rerun = asyncio.Event()
    reasoning = queue_reasoning(
        log,
        speaker,
        ["answered_right"],
        plans=[LESSON] * 3,
        plan_gates=[opened(), rerun],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)
    running = asyncio.create_task(loop.run())

    await opening_done(page)
    await asyncio.wait_for(page.state_where(status_is("clip", "built")), HANG_GUARD_S)
    reasoning.planning.clear()
    await pull_past(source, first)
    await asyncio.wait_for(page.state_where(lambda p: p["current"] == "clip"), HANG_GUARD_S)
    boundary = planner_task()
    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[1].held.wait(), HANG_GUARD_S)
    assert builds(reasoning, "Several epochs") == []
    log.append(("released", None))
    rerun.set()
    await asyncio.wait_for(asyncio.wait([boundary]), HANG_GUARD_S)
    await asyncio.wait_for(page.seen("scene.push", 3), HANG_GUARD_S)

    epochs = next(
        n
        for n, (name, text) in enumerate(log)
        if name == "build" and "\nTitle: Several epochs\n" in str(text)
    )
    assert log.index(("released", None)) < epochs
    assert pushed(log) == ["ratio", "clip", "epochs"]
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)


async def test_the_queue_follows_an_accepted_rerun() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready={**READY, "trust": ""})
    trusted = with_fourth_scene(LESSON).model_dump()
    del trusted["scenes"][2]
    replanned = LessonPlan.model_validate(trusted)
    reasoning = queue_reasoning(
        log, speaker, ["answered_right"], plans=[LESSON, replanned, replanned]
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)
    running = asyncio.create_task(loop.run())

    await opening_done(page)
    await pull_past(source, first)
    await asyncio.wait_for(page.seen("scene.push", 3), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pushed(log) == ["ratio", "clip", "trust"]
    assert builds(reasoning, "Several epochs") == []


async def test_a_barge_in_cancels_neither_the_build_nor_the_planner(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready=READY)
    clip_gate = asyncio.Event()
    rerun = asyncio.Event()
    reasoning = queue_reasoning(
        log,
        speaker,
        ["answered_right"],
        plans=[LESSON] * 3,
        plan_gates=[opened(), rerun],
        visual_gates=[opened(), clip_gate],
    )
    first = asyncio.Event()
    barge = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), barge, SpeechStarted(), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await asyncio.wait_for(page.state_where(status_is("clip", "building")), HANG_GUARD_S)
        builder = builder_task()
        reasoning.planning.clear()
        await pull_past(source, first)
        await asyncio.wait_for(page.state_where(lambda p: p["current"] == "clip"), HANG_GUARD_S)
        planner = planner_task()
        await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.plan_streams[1].held.wait(), HANG_GUARD_S)
        await pull_past(source, barge)
        assert not builder.cancelling() and not planner.cancelling()
        clip_gate.set()
        rerun.set()
        await asyncio.wait_for(page.state_where(status_is("clip", "built")), HANG_GUARD_S)
        await asyncio.wait_for(asyncio.wait([planner]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert not planner.cancelled()
    messages = session_messages(caplog)
    assert "scene.built scene_id=clip steps=3" in messages
    assert "planner.accepted stage=boundary scenes=3" in messages


async def test_eof_and_aclose_end_the_builder_and_the_planner() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready=READY)
    clip_gate = asyncio.Event()
    rerun = asyncio.Event()
    reasoning = queue_reasoning(
        log,
        speaker,
        ["answered_right"],
        plans=[LESSON, LESSON],
        plan_gates=[opened(), rerun],
        visual_gates=[opened(), clip_gate],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)
    running = asyncio.create_task(loop.run())

    await opening_done(page)
    await asyncio.wait_for(page.state_where(status_is("clip", "building")), HANG_GUARD_S)
    builder = builder_task()
    await pull_past(source, first)
    await asyncio.wait_for(page.state_where(lambda p: p["current"] == "clip"), HANG_GUARD_S)
    planner = planner_task()
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert builder.cancelled() and planner.cancelled()
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith(("lesson-", "turn-"))
    ]


async def test_a_ready_report_never_interrupts_the_turn() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, ready=READY)
    reasoning = queue_reasoning(log, speaker, [])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)

    with line_seen("scene.built scene_id=ratio") as built:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        await asyncio.wait_for(built.wait(), HANG_GUARD_S)
    hold.set()
    await asyncio.wait_for(speaker.finished.wait(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert ("flush_playout", None) not in log
    assert reasoning.prompts == []
    assert speaker.utterances == [PREPARED_OPENING]


async def test_a_theme_message_reaches_the_planned_scene_prompt() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), speaker.received, plans=[planned_call(LESSON)]
    )
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )
    page.reply({"type": "theme", "theme": "dark"})
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.visual_started.wait(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert "The page is dark." in reasoning.build_prompts[0].system


async def test_the_builder_uses_the_scene_model_and_the_voice_does_not() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    reasoning = queue_reasoning(log, speaker, ["go_on"])
    cfg = lesson_cfg().model_copy(update={"scene_model": "draw-1", "scene_effort": "medium"})
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="go on"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(page.state_where(status_is("ratio", "built")), HANG_GUARD_S)
    await opening_done(page)
    await turn_after(source, first, "turn-2")
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.build_models and set(reasoning.build_models) == {"draw-1"}
    assert set(reasoning.build_efforts) == {"medium"}
    assert reasoning.models == [None] and reasoning.efforts == [None]


async def test_a_planned_scene_is_built_from_the_plan_alone(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [SEARCH_CALL],
        speaker.received,
        follow_up=spoken_chunks(FOLLOW_DELTAS),
        plans=[planned_call(LESSON)],
        visual=[draft_call(3)],
        visual_finish="tool_calls",
        visual_gates=[held],
    )
    restored = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([restored, EndOfTurn(text=USER_TEXT), stop])
    loop = TurnLoop(
        config(tmp_path).model_copy(update={"planned": True}),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
    )
    loop._lesson.scripts = dict(SCRIPTS)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.visual_started.wait(), HANG_GUARD_S)
    await opening_done(page)
    page.scene_id, page.step, page.revision = "epochs", 3, 3
    page.reply({**RESTORE, "scene_id": "epochs", "step": 3})
    await turn_after(source, restored, "turn-2")
    held.set()
    await asyncio.wait_for(page.state_where(status_is("epochs", "built")), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert ("search_done", MODEL_QUERY) in log
    searched = log.index(("search_done", MODEL_QUERY))
    assert any(kind == "build" for kind, _ in log[searched:])
    assert reasoning.build_prompts
    scenes = {scene.title: scene for scene in LESSON.scenes}
    for prompt in reasoning.build_prompts:
        assert prompt.tool_context == [] and prompt.tool_exchange == []
        title = prompt.user_text.splitlines()[2].removeprefix("Title: ")
        expected = planned_scene_prompt(SUBJECT, LESSON.profile, scenes[title], "light")
        assert prompt.user_text == expected.user_text


async def test_statuses_follow_the_queue() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready=READY)
    reasoning = queue_reasoning(log, speaker, [])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(page.state_where(status_is("clip", "built")), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    def history(scene_id: str) -> list[str]:
        statuses = [
            next(row["status"] for row in payload["scenes"] if row["id"] == scene_id)
            for payload in sent(log, "lesson.state")
        ]
        return [s for n, s in enumerate(statuses) if n == 0 or s != statuses[n - 1]]

    assert history("ratio") == ["planned", "building", "built"]
    assert history("clip") == ["planned", "building", "built"]


RETRY_TAIL = "\n\nThe previous attempt failed its check: {error}\nWrite the whole scene again."
BAND_ERROR = "timeline lacks labels step-3"


async def test_a_failed_build_ahead_is_retried_once_with_its_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready={"ratio": "", "clip": BAND_ERROR})
    reasoning = queue_reasoning(log, speaker, [])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(page.state_where(status_is("clip", "failed")), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    clip = builds(reasoning, "The clip")
    assert len(clip) == 2
    assert clip[1].user_text.endswith(RETRY_TAIL.format(error=BAND_ERROR))
    assert "scene.failed scene_id=clip attempt=2 reason=check" in session_messages(caplog)
    assert loop._lesson.failed == {"clip"}


async def test_scene_one_is_retried_until_the_voice_opens_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready={"ratio": BAND_ERROR, "clip": ""})
    written = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        visual=[draft_call(3)],
        visual_finish="tool_calls",
        scripts=[script_call("ratio")],
        script_gates=[written],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with (
        caplog.at_level(logging.INFO, logger="tutor.session"),
        line_seen("scene.failed scene_id=ratio attempt=2") as failed,
    ):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(page.seen("scene.push", 2), HANG_GUARD_S)
        await asyncio.wait_for(failed.wait(), HANG_GUARD_S)
        assert sent(log, "lesson.cue") == []
        written.set()
        await asyncio.wait_for(page.seen("lesson.cue", 1), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    ratio = builds(reasoning, "The ratio")
    assert len(ratio) == 2
    assert ratio[1].user_text.endswith(RETRY_TAIL.format(error=BAND_ERROR))
    assert "scene.failed scene_id=ratio attempt=2 reason=check" in session_messages(caplog)
    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}


async def test_a_build_that_fails_while_its_scene_is_taught_fails_at_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready={"ratio": "", "clip": BAND_ERROR})
    clip_gate = asyncio.Event()
    reasoning = queue_reasoning(
        log,
        speaker,
        ["answered_right"],
        plans=[LESSON] * 3,
        visual_gates=[opened(), clip_gate],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await asyncio.wait_for(page.state_where(status_is("clip", "building")), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.build_streams[1].held.wait(), HANG_GUARD_S)
        await pull_past(source, first)
        await asyncio.wait_for(page.state_where(lambda p: p["current"] == "clip"), HANG_GUARD_S)
        clip_gate.set()
        await asyncio.wait_for(page.state_where(status_is("clip", "failed")), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(builds(reasoning, "The clip")) == 1
    assert "scene.failed scene_id=clip attempt=1 reason=check" in session_messages(caplog)


async def test_a_draft_with_the_wrong_count_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    reasoning = queue_reasoning(log, speaker, SPOKEN_DELTAS, visual=[draft_call(4)])
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )

    with (
        caplog.at_level(logging.INFO, logger="tutor.session"),
        line_seen("scene.failed scene_id=clip attempt=2") as failed,
    ):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(failed.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert sent(log, "scene.push") == []
    assert "scene.failed scene_id=ratio attempt=2 reason=count" in session_messages(caplog)
    error = "write exactly 3 say lines, one per step; the draft had 4"
    assert builds(reasoning, "The ratio")[1].user_text.endswith(RETRY_TAIL.format(error=error))


async def test_a_draft_that_names_a_source_path_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    named = said_call(["The ratio at one action", "see src/pool.py:12", "The band"])
    reasoning = queue_reasoning(log, speaker, SPOKEN_DELTAS, visual=[named])
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )

    with (
        caplog.at_level(logging.INFO, logger="tutor.session"),
        line_seen("scene.failed scene_id=ratio attempt=2") as failed,
    ):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(failed.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert "ratio" not in pushed(log)
    assert "scene.failed scene_id=ratio attempt=2 reason=position" in session_messages(caplog)
    error = "write no file path, symbol or line number; the scene is concept content only"
    assert builds(reasoning, "The ratio")[1].user_text.endswith(RETRY_TAIL.format(error=error))


async def test_a_page_that_never_reports_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready={})
    reasoning = queue_reasoning(log, speaker, SPOKEN_DELTAS)
    cfg = lesson_cfg().model_copy(update={"scene_ready_timeout_s": 0.01})
    stop = asyncio.Event()
    loop = concept_loop(log, ScriptedSource([stop]), speaker, reasoning, cfg=cfg, transport=page)

    with (
        caplog.at_level(logging.INFO, logger="tutor.session"),
        line_seen("scene.failed scene_id=ratio attempt=2") as failed,
    ):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(failed.wait(), HANG_GUARD_S)
        assert "ratio" not in loop._ready
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pushed(log)[:2] == ["ratio", "ratio"]
    assert "scene.failed scene_id=ratio attempt=2 reason=no_report" in session_messages(caplog)


async def test_a_build_past_its_bound_is_a_failed_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    never = [asyncio.Event(), asyncio.Event()]
    reasoning = queue_reasoning(log, speaker, SPOKEN_DELTAS, visual_gates=never)
    cfg = lesson_cfg().model_copy(update={"scene_timeout_s": 0.01})
    stop = asyncio.Event()
    loop = concept_loop(log, ScriptedSource([stop]), speaker, reasoning, cfg=cfg, transport=page)

    with (
        caplog.at_level(logging.INFO, logger="tutor.session"),
        line_seen("scene.failed scene_id=ratio attempt=2") as failed,
    ):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(failed.wait(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert "ratio" not in pushed(log)
    assert "scene.failed scene_id=ratio attempt=2 reason=timeout" in session_messages(caplog)


class RatioPushFails(FakePage):
    async def send_json(self, payload: dict[str, object]) -> None:
        if payload["type"] == "scene.push" and payload["scene_id"] == "ratio":
            raise ChannelClosed()
        await super().send_json(payload)


async def test_an_unexpected_build_error_marks_the_scene_failed_and_the_queue_moves_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = RatioPushFails(log, ready=READY)
    reasoning = queue_reasoning(log, speaker, SPOKEN_DELTAS)
    stop = asyncio.Event()
    loop = concept_loop(
        log, ScriptedSource([stop]), speaker, reasoning, cfg=lesson_cfg(), transport=page
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(page.seen("scene.push", 1), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pushed(log) == ["clip"]
    assert "scene.task_failed scene_id=ratio error=ChannelClosed" in session_messages(caplog)
    assert "ratio" in loop._lesson.failed


PREPARED_OPENING = [
    "Here are the old and the new policy over three actions.",
    "Each bar is the chance one of them gives an action.",
    "Divide the new chance by the old one at a single action.",
    "What is the ratio where they agree?",
]
PREPARED_QUESTION = PREPARED_OPENING[2:]


def split_loop(
    log: list[tuple[str, object]],
    source: InputPath,
    speaker: FakeSpeaker,
    reasoning: FakeReasoning,
    page: FakePage,
    scripts: dict[str, list[ScriptChunk]] | None = None,
    registry: FakeRegistry | TurnRegistry | None = None,
    clock: Callable[[], float] | None = None,
    cfg: TurnLoopConfig | None = None,
) -> TurnLoop:
    loop = TurnLoop(
        cfg if cfg is not None else lesson_cfg(),
        source,
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        registry if registry is not None else FakeRegistry(log),
        clock if clock is not None else FakeClock(),
    )
    loop._lesson.scripts = dict(SCRIPTS if scripts is None else scripts)
    return loop


def with_chunk(scene_id: str, step: int, question: bool, text: str) -> dict[str, list[ScriptChunk]]:
    rewritten = [
        ScriptChunk(step=step, question=question, text=text)
        if (chunk.step, chunk.question) == (step, question)
        else chunk
        for chunk in SCRIPTS[scene_id]
    ]
    return {**SCRIPTS, scene_id: rewritten}


async def opening_done(page: FakePage) -> None:
    await asyncio.wait_for(page.seen("caption", 1), HANG_GUARD_S)
    await asyncio.wait_for(asyncio.wait([turn_task("turn-1")]), HANG_GUARD_S)


class SetClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class BackloggedSpeaker(FakeSpeaker):
    async def speak(self, chunks: AsyncIterator[Chunk], on_play: OnPlay) -> None:
        async def late(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
            await on_play(chunk, 9000, 3000)

        await super().speak(chunks, late)


async def test_the_opening_plays_prepared_chunks_with_no_voice_call() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [PREPARED_OPENING]
    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}
    assert cue["chunk_id"] == speaker.chunks[0].id
    (caption,) = [c for c in sent(log, "caption") if c["text"] == PREPARED_OPENING[0]]
    assert log.index(("send_json", cue)) == log.index(("send_json", caption)) + 1
    assert reasoning.streams == []
    assert ("ratio", 2) in loop._lesson.asked


async def test_go_on_ends_the_live_stream_at_its_label(caplog: pytest.LogCaptureFixture) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on", "\nWell, then", " more."])],
        plans=[planned_call(LESSON)],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="go on"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        mark = len(log)
        await pull_past(source, first)
        await asyncio.wait_for(asyncio.wait([turn_task("turn-2")]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    (stream,) = reasoning.streams
    assert (stream.yielded, stream.cancels) == (1, 1)
    after = log[mark:]
    assert after.index(("stream_closed", 1)) < after.index(("speak", PREPARED_QUESTION[0]))
    assert speaker.utterances[1] == PREPARED_QUESTION
    assert not any("Well" in text for text in spoken_texts(speaker))
    messages = session_messages(caplog)
    (label,) = [m for m in messages if m.startswith("reply.label")]
    assert re.fullmatch(r"reply\.label turn_id=turn-2 label=go_on ms=\d+", label)
    assert "reply.discarded turn_id=turn-2 chars=0" in messages
    (prompt,) = reasoning.prompts
    assert prompt.system == LIVE_PROMPT
    assert prompt.history == []
    assert "Pending question: What is the ratio where they agree?" in prompt.user_text
    assert prompt.user_text.endswith("The learner now says: go on")
    assert (reasoning.tools, reasoning.max_tokens, reasoning.efforts, reasoning.models) == (
        [None],
        [None],
        [None],
        [None],
    )


async def test_a_side_question_reaction_is_spoken_as_it_streams() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gate = asyncio.Event()
    reaction = spoken_chunks(
        ["side_question\nIt is one number ", "per action. What is the ratio where they agree?"]
    )
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        gate=gate,
        holds_at=1,
        turns=[reaction],
        plans=[planned_call(LESSON)],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="is it one number per action?"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    asked = set(loop._lesson.asked)
    await pull_past(source, first)
    turn = turn_task("turn-2")
    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    assert spoken_texts(speaker) == PREPARED_OPENING
    gate.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1] == [
        "It is one number per action.",
        "What is the ratio where they agree?",
    ]
    assert len(sent(log, "lesson.cue")) == 1
    assert loop._lesson.asked == asked == {("ratio", 2)}


async def test_tags_in_a_reaction_are_neither_spoken_nor_cued(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["side_question\n<step 2>It is one. <scene 2>Fine."])],
        plans=[planned_call(LESSON)],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="is it one?"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await pull_past(source, first)
        await asyncio.wait_for(asyncio.wait([turn_task("turn-2")]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1] == ["It is one.", "Fine."]
    assert len(sent(log, "lesson.cue")) == 1
    dropped = [m for m in session_messages(caplog) if m.startswith("tag.dropped")]
    assert dropped == [
        "tag.dropped turn_id=turn-2 kind=step reason=reaction chars=6",
        "tag.dropped turn_id=turn-2 kind=scene reason=reaction chars=7",
    ]


async def test_a_barge_in_mid_chunk_drops_the_unheard_cue_and_go_on_resumes_from_the_page(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=3)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on"])],
        plans=[planned_call(WATCHED)],
    )
    stop = asyncio.Event()
    loop = split_loop(
        log, ScriptedSource([stop]), speaker, reasoning, page, scripts=WATCHED_SCRIPTS
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        scene, step = sent(log, "lesson.cue")
        page.fire(int(typing.cast(int, scene["cue_id"])))
        page.handlers[0]({"type": "say", "text": "go on"})
        turn = turn_task("turn-2")
        hold.set()
        await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[0] == [
        "Here are the old and the new policy over three actions.",
        "Each bar is the chance one of them gives an action.",
        "Where the two policies agree the ratio is exactly one.",
    ]
    assert scene["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}
    assert step["tag"] == {"kind": "step", "n": 2}
    assert step["chunk_id"] == speaker.chunks[2].id
    messages = session_messages(caplog)
    assert f"lesson.ack cue_id={step['cue_id']} outcome=dropped reason=barrier" in messages
    assert any(
        m.startswith("lesson.synced epoch=1 barrier=1 scene_id=ratio step=1 ") for m in messages
    )
    (sync,) = sent(log, "lesson.sync")
    later = [
        cue
        for cue in sent(log, "lesson.cue")
        if log.index(("send_json", cue)) > log.index(("send_json", sync))
    ]
    assert later[0]["tag"] == {"kind": "step", "n": 2}
    assert later[0]["barrier"] == 1
    assert speaker.utterances[1][0] == "Where the two policies agree the ratio is exactly one."
    assert later[0]["chunk_id"] == speaker.chunks[3].id
    assert all(cue["barrier"] == 1 for cue in later)


async def test_a_question_cut_before_its_end_is_not_pending() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, fires=True)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[planned_call(LESSON)],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)

    with line_seen("cursor.scene scene_id=ratio") as fired:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(fired.wait(), HANG_GUARD_S)
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    page.handlers[0]({"type": "say", "text": "one"})
    assert loop._lesson.asked == set()
    turn = turn_task("turn-2")
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[0] == PREPARED_OPENING[:1]
    assert speaker.utterances[1] == PREPARED_QUESTION
    assert loop._lesson.asked == {("ratio", 2)}


async def asked_after_a_cut(cut_at: float) -> set[tuple[str, int]]:
    log: list[tuple[str, object]] = []
    clock = SetClock()
    speaker = BackloggedSpeaker(log)
    page = FakePage(log, fires=True)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, clock=clock)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    assert loop._lesson.asked == {("ratio", 2)}
    clock.now = cut_at
    page.handlers[0]({"type": "say", "text": "one"})
    asked = set(loop._lesson.asked)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)
    return asked


async def test_a_question_cut_while_its_audio_waits_to_play_is_not_pending() -> None:
    assert await asked_after_a_cut(1.0) == set()
    assert await asked_after_a_cut(13.0) == {("ratio", 2)}


async def test_a_withheld_question_is_never_pending(caplog: pytest.LogCaptureFixture) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    scripts = with_chunk("ratio", 2, True, "Which flag in setup.py sets it?")
    loop = split_loop(
        log,
        ScriptedSource([stop]),
        speaker,
        reasoning,
        page,
        scripts=scripts,
        registry=TurnRegistry(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [PREPARED_OPENING[:2]]
    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    assert "question.dropped turn_id=turn-1 scene_id=ratio n=2 reason=withheld" in messages
    assert loop._lesson.asked == set()


async def test_a_prepared_chunk_naming_a_path_is_withheld(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    scripts = with_chunk("ratio", 1, False, "Look in setup.py for the flags.")
    loop = split_loop(
        log,
        ScriptedSource([stop]),
        speaker,
        reasoning,
        page,
        scripts=scripts,
        registry=TurnRegistry(),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [PREPARED_QUESTION]
    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=model ungrounded=1" in messages
    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}
    assert cue["chunk_id"] == speaker.chunks[0].id
    (caption,) = [c for c in sent(log, "caption") if c["text"] == PREPARED_QUESTION[0]]
    assert log.index(("send_json", cue)) == log.index(("send_json", caption)) + 1


async def test_no_speculation_starts_on_the_split_path() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, PartialTranscript(text="go"), stop])
    cfg = lesson_cfg().model_copy(update={"speculative_reasoning": True})
    loop = split_loop(log, source, speaker, reasoning, page, cfg=cfg)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    await pull_past(source, first)
    speculating = [
        task.get_name() for task in asyncio.all_tasks() if task.get_name().endswith("-speculation")
    ]
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speculating == []
    assert loop._speculations == {}


async def test_a_failed_live_call_ends_the_turn(caplog: pytest.LogCaptureFixture) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["side_question\nIt is one number. "])],
        plans=[planned_call(LESSON)],
        stream_raises=RateLimited(RATE_LIMIT_DETAIL),
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="is it one number?"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await pull_past(source, first)
        await asyncio.wait_for(asyncio.wait([turn_task("turn-2")]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1] == ["It is one number."]
    messages = session_messages(caplog)
    assert "turn.reasoning_failed turn_id=turn-2 error=RateLimited" in messages
    assert any(m.startswith("turn.spoken turn_id=turn-2 ") for m in messages)
    assert states(log)[-1] == "listening"


def said(
    scene_id: str,
    step: int,
    question: bool = False,
    scripts: dict[str, list[ScriptChunk]] = SCRIPTS,
) -> list[str]:
    (text,) = [c.text for c in scripts[scene_id] if (c.step, c.question) == (step, question)]
    return re.split(r"(?<=[.?]) ", text)


def step_cue(n: int) -> dict[str, object]:
    return {"kind": "step", "n": n}


def scene_cue(n: int, scene_id: str) -> dict[str, object]:
    return {"kind": "scene", "n": n, "scene_id": scene_id}


def cued(log: list[tuple[str, object]], speaker: FakeSpeaker) -> list[tuple[object, str]]:
    texts = {chunk.id: chunk.text for chunk in speaker.chunks}
    return [
        (cue["tag"], texts[typing.cast(int, cue["chunk_id"])]) for cue in sent(log, "lesson.cue")
    ]


def ask_on(
    scene_id: str, step: int, ask: str, question: str
) -> tuple[LessonPlan, dict[str, list[ScriptChunk]]]:
    body = LESSON.model_dump()
    (scene,) = [each for each in body["scenes"] if each["id"] == scene_id]
    scene["steps"][step - 1]["ask"] = ask
    chunk = ScriptChunk(step=step, question=True, text=question)
    return LessonPlan.model_validate(body), {**SCRIPTS, scene_id: [*SCRIPTS[scene_id], chunk]}


async def turn_after(source: ScriptedSource, gate: asyncio.Event, turn_id: str) -> None:
    await pull_past(source, gate)
    await asyncio.wait_for(asyncio.wait([turn_task(turn_id)]), HANG_GUARD_S)


async def test_a_right_answer_speaks_the_bridge_then_the_answer_step() -> None:
    plan, scripts = ask_on(
        "epochs",
        2,
        "Where does the ratio drift after several epochs?",
        "Now run several epochs on the same batch. Where does the ratio go if nothing holds it?",
    )
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    labels = [spoken_chunks([label]) for label in ("answered_right", "go_on", "answered_right")]
    reasoning = FakeReasoning(log, [], speaker.received, turns=labels, plans=[planned_call(plan)])
    gates = [asyncio.Event() for _ in labels]
    stop = asyncio.Event()
    lines = ["one", "go on", "it drifts away"]
    source = ScriptedSource(
        [item for gate, line in zip(gates, lines) for item in (gate, EndOfTurn(text=line))] + [stop]
    )
    loop = split_loop(log, source, speaker, reasoning, page, scripts=scripts)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    marks = []
    for n, gate in enumerate(gates, start=2):
        marks.append(len(log))
        await turn_after(source, gate, f"turn-{n}")
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    answered = [*said("ratio", 2), *said("ratio", 3)]
    opened = [*said("clip", 1), *said("clip", 2), *said("clip", 3)]
    assert speaker.utterances[1] == ["Yes, that's right.", *answered, *opened]
    assert cued(log[marks[0] : marks[1]], speaker) == [
        (step_cue(2), said("ratio", 2)[0]),
        (step_cue(3), said("ratio", 3)[0]),
        (scene_cue(2, "clip"), said("clip", 1)[0]),
        (step_cue(2), said("clip", 2)[0]),
        (step_cue(3), said("clip", 3)[0]),
    ]
    assert speaker.utterances[2] == [*said("epochs", 1), *said("epochs", 2, True, scripts)]
    assert speaker.utterances[3] == ["Exactly right.", *said("epochs", 2), *said("epochs", 3)]


async def test_a_wrong_answer_reaction_comes_before_the_answer_step() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reaction = spoken_chunks(["answered_wrong\nNot quite", " right."])
    reasoning = FakeReasoning(
        log, [], speaker.received, turns=[reaction], plans=[planned_call(LESSON)]
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="two"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    mark = len(log)
    await turn_after(source, first, "turn-2")
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1][:3] == ["Not quite right.", *said("ratio", 2)]
    assert cued(log[mark:], speaker)[0] == (step_cue(2), said("ratio", 2)[0])


async def test_tell_me_speaks_its_bridge_and_the_rest_of_the_scene() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["tell_me"])],
        plans=[planned_call(LESSON)],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="just tell me"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    mark = len(log)
    await turn_after(source, first, "turn-2")
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1] == ["Sure, here it is.", *said("ratio", 2), *said("ratio", 3)]
    assert cued(log[mark:], speaker) == [
        (step_cue(2), said("ratio", 2)[0]),
        (step_cue(3), said("ratio", 3)[0]),
    ]


async def test_an_unlabelled_reply_is_spoken_whole_without_label_words(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["Hmm, go_on then. ", "The ratio is one."])],
        plans=[planned_call(LESSON)],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="hmm"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        mark = len(log)
        await turn_after(source, first, "turn-2")
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    spoken = [" ".join(text.split()) for text in speaker.utterances[1]]
    assert spoken == ["Hmm, then.", "The ratio is one."]
    messages = session_messages(caplog)
    assert any(re.fullmatch(r"reply\.unlabelled turn_id=turn-2 ms=\d+", m) for m in messages)
    assert not any(m.startswith("reply.label") for m in messages)
    assert sent(log[mark:], "lesson.cue") == []


async def test_text_after_a_bridge_label_is_discarded_and_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on\nSure thing."])],
        plans=[planned_call(LESSON)],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="okay"), stop])
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        await turn_after(source, first, "turn-2")
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1] == PREPARED_QUESTION
    assert not any("Sure thing" in text for text in spoken_texts(speaker))
    assert "reply.discarded turn_id=turn-2 chars=11" in session_messages(caplog)


async def test_a_learner_who_speaks_before_the_opening_hears_the_reaction_then_scene_one() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["side_question\nHello there."])],
        plans=[planned_call(LESSON)],
        plan_gates=[held],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page)

    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[0].held.wait(), HANG_GUARD_S)
    page.handlers[0]({"type": "say", "text": "hello"})
    turn = turn_task("turn-1")
    held.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [["Hello there.", *PREPARED_OPENING]]
    assert cued(log, speaker) == [(scene_cue(1, "ratio"), PREPARED_OPENING[0])]
    (prompt,) = reasoning.prompts
    assert prompt.user_text.endswith("The learner now says: hello")


async def test_the_turn_after_the_last_step_takes_the_voice_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    turns = [
        spoken_chunks(["answered_right"]),
        spoken_chunks(["go_on"]),
        spoken_chunks(["That is the whole lesson."]),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns, plans=[planned_call(LESSON)])
    gates = [asyncio.Event() for _ in turns]
    stop = asyncio.Event()
    lines = ["one", "go on", "thanks"]
    source = ScriptedSource(
        [item for gate, line in zip(gates, lines) for item in (gate, EndOfTurn(text=line))] + [stop]
    )
    loop = split_loop(log, source, speaker, reasoning, page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        for n, gate in enumerate(gates, start=2):
            await turn_after(source, gate, f"turn-{n}")
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    voice = reasoning.prompts[-1]
    assert (voice.user_text, speaker.utterances[3]) == ("thanks", ["That is the whole lesson."])
    assert "This was the last scene and it is done: close the lesson" in voice.system
    assert not any(m.startswith("reply.label turn_id=turn-4") for m in session_messages(caplog))


async def test_a_scene_that_asks_first_opens_on_a_right_answer() -> None:
    ask = "What does the new axis measure?"
    plan, scripts = ask_on("clip", 1, ask, "A new axis comes next. What do you think it measures?")
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    labels = [spoken_chunks(["answered_right"]), spoken_chunks(["answered_right"])]
    reasoning = FakeReasoning(log, [], speaker.received, turns=labels, plans=[planned_call(plan)])
    first = asyncio.Event()
    second = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource(
        [first, EndOfTurn(text="one"), second, EndOfTurn(text="the ratio"), stop]
    )
    loop = split_loop(log, source, speaker, reasoning, page, scripts=scripts)

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    mark = len(log)
    await turn_after(source, first, "turn-2")
    asked = set(loop._lesson.asked)
    later = len(log)
    await turn_after(source, second, "turn-3")
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    answered = [*said("ratio", 2), *said("ratio", 3)]
    assert speaker.utterances[1] == [
        "Yes, that's right.",
        *answered,
        *said("clip", 1, True, scripts),
    ]
    assert cued(log[mark:later], speaker) == [
        (step_cue(2), said("ratio", 2)[0]),
        (step_cue(3), said("ratio", 3)[0]),
    ]
    assert ("clip", 1) in asked
    assert f"Pending question: {ask}" in reasoning.prompts[1].user_text
    clip = [*said("clip", 1), *said("clip", 2), *said("clip", 3)]
    epochs = [*said("epochs", 1), *said("epochs", 2), *said("epochs", 3)]
    assert speaker.utterances[2] == ["Exactly right.", *clip, *epochs]
    assert cued(log[later:], speaker) == [
        (scene_cue(2, "clip"), said("clip", 1)[0]),
        (step_cue(2), said("clip", 2)[0]),
        (step_cue(3), said("clip", 3)[0]),
        (scene_cue(3, "epochs"), said("epochs", 1)[0]),
        (step_cue(2), said("epochs", 2)[0]),
        (step_cue(3), said("epochs", 3)[0]),
    ]


NO_SCRIPT = spoken_chunks(["The script follows."])
NOT_READY_SAID = re.split(r"(?<=[.?]) ", NOT_READY)
RIGHT_ANSWER = ["Yes, that's right.", *said("ratio", 2), *said("ratio", 3)]


def script_call(scene_id: str) -> list[TurnChunk]:
    body = {"chunks": [chunk.model_dump() for chunk in SCRIPTS[scene_id]]}
    return [
        TurnChunk(
            kind="tool_call",
            text=json.dumps(body),
            tool_call_id=f"call-script-{scene_id}",
            tool_name=SCRIPT_TOOL,
        )
    ]


def scripted_titles(reasoning: FakeReasoning) -> list[str]:
    return [
        line.removeprefix("This scene: ")
        for prompt in reasoning.script_prompts
        for line in prompt.user_text.splitlines()
        if line.startswith("This scene: ")
    ]


def call_at(log: list[tuple[str, object]], kind: str, line: str) -> int:
    return next(n for n, (name, text) in enumerate(log) if name == kind and line in str(text))


def scripter_task() -> asyncio.Task[None]:
    (task,) = [task for task in asyncio.all_tasks() if task.get_name() == "lesson-scripter"]
    return task


async def test_the_scripter_writes_scene_one_then_scene_two_one_at_a_time() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    ratio = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio"), script_call("clip"), script_call("epochs")],
        script_gates=[ratio],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={})

    with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
    assert scripted_titles(reasoning) == ["The ratio"]
    with line_seen("script.written scene_id=clip", "tutor.script") as written:
        ratio.set()
        await asyncio.wait_for(written.wait(), HANG_GUARD_S)
    await opening_done(page)
    await turn_after(source, first, "turn-2")
    for cue_id in (1, 2, 3):
        page.fire(cue_id)
    await asyncio.wait_for(page.state_where(lambda p: p["current"] == "ratio"), HANG_GUARD_S)
    assert scripted_titles(reasoning) == ["The ratio", "The clip"]
    log.append(("fired", "clip"))
    page.fire(4)
    await asyncio.wait_for(reasoning.scripted(3), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert scripted_titles(reasoning) == ["The ratio", "The clip", "Several epochs"]
    assert reasoning.scripts_open == [0, 0, 0]
    assert reasoning.script_models == [lesson_cfg().script_model] * 3
    assert log.index(("fired", "clip")) < call_at(log, "script", "\nThis scene: Several epochs\n")
    assert speaker.utterances[0] == PREPARED_OPENING


async def test_a_scene_is_protected_when_its_script_call_starts() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    rewritten = with_title(LESSON, 1, "The clip band")
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["side_question\nIt is one number per action."])],
        plans=[planned_call(LESSON), planned_call(rewritten)],
        scripts=[script_call("clip")],
        script_gates=[asyncio.Event()],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="is it one number per action?"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={"ratio": SCRIPTS["ratio"]})

    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(reasoning.scripted(1), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.script_streams[0].held.wait(), HANG_GUARD_S)
    await opening_done(page)
    with line_seen("planner.accepted stage=first_answer") as accepted:
        await turn_after(source, first, "turn-2")
        await asyncio.wait_for(accepted.wait(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert scripted_titles(reasoning) == ["The clip"]
    assert "Protected: ratio, clip" in reasoning.plan_prompts[1].user_text
    assert loop._lesson.plan == LESSON
    assert "clip" in loop._lesson.scripting and "clip" not in loop._lesson.committed


async def test_a_scene_scripted_before_the_builder_reaches_it_is_still_built() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True, ready=READY)
    held = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio"), script_call("clip")],
        visual=[draft_call(3)],
        visual_finish="tool_calls",
        visual_gates=[held],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with line_seen("script.written scene_id=clip", "tutor.script") as written:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(written.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.visual_started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.build_streams[0].held.wait(), HANG_GUARD_S)
    assert set(loop._lesson.scripts) == {"ratio", "clip"}
    assert builds(reasoning, "The clip") == [] and "ratio" not in loop._lesson.built
    held.set()
    await asyncio.wait_for(page.seen("scene.push", 2), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pushed(log) == ["ratio", "clip"] and "ratio" in loop._lesson.built
    assert len(builds(reasoning, "The clip")) == 1
    ratio_script = call_at(log, "script", "\nThis scene: The ratio\n")
    assert ratio_script < call_at(log, "build", "\nTitle: The ratio\n")


async def test_the_opening_waits_for_scene_ones_script() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio")],
        script_gates=[gate],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
    assert spoken_texts(speaker) == []
    gate.set()
    await opening_done(page)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [PREPARED_OPENING]
    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == scene_cue(1, "ratio")


async def test_an_answer_that_opens_the_next_scene_waits_after_its_bridge() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    clip = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[planned_call(LESSON)],
        scripts=[script_call("clip")],
        script_gates=[clip],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={"ratio": SCRIPTS["ratio"]})

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    mark = len(log)
    with line_seen("script.waiting turn_id=turn-2 scene_id=clip") as waiting:
        await pull_past(source, first)
        await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
    turn = turn_task("turn-2")
    await asyncio.wait_for(
        page.seen("caption", len(PREPARED_OPENING) + len(RIGHT_ANSWER)), HANG_GUARD_S
    )
    assert speaker.utterances[1] == RIGHT_ANSWER
    clip.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    opened = [*said("clip", 1), *said("clip", 2), *said("clip", 3)]
    assert speaker.utterances[1] == [*RIGHT_ANSWER, *opened]
    assert cued(log[mark:], speaker) == [
        (step_cue(2), said("ratio", 2)[0]),
        (step_cue(3), said("ratio", 3)[0]),
        (scene_cue(2, "clip"), said("clip", 1)[0]),
        (step_cue(2), said("clip", 2)[0]),
        (step_cue(3), said("clip", 3)[0]),
    ]


async def test_a_script_that_fails_twice_speaks_not_ready_and_is_written_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"]), spoken_chunks(["go_on"])],
        plans=[planned_call(LESSON)],
        scripts=[NO_SCRIPT, NO_SCRIPT, script_call("clip")],
    )
    first = asyncio.Event()
    second = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), second, EndOfTurn(text="go on"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={"ratio": SCRIPTS["ratio"]})

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        with line_seen("script.failed scene_id=clip") as failed:
            running = asyncio.create_task(loop.run())
            await asyncio.wait_for(failed.wait(), HANG_GUARD_S)
        await opening_done(page)
        await turn_after(source, first, "turn-2")
        await turn_after(source, second, "turn-3")
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances[1] == [*RIGHT_ANSWER, *NOT_READY_SAID]
    assert speaker.utterances[2] == [*said("clip", 1), *said("clip", 2), *said("clip", 3)]
    messages = session_messages(caplog)
    assert messages.count("script.failed scene_id=clip") == 1
    assert "reply.not_ready turn_id=turn-2 scene_id=clip" in messages
    assert scripted_titles(reasoning) == ["The clip"] * 3
    assert "rejected" not in reasoning.script_prompts[2].user_text
    assert loop._lesson.scripts["clip"] == SCRIPTS["clip"] and loop._lesson.unscripted == set()


async def test_a_script_that_fails_twice_while_the_opening_waits_speaks_not_ready(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gates = [asyncio.Event(), asyncio.Event()]
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        scripts=[NO_SCRIPT, NO_SCRIPT],
        script_gates=gates,
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
            running = asyncio.create_task(loop.run())
            await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
        opening = turn_task("turn-1")
        for gate in gates:
            gate.set()
        await asyncio.wait_for(asyncio.wait([opening]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [NOT_READY_SAID]
    assert sent(log, "lesson.cue") == []
    messages = session_messages(caplog)
    failed = messages.index("script.failed scene_id=ratio")
    assert failed < messages.index("reply.not_ready turn_id=turn-1 scene_id=ratio")


async def test_a_fired_scene_cue_wakes_both_the_builder_and_the_scripter() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, ready=READY)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio"), script_call("clip"), script_call("epochs")],
        visual=[draft_call(3)],
        visual_finish="tool_calls",
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={})

    with line_seen("script.written scene_id=clip", "tutor.script") as written:
        running = asyncio.create_task(loop.run())
        await opening_done(page)
        page.fire(1)
        await asyncio.wait_for(written.wait(), HANG_GUARD_S)
    await asyncio.wait_for(page.state_where(status_is("clip", "built")), HANG_GUARD_S)
    await turn_after(source, first, "turn-2")
    page.fire(2)
    page.fire(3)
    assert scripted_titles(reasoning) == ["The ratio", "The clip"]
    assert builds(reasoning, "Several epochs") == []
    log.append(("fired", "clip"))
    page.fire(4)
    await asyncio.wait_for(reasoning.scripted(3), HANG_GUARD_S)
    await asyncio.wait_for(page.seen("scene.push", 3), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    fired = log.index(("fired", "clip"))
    assert fired < call_at(log, "script", "\nThis scene: Several epochs\n")
    assert fired < call_at(log, "build", "\nTitle: Several epochs\n")
    assert pushed(log) == ["ratio", "clip", "epochs"]


async def test_the_scripter_waits_for_the_connect_plan_but_not_a_rerun() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    connect = asyncio.Event()
    boundary = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["answered_right"])],
        plans=[planned_call(LESSON), planned_call(RETITLED)],
        plan_gates=[connect, boundary],
        scripts=[script_call("ratio"), script_call("clip"), script_call("epochs")],
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="one"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={})

    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[0].held.wait(), HANG_GUARD_S)
    assert reasoning.script_prompts == []
    connect.set()
    await opening_done(page)
    reasoning.planning.clear()
    with line_seen("cursor.scene scene_id=clip") as entered:
        await pull_past(source, first)
        await asyncio.wait_for(entered.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.planning.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.plan_streams[1].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.scripted(3), HANG_GUARD_S)
    assert not boundary.is_set() and "epochs" in loop._lesson.scripting
    with line_seen("planner.accepted stage=boundary") as accepted:
        boundary.set()
        await asyncio.wait_for(accepted.wait(), HANG_GUARD_S)
    await asyncio.wait_for(asyncio.wait([turn_task("turn-2")]), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert scripted_titles(reasoning) == ["The ratio", "The clip", "Several epochs"]
    assert loop._lesson.plan.scenes[2].title == "Several epochs"
    assert loop._lesson.committed == {"ratio"}


async def test_aclose_ends_the_scripter() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(reasoning.scripted(1), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.script_streams[0].held.wait(), HANG_GUARD_S)
    scripter = scripter_task()
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)
    assert scripter.cancelled()
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert reasoning.script_streams[0].closed
    assert not [
        task for task in asyncio.all_tasks() if task.get_name().startswith(("lesson-", "turn-"))
    ]


async def test_a_script_call_that_raises_counts_as_a_failed_script(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        turns=[spoken_chunks(["go_on"])],
        plans=[planned_call(LESSON)],
        scripts=[spoken_chunks([""]), script_call("ratio")],
        script_gates=[gate],
        script_raises=RuntimeError("the script call broke"),
    )
    first = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([first, EndOfTurn(text="go on"), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={"clip": SCRIPTS["clip"]})

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
            running = asyncio.create_task(loop.run())
            await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
        opening = turn_task("turn-1")
        scripter = scripter_task()
        gate.set()
        await asyncio.wait_for(asyncio.wait([opening]), HANG_GUARD_S)
        await turn_after(source, first, "turn-2")
        assert not scripter.done()
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [NOT_READY_SAID, PREPARED_OPENING]
    messages = session_messages(caplog)
    assert messages.count("script.failed scene_id=ratio error=RuntimeError") == 1
    failed = messages.index("script.failed scene_id=ratio error=RuntimeError")
    assert failed < messages.index("reply.not_ready turn_id=turn-1 scene_id=ratio")
    assert not [message for message in messages if message.startswith("script.task_failed")]
    assert scripted_titles(reasoning) == ["The ratio"] * 2
    assert "rejected" not in reasoning.script_prompts[1].user_text
    assert loop._lesson.scripts["ratio"] == SCRIPTS["ratio"] and loop._lesson.unscripted == set()


async def test_a_scripter_that_fails_outside_the_script_call_still_speaks_not_ready(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    broke = RuntimeError("the window broke")

    def window(state: LessonState) -> Scene | None:
        raise broke

    monkeypatch.setattr(LessonState, "next_to_script", window)
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, [], speaker.received, plans=[planned_call(LESSON)])
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
            running = asyncio.create_task(loop.run())
            await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
        opening = turn_task("turn-1")
        await asyncio.wait_for(asyncio.wait([opening]), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert loop._scripter is not None and loop._scripter.exception() is broke
    assert reasoning.script_prompts == []
    assert speaker.utterances == [NOT_READY_SAID]
    messages = session_messages(caplog)
    assert "script.task_failed error=RuntimeError" in messages
    assert not [message for message in messages if message.startswith("script.failed")]
    assert "reply.not_ready turn_id=turn-1 scene_id=ratio" in messages


async def test_a_source_that_ends_during_the_opening_wait_ends_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio")],
        script_gates=[gate],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
            running = asyncio.create_task(loop.run())
            await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
        scripter = scripter_task()
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert scripter.cancelled() and not gate.is_set()
    assert "ratio" not in loop._lesson.scripts
    assert spoken_texts(speaker) == []
    closing = ("reply.not_ready", "script.failed")
    assert not [message for message in session_messages(caplog) if message.startswith(closing)]


async def test_aclose_during_the_opening_wait_speaks_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio")],
        script_gates=[asyncio.Event()],
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
            running = asyncio.create_task(loop.run())
            await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
        opening = turn_task("turn-1")
        scripter = scripter_task()
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)
        stop.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert scripter.cancelled() and opening.done()
    assert spoken_texts(speaker) == []
    closing = ("reply.not_ready", "script.failed")
    assert not [message for message in session_messages(caplog) if message.startswith(closing)]


async def test_a_barge_in_cancels_the_script_wait_and_not_the_scripter() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        [],
        speaker.received,
        plans=[planned_call(LESSON)],
        scripts=[script_call("ratio")],
        script_gates=[gate],
    )
    barge = asyncio.Event()
    stop = asyncio.Event()
    source = ScriptedSource([barge, SpeechStarted(), stop])
    loop = split_loop(log, source, speaker, reasoning, page, scripts={})

    with line_seen("script.waiting turn_id=turn-1 scene_id=ratio") as waiting:
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(waiting.wait(), HANG_GUARD_S)
    opening = turn_task("turn-1")
    scripter = scripter_task()
    await pull_past(source, barge)
    await asyncio.wait_for(asyncio.wait([opening]), HANG_GUARD_S)
    assert opening.cancelled()
    assert not scripter.done() and not scripter.cancelling()
    with line_seen("script.written scene_id=ratio", "tutor.script") as written:
        gate.set()
        await asyncio.wait_for(written.wait(), HANG_GUARD_S)
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert spoken_texts(speaker) == []
    assert loop._lesson.scripts["ratio"] == SCRIPTS["ratio"]


async def test_the_scripter_starts_in_every_planned_session_and_never_without_a_plan() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(
        log, [], speaker.received, plans=[planned_call(LESSON)], scripts=[script_call("ratio")]
    )
    stop = asyncio.Event()
    loop = split_loop(log, ScriptedSource([stop]), speaker, reasoning, page, scripts={})

    running = asyncio.create_task(loop.run())
    await opening_done(page)
    scripters = [task for task in asyncio.all_tasks() if task.get_name() == "lesson-scripter"]
    stop.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(scripters) == 1
    assert scripted_titles(reasoning)[0] == "The ratio"
    assert speaker.utterances == [PREPARED_OPENING]

    unplanned_log: list[tuple[str, object]] = []
    unplanned_speaker = FakeSpeaker(unplanned_log)
    unplanned_reasoning = FakeReasoning(
        unplanned_log, spoken_chunks(SPOKEN_DELTAS), unplanned_speaker.received
    )
    unplanned = concept_loop(
        unplanned_log, SerialSource([CONCEPT_TEXT]), unplanned_speaker, unplanned_reasoning
    )
    await asyncio.wait_for(unplanned.run(), HANG_GUARD_S)
    await asyncio.wait_for(unplanned.aclose(), HANG_GUARD_S)

    assert unplanned._scripter is None
    assert unplanned_reasoning.script_prompts == []
    assert unplanned_speaker.utterances == [SPOKEN_CLAUSES]
