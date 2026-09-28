import asyncio
import json
import logging
import re
import typing
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from tests.fakes import local_peer, numbered_frames
from tests.test_transport import FakeChannel
from tutor.brief import BRIEF_END, BRIEF_MARKER, VisualBrief
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
from tutor.lesson import Cursor, LessonPlan, Scene, Step
from tutor.pedagogy import PedagogyState, Phase, TurnOutcome
from tutor.prompt import SEARCH_CODE_TOOL, Message, TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.scene import SCENE_TOOLS
from tutor.session import (
    BAD_ARGUMENTS,
    CONCEPT_OUTCOME_INSTRUCTION,
    LESSON_SYNC_TIMEOUT_S,
    OUTCOME_INSTRUCTION,
    OUTCOME_MARKER,
    SPOKEN_DEPTH,
    TAIL_LIMIT,
    OutcomeSplitter,
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
VISUAL_TASK = "turn-1-visual"
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
TEACH_DIRECTIVE = PedagogyState(Phase.TEACH).prompt_directive()
CONCRETE_DIRECTIVE = PedagogyState(Phase.CONCRETE).prompt_directive()
INTERROGATE_DIRECTIVE = PedagogyState(Phase.INTERROGATE).prompt_directive()
COVERED_OUTCOME = OUTCOME_MARKER + json.dumps({"signal": "covered", "settling_positions": []})
MISCONCEPTION_OUTCOME = OUTCOME_MARKER + json.dumps(
    {
        "signal": "misconception",
        "settling_positions": [
            {"path": GROUNDED_PATH, "line": 24},
            {"path": SECOND_PATH, "line": 41},
        ],
    }
)
MALFORMED_OUTCOME = OUTCOME_MARKER + "not json at all"
SPLIT_OUTCOME_DELTAS = [
    *SPOKEN_DELTAS[:3],
    "the track.<ou",
    'tcome>{"signal": ',
    '"covered", "settling_positions": []}',
]
CONCEPT_TEXT = "teach me ppo"
CONCEPT_PARTIAL = "teach me"
STARTING_FROM = "I know policy gradients"
CONCEPT_DELTAS = ["We run 10 epochs ", "on 2048 samples. ", "Look in setup.py ", "for the flags."]
CONCEPT_CLAUSES = ["We run 10 epochs on 2048 samples.", "Look in setup.py for the flags."]
TOLD_OUTCOME = OUTCOME_MARKER + json.dumps({"signal": "told", "settling_positions": []})
SETTLING_TEXT = "past epsilon the objective is flat"
SETTLED_OUTCOME = OUTCOME_MARKER + json.dumps(
    {
        "signal": "misconception",
        "settling": SETTLING_TEXT,
        "settling_positions": [{"path": "ppo.py", "line": 12}],
    }
)
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
    return TurnLoopConfig(system=SYSTEM, subject=SUBJECT, root=root)


def concept_cfg() -> TurnLoopConfig:
    return TurnLoopConfig(system=SYSTEM, subject="PPO", starting_from=STARTING_FROM, root=None)


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


class AnsweringTransport(LoggingTransport):
    def __init__(
        self,
        log: list[tuple[str, object]],
        answers: dict[str, str],
        holds: dict[str, asyncio.Event] | None = None,
    ) -> None:
        super().__init__(log)
        self.answers = answers
        self.holds = holds if holds is not None else {}
        self.pushes: dict[str, asyncio.Event] = {}
        self.deliveries: list[asyncio.Task[None]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self._log.append(("send_json", payload))
        if payload["type"] != "scene.push":
            return
        scene_id = str(payload["scene_id"])
        self.pushes.setdefault(scene_id, asyncio.Event()).set()
        if scene_id not in self.answers:
            return
        error = self.answers[scene_id]
        report = {
            "type": "scene.ready",
            "scene_id": scene_id,
            "ok": error == "",
            "steps": len(payload["steps"]) if error == "" else 0,
            "error": error,
        }
        gate = self.holds.get(scene_id)
        if gate is None:
            asyncio.get_running_loop().call_soon(lambda: [h(report) for h in self.handlers])
            return

        async def deliver() -> None:
            await gate.wait()
            for handler in self.handlers:
                handler(report)

        self.deliveries.append(asyncio.create_task(deliver()))


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
    ) -> None:
        self._chunks = chunks
        self._log = log
        self._received = received
        self._gate = gate
        self._holds_at = holds_at
        self._close_gate = close_gate
        self.iterations = 0
        self.cancels = 0
        self.yielded = 0
        self.finish_reason: str | None = None
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
        finally:
            self._log.append(("stream_closed", self.yielded))
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
    ) -> None:
        self._log = log
        self._chunks = chunks
        self._follow_up = follow_up if follow_up is not None else []
        self._received = received
        self._gate = gate
        self._fails = fails
        self._holds_at = holds_at
        self._close_gate = close_gate
        self._turns = deque(turns) if turns is not None else deque()
        self._follow_up_gate = follow_up_gate
        self._visual = visual if visual is not None else []
        self._visual_finish = visual_finish
        self._visual_gates = deque(visual_gates) if visual_gates is not None else deque()
        self.prompts: list[TurnPrompt] = []
        self.tools: list[list[dict] | None] = []
        self.max_tokens: list[int | None] = []
        self.tool_choices: list[str | None] = []
        self.models: list[str | None] = []
        self.efforts: list[str | None] = []
        self.streams: list[FakeStream] = []
        self.started = asyncio.Event()
        self.visual_started = asyncio.Event()
        self.visual_callers: list[asyncio.Task[object] | None] = []

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
    ) -> FakeStream:
        self.prompts.append(prompt)
        self.tools.append(list(tools) if tools is not None else None)
        self.max_tokens.append(max_tokens)
        self.tool_choices.append(tool_choice)
        self.models.append(model)
        self.efforts.append(effort)
        self._log.append(("start_turn", prompt.user_text))
        if prompt.user_text == self._fails:
            raise RateLimited(RATE_LIMIT_DETAIL)
        if tool_choice is not None:
            gate = self._visual_gates.popleft() if self._visual_gates else None
            stream = FakeStream(self._visual, self._log, self._received, gate)
            stream.finish_reason = self._visual_finish
            self.streams.append(stream)
            self.visual_callers.append(asyncio.current_task())
            self.started.set()
            self.visual_started.set()
            return stream
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


def outcome_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [message for message in session_messages(caplog) if message.startswith("turn.outcome")]


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


def visual_task(name: str = VISUAL_TASK) -> asyncio.Task[str]:
    visuals = [task for task in asyncio.all_tasks() if task.get_name() == name]
    assert len(visuals) == 1
    return visuals[0]


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
    cfg = TurnLoopConfig(system=SYSTEM, subject=SUBJECT, root=root, speculative_reasoning=True)
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


async def test_the_prompt_carries_the_directive_for_the_current_phase(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
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

    system = reasoning.prompts[0].system
    assert SYSTEM in system
    assert SUBJECT in system
    assert TEACH_DIRECTIVE in system
    assert OUTCOME_MARKER in system
    assert CONCRETE_DIRECTIVE not in system
    assert INTERROGATE_DIRECTIVE not in system
    assert speaker.utterances == [SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_misconception_in_interrogate_scopes_the_next_turn_to_where_it_settles(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    turns = [
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, MISCONCEPTION_OUTCOME]),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    source = SerialSource([USER_TEXT] * 4)
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

    systems = [prompt.system for prompt in reasoning.prompts]
    assert len(systems) == 4
    assert TEACH_DIRECTIVE in systems[0]
    assert CONCRETE_DIRECTIVE in systems[1]
    assert INTERROGATE_DIRECTIVE in systems[2]
    assert CONCRETE_DIRECTIVE in systems[3]
    assert INTERROGATE_DIRECTIVE not in systems[3]
    assert all(SECOND_PATH not in system for system in systems[:3])
    assert search.calls == []
    assert f"Search {GROUNDED_PATH} line 24, {SECOND_PATH} line 41 first" in systems[3]
    assert outcome_lines(caplog) == [
        "turn.outcome turn_id=turn-1 signal=covered phase=concrete",
        "turn.outcome turn_id=turn-2 signal=covered phase=interrogate",
        "turn.outcome turn_id=turn-3 signal=misconception phase=concrete",
        "turn.outcome turn_id=turn-4 signal=None phase=concrete",
    ]
    assert speaker.utterances == [SPOKEN_CLAUSES] * 4
    spoken = [str(text) for name, text in log if name == "speak"]
    assert all(OUTCOME_MARKER not in text for text in spoken)
    assert all("signal" not in text and "{" not in text for text in spoken)
    await loop.aclose()


@pytest.mark.parametrize("path", ["", "!" + GROUNDED_PATH], ids=["empty", "negated"])
async def test_an_unusable_settling_path_never_reaches_the_directive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, path: str
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    outcome = OUTCOME_MARKER + json.dumps(
        {
            "signal": "misconception",
            "settling_positions": [{"path": GROUNDED_PATH, "line": 24}, {"path": path, "line": 1}],
        }
    )
    turns = [
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, outcome]),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    source = SerialSource([USER_TEXT] * 4)
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
    assert all(GROUNDED_PATH not in prompt.system for prompt in reasoning.prompts)
    assert outcome_lines(caplog) == [
        "turn.outcome turn_id=turn-1 signal=covered phase=concrete",
        "turn.outcome turn_id=turn-2 signal=covered phase=interrogate",
        "turn.outcome turn_id=turn-3 signal=None phase=interrogate",
        "turn.outcome turn_id=turn-4 signal=None phase=interrogate",
    ]
    assert INTERROGATE_DIRECTIVE in reasoning.prompts[3].system
    assert speaker.utterances == [SPOKEN_CLAUSES] * 4
    await loop.aclose()


async def test_a_marker_split_across_deltas_is_still_stripped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPLIT_OUTCOME_DELTAS), speaker.received)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    registry = RecordingRegistry(log)
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

    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert outcome_lines(caplog) == ["turn.outcome turn_id=turn-1 signal=covered phase=concrete"]
    verified = [text for _, text, _ in registry.verified]
    assert verified
    assert all("<" not in text and "{" not in text and "signal" not in text for text in verified)
    await loop.aclose()


async def test_a_malformed_outcome_holds_the_phase_and_finishes_the_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    turns = [spoken_chunks([*SPOKEN_DELTAS, MALFORMED_OUTCOME]), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    source = SerialSource([USER_TEXT, USER_TEXT])
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

    spoken = [message for message in spoken_spans(caplog) if "turn_id=turn-1 " in message]
    assert len(spoken) == 1
    assert "turn.failed turn_id=turn-1" not in "\n".join(session_messages(caplog))
    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert all("not json" not in str(text) for name, text in log if name == "speak")
    assert outcome_lines(caplog) == [
        "turn.outcome turn_id=turn-1 signal=None phase=teach",
        "turn.outcome turn_id=turn-2 signal=None phase=teach",
    ]
    assert TEACH_DIRECTIVE in reasoning.prompts[1].system
    await loop.aclose()


async def test_a_reply_without_a_marker_speaks_everything_and_holds_the_phase(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    source = SerialSource([USER_TEXT, USER_TEXT])
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

    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert outcome_lines(caplog) == [
        "turn.outcome turn_id=turn-1 signal=None phase=teach",
        "turn.outcome turn_id=turn-2 signal=None phase=teach",
    ]
    assert [TEACH_DIRECTIVE in prompt.system for prompt in reasoning.prompts] == [True, True]
    await loop.aclose()


async def test_a_barge_in_leaves_the_phase_where_it_was(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    search = FakeSearch(log, found())
    gate = asyncio.Event()
    deltas = spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME])
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
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
        turn = turn_task()

        await pull_past(source, barge)
        gate.set()
        hold.set()
        await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)
        assert turn.cancelled()
        assert outcome_lines(caplog) == []

        resume.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert outcome_lines(caplog) == ["turn.outcome turn_id=turn-2 signal=covered phase=concrete"]
    assert [TEACH_DIRECTIVE in prompt.system for prompt in reasoning.prompts] == [True, True]
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]], SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_discarded_speculation_never_advances_the_phase(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    close_gate = asyncio.Event()
    speculative_deltas = [*SPOKEN_DELTAS, COVERED_OUTCOME]
    turns = [
        spoken_chunks(speculative_deltas),
        spoken_chunks(SPOKEN_DELTAS),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, close_gate=close_gate, turns=turns)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=PARTIAL_TEXT),
            endpoint,
            EndOfTurn(text=SECOND_TEXT),
            speaker.finished,
            EndOfTurn(text=USER_TEXT),
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
        await asyncio.wait_for(reasoning.streams[0].closing.wait(), HANG_GUARD_S)
        speculation = speculation_task()
        assert ("stream_closed", len(speculative_deltas)) in log

        await pull_past(source, endpoint)
        await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
        assert speculation.cancelled()
        close_gate.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert ("stream_released", len(speculative_deltas)) not in log
    assert [entry for entry in log if entry[0] == "start_turn"] == [
        ("start_turn", PARTIAL_TEXT),
        ("start_turn", SECOND_TEXT),
        ("start_turn", USER_TEXT),
    ]
    assert outcome_lines(caplog) == [
        "turn.outcome turn_id=turn-1 signal=None phase=teach",
        "turn.outcome turn_id=turn-2 signal=None phase=teach",
    ]
    assert [TEACH_DIRECTIVE in prompt.system for prompt in reasoning.prompts] == [True] * 3
    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert search.calls == []
    await loop.aclose()


async def test_a_misconception_in_the_follow_up_is_the_turns_outcome(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    turns = [
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        [SEARCH_CALL],
        spoken_chunks(SPOKEN_DELTAS),
    ]
    follow_up = spoken_chunks([*SPOKEN_DELTAS, MISCONCEPTION_OUTCOME])
    reasoning = FakeReasoning(log, [], speaker.received, follow_up=follow_up, turns=turns)
    source = SerialSource([USER_TEXT] * 4)
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
    assert reasoning.tools == [
        [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS],
        [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS],
        [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS],
        None,
        [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS],
    ]
    assert outcome_lines(caplog) == [
        "turn.outcome turn_id=turn-1 signal=covered phase=concrete",
        "turn.outcome turn_id=turn-2 signal=covered phase=interrogate",
        "turn.outcome turn_id=turn-3 signal=misconception phase=concrete",
        "turn.outcome turn_id=turn-4 signal=None phase=concrete",
    ]
    assert CONCRETE_DIRECTIVE in reasoning.prompts[4].system
    assert (
        f"Search {GROUNDED_PATH} line 24, {SECOND_PATH} line 41 first"
        in reasoning.prompts[4].system
    )
    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [
        SPOKEN_CLAUSES,
        SPOKEN_CLAUSES,
        [lead_in, *SPOKEN_CLAUSES],
        SPOKEN_CLAUSES,
    ]
    spoken = [str(text) for name, text in log if name == "speak"]
    assert all(OUTCOME_MARKER not in text and "{" not in text for text in spoken)
    await loop.aclose()


def states(log: list[tuple[str, object]]) -> list[str]:
    return [str(payload["state"]) for payload in sent(log, "state")]


def outcome_signals() -> list[str]:
    annotation = TurnOutcome.model_fields["signal"].annotation
    literal = next(member for member in typing.get_args(annotation) if member is not type(None))
    names = list(typing.get_args(literal))
    assert names
    return names


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

    assert [(p["state"], p["phase"]) for p in sent(log, "state")] == [
        ("thinking", "teach"),
        ("speaking", "teach"),
        ("listening", "teach"),
    ]
    speaking = next(
        i for i, (n, p) in enumerate(log) if n == "send_json" and p.get("state") == "speaking"
    )
    assert speaking < log.index(("speak", SPOKEN_CLAUSES[0]))
    await loop.aclose()


async def test_the_state_carries_the_phase_after_it_moves() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    turns = [spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
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

    phases = [p["phase"] for p in sent(log, "state")]
    assert phases == ["teach", "teach", "concrete", "concrete", "concrete", "concrete"]
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


def test_the_outcome_instruction_names_every_signal_the_outcome_accepts() -> None:
    for name in outcome_signals():
        assert name in OUTCOME_INSTRUCTION


def test_the_concept_instruction_names_every_signal_and_no_position() -> None:
    for name in outcome_signals():
        assert name in CONCEPT_OUTCOME_INSTRUCTION
    assert "settling_positions" not in CONCEPT_OUTCOME_INSTRUCTION


async def test_a_concept_turn_drops_a_settling_position_before_the_next_directive() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    turns = [
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, SETTLED_OUTCOME]),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    source = SerialSource(
        [CONCEPT_TEXT, "why clip", "quiz me", "the clip makes the gradient larger"]
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

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    systems = [prompt.system for prompt in reasoning.prompts]
    assert CONCEPT_OUTCOME_INSTRUCTION in systems[0]
    assert "settling_positions" not in systems[0]
    assert INTERROGATE_DIRECTIVE in systems[2]
    assert f"Settle the misconception here: {SETTLING_TEXT}" in systems[3]
    assert "ppo.py" not in systems[3]
    assert search.calls == []
    await loop.aclose()


async def test_a_told_signal_in_interrogate_returns_the_next_turn_to_teach(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    turns = [
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, COVERED_OUTCOME]),
        spoken_chunks([*SPOKEN_DELTAS, TOLD_OUTCOME]),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, turns=turns)
    source = SerialSource([CONCEPT_TEXT, "quiz me", "just tell me", "go on"])
    loop = TurnLoop(
        concept_cfg(),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert "turn.outcome turn_id=turn-3 signal=told phase=teach" in outcome_lines(caplog)
    assert TEACH_DIRECTIVE in reasoning.prompts[3].system
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


def test_the_splitter_releases_a_held_prefix_that_is_not_the_marker() -> None:
    splitter = OutcomeSplitter()
    assert splitter.feed("the reader <") == "the reader "
    assert splitter.feed("3 drains <ou") == "<3 drains "
    assert splitter.feed("r track.") == "<our track."
    assert splitter.feed("done <outc") == "done "
    assert splitter.finish() == ("<outc", TurnOutcome())


def test_the_splitter_holds_everything_after_the_marker() -> None:
    splitter = OutcomeSplitter()
    assert splitter.feed('the track.<outcome>{"signal": ') == "the track."
    assert splitter.feed('"correct", "settling_positions": []}\n') == ""
    text, outcome = splitter.finish()
    assert text == ""
    assert outcome == TurnOutcome(signal="correct")


BRIEF_JSON = VisualBrief(
    kind="app", title="Clipped objective", show="surrogate vs ratio"
).model_dump_json()
BRIEF_HEAD = BRIEF_MARKER + BRIEF_JSON + BRIEF_END
NONE_HEAD = BRIEF_MARKER + VisualBrief(kind="none", title="", show="").model_dump_json() + BRIEF_END
BRIEFED_DELTAS = [BRIEF_HEAD[:12], BRIEF_HEAD[12:], "\n", *SPOKEN_DELTAS]
SCENE_STEPS = ["The ratio axis", "The clip band", "The flat region"]
SCENE_HTML = "<!doctype html><p>x</p>"
SCENE_CALL = visual_call(
    "write_scene", json.dumps({"html": SCENE_HTML, "steps": SCENE_STEPS}), "call-v"
)
APP_TITLE = "Clipped objective"
SECOND_CONCEPT_TEXT = "and the ratio"


def visual_user_text(learner: str) -> str:
    return "Subject: PPO\nTitle: Clipped objective\nShow: surrogate vs ratio"


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


def pending(log: list[tuple[str, object]], turn_id: str) -> list[str]:
    return [str(p["title"]) for p in sent(log, "visual.pending") if p["turn_id"] == turn_id]


def shown(log: list[tuple[str, object]]) -> list[str]:
    return [str(p["scene_id"]) for p in sent(log, "scene.show")]


def pushed(log: list[tuple[str, object]]) -> list[str]:
    return [str(p["scene_id"]) for p in sent(log, "scene.push")]


async def test_a_validated_brief_announces_the_pending_visual_before_the_call() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pending(log, "turn-1") == [APP_TITLE]
    (announced,) = sent(log, "visual.pending")
    assert log.index(("start_turn", CONCEPT_TEXT)) < log.index(("send_json", announced))
    assert log.index(("send_json", announced)) < log.index(
        ("start_turn", visual_user_text(CONCEPT_TEXT))
    )
    assert pushed(log) == ["turn-1"]
    assert log.index(("send_json", sent(log, "scene.push")[0])) < log.index(
        ("send_json", sent(log, "scene.show")[0])
    )
    assert shown(log) == ["turn-1"]


async def test_the_visual_call_uses_the_configured_scene_model_and_the_voice_call_does_not() -> (
    None
):
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    cfg = concept_cfg().model_copy(update={"scene_model": "draw-1"})
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, cfg=cfg)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.models == [None, "draw-1"]


async def test_an_empty_scene_model_leaves_the_visual_call_on_the_reasoning_model() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.models == [None, None]


async def test_kind_none_announces_no_pending_visual() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    deltas = spoken_chunks([NONE_HEAD, *SPOKEN_DELTAS])
    reasoning = FakeReasoning(log, deltas, speaker.received, visual=[SCENE_CALL])
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert sent(log, "visual.pending") == []


async def test_a_truncated_visual_clears_the_pending_title() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    cut = visual_call("write_scene", SCENE_CALL.text[:40], "call-v")
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[cut], visual_finish="length"
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pending(log, "turn-1") == [APP_TITLE, ""]
    assert pushed(log) == []


async def test_a_build_that_ends_at_the_cap_with_nothing_clears_the_title_and_leaves_one_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[], visual_finish="length"
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    with caplog.at_level(logging.INFO, logger="tutor.scene"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        (visual,) = reasoning.visual_callers
        assert visual is not None
        await asyncio.wait_for(visual, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert visual.result() == "scene: error: empty reply"
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    assert pushed(log) == []
    assert speaker.utterances == [SPOKEN_CLAUSES]
    scene_lines = [r.getMessage() for r in caplog.records if r.name == "tutor.scene"]
    assert scene_lines == ["scene.empty finish=length"]


class ClosedSettleTransport(LoggingTransport):
    async def send_json(self, payload: dict[str, object]) -> None:
        if payload["type"] == "visual.pending" and payload["title"] == "":
            raise ChannelClosed()
        self._log.append(("send_json", payload))


async def test_a_pending_clear_on_a_closed_channel_is_a_warning_not_a_second_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        fails=visual_user_text(CONCEPT_TEXT),
        visual=[SCENE_CALL],
    )
    loop = concept_loop(
        log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=ClosedSettleTransport(log)
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert pending(log, "turn-1") == [APP_TITLE]
    messages = session_messages(caplog)
    assert "visual.pending_push_failed turn_id=turn-1 error=ChannelClosed" in messages
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == ["scene.failed turn_id=turn-1 error=RateLimited"]


async def test_a_brief_head_starts_the_visual_call_and_is_never_spoken() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [SPOKEN_CLAUSES]
    spoken = [str(text) for name, text in log if name == "speak"]
    captions = [str(payload["text"]) for payload in sent(log, "caption")]
    assert captions == SPOKEN_CLAUSES
    assert all(BRIEF_MARKER not in text for text in [*spoken, *captions])
    assert reasoning.tool_choices == [None, "required"]
    assert reasoning.tools[1] == SCENE_TOOLS
    assert reasoning.max_tokens[1] == 128000
    assert reasoning.efforts[1] == "high"
    assert shown(log) == ["turn-1"]
    assert reasoning.prompts[1].history == []
    assert reasoning.prompts[1].tool_context == []
    assert reasoning.prompts[1].user_text == visual_user_text(CONCEPT_TEXT)
    (push,) = sent(log, "scene.push")
    assert push["title"] == APP_TITLE and push["html"] == SCENE_HTML
    assert push["steps"] == SCENE_STEPS
    (show,) = sent(log, "scene.show")
    assert show["at"] == 1 and show["seq"] > push["seq"]


async def test_kind_none_starts_no_visual_call() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    deltas = spoken_chunks([NONE_HEAD, *SPOKEN_DELTAS])
    reasoning = FakeReasoning(log, deltas, speaker.received, visual=[SCENE_CALL])
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.prompts) == 1
    assert shown(log) == []
    assert speaker.utterances == [SPOKEN_CLAUSES]


async def test_a_missing_head_speaks_everything_and_starts_no_visual_call() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.prompts) == 1
    assert shown(log) == []
    assert speaker.utterances == [SPOKEN_CLAUSES]


async def test_the_visual_call_lands_after_the_turn_has_ended() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[gate],
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[1].held.wait(), HANG_GUARD_S)
    visual = visual_task()

    assert not visual.done()
    assert shown(log) == []
    assert speaker.utterances == [SPOKEN_CLAUSES]

    gate.set()
    await asyncio.wait_for(visual, HANG_GUARD_S)

    assert shown(log) == ["turn-1"]
    assert pending(log, "turn-1") == [APP_TITLE]
    await loop.aclose()


async def test_a_visual_that_finishes_after_a_newer_one_has_landed_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[gate],
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        first = visual_task()
        assert not first.done()
        assert len(shown(log)) == 1
        log.append(("release", None))
        gate.set()
        await asyncio.wait_for(first, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    (push,) = sent(log, "scene.show")
    assert log.index(("send_json", push)) < log.index(("release", None))
    assert "scene.superseded turn_id=turn-1" in session_messages(caplog)
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    assert pending(log, "turn-2") == [APP_TITLE]


async def test_an_older_visual_lands_while_the_newer_one_is_still_drawing() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[first_gate, second_gate],
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    first = visual_task()
    second = visual_task("turn-2-visual")
    assert shown(log) == []

    first_gate.set()
    await asyncio.wait_for(first, HANG_GUARD_S)
    assert len(shown(log)) == 1
    assert not second.done()

    second_gate.set()
    await asyncio.wait_for(second, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert shown(log) == ["turn-1", "turn-2"]
    shows = sent(log, "scene.show")
    assert shows[0]["seq"] < shows[1]["seq"]
    assert pending(log, "turn-1") == [APP_TITLE]
    assert pending(log, "turn-2") == [APP_TITLE]


async def test_a_visual_call_error_is_one_log_line_and_the_voice_is_untouched(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        fails=visual_user_text(CONCEPT_TEXT),
        visual=[SCENE_CALL],
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [SPOKEN_CLAUSES]
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == ["scene.failed turn_id=turn-1 error=RateLimited"]
    assert shown(log) == []
    assert pending(log, "turn-1") == [APP_TITLE, ""]


async def test_a_visual_call_past_the_timeout_is_cancelled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[gate],
    )
    cfg = concept_cfg().model_copy(update={"scene_timeout_s": 0.01})
    loop = concept_loop(
        log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning, cfg=cfg
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())
        await asyncio.wait_for(reasoning.visual_started.wait(), HANG_GUARD_S)
        # The task is looked up through the fake, not by name among pending tasks, because
        # the timeout may already have fired by the time this coroutine resumes.
        (visual,) = reasoning.visual_callers
        assert visual is not None and visual.get_name() == VISUAL_TASK
        await asyncio.wait_for(visual, HANG_GUARD_S)
        await asyncio.wait_for(running, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert visual.result() == "scene: timeout"
    assert ("stream_closed", 0) in log
    assert "scene.timeout turn_id=turn-1" in session_messages(caplog)
    assert ("open_turn", "turn-2") in log
    assert speaker.utterances == [SPOKEN_CLAUSES, SPOKEN_CLAUSES]
    assert shown(log) == ["turn-2"]
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    assert pending(log, "turn-2") == [APP_TITLE]


async def test_a_barge_in_cancels_the_interrupted_turns_visual_call() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    visual_gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        gate=gate,
        holds_at=3,
        visual=[SCENE_CALL],
        visual_gates=[visual_gate],
    )
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource([EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume])
    loop = concept_loop(log, source, speaker, reasoning)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.visual_started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[1].held.wait(), HANG_GUARD_S)
    turn = turn_task()
    drain = drain_task()
    visual = visual_task()

    await pull_past(source, barge)
    gate.set()
    await asyncio.wait_for(asyncio.wait([turn, visual]), HANG_GUARD_S)

    assert turn.cancelled()
    assert drain.cancelled()
    assert visual.cancelled()
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    assert shown(log) == []
    assert ("abandon", "turn-1") in log

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await loop.aclose()
    assert shown(log) == []


@pytest.mark.parametrize("matches", [True, False], ids=["matching", "mismatching"])
async def test_a_speculative_brief_waits_for_the_claim(matches: bool) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    turns = [spoken_chunks(BRIEFED_DELTAS), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(
        log, [], speaker.received, gate=gate, holds_at=3, turns=turns, visual=[SCENE_CALL]
    )
    final = CONCEPT_TEXT if matches else "two"
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [PartialTranscript(text=CONCEPT_PARTIAL), endpoint, EndOfTurn(text=final), speaker.finished]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    assert reasoning.tool_choices == [None]
    reasoning.started.clear()

    await pull_past(source, endpoint)
    if matches:
        await asyncio.wait_for(reasoning.visual_started.wait(), HANG_GUARD_S)
        assert reasoning.tool_choices == [None, "required"]
    else:
        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        assert reasoning.tool_choices == [None, None]
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    if matches:
        assert reasoning.tool_choices == [None, "required"]
        assert shown(log) == ["turn-1"]
        assert reasoning.prompts[1].user_text == visual_user_text(CONCEPT_PARTIAL)
    else:
        assert reasoning.tool_choices == [None, None]
        assert shown(log) == []
    assert speaker.utterances == [SPOKEN_CLAUSES]


@pytest.mark.parametrize("completes", [False, True], ids=["open", "completed"])
async def test_a_mismatched_speculation_leaves_no_brief_behind(completes: bool) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    holds_at = len(BRIEFED_DELTAS) - 1 if completes else 3
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        gate=gate,
        holds_at=holds_at,
        visual=[SCENE_CALL],
    )
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [PartialTranscript(text=CONCEPT_PARTIAL), endpoint, EndOfTurn(text="two"), speaker.finished]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    speculation = speculation_task()
    assert reasoning.tool_choices == [None]
    if completes:
        gate.set()
        await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
        assert not speculation.cancelled() and speculation.exception() is None
        assert reasoning.tool_choices == [None]

    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, None, "required"]
    assert [prompt.user_text for prompt in reasoning.prompts] == [
        CONCEPT_PARTIAL,
        "two",
        visual_user_text("two"),
    ]
    assert shown(log) == ["turn-1"]
    assert speaker.utterances == [SPOKEN_CLAUSES]


async def test_the_next_turn_sees_the_brief_head_in_its_history() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    deltas = spoken_chunks([NONE_HEAD, "\n", *SPOKEN_DELTAS])
    reasoning = FakeReasoning(log, deltas, speaker.received, visual=[SCENE_CALL])
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, None]
    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=NONE_HEAD + "\n" + " ".join(SPOKEN_CLAUSES)),
    ]


@pytest.mark.parametrize("completes", [False, True], ids=["open", "completed"])
async def test_a_mismatched_speculation_leaves_no_head_in_the_transcript(completes: bool) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    holds_at = len(BRIEFED_DELTAS) - 1 if completes else 3
    turns = [
        spoken_chunks(BRIEFED_DELTAS),
        spoken_chunks(SPOKEN_DELTAS),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(
        log, [], speaker.received, gate=gate, holds_at=holds_at, turns=turns, visual=[SCENE_CALL]
    )
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=CONCEPT_PARTIAL),
            endpoint,
            EndOfTurn(text="two"),
            speaker.finished,
            EndOfTurn(text="three"),
        ]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    speculation = speculation_task()
    assert reasoning.tool_choices == [None]
    if completes:
        gate.set()
        await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
        assert not speculation.cancelled() and speculation.exception() is None
        assert reasoning.tool_choices == [None]

    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, None, None]
    assert [prompt.user_text for prompt in reasoning.prompts] == [CONCEPT_PARTIAL, "two", "three"]
    assert reasoning.prompts[2].history == [
        Message(role="user", content="two"),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]
    assert shown(log) == []


async def test_a_missing_head_leaves_the_history_as_before() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, None]
    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]


async def test_a_head_in_both_rounds_starts_one_visual(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    deltas = [TurnChunk(kind="spoken", text=BRIEF_HEAD), SEARCH_CALL]
    reasoning = FakeReasoning(
        log, deltas, speaker.received, follow_up=spoken_chunks(BRIEFED_DELTAS), visual=[SCENE_CALL]
    )
    loop = TurnLoop(
        config(tmp_path),
        SerialSource([USER_TEXT]),
        FakeSearch(log, result),
        speaker,
        CheckingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, None, "required"]
    assert shown(log) == ["turn-1"]
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]


async def test_a_head_only_in_the_follow_up_still_draws(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log,
        [SEARCH_CALL],
        speaker.received,
        follow_up=spoken_chunks(BRIEFED_DELTAS),
        visual=[SCENE_CALL],
    )
    loop = TurnLoop(
        config(tmp_path),
        SerialSource([USER_TEXT]),
        FakeSearch(log, result),
        speaker,
        CheckingTransport(log),
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, None, "required"]
    assert shown(log) == ["turn-1"]
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]


@pytest.mark.parametrize("theme", ["dark", "sepia", ["dark"]])
async def test_a_theme_message_reaches_the_visual_prompt(theme: object) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    transport = CheckingTransport(log)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=transport)
    (handler,) = transport.handlers
    handler({"type": "theme", "theme": theme})

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    system = reasoning.prompts[1].system
    assert ("The page is dark" in system) == (theme == "dark")
    assert ("The page is light" in system) == (theme != "dark")


async def test_an_accepted_theme_is_logged_and_a_rejected_one_is_not(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    transport = CheckingTransport(log)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=transport)
    (handler,) = transport.handlers

    with caplog.at_level(logging.DEBUG, logger="tutor.session"):
        handler({"type": "theme", "theme": "sepia"})
        handler({"type": "theme", "theme": "dark"})

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    messages = session_messages(caplog)
    assert [line for line in messages if line.startswith("session.theme")] == [
        "session.theme theme=dark"
    ]
    assert [line for line in messages if line.startswith("client.message_rejected")] == [
        "client.message_rejected type=theme"
    ]
    assert all("sepia" not in line for line in messages)


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


async def test_aclose_cancels_a_visual_call_in_flight() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[gate],
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    visual = visual_task()
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert visual.cancelled()
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    assert shown(log) == []


async def test_a_failed_check_retries_once_with_the_error_and_then_gives_up(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    transport = CheckingTransport(log, ok=False, error="timeline lacks labels step-3")
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=transport)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.tool_choices == [None, "required", "required"]
    assert "timeline lacks labels step-3" not in reasoning.prompts[1].user_text
    assert reasoning.prompts[2].user_text.endswith(
        "The previous attempt failed its check: timeline lacks labels step-3\n"
        "Write the whole scene again."
    )
    assert pushed(log) == ["turn-1", "turn-1"]
    assert shown(log) == []
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    messages = session_messages(caplog)
    assert "scene.check_failed turn_id=turn-1 attempt=1 steps=0 chars=28" in messages
    assert "scene.check_failed turn_id=turn-1 attempt=2 steps=0 chars=28" in messages
    assert "scene.shown" not in " ".join(messages)


async def test_a_check_the_page_never_answers_fails_within_its_own_bound(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    cfg = concept_cfg().model_copy(update={"scene_ready_timeout_s": 0.01})
    loop = concept_loop(
        log,
        SerialSource([CONCEPT_TEXT]),
        speaker,
        reasoning,
        cfg=cfg,
        transport=LoggingTransport(log),
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        visual = reasoning.visual_callers[0]
        assert visual is not None and visual.get_name() == VISUAL_TASK
        await asyncio.wait_for(visual, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert visual.result() == "scene: error: no report"
    assert reasoning.tool_choices == [None, "required"]
    assert pushed(log) == ["turn-1"]
    assert shown(log) == []
    assert pending(log, "turn-1") == [APP_TITLE, ""]
    messages = session_messages(caplog)
    assert "scene.no_report turn_id=turn-1 attempt=1" in messages
    assert "scene.check_failed" not in " ".join(messages)
    assert loop._ready == {}


async def test_a_push_the_page_replaced_is_superseded_without_a_second_build(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[first_gate, second_gate],
    )
    cfg = concept_cfg().model_copy(update={"scene_ready_timeout_s": 0.2})
    transport = AnsweringTransport(log, answers={"turn-2": ""})
    loop = concept_loop(
        log,
        SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]),
        speaker,
        reasoning,
        cfg=cfg,
        transport=transport,
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        first = visual_task()
        second = visual_task("turn-2-visual")
        first_gate.set()
        await asyncio.wait_for(
            transport.pushes.setdefault("turn-1", asyncio.Event()).wait(), HANG_GUARD_S
        )
        assert pushed(log) == ["turn-1"]
        second_gate.set()
        await asyncio.wait_for(second, HANG_GUARD_S)
        assert shown(log) == ["turn-2"]
        await asyncio.wait_for(first, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert first.result() == "scene: superseded"
    assert reasoning.tool_choices.count("required") == 2
    assert pushed(log) == ["turn-1", "turn-2"]
    messages = session_messages(caplog)
    assert "scene.no_report turn_id=turn-1 attempt=1" in messages
    assert "scene.superseded turn_id=turn-1" in messages
    assert loop._ready == {}


async def test_a_failed_check_is_not_rebuilt_once_a_newer_scene_has_landed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[first_gate, second_gate],
    )
    verdict = asyncio.Event()
    transport = AnsweringTransport(
        log,
        answers={"turn-1": "timeline lacks labels step-3", "turn-2": ""},
        holds={"turn-1": verdict},
    )
    loop = concept_loop(
        log,
        SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]),
        speaker,
        reasoning,
        transport=transport,
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        first = visual_task()
        second = visual_task("turn-2-visual")
        first_gate.set()
        await asyncio.wait_for(
            transport.pushes.setdefault("turn-1", asyncio.Event()).wait(), HANG_GUARD_S
        )
        assert pushed(log) == ["turn-1"]
        second_gate.set()
        await asyncio.wait_for(second, HANG_GUARD_S)
        assert shown(log) == ["turn-2"]
        verdict.set()
        await asyncio.wait_for(first, HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert first.result() == "scene: superseded"
    assert reasoning.tool_choices.count("required") == 2
    assert pushed(log) == ["turn-1", "turn-2"]
    assert shown(log) == ["turn-2"]
    assert "scene.check_failed turn_id=turn-1 attempt=1 steps=0 chars=28" in session_messages(
        caplog
    )
    assert pending(log, "turn-1") == [APP_TITLE, ""]


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
        self, log: list[tuple[str, object]], fires: bool = False, syncs: bool = True
    ) -> None:
        super().__init__(log)
        self.fires = fires
        self.syncs = syncs
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
CUED_DELTAS = [
    "<scene 1>The ratio compares two policies. ",
    "They agree at one. ",
    "<step 2>At one action ",
    "it is a single number.",
]
SCENE_DELTAS = ["<scene 1>The ratio compares two policies. ", "They agree at one."]


def spoken_texts(speaker: FakeSpeaker) -> list[str]:
    return [text for utterance in speaker.utterances for text in utterance]


class TimedSpeaker(FakeSpeaker):
    async def speak(self, chunks: AsyncIterator[Chunk], on_play: OnPlay) -> None:
        async def timed(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
            await on_play(chunk, 300, 900)

        await super().speak(chunks, timed)


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
    assert ("start_turn", SECOND_TEXT) not in log

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.answer_sync()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert log.index(("start_turn", SECOND_TEXT)) > log.index(("send_json", sync))
    assert speaker.utterances[1] == ["Back to the ratio then."]
    assert loop._lesson.sent == []
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
    assert ("start_turn", SECOND_TEXT) not in log

    with caplog.at_level(logging.WARNING, logger="tutor.session"):
        pace.release_once()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert "lesson.sync_timeout epoch=1 barrier=1" in session_messages(caplog)
    assert loop._lesson.sent == []
    assert loop._lesson.acked == Cursor(scene=0, step=0)
    assert ("start_turn", SECOND_TEXT) in log
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
        assert ("start_turn", SECOND_TEXT) not in log
        page.answer_sync()
        await asyncio.wait_for(running, HANG_GUARD_S)

    messages = session_messages(caplog)
    assert "lesson.synced_ignored epoch=1 barrier=1" in messages
    assert "lesson.synced epoch=1 barrier=2 scene_id=None step=0 revision=0 last_cue=0" in messages
    assert ("start_turn", SECOND_TEXT) in log
    assert loop._lesson.sent == []
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
    assert loop._lesson.scene_tag(1) is None
    running = asyncio.create_task(loop.run())
    await asyncio.wait_for(page.arrival("lesson.attach").wait(), HANG_GUARD_S)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply(page_ack(1, "failed", "runtime", None, 0, 0))
        (sync,) = sent(log, "lesson.sync")
        assert log[-1] == ("send_json", sync)
        assert without_seq(sync) == {"type": "lesson.sync", "epoch": 1, "barrier": 1}
        assert loop._lesson.resync and loop._lesson.sent == []
        assert loop._lesson.dropped == ["<scene 1>: runtime"]

        page.arrival("transcript").clear()
        await pull_past(source, after)
        await asyncio.wait_for(page.arrival("transcript").wait(), HANG_GUARD_S)
        assert ("start_turn", CONCEPT_TEXT) not in log
        page.answer_sync()
        await asyncio.wait_for(asyncio.wait([turn_task()]), HANG_GUARD_S)

    assert ("start_turn", CONCEPT_TEXT) in log
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
        assert ("start_turn", CONCEPT_TEXT) not in log
        page.scene_id, page.step, page.revision = "ratio", 1, 3
        page.answer_sync()
        await asyncio.wait_for(asyncio.wait([turn_task()]), HANG_GUARD_S)

    assert "lesson.sync epoch=1 barrier=2 waiting=True reason=barge_in" in session_messages(caplog)
    assert ("start_turn", CONCEPT_TEXT) in log
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
    page.reply(RESTORE)
    assert [without_seq(sync) for sync in sent(log, "lesson.sync")] == [
        {"type": "lesson.sync", "epoch": 1, "barrier": 1}
    ]

    await pull_past(source, restored)
    assert ("open_turn", TURN_TASK) not in log
    assert reasoning.prompts == []

    log.append(("synced", None))
    page.scene_id, page.step, page.revision = "ratio", 1, 3
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


async def test_tags_never_reach_the_speaker_and_every_tag_drops_without_a_plan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, spoken_chunks(TAGGED_DELTAS), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    spoken = spoken_texts(speaker)
    assert spoken and not any("<" in text or ">" in text for text in spoken)
    assert "They agree at one." in " ".join(spoken)
    messages = session_messages(caplog)
    assert loop._lesson.dropped == ["<scene 1>: no_plan", "<step 2>: no_plan"]
    assert any(
        message.startswith("tag.dropped turn_id=turn-1 kind=set reason=unsupported")
        for message in messages
    )
    assert any(
        message.startswith("tag.dropped turn_id=turn-1 kind=step reason=malformed")
        for message in messages
    )
    assert sent(log, "lesson.cue") == []
    assert loop._lesson.sent == []


async def test_each_cue_goes_out_when_the_chunk_after_its_tag_plays() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, spoken_chunks(CUED_DELTAS), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)
    loop._lesson.adopt(LESSON)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    opening, step = sent(log, "lesson.cue")
    assert opening["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}
    assert step["tag"] == {"kind": "step", "n": 2}
    assert [
        (c["cue_id"], c["scene_id"], c["revision"], c["barrier"], c["epoch"])
        for c in (opening, step)
    ] == [(1, None, 0, 0, 1), (2, "ratio", 1, 0, 1)]
    captions = sent(log, "caption")
    assert log.index(("send_json", opening)) == log.index(("send_json", captions[0])) + 1
    (carrier,) = [caption for caption in captions if caption["text"].startswith("At one action")]
    assert log.index(("send_json", step)) == log.index(("send_json", carrier)) + 1
    assert opening["chunk_id"] < step["chunk_id"]
    assert [entry.cue_id for entry in loop._lesson.sent] == [1, 2]


async def test_a_fired_ack_moves_the_cursor_and_logs_the_position(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log, fires=True)
    reasoning = FakeReasoning(log, spoken_chunks(CUED_DELTAS), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)
    loop._lesson.adopt(LESSON)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert loop._lesson.acked == Cursor(scene=1, step=2)
    assert loop._lesson.sent == []
    messages = session_messages(caplog)
    assert "cursor.scene scene_id=ratio n=1 cue_id=1" in messages
    assert "cursor.step scene_id=ratio n=2 cue_id=2" in messages


async def test_a_fired_ack_the_state_refuses_moves_nothing_and_logs_no_cursor_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), speaker.received)
    loop = concept_loop(log, ScriptedSource([]), speaker, reasoning, transport=page)
    loop._lesson.adopt(LESSON)
    assert loop._lesson.scene_tag(1) is None
    assert loop._lesson.step_tag(2) is None

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.reply(page_ack(2, "fired", None, "ratio", 2, 2))
        page.reply(page_ack(1, "fired", None, "gone", 1, 1))

    assert not any(message.startswith("cursor.") for message in session_messages(caplog))
    assert [entry.cue_id for entry in loop._lesson.sent] == [1, 2]
    assert loop._lesson.acked == Cursor()
    await loop.aclose()


async def test_a_trailing_tag_takes_the_last_chunks_timing_and_no_chunk_discards_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = TimedSpeaker(log)
    page = FakePage(log)
    first = spoken_chunks(["<scene 1>The ratio compares two policies. <step 2>"])
    second = spoken_chunks(["<step 3>"])
    reasoning = FakeReasoning(log, [], speaker.received, turns=[first, second])
    loop = TurnLoop(
        concept_cfg(),
        SerialSource([CONCEPT_TEXT, "go on"]),
        FakeSearch(log, found()),
        speaker,
        page,
        reasoning,
        FakeRegistry(log),
        FakeClock(),
    )
    loop._lesson.adopt(LESSON)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
        await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    opening, trailing = sent(log, "lesson.cue")
    assert trailing["tag"] == {"kind": "step", "n": 2}
    assert trailing["chunk_id"] == opening["chunk_id"]
    assert (trailing["lead_ms"], trailing["audio_ms"]) == (950, 0)
    assert "tag.discarded turn_id=turn-2 count=1 reason=no_chunk" in session_messages(caplog)
    assert [entry.cue_id for entry in loop._lesson.sent] == [1, 2]


async def test_a_tag_in_the_follow_up_round_is_stripped_and_cued(tmp_path: Path) -> None:
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
    loop._lesson.adopt(LESSON)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    (cue,) = sent(log, "lesson.cue")
    assert cue["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}
    assert not any("<" in text for text in spoken_texts(speaker))


async def test_a_cue_checked_after_a_failed_ack_in_the_same_reply_is_discarded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    first = spoken_chunks(
        ["<scene 1>The ratio compares two policies. ", "<step 2>At one action it is one number."]
    )
    reasoning = FakeReasoning(log, [], speaker.received, turns=[first])
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning, transport=page)
    loop._lesson.adopt(LESSON)

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
    assert loop._lesson.dropped == ["<scene 1>: runtime"]
    await loop.aclose()


@pytest.mark.parametrize("matches", [True, False], ids=["matching", "mismatching"])
async def test_a_speculations_tags_are_checked_only_when_the_claimed_turn_plays_them(
    matches: bool,
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    page = FakePage(log)
    gate = asyncio.Event()
    turns = [spoken_chunks(SCENE_DELTAS), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(log, [], speaker.received, gate=gate, holds_at=1, turns=turns)
    final = CONCEPT_TEXT if matches else "two"
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [PartialTranscript(text=CONCEPT_PARTIAL), endpoint, EndOfTurn(text=final), speaker.finished]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg, transport=page)
    loop._lesson.adopt(LESSON)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    assert sent(log, "lesson.cue") == [] and loop._lesson.sent == []

    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    if matches:
        (cue,) = sent(log, "lesson.cue")
        assert cue["tag"] == {"kind": "scene", "n": 1, "scene_id": "ratio"}
        first_caption = sent(log, "caption")[0]
        assert log.index(("send_json", cue)) == log.index(("send_json", first_caption)) + 1
        assert [entry.cue_id for entry in loop._lesson.sent] == [1]
        assert "They agree at one." in " ".join(spoken_texts(speaker))
    else:
        assert sent(log, "lesson.cue") == []
        assert loop._lesson.sent == [] and loop._lesson.dropped == []
        assert speaker.utterances == [SPOKEN_CLAUSES]


async def test_a_barge_in_after_a_cue_went_out_drops_it_at_the_page_and_the_next_turn_waits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    page = FakePage(log, syncs=False)
    first = spoken_chunks(
        ["<scene 1>The ratio compares two policies. ", "<step 2>At one action it is one number."]
    )
    second = spoken_chunks(["Back to the ratio then."])
    reasoning = FakeReasoning(log, [], speaker.received, turns=[first, second])
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
    assert ("start_turn", SECOND_TEXT) not in log

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        page.answer_sync()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert speaker.utterances[1] == ["Back to the ratio then."]
    assert loop._lesson.sent == [] and loop._lesson.dropped == []
    assert loop._lesson.acked == Cursor(scene=0, step=0)
    assert "lesson.ack cue_id=1 outcome=dropped reason=barrier" in session_messages(caplog)
    await loop.aclose()


async def test_a_ready_report_never_interrupts_the_turn() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(
        log, spoken_chunks(BRIEFED_DELTAS), speaker.received, visual=[SCENE_CALL]
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert ("flush_playout", None) not in log
    assert speaker.utterances == [SPOKEN_CLAUSES]
    assert shown(log) == ["turn-1"]


async def test_a_superseded_draft_is_never_pushed_to_the_page() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    reasoning = FakeReasoning(
        log,
        spoken_chunks(BRIEFED_DELTAS),
        speaker.received,
        visual=[SCENE_CALL],
        visual_gates=[gate],
    )
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    first = visual_task()
    assert shown(log) == ["turn-2"]
    gate.set()
    await asyncio.wait_for(first, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert first.result() == "scene: superseded"
    assert pushed(log) == ["turn-2"]


COVERED_TAIL = json.dumps({"signal": "covered", "settling": ""})
TOLD_TAIL = json.dumps({"signal": "told", "settling": ""})
TAILED_DELTAS = [*SPOKEN_DELTAS, "\n" + OUTCOME_MARKER, COVERED_TAIL]
TOLD_DELTAS = [*SPOKEN_DELTAS, "\n" + OUTCOME_MARKER, TOLD_TAIL]
SPOKEN_WITH_TAIL = " ".join(SPOKEN_CLAUSES) + "\n" + OUTCOME_MARKER


def test_the_splitter_keeps_the_raw_tail() -> None:
    splitter = OutcomeSplitter()
    assert splitter.feed('x<outcome> {"signal": "covered"} ') == "x"
    assert splitter.tail is None
    assert splitter.finish() == ("", TurnOutcome(signal="covered"))
    assert splitter.tail == '{"signal": "covered"}'
    bare = OutcomeSplitter()
    assert bare.feed("x") == "x"
    assert bare.finish() == ("", TurnOutcome())
    assert bare.tail is None


async def test_the_next_turn_sees_the_outcome_tail_in_its_history() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    reasoning = FakeReasoning(log, spoken_chunks(TAILED_DELTAS), speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=SPOKEN_WITH_TAIL + COVERED_TAIL),
    ]
    assert reasoning.prompts[1].history[1].content.endswith("\n<outcome>" + COVERED_TAIL)


async def test_a_malformed_tail_is_kept_as_written(caplog: pytest.LogCaptureFixture) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    tail = '{"signal": "teach"}</outcome>'
    deltas = spoken_chunks([*SPOKEN_DELTAS, "\n" + OUTCOME_MARKER, tail])
    reasoning = FakeReasoning(log, deltas, speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert outcome_lines(caplog)[0] == "turn.outcome turn_id=turn-1 signal=None phase=teach"
    assert reasoning.prompts[1].history[1] == Message(
        role="assistant", content=SPOKEN_WITH_TAIL + tail
    )


async def test_a_tail_is_capped_at_the_limit() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    frame = json.dumps({"signal": "covered", "settling": ""})
    tail = json.dumps({"signal": "covered", "settling": "a" * (TAIL_LIMIT + 100 - len(frame))})
    assert len(tail) == TAIL_LIMIT + 100
    deltas = spoken_chunks([*SPOKEN_DELTAS, "\n" + OUTCOME_MARKER, tail])
    reasoning = FakeReasoning(log, deltas, speaker.received)
    loop = concept_loop(log, SerialSource([CONCEPT_TEXT, SECOND_CONCEPT_TEXT]), speaker, reasoning)

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.prompts[1].history[1] == Message(
        role="assistant", content=SPOKEN_WITH_TAIL + tail[:TAIL_LIMIT]
    )


async def test_a_cancelled_turn_keeps_no_tail() -> None:
    log: list[tuple[str, object]] = []
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold, hold_at=1)
    reasoning = FakeReasoning(log, spoken_chunks(TAILED_DELTAS), speaker.received)
    barge = asyncio.Event()
    resume = asyncio.Event()
    source = ScriptedSource(
        [EndOfTurn(text=CONCEPT_TEXT), barge, SpeechStarted(), resume, EndOfTurn(text="two")]
    )
    loop = concept_loop(log, source, speaker, reasoning)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    assert ("stream_closed", 6) in log
    assert loop._tails == {"turn-1": COVERED_TAIL}
    assert speaker.utterances == [[SPOKEN_CLAUSES[0]]]
    turn = turn_task()

    await pull_past(source, barge)
    hold.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    assert loop._tails == {}

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=SPOKEN_CLAUSES[0]),
    ]


@pytest.mark.parametrize("completes", [False, True], ids=["open", "completed"])
async def test_a_mismatched_speculation_leaves_no_tail_in_the_transcript(completes: bool) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    holds_at = len(TAILED_DELTAS) - 1 if completes else 3
    turns = [
        spoken_chunks(TAILED_DELTAS),
        spoken_chunks(SPOKEN_DELTAS),
        spoken_chunks(SPOKEN_DELTAS),
    ]
    reasoning = FakeReasoning(log, [], speaker.received, gate=gate, holds_at=holds_at, turns=turns)
    endpoint = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=CONCEPT_PARTIAL),
            endpoint,
            EndOfTurn(text="two"),
            speaker.finished,
            EndOfTurn(text="three"),
        ]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    speculation = speculation_task()
    if completes:
        gate.set()
        await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
        assert not speculation.cancelled() and speculation.exception() is None
        assert loop._tails == {"turn-1": COVERED_TAIL}

    await pull_past(source, endpoint)
    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert [prompt.user_text for prompt in reasoning.prompts] == [CONCEPT_PARTIAL, "two", "three"]
    assert reasoning.prompts[2].history == [
        Message(role="user", content="two"),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]
    assert loop._tails == {}


async def test_a_superseded_speculations_tail_never_reaches_the_transcript() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    turns = [spoken_chunks(TOLD_DELTAS), spoken_chunks(SPOKEN_DELTAS), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(
        log, [], speaker.received, gate=gate, holds_at=len(TOLD_DELTAS) - 1, turns=turns
    )
    settled = asyncio.Event()
    primed = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text="teach"),
            settled,
            PartialTranscript(text=CONCEPT_TEXT),
            primed,
            EndOfTurn(text=CONCEPT_TEXT),
            speaker.finished,
            EndOfTurn(text="two"),
        ]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    first = speculation_task()
    gate.set()
    await asyncio.wait_for(asyncio.wait([first]), HANG_GUARD_S)
    assert not first.cancelled() and first.exception() is None
    assert loop._tails == {"turn-1": TOLD_TAIL}
    reasoning.started.clear()

    await pull_past(source, settled)
    assert loop._tails == {}
    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await pull_past(source, primed)
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert [prompt.user_text for prompt in reasoning.prompts] == ["teach", CONCEPT_TEXT, "two"]
    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert reasoning.prompts[2].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]


async def test_a_completed_speculation_that_is_claimed_keeps_its_tail() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    turns = [spoken_chunks(TAILED_DELTAS), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(
        log, [], speaker.received, gate=gate, holds_at=len(TAILED_DELTAS) - 1, turns=turns
    )
    settled = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=CONCEPT_PARTIAL),
            settled,
            EndOfTurn(text=CONCEPT_TEXT),
            speaker.finished,
            EndOfTurn(text="two"),
        ]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    speculation = speculation_task()
    gate.set()
    await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
    assert not speculation.cancelled() and speculation.exception() is None
    assert loop._tails == {"turn-1": COVERED_TAIL}

    await pull_past(source, settled)
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert [prompt.user_text for prompt in reasoning.prompts] == [CONCEPT_PARTIAL, "two"]
    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert reasoning.prompts[1].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=SPOKEN_WITH_TAIL + COVERED_TAIL),
    ]
    assert loop._tails == {}


async def test_a_speculation_cancelled_by_speech_leaves_no_tail_on_the_claimed_turn() -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    close_gate = asyncio.Event()
    filled = [FILLER_DELTA] * (SPOKEN_DEPTH - 1) + ["x. \n" + OUTCOME_MARKER, TOLD_TAIL]
    turns = [spoken_chunks(filled), spoken_chunks(SPOKEN_DELTAS), spoken_chunks(SPOKEN_DELTAS)]
    reasoning = FakeReasoning(log, [], speaker.received, close_gate=close_gate, turns=turns)
    barge = asyncio.Event()
    resume = asyncio.Event()
    primed = asyncio.Event()
    source = ScriptedSource(
        [
            PartialTranscript(text=CONCEPT_PARTIAL),
            barge,
            SpeechStarted(),
            resume,
            PartialTranscript(text=CONCEPT_TEXT),
            primed,
            EndOfTurn(text=CONCEPT_TEXT),
            speaker.finished,
            EndOfTurn(text="two"),
        ]
    )
    cfg = concept_cfg().model_copy(update={"speculative_reasoning": True})
    loop = concept_loop(log, source, speaker, reasoning, cfg=cfg)
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].closing.wait(), HANG_GUARD_S)
    speculation = speculation_task()
    assert ("stream_closed", len(filled)) in log
    assert loop._tails == {}
    reasoning.started.clear()

    close_gate.set()
    await pull_past(source, barge)
    await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
    assert speculation.cancelled()
    assert loop._tails == {"turn-1": TOLD_TAIL}

    await pull_past(source, resume)
    assert loop._tails == {}
    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await pull_past(source, primed)
    await asyncio.wait_for(running, HANG_GUARD_S)
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert [prompt.user_text for prompt in reasoning.prompts] == [
        CONCEPT_PARTIAL,
        CONCEPT_TEXT,
        "two",
    ]
    assert speaker.utterances == [SPOKEN_CLAUSES] * 2
    assert reasoning.prompts[2].history == [
        Message(role="user", content=CONCEPT_TEXT),
        Message(role="assistant", content=" ".join(SPOKEN_CLAUSES)),
    ]


async def test_a_tool_rounds_tail_yields_to_the_follow_up(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, found())
    turns = [[SEARCH_CALL, *spoken_chunks(TAILED_DELTAS)], spoken_chunks(SPOKEN_DELTAS)]
    follow_up = spoken_chunks(FOLLOW_DELTAS)
    reasoning = FakeReasoning(log, [], speaker.received, follow_up=follow_up, turns=turns)
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
    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert len(reasoning.prompts) == 3
    assert outcome_lines(caplog)[0] == "turn.outcome turn_id=turn-1 signal=None phase=teach"
    history = reasoning.prompts[2].history
    assert [message.role for message in history] == ["user", "assistant"]
    assert OUTCOME_MARKER not in history[1].content
    assert history[1].content.endswith(FOLLOW_DELTAS[-1].strip())
