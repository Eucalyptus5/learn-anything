import asyncio
import json
import logging
import re
from collections import deque
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import av
import numpy as np
import pytest
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from tests.fakes import local_peer, numbered_frames
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
from tutor.lead_in import lead_in_sentence, lead_in_stages, opener_key
from tutor.prompt import SEARCH_CODE_TOOL, TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.session import BAD_ARGUMENTS, OPENER, SPOKEN_DEPTH, TurnLoop, TurnLoopConfig
from tutor.tools.models import (
    GroundingVerdict,
    Position,
    SearchBudget,
    SearchMatch,
    SearchResult,
)
from tutor.tools.provenance import TurnRegistry
from tutor.transport import INBOUND_CAPACITY, Connection

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
GLOBS = ["tutor/**/*.py"]
SPOKEN_DELTAS = ["It lives in ", "the reader, ", "which drains ", "the track."]
SPOKEN_CLAUSES = ["It lives in the reader,", "which drains the track."]
FOLLOW_DELTAS = [" Both call sites ", "drain it."]
PATH_DELTAS = [
    f"It lives in {GROUNDED_PATH}, ",
    f"and the other copy of the same reader sits in {UNGROUNDED_PATH}, ",
    "Both call sites drain it.",
]
PATH_CLAUSES = [
    f"It lives in {GROUNDED_PATH},",
    f"and the other copy of the same reader sits in {UNGROUNDED_PATH},",
    "Both call sites drain it.",
]
LINE_DELTAS = [f"It lives in {GROUNDED_PATH} line 24, ", "and the reader drains the track."]
LINE_CLAUSES = [
    f"It lives in {GROUNDED_PATH} line 24,",
    "and the reader drains the track.",
]
COUNT_DELTAS = [f"It lives in {GROUNDED_PATH} line 24, ", "There are 3 callers of it."]
SPLIT_LINE = 4021
SPLIT_DELTAS = [
    "It lives there. The queue reader that drains ",
    f"the inbound audio track sits on line {SPLIT_LINE} of that same file and it never blocks.",
]
SPLIT_CLAUSES = [
    "It lives there.",
    "The queue reader that drains the inbound audio track sits on line",
    f"{SPLIT_LINE} of that same file and it never blocks.",
]
CHAINED_CLAUSES = [
    "It lives in the reader,",
    "which drains the track.\n Both call sites drain it.",
]
REASONING_TEXT = "weighing two call sites"
MODEL_QUERY = "class Connection"
MODEL_GLOBS = ["tutor/transport.py"]
SEARCH_ARGUMENTS = json.dumps({"query": MODEL_QUERY, "globs": MODEL_GLOBS})
SEARCH_CALL_ID = "call-search-1"
TRUNCATED_ARGUMENTS = json.dumps({"query": "def recv", "globs": MODEL_GLOBS})
BAD_ARGUMENT_BODIES = [
    pytest.param(json.dumps({"query": MODEL_QUERY, "globs": []}), id="empty-globs"),
    pytest.param(json.dumps({"query": MODEL_QUERY}), id="missing-globs"),
    pytest.param(json.dumps({"query": MODEL_QUERY, "globs": MODEL_GLOBS[0]}), id="globs-string"),
    pytest.param(SEARCH_ARGUMENTS.removesuffix("]}"), id="truncated-json"),
]
VISUAL_TOOL = "push_diagram"
VISUAL_ARGUMENTS = json.dumps({"mermaid": "graph TD; reader-->queue"})
VISUAL_CALL_ID = "call-visual-1"
RATE_LIMIT_DETAIL = "quota exhausted for project acct-9"
TURN_TASK = "turn-1"
DRAIN_TASK = "turn-1-drain"
STAGER_TASK = "turn-1-stager"
SPECULATION_TASK = "turn-1-speculation"
SECOND_PATH = "tutor/resample.py"
CARRY_DELTAS = [
    "Line 30 is the reader. ",
    f"It lives in the queue reader inside {GROUNDED_PATH}, ",
    "line 24 is where it drains.",
]
CARRY_CLAUSES = [
    f"It lives in the queue reader inside {GROUNDED_PATH},",
    "line 24 is where it drains.",
]
FIRST_CLAUSE_DELTA = "It lives in the reader, "
FILLER_DELTA = "and "
# One delta reaches the chunker, SPOKEN_DEPTH sit in the queue, and the next put blocks.
QUEUE_FULL_DELTAS = SPOKEN_DEPTH + 2
GROUNDING_SPAN = re.compile(r"^turn\.grounding turn_id=\S+ ms=250$")
SPOKEN_SPAN = re.compile(r"^turn\.spoken turn_id=\S+ ms=500$")

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
        query=USER_TEXT,
        globs=GLOBS,
        matches=[SearchMatch(path=path, line=24, text="        self._inbound = queue")],
        truncated=False,
        oversized=False,
        byte_count=64,
    )


def found_many() -> SearchResult:
    return SearchResult(
        tool="search",
        query=USER_TEXT,
        globs=GLOBS,
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
    def __init__(self, log: list[tuple[str, object]], gate: asyncio.Event | None = None) -> None:
        self._log = log
        self._gate = gate
        self.utterances: list[list[str]] = []
        self.openers: list[str] = []
        self.received = asyncio.Event()
        self.held = asyncio.Event()
        self.finished = asyncio.Event()

    async def speak_opener(self, key: str) -> None:
        self.openers.append(key)
        self._log.append(("speak_opener", key))

    async def speak(self, chunks: AsyncIterator[str]) -> None:
        spoken: list[str] = []
        self.utterances.append(spoken)
        try:
            async for chunk in chunks:
                spoken.append(chunk)
                self._log.append(("speak", chunk))
                self.received.set()
                if self._gate is not None and len(spoken) > 1:
                    self.held.set()
                    await self._gate.wait()
        except asyncio.CancelledError:
            self._log.append(("speak_cancelled", len(spoken)))
            raise
        self.finished.set()


class LoggingTransport:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self._log = log

    def flush_playout(self) -> None:
        self._log.append(("flush_playout", None))


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
    ) -> None:
        self._log = log
        self._chunks = chunks
        self._follow_up = follow_up if follow_up is not None else []
        self._received = received
        self._gate = gate
        self._fails = fails
        self._holds_at = holds_at
        self._close_gate = close_gate
        self.prompts: list[TurnPrompt] = []
        self.tools: list[list[dict] | None] = []
        self.streams: list[FakeStream] = []
        self.started = asyncio.Event()

    def start_turn(self, prompt: TurnPrompt, tools: Sequence[dict] | None = None) -> FakeStream:
        self.prompts.append(prompt)
        self.tools.append(list(tools) if tools is not None else None)
        self._log.append(("start_turn", prompt.user_text))
        if prompt.user_text == self._fails:
            raise RateLimited(RATE_LIMIT_DETAIL)
        answering = bool(prompt.tool_exchange)
        stream = FakeStream(
            self._follow_up if answering else self._chunks,
            self._log,
            self._received,
            None if answering else self._gate,
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


async def test_grounding_lands_before_the_reasoning_call(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [TurnChunk(kind="reasoning", text=REASONING_TEXT)]
    deltas += [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    source = ScriptedSource([EndOfTurn(text=USER_TEXT), speaker.finished])
    clock = FakeClock()
    loop = TurnLoop(
        config(tmp_path),
        source,
        search,
        speaker,
        LoggingTransport(log),
        reasoning,
        FakeRegistry(log),
        clock,
    )

    with caplog.at_level(logging.INFO, logger="tutor.session"):
        await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == [(USER_TEXT, GLOBS, tmp_path, SearchBudget())]

    steps = [name for name, _ in log]
    assert steps.index("open_turn") < steps.index("search_done")
    assert steps.index("search_done") < steps.index("record") < steps.index("start_turn")
    assert steps.index("speak") < steps.index("stream_open")
    assert ("stream_open", True) in log

    prompt = reasoning.prompts[0]
    assert prompt.tool_context == [result]
    assert prompt.user_text == USER_TEXT
    assert SYSTEM in prompt.system
    assert SUBJECT in prompt.system

    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]

    messages = session_messages(caplog)
    assert any(GROUNDING_SPAN.match(message) for message in messages)
    assert any(SPOKEN_SPAN.match(message) for message in messages)
    assert all(USER_TEXT not in message for message in messages)
    await loop.aclose()


async def test_the_cached_opener_is_enqueued_before_the_search_returns(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, result, gate=gate)
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
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
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(search.held.wait(), HANG_GUARD_S)
    assert ("speak_opener", "thinking") in log
    assert ("search_done", USER_TEXT) not in log
    assert not [entry for entry in log if entry[0] == "speak"]

    gate.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    steps = [name for name, _ in log]
    assert steps.index("speak_opener") < steps.index("search_done") < steps.index("start_turn")
    assert ("stream_open", True) in log
    spoken = [entry for entry in log if entry[0] in ("speak_opener", "speak")]
    assert spoken[:3] == [
        ("speak_opener", "thinking"),
        ("speak_opener", opener_key([result])),
        ("speak", lead_in_sentence([result])),
    ]
    assert speaker.openers == ["thinking", opener_key([result])]
    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
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
    reasoning = FakeReasoning(log, [], speaker.received)
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
    assert speaker.utterances == []
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
    reasoning = FakeReasoning(log, [], speaker.received)
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
    assert speaker.utterances == []
    assert reasoning.prompts == []

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert speaker.utterances == []
    assert reasoning.prompts == []


async def test_a_turn_with_no_spoken_chunk_accepts_the_next_turn(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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

    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [[lead_in], [lead_in]]
    assert [prompt.user_text for prompt in reasoning.prompts] == [USER_TEXT, SECOND_TEXT]
    turn_ids = [turn_id for name, turn_id in log if name == "open_turn"]
    assert len(set(turn_ids)) == 2
    assert [turn_id for name, turn_id in log if name == "record"] == turn_ids
    await loop.aclose()


async def test_a_failing_turn_is_reported_and_the_loop_keeps_running(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result, fails=USER_TEXT)
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
    source = ScriptedSource(
        [EndOfTurn(text=USER_TEXT), EndOfTurn(text=SECOND_TEXT), speaker.finished]
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

    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.failed")] == [
        "turn.failed turn_id=turn-1 error=SearchFailed"
    ]
    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
    assert [prompt.user_text for prompt in reasoning.prompts] == [SECOND_TEXT]
    captured = [record.getMessage() for record in caplog.records]
    assert all(USER_TEXT not in message and SECOND_TEXT not in message for message in captured)
    await loop.aclose()


async def test_a_tool_call_chunk_never_reaches_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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

    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
    spoken = [text for name, text in log if name == "speak"]
    assert all(VISUAL_ARGUMENTS not in str(text) for text in spoken)
    assert all(REASONING_TEXT not in str(text) for text in spoken)

    recorded = [payload for name, payload in log if name in ("open_turn", "record", "abandon")]
    assert recorded == ["turn-1", "turn-1"]
    assert len(search.calls) == 1

    assert len(reasoning.prompts) == 2
    exchange = reasoning.prompts[1].tool_exchange
    assert [message.role for message in exchange] == ["assistant", "tool"]
    assert exchange[0].tool_calls[0].id == VISUAL_CALL_ID
    assert exchange[1].tool_call_id == VISUAL_CALL_ID
    assert VISUAL_TOOL in exchange[1].content
    assert "not available" in exchange[1].content

    assert all(VISUAL_ARGUMENTS not in message for message in session_messages(caplog))
    await loop.aclose()


async def test_a_search_code_call_dispatches_one_search_and_one_follow_up(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [
        TurnChunk(
            kind="tool_call",
            text=SEARCH_ARGUMENTS,
            tool_call_id=SEARCH_CALL_ID,
            tool_name="search_code",
        )
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

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert search.calls == [
        (USER_TEXT, GLOBS, tmp_path, SearchBudget()),
        (MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget()),
    ]
    assert len(reasoning.prompts) == 2
    assert reasoning.tools == [[SEARCH_CODE_TOOL], None]

    exchange = reasoning.prompts[1].tool_exchange
    assert exchange[0].tool_calls[0].id == SEARCH_CALL_ID
    assert exchange[0].tool_calls[0].function.name == "search_code"
    assert exchange[0].tool_calls[0].function.arguments == SEARCH_ARGUMENTS
    assert exchange[1].tool_call_id == SEARCH_CALL_ID
    assert exchange[1].content == result.model_dump_json()

    assert [turn_id for name, turn_id in log if name == "record"] == ["turn-1", "turn-1"]
    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
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

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert speaker.utterances == [
        [lead_in_sentence([result]), "It lives there.", "Diagram incoming for the pool."]
    ]
    await loop.aclose()


async def test_a_payload_in_the_stream_never_reaches_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
            lead_in_sentence([result]),
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
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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

    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [[lead_in], [lead_in] + SPOKEN_CLAUSES]

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
    deltas = [
        TurnChunk(
            kind="tool_call",
            text=SEARCH_ARGUMENTS,
            tool_call_id=SEARCH_CALL_ID,
            tool_name="search_code",
        )
    ]
    deltas += [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    follow_up = [TurnChunk(kind="spoken", text=delta) for delta in FOLLOW_DELTAS]
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

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert [stream.iterations for stream in reasoning.streams] == [1, 1]
    assert len([name for name, _ in log if name == "start_turn"]) == 2
    assert speaker.utterances == [[lead_in_sentence([result])] + CHAINED_CLAUSES]
    await loop.aclose()


def turn_task() -> asyncio.Task[None]:
    turns = [task for task in asyncio.all_tasks() if task.get_name() == TURN_TASK]
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
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert speaker.utterances == [[lead_in_sentence([result])]]
    assert ("abandon", "turn-1") in log

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)


async def test_a_barge_in_on_a_full_spoken_queue_cancels_the_drain(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, result)
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
    assert speaker.utterances == [[lead_in_sentence([result]), SPOKEN_CLAUSES[0]]]
    assert ("abandon", "turn-1") in log

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)


async def test_speech_start_mid_synthesis_cancels_flushes_and_takes_the_next_turn(
    tmp_path: Path,
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, result)
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
    lead_in = lead_in_sentence([result])

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    await asyncio.wait_for(speaker.held.wait(), HANG_GUARD_S)
    drain = drain_task()
    assert speaker.utterances == [[lead_in, SPOKEN_CLAUSES[0]]]
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
    assert (
        steps.index("barge")
        < steps.index("flush_playout")
        < steps.index("stream_closed")
        < steps.index("speak_cancelled")
        < log.index(("abandon", "turn-1"))
    )
    assert ("stream_closed", 3) in log
    assert speaker.utterances == [[lead_in, SPOKEN_CLAUSES[0]]]
    assert ("open_turn", "turn-2") not in log

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert log.index(("abandon", "turn-1")) < log.index(("open_turn", "turn-2"))
    assert speaker.utterances == [[lead_in, SPOKEN_CLAUSES[0]], [lead_in, *SPOKEN_CLAUSES]]
    assert [entry for entry in log if entry[0] == "speak_cancelled"] == [("speak_cancelled", 2)]
    await loop.aclose()


async def test_speech_start_during_the_opener_drops_the_queued_playout(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, result, gate=gate)
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
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

    await asyncio.wait_for(search.held.wait(), HANG_GUARD_S)
    assert speaker.openers == ["thinking"]
    assert speaker.utterances == []
    turn = turn_task()

    await pull_past(source, barge)
    gate.set()
    await asyncio.wait_for(asyncio.wait([turn]), HANG_GUARD_S)

    assert turn.cancelled()
    steps = [name for name, _ in log]
    assert (
        steps.index("speak_opener")
        < steps.index("flush_playout")
        < steps.index("search_cancelled")
        < log.index(("abandon", "turn-1"))
    )
    assert "speak" not in steps
    assert "speak_cancelled" not in steps
    assert reasoning.prompts == []

    resume.set()
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert len(search.calls) == 2
    steps = [name for name, _ in log]
    assert steps.index("flush_playout") < steps.index("speak")
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]
    await loop.aclose()


async def test_a_clause_past_the_chunker_at_speech_start_is_never_spoken(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found_many()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
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
    result = found()
    hold = asyncio.Event()
    speaker = FakeSpeaker(log, gate=hold)
    search = FakeSearch(log, result)
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
        lead_in = lead_in_sentence([result])

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
    assert speaker.utterances == [[lead_in, SPOKEN_CLAUSES[0]], [lead_in, *SPOKEN_CLAUSES]]
    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.failed")] == []
    await loop.aclose()


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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in PATH_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
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
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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

    assert search.calls == [
        (USER_TEXT, GLOBS, tmp_path, SearchBudget()),
        (SECOND_TEXT, GLOBS, tmp_path, SearchBudget()),
    ]
    assert [turn_id for name, turn_id in log if name == "record"] == ["turn-1", "turn-2"]

    assert len(reasoning.prompts) == 4
    exchange = reasoning.prompts[1].tool_exchange
    assert exchange[0].tool_calls[0].id == SEARCH_CALL_ID
    assert exchange[1].tool_call_id == SEARCH_CALL_ID
    assert exchange[1].content == BAD_ARGUMENTS

    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [[lead_in] + SPOKEN_CLAUSES, [lead_in] + SPOKEN_CLAUSES]
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

    assert search.calls == [
        (USER_TEXT, GLOBS, tmp_path, SearchBudget()),
        (MODEL_QUERY, MODEL_GLOBS, tmp_path, SearchBudget()),
    ]

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
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert reasoning.tools == [[SEARCH_CODE_TOOL], [SEARCH_CODE_TOOL]]
    assert search.calls == [
        (USER_TEXT, GLOBS, tmp_path, SearchBudget()),
        (SECOND_TEXT, GLOBS, tmp_path, SearchBudget()),
    ]

    lead_in = lead_in_sentence([result])
    assert speaker.utterances == [[lead_in] + SPOKEN_CLAUSES, [lead_in] + SPOKEN_CLAUSES]
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


async def test_an_ungrounded_chunk_is_withheld_from_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in PATH_DELTAS]
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in COUNT_DELTAS]
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in LINE_DELTAS]
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in PATH_DELTAS]
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
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

    assert lead_in_sentence([result]) not in [text for name, text in log if name == "speak"]
    assert speaker.utterances == [SPOKEN_CLAUSES]

    messages = session_messages(caplog)
    assert "turn.chunk_withheld turn_id=turn-1 source=lead_in ungrounded=1" in messages
    assert all(UNPARSEABLE_PATH not in message for message in messages)
    await loop.aclose()


async def test_a_tool_result_answering_an_abandoned_turn_reaches_nobody(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, result, gate=gate, later=found(UNGROUNDED_PATH), holds=MODEL_QUERY)
    deltas = [
        TurnChunk(
            kind="tool_call",
            text=SEARCH_ARGUMENTS,
            tool_call_id=SEARCH_CALL_ID,
            tool_name="search_code",
        )
    ]
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

    assert len(search.calls) == 2
    assert ("search_cancelled", MODEL_QUERY) in log
    assert ("search_done", MODEL_QUERY) not in log
    assert not registry.known("turn-1", Position(path=UNGROUNDED_PATH, line=24))
    assert not registry.known("turn-1", Position(path=UNGROUNDED_PATH))
    assert speaker.utterances == [[lead_in_sentence([result])]]
    assert PATH_CLAUSES[1] not in [text for name, text in log if name == "speak"]


async def test_a_line_number_split_across_the_cut_never_reaches_the_speaker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    registry = TurnRegistry()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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

    clauses, remainder = split_clauses("".join(SPLIT_DELTAS), 3, 12)
    spoken = [text for name, text in log if name == "speak"]
    assert clauses + [remainder] == SPLIT_CLAUSES
    assert speaker.utterances == [[lead_in_sentence([result]), *SPLIT_CLAUSES[:2]]]
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
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

    assert speaker.utterances == [[lead_in_sentence([result]), *stages[:2], *SPOKEN_CLAUSES]]
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in CARRY_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
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
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
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
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert search.calls == []
    assert reasoning.prompts == []
    assert log == []

    await pull_past(source, endpoint)
    await asyncio.wait_for(running, HANG_GUARD_S)

    assert search.calls == [(USER_TEXT, GLOBS, tmp_path, SearchBudget())]
    assert [entry for entry in log if entry[0] == "start_turn"] == [("start_turn", USER_TEXT)]
    assert speaker.utterances == [[lead_in_sentence([result])] + SPOKEN_CLAUSES]
    await loop.aclose()


async def test_a_partial_issues_the_request_and_a_matching_final_reuses_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
    gate = asyncio.Event()
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received, gate=gate)
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
    lead_in = lead_in_sentence([result])
    with caplog.at_level(logging.INFO, logger="tutor.session"):
        running = asyncio.create_task(loop.run())

        await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
        await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
        assert [call[0] for call in search.calls] == [PARTIAL_TEXT]
        assert [entry for entry in log if entry[0] == "start_turn"] == [
            ("start_turn", PARTIAL_TEXT)
        ]
        assert ("stream_open", False) in log
        assert reasoning.streams[0].iterations == 1
        assert reasoning.prompts[0].tool_context == [result]
        assert speaker.openers == []
        assert speaker.utterances == []

        await pull_past(source, endpoint)
        await asyncio.wait_for(speaker.received.wait(), HANG_GUARD_S)
        assert speaker.openers == [OPENER, opener_key([result])]
        assert speaker.utterances == [[lead_in]]
        assert [entry for entry in log if entry[0] == "start_turn"] == [
            ("start_turn", PARTIAL_TEXT)
        ]

        gate.set()
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert [entry for entry in log if entry[0] == "start_turn"] == [("start_turn", PARTIAL_TEXT)]
    assert reasoning.streams[-1].iterations == 1
    assert speaker.utterances == [[lead_in, *SPOKEN_CLAUSES]]
    assert [entry for entry in log if entry[0] == "open_turn"] == [("open_turn", TURN_TASK)]
    assert [entry for entry in log if entry[0] == "record"] == [("record", TURN_TASK)]
    assert "abandon" not in [name for name, _ in log]
    assert len(spoken_spans(caplog)) == 1
    messages = session_messages(caplog)
    assert all(PARTIAL_TEXT not in message for message in messages)
    await loop.aclose()


async def test_a_later_partial_cancels_and_reissues_the_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]
    assert [name for name, _ in log].count("abandon") == 1
    assert len(spoken_spans(caplog)) == 1
    await loop.aclose()


async def test_a_mismatching_final_discards_the_speculation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert search.calls[1][0] == SECOND_TEXT
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]
    assert len(spoken_spans(caplog)) == 1
    await loop.aclose()


async def test_speech_start_cancels_the_speculation(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]
    await loop.aclose()


async def test_aclose_cancels_a_speculation_in_flight(tmp_path: Path) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    search = FakeSearch(log, result)
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
    assert [name for name, _ in log].count("start_turn") == 1


async def test_a_failed_speculation_is_reported_and_the_final_grounds_itself(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, result, gate=gate, fails=PARTIAL_TEXT, holds=PARTIAL_TEXT)
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
        gate.set()
        await asyncio.wait_for(asyncio.wait([speculation]), HANG_GUARD_S)
        assert not speculation.cancelled()
        assert reasoning.prompts == []

        await pull_past(source, endpoint)
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert [call[0] for call in search.calls] == [PARTIAL_TEXT, USER_TEXT]
    assert [entry for entry in log if entry[0] == "start_turn"] == [("start_turn", USER_TEXT)]
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]
    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.speculation_failed")] == [
        "turn.speculation_failed turn_id=turn-1 error=SearchFailed"
    ]
    assert [message for message in messages if message.startswith("turn.failed")] == []
    assert len(spoken_spans(caplog)) == 1
    captured = [record.getMessage() for record in caplog.records]
    assert all(PARTIAL_TEXT not in message for message in captured)
    await loop.aclose()


async def test_a_claimed_speculation_that_fails_fails_the_turn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log: list[tuple[str, object]] = []
    result = found()
    speaker = FakeSpeaker(log)
    gate = asyncio.Event()
    search = FakeSearch(log, result, gate=gate, fails=PARTIAL_TEXT, holds=PARTIAL_TEXT)
    deltas = [TurnChunk(kind="spoken", text=delta) for delta in SPOKEN_DELTAS]
    reasoning = FakeReasoning(log, deltas, speaker.received)
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
        assert speaker.openers == [OPENER]
        turn = next(task for task in asyncio.all_tasks() if task.get_name() == TURN_TASK)

        gate.set()
        await asyncio.wait_for(asyncio.wait([speculation, turn]), HANG_GUARD_S)
        assert not turn.cancelled()
        assert isinstance(turn.exception(), SearchFailed)
        assert isinstance(speculation.exception(), SearchFailed)
        assert speaker.utterances == []
        assert reasoning.prompts == []
        assert [entry for entry in log if entry[0] in ("open_turn", "record", "abandon")] == [
            ("open_turn", TURN_TASK)
        ]

        await pull_past(source, second)
        await asyncio.wait_for(running, HANG_GUARD_S)

    assert [call[0] for call in search.calls] == [PARTIAL_TEXT, SECOND_TEXT]
    assert speaker.openers == [OPENER, OPENER, opener_key([result])]
    assert speaker.utterances == [[lead_in_sentence([result]), *SPOKEN_CLAUSES]]
    assert [prompt.user_text for prompt in reasoning.prompts] == [SECOND_TEXT]
    messages = session_messages(caplog)
    assert [message for message in messages if message.startswith("turn.failed")] == [
        "turn.failed turn_id=turn-1 error=SearchFailed"
    ]
    assert [message for message in messages if message.startswith("turn.speculation_failed")] == []
    assert len(spoken_spans(caplog)) == 1
    assert "turn_id=turn-2" in spoken_spans(caplog)[0]
    captured = [record.getMessage() for record in caplog.records]
    assert all(PARTIAL_TEXT not in message for message in captured)
    await loop.aclose()
