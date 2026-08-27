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
from tutor.lead_in import lead_in_sentence
from tutor.prompt import SEARCH_CODE_TOOL, TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.session import BAD_ARGUMENTS, SPOKEN_DEPTH, TurnLoop, TurnLoopConfig
from tutor.tools.models import SearchBudget, SearchMatch, SearchResult
from tutor.transport import INBOUND_CAPACITY, Connection

HANG_GUARD_S = 20.0
TICK_S = 0.25
SYSTEM = "you tutor an engineer through a codebase"
SUBJECT = "the inbound frame queue"
USER_TEXT = "walk me through tutor/transport.py"
SECOND_TEXT = "and where does tutor/playout.py fit"
GLOBS = ["tutor/**/*.py"]
SPOKEN_DELTAS = ["It lives in ", "the reader, ", "which drains ", "the track."]
SPOKEN_CLAUSES = ["It lives in the reader,", "which drains the track."]
FOLLOW_DELTAS = [" Both call sites ", "drain it."]
CHAINED_CLAUSES = [
    "It lives in the reader,",
    "which drains the track. Both call sites drain it.",
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
DRAIN_TASK = "turn-1-drain"
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


def found() -> SearchResult:
    return SearchResult(
        tool="search",
        query=USER_TEXT,
        globs=GLOBS,
        matches=[
            SearchMatch(path="tutor/transport.py", line=24, text="        self._inbound = queue")
        ],
        truncated=False,
        oversized=False,
        byte_count=64,
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


class ScriptedSource:
    def __init__(self, script: list[InputEvent | asyncio.Event]) -> None:
        self._script = script

    async def events(self) -> AsyncIterator[InputEvent]:
        for item in self._script:
            if isinstance(item, asyncio.Event):
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
    ) -> None:
        self._log = log
        self._result = result
        self._gate = gate
        self._fails = fails
        self.calls: list[tuple[str, list[str], Path, SearchBudget]] = []
        self.entered = asyncio.Event()

    async def __call__(
        self, query: str, globs: Sequence[str], root: Path, budget: SearchBudget
    ) -> SearchResult:
        self.calls.append((query, list(globs), root, budget))
        self.entered.set()
        if self._gate is not None:
            try:
                await self._gate.wait()
            except asyncio.CancelledError:
                self._log.append(("search_cancelled", query))
                raise
        if query == self._fails:
            raise SearchFailed
        self._log.append(("search_done", query))
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


class FakeSpeaker:
    def __init__(self, log: list[tuple[str, object]], gate: asyncio.Event | None = None) -> None:
        self._log = log
        self._gate = gate
        self.utterances: list[list[str]] = []
        self.received = asyncio.Event()
        self.finished = asyncio.Event()

    async def speak(self, chunks: AsyncIterator[str]) -> None:
        spoken: list[str] = []
        self.utterances.append(spoken)
        async for chunk in chunks:
            spoken.append(chunk)
            self._log.append(("speak", chunk))
            self.received.set()
            if self._gate is not None and len(spoken) > 1:
                await self._gate.wait()
        self.finished.set()


class RateLimited(Exception):
    pass


class FakeStream:
    def __init__(
        self,
        chunks: list[TurnChunk],
        log: list[tuple[str, object]],
        received: asyncio.Event,
        gate: asyncio.Event | None = None,
    ) -> None:
        self._chunks = chunks
        self._log = log
        self._received = received
        self._gate = gate
        self.iterations = 0
        self.cancels = 0
        self.yielded = 0
        self.entered = asyncio.Event()
        self.filled = asyncio.Event()

    async def _drain(self) -> AsyncIterator[TurnChunk]:
        self._log.append(("stream_open", self._received.is_set()))
        self.entered.set()
        if self._gate is not None:
            await self._gate.wait()
        for chunk in self._chunks:
            self.yielded += 1
            if self.yielded == QUEUE_FULL_DELTAS:
                self.filled.set()
            yield chunk

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
    ) -> None:
        self._log = log
        self._chunks = chunks
        self._follow_up = follow_up if follow_up is not None else []
        self._received = received
        self._gate = gate
        self._fails = fails
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
    loop = TurnLoop(config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), clock)

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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
    )

    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert [stream.iterations for stream in reasoning.streams] == [1, 1]
    assert len([name for name, _ in log if name == "start_turn"]) == 2
    assert speaker.utterances == [[lead_in_sentence([result])] + CHAINED_CLAUSES]
    await loop.aclose()


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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].entered.wait(), HANG_GUARD_S)
    drains = [task for task in asyncio.all_tasks() if task.get_name() == DRAIN_TASK]
    assert len(drains) == 1

    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert drains[0].cancelled()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
    )
    running = asyncio.create_task(loop.run())

    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].filled.wait(), HANG_GUARD_S)
    drains = [task for task in asyncio.all_tasks() if task.get_name() == DRAIN_TASK]
    assert len(drains) == 1

    await asyncio.wait_for(loop.aclose(), HANG_GUARD_S)

    assert drains[0].cancelled()
    assert reasoning.streams[0].yielded == QUEUE_FULL_DELTAS
    assert speaker.utterances == [[lead_in_sentence([result]), SPOKEN_CLAUSES[0]]]
    assert ("abandon", "turn-1") in log

    release.set()
    await asyncio.wait_for(running, HANG_GUARD_S)


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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
        config(tmp_path), source, search, speaker, reasoning, FakeRegistry(log), FakeClock()
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
