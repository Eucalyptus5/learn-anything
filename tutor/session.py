import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field, ValidationError

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.input_path import EndOfTurn, InputPath, PartialTranscript, SpeechStarted
from tutor.lead_in import lead_in_sentence, lead_in_stages
from tutor.lesson import OPENING_TEXT, LessonPlan, LessonState, lesson_block
from tutor.planner import EMPTY_REPLY, NO_TOOL_CALL, POSITION_REFUSED, plan_prompt, run_planner
from tutor.prompt import (
    SEARCH_CODE_TOOL,
    Message,
    ToolCall,
    ToolCallFunction,
    TurnPrompt,
)
from tutor.reasoning import ReasoningClient, TurnChunk
from tutor.speech import Chunk, OnPlay, Speaker
from tutor.tags import Marker, RawTag, TagSplitter, parse_marker, tag_name
from tutor.tools.models import SearchBudget, SearchResult
from tutor.tools.provenance import TurnRegistry
from tutor.transcript import Transcript
from tutor.transport import Connection
from tutor.visual_tools import VOICE_VISUAL_TOOLS, dispatch_visual_tool
from tutor.visuals import (
    CLIENT_MESSAGE,
    Caption,
    LearnerText,
    LessonAck,
    LessonAttach,
    LessonCheckpoint,
    LessonCue,
    LessonStatePush,
    LessonSync,
    LessonSynced,
    SceneCue,
    ScenePush,
    SceneReady,
    ThemeMessage,
    TurnState,
    VisualChannel,
)

logger = logging.getLogger(__name__)

SearchCall = Callable[[str, Sequence[str], Path, SearchBudget], Awaitable[SearchResult]]
Grounded = tuple[TurnPrompt, asyncio.Queue[str | RawTag | None]]

SEARCH_CODE = "search_code"
TURN_TOOLS: list[dict[str, object]] = [SEARCH_CODE_TOOL, *VOICE_VISUAL_TOOLS]
VISUAL_TOOL_NAMES = frozenset(tool["function"]["name"] for tool in VOICE_VISUAL_TOOLS)
SPOKEN_DEPTH = 32
BAD_ARGUMENTS = "search_code takes a query string and a non-empty list of glob strings"
NO_REPORT = "the page sent no report"
LESSON_SYNC_TIMEOUT_S = 5.0  # a control bound on the page's answer, not a measured latency
PLANNER_TIMEOUT = "planner: error: timeout"
PLANNER_CALL_FAILED = "planner: error: call failed"
PLANNER_REASONS = (
    (NO_TOOL_CALL, "prose"),
    (EMPTY_REPLY, "empty"),
    (POSITION_REFUSED, "position"),
    (PLANNER_TIMEOUT, "timeout"),
    (PLANNER_CALL_FAILED, "call"),
    ("planner: error: unexpected tool", "unexpected_tool"),
    ("planner: error: truncated", "cap"),
)


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


def _planner_reason(result: str) -> str:
    return next(
        (token for prefix, token in PLANNER_REASONS if result.startswith(prefix)), "invalid"
    )


def _search_arguments(text: str) -> tuple[str, list[str]] | None:
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(body, dict):
        return None
    query = body.get("query")
    globs = body.get("globs")
    if not isinstance(query, str) or not query:
        return None
    if not isinstance(globs, list) or not globs:
        return None
    if not all(isinstance(glob, str) and glob for glob in globs):
        return None
    return query, globs


def _assistant_calls(calls: list[TurnChunk]) -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id=call.tool_call_id,
                function=ToolCallFunction(name=call.tool_name, arguments=call.text),
            )
            for call in calls
        ],
    )


async def _queued(queue: asyncio.Queue[str | RawTag | None]) -> AsyncIterator[str | RawTag]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


class Clause(NamedTuple):
    text: str
    markers: list[Marker]


class Played(NamedTuple):
    chunk: Chunk
    lead_ms: int
    audio_ms: int
    at: float


async def _put_all(queue: asyncio.Queue[str | RawTag | None], items: list[str | RawTag]) -> None:
    for item in items:
        if item != "":
            await queue.put(item)


class TurnLoopConfig(BaseModel):
    system: str
    subject: str
    starting_from: str = ""
    root: Path | None = None
    budget: SearchBudget = Field(default_factory=SearchBudget)
    stage_gap_ms: int = Field(default=2000, gt=0)
    tool_round_max_tokens: int = Field(default=2000, gt=0)
    history_turns: int = Field(default=10, ge=0)
    scene_model: str = ""
    scene_effort: str = "high"
    scene_max_tokens: int = Field(default=128000, gt=0)
    scene_timeout_s: float = Field(default=600.0, gt=0)
    scene_ready_timeout_s: float = Field(default=20.0, gt=0)
    speculative_reasoning: bool = False
    planned: bool = True
    planner_model: str = "glm-5.3"
    planner_effort: str = "high"
    planner_max_tokens: int = Field(default=22000, gt=0)
    planner_timeout_s: float = Field(default=300.0, gt=0)


class Speculation:
    def __init__(self, text: str) -> None:
        self.text = text
        self.claimed = False
        self.grounded: asyncio.Future[Grounded] = asyncio.get_running_loop().create_future()
        self.task: asyncio.Task[None]

    def live(self) -> bool:
        if not self.task.done():
            return not self.task.cancelling()
        return not self.task.cancelled() and self.task.exception() is None


class TurnLoop:
    def __init__(
        self,
        cfg: TurnLoopConfig,
        source: InputPath,
        search: SearchCall,
        speaker: Speaker,
        transport: Connection,
        reasoning: ReasoningClient,
        registry: TurnRegistry,
        clock: Callable[[], float] = time.perf_counter,
        pace: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._search = search
        self._speaker = speaker
        self._transport = transport
        self._visuals = VisualChannel(transport)
        self._reasoning = reasoning
        self._registry = registry
        self._clock = clock
        self._pace = pace
        self._transcript = Transcript(cfg.history_turns)
        self._turns: set[asyncio.Task[None]] = set()
        self._drains: dict[str, asyncio.Task[None]] = {}
        self._pumps: dict[str, asyncio.Task[None]] = {}
        self._stagers: dict[str, asyncio.Task[None]] = {}
        self._staging: dict[str, tuple[asyncio.Queue[Clause | None], asyncio.Event]] = {}
        self._due: dict[str, dict[int, list[Marker]]] = {}
        self._played: dict[str, Played] = {}
        self._speculations: dict[str, Speculation] = {}
        self._results: dict[str, list[SearchResult]] = {}
        self._ready: dict[str, asyncio.Future[SceneReady]] = {}
        self._chunk_id = 0
        self._theme = "light"
        self._dispatched = 0
        self._lesson = LessonState()
        self._epoch = 1
        self._barrier = 0
        self._pending_barrier: int | None = None
        self._settled = asyncio.Event()
        self._settled.set()
        self._sync_wait: asyncio.Task[None] | None = None
        self._background: set[asyncio.Task[None]] = set()
        self._plan_ready = asyncio.Event()
        if not cfg.planned:
            self._plan_ready.set()
        self._planner: asyncio.Task[None] | None = None
        self._closing = False
        self._hearing = False
        transport.on_json(self._on_json)

    def _on_json(self, payload: dict[str, object]) -> None:
        try:
            message = CLIENT_MESSAGE.validate_python(payload)
        except ValidationError:
            kind = payload.get("type")
            logger.warning(
                "client.message_rejected type=%s", kind if isinstance(kind, str) else "unknown"
            )
            return
        if isinstance(message, ThemeMessage):
            self._theme = message.theme
            logger.debug("session.theme theme=%s", message.theme)
            return
        if isinstance(message, SceneReady):
            waiting = self._ready.get(message.scene_id)
            if waiting is None or waiting.done():
                logger.info("scene.ready_unexpected scene_id=%s", message.scene_id)
                return
            waiting.set_result(message)
            return
        if isinstance(message, LessonAck):
            self._on_ack(message)
            return
        if isinstance(message, LessonSynced):
            self._on_synced(message)
            return
        if isinstance(message, LessonCheckpoint):
            self._on_checkpoint(message)
            return
        self._interrupt()
        self._dispatch(message.text)

    def _on_ack(self, ack: LessonAck) -> None:
        if ack.epoch != self._epoch:
            logger.info("lesson.ack_ignored epoch=%d cue_id=%d", ack.epoch, ack.cue_id)
            return
        head = self._lesson.sent[0] if self._lesson.sent else None
        self._lesson.acknowledge(ack)
        if head is None or head.cue_id != ack.cue_id:
            return
        logger.info(
            "lesson.ack cue_id=%d outcome=%s reason=%s", ack.cue_id, ack.outcome, ack.reason
        )
        if ack.outcome == "failed":
            self._resync("failed")
        if ack.outcome != "fired" or self._lesson.sent[:1] == [head]:
            return
        if isinstance(head.tag, SceneCue):
            logger.info(
                "cursor.scene scene_id=%s n=%d cue_id=%d", head.tag.scene_id, head.tag.n, ack.cue_id
            )
            self._publish()
            return
        logger.info("cursor.step scene_id=%s n=%d cue_id=%d", ack.scene_id, head.tag.n, ack.cue_id)

    def _on_synced(self, message: LessonSynced) -> None:
        if message.epoch != self._epoch or message.barrier != self._pending_barrier:
            logger.info("lesson.synced_ignored epoch=%d barrier=%d", message.epoch, message.barrier)
            return
        self._pending_barrier = None
        self._lesson.synced(message)
        self._settled.set()
        if self._sync_wait is not None:
            self._sync_wait.cancel()
            self._sync_wait = None
        logger.info(
            "lesson.synced epoch=%d barrier=%d scene_id=%s step=%d revision=%d last_cue=%d",
            message.epoch,
            message.barrier,
            message.scene_id,
            message.step,
            message.revision,
            message.last_cue,
        )

    def _on_checkpoint(self, message: LessonCheckpoint) -> None:
        if message.epoch != self._epoch:
            logger.info("lesson.checkpoint_ignored epoch=%d", message.epoch)
            return
        self._lesson.retain(message)
        self._resync("restore")

    async def run(self) -> None:
        self._keep(self._visuals.push(LessonAttach(epoch=self._epoch)), "lesson-attach")
        if self._cfg.planned:
            self._planner = asyncio.create_task(self._plan(), name="lesson-planner")
            self._planner.add_done_callback(self._planner_done)
        async for event in self._source.events():
            if isinstance(event, SpeechStarted):
                self._hearing = True
                self._interrupt()
            elif isinstance(event, PartialTranscript):
                if self._cfg.speculative_reasoning and event.text:
                    self._prime(event.text)
            elif isinstance(event, EndOfTurn):
                self._hearing = False
                if any(not turn.cancelling() for turn in self._turns):
                    self._interrupt()
                self._dispatch(event.text)
        self._closing = True
        pending = [speculation.task for speculation in self._speculations.values()]
        if self._planner is not None:
            pending.append(self._planner)
        for task in pending:
            task.cancel()
        await asyncio.gather(*self._turns, *pending, return_exceptions=True)

    async def _plan(self) -> None:
        try:
            result: LessonPlan | str = EMPTY_REPLY
            for _ in (1, 2):
                result = await self._planner_call("connect", None, 0)
                if isinstance(result, LessonPlan):
                    self._adopt(result)
                    break
            else:
                logger.info("planner.failed stage=connect reason=%s", _planner_reason(result))
        finally:
            self._plan_ready.set()
        if self._dispatched == 0 and not self._hearing and not self._closing:
            self._dispatch(OPENING_TEXT)

    async def _planner_call(
        self, stage: str, snapshot: LessonPlan | None, protected: int
    ) -> LessonPlan | str:
        ids = [] if snapshot is None else [scene.id for scene in snapshot.scenes[:protected]]
        transcript = self._transcript.since(self._lesson.planned_through, self._cfg.history_turns)
        self._lesson.planned_through = self._transcript.latest()
        prompt = plan_prompt(
            self._cfg.subject,
            self._cfg.starting_from,
            self._cfg.root is not None,
            snapshot,
            ids,
            transcript,
        )
        start = self._clock()
        try:
            async with asyncio.timeout(self._cfg.planner_timeout_s):
                # A reasoning-client error comes back as the value; a cancel still propagates.
                (result,) = await asyncio.gather(
                    run_planner(
                        self._reasoning,
                        prompt,
                        self._cfg.planner_max_tokens,
                        self._cfg.planner_effort,
                        model=self._cfg.planner_model or None,
                    ),
                    return_exceptions=True,
                )
        except TimeoutError:
            result = PLANNER_TIMEOUT
        if isinstance(result, BaseException):
            logger.warning("planner.call_failed stage=%s error=%s", stage, type(result).__name__)
            result = f"{PLANNER_CALL_FAILED} {type(result).__name__}"
        logger.info(
            "planner.result stage=%s ms=%d valid=%s",
            stage,
            _elapsed_ms(start, self._clock()),
            isinstance(result, LessonPlan),
        )
        return result

    def _planner_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("planner.task_failed error=%s", type(task.exception()).__name__)

    def _adopt(self, plan: LessonPlan) -> None:
        self._lesson.adopt(plan)
        logger.info("lesson.planned scenes=%d", len(plan.scenes))
        self._publish()

    def _publish(self) -> None:
        scenes, current = self._lesson.statuses()
        self._keep(
            self._visuals.push(LessonStatePush(scenes=scenes, current=current)), "lesson-state"
        )

    def _dispatch(self, text: str) -> None:
        self._dispatched += 1
        turn_id = f"turn-{self._dispatched}"
        turn = asyncio.create_task(self._turn(turn_id, text), name=turn_id)
        self._turns.add(turn)
        turn.add_done_callback(self._turn_done)

    def _prime(self, text: str) -> None:
        turn_id = f"turn-{self._dispatched + 1}"
        previous = self._speculations.get(turn_id)
        if previous is not None and previous.text == text and previous.live():
            return
        if previous is not None and not previous.task.cancelling():
            previous.task.cancel()
        speculation = Speculation(text)
        speculation.task = asyncio.create_task(
            self._speculate(turn_id, speculation, None if previous is None else previous.task),
            name=f"{turn_id}-speculation",
        )
        speculation.task.add_done_callback(lambda _: self._speculation_done(turn_id, speculation))
        self._speculations[turn_id] = speculation

    def _speculation_done(self, turn_id: str, speculation: Speculation) -> None:
        task = speculation.task
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and not (speculation.claimed and speculation.grounded.done()):
            logger.error(
                "turn.speculation_failed turn_id=%s error=%s", turn_id, type(error).__name__
            )

    def _keep(self, work: Coroutine[Any, Any, None], name: str) -> None:
        task = asyncio.create_task(work, name=name)
        self._background.add(task)
        task.add_done_callback(self._kept)

    def _kept(self, task: asyncio.Task[None]) -> None:
        self._background.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "lesson.send_failed task=%s error=%s",
                task.get_name(),
                type(task.exception()).__name__,
            )

    def _sync(self, reason: str) -> None:
        self._barrier += 1
        sync = LessonSync(epoch=self._epoch, barrier=self._barrier)
        if not self._visuals.push_nowait(sync):
            self._keep(self._visuals.push(sync), "lesson-sync")
        waiting = (
            bool(self._lesson.sent) or self._lesson.resync or self._pending_barrier is not None
        )
        if waiting:
            self._arm(self._barrier)
        logger.info(
            "lesson.sync epoch=%d barrier=%d waiting=%s reason=%s",
            self._epoch,
            self._barrier,
            waiting,
            reason,
        )

    def _resync(self, reason: str) -> None:
        if self._lesson.resync and self._pending_barrier is None:
            self._sync(reason)

    def _interrupt(self) -> None:
        self._sync("barge_in")
        for speculation in self._speculations.values():
            if not speculation.task.cancelling():
                speculation.task.cancel()
        for turn in self._turns:
            if turn.cancelling():
                continue
            drain = self._drains.get(turn.get_name())
            if drain is not None:
                drain.cancel()
            turn.cancel()
        # Playout outlives the turn task, so what is queued drops even with no turn in flight.
        self._transport.flush_playout()

    def _arm(self, barrier: int) -> None:
        self._pending_barrier = barrier
        self._settled.clear()
        if self._sync_wait is not None:
            self._sync_wait.cancel()
        self._sync_wait = asyncio.create_task(self._sync_bound(barrier), name="lesson-sync-bound")

    async def _sync_bound(self, barrier: int) -> None:
        await self._pace(LESSON_SYNC_TIMEOUT_S)
        if self._pending_barrier != barrier:
            return
        self._pending_barrier = None
        self._lesson.forget()
        self._settled.set()
        logger.warning("lesson.sync_timeout epoch=%d barrier=%d", self._epoch, barrier)

    def _turn_done(self, turn: asyncio.Task[None]) -> None:
        self._turns.discard(turn)
        if turn.cancelled():
            return
        error = turn.exception()
        if error is not None:
            logger.error("turn.failed turn_id=%s error=%s", turn.get_name(), type(error).__name__)

    async def aclose(self) -> None:
        self._closing = True
        if self._planner is not None:
            self._planner.cancel()
            await asyncio.gather(self._planner, return_exceptions=True)
        tasks = [*self._turns, *(speculation.task for speculation in self._speculations.values())]
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        background = list(self._background)
        if self._sync_wait is not None:
            background.append(self._sync_wait)
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)

    async def _speculate(
        self, turn_id: str, speculation: Speculation, previous: asyncio.Task[None] | None
    ) -> None:
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        await self._plan_ready.wait()
        await self._settled.wait()
        start = self._clock()
        self._registry.open_turn(turn_id)
        try:
            prompt = self._prompt([], speculation.text, turn_id)
            queue: asyncio.Queue[str | RawTag | None] = asyncio.Queue(SPOKEN_DEPTH)
            speculation.grounded.set_result((prompt, queue))
            logger.info(
                "turn.speculation turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock())
            )
            tools = [SEARCH_CODE_TOOL] if self._cfg.root is not None else []
            await self._drain(turn_id, prompt, queue, tools)
        except asyncio.CancelledError:
            if not speculation.claimed:
                self._registry.abandon(turn_id)
            raise
        finally:
            if not speculation.grounded.done():
                speculation.grounded.cancel()
            if not speculation.claimed:
                self._results.pop(turn_id, None)

    async def _claim(self, turn_id: str, user_text: str) -> asyncio.Future[Grounded] | None:
        speculation = self._speculations.pop(turn_id, None)
        if (
            speculation is not None
            and speculation.live()
            and user_text.startswith(speculation.text)
        ):
            speculation.claimed = True
            self._drains[turn_id] = speculation.task
            return speculation.grounded
        if speculation is not None:
            if not speculation.task.cancelling():
                speculation.task.cancel()
            await asyncio.gather(speculation.task, return_exceptions=True)
        self._registry.open_turn(turn_id)
        return None

    def _prompt(self, results: list[SearchResult], user_text: str, turn_id: str) -> TurnPrompt:
        starting = (
            f"Starting from: {self._cfg.starting_from}\n\n" if self._cfg.starting_from else ""
        )
        block = lesson_block(self._lesson, user_text != OPENING_TEXT, list(self._lesson.dropped))
        system = f"{self._cfg.system}\n\nSubject: {self._cfg.subject}\n\n{starting}{block}"
        return TurnPrompt(
            system=system,
            history=self._transcript.history(before=turn_id),
            tool_context=results,
            user_text=user_text,
        )

    def _state(
        self, state: Literal["listening", "thinking", "speaking"], interrupted: bool = False
    ) -> TurnState:
        return TurnState(state=state, interrupted=interrupted)

    def _caption(self, turn_id: str, barrier: int) -> OnPlay:
        async def on_play(chunk: Chunk, lead_ms: int, audio_ms: int) -> None:
            await self._visuals.push(Caption(turn_id=turn_id, text=chunk.text, lead_ms=lead_ms))
            self._played[turn_id] = Played(chunk, lead_ms, audio_ms, self._clock())
            markers = self._due.get(turn_id, {}).pop(chunk.id, [])
            await self._send_cues(turn_id, barrier, markers, chunk.id, lead_ms, audio_ms)

        return on_play

    async def _trailing(self, turn_id: str, barrier: int, markers: list[Marker]) -> None:
        played = self._played.get(turn_id)
        if played is None:
            logger.info("tag.discarded turn_id=%s count=%d reason=no_chunk", turn_id, len(markers))
            return
        remaining = played.lead_ms + played.audio_ms - _elapsed_ms(played.at, self._clock())
        await self._send_cues(turn_id, barrier, markers, played.chunk.id, max(remaining, 0), 0)

    async def _send_cues(
        self,
        turn_id: str,
        barrier: int,
        markers: list[Marker],
        chunk_id: int,
        lead_ms: int,
        audio_ms: int,
    ) -> None:
        for at, marker in enumerate(markers):
            if barrier != self._barrier:
                logger.info(
                    "tag.discarded turn_id=%s count=%d reason=barrier", turn_id, len(markers) - at
                )
                return
            if marker.kind == "step":
                reason = self._lesson.step_tag(marker.n)
            else:
                reason = self._lesson.scene_tag(marker.n)
            if reason is not None:
                continue
            entry = self._lesson.sent[-1]
            await self._visuals.push(
                LessonCue(
                    epoch=self._epoch,
                    barrier=barrier,
                    cue_id=entry.cue_id,
                    chunk_id=chunk_id,
                    scene_id=entry.scene_id,
                    revision=entry.revision,
                    lead_ms=lead_ms,
                    audio_ms=audio_ms,
                    tag=entry.tag,
                )
            )
            logger.info(
                "lesson.cue turn_id=%s cue_id=%d kind=%s n=%d chunk_id=%d lead_ms=%d",
                turn_id,
                entry.cue_id,
                marker.kind,
                marker.n,
                chunk_id,
                lead_ms,
            )

    async def _turn(self, turn_id: str, user_text: str) -> None:
        start = self._clock()
        self._transcript.learner(turn_id, user_text)
        if user_text != OPENING_TEXT:
            await self._visuals.push(LearnerText(turn_id=turn_id, text=user_text))
        await self._plan_ready.wait()
        await self._settled.wait()
        opens = not self._lesson.opened
        barrier = self._barrier
        grounded = await self._claim(turn_id, user_text)
        self._visuals.set_grounding(self._registry, turn_id)
        await self._visuals.push(self._state("thinking"))
        try:
            queue: asyncio.Queue[str | RawTag | None] | None = None
            if grounded is None:
                prompt = self._prompt([], user_text, turn_id)
            else:
                # A cancelled turn must not cancel the future the speculation is about to
                # resolve, or set_result() raises inside the speculation.
                prompt, queue = await asyncio.shield(grounded)
            self._lesson.dropped.clear()
            await self._speaker.speak(
                self._utterance(turn_id, prompt, queue, barrier), self._caption(turn_id, barrier)
            )
            await self._await_drain(turn_id)
        except asyncio.CancelledError:
            # The drain has to stop before the id is cleared, or a late record() finds no turn.
            await self._stop_turn(turn_id)
            self._registry.abandon(turn_id)
            if turn_id == f"turn-{self._dispatched}":
                try:
                    await self._visuals.push(self._state("listening", interrupted=True))
                except Exception as error:
                    logger.warning(
                        "turn.state_push_failed turn_id=%s error=%s",
                        turn_id,
                        type(error).__name__,
                        exc_info=True,
                    )
            raise
        finally:
            await self._stop_turn(turn_id)
        if opens:
            self._lesson.opened = True
        logger.info("turn.spoken turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock()))
        await self._visuals.push(self._state("listening"))

    async def _await_drain(self, turn_id: str) -> None:
        drain = self._drains.get(turn_id)
        if drain is None:
            return
        if not drain.done():
            await asyncio.gather(drain, return_exceptions=True)
        del self._drains[turn_id]
        error = drain.exception()
        if error is not None:
            logger.error("turn.reasoning_failed turn_id=%s error=%s", turn_id, type(error).__name__)

    async def _stop_turn(self, turn_id: str) -> None:
        self._staging.pop(turn_id, None)
        self._due.pop(turn_id, None)
        self._played.pop(turn_id, None)
        self._results.pop(turn_id, None)
        await self._stop(self._pumps, turn_id)
        await self._stop(self._stagers, turn_id)
        await self._stop(self._drains, turn_id)

    async def _stop(self, tasks: dict[str, asyncio.Task[Any]], turn_id: str) -> None:
        task = tasks.pop(turn_id, None)
        if task is None or task.done():
            return
        if not task.cancelling():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _utterance(
        self,
        turn_id: str,
        prompt: TurnPrompt,
        queue: asyncio.Queue[str | RawTag | None] | None,
        barrier: int,
    ) -> AsyncIterator[Chunk]:
        if queue is None:
            queue = asyncio.Queue(SPOKEN_DEPTH)
            tools = TURN_TOOLS if self._cfg.root is not None else []
            self._drains[turn_id] = asyncio.create_task(
                self._drain(turn_id, prompt, queue, tools), name=f"{turn_id}-drain"
            )
        spoken: asyncio.Queue[Clause | None] = asyncio.Queue(SPOKEN_DEPTH)
        demand = asyncio.Event()
        self._staging[turn_id] = (spoken, demand)
        self._due[turn_id] = {}
        self._pumps[turn_id] = asyncio.create_task(
            self._pump(turn_id, queue, spoken, demand), name=f"{turn_id}-pump"
        )
        first = True
        while True:
            demand.set()
            clause = await spoken.get()
            if clause is None:
                return
            if not clause.text:
                await self._trailing(turn_id, barrier, clause.markers)
                continue
            if first:
                await self._visuals.push(self._state("speaking"))
                first = False
            self._chunk_id += 1
            chunk = Chunk(self._chunk_id, clause.text)
            if clause.markers:
                self._due[turn_id][chunk.id] = clause.markers
            self._transcript.tutor(turn_id, clause.text)
            yield chunk

    def _ground(self, turn_id: str, result: SearchResult) -> None:
        staging = self._staging.get(turn_id)
        if staging is None or turn_id in self._stagers:
            return
        spoken, demand = staging
        lead_in = lead_in_sentence([result])
        sentences = [lead_in, *(s for s in lead_in_stages(result) if s != lead_in)]
        self._stagers[turn_id] = asyncio.create_task(
            self._stage(turn_id, sentences, spoken, demand), name=f"{turn_id}-stager"
        )

    async def _stage(
        self,
        turn_id: str,
        sentences: list[str],
        spoken: asyncio.Queue[Clause | None],
        demand: asyncio.Event,
    ) -> None:
        # The lead-in answers the search as soon as the speaker asks; the per-match sentences
        # after it keep the gap, which runs from the speaker asking for more, not from the
        # previous put, so stage audio never piles up ahead of the model's first clause.
        staged = 0
        for n, sentence in enumerate(sentences):
            await demand.wait()
            if n:
                await self._pace(self._cfg.stage_gap_ms / 1000)
            if not self._admits(turn_id, sentence, "lead_in"):
                continue
            demand.clear()
            await spoken.put(Clause(sentence, []))
            staged += 1
            logger.info("turn.stage turn_id=%s n=%d", turn_id, staged)

    async def _pump(
        self,
        turn_id: str,
        queue: asyncio.Queue[str | RawTag | None],
        spoken: asyncio.Queue[Clause | None],
        demand: asyncio.Event,
    ) -> None:
        scrubber = Scrubber()
        items = clause_chunks(spoken_text(_queued(queue), scrubber))
        markers: list[Marker] = []
        # A clause is pulled only once the speaker has asked for one, so a stalled speaker still
        # backs the model stream up at SPOKEN_DEPTH deltas rather than at the chunker's buffer.
        while True:
            await demand.wait()
            item = await anext(items, None)
            if item is None:
                break
            if isinstance(item, RawTag):
                marker = parse_marker(item)
                if isinstance(marker, str):
                    logger.info(
                        "tag.dropped turn_id=%s kind=%s reason=%s chars=%d",
                        turn_id,
                        tag_name(item),
                        marker,
                        len(item.text),
                    )
                    self._lesson.dropped.append(f"<{tag_name(item)}>: {marker}")
                else:
                    markers.append(marker)
                continue
            if not self._admits(turn_id, item, "model"):
                continue
            await self._stop(self._stagers, turn_id)
            demand.clear()
            await spoken.put(Clause(item, markers))
            markers = []
        await self._stop(self._stagers, turn_id)
        if markers:
            await spoken.put(Clause("", markers))
        await spoken.put(None)
        if scrubber.dropped:
            counts = " ".join(f"{key}={count}" for key, count in sorted(scrubber.dropped.items()))
            logger.info("turn.markup_dropped turn_id=%s %s", turn_id, counts)

    def _admits(self, turn_id: str, text: str, source: str) -> bool:
        verdict = self._registry.verify_chunk(turn_id, text, source=source)
        ungrounded = verdict.ungrounded
        if self._cfg.root is None:
            ungrounded = [position for position in ungrounded if position.path]
        if not ungrounded:
            return True
        logger.warning(
            "turn.chunk_withheld turn_id=%s source=%s ungrounded=%d",
            turn_id,
            source,
            len(ungrounded),
        )
        return False

    async def _drain(
        self,
        turn_id: str,
        prompt: TurnPrompt,
        queue: asyncio.Queue[str | RawTag | None],
        tools: Sequence[dict[str, object]],
    ) -> None:
        try:
            calls: list[TurnChunk] = []
            tags = TagSplitter()
            stream = self._reasoning.start_turn(
                prompt, tools=list(tools) or None, max_tokens=self._cfg.tool_round_max_tokens
            )
            async for chunk in stream:
                if chunk.kind == "spoken":
                    await _put_all(queue, tags.feed(chunk.text))
                elif chunk.kind == "tool_call":
                    calls.append(chunk)
            await _put_all(queue, tags.finish())
            answerable = [call for call in calls if call.tool_call_id and call.tool_name]
            if len(answerable) != len(calls):
                logger.warning("turn.tool_call_incomplete turn_id=%s", turn_id)
            if answerable:
                await queue.put("\n")
                await self._follow_up(turn_id, prompt, answerable, queue)
        except BaseException:
            # Nothing consumes the queue once the turn unwinds, so the sentinel takes a slot
            # instead of waiting for one.
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)
            raise
        await queue.put(None)

    async def _follow_up(
        self,
        turn_id: str,
        prompt: TurnPrompt,
        calls: list[TurnChunk],
        queue: asyncio.Queue[str | RawTag | None],
    ) -> None:
        exchange = [_assistant_calls(calls)]
        for call in calls:
            answer = await self._answer(turn_id, call)
            exchange.append(Message(role="tool", content=answer, tool_call_id=call.tool_call_id))
        tags = TagSplitter()
        follow_up = prompt.model_copy(update={"tool_exchange": exchange})
        # The follow-up carries no tools, so the model cannot open a round this loop will not serve.
        async for chunk in self._reasoning.start_turn(follow_up):
            if chunk.kind == "spoken":
                await _put_all(queue, tags.feed(chunk.text))
        await _put_all(queue, tags.finish())

    async def _check(self, turn_id: str, push: ScenePush) -> SceneReady:
        waiting: asyncio.Future[SceneReady] = asyncio.get_running_loop().create_future()
        self._ready[turn_id] = waiting
        try:
            await self._visuals.push(push)
            async with asyncio.timeout(self._cfg.scene_ready_timeout_s):
                return await waiting
        except TimeoutError:
            return SceneReady(scene_id=turn_id, ok=False, steps=0, error=NO_REPORT)
        finally:
            self._ready.pop(turn_id, None)

    async def _answer(self, turn_id: str, call: TurnChunk) -> str:
        if call.tool_name in VISUAL_TOOL_NAMES:
            return await dispatch_visual_tool(call.tool_name, call.text, self._visuals)
        if call.tool_name != SEARCH_CODE:
            logger.info("turn.tool_unrouted turn_id=%s tool=%s", turn_id, call.tool_name)
            return f"{call.tool_name} is not available in this turn"
        arguments = _search_arguments(call.text)
        if arguments is None:
            logger.warning("turn.tool_arguments turn_id=%s tool=%s", turn_id, call.tool_name)
            return BAD_ARGUMENTS
        query, globs = arguments
        start = self._clock()
        result = await self._search(query, globs, self._cfg.root, self._cfg.budget)
        self._registry.record(turn_id, result)
        self._results.setdefault(turn_id, []).append(result)
        logger.info("turn.search turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock()))
        self._ground(turn_id, result)
        return result.model_dump_json()
