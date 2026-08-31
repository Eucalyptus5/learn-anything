import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.input_path import EndOfTurn, InputPath
from tutor.lead_in import lead_in_sentence, lead_in_stages, opener_key
from tutor.prompt import (
    SEARCH_CODE_TOOL,
    Message,
    ToolCall,
    ToolCallFunction,
    TurnPrompt,
    derive_globs,
)
from tutor.reasoning import ReasoningClient, TurnChunk
from tutor.speech import Speaker
from tutor.tools.models import SearchBudget, SearchResult
from tutor.tools.provenance import TurnRegistry

logger = logging.getLogger(__name__)

SearchCall = Callable[[str, Sequence[str], Path, SearchBudget], Awaitable[SearchResult]]

SEARCH_CODE = "search_code"
OPENER = "thinking"
SPOKEN_DEPTH = 32
BAD_ARGUMENTS = "search_code takes a query string and a non-empty list of glob strings"


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


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


async def _queued(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while True:
        text = await queue.get()
        if text is None:
            return
        yield text


class TurnLoopConfig(BaseModel):
    system: str
    subject: str
    root: Path
    budget: SearchBudget = Field(default_factory=SearchBudget)
    stage_gap_ms: int = Field(default=2000, gt=0)


class TurnLoop:
    def __init__(
        self,
        cfg: TurnLoopConfig,
        source: InputPath,
        search: SearchCall,
        speaker: Speaker,
        reasoning: ReasoningClient,
        registry: TurnRegistry,
        clock: Callable[[], float] = time.perf_counter,
        pace: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._search = search
        self._speaker = speaker
        self._reasoning = reasoning
        self._registry = registry
        self._clock = clock
        self._pace = pace
        self._turns: set[asyncio.Task[None]] = set()
        self._drains: dict[str, asyncio.Task[None]] = {}
        self._pumps: dict[str, asyncio.Task[None]] = {}
        self._stagers: dict[str, asyncio.Task[None]] = {}
        self._dispatched = 0

    async def run(self) -> None:
        async for event in self._source.events():
            if not isinstance(event, EndOfTurn):
                continue
            self._dispatched += 1
            turn_id = f"turn-{self._dispatched}"
            turn = asyncio.create_task(self._turn(turn_id, event.text), name=turn_id)
            self._turns.add(turn)
            turn.add_done_callback(self._turn_done)
        await asyncio.gather(*self._turns, return_exceptions=True)

    def _turn_done(self, turn: asyncio.Task[None]) -> None:
        self._turns.discard(turn)
        if turn.cancelled():
            return
        error = turn.exception()
        if error is not None:
            logger.error("turn.failed turn_id=%s error=%s", turn.get_name(), type(error).__name__)

    async def aclose(self) -> None:
        turns = list(self._turns)
        for turn in turns:
            turn.cancel()
        await asyncio.gather(*turns, return_exceptions=True)

    async def _turn(self, turn_id: str, user_text: str) -> None:
        start = self._clock()
        self._registry.open_turn(turn_id)
        try:
            await self._speaker.speak_opener(OPENER)
            globs = await derive_globs(user_text, self._cfg.root)
            result = await self._search(user_text, globs, self._cfg.root, self._cfg.budget)
            self._registry.record(turn_id, result)
            logger.info(
                "turn.grounding turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock())
            )
            await self._speaker.speak_opener(opener_key([result]))
            prompt = TurnPrompt(
                system=f"{self._cfg.system}\n\nSubject: {self._cfg.subject}",
                tool_context=[result],
                user_text=user_text,
            )
            await self._speaker.speak(self._utterance(turn_id, result, prompt))
            await self._report_drain(turn_id)
        except asyncio.CancelledError:
            # The drain has to stop before the id is cleared, or a late record() finds no turn.
            await self._stop_turn(turn_id)
            self._registry.abandon(turn_id)
            raise
        finally:
            await self._stop_turn(turn_id)
        logger.info("turn.spoken turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock()))

    async def _report_drain(self, turn_id: str) -> None:
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
        await self._stop(self._pumps, turn_id)
        await self._stop(self._stagers, turn_id)
        await self._stop(self._drains, turn_id)

    async def _stop(self, tasks: dict[str, asyncio.Task[None]], turn_id: str) -> None:
        task = tasks.pop(turn_id, None)
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _utterance(
        self, turn_id: str, result: SearchResult, prompt: TurnPrompt
    ) -> AsyncIterator[str]:
        lead_in = lead_in_sentence([result])
        if self._admits(turn_id, lead_in, "lead_in"):
            yield lead_in
        stages = [sentence for sentence in lead_in_stages(result) if sentence != lead_in]
        queue: asyncio.Queue[str | None] = asyncio.Queue(SPOKEN_DEPTH)
        spoken: asyncio.Queue[str | None] = asyncio.Queue(SPOKEN_DEPTH)
        demand = asyncio.Event()
        self._drains[turn_id] = asyncio.create_task(
            self._drain(turn_id, prompt, queue), name=f"{turn_id}-drain"
        )
        self._stagers[turn_id] = asyncio.create_task(
            self._stage(turn_id, stages, spoken, demand), name=f"{turn_id}-stager"
        )
        self._pumps[turn_id] = asyncio.create_task(
            self._pump(turn_id, queue, spoken, demand), name=f"{turn_id}-pump"
        )
        while True:
            demand.set()
            text = await spoken.get()
            if text is None:
                return
            yield text

    async def _stage(
        self,
        turn_id: str,
        sentences: list[str],
        spoken: asyncio.Queue[str | None],
        demand: asyncio.Event,
    ) -> None:
        # The gap runs from the speaker asking for more, not from the previous put, so stage
        # audio never piles up in the playout buffer ahead of the model's first clause.
        staged = 0
        for sentence in sentences:
            await demand.wait()
            await self._pace(self._cfg.stage_gap_ms / 1000)
            if not self._admits(turn_id, sentence, "lead_in"):
                continue
            demand.clear()
            await spoken.put(sentence)
            staged += 1
            logger.info("turn.stage turn_id=%s n=%d", turn_id, staged)

    async def _pump(
        self,
        turn_id: str,
        queue: asyncio.Queue[str | None],
        spoken: asyncio.Queue[str | None],
        demand: asyncio.Event,
    ) -> None:
        scrubber = Scrubber()
        clauses = clause_chunks(spoken_text(_queued(queue), scrubber))
        # A clause is pulled only once the speaker has asked for one, so a stalled speaker still
        # backs the model stream up at SPOKEN_DEPTH deltas rather than at the chunker's buffer.
        while True:
            await demand.wait()
            clause = await anext(clauses, None)
            if clause is None:
                break
            if not self._admits(turn_id, clause, "model"):
                continue
            await self._stop(self._stagers, turn_id)
            demand.clear()
            await spoken.put(clause)
        await self._stop(self._stagers, turn_id)
        await spoken.put(None)
        if scrubber.dropped:
            counts = " ".join(f"{key}={count}" for key, count in sorted(scrubber.dropped.items()))
            logger.info("turn.markup_dropped turn_id=%s %s", turn_id, counts)

    def _admits(self, turn_id: str, text: str, source: str) -> bool:
        verdict = self._registry.verify_chunk(turn_id, text, source=source)
        if verdict.ok:
            return True
        logger.warning(
            "turn.chunk_withheld turn_id=%s source=%s ungrounded=%d",
            turn_id,
            source,
            len(verdict.ungrounded),
        )
        return False

    async def _drain(
        self, turn_id: str, prompt: TurnPrompt, queue: asyncio.Queue[str | None]
    ) -> None:
        try:
            calls: list[TurnChunk] = []
            async for chunk in self._reasoning.start_turn(prompt, tools=[SEARCH_CODE_TOOL]):
                if chunk.kind == "spoken":
                    await queue.put(chunk.text)
                elif chunk.kind == "tool_call":
                    calls.append(chunk)
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
        queue: asyncio.Queue[str | None],
    ) -> None:
        exchange = [_assistant_calls(calls)]
        for call in calls:
            answer = await self._answer(turn_id, call)
            exchange.append(Message(role="tool", content=answer, tool_call_id=call.tool_call_id))
        follow_up = prompt.model_copy(update={"tool_exchange": exchange})
        # The follow-up carries no tools, so the model cannot open a round this loop will not serve.
        async for chunk in self._reasoning.start_turn(follow_up):
            if chunk.kind == "spoken":
                await queue.put(chunk.text)

    async def _answer(self, turn_id: str, call: TurnChunk) -> str:
        if call.tool_name != SEARCH_CODE:
            logger.info("turn.tool_unrouted turn_id=%s tool=%s", turn_id, call.tool_name)
            return f"{call.tool_name} is not available in this turn"
        arguments = _search_arguments(call.text)
        if arguments is None:
            logger.warning("turn.tool_arguments turn_id=%s tool=%s", turn_id, call.tool_name)
            return BAD_ARGUMENTS
        query, globs = arguments
        result = await self._search(query, globs, self._cfg.root, self._cfg.budget)
        self._registry.record(turn_id, result)
        return result.model_dump_json()
