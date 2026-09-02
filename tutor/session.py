import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.input_path import EndOfTurn, InputPath, PartialTranscript, SpeechStarted
from tutor.lead_in import lead_in_sentence, lead_in_stages, opener_key
from tutor.pedagogy import PedagogyState, TurnOutcome, parse_outcome
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
from tutor.transport import Connection

logger = logging.getLogger(__name__)

SearchCall = Callable[[str, Sequence[str], Path, SearchBudget], Awaitable[SearchResult]]
Grounded = tuple[SearchResult, TurnPrompt, asyncio.Queue[str | None]]

SEARCH_CODE = "search_code"
OPENER = "thinking"
SPOKEN_DEPTH = 32
BAD_ARGUMENTS = "search_code takes a query string and a non-empty list of glob strings"
OUTCOME_MARKER = "<outcome>"
OUTCOME_INSTRUCTION = (
    f"End every reply with a line holding exactly {OUTCOME_MARKER} followed by one JSON object "
    'with "signal" (one of covered, follow_up, correct, misconception, or null) and '
    '"settling_positions" (a list of objects with "path" and "line") naming where a '
    "misconception is settled. Nothing after the marker is spoken."
)


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


class OutcomeSplitter:
    def __init__(self) -> None:
        self._held = ""
        self._tail: str | None = None

    def feed(self, delta: str) -> str:
        if self._tail is not None:
            self._tail += delta
            return ""
        text = self._held + delta
        at = text.find(OUTCOME_MARKER)
        if at >= 0:
            self._held = ""
            self._tail = text[at + len(OUTCOME_MARKER) :]
            return text[:at]
        prefixes = range(1, len(OUTCOME_MARKER))
        held = max((n for n in prefixes if text.endswith(OUTCOME_MARKER[:n])), default=0)
        self._held = text[len(text) - held :]
        return text[: len(text) - held]

    def finish(self) -> tuple[str, TurnOutcome]:
        if self._tail is None:
            held, self._held = self._held, ""
            return held, TurnOutcome()
        return "", parse_outcome(self._tail.strip())


class TurnLoopConfig(BaseModel):
    system: str
    subject: str
    root: Path
    budget: SearchBudget = Field(default_factory=SearchBudget)
    stage_gap_ms: int = Field(default=2000, gt=0)
    speculative_reasoning: bool = False


class Speculation:
    def __init__(self, text: str) -> None:
        self.text = text
        self.claimed = False
        self.grounded: asyncio.Future[Grounded] = asyncio.get_running_loop().create_future()
        self.task: asyncio.Task[TurnOutcome]

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
        self._reasoning = reasoning
        self._registry = registry
        self._clock = clock
        self._pace = pace
        self._pedagogy = PedagogyState()
        self._turns: set[asyncio.Task[None]] = set()
        self._drains: dict[str, asyncio.Task[TurnOutcome]] = {}
        self._pumps: dict[str, asyncio.Task[None]] = {}
        self._stagers: dict[str, asyncio.Task[None]] = {}
        self._speculations: dict[str, Speculation] = {}
        self._dispatched = 0

    async def run(self) -> None:
        async for event in self._source.events():
            if isinstance(event, SpeechStarted):
                self._interrupt()
            elif isinstance(event, PartialTranscript):
                if self._cfg.speculative_reasoning and event.text:
                    self._prime(event.text)
            elif isinstance(event, EndOfTurn):
                self._dispatched += 1
                turn_id = f"turn-{self._dispatched}"
                turn = asyncio.create_task(self._turn(turn_id, event.text), name=turn_id)
                self._turns.add(turn)
                turn.add_done_callback(self._turn_done)
        speculations = [speculation.task for speculation in self._speculations.values()]
        for task in speculations:
            task.cancel()
        await asyncio.gather(*self._turns, *speculations, return_exceptions=True)

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

    def _interrupt(self) -> None:
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

    def _turn_done(self, turn: asyncio.Task[None]) -> None:
        self._turns.discard(turn)
        if turn.cancelled():
            return
        error = turn.exception()
        if error is not None:
            logger.error("turn.failed turn_id=%s error=%s", turn.get_name(), type(error).__name__)

    async def aclose(self) -> None:
        tasks = [*self._turns, *(speculation.task for speculation in self._speculations.values())]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _speculate(
        self, turn_id: str, speculation: Speculation, previous: asyncio.Task[TurnOutcome] | None
    ) -> TurnOutcome:
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        start = self._clock()
        self._registry.open_turn(turn_id)
        try:
            globs = await self._globs(speculation.text)
            result = await self._search(speculation.text, globs, self._cfg.root, self._cfg.budget)
            self._registry.record(turn_id, result)
            logger.info(
                "turn.speculation turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock())
            )
            prompt = self._prompt(result, speculation.text)
            queue: asyncio.Queue[str | None] = asyncio.Queue(SPOKEN_DEPTH)
            speculation.grounded.set_result((result, prompt, queue))
            return await self._drain(turn_id, prompt, queue)
        except asyncio.CancelledError:
            if not speculation.claimed:
                self._registry.abandon(turn_id)
            raise
        except Exception as error:
            if speculation.claimed and not speculation.grounded.done():
                speculation.grounded.set_exception(error)
            raise
        finally:
            if not speculation.grounded.done():
                speculation.grounded.cancel()

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

    async def _globs(self, user_text: str) -> list[str]:
        settled = list(dict.fromkeys(p.path for p in self._pedagogy.settling_positions))
        if settled:
            return settled
        return await derive_globs(user_text, self._cfg.root)

    def _prompt(self, result: SearchResult, user_text: str) -> TurnPrompt:
        system = (
            f"{self._cfg.system}\n\nSubject: {self._cfg.subject}\n\n"
            f"{self._pedagogy.prompt_directive()}\n\n{OUTCOME_INSTRUCTION}"
        )
        return TurnPrompt(system=system, tool_context=[result], user_text=user_text)

    async def _turn(self, turn_id: str, user_text: str) -> None:
        start = self._clock()
        grounded = await self._claim(turn_id, user_text)
        try:
            await self._speaker.speak_opener(OPENER)
            queue: asyncio.Queue[str | None] | None = None
            if grounded is None:
                globs = await self._globs(user_text)
                result = await self._search(user_text, globs, self._cfg.root, self._cfg.budget)
                self._registry.record(turn_id, result)
                logger.info(
                    "turn.grounding turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock())
                )
                prompt = self._prompt(result, user_text)
            else:
                # A cancelled turn must not cancel the future the speculation is about to
                # resolve, or set_result() raises inside the speculation.
                result, prompt, queue = await asyncio.shield(grounded)
            await self._speaker.speak_opener(opener_key([result]))
            await self._speaker.speak(self._utterance(turn_id, result, prompt, queue))
            outcome = await self._report_drain(turn_id)
        except asyncio.CancelledError:
            # The drain has to stop before the id is cleared, or a late record() finds no turn.
            await self._stop_turn(turn_id)
            self._registry.abandon(turn_id)
            raise
        finally:
            await self._stop_turn(turn_id)
        logger.info("turn.spoken turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock()))
        phase = self._pedagogy.advance(outcome)
        logger.info("turn.outcome turn_id=%s signal=%s phase=%s", turn_id, outcome.signal, phase)

    async def _report_drain(self, turn_id: str) -> TurnOutcome:
        drain = self._drains.get(turn_id)
        if drain is None:
            return TurnOutcome()
        if not drain.done():
            await asyncio.gather(drain, return_exceptions=True)
        del self._drains[turn_id]
        error = drain.exception()
        if error is not None:
            logger.error("turn.reasoning_failed turn_id=%s error=%s", turn_id, type(error).__name__)
            return TurnOutcome()
        return drain.result()

    async def _stop_turn(self, turn_id: str) -> None:
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
        result: SearchResult,
        prompt: TurnPrompt,
        queue: asyncio.Queue[str | None] | None,
    ) -> AsyncIterator[str]:
        lead_in = lead_in_sentence([result])
        if self._admits(turn_id, lead_in, "lead_in"):
            yield lead_in
        stages = [sentence for sentence in lead_in_stages(result) if sentence != lead_in]
        if queue is None:
            queue = asyncio.Queue(SPOKEN_DEPTH)
            self._drains[turn_id] = asyncio.create_task(
                self._drain(turn_id, prompt, queue), name=f"{turn_id}-drain"
            )
        spoken: asyncio.Queue[str | None] = asyncio.Queue(SPOKEN_DEPTH)
        demand = asyncio.Event()
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
    ) -> TurnOutcome:
        try:
            calls: list[TurnChunk] = []
            splitter = OutcomeSplitter()
            async for chunk in self._reasoning.start_turn(prompt, tools=[SEARCH_CODE_TOOL]):
                if chunk.kind == "spoken":
                    text = splitter.feed(chunk.text)
                    if text:
                        await queue.put(text)
                elif chunk.kind == "tool_call":
                    calls.append(chunk)
            text, outcome = splitter.finish()
            if text:
                await queue.put(text)
            answerable = [call for call in calls if call.tool_call_id and call.tool_name]
            if len(answerable) != len(calls):
                logger.warning("turn.tool_call_incomplete turn_id=%s", turn_id)
            if answerable:
                await queue.put("\n")
                outcome = await self._follow_up(turn_id, prompt, answerable, queue)
        except BaseException:
            # Nothing consumes the queue once the turn unwinds, so the sentinel takes a slot
            # instead of waiting for one.
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(None)
            raise
        await queue.put(None)
        return outcome

    async def _follow_up(
        self,
        turn_id: str,
        prompt: TurnPrompt,
        calls: list[TurnChunk],
        queue: asyncio.Queue[str | None],
    ) -> TurnOutcome:
        exchange = [_assistant_calls(calls)]
        for call in calls:
            answer = await self._answer(turn_id, call)
            exchange.append(Message(role="tool", content=answer, tool_call_id=call.tool_call_id))
        splitter = OutcomeSplitter()
        follow_up = prompt.model_copy(update={"tool_exchange": exchange})
        # The follow-up carries no tools, so the model cannot open a round this loop will not serve.
        async for chunk in self._reasoning.start_turn(follow_up):
            if chunk.kind == "spoken":
                text = splitter.feed(chunk.text)
                if text:
                    await queue.put(text)
        text, outcome = splitter.finish()
        if text:
            await queue.put(text)
        return outcome

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
