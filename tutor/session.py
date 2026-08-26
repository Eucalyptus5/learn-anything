import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from tutor.chunker import clause_chunks
from tutor.input_path import EndOfTurn, InputPath
from tutor.lead_in import lead_in_sentence
from tutor.prompt import TurnPrompt, derive_globs
from tutor.reasoning import ReasoningClient, TurnStream
from tutor.speech import Speaker
from tutor.tools.models import SearchBudget, SearchResult
from tutor.tools.provenance import TurnRegistry

logger = logging.getLogger(__name__)

SearchCall = Callable[[str, Sequence[str], Path, SearchBudget], Awaitable[SearchResult]]


def _elapsed_ms(start: float, now: float) -> int:
    return int((now - start) * 1000)


async def _spoken(stream: TurnStream) -> AsyncIterator[str]:
    async for chunk in stream:
        if chunk.kind == "spoken":
            yield chunk.text


class TurnLoopConfig(BaseModel):
    system: str
    subject: str
    root: Path
    budget: SearchBudget = Field(default_factory=SearchBudget)


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
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._search = search
        self._speaker = speaker
        self._reasoning = reasoning
        self._registry = registry
        self._clock = clock
        self._turns: set[asyncio.Task[None]] = set()
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
            globs = await derive_globs(user_text, self._cfg.root)
            result = await self._search(user_text, globs, self._cfg.root, self._cfg.budget)
            self._registry.record(turn_id, result)
            logger.info(
                "turn.grounding turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock())
            )
            prompt = TurnPrompt(
                system=f"{self._cfg.system}\n\nSubject: {self._cfg.subject}",
                tool_context=[result],
                user_text=user_text,
            )
            stream = self._reasoning.start_turn(prompt)
            await self._speaker.speak(self._utterance(lead_in_sentence([result]), stream))
        except asyncio.CancelledError:
            self._registry.abandon(turn_id)
            raise
        logger.info("turn.spoken turn_id=%s ms=%d", turn_id, _elapsed_ms(start, self._clock()))

    async def _utterance(self, lead_in: str, stream: TurnStream) -> AsyncIterator[str]:
        yield lead_in
        async for clause in clause_chunks(_spoken(stream)):
            yield clause
