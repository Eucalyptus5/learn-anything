"""Barge-in through the real stack: a scripted turn is interrupted mid-answer by an injected
SpeechStarted and the next turn is dispatched in the same step. Reports how fast playout is
flushed, when the last old frame left, when the next turn reached the speaker, and how long the
cancelled clause's native synthesis kept running under the next turn's.
"""

import argparse
import asyncio
import contextlib
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from scripts.bench_turn import (
    SAMPLES,
    SUBJECT,
    UTTERANCES,
    WARMUP,
    BenchTransport,
    Enqueued,
    MeteredReasoning,
    OutcomeWatch,
    ScriptedSource,
    TaggedSynth,
    TurnRecord,
    is_model,
    model_text,
)
from tutor.app import Models, build_loop, load_models
from tutor.config import Settings, settings
from tutor.input_path import EndOfTurn, SpeechStarted
from tutor.reasoning import ReasoningClient
from tutor.session import TurnLoop
from tutor.tts import KokoroSynthesizer

Interval = tuple[float, float, str]


def last_old_frame_ms(frames: Sequence[tuple[float, bool]], t_c: float, t_flush: float) -> int:
    last = max((t for t, real in frames if real and t <= t_flush), default=None)
    if last is None:
        return 0
    return max(0, int((last - t_c) * 1000))


def overlap_ms(intervals: Sequence[Interval], t_c: float) -> tuple[int, int] | None:
    old = next((end for start, end, _ in intervals if start <= t_c < end), None)
    new = min(((start, end) for start, end, _ in intervals if start > t_c), default=None)
    if old is None or new is None:
        return None
    start, end = new
    return max(0, int((old - start) * 1000)), int((end - start) * 1000)


def accept_ms(ledger: Sequence[Enqueued], t_c: float) -> int | None:
    at = next((entry.at for entry in ledger if entry.at > t_c and entry.text is None), None)
    return None if at is None else int((at - t_c) * 1000)


class Sample(BaseModel):
    flush_ms: int | None = None
    last_frame_ms: int | None = None
    accept_ms: int | None = None
    next_first_sound_ms: int | None = None
    overlap_ms: int | None = None
    first_synth_ms: int | None = None
    missed: bool


class BusySynth(TaggedSynth):
    def __init__(self, inner: KokoroSynthesizer) -> None:
        super().__init__(inner)
        self._starts: list[float] = []
        self.intervals: list[Interval] = []
        self.errors: list[str] = []

    @property
    def busy(self) -> bool:
        return len(self._starts) > len(self.intervals)

    def synthesize(self, text: str) -> np.ndarray:
        start = time.perf_counter()
        self._starts.append(start)
        try:
            return super().synthesize(text)
        except Exception as error:
            self.errors.append(type(error).__name__)
            raise
        finally:
            self.intervals.append((start, time.perf_counter(), text))


class FlushTransport(BenchTransport):
    def __init__(self, synth: BusySynth, openers: dict[str, np.ndarray]) -> None:
        super().__init__(synth, openers)
        self.flushes: list[float] = []

    def flush_playout(self) -> None:
        self.flushes.append(time.perf_counter())
        super().flush_playout()


class FailureWatch(OutcomeWatch):
    def __init__(self) -> None:
        super().__init__()
        self.failed: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg.startswith("turn.failed "):
            self.failed.add(str(record.args[0]))
        return super().filter(record)


class Pending(NamedTuple):
    turn_id: str
    record: TurnRecord
    ledger_since: int
    frames_since: int


class Bench:
    def __init__(
        self,
        loop_task: asyncio.Task[None],
        drain_task: asyncio.Task[None],
        source: ScriptedSource,
        transport: FlushTransport,
        reasoning: MeteredReasoning,
        synth: BusySynth,
        watch: FailureWatch,
        loop: TurnLoop,
    ) -> None:
        self._loop_task = loop_task
        self._drain_task = drain_task
        self._source = source
        self._transport = transport
        self.reasoning = reasoning
        self.synth = synth
        self.watch = watch
        self._loop = loop
        self._dispatched = 0

    @classmethod
    async def boot(cls, cfg: Settings, root: Path, subject: str) -> "Bench":
        cfg = cfg.model_copy(update={"subject": subject, "repo_root": root})
        loaded = await asyncio.to_thread(load_models)
        synth = BusySynth(loaded.synth)
        models = Models(
            partial=loaded.partial, final=loaded.final, synth=synth, openers=loaded.openers
        )
        reasoning = MeteredReasoning(ReasoningClient(cfg))
        transport = FlushTransport(synth, models.openers)
        source = ScriptedSource()
        loop = build_loop(cfg, models, reasoning, source, transport)
        watch = FailureWatch()
        logging.getLogger("tutor.session").addFilter(watch)
        loop_task = asyncio.create_task(loop.run(), name="bench-loop")
        drain_task = asyncio.create_task(transport.drain(), name="bench-drain")
        return cls(loop_task, drain_task, source, transport, reasoning, synth, watch, loop)

    def dispatch(self, text: str) -> Pending:
        record = self.reasoning.begin()
        self._dispatched += 1
        pending = Pending(
            f"turn-{self._dispatched}",
            record,
            len(self._transport.ledger),
            len(self._transport.frames),
        )
        self._source.inject(EndOfTurn(text=text))
        return pending

    async def barge_in(self, pending: Pending, next_text: str) -> tuple[Sample, Pending]:
        transport = self._transport
        synth = self.synth
        seen = self.watch.seen

        checked = pending.ledger_since
        substance = False

        # model_text() parses the outcome tail, so it runs once per enqueued buffer, not per poll.
        def armed() -> bool:
            nonlocal checked, substance
            if pending.turn_id in seen:
                return True
            if not substance and len(transport.ledger) != checked:
                checked = len(transport.ledger)
                model = model_text(pending.record.streams)
                played = transport.ledger[pending.ledger_since : checked]
                substance = any(is_model(entry.text, model) for entry in played)
            return substance and synth.busy

        await transport.wait_frames(armed)
        if pending.turn_id in seen:
            return Sample(missed=True), self.dispatch(next_text)

        flushes_since = len(transport.flushes)
        t_c = time.perf_counter()
        self._source.inject(SpeechStarted())
        following = self.dispatch(next_text)

        def settled() -> bool:
            if following.turn_id in seen:
                return True
            if len(transport.flushes) == flushes_since:
                return False
            t_flush = transport.flushes[flushes_since]
            return (
                accept_ms(transport.ledger[following.ledger_since :], t_c) is not None
                and any(
                    real and t > t_flush for t, real in transport.frames[following.frames_since :]
                )
                and overlap_ms(synth.intervals, t_c) is not None
            )

        await transport.wait_frames(settled)

        t_flush = transport.flushes[flushes_since]
        first_new = next(
            (t for t, real in transport.frames[following.frames_since :] if real and t > t_flush),
            None,
        )
        overlap = overlap_ms(synth.intervals, t_c)
        return Sample(
            flush_ms=int((t_flush - t_c) * 1000),
            last_frame_ms=last_old_frame_ms(transport.frames[pending.frames_since :], t_c, t_flush),
            accept_ms=accept_ms(transport.ledger[following.ledger_since :], t_c),
            next_first_sound_ms=None if first_new is None else int((first_new - t_c) * 1000),
            overlap_ms=None if overlap is None else overlap[0],
            first_synth_ms=None if overlap is None else overlap[1],
            missed=False,
        ), following

    async def aclose(self) -> None:
        self._source.inject(SpeechStarted())
        self._source.close()
        try:
            await self._loop_task
        finally:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            logging.getLogger("tutor.session").removeFilter(self.watch)
            await self._loop.aclose()
            await self.reasoning.aclose()
            await self._transport.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--root", type=Path, default=REPO / "tests" / "data" / "fixture_repo")
    parser.add_argument("--model", default=None)
    return parser


def report(cfg: Settings, args: argparse.Namespace, samples: list[Sample], bench: Bench) -> None:
    print(
        "t_c: perf_counter read just before SpeechStarted is injected while the interrupted turn "
        "is mid-answer: a buffer synthesized from a model-authored clause is already enqueued, "
        "the synthesizer is inside a synthesize call, and no turn.outcome has been logged."
    )
    print(
        "flush latency: t_c to the loop's flush_playout call. last audio frame: t_c to the last "
        "48 kHz frame carrying the interrupted turn's audio (the real frames before the flush), "
        "floored at 0; the track ticks every 20 ms, and the browser's jitter buffer is excluded."
    )
    print(
        "next turn accepted: t_c to the next turn's cached opener reaching play(). next turn first "
        "sound: t_c to the first real frame after the flush."
    )
    print(
        "synth overlap: how long the cancelled clause's native synthesis kept running after the "
        "next turn's first synthesize call started. first synth after barge-in: that call's "
        "duration."
    )
    print(
        f"model={cfg.reasoning_model}  samples={len(samples)} (plus {WARMUP} discarded warm-ups)  "
        f"subject={args.subject!r}  root={args.root}"
    )
    summarize("flush latency", [s.flush_ms for s in samples if s.flush_ms is not None])
    summarize("last audio frame", [s.last_frame_ms for s in samples if s.last_frame_ms is not None])
    summarize("next turn accepted", [s.accept_ms for s in samples if s.accept_ms is not None])
    summarize(
        "next turn first sound",
        [s.next_first_sound_ms for s in samples if s.next_first_sound_ms is not None],
    )
    summarize("synth overlap", [s.overlap_ms for s in samples if s.overlap_ms is not None])
    summarize(
        "first synth after barge-in",
        [s.first_synth_ms for s in samples if s.first_synth_ms is not None],
    )
    missed = sum(1 for s in samples if s.missed)
    print(f"missed {missed}/{len(samples)}")
    print(f"synth errors {len(bench.synth.errors)}")
    print(f"turn.failed {len(bench.watch.failed)}")
    ledger = bench.reasoning.ledger
    print(
        f"turns={ledger.turns} prompt_tokens={ledger.total.prompt_tokens} "
        f"completion_tokens={ledger.total.completion_tokens} "
        f"cost_usd={ledger.cost_usd():.6f} list_usd={ledger.cost_usd(list_price=True):.6f}"
    )
    print(
        "the ledger counts only streams that ran to completion; a cancelled stream's tokens are "
        "billed but not counted here."
    )


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No barge-in latency can be reported. Populate .env and rerun.")
        return 2
    if args.model:
        cfg = cfg.model_copy(update={"reasoning_model": args.model})

    bench = await Bench.boot(cfg, args.root.resolve(), args.subject)
    samples: list[Sample] = []
    total = WARMUP + args.samples
    try:
        pending = bench.dispatch(UTTERANCES[0])
        for n in range(total):
            sample, pending = await bench.barge_in(pending, UTTERANCES[(n + 1) % len(UTTERANCES)])
            if n >= WARMUP:
                samples.append(sample)
            print(
                f"turn={n + 1}/{total} flush_ms={sample.flush_ms} "
                f"last_frame_ms={sample.last_frame_ms} accept_ms={sample.accept_ms} "
                f"next_first_sound_ms={sample.next_first_sound_ms} "
                f"overlap_ms={sample.overlap_ms} first_synth_ms={sample.first_synth_ms} "
                f"missed={sample.missed}",
                file=sys.stderr,
            )
    finally:
        await bench.aclose()
    report(cfg, args, samples, bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
