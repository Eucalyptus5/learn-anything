"""Diagram lead through the real stack: a scripted reasoning client answers every turn with one
``push_diagram`` call and then a fixed three-clause explanation, and the harness measures the gap
from the ``diagram.push`` payload reaching ``Connection.send_json`` to the first explaining clause
reaching ``Connection.play``. The model answers in zero time, so a live model only widens the lead.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from scripts.bench_turn import (
    SAMPLES,
    SUBJECT,
    WARMUP,
    BenchTransport,
    Enqueued,
    OutcomeWatch,
    ScriptedSource,
    TaggedSynth,
    is_model,
    model_text,
    substance_frame_index,
    substance_frame_time,
)
from tutor.app import Models, build_loop, load_models
from tutor.config import Settings
from tutor.input_path import EndOfTurn
from tutor.prompt import TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.session import TurnLoop
from tutor.signaling import SessionRequest

UTTERANCE = "walk me through how the http client gets a connection"
FLOWCHART = """flowchart TD
  n1[start] --> n2[parse args]
  n2 --> n3[load config]
  n3 --> n4{config ok}
  n4 -->|yes| n5[open mic]
  n4 -->|no| n6[report error]
  n6 --> n30[exit]
  n5 --> n7[start vad]
  n7 --> n8[start recognizer]
  n8 --> n9[start synthesizer]
  n9 --> n10[join room]
  n10 --> n11{frame arrives}
  n11 -->|speech| n12[buffer frame]
  n11 -->|silence| n13[end of turn]
  n12 --> n11
  n13 --> n14[transcribe]
  n14 --> n15[build prompt]
  n15 --> n16[call model]
  n16 --> n17{tool call}
  n17 -->|search| n18[run ripgrep]
  n17 -->|read| n19[read file]
  n17 -->|diagram| n20[push diagram]
  n17 -->|none| n21[stream text]
  n18 --> n22[ground result]
  n19 --> n22
  n20 --> n22
  n22 --> n16
  n21 --> n23[chunk sentences]
  n23 --> n24[synthesize]
  n24 --> n25[send audio]
  n25 --> n26{barge in}
  n26 -->|yes| n27[flush queue]
  n26 -->|no| n28[finish turn]
  n27 --> n29[cancel task]
  n29 --> n11
  n28 --> n11"""
PUSH_ARGUMENTS = json.dumps({"id": "lead", "kind": "flowchart", "source": FLOWCHART})
BUDGET_MS = 100
FOLLOW_UP_DELTAS = [
    "The reader pulls frames off the track, ",
    "hands each one to the queue, ",
    "and the playout drains it.",
    "\n<outcome>",
    '{"signal": "covered", "settling_positions": []}',
]
EXPLANATION = model_text([FOLLOW_UP_DELTAS])


class ScriptedStream:
    def __init__(self, chunks: list[TurnChunk]) -> None:
        self._chunks = chunks
        self.usage = None

    async def __aiter__(self) -> AsyncIterator[TurnChunk]:
        for chunk in self._chunks:
            yield chunk


class ScriptedReasoning:
    def start_turn(
        self, prompt: TurnPrompt, tools: Sequence[dict] | None = None, max_tokens: int | None = None
    ) -> ScriptedStream:
        if prompt.tool_exchange:
            return ScriptedStream([TurnChunk(kind="spoken", text=d) for d in FOLLOW_UP_DELTAS])
        return ScriptedStream(
            [
                TurnChunk(
                    kind="tool_call",
                    text=PUSH_ARGUMENTS,
                    tool_call_id="call-lead",
                    tool_name="push_diagram",
                )
            ]
        )


class LeadTransport(BenchTransport):
    def __init__(self, synth: TaggedSynth, openers: dict[str, np.ndarray]) -> None:
        super().__init__(synth, openers)
        self.pushes: list[tuple[float, str]] = []

    async def send_json(self, payload: dict[str, object]) -> None:
        self.pushes.append((time.perf_counter(), str(payload["type"])))


def explanation_index(played: Sequence[Enqueued], explanation: str) -> int | None:
    return next((n for n, entry in enumerate(played) if is_model(entry.text, explanation)), None)


def lead_ms(push_at: float, voice_at: float) -> int:
    return int((voice_at - push_at) * 1000)


def verdict(leads: Sequence[int]) -> str:
    if not leads:
        return "verdict: no lead measured, FAIL"
    median = int(np.median(leads))
    line = f"verdict: median lead {median}ms against the {BUDGET_MS}ms render budget"
    return line if median >= BUDGET_MS else f"{line}, FAIL"


class LeadSample(BaseModel):
    lead_ms: int | None
    playout_ms: int | None
    push_ms: int | None
    voice_ms: int | None
    pushes: int


class Bench:
    def __init__(
        self,
        loop_task: asyncio.Task[None],
        drain_task: asyncio.Task[None],
        source: ScriptedSource,
        transport: LeadTransport,
        synth: TaggedSynth,
        watch: OutcomeWatch,
        loop: TurnLoop,
    ) -> None:
        self._loop_task = loop_task
        self._drain_task = drain_task
        self._source = source
        self._transport = transport
        self._synth = synth
        self._watch = watch
        self._loop = loop
        self._dispatched = 0

    @classmethod
    async def boot(cls, cfg: Settings, root: Path) -> "Bench":
        request = SessionRequest(subject=SUBJECT, folder=root)
        loaded = await asyncio.to_thread(load_models)
        synth = TaggedSynth(loaded.synth)
        models = Models(
            partial=loaded.partial, final=loaded.final, synth=synth, openers=loaded.openers
        )
        transport = LeadTransport(synth, models.openers)
        source = ScriptedSource()
        loop = build_loop(cfg, models, ScriptedReasoning(), source, transport, request)
        watch = OutcomeWatch()
        logging.getLogger("tutor.session").addFilter(watch)
        loop_task = asyncio.create_task(loop.run(), name="bench-loop")
        drain_task = asyncio.create_task(transport.drain(), name="bench-drain")
        return cls(loop_task, drain_task, source, transport, synth, watch, loop)

    async def turn(self, text: str) -> LeadSample:
        transport = self._transport
        transport.flush_playout()
        since = len(transport.frames)
        await transport.wait_frames(lambda: transport.idle_since(since))
        ledger_since = len(transport.ledger)
        pushes_since = len(transport.pushes)
        self._synth.last = None
        self._dispatched += 1
        turn_id = f"turn-{self._dispatched}"

        t0 = time.perf_counter()
        self._source.inject(EndOfTurn(text=text))
        await self._watch.wait(turn_id)
        outcome_at = len(transport.frames)

        played = transport.ledger[ledger_since:]
        lengths = [entry.samples for entry in played]
        m = explanation_index(played, EXPLANATION)
        needed = 1 if m is None else substance_frame_index(lengths, m) + 1
        await transport.wait_frames(
            lambda: (
                len(transport.real_times_since(since)) >= needed or transport.idle_since(outcome_at)
            )
        )

        pushes = transport.pushes[pushes_since:]
        push_at = pushes[0][0] if pushes else None
        real_times = transport.real_times_since(since)
        voice_at = substance_frame_time(lengths, m, real_times) if m is not None else None
        clause_at = played[m].at if m is not None else None
        return LeadSample(
            lead_ms=None if push_at is None or clause_at is None else lead_ms(push_at, clause_at),
            playout_ms=None if push_at is None or voice_at is None else lead_ms(push_at, voice_at),
            push_ms=None if push_at is None else int((push_at - t0) * 1000),
            voice_ms=None if voice_at is None else int((voice_at - t0) * 1000),
            pushes=len(pushes),
        )

    async def aclose(self) -> None:
        self._source.close()
        try:
            await self._loop_task
        finally:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            logging.getLogger("tutor.session").removeFilter(self._watch)
            await self._loop.aclose()
            await self._transport.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--root", type=Path, default=REPO / "tests" / "data" / "fixture_repo")
    return parser


def report(args: argparse.Namespace, samples: list[LeadSample]) -> None:
    print(
        "diagram lead: from the diagram.push payload reaching Connection.send_json to the first "
        "clause explaining it reaching Connection.play, the earliest that clause can sound on an "
        "empty playout buffer. The scripted model answers in zero time; a live model only widens "
        "the lead."
    )
    print(
        "playout gap: from the same push to the first 48 kHz frame carrying that clause leaving "
        "the playout track; it includes the opener and the lead-in still draining ahead of it."
    )
    print(f"samples={len(samples)} (plus {WARMUP} discarded warm-ups)  root={args.root}")
    leads = [s.lead_ms for s in samples if s.lead_ms is not None]
    summarize("diagram lead", leads)
    summarize("playout gap", [s.playout_ms for s in samples if s.playout_ms is not None])
    summarize("push after end of turn", [s.push_ms for s in samples if s.push_ms is not None])
    summarize("voice after end of turn", [s.voice_ms for s in samples if s.voice_ms is not None])
    unpushed = sum(1 for s in samples if not s.pushes)
    print(f"turns without a push {unpushed}/{len(samples)}")
    print(verdict(leads))


async def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    root = args.root.resolve()
    cfg = Settings(reasoning_api_base="scripted", reasoning_api_key="scripted")
    bench = await Bench.boot(cfg, root)
    samples: list[LeadSample] = []
    try:
        for n in range(WARMUP + args.samples):
            sample = await bench.turn(UTTERANCE)
            if n >= WARMUP:
                samples.append(sample)
            print(
                f"turn={n + 1}/{WARMUP + args.samples} lead_ms={sample.lead_ms} "
                f"playout_ms={sample.playout_ms} push_ms={sample.push_ms} "
                f"voice_ms={sample.voice_ms} pushes={sample.pushes}",
                file=sys.stderr,
            )
    finally:
        await bench.aclose()
    report(args, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
