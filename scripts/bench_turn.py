"""End to end turn latency through the real stack: scripted text in, timestamped 48 kHz frames
out of the playout track. Reports time to first sound, time to substance, the visual's landing
and validity, and the cost per turn. Without ``--root`` the turns are a concept lesson on PPO;
with it they walk the fixture repo, the only tree searched. Text only on the wire. ``--soak
MINUTES`` runs the same turns for a stated number of minutes while sampling power, thermal
pressure, cluster frequency, process RSS and swap.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import statistics
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection
from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from tutor.app import Models, build_loop, load_models
from tutor.brief import BriefSplitter, VisualBrief
from tutor.chunker import Scrubber
from tutor.config import Settings, settings
from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.cost import TurnUsage, UsageLedger, turn_cost_usd
from tutor.input_path import EndOfTurn, InputEvent
from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient, TurnChunk, TurnStream
from tutor.session import OutcomeSplitter, TurnLoop
from tutor.signaling import SessionRequest
from tutor.transport import Connection
from tutor.tts import KokoroSynthesizer

WARMUP = 3
SAMPLES = 30
SUBJECT = "a small http client with a bounded connection pool"
PPO_SUBJECT = "PPO"
STARTING_FROM = "I know policy gradients and the advantage; I have not read the PPO paper"
IDLE_FRAMES = 5
SOAK_SAMPLE_S = 10
SOAK_EDGE_MIN = 5
POWERMETRICS_HEADER = "*** Sampled system activity"
POWERMETRICS_FIELDS = {
    "CPU Power": "cpu_power_mw",
    "Combined Power (CPU + GPU + ANE)": "combined_power_mw",
    "P-Cluster HW active frequency": "p_cluster_mhz",
    "E-Cluster HW active frequency": "e_cluster_mhz",
    "Current pressure level": "pressure",
}
OUTBOUND_RATIO = WEBRTC_SAMPLE_RATE // TTS_SAMPLE_RATE
UTTERANCES = (
    "walk me through how the http client gets a connection",
    "what happens when the pool is exhausted",
    "quiz me on the release path",
    "why does acquire_or_wait call release with None",
    "how does with_connection make sure a connection goes back to the pool",
    "what is different between get and post in HttpClient",
)
PPO_UTTERANCES = (
    "teach me ppo",
    "why does it clip the ratio instead of using it directly",
    "what happens when the advantage is negative",
    "quiz me on the clipped objective",
    "the clip makes the gradient larger past epsilon",
    "just tell me",
)
PUSH_TYPES = frozenset({"diagram.push", "app.push"})


def utterances_for(root: Path | None) -> tuple[str, ...]:
    return PPO_UTTERANCES if root is None else UTTERANCES


def subject_for(root: Path | None, subject: str | None) -> str:
    if subject is not None:
        return subject
    return PPO_SUBJECT if root is None else SUBJECT


def substance_frame_index(lengths: Sequence[int], m: int) -> int:
    return OUTBOUND_RATIO * sum(lengths[:m]) // WEBRTC_FRAME_SAMPLES


def substance_frame_time(
    lengths: Sequence[int], m: int, real_times: Sequence[float]
) -> float | None:
    index = substance_frame_index(lengths, m)
    if index >= len(real_times):
        return None
    return real_times[index]


def model_text(streams: Sequence[Sequence[str]]) -> str:
    pieces: list[str] = []
    for n, deltas in enumerate(streams):
        if n:
            pieces.append("\n")
        head = BriefSplitter()
        splitter = OutcomeSplitter()
        pieces.extend(
            text for text in (splitter.feed(head.feed(delta)) for delta in deltas) if text
        )
        if text := splitter.feed(head.finish()):
            pieces.append(text)
        tail, _ = splitter.finish()
        if tail:
            pieces.append(tail)
    scrubber = Scrubber()
    scrubbed = [text for text in (scrubber.feed(piece) for piece in pieces) if text]
    if flushed := scrubber.flush():
        scrubbed.append(flushed)
    return "".join(scrubbed)


def is_model(text: str | None, model: str) -> bool:
    return text is not None and text in model


def is_valid(result: str) -> bool:
    return result.endswith(": sent")


def is_truncated(result: str) -> bool:
    return result.startswith("visual: error: truncated")


def floored(pcm: np.ndarray) -> np.ndarray:
    return np.where(pcm == 0, 1, pcm).astype(np.int16)


class Sample(BaseModel):
    first_sound_ms: int | None
    substance_ms: int | None
    first_content_delta_ms: int | None
    stages: int
    silent: bool
    brief: str | None
    brief_gap_ms: int | None
    visual_landed_ms: int | None
    visual_valid: bool | None
    visual_truncated: bool
    audio_ms: int
    voice_usd: float | None
    visual_usd: float | None


class Enqueued(NamedTuple):
    at: float
    text: str | None
    samples: int


class Push(NamedTuple):
    at: float
    type: str
    title: str


class PowerSample(BaseModel):
    cpu_power_mw: int
    combined_power_mw: int
    p_cluster_mhz: int
    e_cluster_mhz: int
    pressure: str


class SoakRecord(NamedTuple):
    at: float
    power: PowerSample
    rss_mb: int
    swap_used_mb: int


def frame_powermetrics(block: list[str], line: str) -> str | None:
    if line.startswith(POWERMETRICS_HEADER):
        block.clear()
    block.append(line)
    # pressure is the last line powermetrics prints per block for these samplers
    if line.startswith("Current pressure level"):
        return "".join(block)
    return None


def powermetrics_blocks(lines: Iterable[str]) -> Iterator[str]:
    block: list[str] = []
    for line in lines:
        if (done := frame_powermetrics(block, line)) is not None:
            yield done


def parse_powermetrics(block: str) -> PowerSample | None:
    found: dict[str, str] = {}
    for line in block.splitlines():
        key, _, value = line.partition(":")
        if key in POWERMETRICS_FIELDS and value.split():
            found[POWERMETRICS_FIELDS[key]] = value.split()[0]
    try:
        return PowerSample(**found)
    except ValidationError:
        return None


def parse_swap(text: str) -> int:
    used = text.split("used = ")[1].split("M")[0]
    return int(float(used))


def edge_buckets(
    offsets: Sequence[float], minutes: int, edge_min: int
) -> tuple[list[int], list[int]]:
    head = edge_min * 60
    tail = minutes * 60 - head
    first = [n for n, at in enumerate(offsets) if at < head]
    last = [n for n, at in enumerate(offsets) if at >= tail]
    return first, last


async def command_output(*argv: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return out.decode()


async def process_rss_mb() -> int:
    return int(await command_output("ps", "-o", "rss=", "-p", str(os.getpid()))) // 1024


async def machine_state() -> str:
    load, swap, rss = await asyncio.gather(
        command_output("sysctl", "-n", "vm.loadavg"),
        command_output("sysctl", "-n", "vm.swapusage"),
        process_rss_mb(),
    )
    return f"loadavg={load.strip()} swapusage={swap.strip()} rss_mb={rss}"


class Sampler:
    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        pipe: asyncio.ReadTransport,
        stream: asyncio.StreamReader,
    ) -> None:
        self._proc = proc
        self._pipe = pipe
        self._stream = stream
        self._origin = time.perf_counter()
        self.records: list[SoakRecord] = []
        self.sampled = asyncio.Event()
        self._reader = asyncio.create_task(self._read(), name="soak-sampler")

    @classmethod
    async def start(cls) -> "Sampler":
        # stdout=PIPE hides the read transport; the PermissionError path in aclose needs it
        read_fd, write_fd = os.pipe()
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "powermetrics",
            "--samplers",
            "cpu_power,thermal",
            "-i",
            str(SOAK_SAMPLE_S * 1000),
            stdout=write_fd,
        )
        os.close(write_fd)
        stream = asyncio.StreamReader()
        pipe, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(stream), os.fdopen(read_fd, "rb", buffering=0)
        )
        return cls(proc, pipe, stream)

    async def _read(self) -> None:
        block: list[str] = []
        try:
            while True:
                raw = await self._stream.readline()
                if not raw:
                    return
                done = frame_powermetrics(block, raw.decode(errors="replace"))
                if done is None:
                    continue
                sample = parse_powermetrics(done)
                if sample is not None:
                    await self._record(sample)
        finally:
            self.sampled.set()

    async def _record(self, power: PowerSample) -> None:
        at = time.perf_counter() - self._origin
        rss, swap = await asyncio.gather(
            process_rss_mb(), command_output("sysctl", "-n", "vm.swapusage")
        )
        self.records.append(SoakRecord(at, power, rss, parse_swap(swap)))
        self.sampled.set()

    async def first_block(self) -> bool:
        while not self.records and not self._reader.done():
            self.sampled.clear()
            await self.sampled.wait()
        return bool(self.records)

    def rebase(self) -> None:
        self._origin = time.perf_counter()
        self.records = []

    async def aclose(self) -> None:
        try:
            if self._proc.returncode is None:
                try:
                    self._proc.terminate()
                except PermissionError:
                    print(
                        "powermetrics runs as root and refused SIGTERM; closing its stdout so it "
                        "exits on its next write",
                        file=sys.stderr,
                    )
                    self._pipe.close()
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(SOAK_SAMPLE_S * 2):
                    await self._proc.wait()
        finally:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._pipe.close()


class ScriptedSource:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[InputEvent | None] = asyncio.Queue()

    def inject(self, event: InputEvent) -> None:
        self._queue.put_nowait(event)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[InputEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event


class TurnRecord:
    def __init__(self) -> None:
        self.streams: list[list[str]] = []
        self.requested: float | None = None
        self.first_spoken_ms: int | None = None
        self.brief: VisualBrief | None = None
        self.brief_at: float | None = None
        self.voice_usage: TurnUsage | None = None
        self.visual_usage: TurnUsage | None = None
        self.visual_first_chunk_ms: int | None = None
        self.visual_task: asyncio.Task[str] | None = None


def add_usage(total: TurnUsage | None, usage: TurnUsage) -> TurnUsage:
    if total is None:
        return usage
    return TurnUsage(
        prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
        completion_tokens=total.completion_tokens + usage.completion_tokens,
        cached_tokens=total.cached_tokens + usage.cached_tokens,
        reasoning_chars=total.reasoning_chars + usage.reasoning_chars,
    )


class MeteredStream:
    def __init__(
        self,
        inner: TurnStream,
        record: TurnRecord,
        ledgers: Sequence[UsageLedger],
        kind: str,
    ) -> None:
        self._inner = inner
        self._record = record
        self._ledgers = ledgers
        self._kind = kind

    @property
    def finish_reason(self) -> str | None:
        return self._inner.finish_reason

    @property
    def first_chunk_ms(self) -> int | None:
        return self._inner.first_chunk_ms

    async def __aiter__(self) -> AsyncIterator[TurnChunk]:
        record = self._record
        voice = self._kind == "voice"
        deltas: list[str] = []
        head = BriefSplitter()
        if voice:
            record.streams.append(deltas)
        async for chunk in self._inner:
            if voice and chunk.kind == "spoken":
                if record.first_spoken_ms is None:
                    record.first_spoken_ms = int((time.perf_counter() - record.requested) * 1000)
                head.feed(chunk.text)
                if head.brief is not None and record.brief is None:
                    record.brief = head.brief
                    record.brief_at = time.perf_counter()
                deltas.append(chunk.text)
            yield chunk
        if not voice:
            record.visual_first_chunk_ms = self._inner.first_chunk_ms
        usage = self._inner.usage
        if usage is None:
            return
        for ledger in self._ledgers:
            ledger.add(usage)
        if voice:
            record.voice_usage = add_usage(record.voice_usage, usage)
        else:
            record.visual_usage = add_usage(record.visual_usage, usage)


class MeteredReasoning:
    def __init__(self, inner: ReasoningClient) -> None:
        self._inner = inner
        self.ledger = UsageLedger()
        self.ledgers = {"voice": UsageLedger(), "visual": UsageLedger()}
        self.record = TurnRecord()

    def begin(self) -> TurnRecord:
        self.record = TurnRecord()
        return self.record

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
    ) -> MeteredStream:
        record = self.record
        if record.requested is None:
            record.requested = time.perf_counter()
        kind = "voice" if tool_choice is None else "visual"
        if kind == "visual":
            record.visual_task = asyncio.current_task()
        inner = self._inner.start_turn(
            prompt, tools=tools, effort=effort, max_tokens=max_tokens, tool_choice=tool_choice
        )
        return MeteredStream(inner, record, (self.ledger, self.ledgers[kind]), kind)

    async def aclose(self) -> None:
        await self._inner.aclose()


class TaggedSynth:
    def __init__(self, inner: KokoroSynthesizer) -> None:
        self._inner = inner
        self.last: str | None = None

    def synthesize(self, text: str) -> np.ndarray:
        self.last = text
        return self._inner.synthesize(text)


class BenchTransport(Connection):
    def __init__(self, synth: TaggedSynth) -> None:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        super().__init__(pc)
        self._track = pc.getSenders()[0].track
        self._synth = synth
        self.ledger: list[Enqueued] = []
        self.pushes: list[Push] = []
        self.frames: list[tuple[float, bool]] = []
        self.emitted = asyncio.Event()

    async def play(self, pcm: np.ndarray) -> None:
        at = time.perf_counter()
        self.ledger.append(Enqueued(at, self._synth.last, len(pcm)))
        await super().play(floored(pcm))

    async def send_json(self, payload: dict[str, object]) -> None:
        if payload["type"] in PUSH_TYPES:
            at = time.perf_counter()
            self.pushes.append(Push(at, str(payload["type"]), str(payload["title"])))

    async def drain(self) -> None:
        while True:
            frame = await self._track.recv()
            self.frames.append((time.perf_counter(), bool(frame.to_ndarray().any())))
            self.emitted.set()

    async def wait_frames(self, ready: Callable[[], bool]) -> None:
        while not ready():
            self.emitted.clear()
            await self.emitted.wait()

    def idle_since(self, since: int) -> bool:
        tail = self.frames[max(since, len(self.frames) - IDLE_FRAMES) :]
        return len(tail) == IDLE_FRAMES and not any(real for _, real in tail)

    def real_times_since(self, since: int) -> list[float]:
        return [t for t, real in self.frames[since:] if real]


class OutcomeWatch(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()
        self.event = asyncio.Event()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg.startswith(("turn.outcome ", "turn.failed ")):
            self.seen.add(str(record.args[0]))
            self.event.set()
        return True

    async def wait(self, turn_id: str) -> None:
        while turn_id not in self.seen:
            self.event.clear()
            await self.event.wait()


async def settle_visual(turn_id: str, timeout_s: float) -> bool:
    name = f"{turn_id}-visual"
    task = next((t for t in asyncio.all_tasks() if t.get_name() == name), None)
    if task is None:
        return False
    done, _ = await asyncio.wait({task}, timeout=timeout_s)
    return bool(done)


class Bench:
    def __init__(
        self,
        loop_task: asyncio.Task[None],
        drain_task: asyncio.Task[None],
        source: ScriptedSource,
        transport: BenchTransport,
        reasoning: MeteredReasoning,
        synth: TaggedSynth,
        watch: OutcomeWatch,
        loop: TurnLoop,
        visual_timeout_s: float,
    ) -> None:
        self._loop_task = loop_task
        self._drain_task = drain_task
        self._source = source
        self._transport = transport
        self.reasoning = reasoning
        self._synth = synth
        self._watch = watch
        self._loop = loop
        self._visual_timeout_s = visual_timeout_s
        self._dispatched = 0

    @classmethod
    async def boot(
        cls, cfg: Settings, root: Path | None, subject: str, starting_from: str
    ) -> "Bench":
        request = SessionRequest(subject=subject, folder=root, starting_from=starting_from)
        loaded = await asyncio.to_thread(load_models)
        synth = TaggedSynth(loaded.synth)
        models = Models(partial=loaded.partial, final=loaded.final, synth=synth)
        reasoning = MeteredReasoning(ReasoningClient(cfg))
        transport = BenchTransport(synth)
        source = ScriptedSource()
        loop = build_loop(cfg, models, reasoning, source, transport, request)
        watch = OutcomeWatch()
        logging.getLogger("tutor.session").addFilter(watch)
        loop_task = asyncio.create_task(loop.run(), name="bench-loop")
        drain_task = asyncio.create_task(transport.drain(), name="bench-drain")
        return cls(
            loop_task,
            drain_task,
            source,
            transport,
            reasoning,
            synth,
            watch,
            loop,
            cfg.visual_timeout_s,
        )

    async def turn(self, text: str) -> Sample:
        transport = self._transport
        transport.flush_playout()
        since = len(transport.frames)
        await transport.wait_frames(lambda: transport.idle_since(since))
        ledger_since = len(transport.ledger)
        pushes_since = len(transport.pushes)
        record = self.reasoning.begin()
        self._synth.last = None
        self._dispatched += 1
        turn_id = f"turn-{self._dispatched}"

        t0 = time.perf_counter()
        self._source.inject(EndOfTurn(text=text))
        await self._watch.wait(turn_id)
        outcome_at = len(transport.frames)

        played = transport.ledger[ledger_since:]
        model = model_text(record.streams)
        lengths = [entry.samples for entry in played]
        m = next((n for n, entry in enumerate(played) if is_model(entry.text, model)), None)
        needed = 1 if m is None else substance_frame_index(lengths, m) + 1
        await transport.wait_frames(
            lambda: (
                len(transport.real_times_since(since)) >= needed or transport.idle_since(outcome_at)
            )
        )

        await settle_visual(turn_id, self._visual_timeout_s)

        real_times = transport.real_times_since(since)
        first_sound = real_times[0] if real_times else None
        substance = substance_frame_time(lengths, m, real_times) if m is not None else None
        before = played[:m] if m is not None else played
        pushes = transport.pushes[pushes_since:]
        task = record.visual_task
        result = None
        if task is not None and task.done() and not task.cancelled() and task.exception() is None:
            result = task.result()
        brief_at = record.brief_at
        voice, visual = record.voice_usage, record.visual_usage
        if task is None:
            visual_usd = 0.0
        elif visual is None:
            visual_usd = None
        else:
            visual_usd = turn_cost_usd(visual, list_price=True)
        return Sample(
            first_sound_ms=None if first_sound is None else int((first_sound - t0) * 1000),
            substance_ms=None if substance is None else int((substance - t0) * 1000),
            first_content_delta_ms=record.first_spoken_ms,
            stages=max(sum(1 for entry in before if entry.text is not None) - 1, 0),
            silent=m is None,
            brief=record.brief.kind if record.brief is not None else None,
            brief_gap_ms=(
                None if m is None or brief_at is None else int((played[m].at - brief_at) * 1000)
            ),
            visual_landed_ms=None if not pushes else int((pushes[0].at - t0) * 1000),
            visual_valid=None if task is None else result is not None and is_valid(result),
            visual_truncated=result is not None and is_truncated(result),
            audio_ms=sum(entry.samples for entry in played) * 1000 // TTS_SAMPLE_RATE,
            voice_usd=None if voice is None else turn_cost_usd(voice, list_price=True),
            visual_usd=visual_usd,
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
            await self.reasoning.aclose()
            await self._transport.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--soak", type=int, default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--starting-from", default=STARTING_FROM)
    parser.add_argument("--model", default=None)
    return parser


def ledger_line(prefix: str, ledger: UsageLedger) -> str:
    return (
        f"{prefix}turns={ledger.turns} prompt_tokens={ledger.total.prompt_tokens} "
        f"completion_tokens={ledger.total.completion_tokens} "
        f"cost_usd={ledger.cost_usd():.6f} list_usd={ledger.cost_usd(list_price=True):.6f}"
    )


def report(
    cfg: Settings, args: argparse.Namespace, samples: list[Sample], reasoning: MeteredReasoning
) -> None:
    print(
        "time to first sound: first 48 kHz frame carrying the first synthesized clause leaving "
        "the playout track after the scripted EndOfTurn is injected; this is the first spoken "
        "word. Excludes endpointing, recognition and the browser."
    )
    print(
        "time to substance: frame carrying the first sample synthesized from a model-authored "
        "clause; the lead-in and stage sentences do not count. Frame granularity 20 ms."
    )
    print(
        "first content delta: the model's first spoken delta after the first request; it is the "
        "head token when a brief is present."
    )
    print(
        "a visual lands when its diagram.push or app.push payload reaches Connection.send_json "
        "after the EndOfTurn; brief to first clause runs from the head closing to the first "
        "model clause reaching Connection.play. Visuals are serialized: each turn waits for its "
        "own visual task, up to visual_timeout_s, before the next turn is injected, so a visual "
        "never runs under the following turn's voice call and nothing is superseded."
    )
    print(
        "a one-LSB floor is applied to every buffer before playout so an all-zero frame is exactly "
        "padding; it is inaudible and only serves attribution."
    )
    print(
        "cost is list price from the usage chunk that ends each stream; a stream cancelled or "
        "failed before that chunk has an unknown cost and is left out of the cost lines."
    )
    print(
        f"model={cfg.reasoning_model}  samples={len(samples)} (plus {WARMUP} discarded warm-ups)  "
        f"subject={subject_for(args.root, args.subject)!r}  root={args.root}  "
        f"starting_from={args.starting_from!r}  visual_timeout_s={cfg.visual_timeout_s}  "
        f"visual_max_tokens={cfg.visual_max_tokens}"
    )
    summarize(
        "time to first sound", [s.first_sound_ms for s in samples if s.first_sound_ms is not None]
    )
    summarize("time to substance", [s.substance_ms for s in samples if s.substance_ms is not None])
    summarize(
        "first content delta (model, from first request)",
        [s.first_content_delta_ms for s in samples if s.first_content_delta_ms is not None],
    )
    landed = [s for s in samples if s.visual_landed_ms is not None]
    summarize("visual landing", [s.visual_landed_ms for s in landed])
    summarize("audio length, those turns", [s.audio_ms for s in landed])
    summarize(
        "brief to first clause", [s.brief_gap_ms for s in samples if s.brief_gap_ms is not None]
    )
    priced = [s for s in samples if s.voice_usd is not None and s.visual_usd is not None]
    summarize_usd("cost per turn (list)", [s.voice_usd + s.visual_usd for s in priced])
    summarize_usd(
        "voice cost per turn (list)", [s.voice_usd for s in samples if s.voice_usd is not None]
    )
    summarize_usd(
        "visual cost per turn (list)", [s.visual_usd for s in samples if s.visual_usd is not None]
    )
    print(f"turns with unknown cost {len(samples) - len(priced)}/{len(samples)}")
    stages = [s.stages for s in samples if not s.silent]
    if stages:
        print(f"stage sentences before substance median={int(statistics.median(stages))}")
    silent = sum(1 for s in samples if s.silent)
    print(f"silent turns {silent}/{len(samples)}")
    briefed = sum(1 for s in samples if s.brief is not None)
    calls = [s for s in samples if s.brief in ("diagram", "app")]
    print(f"turns with a brief {briefed}/{len(samples)}")
    print(f"visual calls {len(calls)}/{len(samples)}")
    print(f"visuals landed {len(landed)}/{len(calls)}")
    print(f"visuals valid {sum(1 for s in calls if s.visual_valid)}/{len(calls)}")
    print(f"visuals truncated {sum(1 for s in calls if s.visual_truncated)}/{len(calls)}")
    print(ledger_line("", reasoning.ledger))
    print(ledger_line("voice ", reasoning.ledgers["voice"]))
    print(ledger_line("visual ", reasoning.ledgers["visual"]))


def summarize_usd(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label:34s} no samples")
        return
    ordered = sorted(values)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    print(
        f"{label:34s} n={len(ordered):3d} median={statistics.median(ordered):.6f} "
        f"p95={p95:.6f} min={ordered[0]:.6f} max={ordered[-1]:.6f}"
    )


def summarize_series(label: str, values: list[int], unit: str) -> None:
    if not values:
        print(f"{label:34s} no samples")
        return
    ordered = sorted(values)
    print(
        f"{label:34s} n={len(ordered):3d} min={ordered[0]:6d}{unit} "
        f"median={int(statistics.median(ordered)):6d}{unit} max={ordered[-1]:6d}{unit}"
    )


def present(samples: Sequence[Sample], key: Callable[[Sample], int | None]) -> list[int]:
    return [value for value in map(key, samples) if value is not None]


def report_soak(
    cfg: Settings,
    args: argparse.Namespace,
    turns: list[tuple[float, Sample]],
    records: list[SoakRecord],
    ledger: UsageLedger,
) -> None:
    print(
        f"soak: {args.soak} min of scripted turns through the assembled loop, every turn kept "
        f"with its offset from the first turn; powermetrics sampled every {SOAK_SAMPLE_S} s "
        "alongside this process's rss and system swap."
    )
    print(
        "throttling: powermetrics on this machine prints no throttling line; the thermal "
        "pressure level (Nominal, Moderate, Heavy, Trapping, Sleeping) and the P-cluster active "
        "frequency are the throttling signals."
    )
    print(
        f"model={cfg.reasoning_model}  subject={subject_for(args.root, args.subject)!r}  "
        f"root={args.root}  starting_from={args.starting_from!r}"
    )
    summarize_series("cpu power", [r.power.cpu_power_mw for r in records], "mW")
    summarize_series("combined power", [r.power.combined_power_mw for r in records], "mW")
    summarize_series("p-cluster frequency", [r.power.p_cluster_mhz for r in records], "MHz")
    summarize_series("e-cluster frequency", [r.power.e_cluster_mhz for r in records], "MHz")
    rss = [r.rss_mb for r in records]
    swap = [r.swap_used_mb for r in records]
    early, late = edge_buckets([r.at for r in records], args.soak, SOAK_EDGE_MIN)
    summarize_series("process rss", rss, "MB")
    summarize_series("swap used", swap, "MB")
    summarize_series(f"process rss, first {SOAK_EDGE_MIN} min", [rss[n] for n in early], "MB")
    summarize_series(f"process rss, last {SOAK_EDGE_MIN} min", [rss[n] for n in late], "MB")
    summarize_series(f"swap used, first {SOAK_EDGE_MIN} min", [swap[n] for n in early], "MB")
    summarize_series(f"swap used, last {SOAK_EDGE_MIN} min", [swap[n] for n in late], "MB")
    levels = Counter(r.power.pressure for r in records)
    counts = " ".join(f"{level}={count}" for level, count in sorted(levels.items()))
    print(f"{'pressure levels':34s} n={len(records):3d} {counts}")

    samples = [sample for _, sample in turns]
    first, last = edge_buckets([at for at, _ in turns], args.soak, SOAK_EDGE_MIN)
    head = [samples[n] for n in first]
    tail = [samples[n] for n in last]
    summarize("time to first sound", present(samples, lambda s: s.first_sound_ms))
    summarize("time to substance", present(samples, lambda s: s.substance_ms))
    summarize(f"first sound, first {SOAK_EDGE_MIN} min", present(head, lambda s: s.first_sound_ms))
    summarize(f"first sound, last {SOAK_EDGE_MIN} min", present(tail, lambda s: s.first_sound_ms))
    summarize(f"substance, first {SOAK_EDGE_MIN} min", present(head, lambda s: s.substance_ms))
    summarize(f"substance, last {SOAK_EDGE_MIN} min", present(tail, lambda s: s.substance_ms))
    summarize(
        "first content delta (model, from first request)",
        present(samples, lambda s: s.first_content_delta_ms),
    )
    print(f"turns={len(samples)}")
    silent = sum(1 for s in samples if s.silent)
    print(f"silent turns {silent}/{len(samples)}")
    print(ledger_line("", ledger))


def turn_line(sample: Sample) -> str:
    return (
        f"first_sound_ms={sample.first_sound_ms} substance_ms={sample.substance_ms} "
        f"stages={sample.stages} silent={sample.silent} brief={sample.brief} "
        f"visual_landed_ms={sample.visual_landed_ms} visual_valid={sample.visual_valid} "
        f"visual_truncated={sample.visual_truncated} audio_ms={sample.audio_ms}"
    )


async def soak(cfg: Settings, args: argparse.Namespace) -> int:
    sampler = await Sampler.start()
    try:
        if not await sampler.first_block():
            print(
                "powermetrics needs root: run this from a terminal where sudo can prompt, "
                "never under sudo as a whole"
            )
            return 2
        root = None if args.root is None else args.root.resolve()
        utterances = utterances_for(root)
        bench = await Bench.boot(cfg, root, subject_for(root, args.subject), args.starting_from)
        turns: list[tuple[float, Sample]] = []
        try:
            print(f"soak start {await machine_state()}", flush=True)
            started = time.perf_counter()
            sampler.rebase()
            while (at := time.perf_counter() - started) < args.soak * 60:
                n = len(turns)
                sample = await bench.turn(utterances[n % len(utterances)])
                turns.append((at, sample))
                print(f"turn={n + 1} t={int(at)}s {turn_line(sample)}", file=sys.stderr)
            print(f"soak end {await machine_state()}", flush=True)
        finally:
            await bench.aclose()
    finally:
        await sampler.aclose()
    report_soak(cfg, args, turns, sampler.records, bench.reasoning.ledger)
    return 0


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No turn latency can be reported. Populate .env and rerun.")
        return 2
    if args.model:
        cfg = cfg.model_copy(update={"reasoning_model": args.model})
    if args.soak is not None:
        return await soak(cfg, args)

    root = None if args.root is None else args.root.resolve()
    utterances = utterances_for(root)
    bench = await Bench.boot(cfg, root, subject_for(root, args.subject), args.starting_from)
    samples: list[Sample] = []
    try:
        for n in range(WARMUP + args.samples):
            sample = await bench.turn(utterances[n % len(utterances)])
            if n >= WARMUP:
                samples.append(sample)
            print(f"turn={n + 1}/{WARMUP + args.samples} {turn_line(sample)}", file=sys.stderr)
    finally:
        await bench.aclose()
    report(cfg, args, samples, bench.reasoning)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
