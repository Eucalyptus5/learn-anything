"""End to end turn latency through the real stack: scripted text in, timestamped 48 kHz frames
out of the playout track. Reports time to first sound and time to substance per turn. Text only
on the wire; the fixture repo is the only tree searched. ``--soak MINUTES`` runs the same turns
for a stated number of minutes while sampling power, thermal pressure, cluster frequency, process
RSS and swap.
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
from tutor.chunker import Scrubber
from tutor.config import Settings, settings
from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.cost import UsageLedger
from tutor.input_path import EndOfTurn, InputEvent
from tutor.reasoning import ReasoningClient, TurnChunk, TurnStream
from tutor.session import OutcomeSplitter, TurnLoop
from tutor.signaling import SessionRequest
from tutor.transport import Connection
from tutor.tts import KokoroSynthesizer

WARMUP = 3
SAMPLES = 30
SUBJECT = "a small http client with a bounded connection pool"
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
        splitter = OutcomeSplitter()
        pieces.extend(text for text in (splitter.feed(delta) for delta in deltas) if text)
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


def floored(pcm: np.ndarray) -> np.ndarray:
    return np.where(pcm == 0, 1, pcm).astype(np.int16)


class Sample(BaseModel):
    first_sound_ms: int | None
    substance_ms: int | None
    first_spoken_delta_ms: int | None
    stages: int
    silent: bool


class Enqueued(NamedTuple):
    at: float
    text: str | None
    samples: int


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


class MeteredStream:
    def __init__(self, inner: TurnStream, record: TurnRecord, ledger: UsageLedger) -> None:
        self._inner = inner
        self._record = record
        self._ledger = ledger

    async def __aiter__(self) -> AsyncIterator[TurnChunk]:
        deltas: list[str] = []
        self._record.streams.append(deltas)
        async for chunk in self._inner:
            if chunk.kind == "spoken":
                if self._record.first_spoken_ms is None:
                    since = time.perf_counter() - self._record.requested
                    self._record.first_spoken_ms = int(since * 1000)
                deltas.append(chunk.text)
            yield chunk
        if self._inner.usage is not None:
            self._ledger.add(self._inner.usage)


class MeteredReasoning:
    def __init__(self, inner: ReasoningClient) -> None:
        self._inner = inner
        self.ledger = UsageLedger()
        self.record = TurnRecord()

    def begin(self) -> TurnRecord:
        self.record = TurnRecord()
        return self.record

    def start_turn(self, *args: object, **kwargs: object) -> MeteredStream:
        if self.record.requested is None:
            self.record.requested = time.perf_counter()
        return MeteredStream(self._inner.start_turn(*args, **kwargs), self.record, self.ledger)

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
    def __init__(self, synth: TaggedSynth, openers: dict[str, np.ndarray]) -> None:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        super().__init__(pc)
        self._track = pc.getSenders()[0].track
        self._synth = synth
        self._openers = list(openers.values())
        self.ledger: list[Enqueued] = []
        self.frames: list[tuple[float, bool]] = []
        self.emitted = asyncio.Event()

    async def play(self, pcm: np.ndarray) -> None:
        at = time.perf_counter()
        opener = any(pcm is cached for cached in self._openers)
        self.ledger.append(Enqueued(at, None if opener else self._synth.last, len(pcm)))
        await super().play(floored(pcm))

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
    ) -> None:
        self._loop_task = loop_task
        self._drain_task = drain_task
        self._source = source
        self._transport = transport
        self.reasoning = reasoning
        self._synth = synth
        self._watch = watch
        self._loop = loop
        self._dispatched = 0

    @classmethod
    async def boot(cls, cfg: Settings, root: Path, subject: str) -> "Bench":
        request = SessionRequest(subject=subject, folder=root)
        loaded = await asyncio.to_thread(load_models)
        synth = TaggedSynth(loaded.synth)
        models = Models(
            partial=loaded.partial, final=loaded.final, synth=synth, openers=loaded.openers
        )
        reasoning = MeteredReasoning(ReasoningClient(cfg))
        transport = BenchTransport(synth, models.openers)
        source = ScriptedSource()
        loop = build_loop(cfg, models, reasoning, source, transport, request)
        watch = OutcomeWatch()
        logging.getLogger("tutor.session").addFilter(watch)
        loop_task = asyncio.create_task(loop.run(), name="bench-loop")
        drain_task = asyncio.create_task(transport.drain(), name="bench-drain")
        return cls(loop_task, drain_task, source, transport, reasoning, synth, watch, loop)

    async def turn(self, text: str) -> Sample:
        transport = self._transport
        transport.flush_playout()
        since = len(transport.frames)
        await transport.wait_frames(lambda: transport.idle_since(since))
        ledger_since = len(transport.ledger)
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

        real_times = transport.real_times_since(since)
        first_sound = real_times[0] if real_times else None
        substance = substance_frame_time(lengths, m, real_times) if m is not None else None
        before = played[:m] if m is not None else played
        return Sample(
            first_sound_ms=None if first_sound is None else int((first_sound - t0) * 1000),
            substance_ms=None if substance is None else int((substance - t0) * 1000),
            first_spoken_delta_ms=record.first_spoken_ms,
            stages=max(sum(1 for entry in before if entry.text is not None) - 1, 0),
            silent=m is None,
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
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--root", type=Path, default=REPO / "tests" / "data" / "fixture_repo")
    parser.add_argument("--model", default=None)
    return parser


def report(
    cfg: Settings, args: argparse.Namespace, samples: list[Sample], ledger: UsageLedger
) -> None:
    print(
        "time to first sound: first 48 kHz frame with real audio leaving the playout track after "
        "the scripted EndOfTurn is injected; excludes endpointing, recognition and the browser."
    )
    print(
        "time to substance: frame carrying the first sample synthesized from a model-authored "
        "clause; openers, the lead-in and stage sentences do not count. Frame granularity 20 ms."
    )
    print(
        "a one-LSB floor is applied to every buffer before playout so an all-zero frame is exactly "
        "padding; it is inaudible and only serves attribution."
    )
    print(
        f"model={cfg.reasoning_model}  samples={len(samples)} (plus {WARMUP} discarded warm-ups)  "
        f"subject={args.subject!r}  root={args.root}"
    )
    summarize(
        "time to first sound", [s.first_sound_ms for s in samples if s.first_sound_ms is not None]
    )
    summarize("time to substance", [s.substance_ms for s in samples if s.substance_ms is not None])
    summarize(
        "first spoken delta (model, from first request)",
        [s.first_spoken_delta_ms for s in samples if s.first_spoken_delta_ms is not None],
    )
    stages = [s.stages for s in samples if not s.silent]
    if stages:
        print(f"stage sentences before substance median={int(statistics.median(stages))}")
    silent = sum(1 for s in samples if s.silent)
    print(f"silent turns {silent}/{len(samples)}")
    print(
        f"turns={ledger.turns} prompt_tokens={ledger.total.prompt_tokens} "
        f"completion_tokens={ledger.total.completion_tokens} "
        f"cost_usd={ledger.cost_usd():.6f} list_usd={ledger.cost_usd(list_price=True):.6f}"
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
    print(f"model={cfg.reasoning_model}  subject={args.subject!r}  root={args.root}")
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
        "first spoken delta (model, from first request)",
        present(samples, lambda s: s.first_spoken_delta_ms),
    )
    print(f"turns={len(samples)}")
    silent = sum(1 for s in samples if s.silent)
    print(f"silent turns {silent}/{len(samples)}")
    print(
        f"turns={ledger.turns} prompt_tokens={ledger.total.prompt_tokens} "
        f"completion_tokens={ledger.total.completion_tokens} "
        f"cost_usd={ledger.cost_usd():.6f} list_usd={ledger.cost_usd(list_price=True):.6f}"
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
        bench = await Bench.boot(cfg, args.root.resolve(), args.subject)
        turns: list[tuple[float, Sample]] = []
        try:
            print(f"soak start {await machine_state()}", flush=True)
            started = time.perf_counter()
            sampler.rebase()
            while (at := time.perf_counter() - started) < args.soak * 60:
                n = len(turns)
                sample = await bench.turn(UTTERANCES[n % len(UTTERANCES)])
                turns.append((at, sample))
                print(
                    f"turn={n + 1} t={int(at)}s first_sound_ms={sample.first_sound_ms} "
                    f"substance_ms={sample.substance_ms} stages={sample.stages} "
                    f"silent={sample.silent}",
                    file=sys.stderr,
                )
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

    bench = await Bench.boot(cfg, args.root.resolve(), args.subject)
    samples: list[Sample] = []
    try:
        for n in range(WARMUP + args.samples):
            sample = await bench.turn(UTTERANCES[n % len(UTTERANCES)])
            if n >= WARMUP:
                samples.append(sample)
            print(
                f"turn={n + 1}/{WARMUP + args.samples} first_sound_ms={sample.first_sound_ms} "
                f"substance_ms={sample.substance_ms} stages={sample.stages} "
                f"silent={sample.silent}",
                file=sys.stderr,
            )
    finally:
        await bench.aclose()
    report(cfg, args, samples, bench.reasoning.ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
