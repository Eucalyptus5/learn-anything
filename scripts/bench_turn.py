"""End to end turn latency through the real stack: scripted text in, timestamped 48 kHz frames
out of the playout track. Reports time to first sound, time to substance and the cost per turn.
Without ``--root`` the turns are a concept lesson on PPO; with it they walk the fixture repo, the
only tree searched. Text only on the wire. Each turn's reply length is appended to the capture
file (``--capture``, default ``scratch/bench/replies.jsonl``). ``--soak MINUTES`` runs the same
turns for a stated number of minutes while sampling power, thermal pressure, cluster frequency,
process RSS and swap. ``--plan`` answers the planner from a fixed lesson and ``--stub-scenes``
answers every scene build with a stub; ``--scenarios`` replays each case's setup and reads the
tags of the one measured reply, writing it to ``--out``; ``--connect`` times connect to the first
audio frame and to scene one checked on the real planner, builder and voice. The page is simulated.
"""

import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, Self

import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from tutor.app import Models, build_loop, load_models
from tutor.chunker import Scrubber
from tutor.config import Settings, settings
from tutor.constants import TTS_SAMPLE_RATE, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE
from tutor.cost import TurnUsage, UsageLedger, turn_cost_usd
from tutor.input_path import EndOfTurn, InputEvent
from tutor.lesson import OPENING_TEXT, REPEAT, Cursor, LessonPlan, LessonState
from tutor.planner import PLAN_TOOL
from tutor.prompt import Message, TurnPrompt
from tutor.reasoning import ReasoningClient, TurnChunk, TurnStream
from tutor.scene import SCENE_TOOL
from tutor.session import TurnLoop, TurnLoopConfig
from tutor.signaling import SessionRequest
from tutor.tags import TAG_NAMES, TagSplitter, parse_marker, tag_name
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
SILENT_FRAME_SAMPLES = TTS_SAMPLE_RATE // 50
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
CAPTURE_DIR = REPO / "scratch" / "bench"
LISTENING_OUTCOME = REPO / "scratch" / "experiments" / "12" / "listening" / "outcome.json"
SCENE_READY_TIMEOUT_S = TurnLoopConfig.model_fields["scene_ready_timeout_s"].default
STUB_COUNT = re.compile(r"exactly (\d+), one say line each")
STUB_HTML = '<!doctype html><div id="stub"></div>'
SAY_LINE_CHARS = 120  # write_scene's StepLine limit; a plan's show line may run to 300
GROUPS = ("opening", "progress", "question", "answer", "boundary", "tangent")
FLOOR = 27


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
        tags = TagSplitter()
        for delta in deltas:
            pieces.extend(item for item in tags.feed(delta) if isinstance(item, str))
    scrubber = Scrubber()
    scrubbed = [text for text in (scrubber.feed(piece) for piece in pieces) if text]
    if flushed := scrubber.flush():
        scrubbed.append(flushed)
    return "".join(scrubbed)


def is_model(text: str | None, model: str) -> bool:
    return text is not None and text in model


def is_truncated(result: str) -> bool:
    return result.startswith(("scene: error: truncated", "planner: error: truncated"))


def floored(pcm: np.ndarray) -> np.ndarray:
    return np.where(pcm == 0, 1, pcm).astype(np.int16)


class Capture:
    def __init__(self, path: Path, started: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="ascii")
        self._started = started

    def write(self, turn: int, utterance: int, sample: bool, reply_chars: int) -> None:
        line = {
            "started": self._started,
            "turn": turn,
            "utterance": utterance,
            "sample": sample,
            "reply_chars": reply_chars,
        }
        self._file.write(json.dumps(line, ensure_ascii=True) + "\n")
        self._file.flush()


class Sample(BaseModel):
    first_sound_ms: int | None
    substance_ms: int | None
    first_content_delta_ms: int | None
    stages: int
    silent: bool
    audio_ms: int
    voice_usd: float | None
    utterance: int
    scenario_id: str | None = None
    group: str | None = None
    tags: list[str] | None = None
    tags_valid: bool | None = None
    placement_errors: int | None = None
    dropped: dict[str, int] | None = None
    progress_ok: bool | None = None
    asked: bool | None = None
    scene_ok: bool | None = None
    leaked: int | None = None
    opening: bool | None = None
    scenario_pass: bool | None = None
    seen_pass: bool | None = None


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
        self.voice_usage: TurnUsage | None = None
        self.visual_usage: TurnUsage | None = None
        self.visual_first_chunk_ms: int | None = None
        self.planner_usage: TurnUsage | None = None
        self.planner_first_chunk_ms: int | None = None


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
        if voice:
            record.streams.append(deltas)
        async for chunk in self._inner:
            if voice and chunk.kind == "spoken":
                if record.first_spoken_ms is None:
                    record.first_spoken_ms = int((time.perf_counter() - record.requested) * 1000)
                deltas.append(chunk.text)
            yield chunk
        if self._kind == "visual":
            record.visual_first_chunk_ms = self._inner.first_chunk_ms
        elif self._kind == "planner":
            record.planner_first_chunk_ms = self._inner.first_chunk_ms
        usage = self._inner.usage
        if usage is None:
            return
        for ledger in self._ledgers:
            ledger.add(usage)
        if voice:
            record.voice_usage = add_usage(record.voice_usage, usage)
        elif self._kind == "planner":
            record.planner_usage = add_usage(record.planner_usage, usage)
        else:
            record.visual_usage = add_usage(record.visual_usage, usage)


class CannedStream:
    def __init__(self, chunks: list[TurnChunk], finish_reason: str) -> None:
        self._chunks = chunks
        self.finish_reason = finish_reason
        self.usage: TurnUsage | None = None
        self.first_chunk_ms: int | None = None

    async def __aiter__(self) -> AsyncIterator[TurnChunk]:
        for chunk in self._chunks:
            yield chunk


def stub_scene(prompt: TurnPrompt) -> TurnChunk:
    found = STUB_COUNT.search(prompt.user_text)
    count = int(found.group(1))
    lines = prompt.user_text[found.end() :].splitlines()[1 : count + 1]
    said = [
        line.partition(". ")[2][:SAY_LINE_CHARS] or f"Step {n}" for n, line in enumerate(lines, 1)
    ]
    steps = [*said, *(f"Step {n}" for n in range(len(said) + 1, count + 1))]
    body = {"html": STUB_HTML, "steps": steps}
    return TurnChunk(
        kind="tool_call", text=json.dumps(body), tool_call_id="call-scene", tool_name=SCENE_TOOL
    )


class MeteredReasoning:
    def __init__(
        self, inner: ReasoningClient, plan: LessonPlan | None = None, stub_scenes: bool = False
    ) -> None:
        self._inner = inner
        self._plan = plan
        self._stub_scenes = stub_scenes
        self._scripted: list[str] = []
        self.ledger = UsageLedger()
        self.ledgers = {"voice": UsageLedger(), "visual": UsageLedger(), "planner": UsageLedger()}
        self.record = TurnRecord()

    def begin(self) -> TurnRecord:
        self.record = TurnRecord()
        return self.record

    def script(self, replies: Iterable[str]) -> None:
        self._scripted.extend(replies)

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
    ) -> MeteredStream | CannedStream:
        name = tools[0]["function"]["name"] if tools else None
        kind = {PLAN_TOOL: "planner", SCENE_TOOL: "visual"}.get(name, "voice")
        if kind == "planner" and self._plan is not None:
            call = TurnChunk(
                kind="tool_call",
                text=self._plan.model_dump_json(),
                tool_call_id="call-plan",
                tool_name=PLAN_TOOL,
            )
            return CannedStream([call], "tool_calls")
        if kind == "visual" and self._stub_scenes:
            return CannedStream([stub_scene(prompt)], "tool_calls")
        if kind == "voice" and self._scripted:
            reply = self._scripted.pop(0)
            pieces = [piece for piece in re.split(r"(?<= )", reply) if piece]
            return CannedStream([TurnChunk(kind="spoken", text=piece) for piece in pieces], "stop")
        record = self.record
        # A build or a planner call runs beside the turn; only a voice call starts its clock.
        if kind == "voice" and record.requested is None:
            record.requested = time.perf_counter()
        inner = self._inner.start_turn(
            prompt,
            tools=tools,
            effort=effort,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            model=model,
        )
        return MeteredStream(inner, record, (self.ledger, self.ledgers[kind]), kind)

    async def aclose(self) -> None:
        await self._inner.aclose()


class SilentSynth:
    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(SILENT_FRAME_SAMPLES, dtype=np.int16)


class TaggedSynth:
    def __init__(self, inner: KokoroSynthesizer | SilentSynth) -> None:
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
        self.frames: list[tuple[float, bool]] = []
        self.first_sound: float | None = None
        self.emitted = asyncio.Event()
        self.checks: list[tuple[float, str, bool]] = []
        self.simulated: Counter[str] = Counter()
        self._held: dict[int, tuple[dict[str, object], asyncio.TimerHandle]] = {}
        self._epoch = 0
        self._barrier = 0
        self._revision = 0
        self._scene_id: str | None = None
        self._step = 0
        self._last_cue = 0

    async def play(self, pcm: np.ndarray) -> None:
        at = time.perf_counter()
        self.ledger.append(Enqueued(at, self._synth.last, len(pcm)))
        await super().play(floored(pcm))

    async def send_json(self, payload: dict[str, object]) -> None:
        kind = payload["type"]
        if kind == "scene.push":
            report = {
                "type": "scene.ready",
                "scene_id": payload["scene_id"],
                "ok": True,
                "steps": len(payload["steps"]),
                "error": "",
            }
            self.checks.append((time.perf_counter(), str(payload["scene_id"]), True))
            self._answer(report, "ready")
        elif kind == "lesson.attach":
            self._drop("barrier")
            self._epoch = payload["epoch"]
            self._barrier = self._revision = self._step = self._last_cue = 0
            self._scene_id = None
        elif kind == "lesson.cue":
            cue_id = int(payload["cue_id"])
            lead_s = int(payload["lead_ms"]) / 1000
            timer = asyncio.get_running_loop().call_later(lead_s, self._due, cue_id)
            self._held[cue_id] = (payload, timer)
        elif kind == "lesson.sync":
            # The page answers over the network, after the session has armed its wait.
            asyncio.get_running_loop().call_soon(self._sync, payload)

    def _answer(self, message: dict[str, object], label: str) -> None:
        self.simulated[label] += 1
        for handler in self._handlers:
            handler(message)

    def _ack(self, cue: dict[str, object], outcome: str, reason: str | None) -> None:
        ack = {
            "type": "lesson.ack",
            "epoch": cue["epoch"],
            "barrier": cue["barrier"],
            "cue_id": cue["cue_id"],
            "outcome": outcome,
            "reason": reason,
            "scene_id": self._scene_id,
            "step": self._step,
            "revision": self._revision,
        }
        self._answer(ack, "_".join(part for part in ("ack", outcome, reason) if part))

    def _due(self, cue_id: int) -> None:
        for held in sorted(n for n in self._held if n <= cue_id):
            cue, timer = self._held.pop(held)
            timer.cancel()
            tag = cue["tag"]
            if cue["epoch"] != self._epoch:
                reason = "stale_epoch"
            elif cue["barrier"] != self._barrier:
                reason = "stale_barrier"
            elif cue["revision"] != self._revision or cue["scene_id"] != self._scene_id:
                reason = "stale_revision"
            elif tag["kind"] == "step" and tag["n"] <= self._step:
                reason = "range"
            else:
                reason = None
            if reason is not None:
                self._ack(cue, "dropped", reason)
                continue
            if tag["kind"] == "scene":
                self._scene_id, self._step = tag["scene_id"], 1
            else:
                self._step = tag["n"]
            self._revision += 1
            self._last_cue = cue["cue_id"]
            self._ack(cue, "fired", None)

    def _drop(self, reason: str) -> None:
        for held in sorted(self._held):
            cue, timer = self._held.pop(held)
            timer.cancel()
            self._ack(cue, "dropped", reason)

    def _sync(self, sync: dict[str, object]) -> None:
        if sync["epoch"] != self._epoch:
            return
        self._drop("barrier")
        self._barrier = sync["barrier"]
        synced = {
            "type": "lesson.synced",
            "epoch": self._epoch,
            "barrier": self._barrier,
            "scene_id": self._scene_id,
            "step": self._step,
            "revision": self._revision,
            "last_cue": self._last_cue,
        }
        self._answer(synced, "synced")

    async def drain(self) -> None:
        while True:
            frame = await self._track.recv()
            at = time.perf_counter()
            real = bool(frame.to_ndarray().any())
            self.frames.append((at, real))
            if real and self.first_sound is None:
                self.first_sound = at
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

    async def close(self) -> None:
        for _, timer in self._held.values():
            timer.cancel()
        self._held.clear()
        await super().close()


class OutcomeWatch(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()
        self.planned_at: float | None = None
        self.unplanned = False
        self.event = asyncio.Event()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg.startswith(("turn.spoken ", "turn.failed ")):
            self.seen.add(str(record.args[0]))
            self.event.set()
        elif record.msg.startswith("lesson.planned ") and self.planned_at is None:
            self.planned_at = time.perf_counter()
        elif record.msg.startswith("planner.failed stage=connect "):
            self.unplanned = True
        return True

    async def wait(self, turn_id: str) -> None:
        while turn_id not in self.seen:
            self.event.clear()
            await self.event.wait()


def setup_settled(lesson: LessonState) -> bool:
    # Only the shown scene's build reaches the prompt; one further ahead may wait for a scene
    # that a setup short of its target never shows.
    scene = lesson.current()
    return (
        not lesson.sent
        and scene is not None
        and (scene.id in lesson.built or scene.id in lesson.failed)
    )


class Measured(NamedTuple):
    sample: Sample
    reply: str
    spoken: list[str]


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
        started: float,
        capture: Capture | None,
    ) -> None:
        self._loop_task = loop_task
        self._drain_task = drain_task
        self._source = source
        self.transport = transport
        self.reasoning = reasoning
        self._synth = synth
        self.watch = watch
        self._loop = loop
        self.started = started
        self._capture = capture
        self._dispatched = 0

    @classmethod
    def boot(
        cls,
        cfg: Settings,
        models: Models,
        inner: ReasoningClient,
        request: SessionRequest,
        planned: bool,
        plan: LessonPlan | None = None,
        stub_scenes: bool = False,
        replies: Sequence[str] = (),
        capture: Capture | None = None,
    ) -> "Bench":
        reasoning = MeteredReasoning(inner, plan, stub_scenes)
        reasoning.script(replies)
        transport = BenchTransport(models.synth)
        source = ScriptedSource()
        loop = build_loop(cfg, models, reasoning, source, transport, request, planned=planned)
        watch = OutcomeWatch()
        logging.getLogger("tutor.session").addFilter(watch)
        reasoning.begin()
        started = time.perf_counter()
        loop_task = asyncio.create_task(loop.run(), name="bench-loop")
        drain_task = asyncio.create_task(transport.drain(), name="bench-drain")
        return cls(
            loop_task,
            drain_task,
            source,
            transport,
            reasoning,
            models.synth,
            watch,
            loop,
            started,
            capture,
        )

    @property
    def lesson(self) -> LessonState:
        return self._loop._lesson

    def history(self) -> list[Message]:
        return self._loop._transcript.history(before=f"turn-{self._dispatched + 1}")

    async def turn(self, learner: str | None, utterance: int = 0, sample: bool = True) -> Measured:
        transport = self.transport
        if learner is None:
            since = ledger_since = 0
            record, t0 = self.reasoning.record, self.started
        else:
            transport.flush_playout()
            since = len(transport.frames)
            await transport.wait_frames(lambda: transport.idle_since(since))
            ledger_since = len(transport.ledger)
            record = self.reasoning.begin()
            self._synth.last = None
            t0 = time.perf_counter()
            self._source.inject(EndOfTurn(text=learner))
        self._dispatched += 1
        await self.watch.wait(f"turn-{self._dispatched}")
        spoken_at = len(transport.frames)

        played = transport.ledger[ledger_since:]
        model = model_text(record.streams)
        lengths = [entry.samples for entry in played]
        m = next((n for n, entry in enumerate(played) if is_model(entry.text, model)), None)
        needed = 1 if m is None else substance_frame_index(lengths, m) + 1
        await transport.wait_frames(
            lambda: (
                len(transport.real_times_since(since)) >= needed or transport.idle_since(spoken_at)
            )
        )

        reply = "\n".join("".join(deltas) for deltas in record.streams)
        if self._capture is not None:
            self._capture.write(self._dispatched, utterance, sample, len(reply))

        real_times = transport.real_times_since(since)
        first_sound = real_times[0] if real_times else None
        substance = substance_frame_time(lengths, m, real_times) if m is not None else None
        before = played[:m] if m is not None else played
        voice = record.voice_usage
        measured = Sample(
            first_sound_ms=None if first_sound is None else int((first_sound - t0) * 1000),
            substance_ms=None if substance is None else int((substance - t0) * 1000),
            first_content_delta_ms=record.first_spoken_ms,
            stages=max(sum(1 for entry in before if entry.text is not None) - 1, 0),
            silent=m is None,
            audio_ms=sum(entry.samples for entry in played) * 1000 // TTS_SAMPLE_RATE,
            voice_usd=None if voice is None else turn_cost_usd(voice, list_price=True),
            utterance=utterance,
        )
        return Measured(measured, reply, [entry.text for entry in played if entry.text is not None])

    async def settle(self) -> None:
        await self.transport.wait_frames(lambda: setup_settled(self.lesson))

    async def aclose(self) -> None:
        await self._loop.aclose()
        self._source.close()
        try:
            await self._loop_task
        finally:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            logging.getLogger("tutor.session").removeFilter(self.watch)
            await self.transport.close()


class TagRead(NamedTuple):
    tags: list[str]
    valid: bool
    dropped: dict[str, int]
    placement_errors: int
    leaked: int
    cues: list[str]
    cue_placement_errors: int


def classify_tags(
    reply: str, spoken: list[str], state: LessonState, learner_spoke: bool, sentence_rule: bool
) -> TagRead:
    ends = ".?!" if sentence_rule else ".?!;:,"
    probe = copy.deepcopy(state)
    probe.begin_turn(learner_spoke)
    splitter = TagSplitter()
    items = [*splitter.feed(reply), *splitter.finish()]
    tags: list[str] = []
    cues: list[str] = []
    dropped: Counter[str] = Counter()
    valid = True
    misplaced = 0
    cue_misplaced = 0
    at = 0
    since_cue = ""
    for item in items:
        if isinstance(item, str):
            continue
        marker = parse_marker(item)
        kind = tag_name(item)
        cue = None
        if isinstance(marker, str):
            valid = False
            tags.append(kind)
            dropped[f"{kind}:{marker}"] += 1
        else:
            check = probe.step_tag if marker.kind == "step" else probe.scene_tag
            reason = check(marker.n)
            if reason is None:
                cue = f"{marker.kind} {marker.n}"
            if reason != REPEAT:
                tags.append(f"{marker.kind} {marker.n}")
            if reason not in (None, REPEAT):
                valid = False
                dropped[f"{marker.kind}:{reason}"] += 1
        start = reply.index(f"<{item.text}>", at)
        text = reply[at:start].rstrip()
        if text and text[-1] not in ends:
            misplaced += 1
        # A tag that became no cue is gone from what the page receives, so a cue is placed by
        # the speech since the previous cue, not by whatever tag was written just before it.
        since_cue += reply[at:start]
        if cue is not None:
            cues.append(cue)
            heard = since_cue.rstrip()
            if heard and heard[-1] not in ends:
                cue_misplaced += 1
            since_cue = ""
        at = start + len(item.text) + 2
    said = [text.lower() for text in spoken]
    leaked = sum(
        1 for text in said for name in TAG_NAMES if f"<{name}" in text or f"</{name}" in text
    )
    return TagRead(tags, valid, dict(dropped), misplaced, leaked, cues, cue_misplaced)


def qualified(passed: int, leaked: int, groups: dict[str, int]) -> bool:
    covered = all(groups.get(group, 0) == SAMPLES // len(GROUPS) for group in GROUPS)
    return passed >= FLOOR and leaked == 0 and covered


def scenario_verdict(passed: int, leaked: int, n: int, groups: dict[str, int]) -> str:
    if n < SAMPLES:
        return f"verdict: scenarios passed {passed}/{n}, not a qualification run"
    line = (
        f"verdict: scenarios passed {passed}/{n} against the floor of {FLOOR}, leaked {leaked}/{n}"
    )
    return line + ("" if qualified(passed, leaked, groups) else ", FAIL")


def seen_verdict(passed: int, leaked: int, n: int, groups: dict[str, int]) -> str:
    line = f"verdict as seen: scenarios passed {passed}/{n} on the cues the page receives"
    if n < SAMPLES:
        return f"{line}, not a qualification run"
    line += f", against the floor of {FLOOR}, leaked {leaked}/{n}"
    return line + ("" if qualified(passed, leaked, groups) else ", FAIL")


class SetupTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learner: str | None
    reply: str


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    group: Literal["opening", "progress", "question", "answer", "boundary", "tangent"]
    setup: list[SetupTurn]
    target: Cursor
    asked: list[int]
    learner: str | None
    sequences: list[list[str]]
    question: bool
    terms: list[str]
    reveal_forbidden: list[str]
    requires_scene: int | None
    forbids_scene: bool


class ScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: list[str]
    tags_valid: bool
    placement_errors: int
    dropped: dict[str, int]
    progress_ok: bool
    scene_ok: bool
    reveal_ok: bool
    asked: bool | None
    leaked: int
    opening: bool
    passed: bool
    seen_tags: list[str]
    seen_passed: bool


def tag_checks(tags: list[str], scenario: Scenario) -> tuple[bool, bool, bool]:
    scenes = [tag for tag in tags if tag.startswith("scene ")]
    if scenario.requires_scene is not None:
        scene_ok = f"scene {scenario.requires_scene}" in scenes
    else:
        scene_ok = not (scenario.forbids_scene and scenes)
    return tags in scenario.sequences, scene_ok, not set(tags) & set(scenario.reveal_forbidden)


def evaluate_scenario(
    reply: str, spoken: list[str], state: LessonState, scenario: Scenario, sentence_rule: bool
) -> ScenarioResult:
    read = classify_tags(reply, spoken, state, scenario.learner is not None, sentence_rule)
    said = " ".join(spoken).strip()
    asked = None
    if scenario.question:
        asked = said.endswith("?") and all(term.lower() in said.lower() for term in scenario.terms)
    progress_ok, scene_ok, reveal_ok = tag_checks(read.tags, scenario)
    passed = (
        read.valid
        and progress_ok
        and scene_ok
        and reveal_ok
        and asked is not False
        and read.placement_errors == 0
        and read.leaked == 0
    )
    seen_passed = (
        all(tag_checks(read.cues, scenario))
        and asked is not False
        and read.cue_placement_errors == 0
        and read.leaked == 0
    )
    return ScenarioResult(
        tags=read.tags,
        tags_valid=read.valid,
        placement_errors=read.placement_errors,
        dropped=read.dropped,
        progress_ok=progress_ok,
        scene_ok=scene_ok,
        reveal_ok=reveal_ok,
        asked=asked,
        leaked=read.leaked,
        opening=scenario.learner is None,
        passed=passed,
        seen_tags=read.cues,
        seen_passed=seen_passed,
    )


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_digest: str
    cases: list[Scenario]

    @model_validator(mode="after")
    def _one_opening_first(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("case ids repeat")
        for case in self.cases:
            learners = [turn.learner for turn in case.setup] + [case.learner]
            if learners[0] is not None or None in learners[1:]:
                raise ValueError(f"{case.id}: only a case's first turn is the spontaneous opening")
        return self


class ListeningOutcome(BaseModel):
    sentence_boundaries_only: bool


class HarnessError(Exception):
    pass


def check_setup(acked: Cursor, history: list[Message], case: Scenario) -> None:
    if acked != case.target:
        raise HarnessError(
            f"{case.id}: the setup reached scene {acked.scene} step {acked.step}, "
            f"not scene {case.target.scene} step {case.target.step}"
        )
    expected: list[tuple[str, str]] = []
    for turn in case.setup:
        learner = OPENING_TEXT if turn.learner is None else turn.learner
        expected.append(("user", " ".join(learner.split())))
        splitter = TagSplitter()
        items = [*splitter.feed(turn.reply), *splitter.finish()]
        said = " ".join("".join(item for item in items if isinstance(item, str)).split())
        if said:
            expected.append(("assistant", said))
    found = [(message.role, " ".join(message.content.split())) for message in history]
    if found != expected:
        raise HarnessError(f"{case.id}: the transcript does not hold the setup's turns")


def run_order(cases: list[Scenario]) -> tuple[list[Scenario], list[Scenario]]:
    measured = sorted(cases, key=lambda case: GROUPS.index(case.group))
    warmups = [case for case in measured if case.group != "opening"][:WARMUP]
    return warmups, measured


def prepare_out(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"{path} already holds a run")


def write_case(
    out: Path, case_id: str, reply: str, spoken: list[str], result: ScenarioResult
) -> None:
    body = {"reply": reply, "spoken": spoken, "result": result.model_dump(mode="json")}
    (out / f"{case_id}.json").write_text(json.dumps(body, indent=2, ensure_ascii=True) + "\n")


async def run_scenarios(
    cases: list[Scenario],
    out: Path,
    run_case: Callable[[Scenario], Awaitable[tuple[str, list[str], ScenarioResult]]],
) -> list[ScenarioResult]:
    warmups, measured = run_order(cases)
    for case in warmups:
        await run_case(case)
    results: list[ScenarioResult] = []
    for case in measured:
        reply, spoken, result = await run_case(case)
        write_case(out, case.id, reply, spoken, result)
        results.append(result)
    return results


class ConnectSample(BaseModel):
    plan_ms: int | None
    plan_valid: bool
    first_audio_frame_ms: int | None
    scene_checked_ms: int | None


def first_scene_checked(
    checks: Sequence[tuple[float, str, bool]], plan: LessonPlan
) -> float | None:
    first = plan.scenes[0].id
    return next((at for at, scene_id, ok in checks if ok and scene_id == first), None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--soak", type=int, default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--starting-from", default=STARTING_FROM)
    parser.add_argument("--model", default=None)
    parser.add_argument("--capture", type=Path, default=None)
    parser.add_argument("--silent-synth", action="store_true", default=False)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--stub-scenes", action="store_true", default=False)
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--connect", action="store_true", default=False)
    return parser


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    synthetic = [
        flag
        for flag, given in (
            ("--silent-synth", args.silent_synth),
            ("--stub-scenes", args.stub_scenes),
            ("--plan", args.plan is not None),
            ("--scenarios", args.scenarios is not None),
        )
        if given
    ]
    mixed = synthetic + [
        flag
        for flag, given in (("--out", args.out is not None), ("--connect", args.connect))
        if given
    ]
    if args.soak is not None and mixed:
        parser.error(
            f"--soak runs the unplanned loop on Kokoro and takes none of {', '.join(mixed)}"
        )
    if args.connect and synthetic:
        parser.error(f"--connect times the real path and takes none of {', '.join(synthetic)}")
    if args.scenarios is not None and (
        args.plan is None or not args.stub_scenes or args.out is None
    ):
        parser.error("--scenarios needs --plan, --stub-scenes and --out")
    if args.out is not None and args.scenarios is None:
        parser.error("--out holds the reply files of a --scenarios run")
    return args


def planned(args: argparse.Namespace) -> bool:
    return args.plan is not None or args.connect


def ledger_line(prefix: str, ledger: UsageLedger) -> str:
    return (
        f"{prefix}turns={ledger.turns} prompt_tokens={ledger.total.prompt_tokens} "
        f"completion_tokens={ledger.total.completion_tokens} "
        f"cost_usd={ledger.cost_usd():.6f} list_usd={ledger.cost_usd(list_price=True):.6f}"
    )


def pool(into: UsageLedger, ledger: UsageLedger) -> None:
    into.turns += ledger.turns
    into.total = add_usage(into.total, ledger.total)


def report(
    cfg: Settings,
    args: argparse.Namespace,
    samples: list[Sample],
    reasoning: MeteredReasoning,
) -> None:
    print(
        "time to first sound: first 48 kHz frame carrying the first synthesized clause leaving "
        "the playout track after the scripted EndOfTurn is injected, or after the loop starts for "
        "the spontaneous opening; this is the first spoken word. Excludes endpointing, "
        "recognition and the browser."
    )
    print(
        "time to substance: frame carrying the first sample synthesized from a model-authored "
        "clause; the lead-in and stage sentences do not count. Frame granularity 20 ms."
    )
    print("first content delta: the model's first spoken delta after the turn's first voice call.")
    print(
        "planner calls and scene builds run beside the turns, not inside them: a turn's cost is "
        "its voice calls, and the planner and visual ledger lines carry the rest. With --plan the "
        "planner is answered from the file and with --stub-scenes every build is a stub; neither "
        "reaches the model. The page is simulated: it answers each scene.push with a passing "
        "scene.ready, fires each lesson.cue after its lead and answers each lesson.sync."
    )
    print(
        "a one-LSB floor is applied to every buffer before playout so an all-zero frame is exactly "
        "padding; it is inaudible and only serves attribution."
    )
    print(
        "cost is list price from the usage chunk that ends each stream; a stream cancelled or "
        "failed before that chunk has an unknown cost and is left out of the cost lines."
    )
    synth_field = "  synth=silent" if args.silent_synth else ""
    print(
        f"model={cfg.reasoning_model}{synth_field}  samples={len(samples)} "
        f"(plus {WARMUP} discarded warm-ups)  "
        f"subject={subject_for(args.root, args.subject)!r}  root={args.root}  "
        f"starting_from={args.starting_from!r}  planned={planned(args)}  plan={args.plan}  "
        f"stub_scenes={args.stub_scenes}  "
        f"planner_model={cfg.planner_model or cfg.reasoning_model}  "
        f"planner_effort={cfg.planner_effort}  scene_timeout_s={cfg.scene_timeout_s}  "
        f"scene_max_tokens={cfg.scene_max_tokens}  "
        f"scene_effort={cfg.scene_effort}  scene_model={cfg.scene_model or cfg.reasoning_model}"
    )

    def measured(label: str, values: list[int]) -> None:
        if args.silent_synth:
            print(f"{label:34s} not measured")
        else:
            summarize(label, values)

    measured(
        "time to first sound", [s.first_sound_ms for s in samples if s.first_sound_ms is not None]
    )
    measured("time to substance", [s.substance_ms for s in samples if s.substance_ms is not None])
    summarize(
        "first content delta (model, from first request)",
        [s.first_content_delta_ms for s in samples if s.first_content_delta_ms is not None],
    )
    measured("audio length per turn", [s.audio_ms for s in samples])
    priced = [s.voice_usd for s in samples if s.voice_usd is not None]
    summarize_usd("cost per turn (list)", priced)
    print(f"turns with unknown cost {len(samples) - len(priced)}/{len(samples)}")
    stages = [s.stages for s in samples if not s.silent]
    if stages:
        print(f"stage sentences before substance median={int(statistics.median(stages))}")
    silent = sum(1 for s in samples if s.silent)
    print(f"silent turns {silent}/{len(samples)}")
    print(ledger_line("", reasoning.ledger))
    print(ledger_line("voice ", reasoning.ledgers["voice"]))
    print(ledger_line("visual ", reasoning.ledgers["visual"]))
    print(ledger_line("planner ", reasoning.ledgers["planner"]))


def report_page(simulated: Counter[str]) -> None:
    counts = " ".join(f"{label}={count}" for label, count in sorted(simulated.items()))
    print(f"simulated page answers, no browser: {counts or 'none'}")


def report_scenarios(samples: list[Sample], sentence_rule: bool) -> None:
    n = len(samples)
    print(
        "scenario checks read each tag where the reply wrote it: a tag is placed at the reply's "
        f"start, after another tag, or after one of {'.?!' if sentence_rule else '.?!;:,'}. "
        "Asked is a proxy, a final question mark and every required term, case-insensitive, "
        "not a semantic judge. The setup replies, the plan and the builds are scripted; only the "
        "measured reply reaches the model."
    )
    placed = sum(1 for s in samples if s.tags_valid and s.placement_errors == 0)
    print(f"tags valid and placed {placed}/{n}")
    print(f"tags as scripted {sum(1 for s in samples if s.progress_ok)}/{n}")
    asking = [s for s in samples if s.asked is not None]
    print(f"asked, proxy {sum(1 for s in asking if s.asked)}/{len(asking)}")
    scenes = [s for s in samples if s.group in ("boundary", "tangent")]
    print(f"scene tags as required {sum(1 for s in scenes if s.scene_ok)}/{len(scenes)}")
    openings = [s for s in samples if s.opening]
    print(f"openings passed {sum(1 for s in openings if s.scenario_pass)}/{len(openings)}")
    dropped = sum((Counter(s.dropped) for s in samples), Counter())
    counts = " ".join(f"{key}={count}" for key, count in sorted(dropped.items()))
    print(f"tags dropped: {counts or 'none'}")
    groups = Counter(s.group for s in samples)
    for group in GROUPS:
        passed = sum(1 for s in samples if s.group == group and s.scenario_pass)
        print(f"{group} passed {passed}/{groups[group]}")
    leaked = sum(1 for s in samples if s.leaked)
    passed = sum(1 for s in samples if s.scenario_pass)
    print(seen_verdict(sum(1 for s in samples if s.seen_pass), leaked, n, dict(groups)))
    print(scenario_verdict(passed, leaked, n, dict(groups)))


def connect_series(label: str, values: list[int | None]) -> str:
    observed = sorted(value for value in values if value is not None)
    line = f"{label}: observed {len(observed)} missing {len(values) - len(observed)}"
    if observed:
        p95 = observed[max(0, int(len(observed) * 0.95) - 1)]
        line += f" median={int(statistics.median(observed))}ms p95={p95}ms max={observed[-1]}ms"
    return line


def report_connect(cfg: Settings, samples: list[ConnectSample], bound_s: float) -> None:
    print(
        "connect: each sample is a fresh loop on the real planner, the interim HTML builder, the "
        "voice and Kokoro, with t0 taken immediately before the loop's task is created. First "
        "audio is the first non-silent frame leaving the outbound track; scene checked is the "
        "simulated page passing the plan's first scene. Excluded: model loading before t0, "
        "browser output, a real iframe check and frame visibility. A sample that reaches the "
        "bound is kept with its nulls."
    )
    print(
        f"planner_model={cfg.planner_model or cfg.reasoning_model}  "
        f"planner_effort={cfg.planner_effort}  "
        f"scene_model={cfg.scene_model or cfg.reasoning_model}  scene_effort={cfg.scene_effort}  "
        f"voice_model={cfg.reasoning_model}  builder=interim-html  simulated page  "
        f"bound_s={bound_s:g}  samples={len(samples)} (plus {WARMUP} discarded warm-ups)"
    )
    print(
        connect_series(
            "connect to first outbound synthesized audio frame",
            [s.first_audio_frame_ms for s in samples],
        )
    )
    print(
        connect_series(
            "connect to scene checked (interim HTML builder, simulated page)",
            [s.scene_checked_ms for s in samples],
        )
    )
    print(f"plan valid {sum(1 for s in samples if s.plan_valid)}/{len(samples)}")
    complete = sum(
        1 for s in samples if s.first_audio_frame_ms is not None and s.scene_checked_ms is not None
    )
    print(f"measurement complete {complete}/{len(samples)}")


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
        f"soak: {args.soak} min of scripted turns through the assembled loop with no plan, on "
        "Kokoro, every turn kept "
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
        f"stages={sample.stages} silent={sample.silent} audio_ms={sample.audio_ms}"
    )


async def soak(
    cfg: Settings,
    args: argparse.Namespace,
    models: Models,
    inner: ReasoningClient,
    request: SessionRequest,
) -> int:
    sampler = await Sampler.start()
    try:
        if not await sampler.first_block():
            print(
                "powermetrics needs root: run this from a terminal where sudo can prompt, "
                "never under sudo as a whole"
            )
            return 2
        utterances = utterances_for(request.folder)
        bench = Bench.boot(cfg, models, inner, request, False)
        turns: list[tuple[float, Sample]] = []
        try:
            print(f"soak start {await machine_state()}", flush=True)
            started = time.perf_counter()
            sampler.rebase()
            while (at := time.perf_counter() - started) < args.soak * 60:
                n = len(turns)
                heard = await bench.turn(utterances[n % len(utterances)], n % len(utterances))
                turns.append((at, heard.sample))
                print(f"turn={n + 1} t={int(at)}s {turn_line(heard.sample)}", file=sys.stderr)
            print(f"soak end {await machine_state()}", flush=True)
        finally:
            await bench.aclose()
    finally:
        await sampler.aclose()
    report_soak(cfg, args, turns, sampler.records, bench.reasoning.ledger)
    return 0


async def run_utterances(
    cfg: Settings,
    args: argparse.Namespace,
    models: Models,
    inner: ReasoningClient,
    request: SessionRequest,
    plan: LessonPlan | None,
) -> int:
    utterances = utterances_for(request.folder)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    capture = Capture(
        CAPTURE_DIR / "replies.jsonl" if args.capture is None else args.capture, started
    )
    total = WARMUP + args.samples
    bench = Bench.boot(
        cfg, models, inner, request, planned(args), plan, args.stub_scenes, capture=capture
    )
    samples: list[Sample] = []
    try:
        print(f"start {await machine_state()}", flush=True)
        if planned(args):
            await bench.turn(None, sample=False)
        for n in range(total):
            heard = await bench.turn(
                utterances[n % len(utterances)], n % len(utterances), n >= WARMUP
            )
            if n >= WARMUP:
                samples.append(heard.sample)
            print(f"turn={n + 1}/{total} {turn_line(heard.sample)}", file=sys.stderr)
    finally:
        await bench.aclose()
    report(cfg, args, samples, bench.reasoning)
    report_page(bench.transport.simulated)
    return 0


async def run_fixture(
    cfg: Settings,
    args: argparse.Namespace,
    models: Models,
    inner: ReasoningClient,
    request: SessionRequest,
    plan: LessonPlan,
    fixture: Fixture,
    sentence_rule: bool,
) -> int:
    runs: list[tuple[Sample, MeteredReasoning, Counter[str]]] = []

    async def run_case(case: Scenario) -> tuple[str, list[str], ScenarioResult]:
        replies = [turn.reply for turn in case.setup]
        bench = Bench.boot(
            cfg, models, inner, request, planned(args), plan, args.stub_scenes, replies
        )
        try:
            for turn in case.setup:
                await bench.turn(turn.learner)
                await bench.settle()
            check_setup(bench.lesson.acked, bench.history(), case)
            # The opening starts on its own, so its state is taken as the plan adopted afresh.
            if case.learner is None:
                state = LessonState()
                state.adopt(plan)
            else:
                state = copy.deepcopy(bench.lesson)
            heard = await bench.turn(case.learner)
        finally:
            await bench.aclose()
        result = evaluate_scenario(heard.reply, heard.spoken, state, case, sentence_rule)
        fields = result.model_dump(exclude={"reveal_ok", "passed", "seen_tags", "seen_passed"})
        sample = heard.sample.model_copy(
            update={
                **fields,
                "scenario_id": case.id,
                "group": case.group,
                "scenario_pass": result.passed,
                "seen_pass": result.seen_passed,
            }
        )
        runs.append((sample, bench.reasoning, bench.transport.simulated))
        print(
            f"case={case.id} {turn_line(sample)} tags={result.tags} passed={result.passed}",
            file=sys.stderr,
        )
        return heard.reply, heard.spoken, result

    print(f"start {await machine_state()}", flush=True)
    results = await run_scenarios(fixture.cases, args.out, run_case)
    measured = runs[len(runs) - len(results) :]
    totals = MeteredReasoning(inner)
    for _, reasoning, _ in measured:
        pool(totals.ledger, reasoning.ledger)
        for kind, ledger in reasoning.ledgers.items():
            pool(totals.ledgers[kind], ledger)
    samples = [sample for sample, _, _ in measured]
    report(cfg, args, samples, totals)
    report_page(sum((simulated for _, _, simulated in measured), Counter()))
    report_scenarios(samples, sentence_rule)
    return 0


async def connect_sample(
    cfg: Settings,
    models: Models,
    inner: ReasoningClient,
    request: SessionRequest,
    bound_s: float,
) -> tuple[ConnectSample, Counter[str]]:
    bench = Bench.boot(cfg, models, inner, request, True)
    transport, lesson, watch = bench.transport, bench.lesson, bench.watch

    def checked() -> float | None:
        return None if lesson.plan is None else first_scene_checked(transport.checks, lesson.plan)

    # A plan that failed or a first scene given up on cannot be observed any later.
    def settled() -> bool:
        if transport.first_sound is None:
            return False
        if lesson.plan is None:
            return watch.unplanned
        return checked() is not None or lesson.plan.scenes[0].id in lesson.failed

    def since_t0(at: float | None) -> int | None:
        return None if at is None else int((at - bench.started) * 1000)

    try:
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(bound_s):
                await transport.wait_frames(settled)
        sample = ConnectSample(
            plan_ms=since_t0(watch.planned_at),
            plan_valid=watch.planned_at is not None,
            first_audio_frame_ms=since_t0(transport.first_sound),
            scene_checked_ms=since_t0(checked()),
        )
        return sample, transport.simulated
    finally:
        await bench.aclose()


async def run_connect(
    cfg: Settings,
    args: argparse.Namespace,
    models: Models,
    inner: ReasoningClient,
    request: SessionRequest,
) -> int:
    bound_s = 2 * cfg.planner_timeout_s + cfg.scene_timeout_s + SCENE_READY_TIMEOUT_S + 60
    total = WARMUP + args.samples
    samples: list[ConnectSample] = []
    simulated: Counter[str] = Counter()
    print(f"start {await machine_state()}", flush=True)
    for n in range(total):
        sample, page = await connect_sample(cfg, models, inner, request, bound_s)
        if n >= WARMUP:
            samples.append(sample)
            simulated += page
        fields = " ".join(f"{key}={value}" for key, value in sample.model_dump().items())
        print(f"connect={n + 1}/{total} {fields}", file=sys.stderr)
    report_page(simulated)
    report_connect(cfg, samples, bound_s)
    return 0


async def main() -> int:
    args = parse_args(sys.argv[1:])
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No turn latency can be reported. Populate .env and rerun.")
        return 2
    if args.model:
        cfg = cfg.model_copy(update={"reasoning_model": args.model})
    root = None if args.root is None else args.root.resolve()
    request = SessionRequest(
        subject=subject_for(root, args.subject), folder=root, starting_from=args.starting_from
    )
    plan = None if args.plan is None else LessonPlan.model_validate_json(args.plan.read_text())
    if args.scenarios is not None:
        fixture = Fixture.model_validate_json(args.scenarios.read_text())
        digest = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        if fixture.plan_digest != digest:
            print(f"{args.scenarios} was written against another plan: {args.plan} is {digest}")
            return 2
        outcome = ListeningOutcome.model_validate_json(LISTENING_OUTCOME.read_text())
        prepare_out(args.out)
    loaded = await asyncio.to_thread(load_models)
    synth = TaggedSynth(SilentSynth() if args.silent_synth else loaded.synth)
    models = Models(partial=loaded.partial, final=loaded.final, synth=synth)
    inner = ReasoningClient(cfg)
    try:
        if args.soak is not None:
            return await soak(cfg, args, models, inner, request)
        if args.connect:
            return await run_connect(cfg, args, models, inner, request)
        if args.scenarios is None:
            return await run_utterances(cfg, args, models, inner, request, plan)
        try:
            return await run_fixture(
                cfg, args, models, inner, request, plan, fixture, outcome.sentence_boundaries_only
            )
        except HarnessError as error:
            print(f"harness error: {error}")
            return 1
    finally:
        await inner.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
