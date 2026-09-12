import asyncio
import logging
import os
import re
import signal
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from tests.fakes import FakeSynthesizer, FakeTransport
from tests.test_input_path import CANONICAL, FakeVad, scripted_frames
from tests.test_session import SPOKEN_DELTAS, FakeReasoning, spoken_chunks
from tutor import app
from tutor.app import Models, Sessions, build_loop
from tutor.config import Settings
from tutor.input_path import EndOfTurn, InputEvent, InputPath
from tutor.openers import OPENER_PHRASES, synthesize_openers
from tutor.prompt import SYSTEM_PROMPT
from tutor.session import OPENER, TurnLoop
from tutor.signaling import SessionRequest

HANG_GUARD_S = 20.0
FAKE_BASE = "https://reasoning.invalid/v1"
FAKE_KEY = "sk-test-not-a-real-key"
FIXTURE_REPO = Path(__file__).resolve().parent / "data" / "fixture_repo"
SUBJECT = "a small http client with a bounded connection pool"
REQUEST = SessionRequest(subject=SUBJECT, folder=FIXTURE_REPO)
USER_TEXT = "walk me through src/pool.py"
READY_PREFIX = "tutor_ready "
READY_LINE = re.compile(r"^tutor_ready host=127\.0\.0\.1 port=\d+$")


class FakeTranscriber:
    def transcribe(self, audio: np.ndarray) -> str:
        return USER_TEXT


class RecordingVad(FakeVad):
    def __init__(self, probabilities: list[float]) -> None:
        super().__init__(probabilities)
        self.frames: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> float:
        self.frames.append(frame)
        return super().__call__(frame)


class FakeConnection(FakeTransport):
    def __init__(self, audio: list[np.ndarray], hold: asyncio.Event | None = None) -> None:
        super().__init__()
        self._audio = audio
        self._hold = hold
        self.closed = False

    async def frames(self) -> AsyncIterator[np.ndarray]:
        try:
            for frame in self._audio:
                yield frame
            if self._hold is not None:
                await self._hold.wait()
        finally:
            self.closed = True


class EndOfTurnSource:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    async def events(self) -> AsyncIterator[InputEvent]:
        for text in self._texts:
            yield EndOfTurn(text=text)


class FakeReasoningClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class ReadyFilter(logging.Filter):
    def __init__(self, ready: threading.Event) -> None:
        super().__init__()
        self._ready = ready

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage().startswith(READY_PREFIX):
            self._ready.set()
        return True


class Interrupter(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.ready = threading.Event()
        self.armed = True

    def run(self) -> None:
        self.ready.wait()
        if self.armed:
            os.kill(os.getpid(), signal.SIGINT)

    def disarm(self) -> None:
        self.armed = False
        self.ready.set()


@pytest.fixture
def closes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    order: list[str] = []

    class RecordingInputPath(InputPath):
        async def aclose(self) -> None:
            order.append("input_path")
            await super().aclose()

    class RecordingTurnLoop(TurnLoop):
        async def aclose(self) -> None:
            order.append("loop")
            await super().aclose()

    monkeypatch.setattr(app, "InputPath", RecordingInputPath)
    monkeypatch.setattr(app, "TurnLoop", RecordingTurnLoop)
    return order


@pytest.fixture
def vads(monkeypatch: pytest.MonkeyPatch) -> list[RecordingVad]:
    handed: list[RecordingVad] = []

    def fake_vad(path: Path) -> RecordingVad:
        handed.append(RecordingVad(CANONICAL))
        return handed[-1]

    monkeypatch.setattr(app, "SileroVad", fake_vad)
    return handed


def settings_for(tmp_path: Path, **extra: str) -> Settings:
    lines = [f"REASONING_API_BASE={FAKE_BASE}", f"REASONING_API_KEY={FAKE_KEY}"]
    lines.extend(f"{key}={value}" for key, value in extra.items())
    path = tmp_path / "app.env"
    path.write_text("\n".join(lines) + "\n")
    return Settings(_env_file=path)


def fake_models() -> Models:
    return Models(
        partial=FakeTranscriber(),
        final=FakeTranscriber(),
        synth=FakeSynthesizer(),
        openers=synthesize_openers(FakeSynthesizer()),
    )


def test_load_models_synthesizes_the_openers_once_in_the_loading_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synths: list[FakeSynthesizer] = []

    def fake_kokoro(weights: Path, voices: Path) -> FakeSynthesizer:
        synths.append(FakeSynthesizer())
        return synths[-1]

    monkeypatch.setattr(app, "load_whisper", lambda path, threads: None)
    monkeypatch.setattr(app, "Transcriber", lambda model: FakeTranscriber())
    monkeypatch.setattr(app, "KokoroSynthesizer", fake_kokoro)

    models = app.load_models()

    (synth,) = synths
    assert models.synth is synth
    assert synth.calls == list(OPENER_PHRASES.values())
    assert set(models.openers) == set(OPENER_PHRASES)


def test_main_boots_from_injected_settings_and_shuts_down_on_sigint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = settings_for(tmp_path, SIGNALING_PORT="0", LOG_LEVEL="DEBUG")
    clients: list[FakeReasoningClient] = []

    def recording_client(cfg: Settings) -> FakeReasoningClient:
        clients.append(FakeReasoningClient(cfg))
        return clients[-1]

    monkeypatch.setattr(app, "load_models", fake_models)
    monkeypatch.setattr(app, "ReasoningClient", recording_client)
    interrupter = Interrupter()
    ready = ReadyFilter(interrupter.ready)
    logging.getLogger("tutor.app").addFilter(ready)
    interrupter.start()
    try:
        app.main(cfg)
    finally:
        interrupter.disarm()
        logging.getLogger("tutor.app").removeFilter(ready)
        interrupter.join(HANG_GUARD_S)

    (client,) = clients
    assert client.cfg is cfg
    assert client.closed
    assert FAKE_KEY not in caplog.text
    (ready_line,) = [m for m in caplog.messages if m.startswith(READY_PREFIX)]
    assert READY_LINE.match(ready_line)
    assert not ready_line.endswith("port=0")


async def test_build_loop_speaks_the_thinking_opener_on_end_of_turn(tmp_path: Path) -> None:
    cfg = settings_for(tmp_path)
    log: list[tuple[str, object]] = []
    transport = FakeTransport()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), asyncio.Event())

    models = fake_models()
    loop = build_loop(cfg, models, reasoning, EndOfTurnSource([USER_TEXT]), transport, REQUEST)
    assert models.synth.calls == []
    assert transport.played == []
    await asyncio.wait_for(loop.run(), HANG_GUARD_S)

    assert transport.played[0] is models.openers[OPENER]
    assert len(transport.played) > 1
    (prompt,) = reasoning.prompts
    assert prompt.system.startswith(SYSTEM_PROMPT)
    assert f"Subject: {SUBJECT}" in prompt.system
    assert prompt.user_text == USER_TEXT


async def test_session_ends_on_its_own_when_the_frames_end(
    tmp_path: Path, closes: list[str], vads: list[RecordingVad]
) -> None:
    cfg = settings_for(tmp_path)
    log: list[tuple[str, object]] = []
    connection = FakeConnection(scripted_frames(len(CANONICAL)))
    models = fake_models()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), asyncio.Event())
    sessions = Sessions(cfg, models, reasoning)

    task = sessions.start(connection, REQUEST)
    await asyncio.wait_for(task, HANG_GUARD_S)

    assert task.result() is None
    assert connection.closed
    assert connection.played[0].size == len(OPENER_PHRASES[OPENER])
    assert ("stream_closed", len(SPOKEN_DELTAS)) in log
    (vad,) = vads
    assert vad.resets == 1
    assert sessions.live == 0
    assert closes == ["loop", "input_path"]


async def test_concurrent_sessions_each_get_their_own_vad(
    tmp_path: Path, vads: list[RecordingVad]
) -> None:
    cfg = settings_for(tmp_path)
    log: list[tuple[str, object]] = []
    positive = scripted_frames(len(CANONICAL))
    negative = [-frame for frame in positive]
    models = fake_models()
    reasoning = FakeReasoning(log, spoken_chunks(SPOKEN_DELTAS), asyncio.Event())
    sessions = Sessions(cfg, models, reasoning)

    first = sessions.start(FakeConnection(positive), REQUEST)
    second = sessions.start(FakeConnection(negative), REQUEST)
    assert sessions.live == 2
    await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), HANG_GUARD_S)

    assert len(vads) == 2
    assert [first.result(), second.result()] == [None, None]
    assert vads[0] is not vads[1]
    assert len(vads[0].frames) == len(CANONICAL)
    assert len(vads[1].frames) == len(CANONICAL)
    assert all(frame[0] > 0 for frame in vads[0].frames)
    assert all(frame[0] < 0 for frame in vads[1].frames)
    assert [vad.resets for vad in vads] == [1, 1]
    assert sessions.live == 0


async def test_shutdown_cancels_a_live_session_and_closes_its_collaborators(
    tmp_path: Path, closes: list[str], vads: list[RecordingVad]
) -> None:
    cfg = settings_for(tmp_path)
    log: list[tuple[str, object]] = []
    connection = FakeConnection(scripted_frames(len(CANONICAL)), hold=asyncio.Event())
    models = fake_models()
    reasoning = FakeReasoning(
        log, spoken_chunks(SPOKEN_DELTAS), asyncio.Event(), gate=asyncio.Event()
    )
    sessions = Sessions(cfg, models, reasoning)

    task = sessions.start(connection, REQUEST)
    await asyncio.wait_for(reasoning.started.wait(), HANG_GUARD_S)
    await asyncio.wait_for(reasoning.streams[0].held.wait(), HANG_GUARD_S)
    assert sessions.live == 1
    await asyncio.wait_for(sessions.aclose(), HANG_GUARD_S)

    assert task.cancelled()
    assert connection.closed
    assert ("stream_closed", 0) in log
    (vad,) = vads
    assert vad.resets == 1
    assert sessions.live == 0
    assert closes == ["loop", "input_path"]


async def test_a_failing_session_is_logged_by_type_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, vads: list[RecordingVad]
) -> None:
    caplog.set_level(logging.DEBUG)
    cfg = settings_for(tmp_path)
    connection = FakeConnection(scripted_frames(len(CANONICAL)))
    models = fake_models()

    class Exploding:
        def transcribe(self, audio: np.ndarray) -> str:
            raise RuntimeError(FAKE_KEY)

    models.final = Exploding()
    sessions = Sessions(cfg, models, FakeReasoningClient(cfg))

    task = sessions.start(connection, REQUEST)
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), HANG_GUARD_S)

    assert isinstance(task.exception(), RuntimeError)
    assert "session.failed error=RuntimeError" in caplog.messages
    assert FAKE_KEY not in caplog.text
    assert sessions.live == 0
