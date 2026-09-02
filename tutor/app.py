import asyncio
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aiohttp import web

from tutor import signaling
from tutor.config import Settings, settings
from tutor.input_path import InputPath
from tutor.openers import synthesize_openers
from tutor.prompt import SYSTEM_PROMPT
from tutor.reasoning import ReasoningClient
from tutor.session import TurnLoop, TurnLoopConfig
from tutor.speech import Speaker
from tutor.stt import FINAL_CPU_THREADS, PARTIAL_CPU_THREADS, Transcriber, load_whisper
from tutor.tools.provenance import TurnRegistry
from tutor.tools.search import search
from tutor.transport import Connection
from tutor.tts import KokoroSynthesizer
from tutor.vad import SileroVad

logger = logging.getLogger(__name__)

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
VAD_MODEL = MODELS_ROOT / "silero" / "silero_vad.onnx"
WHISPER_DIR = MODELS_ROOT / "whisper"
KOKORO_WEIGHTS = MODELS_ROOT / "kokoro" / "kokoro-v1.0.fp16.onnx"
KOKORO_VOICES = MODELS_ROOT / "kokoro" / "voices-v1.0.bin"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


@dataclass
class Models:
    partial: Transcriber
    final: Transcriber
    synth: KokoroSynthesizer
    openers: dict[str, np.ndarray]


def load_models() -> Models:
    synth = KokoroSynthesizer(KOKORO_WEIGHTS, KOKORO_VOICES)
    return Models(
        partial=Transcriber(load_whisper(WHISPER_DIR, PARTIAL_CPU_THREADS)),
        final=Transcriber(load_whisper(WHISPER_DIR, FINAL_CPU_THREADS)),
        synth=synth,
        openers=synthesize_openers(synth),
    )


def build_loop(
    cfg: Settings,
    models: Models,
    reasoning: ReasoningClient,
    source: InputPath,
    transport: Connection,
) -> TurnLoop:
    speaker = Speaker(models.synth, transport, models.openers)
    loop_cfg = TurnLoopConfig(
        system=SYSTEM_PROMPT, subject=cfg.subject, root=cfg.repo_root.resolve()
    )
    return TurnLoop(loop_cfg, source, search, speaker, transport, reasoning, TurnRegistry())


class Sessions:
    def __init__(self, cfg: Settings, models: Models, reasoning: ReasoningClient) -> None:
        self._cfg = cfg
        self._models = models
        self._reasoning = reasoning
        self._tasks: set[asyncio.Task[None]] = set()
        self._started = 0

    @property
    def live(self) -> int:
        return len(self._tasks)

    def start(self, connection: Connection) -> asyncio.Task[None]:
        self._started += 1
        name = f"session-{self._started}"
        task = asyncio.create_task(self._session(connection), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        logger.info("session.started name=%s live=%d", name, len(self._tasks))
        return task

    def _done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        logger.info("session.ended name=%s live=%d", task.get_name(), len(self._tasks))
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("session.failed error=%s", type(error).__name__)

    async def _session(self, connection: Connection) -> None:
        vad = await asyncio.to_thread(SileroVad, VAD_MODEL)
        source = InputPath(connection, vad, self._models.partial, self._models.final)
        try:
            loop = build_loop(self._cfg, self._models, self._reasoning, source, connection)
            try:
                await loop.run()
            finally:
                await loop.aclose()
        finally:
            await source.aclose()

    async def aclose(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            if not task.cancelling():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _serve(cfg: Settings) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    models = await asyncio.to_thread(load_models)
    reasoning = ReasoningClient(cfg)
    sessions = Sessions(cfg, models, reasoning)
    runner = web.AppRunner(signaling.create_app(on_connection=sessions.start))
    try:
        await runner.setup()
        await web.TCPSite(runner, signaling.HOST, cfg.signaling_port).start()
        logger.info("tutor_ready host=%s port=%d", signaling.HOST, runner.addresses[0][1])
        await stop.wait()
    finally:
        await sessions.aclose()
        await runner.cleanup()
        await reasoning.aclose()


def main(cfg: Settings | None = None) -> None:
    if cfg is None:
        cfg = settings()
    logging.basicConfig(level=cfg.log_level, format=LOG_FORMAT)
    asyncio.run(_serve(cfg))
