# Learn Anything

A voice-first tutor for any subject: a concept like PPO, a paper, an algorithm, or a folder of
source code. It holds a spoken, full-duplex conversation, explains the thing aloud, and puts
diagrams and small interactive apps on a browser canvas alongside the speech. When the subject is
a codebase, it walks through the real files, and every path or line it speaks has to come from a
search it ran in that turn.

Speech recognition, synthesis, voice activity detection, media transport and code search all run
on the machine. The only network call is the reasoning model, and it receives text, never audio.

## What runs today

`uv run tutor` boots one process:

- an aiohttp signaling server on `127.0.0.1:8080` serving the page and answering WebRTC offers;
- aiortc carrying microphone audio in and synthesized speech out, with a data channel for
  captions, state chips, the transcript and visual payloads;
- Silero VAD over raw onnxruntime on 32 ms frames, with a trailing-silence endpointer and a
  speculative transcription that starts inside the silence window;
- faster-whisper `base.en` int8 for recognition;
- Kokoro-82M ONNX for synthesis, driven clause by clause through a worker thread so the event
  loop never blocks on a native call;
- ripgrep as the grounding tool when a folder is given;
- an OpenAI-compatible reasoning endpoint, `glm-5.3-flash` by default, configured from `.env`.

A turn is one streamed voice call built from the last ten turns and the current pedagogy
directive. If the model opens its reply with a visual brief, a second call runs beside the voice
with the same history and lands either a mermaid diagram or an app built on plotly, KaTeX, p5 or
d3. Those libraries are vendored and served from `/vendor/`; the visual itself renders only inside
a sandboxed iframe with no network and no bridge to the host page. A visual that finishes after a
newer turn's visual has already landed is dropped.

In folder mode the model gets a `search_code` tool. A gate between the model's text and the
synthesizer withholds any clause that names a file path, symbol or line that did not come back
from a tool result in the current turn. That rule lives in code, not in the prompt.

Barge-in is real: speaking while the tutor is talking cancels the in-flight reasoning stream,
drops queued playout, and starts the next turn.

## Measured

Every number below comes from a harness under `scripts/` on one M2 laptop, most of them taken
with the machine deep in swap. The sample count is stated with each figure.

- Transport loopback, Opus encode through the jitter buffer, decode and the inbound reframer:
  median 75 ms, p95 78 ms, n=30, two runs agreeing within 1 ms. Peer connect: median 31 ms,
  n=30. Excludes browser capture and playout.
- Barge-in, injected while a clause is being synthesized: from the injection to the last frame of
  the interrupted turn, 0 ms in every sample; to the next turn's first frame on the wire, median
  19 ms, p95 28 ms, n=29. Measured on an earlier shape of the turn loop that led each turn with a
  cached opener; the flush path is unchanged since. The abandoned Kokoro call keeps running on
  its thread and overlaps the next turn's first synthesis by a median of 1.4 s.
- Concept turn, end of speech to first synthesized sound: median 4.4 s and 4.5 s, n=30 each.
  Folder turn with a search round, first model-authored sound: median 8.4 s, n=30.
- Visual call alone: a diagram lands median 7.9 s after the brief, an app median 14.6 s, n=30
  each, 30 of 30 valid in both runs.
- Cost at list price: median 0.0004 USD per turn, worst turn 0.0012 USD; a hundred-turn hour with
  every turn briefed is about 0.11 USD.
- Silero per 512-sample frame: median 98 us, n=269. Whisper on 2.9 s of speech: median 405 ms,
  n=30. Kokoro runs at RTF 0.75 to 0.91 here and is the local bottleneck.
- Thirty-minute soak, 95 turns: first sound did not degrade between the first and last five
  minutes. Thermal pressure was Heavy for 43 percent of 180 powermetrics samples.

## Known limits

- The model omits the visual brief on most turns once there is history: 27 of 30 and 28 of 30
  turns in the two bench runs. Fresh sessions draw; long ones mostly stop.
- The pedagogy machine never leaves the teach phase with this model. The outcome tail it is
  asked to write fails validation or is absent on every measured turn, so interrogation and
  misconception handling are unreachable in a real session until that is fixed.
- Four to eight seconds from end of speech to the first substantive word. The design accepted
  about six; folder mode misses it at the median.
- Tested only on arm64 macOS with Chrome. The browser's echo cancellation is relied on and has
  not been checked by a listener with the real page.
- The visual history in the page is unbounded, and a theme flip does not re-render a diagram
  already on screen.

## Setup

```
uv sync
cp .env.example .env
```

Fill `REASONING_API_BASE` and `REASONING_API_KEY`. The other keys in `.env.example` have
defaults.

Model weights live under `models/` and are not committed:

```
uv run python scripts/fetch_models.py
```

fetches the Silero VAD graph against a pinned sha256. Whisper pulls `Systran/faster-whisper-base.en`
into `models/whisper/` on first load and never contacts the network again. Kokoro is not fetched
by a script: put `kokoro-v1.0.fp16.onnx` and `voices-v1.0.bin` from the kokoro-onnx
`model-files-v1.1` release under `models/kokoro/`.

The browser libraries are vendored the same way:

```
uv run python scripts/fetch_client_assets.py
```

downloads sha256-pinned tarballs from the npm registry into `client/vendor/`. ripgrep has to be
on `PATH`. Then:

```
uv run tutor
```

and open `http://127.0.0.1:8080` in Chrome. Type a subject, optionally a folder and a starting
point, and talk.

## Tests

```
uv run pytest -q
uv run ruff format && uv run ruff check
```

875 tests in about six seconds. Every model is a fake in the suite; ripgrep is real and runs
against `tests/data/fixture_repo`. No test sleeps or asserts a wall-clock latency. Timing lives
in `scripts/bench_*.py`, run on purpose, and each prints its sample count.

## Layout

`tutor/` is the package and `tests/` mirrors it. `client/` holds the page, the two frames and
the vendored libraries. `scripts/` holds the fetchers and the benchmark harnesses. `models/`,
`captures/` and `.env` stay local.
