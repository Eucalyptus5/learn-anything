# Learn Anything

A voice-first tutor for learning an unfamiliar codebase. It holds a spoken, full-duplex
conversation about a target repository: it teaches a subsystem, walks you through the real
source, then interrogates you on it - and pushes diagrams to a browser canvas in step with
what it is saying.

The intended shape is a hybrid: speech recognition, speech synthesis, voice activity
detection, media transport, and code search all run locally on the machine; only the
reasoning model is a network call, and it receives text, never audio.

> Status: early. The design exists, nothing is built, and the design's own numbers have not
> been measured yet. The stack below is a candidate, not a decision.

## Teaching loop

The tutor works in three phases and moves between them deliberately rather than drifting.

1. **Teach.** Introduce one subsystem - an event bus, a consensus loop, an allocator - as
   design patterns and control flow, with a diagram appearing as it is described.
2. **Explore.** Go into the actual files. Entry points, invariants, how a failure propagates.
   It talks about exact lines without reading syntax aloud.
3. **Recall.** Stop teaching and ask. A realistic edge case, a race, a failure mode, answered
   out loud. A wrong answer sends the session back to phase two, at the lines that settle it.

Every claim the tutor makes about the code has to come from a tool result in that turn. It does
not get to guess a file path, a symbol, or a line number.

## Setup

```
uv sync
cp .env.example .env
```

The dependency list is deliberately empty. Components land in `pyproject.toml` one at a time,
each after it has been checked to exist, to run natively on arm64 macOS, and to behave in a
real-time loop.

## Tests

```
uv run pytest -q
```

Unit tests never sleep and never assert a wall-clock latency. Timing lives in a separate
benchmark harness that gets run on purpose and reports its sample count.

## Layout

`tutor/` is the package; `tests/` mirrors it. The internal module layout follows the subsystems
as they get built and validated, so it is not fixed in advance.
