"""Streaming time-to-first-token for the reasoning model, per reasoning mode.

Text only. Sends a synthetic tutor prompt and a synthetic user turn; never reads a
transcript, a recording, or any file from the target repository.
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
WARMUP = 1
SAMPLES = 30

SYSTEM_PROMPT_BODY = """
You are a demanding systems architect running a spoken, hands-free code walkthrough for an
experienced engineer. You direct the curriculum. You do not wait to be asked.

Rhythm. You move through three phases and you name the phase you are in.
Teach: introduce one subsystem, give its design pattern and its control flow, and push a
diagram to the canvas at the same moment you begin speaking about it.
Explore: walk the engineer into concrete files. Name entry points, structural invariants,
and failure propagation paths. Never read raw syntax aloud; describe the mechanism.
Reverse Feynman: stop lecturing and interrogate. Pose an edge case, a race, or a failure
mode, and make the engineer explain the mechanism back to you. On a misconception, cut in,
correct it in one sentence, and drop back to Explore on the exact lines that settle it.

Grounding. Every file path, symbol name, and line number you speak comes from a tool result
in the current turn. You have never seen this repository before this session. If a tool has
not returned a position, you do not have one, and you say so and call the tool.

Speech. You are being synthesized to audio and interrupted freely. Keep each turn under
four sentences unless the engineer asks for depth. No lists, no markdown, no code blocks,
no headings; none of it survives text to speech. Numbers spoken as words. When you need a
file, say its name naturally rather than spelling a path character by character.

Interruption. If the engineer speaks while you are speaking, you stop. You do not repeat
the sentence you were cut off in. You answer what they just said.

Visuals. When a topology, a lifecycle, or a state machine is the point, emit a diagram
payload before the sentence that explains it, so the picture is on screen slightly ahead of
your voice. Diagrams are structural, never decorative.

Tools. You have lexical search over the repository, structural search over its syntax
trees, a ranked symbol map, and a file reader that returns numbered lines. Search before
you assert. Cap what you pull; a wide search that floods your context makes you slower and
less accurate, and the engineer hears the pause.
""".strip()

USER_TURN = """
Alright, before we go further into the transport layer I want to back up. You said earlier
that the session actor owns the audio buffers and that the synthesis worker only ever gets
a handle to them. Walk me through what actually happens on a barge-in. The engineer is mid
sentence, the tutor is mid sentence, the voice detector fires. Who cancels whom, what
happens to the audio already queued in the output device buffer, and what stops the
half-finished model response from being spoken after the interruption is handled? I want
the ordering, and I want to know which of those steps can fail independently.
""".strip()


class Config(BaseModel):
    base_url: str
    api_key: str
    model: str


def load_config(model: str) -> Config | None:
    env = REPO / ".env"
    values: dict[str, str] = {}
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    base_url = os.environ.get("REASONING_API_BASE") or values.get("REASONING_API_BASE", "")
    api_key = os.environ.get("REASONING_API_KEY") or values.get("REASONING_API_KEY", "")
    if not base_url or not api_key:
        return None
    return Config(base_url=base_url, api_key=api_key, model=model)


def pad_to_tokens(body: str, target_words: int) -> str:
    words = body.split()
    out = list(words)
    while len(out) < target_words:
        out.extend(words[: target_words - len(out)])
    return " ".join(out)


class Sample(BaseModel):
    first_chunk_ms: int
    first_spoken_ms: int | None
    total_ms: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_chars: int


async def one_turn(client: AsyncOpenAI, cfg: Config, effort: str | None) -> Sample:
    kwargs = {}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    start = time.perf_counter()
    first_chunk: float | None = None
    first_spoken: float | None = None
    reasoning_chars = 0
    prompt_tokens = 0
    completion_tokens = 0
    stream = await client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": pad_to_tokens(SYSTEM_PROMPT_BODY, 1500)},
            {"role": "user", "content": pad_to_tokens(USER_TURN, 225)},
        ],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=150,
        **kwargs,
    )
    async for chunk in stream:
        now = time.perf_counter()
        if first_chunk is None:
            first_chunk = now
        if chunk.usage is not None:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            reasoning_chars += len(reasoning)
        if delta.content and first_spoken is None:
            first_spoken = now
    end = time.perf_counter()
    return Sample(
        first_chunk_ms=int((first_chunk - start) * 1000) if first_chunk else -1,
        first_spoken_ms=int((first_spoken - start) * 1000) if first_spoken else None,
        total_ms=int((end - start) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_chars=reasoning_chars,
    )


def summarize(label: str, values: list[int]) -> None:
    if not values:
        print(f"{label:34s} no samples")
        return
    ordered = sorted(values)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    print(
        f"{label:34s} n={len(ordered):3d} median={int(statistics.median(ordered)):6d}ms "
        f"p95={p95:6d}ms min={ordered[0]:6d}ms max={ordered[-1]:6d}ms"
    )


async def run_mode(client: AsyncOpenAI, cfg: Config, effort: str | None) -> None:
    name = effort or "default"
    for _ in range(WARMUP):
        await one_turn(client, cfg, effort)
    samples: list[Sample] = []
    for i in range(SAMPLES):
        samples.append(await one_turn(client, cfg, effort))
        print(f"  {name}: {i + 1}/{SAMPLES}", end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    print(f"\nreasoning_effort={name}  prompt_tokens={samples[0].prompt_tokens}")
    summarize("  time to first chunk", [s.first_chunk_ms for s in samples])
    summarize(
        "  time to first spoken token", [s.first_spoken_ms for s in samples if s.first_spoken_ms]
    )
    summarize("  full response", [s.total_ms for s in samples])
    silent = sum(1 for s in samples if s.first_spoken_ms is None)
    if silent:
        print(f"  turns that never produced spoken content: {silent}/{len(samples)}")
    reasoning = [s.reasoning_chars for s in samples]
    print(f"  reasoning chars median={int(statistics.median(reasoning))}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="z-ai/glm-5.3-flash")
    parser.add_argument("--modes", default="low,high,max")
    args = parser.parse_args()

    cfg = load_config(args.model)
    if cfg is None:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No reasoning-model latency can be reported. Populate .env and rerun.")
        return 2

    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    print(f"model={cfg.model}  samples={SAMPLES} (plus {WARMUP} discarded warm-up)")
    for effort in args.modes.split(","):
        await run_mode(client, cfg, effort.strip() or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
