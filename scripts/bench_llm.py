"""Streaming time-to-first-token for the reasoning model, per reasoning mode.

Text only. Sends a synthetic tutor prompt and a synthetic user turn; never reads a
transcript, a recording, or any file from the target repository.
"""

import argparse
import asyncio
import statistics
import sys
import time
import uuid
from pathlib import Path

from openai import RateLimitError
from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tutor.config import settings
from tutor.cost import TurnUsage
from tutor.prompt import TurnPrompt
from tutor.reasoning import ReasoningClient

WARMUP = 1
RETRY_BACKOFF_S = 20
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

TERSE_REASONING = (
    "Reasoning budget. You are being synthesized to audio and the engineer waits in silence while "
    "you think. Keep internal reasoning to one short sentence at most, then start speaking. Never "
    "deliberate at length before answering; think while you talk, not before."
)

USER_TURN = """
Alright, before we go further into the transport layer I want to back up. You said earlier
that the session actor owns the audio buffers and that the synthesis worker only ever gets
a handle to them. Walk me through what actually happens on a barge-in. The engineer is mid
sentence, the tutor is mid sentence, the voice detector fires. Who cancels whom, what
happens to the audio already queued in the output device buffer, and what stops the
half-finished model response from being spoken after the interruption is handled? I want
the ordering, and I want to know which of those steps can fail independently.
""".strip()


def pad_to_tokens(body: str, target_words: int) -> str:
    words = body.split()
    out = list(words)
    while len(out) < target_words:
        out.extend(words[: target_words - len(out)])
    return " ".join(out)


class Sample(BaseModel):
    first_chunk_ms: int | None
    first_spoken_ms: int | None
    total_ms: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_chars: int


async def one_turn(
    client: ReasoningClient, mode: str, max_tokens: int, system_prompt: str
) -> Sample:
    prompt = TurnPrompt(system=system_prompt, user_text=pad_to_tokens(USER_TURN, 225))
    start = time.perf_counter()
    stream = client.start_turn(prompt, effort=mode, max_tokens=max_tokens)
    async for _ in stream:
        pass
    total_ms = int((time.perf_counter() - start) * 1000)
    usage = stream.usage or TurnUsage(prompt_tokens=0, completion_tokens=0)
    return Sample(
        first_chunk_ms=stream.first_chunk_ms,
        first_spoken_ms=stream.first_spoken_ms,
        total_ms=total_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        reasoning_chars=usage.reasoning_chars,
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


async def run_mode(
    client: ReasoningClient,
    mode: str,
    samples_n: int,
    max_tokens: int,
    terse: bool,
    cache_bust: bool,
) -> None:
    name = f"{mode}{' +terse' if terse else ''}"
    system_prompt = pad_to_tokens(SYSTEM_PROMPT_BODY, 1500)
    if cache_bust:
        system_prompt = uuid.uuid4().hex + " " + system_prompt
    if terse:
        system_prompt = system_prompt + "\n\n" + TERSE_REASONING
    for _ in range(WARMUP):
        while True:
            try:
                await one_turn(client, mode, max_tokens, system_prompt)
                break
            except RateLimitError:
                await asyncio.sleep(RETRY_BACKOFF_S)
    samples: list[Sample] = []
    retries = 0
    for i in range(samples_n):
        while True:
            try:
                samples.append(await one_turn(client, mode, max_tokens, system_prompt))
                break
            except RateLimitError:
                retries += 1
                await asyncio.sleep(RETRY_BACKOFF_S)
        print(f"  {name}: {i + 1}/{samples_n}", end="\r", file=sys.stderr)
    print(" " * 40, end="\r", file=sys.stderr)
    print(f"\nreasoning_effort={name}  prompt_tokens={samples[0].prompt_tokens}")
    summarize(
        "  time to first chunk", [s.first_chunk_ms for s in samples if s.first_chunk_ms is not None]
    )
    summarize(
        "  time to first spoken token", [s.first_spoken_ms for s in samples if s.first_spoken_ms]
    )
    summarize("  full response", [s.total_ms for s in samples])
    silent = sum(1 for s in samples if s.first_spoken_ms is None)
    if silent:
        print(f"  turns that never produced spoken content: {silent}/{len(samples)}")
    if retries:
        print(f"  rate-limit retries (excluded from timings): {retries}")
    cached = [s.cached_tokens for s in samples]
    print(
        f"  cached prompt tokens median={int(statistics.median(cached))}/{samples[0].prompt_tokens}"
    )
    reasoning = [s.reasoning_chars for s in samples]
    print(f"  reasoning chars median={int(statistics.median(reasoning))}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--modes", default="low")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--terse", action="store_true")
    parser.add_argument("--cache-bust", action="store_true")
    args = parser.parse_args()

    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No reasoning-model latency can be reported. Populate .env and rerun.")
        return 2
    if args.model:
        cfg = cfg.model_copy(update={"reasoning_model": args.model})

    client = ReasoningClient(cfg)
    print(
        f"model={cfg.reasoning_model}  samples={args.samples} (plus {WARMUP} discarded warm-up)  max_tokens={args.max_tokens}"
    )
    for mode in args.modes.split(","):
        await run_mode(
            client, mode.strip(), args.samples, args.max_tokens, args.terse, args.cache_bust
        )
    await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
