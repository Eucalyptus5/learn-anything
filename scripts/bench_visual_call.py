"""Visual call landing and first-try validity through the real reasoning model: a fixed PPO
snapshot and a fixed brief per kind go through ``run_visual_call`` with ``tool_choice`` required,
the validated payload is recorded on a channel that sends nothing, and the harness reports the
landing time of the calls that landed, the result time of every call, the first chunk, the
output tokens and the list-price cost per call. Text only on the wire; no session, no browser,
no frame render.
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from scripts.bench_turn import (
    SAMPLES,
    WARMUP,
    MeteredReasoning,
    is_truncated,
    is_valid,
    summarize_series,
    summarize_usd,
)
from tutor.brief import VisualBrief
from tutor.config import Settings, settings
from tutor.cost import turn_cost_usd
from tutor.prompt import SYSTEM_PROMPT, Message, TurnPrompt
from tutor.reasoning import ReasoningClient
from tutor.visual_call import run_visual_call, visual_prompt
from tutor.visuals import ChannelPayload

DIAGRAM_BUDGET_MS = 20000
APP_BUDGET_MS = 90000
VALID_FLOOR = 27
KINDS = ("diagram", "app")
BUDGETS = {"diagram": DIAGRAM_BUDGET_MS, "app": APP_BUDGET_MS}
DIAGRAM_BRIEF = VisualBrief(
    kind="diagram",
    title="PPO update loop",
    show=(
        "collect rollouts, estimate advantages with GAE, then several epochs of minibatch "
        "clipped updates, as a flowchart"
    ),
)
APP_BRIEF = VisualBrief(
    kind="app",
    title="Clipped objective",
    show=(
        "the clipped surrogate objective against the probability ratio from 0.5 to 1.5 for "
        "advantage plus one and minus one, epsilon 0.2, the flat regions visible"
    ),
)
BRIEFS = {"diagram": DIAGRAM_BRIEF, "app": APP_BRIEF}
SNAPSHOT = TurnPrompt(
    system=SYSTEM_PROMPT,
    history=[
        Message(role="user", content="teach me ppo"),
        Message(
            role="assistant",
            content=(
                "PPO is a policy gradient method that keeps each update close to the policy "
                "that collected the data. It maximizes a clipped surrogate objective: the "
                "probability ratio between the new and the old policy times the advantage, "
                "with the ratio clipped to one plus or minus epsilon. What do you think the "
                "clip is protecting against?"
            ),
        ),
        Message(role="user", content="a step that is too big"),
        Message(
            role="assistant",
            content=(
                "Right. Without the clip one large ratio could drag the policy far from the "
                "data it was trained on, and the advantage estimates would no longer describe "
                "that policy. The clip removes the incentive to push the ratio past epsilon in "
                "the direction the advantage rewards."
            ),
        ),
    ],
    user_text="why does it clip",
)


class RecordingChannel:
    def __init__(self) -> None:
        self.pushes: list[tuple[float, ChannelPayload]] = []

    async def push(self, payload: ChannelPayload) -> None:
        self.pushes.append((time.perf_counter(), payload))


class CallSample(BaseModel):
    first_chunk_ms: int | None
    landing_ms: int | None
    result_ms: int
    valid: bool
    truncated: bool
    output_tokens: int | None
    cost_usd: float | None
    result: str


async def one_call(reasoning: MeteredReasoning, kind: str, max_tokens: int) -> CallSample:
    record = reasoning.begin()
    channel = RecordingChannel()
    t0 = time.perf_counter()
    result = await run_visual_call(
        reasoning,
        visual_prompt(SNAPSHOT, BRIEFS[kind], None, "light"),
        channel,
        max_tokens,
        lambda: True,
    )
    result_ms = int((time.perf_counter() - t0) * 1000)
    usage = record.visual_usage
    return CallSample(
        first_chunk_ms=record.visual_first_chunk_ms,
        landing_ms=None if not channel.pushes else int((channel.pushes[0][0] - t0) * 1000),
        result_ms=result_ms,
        valid=is_valid(result),
        truncated=is_truncated(result),
        output_tokens=None if usage is None else usage.completion_tokens,
        cost_usd=None if usage is None else turn_cost_usd(usage, list_price=True),
        result=result,
    )


def verdict(kind: str, landing_ms: int | None, valid: int, n: int) -> str:
    budget = BUDGETS[kind]
    landing = "none" if landing_ms is None else f"{landing_ms}ms"
    line = (
        f"verdict: {kind} median landing {landing} against the {budget}ms budget, "
        f"valid {valid}/{n} against the floor of {VALID_FLOOR}"
    )
    failed = landing_ms is None or landing_ms > budget or valid < VALID_FLOOR
    return f"{line}, FAIL" if failed else line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=["diagram", "app"])
    return parser


def report(cfg: Settings, args: argparse.Namespace, samples: dict[str, list[CallSample]]) -> None:
    print(
        "landing: from the request leaving run_visual_call to the validated payload reaching a "
        "recording channel that sends nothing, over the calls that landed; the voice call, the "
        "session, the browser and the frame's render are excluded. result, every call: from the "
        "same request to run_visual_call returning, failures included."
    )
    print(
        "first-try validity: the result string ends in sent, so a payload failing the push "
        "model's validation, a stream with no tool call and a truncated stream all count invalid. "
        "cost is list price from the usage chunk that ends the stream; a stream that ended "
        "without that chunk has an unknown cost and is left out of the cost line."
    )
    print(
        f"model={cfg.reasoning_model}  samples={args.samples} (plus {WARMUP} discarded "
        f"warm-ups)  visual_max_tokens={cfg.visual_max_tokens}  theme=light  previous=none"
    )
    for kind in args.kinds:
        calls = samples[kind]
        landing = [s.landing_ms for s in calls if s.landing_ms is not None]
        costs = [s.cost_usd for s in calls if s.cost_usd is not None]
        valid = sum(1 for s in calls if s.valid)
        print(f"kind={kind}")
        summarize("landing", landing)
        summarize("result, every call", [s.result_ms for s in calls])
        summarize("first chunk", [s.first_chunk_ms for s in calls if s.first_chunk_ms is not None])
        summarize_series(
            "output tokens", [s.output_tokens for s in calls if s.output_tokens is not None], ""
        )
        summarize_usd("cost usd (list)", costs)
        print(f"valid {valid}/{len(calls)}")
        print(f"truncated {sum(1 for s in calls if s.truncated)}/{len(calls)}")
        print(f"unknown cost {len(calls) - len(costs)}/{len(calls)}")
        median = int(statistics.median(landing)) if landing else None
        print(verdict(kind, median, valid, len(calls)))


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No visual call latency can be reported. Populate .env and rerun.")
        return 2
    reasoning = MeteredReasoning(ReasoningClient(cfg))
    samples: dict[str, list[CallSample]] = {kind: [] for kind in args.kinds}
    total = WARMUP + args.samples
    try:
        for kind in args.kinds:
            for n in range(total):
                sample = await one_call(reasoning, kind, cfg.visual_max_tokens)
                if n >= WARMUP:
                    samples[kind].append(sample)
                print(
                    f"kind={kind} call={n + 1}/{total} landing_ms={sample.landing_ms} "
                    f"result_ms={sample.result_ms} "
                    f"first_chunk_ms={sample.first_chunk_ms} valid={sample.valid} "
                    f"truncated={sample.truncated} tokens={sample.output_tokens} "
                    f"result={sample.result}",
                    file=sys.stderr,
                )
    finally:
        await reasoning.aclose()
    report(cfg, args, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
