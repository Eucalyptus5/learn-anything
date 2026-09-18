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
import re
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ValidationError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.bench_llm import summarize
from scripts.bench_turn import (
    PPO_UTTERANCES,
    SAMPLES,
    WARMUP,
    MeteredReasoning,
    is_truncated,
    is_valid,
    summarize_series,
    summarize_usd,
)
from tutor.brief import BRIEF_END, BRIEF_MARKER, VisualBrief
from tutor.config import Settings, settings
from tutor.cost import turn_cost_usd
from tutor.prompt import SYSTEM_PROMPT, Message, TurnPrompt
from tutor.reasoning import ReasoningClient
from tutor.session import OUTCOME_MARKER
from tutor.transcript import Transcript
from tutor.visual_call import run_visual_call, visual_prompt
from tutor.visuals import AppPush, ChannelPayload

DIAGRAM_BUDGET_MS = 20000
APP_BUDGET_MS = 90000
VALID_FLOOR = 27
LESSON_TAG = "/lesson.js"
STEP_KEY = re.compile(r"""["']?say["']?\s*:""")
NO_TOOL_CALL = "visual: error: no tool call"
KINDS = ("diagram", "app")
BUDGETS = {"diagram": DIAGRAM_BUDGET_MS, "app": APP_BUDGET_MS}
HISTORIES = ("fixed", "session")
HISTORY_TURNS = 10
OUTCOME_END = "</outcome>"
NONE_BRIEF = VisualBrief(kind="none", title="", show="")
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
SESSION_TURNS = (
    (
        0,
        VisualBrief(
            kind="diagram",
            title="PPO update loop",
            show=(
                "collect rollouts, estimate advantages with GAE, then several epochs of "
                "clipped minibatch updates, as a flowchart"
            ),
        ),
        (
            "PPO is a policy gradient method that keeps each update close to the policy that "
            "collected the data. You gather rollouts with the current policy, estimate the "
            "advantage of each action, then take several minibatch epochs on a clipped surrogate "
            "objective: the probability ratio between the new and the old policy times the "
            "advantage, with the ratio clipped to one plus or minus epsilon. What do you think the "
            "clip is protecting against?"
        ),
        '{"signal": "covered", "settling": "null"}',
    ),
    (
        1,
        VisualBrief(
            kind="app",
            title="Clipped objective",
            show=(
                "the clipped surrogate against the ratio from 0.5 to 1.5 for advantage plus "
                "one, epsilon 0.2, the flat region past 1.2 visible"
            ),
        ),
        (
            "Using the ratio directly would let one sample with a large advantage pull the policy "
            "as far as the optimizer can move it in one epoch, and the advantage was estimated "
            "under the old policy, so past some distance it describes nothing. Clipping the ratio "
            "to one plus or minus epsilon removes the incentive to move further in the direction "
            "the advantage rewards: the objective goes flat there and that sample's gradient is "
            "zero. Where does the flat region sit when the advantage is negative?"
        ),
        '{"signal": "follow_up", "settling": "null"}',
    ),
    (
        2,
        NONE_BRIEF,
        (
            "With a negative advantage the objective rewards lowering the ratio, and the minimum in "
            "the surrogate takes the clipped term once the ratio drops below one minus epsilon. So "
            "the policy can push a bad action's probability down until the ratio reaches one minus "
            "epsilon and no further, the mirror of the positive case. The minimum is what makes "
            "the bound pessimistic in both directions."
        ),
        '{"signal": "covered", "settling": "null"}',
    ),
    (
        3,
        NONE_BRIEF,
        (
            "Here is one. The old policy gave an action probability 0.2, the new one gives it 0.3, "
            "epsilon is 0.2 and the advantage is plus one. What is the ratio, what is the clipped "
            "ratio, and which of the two does the objective use?"
        ),
        '{"signal": "follow_up", "settling": "null"}',
    ),
    (
        4,
        VisualBrief(
            kind="app",
            title="Gradient past epsilon",
            show=(
                "the clipped objective and its slope against the ratio for advantage plus one, "
                "epsilon 0.2, the slope dropping to zero at 1.2"
            ),
        ),
        (
            "Not quite, and it is worth being exact here. Past epsilon the clipped term is a "
            "constant, so its gradient is zero, not larger; the minimum with the unclipped term "
            "picks the clipped one exactly when moving further would improve the objective, which "
            "is the case the clip exists to stop. Inside the band the gradient is the plain policy "
            "gradient and outside it there is none. Try the same reasoning with the advantage "
            "negative."
        ),
        (
            '{"signal": "misconception", "settling": "Past epsilon the clipped term is constant, '
            'so the gradient from that sample is zero, not larger."}'
        ),
    ),
    (
        5,
        NONE_BRIEF,
        (
            "For a negative advantage the objective is the advantage times the ratio while the "
            "ratio is above one minus epsilon, and a constant below it. The gradient pushes the "
            "probability down until the ratio reaches one minus epsilon and then stops. It is the "
            "positive case reflected about a ratio of one."
        ),
        '{"signal": "told", "settling": "null"}',
    ),
    (
        0,
        VisualBrief(
            kind="diagram",
            title="Rollout to update",
            show=(
                "one iteration as a flowchart: rollouts, returns and advantages, minibatch "
                "epochs, the updated policy becoming the old policy"
            ),
        ),
        (
            "Take it as a loop. Collect a batch of trajectories with the current policy, compute "
            "returns and advantages with GAE, then for a few epochs sample minibatches and step on "
            "the clipped objective plus a value loss and an entropy bonus. After those epochs the "
            "batch is discarded and the loop starts again with the updated policy as the new old "
            "policy. Which step is the one PPO changed relative to plain policy gradient?"
        ),
        '{"signal": "covered", "settling": "null"}',
    ),
    (
        1,
        NONE_BRIEF,
        (
            "Because the ratio alone has no ceiling. One minibatch step can move the probability "
            "of a well-rewarded action far beyond where the advantage was estimated, and the next "
            "step compounds it. The clip makes the objective indifferent to moves past epsilon, "
            "which is a cheap stand-in for the trust region constraint TRPO enforces with a "
            "second-order solve."
        ),
        '{"signal": "told", "settling": "null"}',
    ),
    (
        2,
        VisualBrief(
            kind="app",
            title="Negative advantage",
            show=(
                "the clipped surrogate against the ratio from 0.5 to 1.5 for advantage minus "
                "one, epsilon 0.2, flat below 0.8"
            ),
        ),
        (
            "When the advantage is negative the surrogate rewards lowering the ratio, and the "
            "minimum picks the clipped term once the ratio is below one minus epsilon. The picture "
            "is the mirror of the positive case: a slope down to one minus epsilon and flat beyond "
            "it. So a bad action's probability falls only as far as the band allows in one update."
        ),
        '{"signal": "covered", "settling": "null"}',
    ),
    (
        3,
        NONE_BRIEF,
        (
            "One more. Advantage minus one, old probability 0.5, new probability 0.35, epsilon "
            "0.2. Is the ratio inside the band, and does the objective use the clipped or the "
            "unclipped term?"
        ),
        '{"signal": "follow_up", "settling": "null"}',
    ),
)


def session_history() -> list[Message]:
    transcript = Transcript(HISTORY_TURNS)
    for n, (utterance, brief, reply, outcome) in enumerate(SESSION_TURNS, start=1):
        turn_id = f"turn-{n}"
        transcript.learner(turn_id, PPO_UTTERANCES[utterance])
        transcript.head(turn_id, f"{BRIEF_MARKER}{brief.model_dump_json()}{BRIEF_END}")
        transcript.tutor(turn_id, reply)
        transcript.tail(turn_id, f"{OUTCOME_MARKER}{outcome}{OUTCOME_END}")
    return transcript.history(before=f"turn-{len(SESSION_TURNS) + 1}")


SESSION_SNAPSHOT = TurnPrompt(
    system=SYSTEM_PROMPT, history=session_history(), user_text=SNAPSHOT.user_text
)
SNAPSHOTS = {"fixed": SNAPSHOT, "session": SESSION_SNAPSHOT}


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
    uses_helper: bool | None
    steps: int | None
    prose_only: bool


async def one_call(
    reasoning: MeteredReasoning,
    kind: str,
    max_tokens: int,
    model: str | None = None,
    history: str = "fixed",
) -> CallSample:
    record = reasoning.begin()
    channel = RecordingChannel()
    t0 = time.perf_counter()
    result = await run_visual_call(
        reasoning,
        visual_prompt(SNAPSHOTS[history], BRIEFS[kind], None, "light"),
        channel,
        max_tokens,
        lambda: True,
        model=model,
    )
    result_ms = int((time.perf_counter() - t0) * 1000)
    usage = record.visual_usage
    html = next((p.html for _, p in channel.pushes if isinstance(p, AppPush)), None)
    return CallSample(
        first_chunk_ms=record.visual_first_chunk_ms,
        landing_ms=None if not channel.pushes else int((channel.pushes[0][0] - t0) * 1000),
        result_ms=result_ms,
        valid=is_valid(result),
        truncated=is_truncated(result),
        output_tokens=None if usage is None else usage.completion_tokens,
        cost_usd=None if usage is None else turn_cost_usd(usage, list_price=True),
        result=result,
        uses_helper=None if html is None else LESSON_TAG in html,
        steps=None if html is None else len(STEP_KEY.findall(html)),
        prose_only=result == NO_TOOL_CALL,
    )


def verdict(kind: str, landing_ms: int | None, valid: int, n: int, prose_only: int) -> str:
    budget = BUDGETS[kind]
    landing = "none" if landing_ms is None else f"{landing_ms}ms"
    line = (
        f"verdict: {kind} median landing {landing} against the {budget}ms budget, "
        f"valid {valid}/{n} against the floor of {VALID_FLOOR}, prose only {prose_only}/{n}"
    )
    failed = landing_ms is None or landing_ms > budget or valid < VALID_FLOOR
    return f"{line}, FAIL" if failed else line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=["diagram", "app"])
    parser.add_argument("--history", choices=HISTORIES, default="fixed")
    return parser


def report(cfg: Settings, args: argparse.Namespace, samples: dict[str, list[CallSample]]) -> None:
    print(
        "landing: from the request leaving run_visual_call to the validated payload reaching a "
        "recording channel that sends nothing, over the calls that landed; the voice call, the "
        "session, the browser and the frame's render are excluded. result, every call: from the "
        "same request to run_visual_call returning, failures included.",
        "history=session is ten turns as Transcript.history sends them, each assistant message "
        "opening with the brief head and closing with the outcome tail as the model writes it; "
        "history=fixed is the two-turn history the 2026-09-13 runs used.",
    )
    print(
        "first-try validity: the result string ends in sent, so a payload failing the push "
        "model's validation, a stream with no tool call and a truncated stream all count invalid. "
        "cost is list price from the usage chunk that ends the stream; a stream that ended "
        "without that chunk has an unknown cost and is left out of the cost line.",
        "helper is the share of landed apps whose document loads /lesson.js; steps counts the "
        "say keys in the document.",
        "prose only counts the streams that ended with no tool call at all, which the session "
        "logs as visual.prose with no visual.call line after it.",
    )
    print(
        f"model={cfg.reasoning_model}  visual_model={cfg.visual_model or cfg.reasoning_model}  "
        f"samples={args.samples} (plus {WARMUP} discarded warm-ups)  "
        f"visual_max_tokens={cfg.visual_max_tokens}  theme=light  previous=none  "
        f"history={args.history}"
    )
    for kind in args.kinds:
        calls = samples[kind]
        landing = [s.landing_ms for s in calls if s.landing_ms is not None]
        costs = [s.cost_usd for s in calls if s.cost_usd is not None]
        valid = sum(1 for s in calls if s.valid)
        prose = sum(1 for s in calls if s.prose_only)
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
        print(f"prose only {prose}/{len(calls)}")
        apps = [s for s in calls if s.uses_helper is not None]
        if apps:
            print(f"helper {sum(1 for s in apps if s.uses_helper)}/{len(apps)}")
            summarize_series("steps", [s.steps for s in apps if s.steps is not None], "")
        print(f"unknown cost {len(calls) - len(costs)}/{len(calls)}")
        median = int(statistics.median(landing)) if landing else None
        print(verdict(kind, median, valid, len(calls), prose))


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
                sample = await one_call(
                    reasoning, kind, cfg.visual_max_tokens, cfg.visual_model or None, args.history
                )
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
