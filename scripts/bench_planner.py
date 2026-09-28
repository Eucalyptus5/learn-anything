"""Planner call time and validity through the real reasoning model: n forced write_plan calls
at one stage, connect from the fixed form or a rerun from a fixed transcript and a plan file,
timed and validated, every valid plan written to a run directory; or one connect call per topic
of a list, each plan saved under the topic's id."""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.bench_llm import summarize
from scripts.bench_turn import (
    SAMPLES,
    STARTING_FROM,
    WARMUP,
    MeteredReasoning,
    is_truncated,
    summarize_series,
    summarize_usd,
)
from tutor.config import Settings, settings
from tutor.cost import turn_cost_usd
from tutor.lesson import OPENING_TEXT, LessonPlan, rerun_conflict, splice_rerun
from tutor.planner import EMPTY_REPLY, NO_TOOL_CALL, plan_prompt, run_planner
from tutor.prompt import Message, TurnPrompt
from tutor.reasoning import ReasoningClient
from tutor.visuals import SCENE_ID

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "scratch" / "bench" / "plans"
SUBJECT = "PPO"
STAGES = ("connect", "first_answer", "boundary")
FLOOR = 27

FIRST_ANSWER_TRANSCRIPT = [
    Message(role="user", content=OPENING_TEXT),
    Message(
        role="assistant",
        content=(
            "We start with why one policy update can wreck a policy. Picture the expected "
            "reward against the size of the step. Before I show the curve, where do you think "
            "the reward goes as the step grows: up for ever, or up and then down?"
        ),
    ),
    Message(role="user", content="I guess it keeps going up as long as the advantage is positive."),
]
BOUNDARY_TRANSCRIPT = [
    *FIRST_ANSWER_TRANSCRIPT,
    Message(
        role="assistant",
        content=(
            "Not quite. Past a point the new policy lands where the old advantage estimates "
            "say nothing, and the reward falls off a cliff. Small steps climb; one huge step "
            "lands below where we started. Now the ratio: the new probability over the old."
        ),
    ),
    Message(role="user", content="so the ratio is like a measure of the step size"),
    Message(
        role="assistant",
        content=(
            "Yes, per action. When it drifts far from one the estimate stops meaning anything, "
            "and that is what clipping is for. Next we look at the clipped objective itself."
        ),
    ),
]


class PlanSample(BaseModel):
    plan_ms: int
    first_chunk_ms: int | None
    valid: bool
    accepted: bool | None
    touched_protected: bool | None
    appended: bool | None
    output_tokens: int | None
    cost_usd: float | None
    scenes: int | None
    steps: list[int]
    asking: int | None
    empty: bool
    prose_only: bool
    truncated: bool
    result: str


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SCENE_ID)
    domain: str = Field(default="", max_length=80)
    subject: str = Field(min_length=1, max_length=200)
    starting_from: str = Field(default="", max_length=400)
    folder: bool = False


class TopicFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sealed_at: str = ""
    topics: list[Topic]


def protected_for(stage: str, plan: LessonPlan) -> list[str]:
    ahead = {"first_answer": 1, "boundary": 2}[stage]
    return [scene.id for scene in plan.scenes[:ahead]]


def transcript_for(stage: str) -> list[Message]:
    return FIRST_ANSWER_TRANSCRIPT if stage == "first_answer" else BOUNDARY_TRANSCRIPT


async def one_plan(
    reasoning: MeteredReasoning,
    prompt: TurnPrompt,
    max_tokens: int,
    effort: str,
    model: str | None,
    current: LessonPlan | None,
    protected: list[str],
) -> tuple[PlanSample, LessonPlan | None]:
    record = reasoning.begin()
    t0 = time.perf_counter()
    outcome = await run_planner(reasoning, prompt, max_tokens, effort, model=model)
    plan_ms = int((time.perf_counter() - t0) * 1000)
    plan = outcome if isinstance(outcome, LessonPlan) else None
    result = "planner: sent" if plan is not None else outcome
    usage = record.planner_usage
    accepted = touched = appended = None
    if current is not None and plan is not None:
        spliced = splice_rerun(current, plan, len(protected))
        touched = rerun_conflict(current, plan, len(protected)) is not None
        accepted = isinstance(spliced, LessonPlan)
        if accepted:
            appended = len(spliced.scenes) > len(current.scenes)
    elif current is not None:
        accepted = False
    return (
        PlanSample(
            plan_ms=plan_ms,
            first_chunk_ms=record.planner_first_chunk_ms,
            valid=plan is not None,
            accepted=accepted,
            touched_protected=touched,
            appended=appended,
            output_tokens=None if usage is None else usage.completion_tokens,
            cost_usd=None if usage is None else turn_cost_usd(usage, list_price=True),
            scenes=None if plan is None else len(plan.scenes),
            steps=[] if plan is None else [len(scene.steps) for scene in plan.scenes],
            asking=(
                None
                if plan is None
                else sum(1 for scene in plan.scenes for step in scene.steps if step.ask)
            ),
            empty=result == EMPTY_REPLY,
            prose_only=result == NO_TOOL_CALL,
            truncated=is_truncated(result),
            result=result,
        ),
        plan,
    )


def plan_json(plan: LessonPlan) -> str:
    return json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n"


def write_plan(out: Path, label: str, n: int, plan: LessonPlan) -> Path:
    run = out / label
    run.mkdir(parents=True, exist_ok=True)
    path = run / f"plan-{n:02d}.json"
    path.write_text(plan_json(plan), encoding="ascii")
    return path


def load_topics(path: Path) -> list[Topic]:
    topics = TopicFile.model_validate_json(path.read_text()).topics
    ids = [topic.id for topic in topics]
    if not topics or len(set(ids)) != len(ids):
        raise ValueError("a topic list is non-empty and its ids are unique")
    return topics


async def run_topics(
    reasoning: MeteredReasoning,
    topics: list[Topic],
    out: Path,
    max_tokens: int,
    effort: str,
    model: str | None,
) -> list[tuple[Topic, PlanSample]]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(str(out))
    out.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Topic, PlanSample]] = []
    for topic in topics:
        prompt = plan_prompt(topic.subject, topic.starting_from, topic.folder, None, [], [])
        sample, plan = await one_plan(reasoning, prompt, max_tokens, effort, model, None, [])
        if plan is not None:
            (out / f"{topic.id}.json").write_text(plan_json(plan), encoding="ascii")
        else:
            error = sample.result.encode("ascii", "backslashreplace").decode("ascii")
            (out / f"{topic.id}.error.txt").write_text(error + "\n", encoding="ascii")
        rows.append((topic, sample))
    return rows


def verdict(plan_ms: int | None, good: int, n: int, stage: str) -> str:
    time_text = "none" if plan_ms is None else f"{plan_ms}ms"
    what = "valid" if stage == "connect" else "accepted"
    line = (
        f"verdict: median plan {time_text}, no budget yet, {what} {good}/{n} "
        f"against the floor of {FLOOR}"
    )
    return line if good >= FLOOR else line + ", FAIL"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--stage", choices=STAGES, default="connect")
    parser.add_argument("--plan", default=None)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--starting-from", default=STARTING_FROM)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--topics", default=None)
    return parser


def report(cfg: Settings, args: argparse.Namespace, samples: list[PlanSample], run: Path) -> None:
    print(
        "plan: from the request leaving run_planner to a validated plan, over every call; a "
        "call that returned an error string is in the plan line too. plan (valid) is the same "
        "span over the calls that returned a plan, and the verdict's median is that one. The "
        "session, the opening turn and the build are excluded. first chunk is the first delta "
        "of any kind.",
    )
    print(
        "validity: the call returned a plan the boundary accepted; at a rerun stage accepted "
        "means the reply yields a valid plan once the protected prefix is restored from the plan "
        "it was given, touched protected counts the replies whose own copy of the protected "
        "prefix differed from that plan, appended counts the accepted replies whose spliced plan "
        "is longer than the plan they were given. prose only, empty, truncated and other errors "
        "partition the failures. cost is the flash list price in tutor/cost.py applied to the "
        "tokens whatever the planner_model, so a non-flash run is not priced at its own rate.",
    )
    model = args.model or cfg.planner_model or cfg.reasoning_model
    print(
        f"model={cfg.reasoning_model}  planner_model={model}  effort={args.effort}  "
        f"stage={args.stage}  samples={args.samples} (plus {WARMUP} discarded warm-ups)  "
        f"planner_max_tokens={cfg.planner_max_tokens}  subject={args.subject!r}  "
        f"starting_from={args.starting_from!r}  plan={args.plan}  out={run}"
    )
    n = len(samples)
    valid = sum(1 for s in samples if s.valid)
    plans = [s for s in samples if s.valid]
    summarize("plan", [s.plan_ms for s in samples])
    summarize("plan (valid)", [s.plan_ms for s in samples if s.valid])
    summarize("first chunk", [s.first_chunk_ms for s in samples if s.first_chunk_ms is not None])
    summarize_series(
        "output tokens", [s.output_tokens for s in samples if s.output_tokens is not None], ""
    )
    costs = [s.cost_usd for s in samples if s.cost_usd is not None]
    summarize_usd("cost usd (list, flash rate)", costs)
    print(f"valid {valid}/{n}")
    print(f"truncated {sum(1 for s in samples if s.truncated)}/{n}")
    print(f"prose only {sum(1 for s in samples if s.prose_only)}/{n}")
    print(f"empty {sum(1 for s in samples if s.empty)}/{n}")
    other = sum(1 for s in samples if not (s.valid or s.truncated or s.prose_only or s.empty))
    print(f"other errors {other}/{n}")
    summarize_series("scenes per plan", [s.scenes for s in plans if s.scenes is not None], "")
    summarize_series("steps per scene", [k for s in plans for k in s.steps], "")
    asking = sum(s.asking or 0 for s in plans)
    print(f"asking steps {asking}/{sum(sum(s.steps) for s in plans)}")
    if args.stage != "connect":
        print(f"accepted {sum(1 for s in samples if s.accepted)}/{n}")
        print(f"touched protected {sum(1 for s in samples if s.touched_protected)}/{n}")
        print(f"appended {sum(1 for s in samples if s.appended)}/{n}")
    print(f"unknown cost {n - len(costs)}/{n}")
    times = [s.plan_ms for s in plans] or [s.plan_ms for s in samples]
    good = valid if args.stage == "connect" else sum(1 for s in samples if s.accepted)
    print(verdict(int(statistics.median(times)) if times else None, good, n, args.stage))


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No planner call can be reported. Populate .env and rerun.")
        return 2
    model = args.model or cfg.planner_model or None
    if args.topics is not None:
        if args.stage != "connect" or args.plan is not None:
            print("BLOCKED: --topics runs the connect stage only, with no --plan.")
            return 2
        try:
            topics = load_topics(Path(args.topics))
        except (OSError, ValueError) as exc:
            print(f"BLOCKED: the topic list does not load: {type(exc).__name__}.")
            return 2
        out = Path(args.out)
        reasoning = MeteredReasoning(ReasoningClient(cfg))
        try:
            rows = await run_topics(
                reasoning, topics, out, cfg.planner_max_tokens, args.effort, model
            )
        except FileExistsError:
            print(f"BLOCKED: {out} is not empty; a topic run writes into an empty directory.")
            return 2
        finally:
            await reasoning.aclose()
        print(
            f"model={cfg.reasoning_model}  planner_model={model or cfg.reasoning_model}  "
            f"effort={args.effort}  topics={len(topics)}  "
            f"planner_max_tokens={cfg.planner_max_tokens}  out={out}"
        )
        for topic, sample in rows:
            print(
                f"topic={topic.id} valid={sample.valid} plan_ms={sample.plan_ms} "
                f"tokens={sample.output_tokens} scenes={sample.scenes} steps={sample.steps} "
                f"result={sample.result}"
            )
        print(f"valid {sum(1 for _, sample in rows if sample.valid)}/{len(rows)}")
        return 0
    current = None
    protected: list[str] = []
    transcript: list[Message] = []
    if args.stage != "connect":
        if args.plan is None:
            print("BLOCKED: a rerun stage needs --plan, a plan file from a connect run.")
            return 2
        current = LessonPlan.model_validate_json(Path(args.plan).read_text())
        protected = protected_for(args.stage, current)
        transcript = transcript_for(args.stage)
    stamp = time.strftime("%Y%m%d-%H%M")
    label = f"{model or cfg.reasoning_model}-{args.effort}-{args.stage}-{stamp}"
    run = Path(args.out) / label
    reasoning = MeteredReasoning(ReasoningClient(cfg))
    samples: list[PlanSample] = []
    total = WARMUP + args.samples
    try:
        for n in range(total):
            prompt = plan_prompt(
                args.subject, args.starting_from, False, current, protected, transcript
            )
            sample, plan = await one_plan(
                reasoning, prompt, cfg.planner_max_tokens, args.effort, model, current, protected
            )
            if n >= WARMUP:
                samples.append(sample)
            if plan is not None:
                write_plan(Path(args.out), label, n + 1, plan)
            print(
                f"call={n + 1}/{total} plan_ms={sample.plan_ms} "
                f"first_chunk_ms={sample.first_chunk_ms} valid={sample.valid} "
                f"accepted={sample.accepted} tokens={sample.output_tokens} "
                f"scenes={sample.scenes} steps={sample.steps} asking={sample.asking} "
                f"result={sample.result}",
                file=sys.stderr,
            )
    finally:
        await reasoning.aclose()
    report(cfg, args, samples, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
