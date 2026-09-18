"""Scene build time and validity through the real reasoning model: one fixed PPO brief goes
through ``run_scene_build`` at the configured effort and budget, the harness records the wall
time to a draft, the first chunk, the output tokens and the list-price cost per call, whether
the draft loads the helper and gsap and how many steps it names, and writes every document to a
directory for the review page. Text only on the wire; no session, no browser, no check.
"""

import argparse
import asyncio
import json
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
    summarize_series,
    summarize_usd,
)
from tutor.brief import VisualBrief
from tutor.config import Settings, settings
from tutor.cost import turn_cost_usd
from tutor.reasoning import ReasoningClient
from tutor.scene import NO_TOOL_CALL, SceneDraft, run_scene_build, scene_prompt

VALID_FLOOR = 27
SUBJECT = "PPO"
LESSON_TAG = '<script src="/lesson.js">'
GSAP_TAG = '<script src="/vendor/gsap.min.js">'
OUT_DIR = REPO / "scratch" / "bench" / "scenes"
BRIEF = VisualBrief(
    kind="app",
    title="Clipped objective",
    show=(
        "the clipped surrogate objective against the probability ratio from 0.5 to 1.5 for "
        "advantage plus one and minus one, epsilon 0.2, the flat regions visible"
    ),
)


class BuildSample(BaseModel):
    build_ms: int
    first_chunk_ms: int | None
    valid: bool
    truncated: bool
    prose_only: bool
    output_tokens: int | None
    cost_usd: float | None
    steps: int | None
    uses_helper: bool | None
    uses_gsap: bool | None
    result: str


async def one_build(
    reasoning: MeteredReasoning,
    brief: VisualBrief,
    max_tokens: int,
    effort: str,
    model: str | None = None,
) -> tuple[BuildSample, SceneDraft | None]:
    record = reasoning.begin()
    t0 = time.perf_counter()
    outcome = await run_scene_build(
        reasoning, scene_prompt(SUBJECT, brief, "light"), max_tokens, effort, model=model
    )
    build_ms = int((time.perf_counter() - t0) * 1000)
    draft = outcome if isinstance(outcome, SceneDraft) else None
    result = "scene: sent" if draft is not None else outcome
    usage = record.visual_usage
    sample = BuildSample(
        build_ms=build_ms,
        first_chunk_ms=record.visual_first_chunk_ms,
        valid=draft is not None,
        truncated=is_truncated(result),
        prose_only=result == NO_TOOL_CALL,
        output_tokens=None if usage is None else usage.completion_tokens,
        cost_usd=None if usage is None else turn_cost_usd(usage, list_price=True),
        steps=None if draft is None else len(draft.steps),
        uses_helper=None if draft is None else LESSON_TAG in draft.html,
        uses_gsap=None if draft is None else GSAP_TAG in draft.html,
        result=result,
    )
    return sample, draft


def write_scene(out: Path, label: str, n: int, draft: SceneDraft) -> Path:
    run = out / label
    run.mkdir(parents=True, exist_ok=True)
    path = run / f"{n:02d}.html"
    path.write_text(draft.html)
    (run / f"{n:02d}.json").write_text(json.dumps({"steps": draft.steps}))
    return path


def verdict(build_ms: int | None, valid: int, n: int, prose_only: int) -> str:
    build = "none" if build_ms is None else f"{build_ms}ms"
    line = (
        f"verdict: median build {build}, no budget yet, valid {valid}/{n} against the floor "
        f"of {VALID_FLOOR}, prose only {prose_only}/{n}"
    )
    return f"{line}, FAIL" if valid < VALID_FLOOR else line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--model", default="")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--out", default=str(OUT_DIR))
    return parser


def report(cfg: Settings, args: argparse.Namespace, samples: list[BuildSample], run: Path) -> None:
    print(
        "build: from the request leaving run_scene_build to a validated draft, over every call; "
        "a call that returned an error string is in the build line too, since a fast failure "
        "is still a wall time the session waits. The session, the page and the check are "
        "excluded; the check pass rate is read on /scene_review.html from the files under out.",
    )
    print(
        "validity: the call returned a draft, three to five steps and a document; helper and "
        "gsap count the drafts whose document carries the two required tags; prose only counts "
        "the streams that ended with no tool call; other errors counts the rest, an unexpected "
        "tool, arguments that were not JSON or a draft the boundary rejected. cost is the list "
        "price at the flash rate in tutor/cost.py applied to the tokens; the cost line is the "
        "flash list price applied to the tokens whatever the scene_model, so a non-flash run is "
        "not priced at its own rate.",
    )
    model = args.model or cfg.scene_model or cfg.reasoning_model
    print(
        f"model={cfg.reasoning_model}  scene_model={model}  effort={args.effort}  "
        f"samples={args.samples} (plus {WARMUP} discarded warm-ups)  "
        f"scene_max_tokens={cfg.scene_max_tokens}  theme=light  brief={BRIEF.title}  out={run}"
    )
    valid = sum(1 for s in samples if s.valid)
    prose = sum(1 for s in samples if s.prose_only)
    drafts = [s for s in samples if s.valid]
    summarize("build", [s.build_ms for s in samples])
    summarize("first chunk", [s.first_chunk_ms for s in samples if s.first_chunk_ms is not None])
    summarize_series(
        "output tokens", [s.output_tokens for s in samples if s.output_tokens is not None], ""
    )
    costs = [s.cost_usd for s in samples if s.cost_usd is not None]
    summarize_usd("cost usd (list, flash rate)", costs)
    print(f"valid {valid}/{len(samples)}")
    print(f"truncated {sum(1 for s in samples if s.truncated)}/{len(samples)}")
    print(f"prose only {prose}/{len(samples)}")
    other = sum(1 for s in samples if not (s.valid or s.truncated or s.prose_only))
    print(f"other errors {other}/{len(samples)}")
    print(f"helper {sum(1 for s in drafts if s.uses_helper)}/{len(drafts)}")
    print(f"gsap {sum(1 for s in drafts if s.uses_gsap)}/{len(drafts)}")
    summarize_series("steps", [s.steps for s in drafts if s.steps is not None], "")
    print(f"unknown cost {len(samples) - len(costs)}/{len(samples)}")
    builds = [s.build_ms for s in samples]
    print(verdict(int(statistics.median(builds)) if builds else None, valid, len(samples), prose))


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = settings()
    except ValidationError:
        print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
        print("No scene build can be reported. Populate .env and rerun.")
        return 2
    model = args.model or cfg.scene_model or None
    stamp = time.strftime("%Y%m%d-%H%M")
    label = f"{model or cfg.reasoning_model}-{args.effort}-{stamp}"
    run = Path(args.out) / label
    reasoning = MeteredReasoning(ReasoningClient(cfg))
    samples: list[BuildSample] = []
    total = WARMUP + args.samples
    try:
        for n in range(total):
            sample, draft = await one_build(
                reasoning, BRIEF, cfg.scene_max_tokens, args.effort, model
            )
            if n >= WARMUP:
                samples.append(sample)
            if draft is not None:
                write_scene(Path(args.out), label, n + 1, draft)
            print(
                f"call={n + 1}/{total} build_ms={sample.build_ms} "
                f"first_chunk_ms={sample.first_chunk_ms} valid={sample.valid} "
                f"truncated={sample.truncated} tokens={sample.output_tokens} "
                f"steps={sample.steps} helper={sample.uses_helper} gsap={sample.uses_gsap} "
                f"result={sample.result}",
                file=sys.stderr,
            )
    finally:
        await reasoning.aclose()
    report(cfg, args, samples, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
