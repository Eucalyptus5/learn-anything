"""Wall-clock latency of tutor.tools.search.search() from the caller's point of view: one
perf_counter span per call, three warm-ups discarded, median and p95 over the rest.
"""

import argparse
import asyncio
import statistics
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tutor.tools.models import SearchBudget, SearchResult
from tutor.tools.search import search

QUERY = "acquire"
GLOBS = ["**/*"]
WARMUPS = 3


async def time_runs(run: Callable[[], Awaitable[object]], samples: int) -> list[float]:
    timings: list[float] = []
    for index in range(WARMUPS + samples):
        started = time.perf_counter()
        await run()
        elapsed = time.perf_counter() - started
        if index >= WARMUPS:
            timings.append(elapsed)
    return timings


def summarize(seconds: list[float]) -> tuple[int, int]:
    ms = sorted(round(v * 1000) for v in seconds)
    median = int(statistics.median(ms))
    p95 = ms[max(0, int(len(ms) * 0.95) - 1)]
    return median, p95


def file_count(root: Path) -> int:
    result = subprocess.run(
        ["rg", "--files", "--no-require-git"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return len(result.stdout.splitlines())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    return parser


async def main() -> int:
    args = build_parser().parse_args()

    print(f"root={args.root} files={file_count(args.root)}")

    budget = SearchBudget()
    results: list[SearchResult] = []

    async def run() -> None:
        results.append(await search(QUERY, GLOBS, args.root, budget))

    seconds = await time_runs(run, args.samples)

    median, p95 = summarize(seconds)
    ms = sorted(round(v * 1000) for v in seconds)
    last = results[-1]
    print(
        f"search {QUERY!r} globs={GLOBS} n={len(seconds)} median={median}ms p95={p95}ms "
        f"min={ms[0]}ms max={ms[-1]}ms matches={len(last.matches)} truncated={last.truncated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
