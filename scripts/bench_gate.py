"""What a fail-closed provenance gate would cost in withheld speech. Runs three candidate
policies over two corpora that are reported separately and never pooled: the lead-in sentences
the tutor emits without a model, and clauses captured from real reasoning turns. Reports the
withheld fraction per policy per source, plus the unbound digit and path-shaped tokens behind it.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

import openai
from pydantic import ValidationError

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.config import Settings, settings
from tutor.cost import UsageLedger
from tutor.lead_in import lead_in_sentence
from tutor.prompt import SEARCH_CODE_TOOL, SYSTEM_PROMPT, Message, TurnPrompt
from tutor.reasoning import ReasoningClient, TurnChunk
from tutor.session import BAD_ARGUMENTS, SEARCH_CODE, _assistant_calls, _search_arguments
from tutor.tools.models import SearchBudget, SearchResult
from tutor.tools.provenance import TurnRegistry, extract_positions
from tutor.tools.search import search

POLICIES = ("current", "digits", "pathshape")
GLOBS = ["**/*.py"]
SOURCES = ("lead_in", "model")
SUBJECT = "a small http client with a bounded connection pool"
RETRY_BACKOFF_S = 20
MAX_ATTEMPTS = 5
TOP_TOKENS = 10

LEADING = "`\"'([{"
TRAILING = "`\"')]}.,;:!?"
DIGIT_TOKEN = re.compile(r"^[#@L]?(\d+)$")
LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
PATH_SHAPED = re.compile(r"^[\w.-]+\.[A-Za-z0-9]{1,6}$")

UTTERANCES = (
    "acquire",
    "release",
    "class ConnectionPool",
    "def acquire",
    "def release",
    "acquire_or_wait",
    "with_connection",
    "acquire_batch",
    "send_request",
    "request_count",
    "in_use",
    "available",
    "retries",
    "pool exhausted",
    "RuntimeError",
    "DEFAULT_TIMEOUT_S",
    "acquire_timeout_s",
    "SMALL_MODULE_VERSION",
    "import",
    "Release Notes",
    "leak",
    "socket",
    "stats",
    "HttpClient",
    "walk me through how the http client gets a connection",
    "what happens when the pool is exhausted",
    "quiz me on the release path",
    "where is the retry loop and what does it retry",
    "explain pool_helpers.py to me",
    "teach me the pooling design in python",
)


def _strip(token: str) -> str:
    return token.lstrip(LEADING).rstrip(TRAILING)


def _tokens(chunk: str) -> tuple[list[tuple[str, int]], list[str]]:
    digits: list[tuple[str, int]] = []
    shapes: list[str] = []
    for raw in chunk.split():
        token = _strip(raw)
        found = DIGIT_TOKEN.match(token)
        if found:
            digits.append((token, int(found.group(1))))
            continue
        head = LINE_SUFFIX.sub("", token)
        if PATH_SHAPED.match(head) or "/" in head:
            shapes.append(head)
    return digits, shapes


def _bound(chunks: list[str], index: int) -> tuple[set[int], set[str]]:
    previous = chunks[index - 1] if index else ""
    positions = extract_positions(f"{previous} {chunks[index]}")
    return (
        {position.line for position in positions if position.line is not None},
        {position.path for position in positions},
    )


def classify(
    chunks: list[str], registry: TurnRegistry, turn_id: str, source: str, policy: str
) -> list[str]:
    verdicts: list[str] = []
    for index, chunk in enumerate(chunks):
        verdict = registry.verify_chunk(turn_id, chunk, source=source)
        if not verdict.ok:
            verdicts.append("withheld_bound")
            continue
        if policy == "current":
            verdicts.append("admitted")
            continue
        lines, paths = _bound(chunks, index)
        digits, shapes = _tokens(chunk)
        if any(value not in lines for _, value in digits):
            verdicts.append("withheld_digit")
        elif policy == "pathshape" and any(head not in paths for head in shapes):
            verdicts.append("withheld_pathshape")
        else:
            verdicts.append("admitted")
    return verdicts


def unbound_tokens(chunks: list[str]) -> tuple[Counter[str], Counter[str]]:
    unbound_digits: Counter[str] = Counter()
    unbound_shapes: Counter[str] = Counter()
    for index, chunk in enumerate(chunks):
        lines, paths = _bound(chunks, index)
        digits, shapes = _tokens(chunk)
        unbound_digits.update(token for token, value in digits if value not in lines)
        unbound_shapes.update(head for head in shapes if head not in paths)
    return unbound_digits, unbound_shapes


def report_policy(source: str, policy: str, verdicts: list[str]) -> None:
    counts = Counter(verdicts)
    total = len(verdicts)
    withheld = total - counts["admitted"]
    fraction = withheld / total if total else 0.0
    print(
        f"source={source} policy={policy} n={total} withheld={withheld} "
        f"fraction={fraction:.3f} bound={counts['withheld_bound']} "
        f"digit={counts['withheld_digit']} pathshape={counts['withheld_pathshape']}"
    )


def report_tokens(source: str, kind: str, counts: Counter[str]) -> None:
    top = " ".join(f"{token}:{count}" for token, count in counts.most_common(TOP_TOKENS))
    print(
        f"tokens source={source} {kind} distinct={len(counts)} "
        f"occurrences={sum(counts.values())} top={top}"
    )


async def deterministic_sentences(root: Path) -> list[tuple[str, SearchResult]]:
    budget = SearchBudget()
    sentences: list[tuple[str, SearchResult]] = []
    for text in UTTERANCES:
        result = await search(text, GLOBS, root, budget)
        sentences.append((lead_in_sentence([result]), result))
    return sentences


def report_deterministic(sentences: list[tuple[str, SearchResult]]) -> None:
    print(f"corpus=deterministic searches={len(sentences)}")
    for policy in POLICIES:
        verdicts: list[str] = []
        for index, (sentence, result) in enumerate(sentences):
            turn_id = f"lead-in-{index + 1}"
            registry = TurnRegistry()
            registry.open_turn(turn_id)
            registry.record(turn_id, result)
            verdicts.extend(classify([sentence], registry, turn_id, "lead_in", policy))
        report_policy("lead_in", policy, verdicts)
    digits: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    for sentence, _ in sentences:
        turn_digits, turn_shapes = unbound_tokens([sentence])
        digits.update(turn_digits)
        shapes.update(turn_shapes)
    report_tokens("lead_in", "unbound_digit", digits)
    report_tokens("lead_in", "unbound_pathshape", shapes)


def report_corpus(path: Path, corpus: dict) -> None:
    turns = corpus["turns"]
    texts: dict[str, list[list[str]]] = {source: [] for source in SOURCES}
    verdicts: dict[tuple[str, str], list[str]] = {
        (source, policy): [] for source in SOURCES for policy in POLICIES
    }
    for turn in turns:
        results = [SearchResult.model_validate(raw) for raw in turn["results"]]
        chunks = {
            source: [chunk["text"] for chunk in turn["chunks"] if chunk["source"] == source]
            for source in SOURCES
        }
        for source in SOURCES:
            texts[source].append(chunks[source])
        for policy in POLICIES:
            registry = TurnRegistry()
            registry.open_turn(turn["turn_id"])
            for result in results:
                registry.record(turn["turn_id"], result)
            for source in SOURCES:
                verdicts[(source, policy)].extend(
                    classify(chunks[source], registry, turn["turn_id"], source, policy)
                )

    print(
        f"corpus=model file={path} turns={len(turns)} "
        f"dropped_turns={corpus['dropped_turns']} "
        f"tool_calls={sum(turn['tool_calls'] for turn in turns)} "
        f"retries={sum(turn['retries'] for turn in turns)} "
        f"lead_in_chunks={sum(len(chunks) for chunks in texts['lead_in'])} "
        f"model_chunks={sum(len(chunks) for chunks in texts['model'])}"
    )
    for source in SOURCES:
        for policy in POLICIES:
            report_policy(source, policy, verdicts[(source, policy)])
    for source in SOURCES:
        digits: Counter[str] = Counter()
        shapes: Counter[str] = Counter()
        for chunks in texts[source]:
            turn_digits, turn_shapes = unbound_tokens(chunks)
            digits.update(turn_digits)
            shapes.update(turn_shapes)
        report_tokens(source, "unbound_digit", digits)
        report_tokens(source, "unbound_pathshape", shapes)
    usage = corpus["usage"]
    print(
        f"usage turns={usage['turns']} prompt_tokens={usage['prompt_tokens']} "
        f"completion_tokens={usage['completion_tokens']} cached_tokens={usage['cached_tokens']} "
        f"reasoning_chars={usage['reasoning_chars']} cost_usd={usage['cost_usd']:.6f} "
        f"list_cost_usd={usage['list_cost_usd']:.6f}"
    )


async def _replay(deltas: list[str]) -> AsyncIterator[str]:
    for delta in deltas:
        yield delta


async def _stream(
    client: ReasoningClient, prompt: TurnPrompt, tools: list[dict] | None, ledger: UsageLedger
) -> tuple[list[str], list[TurnChunk]]:
    deltas: list[str] = []
    calls: list[TurnChunk] = []
    stream = client.start_turn(prompt, tools=tools)
    async for chunk in stream:
        if chunk.kind == "spoken":
            deltas.append(chunk.text)
        elif chunk.kind == "tool_call":
            calls.append(chunk)
    if stream.usage is not None:
        ledger.add(stream.usage)
    return deltas, calls


async def _answer(
    registry: TurnRegistry,
    turn_id: str,
    call: TurnChunk,
    results: list[SearchResult],
    root: Path,
    budget: SearchBudget,
) -> str:
    if call.tool_name != SEARCH_CODE:
        return f"{call.tool_name} is not available in this turn"
    arguments = _search_arguments(call.text)
    if arguments is None:
        return BAD_ARGUMENTS
    query, globs = arguments
    result = await search(query, globs, root, budget)
    registry.record(turn_id, result)
    results.append(result)
    return result.model_dump_json()


async def capture_turn(
    client: ReasoningClient,
    registry: TurnRegistry,
    turn_id: str,
    user_text: str,
    root: Path,
    system: str,
    ledger: UsageLedger,
) -> dict | None:
    budget = SearchBudget()
    for attempt in range(MAX_ATTEMPTS):
        registry.open_turn(turn_id)
        result = await search(user_text, GLOBS, root, budget)
        registry.record(turn_id, result)
        results = [result]
        prompt = TurnPrompt(system=system, tool_context=[result], user_text=user_text)
        lead_in = lead_in_sentence([result])
        try:
            deltas, calls = await _stream(client, prompt, [SEARCH_CODE_TOOL], ledger)
            answerable = [call for call in calls if call.tool_call_id and call.tool_name]
            if answerable:
                exchange = [_assistant_calls(answerable)]
                for call in answerable:
                    answer = await _answer(registry, turn_id, call, results, root, budget)
                    exchange.append(
                        Message(role="tool", content=answer, tool_call_id=call.tool_call_id)
                    )
                follow_up = prompt.model_copy(update={"tool_exchange": exchange})
                more, _ = await _stream(client, follow_up, None, ledger)
                deltas.append("\n")
                deltas.extend(more)
        except (openai.RateLimitError, openai.APIConnectionError):
            if attempt + 1 < MAX_ATTEMPTS:
                await asyncio.sleep(RETRY_BACKOFF_S)
            continue
        chunks = [{"source": "lead_in", "text": lead_in}]
        async for clause in clause_chunks(spoken_text(_replay(deltas), Scrubber())):
            chunks.append({"source": "model", "text": clause})
        return {
            "turn_id": turn_id,
            "user_text": user_text,
            "results": [item.model_dump() for item in results],
            "chunks": chunks,
            "tool_calls": len(calls),
            "retries": attempt,
        }
    return None


async def capture(
    client: ReasoningClient, cfg: Settings, args: argparse.Namespace, system: str
) -> None:
    ledger = UsageLedger()
    registry = TurnRegistry()
    turns: list[dict] = []
    dropped = 0
    for index in range(args.turns):
        started = time.perf_counter()
        turn_id = f"turn-{index + 1}"
        user_text = UTTERANCES[index % len(UTTERANCES)]
        turn = await capture_turn(client, registry, turn_id, user_text, args.root, system, ledger)
        if turn is None:
            dropped += 1
        else:
            turns.append(turn)
        corpus = {
            "model": cfg.reasoning_model,
            "effort": cfg.reasoning_effort,
            "max_tokens": cfg.reasoning_max_tokens,
            "root": str(args.root),
            "dropped_turns": dropped,
            "turns": turns,
            "usage": {
                "turns": ledger.turns,
                "prompt_tokens": ledger.total.prompt_tokens,
                "completion_tokens": ledger.total.completion_tokens,
                "cached_tokens": ledger.total.cached_tokens,
                "reasoning_chars": ledger.total.reasoning_chars,
                "cost_usd": ledger.cost_usd(),
                "list_cost_usd": ledger.cost_usd(list_price=True),
            },
        }
        args.corpus.parent.mkdir(parents=True, exist_ok=True)
        args.corpus.write_text(json.dumps(corpus, indent=1))
        spoken = sum(1 for chunk in turn["chunks"] if chunk["source"] == "model") if turn else 0
        print(
            f"turn={index + 1}/{args.turns} model_chunks={spoken} "
            f"tool_calls={turn['tool_calls'] if turn else 0} "
            f"retries={turn['retries'] if turn else MAX_ATTEMPTS} "
            f"ms={int((time.perf_counter() - started) * 1000)}",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=REPO / "captures" / "gate_corpus.json")
    parser.add_argument("--root", type=Path, default=REPO / "tests" / "data" / "fixture_repo")
    return parser


async def main() -> int:
    args = build_parser().parse_args()

    report_deterministic(await deterministic_sentences(args.root))

    if args.turns > 0:
        try:
            cfg = settings()
        except ValidationError:
            print("BLOCKED: REASONING_API_BASE or REASONING_API_KEY is empty in .env.")
            print("No model corpus can be captured. Populate .env and rerun.")
            return 2
        system = f"{SYSTEM_PROMPT}\n\nSubject: {SUBJECT}"
        client = ReasoningClient(cfg)
        try:
            await capture(client, cfg, args, system)
        finally:
            await client.aclose()

    if args.corpus.exists():
        report_corpus(args.corpus, json.loads(args.corpus.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
