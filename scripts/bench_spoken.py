"""What markup reaches the speech synthesizer. Classifies every model-authored chunk in the
captured corpus by the markup it carries, separates glued sentences from dotted names with a
tool-call control, and with --synth measures whether Kokoro voices each marker or drops it.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.constants import TTS_SAMPLE_RATE
from tutor.tts import KokoroSynthesizer

CLASSES = ("fence", "json", "tag", "backtick", "bold", "glued")
UNION = ("fence", "json", "tag", "backtick", "bold")
DROPPED = ("fence", "json", "json_unbalanced", "tag", "backtick", "bold")
FENCE = "```"
JSON_OPEN = re.compile(r'\{\s*"')
TAG = re.compile(r"</?[a-z]+>")
BOLD = "**"
GLUED = re.compile(r"[a-z][.?!][A-Z]")
WEIGHTS = REPO / "models" / "kokoro" / "kokoro-v1.0.fp16.onnx"
VOICES = REPO / "models" / "kokoro" / "voices-v1.0.bin"
SYNTH_RUNS = 3
DELTA_WIDTH = 4

PAIRS = (
    ("the acquire method returns a connection", "the `acquire` method returns a connection"),
    ("call acquire and release in that order", "call acquire() and release() in that order"),
    ("the pool is a free list", "the pool is a **free list**"),
    ("it is a free list semaphore", "it is a free list semaphore</diagram>"),
    ("precisely. Diagram incoming for the pool", "precisely.Diagram incoming for the pool"),
    ("idle stack then pop then in use set", "idle stack \u2192 pop \u2192 in_use set"),
)


def classes(text: str) -> set[str]:
    found: set[str] = set()
    if FENCE in text:
        found.add("fence")
    if "`" in text.replace(FENCE, ""):
        found.add("backtick")
    if JSON_OPEN.search(text):
        found.add("json")
    if TAG.search(text):
        found.add("tag")
    if BOLD in text:
        found.add("bold")
    if GLUED.search(text):
        found.add("glued")
    return found


def _model_texts(turn: dict) -> list[str]:
    return [chunk["text"] for chunk in turn["chunks"] if chunk["source"] == "model"]


def glue_control(turns: list[dict]) -> tuple[int, int, int, int]:
    with_calls = glued_with = without = glued_without = 0
    for turn in turns:
        glued = any("glued" in classes(text) for text in _model_texts(turn))
        if turn["tool_calls"] > 0:
            with_calls += 1
            glued_with += glued
        else:
            without += 1
            glued_without += glued
    return with_calls, glued_with, without, glued_without


def report_corpus(path: Path, corpus: dict) -> None:
    turns = corpus["turns"]
    labelled = [[classes(text) for text in _model_texts(turn)] for turn in turns]
    print(
        f"corpus=model file={path} turns={len(turns)} "
        f"model_chunks={sum(len(turn) for turn in labelled)}"
    )
    for name in CLASSES:
        chunk_count = sum(name in found for turn in labelled for found in turn)
        turn_count = sum(any(name in found for found in turn) for turn in labelled)
        print(f"class={name} chunks={chunk_count} turns={turn_count}")
    with_calls, glued_with, without, glued_without = glue_control(turns)
    print(
        f"glued turns: with_tool_calls={with_calls} glued={glued_with} "
        f"without={without} glued={glued_without}"
    )
    union_chunks = sum(any(name in found for name in UNION) for turn in labelled for found in turn)
    union_turns = sum(any(name in found for name in UNION for found in turn) for turn in labelled)
    print(f"union chunks={union_chunks} turns={union_turns}")


def cut(text: str, width: int = DELTA_WIDTH) -> list[str]:
    return [text[i : i + width] for i in range(0, len(text), width)]


async def _stream(deltas: list[str]) -> AsyncIterator[str]:
    for delta in deltas:
        yield delta


async def _replay(turns: list[dict]) -> tuple[int, int, Counter[str], int, Counter[str]]:
    chunks_in = chunks_out = dropped_turns = 0
    dropped_chars: Counter[str] = Counter()
    remaining: Counter[str] = Counter()
    for turn in turns:
        texts = _model_texts(turn)
        scrubber = Scrubber()
        deltas = cut(" ".join(texts))
        clauses = [c async for c in clause_chunks(spoken_text(_stream(deltas), scrubber))]
        chunks_in += len(texts)
        chunks_out += len(clauses)
        dropped_chars.update(scrubber.dropped_chars)
        dropped_turns += bool(scrubber.dropped)
        for clause in clauses:
            remaining.update(classes(clause))
    return chunks_in, chunks_out, dropped_chars, dropped_turns, remaining


def replay(turns: list[dict]) -> tuple[int, int, Counter[str], int, Counter[str]]:
    return asyncio.run(_replay(turns))


def report_replay(turns: list[dict]) -> None:
    chunks_in, chunks_out, dropped_chars, dropped_turns, remaining = replay(turns)
    print(f"replay turns={len(turns)} chunks_in={chunks_in} chunks_out={chunks_out}")
    for name in DROPPED:
        print(f"dropped class={name} chars={dropped_chars[name]}")
    print(f"dropped turns={dropped_turns}")
    for name in CLASSES:
        print(f"remaining class={name} chunks={remaining[name]}")


def _synth_median(synthesizer: KokoroSynthesizer, text: str) -> int:
    runs = [synthesizer.synthesize(text) for _ in range(SYNTH_RUNS)]
    lengths = [len(audio) * 1000 // TTS_SAMPLE_RATE for audio in runs]
    median = int(statistics.median(lengths))
    identical = all(np.array_equal(runs[0], audio) for audio in runs[1:])
    print(f"synth text={text!r} ms={lengths} median_ms={median} identical={identical}")
    return median


def report_synth() -> None:
    synthesizer = KokoroSynthesizer(WEIGHTS, VOICES)
    print(f"synth weights={WEIGHTS} runs={SYNTH_RUNS} sample_rate={TTS_SAMPLE_RATE}")
    for clean, marked in PAIRS:
        clean_ms = _synth_median(synthesizer, clean)
        marked_ms = _synth_median(synthesizer, marked)
        print(f"pair clean_ms={clean_ms} marked_ms={marked_ms} diff_ms={marked_ms - clean_ms}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=REPO / "captures" / "gate_corpus.json")
    parser.add_argument("--synth", action="store_true")
    parser.add_argument("--replay", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.corpus.exists():
        corpus = json.loads(args.corpus.read_text())
        report_corpus(args.corpus, corpus)
        if args.replay:
            report_replay(corpus["turns"])
    if args.synth:
        report_synth()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
