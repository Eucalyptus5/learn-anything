import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_spoken.py"
_spec = importlib.util.spec_from_file_location("bench_spoken", SCRIPT)
bench_spoken = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_spoken)

GLUE_TURNS = [
    {
        "turn_id": "turn-1",
        "tool_calls": 1,
        "chunks": [
            {"source": "lead_in", "text": "The search found acquire in src/pool.py."},
            {"source": "model", "text": "precisely.Diagram incoming for the pool"},
        ],
    },
    {
        "turn_id": "turn-2",
        "tool_calls": 1,
        "chunks": [
            {"source": "lead_in", "text": "The search found release in src/pool.py."},
            {"source": "model", "text": "the pool hands out one connection at a time."},
        ],
    },
    {
        "turn_id": "turn-3",
        "tool_calls": 0,
        "chunks": [
            {"source": "lead_in", "text": "The search found retries in src/pool.py."},
            {"source": "model", "text": "the retry loop sleeps between attempts."},
        ],
    },
    {
        "turn_id": "turn-4",
        "tool_calls": 0,
        "chunks": [
            {"source": "lead_in", "text": "The search found HttpClient in src/client.py."},
            {"source": "model", "text": "HttpClient.get borrows a connection and returns it."},
        ],
    },
]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the pool hands out one connection at a time.", set()),
        ("the `acquire` method", {"backtick"}),
        ('lifecycle.\n\n```json\n{"type":"diagram"', {"fence", "json"}),
        ('attempts.{"diagram":{"title"', {"json"}),
        ("about the failure path.\n</turn>", {"tag"}),
        ("the pool is a **free list**", {"bold"}),
        ("precisely.Diagram incoming", {"glued"}),
        ("Your answer, engineer?We're in Explore", {"glued"}),
        ("HttpClient.get", set()),
    ],
)
def test_classes_over_the_nine_chunk_table(text: str, expected: set[str]) -> None:
    assert bench_spoken.classes(text) == expected


def test_glue_control_over_the_four_turn_table() -> None:
    assert bench_spoken.glue_control(GLUE_TURNS) == (2, 1, 2, 0)
