import importlib.util
from pathlib import Path

from tutor.prompt import Message, TurnPrompt
from tutor.session import OutcomeSplitter
from tutor.visuals import DiagramPush

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_visual_lead.py"
_spec = importlib.util.spec_from_file_location("bench_visual_lead", SCRIPT)
bench_visual_lead = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_visual_lead)

Enqueued = bench_visual_lead.Enqueued


def _prompt(exchange: list[Message]) -> TurnPrompt:
    return TurnPrompt(system="s", user_text="walk me through it", tool_exchange=exchange)


async def test_the_first_round_is_one_push_diagram_call() -> None:
    stream = bench_visual_lead.ScriptedReasoning().start_turn(_prompt([]), tools=[], max_tokens=1)

    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 1
    assert chunks[0].kind == "tool_call"
    assert chunks[0].tool_name == "push_diagram"
    assert chunks[0].tool_call_id
    push = DiagramPush.model_validate_json(chunks[0].text)
    assert push.kind == "flowchart"
    assert push.source == bench_visual_lead.FLOWCHART


async def test_the_follow_up_round_is_speech_and_an_outcome() -> None:
    exchange = [Message(role="tool", content="push_diagram: sent", tool_call_id="call-lead")]
    stream = bench_visual_lead.ScriptedReasoning().start_turn(_prompt(exchange))

    chunks = [chunk async for chunk in stream]

    assert chunks
    assert all(chunk.kind == "spoken" for chunk in chunks)
    joined = "".join(chunk.text for chunk in chunks)
    assert joined.endswith(bench_visual_lead.FOLLOW_UP_DELTAS[-1])
    splitter = OutcomeSplitter()
    spoken = "".join(splitter.feed(chunk.text) for chunk in chunks)
    tail, outcome = splitter.finish()
    assert outcome.signal == "covered"
    assert spoken + tail == bench_visual_lead.EXPLANATION


def test_the_explanation_index_skips_openers_and_the_lead_in() -> None:
    explanation = bench_visual_lead.EXPLANATION
    played = [
        Enqueued(1.0, None, 4800),
        Enqueued(2.0, "The match is in src/pool.py line 12.", 9600),
        Enqueued(3.0, "The reader pulls frames off the track,", 7200),
        Enqueued(4.0, "hands each one to the queue,", 7200),
    ]

    assert bench_visual_lead.explanation_index(played, explanation) == 2
    assert bench_visual_lead.explanation_index(played[:2], explanation) is None
    assert bench_visual_lead.explanation_index([], explanation) is None


def test_lead_is_negative_when_the_voice_precedes_the_push() -> None:
    assert bench_visual_lead.lead_ms(2.0, 1.5) == -500
    assert bench_visual_lead.lead_ms(1.0, 1.25) == 250


def test_the_verdict_fails_under_the_budget_and_clears_above_it() -> None:
    budget = bench_visual_lead.BUDGET_MS

    assert bench_visual_lead.verdict([]).endswith(", FAIL")
    assert bench_visual_lead.verdict([budget - 1, budget - 50, budget + 10]).endswith(", FAIL")
    assert not bench_visual_lead.verdict([budget, budget, budget]).endswith(", FAIL")
    assert not bench_visual_lead.verdict([budget * 3]).endswith(", FAIL")
    assert str(budget) in bench_visual_lead.verdict([budget])


def test_parser_defaults() -> None:
    args = bench_visual_lead.build_parser().parse_args([])

    assert args.samples == 30
    assert args.root.parts[-3:] == ("tests", "data", "fixture_repo")
