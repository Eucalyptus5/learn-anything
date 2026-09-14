import importlib.util
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from tutor.brief import VisualBrief
from tutor.config import Settings
from tutor.cost import TurnUsage
from tutor.prompt import TurnPrompt
from tutor.reasoning import TurnChunk

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_visual_call.py"
_spec = importlib.util.spec_from_file_location("bench_visual_call", SCRIPT)
bench_visual_call = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_visual_call)

DIAGRAM_ARGUMENTS = json.dumps(
    {
        "id": "loop",
        "kind": "flowchart",
        "source": "flowchart TD\n  a[collect rollouts] --> b[estimate advantages]",
        "title": "PPO update loop",
    }
)


class ScriptedStream:
    def __init__(
        self,
        chunks: list[TurnChunk],
        usage: TurnUsage | None,
        finish_reason: str | None,
        first_chunk_ms: int | None,
    ) -> None:
        self._chunks = chunks
        self.usage = usage
        self.finish_reason = finish_reason
        self.first_chunk_ms = first_chunk_ms

    async def __aiter__(self) -> AsyncIterator[TurnChunk]:
        for chunk in self._chunks:
            yield chunk


class ScriptedReasoning:
    def __init__(self, stream: ScriptedStream) -> None:
        self._stream = stream
        self.tool_choices: list[str | None] = []

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
    ) -> ScriptedStream:
        self.tool_choices.append(tool_choice)
        return self._stream

    async def aclose(self) -> None:
        return None


def test_the_two_briefs_validate() -> None:
    diagram = bench_visual_call.DIAGRAM_BRIEF
    app = bench_visual_call.APP_BRIEF

    assert diagram.kind == "diagram"
    assert app.kind == "app"
    assert VisualBrief.model_validate(diagram.model_dump()) == diagram
    assert VisualBrief.model_validate(app.model_dump()) == app


def test_a_sent_result_counts_as_valid_and_an_error_does_not() -> None:
    is_valid = bench_visual_call.is_valid
    is_truncated = bench_visual_call.is_truncated

    assert is_valid("push_diagram: sent")
    assert is_valid("push_app: sent")
    assert not is_valid("push_app: error: html: too long")
    assert not is_valid("visual: error: no tool call")
    assert not is_valid("visual: error: truncated at 3000 tokens")
    assert is_truncated("visual: error: truncated at 3000 tokens")
    assert not is_truncated("push_app: sent")


def test_the_verdict_fails_under_the_validity_floor() -> None:
    verdict = bench_visual_call.verdict
    app_budget = bench_visual_call.APP_BUDGET_MS
    diagram_budget = bench_visual_call.DIAGRAM_BUDGET_MS

    assert verdict("app", 50000, valid=26, n=30).endswith(", FAIL")
    assert not verdict("app", 50000, valid=27, n=30).endswith(", FAIL")
    assert verdict("app", app_budget + 1, 30, 30).endswith(", FAIL")
    assert not verdict("app", app_budget, 30, 30).endswith(", FAIL")
    assert verdict("diagram", diagram_budget + 1, 30, 30).endswith(", FAIL")
    assert not verdict("diagram", diagram_budget, 30, 30).endswith(", FAIL")
    assert verdict("diagram", None, 30, 30).endswith(", FAIL")
    assert "median landing none against" in verdict("diagram", None, 30, 30)


def test_parser_defaults() -> None:
    args = bench_visual_call.build_parser().parse_args([])

    assert args.samples == 30
    assert args.kinds == ["diagram", "app"]


async def test_a_scripted_push_lands_as_a_valid_sample() -> None:
    push = TurnChunk(
        kind="tool_call", text=DIAGRAM_ARGUMENTS, tool_call_id="call-1", tool_name="push_diagram"
    )
    stream = ScriptedStream(
        [push], TurnUsage(prompt_tokens=10, completion_tokens=20), "tool_calls", 5
    )
    scripted = ScriptedReasoning(stream)
    reasoning = bench_visual_call.MeteredReasoning(scripted)

    sample = await bench_visual_call.one_call(reasoning, "diagram", 3000)

    assert scripted.tool_choices == ["required"]
    assert sample.valid is True
    assert sample.truncated is False
    assert sample.landing_ms is not None
    assert sample.result_ms >= sample.landing_ms
    assert sample.output_tokens == 20
    assert sample.first_chunk_ms == 5
    assert sample.cost_usd is not None and sample.cost_usd > 0
    assert sample.result == "push_diagram: sent"

    prose = ScriptedStream(
        [TurnChunk(kind="spoken", text="here is a picture")],
        TurnUsage(prompt_tokens=10, completion_tokens=4),
        "stop",
        3,
    )
    reasoning = bench_visual_call.MeteredReasoning(ScriptedReasoning(prose))

    sample = await bench_visual_call.one_call(reasoning, "diagram", 3000)

    assert sample.valid is False
    assert sample.landing_ms is None
    assert isinstance(sample.result_ms, int)
    assert sample.result == "visual: error: no tool call"


async def test_a_stream_without_a_usage_chunk_has_an_unknown_cost() -> None:
    stream = ScriptedStream([TurnChunk(kind="spoken", text="cut off")], None, None, None)
    reasoning = bench_visual_call.MeteredReasoning(ScriptedReasoning(stream))

    sample = await bench_visual_call.one_call(reasoning, "app", 3000)

    assert sample.cost_usd is None
    assert sample.output_tokens is None
    assert sample.landing_ms is None


def test_the_report_prints_a_verdict_per_kind(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_visual_call.build_parser().parse_args([])
    sent = bench_visual_call.CallSample(
        first_chunk_ms=400,
        landing_ms=3000,
        result_ms=3010,
        valid=True,
        truncated=False,
        output_tokens=250,
        cost_usd=0.0002,
        result="push_diagram: sent",
    )
    prose = sent.model_copy(
        update={
            "landing_ms": None,
            "result_ms": 800,
            "valid": False,
            "cost_usd": None,
            "result": "visual: error: no tool call",
        }
    )
    truncated = sent.model_copy(
        update={
            "landing_ms": None,
            "result_ms": 95000,
            "valid": False,
            "truncated": True,
            "result": "visual: error: truncated at 3000 tokens",
        }
    )

    bench_visual_call.report(cfg, args, {"diagram": [sent, prose], "app": [truncated]})

    lines = capsys.readouterr().out.splitlines()
    assert "kind=diagram" in lines
    assert "kind=app" in lines
    landing, no_landing = [line for line in lines if line.startswith("landing ")]
    assert "n=  1 median=  3000ms" in landing
    assert no_landing.endswith("no samples")
    every, every_app = [line for line in lines if line.startswith("result, every call")]
    assert "n=  2" in every and "min=   800ms" in every
    assert "n=  1 median= 95000ms" in every_app
    verdicts = [line for line in lines if line.startswith("verdict:")]
    assert len(verdicts) == 2
    assert verdicts[0].startswith("verdict: diagram median landing 3000ms")
    assert "valid 1/2" in verdicts[0]
    assert verdicts[0].endswith(", FAIL")
    assert verdicts[1].startswith("verdict: app median landing none against")
    assert verdicts[1].endswith(", FAIL")
    assert "valid 1/2" in lines
    assert "truncated 1/1" in lines
    assert "unknown cost 1/2" in lines
    assert "unknown cost 0/1" in lines
