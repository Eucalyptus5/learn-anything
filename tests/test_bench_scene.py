import importlib.util
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from tutor.config import Settings
from tutor.cost import TurnUsage
from tutor.prompt import TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.scene import SCENE_TOOL, SceneDraft

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_scene.py"
_spec = importlib.util.spec_from_file_location("bench_scene", SCRIPT)
bench_scene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_scene)

STEPS = ["The axes", "The curve", "The band"]
HTML = (
    '<!doctype html><script src="/lesson.js"></script>'
    '<script src="/vendor/gsap.min.js"></script><div id="s"></div>'
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
        self.prompts: list[TurnPrompt] = []
        self.efforts: list[str | None] = []
        self.models: list[str | None] = []
        self.max_tokens: list[int | None] = []

    def start_turn(
        self,
        prompt: TurnPrompt,
        tools: Sequence[dict] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
    ) -> ScriptedStream:
        self.prompts.append(prompt)
        self.efforts.append(effort)
        self.models.append(model)
        self.max_tokens.append(max_tokens)
        return self._stream

    async def aclose(self) -> None:
        return None


def scene_call(html: str, steps: list[str]) -> TurnChunk:
    return TurnChunk(
        kind="tool_call",
        text=json.dumps({"html": html, "steps": steps}),
        tool_call_id="call-1",
        tool_name=SCENE_TOOL,
    )


def test_the_brief_validates_and_names_the_numbers() -> None:
    brief = bench_scene.BRIEF
    assert brief.kind == "app" and brief.title == "Clipped objective"
    assert "0.5 to 1.5" in brief.show and "0.2" in brief.show


async def test_a_scripted_draft_is_a_valid_sample_and_is_written(tmp_path: Path) -> None:
    stream = ScriptedStream(
        [scene_call(HTML, STEPS)],
        TurnUsage(prompt_tokens=10, completion_tokens=2000),
        "tool_calls",
        5,
    )
    scripted = ScriptedReasoning(stream)
    reasoning = bench_scene.MeteredReasoning(scripted)

    sample, draft = await bench_scene.one_build(
        reasoning, bench_scene.BRIEF, 32000, "high", "draw-1"
    )

    assert scripted.efforts == ["high"] and scripted.models == ["draw-1"]
    assert scripted.max_tokens == [32000]
    assert sample.valid is True and sample.truncated is False and sample.prose_only is False
    assert sample.empty is False
    assert sample.output_tokens == 2000 and sample.first_chunk_ms == 5
    assert sample.cost_usd is not None and sample.cost_usd > 0
    assert sample.steps == 3 and sample.uses_helper is True and sample.uses_gsap is True
    assert sample.result == "scene: sent"
    assert draft == SceneDraft(html=HTML, steps=STEPS)

    path = bench_scene.write_scene(tmp_path, "run", 4, draft)

    assert path == tmp_path / "run" / "04.html"
    assert path.read_text() == HTML
    assert json.loads((tmp_path / "run" / "04.json").read_text()) == {"steps": STEPS}


async def test_prose_and_a_bare_document_are_read_for_what_they_are() -> None:
    prose = ScriptedStream(
        [TurnChunk(kind="spoken", text="here is a scene")],
        TurnUsage(prompt_tokens=10, completion_tokens=4),
        "stop",
        3,
    )
    reasoning = bench_scene.MeteredReasoning(ScriptedReasoning(prose))

    sample, draft = await bench_scene.one_build(reasoning, bench_scene.BRIEF, 32000, "high")

    assert sample.valid is False and sample.prose_only is True and draft is None
    assert sample.empty is False
    assert sample.steps is None and sample.uses_helper is None and sample.uses_gsap is None
    assert sample.result == "scene: error: no tool call"

    bare = ScriptedStream(
        [scene_call("<!doctype html><p>x</p>", STEPS)],
        TurnUsage(prompt_tokens=10, completion_tokens=20),
        "tool_calls",
        5,
    )
    reasoning = bench_scene.MeteredReasoning(ScriptedReasoning(bare))

    sample, draft = await bench_scene.one_build(reasoning, bench_scene.BRIEF, 32000, "high")

    assert sample.valid is True
    assert sample.uses_helper is False and sample.uses_gsap is False


async def test_a_stream_that_ends_at_the_cap_with_nothing_is_empty_not_prose() -> None:
    capped = ScriptedStream(
        [], TurnUsage(prompt_tokens=4277, completion_tokens=128000), "length", 4000
    )
    reasoning = bench_scene.MeteredReasoning(ScriptedReasoning(capped))

    sample, draft = await bench_scene.one_build(reasoning, bench_scene.BRIEF, 128000, "high")

    assert draft is None and sample.valid is False
    assert sample.empty is True and sample.prose_only is False and sample.truncated is False
    assert sample.output_tokens == 128000 and sample.first_chunk_ms == 4000
    assert sample.steps is None and sample.uses_helper is None and sample.uses_gsap is None
    assert sample.result == "scene: error: empty reply"


async def test_a_stream_without_a_usage_chunk_has_an_unknown_cost() -> None:
    stream = ScriptedStream([TurnChunk(kind="spoken", text="cut off")], None, None, None)
    reasoning = bench_scene.MeteredReasoning(ScriptedReasoning(stream))

    sample, _ = await bench_scene.one_build(reasoning, bench_scene.BRIEF, 32000, "high")

    assert sample.cost_usd is None and sample.output_tokens is None


def test_the_verdict_fails_only_under_the_validity_floor() -> None:
    assert bench_scene.verdict(90000, 30, 30, 0) == (
        "verdict: median build 90000ms, no budget yet, valid 30/30 against the floor of 27, "
        "prose only 0/30"
    )
    assert bench_scene.verdict(90000, 26, 30, 4).endswith("prose only 4/30, FAIL")
    assert bench_scene.verdict(None, 0, 30, 30).startswith("verdict: median build none")


def test_parser_defaults() -> None:
    args = bench_scene.build_parser().parse_args([])
    assert args.samples == 30 and args.model == "" and args.effort == "high"
    assert args.out == str(bench_scene.OUT_DIR)


def test_the_report_names_the_model_the_effort_and_the_run_directory(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_scene.build_parser().parse_args(["--model", "draw-1", "--effort", "medium"])
    samples = [
        bench_scene.BuildSample(
            build_ms=90000,
            first_chunk_ms=200,
            valid=True,
            truncated=False,
            prose_only=False,
            output_tokens=3000,
            cost_usd=0.0015,
            steps=4,
            uses_helper=True,
            uses_gsap=True,
            empty=False,
            result="scene: sent",
        ),
        bench_scene.BuildSample(
            build_ms=421000,
            first_chunk_ms=4000,
            valid=False,
            truncated=False,
            prose_only=False,
            output_tokens=128000,
            cost_usd=0.03,
            steps=None,
            uses_helper=None,
            uses_gsap=None,
            empty=True,
            result="scene: error: empty reply",
        ),
    ]

    bench_scene.report(cfg, args, samples, Path("/tmp/run"))

    lines = capsys.readouterr().out.splitlines()
    (header,) = [line for line in lines if line.startswith("model=")]
    assert "scene_model=draw-1" in header and "effort=medium" in header
    assert "scene_max_tokens=128000" in header and "out=/tmp/run" in header
    assert any(line.startswith("build ") for line in lines)
    assert "valid 1/2" in lines and "helper 1/1" in lines and "gsap 1/1" in lines
    assert lines.index("prose only 0/2") + 1 == lines.index("empty 1/2")
    assert "other errors 0/2" in lines
    assert any(line.startswith("steps ") for line in lines)
    assert any("flash" in line and "not priced" in line for line in lines)
    (validity,) = [line for line in lines if line.startswith("validity:")]
    assert "empty counts the streams that ended with neither" in validity
    (verdict,) = [line for line in lines if line.startswith("verdict:")]
    assert verdict.endswith("valid 1/2 against the floor of 27, prose only 0/2, FAIL")
