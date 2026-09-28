import importlib.util
import json
from pathlib import Path

import pytest

from tests.test_bench_turn import ScriptedReasoning, ScriptedStream
from tests.test_lesson import plan, scene
from tutor.config import Settings
from tutor.cost import TurnUsage
from tutor.lesson import LessonPlan
from tutor.planner import EMPTY_REPLY, NO_TOOL_CALL, PLAN_TOOL, plan_prompt
from tutor.reasoning import TurnChunk

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_planner.py"
_spec = importlib.util.spec_from_file_location("bench_planner", SCRIPT)
bench_planner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_planner)

PLAN = LessonPlan.model_validate(plan(3))
USAGE = TurnUsage(prompt_tokens=900, completion_tokens=2100)
TOPICS = [
    {
        "id": "ml-ppo",
        "domain": "machine learning",
        "subject": "PPO",
        "starting_from": "I know policy gradients",
        "folder": False,
    },
    {
        "id": "sys-raft",
        "domain": "systems",
        "subject": "Raft leader election",
        "starting_from": "",
        "folder": True,
    },
]


def call(body: dict[str, object]) -> TurnChunk:
    return TurnChunk(kind="tool_call", text=json.dumps(body), tool_call_id="c", tool_name=PLAN_TOOL)


def metered(*streams: ScriptedStream) -> "bench_planner.MeteredReasoning":
    return bench_planner.MeteredReasoning(ScriptedReasoning(list(streams)))


def test_parser_defaults() -> None:
    args = bench_planner.build_parser().parse_args([])
    assert args.samples == 30 and args.model == "" and args.effort == "high"
    assert args.stage == "connect" and args.plan is None and args.topics is None
    assert args.subject == "PPO" and args.starting_from == bench_planner.STARTING_FROM
    assert args.out.endswith("scratch/bench/plans")
    rerun = bench_planner.build_parser().parse_args(["--stage", "boundary", "--plan", "p.json"])
    assert rerun.stage == "boundary" and rerun.plan == "p.json"
    topics = bench_planner.build_parser().parse_args(["--topics", "t.json", "--out", "o"])
    assert topics.topics == "t.json" and topics.out == "o"


def test_the_fixed_transcripts_are_ascii_and_the_boundary_extends_the_first_answer() -> None:
    first = bench_planner.FIRST_ANSWER_TRANSCRIPT
    boundary = bench_planner.BOUNDARY_TRANSCRIPT
    assert boundary[: len(first)] == first and len(boundary) > len(first)
    assert first[0].role == "user" and first[-1].role == "user"
    assert all(m.content.isascii() for m in boundary)
    assert bench_planner.protected_for("first_answer", PLAN) == ["scene-1"]
    assert bench_planner.protected_for("boundary", PLAN) == ["scene-1", "scene-2"]


async def test_a_scripted_plan_is_a_valid_sample_and_is_written(tmp_path: Path) -> None:
    stream = ScriptedStream(
        [call(plan(2))], usage=USAGE, finish_reason="tool_calls", first_chunk_ms=350
    )
    prompt = plan_prompt("PPO", bench_planner.STARTING_FROM, False, None, [], [])
    sample, written = await bench_planner.one_plan(
        metered(stream), prompt, 8000, "high", None, None, []
    )

    assert sample.valid and written == LessonPlan.model_validate(plan(2))
    assert sample.scenes == 2 and sample.steps == [3, 3] and sample.asking == 2
    assert sample.first_chunk_ms == 350 and sample.output_tokens == 2100
    assert sample.cost_usd is not None and sample.cost_usd > 0
    assert sample.accepted is None and sample.touched_protected is None and sample.appended is None
    assert not (sample.empty or sample.prose_only or sample.truncated)
    path = bench_planner.write_plan(tmp_path, "run", 4, written)
    assert path == tmp_path / "run" / "plan-04.json"
    assert LessonPlan.model_validate_json(path.read_text()) == written


async def test_prose_empty_and_a_cut_stream_are_read_for_what_they_are() -> None:
    prompt = plan_prompt("PPO", bench_planner.STARTING_FROM, False, None, [], [])
    prose = ScriptedStream(
        [TurnChunk(kind="spoken", text="a plan")], usage=USAGE, finish_reason="stop"
    )
    empty = ScriptedStream([], usage=USAGE, finish_reason="length")
    cut = ScriptedStream([call(plan(2))], usage=USAGE, finish_reason="length")
    for stream, check in (
        (prose, lambda s: s.prose_only and s.result == NO_TOOL_CALL),
        (empty, lambda s: s.empty and s.result == EMPTY_REPLY),
        (cut, lambda s: s.truncated and s.result.startswith("planner: error: truncated")),
    ):
        sample, written = await bench_planner.one_plan(
            metered(stream), prompt, 8000, "high", None, None, []
        )
        assert not sample.valid and written is None and check(sample)


async def test_a_rerun_sample_is_judged_on_the_splice_and_notes_a_touched_prefix() -> None:
    prompt = plan_prompt(
        "PPO",
        bench_planner.STARTING_FROM,
        False,
        PLAN,
        ["scene-1"],
        bench_planner.FIRST_ANSWER_TRANSCRIPT,
    )
    kept = ScriptedStream([call(plan(4))], usage=USAGE, finish_reason="tool_calls")
    sample, _ = await bench_planner.one_plan(
        metered(kept), prompt, 8000, "high", None, PLAN, ["scene-1"]
    )
    assert sample.accepted and not sample.touched_protected and sample.appended
    retitled = plan(3)
    retitled["scenes"][0]["title"] = "Other"
    touched = ScriptedStream([call(retitled)], usage=USAGE, finish_reason="tool_calls")
    sample, written = await bench_planner.one_plan(
        metered(touched), prompt, 8000, "high", None, PLAN, ["scene-1"]
    )
    assert sample.valid and sample.accepted and sample.touched_protected
    assert sample.appended is False
    assert written == LessonPlan.model_validate(retitled)
    crowded = {**plan(), "scenes": [scene(n) for n in range(2, 14)]}
    overfull = ScriptedStream([call(crowded)], usage=USAGE, finish_reason="tool_calls")
    sample, _ = await bench_planner.one_plan(
        metered(overfull), prompt, 8000, "high", None, PLAN, ["scene-1"]
    )
    assert sample.valid and sample.accepted is False and sample.appended is None


def test_valid_unicode_plan_text_is_saved_as_ascii_json(tmp_path: Path) -> None:
    body = plan(2)
    body["profile"] = "Uses \u03c0 and \u201ccurves\u201d."
    lesson = LessonPlan.model_validate(body)
    path = bench_planner.write_plan(tmp_path, "unicode", 1, lesson)
    assert path.read_bytes().isascii()
    assert LessonPlan.model_validate_json(path.read_text(encoding="ascii")) == lesson


def test_the_verdict_fails_only_under_the_floor() -> None:
    assert bench_planner.verdict(41000, 27, 30, "connect") == (
        "verdict: median plan 41000ms, no budget yet, valid 27/30 against the floor of 27"
    )
    assert bench_planner.verdict(41000, 26, 30, "connect").endswith(", FAIL")
    assert "accepted 28/30" in bench_planner.verdict(9000, 28, 30, "boundary")
    assert bench_planner.verdict(None, 0, 0, "connect").startswith("verdict: median plan none")


def test_the_report_names_the_model_the_stage_and_the_run_directory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_planner.build_parser().parse_args(["--model", "glm-5.3", "--samples", "2"])
    samples = [
        bench_planner.PlanSample(
            plan_ms=40000,
            first_chunk_ms=300,
            valid=True,
            accepted=None,
            touched_protected=None,
            appended=None,
            output_tokens=2000,
            cost_usd=0.001,
            scenes=4,
            steps=[3, 4, 5, 3],
            asking=6,
            empty=False,
            prose_only=False,
            truncated=False,
            result="planner: sent",
        ),
        bench_planner.PlanSample(
            plan_ms=42000,
            first_chunk_ms=None,
            valid=False,
            accepted=None,
            touched_protected=None,
            appended=None,
            output_tokens=None,
            cost_usd=None,
            scenes=None,
            steps=[],
            asking=None,
            empty=True,
            prose_only=False,
            truncated=False,
            result=EMPTY_REPLY,
        ),
    ]
    bench_planner.report(cfg, args, samples, Path("scratch/bench/plans/run"))
    out = capsys.readouterr().out
    assert (
        "model=glm-5.3-flash  planner_model=glm-5.3  effort=high  stage=connect  samples=2 "
        "(plus 3 discarded warm-ups)  planner_max_tokens=64000  subject='PPO'"
    ) in out
    assert "out=scratch/bench/plans/run" in out
    assert "valid 1/2" in out and "empty 1/2" in out and "prose only 0/2" in out
    assert "scenes per plan" in out and "steps per scene" in out and "asking steps 6/15" in out
    assert "cost usd (list, flash rate)" in out
    assert "plan (valid)" in out
    assert "verdict: median plan 40000ms" in out


def test_a_topic_list_is_read_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "held_in.json"
    path.write_text(json.dumps({"sealed_at": "2026-09-26T10:00:00", "topics": TOPICS}))
    topics = bench_planner.load_topics(path)
    assert [t.id for t in topics] == ["ml-ppo", "sys-raft"] and topics[1].folder
    assert topics[1].domain == "systems"
    minimal = tmp_path / "minimal.json"
    minimal.write_text(json.dumps({"topics": [{"id": "calc-chain", "subject": "The chain rule"}]}))
    (only,) = bench_planner.load_topics(minimal)
    assert only.starting_from == "" and only.folder is False and only.domain == ""
    for body in (
        {"topics": []},
        {"topics": [TOPICS[0], TOPICS[0]]},
        {"topics": [{**TOPICS[0], "notes": "x"}]},
        {"topics": [{**TOPICS[0], "id": "ML PPO"}]},
        {"topics": [{"id": "x", "subject": ""}]},
        {"topics": [{"id": "x", "subject": "s" * 201}]},
        {"topics": [{**TOPICS[0], "domain": "d" * 81}]},
        {"topics": [{**TOPICS[0], "starting_from": "s" * 401}]},
        {"topics": [{**TOPICS[0], "folder": "maybe"}]},
        {"topics": TOPICS, "seed": 1},
        {"sealed_at": 1, "topics": TOPICS},
        TOPICS,
    ):
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError):
            bench_planner.load_topics(path)


async def test_each_topic_gets_one_connect_plan_or_its_error_string(tmp_path: Path) -> None:
    good = ScriptedStream([call(plan(2))], usage=USAGE, finish_reason="tool_calls")
    prose = ScriptedStream(
        [TurnChunk(kind="spoken", text="a plan")], usage=USAGE, finish_reason="stop"
    )
    scripted = ScriptedReasoning([good, prose])
    topics = [bench_planner.Topic.model_validate(t) for t in TOPICS]
    out = tmp_path / "topics"
    rows = await bench_planner.run_topics(
        bench_planner.MeteredReasoning(scripted), topics, out, 8000, "high", None
    )

    assert [(topic.id, sample.valid) for topic, sample in rows] == [
        ("ml-ppo", True),
        ("sys-raft", False),
    ]
    assert sorted(p.name for p in out.iterdir()) == ["ml-ppo.json", "sys-raft.error.txt"]
    saved = LessonPlan.model_validate_json((out / "ml-ppo.json").read_text())
    assert saved == LessonPlan.model_validate(plan(2))
    assert (out / "sys-raft.error.txt").read_text() == NO_TOOL_CALL + "\n"
    assert all(p.read_bytes().isascii() for p in out.iterdir())
    assert [p.user_text for p in scripted.prompts] == [
        "Subject: PPO\nStarting from: I know policy gradients",
        (
            "Subject: Raft leader election\n"
            "A folder of source code is attached; the tutor can search it while teaching."
        ),
    ]
    assert all(p.history == [] for p in scripted.prompts)


async def test_a_topic_run_refuses_an_output_directory_that_is_not_empty(tmp_path: Path) -> None:
    (tmp_path / "old.json").write_text("{}")
    scripted = ScriptedReasoning([])
    topic = bench_planner.Topic.model_validate(TOPICS[0])
    with pytest.raises(FileExistsError):
        await bench_planner.run_topics(
            bench_planner.MeteredReasoning(scripted), [topic], tmp_path, 8000, "high", None
        )
    assert scripted.prompts == []
