import ast
import importlib.util
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from tests.test_session import LESSON, WATCHED
from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.config import Settings
from tutor.cost import TurnUsage, UsageLedger
from tutor.input_path import EndOfTurn, SpeechStarted
from tutor.lesson import OPENING_TEXT, BuiltScene, Cursor, LessonState, Step
from tutor.planner import PLAN_TOOL, PLAN_TOOLS
from tutor.prompt import Message, TurnPrompt
from tutor.reasoning import TurnChunk
from tutor.scene import SCENE_TOOL, SCENE_TOOLS, SceneDraft, planned_scene_prompt
from tutor.visuals import LessonAck

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_turn.py"
_spec = importlib.util.spec_from_file_location("bench_turn", SCRIPT)
bench_turn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_turn)

TIMES = [float(n) for n in range(60)]
PROMPT = TurnPrompt(system="s", user_text="teach me ppo")

DELTAS = [
    "The `acquire` method pops ",
    "from the available list. Then it adds",
    " the connection to in_use, and returns it.",
]


async def _loop_pump(deltas: list[str]) -> list[str]:
    async def replay() -> AsyncIterator[str]:
        for delta in deltas:
            yield delta

    return [clause async for clause in clause_chunks(spoken_text(replay(), Scrubber()))]


def test_substance_frame_is_the_first_frame_holding_the_model_buffer() -> None:
    assert bench_turn.substance_frame_time([24000, 12000], 1, TIMES) == 50.0


def test_substance_frame_for_the_first_buffer_is_frame_zero() -> None:
    assert bench_turn.substance_frame_time([24000, 12000], 0, TIMES) == 0.0


def test_substance_frame_when_the_boundary_falls_inside_a_frame() -> None:
    assert bench_turn.substance_frame_time([500, 100], 1, TIMES) == 1.0
    assert bench_turn.substance_frame_time([480, 100], 1, TIMES) == 1.0
    assert bench_turn.substance_frame_time([479, 100], 1, TIMES) == 0.0


def test_substance_frame_not_yet_emitted_is_none() -> None:
    assert bench_turn.substance_frame_time([24000, 12000], 1, TIMES[:50]) is None


async def test_loop_clauses_classify_as_model_and_lead_in_does_not() -> None:
    clauses = await _loop_pump(DELTAS)
    model = bench_turn.model_text([DELTAS])

    assert clauses
    assert all(bench_turn.is_model(clause, model) for clause in clauses)
    assert "`" not in model
    assert not bench_turn.is_model("The match is in src/pool.py line 12.", model)
    assert not bench_turn.is_model(None, model)


def test_model_text_reads_a_tagged_reply_as_the_loop_speaks_it() -> None:
    model = bench_turn.model_text([["<step 2>The band sits", "<step 3> at 0.8. <scene 2>Then"]])

    assert bench_turn.is_model("The band sits at 0.8.", model)
    assert "<" not in model


async def test_follow_up_stream_joins_the_first_with_a_newline() -> None:
    first = ["Two things to check"]
    second = ["and here is the second one."]

    assert (
        bench_turn.model_text([first, second]) == "Two things to check\nand here is the second one."
    )


async def test_scripted_source_yields_in_order_and_ends_on_close() -> None:
    source = bench_turn.ScriptedSource()
    source.inject(EndOfTurn(text="one"))
    source.inject(SpeechStarted())
    source.inject(EndOfTurn(text="two"))
    source.close()

    events = [event async for event in source.events()]

    assert events == [EndOfTurn(text="one"), SpeechStarted(), EndOfTurn(text="two")]


def test_floor_lifts_every_zero_sample_and_leaves_the_original_alone() -> None:
    pcm = np.array([0, 5, 0, -3, 0], dtype=np.int16)

    floored = bench_turn.floored(pcm)

    assert floored.dtype == np.int16
    assert floored.tolist() == [1, 5, 1, -3, 1]
    assert pcm.tolist() == [0, 5, 0, -3, 0]
    assert floored is not pcm


def test_parser_defaults() -> None:
    args = bench_turn.build_parser().parse_args([])

    assert args.root is None
    assert args.model is None
    assert args.subject is None
    assert args.starting_from == bench_turn.STARTING_FROM
    assert not hasattr(args, "history")
    assert args.capture is None
    assert args.silent_synth is False


def test_the_silent_synth_flag_parses() -> None:
    args = bench_turn.build_parser().parse_args(["--silent-synth"])

    assert args.silent_synth is True


def test_concept_mode_uses_the_ppo_utterances() -> None:
    assert bench_turn.utterances_for(None) is bench_turn.PPO_UTTERANCES
    assert bench_turn.PPO_UTTERANCES == (
        "teach me ppo",
        "why does it clip the ratio instead of using it directly",
        "what happens when the advantage is negative",
        "quiz me on the clipped objective",
        "the clip makes the gradient larger past epsilon",
        "just tell me",
    )
    assert bench_turn.utterances_for(Path("x")) is bench_turn.UTTERANCES


def test_silent_synth_returns_one_frame_of_int16_zeros() -> None:
    synth = bench_turn.SilentSynth()

    frame = synth.synthesize("PPO clips.")

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.int16
    assert bench_turn.SILENT_FRAME_SAMPLES == 480
    assert len(frame) == bench_turn.SILENT_FRAME_SAMPLES
    assert not frame.any()

    other = synth.synthesize("PPO clips.")
    frame[0] = 1
    assert other[0] == 0


class ScriptedStream:
    def __init__(
        self,
        chunks: list[TurnChunk],
        usage: TurnUsage | None = None,
        finish_reason: str | None = None,
        first_chunk_ms: int | None = None,
    ) -> None:
        self._chunks = chunks
        self.usage = usage
        self.finish_reason = finish_reason
        self.first_chunk_ms = first_chunk_ms

    async def __aiter__(self) -> AsyncIterator[TurnChunk]:
        for chunk in self._chunks:
            yield chunk


class ScriptedReasoning:
    def __init__(self, streams: list[ScriptedStream]) -> None:
        self._streams = streams
        self.prompts: list[TurnPrompt] = []

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
        return self._streams.pop(0)


def _spoken(*texts: str) -> list[TurnChunk]:
    return [TurnChunk(kind="spoken", text=text) for text in texts]


async def test_a_voice_stream_records_its_deltas_and_sums_its_usage() -> None:
    first = ScriptedStream(
        _spoken("PPO ", "clips."), usage=TurnUsage(prompt_tokens=10, completion_tokens=20)
    )
    second = ScriptedStream(
        _spoken("Then it stops."), usage=TurnUsage(prompt_tokens=30, completion_tokens=5)
    )
    reasoning = bench_turn.MeteredReasoning(ScriptedReasoning([first, second]))
    record = reasoning.begin()

    chunks = [chunk async for chunk in reasoning.start_turn(PROMPT, tools=[], max_tokens=10)]
    assert len(chunks) == 2
    assert record.first_spoken_ms is not None
    assert record.voice_usage == TurnUsage(prompt_tokens=10, completion_tokens=20)

    async for _ in reasoning.start_turn(PROMPT):
        pass

    assert record.voice_usage == TurnUsage(prompt_tokens=40, completion_tokens=25)
    assert record.visual_usage is None
    assert record.streams == [["PPO ", "clips."], ["Then it stops."]]
    assert reasoning.ledgers["voice"].turns == 2
    assert reasoning.ledgers["voice"].total.prompt_tokens == 40
    assert reasoning.ledgers["visual"].turns == 0
    assert reasoning.ledger.turns == 2


async def test_a_visual_stream_is_attributed_to_the_visual_ledger() -> None:
    call = TurnChunk(kind="tool_call", text="{}", tool_call_id="c", tool_name="push_diagram")
    stream = ScriptedStream(
        [call],
        usage=TurnUsage(prompt_tokens=30, completion_tokens=40),
        finish_reason="tool_calls",
        first_chunk_ms=7,
    )
    reasoning = bench_turn.MeteredReasoning(ScriptedReasoning([stream]))
    record = reasoning.begin()

    metered = reasoning.start_turn(PROMPT, tools=SCENE_TOOLS, max_tokens=10, tool_choice="required")
    chunks = [chunk async for chunk in metered]

    assert chunks == [call]
    assert metered.finish_reason == "tool_calls"
    assert metered.first_chunk_ms == 7
    assert record.visual_first_chunk_ms == 7
    assert record.visual_usage == TurnUsage(prompt_tokens=30, completion_tokens=40)
    assert record.voice_usage is None
    assert record.first_spoken_ms is None
    assert record.streams == []
    assert reasoning.ledgers["visual"].turns == 1
    assert reasoning.ledgers["visual"].total.completion_tokens == 40
    assert reasoning.ledgers["voice"].turns == 0
    assert reasoning.ledger.turns == 1


def test_summarize_usd_prints_six_decimals(capsys) -> None:
    bench_turn.summarize_usd("cost per turn (list)", [0.001, 0.0025, 0.0005])
    bench_turn.summarize_usd("visual cost per turn (list)", [])

    filled, empty = capsys.readouterr().out.splitlines()
    assert filled.startswith("cost per turn (list)")
    assert "n=  3" in filled
    assert "median=0.001000" in filled
    assert "p95=0.001000" in filled
    assert "min=0.000500" in filled
    assert "max=0.002500" in filled
    assert "ms" not in filled
    assert empty.endswith("no samples")


def test_a_silent_run_reports_the_audio_lines_as_not_measured(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_turn.build_parser().parse_args(["--silent-synth"])
    sample = bench_turn.Sample(
        first_sound_ms=900,
        substance_ms=900,
        first_content_delta_ms=600,
        stages=0,
        silent=False,
        audio_ms=4000,
        voice_usd=0.0001,
        utterance=0,
    )

    bench_turn.report(cfg, args, [sample], bench_turn.MeteredReasoning(ScriptedReasoning([])))

    lines = capsys.readouterr().out.splitlines()
    (model,) = [line for line in lines if line.startswith("model=")]
    assert "synth=silent" in model
    for label in (
        "time to first sound",
        "time to substance",
        "audio length per turn",
    ):
        (line,) = [line for line in lines if line.startswith(f"{label:34s}")]
        assert line.endswith("not measured")
    (delta,) = [
        line
        for line in lines
        if line.startswith(f"{'first content delta (model, from first request)':34s}")
    ]
    assert not delta.endswith("not measured")
    assert any(line.startswith("cost per turn (list)") for line in lines)
    assert any(line.startswith("turns with unknown cost") for line in lines)
    assert any(line.startswith("silent turns") for line in lines)


def test_a_kokoro_run_leaves_the_model_line_alone(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_turn.build_parser().parse_args([])
    sample = bench_turn.Sample(
        first_sound_ms=900,
        substance_ms=900,
        first_content_delta_ms=600,
        stages=0,
        silent=False,
        audio_ms=4000,
        voice_usd=0.0001,
        utterance=0,
    )

    bench_turn.report(cfg, args, [sample], bench_turn.MeteredReasoning(ScriptedReasoning([])))

    lines = capsys.readouterr().out.splitlines()
    (model,) = [line for line in lines if line.startswith("model=")]
    assert "synth=" not in model
    (first_sound,) = [line for line in lines if line.startswith(f"{'time to first sound':34s}")]
    assert "n=  1" in first_sound


def test_is_truncated_reads_a_scene_result() -> None:
    assert bench_turn.is_truncated("scene: error: truncated at 32000 tokens")
    assert not bench_turn.is_truncated("scene: sent")
    assert bench_turn.is_truncated("planner: error: truncated at 8000 tokens")
    assert not bench_turn.is_truncated("planner: sent")


POWERMETRICS_BLOCKS = """Machine model: Mac14,2
OS version: 25C56
Boot arguments: 
Boot time: Thu Sep 10 23:42:17 2026



*** Sampled system activity (Fri Sep 11 13:41:46 2026 -0700) (1002.90ms elapsed) ***


**** Processor usage ****

E-Cluster HW active frequency: 1248 MHz
E-Cluster HW active residency:  77.43% (600 MHz:   0% 912 MHz:  54% 2424 MHz: 8.7%)
E-Cluster idle residency:  22.57%
CPU 0 frequency: 1359 MHz

P-Cluster HW active frequency: 1145 MHz
P-Cluster HW active residency:  56.23% (660 MHz:  36% 924 MHz: .54% 3504 MHz:   0%)
P-Cluster idle residency:  43.77%
CPU 4 frequency: 2039 MHz

CPU Power: 823 mW
GPU Power: 26 mW
ANE Power: 323 mW
Combined Power (CPU + GPU + ANE): 1172 mW


**** Thermal pressure ****

Current pressure level: Nominal


*** Sampled system activity (Fri Sep 11 13:41:47 2026 -0700) (1010.09ms elapsed) ***


**** Processor usage ****

E-Cluster HW active frequency: 1049 MHz
E-Cluster HW active residency:  77.05% (600 MHz:   0% 912 MHz:  65% 2424 MHz: 1.8%)
E-Cluster idle residency:  22.95%
CPU 0 frequency: 1045 MHz

P-Cluster HW active frequency: 715 MHz
P-Cluster HW active residency:  51.36% (660 MHz:  47% 924 MHz: .40% 3504 MHz:   0%)
P-Cluster idle residency:  48.64%
CPU 4 frequency: 1220 MHz

CPU Power: 205 mW
GPU Power: 19 mW
ANE Power: 221 mW
Combined Power (CPU + GPU + ANE): 445 mW


**** Thermal pressure ****

Current pressure level: Nominal
"""


def _blocks(text: str) -> list[str]:
    return list(bench_turn.powermetrics_blocks(text.splitlines(keepends=True)))


def test_powermetrics_blocks_frames_on_the_pressure_line() -> None:
    blocks = _blocks(POWERMETRICS_BLOCKS)

    assert len(blocks) == 2
    assert all(block.startswith(bench_turn.POWERMETRICS_HEADER) for block in blocks)
    assert all(block.rstrip().endswith("Current pressure level: Nominal") for block in blocks)
    assert "Machine model" not in blocks[0]


def test_powermetrics_blocks_holds_a_block_until_its_pressure_line() -> None:
    lines = POWERMETRICS_BLOCKS.splitlines(keepends=True)
    cut = lines[: len(lines) - 1]

    assert len(list(bench_turn.powermetrics_blocks(cut))) == 1


def test_parse_powermetrics_reads_each_block() -> None:
    first, second = _blocks(POWERMETRICS_BLOCKS)

    a = bench_turn.parse_powermetrics(first)
    b = bench_turn.parse_powermetrics(second)

    assert a == bench_turn.PowerSample(
        cpu_power_mw=823,
        combined_power_mw=1172,
        p_cluster_mhz=1145,
        e_cluster_mhz=1248,
        pressure="Nominal",
    )
    assert b == bench_turn.PowerSample(
        cpu_power_mw=205,
        combined_power_mw=445,
        p_cluster_mhz=715,
        e_cluster_mhz=1049,
        pressure="Nominal",
    )


def test_parse_powermetrics_without_pressure_is_none() -> None:
    first, _ = _blocks(POWERMETRICS_BLOCKS)
    cut = first[: first.index("**** Thermal pressure ****")]

    assert bench_turn.parse_powermetrics(cut) is None


def test_parse_swap_truncates_to_whole_megabytes() -> None:
    line = "total = 5120.00M  used = 3972.81M  free = 1147.19M  (encrypted)\n"

    assert bench_turn.parse_swap(line) == 3972


def test_edge_buckets_split_the_first_and_last_window() -> None:
    offsets = [0.0, 100.0, 299.0, 300.0, 1500.0, 1501.0, 1799.0]

    first, last = bench_turn.edge_buckets(offsets, minutes=30, edge_min=5)

    assert first == [0, 1, 2]
    assert last == [4, 5, 6]


def test_edge_buckets_overlap_on_a_short_run() -> None:
    offsets = [0.0, 60.0, 120.0]

    first, last = bench_turn.edge_buckets(offsets, minutes=2, edge_min=5)

    assert first == [0, 1, 2]
    assert last == [0, 1, 2]


def test_parser_soak_defaults_to_none() -> None:
    args = bench_turn.build_parser().parse_args([])

    assert args.soak is None
    assert bench_turn.build_parser().parse_args(["--soak", "30"]).soak == 30


def _record(at: float, rss_mb: int, swap_used_mb: int) -> bench_turn.SoakRecord:
    power = bench_turn.PowerSample(
        cpu_power_mw=500,
        combined_power_mw=800,
        p_cluster_mhz=1000,
        e_cluster_mhz=1000,
        pressure="Nominal",
    )
    return bench_turn.SoakRecord(at, power, rss_mb, swap_used_mb)


def test_report_soak_buckets_rss_and_swap_at_the_edges(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    args = bench_turn.build_parser().parse_args(["--soak", "30"])
    records = [
        _record(0.0, 1500, 100),
        _record(120.0, 1600, 110),
        _record(1560.0, 2400, 300),
        _record(1680.0, 2500, 310),
    ]

    bench_turn.report_soak(cfg, args, [], records, UsageLedger())

    lines = capsys.readouterr().out.splitlines()
    rss, rss_first, rss_last = [line for line in lines if line.startswith("process rss")]
    swap, swap_first, swap_last = [line for line in lines if line.startswith("swap used")]
    assert "n=  4 min=  1500MB median=  2000MB max=  2500MB" in rss
    assert rss_first.startswith("process rss, first 5 min")
    assert "n=  2 min=  1500MB median=  1550MB max=  1600MB" in rss_first
    assert rss_last.startswith("process rss, last 5 min")
    assert "n=  2 min=  2400MB median=  2450MB max=  2500MB" in rss_last
    assert "n=  4 min=   100MB median=   205MB max=   310MB" in swap
    assert swap_first.startswith("swap used, first 5 min")
    assert "n=  2 min=   100MB median=   105MB max=   110MB" in swap_first
    assert swap_last.startswith("swap used, last 5 min")
    assert "n=  2 min=   300MB median=   305MB max=   310MB" in swap_last


def _prompt_with_history() -> TurnPrompt:
    return TurnPrompt(
        system="s",
        history=[
            Message(role="user", content="teach me ppo"),
            Message(role="assistant", content="PPO clips."),
        ],
        user_text="why does it clip the ratio instead of using it directly",
    )


def test_capture_writes_one_ascii_json_line_per_turn(tmp_path) -> None:
    path = tmp_path / "replies.jsonl"
    capture = bench_turn.Capture(path, "2026-09-16T10:00:00")

    capture.write(1, 0, False, 120)
    capture.write(4, 3, True, 80)

    raw = path.read_bytes()
    assert raw.isascii()
    lines = raw.decode().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert list(first) == ["started", "turn", "utterance", "sample", "reply_chars"]
    assert list(second) == list(first)
    assert first["started"] == "2026-09-16T10:00:00"
    assert first["turn"] == 1 and first["utterance"] == 0 and first["sample"] is False
    assert second["turn"] == 4 and second["utterance"] == 3 and second["sample"] is True
    assert first["reply_chars"] == 120
    assert second["reply_chars"] == 80


async def test_a_planner_stream_is_attributed_to_the_planner_ledger() -> None:
    usage = TurnUsage(prompt_tokens=10, completion_tokens=20)
    chunk = TurnChunk(kind="tool_call", text="{}", tool_call_id="c", tool_name="write_plan")
    stream = ScriptedStream([chunk], usage=usage, first_chunk_ms=12)
    reasoning = bench_turn.MeteredReasoning(ScriptedReasoning([stream]))
    record = reasoning.begin()
    metered = reasoning.start_turn(_prompt_with_history(), tools=PLAN_TOOLS, tool_choice="required")
    async for _ in metered:
        pass
    assert reasoning.ledgers["planner"].turns == 1
    assert reasoning.ledgers["planner"].total.prompt_tokens == 10
    assert reasoning.ledgers["visual"].turns == 0
    assert record.planner_usage == usage and record.planner_first_chunk_ms == 12
    assert record.visual_usage is None


def test_the_harness_imports_nothing_from_the_brief_or_the_phase_machine() -> None:
    source = Path(bench_turn.__file__).read_text()
    for module in ("tutor.brief", "tutor.pedagogy"):
        assert module not in source, module
    for name in ("OUTCOME_MARKER", "TAIL_LIMIT", "OutcomeSplitter", "BriefSplitter"):
        assert name not in source, name
    imported = {
        (node.module, alias.name)
        for node in ast.walk(ast.parse(Path(__file__).read_text()))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not {module for module, _ in imported} & {"tutor.brief", "tutor.pedagogy"}
    assert ("tutor.session", "OutcomeSplitter") not in imported


async def test_the_report_counts_the_scene_builds_in_the_visual_ledger(capsys) -> None:
    call = TurnChunk(kind="tool_call", text="{}", tool_call_id="c", tool_name=SCENE_TOOL)
    stream = ScriptedStream([call], usage=TurnUsage(prompt_tokens=30, completion_tokens=40))
    reasoning = bench_turn.MeteredReasoning(ScriptedReasoning([stream]))
    async for _ in reasoning.start_turn(PROMPT, tools=SCENE_TOOLS, tool_choice="required"):
        pass
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    sample = bench_turn.Sample(
        first_sound_ms=900,
        substance_ms=900,
        first_content_delta_ms=600,
        stages=0,
        silent=False,
        audio_ms=4000,
        voice_usd=0.0001,
        utterance=0,
    )

    bench_turn.report(cfg, bench_turn.parse_args([]), [sample], reasoning)

    lines = capsys.readouterr().out.splitlines()
    (visual,) = [line for line in lines if line.startswith("visual turns=")]
    assert visual.startswith("visual turns=1 prompt_tokens=30 completion_tokens=40 ")
    (voice,) = [line for line in lines if line.startswith("voice turns=")]
    assert voice.startswith("voice turns=0 ")
    assert any(line.startswith("turns=1 ") for line in lines)


STATE = LessonState()
STATE.adopt(LESSON)
WATCHED_STATE = LessonState()
WATCHED_STATE.adopt(WATCHED)


def state_at(scene: int, step: int) -> LessonState:
    state = LessonState()
    state.adopt(LESSON)
    revision = 0
    for n in range(1, scene + 1):
        assert state.scene_tag(n) is None
        revision += 1
        state.acknowledge(
            LessonAck(
                epoch=1,
                barrier=0,
                cue_id=state.sent[0].cue_id,
                outcome="fired",
                reason=None,
                scene_id=LESSON.scenes[n - 1].id,
                step=1,
                revision=revision,
            )
        )
    for n in range(2, step + 1):
        state.begin_turn(True)
        assert state.step_tag(n) is None
        revision += 1
        state.acknowledge(
            LessonAck(
                epoch=1,
                barrier=0,
                cue_id=state.sent[0].cue_id,
                outcome="fired",
                reason=None,
                scene_id=LESSON.scenes[scene - 1].id,
                step=n,
                revision=revision,
            )
        )
    return state


def scenario(
    group: str, target: Cursor, sequences: list[list[str]], **fields: object
) -> "bench_turn.Scenario":
    body = {
        "id": f"{group}-1",
        "group": group,
        "setup": [],
        "target": target.model_dump(),
        "asked": [],
        "learner": None if group == "opening" else "go on",
        "sequences": sequences,
        "question": False,
        "terms": [],
        "reveal_forbidden": [],
        "requires_scene": None,
        "forbids_scene": False,
        **fields,
    }
    return bench_turn.Scenario.model_validate(body)


def test_classify_tags_reads_validity_drops_placement_and_leaks() -> None:
    reply = "<scene 1>The ratio compares two policies. <step 2>At one action<step 3> it is one. <set lr 0.9>"
    spoken = ["The ratio compares two policies.", "At one action it is one."]
    read = bench_turn.classify_tags(reply, spoken, WATCHED_STATE, False, sentence_rule=False)
    assert read == bench_turn.TagRead(
        tags=["scene 1", "step 2", "step 3", "set"],
        valid=False,
        dropped={"set:unsupported": 1},
        placement_errors=1,
        leaked=0,
        cues=["scene 1", "step 2", "step 3"],
        cue_placement_errors=1,
    )
    assert WATCHED_STATE.sent == []


def test_a_step_tag_naming_the_current_step_is_neither_listed_nor_dropped() -> None:
    reply = "<scene 1> <step 1>Three bars. <step 2>Then one moves."
    read = bench_turn.classify_tags(reply, [], WATCHED_STATE, False, sentence_rule=True)
    assert (read.tags, read.valid, read.dropped) == (["scene 1", "step 2"], True, {})
    assert read.placement_errors == 0
    held = bench_turn.classify_tags(
        "<step 2>Then one moves <step 2>and settles.", [], state_at(1, 1), True, sentence_rule=True
    )
    assert (held.tags, held.valid, held.dropped) == (["step 2"], True, {})
    assert held.placement_errors == 1
    assert (held.cues, held.cue_placement_errors) == (["step 2"], 0)
    below = bench_turn.classify_tags(
        "<step 3>It settles. <step 2>Again.", [], state_at(1, 1), True, sentence_rule=True
    )
    assert (below.tags, below.valid, below.dropped) == (
        ["step 3", "step 2"],
        False,
        {"step:not_rising": 1},
    )
    assert below.cues == ["step 3"]


def test_a_tag_after_a_comma_is_misplaced_only_under_the_sentence_rule() -> None:
    reply = "<scene 1>The ratio, <step 2>compared at one action."
    spoken = ["The ratio,", "compared at one action."]
    assert (
        bench_turn.classify_tags(reply, spoken, STATE, False, sentence_rule=False).placement_errors
        == 0
    )
    assert (
        bench_turn.classify_tags(reply, spoken, STATE, False, sentence_rule=True).placement_errors
        == 1
    )


def test_placement_is_read_where_the_reply_wrote_each_tag() -> None:
    reply = '<scene 1><step 2>It falls, <draw t = label "x > 1">then rises<step 2> again.'
    assert (
        bench_turn.classify_tags(reply, [], STATE, False, sentence_rule=False).placement_errors == 1
    )
    assert (
        bench_turn.classify_tags(reply, [], STATE, False, sentence_rule=True).placement_errors == 2
    )


@pytest.mark.parametrize(
    ("reply", "cues", "strict", "seen"),
    [
        pytest.param(
            "<step 2>The band sits at 0.8 and 1.2. Below it<step 1> the ratio is clipped.",
            ["step 2"],
            1,
            0,
            id="refused-mid-sentence",
        ),
        pytest.param(
            "<step 2>The band sits<step 3> at 0.8 and 1.2.",
            ["step 2", "step 3"],
            1,
            1,
            id="cue-mid-sentence",
        ),
        pytest.param(
            "<step 2>The band sits<step 1><step 3> at 0.8 and 1.2.",
            ["step 2", "step 3"],
            1,
            1,
            id="cue-behind-a-refused-tag-mid-sentence",
        ),
    ],
)
def test_placement_as_seen_is_read_where_each_cue_lands_in_the_speech(
    reply: str, cues: list[str], strict: int, seen: int
) -> None:
    read = bench_turn.classify_tags(reply, [], state_at(2, 1), True, sentence_rule=True)
    assert read.cues == cues
    assert (read.placement_errors, read.cue_placement_errors) == (strict, seen)


def test_any_tag_text_in_the_spoken_stream_is_a_leak() -> None:
    read = bench_turn.classify_tags(
        "<scene 1>Hi.", ["<scene 1>Hi.", "<set lr", "<STEP 2>"], STATE, False, sentence_rule=False
    )
    assert read.leaked == 3


def test_closing_tag_text_in_the_spoken_stream_is_a_leak() -> None:
    read = bench_turn.classify_tags(
        "<step 3>Why is it dishonest?</step 3></scene 3>",
        ["Why is it dishonest?</step 3></scene 3>", "</Draw>"],
        STATE,
        False,
        sentence_rule=False,
    )
    assert (read.tags, read.leaked) == (["step 3"], 3)


def test_a_malformed_tag_is_counted_under_its_name_alone() -> None:
    read = bench_turn.classify_tags(
        "<Step 2>Hi. <step2>There.", ["Hi.", "There."], STATE, False, sentence_rule=False
    )
    assert read.tags == ["step", "step"] and read.dropped == {"step:malformed": 2}
    assert read.cues == []


def test_the_asking_step_is_refused_without_learner_text_and_taken_with_it() -> None:
    reply = "<step 2>They agree at one."
    silent = bench_turn.classify_tags(reply, [], state_at(1, 1), False, sentence_rule=True)
    assert (silent.tags, silent.valid, silent.dropped) == (
        ["step 2"],
        False,
        {"step:not_answered": 1},
    )
    spoke = bench_turn.classify_tags(reply, [], state_at(1, 1), True, sentence_rule=True)
    assert (spoke.tags, spoke.valid, spoke.dropped) == (["step 2"], True, {})


def test_a_scenario_takes_learner_text_from_its_case() -> None:
    reply = "<step 2>They agree at one."
    spoken = ["They agree at one."]
    answered = scenario("answer", Cursor(scene=1, step=1), [["step 2"]])
    unprompted = scenario("answer", Cursor(scene=1, step=1), [["step 2"]], learner=None)
    state = state_at(1, 1)
    taken = bench_turn.evaluate_scenario(reply, spoken, state, answered, sentence_rule=True)
    refused = bench_turn.evaluate_scenario(reply, spoken, state, unprompted, sentence_rule=True)
    assert (taken.tags_valid, taken.dropped, taken.passed) == (True, {}, True)
    assert (refused.tags, refused.tags_valid, refused.passed) == (["step 2"], False, False)
    assert refused.dropped == {"step:not_answered": 1}
    assert state.sent == []


QUESTION = {"question": True, "terms": ["ratio", "agree"], "reveal_forbidden": ["step 2"]}


@pytest.mark.parametrize(
    ("case", "at", "reply", "spoken", "failing"),
    [
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2", "step 3"]]),
            (2, 1),
            "<step 2>The band sits at 0.8 and 1.2. <step 3>Outside it the objective is flat.",
            ["The band sits at 0.8 and 1.2.", "Outside it the objective is flat."],
            None,
            id="progress",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2", "step 3"]]),
            (2, 1),
            "<step 3>Outside the band it is flat.",
            ["Outside the band it is flat."],
            ("progress_ok", False),
            id="early-tag",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2", "step 3"]]),
            (2, 1),
            "<step 2>The band sits<step 3> at 0.8 and 1.2.",
            ["The band sits", "at 0.8 and 1.2."],
            ("placement_errors", 1),
            id="mid-phrase",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2"]]),
            (2, 1),
            "<step 2>The band sits at 0.8 and 1.2.",
            ["<step 2>The band sits at 0.8 and 1.2."],
            ("leaked", 1),
            id="leak",
        ),
        pytest.param(
            scenario("question", Cursor(scene=1, step=1), [[]], **QUESTION),
            (1, 1),
            "Where do the two policies agree, and what is the ratio there?",
            ["Where do the two policies agree, and what is the ratio there?"],
            None,
            id="question-only",
        ),
        pytest.param(
            scenario("question", Cursor(scene=1, step=1), [[]], **QUESTION),
            (1, 1),
            "<step 2>They agree where the ratio is one. Where do they agree, and what is the ratio?",
            ["They agree where the ratio is one.", "Where do they agree, and what is the ratio?"],
            ("reveal_ok", False),
            id="revealed-before-the-question",
        ),
        pytest.param(
            scenario("question", Cursor(scene=1, step=1), [[]], **QUESTION),
            (1, 1),
            "What do you think?",
            ["What do you think?"],
            ("asked", False),
            id="question-without-its-terms",
        ),
        pytest.param(
            scenario("boundary", Cursor(scene=1, step=3), [["scene 2"]], requires_scene=2),
            (1, 3),
            "That is the ratio. <scene 2>Now the clip bounds it.",
            ["That is the ratio.", "Now the clip bounds it."],
            None,
            id="boundary",
        ),
        pytest.param(
            scenario("boundary", Cursor(scene=1, step=3), [["scene 2"]], requires_scene=2),
            (1, 3),
            "That is the ratio.",
            ["That is the ratio."],
            ("scene_ok", False),
            id="boundary-not-crossed",
        ),
        pytest.param(
            scenario("tangent", Cursor(scene=1, step=3), [[]], forbids_scene=True),
            (1, 3),
            "Good question. <scene 2>Now the clip.",
            ["Good question.", "Now the clip."],
            ("scene_ok", False),
            id="tangent-opens-a-scene",
        ),
        pytest.param(
            scenario("opening", Cursor(), [["scene 1"]]),
            (0, 0),
            "<scene 1>Start with the ratio.",
            ["Start with the ratio."],
            None,
            id="opening",
        ),
        pytest.param(
            scenario("opening", Cursor(), [["scene 1"]]),
            (0, 0),
            "Start with the ratio.",
            ["Start with the ratio."],
            ("progress_ok", False),
            id="opening-without-its-scene",
        ),
    ],
)
def test_a_scenario_passes_only_when_every_applicable_check_holds(
    case: "bench_turn.Scenario",
    at: tuple[int, int],
    reply: str,
    spoken: list[str],
    failing: tuple[str, object] | None,
) -> None:
    state = state_at(*at)
    result = bench_turn.evaluate_scenario(reply, spoken, state, case, sentence_rule=False)
    assert result.passed is (failing is None)
    if failing is not None:
        field, value = failing
        assert getattr(result, field) == value
    assert result.opening is (case.group == "opening")
    assert state.sent == []


@pytest.mark.parametrize(
    ("case", "at", "reply", "spoken", "failing", "seen_tags", "seen_passed"),
    [
        pytest.param(
            scenario("opening", Cursor(), [["scene 1"]]),
            (0, 0),
            "<scene 1>Start with the ratio. <step 2>They agree at one.",
            ["Start with the ratio.", "They agree at one."],
            ("tags", ["scene 1", "step 2"]),
            ["scene 1"],
            True,
            id="asking-step-with-no-learner-text",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2"]]),
            (2, 1),
            "<step 2>The band sits at 0.8 and 1.2. <step 1>Back on the axis.",
            ["The band sits at 0.8 and 1.2.", "Back on the axis."],
            ("dropped", {"step:not_rising": 1}),
            ["step 2"],
            True,
            id="not-rising",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2"]]),
            (2, 1),
            "<step 1>The axis. <Step 2>The band. <step 2>It sits at 0.8 and 1.2.",
            ["The axis.", "The band.", "It sits at 0.8 and 1.2."],
            ("tags", ["step", "step 2"]),
            ["step 2"],
            True,
            id="malformed-and-repeat",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2"]]),
            (2, 1),
            "<step 2>The band sits at 0.8 and 1.2. Below it<step 1> the ratio is clipped.",
            ["The band sits at 0.8 and 1.2.", "Below it the ratio is clipped."],
            ("placement_errors", 1),
            ["step 2"],
            True,
            id="refused-mid-sentence",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2", "step 3"]]),
            (2, 1),
            "<step 2>The band sits<step 3> at 0.8 and 1.2.",
            ["The band sits", "at 0.8 and 1.2."],
            ("placement_errors", 1),
            ["step 2", "step 3"],
            False,
            id="cue-mid-sentence",
        ),
        pytest.param(
            scenario("progress", Cursor(scene=2, step=1), [["step 2"]]),
            (2, 1),
            "<step 2>The band sits at 0.8 and 1.2.",
            ["<step 2>The band sits at 0.8 and 1.2."],
            ("leaked", 1),
            ["step 2"],
            False,
            id="leak",
        ),
        pytest.param(
            scenario("answer", Cursor(scene=1, step=1), [["step 2"]], learner=None),
            (1, 1),
            "<step 2>They agree at one.",
            ["They agree at one."],
            ("dropped", {"step:not_answered": 1}),
            [],
            False,
            id="required-tag-refused",
        ),
        pytest.param(
            scenario("question", Cursor(scene=1, step=1), [[]], **QUESTION),
            (1, 1),
            "Think about where they agree and what the ratio is there.",
            ["Think about where they agree and what the ratio is there."],
            ("asked", False),
            [],
            False,
            id="not-asked",
        ),
        pytest.param(
            scenario("boundary", Cursor(), [["scene 2"], []], requires_scene=2),
            (0, 0),
            "Start here. <scene 2>Now the clip bounds it.",
            ["Start here.", "Now the clip bounds it."],
            ("dropped", {"scene:not_next": 1}),
            [],
            False,
            id="required-scene-refused",
        ),
        pytest.param(
            scenario("tangent", Cursor(scene=1, step=3), [[], ["scene 2"]], forbids_scene=True),
            (1, 3),
            "Good question. <scene 2>Now the clip.",
            ["Good question.", "Now the clip."],
            ("scene_ok", False),
            ["scene 2"],
            False,
            id="forbidden-scene-cue",
        ),
        pytest.param(
            scenario("question", Cursor(scene=1, step=1), [[], ["step 2"]], **QUESTION),
            (1, 1),
            "<step 2>They agree where the ratio is one. Where do they agree, and what is the ratio?",
            ["They agree where the ratio is one.", "Where do they agree, and what is the ratio?"],
            ("reveal_ok", False),
            ["step 2"],
            False,
            id="reveal-forbidden-cue",
        ),
        pytest.param(
            scenario("question", Cursor(scene=1, step=1), [[]], **QUESTION, learner=None),
            (1, 1),
            "<step 2>They agree where the ratio is one. Where do they agree, and what is the ratio?",
            ["They agree where the ratio is one.", "Where do they agree, and what is the ratio?"],
            ("reveal_ok", False),
            [],
            True,
            id="reveal-forbidden-tag-refused",
        ),
    ],
)
def test_the_as_seen_pass_reads_only_the_tags_that_became_cues(
    case: "bench_turn.Scenario",
    at: tuple[int, int],
    reply: str,
    spoken: list[str],
    failing: tuple[str, object],
    seen_tags: list[str],
    seen_passed: bool,
) -> None:
    state = state_at(*at)
    result = bench_turn.evaluate_scenario(reply, spoken, state, case, sentence_rule=True)
    field, value = failing
    assert result.passed is False and getattr(result, field) == value
    assert (result.seen_tags, result.seen_passed) == (seen_tags, seen_passed)
    assert state.sent == []


def test_the_verdict_fails_under_the_floor_on_any_leak_or_missing_coverage() -> None:
    groups = {group: 5 for group in bench_turn.GROUPS}
    assert bench_turn.scenario_verdict(28, 0, 30, groups) == (
        "verdict: scenarios passed 28/30 against the floor of 27, leaked 0/30"
    )
    assert bench_turn.scenario_verdict(26, 0, 30, groups).endswith(", FAIL")
    assert bench_turn.scenario_verdict(30, 1, 30, groups).endswith(", FAIL")
    assert bench_turn.scenario_verdict(30, 0, 30, {**groups, "boundary": 0}).endswith(", FAIL")
    assert bench_turn.scenario_verdict(10, 0, 10, groups) == (
        "verdict: scenarios passed 10/10, not a qualification run"
    )


def test_warm_ups_repeat_non_opening_cases_and_every_case_is_measured() -> None:
    cases = [
        scenario(group, Cursor(), [[]], id=f"{group}-{n}")
        for group in reversed(bench_turn.GROUPS)
        for n in range(5)
    ]
    warmups, measured = bench_turn.run_order(cases)
    assert [case.id for case in measured] == [
        f"{group}-{n}" for group in bench_turn.GROUPS for n in range(5)
    ]
    assert [case.id for case in warmups] == ["progress-0", "progress-1", "progress-2"]


async def test_each_measured_case_is_written_once_and_the_warm_ups_are_not(tmp_path: Path) -> None:
    cases = [
        scenario(group, Cursor(), [[]], id=f"{group}-{n}")
        for group in bench_turn.GROUPS
        for n in range(5)
    ]
    ran: list[str] = []

    async def run_case(
        case: "bench_turn.Scenario",
    ) -> tuple[str, list[str], "bench_turn.ScenarioResult"]:
        ran.append(case.id)
        result = bench_turn.evaluate_scenario("Hi.", ["Hi."], STATE, case, sentence_rule=False)
        return "Hi.", ["Hi."], result

    out = tmp_path / "runs"
    bench_turn.prepare_out(out)
    results = await bench_turn.run_scenarios(cases, out, run_case)

    assert len(ran) == bench_turn.WARMUP + 30 and len(results) == 30
    assert sorted(path.name for path in out.iterdir()) == sorted(
        f"{case.id}.json" for case in cases
    )
    assert json.loads((out / "opening-0.json").read_text()) == {
        "reply": "Hi.",
        "spoken": ["Hi."],
        "result": results[0].model_dump(mode="json"),
    }


def test_a_non_empty_out_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "runs"
    bench_turn.prepare_out(out)
    assert out.is_dir()
    bench_turn.prepare_out(out)
    (out / "old.json").write_text("{}")
    with pytest.raises(FileExistsError):
        bench_turn.prepare_out(out)


def test_a_setup_that_misses_its_target_or_its_history_is_a_harness_error() -> None:
    case = scenario(
        "progress",
        Cursor(scene=1, step=2),
        [["step 3"]],
        setup=[
            {"learner": None, "reply": "<scene 1>Start with the ratio."},
            {"learner": "they agree at one", "reply": "<step 2>Right, the ratio is one there."},
        ],
    )
    history = [
        Message(role="user", content=OPENING_TEXT),
        Message(role="assistant", content="Start with the ratio."),
        Message(role="user", content="they agree at one"),
        Message(role="assistant", content="Right, the ratio is one there."),
    ]
    bench_turn.check_setup(state_at(1, 2).acked, history, case)
    with pytest.raises(bench_turn.HarnessError):
        bench_turn.check_setup(state_at(1, 1).acked, history, case)
    with pytest.raises(bench_turn.HarnessError):
        bench_turn.check_setup(state_at(1, 2).acked, history[:2], case)


async def test_setup_replies_the_plan_and_stub_builds_never_reach_the_model() -> None:
    measured = ScriptedStream(
        _spoken("The real reply."), usage=TurnUsage(prompt_tokens=5, completion_tokens=3)
    )
    inner = ScriptedReasoning([measured])
    reasoning = bench_turn.MeteredReasoning(inner, plan=LESSON, stub_scenes=True)
    reasoning.script(["<scene 1>Start with the ratio.", "<step 2>At one action."])
    build = planned_scene_prompt("PPO", LESSON.profile, LESSON.scenes[1], "light")

    plan = [c async for c in reasoning.start_turn(PROMPT, tools=PLAN_TOOLS, tool_choice="required")]
    stub = [c async for c in reasoning.start_turn(build, tools=SCENE_TOOLS, tool_choice="required")]
    first = [c.text async for c in reasoning.start_turn(PROMPT)]
    second = [c.text async for c in reasoning.start_turn(PROMPT)]
    real = [c.text async for c in reasoning.start_turn(PROMPT)]

    assert plan == [
        TurnChunk(
            kind="tool_call",
            text=LESSON.model_dump_json(),
            tool_call_id="call-plan",
            tool_name=PLAN_TOOL,
        )
    ]
    (drafted,) = stub
    assert drafted.tool_name == SCENE_TOOL and len(json.loads(drafted.text)["steps"]) == 3
    assert first == ["<scene ", "1>Start ", "with ", "the ", "ratio."]
    assert "".join(second) == "<step 2>At one action."
    assert real == ["The real reply."]
    assert inner.prompts == [PROMPT]
    assert reasoning.ledger.turns == 1 and reasoning.ledgers["voice"].turns == 1
    assert reasoning.ledgers["planner"].turns == 0 and reasoning.ledgers["visual"].turns == 0


def test_metering_is_armed_before_the_loop_runs() -> None:
    boot = inspect.getsource(bench_turn.Bench.boot)
    assert boot.index("reasoning.begin(") < boot.index("loop.run()")


def test_the_loop_plans_exactly_with_a_plan_file_or_the_connect_run() -> None:
    assert bench_turn.planned(bench_turn.parse_args([])) is False
    assert bench_turn.planned(bench_turn.parse_args(["--silent-synth"])) is False
    assert bench_turn.planned(bench_turn.parse_args(["--plan", "p.json"])) is True
    assert bench_turn.planned(bench_turn.parse_args(["--connect"])) is True
    assert "planned=planned" in inspect.getsource(bench_turn.Bench.boot)


def test_connect_times_scene_one_by_the_plans_first_scene_id() -> None:
    checks = [(1.0, "clip", True), (2.0, "ratio", False), (3.0, "ratio", True)]
    assert bench_turn.first_scene_checked(checks, LESSON) == 3.0
    assert bench_turn.first_scene_checked(checks[:2], LESSON) is None


def test_the_connect_report_keeps_a_sample_at_its_bound_with_its_nulls(capsys) -> None:
    cfg = Settings(_env_file=None, reasoning_api_base="http://x", reasoning_api_key="k")
    samples = [
        bench_turn.ConnectSample(
            plan_ms=900, plan_valid=True, first_audio_frame_ms=2000, scene_checked_ms=40000
        ),
        bench_turn.ConnectSample(
            plan_ms=None, plan_valid=False, first_audio_frame_ms=None, scene_checked_ms=None
        ),
    ]

    bench_turn.report_connect(cfg, samples, 420.0)

    lines = capsys.readouterr().out.splitlines()
    assert any("builder=interim-html" in line and "simulated page" in line for line in lines)
    assert (
        "connect to first outbound synthesized audio frame: observed 1 missing 1 "
        "median=2000ms p95=2000ms max=2000ms"
    ) in lines
    assert (
        "connect to scene checked (interim HTML builder, simulated page): observed 1 missing 1 "
        "median=40000ms p95=40000ms max=40000ms"
    ) in lines
    assert "plan valid 1/2" in lines
    assert lines[-1] == "measurement complete 1/2"


@pytest.mark.parametrize(
    "flags",
    [
        ["--connect", "--silent-synth"],
        ["--connect", "--stub-scenes"],
        ["--connect", "--plan", "p.json"],
        ["--connect", "--scenarios", "s.json"],
        ["--scenarios", "s.json"],
        ["--scenarios", "s.json", "--plan", "p.json"],
        ["--scenarios", "s.json", "--plan", "p.json", "--stub-scenes"],
        ["--out", "o"],
        ["--soak", "30", "--connect"],
        [
            "--soak",
            "30",
            "--scenarios",
            "s.json",
            "--plan",
            "p.json",
            "--stub-scenes",
            "--out",
            "o",
        ],
        ["--soak", "30", "--plan", "p.json"],
        ["--soak", "30", "--stub-scenes"],
        ["--soak", "30", "--silent-synth"],
    ],
)
def test_the_parser_refuses_combinations_that_would_mislabel_a_measurement(
    flags: list[str],
) -> None:
    with pytest.raises(SystemExit):
        bench_turn.parse_args(flags)


def test_a_stub_scene_has_the_count_the_planned_prompt_asks_for() -> None:
    for count in (3, 4, 5):
        prompt = TurnPrompt(
            system="s", user_text=f"Steps, in this order, exactly {count}, one say line each:"
        )
        body = json.loads(bench_turn.stub_scene(prompt).text)
        assert len(body["steps"]) == count and body["html"]


REPORTED = bench_turn.Sample(
    first_sound_ms=900,
    substance_ms=900,
    first_content_delta_ms=600,
    stages=0,
    silent=False,
    audio_ms=4000,
    voice_usd=0.0001,
    utterance=0,
    scenario_id="question-0",
    group="question",
    tags=[],
    tags_valid=True,
    placement_errors=0,
    dropped={},
    progress_ok=True,
    asked=True,
    scene_ok=True,
    leaked=0,
    opening=False,
    scenario_pass=True,
    seen_pass=True,
)


def test_the_scenario_report_counts_each_check_over_its_own_cases(capsys) -> None:
    kept = REPORTED
    missed = kept.model_copy(
        update={
            "scenario_id": "boundary-0",
            "group": "boundary",
            "tags": ["step 2"],
            "tags_valid": False,
            "dropped": {"step:not_rising": 1},
            "asked": None,
            "scene_ok": False,
            "leaked": 1,
            "scenario_pass": False,
            "seen_pass": False,
        }
    )

    bench_turn.report_scenarios([kept, missed], sentence_rule=True)

    lines = capsys.readouterr().out.splitlines()
    assert "tags valid and placed 1/2" in lines
    assert "asked, proxy 1/1" in lines
    assert "scene tags as required 0/1" in lines
    assert "tags dropped: step:not_rising=1" in lines
    assert "boundary passed 0/1" in lines
    assert lines[-2:] == [
        (
            "verdict as seen: scenarios passed 1/2 on the cues the page receives, "
            "not a qualification run"
        ),
        "verdict: scenarios passed 1/2, not a qualification run",
    ]


SEEN = "verdict as seen: scenarios passed {}/{} on the cues the page receives"


@pytest.mark.parametrize(
    ("n", "strict", "seen", "leaks", "spread", "verdicts"),
    [
        pytest.param(
            30,
            27,
            29,
            0,
            6,
            [
                SEEN.format(29, 30) + ", against the floor of 27, leaked 0/30",
                "verdict: scenarios passed 27/30 against the floor of 27, leaked 0/30",
            ],
            id="both-pass",
        ),
        pytest.param(
            30,
            24,
            26,
            0,
            6,
            [
                SEEN.format(26, 30) + ", against the floor of 27, leaked 0/30, FAIL",
                "verdict: scenarios passed 24/30 against the floor of 27, leaked 0/30, FAIL",
            ],
            id="both-under-the-floor",
        ),
        pytest.param(
            30,
            24,
            28,
            0,
            6,
            [
                SEEN.format(28, 30) + ", against the floor of 27, leaked 0/30",
                "verdict: scenarios passed 24/30 against the floor of 27, leaked 0/30, FAIL",
            ],
            id="only-strict-under-the-floor",
        ),
        pytest.param(
            30,
            28,
            28,
            1,
            6,
            [
                SEEN.format(28, 30) + ", against the floor of 27, leaked 1/30, FAIL",
                "verdict: scenarios passed 28/30 against the floor of 27, leaked 1/30, FAIL",
            ],
            id="a-leak",
        ),
        pytest.param(
            30,
            30,
            30,
            0,
            5,
            [
                SEEN.format(30, 30) + ", against the floor of 27, leaked 0/30, FAIL",
                "verdict: scenarios passed 30/30 against the floor of 27, leaked 0/30, FAIL",
            ],
            id="a-group-not-covered",
        ),
        pytest.param(
            12,
            9,
            11,
            0,
            6,
            [
                SEEN.format(11, 12) + ", not a qualification run",
                "verdict: scenarios passed 9/12, not a qualification run",
            ],
            id="short-run",
        ),
    ],
)
def test_the_as_seen_verdict_comes_just_before_the_strict_verdict(
    capsys, n: int, strict: int, seen: int, leaks: int, spread: int, verdicts: list[str]
) -> None:
    samples = [
        REPORTED.model_copy(
            update={
                "group": bench_turn.GROUPS[i % spread],
                "scenario_pass": i < strict,
                "seen_pass": i < seen,
                "leaked": int(i >= n - leaks),
            }
        )
        for i in range(n)
    ]

    bench_turn.report_scenarios(samples, sentence_rule=True)

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 9 + len(bench_turn.GROUPS)
    assert lines[-2:] == verdicts


def test_a_stub_scene_says_the_planned_prompts_show_lines() -> None:
    steps = [*LESSON.scenes[1].steps[:2], Step(show="x" * 300)]
    scene = LESSON.scenes[1].model_copy(update={"steps": steps})
    prompt = planned_scene_prompt("PPO", LESSON.profile, scene, "light", "write 3 say lines")

    body = json.loads(bench_turn.stub_scene(prompt).text)

    assert body["steps"] == [
        "The ratio axis from 0.5 to 2.0",
        "The clip band at 0.8 and 1.2",
        "x" * bench_turn.SAY_LINE_CHARS,
    ]
    SceneDraft.model_validate(body)


def test_a_setup_short_of_its_scene_settles_and_then_misses_its_target() -> None:
    case = scenario(
        "progress",
        Cursor(scene=1, step=1),
        [["step 2"]],
        setup=[{"learner": None, "reply": "<scene1>Start with the ratio."}],
    )
    history = [
        Message(role="user", content=OPENING_TEXT),
        Message(role="assistant", content="Start with the ratio."),
    ]
    state = state_at(0, 0)
    state.committed = {"ratio", "clip"}
    assert not bench_turn.setup_settled(state)

    state.built["ratio"] = BuiltScene(
        scene_id="ratio", version=1, say=["a", "b", "c"], html="<p>ratio</p>"
    )

    assert bench_turn.setup_settled(state)
    with pytest.raises(bench_turn.HarnessError, match="progress-1"):
        bench_turn.check_setup(state.acked, history, case)


def test_a_setup_cue_still_waiting_on_its_ack_is_not_settled() -> None:
    state = state_at(1, 1)
    state.built["ratio"] = BuiltScene(
        scene_id="ratio", version=1, say=["a", "b", "c"], html="<p>ratio</p>"
    )
    assert bench_turn.setup_settled(state)

    state.begin_turn(True)
    assert state.step_tag(2) is None

    assert not bench_turn.setup_settled(state)
