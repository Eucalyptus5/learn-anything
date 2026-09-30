import logging
from collections.abc import AsyncIterator, Sequence

import pytest

from tutor.chunker import Scrubber, clause_chunks, spoken_text
from tutor.tags import (
    TAG_MAX_CHARS,
    TAG_NAMES,
    Marker,
    RawTag,
    TagSplitter,
    parse_marker,
    tag_name,
)


def merged(items: list[str | RawTag]) -> list[str | RawTag]:
    out: list[str | RawTag] = []
    for item in items:
        if isinstance(item, str) and out and isinstance(out[-1], str):
            out[-1] += item
        elif item != "":
            out.append(item)
    return out


def fed(*deltas: str) -> list[str | RawTag]:
    splitter = TagSplitter()
    out: list[str | RawTag] = []
    for delta in deltas:
        out.extend(splitter.feed(delta))
    out.extend(splitter.finish())
    return merged(out)


def spoken(items: list[str | RawTag]) -> str:
    return "".join(item for item in items if isinstance(item, str))


REPLIES = [
    "Look at the band, <step 2> which clips the ratio.",
    "Done. <scene 2> Here is the next picture, and <step 2> its first move.",
    'Watch this. <draw t = label "x > 1"> It holds. <set lr 0.9> Now it moves, <look t> there.',
    "x < 3 and a<b, so <step 3> the band holds. <clear t> Next one?",
    "It falls, <step 2> then rises. <step 3> And it stops.",
]


def test_the_names_and_the_cap_are_fixed() -> None:
    assert TAG_NAMES == (
        "step",
        "scene",
        "set",
        "point",
        "look",
        "orbit",
        "hand",
        "take",
        "draw",
        "clear",
    )
    assert TAG_MAX_CHARS == 240


def test_every_tag_name_splits_bare_or_with_arguments() -> None:
    for name in TAG_NAMES:
        assert fed(f"Done. <{name} x 1> Next.") == ["Done. ", RawTag(f"{name} x 1"), " Next."]
        assert fed(f"Done. <{name}> Next.") == ["Done. ", RawTag(name), " Next."]
        assert fed(f"Done. <{name}2> Next.") == ["Done. ", RawTag(f"{name}2"), " Next."]


def test_a_tag_split_across_deltas_is_held_then_released() -> None:
    assert fed("Look. <st", "ep 2", "> there") == ["Look. ", RawTag("step 2"), " there"]
    assert fed("<", "scene", " 1", "2>", "x") == [RawTag("scene 12"), "x"]
    splitter = TagSplitter()
    assert splitter.feed("Look. <step") == ["Look. "]
    assert splitter.feed(" 3") == []
    assert splitter.feed(">") == [RawTag("step 3")]
    assert splitter.feed(" on") == [" on"]


def test_every_reply_splits_the_same_when_fed_one_character_at_a_time() -> None:
    for reply in REPLIES:
        assert fed(*reply) == fed(reply), reply


def test_a_less_than_sign_that_starts_no_tag_passes_through() -> None:
    assert fed("x < 3 and a<b, so ", "y<", "z.") == ["x < 3 and a<b, so y<z."]
    assert fed("a <", " b") == ["a < b"]
    assert fed("the <stepping stones> path") == ["the <stepping stones> path"]
    assert fed("<b>bold</b>") == ["<b>bold</b>"]
    assert spoken(fed("a <<step 2> b")) == "a < b"


def test_a_close_inside_double_quotes_does_not_end_the_tag() -> None:
    body = 'draw t = label "a > b"'
    assert fed(f"Done. <{body}> Next.") == ["Done. ", RawTag(body), " Next."]
    assert fed('Done. <draw t = label "a ', '> b"> Next.') == ["Done. ", RawTag(body), " Next."]
    escaped = 'draw t = label "say \\"x > y\\" now"'
    assert fed(f"Done. <{escaped}> Next.") == ["Done. ", RawTag(escaped), " Next."]


def test_a_tag_past_the_cap_is_discarded_to_its_close_with_one_line(caplog) -> None:
    body = "draw t = label " + "x" * TAG_MAX_CHARS
    with caplog.at_level(logging.INFO, logger="tutor.tags"):
        whole = fed(f"Done. <{body}> Next.")
        split = fed("Done. <draw ", "x" * 300, ' "> more', '"> Next.')
    assert whole == ["Done.  Next."] and split == ["Done.  Next."]
    assert caplog.messages == [
        f"tag.unterminated chars={len(body) + 2}",
        f"tag.unterminated chars={6 + 300 + 8 + 2}",
    ]


def test_the_cap_counts_the_whole_tag_with_its_brackets(caplog) -> None:
    fits = "draw t = label " + "x" * (TAG_MAX_CHARS - 17)
    over = fits + "x"
    with caplog.at_level(logging.INFO, logger="tutor.tags"):
        assert fed(f"<{fits}>") == [RawTag(fits)]
        assert fed(f"<{fits}", ">") == [RawTag(fits)]
        assert fed(f"<{over}>") == []
        assert fed(f"<{over}", ">") == []
    assert caplog.messages == [f"tag.unterminated chars={TAG_MAX_CHARS + 1}"] * 2


def test_a_capitalised_or_newline_split_tag_is_split_out_and_malformed() -> None:
    assert fed("Done. <Step 2> Next.") == ["Done. ", RawTag("Step 2"), " Next."]
    assert fed("Done. <step\n2> Next.") == ["Done. ", RawTag("step\n2"), " Next."]
    assert fed("Done. <SCENE", " 2> Next.") == ["Done. ", RawTag("SCENE 2"), " Next."]
    assert fed("Done. <Scene-", "2> Next.") == ["Done. ", RawTag("Scene-2"), " Next."]
    assert parse_marker(RawTag("Step 2")) == "malformed"
    assert parse_marker(RawTag("step\n2")) == "malformed"
    assert parse_marker(RawTag("Draw t")) == "unsupported"


def test_anything_held_at_the_end_is_dropped_with_one_line(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="tutor.tags"):
        splitter = TagSplitter()
        assert splitter.feed("and so <step 4") == ["and so "]
        assert splitter.finish() == []
        prefix = TagSplitter()
        assert prefix.feed("then <sc") == ["then "]
        assert prefix.finish() == []
        skipped = TagSplitter()
        assert skipped.feed("so <draw " + "x" * 300) == ["so "]
        assert skipped.finish() == []
        assert TagSplitter().finish() == []
    assert caplog.messages == [
        "tag.unterminated chars=7",
        "tag.unterminated chars=3",
        "tag.unterminated chars=306",
    ]


def test_no_tag_text_is_ever_released_as_speech() -> None:
    for reply in REPLIES:
        text = spoken(fed(reply)) + spoken(fed(*reply))
        for name in TAG_NAMES:
            assert f"<{name}" not in text, (reply, name)
        assert ">" not in text.replace("x > 1", ""), reply


def test_parse_marker_reads_step_and_scene_and_names_every_other_reason() -> None:
    assert parse_marker(RawTag("step 3")) == Marker("step", 3)
    assert parse_marker(RawTag("scene 2")) == Marker("scene", 2)
    assert parse_marker(RawTag("scene 12")) == Marker("scene", 12)
    for text in (
        "step",
        "step 0",
        "step 03",
        "step 123",
        "step  3",
        "step 3 now",
        "step two",
        "scene -1",
        "scene 2 ",
        "scene",
        "step2",
        "Scene-2",
    ):
        assert parse_marker(RawTag(text)) == "malformed", text
    for name in TAG_NAMES[2:]:
        assert parse_marker(RawTag(f"{name} x")) == "unsupported", name
        assert parse_marker(RawTag(name)) == "unsupported", name


def test_a_tag_is_named_by_its_leading_letters_and_nothing_after_them() -> None:
    for text in ("step 2", "Step 2", "step\n2", "step2", "STEP", "step:x"):
        assert tag_name(RawTag(text)) == "step", text
    assert tag_name(RawTag("Scene-2")) == "scene"
    assert tag_name(RawTag("draw:the-learner-said-their-password-is-hunter2")) == "draw"
    assert tag_name(RawTag("sketch 2")) == "unknown"
    assert tag_name(RawTag("")) == "unknown"


async def test_a_reply_through_the_splitter_and_the_chunker_speaks_no_tag() -> None:
    splitter = TagSplitter()

    async def deltas() -> AsyncIterator[str | RawTag]:
        for delta in (
            "Watch the ratio, <st",
            "ep 2> it climbs past one, <set lr 0.",
            '9> and <draw t = label "x > 1"> then',
            " it stops.",
        ):
            for item in splitter.feed(delta):
                yield item
        for item in splitter.finish():
            yield item

    out = [item async for item in clause_chunks(spoken_text(deltas(), Scrubber()))]
    assert [item for item in out if isinstance(item, RawTag)] == [
        RawTag("step 2"),
        RawTag("set lr 0.9"),
        RawTag('draw t = label "x > 1"'),
    ]
    assert not any("<" in item or ">" in item for item in out if isinstance(item, str))


def test_a_tag_inside_a_sentence_waits_for_the_sentence_to_end(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="tutor.tags"):
        assert fed("It falls, <step 2> then rises. Next one.") == [
            "It falls,  then rises.",
            RawTag("step 2"),
            " Next one.",
        ]
        assert fed("It falls, <draw:secret> then rises.") == [
            "It falls,  then rises.",
            RawTag("draw:secret"),
        ]
    assert caplog.messages == ["tag.deferred name=step", "tag.deferred name=draw"]


def test_a_sentence_end_split_across_deltas_still_releases_the_tag() -> None:
    assert fed("It falls, <step 2> then", " rises.", " Next") == [
        "It falls,  then rises.",
        RawTag("step 2"),
        " Next",
    ]
    assert fed("It falls <step 2> and rises.<step 3> Next.") == [
        "It falls  and rises.",
        RawTag("step 2"),
        RawTag("step 3"),
        " Next.",
    ]


def test_a_tag_at_a_sentence_start_is_not_deferred(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="tutor.tags"):
        assert fed("Done. <step 2> Next.") == ["Done. ", RawTag("step 2"), " Next."]
        assert fed("<scene 1>Here we go.") == [RawTag("scene 1"), "Here we go."]
    assert caplog.messages == []


def test_deferred_tags_come_out_in_order_at_the_end_of_the_reply() -> None:
    assert fed("It falls <step 2> and <step 3> rises") == [
        "It falls  and  rises",
        RawTag("step 2"),
        RawTag("step 3"),
    ]


def test_a_deferred_tag_between_two_words_leaves_a_space() -> None:
    assert fed("The band<step 2>clips it.") == ["The band clips it.", RawTag("step 2")]
    assert fed(*"The band<step 2>clips it.") == ["The band clips it.", RawTag("step 2")]


def test_a_decimal_point_ends_no_sentence_and_a_question_mark_does() -> None:
    assert fed("Is it 0.5 <step 2> here? Yes.") == [
        "Is it 0.5  here?",
        RawTag("step 2"),
        " Yes.",
    ]


def test_a_sentence_may_end_inside_a_closing_quote_or_bracket() -> None:
    assert fed('He said "stop." <step 2> Next.') == [
        'He said "stop." ',
        RawTag("step 2"),
        " Next.",
    ]
    assert fed('It falls <step 2> and "rises." Next.') == [
        'It falls  and "rises."',
        RawTag("step 2"),
        " Next.",
    ]


LEAKING_TAIL = (
    "<step 3>An arrow carries those rollouts into the trainable pi theta box, stamped K equals 4 "
    "passes, meaning we plan to squeeze four gradient updates from this one batch. After pass 1, "
    "why is pass 2 already slightly dishonest?</step 3></scene 3>"
)


async def through_the_live_path(deltas: Sequence[str]) -> list[str | RawTag]:
    splitter = TagSplitter()

    async def items() -> AsyncIterator[str | RawTag]:
        for delta in deltas:
            for item in splitter.feed(delta):
                yield item
        for item in splitter.finish():
            yield item

    return [item async for item in clause_chunks(spoken_text(items(), Scrubber()))]


@pytest.mark.parametrize("deltas", [[LEAKING_TAIL], list(LEAKING_TAIL)], ids=["whole", "by-char"])
async def test_closing_tags_after_the_last_question_are_never_spoken(
    deltas: list[str], caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="tutor.tags"):
        out = await through_the_live_path(deltas)
    said = " ".join(item for item in out if isinstance(item, str))
    assert out[-1] == "After pass 1, why is pass 2 already slightly dishonest?"
    assert said.endswith("dishonest?")
    assert "<" not in said and ">" not in said
    assert "step 3" not in said and "scene 3" not in said
    assert [item for item in out if isinstance(item, RawTag)] == [RawTag("step 3")]
    assert caplog.messages == ["tag.closing name=step", "tag.closing name=scene"]


@pytest.mark.parametrize("args", ["", " 3"])
@pytest.mark.parametrize("name", TAG_NAMES)
def test_a_closing_tag_of_every_kind_is_dropped_whole_or_split(name: str, args: str) -> None:
    for spelled in (name, name.upper(), name.capitalize()):
        reply = f"Is it so?</{spelled}{args}> It holds</{spelled}{args}>then falls."
        assert fed(reply) == fed(*reply) == ["Is it so? It holds then falls."], reply
