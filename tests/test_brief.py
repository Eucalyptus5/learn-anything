import logging

import pytest

from tutor.brief import (
    BRIEF_END,
    BRIEF_LIMIT,
    BRIEF_MARKER,
    BriefSplitter,
    VisualBrief,
    parse_brief,
)
from tutor.session import OUTCOME_MARKER, OutcomeSplitter

BODY = (
    '{"kind": "app", "title": "Clipped objective", '
    '"show": "the surrogate vs the ratio for A>0 and A<0, epsilon 0.2"}'
)
HEAD = BRIEF_MARKER + BODY + BRIEF_END
SPEECH = "PPO clips the ratio. Past one plus epsilon the objective is flat."
QUESTION = "Where does the clipped objective go flat?"
OUTCOME = OUTCOME_MARKER + '{"signal": "covered", "settling_positions": []}'
LONG_SHOW = "the surrogate against the ratio " * 18
LONG_BODY = '{"kind": "app", "title": "Clipped objective", "show": "' + LONG_SHOW + 'ratio"}'


def feed_all(splitter: BriefSplitter, deltas: list[str]) -> str:
    out = "".join(splitter.feed(d) for d in deltas)
    return out + splitter.finish()


def pieces(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def test_a_head_is_parsed_and_the_text_after_it_is_released() -> None:
    s = BriefSplitter()
    assert feed_all(s, [HEAD, "\n", SPEECH]) == SPEECH
    assert s.brief == VisualBrief(
        kind="app",
        title="Clipped objective",
        show="the surrogate vs the ratio for A>0 and A<0, epsilon 0.2",
    )


def test_a_marker_split_across_deltas_is_still_parsed() -> None:
    s = BriefSplitter()
    parts = [HEAD[:3], HEAD[3:20], HEAD[20:-4], HEAD[-4:], " ", SPEECH]
    assert feed_all(s, parts) == SPEECH
    assert s.brief is not None and s.brief.kind == "app"


def test_text_without_a_head_passes_through_untouched() -> None:
    s = BriefSplitter()
    assert feed_all(s, ["PPO ", "clips ", "the ratio."]) == "PPO clips the ratio."
    assert s.brief is None


def test_a_prefix_of_the_marker_is_held_then_released_when_it_is_not_the_marker() -> None:
    s = BriefSplitter()
    assert s.feed("<vis") == ""
    assert s.feed("ible light") == "<visible light"
    assert s.finish() == ""
    assert s.brief is None


def test_leading_whitespace_before_the_head_is_ignored() -> None:
    s = BriefSplitter()
    assert feed_all(s, ["\n  ", HEAD, SPEECH]) == SPEECH
    assert s.brief is not None


def test_a_head_without_the_end_tag_closes_on_its_last_brace() -> None:
    s = BriefSplitter()
    assert feed_all(s, [BRIEF_MARKER + BODY, "\n", QUESTION]) == QUESTION
    assert s.brief is not None and s.brief.kind == "app"


def test_a_short_reply_with_no_end_tag_keeps_its_question_and_its_outcome() -> None:
    head = BriefSplitter()
    outcome = OutcomeSplitter()
    spoken = ""
    for delta in [*pieces(BRIEF_MARKER + BODY + "\n" + QUESTION + " ", 7), OUTCOME]:
        spoken += outcome.feed(head.feed(delta))
    spoken += outcome.feed(head.finish())
    tail, parsed = outcome.finish()
    assert (spoken + tail).strip() == QUESTION
    assert head.brief is not None and head.brief.title == "Clipped objective"
    assert parsed.signal == "covered"


def test_a_late_end_tag_split_across_deltas_is_swallowed() -> None:
    s = BriefSplitter()
    assert feed_all(s, [BRIEF_MARKER + BODY, " </vis", "ual>", " ", SPEECH]) == SPEECH
    assert s.brief is not None


def test_a_nested_brace_inside_a_string_does_not_close_the_head() -> None:
    s = BriefSplitter()
    body = (
        '{"kind": "app", "title": "Sets", "show": "the set {a, b} and the escaped quote \\" here"}'
    )
    assert feed_all(s, [BRIEF_MARKER + body + " " + SPEECH]) == SPEECH
    assert s.brief is not None and s.brief.title == "Sets"


def test_a_head_past_the_limit_never_leaks_its_tail_to_speech(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert len(LONG_BODY) > BRIEF_LIMIT
    s = BriefSplitter()
    with caplog.at_level(logging.INFO, logger="tutor.brief"):
        out = feed_all(s, [*pieces(BRIEF_MARKER + LONG_BODY + BRIEF_END + " " + SPEECH, 5)])
    assert out == SPEECH
    assert s.brief is None
    assert any(m.startswith("brief.oversized chars=") for m in caplog.messages)


def test_a_head_past_the_limit_with_no_end_tag_still_releases_the_speech() -> None:
    s = BriefSplitter()
    assert feed_all(s, [*pieces(BRIEF_MARKER + LONG_BODY + "\n" + SPEECH, 5)]) == SPEECH
    assert s.brief is None


def test_an_unterminated_head_swallows_the_reply(caplog: pytest.LogCaptureFixture) -> None:
    s = BriefSplitter()
    with caplog.at_level(logging.INFO, logger="tutor.brief"):
        out = feed_all(s, [BRIEF_MARKER + '{"show": "' + "x" * BRIEF_LIMIT, SPEECH])
    assert out == ""
    assert s.brief is None
    assert any(m.startswith("brief.unterminated chars=") for m in caplog.messages)


def test_a_malformed_head_yields_no_brief_and_speech_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    s = BriefSplitter()
    with caplog.at_level(logging.INFO, logger="tutor.brief"):
        out = feed_all(s, [BRIEF_MARKER + "not json" + BRIEF_END + SPEECH])
    assert out == SPEECH
    assert s.brief is None
    assert any(m.startswith("brief.unparsed chars=") for m in caplog.messages)


def test_kind_none_is_a_brief() -> None:
    s = BriefSplitter()
    head = BRIEF_MARKER + '{"kind": "none", "title": "", "show": ""}' + BRIEF_END
    assert feed_all(s, [head, SPEECH]) == SPEECH
    assert s.brief == VisualBrief(kind="none", title="", show="")


def test_finish_drops_an_open_head_and_releases_held_text() -> None:
    open_head = BriefSplitter()
    assert open_head.feed(BRIEF_MARKER + '{"kind"') == ""
    assert open_head.finish() == ""
    held = BriefSplitter()
    assert held.feed("<visu") == ""
    assert held.finish() == "<visu"


def test_only_the_first_head_counts() -> None:
    s = BriefSplitter()
    second = BRIEF_MARKER + '{"kind": "diagram", "title": "t", "show": "s"}' + BRIEF_END
    assert feed_all(s, [HEAD, " ", second, SPEECH]) == second + SPEECH
    assert s.brief is not None and s.brief.kind == "app"


def test_title_and_show_length_caps() -> None:
    assert parse_brief('{"kind": "app", "title": "' + "t" * 81 + '", "show": "s"}') is None
    assert parse_brief('{"kind": "app", "title": "t", "show": "' + "s" * 401 + '"}') is None
    assert parse_brief('{"kind": "app", "title": "t", "show": "s", "extra": 1}') is None
