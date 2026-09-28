import asyncio
from collections import Counter
from collections.abc import AsyncIterator

import pytest

from tutor.chunker import Scrubber, clause_chunks, split_clauses, spoken_text
from tutor.tags import RawTag


def _run(count: int, *marked: int, mark: str = ",") -> list[str]:
    return [f"w{i}{mark if i in marked else ''}" for i in range(1, count + 1)]


def test_splits_on_period_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses("This is a full sentence. ", max_words=40)
    assert clauses == ["This is a full sentence."]
    assert remainder == ""


def test_splits_on_question_mark_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses("Is this the right function? ", max_words=40)
    assert clauses == ["Is this the right function?"]
    assert remainder == ""


def test_splits_on_exclamation_mark_followed_by_whitespace() -> None:
    clauses, remainder = split_clauses("Watch out for that bug! ", max_words=40)
    assert clauses == ["Watch out for that bug!"]
    assert remainder == ""


def test_question_and_exclamation_marks_each_end_a_sentence() -> None:
    clauses, remainder = split_clauses("Why? Because it blocks! Then ", max_words=40)
    assert clauses == ["Why?", "Because it blocks!"]
    assert remainder == "Then "


def test_semicolon_is_no_sentence_end() -> None:
    text = "First we parse the tokens; then we emit clauses "
    assert split_clauses(text, max_words=40) == ([], text)


def test_colon_is_no_sentence_end() -> None:
    text = "Here is the plan: read the file first "
    assert split_clauses(text, max_words=40) == ([], text)


def test_comma_is_no_sentence_end() -> None:
    text = "After the buffer fills, we flush it downstream "
    assert split_clauses(text, max_words=40) == ([], text)


def test_first_chunk_runs_past_commas_to_the_sentence_end() -> None:
    clauses, remainder = split_clauses(
        "Start in pool.py, line 142, the acquire path. The pool ", max_words=40
    )
    assert clauses == ["Start in pool.py, line 142, the acquire path."]
    assert remainder == "The pool "


def test_a_one_word_sentence_is_its_own_chunk() -> None:
    clauses, remainder = split_clauses("Yes. That is exactly right. ", max_words=40)
    assert clauses == ["Yes.", "That is exactly right."]
    assert remainder == ""


@pytest.mark.parametrize(
    ("text", "sentence", "rest"),
    [
        ('He said "stop." Next', 'He said "stop."', "Next"),
        ("(see above.) Then", "(see above.)", "Then"),
        ("It said 'go!' Then", "It said 'go!'", "Then"),
        ("[is it done?] Then", "[is it done?]", "Then"),
        ('(he said "no.") Then', '(he said "no.")', "Then"),
    ],
)
def test_a_sentence_end_inside_closers_ends_the_chunk(text: str, sentence: str, rest: str) -> None:
    assert split_clauses(text, max_words=40) == ([sentence], rest)


def test_closers_with_no_sentence_end_inside_are_no_cut() -> None:
    text = 'He said "stop" then ( see above ) and '
    assert split_clauses(text, max_words=40) == ([], text)


def test_does_not_split_inside_eg_abbreviation() -> None:
    text = "Some languages e.g. Python are dynamically typed. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["Some languages e.g. Python are dynamically typed."]
    assert remainder == ""


def test_does_not_split_inside_ie_abbreviation() -> None:
    text = "Use the fast path i.e. skip the cache entirely. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["Use the fast path i.e. skip the cache entirely."]
    assert remainder == ""


def test_does_not_split_inside_etc_abbreviation() -> None:
    text = "It handles files sockets pipes etc. without issue. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["It handles files sockets pipes etc. without issue."]
    assert remainder == ""


def test_does_not_split_inside_vs_abbreviation() -> None:
    text = "This is a tradeoff of latency vs. throughput here. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["This is a tradeoff of latency vs. throughput here."]
    assert remainder == ""


@pytest.mark.parametrize("wrapped", ["(e.g.)", '"etc."', "[vs.]", "'i.e.'", '("Dr.")'])
def test_an_abbreviation_inside_brackets_or_quotes_ends_no_sentence(wrapped: str) -> None:
    text = f"Use a list {wrapped} for this and go. "
    assert split_clauses(text, max_words=40) == ([text.strip()], "")


def test_does_not_split_inside_dr_abbreviation() -> None:
    text = "Dr. Smith wrote the original paper on this. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["Dr. Smith wrote the original paper on this."]
    assert remainder == ""


def test_does_not_split_inside_decimal_number() -> None:
    text = "The constant is 3.14 and it never changes. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["The constant is 3.14 and it never changes."]
    assert remainder == ""


def test_does_not_split_inside_version_number() -> None:
    text = "We pinned the dependency to 1.15.0 for now. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["We pinned the dependency to 1.15.0 for now."]
    assert remainder == ""


def test_does_not_split_inside_module_path() -> None:
    text = "The fix lives in connection_pool.py near the top. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["The fix lives in connection_pool.py near the top."]
    assert remainder == ""


def test_does_not_split_inside_repo_relative_path() -> None:
    text = "Open tutor/tts.py and look at the queue. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["Open tutor/tts.py and look at the queue."]
    assert remainder == ""


def test_does_not_split_inside_attribute_access() -> None:
    text = "The call to self.acquire blocks until it succeeds. "
    clauses, remainder = split_clauses(text, max_words=40)
    assert clauses == ["The call to self.acquire blocks until it succeeds."]
    assert remainder == ""


def test_returns_tail_as_remainder_when_no_boundary_present() -> None:
    clauses, remainder = split_clauses("still waiting on more tokens", max_words=40)
    assert clauses == []
    assert remainder == "still waiting on more tokens"


def test_trailing_punctuation_with_no_whitespace_stays_in_remainder() -> None:
    clauses, remainder = split_clauses("The value is 3.", max_words=40)
    assert clauses == []
    assert remainder == "The value is 3."


def test_a_45_word_run_with_no_punctuation_cuts_after_word_40() -> None:
    words = _run(45)
    clauses, remainder = split_clauses(" ".join(words) + " ", max_words=40)
    assert clauses == [" ".join(words[:40])]
    assert remainder == " ".join(words[40:]) + " "


@pytest.mark.parametrize("mark", [",", ";", ":"])
def test_a_run_past_the_cap_cuts_after_its_clause_boundary(mark: str) -> None:
    words = _run(45, 30, mark=mark)
    clauses, remainder = split_clauses(" ".join(words) + " ", max_words=40)
    assert clauses == [" ".join(words[:30])]
    assert remainder == " ".join(words[30:]) + " "


def test_a_run_past_the_cap_cuts_after_its_last_clause_boundary() -> None:
    words = _run(45, 12, 30)
    clauses, remainder = split_clauses(" ".join(words) + " ", max_words=40)
    assert clauses == [" ".join(words[:30])]
    assert remainder == " ".join(words[30:]) + " "


def test_words_carried_past_a_clause_cut_count_toward_the_next_cap() -> None:
    words = _run(75, 30)
    clauses, remainder = split_clauses(" ".join(words) + " ", max_words=40)
    assert clauses == [" ".join(words[:30]), " ".join(words[30:70])]
    assert remainder == " ".join(words[70:]) + " "


def test_the_cap_waits_for_a_word_past_it() -> None:
    text = " ".join(_run(40, 30)) + " "
    assert split_clauses(text, max_words=40) == ([], text)


def test_cap_cut_remainder_can_hold_a_partial_word() -> None:
    text = "one two three four five six seven eight nine te"
    clauses, remainder = split_clauses(text, max_words=9)

    assert clauses == ["one two three four five six seven eight nine"]
    assert remainder == "te"


def test_return_shape_is_list_and_string_tuple() -> None:
    result = split_clauses("Hello there friend. ", max_words=40)
    assert isinstance(result, tuple)
    assert len(result) == 2
    clauses, remainder = result
    assert isinstance(clauses, list)
    assert isinstance(remainder, str)


async def _stream(pieces: list[str]) -> AsyncIterator[str]:
    for piece in pieces:
        yield piece


async def test_mid_word_token_cuts_still_yield_whole_clauses() -> None:
    pieces = [
        "The sys",
        "tem logs each reque",
        "st.",
        " Then it wri",
        "tes the response to di",
        "sk. ",
    ]
    clauses = [clause async for clause in clause_chunks(_stream(pieces))]
    assert clauses == ["The system logs each request.", "Then it writes the response to disk."]


async def test_every_chunk_including_the_first_runs_to_a_sentence_end() -> None:
    text = (
        "Yes. That is right, and then also, we look at the second clause, which is longer. Done. "
    )
    tokens = [f"{word} " for word in text.split()]

    clauses = [clause async for clause in clause_chunks(_stream(tokens))]

    assert clauses == [
        "Yes.",
        "That is right, and then also, we look at the second clause, which is longer.",
        "Done.",
    ]


async def test_a_sentence_end_at_the_buffer_end_waits_for_the_next_delta() -> None:
    clauses = [
        clause async for clause in clause_chunks(_stream(["The value is 3.", "14 exactly. "]))
    ]
    assert clauses == ["The value is 3.14 exactly."]


async def test_a_sentence_end_at_the_end_of_the_stream_is_flushed() -> None:
    clauses = [clause async for clause in clause_chunks(_stream(["The value is ", "3."]))]
    assert clauses == ["The value is 3."]


async def test_trailing_remainder_is_flushed_stripped_at_source_exhaustion() -> None:
    tokens = ["This ", "finishes ", "quickly. ", "Wrap up"]

    clauses = [clause async for clause in clause_chunks(_stream(tokens))]

    assert clauses == ["This finishes quickly.", "Wrap up"]


async def test_empty_source_yields_nothing() -> None:
    clauses = [clause async for clause in clause_chunks(_stream([]))]
    assert clauses == []


async def test_cancelling_the_consumer_propagates_and_does_not_flush_the_buffer() -> None:
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def source() -> AsyncIterator[str]:
        entered.set()
        yield "partial fragment without a boundary "
        await blocked.wait()
        yield "unreachable "

    received: list[str] = []

    async def consume() -> None:
        async for clause in clause_chunks(source()):
            received.append(clause)

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert received == []
    assert task.cancelled()


async def test_one_token_with_several_sentences_yields_each_sentence() -> None:
    text = "Yes, that is right. And then also, we look at the second clause. Which is longer! "

    clauses = [clause async for clause in clause_chunks(_stream([text]))]

    assert clauses == [
        "Yes, that is right.",
        "And then also, we look at the second clause.",
        "Which is longer!",
    ]


async def test_default_cap_cuts_a_boundary_free_run_every_40_words() -> None:
    tokens = [f"w{i} " for i in range(90)]

    clauses = [clause async for clause in clause_chunks(_stream(tokens))]

    assert clauses == [
        " ".join(f"w{i}" for i in range(40)),
        " ".join(f"w{i}" for i in range(40, 80)),
        " ".join(f"w{i}" for i in range(80, 90)),
    ]


async def test_a_streamed_run_carries_words_past_a_clause_cut_into_the_next_chunk() -> None:
    words = _run(75, 30)

    clauses = [clause async for clause in clause_chunks(_stream([f"{w} " for w in words]))]

    assert clauses == [" ".join(words[:30]), " ".join(words[30:70]), " ".join(words[70:])]


def _scrub(deltas: list[str]) -> tuple[str, Scrubber]:
    scrubber = Scrubber()
    text = "".join(scrubber.feed(delta) for delta in deltas) + scrubber.flush()
    return text, scrubber


def test_a_fenced_block_is_dropped_whole() -> None:
    text, scrubber = _scrub(
        ["lifecycle.\n\n``", '`json\n{"type":', '"diagram"}\n``', "`\n\nPhase: Teach."]
    )
    assert text == "lifecycle.\n\n \n\nPhase: Teach."
    assert scrubber.dropped == Counter({"fence": 1})


def test_a_json_payload_is_dropped_to_its_closing_brace() -> None:
    text, scrubber = _scrub(['attempts.{"diag', 'ram":{"a":1}}We', "'re in"])
    assert text == "attempts. We're in"
    assert scrubber.dropped == Counter({"json": 1})


def test_an_unbalanced_payload_gives_the_turn_back() -> None:
    text, scrubber = _scrub(['{"' + "x" * 9000, " and so on."])
    assert text.endswith(" and so on.")
    assert scrubber.dropped == Counter({"json": 1, "json_unbalanced": 1})


def test_inline_markers_are_removed_from_prose() -> None:
    text, scrubber = _scrub(["the `acq", "uire` method is **free", "** now"])
    assert text == "the acquire method is free now"
    assert scrubber.dropped == Counter({"backtick": 2, "bold": 2})


def test_a_tag_shaped_token_is_dropped() -> None:
    text, scrubber = _scrub(["path.\n<", "/turn>"])
    assert text == "path.\n"
    assert scrubber.dropped == Counter({"tag": 1})

    text, scrubber = _scrub(["a < b and", " c > d"])
    assert text == "a < b and c > d"
    assert scrubber.dropped == Counter()


def test_a_lone_trailing_backtick_is_held_then_released() -> None:
    scrubber = Scrubber()
    fed = scrubber.feed("ends here `")
    assert fed == "ends here "
    assert scrubber.flush() == ""
    assert scrubber.dropped == Counter({"backtick": 1})


def test_the_first_delta_can_open_a_fence() -> None:
    text, scrubber = _scrub(["```", "json\n{}\n```", "Speech."])
    assert text == " Speech."
    assert scrubber.dropped == Counter({"fence": 1})


def test_a_payload_with_a_space_after_the_brace_is_dropped() -> None:
    text, scrubber = _scrub(['forever.\n\n{ "d', 'iagram": 1}Next.'])
    assert text == "forever.\n\n Next."
    assert scrubber.dropped == Counter({"json": 1})


def test_dropped_chars_counts_what_each_class_removed() -> None:
    text, scrubber = _scrub(['a ```x``` b {"k":1} c </turn> `d` **e**'])
    assert text == "a   b   c  d e"
    assert scrubber.dropped == Counter({"fence": 1, "json": 1, "tag": 1, "backtick": 2, "bold": 2})
    assert scrubber.dropped_chars == Counter(
        {"fence": 7, "json": 7, "tag": 7, "backtick": 2, "bold": 4}
    )


async def test_spoken_text_feeds_the_chunker_clean_clauses() -> None:
    pieces = [
        "The pool is a free list. ",
        '```json\n{"a":1}\n```',
        " The acquire path takes a slot from it.",
    ]
    clauses = [clause async for clause in clause_chunks(spoken_text(_stream(pieces), Scrubber()))]
    assert clauses == ["The pool is a free list.", "The acquire path takes a slot from it."]


async def tokens(*items: str | RawTag) -> AsyncIterator[str | RawTag]:
    for item in items:
        yield item


async def test_a_tag_token_flushes_the_clause_before_it_whatever_its_length() -> None:
    out = [
        item
        async for item in clause_chunks(
            tokens("It falls", RawTag("step 2"), " and then it rises again, slowly.")
        )
    ]
    assert out == ["It falls", RawTag("step 2"), "and then it rises again, slowly."]


async def test_a_tag_token_with_nothing_before_it_passes_alone() -> None:
    out = [
        item
        async for item in clause_chunks(
            tokens(RawTag("scene 1"), "Here is the curve, and here is its slope.")
        )
    ]
    assert out == [RawTag("scene 1"), "Here is the curve, and here is its slope."]


async def test_adjacent_tag_tokens_pass_in_order_with_no_empty_clause() -> None:
    out = [
        item
        async for item in clause_chunks(
            tokens("Done here.", RawTag("step 2"), " ", RawTag("step 3"), " Now.")
        )
    ]
    assert out == ["Done here.", RawTag("step 2"), RawTag("step 3"), "Now."]


async def test_spoken_text_passes_a_tag_token_through_and_never_scrubs_it() -> None:
    scrubber = Scrubber()
    out = [
        item
        async for item in spoken_text(
            tokens("Look **here**", RawTag("set lr 0.9"), " now"), scrubber
        )
    ]
    assert out == ["Look here", RawTag("set lr 0.9"), " now"]
    assert scrubber.dropped == Counter({"bold": 2})
